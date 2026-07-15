"""Tier 2: shot attempt detection + the outcome honesty ladder. Design doc §5.2.

Attempt detection (reliable half): a launch from a plausible shooting zone whose
projected goal-line crossing falls within the goal mouth, at speed. Outcome
classification (honesty-ladder half): what tracking alone can distinguish —

- ``goal``       only via Tier 1 corroboration (a KICKOFF follows); indirect.
- ``saved``/``blocked``  flight reversal at an opponent (keeper vs outfielder).
- ``off_target`` tracked crossing outside the mouth, or a GOAL_KICK follows.
- ``unresolved`` flight lost, no corroborating restart (broadcast cut).

The 2D blind spot is structural: no ball height, so over-the-bar and top-corner are
identical trajectories; the restart that follows is what disambiguates, ex post.
``outcome_confidence`` is therefore separate from the attempt ``confidence``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Sequence, Tuple

from ..patterns.tracking import MatchMeta, Role, TeamSide
from .base import EventDetector, EventDetectorRegistry, combine_evidence, soft_threshold
from .kinematics import BallLaunch, KinematicSegment, KinematicSeries, detect_launches
from .model import EventType, MatchEvent
from .restarts import GOAL_HALF_WIDTH_M


@dataclass
class ShotConfig:
    """Thresholds; provenance in design doc §10 (zone/speed engine-original)."""

    max_shot_dist_m: float = 35.0
    min_shot_speed_ms: float = 9.0
    mouth_margin_m: float = 1.0        # + boundary noise on top
    attempt_score_min: float = 0.25    # product of the soft factors must clear this
    aftermath_s: float = 4.0
    blocker_radius_m: float = 2.5
    keeper_line_dist_m: float = 6.0    # reversal this close to the goal line = save
    reversal_speed_drop: float = 0.5   # toward-goal speed shed to count as a reversal
    lost_frames: int = 2
    near_goal_m: float = 6.0           # flight lost this close to the mouth = "died at goal"
    restart_corroboration_s: float = 90.0
    min_confidence: float = 0.2


@dataclass
class _Aftermath:
    end_index: int
    crossed: Optional[str] = None        # "in_mouth" | "wide" | None
    blocker_track_id: Optional[int] = None
    blocker_is_keeper: bool = False
    died_near_goal: bool = False


@EventDetectorRegistry.register
class ShotDetector(EventDetector):
    """Goalward-launch shot detection with restart-corroborated outcomes."""

    detector_id: ClassVar[str] = "shots"
    display_name: ClassVar[str] = "Shots"
    event_types: ClassVar[Tuple[EventType, ...]] = (EventType.SHOT,)
    stage: ClassVar[int] = 2  # after restarts (kickoff/goal-kick corroboration)

    def __init__(self, config: Optional[ShotConfig] = None):
        self.config = config or ShotConfig()

    def detect(
        self, kin: KinematicSeries, context: Sequence[MatchEvent] = ()
    ) -> List[MatchEvent]:
        cfg = self.config
        restarts = sorted(
            (e for e in context if e.event_type in
             (EventType.KICKOFF, EventType.GOAL_KICK, EventType.CORNER)),
            key=lambda e: e.start_s,
        )
        events: List[MatchEvent] = []
        for seg in kin.segments:
            for launch in detect_launches(seg, kin.config):
                event = self._classify_launch(kin, seg, launch, restarts)
                if event is not None and event.confidence >= cfg.min_confidence:
                    events.append(event)
        return events

    def _classify_launch(
        self, kin: KinematicSeries, seg: KinematicSegment,
        launch: BallLaunch, restarts: List[MatchEvent],
    ) -> Optional[MatchEvent]:
        cfg = self.config
        team = launch.kicker_team
        if team is None:
            return None
        meta = kin.meta
        noise = kin.config.boundary_noise_m
        direction = meta.attack_direction(team, seg.period)
        goal_x = direction * meta.pitch_length_m / 2

        dist = math.hypot(goal_x - launch.x, launch.y)
        if dist > cfg.max_shot_dist_m * 1.3:
            return None
        if launch.vx * direction <= 0.5:
            return None  # not travelling toward the goal line
        y_proj = launch.y + launch.vy * (goal_x - launch.x) / launch.vx
        mouth_limit = GOAL_HALF_WIDTH_M + cfg.mouth_margin_m + noise

        mouth_factor = soft_threshold(abs(y_proj), mouth_limit, 1.5, above=False)
        speed_factor = soft_threshold(launch.speed, cfg.min_shot_speed_ms, 2.0)
        dist_factor = soft_threshold(dist, cfg.max_shot_dist_m, 5.0, above=False)
        if mouth_factor * speed_factor * dist_factor < cfg.attempt_score_min:
            return None

        kicker_factor = soft_threshold(
            launch.kicker_dist_m if launch.kicker_dist_m is not None else 99.0,
            2.5, 1.0, above=False,
        )
        confidence = combine_evidence([mouth_factor, speed_factor, dist_factor, kicker_factor])

        aftermath = self._follow_aftermath(seg, launch, team, direction, goal_x, noise)
        outcome, outcome_conf, restart_after = self._resolve_outcome(
            launch, aftermath, restarts, mouth_factor
        )
        return MatchEvent(
            event_type=EventType.SHOT,
            team=team,
            start_s=launch.t_s,
            end_s=seg.grid[aftermath.end_index],
            confidence=confidence,
            period=seg.period,
            x=launch.x,
            y=launch.y,
            metadata={
                "distance_m": round(dist, 1),
                "launch_speed_ms": round(launch.speed, 1),
                "y_projected": round(y_proj, 2),
                "shooter_track_id": launch.kicker_track_id,
                "outcome": outcome,
                "outcome_confidence": round(outcome_conf, 2),
                "in_mouth_observed": aftermath.crossed == "in_mouth",
                "died_near_goal": aftermath.died_near_goal,
                "blocker_track_id": aftermath.blocker_track_id,
                "restart_after": restart_after,
            },
        )

    def _follow_aftermath(
        self, seg: KinematicSegment, launch: BallLaunch,
        team: TeamSide, direction: int, goal_x: float, noise: float,
    ) -> _Aftermath:
        cfg = self.config
        u_launch = launch.vx * direction
        lost = 0
        end = launch.index
        for j in range(launch.index + 1, len(seg)):
            if seg.grid[j] > launch.t_s + cfg.aftermath_s:
                break
            end = j
            pos = seg.ball_pos(j)
            if pos is None:
                lost += 1
                if lost >= cfg.lost_frames:
                    prev = seg.ball_pos(j - lost)
                    died_near = prev is not None and math.hypot(
                        goal_x - prev[0], max(0.0, abs(prev[1]) - GOAL_HALF_WIDTH_M)
                    ) <= cfg.near_goal_m
                    return _Aftermath(end_index=j, died_near_goal=died_near)
                continue
            lost = 0
            prev = seg.ball_pos(j - 1)
            if prev is not None:
                c0, c1 = prev[0] * direction, pos[0] * direction
                line = goal_x * direction
                if c0 <= line < c1:
                    frac = (line - c0) / (c1 - c0)
                    y_c = prev[1] + frac * (pos[1] - prev[1])
                    in_mouth = abs(y_c) <= GOAL_HALF_WIDTH_M + noise
                    return _Aftermath(end_index=j, crossed="in_mouth" if in_mouth else "wide")
            u = (seg.ball_vx[j] or 0.0) * direction
            if u < (1.0 - cfg.reversal_speed_drop) * u_launch - 1e-9 and u < 1.0:
                blocker = seg.nearest_player(
                    j, pos[0], pos[1], team=team.opponent(), max_dist_m=cfg.blocker_radius_m
                )
                if blocker is not None:
                    tr = blocker[0]
                    is_keeper = tr.role is Role.GOALKEEPER
                    bp = tr.pos(j)
                    near_line = bp is not None and abs(goal_x - bp[0]) <= cfg.keeper_line_dist_m
                    return _Aftermath(
                        end_index=j,
                        blocker_track_id=tr.track_id,
                        blocker_is_keeper=is_keeper or near_line,
                    )
        return _Aftermath(end_index=end)

    def _resolve_outcome(
        self, launch: BallLaunch, aftermath: _Aftermath,
        restarts: List[MatchEvent], mouth_factor: float,
    ) -> Tuple[str, float, Optional[str]]:
        cfg = self.config
        restart = next(
            (e for e in restarts
             if launch.t_s < e.start_s <= launch.t_s + cfg.restart_corroboration_s),
            None,
        )
        restart_type = restart.event_type.value if restart else None

        if restart is not None and restart.event_type is EventType.KICKOFF:
            if aftermath.crossed == "in_mouth" or aftermath.died_near_goal:
                return "goal", combine_evidence([0.9, mouth_factor, restart.confidence]), restart_type
            return "goal", 0.45, restart_type  # kickoff followed, mouth evidence missing
        if aftermath.blocker_track_id is not None:
            outcome = "saved" if aftermath.blocker_is_keeper else "blocked"
            conf = 0.55 if aftermath.blocker_is_keeper else 0.5
            return outcome, conf, restart_type
        if aftermath.crossed == "wide":
            return "off_target", 0.7, restart_type
        if restart is not None and restart.event_type is EventType.GOAL_KICK:
            return "off_target", 0.6, restart_type
        if restart is not None and restart.event_type is EventType.CORNER:
            # kept out and put behind by a defender we didn't see touch it
            return "saved", 0.4, restart_type
        if aftermath.crossed == "in_mouth" or aftermath.died_near_goal:
            # goal-mouth evidence without the kickoff: honest limbo, lean nothing
            return "unresolved", 0.25, restart_type
        return "unresolved", 0.2, restart_type
