"""Tier 2: set-piece organization — the primary dead-ball deliverable (§5.3).

Together with the Tier 1 restarts this is what "dead-ball phase detection" means
in this design: recognize that a dead-ball restart is happening and roughly type
it. Boundary restarts arrive typed from Tier 1; this detector covers the
remainder — free-kick-shaped restarts — via their organizational signature (a
wall, box loading, or mere staticness), WITHOUT claiming to know why the game
stopped. The event type deliberately says ``set_piece_setup``, not ``free_kick``:
offside restarts and drop balls produce the same picture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median, pstdev
from typing import ClassVar, List, Optional, Sequence, Tuple

from ..patterns.tracking import TeamSide
from .base import EventDetector, EventDetectorRegistry, combine_evidence, soft_threshold
from .kinematics import BallLaunch, KinematicSegment, KinematicSeries, detect_launches
from .model import EventType, MatchEvent

PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_HALF_WIDTH_M = 20.16


@dataclass
class SetPieceConfig:
    """Thresholds; wall geometry formalises the 9.15 m law (design doc §10)."""

    dead_ball_speed_ms: float = 0.5
    min_setup_s: float = 6.0
    tail_window_s: float = 3.0
    static_speed_ms: float = 0.7
    min_visible_players: int = 8
    wall_min_players: int = 3
    wall_annulus_m: Tuple[float, float] = (6.0, 12.0)
    wall_gap_m: float = 2.2            # max adjacent spacing along the wall line
    wall_perp_spread_m: float = 1.0    # max std of offsets across the wall line
    box_load_min_each: int = 3
    box_load_max_goal_dist_m: float = 40.0
    runners_window_s: float = 2.0
    resumption_slack_s: float = 1.5
    static_only_cap: float = 0.55  # staticness alone must never look like a wall
    min_confidence: float = 0.25


@EventDetectorRegistry.register
class SetPieceDetector(EventDetector):
    """Dead spell + static organization + delivery => set-piece restart."""

    detector_id: ClassVar[str] = "setpieces"
    display_name: ClassVar[str] = "Set-piece organization"
    event_types: ClassVar[Tuple[EventType, ...]] = (EventType.SET_PIECE_SETUP,)
    stage: ClassVar[int] = 4  # after restarts: skip already-typed dead spells

    def __init__(self, config: Optional[SetPieceConfig] = None):
        self.config = config or SetPieceConfig()

    def detect(
        self, kin: KinematicSeries, context: Sequence[MatchEvent] = ()
    ) -> List[MatchEvent]:
        cfg = self.config
        tier1 = [e for e in context if e.tier == 1]
        events: List[MatchEvent] = []
        for seg in kin.segments:
            launches = detect_launches(seg, kin.config)
            for start_i, end_i in self._dead_runs(seg):
                run_t0, run_t1 = seg.grid[start_i], seg.grid[end_i]
                resumption = next(
                    (l for l in launches
                     if run_t1 - 0.4 <= l.t_s <= run_t1 + cfg.resumption_slack_s),
                    None,
                )
                end_t = resumption.t_s if resumption else run_t1
                if any(e.overlaps(run_t0, end_t) for e in tier1):
                    continue  # already typed by a boundary restart / kickoff
                event = self._build_event(kin, seg, start_i, end_i, resumption)
                if event is not None and event.confidence >= cfg.min_confidence:
                    events.append(event)
        return events

    # -- dead spells -------------------------------------------------------------

    def _dead_runs(self, seg: KinematicSegment) -> List[Tuple[int, int]]:
        """[start, end] index spans where the ball is at rest or untracked."""
        cfg = self.config
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i in range(len(seg)):
            speed = seg.ball_speed(i)
            dead = seg.ball_pos(i) is None or (speed is not None and speed < cfg.dead_ball_speed_ms)
            if dead and start is None:
                start = i
            elif not dead and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(seg) - 1))
        return [
            (a, b) for a, b in runs
            if seg.grid[b] - seg.grid[a] >= cfg.min_setup_s - 1e-9
        ]

    # -- event assembly -------------------------------------------------------------

    def _build_event(
        self, kin: KinematicSeries, seg: KinematicSegment,
        start_i: int, end_i: int, resumption: Optional[BallLaunch],
    ) -> Optional[MatchEvent]:
        cfg = self.config
        tail_lo = max(start_i, end_i - int(round(cfg.tail_window_s / kin.frame_interval_s())))
        tail_mid = (tail_lo + end_i) // 2

        speeds = [
            s for tr, _, _ in seg.visible_players(tail_mid)
            if (s := tr.speed(tail_mid)) is not None
        ]
        if len(speeds) < cfg.min_visible_players:
            return None
        static_factor = soft_threshold(median(speeds), cfg.static_speed_ms, 0.3, above=False)

        dead_xy = self._dead_ball_point(seg, start_i, end_i, resumption)
        wall = self._find_wall(kin, seg, tail_mid, dead_xy) if dead_xy else None
        box_loaded = self._box_loaded(kin, seg, tail_mid, dead_xy)

        if wall is not None:
            organization, org_factor = "wall", 0.9
        elif box_loaded:
            organization, org_factor = "box_load", 0.75
        else:
            organization, org_factor = "static_only", 0.45

        valid = sum(1 for i in range(start_i, end_i + 1) if seg.ball_pos(i) is not None)
        ball_observed = valid / max(1, end_i - start_i + 1)
        dead_evidence = 0.5 + 0.5 * ball_observed

        confidence = combine_evidence([static_factor, org_factor, dead_evidence])
        if organization == "static_only":
            confidence *= cfg.static_only_cap
        taker_team = resumption.kicker_team if resumption else None
        runners = (
            self._runners_into_box(kin, seg, resumption, taker_team)
            if resumption is not None else None
        )
        return MatchEvent(
            event_type=EventType.SET_PIECE_SETUP,
            team=taker_team,
            start_s=seg.grid[start_i],
            end_s=resumption.t_s if resumption else seg.grid[end_i],
            confidence=confidence,
            period=seg.period,
            x=dead_xy[0] if dead_xy else None,
            y=dead_xy[1] if dead_xy else None,
            metadata={
                "organization": organization,
                "wall_team": wall[0].value if wall else None,
                "wall_size": wall[1] if wall else None,
                "box_loaded": box_loaded,
                "taker_track_id": resumption.kicker_track_id if resumption else None,
                "resumption_observed": resumption is not None,
                "runners_into_box": runners,
                "ball_observed_fraction": round(ball_observed, 2),
            },
        )

    def _dead_ball_point(
        self, seg: KinematicSegment, start_i: int, end_i: int,
        resumption: Optional[BallLaunch],
    ) -> Optional[Tuple[float, float]]:
        xs = [p[0] for i in range(start_i, end_i + 1) if (p := seg.ball_pos(i)) is not None]
        ys = [p[1] for i in range(start_i, end_i + 1) if (p := seg.ball_pos(i)) is not None]
        if len(xs) >= 3:
            return median(xs), median(ys)
        if resumption is not None:
            return resumption.x, resumption.y
        return None

    def _find_wall(
        self, kin: KinematicSeries, seg: KinematicSegment, i: int,
        dead_xy: Tuple[float, float],
    ) -> Optional[Tuple[TeamSide, int]]:
        """A tight, collinear same-team cluster in the 9.15 m annulus, goal-side."""
        cfg = self.config
        lo, hi = cfg.wall_annulus_m
        bx, by = dead_xy
        for team in (TeamSide.HOME, TeamSide.AWAY):
            direction = kin.meta.attack_direction(team, seg.period)
            own_goal = (-direction * kin.meta.pitch_length_m / 2, 0.0)
            ball_goal_d = math.hypot(own_goal[0] - bx, own_goal[1] - by)
            members = []
            for tr, x, y in seg.visible_players(i, team):
                d = math.hypot(x - bx, y - by)
                if lo <= d <= hi and math.hypot(own_goal[0] - x, own_goal[1] - y) < ball_goal_d:
                    members.append((x, y))
            if len(members) < cfg.wall_min_players:
                continue
            if self._is_line(members):
                return team, len(members)
        return None

    def _is_line(self, points: List[Tuple[float, float]]) -> bool:
        """Principal-axis fit: tight perpendicular spread, no gaps along the line."""
        cfg = self.config
        n = len(points)
        mx = sum(x for x, _ in points) / n
        my = sum(y for _, y in points) / n
        sxx = sum((x - mx) ** 2 for x, _ in points) / n
        syy = sum((y - my) ** 2 for _, y in points) / n
        sxy = sum((x - mx) * (y - my) for x, y in points) / n
        theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
        ux, uy = math.cos(theta), math.sin(theta)
        along = sorted((x - mx) * ux + (y - my) * uy for x, y in points)
        perp = [-(x - mx) * uy + (y - my) * ux for x, y in points]
        if pstdev(perp) > cfg.wall_perp_spread_m:
            return False
        return all(b - a <= cfg.wall_gap_m for a, b in zip(along, along[1:]))

    def _box_loaded(
        self, kin: KinematicSeries, seg: KinematicSegment, i: int,
        dead_xy: Optional[Tuple[float, float]],
    ) -> bool:
        cfg = self.config
        half_len = kin.meta.pitch_length_m / 2
        for sign in (1, -1):
            goal = (sign * half_len, 0.0)
            if dead_xy is not None and math.hypot(
                dead_xy[0] - goal[0], dead_xy[1] - goal[1]
            ) > cfg.box_load_max_goal_dist_m:
                continue
            counts = {TeamSide.HOME: 0, TeamSide.AWAY: 0}
            for tr, x, y in seg.visible_players(i):
                if x * sign >= half_len - PENALTY_BOX_DEPTH_M and abs(y) <= PENALTY_BOX_HALF_WIDTH_M:
                    counts[tr.team] += 1
            if all(c >= cfg.box_load_min_each for c in counts.values()):
                return True
        return False

    def _runners_into_box(
        self, kin: KinematicSeries, seg: KinematicSegment,
        resumption: BallLaunch, taker_team: Optional[TeamSide],
    ) -> int:
        """Players entering a penalty box shortly after the delivery."""
        cfg = self.config
        half_len = kin.meta.pitch_length_m / 2
        i0 = resumption.index
        i1 = min(len(seg) - 1, i0 + int(round(cfg.runners_window_s / kin.frame_interval_s())))

        def in_box(x: float, y: float, sign: int) -> bool:
            return x * sign >= half_len - PENALTY_BOX_DEPTH_M and abs(y) <= PENALTY_BOX_HALF_WIDTH_M

        best = 0
        for sign in (1, -1):
            runners = 0
            for tr, x1, y1 in seg.visible_players(i1, taker_team):
                p0 = tr.pos(i0)
                if p0 is not None and not in_box(p0[0], p0[1], sign) and in_box(x1, y1, sign):
                    runners += 1
            best = max(best, runners)
        return best
