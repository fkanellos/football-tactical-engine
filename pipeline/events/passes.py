"""Tier 2: pass detection and the subtype ladder. Design doc §5.1.

A pass is a launch -> flight -> reception arc: a launch attributed to player A, a
flight during which the ball separates from A (else it is a carry — rejected), and
a reception by the first player controlling the ball. An ``intercepted`` outcome is
a turnover with a timestamp and a location — the signal Phase 4's possession
machinery is designed to consume after the §8 refactor.

Subtypes (cutback / cross / through_ball / length buckets) ride in metadata with
their own ``subtype_confidence``: the subtype is always less certain than the pass.
Known recall floor: 5 Hz + smoothing hides soft kicks (< ~5 m/s) and sub-second
one-twos; ball dropouts mid-flight resolve to honest ``unresolved`` outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Tuple

from ..patterns.features import to_team_relative
from ..patterns.tracking import MatchMeta, Role, TeamSide
from .base import EventDetector, EventDetectorRegistry, combine_evidence, soft_threshold
from .kinematics import BallLaunch, KinematicSegment, KinematicSeries, detect_launches
from .model import EventType, MatchEvent


@dataclass
class PassConfig:
    """Thresholds; provenance in design doc §10 (bucket edges engine-original)."""

    control_radius_m: float = 2.0       # reception radius (possession-radius basis)
    separation_min_m: float = 2.5       # ball must leave the kicker by this, else carry
    max_flight_s: float = 6.0
    max_lost_frames: int = 3            # ball missing this long ends the flight
    reception_speed_ms: float = 6.0     # arriving ball may still move this fast at the foot
    reception_slow_factor: float = 0.8  # ...or must have shed this share of launch speed
    shot_exclusion_s: float = 0.3       # launches this close to a SHOT are the shot
    min_confidence: float = 0.2

    # subtype geometry (design doc §5.1 table), team-relative coordinates
    short_max_m: float = 15.0
    long_min_m: float = 30.0
    through_line_margin_m: float = 1.0
    through_min_x_rel: float = 55.0
    min_line_players: int = 4           # opposing outfielders needed to place the line
    cross_min_y_rel: float = 18.0
    cross_min_x_rel: float = 70.0
    cross_target_x_rel: float = 86.5
    cross_target_y_rel: float = 21.0
    cross_centre_gain_m: float = 4.0
    cutback_min_x_rel: float = 94.0
    cutback_min_y_rel: float = 12.0
    cutback_target_x_min: float = 80.0
    cutback_target_y_rel: float = 12.0


@dataclass
class _Flight:
    """Resolved flight of one launch (internal)."""

    end_index: int
    receiver_track_id: Optional[int]
    receiver_team: Optional[TeamSide]
    outcome: str            # completed | intercepted | out_of_play | unresolved
    max_separation_m: float
    rx: Optional[float]
    ry: Optional[float]
    valid_fraction: float


@EventDetectorRegistry.register
class PassDetector(EventDetector):
    """Launch -> flight -> reception pass detection."""

    detector_id: ClassVar[str] = "passes"
    display_name: ClassVar[str] = "Passes"
    event_types: ClassVar[Tuple[EventType, ...]] = (EventType.PASS,)
    stage: ClassVar[int] = 3  # after shots: a launch is one or the other

    def __init__(self, config: Optional[PassConfig] = None):
        self.config = config or PassConfig()

    def detect(
        self, kin: KinematicSeries, context: Sequence[MatchEvent] = ()
    ) -> List[MatchEvent]:
        cfg = self.config
        shot_times = [e.start_s for e in context if e.event_type is EventType.SHOT]
        events: List[MatchEvent] = []
        for seg in kin.segments:
            launches = detect_launches(seg, kin.config)
            for pos_in_list, launch in enumerate(launches):
                if launch.kicker_team is None:
                    continue  # an unattributed kick is a loose ball, not a pass
                if any(abs(launch.t_s - t) < cfg.shot_exclusion_s for t in shot_times):
                    continue
                next_launch = launches[pos_in_list + 1] if pos_in_list + 1 < len(launches) else None
                flight = self._resolve_flight(kin, seg, launch, next_launch)
                if flight is None:
                    continue
                events.append(self._build_event(kin.meta, seg, launch, flight))
        return [e for e in events if e.confidence >= cfg.min_confidence]

    # -- flight resolution -------------------------------------------------------

    def _resolve_flight(
        self, kin: KinematicSeries, seg: KinematicSegment,
        launch: BallLaunch, next_launch: Optional[BallLaunch],
    ) -> Optional[_Flight]:
        cfg = self.config
        half_len = kin.meta.pitch_length_m / 2
        half_wid = kin.meta.pitch_width_m / 2
        kicker = next(
            (tr for tr in seg.players if tr.track_id == launch.kicker_track_id), None
        )
        max_sep = 0.0
        lost = 0
        n_frames = 0
        n_valid = 0
        last_valid: Optional[Tuple[int, float, float]] = None

        for j in range(launch.index + 1, len(seg)):
            if seg.grid[j] > launch.t_s + cfg.max_flight_s:
                break
            n_frames += 1
            pos = seg.ball_pos(j)
            if pos is None:
                lost += 1
                if lost >= cfg.max_lost_frames:
                    return self._flight(j, None, None, "unresolved", max_sep, last_valid, n_valid, n_frames)
                continue
            lost = 0
            n_valid += 1
            last_valid = (j, pos[0], pos[1])
            if abs(pos[0]) > half_len or abs(pos[1]) > half_wid:
                return self._flight(j, None, None, "out_of_play", max_sep, last_valid, n_valid, n_frames)
            if kicker is not None:
                kp = kicker.pos(j)
                if kp is not None:
                    max_sep = max(max_sep, math.hypot(kp[0] - pos[0], kp[1] - pos[1]))
            if next_launch is not None and j >= next_launch.index:
                # one-touch: the next launch's kicker received this pass
                receiver = next(
                    (tr for tr in seg.players if tr.track_id == next_launch.kicker_track_id),
                    None,
                )
                if receiver is not None and receiver.track_id != launch.kicker_track_id:
                    return self._resolved(j, receiver, launch, max_sep, pos, n_valid, n_frames)
                return self._flight(j, None, None, "unresolved", max_sep, last_valid, n_valid, n_frames)
            receiver = self._reception_at(seg, j, pos, launch, max_sep)
            if receiver is not None:
                return self._resolved(j, receiver, launch, max_sep, pos, n_valid, n_frames)
        if max_sep < cfg.separation_min_m:
            return None  # never left the kicker: a carry, not a pass
        return self._flight(
            min(launch.index + int(cfg.max_flight_s / kin.frame_interval_s()), len(seg) - 1),
            None, None, "unresolved", max_sep, last_valid, n_valid, max(1, n_frames),
        )

    def _reception_at(
        self, seg: KinematicSegment, j: int, pos: Tuple[float, float],
        launch: BallLaunch, max_sep: float,
    ):
        cfg = self.config
        speed = seg.ball_speed(j)
        slowed = speed is None or speed <= max(
            cfg.reception_speed_ms, cfg.reception_slow_factor * launch.speed
        )
        if not slowed:
            return None
        candidates = seg.players_within(j, pos[0], pos[1], cfg.control_radius_m)
        candidates = [
            (tr, d) for tr, d in candidates
            if tr.track_id != launch.kicker_track_id or max_sep >= cfg.separation_min_m
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda c: c[1])[0]

    def _resolved(self, j, receiver, launch, max_sep, pos, n_valid, n_frames) -> Optional[_Flight]:
        cfg = self.config
        if max_sep < cfg.separation_min_m:
            return None  # controlled back by/near the kicker without real separation
        if receiver.track_id == launch.kicker_track_id:
            return None
        outcome = "completed" if receiver.team is launch.kicker_team else "intercepted"
        return _Flight(
            end_index=j, receiver_track_id=receiver.track_id, receiver_team=receiver.team,
            outcome=outcome, max_separation_m=max_sep, rx=pos[0], ry=pos[1],
            valid_fraction=n_valid / max(1, n_frames),
        )

    def _flight(self, j, rid, rteam, outcome, max_sep, last_valid, n_valid, n_frames) -> Optional[_Flight]:
        if max_sep < self.config.separation_min_m:
            return None
        return _Flight(
            end_index=j, receiver_track_id=rid, receiver_team=rteam, outcome=outcome,
            max_separation_m=max_sep,
            rx=last_valid[1] if last_valid else None,
            ry=last_valid[2] if last_valid else None,
            valid_fraction=n_valid / max(1, n_frames),
        )

    # -- event assembly ------------------------------------------------------------

    def _build_event(
        self, meta: MatchMeta, seg: KinematicSegment, launch: BallLaunch, flight: _Flight
    ) -> MatchEvent:
        cfg = self.config
        launch_factor = soft_threshold(launch.speed, cfg.reception_speed_ms - 1.5, 1.5)
        kicker_factor = soft_threshold(
            launch.kicker_dist_m if launch.kicker_dist_m is not None else 99.0,
            2.5, 1.0, above=False,
        )
        separation_factor = soft_threshold(flight.max_separation_m, cfg.separation_min_m, 1.0)
        reception_factor = {
            "completed": 0.9, "intercepted": 0.85, "out_of_play": 0.55, "unresolved": 0.35,
        }[flight.outcome]
        confidence = combine_evidence(
            [launch_factor, kicker_factor, separation_factor, reception_factor],
            quality=0.5 + 0.5 * flight.valid_fraction,
        )

        metadata = {
            "passer_track_id": launch.kicker_track_id,
            "receiver_track_id": flight.receiver_track_id,
            "outcome": flight.outcome,
            "launch_speed_ms": round(launch.speed, 1),
        }
        if flight.rx is not None:
            metadata.update(self._classify_subtype(meta, seg, launch, flight))
        return MatchEvent(
            event_type=EventType.PASS,
            team=launch.kicker_team,
            start_s=launch.t_s,
            end_s=seg.grid[flight.end_index],
            confidence=confidence,
            period=seg.period,
            x=launch.x,
            y=launch.y,
            metadata=metadata,
        )

    def _classify_subtype(
        self, meta: MatchMeta, seg: KinematicSegment, launch: BallLaunch, flight: _Flight
    ) -> dict:
        cfg = self.config
        team = launch.kicker_team
        direction = meta.attack_direction(team, seg.period)
        x0, y0 = to_team_relative(launch.x, launch.y, direction)
        x1, y1 = to_team_relative(flight.rx, flight.ry, direction)
        length = math.hypot(x1 - x0, y1 - y0)
        forward = x1 > x0 + 2.0
        out = {"length_m": round(length, 1), "forward": forward}

        # cutback: byline strip, wide, travelling backward into the central box zone
        if (
            x0 >= cfg.cutback_min_x_rel
            and abs(y0) >= cfg.cutback_min_y_rel
            and x1 < x0 - 1.0
            and x1 >= cfg.cutback_target_x_min
            and abs(y1) <= cfg.cutback_target_y_rel
        ):
            out["subtype"] = "cutback"
            out["subtype_confidence"] = round(combine_evidence([
                soft_threshold(x0, cfg.cutback_min_x_rel, 2.0),
                soft_threshold(abs(y1), cfg.cutback_target_y_rel, 3.0, above=False),
            ]), 2)
            return out

        # cross: wide channel origin, into the box, gaining the centre
        if (
            abs(y0) >= cfg.cross_min_y_rel
            and x0 >= cfg.cross_min_x_rel
            and x1 >= cfg.cross_target_x_rel
            and abs(y1) <= cfg.cross_target_y_rel
            and abs(y1) <= abs(y0) - cfg.cross_centre_gain_m
        ):
            out["subtype"] = "cross"
            out["subtype_confidence"] = round(combine_evidence([
                soft_threshold(abs(y0), cfg.cross_min_y_rel, 2.5),
                soft_threshold(x1, cfg.cross_target_x_rel, 3.0),
            ]), 2)
            return out

        # through ball: forward, received beyond the opposing defensive line
        line_rel = self._opposing_line(meta, seg, launch.index, team)
        out["line_visible"] = line_rel is not None
        if (
            line_rel is not None
            and forward
            and x1 >= cfg.through_min_x_rel
            and x1 > line_rel + cfg.through_line_margin_m
        ):
            out["subtype"] = "through_ball"
            out["subtype_confidence"] = round(combine_evidence([
                soft_threshold(x1, line_rel + cfg.through_line_margin_m, 2.0),
                soft_threshold(length, 8.0, 3.0),
            ]), 2)
            return out

        if length < cfg.short_max_m:
            out["subtype"] = "short_pass"
        elif length >= cfg.long_min_m:
            out["subtype"] = "long_pass"
        else:
            out["subtype"] = "medium_pass"
        out["subtype_confidence"] = 0.7  # bucket membership is only as sharp as its edges
        return out

    def _opposing_line(
        self, meta: MatchMeta, seg: KinematicSegment, index: int, team: TeamSide
    ) -> Optional[float]:
        """Opposing defensive line at the launch, in the KICKER's team-relative x'.

        Second-deepest opposing outfielder (matching ``def_line_height``'s
        convention); None when fewer than ``min_line_players`` are visible —
        the subtype is then withheld rather than guessed (design doc §5.1).
        """
        opponent = team.opponent()
        opp_dir = meta.attack_direction(opponent, seg.period)
        xs = sorted(
            to_team_relative(x, y, opp_dir)[0]
            for tr, x, y in seg.visible_players(index, opponent)
            if tr.role is Role.OUTFIELD
        )
        if len(xs) < self.config.min_line_players:
            return None
        return 105.0 - xs[1]
