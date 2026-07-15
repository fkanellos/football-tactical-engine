"""Tier 1: boundary-crossing restart classification. Design doc §4.

Throw-in / corner / goal kick from boundary crossings + restart morphology;
kickoff from centre-spot geometry. The evidence model (§4.1): the trajectory
crossing is WEAK evidence near the line (position noise), the resumption location
is STRONG evidence (hypotheses are tens of metres apart) — so the crossing seeds a
candidate and the morphology classifies it. No hard in/out boolean exists anywhere;
marginal crossings yield lower-confidence events, and marginal crossings where play
just continues yield nothing.

Deliberate contract deviation (§4.3): this detector is NOT segment-local — the
broadcast usually cuts between the ball going out and the restart, so exits are
linked to resumptions across segment boundaries, with ``resumption_observed:
false`` and a confidence discount when the restart happened off-camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Tuple

from ..patterns.tracking import MatchMeta, TeamSide
from .base import EventDetector, EventDetectorRegistry, combine_evidence, soft_threshold
from .kinematics import BallLaunch, KinematicSegment, KinematicSeries, detect_launches
from .model import EventType, MatchEvent

GOAL_HALF_WIDTH_M = 3.66
SIX_YARD_DEPTH_M = 5.5
SIX_YARD_HALF_WIDTH_M = 9.16


@dataclass
class RestartConfig:
    """Thresholds; provenance in design doc §10 (rule geometry + noise margins)."""

    touch_radius_m: float = 2.5        # last-touch attribution radius
    touch_lookback_s: float = 1.0
    near_line_m: float = 1.5           # ball lost this close to a line => exit candidate
    min_outward_speed_ms: float = 0.3  # ...if moving toward the line at least this fast
    definite_out_m: float = 1.0        # 2 sigma: excursion that alone confirms out
    min_dead_s: float = 2.0            # quicker returns with no morphology are noise
    max_restart_gap_s: float = 90.0    # exit -> resumption linking horizon
    in_play_speed_ms: float = 3.0      # sustained in-pitch ball movement = play resumed
    in_play_sustain_s: float = 1.0
    throw_in_tolerance_m: float = 4.0  # resumption distance to the crossing point
    corner_tolerance_m: float = 3.0    # resumption distance to a corner point
    goal_kick_tolerance_m: float = 3.0 # resumption distance beyond the six-yard box
    unobserved_resumption_factor: float = 0.7  # discount when the restart was off-camera
    kickoff_spot_radius_m: float = 2.0
    kickoff_rest_s: float = 1.0
    kickoff_rest_speed_ms: float = 0.5
    kickoff_own_half_fraction: float = 0.75
    kickoff_own_half_margin_m: float = 2.0
    min_confidence: float = 0.2


@dataclass
class _Exit:
    """One candidate ball exit over a boundary line (internal)."""

    seg_idx: int
    index: int
    t_s: float
    axis: str            # "y" (touchline) | "x" (goal line)
    sign: int            # which of the two parallel lines
    cross_x: float
    cross_y: float
    excursion_m: float
    observed_out: bool   # False for near-line tracking loss
    returned_t: Optional[float] = None  # ball back inside (same segment)
    in_mouth: bool = False


@dataclass
class _Resumption:
    launch: BallLaunch
    period: int


@EventDetectorRegistry.register
class RestartDetector(EventDetector):
    """Throw-in / corner / goal kick / kickoff classification."""

    detector_id: ClassVar[str] = "restarts"
    display_name: ClassVar[str] = "Boundary restarts"
    event_types: ClassVar[Tuple[EventType, ...]] = (
        EventType.THROW_IN, EventType.CORNER, EventType.GOAL_KICK, EventType.KICKOFF,
    )
    stage: ClassVar[int] = 1

    def __init__(self, config: Optional[RestartConfig] = None):
        self.config = config or RestartConfig()

    def detect(
        self, kin: KinematicSeries, context: Sequence[MatchEvent] = ()
    ) -> List[MatchEvent]:
        events: List[MatchEvent] = []
        launches: List[Tuple[int, BallLaunch]] = []  # (seg_idx, launch), time-ordered
        for si, seg in enumerate(kin.segments):
            for launch in detect_launches(seg, kin.config):
                launches.append((si, launch))

        exits = self._find_exits(kin)
        consumed_until = -1e9
        for exit_ in exits:
            if exit_.t_s < consumed_until or exit_.in_mouth:
                continue  # in-mouth crossings belong to the shot detector (§5.2)
            event, end_t = self._classify_exit(kin, exit_, launches)
            if event is not None:
                events.append(event)
            if end_t is not None:
                consumed_until = end_t

        events.extend(self._detect_kickoffs(kin, launches))
        if self.config.min_confidence > 0:
            events = [e for e in events if e.confidence >= self.config.min_confidence]
        return sorted(events, key=lambda e: e.start_s)

    # -- exits -----------------------------------------------------------------

    def _find_exits(self, kin: KinematicSeries) -> List[_Exit]:
        cfg = self.config
        half_len = kin.meta.pitch_length_m / 2
        half_wid = kin.meta.pitch_width_m / 2
        noise = kin.config.boundary_noise_m
        exits: List[_Exit] = []
        for si, seg in enumerate(kin.segments):
            for i in range(1, len(seg)):
                p0, p1 = seg.ball_pos(i - 1), seg.ball_pos(i)
                if p0 is not None and p1 is not None:
                    exit_ = self._crossing_at(si, seg, i, p0, p1, half_len, half_wid, noise)
                    if exit_ is not None:
                        exits.append(exit_)
                        continue
                if p0 is not None and p1 is None:
                    exit_ = self._near_line_loss(si, seg, i - 1, p0, half_len, half_wid, noise)
                    if exit_ is not None:
                        exits.append(exit_)
        return sorted(exits, key=lambda e: e.t_s)

    def _crossing_at(
        self, si: int, seg: KinematicSegment, i: int,
        p0: Tuple[float, float], p1: Tuple[float, float],
        half_len: float, half_wid: float, noise: float,
    ) -> Optional[_Exit]:
        for axis, sign, line in (
            ("y", 1, half_wid), ("y", -1, half_wid), ("x", 1, half_len), ("x", -1, half_len),
        ):
            c0 = (p0[1] if axis == "y" else p0[0]) * sign
            c1 = (p1[1] if axis == "y" else p1[0]) * sign
            if not (c0 <= line < c1):
                continue
            frac = (line - c0) / (c1 - c0)
            cx = p0[0] + frac * (p1[0] - p0[0])
            cy = p0[1] + frac * (p1[1] - p0[1])
            # the crossing must be on the actual pitch edge, not the line extended
            if axis == "y" and abs(cx) > half_len + 1.0:
                continue
            if axis == "x" and abs(cy) > half_wid + 1.0:
                continue
            excursion = c1 - line
            returned_t: Optional[float] = None
            for j in range(i + 1, len(seg)):
                pj = seg.ball_pos(j)
                if pj is None:
                    break
                cj = (pj[1] if axis == "y" else pj[0]) * sign
                if cj <= line:
                    returned_t = seg.grid[j]
                    break
                excursion = max(excursion, cj - line)
            return _Exit(
                seg_idx=si, index=i, t_s=seg.grid[i], axis=axis, sign=sign,
                cross_x=cx, cross_y=cy, excursion_m=excursion, observed_out=True,
                returned_t=returned_t,
                in_mouth=(axis == "x" and abs(cy) <= GOAL_HALF_WIDTH_M + noise),
            )
        return None

    def _near_line_loss(
        self, si: int, seg: KinematicSegment, i: int, p: Tuple[float, float],
        half_len: float, half_wid: float, noise: float,
    ) -> Optional[_Exit]:
        """Ball lost by the tracker close to a line while moving outward."""
        cfg = self.config
        # require a real loss: at least the next two frames missing (or segment end)
        if i + 2 < len(seg) and (seg.ball_pos(i + 1) is not None or seg.ball_pos(i + 2) is not None):
            return None
        vx, vy = seg.ball_vx[i], seg.ball_vy[i]
        best: Optional[Tuple[str, int, float, float]] = None  # axis, sign, inside_dist, v_out
        for axis, sign, line in (
            ("y", 1, half_wid), ("y", -1, half_wid), ("x", 1, half_len), ("x", -1, half_len),
        ):
            c = (p[1] if axis == "y" else p[0]) * sign
            inside = line - c
            if inside < 0 or inside > cfg.near_line_m:
                continue
            v_out = ((vy if axis == "y" else vx) or 0.0) * sign
            if v_out < cfg.min_outward_speed_ms:
                continue
            if best is None or inside < best[2]:
                best = (axis, sign, inside, v_out)
        if best is None:
            return None
        axis, sign, _, _ = best
        # project the last position onto the line for the crossing estimate
        line = half_wid if axis == "y" else half_len
        cx = p[0] if axis == "y" else math.copysign(line, sign)
        cy = math.copysign(line, sign) if axis == "y" else p[1]
        return _Exit(
            seg_idx=si, index=i, t_s=seg.grid[i], axis=axis, sign=sign,
            cross_x=cx, cross_y=cy, excursion_m=0.0, observed_out=False,
            in_mouth=(axis == "x" and abs(cy) <= GOAL_HALF_WIDTH_M + noise),
        )

    # -- classification ----------------------------------------------------------

    def _classify_exit(
        self, kin: KinematicSeries, exit_: _Exit, launches: List[Tuple[int, BallLaunch]],
    ) -> Tuple[Optional[MatchEvent], Optional[float]]:
        cfg = self.config
        noise = kin.config.boundary_noise_m
        seg = kin.segments[exit_.seg_idx]

        # noise guard (§4.1): marginal excursion + quick return + no morphology = nothing
        if (
            exit_.observed_out
            and exit_.excursion_m < cfg.definite_out_m
            and exit_.returned_t is not None
            and exit_.returned_t - exit_.t_s < cfg.min_dead_s
        ):
            return None, None

        last_touch = self._last_touch(seg, exit_)
        resumption, in_play_t = self._find_resumption(kin, exit_, launches)

        exit_clarity = (
            soft_threshold(exit_.excursion_m, 0.0, noise) if exit_.observed_out else 0.45
        )
        quality = self._ball_quality_around(seg, exit_.index)

        if exit_.axis == "y":
            event = self._classify_throw_in(exit_, last_touch, resumption, exit_clarity, quality, seg)
        else:
            event = self._classify_goal_line(
                kin.meta, exit_, last_touch, resumption, exit_clarity, quality, seg
            )
        if event is None:
            return None, None

        if resumption is not None:
            end_t = resumption.launch.t_s
        elif in_play_t is not None:
            end_t = in_play_t
        else:
            end_t = exit_.t_s + cfg.min_dead_s
        event = MatchEvent(
            event_type=event.event_type, team=event.team,
            start_s=exit_.t_s, end_s=end_t,
            confidence=event.confidence, period=event.period,
            x=event.x, y=event.y, metadata=event.metadata,
        )
        return event, end_t

    def _classify_throw_in(
        self, exit_: _Exit, last_touch, resumption: Optional[_Resumption],
        exit_clarity: float, quality: float, seg: KinematicSegment,
    ) -> Optional[MatchEvent]:
        cfg = self.config
        touch_team = last_touch[0].team if last_touch else None
        expected = touch_team.opponent() if touch_team else None

        geometry = 0.5
        taker_team = None
        conflict = False
        if resumption is not None:
            launch = resumption.launch
            d = math.hypot(launch.x - exit_.cross_x, launch.y - exit_.cross_y)
            geometry = soft_threshold(d, cfg.throw_in_tolerance_m, 2.0, above=False)
            taker_team = launch.kicker_team
        team = expected
        attr = 0.55
        if taker_team is not None and expected is not None:
            if taker_team is expected:
                attr = 0.9
            else:  # the thrower IS ground truth; the touch attribution was wrong
                team, attr, conflict = taker_team, 0.3, True
        elif taker_team is not None:
            team, attr = taker_team, 0.7
        elif expected is not None:
            attr = 0.6

        confidence = combine_evidence([exit_clarity, geometry, attr], quality)
        if resumption is None:
            confidence *= cfg.unobserved_resumption_factor
        return MatchEvent(
            event_type=EventType.THROW_IN, team=team,
            start_s=exit_.t_s, end_s=exit_.t_s, confidence=confidence, period=seg.period,
            x=exit_.cross_x, y=exit_.cross_y,
            metadata=self._restart_metadata(exit_, last_touch, resumption, conflict),
        )

    def _classify_goal_line(
        self, meta: MatchMeta, exit_: _Exit, last_touch, resumption: Optional[_Resumption],
        exit_clarity: float, quality: float, seg: KinematicSegment,
    ) -> Optional[MatchEvent]:
        cfg = self.config
        half_len = meta.pitch_length_m / 2
        half_wid = meta.pitch_width_m / 2
        defending = self._defending_team(meta, seg.period, exit_.sign)
        attacking = defending.opponent()
        touch_team = last_touch[0].team if last_touch else None

        geom_corner, geom_gk = 0.5, 0.5
        if resumption is not None:
            lx, ly = resumption.launch.x, resumption.launch.y
            d_corner = min(
                math.hypot(lx - exit_.sign * half_len, ly - s * half_wid) for s in (1, -1)
            )
            geom_corner = soft_threshold(d_corner, cfg.corner_tolerance_m, 2.0, above=False)
            # distance outside that goal's six-yard box (0 inside)
            dx = max(0.0, (half_len - SIX_YARD_DEPTH_M) - lx * exit_.sign)
            dy = max(0.0, abs(ly) - SIX_YARD_HALF_WIDTH_M)
            d_box = math.hypot(dx, dy)
            geom_gk = soft_threshold(d_box, cfg.goal_kick_tolerance_m, 2.0, above=False)

        def attr_for(expected_touch: TeamSide) -> float:
            if touch_team is None:
                return 0.55
            return 0.9 if touch_team is expected_touch else 0.3

        attr_corner = attr_for(defending)   # corner <= defender touched last
        attr_gk = attr_for(attacking)       # goal kick <= attacker touched last
        score_corner = geom_corner * (0.4 + 0.6 * attr_corner)
        score_gk = geom_gk * (0.4 + 0.6 * attr_gk)

        if score_corner >= score_gk:
            event_type, team = EventType.CORNER, attacking
            geometry, attr = geom_corner, attr_corner
        else:
            event_type, team = EventType.GOAL_KICK, defending
            geometry, attr = geom_gk, attr_gk
        conflict = attr < 0.5 and touch_team is not None

        confidence = combine_evidence([exit_clarity, geometry, attr], quality)
        if resumption is None:
            confidence *= cfg.unobserved_resumption_factor
        return MatchEvent(
            event_type=event_type, team=team,
            start_s=exit_.t_s, end_s=exit_.t_s, confidence=confidence, period=seg.period,
            x=exit_.cross_x, y=exit_.cross_y,
            metadata=self._restart_metadata(exit_, last_touch, resumption, conflict),
        )

    # -- kickoffs ------------------------------------------------------------------

    def _detect_kickoffs(
        self, kin: KinematicSeries, launches: List[Tuple[int, BallLaunch]],
    ) -> List[MatchEvent]:
        cfg = self.config
        events: List[MatchEvent] = []
        for si, seg in enumerate(kin.segments):
            seg_launches = [l for s, l in launches if s == si]
            i = 0
            while i < len(seg):
                if not self._at_centre_rest(seg, i):
                    i += 1
                    continue
                j = i
                while j + 1 < len(seg) and self._at_centre_rest(seg, j + 1):
                    j += 1
                rest_len = seg.grid[j] - seg.grid[i]
                launch = next(
                    (l for l in seg_launches
                     if seg.grid[j] - 0.2 <= l.t_s <= seg.grid[j] + 3.0
                     and math.hypot(l.x, l.y) <= cfg.kickoff_spot_radius_m + 2.0),
                    None,
                )
                if launch is not None and rest_len >= cfg.kickoff_rest_s - 1e-9:
                    rest_factor = soft_threshold(rest_len, cfg.kickoff_rest_s, 0.5)
                    sep_factor = self._half_separation(kin.meta, seg, j)
                    kicker_factor = 0.9 if launch.kicker_team is not None else 0.5
                    confidence = combine_evidence([rest_factor, sep_factor, kicker_factor])
                    events.append(
                        MatchEvent(
                            event_type=EventType.KICKOFF,
                            team=launch.kicker_team,
                            start_s=seg.grid[i], end_s=launch.t_s,
                            confidence=confidence, period=seg.period,
                            x=0.0, y=0.0,
                            metadata={
                                "rest_s": round(rest_len, 1),
                                "taker_track_id": launch.kicker_track_id,
                                "resumption_observed": True,
                            },
                        )
                    )
                i = j + 1
        return events

    def _at_centre_rest(self, seg: KinematicSegment, i: int) -> bool:
        cfg = self.config
        pos = seg.ball_pos(i)
        if pos is None or math.hypot(pos[0], pos[1]) > cfg.kickoff_spot_radius_m:
            return False
        speed = seg.ball_speed(i)
        return speed is None or speed < cfg.kickoff_rest_speed_ms

    def _half_separation(self, meta: MatchMeta, seg: KinematicSegment, i: int) -> float:
        cfg = self.config
        fractions = []
        for team in (TeamSide.HOME, TeamSide.AWAY):
            direction = meta.attack_direction(team, seg.period)
            visible = seg.visible_players(i, team)
            outfield = [(x, y) for tr, x, y in visible if tr.role.value == "outfield"]
            if len(outfield) < 4:
                continue
            own_half = sum(
                1 for x, _ in outfield
                if direction * x + meta.pitch_length_m / 2
                <= meta.pitch_length_m / 2 + cfg.kickoff_own_half_margin_m
            )
            fractions.append(own_half / len(outfield))
        if not fractions:
            return 0.6  # separation unobservable: mild discount, don't drop
        return soft_threshold(min(fractions), cfg.kickoff_own_half_fraction, 0.08)

    # -- shared helpers ---------------------------------------------------------------

    def _last_touch(self, seg: KinematicSegment, exit_: _Exit):
        cfg = self.config
        lookback = max(1, int(round(cfg.touch_lookback_s / (seg.grid[1] - seg.grid[0])))) \
            if len(seg) > 1 else 1
        best = None
        for j in range(max(0, exit_.index - lookback), exit_.index + 1):
            pos = seg.ball_pos(j)
            if pos is None:
                continue
            near = seg.nearest_player(j, pos[0], pos[1], max_dist_m=cfg.touch_radius_m)
            if near is not None and (best is None or near[1] < best[1]):
                best = near
        return best  # (PlayerTrack, dist) | None

    def _find_resumption(
        self, kin: KinematicSeries, exit_: _Exit, launches: List[Tuple[int, BallLaunch]],
    ) -> Tuple[Optional[_Resumption], Optional[float]]:
        """First launch after the exit, unless in-play ball movement precedes it
        (=> the restart itself was cut out; classification loses its morphology)."""
        cfg = self.config
        candidate: Optional[Tuple[int, BallLaunch]] = None
        for si, launch in launches:
            if exit_.t_s + 0.2 < launch.t_s <= exit_.t_s + cfg.max_restart_gap_s:
                candidate = (si, launch)
                break
        if candidate is None:
            return None, None
        in_play_t = self._in_play_before(kin, exit_, candidate[1].t_s)
        if in_play_t is not None:
            return None, in_play_t
        return _Resumption(candidate[1], kin.segments[candidate[0]].period), None

    def _in_play_before(
        self, kin: KinematicSeries, exit_: _Exit, until_t: float
    ) -> Optional[float]:
        cfg = self.config
        half_len = kin.meta.pitch_length_m / 2
        half_wid = kin.meta.pitch_width_m / 2
        need = max(1, int(round(cfg.in_play_sustain_s / kin.frame_interval_s())))
        run = 0
        for seg in kin.segments:
            for i, t in enumerate(seg.grid):
                if t <= exit_.t_s + 0.2 or t >= until_t - 1e-9:
                    continue
                pos = seg.ball_pos(i)
                speed = seg.ball_speed(i)
                inside = pos is not None and abs(pos[0]) <= half_len and abs(pos[1]) <= half_wid
                if inside and speed is not None and speed >= cfg.in_play_speed_ms:
                    run += 1
                    if run >= need:
                        return t
                else:
                    run = 0
        return None

    def _ball_quality_around(self, seg: KinematicSegment, index: int) -> float:
        lo, hi = max(0, index - 5), min(len(seg), index + 6)
        valid = sum(1 for j in range(lo, hi) if seg.ball_pos(j) is not None)
        return 0.5 + 0.5 * (valid / max(1, hi - lo))

    @staticmethod
    def _defending_team(meta: MatchMeta, period: int, x_sign: int) -> TeamSide:
        for team in (TeamSide.HOME, TeamSide.AWAY):
            if meta.attack_direction(team, period) * x_sign < 0:
                return team
        raise AssertionError("unreachable")

    def _restart_metadata(self, exit_: _Exit, last_touch, resumption, conflict: bool):
        return {
            "crossing_xy": [round(exit_.cross_x, 1), round(exit_.cross_y, 1)],
            "excursion_m": round(exit_.excursion_m, 2),
            "exit_observed": exit_.observed_out,
            "resumption_observed": resumption is not None,
            "resumption_xy": (
                [round(resumption.launch.x, 1), round(resumption.launch.y, 1)]
                if resumption else None
            ),
            "last_touch_team": last_touch[0].team.value if last_touch and last_touch[0].team else None,
            "last_touch_track_id": last_touch[0].track_id if last_touch else None,
            "taker_track_id": resumption.launch.kicker_track_id if resumption else None,
            "attribution_conflict": conflict,
        }
