"""Flank overload detector. Design doc §3.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

from ..features import FeatureSeries, Lane
from ..tracking import MatchMeta, TeamSide
from .base import (
    DetectorRegistry,
    PatternDetector,
    PatternEvent,
    clamp01,
    episode_confidence,
    episodes_from_scores,
    soft_threshold,
)

_SIDE_LANES = {
    "left": (Lane.WIDE_LEFT, Lane.HALFSPACE_LEFT),
    "right": (Lane.WIDE_RIGHT, Lane.HALFSPACE_RIGHT),
}
_SIDE_SIGN = {"left": 1.0, "right": -1.0}


@dataclass
class FlankOverloadConfig:
    """Thresholds; provenance in the design doc's sourcing appendix (§9).

    The five-lane scheme is genuine coaching convention (half-spaces /
    Spielverlagerung / juego de posición) but no source defines lane widths in
    metres; equal 13.6m fifths are our simplification — the convention's
    pitch-marking anchors imply ~13.8/11/18.3/11/13.8m. Occupancy and
    superiority counts are engine-original (no published thresholds)."""

    min_ball_x_rel_m: float = 35.0        # middle/final third of the attacking team
    wide_lane_min_abs_y_m: float = 13.6   # |y'| beyond this = ball is in a wide lane
    min_attackers_ball_side: int = 3      # in ball-side wide + half-space lanes
    min_local_superiority: int = 1        # (teammates incl. holder - opponents) near ball
    min_centroid_shift_m: float = 5.0     # centroid_y' displaced toward the flank
    min_duration_s: float = 3.0
    merge_gap_s: float = 2.0
    score_enter: float = 0.30
    score_exit: float = 0.15


@DetectorRegistry.register
class FlankOverloadDetector(PatternDetector):
    """In-possession team commits numerical superiority to one flank.

    Heuristic (per frame, for the in-possession team A, scored separately per
    side) — soft-scored product:
    1. ball in a wide lane, in A's middle/final third;
    2. >= min_attackers_ball_side of A's players in the ball-side wide+half-space
       lanes AND local_superiority_10m >= min_local_superiority;
    3. shape leans over: centroid_y' shifted >= min_centroid_shift_m toward
       that flank;
    4. possession clearly A's (DEAD/CONTESTED frames score None — filters
       throw-in huddles / corner scrambles).

    The match-level aggregation BY SIDE is the tactically interesting output
    ("9 of 12 overloads on their left") — metadata["side"] is mandatory and is
    expressed from A's attacking perspective; Phase 5 templates mirror it for the
    defending coach ("their left / your right").

    Known failure modes: overload-to-isolate (load left to free the right winger)
    is invisible — the isolated far-side player is off-camera and the switch pass
    is an event we cannot see; v1 detects the overload, not its purpose. Ball-side
    counts are trustworthy (camera follows the ball); far-side lane counts are not.
    """

    pattern_id: ClassVar[str] = "flank_overload"
    display_name: ClassVar[str] = "Flank overload"
    required_features: ClassVar[Tuple[str, ...]] = (
        "possession.state",
        "ball.x",
        "ball.y",
        "team.lane_occupancy",
        "team.local_superiority_10m",
        "team.centroid_y",
        "quality.score",
    )

    _N_COMPONENTS = 5

    def __init__(self, config: Optional[FlankOverloadConfig] = None):
        self.config = config or FlankOverloadConfig()

    def detect(self, series: FeatureSeries, meta: MatchMeta) -> List[PatternEvent]:
        events: List[PatternEvent] = []
        for seg in series.segments():
            if len(seg) < 2:
                continue
            for team in (TeamSide.HOME, TeamSide.AWAY):
                for side in ("left", "right"):
                    events.extend(self._detect_side(seg, meta, team, side))
        return events

    def _detect_side(
        self, seg: FeatureSeries, meta: MatchMeta, team: TeamSide, side: str
    ) -> List[PatternEvent]:
        cfg = self.config
        period = seg.frames[0].period
        times = seg.times()
        sign = _SIDE_SIGN[side]
        lanes = _SIDE_LANES[side]
        scores: List[Optional[float]] = []
        complete: List[bool] = []
        attackers_col: List[Optional[int]] = []

        for f in seg.frames:
            if f.possession.state.side() is not team or not f.ball.valid:
                scores.append(None)
                complete.append(False)
                attackers_col.append(None)
                continue
            tf = f.team(team)
            ball_x = f.ball.x_rel(team, meta, period)
            ball_y = f.ball.y_rel(team, meta, period)
            attackers = sum(tf.lane_occupancy.get(lane, 0) for lane in lanes)
            attackers_col.append(attackers)
            centroid_shift = None if tf.centroid_y is None else sign * tf.centroid_y
            superiority = (
                None
                if tf.local_superiority_10m is None
                else float(tf.local_superiority_10m)
            )
            score = (
                soft_threshold(
                    None if ball_y is None else sign * ball_y,
                    cfg.wide_lane_min_abs_y_m, 2.0,
                )
                * soft_threshold(ball_x, cfg.min_ball_x_rel_m, 5.0)
                * soft_threshold(float(attackers), cfg.min_attackers_ball_side, 0.75)
                * soft_threshold(superiority, cfg.min_local_superiority, 0.75)
                * soft_threshold(centroid_shift, cfg.min_centroid_shift_m, 2.0)
            )
            scores.append(score)
            complete.append(None not in (ball_x, ball_y, superiority, centroid_shift))

        spans = episodes_from_scores(
            scores, times, cfg.score_enter, cfg.score_exit, cfg.min_duration_s, cfg.merge_gap_s
        )
        events = []
        for a, b in spans:
            ep = seg.frames[a:b]
            ep_scores = [s for s in scores[a:b] if s is not None]
            confidence = episode_confidence(
                ep_scores, [f.quality.score for f in ep], complete[a:b], self._N_COMPONENTS
            )
            att = [x for x in attackers_col[a:b] if x is not None]
            sup = [
                f.team(team).local_superiority_10m
                for f in ep
                if f.team(team).local_superiority_10m is not None
            ]
            shift = [
                _SIDE_SIGN[side] * f.team(team).centroid_y
                for f in ep
                if f.team(team).centroid_y is not None
            ]
            terms = []
            if att:
                terms.append(clamp01((max(att) - 2) / 3.0))
            if sup:
                terms.append(clamp01(max(sup) / 3.0))
            if shift:
                terms.append(clamp01((sum(shift) / len(shift)) / 12.0))
            events.append(
                PatternEvent(
                    pattern_id=self.pattern_id,
                    team=team,
                    start_s=times[a],
                    end_s=times[b - 1],
                    confidence=confidence,
                    intensity=clamp01(sum(terms) / len(terms)) if terms else 0.0,
                    period=period,
                    metadata={
                        "side": side,
                        "max_attackers_in_zone": max(att) if att else None,
                        "max_local_superiority": max(sup) if sup else None,
                    },
                )
            )
        return events
