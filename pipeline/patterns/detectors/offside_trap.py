"""Offside trap / line step-up detector. Design doc §3.5.

HONESTY NOTE — weakest detector by design intent, kept anyway. At 2-5 Hz a ~1 s
coordinated step-up spans 2-5 samples of a derivative of a noisy line estimate, and
the confirming outcome (flag/whistle) is invisible without event data. Expect high
false-positive rates on the discrete events; confidence is hard-capped to reflect
that. The robust deliverable this class ALSO produces is the line-management
TENDENCY profile (aggregates, not events) — which is what actually feeds a "play
early balls in behind" recommendation, and may end up being the primary output.

Implementation note vs the original sketch: "steps in unison" is checked via
positional line flatness through the step (a flat line that rises together stays
flat) rather than per-player velocity dispersion — per-player velocities at low Hz
with track switching are too noisy to be a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ..features import FeatureSeries
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


@dataclass
class OffsideTrapConfig:
    """Thresholds; provenance in the design doc's sourcing appendix."""

    min_line_step_speed_ms: float = 1.5   # def_line_velocity sustained upward
    min_line_gain_m: float = 2.0          # total height gained around the step
    gain_lookahead_s: float = 2.0         # window after episode start to measure gain
    max_line_flatness_m: float = 2.5      # "in unison" proxy (see module docstring)
    max_ball_dist_from_line_m: float = 35.0  # opponent threatening the line
    through_ball_speed_ms: float = 8.0    # ball surge that corroborates a trap
    through_ball_window_s: float = 1.5
    min_duration_s: float = 0.5
    merge_gap_s: float = 1.0
    score_enter: float = 0.30
    score_exit: float = 0.15
    confidence_cap: float = 0.7           # honesty cap — see module docstring


@DetectorRegistry.register
class OffsideTrapDetector(PatternDetector):
    """Back line steps up in unison to catch runners offside.

    Event heuristic (per frame, for the out-of-possession team D) — soft product:
    1. opponent possession, ball within max_ball_dist_from_line_m of D's line;
    2. rapid rise: def_line_velocity >= min_line_step_speed_ms;
    3. in unison: def_line_flatness <= max_line_flatness_m through the step.
    Post-filters per episode: the line must actually GAIN >= min_line_gain_m
    within gain_lookahead_s of the episode start. A ball surge toward D's goal
    (>= through_ball_speed_ms within through_ball_window_s of the step) is
    corroborating metadata, mildly boosting confidence — never required.

    ML outlook: first detector to replace with a learned sequence model once
    labelled clips exist; the heuristic mostly serves to bootstrap candidate
    clips for labelling.
    """

    pattern_id: ClassVar[str] = "offside_trap"
    display_name: ClassVar[str] = "Offside trap / line step-up"
    required_features: ClassVar[Tuple[str, ...]] = (
        "possession.state",
        "ball.x",
        "ball.vx",
        "team.def_line_height",
        "team.def_line_velocity",
        "team.def_line_flatness",
        "quality.score",
    )

    _N_COMPONENTS = 3

    def __init__(self, config: Optional[OffsideTrapConfig] = None):
        self.config = config or OffsideTrapConfig()

    def detect(self, series: FeatureSeries, meta: MatchMeta) -> List[PatternEvent]:
        events: List[PatternEvent] = []
        for seg in series.segments():
            if len(seg) < 3:
                continue
            for team in (TeamSide.HOME, TeamSide.AWAY):
                events.extend(self._detect_for_team(seg, meta, team))
        return events

    def _detect_for_team(
        self, seg: FeatureSeries, meta: MatchMeta, team: TeamSide
    ) -> List[PatternEvent]:
        cfg = self.config
        period = seg.frames[0].period
        times = seg.times()
        scores: List[Optional[float]] = []
        complete: List[bool] = []

        for f in seg.frames:
            poss = f.possession.state.side()
            if poss is None or poss is team or not f.ball.valid:
                scores.append(None)
                complete.append(False)
                continue
            tf = f.team(team)
            ball_x = f.ball.x_rel(team, meta, period)
            dist_to_line = (
                None
                if ball_x is None or tf.def_line_height is None
                else ball_x - tf.def_line_height
            )
            score = (
                soft_threshold(dist_to_line, cfg.max_ball_dist_from_line_m, 5.0, above=False)
                * soft_threshold(tf.def_line_velocity, cfg.min_line_step_speed_ms, 0.5)
                * soft_threshold(tf.def_line_flatness, cfg.max_line_flatness_m, 1.0, above=False)
            )
            scores.append(score)
            complete.append(
                None not in (dist_to_line, tf.def_line_velocity, tf.def_line_flatness)
            )

        spans = episodes_from_scores(
            scores, times, cfg.score_enter, cfg.score_exit, cfg.min_duration_s, cfg.merge_gap_s
        )
        events = []
        for a, b in spans:
            gain = self._line_gain(seg, a, team)
            if gain is None or gain < cfg.min_line_gain_m:
                continue
            ep = seg.frames[a:b]
            ep_scores = [s for s in scores[a:b] if s is not None]
            confidence = episode_confidence(
                ep_scores, [f.quality.score for f in ep], complete[a:b], self._N_COMPONENTS
            )
            through_ball = self._through_ball(seg, meta, a, b, team, period)
            if through_ball:
                confidence *= 1.15
            confidence = min(confidence, cfg.confidence_cap)
            vels = [
                f.team(team).def_line_velocity
                for f in ep
                if f.team(team).def_line_velocity is not None
            ]
            intensity = clamp01(
                (clamp01(gain / 6.0) + (clamp01(max(vels) / 4.0) if vels else 0.0)) / 2.0
            )
            events.append(
                PatternEvent(
                    pattern_id=self.pattern_id,
                    team=team,
                    start_s=times[a],
                    end_s=times[b - 1],
                    confidence=confidence,
                    intensity=intensity,
                    period=period,
                    metadata={"line_gain_m": round(gain, 1), "through_ball_detected": through_ball},
                )
            )
        return events

    def _line_gain(self, seg: FeatureSeries, a: int, team: TeamSide) -> Optional[float]:
        cfg = self.config
        t0 = seg.frames[a].timestamp_s
        start = seg.frames[a].team(team).def_line_height
        if start is None:
            return None
        peak = start
        for f in seg.frames[a:]:
            if f.timestamp_s > t0 + cfg.gain_lookahead_s:
                break
            h = f.team(team).def_line_height
            if h is not None and h > peak:
                peak = h
        return peak - start

    def _through_ball(
        self, seg: FeatureSeries, meta: MatchMeta, a: int, b: int, team: TeamSide, period: int
    ) -> bool:
        """Ball surging toward ``team``'s goal shortly after the step starts."""
        cfg = self.config
        t0 = seg.frames[a].timestamp_s
        t1 = seg.frames[b - 1].timestamp_s + cfg.through_ball_window_s
        opponent = team.opponent()
        for f in seg.frames[a:]:
            if f.timestamp_s > t1:
                break
            v = f.ball.vx_rel(opponent, meta, period)  # + = toward `team`'s goal
            if v is not None and v >= cfg.through_ball_speed_ms:
                return True
        return False

    # -- the robust aggregate output (design doc §3.5) --------------------------

    def line_tendency(
        self, series: FeatureSeries, meta: MatchMeta, team: TeamSide
    ) -> Dict[str, Any]:
        """Line-management tendency profile for ``team`` (robust at low Hz).

        Returns:
        - "mean_line_height_by_ball_dist_m": mean def_line_height binned by ball
          distance to the line (bins: 0-10, 10-20, 20-35, 35+ m), out-of-possession
          frames only;
        - "step_up_rate_per_90_observed": detected step-up events per observed 90;
        - "mean_line_height_m": overall out-of-possession mean.
        """
        bins: Dict[str, List[float]] = {"0-10": [], "10-20": [], "20-35": [], "35+": []}
        all_heights: List[float] = []
        for f in series.frames:
            poss = f.possession.state.side()
            if poss is None or poss is team or not f.ball.valid:
                continue
            tf = f.team(team)
            ball_x = f.ball.x_rel(team, meta, f.period)
            if tf.def_line_height is None or ball_x is None:
                continue
            all_heights.append(tf.def_line_height)
            d = ball_x - tf.def_line_height
            if d < 10:
                bins["0-10"].append(tf.def_line_height)
            elif d < 20:
                bins["10-20"].append(tf.def_line_height)
            elif d < 35:
                bins["20-35"].append(tf.def_line_height)
            else:
                bins["35+"].append(tf.def_line_height)

        observed = series.observed_seconds()
        n_steps = sum(1 for e in self.detect(series, meta) if e.team is team)
        return {
            "mean_line_height_by_ball_dist_m": {
                k: (round(sum(v) / len(v), 1) if v else None) for k, v in bins.items()
            },
            "mean_line_height_m": (
                round(sum(all_heights) / len(all_heights), 1) if all_heights else None
            ),
            "step_up_rate_per_90_observed": (
                round(n_steps * 5400.0 / observed, 2) if observed > 0 else 0.0
            ),
        }
