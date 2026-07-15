"""High press detector. Design doc §3.2."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import ClassVar, List, Optional, Tuple

from ..features import FeatureSeries, PossessionState
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
class HighPressConfig:
    """Thresholds; provenance in the design doc's sourcing appendix."""

    ball_deep_in_opponent_third_m: float = 70.0  # ball beyond presser's attacking 70m line
    min_def_line_height_m: float = 40.0
    min_defenders_within_15m: int = 3
    max_nearest_defender_dist_m: float = 6.0
    min_closing_speed_ms: float = 0.5
    min_duration_s: float = 3.0
    merge_gap_s: float = 2.0
    counter_press_window_s: float = 5.0  # within this of a turnover => tag as counter-press
    score_enter: float = 0.30
    score_exit: float = 0.15


@DetectorRegistry.register
class HighPressDetector(PatternDetector):
    """Out-of-possession team presses opponent build-up high up the pitch.

    Heuristic (per frame, for the out-of-possession team D) — soft-scored product:
    1. opponent possession AND ball within D's attacking ~35m
       (ball x' in D's frame >= ball_deep_in_opponent_third_m);
    2. D's line is high: def_line_height >= min_def_line_height_m
       (missing back line => mild discount, not a veto — "discount, don't drop");
    3. bodies committed: defenders_within_15m >= 3, nearest defender <= 6m;
    4. actively closing: press_closing_speed > min_closing_speed_ms.
    Episodes via hysteresis; confidence = margin x quality x completeness;
    intensity from pressers committed + closing speed + line height.

    metadata: "trigger" ("restart" if a DEAD spell occurred within 6s before the
    episode, else "open_play"), "is_counter_press" when the episode's median
    time-since-turnover < counter_press_window_s (crude v1 split; a dedicated
    counter-press detector is future work).

    Known failure modes: mid-block pressing traps leak in/out of this definition;
    close-up camera at goal kicks hides the line; the counter-press split by time
    alone is rough.
    """

    pattern_id: ClassVar[str] = "high_press"
    display_name: ClassVar[str] = "High press"
    required_features: ClassVar[Tuple[str, ...]] = (
        "possession.state",
        "possession.time_since_turnover_s",
        "ball.x",
        "team.def_line_height",
        "team.defenders_within_15m",
        "team.nearest_defender_dist",
        "team.press_closing_speed",
        "quality.score",
    )

    _N_COMPONENTS = 5

    def __init__(self, config: Optional[HighPressConfig] = None):
        self.config = config or HighPressConfig()

    def detect(self, series: FeatureSeries, meta: MatchMeta) -> List[PatternEvent]:
        events: List[PatternEvent] = []
        for seg in series.segments():
            if len(seg) < 2:
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
            factors = [
                soft_threshold(ball_x, cfg.ball_deep_in_opponent_third_m, 5.0),
                soft_threshold(tf.def_line_height, cfg.min_def_line_height_m, 5.0),
                soft_threshold(
                    None if tf.defenders_within_15m is None else float(tf.defenders_within_15m),
                    cfg.min_defenders_within_15m, 0.75,
                ),
                soft_threshold(
                    tf.nearest_defender_dist, cfg.max_nearest_defender_dist_m, 1.5, above=False
                ),
                soft_threshold(tf.press_closing_speed, cfg.min_closing_speed_ms, 0.5),
            ]
            score = 1.0
            for x in factors:
                score *= x
            scores.append(score)
            complete.append(
                None
                not in (
                    ball_x,
                    tf.def_line_height,
                    tf.defenders_within_15m,
                    tf.nearest_defender_dist,
                    tf.press_closing_speed,
                )
            )

        spans = episodes_from_scores(
            scores, times, cfg.score_enter, cfg.score_exit, cfg.min_duration_s, cfg.merge_gap_s
        )
        events = []
        for a, b in spans:
            ep_scores = [s for s in scores[a:b] if s is not None]
            confidence = episode_confidence(
                ep_scores,
                [f.quality.score for f in seg.frames[a:b]],
                complete[a:b],
                self._N_COMPONENTS,
            )
            events.append(
                PatternEvent(
                    pattern_id=self.pattern_id,
                    team=team,
                    start_s=times[a],
                    end_s=times[b - 1],
                    confidence=confidence,
                    intensity=self._intensity(seg, a, b, team),
                    period=period,
                    metadata=self._metadata(seg, a, b, team),
                )
            )
        return events

    def _intensity(self, seg: FeatureSeries, a: int, b: int, team: TeamSide) -> float:
        cfg = self.config
        parts: List[float] = []
        for f in seg.frames[a:b]:
            tf = f.team(team)
            terms = []
            if tf.defenders_within_15m is not None:
                terms.append(clamp01((tf.defenders_within_15m - 2) / 4.0))
            if tf.press_closing_speed is not None:
                terms.append(clamp01(tf.press_closing_speed / 2.5))
            if tf.def_line_height is not None:
                terms.append(clamp01((tf.def_line_height - cfg.min_def_line_height_m) / 20.0))
            if terms:
                parts.append(sum(terms) / len(terms))
        return clamp01(sum(parts) / len(parts)) if parts else 0.0

    def _metadata(self, seg: FeatureSeries, a: int, b: int, team: TeamSide) -> dict:
        cfg = self.config
        tst = [
            f.possession.time_since_turnover_s
            for f in seg.frames[a:b]
            if f.possession.time_since_turnover_s is not None
        ]
        is_counter_press = bool(tst) and median(tst) < cfg.counter_press_window_s
        start_t = seg.frames[a].timestamp_s
        recent_dead = any(
            f.possession.state is PossessionState.DEAD
            for f in seg.frames
            if start_t - 6.0 <= f.timestamp_s < start_t
        )
        return {
            "trigger": "restart" if recent_dead else "open_play",
            "is_counter_press": is_counter_press,
        }
