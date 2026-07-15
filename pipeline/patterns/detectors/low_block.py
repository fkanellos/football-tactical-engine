"""Low block detector. Design doc §3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

from ..features import FeatureSeries
from ..tracking import MatchMeta, TeamSide
from .base import (
    DetectorRegistry,
    MISSING_FEATURE_FACTOR,
    PatternDetector,
    PatternEvent,
    clamp01,
    episode_confidence,
    episodes_from_scores,
    soft_threshold,
)


@dataclass
class LowBlockConfig:
    """Thresholds; provenance in the design doc's sourcing appendix.

    Compactness is judged two ways and the MORE GENEROUS wins: (a) adaptively —
    hull area in this team's own bottom ``hull_area_quantile`` of out-of-possession
    frames this match (absolute cutoffs transfer badly between teams), and (b)
    against ``max_hull_area_m2`` as an absolute sanity fallback, which also covers
    the degenerate case where a team is compact ALL match and its own quantile
    stops meaning anything.
    """

    max_def_line_height_m: float = 28.0
    max_width_m: float = 40.0
    hull_area_quantile: float = 0.25
    max_hull_area_m2: float = 700.0
    min_players_behind_ball: int = 8
    min_visible_outfield: int = 9     # below this, don't trust n_behind_ball at all
    min_duration_s: float = 12.0      # a low block is a state, not a moment
    merge_gap_s: float = 4.0
    score_enter: float = 0.30
    score_exit: float = 0.15


@DetectorRegistry.register
class LowBlockDetector(PatternDetector):
    """Sustained deep, compact out-of-possession shape.

    Heuristic (per frame, for the out-of-possession team D) — soft-scored product:
    1. opponent possession, ball in D's half;
    2. deep: def_line_height <= max_def_line_height_m;
    3. compact: width <= max_width_m AND hull small (adaptive-or-absolute, see
       config docstring);
    4. numbers home: n_behind_ball >= min_players_behind_ball, trusted only when
       n_visible_outfield >= min_visible_outfield (else mild discount).
    Long min_duration is the point: short deep spells are just normal defending.

    intensity: depth + compactness + duration, normalised.
    metadata: "mean_line_height_m", "mean_width_m".

    Known failure modes: cannot distinguish a deliberate bus-park from a team
    pinned back and failing to escape — match-level recurrence is the v1
    disambiguator, surfaced via aggregation rather than solved here. Expected to
    be our most reliable detector (camera framing near the box shows most
    defenders).
    """

    pattern_id: ClassVar[str] = "low_block"
    display_name: ClassVar[str] = "Low block"
    required_features: ClassVar[Tuple[str, ...]] = (
        "possession.state",
        "ball.x",
        "team.def_line_height",
        "team.width",
        "team.hull_area",
        "team.n_behind_ball",
        "team.n_visible_outfield",
        "quality.score",
    )

    _N_COMPONENTS = 5

    def __init__(self, config: Optional[LowBlockConfig] = None):
        self.config = config or LowBlockConfig()

    def detect(self, series: FeatureSeries, meta: MatchMeta) -> List[PatternEvent]:
        events: List[PatternEvent] = []
        adaptive = {
            team: self._adaptive_hull_threshold(series, team)
            for team in (TeamSide.HOME, TeamSide.AWAY)
        }
        for seg in series.segments():
            if len(seg) < 2:
                continue
            for team in (TeamSide.HOME, TeamSide.AWAY):
                events.extend(self._detect_for_team(seg, meta, team, adaptive[team]))
        return events

    def _adaptive_hull_threshold(
        self, series: FeatureSeries, team: TeamSide
    ) -> Optional[float]:
        """This team's hull-area quantile over its out-of-possession frames,
        computed match-wide (not per segment) so short segments aren't self-judging."""
        hulls = sorted(
            f.team(team).hull_area
            for f in series.frames
            if f.team(team).hull_area is not None
            and f.possession.state.side() not in (None, team)
        )
        if not hulls:
            return None
        idx = int(self.config.hull_area_quantile * (len(hulls) - 1))
        return hulls[idx]

    def _detect_for_team(
        self,
        seg: FeatureSeries,
        meta: MatchMeta,
        team: TeamSide,
        adaptive_hull: Optional[float],
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

            hull_factor = self._hull_factor(tf.hull_area, adaptive_hull)
            if tf.n_visible_outfield >= cfg.min_visible_outfield and tf.n_behind_ball is not None:
                behind_factor = soft_threshold(
                    float(tf.n_behind_ball), cfg.min_players_behind_ball, 0.75
                )
                behind_ok = True
            else:
                behind_factor = MISSING_FEATURE_FACTOR
                behind_ok = False

            score = (
                soft_threshold(ball_x, 52.5, 5.0, above=False)
                * soft_threshold(tf.def_line_height, cfg.max_def_line_height_m, 4.0, above=False)
                * soft_threshold(tf.width, cfg.max_width_m, 4.0, above=False)
                * hull_factor
                * behind_factor
            )
            scores.append(score)
            complete.append(
                behind_ok
                and None not in (ball_x, tf.def_line_height, tf.width, tf.hull_area)
            )

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
            lines = [f.team(team).def_line_height for f in ep if f.team(team).def_line_height is not None]
            widths = [f.team(team).width for f in ep if f.team(team).width is not None]
            duration = times[b - 1] - times[a]
            intensity_terms = [clamp01(duration / 30.0)]
            if lines:
                intensity_terms.append(
                    clamp01((cfg.max_def_line_height_m - sum(lines) / len(lines)) / 15.0)
                )
            if widths:
                intensity_terms.append(
                    clamp01((cfg.max_width_m - sum(widths) / len(widths)) / 15.0)
                )
            events.append(
                PatternEvent(
                    pattern_id=self.pattern_id,
                    team=team,
                    start_s=times[a],
                    end_s=times[b - 1],
                    confidence=confidence,
                    intensity=clamp01(sum(intensity_terms) / len(intensity_terms)),
                    period=period,
                    metadata={
                        "mean_line_height_m": round(sum(lines) / len(lines), 1) if lines else None,
                        "mean_width_m": round(sum(widths) / len(widths), 1) if widths else None,
                    },
                )
            )
        return events

    def _hull_factor(self, hull: Optional[float], adaptive: Optional[float]) -> float:
        cfg = self.config
        if hull is None:
            return MISSING_FEATURE_FACTOR
        absolute = soft_threshold(hull, cfg.max_hull_area_m2, 100.0, above=False)
        if adaptive is None or adaptive <= 0:
            return absolute
        adaptive_score = soft_threshold(hull, adaptive, max(20.0, 0.15 * adaptive), above=False)
        return max(absolute, adaptive_score)
