"""Counter-attack detector. Design doc §3.6."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import ClassVar, List, Optional, Tuple

from ..features import FeatureSeries
from ..tracking import MatchMeta, TeamSide
from .base import (
    DetectorRegistry,
    PatternDetector,
    PatternEvent,
    clamp01,
    episode_confidence,
    soft_threshold,
)


@dataclass
class CounterAttackConfig:
    """Thresholds; provenance in the design doc's sourcing appendix."""

    window_after_turnover_s: float = 10.0
    min_progression_m: float = 25.0       # ball gain toward opponent goal...
    final_third_x_rel_m: float = 70.0     # ...or reaching the final third
    min_mean_progression_speed_ms: float = 4.0
    min_runners: int = 2                  # n_forward_runners peak within the burst
    score_min: float = 0.30               # anchor fires iff soft-product >= this
    disorganised_margin: int = 1          # opponents-behind-ball below their median


@DetectorRegistry.register
class CounterAttackDetector(PatternDetector):
    """Fast transition toward goal after regaining possession.

    Heuristic — anchored on confirmed turnovers rather than sliding windows:
    1. for each turnover won by team A (possession flip surviving the state
       machine's persistence check; flips are backdated to when the new holder
       first appeared), open a window_after_turnover_s window;
    2. within it, find the ball's peak progression (team-relative x') and score:
       gain >= min_progression_m OR final third reached; mean progression speed
       to the peak >= min_mean_progression_speed_ms; peak n_forward_runners >=
       min_runners. Product of soft margins must clear score_min;
    3. supporting (raises intensity, not required): opponents goal-side of the
       ball at the peak below their own match median — the "defence not set"
       signal.

    intensity: territory gained / time taken + runner count (+ disorganisation).
    metadata: "turnover_location" (deep/middle/high in A's frame — a team that
    counters from deep turnovers vs from a high press are different scouting
    stories, Phase 5 keys on this), "n_runners", "reached_final_third".

    Known failure modes: inherits every weakness of possession inference (false
    turnovers => false counter windows — mitigated by the persistence rule);
    broadcast replays truncate exactly these sequences, so rates undercount
    (report per OBSERVED minute); counter vs fast build-up vs hopeful clearance
    is a spectrum the speed threshold slices arbitrarily (sequence-model
    territory, later). Restarts do not create anchors: the possession state
    machine re-establishes possession after DEAD spells without a turnover.
    """

    pattern_id: ClassVar[str] = "counter_attack"
    display_name: ClassVar[str] = "Counter-attack"
    required_features: ClassVar[Tuple[str, ...]] = (
        "possession.state",
        "possession.time_since_turnover_s",
        "possession.turnover_won_by",
        "ball.x",
        "team.n_forward_runners",
        "team.n_behind_ball",
        "quality.score",
    )

    _N_COMPONENTS = 3

    def __init__(self, config: Optional[CounterAttackConfig] = None):
        self.config = config or CounterAttackConfig()

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
        opponent = team.opponent()

        # opponent's usual rest-defence numbers, for the disorganisation signal
        opp_behind = [
            f.team(opponent).n_behind_ball
            for f in seg.frames
            if f.possession.state.side() is team
            and f.team(opponent).n_behind_ball is not None
        ]
        opp_behind_median = median(opp_behind) if opp_behind else None

        events: List[PatternEvent] = []
        last_end = -1e9
        for i0, anchor in enumerate(seg.frames):
            if anchor.possession.turnover_won_by is not team:
                continue
            t0 = anchor.timestamp_s
            if t0 < last_end:
                continue  # same-(pattern, team) events must not overlap

            x0 = anchor.ball.x_rel(team, meta, period)
            if x0 is None:
                continue

            # scan the window for peak ball progression and peak runners
            peak_x, peak_t, peak_i = x0, t0, i0
            runners_peak = 0
            for j in range(i0, len(seg.frames)):
                f = seg.frames[j]
                if f.timestamp_s > t0 + cfg.window_after_turnover_s:
                    break
                if f.possession.state.side() is not team:
                    if f.possession.state.side() is opponent:
                        break  # lost it again — the window ends with possession
                    continue
                runners_peak = max(runners_peak, f.team(team).n_forward_runners)
                x = f.ball.x_rel(team, meta, period)
                if x is not None and x > peak_x:
                    peak_x, peak_t, peak_i = x, f.timestamp_s, j

            gain = peak_x - x0
            if peak_t <= t0:
                continue
            speed = gain / (peak_t - t0)
            score = (
                max(
                    soft_threshold(gain, cfg.min_progression_m, 8.0),
                    soft_threshold(peak_x, cfg.final_third_x_rel_m, 5.0),
                )
                * soft_threshold(speed, cfg.min_mean_progression_speed_ms, 1.5)
                * soft_threshold(float(runners_peak), cfg.min_runners, 0.75)
            )
            if score < cfg.score_min:
                continue

            window = seg.frames[i0 : peak_i + 1]
            confidence = episode_confidence(
                [score], [f.quality.score for f in window], [True] * len(window),
                self._N_COMPONENTS,
            )
            disorganised = False
            if opp_behind_median is not None:
                at_peak = seg.frames[peak_i].team(opponent).n_behind_ball
                disorganised = (
                    at_peak is not None
                    and at_peak <= opp_behind_median - cfg.disorganised_margin
                )
            intensity_terms = [
                clamp01(gain / 45.0),
                clamp01(speed / 8.0),
                clamp01(runners_peak / 4.0),
            ]
            intensity = clamp01(
                sum(intensity_terms) / len(intensity_terms) + (0.1 if disorganised else 0.0)
            )
            if x0 < 35.0:
                turnover_location = "deep"
            elif x0 < cfg.final_third_x_rel_m:
                turnover_location = "middle"
            else:
                turnover_location = "high"
            events.append(
                PatternEvent(
                    pattern_id=self.pattern_id,
                    team=team,
                    start_s=t0,
                    end_s=peak_t,
                    confidence=confidence,
                    intensity=intensity,
                    period=period,
                    metadata={
                        "turnover_location": turnover_location,
                        "n_runners": runners_peak,
                        "reached_final_third": peak_x >= cfg.final_third_x_rel_m,
                        "gain_m": round(gain, 1),
                    },
                )
            )
            last_end = peak_t
        return events
