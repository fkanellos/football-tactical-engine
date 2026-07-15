"""Data model for opponent scouting profiles.

A scouting profile aggregates one team's per-match ``MatchPatternProfile``s
(Phase 4 output) across N historical matches into recency-weighted tendencies
with explicit uncertainty. Design: /docs/opponent-scouting-design.md.

The unit of aggregation is the ``MatchObservation`` — a flat, per-(match,
pattern) row extracted from a MatchPatternProfile. Matches where a pattern
never fired yield an explicit zero observation (rate 0, absent), because
"they never once parked the bus in 8 matches" is exactly as informative as
"they pressed in all 8"; silently skipping absent patterns would bias every
prevalence estimate upward.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..patterns.runner import MatchPatternProfile
from ..patterns.tracking import TeamSide


@dataclass(frozen=True)
class MatchScoutingInput:
    """One historical match of the scouted team, as fed to the aggregator.

    ``scouted_side`` says which side of the match profile is the team being
    scouted (they are home in some matches, away in others).
    ``context_tags`` are optional analyst-supplied labels ("vs_top_side",
    "away", "down_to_ten") — the honest v1 alternative to opponent-strength
    modelling (design doc §2.3): the system surfaces per-match values and the
    analyst brings the context.
    """

    match_id: str
    date: dt.date
    scouted_side: TeamSide
    profile: MatchPatternProfile
    context_tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchObservation:
    """One (match, pattern) row: what the scouted team showed in that match."""

    match_id: str
    date: dt.date
    observed_minutes: float
    count: int
    rate_per_90: float
    mean_confidence: float
    mean_intensity: float
    breakdowns: Dict[str, Dict[str, int]] = field(default_factory=dict)
    context_tags: Tuple[str, ...] = ()


def observations_from_profile(
    match: MatchScoutingInput, pattern_ids: Sequence[str]
) -> Dict[str, MatchObservation]:
    """Flatten one match profile into per-pattern observations.

    Every id in ``pattern_ids`` gets a row; patterns with no aggregate in the
    profile become explicit zero observations (see module docstring).
    """
    out: Dict[str, MatchObservation] = {}
    for pid in pattern_ids:
        agg = match.profile.get(pid, match.scouted_side)
        if agg is None:
            out[pid] = MatchObservation(
                match_id=match.match_id,
                date=match.date,
                observed_minutes=match.profile.observed_minutes,
                count=0,
                rate_per_90=0.0,
                mean_confidence=0.0,
                mean_intensity=0.0,
                context_tags=match.context_tags,
            )
        else:
            out[pid] = MatchObservation(
                match_id=match.match_id,
                date=match.date,
                observed_minutes=match.profile.observed_minutes,
                count=agg.count,
                rate_per_90=agg.rate_per_90_observed,
                mean_confidence=agg.mean_confidence,
                mean_intensity=agg.mean_intensity,
                breakdowns={k: dict(v) for k, v in agg.breakdowns.items()},
                context_tags=match.context_tags,
            )
    return out


@dataclass(frozen=True)
class PerMatchRow:
    """Transparency row kept in every tendency: the analyst sees the spread,
    not just the summary (design doc §2.3)."""

    match_id: str
    date: dt.date
    weight: float
    count: int
    rate_per_90: float
    present: bool
    context_tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PatternTendency:
    """One pattern's aggregated tendency for the scouted team.

    Uncertainty is carried explicitly and in two forms:
    - ``n_eff``: effective sample size after recency/volume weighting — 8
      well-observed recent matches and 8 stale fragments are not the same
      evidence, and this number is what downstream gates check;
    - ``prevalence_ci``: Wilson interval on the weighted share of matches where
      the pattern was present — "seen in 2 of 3" vs "8 of 10" get very
      different intervals even when the point estimates look similar.
    """

    pattern_id: str
    n_matches: int
    n_eff: float
    weighted_rate_per_90: float
    rate_spread: float  # weighted std of per-match rates: consistency signal
    prevalence: float
    prevalence_ci: Tuple[float, float]
    mean_confidence: float  # weighted, over matches where the pattern appeared
    mean_intensity: float
    breakdowns: Dict[str, Dict[str, float]] = field(default_factory=dict)  # weighted
    per_match: Tuple[PerMatchRow, ...] = ()


@dataclass(frozen=True)
class MatchWeightRow:
    """How much each input match counted, and why (provenance for the report)."""

    match_id: str
    date: dt.date
    age_days: int
    observed_minutes: float
    weight: float
    context_tags: Tuple[str, ...] = ()


@dataclass
class TeamScoutingProfile:
    """The aggregated tendency profile for one team — the scouting twin of a
    single-match MatchPatternProfile, and the input to both the pre-match
    report and the live-detection priors."""

    team_name: str
    as_of: dt.date
    n_matches: int
    total_observed_minutes: float
    matches: List[MatchWeightRow] = field(default_factory=list)
    tendencies: Dict[str, PatternTendency] = field(default_factory=dict)
    half_life_days: float = 0.0  # echo of the config used, for provenance

    def get(self, pattern_id: str) -> Optional[PatternTendency]:
        return self.tendencies.get(pattern_id)
