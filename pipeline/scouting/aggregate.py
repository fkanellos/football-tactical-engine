"""Aggregation math: per-match profiles -> recency-weighted team tendencies.

Design: /docs/opponent-scouting-design.md §2. The choices in one paragraph:
matches are weighted by exponential recency decay (half-life, default 60 days)
times an observed-minutes volume factor (a 30-observed-minute broadcast tells
less than a full match); rates average over ALL matches including zeros;
presence is a per-match boolean (rate and confidence over thresholds) whose
weighted share is the tendency's ``prevalence``, with a Wilson interval at the
effective sample size supplying honest uncertainty; confidence/intensity
average only over matches where the pattern actually appeared. Everything is
pure stdlib arithmetic — deliberately: this module must be trivially auditable
because every scouting claim traces back to it.

Opponent-strength normalisation is deliberately ABSENT (design doc §2.3): with
3-10 input matches there are not enough degrees of freedom to fit an adjustment,
and we carry no reliable opponent-strength signal in the first place. Instead
the per-match rows (with analyst context tags) are preserved in every tendency
so heterogeneity is *visible* rather than silently corrected.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .models import (
    MatchObservation,
    MatchScoutingInput,
    MatchWeightRow,
    PatternTendency,
    PerMatchRow,
    TeamScoutingProfile,
    observations_from_profile,
)


@dataclass(frozen=True)
class ScoutingConfig:
    """Aggregation tunables.

    ``half_life_days=60``: last week's match keeps ~92% weight, one from 8
    months ago ~6% — matching the intuition in the design brief without a
    hard cutoff. ``presence_min_rate_per_90`` (with per-pattern overrides) plus
    ``presence_min_confidence`` define "the team showed this pattern in that
    match"; both must hold so one noisy low-confidence episode doesn't count
    as a tendency sighting.
    """

    half_life_days: float = 60.0
    reference_minutes: float = 60.0  # observed minutes for full volume weight
    presence_min_rate_per_90: float = 3.0
    presence_rate_overrides: Dict[str, float] = field(default_factory=dict)
    presence_min_confidence: float = 0.30
    wilson_z: float = 1.96
    min_n_eff_for_rules: float = 2.0  # gate for feeding rules/priors downstream


# ---------------------------------------------------------------------------
# Primitives (unit-tested individually)
# ---------------------------------------------------------------------------


def recency_weight(age_days: float, half_life_days: float) -> float:
    """Exponential decay: 0.5 ** (age / half_life). age<0 clamps to 1."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    return 0.5 ** (max(0.0, age_days) / half_life_days)


def volume_weight(observed_minutes: float, reference_minutes: float) -> float:
    """Linear up to the reference, capped at 1 (more footage than the
    reference is not more evidence about *tendency*, just about that match)."""
    if reference_minutes <= 0:
        raise ValueError("reference_minutes must be positive")
    return max(0.0, min(1.0, observed_minutes / reference_minutes))


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish's n_eff = (Σw)² / Σw²: equal weights -> n, one dominant -> ~1."""
    total = sum(weights)
    if total <= 0:
        return 0.0
    return total * total / sum(w * w for w in weights)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def weighted_std(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted population standard deviation."""
    total = sum(weights)
    if total <= 0:
        return 0.0
    mean = weighted_mean(values, weights)
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total
    return math.sqrt(var)


def wilson_interval(p_hat: float, n_eff: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval at an effective sample size.

    Chosen over Wald (degenerate at p̂∈{0,1}, exactly where scouting lives:
    "pressed in all 5") and over Jeffreys (needs an incomplete-beta inverse —
    numerical machinery this module doesn't otherwise need). n_eff=0 -> (0,1):
    no data, total uncertainty.
    """
    if n_eff <= 0:
        return (0.0, 1.0)
    p_hat = max(0.0, min(1.0, p_hat))
    z2 = z * z
    denom = 1.0 + z2 / n_eff
    centre = (p_hat + z2 / (2.0 * n_eff)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n_eff + z2 / (4.0 * n_eff * n_eff))
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def match_weight(obs_date: dt.date, observed_minutes: float, as_of: dt.date, cfg: ScoutingConfig) -> float:
    return recency_weight((as_of - obs_date).days, cfg.half_life_days) * volume_weight(
        observed_minutes, cfg.reference_minutes
    )


def is_present(obs: MatchObservation, pattern_id: str, cfg: ScoutingConfig) -> bool:
    min_rate = cfg.presence_rate_overrides.get(pattern_id, cfg.presence_min_rate_per_90)
    return obs.rate_per_90 >= min_rate and obs.mean_confidence >= cfg.presence_min_confidence


def aggregate_pattern(
    pattern_id: str,
    observations: Sequence[MatchObservation],
    weights: Sequence[float],
    cfg: ScoutingConfig,
) -> PatternTendency:
    """Weighted aggregation of one pattern across matches. See module doc."""
    if len(observations) != len(weights):
        raise ValueError("observations and weights must align")
    n_eff = effective_sample_size(weights)
    rates = [o.rate_per_90 for o in observations]
    present_flags = [is_present(o, pattern_id, cfg) for o in observations]

    prevalence = weighted_mean([1.0 if p else 0.0 for p in present_flags], weights)

    # confidence/intensity describe the pattern WHEN IT APPEARS: averaging in
    # zero-count matches would say "their press is weak" when the truth is
    # "they pressed rarely" — prevalence already carries the latter.
    appeared = [(o, w) for o, w in zip(observations, weights) if o.count > 0]
    conf = weighted_mean([o.mean_confidence for o, _ in appeared], [w for _, w in appeared])
    inten = weighted_mean([o.mean_intensity for o, _ in appeared], [w for _, w in appeared])

    breakdowns: Dict[str, Dict[str, float]] = {}
    for o, w in zip(observations, weights):
        for key, counts in o.breakdowns.items():
            slot = breakdowns.setdefault(key, {})
            for value, count in counts.items():
                slot[value] = slot.get(value, 0.0) + w * count

    per_match = tuple(
        PerMatchRow(
            match_id=o.match_id,
            date=o.date,
            weight=round(w, 4),
            count=o.count,
            rate_per_90=o.rate_per_90,
            present=p,
            context_tags=o.context_tags,
        )
        for o, w, p in sorted(
            zip(observations, weights, present_flags), key=lambda row: row[0].date, reverse=True
        )
    )

    return PatternTendency(
        pattern_id=pattern_id,
        n_matches=len(observations),
        n_eff=round(n_eff, 3),
        weighted_rate_per_90=round(weighted_mean(rates, weights), 3),
        rate_spread=round(weighted_std(rates, weights), 3),
        prevalence=round(prevalence, 3),
        prevalence_ci=tuple(round(x, 3) for x in wilson_interval(prevalence, n_eff, cfg.wilson_z)),
        mean_confidence=round(conf, 3),
        mean_intensity=round(inten, 3),
        breakdowns={k: {v: round(c, 3) for v, c in vals.items()} for k, vals in breakdowns.items()},
        per_match=per_match,
    )


def build_team_profile(
    team_name: str,
    matches: Sequence[MatchScoutingInput],
    pattern_ids: Sequence[str],
    as_of: dt.date,
    config: Optional[ScoutingConfig] = None,
) -> TeamScoutingProfile:
    """Aggregate N historical matches into a TeamScoutingProfile.

    Degrades gracefully by construction (design doc §5): with 0 matches the
    profile is empty (callers get no tendencies, priors stay neutral); with 1-2
    matches tendencies exist but their n_eff fails the downstream gates; the
    profile never *requires* a library.
    """
    cfg = config or ScoutingConfig()
    profile = TeamScoutingProfile(
        team_name=team_name,
        as_of=as_of,
        n_matches=len(matches),
        total_observed_minutes=round(sum(m.profile.observed_minutes for m in matches), 1),
        half_life_days=cfg.half_life_days,
    )
    if not matches:
        return profile

    ordered = sorted(matches, key=lambda m: m.date, reverse=True)
    weights = [
        match_weight(m.date, m.profile.observed_minutes, as_of, cfg) for m in ordered
    ]
    profile.matches = [
        MatchWeightRow(
            match_id=m.match_id,
            date=m.date,
            age_days=(as_of - m.date).days,
            observed_minutes=m.profile.observed_minutes,
            weight=round(w, 4),
            context_tags=m.context_tags,
        )
        for m, w in zip(ordered, weights)
    ]

    per_match_obs: List[Dict[str, MatchObservation]] = [
        observations_from_profile(m, pattern_ids) for m in ordered
    ]
    for pid in pattern_ids:
        profile.tendencies[pid] = aggregate_pattern(
            pid, [obs[pid] for obs in per_match_obs], weights, cfg
        )
    return profile
