"""Pre-match scouting report: schema, adapter into the Phase 5 rules engine.

Design: /docs/opponent-scouting-design.md §4. The report is the day-before
deliverable ("this opponent's tendencies across their last N matches"),
distinct from Phase 5's in-game recommendations. It reuses the SAME CounterRule
mechanism: ``scouting_profile_as_match_profile`` adapts the aggregated profile
into the ``MatchPatternProfile`` shape the RecommendationEngine already
consumes, gated so that only tendencies with enough effective evidence are
visible to rules at all. Nothing about the engine changes — which is the point:
one rules table, authored once, drives both the pre-match report and (later)
in-game evidence.

What genuinely does NOT carry over from the single-match flow (and is recorded
as such rather than papered over): per-event clip spans. A scouting aggregate's
evidence lives in multiple matches, so ``Evidence.clip_spans`` (single-match
(start, end, period) tuples) cannot represent it; the report carries per-match
attribution rows instead, and extending Evidence with match-qualified spans is
noted as a Phase 5 v2 change, not silently faked here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..patterns.runner import MatchPatternProfile, PatternAggregate
from ..patterns.tracking import TeamSide
from .aggregate import ScoutingConfig
from .models import TeamScoutingProfile

SCHEMA = "scouting-report/v1"


def scouting_profile_as_match_profile(
    profile: TeamScoutingProfile,
    as_side: TeamSide,
    config: Optional[ScoutingConfig] = None,
) -> MatchPatternProfile:
    """Adapt an aggregated scouting profile into the shape Phase 5 consumes.

    ``as_side`` labels which TeamSide the scouted team occupies in the pseudo
    profile; call the engine with ``for_team`` = the other side and
    scope=OPPONENT triggers resolve exactly as in the single-match flow.

    Gating: tendencies with ``n_eff < min_n_eff_for_rules`` are omitted
    entirely — a rule must not fire off one stale match. Rates/confidences are
    the weighted aggregates; ``count`` is the total raw event count across
    matches (evidence display, not a rate); weighted breakdowns are rounded
    back to ints because the aggregate schema (and rule ``where`` filters)
    count events. ``events`` stays empty — see module docstring.
    """
    cfg = config or ScoutingConfig()
    out = MatchPatternProfile(
        match_id=f"scouting:{profile.team_name}:{profile.as_of.isoformat()}",
        observed_minutes=profile.total_observed_minutes,
    )
    for pid in sorted(profile.tendencies):
        t = profile.tendencies[pid]
        if t.n_eff < cfg.min_n_eff_for_rules:
            continue
        if t.weighted_rate_per_90 <= 0.0:
            continue
        agg = PatternAggregate(pattern_id=pid, team=as_side)
        agg.count = sum(row.count for row in t.per_match)
        agg.rate_per_90_observed = t.weighted_rate_per_90
        agg.mean_confidence = t.mean_confidence
        agg.mean_intensity = t.mean_intensity
        agg.breakdowns = {
            key: {value: int(round(c)) for value, c in counts.items() if round(c) >= 1}
            for key, counts in t.breakdowns.items()
        }
        agg.breakdowns = {k: v for k, v in agg.breakdowns.items() if v}
        out.aggregates.append(agg)
    return out


@dataclass
class ScoutingReport:
    """The rendered pre-match deliverable (JSON; the UI formats it).

    ``tendencies`` rows carry the full uncertainty story (prevalence x/N with
    interval, n_eff, per-match attribution); ``recommendations`` are rendered
    CounterRule outputs from the adapted profile; ``caveats`` is not decoration
    — the builder fills it from the profile's actual evidence sufficiency, and
    a UI must render it with the report, not behind a disclosure.
    """

    team_name: str
    as_of: dt.date
    n_matches: int
    total_observed_minutes: float
    matches: List[Dict[str, Any]] = field(default_factory=list)
    tendencies: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "team": self.team_name,
            "as_of": self.as_of.isoformat(),
            "n_matches": self.n_matches,
            "total_observed_minutes": self.total_observed_minutes,
            "matches": self.matches,
            "tendencies": self.tendencies,
            "recommendations": self.recommendations,
            "caveats": self.caveats,
        }


def build_scouting_report(
    profile: TeamScoutingProfile,
    recommendations: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[ScoutingConfig] = None,
) -> ScoutingReport:
    """Assemble the report from an aggregated profile.

    ``recommendations`` are rendered rule outputs supplied by the caller (run
    the RecommendationEngine over ``scouting_profile_as_match_profile(...)``);
    they are an input here because the engine is Phase 5's component and this
    module does not reach around its interface.
    """
    cfg = config or ScoutingConfig()
    report = ScoutingReport(
        team_name=profile.team_name,
        as_of=profile.as_of,
        n_matches=profile.n_matches,
        total_observed_minutes=profile.total_observed_minutes,
        recommendations=list(recommendations or []),
    )
    report.matches = [
        {
            "match_id": m.match_id,
            "date": m.date.isoformat(),
            "age_days": m.age_days,
            "observed_minutes": m.observed_minutes,
            "weight": m.weight,
            "context_tags": list(m.context_tags),
        }
        for m in profile.matches
    ]
    for pid in sorted(profile.tendencies):
        t = profile.tendencies[pid]
        report.tendencies.append(
            {
                "pattern": pid,
                "rate_per_90": t.weighted_rate_per_90,
                "rate_spread": t.rate_spread,
                "prevalence": t.prevalence,
                "prevalence_ci": list(t.prevalence_ci),
                "seen_in": f"{sum(1 for r in t.per_match if r.present)} of {t.n_matches}",
                "n_eff": t.n_eff,
                "rules_eligible": t.n_eff >= cfg.min_n_eff_for_rules,
                "mean_confidence": t.mean_confidence,
                "mean_intensity": t.mean_intensity,
                "breakdowns": t.breakdowns,
                "per_match": [
                    {
                        "match_id": r.match_id,
                        "date": r.date.isoformat(),
                        "weight": r.weight,
                        "rate_per_90": r.rate_per_90,
                        "present": r.present,
                        "context_tags": list(r.context_tags),
                    }
                    for r in t.per_match
                ],
            }
        )

    if profile.n_matches == 0:
        report.caveats.append("No historical footage: this report is empty by design.")
    elif profile.n_matches <= 2:
        report.caveats.append(
            f"Only {profile.n_matches} match(es) of footage: tendencies are indicative "
            "at best; no rule triggers or live priors are derived at this sample size."
        )
    weak = [
        pid
        for pid, t in sorted(profile.tendencies.items())
        if 0 < t.n_eff < cfg.min_n_eff_for_rules
    ]
    if weak and profile.n_matches > 2:
        report.caveats.append(
            "Below the evidence bar (excluded from rules/priors): " + ", ".join(weak)
        )
    report.caveats.append(
        "Rates are per OBSERVED 90 (broadcast footage never shows the full match); "
        "confidence/intensity describe the pattern when it appeared, prevalence how "
        "often it appeared."
    )
    return report
