"""Opponent scouting: multi-match tendency profiles, live priors, pre-match reports.

Design: /docs/opponent-scouting-design.md. Position in the architecture:

    N x MatchPatternProfile (Phase 4, batch, historical footage)
        -> aggregate.build_team_profile -> TeamScoutingProfile
            -> report.build_scouting_report      (day-before deliverable)
            -> report.scouting_profile_as_...    (Phase 5 rules reuse)
            -> priors.PriorAdjuster              (live detection priors, Part A hook)

    models.py     MatchScoutingInput / MatchObservation / PatternTendency /
                  TeamScoutingProfile
    aggregate.py  recency+volume weighting, n_eff, Wilson intervals, tendency math
    priors.py     PatternPrior + PriorAdjuster (implements the streaming layer's
                  ConfidencePrior protocol: bounded log-odds shift, absolute floor)
    report.py     scouting-report/v1 schema + adapter into MatchPatternProfile

Everything degrades gracefully with 0/1/few historical matches: empty profiles
produce empty reports and neutral priors, never errors — scouting is strictly
additive to the live system.
"""

from .aggregate import (
    ScoutingConfig,
    build_team_profile,
    effective_sample_size,
    recency_weight,
    weighted_mean,
    weighted_std,
    wilson_interval,
)
from .models import (
    MatchObservation,
    MatchScoutingInput,
    MatchWeightRow,
    PatternTendency,
    PerMatchRow,
    TeamScoutingProfile,
    observations_from_profile,
)
from .priors import PatternPrior, PriorAdjuster, PriorConfig
from .report import (
    SCHEMA,
    ScoutingReport,
    build_scouting_report,
    scouting_profile_as_match_profile,
)

__all__ = [
    "MatchObservation",
    "MatchScoutingInput",
    "MatchWeightRow",
    "PatternPrior",
    "PatternTendency",
    "PerMatchRow",
    "PriorAdjuster",
    "PriorConfig",
    "SCHEMA",
    "ScoutingConfig",
    "ScoutingReport",
    "TeamScoutingProfile",
    "build_scouting_report",
    "build_team_profile",
    "effective_sample_size",
    "observations_from_profile",
    "recency_weight",
    "scouting_profile_as_match_profile",
    "weighted_mean",
    "weighted_std",
    "wilson_interval",
]
