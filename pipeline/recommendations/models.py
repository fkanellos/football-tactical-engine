"""Data model for the Phase 5 rules engine. Design doc §5.2 / §5.4."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..patterns.detectors.base import PatternEvent
from ..patterns.tracking import TeamSide


class TriggerScope(Enum):
    """Whose pattern the rule reacts to, relative to the team being advised."""

    OPPONENT = "opponent"  # counter THEIR tendency (the main use-case)
    SELF = "self"          # address OUR OWN detected habit


@dataclass(frozen=True)
class TriggerSpec:
    """Predicate over a MatchPatternProfile.

    Thresholds are PER-RULE on purpose: "how often is often enough to gameplan
    against" is analyst judgment and differs per pattern (design doc §5.2).
    ``where`` filters against aggregate ``breakdowns``, e.g. {"side": "left"} —
    sides are recorded from the ATTACKING team's perspective (see
    FlankOverloadDetector); templates handle the mirroring for the reader.
    """

    pattern_id: str
    scope: TriggerScope = TriggerScope.OPPONENT
    min_occurrences: int = 1
    min_rate_per_90: float = 0.0
    min_mean_confidence: float = 0.0
    min_mean_intensity: float = 0.0
    where: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationTemplate:
    """Analyst-authored content with {placeholder} slots.

    Available placeholders: any trigger-matched aggregate stat (count,
    rate_per_90, mean_intensity, ...), any ``where``-matched breakdown key, plus
    derived conveniences like {mirrored_side} so rendered text can say
    "their left / your right" and never make the coach do the mental flip.
    """

    headline: str
    rationale: str
    suggested_actions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterRule:
    """One authored rule: trigger -> recommendation, with provenance.

    ``priority`` is the analyst's judgment of how actionable the advice is when it
    fires, in [0, 1] — it is NOT computed from data. ``requires`` names squad
    capabilities the advice presupposes (checked against SquadProfile; unmet =>
    demoted with the gap stated, not dropped). Rules sharing a
    ``conflicts_group`` are mutually exclusive: highest score wins.
    """

    rule_id: str
    trigger: TriggerSpec
    recommendation: RecommendationTemplate
    priority: float = 0.5
    requires: Tuple[str, ...] = ()
    conflicts_group: Optional[str] = None
    tags: Tuple[str, ...] = ()
    author: str = "unknown"
    provenance: str = "starter"  # "starter" (illustrative) vs "analyst" (curated)


@dataclass
class SquadProfile:
    """Hand-maintained capability sheet for the advised team.

    The engine cannot infer your squad's traits from opponent tracking data —
    this is deliberately manual input (design doc §5.4). ``capabilities`` are
    free-form flags referenced by rules' ``requires``, e.g. "has_target_striker",
    "wingers_defend_well", "comfortable_playing_out".
    """

    team_name: str
    capabilities: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class Evidence:
    """The audit trail attached to every recommendation.

    ``clip_spans`` are the underlying PatternEvents' (start_s, end_s, period)
    spans so the UI can deep-link each recommendation straight to the video
    moments that justify it — the claim->clips link is what earns a coach's
    trust (design doc §5.3).
    """

    pattern_id: str
    team: TeamSide
    stats: Dict[str, Any] = field(default_factory=dict)
    clip_spans: Tuple[Tuple[float, float, int], ...] = ()
    events: Tuple[PatternEvent, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    """One rendered suggestion, ready for the UI."""

    rule_id: str
    for_team: TeamSide
    headline: str
    rationale: str
    suggested_actions: Tuple[str, ...]
    score: float               # priority x trigger strength (ranking key)
    evidence: Evidence
    unmet_requirements: Tuple[str, ...] = ()  # non-empty => demoted, gap stated
    tags: Tuple[str, ...] = ()
