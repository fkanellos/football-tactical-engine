"""Recommendation engine: MatchPatternProfile + rules -> ranked Recommendations.

The engine never touches frames, features, or detectors — its entire world is the
MatchPatternProfile (design doc §4.2). Its job is mechanical on purpose: match
triggers, score, resolve conflicts, render with evidence. All football judgment
lives in the rules table and SquadProfile (design doc §5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..patterns.runner import MatchPatternProfile, PatternAggregate
from ..patterns.tracking import TeamSide
from .models import CounterRule, Evidence, Recommendation, SquadProfile, TriggerSpec
from .rules import RulesRepository


@dataclass
class EngineConfig:
    max_recommendations: int = 8
    min_score: float = 0.15          # below this, evidence is too marginal to show
    demotion_per_unmet_requirement: float = 0.5  # multiplier per missing capability
    max_clip_spans_per_recommendation: int = 6   # top spans by event confidence


class RecommendationEngine:
    """recommend(profile, for_team, squad) -> ranked list[Recommendation].

    Pipeline (design doc §5.3):
    1. MATCH   — resolve each rule's trigger scope to the right team's aggregates
                 (scope=OPPONENT => patterns exhibited by for_team's opponent),
                 apply ``where`` filters against breakdowns, check thresholds.
    2. SCORE   — score = rule.priority x trigger strength. Trigger strength uses
                 the same soft-margin idea as the detectors: how far above the
                 rule's thresholds the observed rate / confidence / intensity sit,
                 so marginal evidence yields low-ranked suggestions instead of
                 binary flapping at the threshold.
    3. FILTER  — check rule.requires against SquadProfile: unmet requirements
                 DEMOTE (multiply score) and are surfaced on the Recommendation
                 ("assumes a target striker"), never silently dropped — the
                 analyst decides whether the assumption holds.
    4. RESOLVE — within each conflicts_group keep the highest score; cap at
                 max_recommendations; drop scores below min_score.
    5. RENDER  — fill template placeholders from aggregate stats and matched
                 breakdown values (plus derived ones like {mirrored_side}), and
                 attach Evidence with the underlying event clip spans so the UI
                 can deep-link every claim to video.
    """

    def __init__(self, rules: RulesRepository, config: Optional[EngineConfig] = None):
        self.rules = rules
        self.config = config or EngineConfig()

    def recommend(
        self,
        profile: MatchPatternProfile,
        for_team: TeamSide,
        squad: Optional[SquadProfile] = None,
    ) -> List[Recommendation]:
        """Produce ranked, rendered recommendations for ``for_team``.

        ``profile`` will typically come from scouting the OPPONENT's recent
        matches, not the advised team's own — the caller owns that choice; the
        engine only cares which side of the profile each rule's scope points at.
        """
        raise NotImplementedError

    # -- internals (signatures pinned for the test suite) ---------------------

    def _trigger_strength(self, spec: TriggerSpec, agg: PatternAggregate) -> float:
        """Soft margin above the rule's thresholds, in [0, 1]. 0 = trigger not met."""
        raise NotImplementedError

    def _render(
        self, rule: CounterRule, agg: PatternAggregate, for_team: TeamSide
    ) -> Recommendation:
        """Fill placeholders; compute {mirrored_side}; build Evidence with the
        top clip spans by event confidence."""
        raise NotImplementedError

    def _resolve_conflicts(self, candidates: List[Recommendation]) -> List[Recommendation]:
        raise NotImplementedError

    @staticmethod
    def _placeholder_values(agg: PatternAggregate, where_matches: Dict[str, str]) -> Dict[str, object]:
        """Stats + breakdown values + derived conveniences available to templates.

        Must stay in sync with RulesRepository.validate's placeholder check —
        validation at load time is what lets analysts edit YAML fearlessly.
        """
        raise NotImplementedError
