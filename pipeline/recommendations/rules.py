"""Rule repository: loading, validation, and starter rules.

Rules live in YAML DATA, not Python logic, because the person curating them (the
football analyst) shouldn't need to touch code (design doc §5.1). The Python
starter rules below are ILLUSTRATIVE PLACEHOLDERS to make the format concrete —
they carry ``provenance="starter"`` and are expected to be rewritten or replaced
by analyst-authored YAML (see rules/starter_rules.yaml for the file format).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import CounterRule, RecommendationTemplate, TriggerSpec, TriggerScope


class RulesRepository:
    """Loads, validates, and serves the rule table."""

    def __init__(self, rules: List[CounterRule]):
        self.rules = rules

    @classmethod
    def from_yaml(cls, path: str) -> "RulesRepository":
        """Load rules from an analyst-authored YAML file (format: see
        rules/starter_rules.yaml). Calls ``validate`` before returning."""
        raise NotImplementedError

    @classmethod
    def starter(cls) -> "RulesRepository":
        """Repository holding only the illustrative starter rules below."""
        raise NotImplementedError

    def validate(self, known_pattern_ids: List[str]) -> List[str]:
        """Return human-readable problems (empty = valid):
        - trigger.pattern_id not in known_pattern_ids (catches typos and detector
          renames — pattern_id is part of the Phase 4/5 contract);
        - template placeholders that no aggregate stat / breakdown key can fill;
        - duplicate rule_ids; priorities outside [0, 1].
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Starter rules — placeholders demonstrating the schema, NOT tactical claims.
# The analyst curates the real table (design doc §5.1, §5.4).
# ---------------------------------------------------------------------------

STARTER_RULES: List[CounterRule] = [
    CounterRule(
        rule_id="double-up-vs-flank-overload",
        trigger=TriggerSpec(
            pattern_id="flank_overload",
            scope=TriggerScope.OPPONENT,
            min_rate_per_90=6.0,
            min_mean_confidence=0.5,
        ),
        recommendation=RecommendationTemplate(
            headline="Double up on their {side} flank (your {mirrored_side})",
            rationale=(
                "They overload the {side} flank ({count} episodes, {rate_per_90:.0f}/90 "
                "observed). Doubling your {mirrored_side}-sided winger onto their "
                "overlapping full-back blunts the 2v1."
            ),
            suggested_actions=(
                "Winger tracks the overlapping full-back; full-back stays on the touchline man",
                "Nearest CM shuttles across to screen the half-space cutback",
            ),
        ),
        priority=0.8,
        conflicts_group="flank-defence",
    ),
    CounterRule(
        rule_id="bypass-vs-high-press",
        trigger=TriggerSpec(
            pattern_id="high_press",
            scope=TriggerScope.OPPONENT,
            min_rate_per_90=8.0,
            min_mean_intensity=0.5,
        ),
        recommendation=RecommendationTemplate(
            headline="Prepare press-bypass patterns for build-up",
            rationale=(
                "They press high and hard ({count} episodes, mean intensity "
                "{mean_intensity:.2f}). Rehearsed exits beat improvisation against an "
                "organised press."
            ),
            suggested_actions=(
                "Direct ball to the target striker with midfield set for second balls",
                "Third-man combinations through the pressing line's blind side",
                "GK short-goal-kick exit patterns against their first pressing wave",
            ),
        ),
        priority=0.9,
        requires=("has_target_striker",),
    ),
    CounterRule(
        rule_id="stretch-vs-low-block",
        trigger=TriggerSpec(
            pattern_id="low_block",
            scope=TriggerScope.OPPONENT,
            min_rate_per_90=2.0,
            min_mean_confidence=0.6,
        ),
        recommendation=RecommendationTemplate(
            headline="Plan for a deep block: width, switches, and box presence",
            rationale=(
                "They defend in a sustained low block ({total_duration_s:.0f}s total "
                "observed). Central penetration will be scarce; stretch them laterally "
                "and attack the box on arrival."
            ),
            suggested_actions=(
                "Maximum width from wingers to widen the block before penetrating",
                "Fast switches of play to attack before the block slides across",
                "Runners arriving in the box on crosses rather than static presence",
            ),
        ),
        priority=0.7,
    ),
    CounterRule(
        rule_id="rest-defence-vs-counter",
        trigger=TriggerSpec(
            pattern_id="counter_attack",
            scope=TriggerScope.OPPONENT,
            min_rate_per_90=4.0,
            min_mean_intensity=0.5,
        ),
        recommendation=RecommendationTemplate(
            headline="Set a rest defence — they hurt teams in transition",
            rationale=(
                "{count} counter-attacks detected ({rate_per_90:.0f}/90 observed). "
                "Structure in possession must anticipate losing the ball."
            ),
            suggested_actions=(
                "Keep 3+1 behind the ball in possession (back three plus a screener)",
                "Immediate 5-second counter-press trigger on loss in their half",
                "Full-backs stagger — never both beyond the ball line at once",
            ),
        ),
        priority=0.85,
    ),
    CounterRule(
        rule_id="runs-in-behind-vs-high-line",
        trigger=TriggerSpec(
            pattern_id="offside_trap",
            scope=TriggerScope.OPPONENT,
            min_occurrences=3,
            min_mean_confidence=0.4,  # deliberately low bar: detector is weak, evidence
                                      # is corroborated by the line-tendency profile
        ),
        recommendation=RecommendationTemplate(
            headline="Attack the space behind their line",
            rationale=(
                "Their back line steps up aggressively ({count} step-up events). A high "
                "line trades space in behind for compactness — take the trade."
            ),
            suggested_actions=(
                "Striker starts on the last shoulder, times runs off the trap trigger",
                "Early through balls before their line resets",
                "Runners from midfield beyond the striker, harder to track for offside",
            ),
        ),
        priority=0.75,
        requires=("has_pacey_forward",),
    ),
]


def merge_rule_sources(
    starter: List[CounterRule],
    analyst_rules: Optional[List[CounterRule]] = None,
) -> List[CounterRule]:
    """Analyst rules override starter rules on rule_id collision; provenance kept."""
    raise NotImplementedError
