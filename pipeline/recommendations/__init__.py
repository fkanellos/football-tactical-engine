"""Phase 5: rule-based counter-strategy recommendations.

Design philosophy (design doc §5.1): the data supplies EVIDENCE, the analyst
supplies STRATEGY. Rules are authored content (YAML), never inferred; the engine's
job is to fire rules only when the match evidence supports their trigger, quantify
and cite that evidence (with deep-links to the underlying video clips), rank, and
resolve conflicts.

    models.py   CounterRule / TriggerSpec / Recommendation / SquadProfile
    rules.py    rule repository: YAML loading + validation + starter examples
    engine.py   trigger matching, scoring, conflict resolution, rendering
"""
