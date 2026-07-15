"""Tests for the scouting -> live priors: the math AND the safeguards.

The safeguards are the design (opponent-scouting design doc §3): every test in
SafeguardsTest pins a property that keeps a prior from making the live system
see what it expects instead of what is happening.
"""

from __future__ import annotations

import datetime as dt
import unittest

from pipeline.patterns.streaming.detector import (
    EpisodeStateMachine,
    StreamingEpisodeConfig,
)
from pipeline.patterns.streaming.events import LiveEventKind as K
from pipeline.patterns.tracking import TeamSide
from pipeline.scouting.aggregate import build_team_profile
from pipeline.scouting.models import MatchScoutingInput
from pipeline.scouting.priors import PatternPrior, PriorAdjuster, PriorConfig
from pipeline.tests.test_scouting_aggregate import AS_OF, make_agg, match_input

DT = 0.2


def adjuster(p_team=0.85, n_eff=8.0, baseline=0.35, **cfg_overrides):
    cfg = PriorConfig(**cfg_overrides) if cfg_overrides else PriorConfig()
    return PriorAdjuster(
        PatternPrior("high_press", p_team=p_team, n_eff=n_eff),
        config=cfg,
        baseline_prevalence=baseline,
    )


def episode_config(**overrides):
    defaults = dict(
        enter_threshold=0.30, exit_threshold=0.15, min_duration_s=3.0,
        merge_gap_s=2.0, n_components=5, provisional_after_s=1.5,
    )
    defaults.update(overrides)
    return StreamingEpisodeConfig(**defaults)


class PatternPriorTest(unittest.TestCase):
    def test_jeffreys_shrinkage(self):
        # "pressed in both matches we have" must NOT become certainty
        t_prior = PatternPrior.from_tendency(
            type("T", (), {"pattern_id": "p", "prevalence": 1.0, "n_eff": 2.0})()
        )
        self.assertAlmostEqual(t_prior.p_team, 2.5 / 3.0, places=6)

    def test_empty_profile_is_uninformative(self):
        p = PatternPrior.from_tendency(
            type("T", (), {"pattern_id": "p", "prevalence": 0.0, "n_eff": 0.0})()
        )
        self.assertAlmostEqual(p.p_team, 0.5)


class AdjusterMathTest(unittest.TestCase):
    def test_thin_history_is_identity(self):
        a = adjuster(p_team=0.9, n_eff=1.0)  # below min_n_eff=2
        self.assertEqual(a.shift, 0.0)
        self.assertEqual(a.adjust_confidence(0.44), 0.44)
        self.assertEqual(a.provisional_after_scale(), 1.0)
        self.assertFalse(a.describe()["applied"])

    def test_positive_prior_raises_confidence(self):
        a = adjuster(p_team=0.85, n_eff=8.0)
        self.assertGreater(a.adjust_confidence(0.5), 0.5)
        # monotone: adjustment must never reorder raw confidences
        adj = [a.adjust_confidence(c) for c in (0.1, 0.3, 0.5, 0.7, 0.9)]
        self.assertEqual(adj, sorted(adj))

    def test_negative_prior_lowers_confidence(self):
        a = adjuster(p_team=0.05, n_eff=8.0)
        self.assertLess(a.adjust_confidence(0.5), 0.5)

    def test_cap_binds_for_extreme_priors(self):
        a = adjuster(p_team=0.99, n_eff=50.0, baseline=0.05)
        self.assertAlmostEqual(a.shift, 0.85)  # kappa, not the raw ~4 logits
        # capped shift means 0.5 raw can reach at most ~0.70
        self.assertAlmostEqual(a.adjust_confidence(0.5), 0.7006, places=3)

    def test_log_odds_shape_protects_low_evidence(self):
        # the same max shift that moves 0.5->0.70 barely moves 0.05
        a = adjuster(p_team=0.99, n_eff=50.0, baseline=0.05)
        self.assertLess(a.adjust_confidence(0.05), 0.12)

    def test_provisional_scale_bounds(self):
        self.assertAlmostEqual(
            adjuster(p_team=0.99, n_eff=50.0, baseline=0.05).provisional_after_scale(), 0.75
        )
        self.assertAlmostEqual(
            adjuster(p_team=0.01, n_eff=50.0, baseline=0.5).provisional_after_scale(), 1.25
        )
        self.assertAlmostEqual(adjuster(n_eff=0.0).provisional_after_scale(), 1.0)


class SafeguardsTest(unittest.TestCase):
    def test_floor_ignores_the_prior_entirely(self):
        strong = adjuster(p_team=0.99, n_eff=50.0, baseline=0.05)
        weak = adjuster(p_team=0.01, n_eff=50.0, baseline=0.5)
        self.assertFalse(strong.confirm_allowed(0.39))  # no prior buys confirmation
        self.assertTrue(weak.confirm_allowed(0.41))     # no prior vetoes real evidence

    def test_machine_never_confirms_below_floor_despite_strong_prior(self):
        m = EpisodeStateMachine(
            "high_press", TeamSide.HOME, episode_config(),
            prior=adjuster(p_team=0.99, n_eff=50.0, baseline=0.05),
        )
        updates = []
        # sustained pattern but poor data quality: raw confidence ~0.3 < floor 0.4
        for i in range(30):
            updates.extend(m.push(i * DT, 0.5, quality=0.35))
        updates.extend(m.flush())
        kinds = [u.kind for u in updates]
        self.assertIn(K.PROVISIONAL, kinds)      # it may still *suggest*
        self.assertNotIn(K.CONFIRMED, kinds)     # it may not *assert*
        self.assertNotIn(K.CLOSED, kinds)
        self.assertEqual(kinds[-1], K.RETRACTED)

    def test_events_carry_raw_and_adjusted_and_audit_block(self):
        m = EpisodeStateMachine(
            "high_press", TeamSide.HOME, episode_config(),
            prior=adjuster(p_team=0.85, n_eff=8.0),
        )
        updates = []
        for i in range(30):
            updates.extend(m.push(i * DT, 0.5, quality=0.9))
        provisional = updates[0]
        self.assertGreater(provisional.confidence, provisional.raw_confidence)
        self.assertTrue(provisional.prior["applied"])
        self.assertIn("shift", provisional.prior)

    def test_expected_pattern_alerts_earlier_unexpected_later(self):
        def first_alert_time(prior):
            m = EpisodeStateMachine("p", TeamSide.HOME, episode_config(), prior=prior)
            for i in range(60):
                for u in m.push(i * DT, 0.5, 0.9):
                    if u.kind is K.PROVISIONAL:
                        return i * DT
            return None

        neutral = first_alert_time(None)
        expected = first_alert_time(adjuster(p_team=0.99, n_eff=50.0, baseline=0.05))
        unexpected = first_alert_time(adjuster(p_team=0.01, n_eff=50.0, baseline=0.5))
        self.assertLess(expected, neutral)
        self.assertGreater(unexpected, neutral)
        # and the swing is bounded: 1.5s base within [0.75x, 1.25x]
        self.assertGreaterEqual(expected, 1.5 * 0.75 - DT)
        self.assertLessEqual(unexpected, 1.5 * 1.25 + DT)


class EndToEndPriorTest(unittest.TestCase):
    """Footage library -> tendency -> prior -> live machine, in one breath."""

    def test_press_heavy_opponent_produces_working_positive_prior(self):
        matches = [
            match_input(f"m{i}", days_ago=7 * (i + 1),
                        aggs=[make_agg("high_press", rate=11.0, conf=0.7)] if i != 3 else [])
            for i in range(8)  # pressed in 7 of 8
        ]
        profile = build_team_profile("FC Press", matches, ["high_press"], AS_OF)
        tendency = profile.tendencies["high_press"]
        self.assertGreater(tendency.n_eff, 4.0)

        adj = PriorAdjuster(PatternPrior.from_tendency(tendency))
        self.assertGreater(adj.shift, 0.0)
        self.assertGreater(adj.adjust_confidence(0.5), 0.5)
        self.assertLess(adj.provisional_after_scale(), 1.0)
        # and the audit block tells the whole story
        d = adj.describe()
        self.assertTrue(d["applied"])
        self.assertGreater(d["p_team"], 0.7)


if __name__ == "__main__":
    unittest.main()
