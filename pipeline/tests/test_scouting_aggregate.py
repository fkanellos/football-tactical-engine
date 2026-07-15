"""Tests for the scouting aggregation math (weights, n_eff, Wilson, tendencies)."""

from __future__ import annotations

import datetime as dt
import unittest

from pipeline.patterns.runner import MatchPatternProfile, PatternAggregate
from pipeline.patterns.tracking import TeamSide
from pipeline.scouting.aggregate import (
    ScoutingConfig,
    build_team_profile,
    effective_sample_size,
    recency_weight,
    volume_weight,
    weighted_mean,
    weighted_std,
    wilson_interval,
)
from pipeline.scouting.models import MatchScoutingInput, observations_from_profile

AS_OF = dt.date(2026, 7, 15)


def make_profile(match_id, observed_minutes=60.0, aggs=()):
    profile = MatchPatternProfile(match_id=match_id, observed_minutes=observed_minutes)
    profile.aggregates = list(aggs)
    return profile


def make_agg(pid, team=TeamSide.HOME, count=6, rate=9.0, conf=0.6, inten=0.5, breakdowns=None):
    agg = PatternAggregate(pattern_id=pid, team=team, count=count)
    agg.rate_per_90_observed = rate
    agg.mean_confidence = conf
    agg.mean_intensity = inten
    agg.breakdowns = breakdowns or {}
    return agg


def match_input(match_id, days_ago, aggs=(), observed_minutes=60.0, tags=()):
    return MatchScoutingInput(
        match_id=match_id,
        date=AS_OF - dt.timedelta(days=days_ago),
        scouted_side=TeamSide.HOME,
        profile=make_profile(match_id, observed_minutes, aggs),
        context_tags=tuple(tags),
    )


class PrimitivesTest(unittest.TestCase):
    def test_recency_weight(self):
        self.assertEqual(recency_weight(0, 60), 1.0)
        self.assertAlmostEqual(recency_weight(60, 60), 0.5)
        self.assertAlmostEqual(recency_weight(240, 60), 0.0625)  # ~8 months
        self.assertAlmostEqual(recency_weight(7, 60), 0.922, places=3)  # last week
        self.assertEqual(recency_weight(-3, 60), 1.0)  # future-dated clamps

    def test_volume_weight(self):
        self.assertEqual(volume_weight(30, 60), 0.5)
        self.assertEqual(volume_weight(90, 60), 1.0)
        self.assertEqual(volume_weight(0, 60), 0.0)

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size([1.0] * 8), 8.0)
        self.assertAlmostEqual(effective_sample_size([1.0, 0.001, 0.001]), 1.004, places=3)
        self.assertEqual(effective_sample_size([]), 0.0)

    def test_weighted_mean_and_std(self):
        self.assertAlmostEqual(weighted_mean([10, 0], [1, 1]), 5.0)
        self.assertAlmostEqual(weighted_mean([10, 0], [3, 1]), 7.5)
        self.assertAlmostEqual(weighted_std([5, 5, 5], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(weighted_std([0, 10], [1, 1]), 5.0)

    def test_wilson_interval_known_values(self):
        lo, hi = wilson_interval(0.5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        # the case Wald botches and scouting hits constantly: seen in ALL matches
        lo, hi = wilson_interval(1.0, 5)
        self.assertAlmostEqual(lo, 0.5655, places=3)
        self.assertEqual(hi, 1.0)
        # "2 of 3" vs "8 of 10": similar point estimate, very different certainty
        lo3, _ = wilson_interval(2 / 3, 3)
        lo10, _ = wilson_interval(0.8, 10)
        self.assertLess(lo3, 0.30)
        self.assertGreater(lo10, 0.45)

    def test_wilson_no_data(self):
        self.assertEqual(wilson_interval(0.5, 0), (0.0, 1.0))


class ObservationsTest(unittest.TestCase):
    def test_absent_pattern_yields_explicit_zero(self):
        m = match_input("m1", 3, aggs=[make_agg("high_press")])
        obs = observations_from_profile(m, ["high_press", "low_block"])
        self.assertEqual(obs["high_press"].count, 6)
        self.assertEqual(obs["low_block"].count, 0)
        self.assertEqual(obs["low_block"].rate_per_90, 0.0)

    def test_scouted_side_selects_the_right_team(self):
        aggs = [make_agg("high_press", team=TeamSide.AWAY, rate=12.0)]
        m = MatchScoutingInput(
            match_id="m1", date=AS_OF, scouted_side=TeamSide.AWAY,
            profile=make_profile("m1", aggs=aggs),
        )
        obs = observations_from_profile(m, ["high_press"])
        self.assertEqual(obs["high_press"].rate_per_90, 12.0)
        # and the HOME side of the same profile would see nothing
        m_home = MatchScoutingInput(
            match_id="m1", date=AS_OF, scouted_side=TeamSide.HOME,
            profile=m.profile,
        )
        self.assertEqual(observations_from_profile(m_home, ["high_press"])["high_press"].count, 0)


class BuildTeamProfileTest(unittest.TestCase):
    PATTERNS = ["high_press", "low_block"]

    def test_empty_library_degrades_gracefully(self):
        profile = build_team_profile("FC Test", [], self.PATTERNS, AS_OF)
        self.assertEqual(profile.n_matches, 0)
        self.assertEqual(profile.tendencies, {})

    def test_recency_downweights_old_matches(self):
        # recent match: no pressing; 8-month-old match: heavy pressing
        matches = [
            match_input("recent", 3, aggs=[]),
            match_input("old", 240, aggs=[make_agg("high_press", rate=15.0)]),
        ]
        profile = build_team_profile("FC Test", matches, self.PATTERNS, AS_OF)
        t = profile.tendencies["high_press"]
        # equal weighting would give 7.5; recency must pull it near zero
        self.assertLess(t.weighted_rate_per_90, 1.5)
        self.assertLess(t.prevalence, 0.1)
        # both matches are still visible to the analyst, newest first
        self.assertEqual([r.match_id for r in t.per_match], ["recent", "old"])

    def test_short_footage_downweights(self):
        matches = [
            match_input("full", 2, aggs=[make_agg("high_press", rate=2.0)], observed_minutes=70),
            match_input("clip", 2, aggs=[make_agg("high_press", rate=20.0)], observed_minutes=15),
        ]
        profile = build_team_profile("FC Test", matches, self.PATTERNS, AS_OF)
        t = profile.tendencies["high_press"]
        # the 15-minute fragment must not dominate: weight 0.25 vs 1.0
        self.assertLess(t.weighted_rate_per_90, 7.0)
        self.assertLess(t.n_eff, 1.5)

    def test_prevalence_and_interval_reflect_sample_size(self):
        def library(n, rate=10.0):
            return [
                match_input(f"m{i}", days_ago=7 * i, aggs=[make_agg("high_press", rate=rate)])
                for i in range(n)
            ]

        small = build_team_profile("FC", library(3), self.PATTERNS, AS_OF).tendencies["high_press"]
        large = build_team_profile("FC", library(9), self.PATTERNS, AS_OF).tendencies["high_press"]
        self.assertEqual(small.prevalence, 1.0)
        self.assertEqual(large.prevalence, 1.0)
        # same point estimate, honesty lives in the lower bound
        self.assertLess(small.prevalence_ci[0], large.prevalence_ci[0])

    def test_confidence_only_over_matches_where_pattern_appeared(self):
        matches = [
            match_input("a", 1, aggs=[make_agg("high_press", rate=10.0, conf=0.8)]),
            match_input("b", 2, aggs=[]),  # never pressed
        ]
        t = build_team_profile("FC", matches, self.PATTERNS, AS_OF).tendencies["high_press"]
        self.assertAlmostEqual(t.mean_confidence, 0.8, places=3)
        self.assertLess(t.prevalence, 1.0)  # ...but prevalence says "not always"

    def test_presence_requires_confidence_not_just_rate(self):
        # a high rate of junk-confidence detections must not count as a sighting
        matches = [match_input("a", 1, aggs=[make_agg("high_press", rate=12.0, conf=0.1)])]
        t = build_team_profile("FC", matches, self.PATTERNS, AS_OF).tendencies["high_press"]
        self.assertEqual(t.prevalence, 0.0)

    def test_breakdowns_weighted_and_kept(self):
        matches = [
            match_input("a", 1, aggs=[make_agg("flank_overload", breakdowns={"side": {"left": 8, "right": 2}})]),
            match_input("b", 2, aggs=[make_agg("flank_overload", breakdowns={"side": {"left": 6}})]),
        ]
        t = build_team_profile("FC", matches, ["flank_overload"], AS_OF).tendencies["flank_overload"]
        self.assertGreater(t.breakdowns["side"]["left"], t.breakdowns["side"].get("right", 0.0))

    def test_context_tags_travel_to_per_match_rows(self):
        matches = [match_input("a", 1, aggs=[make_agg("high_press")], tags=("vs_top_side",))]
        t = build_team_profile("FC", matches, ["high_press"], AS_OF).tendencies["high_press"]
        self.assertEqual(t.per_match[0].context_tags, ("vs_top_side",))


if __name__ == "__main__":
    unittest.main()
