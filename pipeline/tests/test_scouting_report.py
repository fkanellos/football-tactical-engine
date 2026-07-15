"""Tests for the scouting report schema and the Phase 5 adapter."""

from __future__ import annotations

import unittest

from pipeline.patterns.tracking import TeamSide
from pipeline.scouting.aggregate import ScoutingConfig, build_team_profile
from pipeline.scouting.report import (
    SCHEMA,
    build_scouting_report,
    scouting_profile_as_match_profile,
)
from pipeline.tests.test_scouting_aggregate import AS_OF, make_agg, match_input

PATTERNS = ["high_press", "low_block", "flank_overload"]


def press_library(n_matches=6):
    """n recent matches: consistent press, one-off low block, no overloads."""
    matches = []
    for i in range(n_matches):
        aggs = [
            make_agg(
                "high_press", rate=10.0, conf=0.7, count=7,
                breakdowns={"trigger": {"open_play": 5, "restart": 2}},
            )
        ]
        if i == 0:
            aggs.append(make_agg("low_block", rate=6.0, conf=0.6, count=4))
        matches.append(match_input(f"m{i}", days_ago=7 * (i + 1), aggs=aggs))
    return matches


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.profile = build_team_profile("FC Press", press_library(), PATTERNS, AS_OF)

    def test_consistent_tendency_becomes_an_aggregate(self):
        pseudo = scouting_profile_as_match_profile(self.profile, TeamSide.AWAY)
        agg = pseudo.get("high_press", TeamSide.AWAY)
        self.assertIsNotNone(agg)
        self.assertAlmostEqual(agg.rate_per_90_observed, 10.0, places=1)
        self.assertEqual(agg.count, 42)  # 7 events x 6 matches: total evidence
        self.assertGreater(agg.mean_confidence, 0.6)
        # breakdowns survive as int counts for rule `where` filters
        self.assertGreater(agg.breakdowns["trigger"]["open_play"], 0)

    def test_never_seen_pattern_is_absent(self):
        pseudo = scouting_profile_as_match_profile(self.profile, TeamSide.AWAY)
        self.assertIsNone(pseudo.get("flank_overload", TeamSide.AWAY))

    def test_one_off_sighting_fails_the_evidence_gate(self):
        # low block appeared once in six matches: n_eff of the *tendency* is
        # fine, but a rule should still see the rate; what must NOT pass the
        # gate is a tendency built on too little effective evidence overall
        thin_profile = build_team_profile(
            "FC Thin", press_library(1), PATTERNS, AS_OF
        )
        pseudo = scouting_profile_as_match_profile(thin_profile, TeamSide.AWAY)
        self.assertIsNone(pseudo.get("high_press", TeamSide.AWAY))  # n_eff < 2

    def test_profile_identity_and_minutes(self):
        pseudo = scouting_profile_as_match_profile(self.profile, TeamSide.AWAY)
        self.assertTrue(pseudo.match_id.startswith("scouting:FC Press:"))
        self.assertAlmostEqual(pseudo.observed_minutes, 360.0, places=0)


class ReportTest(unittest.TestCase):
    def test_report_shape_and_schema(self):
        profile = build_team_profile("FC Press", press_library(), PATTERNS, AS_OF)
        report = build_scouting_report(profile)
        doc = report.to_json_dict()
        self.assertEqual(doc["schema"], SCHEMA)
        self.assertEqual(doc["n_matches"], 6)
        self.assertEqual(len(doc["matches"]), 6)
        rows = {row["pattern"]: row for row in doc["tendencies"]}
        press = rows["high_press"]
        self.assertEqual(press["seen_in"], "6 of 6")
        self.assertTrue(press["rules_eligible"])
        self.assertEqual(len(press["per_match"]), 6)
        self.assertLess(press["prevalence_ci"][0], 1.0)  # certainty never claimed
        self.assertTrue(any("OBSERVED 90" in c for c in doc["caveats"]))

    def test_thin_library_caveat(self):
        profile = build_team_profile("FC Thin", press_library(2), PATTERNS, AS_OF)
        report = build_scouting_report(profile)
        self.assertTrue(any("2 match(es)" in c for c in report.caveats))

    def test_empty_library_caveat(self):
        report = build_scouting_report(build_team_profile("FC None", [], PATTERNS, AS_OF))
        self.assertTrue(any("No historical footage" in c for c in report.caveats))
        self.assertEqual(report.tendencies, [])

    def test_recommendations_pass_through(self):
        profile = build_team_profile("FC Press", press_library(), PATTERNS, AS_OF)
        recs = [{"rule_id": "counter-the-press", "headline": "Play over the first line"}]
        report = build_scouting_report(profile, recommendations=recs)
        self.assertEqual(report.to_json_dict()["recommendations"], recs)


if __name__ == "__main__":
    unittest.main()
