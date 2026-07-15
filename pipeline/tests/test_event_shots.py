"""Shot semantics: attempt detection + the outcome honesty ladder."""

from __future__ import annotations

import unittest
from functools import lru_cache
from typing import List

from pipeline.events.kinematics import KinematicExtractor
from pipeline.events.model import EventType, MatchEvent
from pipeline.events.restarts import RestartDetector
from pipeline.events.shots import ShotDetector
from pipeline.events.testing import synthetic
from pipeline.patterns.tracking import TeamSide


@lru_cache(maxsize=None)
def shots_for(outcome: str) -> List[MatchEvent]:
    kin = KinematicExtractor().extract(synthetic.shot_scenario(outcome))
    restarts = RestartDetector().detect(kin)
    return ShotDetector().detect(kin, context=tuple(restarts))


class ShotDetectionTest(unittest.TestCase):
    def test_goal_needs_the_kickoff_corroboration(self):
        events = shots_for("goal")
        self.assertEqual(len(events), 1, f"expected one shot, got {events}")
        event = events[0]
        self.assertIs(event.team, TeamSide.HOME)
        self.assertEqual(event.metadata["shooter_track_id"], 109)
        self.assertEqual(event.metadata["outcome"], "goal")
        self.assertEqual(event.metadata["restart_after"], "kickoff")
        self.assertTrue(event.metadata["in_mouth_observed"])
        self.assertGreaterEqual(event.confidence, 0.5)
        self.assertGreaterEqual(event.metadata["outcome_confidence"], 0.5)

    def test_save_via_flight_reversal_at_the_keeper(self):
        events = shots_for("saved")
        self.assertEqual(len(events), 1, f"expected one shot, got {events}")
        event = events[0]
        self.assertEqual(event.metadata["outcome"], "saved")
        self.assertEqual(event.metadata["blocker_track_id"], 200)
        self.assertGreaterEqual(event.confidence, 0.5)

    def test_wide_miss_corroborated_by_the_goal_kick(self):
        events = shots_for("wide")
        self.assertEqual(len(events), 1, f"expected one shot, got {events}")
        event = events[0]
        self.assertEqual(event.metadata["outcome"], "off_target")
        self.assertEqual(event.metadata["restart_after"], "goal_kick")
        self.assertFalse(event.metadata["in_mouth_observed"])

    def test_outcome_confidence_is_a_separate_axis(self):
        """Detection confidence and outcome confidence must not be conflated."""
        event = shots_for("goal")[0]
        self.assertIn("outcome_confidence", event.metadata)
        self.assertNotEqual(round(event.confidence, 3),
                            round(event.metadata["outcome_confidence"], 3))

    def test_passes_do_not_read_as_shots(self):
        for variant in ("short", "long", "cross"):
            kin = KinematicExtractor().extract(synthetic.pass_scenario(variant))
            self.assertEqual(
                ShotDetector().detect(kin), [],
                f"pass variant {variant!r} produced a shot",
            )


if __name__ == "__main__":
    unittest.main()
