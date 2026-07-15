"""Pass semantics: arc detection, outcomes, the subtype ladder, dribble rejection."""

from __future__ import annotations

import unittest
from functools import lru_cache
from typing import List

from pipeline.events.kinematics import KinematicExtractor
from pipeline.events.model import EventType, MatchEvent
from pipeline.events.passes import PassDetector
from pipeline.events.restarts import RestartDetector
from pipeline.events.shots import ShotDetector
from pipeline.events.testing import synthetic
from pipeline.patterns.tracking import TeamSide


@lru_cache(maxsize=None)
def passes_for(variant: str) -> List[MatchEvent]:
    kin = KinematicExtractor().extract(synthetic.pass_scenario(variant))
    return PassDetector().detect(kin)


def the_pass(variant: str) -> MatchEvent:
    events = passes_for(variant)
    assert len(events) >= 1, f"expected a pass in {variant!r}, got {events}"
    return events[0]


class PassDetectionTest(unittest.TestCase):
    def test_short_pass_completed(self):
        event = the_pass("short")
        self.assertIs(event.team, TeamSide.HOME)
        self.assertEqual(event.metadata["passer_track_id"], 105)
        self.assertEqual(event.metadata["receiver_track_id"], 106)
        self.assertEqual(event.metadata["outcome"], "completed")
        self.assertEqual(event.metadata["subtype"], "short_pass")
        self.assertGreaterEqual(event.confidence, 0.5)
        self.assertAlmostEqual(event.start_s, 5.2, delta=0.5)

    def test_long_pass(self):
        event = the_pass("long")
        self.assertEqual(event.metadata["subtype"], "long_pass")
        self.assertGreaterEqual(event.metadata["length_m"], 30.0)
        self.assertEqual(event.metadata["outcome"], "completed")

    def test_through_ball_beyond_the_line(self):
        event = the_pass("through")
        self.assertEqual(event.metadata["subtype"], "through_ball")
        self.assertTrue(event.metadata["forward"])
        self.assertTrue(event.metadata["line_visible"])
        self.assertEqual(event.metadata["receiver_track_id"], 109)

    def test_cross_into_the_box(self):
        event = the_pass("cross")
        self.assertEqual(event.metadata["subtype"], "cross")
        self.assertEqual(event.metadata["receiver_track_id"], 109)

    def test_cutback_from_the_byline(self):
        event = the_pass("cutback")
        self.assertEqual(event.metadata["subtype"], "cutback")
        self.assertFalse(event.metadata["forward"])

    def test_interception_is_a_located_turnover(self):
        event = the_pass("intercepted")
        self.assertEqual(event.metadata["outcome"], "intercepted")
        self.assertEqual(event.metadata["receiver_track_id"], 208)
        self.assertIs(event.team, TeamSide.HOME)  # the passer's team, who lost it

    def test_dribble_is_not_a_pass(self):
        self.assertEqual(passes_for("dribble"), [])

    def test_shot_context_excludes_the_launch(self):
        """A launch already classified as a shot must not double as a pass."""
        kin = KinematicExtractor().extract(synthetic.shot_scenario("saved"))
        shots = ShotDetector().detect(kin, context=tuple(RestartDetector().detect(kin)))
        self.assertTrue(shots)
        events = PassDetector().detect(kin, context=tuple(shots))
        shot_t = shots[0].start_s
        self.assertFalse(
            [e for e in events if abs(e.start_s - shot_t) < 0.3],
            "the shot launch leaked into the pass stream",
        )


if __name__ == "__main__":
    unittest.main()
