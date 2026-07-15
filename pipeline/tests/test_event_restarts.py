"""Tier 1 restart semantics: classification, noise bands, morphology-over-touch."""

from __future__ import annotations

import unittest
from functools import lru_cache
from typing import List

from pipeline.events.kinematics import KinematicExtractor
from pipeline.events.model import EventType, MatchEvent
from pipeline.events.restarts import RestartDetector
from pipeline.events.testing import synthetic
from pipeline.patterns.tracking import TeamSide


@lru_cache(maxsize=None)
def run(scenario: str) -> List[MatchEvent]:
    match = {
        "throw_in": lambda: synthetic.throw_in_scenario(),
        "throw_in_ambiguous": lambda: synthetic.throw_in_scenario(excursion_m=0.3),
        "stays_in": synthetic.ball_stays_in_scenario,
        "corner": synthetic.corner_scenario,
        "goal_kick": lambda: synthetic.goal_kick_scenario(),
        "deflected": lambda: synthetic.goal_kick_scenario(deflected=True),
        "kickoff": synthetic.kickoff_scenario,
    }[scenario]()
    kin = KinematicExtractor().extract(match)
    return RestartDetector().detect(kin)


class ThrowInTest(unittest.TestCase):
    def test_clean_throw_in(self):
        events = run("throw_in")
        throws = [e for e in events if e.event_type is EventType.THROW_IN]
        self.assertEqual(len(throws), 1, f"expected one throw-in, got {events}")
        event = throws[0]
        self.assertIs(event.team, TeamSide.HOME)  # AWAY touched it last
        self.assertAlmostEqual(event.x, 10.4, delta=1.5)
        self.assertAlmostEqual(event.y, 34.0, delta=0.5)
        self.assertGreaterEqual(event.confidence, 0.6)
        self.assertTrue(event.metadata["resumption_observed"])
        self.assertEqual(event.metadata["last_touch_team"], "away")
        self.assertFalse(event.metadata["attribution_conflict"])
        self.assertGreater(event.end_s, event.start_s)  # spans the dead time

    def test_ambiguous_crossing_is_a_band_not_a_boolean(self):
        """0.3 m excursion is inside position noise: still a throw-in (the restart
        morphology is decisive) but at visibly lower confidence than a clean exit."""
        clean = [e for e in run("throw_in") if e.event_type is EventType.THROW_IN][0]
        ambiguous = [e for e in run("throw_in_ambiguous") if e.event_type is EventType.THROW_IN]
        self.assertEqual(len(ambiguous), 1)
        self.assertLess(ambiguous[0].metadata["excursion_m"], 0.5)
        self.assertLess(ambiguous[0].confidence, clean.confidence - 0.03)
        self.assertGreaterEqual(ambiguous[0].confidence, 0.3)

    def test_silent_when_ball_stays_in(self):
        self.assertEqual(run("stays_in"), [])


class GoalLineTest(unittest.TestCase):
    def test_corner_off_a_defender(self):
        events = run("corner")
        corners = [e for e in events if e.event_type is EventType.CORNER]
        self.assertEqual(len(corners), 1, f"expected one corner, got {events}")
        event = corners[0]
        self.assertIs(event.team, TeamSide.HOME)  # attacking team takes it
        self.assertEqual(event.metadata["last_touch_team"], "away")
        self.assertFalse(event.metadata["attribution_conflict"])
        self.assertGreaterEqual(event.confidence, 0.5)
        self.assertFalse([e for e in events if e.event_type is EventType.GOAL_KICK])

    def test_goal_kick_off_an_attacker(self):
        events = run("goal_kick")
        kicks = [e for e in events if e.event_type is EventType.GOAL_KICK]
        self.assertEqual(len(kicks), 1, f"expected one goal kick, got {events}")
        event = kicks[0]
        self.assertIs(event.team, TeamSide.AWAY)  # defending team restarts
        self.assertEqual(event.metadata["last_touch_team"], "home")
        self.assertGreaterEqual(event.confidence, 0.5)

    def test_resumption_geometry_overrules_last_touch(self):
        """Attacker touched last (says goal kick) but play resumes from the corner
        arc: an unseen deflection — classify CORNER, flag the conflict (§4.2)."""
        events = run("deflected")
        corners = [e for e in events if e.event_type is EventType.CORNER]
        self.assertEqual(len(corners), 1, f"expected the deflection corner, got {events}")
        event = corners[0]
        self.assertIs(event.team, TeamSide.HOME)
        self.assertTrue(event.metadata["attribution_conflict"])
        goal_kick_conf = [e for e in run("goal_kick") if e.event_type is EventType.GOAL_KICK][0]
        self.assertLess(event.confidence, goal_kick_conf.confidence)


class KickoffTest(unittest.TestCase):
    def test_kickoff_detected(self):
        events = run("kickoff")
        kickoffs = [e for e in events if e.event_type is EventType.KICKOFF]
        self.assertEqual(len(kickoffs), 1, f"expected one kickoff, got {events}")
        event = kickoffs[0]
        self.assertIs(event.team, TeamSide.HOME)
        self.assertAlmostEqual(event.x, 0.0, delta=0.5)
        self.assertGreaterEqual(event.confidence, 0.5)

    def test_no_other_restarts_invented(self):
        events = run("kickoff")
        self.assertFalse([e for e in events if e.event_type is not EventType.KICKOFF])


if __name__ == "__main__":
    unittest.main()
