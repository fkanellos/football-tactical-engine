"""Stoppage signal: uniform dead-phase coverage, no cause claims, cross-refs."""

from __future__ import annotations

import unittest

from pipeline.events.kinematics import KinematicExtractor
from pipeline.events.model import EventType
from pipeline.events.restarts import RestartDetector
from pipeline.events.setpieces import SetPieceDetector
from pipeline.events.stoppages import StoppageDetector
from pipeline.events.testing import synthetic


class StoppageTest(unittest.TestCase):
    def test_sudden_stop_is_detected_without_cause_claims(self):
        kin = KinematicExtractor().extract(synthetic.stoppage_scenario())
        events = StoppageDetector().detect(kin)
        self.assertEqual(len(events), 1, f"expected one stoppage, got {events}")
        event = events[0]
        self.assertAlmostEqual(event.start_s, 8.4, delta=1.0)
        self.assertIsNone(event.team)
        self.assertIsNone(event.metadata["explained_by"])
        self.assertGreaterEqual(event.confidence, 0.4)
        self.assertAlmostEqual(event.x, 0.0, delta=1.0)
        self.assertAlmostEqual(event.y, 5.0, delta=1.0)
        # the narrow scope is a contract: no cause vocabulary in the metadata
        self.assertNotIn("possible_causes", event.metadata)

    def test_cross_references_the_set_piece(self):
        """Coverage stays uniform: the set-piece dead spell also gets a stoppage,
        cross-referenced via explained_by instead of being suppressed."""
        kin = KinematicExtractor().extract(synthetic.set_piece_scenario())
        restarts = RestartDetector().detect(kin)
        setups = SetPieceDetector().detect(kin, context=tuple(restarts))
        self.assertTrue(setups, "fixture sanity: the setup must be detected")
        events = StoppageDetector().detect(kin, context=tuple(restarts) + tuple(setups))
        explained = [e for e in events if e.metadata["explained_by"] == "set_piece_setup"]
        self.assertTrue(explained, f"expected an explained stoppage, got {events}")

    def test_static_players_with_a_live_ball_is_not_a_stoppage(self):
        kin = KinematicExtractor().extract(synthetic.slow_circulation_scenario())
        self.assertEqual(StoppageDetector().detect(kin), [])


if __name__ == "__main__":
    unittest.main()
