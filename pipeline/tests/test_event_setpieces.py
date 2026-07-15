"""Set-piece organization: wall geometry, taker attribution, tier-1 dedup."""

from __future__ import annotations

import unittest

from pipeline.events.kinematics import KinematicExtractor
from pipeline.events.model import EventType
from pipeline.events.restarts import RestartDetector
from pipeline.events.setpieces import SetPieceDetector
from pipeline.events.testing import synthetic
from pipeline.patterns.tracking import TeamSide


class SetPieceTest(unittest.TestCase):
    def test_free_kick_wall_setup(self):
        kin = KinematicExtractor().extract(synthetic.set_piece_scenario())
        restarts = RestartDetector().detect(kin)
        events = SetPieceDetector().detect(kin, context=tuple(restarts))
        self.assertEqual(len(events), 1, f"expected one setup, got {events}")
        event = events[0]
        self.assertEqual(event.metadata["organization"], "wall")
        self.assertEqual(event.metadata["wall_team"], "away")
        self.assertGreaterEqual(event.metadata["wall_size"], 3)
        self.assertIs(event.team, TeamSide.HOME)  # the taker's side
        self.assertAlmostEqual(event.start_s, 6.4, delta=1.0)
        self.assertAlmostEqual(event.end_s, 20.2, delta=1.0)
        self.assertAlmostEqual(event.x, 30.0, delta=1.5)
        self.assertAlmostEqual(event.y, 10.0, delta=1.5)
        self.assertGreaterEqual(event.metadata["runners_into_box"], 1)
        self.assertGreaterEqual(event.confidence, 0.5)
        self.assertTrue(event.metadata["resumption_observed"])

    def test_boundary_restarts_are_not_double_reported(self):
        """A corner's dead time is already typed by Tier 1 — no setup event."""
        kin = KinematicExtractor().extract(synthetic.corner_scenario())
        restarts = RestartDetector().detect(kin)
        self.assertTrue(restarts, "fixture sanity: the corner must be detected")
        events = SetPieceDetector().detect(kin, context=tuple(restarts))
        self.assertEqual(events, [])

    def test_without_context_it_degrades_not_crashes(self):
        """Contract: an empty context makes detectors conservative, not wrong —
        the corner dead spell is then reported as an (untyped) setup."""
        kin = KinematicExtractor().extract(synthetic.corner_scenario())
        events = SetPieceDetector().detect(kin)
        for event in events:
            self.assertIs(event.event_type, EventType.SET_PIECE_SETUP)


if __name__ == "__main__":
    unittest.main()
