"""End-to-end event pipeline: staged context flow + the match-events/v1 contract."""

from __future__ import annotations

import json
import unittest

from pipeline.events import EventInferencePipeline
from pipeline.events.model import EVENT_SCHEMA_VERSION, EventType
from pipeline.events.testing import synthetic


class EventPipelineTest(unittest.TestCase):
    def test_goal_story_end_to_end(self):
        """The corroboration chain must survive the full staged run: the kickoff
        (stage 1) upgrades the shot's outcome (stage 2) to a goal."""
        stream = EventInferencePipeline().run(synthetic.shot_scenario("goal"))
        shots = stream.of_type(EventType.SHOT)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].metadata["outcome"], "goal")
        self.assertTrue(stream.of_type(EventType.KICKOFF))

    def test_json_contract(self):
        stream = EventInferencePipeline().run(synthetic.shot_scenario("goal"))
        doc = stream.to_json_dict()
        self.assertEqual(doc["schema"], EVENT_SCHEMA_VERSION)
        self.assertEqual(doc["match_id"], "synthetic-shot-goal")
        json.dumps(doc)  # must be serializable as-is
        starts = [e["start"] for e in doc["events"]]
        self.assertEqual(starts, sorted(starts))
        for event in doc["events"]:
            self.assertIn(event["tier"], (1, 2, 3))
            self.assertIn("confidence", event)

    def test_detectors_registered_in_stage_order(self):
        detectors = EventInferencePipeline().detectors
        self.assertEqual(
            [d.detector_id for d in detectors],
            ["restarts", "shots", "passes", "setpieces", "stoppages"],
        )


if __name__ == "__main__":
    unittest.main()
