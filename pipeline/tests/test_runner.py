"""End-to-end Phase 4 test: raw synthetic tracking -> MatchPatternProfile."""

from __future__ import annotations

import unittest

from pipeline.patterns.detectors.base import DetectorRegistry
from pipeline.patterns.runner import PatternDetectionPipeline
from pipeline.patterns.testing import synthetic
from pipeline.patterns.tracking import TeamSide


class PipelineEndToEndTest(unittest.TestCase):
    def test_full_run_on_counter_scenario(self):
        pipeline = PatternDetectionPipeline(DetectorRegistry.build_all())
        profile = pipeline.run(synthetic.counter_attack_scenario())

        agg = profile.get("counter_attack", TeamSide.HOME)
        self.assertIsNotNone(agg)
        self.assertEqual(agg.count, 1)
        self.assertGreater(agg.rate_per_90_observed, 0)
        self.assertIn("turnover_location", agg.breakdowns)
        self.assertEqual(agg.breakdowns["turnover_location"], {"deep": 1})
        self.assertIn("0-15min", agg.phase_counts)

        # nothing spurious for AWAY
        self.assertIsNone(profile.get("counter_attack", TeamSide.AWAY))
        # tendencies present for the offside-trap detector
        self.assertIn("offside_trap", profile.tendencies)
        self.assertGreater(profile.observed_minutes, 0)

    def test_flank_side_breakdown_feeds_phase5(self):
        pipeline = PatternDetectionPipeline(DetectorRegistry.build_all())
        profile = pipeline.run(synthetic.flank_overload_scenario())
        agg = profile.get("flank_overload", TeamSide.HOME)
        self.assertIsNotNone(agg)
        self.assertEqual(agg.breakdowns.get("side", {}).get("left"), agg.count)


if __name__ == "__main__":
    unittest.main()
