"""Tests for the structural latency accounting of the streaming feature layer."""

from __future__ import annotations

import unittest

from pipeline.patterns.features import FeatureExtractorConfig
from pipeline.patterns.streaming.features import feature_latencies


class FeatureLatencyTest(unittest.TestCase):
    def test_default_config_latency_table(self):
        # 5 Hz, 1s smoothing window (w=5, half=2), 2s turnover persistence:
        # these are the numbers quoted in the live-architecture design doc §3.2
        lat = feature_latencies()
        self.assertAlmostEqual(lat["positions_shape"].delay_s, 0.5)
        self.assertAlmostEqual(lat["velocities"].delay_s, 0.7)
        self.assertAlmostEqual(lat["line_velocity_closing_speed"].delay_s, 1.1)
        self.assertAlmostEqual(lat["possession.state"].delay_s, 0.5)
        self.assertAlmostEqual(lat["possession.turnover"].delay_s, 2.5)

    def test_latency_scales_with_config(self):
        cfg = FeatureExtractorConfig(resample_hz=2.0, smoothing_window_s=2.0)
        lat = feature_latencies(cfg)
        # dt=0.5; w=4 bumped to 5, half=2 -> positions 0.25 + 1.0 = 1.25
        self.assertAlmostEqual(lat["positions_shape"].delay_s, 1.25)
        self.assertAlmostEqual(lat["velocities"].delay_s, 1.75)

    def test_warmups_are_bounded_by_detector_gates(self):
        # every warm-up must sit comfortably under the shortest episode gate
        # (3s min_duration), or the startup-transient story in the doc is wrong
        for name, lat in feature_latencies().items():
            with self.subTest(feature=name):
                self.assertLessEqual(lat.warmup_s, 3.0)


if __name__ == "__main__":
    unittest.main()
