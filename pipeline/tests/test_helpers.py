"""Unit tests for the shared detection idiom (base.py helpers)."""

from __future__ import annotations

import unittest

from pipeline.patterns.detectors.base import (
    MISSING_FEATURE_FACTOR,
    episode_confidence,
    episodes_from_scores,
    soft_threshold,
)


class SoftThresholdTest(unittest.TestCase):
    def test_midpoint_is_half(self):
        self.assertAlmostEqual(soft_threshold(10.0, 10.0, 2.0), 0.5)

    def test_one_softness_past_threshold(self):
        self.assertAlmostEqual(soft_threshold(12.0, 10.0, 2.0), 0.7310585786, places=6)

    def test_below_inverts(self):
        self.assertAlmostEqual(
            soft_threshold(8.0, 10.0, 2.0, above=False), 0.7310585786, places=6
        )
        self.assertLess(soft_threshold(12.0, 10.0, 2.0, above=False), 0.5)

    def test_monotonic(self):
        values = [soft_threshold(v, 5.0, 1.0) for v in (0.0, 2.0, 5.0, 8.0, 20.0)]
        self.assertEqual(values, sorted(values))

    def test_none_returns_missing_factor(self):
        self.assertEqual(soft_threshold(None, 5.0, 1.0), MISSING_FEATURE_FACTOR)

    def test_extreme_values_do_not_overflow(self):
        self.assertAlmostEqual(soft_threshold(1e9, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(soft_threshold(-1e9, 0.0, 1.0), 0.0)


class EpisodesFromScoresTest(unittest.TestCase):
    TIMES = [i * 0.2 for i in range(50)]  # 10s at 5 Hz

    def test_simple_episode(self):
        scores = [0.0] * 10 + [0.5] * 20 + [0.0] * 20
        spans = episodes_from_scores(scores, self.TIMES, 0.3, 0.15, 1.0, 1.0)
        self.assertEqual(spans, [(10, 30)])

    def test_hysteresis_keeps_episode_alive_between_thresholds(self):
        # dips to 0.2 (between exit 0.15 and enter 0.3) must NOT end the episode
        scores = [0.0] * 10 + [0.5] * 5 + [0.2] * 5 + [0.5] * 10 + [0.0] * 20
        spans = episodes_from_scores(scores, self.TIMES, 0.3, 0.15, 1.0, 0.0)
        self.assertEqual(spans, [(10, 30)])

    def test_min_duration_filters_short_episodes(self):
        scores = [0.0] * 10 + [0.5] * 3 + [0.0] * 37  # 0.4s long
        spans = episodes_from_scores(scores, self.TIMES, 0.3, 0.15, 1.0, 0.0)
        self.assertEqual(spans, [])

    def test_merge_gap(self):
        scores = [0.5] * 10 + [0.0] * 3 + [0.5] * 10 + [0.0] * 27
        merged = episodes_from_scores(scores, self.TIMES, 0.3, 0.15, 1.0, 1.0)
        self.assertEqual(merged, [(0, 23)])
        separate = episodes_from_scores(scores, self.TIMES, 0.3, 0.15, 1.0, 0.3)
        self.assertEqual(separate, [(0, 10), (13, 23)])

    def test_none_ends_episode(self):
        scores = [0.5] * 10 + [None] * 10 + [0.5] * 10 + [0.0] * 20
        spans = episodes_from_scores(scores, self.TIMES, 0.3, 0.15, 1.0, 0.5)
        self.assertEqual(spans, [(0, 10), (20, 30)])


class EpisodeConfidenceTest(unittest.TestCase):
    def test_perfect_episode(self):
        # frame scores of 0.9^k with full quality/completeness -> ~0.9
        conf = episode_confidence([0.9 ** 5] * 10, [1.0] * 10, [True] * 10, 5)
        self.assertAlmostEqual(conf, 0.9, places=5)

    def test_quality_scales_confidence(self):
        full = episode_confidence([0.5] * 10, [1.0] * 10, [True] * 10, 5)
        degraded = episode_confidence([0.5] * 10, [0.5] * 10, [True] * 10, 5)
        self.assertAlmostEqual(degraded, full * 0.5, places=6)

    def test_incompleteness_penalises(self):
        complete = episode_confidence([0.5] * 10, [1.0] * 10, [True] * 10, 5)
        incomplete = episode_confidence([0.5] * 10, [1.0] * 10, [False] * 10, 5)
        self.assertAlmostEqual(incomplete, complete * 0.4, places=6)

    def test_empty_is_zero(self):
        self.assertEqual(episode_confidence([], [], [], 5), 0.0)


if __name__ == "__main__":
    unittest.main()
