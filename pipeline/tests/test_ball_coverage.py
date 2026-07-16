"""Tests for ball detection coverage, gap statistics, and plausibility metrics."""

from __future__ import annotations

import random
import unittest

from pipeline.ball.coverage import (
    coverage_report,
    find_gaps,
    gap_contexts,
    zone_counts,
)
from pipeline.ball.ingest import ProbeFrame, probe_to_samples, project_pixel
from pipeline.ball.model import BallSample, missing, out_of_bounds_m
from pipeline.ball.plausibility import kinematic_flags, plausibility_report
from pipeline.ball.testing import synthetic


def _identityish_homography():
    # Simple scale map: 1280x720 image -> pitch-sized world, origin recentred.
    return [
        [105.0 / 1280.0, 0.0, -52.5],
        [0.0, 68.0 / 720.0, -34.0],
        [0.0, 0.0, 1.0],
    ]


class GapDetectionTest(unittest.TestCase):
    def test_full_detection_has_no_gaps(self):
        samples = synthetic.ground_pass((0.0, 0.0), 0.0, 10.0, 2.0)
        report = coverage_report(samples)
        self.assertEqual(report.n_gaps, 0)
        self.assertEqual(report.detection_rate, 1.0)
        self.assertIsNone(report.median_gap_s)

    def test_span_dropout_is_one_gap_with_true_duration(self):
        samples = synthetic.ground_pass((0.0, 0.0), 0.0, 10.0, 4.0)
        corrupted = synthetic.drop_span(samples, 1.0, 2.0)
        gaps = find_gaps(corrupted)
        self.assertEqual(len(gaps), 1)
        # Duration measured between bracketing detections: slightly more than
        # the blackout window itself (one frame step each side).
        self.assertGreaterEqual(gaps[0].duration_s, 1.0)
        self.assertLess(gaps[0].duration_s, 1.2)
        self.assertFalse(gaps[0].at_edge)

    def test_leading_gap_is_flagged_at_edge(self):
        samples = [missing(0.0), missing(0.04)] + list(
            synthetic.ground_pass((0.0, 0.0), 0.0, 8.0, 1.0, t0=0.08)
        )
        gaps = find_gaps(samples)
        self.assertEqual(len(gaps), 1)
        self.assertTrue(gaps[0].at_edge)

    def test_isolated_vs_blackout_fractions_separate_regimes(self):
        samples = synthetic.ground_pass((0.0, 0.0), 0.0, 6.0, 12.0)
        # Regime A: isolated misses only.
        isolated = synthetic.drop_random(samples, 0.1, random.Random(1))
        rep_a = coverage_report(isolated)
        # Regime B: one long blackout (a replay/close-up).
        blackout = synthetic.drop_span(samples, 3.0, 9.0)
        rep_b = coverage_report(blackout)
        self.assertGreater(rep_a.isolated_miss_fraction, 0.8)
        self.assertEqual(rep_a.blackout_frame_fraction, 0.0)
        self.assertEqual(rep_b.isolated_miss_fraction, 0.0)
        self.assertGreater(rep_b.blackout_frame_fraction, 0.4)

    def test_gap_context_reports_implied_speed_of_struck_ball(self):
        # Ball at rest, then reappears 30 m away ~0.6 s later: a struck ball
        # whose flight was missed (motion-blur signature).
        pre = synthetic.hold((0.0, 0.0), 1.0)
        t0, _ = synthetic.end_of(pre)
        gap = [missing(t0 + i * 0.04) for i in range(15)]
        post = synthetic.hold((30.0, 0.0), 1.0, t0=t0 + 15 * 0.04)
        contexts = gap_contexts(pre + gap + post)
        self.assertEqual(len(contexts), 1)
        ctx = contexts[0]
        self.assertAlmostEqual(ctx.displacement_m, 30.0, places=5)
        self.assertGreater(ctx.implied_speed_ms, 30.0)
        self.assertFalse(ctx.near_boundary)

    def test_gap_context_flags_boundary_exit(self):
        pre = synthetic.hold((51.0, 20.0), 0.5)  # 1.5 m from the goal line
        t0, _ = synthetic.end_of(pre)
        gap = [missing(t0 + i * 0.04) for i in range(75)]  # 3 s unseen
        post = synthetic.hold((49.0, 20.0), 0.5, t0=t0 + 75 * 0.04)
        contexts = gap_contexts(pre + gap + post)
        self.assertTrue(contexts[0].near_boundary)

    def test_zone_counts_place_detections(self):
        left = synthetic.hold((-40.0, 0.0), 1.0)
        centre = synthetic.hold((0.0, 0.0), 1.0, t0=2.0)
        grid = zone_counts(left + centre)
        self.assertEqual(grid[0][1], len(left))
        self.assertEqual(grid[1][1], len(centre))
        self.assertEqual(grid[2][1], 0)


class PlausibilityTest(unittest.TestCase):
    def test_clean_pass_has_no_flags(self):
        samples = synthetic.ground_pass((0.0, 0.0), 30.0, 14.0, 2.0)
        report = plausibility_report(samples)
        self.assertEqual(report.overspeed_count, 0)
        self.assertEqual(report.teleport_count, 0)
        self.assertEqual(report.out_of_bounds_count, 0)
        self.assertLess(report.max_speed_ms, 15.0)

    def test_identity_switch_to_debris_is_a_teleport(self):
        samples = synthetic.ground_pass((0.0, 0.0), 0.0, 8.0, 2.0)
        # One frame jumps to a static white object 35 m away and back.
        corrupted = synthetic.inject_static_false_positive(
            samples, (10.0, -30.0), 1.0, 1.04, only_when_missing=False
        )
        report = plausibility_report(corrupted)
        self.assertGreaterEqual(report.teleport_count, 1)

    def test_large_displacement_across_long_gap_is_not_a_teleport(self):
        pre = synthetic.hold((0.0, 0.0), 1.0)
        t0, _ = synthetic.end_of(pre)
        post = synthetic.hold((40.0, 0.0), 1.0, t0=t0 + 2.0)
        report = plausibility_report(pre + post)
        self.assertEqual(report.teleport_count, 0)
        self.assertEqual(report.overspeed_count, 0)  # 40 m / ~2 s = 20 m/s

    def test_out_of_bounds_flags_far_outside_only(self):
        near_line = synthetic.hold((53.0, 0.0), 0.5)  # 0.5 m out: throw-in range
        far_out = synthetic.hold((65.0, 0.0), 0.5, t0=1.0)  # 12.5 m out: nonsense
        report = plausibility_report(near_line + far_out)
        self.assertEqual(report.out_of_bounds_count, len(far_out))
        self.assertAlmostEqual(out_of_bounds_m((53.0, 0.0)), 0.5)


class IngestTest(unittest.TestCase):
    def test_probe_frames_join_homographies_into_pitch_samples(self):
        h = _identityish_homography()
        probe = [
            ProbeFrame(index=0, boxes=((630.0, 350.0, 640.0, 360.0, 0.6),)),
            ProbeFrame(index=1, boxes=()),
            ProbeFrame(
                index=2,
                boxes=(
                    (100.0, 100.0, 110.0, 110.0, 0.2),
                    (630.0, 350.0, 640.0, 360.0, 0.7),
                ),
            ),
        ]
        samples = probe_to_samples(probe, {0: h, 1: h, 2: h}, fps=25.0, conf_floor=0.1)
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples[0].detected)
        self.assertFalse(samples[1].detected)
        self.assertTrue(samples[2].detected)
        self.assertEqual(samples[2].n_candidates, 2)
        # Highest-confidence candidate selected: bottom-centre (635, 360)
        # projects near the pitch centre under the scale map.
        self.assertAlmostEqual(samples[2].xy[0], 635.0 * 105.0 / 1280.0 - 52.5, places=6)
        self.assertAlmostEqual(samples[2].t, 2.0 / 25.0)

    def test_missing_homography_yields_no_position_but_keeps_candidates(self):
        probe = [ProbeFrame(index=0, boxes=((0.0, 0.0, 5.0, 5.0, 0.9),))]
        samples = probe_to_samples(probe, {0: None}, fps=25.0)
        self.assertFalse(samples[0].detected)
        self.assertEqual(samples[0].n_candidates, 1)

    def test_project_pixel_rejects_degenerate_w(self):
        h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        self.assertIsNone(project_pixel(h, (1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
