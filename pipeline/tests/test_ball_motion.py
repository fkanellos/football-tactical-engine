"""Tests for the ball motion model: smoothing, coasting, reanchoring, gap-bridging."""

from __future__ import annotations

import math
import random
import unittest

from pipeline.ball.model import BallSample
from pipeline.ball.motion import (
    BallMotionConfig,
    BallSmoother,
    BallTrackStatus,
    interpolate_gaps,
    smooth_series,
    smooth_series_bidirectional,
)
from pipeline.ball.testing import synthetic


def _rmse(smoothed, truth):
    errs = []
    for sm, tr in zip(smoothed, truth):
        if sm.xy is None or tr.xy is None:
            continue
        errs.append((sm.xy[0] - tr.xy[0]) ** 2 + (sm.xy[1] - tr.xy[1]) ** 2)
    return math.sqrt(sum(errs) / len(errs)) if errs else float("inf")


class SmootherTrackingTest(unittest.TestCase):
    def test_batch_smoothing_reduces_noise_on_a_pass(self):
        truth = synthetic.ground_pass((0.0, 0.0), 25.0, 14.0, 2.5)
        noisy = synthetic.add_noise(truth, 0.5, random.Random(3))
        smoothed = smooth_series_bidirectional(noisy)
        raw_rmse = _rmse(
            [type("S", (), {"xy": s.xy})() for s in noisy], truth
        )
        smooth_rmse = _rmse(smoothed, truth)
        self.assertLess(smooth_rmse, raw_rmse * 0.8)

    def test_velocity_estimate_converges_on_constant_velocity_flight(self):
        truth = synthetic.flight((0.0, 0.0), (30.0, 0.0), 1.5)  # 20 m/s along +x
        smoothed = smooth_series(truth)
        vx, vy = smoothed[-1].velocity
        self.assertAlmostEqual(vx, 20.0, delta=2.0)
        self.assertAlmostEqual(vy, 0.0, delta=1.0)

    def test_coasts_through_short_gap_and_recovers(self):
        truth = synthetic.flight((0.0, 0.0), (40.0, 0.0), 2.0)  # 20 m/s
        gappy = synthetic.drop_span(truth, 0.8, 1.2)
        smoothed = smooth_series(gappy)
        statuses = [s.status for s in smoothed]
        self.assertIn(BallTrackStatus.COASTED, statuses)
        # During the gap the coasted prediction stays near the true line.
        for sm, tr in zip(smoothed, truth):
            if sm.status is BallTrackStatus.COASTED and sm.xy is not None:
                self.assertLess(abs(sm.xy[1] - 0.0), 1.0)
                self.assertLess(abs(sm.xy[0] - tr.xy[0]), 6.0)
        # After the gap it re-locks without a LOST.
        self.assertNotIn(BallTrackStatus.LOST, statuses)

    def test_lost_after_max_coast(self):
        truth = synthetic.hold((0.0, 0.0), 4.0)
        gappy = synthetic.drop_span(truth, 1.0, 3.5)  # 2.5 s >> max_coast_s
        smoothed = smooth_series(gappy)
        self.assertIn(BallTrackStatus.LOST, [s.status for s in smoothed])
        lost = [s for s in smoothed if s.status is BallTrackStatus.LOST]
        self.assertTrue(all(s.xy is None for s in lost))

    def test_kick_tracked_within_gate_at_25hz(self):
        # At 25 Hz a 20 m/s kick moves 0.8 m/frame — inside the dt-aware gate,
        # so the filter should simply track it, no rejection needed.
        held = synthetic.hold((0.0, 0.0), 1.0)
        t0, p = synthetic.end_of(held)
        struck = synthetic.ground_pass(p, 0.0, 20.0, 1.5, t0=t0)
        smoothed = smooth_series(held + struck)
        self.assertNotIn(BallTrackStatus.LOST, [s.status for s in smoothed])
        truth = (held + struck)[-1]
        est = smoothed[-1]
        self.assertLess(math.hypot(est.xy[0] - truth.xy[0], est.xy[1] - truth.xy[1]), 2.0)

    def test_kick_reanchors_at_5hz_grid(self):
        # At the 5 Hz analysis grid the same kick steps ~4 m/frame: the second
        # step exceeds the gate and the filter must REANCHOR onto the new
        # motion instead of dragging behind it.
        held = synthetic.hold((0.0, 0.0), 1.0, hz=5.0)
        t0, p = synthetic.end_of(held)
        struck = synthetic.ground_pass(p, 0.0, 20.0, 2.0, t0=t0, hz=5.0)
        smoothed = smooth_series(held + struck)
        statuses = [s.status for s in smoothed]
        self.assertIn(BallTrackStatus.REANCHORED, statuses)
        # Within ~1 s of the kick the estimate is back on the ball.
        idx_check = len(held) + 5
        truth = (held + struck)[idx_check]
        est = smoothed[idx_check]
        self.assertIsNotNone(est.xy)
        self.assertLess(math.hypot(est.xy[0] - truth.xy[0], est.xy[1] - truth.xy[1]), 3.0)

    def test_single_frame_false_positive_is_gated_out(self):
        truth = synthetic.ground_pass((0.0, 0.0), 0.0, 6.0, 2.0)
        corrupted = synthetic.inject_static_false_positive(
            truth, (15.0, -25.0), 1.0, 1.04, only_when_missing=False
        )
        smoothed = smooth_series(corrupted)
        fp_index = round(1.0 * 25)
        est = smoothed[fp_index]
        # The estimate must not jump to the false positive.
        self.assertIsNotNone(est.xy)
        self.assertLess(math.hypot(est.xy[0] - truth[fp_index].xy[0],
                                   est.xy[1] - truth[fp_index].xy[1]), 2.0)

    def test_low_quality_measurements_are_ignored(self):
        samples = synthetic.hold((0.0, 0.0), 1.0)
        qualities = [0.0] * len(samples)
        smoothed = smooth_series(samples, qualities)
        self.assertTrue(all(s.status is BallTrackStatus.LOST for s in smoothed))

    def test_reset_forgets_state_across_segments(self):
        smoother = BallSmoother()
        smoother.update(0.0, (0.0, 0.0))
        smoother.update(0.04, (0.4, 0.0))
        smoother.reset()
        out = smoother.update(10.0, (50.0, 20.0))
        self.assertEqual(out.status, BallTrackStatus.MEASURED)
        self.assertEqual(out.velocity, (0.0, 0.0))


class InterpolationTest(unittest.TestCase):
    def test_short_gap_is_bridged_and_flagged(self):
        truth = synthetic.flight((0.0, 0.0), (20.0, 10.0), 2.0)
        gappy = synthetic.drop_span(truth, 0.8, 1.4)
        filled = interpolate_gaps(gappy, max_gap_s=1.0)
        bridged = [s for s in filled if s.interpolated]
        self.assertTrue(bridged)
        # Linear bridge of a constant-velocity flight is exact.
        for s, tr in zip(filled, truth):
            if s.interpolated:
                self.assertAlmostEqual(s.xy[0], tr.xy[0], delta=1e-6)
                self.assertAlmostEqual(s.xy[1], tr.xy[1], delta=1e-6)
                self.assertLess(s.confidence, tr.confidence)

    def test_long_gap_stays_missing(self):
        truth = synthetic.flight((0.0, 0.0), (20.0, 10.0), 4.0)
        gappy = synthetic.drop_span(truth, 1.0, 3.0)
        filled = interpolate_gaps(gappy, max_gap_s=1.0)
        still_missing = [s for s in filled if not s.detected]
        self.assertEqual(len(still_missing), len([s for s in gappy if not s.detected]))

    def test_edge_gaps_never_bridged(self):
        truth = synthetic.hold((0.0, 0.0), 1.0)
        gappy = synthetic.drop_span(truth, 0.0, 0.2)  # leading gap
        filled = interpolate_gaps(gappy, max_gap_s=5.0)
        self.assertFalse(filled[0].detected)

    def test_original_samples_untouched(self):
        truth = synthetic.hold((5.0, 5.0), 1.0)
        gappy = synthetic.drop_span(truth, 0.4, 0.6)
        filled = interpolate_gaps(gappy, max_gap_s=1.0)
        for orig, new in zip(gappy, filled):
            if orig.detected:
                self.assertEqual(orig, new)


if __name__ == "__main__":
    unittest.main()
