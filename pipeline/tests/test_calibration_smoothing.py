"""Tests for pose-space temporal smoothing against synthetic camera trajectories."""

from __future__ import annotations

import math
import random
import unittest

from pipeline.calibration.camera import decompose_homography
from pipeline.calibration.quality import score_frame, world_jitter_m
from pipeline.calibration.smoothing import (
    CameraSmoother,
    SmoothedFrameStatus,
    SmootherConfig,
    smooth_sequence,
    smooth_sequence_bidirectional,
)
from pipeline.calibration.testing import synthetic

PP = (640.0, 360.0)


def _measurements(trajectory, noise=synthetic.REALISTIC, seed=11):
    """(pose, quality) per frame, exactly as the real wiring would produce them."""
    out = []
    for fit, obs in synthetic.simulate_sequence(trajectory, noise, seed):
        if fit is None:
            out.append((None, 0.0))
            continue
        pose = decompose_homography(fit.h, PP)
        q = score_frame(
            fit.h, 1280, 720, correspondences=obs, nullspace_gap=fit.nullspace_gap
        )
        out.append((pose, q.quality))
    return out


def _pan_rmse(trajectory, poses):
    errs = [
        (p.pan_deg - t.pan_deg) ** 2
        for t, p in zip(trajectory, poses)
        if p is not None
    ]
    return math.sqrt(sum(errs) / len(errs))


class StaticJitterTest(unittest.TestCase):
    def test_smoothing_cuts_static_camera_jitter(self):
        """On a static stretch, frame-to-frame world motion of fixed pixels is
        pure estimation error; the smoother must remove most of it."""
        traj = synthetic.static_trajectory(synthetic.near_goal_pose(), 60)
        meas = _measurements(traj, seed=3)
        smoothed = smooth_sequence_bidirectional(meas)

        def median_jitter(hs):
            js = []
            for i in range(1, len(hs)):
                if hs[i - 1] is None or hs[i] is None:
                    continue
                j = world_jitter_m(hs[i - 1], hs[i], 1280, 720)
                if j is not None:
                    js.append(j)
            js.sort()
            return js[len(js) // 2]

        raw_fits = synthetic.per_frame_fits(traj, synthetic.REALISTIC, seed=3)
        raw = median_jitter([f.h if f else None for f in raw_fits])
        smooth = median_jitter([f.h for f in smoothed])
        self.assertGreater(raw, 0.3)  # the problem is real at this framing
        self.assertLess(smooth, raw * 0.35)

    def test_wide_static_pan_variance_shrinks(self):
        traj = synthetic.static_trajectory(synthetic.main_camera_pose(), 60)
        meas = _measurements(traj, seed=5)
        smoothed = smooth_sequence_bidirectional(meas)
        raw_err = _pan_rmse(traj, [p for p, _ in meas])
        smooth_err = _pan_rmse(traj, [f.pose for f in smoothed])
        self.assertLess(smooth_err, raw_err * 0.7)


class TrackingTest(unittest.TestCase):
    def test_bidirectional_beats_raw_over_full_broadcast_sequence(self):
        traj = synthetic.broadcast_trajectory(200)
        meas = _measurements(traj, seed=11)
        smoothed = smooth_sequence_bidirectional(meas)
        raw_poses = [p for p, _ in meas]
        common = [
            i for i in range(len(traj))
            if raw_poses[i] is not None and smoothed[i].pose is not None
        ]
        raw_rmse = math.sqrt(
            sum((raw_poses[i].pan_deg - traj[i].pan_deg) ** 2 for i in common) / len(common)
        )
        smooth_rmse = math.sqrt(
            sum((smoothed[i].pose.pan_deg - traj[i].pan_deg) ** 2 for i in common) / len(common)
        )
        self.assertLess(smooth_rmse, raw_rmse)

    def test_smoother_follows_a_continuous_pan(self):
        traj = synthetic.pan_zoom_trajectory(80, 0.0, 30.0)
        meas = _measurements(traj, noise=synthetic.ObservationNoise(noise_px=1.0), seed=2)
        smoothed = smooth_sequence_bidirectional(meas)
        # error at the end of the pan must not have accumulated lag
        tail = [f.pose.pan_deg for f in smoothed[-5:] if f.pose is not None]
        self.assertTrue(tail)
        self.assertAlmostEqual(tail[-1], 30.0, delta=1.5)


class GatingTest(unittest.TestCase):
    def _static_pose_measurements(self, n, pose=None):
        pose = pose or synthetic.main_camera_pose()
        return [(pose, 0.9)] * n

    def test_single_outlier_is_gated_out(self):
        pose = synthetic.main_camera_pose()
        garbage = synthetic.main_camera_pose(pan_deg=20.0, tilt_deg=25.0)
        meas = self._static_pose_measurements(10) + [(garbage, 0.9)] + self._static_pose_measurements(10)
        smoothed = smooth_sequence(meas)
        self.assertIs(smoothed[10].status, SmoothedFrameStatus.COASTED)
        for frame in smoothed:
            if frame.pose is not None:
                self.assertLess(abs(frame.pose.pan_deg - pose.pan_deg), 1.0)

    def test_low_quality_measurements_are_ignored(self):
        pose = synthetic.main_camera_pose()
        garbage = synthetic.main_camera_pose(pan_deg=20.0)
        meas = self._static_pose_measurements(5) + [(garbage, 0.05)] * 3 + self._static_pose_measurements(5)
        smoothed = smooth_sequence(meas)
        for frame in smoothed:
            if frame.pose is not None:
                self.assertLess(abs(frame.pose.pan_deg), 1.0)

    def test_persistent_jump_forces_reanchor(self):
        """A genuine fast camera move must not be smoothed away forever: after
        ``reanchor_run`` consistent rejections the filter snaps to the new state."""
        before = synthetic.main_camera_pose(pan_deg=0.0)
        after = synthetic.main_camera_pose(pan_deg=25.0)
        meas = [(before, 0.9)] * 10 + [(after, 0.9)] * 10
        smoothed = smooth_sequence(meas)
        statuses = [f.status for f in smoothed]
        self.assertIn(SmoothedFrameStatus.REANCHORED, statuses)
        self.assertAlmostEqual(smoothed[-1].pose.pan_deg, 25.0, delta=0.5)

    def test_lost_after_max_coast(self):
        pose = synthetic.main_camera_pose()
        config = SmootherConfig(max_coast_frames=5)
        meas = [(pose, 0.9)] * 3 + [(None, 0.0)] * 10
        smoothed = smooth_sequence(meas, config)
        self.assertIs(smoothed[-1].status, SmoothedFrameStatus.LOST)
        self.assertIsNone(smoothed[-1].pose)
        # but it coasted for a while first
        self.assertIs(smoothed[4].status, SmoothedFrameStatus.COASTED)
        self.assertIsNotNone(smoothed[4].pose)

    def test_reset_severs_segments(self):
        """Nothing bridges a broadcast cut: after reset() the first measurement
        re-initialises the filter with no memory of the previous segment."""
        smoother = CameraSmoother()
        a = synthetic.main_camera_pose(pan_deg=0.0)
        b = synthetic.main_camera_pose(pan_deg=35.0)
        for _ in range(10):
            smoother.update(a, 0.9)
        smoother.reset()
        first = smoother.update(b, 0.9)
        self.assertIs(first.status, SmoothedFrameStatus.MEASURED)
        self.assertAlmostEqual(first.pose.pan_deg, 35.0, places=6)


class BidirectionalTest(unittest.TestCase):
    def test_gap_in_middle_is_bridged_from_both_sides(self):
        pose = synthetic.main_camera_pose()
        meas = [(pose, 0.9)] * 10 + [(None, 0.0)] * 6 + [(pose, 0.9)] * 10
        smoothed = smooth_sequence_bidirectional(meas)
        for frame in smoothed:
            self.assertIsNotNone(frame.pose)
            self.assertLess(abs(frame.pose.pan_deg), 0.5)

    def test_all_lost_stays_lost(self):
        smoothed = smooth_sequence_bidirectional([(None, 0.0)] * 5)
        for frame in smoothed:
            self.assertIs(frame.status, SmoothedFrameStatus.LOST)


if __name__ == "__main__":
    unittest.main()
