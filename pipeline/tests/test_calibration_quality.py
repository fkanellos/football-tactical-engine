"""Tests for the no-ground-truth calibration quality signals and combined score."""

from __future__ import annotations

import math
import random
import unittest

from pipeline.calibration.camera import (
    compose_homography,
    decompose_homography,
    fit_homography,
    project,
    rescale_homography,
)
from pipeline.calibration.pitch import LANDMARKS, visible_landmarks
from pipeline.calibration.quality import (
    convex_hull,
    landmark_motion_px,
    plausibility_metrics,
    polygon_area,
    reprojection_metrics,
    score_frame,
    static_spans,
    support_metrics,
    teleport_count,
    world_jitter_m,
)
from pipeline.calibration.testing import synthetic


def _clean_correspondences(pose):
    h = compose_homography(pose)
    vis = visible_landmarks(h, 1280, 720)
    return h, [(w, project(h, w)) for w in vis.values()]


class PolygonHelpersTest(unittest.TestCase):
    def test_hull_and_area_of_square(self):
        pts = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]
        hull = convex_hull(pts)
        self.assertEqual(len(hull), 4)
        self.assertAlmostEqual(polygon_area(hull), 4.0)


class ReprojectionTest(unittest.TestCase):
    def test_zero_residual_for_perfect_fit(self):
        h, corr = _clean_correspondences(synthetic.main_camera_pose())
        m = reprojection_metrics(h, corr)
        self.assertLess(m.rms_px, 1e-9)
        self.assertLess(m.rms_m, 1e-9)

    def test_corrupted_observations_show_up_in_metres(self):
        pose = synthetic.main_camera_pose()
        h = compose_homography(pose)
        obs = synthetic.observe_landmarks(pose, synthetic.DEGRADED, random.Random(4))
        m = reprojection_metrics(h, obs)
        self.assertGreater(m.rms_m, 0.2)
        self.assertGreater(m.max_px, m.rms_px)

    def test_pixel_frame_mismatch_is_caught_by_residuals(self):
        """The OFI-run bug (design §3.4): homography in a virtual 1920x1080 frame,
        detections in real 1280x720 pixels. Against keypoints expressed in the
        real frame, the mismatched H shows metre-scale residuals; the repaired
        (rescaled) H drops back to zero."""
        pose_720 = synthetic.main_camera_pose()
        h_720 = compose_homography(pose_720)
        vis = visible_landmarks(h_720, 1280, 720)
        corr_720 = [(w, project(h_720, w)) for w in vis.values()]
        h_1080 = rescale_homography(h_720, (1280.0, 720.0), (1920.0, 1080.0))
        mismatched = reprojection_metrics(h_1080, corr_720)
        repaired = reprojection_metrics(
            rescale_homography(h_1080, (1920.0, 1080.0), (1280.0, 720.0)), corr_720
        )
        self.assertGreater(mismatched.rms_m, 1.0)
        self.assertLess(repaired.rms_m, 1e-6)


class SupportTest(unittest.TestCase):
    def test_collinear_points_score_zero_collinearity(self):
        m = support_metrics([(0.0, y) for y in (-20.0, -10.0, 0.0, 10.0)])
        self.assertLess(m.collinearity, 1e-9)

    def test_near_goal_support_is_tighter_than_wide(self):
        wide = support_metrics(
            [w for w in visible_landmarks(
                compose_homography(synthetic.main_camera_pose()), 1280, 720).values()]
        )
        tight = support_metrics(
            [w for w in visible_landmarks(
                compose_homography(synthetic.near_goal_pose()), 1280, 720).values()]
        )
        self.assertGreater(wide.spread_major_m, tight.spread_major_m)
        self.assertGreater(wide.world_coverage, tight.world_coverage)


class PlausibilityTest(unittest.TestCase):
    def test_true_camera_is_plausible(self):
        h = compose_homography(synthetic.main_camera_pose())
        p = plausibility_metrics(h, 1280, 720)
        self.assertTrue(p.decomposable)
        self.assertAlmostEqual(p.camera_height_m, 18.0, delta=0.1)
        self.assertAlmostEqual(p.roll_deg, 0.0, delta=0.1)
        self.assertTrue(p.chirality_ok)
        # main-camera horizon sits in the top strip of the frame
        self.assertLess(p.horizon_v_fraction, 0.3)
        self.assertGreater(p.pitch_fraction_of_frame, 0.5)
        self.assertIsNotNone(p.visible_pitch_area_m2)
        self.assertGreater(p.visible_pitch_area_m2, 200.0)

    def test_low_flat_camera_is_flagged_by_horizon_position(self):
        # A pitch-level camera looking flat: the horizon drops to mid-frame —
        # physically fine for a fan cam, implausible for the calibrated main
        # broadcast view.
        pose = synthetic.main_camera_pose(tilt_deg=0.5, position=(0.0, -55.0, 2.0))
        p = plausibility_metrics(compose_homography(pose), 1280, 720)
        self.assertGreater(p.horizon_v_fraction, 0.4)


class SequenceMetricsTest(unittest.TestCase):
    def test_landmark_motion_and_static_spans(self):
        pose = synthetic.main_camera_pose()
        series = []
        # 10 static frames, then a pan.
        for i in range(10):
            h = compose_homography(pose)
            series.append({n: project(h, w) for n, w in visible_landmarks(h, 1280, 720).items()})
        for i in range(10):
            moved = synthetic.main_camera_pose(pan_deg=1.5 * (i + 1))
            h = compose_homography(moved)
            series.append({n: project(h, w) for n, w in visible_landmarks(h, 1280, 720).items()})
        self.assertEqual(landmark_motion_px(series[0], series[1]), 0.0)
        self.assertGreater(landmark_motion_px(series[10], series[11]), 5.0)
        spans = static_spans(series, max_median_motion_px=1.5, min_span_frames=5)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][0], 0)
        self.assertGreaterEqual(spans[0][1], 9)

    def test_world_jitter_zero_for_identical_h(self):
        h = compose_homography(synthetic.main_camera_pose())
        self.assertAlmostEqual(world_jitter_m(h, h, 1280, 720), 0.0)

    def test_world_jitter_sees_noisy_fits(self):
        pose = synthetic.main_camera_pose()
        rng = random.Random(9)
        fits = [
            fit_homography(synthetic.observe_landmarks(pose, synthetic.REALISTIC, rng))
            for _ in range(2)
        ]
        j = world_jitter_m(fits[0].h, fits[1].h, 1280, 720)
        self.assertGreater(j, 0.05)

    def test_teleport_count(self):
        smooth = [(t * 0.2, 2.0 * t * 0.2, 0.0) for t in range(20)]  # 2 m/s
        self.assertEqual(teleport_count(smooth), 0)
        jumpy = list(smooth)
        jumpy[10] = (jumpy[10][0], jumpy[10][1] + 15.0, 5.0)  # ~80 m/s excursion
        self.assertGreaterEqual(teleport_count(jumpy), 1)


class ScoreFrameTest(unittest.TestCase):
    def test_no_homography_scores_zero(self):
        q = score_frame(None, 1280, 720)
        self.assertEqual(q.quality, 0.0)

    def test_regime_ordering_wide_beats_near_goal(self):
        """The gate must separate the OFI failure regime from healthy frames."""
        def quality_for(pose, seed):
            obs = synthetic.observe_landmarks(pose, synthetic.REALISTIC, random.Random(seed))
            fit = fit_homography(obs)
            if fit is None:
                return None
            return score_frame(
                fit.h, 1280, 720, correspondences=obs, nullspace_gap=fit.nullspace_gap
            ).quality

        wide = [q for q in (quality_for(synthetic.main_camera_pose(), s) for s in range(6)) if q is not None]
        tight = [q for q in (quality_for(synthetic.near_goal_pose(), s) for s in range(6)) if q is not None]
        self.assertTrue(wide and tight)
        median_wide = sorted(wide)[len(wide) // 2]
        median_tight = sorted(tight)[len(tight) // 2]
        self.assertGreater(median_wide, 0.5)
        self.assertLess(median_tight, 0.35)
        self.assertGreater(median_wide, 2.0 * median_tight)

    def test_missing_evidence_caps_quality(self):
        h, corr = _clean_correspondences(synthetic.main_camera_pose())
        with_evidence = score_frame(h, 1280, 720, correspondences=corr, nullspace_gap=5e-2)
        bare = score_frame(h, 1280, 720)
        self.assertLess(bare.quality, with_evidence.quality)
        self.assertLessEqual(bare.quality, 0.62)
        self.assertIsNone(bare.residual_score)

    def test_temporal_reference_rewards_consistency(self):
        pose = synthetic.main_camera_pose()
        h, corr = _clean_correspondences(pose)
        near = score_frame(
            h, 1280, 720, correspondences=corr, nullspace_gap=5e-2,
            reference_pose_vector=pose.as_vector(),
        )
        far_ref = synthetic.main_camera_pose(pan_deg=15.0)
        far = score_frame(
            h, 1280, 720, correspondences=corr, nullspace_gap=5e-2,
            reference_pose_vector=far_ref.as_vector(),
        )
        self.assertGreater(near.temporal_score, 0.9)
        self.assertEqual(far.temporal_score, 0.0)
        self.assertGreater(near.quality, far.quality)


if __name__ == "__main__":
    unittest.main()
