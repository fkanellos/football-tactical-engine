"""Tests for the calibration geometry core: linalg, fitting, pose decomposition."""

from __future__ import annotations

import math
import random
import unittest

from pipeline.calibration.camera import (
    CameraPose,
    compose_homography,
    decompose_homography,
    fit_homography,
    project,
    rescale_homography,
    unproject,
)
from pipeline.calibration.linalg import jacobi_eigh, nearest_rotation
from pipeline.calibration.pitch import LANDMARKS, visible_landmarks
from pipeline.calibration.testing import synthetic


class LinalgTest(unittest.TestCase):
    def test_jacobi_recovers_known_spectrum(self):
        # diag(1, 4, 9) conjugated by a rotation has eigenvalues {1, 4, 9}.
        c, s = math.cos(0.7), math.sin(0.7)
        r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        d = [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]
        rt = [[r[j][i] for j in range(3)] for i in range(3)]
        m = [[sum(r[i][k] * d[k][k] * rt[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        values, vectors = jacobi_eigh(m)
        for got, want in zip(values, [1.0, 4.0, 9.0]):
            self.assertAlmostEqual(got, want, places=9)
        # eigenvectors are unit and satisfy M v = lambda v
        for lam, vec in zip(values, vectors):
            mv = [sum(m[i][k] * vec[k] for k in range(3)) for i in range(3)]
            for a, b in zip(mv, [lam * x for x in vec]):
                self.assertAlmostEqual(a, b, places=8)

    def test_nearest_rotation_cleans_perturbation(self):
        c, s = math.cos(0.3), math.sin(0.3)
        r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        rng = random.Random(1)
        noisy = [[r[i][j] + rng.uniform(-0.02, 0.02) for j in range(3)] for i in range(3)]
        cleaned = nearest_rotation(noisy)
        self.assertIsNotNone(cleaned)
        # orthonormal columns
        for i in range(3):
            col = [cleaned[k][i] for k in range(3)]
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in col)), 1.0, places=9)
        # close to the original rotation
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(cleaned[i][j], r[i][j], delta=0.05)


class PoseRoundTripTest(unittest.TestCase):
    def test_compose_decompose_round_trip_across_pose_grid(self):
        for pan in (-40.0, -15.0, 0.0, 22.0, 41.0):
            for tilt in (6.0, 12.0, 24.0):
                for focal in (900.0, 1400.0, 3200.0):
                    pose = synthetic.main_camera_pose(
                        pan_deg=pan, tilt_deg=tilt, roll_deg=1.2, focal_px=focal
                    )
                    h = compose_homography(pose)
                    back = decompose_homography(h, pose.principal_point)
                    self.assertIsNotNone(back, f"pan={pan} tilt={tilt} f={focal}")
                    self.assertAlmostEqual(back.pan_deg, pan, places=6)
                    self.assertAlmostEqual(back.tilt_deg, tilt, places=6)
                    self.assertAlmostEqual(back.roll_deg, 1.2, places=6)
                    self.assertAlmostEqual(back.focal_px, focal, places=3)
                    for got, want in zip(back.position, pose.position):
                        self.assertAlmostEqual(got, want, places=6)

    def test_positive_pan_points_at_right_goal(self):
        pose = synthetic.near_goal_pose(side="right")
        h = compose_homography(pose)
        names = set(visible_landmarks(h, 1280, 720))
        self.assertTrue(any("right" in n for n in names))
        self.assertFalse(any(n.endswith("left_top_corner") for n in names))

    def test_identity_is_not_a_camera(self):
        h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertIsNone(decompose_homography(h, (640.0, 360.0)))

    def test_pose_vector_round_trip(self):
        pose = synthetic.main_camera_pose(pan_deg=13.0, focal_px=2100.0)
        back = CameraPose.from_vector(pose.as_vector(), pose.principal_point)
        self.assertAlmostEqual(back.focal_px, 2100.0, places=6)
        self.assertAlmostEqual(back.pan_deg, 13.0, places=9)


class ProjectionTest(unittest.TestCase):
    def test_project_unproject_inverse(self):
        pose = synthetic.main_camera_pose()
        h = compose_homography(pose)
        for wxy in [(0.0, 0.0), (20.0, -10.0), (-30.0, 25.0)]:
            p = project(h, wxy)
            self.assertIsNotNone(p)
            back = unproject(h, p)
            self.assertAlmostEqual(back[0], wxy[0], places=6)
            self.assertAlmostEqual(back[1], wxy[1], places=6)

    def test_rescale_homography_matches_scaled_pixels(self):
        # A camera calibrated in a 1920x1080 frame, re-expressed for 1280x720:
        # every projection must land at exactly 2/3 of the 1080p pixel position.
        pose_1080 = CameraPose(
            pan_deg=10.0, tilt_deg=12.0, roll_deg=0.0, focal_px=2100.0,
            position=(0.0, -55.0, 18.0), principal_point=(960.0, 540.0),
        )
        h1080 = compose_homography(pose_1080)
        h720 = rescale_homography(h1080, (1920.0, 1080.0), (1280.0, 720.0))
        for wxy in [(0.0, 0.0), (30.0, 15.0), (-40.0, -20.0)]:
            p_full = project(h1080, wxy)
            p_scaled = project(h720, wxy)
            self.assertAlmostEqual(p_scaled[0], p_full[0] * 2.0 / 3.0, places=6)
            self.assertAlmostEqual(p_scaled[1], p_full[1] * 2.0 / 3.0, places=6)


class HomographyFitTest(unittest.TestCase):
    def test_exact_recovery_from_clean_correspondences(self):
        pose = synthetic.main_camera_pose()
        h = compose_homography(pose)
        vis = visible_landmarks(h, 1280, 720)
        corr = [(w, project(h, w)) for w in vis.values()]
        fit = fit_homography(corr)
        self.assertIsNotNone(fit)
        self.assertLess(fit.rms_residual_px, 1e-6)
        # action equivalence on held-out world points
        for wxy in [(-52.5, 34.0), (52.5, -34.0), (10.0, 5.0)]:
            a, b = project(h, wxy), project(fit.h, wxy)
            self.assertAlmostEqual(a[0], b[0], places=3)
            self.assertAlmostEqual(a[1], b[1], places=3)

    def test_too_few_points_returns_none(self):
        pose = synthetic.main_camera_pose()
        h = compose_homography(pose)
        corr = [((x, y), project(h, (x, y))) for x, y in [(0, 0), (10, 0), (0, 10)]]
        self.assertIsNone(fit_homography(corr))

    def test_collinear_points_flagged_by_nullspace_gap(self):
        pose = synthetic.main_camera_pose()
        h = compose_homography(pose)
        # 6 points along the halfway line: a one-parameter family of homographies
        # fits them, so the nullspace gap must collapse toward zero.
        corr = [((0.0, y), project(h, (0.0, y))) for y in (-30, -18, -6, 6, 18, 30)]
        fit = fit_homography(corr)
        if fit is not None:
            self.assertLess(fit.nullspace_gap, 1e-8)

    def test_conditioning_separates_wide_from_near_goal(self):
        wide = synthetic.main_camera_pose()
        tight = synthetic.near_goal_pose()
        gaps = {}
        for name, pose in (("wide", wide), ("tight", tight)):
            h = compose_homography(pose)
            corr = [(w, project(h, w)) for w in visible_landmarks(h, 1280, 720).values()]
            gaps[name] = fit_homography(corr).nullspace_gap
        # measured: ~5e-2 vs ~2e-3; assert an order of magnitude of headroom
        self.assertGreater(gaps["wide"], gaps["tight"] * 5.0)

    def test_near_goal_fit_interpolates_but_extrapolates_garbage(self):
        """The core degeneracy phenomenon (design §4 H2): residuals tiny at the
        supporting landmarks, error explodes at the far end of the pitch."""
        pose = synthetic.near_goal_pose()
        h_true = compose_homography(pose)
        obs = synthetic.observe_landmarks(pose, synthetic.REALISTIC, random.Random(3))
        fit = fit_homography(obs)
        self.assertIsNotNone(fit)
        self.assertLess(fit.rms_residual_px, 6.0)
        centre_true = project(h_true, (0.0, 0.0))
        centre_fit = project(fit.h, (0.0, 0.0))
        err = math.hypot(centre_fit[0] - centre_true[0], centre_fit[1] - centre_true[1])
        self.assertGreater(err, 30.0)


class SyntheticHarnessTest(unittest.TestCase):
    def test_observation_determinism(self):
        pose = synthetic.main_camera_pose()
        a = synthetic.observe_landmarks(pose, synthetic.REALISTIC, random.Random(5))
        b = synthetic.observe_landmarks(pose, synthetic.REALISTIC, random.Random(5))
        self.assertEqual(a, b)

    def test_scenario_visibility_counts(self):
        wide = compose_homography(synthetic.main_camera_pose())
        tight = compose_homography(synthetic.near_goal_pose())
        n_wide = len(visible_landmarks(wide, 1280, 720))
        n_tight = len(visible_landmarks(tight, 1280, 720))
        self.assertGreaterEqual(n_wide, 8)
        self.assertLessEqual(n_tight, 8)

    def test_hallucinations_are_added_and_wrong(self):
        pose = synthetic.main_camera_pose()
        noise = synthetic.ObservationNoise(noise_px=0.0, n_hallucinations=2)
        obs = synthetic.observe_landmarks(pose, noise, random.Random(2))
        clean = synthetic.observe_landmarks(pose, synthetic.CLEAN, random.Random(2))
        self.assertEqual(len(obs), len(clean) + 2)
        h = compose_homography(pose)
        worst = max(
            math.hypot(project(h, w)[0] - uv[0], project(h, w)[1] - uv[1])
            for (w, uv) in obs
            if project(h, w) is not None
        )
        self.assertGreater(worst, 50.0)

    def test_broadcast_trajectory_is_fixed_mount(self):
        traj = synthetic.broadcast_trajectory(40)
        positions = {p.position for p in traj}
        self.assertEqual(len(positions), 1)


if __name__ == "__main__":
    unittest.main()
