"""Kinematic layer: grid alignment with the feature layer, launch detection."""

from __future__ import annotations

import unittest

from pipeline.events.kinematics import KinematicExtractor, detect_launches
from pipeline.events.testing import synthetic
from pipeline.patterns.features import FeatureExtractor
from pipeline.patterns.tracking import TeamSide


class GridAlignmentTest(unittest.TestCase):
    def test_same_grid_as_feature_layer(self):
        """Event timestamps must land on the FrameFeatures grid (design doc §3)."""
        match = synthetic.throw_in_scenario()
        kin = KinematicExtractor().extract(match)
        series = FeatureExtractor().extract(match)
        kin_times = [t for seg in kin.segments for t in seg.grid]
        self.assertEqual(kin_times, series.times())

    def test_single_segment_for_continuous_frames(self):
        kin = KinematicExtractor().extract(synthetic.throw_in_scenario())
        self.assertEqual(len(kin.segments), 1)
        self.assertEqual(kin.segments[0].period, 1)


class LaunchDetectionTest(unittest.TestCase):
    def test_pass_launch_detected_and_attributed(self):
        kin = KinematicExtractor().extract(synthetic.pass_scenario("short"))
        launches = detect_launches(kin.segments[0], kin.config)
        self.assertTrue(launches, "expected the scripted pass to register a launch")
        launch = launches[0]
        self.assertAlmostEqual(launch.t_s, 5.2, delta=0.5)
        self.assertEqual(launch.kicker_track_id, 105)
        self.assertIs(launch.kicker_team, TeamSide.HOME)
        self.assertGreaterEqual(launch.speed, 4.5)

    def test_carried_ball_is_not_a_launch(self):
        """A 4 m/s dribble stays below the launch floor."""
        kin = KinematicExtractor().extract(synthetic.pass_scenario("dribble"))
        self.assertEqual(detect_launches(kin.segments[0], kin.config), [])


if __name__ == "__main__":
    unittest.main()
