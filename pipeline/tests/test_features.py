"""Tests for the feature extraction layer against synthetic tracking data."""

from __future__ import annotations

import unittest

from pipeline.patterns.features import FeatureExtractor, PossessionState
from pipeline.patterns.testing import synthetic
from pipeline.patterns.tracking import MatchTracking, TeamSide


def extract(match: MatchTracking):
    return FeatureExtractor().extract(match)


class PossessionStateMachineTest(unittest.TestCase):
    def test_confirmed_turnover_is_backdated(self):
        series = extract(synthetic.counter_attack_scenario())
        flips = [
            f for f in series.frames if f.possession.turnover_won_by is TeamSide.HOME
        ]
        self.assertEqual(len(flips), 1)
        # interception happens at t=8; ball smoothing smears it slightly
        self.assertAlmostEqual(flips[0].timestamp_s, 8.0, delta=1.2)
        # before the interception AWAY has it, well after HOME does
        at_4 = next(f for f in series.frames if abs(f.timestamp_s - 4.0) < 0.01)
        self.assertIs(at_4.possession.state, PossessionState.AWAY)
        at_14 = next(f for f in series.frames if abs(f.timestamp_s - 14.0) < 0.01)
        self.assertIs(at_14.possession.state, PossessionState.HOME)

    def test_flicker_does_not_flip_possession(self):
        series = extract(synthetic.flickering_possession_scenario())
        flips = [f for f in series.frames if f.possession.turnover_won_by is not None]
        self.assertEqual(flips, [])
        at_20 = next(f for f in series.frames if abs(f.timestamp_s - 20.0) < 0.01)
        self.assertIs(at_20.possession.state, PossessionState.AWAY)

    def test_time_since_turnover_advances(self):
        series = extract(synthetic.counter_attack_scenario())
        flip_t = next(
            f.timestamp_s
            for f in series.frames
            if f.possession.turnover_won_by is TeamSide.HOME
        )
        at_later = next(
            f for f in series.frames if abs(f.timestamp_s - (flip_t + 5.0)) < 0.11
        )
        self.assertIsNotNone(at_later.possession.time_since_turnover_s)
        self.assertAlmostEqual(
            at_later.possession.time_since_turnover_s, 5.0, delta=0.3
        )


class TeamShapeTest(unittest.TestCase):
    def test_def_line_height_is_second_deepest(self):
        series = extract(synthetic.low_block_scenario())
        deep = next(f for f in series.frames if abs(f.timestamp_s - 25.0) < 0.01)
        # back four scripted at x'=12 in the deep phase
        self.assertAlmostEqual(deep.home.def_line_height, 12.0, delta=1.0)
        self.assertAlmostEqual(deep.home.width, 32.0, delta=2.0)
        self.assertLess(deep.home.hull_area, 700.0)
        self.assertEqual(deep.home.n_visible_outfield, 10)
        self.assertEqual(deep.home.n_behind_ball, 10)

    def test_team_relative_frame_for_away(self):
        # AWAY attacks -x; their build-up players sit near their OWN goal, so
        # AWAY-relative centroid_x must be small (deep), not ~80 (high).
        series = extract(synthetic.high_press_scenario())
        mid = next(f for f in series.frames if abs(f.timestamp_s - 2.0) < 0.01)
        self.assertIsNotNone(mid.away.centroid_x)
        self.assertLess(mid.away.centroid_x, 40.0)

    def test_shape_none_when_too_few_visible(self):
        series = extract(synthetic.high_press_low_visibility_scenario())
        mid = next(f for f in series.frames if abs(f.timestamp_s - 12.0) < 0.01)
        self.assertEqual(mid.home.n_visible_outfield, 4)
        self.assertIsNone(mid.home.def_line_height)
        self.assertIsNone(mid.home.hull_area)
        self.assertLess(mid.quality.score, 0.6)


class SegmentationTest(unittest.TestCase):
    def test_broadcast_gap_splits_segments(self):
        match = synthetic.counter_attack_scenario()
        # simulate a 3s broadcast cut from t=3..6
        match = MatchTracking(
            meta=match.meta,
            frames=[f for f in match.frames if not (3.0 < f.timestamp_s < 6.0)],
        )
        series = extract(match)
        segment_ids = {f.segment_id for f in series.frames}
        self.assertEqual(len(segment_ids), 2)
        self.assertLess(series.observed_seconds(), 23.5)

    def test_possession_resets_across_segments(self):
        match = synthetic.counter_attack_scenario()
        match = MatchTracking(
            meta=match.meta,
            frames=[f for f in match.frames if not (3.0 < f.timestamp_s < 6.0)],
        )
        series = extract(match)
        # the flip should still be found inside the second segment
        flips = [
            f for f in series.frames if f.possession.turnover_won_by is TeamSide.HOME
        ]
        self.assertEqual(len(flips), 1)


class PressureSignalsTest(unittest.TestCase):
    def test_pressers_register_and_close(self):
        series = extract(synthetic.high_press_scenario())
        late = next(f for f in series.frames if abs(f.timestamp_s - 12.0) < 0.01)
        self.assertIsNotNone(late.home.nearest_defender_dist)
        self.assertLess(late.home.nearest_defender_dist, 8.0)
        self.assertGreaterEqual(late.home.defenders_within_15m, 4)
        self.assertIsNotNone(late.home.press_closing_speed)
        self.assertGreater(late.home.press_closing_speed, 0.4)

    def test_no_closing_when_passive(self):
        series = extract(synthetic.passive_buildup_scenario())
        mid = next(f for f in series.frames if abs(f.timestamp_s - 10.0) < 0.01)
        self.assertIsNotNone(mid.home.press_closing_speed)
        self.assertAlmostEqual(mid.home.press_closing_speed, 0.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
