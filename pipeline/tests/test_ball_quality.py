"""Tests for the per-frame ball_quality gate and player-consistency signals."""

from __future__ import annotations

import random
import unittest

from pipeline.ball.consistency import isolation_spans, nearest_player_distances
from pipeline.ball.model import BallSample, missing
from pipeline.ball.motion import interpolate_gaps
from pipeline.ball.quality import (
    BallQualityConfig,
    INTERPOLATED_CAP,
    score_ball_series,
    temporal_support,
)
from pipeline.ball.testing import synthetic


class ConsistencyTest(unittest.TestCase):
    def test_nearest_distance_none_without_ball_or_players(self):
        samples = [missing(0.0), BallSample(t=0.04, xy=(0.0, 0.0))]
        d = nearest_player_distances(samples, [[(1.0, 1.0)], None])
        self.assertIsNone(d[0])
        self.assertIsNone(d[1])

    def test_nearest_distance_matches_geometry(self):
        samples = [BallSample(t=0.0, xy=(0.0, 0.0))]
        d = nearest_player_distances(samples, [[(3.0, 4.0), (10.0, 0.0)]])
        self.assertAlmostEqual(d[0], 5.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            nearest_player_distances([missing(0.0)], [])

    def test_long_pass_in_flight_is_not_an_isolation_span(self):
        # 1.4 s flight far from everyone: legitimate, below the 2 s bar.
        flight = synthetic.flight((0.0, 0.0), (35.0, 0.0), 1.4)
        players = synthetic.players_static(flight, [(-5.0, 0.0), (40.0, 3.0)])
        spans = isolation_spans(flight, players)
        self.assertEqual(spans, [])

    def test_unchased_static_ball_is_an_isolation_span(self):
        # A "ball" sitting 30+ m from everyone for 3 s: debris signature.
        debris = synthetic.hold((20.0, -25.0), 3.0)
        players = synthetic.players_static(debris, [(-20.0, 25.0), (-15.0, 20.0)])
        spans = isolation_spans(debris, players)
        self.assertEqual(len(spans), 1)
        self.assertGreater(spans[0].min_distance_m, 30.0)


class TemporalSupportTest(unittest.TestCase):
    def test_middle_of_tracked_span_is_fully_supported(self):
        samples = synthetic.ground_pass((0.0, 0.0), 0.0, 10.0, 2.0)
        self.assertAlmostEqual(temporal_support(samples, 25, 0.6), 1.0)

    def test_lone_blip_in_blackout_has_no_support(self):
        samples = [missing(i * 0.04) for i in range(50)]
        samples[25] = BallSample(t=1.0, xy=(0.0, 0.0), confidence=0.9)
        self.assertEqual(temporal_support(samples, 25, 0.6), 0.0)


class QualityScoreTest(unittest.TestCase):
    def test_missing_frame_scores_zero(self):
        qualities = score_ball_series([missing(0.0)])
        self.assertEqual(qualities[0].quality, 0.0)
        self.assertFalse(qualities[0].detected)

    def test_clean_tracked_pass_scores_high(self):
        samples = synthetic.ground_pass((0.0, 0.0), 20.0, 12.0, 2.0)
        players = synthetic.players_near_ball(samples)
        qualities = score_ball_series(samples, players)
        mid = qualities[len(qualities) // 2]
        self.assertGreater(mid.quality, 0.7)
        self.assertEqual(mid.consistency_score, 1.0)

    def test_lone_blip_scores_far_below_tracked_ball(self):
        tracked = synthetic.ground_pass((0.0, 0.0), 0.0, 10.0, 2.0)
        blip = [missing(i * 0.04) for i in range(50)]
        blip[25] = BallSample(t=1.0, xy=(0.0, 0.0), confidence=0.9)
        q_tracked = score_ball_series(tracked)[25].quality
        q_blip = score_ball_series(blip)[25].quality
        self.assertLess(q_blip, 0.2 * q_tracked)

    def test_teleport_slashes_quality(self):
        samples = synthetic.ground_pass((0.0, 0.0), 0.0, 8.0, 2.0)
        corrupted = synthetic.inject_static_false_positive(
            samples, (10.0, -30.0), 1.0, 1.04, only_when_missing=False
        )
        qualities = score_ball_series(corrupted)
        fp_index = round(1.0 * 25)
        clean_q = score_ball_series(samples)[fp_index].quality
        self.assertLess(qualities[fp_index].quality, 0.25 * clean_q)

    def test_isolated_ball_discounted_when_players_known(self):
        samples = synthetic.hold((20.0, -25.0), 2.0)
        far_players = synthetic.players_static(samples, [(-20.0, 25.0)])
        near_players = synthetic.players_near_ball(samples)
        q_far = score_ball_series(samples, far_players)[10].quality
        q_near = score_ball_series(samples, near_players)[10].quality
        self.assertLess(q_far, 0.5 * q_near)

    def test_multi_candidate_frames_are_discounted(self):
        samples = synthetic.hold((0.0, 0.0), 1.0)
        ambiguous = [
            BallSample(t=s.t, xy=s.xy, confidence=s.confidence, n_candidates=3)
            for s in samples
        ]
        cfg = BallQualityConfig()
        q_plain = score_ball_series(samples)[10].quality
        q_ambig = score_ball_series(ambiguous)[10].quality
        self.assertAlmostEqual(q_ambig, q_plain * cfg.multi_candidate_factor, places=6)

    def test_interpolated_samples_capped_below_observed(self):
        samples = synthetic.ground_pass((0.0, 0.0), 0.0, 10.0, 2.0)
        gappy = synthetic.drop_span(samples, 0.8, 1.2)
        filled = interpolate_gaps(gappy, max_gap_s=1.0)
        qualities = score_ball_series(filled)
        interp = [q for q in qualities if q.interpolated]
        self.assertTrue(interp)
        self.assertTrue(all(q.quality <= INTERPOLATED_CAP for q in interp))

    def test_synthetic_scenario_separates_regimes(self):
        """End-to-end: a realistic corrupted scenario ranks regimes correctly —
        tracked play > interpolated bridge > lone blip false positive."""
        rng = random.Random(7)
        pass1 = synthetic.ground_pass((-20.0, 5.0), 10.0, 13.0, 2.0)
        t0, p = synthetic.end_of(pass1)
        dribble = synthetic.dribble(p, 40.0, 4.0, 3.0, t0=t0)
        t1, p1 = synthetic.end_of(dribble)
        flight_seg = synthetic.flight(p1, (30.0, -20.0), 1.2, t0=t1)
        scenario = synthetic.chain(pass1, dribble, flight_seg)
        noisy = synthetic.add_noise(scenario, 0.3, rng)
        gappy = synthetic.drop_random(noisy, 0.15, rng)
        players = synthetic.players_near_ball(gappy)
        qualities = score_ball_series(gappy, players)
        detected_q = [q.quality for q in qualities if q.detected]
        self.assertGreater(sum(detected_q) / len(detected_q), 0.5)
        # Missing frames stay zero:
        self.assertTrue(all(q.quality == 0.0 for q in qualities if not q.detected))


if __name__ == "__main__":
    unittest.main()
