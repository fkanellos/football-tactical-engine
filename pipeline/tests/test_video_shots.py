"""Tests for shot-boundary detection and segmentation."""

from __future__ import annotations

import unittest

from pipeline.video.shots import (
    FrameSignature,
    adaptive_threshold,
    consecutive_distances,
    detect_shot_boundaries,
    histogram_distance,
    segment_for_frame,
    segments_from_boundaries,
    shot_report,
)
from pipeline.video.testing import synthetic


class HistogramDistanceTest(unittest.TestCase):
    def test_identical_histograms_are_zero(self):
        h = (0.2, 0.3, 0.5)
        self.assertEqual(histogram_distance(h, h), 0.0)

    def test_disjoint_histograms_are_one(self):
        self.assertAlmostEqual(histogram_distance((1.0, 0.0), (0.0, 1.0)), 1.0)

    def test_half_the_mass_moving_reads_as_half(self):
        self.assertAlmostEqual(histogram_distance((1.0, 0.0), (0.5, 0.5)), 0.5)

    def test_symmetric(self):
        a, b = (0.7, 0.2, 0.1), (0.1, 0.6, 0.3)
        self.assertAlmostEqual(histogram_distance(a, b), histogram_distance(b, a))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            histogram_distance((0.5, 0.5), (1.0,))


class SignatureValidationTest(unittest.TestCase):
    def test_empty_histogram_rejected(self):
        with self.assertRaises(ValueError):
            FrameSignature(0, 0.0, ())


class DistanceSeriesTest(unittest.TestCase):
    def test_first_distance_is_zero_by_convention(self):
        sigs, _ = synthetic.cut_sequence()
        self.assertEqual(consecutive_distances(sigs)[0], 0.0)

    def test_series_length_matches_frames(self):
        sigs, _ = synthetic.cut_sequence()
        self.assertEqual(len(consecutive_distances(sigs)), len(sigs))

    def test_empty_input(self):
        self.assertEqual(consecutive_distances([]), [])


class DetectionTest(unittest.TestCase):
    def test_single_shot_yields_no_boundaries(self):
        sigs, truth = synthetic.single_shot()
        self.assertEqual(truth, [])
        self.assertEqual(detect_shot_boundaries(sigs), [])

    def test_hard_cuts_are_found_exactly(self):
        sigs, truth = synthetic.cut_sequence(shot_lengths=(60, 80, 50))
        found = [b.index for b in detect_shot_boundaries(sigs)]
        self.assertEqual(found, truth)

    def test_many_cuts_found_across_seeds(self):
        # The one test that would catch a threshold tuned to a single fixture.
        for seed in range(6):
            sigs, truth = synthetic.cut_sequence(
                shot_lengths=(40, 55, 35, 70, 45), seed=seed
            )
            found = [b.index for b in detect_shot_boundaries(sigs)]
            self.assertEqual(found, truth, f"seed {seed}")

    def test_pan_is_not_a_cut(self):
        sigs, truth = synthetic.pan_sequence(drift_per_frame=0.03)
        self.assertEqual(truth, [])
        self.assertEqual(detect_shot_boundaries(sigs), [])

    def test_flash_does_not_split_a_shot(self):
        sigs, truth = synthetic.flash_sequence()
        self.assertEqual(truth, [])
        # Without min_shot_frames a flash reads as two cuts one frame apart.
        naive = detect_shot_boundaries(sigs, min_shot_frames=1)
        self.assertGreaterEqual(len(naive), 2)
        self.assertEqual(detect_shot_boundaries(sigs), [])

    def test_slow_dissolve_is_missed_which_is_the_documented_limitation(self):
        # NOT an aspiration test. A 15-frame cross-fade moves ~7% of the colour
        # mass per frame, under any threshold that leaves a camera pan alone, so
        # a frame-to-frame method cannot see it and this module says so in its
        # docstring. Pinned as a test because the limitation has a downstream
        # cost — a segment silently spans the transition — and a future learned
        # detector (TransNetV2-class) should make this test fail loudly when it
        # lands, rather than slipping in unnoticed.
        sigs, _ = synthetic.dissolve_sequence(shot_len=80, dissolve_frames=15)
        self.assertEqual(detect_shot_boundaries(sigs), [])

    def test_fast_dissolve_is_caught_and_lands_inside_the_transition(self):
        # The boundary between "cut" and "dissolve" is a speed, not a kind. A
        # 3-frame fade moves enough mass per frame to clear the threshold.
        shot_len, dissolve = 80, 3
        sigs, _ = synthetic.dissolve_sequence(shot_len, dissolve)
        found = detect_shot_boundaries(sigs)
        self.assertEqual(len(found), 1)
        # Frame-exact accuracy is not claimed for gradual transitions.
        self.assertGreaterEqual(found[0].index, shot_len)
        self.assertLessEqual(found[0].index, shot_len + dissolve + 1)

    def test_rapid_montage_is_not_mistaken_for_a_transient(self):
        # The counterpart to the flash test, and the reason transient rejection
        # compares endpoints instead of just counting crossings. Three short
        # shots cluster exactly like a flash does; what separates them is that
        # the scene does NOT come back. Rejecting these would silently swallow
        # replay montages, which are precisely the frames that must not be fed
        # to the tactical layer as continuous play.
        sigs, truth = synthetic.cut_sequence(shot_lengths=(40, 5, 5, 40))
        found = detect_shot_boundaries(sigs)
        self.assertGreater(len(found), 0)
        # Sub-min_shot_frames shots collapse, so not every truth boundary
        # survives — but the clip must not read as one continuous take.
        self.assertLessEqual(len(found), len(truth))
        segs = segments_from_boundaries(sigs, found)
        self.assertGreater(len(segs), 1)

    def test_too_few_frames_is_not_an_error(self):
        sigs, _ = synthetic.single_shot(n_frames=1)
        self.assertEqual(detect_shot_boundaries(sigs), [])
        self.assertEqual(detect_shot_boundaries([]), [])

    def test_boundary_records_what_it_had_to_clear(self):
        sigs, _ = synthetic.cut_sequence()
        b = detect_shot_boundaries(sigs)[0]
        self.assertGreater(b.distance, b.threshold)
        self.assertGreater(b.margin, 1.0)


class ThresholdTest(unittest.TestCase):
    def test_absolute_floor_protects_a_quiet_series(self):
        # A perfectly static shot has median and MAD of ~0; without the floor,
        # any compression noise clears the adaptive term.
        sigs = [FrameSignature(i, i / 25.0, (0.5, 0.5)) for i in range(50)]
        self.assertGreaterEqual(adaptive_threshold(consecutive_distances(sigs)), 0.15)

    def test_threshold_rises_with_a_noisier_series(self):
        quiet, _ = synthetic.single_shot(noise=0.01, seed=1)
        noisy, _ = synthetic.single_shot(noise=0.30, seed=1)
        self.assertGreater(
            adaptive_threshold(consecutive_distances(noisy), min_distance=0.0),
            adaptive_threshold(consecutive_distances(quiet), min_distance=0.0),
        )

    def test_degenerate_inputs(self):
        self.assertEqual(adaptive_threshold([]), 0.15)
        self.assertEqual(adaptive_threshold([0.0]), 0.15)


class SegmentationTest(unittest.TestCase):
    def test_segments_tile_the_clip_without_gaps_or_overlap(self):
        sigs, _ = synthetic.cut_sequence(shot_lengths=(60, 80, 50))
        segs = segments_from_boundaries(sigs, detect_shot_boundaries(sigs))
        self.assertEqual(segs[0].start_index, 0)
        self.assertEqual(segs[-1].end_index, len(sigs))
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(a.end_index, b.start_index)

    def test_segment_frame_counts_match_the_shots(self):
        lengths = (60, 80, 50)
        sigs, _ = synthetic.cut_sequence(shot_lengths=lengths)
        segs = segments_from_boundaries(sigs, detect_shot_boundaries(sigs))
        self.assertEqual([s.n_frames for s in segs], list(lengths))

    def test_single_shot_is_one_segment_spanning_everything(self):
        sigs, _ = synthetic.single_shot(n_frames=120)
        segs = segments_from_boundaries(sigs, [])
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].n_frames, 120)

    def test_end_time_is_the_last_frame_not_the_cut(self):
        sigs, _ = synthetic.cut_sequence(shot_lengths=(60, 40))
        segs = segments_from_boundaries(sigs, detect_shot_boundaries(sigs))
        self.assertAlmostEqual(segs[0].end_time_s, sigs[59].timestamp_s)
        self.assertAlmostEqual(segs[1].start_time_s, sigs[60].timestamp_s)

    def test_lookup_finds_the_owning_segment(self):
        sigs, _ = synthetic.cut_sequence(shot_lengths=(60, 80))
        segs = segments_from_boundaries(sigs, detect_shot_boundaries(sigs))
        self.assertIs(segment_for_frame(segs, 0), segs[0])
        self.assertIs(segment_for_frame(segs, 59), segs[0])
        self.assertIs(segment_for_frame(segs, 60), segs[1])

    def test_lookup_outside_every_segment_is_none_not_a_guess(self):
        sigs, _ = synthetic.single_shot(n_frames=50)
        segs = segments_from_boundaries(sigs, [])
        self.assertIsNone(segment_for_frame(segs, 999))
        self.assertIsNone(segment_for_frame(segs, -1))

    def test_empty_input(self):
        self.assertEqual(segments_from_boundaries([], []), [])


class ReportTest(unittest.TestCase):
    def test_single_shot_is_flagged(self):
        sigs, _ = synthetic.single_shot(n_frames=250)
        report = shot_report(sigs)
        self.assertTrue(report.single_shot)
        self.assertEqual(report.n_boundaries, 0)
        self.assertEqual(report.cuts_per_minute, 0.0)

    def test_multi_shot_report_counts_and_rates(self):
        sigs, truth = synthetic.cut_sequence(shot_lengths=(60, 80, 50))
        report = shot_report(sigs)
        self.assertFalse(report.single_shot)
        self.assertEqual(report.n_boundaries, len(truth))
        self.assertEqual(report.n_segments, len(truth) + 1)
        self.assertEqual(report.longest_segment.n_frames, 80)
        self.assertGreater(report.cuts_per_minute, 0.0)

    def test_cuts_per_minute_matches_hand_arithmetic(self):
        # 3 shots of 250 frames at 25 fps = 30 s span, 2 cuts => 4.0 per minute.
        sigs, _ = synthetic.cut_sequence(shot_lengths=(250, 250, 250))
        report = shot_report(sigs)
        self.assertEqual(report.n_boundaries, 2)
        self.assertAlmostEqual(report.cuts_per_minute, 2 / (report.duration_s / 60), 6)


if __name__ == "__main__":
    unittest.main()
