"""Tests for labelled ball ground truth vs probe detections (pixel-space eval).

Synthetic throughout: a hand-built ground-truth document and hand-built probe
frames. The point is to pin the *arithmetic and the join*, especially the two
things that are easy to get quietly wrong — absent-labelled frames scoring
detections as false positives (not as skips), and a frame-index join that must
fail loudly rather than score a shifted pairing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.ball.evaluation import (
    SCHEMA,
    BallGroundTruth,
    GroundTruthLabel,
    evaluate_detections,
    format_sweep,
    load_ground_truth,
    parse_ground_truth,
    sweep,
)
from pipeline.ball.ingest import ProbeFrame

WIDTH, HEIGHT = 1280, 720


def _box(cx, cy, conf, size=8.0):
    """A probe candidate centred on (cx, cy) — the comparison point for eval."""
    h = size / 2.0
    return (cx - h, cy - h, cx + h, cy + h, conf)


def _gt(labels, n_frames_total=None, sampled=()):
    return BallGroundTruth(
        labels=tuple(labels),
        width=WIDTH,
        height=HEIGHT,
        n_frames_total=n_frames_total,
        stride=8,
        sampled_indices=tuple(sampled),
    )


def _doc(labels, **over):
    doc = {
        "schema": SCHEMA,
        "clip_id": "synthetic",
        "resolution": {"width": WIDTH, "height": HEIGHT},
        "n_frames_total": 24,
        "stride": 8,
        "sampled_indices": [0, 8, 16],
        "labels": labels,
    }
    doc.update(over)
    return doc


class GroundTruthParsingTest(unittest.TestCase):
    def test_parses_visible_and_absent_labels(self):
        gt = parse_ground_truth(
            _doc(
                [
                    {"index": 0, "frame": "000001", "status": "visible", "x": 100.5, "y": 200.25},
                    {"index": 8, "frame": "000009", "status": "absent", "x": None, "y": None},
                ]
            )
        )
        self.assertEqual(gt.resolution, (WIDTH, HEIGHT))
        self.assertEqual(gt.stride, 8)
        self.assertTrue(gt.labels[0].visible)
        self.assertEqual(gt.labels[0].xy, (100.5, 200.25))
        self.assertFalse(gt.labels[1].visible)
        self.assertIsNone(gt.labels[1].xy)

    def test_missing_resolution_is_rejected_not_assumed(self):
        doc = _doc([])
        del doc["resolution"]
        with self.assertRaises(ValueError) as ctx:
            parse_ground_truth(doc)
        self.assertIn("resolution", str(ctx.exception))

    def test_wrong_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_ground_truth(_doc([], schema="pattern-events/v1"))

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_ground_truth(_doc([{"index": 0, "status": "maybe", "x": 1, "y": 2}]))

    def test_label_clicked_at_another_resolution_is_rejected(self):
        # Re-extracting the same folder name at a different size is invisible
        # downstream: the rescaled point is still inside the image, so it reads
        # as a blind detector rather than as a broken pixel frame.
        with self.assertRaises(ValueError) as ctx:
            parse_ground_truth(
                _doc([{"index": 0, "status": "visible", "x": 960.0, "y": 540.0,
                       "width": 1920, "height": 1080}])
            )
        self.assertIn("different", str(ctx.exception))

    def test_matching_per_label_resolution_is_accepted(self):
        gt = parse_ground_truth(
            _doc([{"index": 0, "status": "visible", "x": 640.0, "y": 360.0,
                   "width": WIDTH, "height": HEIGHT}])
        )
        self.assertEqual(gt.labels[0].xy, (640.0, 360.0))

    def test_missing_coordinates_raise_value_error_not_key_error(self):
        with self.assertRaises(ValueError):
            parse_ground_truth(_doc([{"index": 0, "status": "visible"}]))
        with self.assertRaises(ValueError):
            parse_ground_truth(_doc([{"status": "absent"}]))

    def test_duplicate_index_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_ground_truth(
                _doc(
                    [
                        {"index": 4, "status": "absent"},
                        {"index": 4, "status": "visible", "x": 1.0, "y": 2.0},
                    ]
                )
            )

    def test_round_trips_through_a_file(self):
        doc = _doc([{"index": 16, "status": "visible", "x": 640.0, "y": 360.0}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.json"
            path.write_text(json.dumps(doc))
            gt = load_ground_truth(str(path))
        self.assertEqual(gt.labels[0].index, 16)
        self.assertEqual(gt.n_frames_total, 24)
        self.assertEqual(gt.sampled_indices, (0, 8, 16))


class DetectionEvaluationTest(unittest.TestCase):
    def test_perfect_detector_scores_one(self):
        gt = _gt(
            [
                GroundTruthLabel(0, (100.0, 100.0)),
                GroundTruthLabel(8, (200.0, 150.0)),
                GroundTruthLabel(16, None),
            ]
        )
        probe = [
            ProbeFrame(0, (_box(101.0, 99.0, 0.5),)),
            ProbeFrame(8, (_box(200.0, 150.0, 0.6),)),
            ProbeFrame(16, ()),
        ]
        rep = evaluate_detections(gt, probe)
        self.assertEqual((rep.true_positives, rep.false_positives), (2, 0))
        self.assertEqual((rep.false_negatives, rep.true_negatives), (0, 1))
        self.assertEqual(rep.precision, 1.0)
        self.assertEqual(rep.recall, 1.0)
        self.assertEqual(rep.f1, 1.0)
        self.assertEqual(rep.absent_frame_fp_rate, 0.0)

    def test_detection_on_an_absent_frame_is_a_false_positive(self):
        # The requirement absent-labels exist for: nothing else in the package
        # can tell "the detector found debris" from "the detector found the ball".
        gt = _gt([GroundTruthLabel(0, None), GroundTruthLabel(8, None)])
        probe = [
            ProbeFrame(0, (_box(500.0, 500.0, 0.5), _box(510.0, 505.0, 0.4))),
            ProbeFrame(8, ()),
        ]
        rep = evaluate_detections(gt, probe, selection="any")
        self.assertEqual(rep.false_positives, 2)
        self.assertEqual(rep.true_negatives, 1)
        self.assertEqual(rep.n_absent, 2)
        self.assertEqual(rep.absent_frame_fp_rate, 0.5)
        self.assertEqual(rep.absent_frame_candidates_per_frame, 1.0)
        self.assertIsNone(rep.recall)  # no visible-ball labels: recall undefined
        self.assertEqual(rep.precision, 0.0)

    def test_skipped_frames_are_reported_but_never_scored(self):
        # Frames 8 and 16 were shown under the stride and skipped: they must not
        # become silent true negatives (which would inflate precision for free).
        gt = _gt([GroundTruthLabel(0, (10.0, 10.0))], sampled=(0, 8, 16))
        probe = [
            ProbeFrame(0, (_box(10.0, 10.0, 0.9),)),
            ProbeFrame(8, (_box(400.0, 400.0, 0.9),)),
            ProbeFrame(16, ()),
        ]
        rep = evaluate_detections(gt, probe)
        self.assertEqual((rep.n_sampled, rep.n_labelled, rep.n_skipped), (3, 1, 2))
        self.assertEqual(rep.false_positives, 0)  # frame 8's detection is unscored
        self.assertEqual(rep.true_negatives, 0)  # frame 16 is not a labelled absence

    def test_near_miss_beyond_tolerance_is_both_a_miss_and_a_false_positive(self):
        gt = _gt([GroundTruthLabel(0, (100.0, 100.0))])
        probe = [ProbeFrame(0, (_box(130.0, 100.0, 0.9),))]  # 30 px away: confetti
        rep = evaluate_detections(gt, probe, tolerance_px=8.0)
        self.assertEqual(rep.true_positives, 0)
        self.assertEqual(rep.false_negatives, 1)
        self.assertEqual(rep.false_positives, 1)
        self.assertEqual(rep.recall, 0.0)
        self.assertEqual(rep.precision, 0.0)
        self.assertEqual(rep.f1, 0.0)  # defined and bad, not undefined

    def test_candidate_pressure_on_absent_frames_survives_best_selection(self):
        # selection="best" scores one box per frame, but the debris *pressure*
        # is a property of the footage: all three candidates must be counted.
        gt = _gt([GroundTruthLabel(0, None)])
        probe = [ProbeFrame(0, (_box(1.0, 1.0, 0.5), _box(2.0, 2.0, 0.4), _box(3.0, 3.0, 0.05)))]
        rep = evaluate_detections(gt, probe, conf_floor=0.1, selection="best")
        self.assertEqual(rep.false_positives, 1)  # one box scored
        self.assertEqual(rep.absent_frame_candidates_per_frame, 2.0)  # two above the floor
        self.assertEqual(rep.absent_frame_fp_rate, 1.0)

    def test_tolerance_sets_the_verdict(self):
        gt = _gt([GroundTruthLabel(0, (100.0, 100.0))])
        probe = [ProbeFrame(0, (_box(106.0, 100.0, 0.9),))]
        self.assertEqual(evaluate_detections(gt, probe, tolerance_px=4.0).true_positives, 0)
        tight = evaluate_detections(gt, probe, tolerance_px=8.0)
        self.assertEqual(tight.true_positives, 1)
        self.assertAlmostEqual(tight.median_error_px, 6.0)

    def test_best_vs_any_separates_selection_error_from_detection_error(self):
        # The ball IS detected, at lower confidence than a piece of debris.
        gt = _gt([GroundTruthLabel(0, (100.0, 100.0))])
        probe = [ProbeFrame(0, (_box(100.0, 100.0, 0.2), _box(600.0, 300.0, 0.8)))]
        best = evaluate_detections(gt, probe, conf_floor=0.1, selection="best")
        any_ = evaluate_detections(gt, probe, conf_floor=0.1, selection="any")
        self.assertEqual(best.recall, 0.0)  # highest-conf candidate is the debris
        self.assertEqual(any_.recall, 1.0)  # the detector did see it
        self.assertEqual(any_.false_positives, 1)  # ...and the debris still costs

    def test_conf_floor_filters_candidates(self):
        gt = _gt([GroundTruthLabel(0, (100.0, 100.0))])
        probe = [ProbeFrame(0, (_box(100.0, 100.0, 0.15),))]
        self.assertEqual(evaluate_detections(gt, probe, conf_floor=0.1).recall, 1.0)
        strict = evaluate_detections(gt, probe, conf_floor=0.3)
        self.assertEqual(strict.recall, 0.0)
        self.assertEqual(strict.false_positives, 0)  # filtered out, not mis-scored

    def test_localisation_error_stats(self):
        gt = _gt([GroundTruthLabel(i, (100.0, 100.0)) for i in range(4)])
        probe = [ProbeFrame(i, (_box(100.0 + d, 100.0, 0.9),)) for i, d in enumerate([0, 1, 2, 5])]
        rep = evaluate_detections(gt, probe, tolerance_px=8.0)
        self.assertEqual(rep.true_positives, 4)
        self.assertAlmostEqual(rep.median_error_px, 1.5)
        self.assertAlmostEqual(rep.mean_error_px, 2.0)
        self.assertAlmostEqual(rep.p90_error_px, 4.1)

    def test_undefined_metrics_are_none_not_zero(self):
        gt = _gt([GroundTruthLabel(0, (100.0, 100.0))])
        rep = evaluate_detections(gt, [ProbeFrame(0, ())])
        self.assertIsNone(rep.precision)  # no detections at all
        self.assertEqual(rep.recall, 0.0)
        self.assertIsNone(rep.f1)
        self.assertIsNone(rep.median_error_px)
        self.assertIsNone(rep.absent_frame_fp_rate)  # no absent labels


class JoinIntegrityTest(unittest.TestCase):
    def test_frame_count_mismatch_refuses_to_score(self):
        gt = _gt([GroundTruthLabel(0, (10.0, 10.0))], n_frames_total=858)
        probe = [ProbeFrame(0, (_box(10.0, 10.0, 0.9),))]
        with self.assertRaises(ValueError) as ctx:
            evaluate_detections(gt, probe)
        self.assertIn("frame-count mismatch", str(ctx.exception))

    def test_matching_frame_count_scores_normally(self):
        gt = _gt([GroundTruthLabel(0, (10.0, 10.0))], n_frames_total=2)
        probe = [ProbeFrame(0, (_box(10.0, 10.0, 0.9),)), ProbeFrame(1, ())]
        self.assertEqual(evaluate_detections(gt, probe).true_positives, 1)

    def test_label_without_a_probe_frame_is_an_error(self):
        gt = _gt([GroundTruthLabel(99, (10.0, 10.0))])
        with self.assertRaises(ValueError):
            evaluate_detections(gt, [ProbeFrame(0, ())])

    def test_same_index_different_filename_is_refused(self):
        # Two folders of equal length join cleanly on index and score as a blind
        # detector. The stem both sides recorded is what catches it.
        gt = _gt([GroundTruthLabel(0, (10.0, 10.0), frame="000001")], n_frames_total=1)
        probe = [ProbeFrame(0, (_box(10.0, 10.0, 0.9),), frame="004301")]
        with self.assertRaises(ValueError) as ctx:
            evaluate_detections(gt, probe)
        self.assertIn("frame-name mismatch", str(ctx.exception))

    def test_matching_filenames_score_normally(self):
        gt = _gt([GroundTruthLabel(0, (10.0, 10.0), frame="000001")], n_frames_total=1)
        probe = [ProbeFrame(0, (_box(10.0, 10.0, 0.9),), frame="000001")]
        self.assertEqual(evaluate_detections(gt, probe).true_positives, 1)

    def test_stem_check_is_skipped_when_either_side_lacks_one(self):
        # Older probe dumps and hand-built fixtures carry no stem; absence must
        # not become an error, only a missed opportunity to verify.
        gt = _gt([GroundTruthLabel(0, (10.0, 10.0), frame="000001")], n_frames_total=1)
        probe = [ProbeFrame(0, (_box(10.0, 10.0, 0.9),), frame=None)]
        self.assertEqual(evaluate_detections(gt, probe).true_positives, 1)

    def test_detection_outside_the_labelled_frame_is_counted(self):
        # A resolution mismatch (labels at 1280x720, probe boxes at 1920x1080)
        # surfaces here rather than as an unexplained recall collapse.
        gt = _gt([GroundTruthLabel(0, (100.0, 100.0))])
        probe = [ProbeFrame(0, (_box(1500.0, 900.0, 0.9),))]
        self.assertEqual(evaluate_detections(gt, probe).out_of_frame_detections, 1)


class SweepTest(unittest.TestCase):
    def test_sweep_reports_one_row_per_floor_and_precision_rises(self):
        gt = _gt(
            [
                GroundTruthLabel(0, (100.0, 100.0)),
                GroundTruthLabel(1, None),
            ]
        )
        probe = [
            ProbeFrame(0, (_box(100.0, 100.0, 0.5),)),
            ProbeFrame(1, (_box(600.0, 600.0, 0.15),)),  # low-confidence debris
        ]
        reports = sweep(gt, probe, conf_floors=(0.1, 0.3))
        self.assertEqual([r.conf_floor for r in reports], [0.1, 0.3])
        self.assertEqual(reports[0].precision, 0.5)  # debris survives the floor
        self.assertEqual(reports[1].precision, 1.0)  # ...and is filtered at 0.3
        self.assertEqual(reports[1].recall, 1.0)

    def test_format_sweep_renders_a_row_per_report(self):
        gt = _gt([GroundTruthLabel(0, (100.0, 100.0))])
        probe = [ProbeFrame(0, (_box(100.0, 100.0, 0.9),))]
        text = format_sweep(sweep(gt, probe, conf_floors=(0.1, 0.2)))
        self.assertEqual(len(text.splitlines()), 4)  # header + rule + 2 rows
        self.assertIn("conf", text)


if __name__ == "__main__":
    unittest.main()
