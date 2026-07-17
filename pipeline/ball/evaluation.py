"""Hand-labelled ball ground truth vs probe detections: real recall/precision.

Everything else in this package measures the ball stream *without* labels — that
is the point of the coverage/plausibility/consistency layer (design §6). This
module is the one place labels exist, and it buys the one thing no self-
consistency signal can: whether a detection is actually the ball, and whether a
frame with no detection actually had one. ~100 hand-clicked frames
(``tools/annotator.html`` ball mode, design §6.4/§6.6) turn the probe's
"detection rate" into measured recall and precision, and pin the §6.2 quality
ramps to something real.

**Pixel space, deliberately.** Ground truth is the ball centre in the frame's
native decoded pixels, and detections are compared to it there — never in pitch
coordinates. Going through the homography would fold calibration error (and the
airborne ground-plane bias, ingest.py's module docstring) into what is supposed
to be a *detector* measurement. Two failures, one number, no way to tell them
apart: exactly the mistake this repo keeps writing docs about.

**The join key is the frame index** — position in the sorted frames folder,
identically defined by ``research/ball_probe.py`` (``sorted(Path.iterdir())``
filtered to image suffixes, enumerated from 0) and by the annotator (sorted file
list, same filter, same enumeration). ``evaluate_detections`` verifies the join
rather than assuming it: both files also record the source filename *stem* per
frame, so a disagreement on frame count, or on which filename an index refers
to, is a hard error. Two folders of 858 frames join "successfully" on index and
produce a plausible-looking 2% recall table; the stem check turns that into an
exception instead of a conclusion about the detector.

**Absent frames are evidence.** A frame labelled "no ball visible" is not a
skipped frame: any detection there is a false positive, and that is the only way
the debris hypothesis (H2, design §4) gets a real precision number. The label
schema keeps skip and absent distinct; so does the arithmetic here — skipped
frames are counted and reported, never scored.

**Localisation comparison uses the bbox *centre*.** ingest.py projects the bbox
*bottom-centre* because that is the ground-contact point the homography wants;
here the question is "did the detector find the ball", so centre-vs-centre is
the right comparison and the tolerance absorbs the rest.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ingest import ProbeFrame
from .model import Point2, median

SCHEMA = "ball-ground-truth/v1"

#: Default match radius. The ball is 4-10 px at our framing (design §3), so this
#: is roughly one ball diameter at the favourable end: tight enough that a
#: confetti hit 30 px away cannot be scored as the ball, loose enough to absorb
#: hand-click error on a 4 px blob. WASB reports soccer F1 at a 4 px tolerance
#: (design §5.2) — sweep both before quoting a number against the literature.
DEFAULT_TOLERANCE_PX = 8.0


@dataclass(frozen=True)
class GroundTruthLabel:
    """One hand-labelled frame.

    ``xy is None`` means the annotator looked and the ball was *not visible* —
    a positive claim about the frame, not an absence of work. Frames the
    annotator skipped are not labels at all and never reach this type.
    """

    index: int
    xy: Optional[Point2]  # ball centre in native pixels; None = ball not visible
    frame: Optional[str] = None  # source filename stem, for eyeballing a miss

    @property
    def visible(self) -> bool:
        return self.xy is not None


@dataclass(frozen=True)
class BallGroundTruth:
    """A labelling session: the labels, plus what was offered and what it was of.

    ``resolution`` is recorded, required, and checked. Every ball number this
    repo has ever been confused by traces back to an unstated pixel frame (the
    wrapper's 640 inference default, the radar's hardcoded 1920x1080, the
    calibration dump's fixed 1280x720) — so a ground-truth file without an
    explicit resolution is rejected at load, not assumed.

    ``sampled_indices`` are the frames the annotator was *shown* under the
    uniform stride. Labels are a subset; the difference is skips, which are
    reported and never scored. Uniform sampling is what makes recall mean
    anything: labelling only frames where the ball is easy to see measures the
    annotator, not the detector.
    """

    labels: Tuple[GroundTruthLabel, ...]
    width: int
    height: int
    n_frames_total: Optional[int] = None  # frames in the folder the probe also saw
    stride: Optional[int] = None
    sampled_indices: Tuple[int, ...] = ()
    clip_id: Optional[str] = None
    annotator: Optional[str] = None

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def by_index(self) -> Dict[int, GroundTruthLabel]:
        return {lab.index: lab for lab in self.labels}


@dataclass(frozen=True)
class DetectionEval:
    """Measured detector performance on the labelled frames.

    Counts are *detection*-level for false positives and *frame*-level for the
    rest, which is the only combination that makes both ratios mean what they
    say: a frame with three confetti detections is three chances to be wrong.

    ``precision``/``recall``/``f1`` are None rather than 0.0 when undefined
    (no detections at all, or no visible-ball labels) — a detector that finds
    nothing has undefined precision, not perfect or zero precision, and the
    difference matters when the honest answer is §7.5's "structurally absent".
    """

    conf_floor: float
    tolerance_px: float
    selection: str  # "best" (what ingest does) | "any" (upper bound on recall)

    n_sampled: int
    n_labelled: int
    n_skipped: int
    n_visible: int
    n_absent: int

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]

    #: fraction of *ball-not-visible* frames carrying >= 1 scored detection — the
    #: debris number (H2), and the one measurement only absent-labels can produce
    absent_frame_fp_rate: Optional[float]
    #: mean *candidates above the floor* per absent frame — measured before the
    #: selection policy, so it stays the false-positive-pressure number
    #: (design §6.1's n_candidates) rather than collapsing to the rate above
    #: whenever selection="best" keeps at most one box per frame
    absent_frame_candidates_per_frame: Optional[float]

    median_error_px: Optional[float]
    mean_error_px: Optional[float]
    p90_error_px: Optional[float]

    #: detections outside the ground truth's own frame rectangle: a resolution
    #: mismatch between probe and labels shows up here before it shows up as a
    #: mysteriously bad recall
    out_of_frame_detections: int


def load_ground_truth(path: str) -> BallGroundTruth:
    """Read a ``ball-ground-truth/v1`` JSON export from the annotator."""
    with open(path) as f:
        doc = json.load(f)
    return parse_ground_truth(doc)


def parse_ground_truth(doc: dict) -> BallGroundTruth:
    """Parse an annotator document; raise loudly on anything under-specified."""
    schema = doc.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, got {schema!r}")
    res = doc.get("resolution") or {}
    try:
        width, height = int(res["width"]), int(res["height"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "ground truth carries no explicit resolution; pixel labels without a "
            "stated pixel frame are unusable (design §6.6)"
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"nonsensical resolution {width}x{height}")

    labels: List[GroundTruthLabel] = []
    for raw in doc.get("labels", []):
        status = raw.get("status")
        if status not in ("visible", "absent"):
            raise ValueError(f"label {raw!r}: status must be 'visible' or 'absent'")
        try:
            idx = int(raw["index"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"label {raw!r}: missing or non-integer 'index'")
        if status == "absent":
            xy = None
        else:
            try:
                xy = (float(raw["x"]), float(raw["y"]))
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"label {raw!r}: status 'visible' needs numeric x and y")
        # The annotator stamps the decoded size on every label. If one disagrees
        # with the document's stated resolution, the labels were made across a
        # re-extraction and the coordinates are in two different pixel frames —
        # which is invisible downstream (a rescaled point is still inside the
        # image) and shows up only as an inexplicably blind detector.
        lw, lh = raw.get("width"), raw.get("height")
        if lw is not None and lh is not None and (int(lw), int(lh)) != (width, height):
            raise ValueError(
                f"label {raw!r} was clicked at {int(lw)}x{int(lh)} but the document "
                f"declares {width}x{height}; these coordinates are in different "
                "pixel frames and cannot be scored together"
            )
        labels.append(GroundTruthLabel(index=idx, xy=xy, frame=raw.get("frame")))
    labels.sort(key=lambda lab: lab.index)

    seen = set()
    for lab in labels:
        if lab.index in seen:
            raise ValueError(f"duplicate label for frame index {lab.index}")
        seen.add(lab.index)

    n_total = doc.get("n_frames_total")
    stride = doc.get("stride")
    return BallGroundTruth(
        labels=tuple(labels),
        width=width,
        height=height,
        n_frames_total=int(n_total) if n_total is not None else None,
        stride=int(stride) if stride is not None else None,
        sampled_indices=tuple(int(i) for i in doc.get("sampled_indices", [])),
        clip_id=doc.get("clip_id"),
        annotator=doc.get("annotator"),
    )


def evaluate_detections(
    ground_truth: BallGroundTruth,
    probe_frames: Sequence[ProbeFrame],
    conf_floor: float = 0.1,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    selection: str = "best",
) -> DetectionEval:
    """Score probe detections against hand labels, joined on frame index.

    ``selection="best"`` keeps only the highest-confidence candidate per frame —
    the policy ``ingest.probe_to_samples`` actually applies, so this is the
    number that predicts downstream behaviour. ``selection="any"`` credits a hit
    if *any* candidate lands on the ball: the gap between the two is selection
    error (the detector saw it and we picked debris) as opposed to detection
    error (it was never found), and those have completely different fixes —
    trajectory-gated selection (§7.2) vs a better detector (§7.4).

    Raises when the ground truth and the probe disagree about the frame count:
    both derive the index from sorted position in the same folder, so a mismatch
    means they are not looking at the same folder and every joined number would
    be silently wrong.
    """
    if selection not in ("best", "any"):
        raise ValueError(f"selection must be 'best' or 'any', got {selection!r}")

    n_probe = len(probe_frames)
    if ground_truth.n_frames_total is not None and ground_truth.n_frames_total != n_probe:
        raise ValueError(
            f"frame-count mismatch: ground truth was labelled over "
            f"{ground_truth.n_frames_total} frames, probe output has {n_probe}. "
            "The join key is sorted-position in the frames folder — different "
            "counts mean different folders (or a different extension filter), "
            "and the indices do not line up. Re-extract or re-label; do not "
            "score this pair."
        )

    by_index = {fr.index: fr for fr in probe_frames}
    labels = ground_truth.by_index()
    missing_from_probe = sorted(i for i in labels if i not in by_index)
    if missing_from_probe:
        raise ValueError(
            f"{len(missing_from_probe)} labelled frame(s) absent from probe output "
            f"(first: {missing_from_probe[0]}); the index join is broken"
        )
    # Matching counts are necessary, not sufficient: two different 858-frame
    # folders join cleanly on index and score as a blind detector. Both sides
    # recorded the filename, so check it.
    for idx, label in labels.items():
        theirs = by_index[idx].frame
        if label.frame and theirs and label.frame != theirs:
            raise ValueError(
                f"frame-name mismatch at index {idx}: labelled {label.frame!r}, "
                f"probe saw {theirs!r}. Same index, different frame — the label "
                "file and the probe output came from different folders."
            )

    tp = fp = fn = tn = 0
    n_visible = n_absent = 0
    absent_with_detection = 0
    absent_detection_count = 0
    out_of_frame = 0
    errors: List[float] = []

    for idx, label in sorted(labels.items()):
        dets = _select(by_index[idx].boxes, conf_floor, selection)
        for box in dets:
            if not _inside_frame(_centre(box), ground_truth.width, ground_truth.height):
                out_of_frame += 1

        if label.visible:
            n_visible += 1
            best_hit, best_dist = None, None
            for box in dets:
                d = math.dist(_centre(box), label.xy)
                if d <= tolerance_px and (best_dist is None or d < best_dist):
                    best_hit, best_dist = box, d
            if best_hit is not None:
                tp += 1
                errors.append(best_dist)
                fp += len(dets) - 1
            else:
                fn += 1
                fp += len(dets)
        else:
            n_absent += 1
            n_det = len(dets)
            # Pressure is counted from every candidate above the floor, not from
            # the selected one: "how much ball-coloured debris is this frame
            # offering" is a property of the footage, not of our selection.
            absent_detection_count += sum(1 for b in by_index[idx].boxes if b[4] >= conf_floor)
            if n_det:
                fp += n_det
                absent_with_detection += 1
            else:
                tn += 1

    n_labelled = len(labels)
    offered = set(ground_truth.sampled_indices) | set(labels)
    n_sampled = len(offered) if ground_truth.sampled_indices else n_labelled
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    # F1 is None only when a component is undefined. A detector that scores
    # P=0 and R=0 has an F1 of 0.0 — a real, bad number, not an absent one.
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return DetectionEval(
        conf_floor=conf_floor,
        tolerance_px=tolerance_px,
        selection=selection,
        n_sampled=n_sampled,
        n_labelled=n_labelled,
        n_skipped=max(0, n_sampled - n_labelled),
        n_visible=n_visible,
        n_absent=n_absent,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        absent_frame_fp_rate=_ratio(absent_with_detection, n_absent),
        absent_frame_candidates_per_frame=(
            absent_detection_count / n_absent if n_absent else None
        ),
        median_error_px=median(errors) if errors else None,
        mean_error_px=(sum(errors) / len(errors)) if errors else None,
        p90_error_px=_percentile(errors, 0.9) if errors else None,
        out_of_frame_detections=out_of_frame,
    )


def sweep(
    ground_truth: BallGroundTruth,
    probe_frames: Sequence[ProbeFrame],
    conf_floors: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    selection: str = "best",
) -> List[DetectionEval]:
    """One evaluation per confidence floor — design §9.2's floor sweep.

    Run it once per ``ball_probe_<imgsz>.jsonl`` to get the 640-vs-1280 table
    that decides H1, the cheapest decisive number in the plan (§9.5).
    """
    return [
        evaluate_detections(ground_truth, probe_frames, floor, tolerance_px, selection)
        for floor in conf_floors
    ]


def format_sweep(reports: Sequence[DetectionEval]) -> str:
    """Render a sweep as a fixed-width table for a notebook or a design doc."""
    head = (
        f"{'conf':>5} {'P':>7} {'R':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4} "
        f"{'TN':>4} {'medErr':>7} {'absFP%':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in reports:
        lines.append(
            f"{r.conf_floor:>5.2f} {_pct(r.precision):>7} {_pct(r.recall):>7} "
            f"{_pct(r.f1):>7} {r.true_positives:>4} {r.false_positives:>4} "
            f"{r.false_negatives:>4} {r.true_negatives:>4} "
            f"{_px(r.median_error_px):>7} {_pct(r.absent_frame_fp_rate):>7}"
        )
    return "\n".join(lines)


def _select(
    boxes: Sequence[Tuple[float, float, float, float, float]],
    conf_floor: float,
    selection: str,
) -> List[Tuple[float, float, float, float, float]]:
    candidates = [b for b in boxes if b[4] >= conf_floor]
    if selection == "any" or not candidates:
        return candidates
    return [max(candidates, key=lambda b: b[4])]


def _centre(box: Tuple[float, float, float, float, float]) -> Point2:
    l, t, r, b = box[0], box[1], box[2], box[3]
    return ((l + r) / 2.0, (t + b) / 2.0)


def _inside_frame(xy: Point2, width: int, height: int) -> bool:
    return 0.0 <= xy[0] <= width and 0.0 <= xy[1] <= height


def _ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{100.0 * value:.1f}"


def _px(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.1f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m pipeline.ball.evaluation gt.json ball_probe_1280.jsonl``."""
    import argparse

    from .ingest import load_ball_probe

    p = argparse.ArgumentParser(description="Ball detection recall/precision vs hand labels.")
    p.add_argument("ground_truth", help="ball-ground-truth/v1 JSON from tools/annotator.html")
    p.add_argument("probe", nargs="+", help="ball_probe_<imgsz>.jsonl file(s)")
    p.add_argument("--conf", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    p.add_argument("--tolerance-px", type=float, default=DEFAULT_TOLERANCE_PX)
    p.add_argument("--selection", choices=["best", "any"], default="best")
    args = p.parse_args(argv)

    try:
        gt = load_ground_truth(args.ground_truth)
    except ValueError as err:
        print(f"error: {args.ground_truth}: {err}")
        return 1
    print(
        f"ground truth: {len(gt.labels)} labels "
        f"({sum(1 for l in gt.labels if l.visible)} visible, "
        f"{sum(1 for l in gt.labels if not l.visible)} absent) "
        f"at {gt.width}x{gt.height}, stride {gt.stride}, "
        f"over {gt.n_frames_total} frames"
    )
    failed = False
    for probe_path in args.probe:
        frames = load_ball_probe(probe_path)
        try:
            reports = sweep(gt, frames, args.conf, args.tolerance_px, args.selection)
        except ValueError as err:
            # A broken join is the one failure that must not be mistaken for a
            # result, so it exits non-zero rather than printing a table.
            print(f"\nerror: {probe_path}: {err}")
            failed = True
            continue
        print(
            f"\n{probe_path}  (selection={args.selection}, "
            f"tolerance={args.tolerance_px:g} px)"
        )
        print(format_sweep(reports))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
