#!/usr/bin/env python3
"""Offline analysis of the ball probe dumps — docs/ball-tracking-design.md §9.2.

The probe (research/ball_probe.py) answers "what did a stock COCO detector see";
this answers "was any of it the ball, at which inference resolution, and what
shape are the holes". No GPU, no tracklab — it reads the committed jsonl dumps
and the committed hand labels and prints the tables that §9.2 asks for:

    1. the resolution curve       — raw, confidence-independent recall per imgsz
    2. the floor x imgsz sweep    — precision/recall/F1 (pipeline.ball.evaluation)
    3. the gap structure          — interpolable vs blackout, at one operating point
    4. the false-positive anatomy — where in the frame the wrong answers live

(1) exists because it is the one number no threshold choice can flatter: it asks
only whether a candidate landed within tolerance of the label, ignoring
confidence entirely. (2) is the operating-point table you actually deploy from.
The two disagree on purpose — the gap between them is selection error, i.e. the
detector had the ball in its list and confidence chose confetti.

Usage (from the repo root, no arguments needed — the defaults are the committed
artifacts):

    python research/ball_analysis.py
    python research/ball_analysis.py --gap-imgsz 640 --gap-conf 0.10

Every number printed here is reproducible from tracked files. That is the point:
the first real perception measurement in this project should not live in
somebody's shell history.
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
# Runnable as `python research/ball_analysis.py` from anywhere, like ball_probe.py.
sys.path.insert(0, str(REPO))
DEFAULT_GT = REPO / "research" / "ofi_frames.ball-gt.json"
DEFAULT_DUMPS = REPO / "research" / "ball_probe_out"

#: Gaps at or below this many frames are bridgeable by pipeline.ball.motion's
#: alpha-beta smoother without inventing a trajectory (design §6.5). At 25 fps
#: this is 200 ms — about one touch.
INTERPOLABLE_MAX_FRAMES = 5
#: Above this, the smoother must not bridge and every ball-dependent consumer
#: has to be gated off instead (design §6.3). 25 frames = 1 s.
BLACKOUT_MIN_FRAMES = 25


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ground-truth", default=str(DEFAULT_GT))
    p.add_argument("--dumps", default=str(DEFAULT_DUMPS),
                   help="folder of ball_probe_<imgsz>.jsonl files")
    p.add_argument("--conf", type=float, nargs="+",
                   default=[0.05, 0.10, 0.20, 0.30])
    p.add_argument("--tolerance-px", type=float, default=8.0)
    p.add_argument("--gap-imgsz", type=int, default=640,
                   help="imgsz for the gap-structure section")
    p.add_argument("--gap-conf", type=float, default=0.10)
    return p.parse_args()


def discover_dumps(folder: Path) -> List[Tuple[int, Path]]:
    """Every ball_probe_<imgsz>.jsonl in the folder, ordered by resolution."""
    found = []
    for path in folder.glob("ball_probe_*.jsonl"):
        m = re.fullmatch(r"ball_probe_(\d+)", path.stem)
        if m and path.stat().st_size > 0:
            found.append((int(m.group(1)), path))
    return sorted(found)


def load_records(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_recall(labels: Sequence[dict], records: Sequence[dict],
               tolerance_px: float) -> Tuple[int, int, int]:
    """Hits within tolerance ignoring confidence, plus frames with no candidate.

    The conf-independent ceiling: no threshold, no selection rule, and no
    downstream cleverness can beat this. If it is low, the detector is blind
    and everything else is bookkeeping.
    """
    hits = empty = 0
    for lab in labels:
        rec = records[lab["index"]]
        if not rec["balls"]:
            empty += 1
            continue
        best = min(math.hypot((b[0] + b[2]) / 2 - lab["x"],
                              (b[1] + b[3]) / 2 - lab["y"])
                   for b in rec["balls"])
        if best <= tolerance_px:
            hits += 1
    return hits, len(labels), empty


def gap_structure(records: Sequence[dict], conf: float) -> dict:
    """Runs of consecutive frames with no candidate above the floor.

    Computed over *all* frames, not just labelled ones, because gap length is a
    property of the stream rather than of the sample. It therefore inherits the
    operating point's false positives: a confetti hit in the middle of a real
    hole splits one blackout into two bridgeable gaps, so these numbers are
    optimistic by roughly the complement of precision. Stated, not corrected —
    correcting it would need labels on every frame, which is the thing we
    deliberately did not buy.
    """
    hit = [any(b[4] >= conf for b in r["balls"]) for r in records]
    gaps, run = [], 0
    for h in hit:
        if h:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    if run:
        gaps.append(run)
    blackouts = [g for g in gaps if g > BLACKOUT_MIN_FRAMES]
    return {
        "n_frames": len(hit),
        "n_detected": sum(hit),
        "gaps": gaps,
        "interpolable": sum(1 for g in gaps if g <= INTERPOLABLE_MAX_FRAMES),
        "blackouts": blackouts,
        "blackout_frames": sum(blackouts),
    }


def vertical_anatomy(records: Sequence[dict], conf: float,
                     bands: Sequence[Tuple[int, int]]) -> List[dict]:
    """Candidate count and size per horizontal band of the frame.

    Ball-sized blobs on grass and ball-shaped things in the stands are different
    failure modes with different fixes, and image row separates them almost
    perfectly in a broadcast framing.
    """
    rows = []
    total = sum(1 for r in records for b in r["balls"] if b[4] >= conf)
    for lo, hi in bands:
        widths = [b[2] - b[0] for r in records for b in r["balls"]
                  if b[4] >= conf and lo <= (b[1] + b[3]) / 2 < hi]
        rows.append({
            "band": (lo, hi),
            "n": len(widths),
            "share": (len(widths) / total) if total else None,
            "median_w": statistics.median(widths) if widths else None,
        })
    return rows


def main():
    args = parse_args()
    from pipeline.ball.evaluation import format_sweep, parse_ground_truth, sweep
    from pipeline.ball.ingest import load_ball_probe

    doc = json.load(open(args.ground_truth))
    gt = parse_ground_truth(doc)
    visible = [l for l in doc["labels"] if l["status"] == "visible"]
    dumps = discover_dumps(Path(args.dumps))
    if not dumps:
        raise SystemExit(f"no ball_probe_<imgsz>.jsonl under {args.dumps}")

    print(f"ground truth: {len(gt.labels)} labels "
          f"({len(visible)} visible, {len(gt.labels) - len(visible)} absent) "
          f"at {gt.width}x{gt.height}, stride {gt.stride}, "
          f"over {gt.n_frames_total} frames")
    print(f"tolerance: {args.tolerance_px:g} px\n")

    # 1. The resolution curve.
    print("=" * 62)
    print("1. RESOLUTION CURVE — raw recall, confidence ignored")
    print("=" * 62)
    print(f"{'imgsz':>6} {'hits':>6} {'raw recall':>11} {'no candidate':>13} "
          f"{'candidates':>11}")
    curve = {}
    for imgsz, path in dumps:
        records = load_records(path)
        hits, n, empty = raw_recall(visible, records, args.tolerance_px)
        n_cand = sum(len(r["balls"]) for r in records)
        curve[imgsz] = hits / n if n else None
        print(f"{imgsz:>6} {hits:>6} {100 * hits / n:>10.1f}% "
              f"{empty:>8}/{n:<4} {n_cand:>11}")
    best_sz = max(curve, key=lambda k: curve[k] or -1)
    print(f"\nbest: imgsz {best_sz} at {100 * curve[best_sz]:.1f}% raw recall")

    # 2. The operating-point sweep.
    print("\n" + "=" * 62)
    print("2. OPERATING POINTS — precision/recall per floor")
    print("=" * 62)
    for imgsz, path in dumps:
        frames = load_ball_probe(str(path))
        print(f"\nimgsz {imgsz} (selection=best):")
        print(format_sweep(sweep(gt, frames, args.conf, args.tolerance_px, "best")))

    # 3. Gap structure at the chosen operating point.
    gap_path = dict(dumps).get(args.gap_imgsz)
    if gap_path:
        print("\n" + "=" * 62)
        print(f"3. GAP STRUCTURE — imgsz {args.gap_imgsz}, conf >= {args.gap_conf:g}")
        print("=" * 62)
        g = gap_structure(load_records(gap_path), args.gap_conf)
        gaps = g["gaps"]
        print(f"detected in {g['n_detected']}/{g['n_frames']} frames "
              f"({100 * g['n_detected'] / g['n_frames']:.1f}%), {len(gaps)} gaps")
        if gaps:
            print(f"  median {statistics.median(gaps):.0f}f  "
                  f"mean {statistics.mean(gaps):.1f}f  "
                  f"max {max(gaps)}f ({max(gaps) / 25:.1f}s)")
            print(f"  interpolable (<={INTERPOLABLE_MAX_FRAMES}f): "
                  f"{g['interpolable']}/{len(gaps)} "
                  f"({100 * g['interpolable'] / len(gaps):.0f}%)")
            print(f"  blackouts (>{BLACKOUT_MIN_FRAMES}f): {len(g['blackouts'])} "
                  f"spanning {g['blackout_frames']} frames "
                  f"({100 * g['blackout_frames'] / g['n_frames']:.0f}% of clip)")
            print(f"  histogram: {dict(sorted(Counter(gaps).items()))}")
        print("  NOTE: computed over unlabelled frames, so false positives split\n"
              "  real holes — optimistic by roughly (1 - precision).")

    # 4. Where the wrong answers live.
    print("\n" + "=" * 62)
    print("4. FALSE-POSITIVE ANATOMY — candidates by image row")
    print("=" * 62)
    bands = [(0, 100), (100, 200), (200, 300), (300, 400),
             (400, 500), (500, 600), (600, gt.height)]
    for imgsz, path in dumps:
        records = load_records(path)
        print(f"\nimgsz {imgsz} at conf >= 0.05:")
        for row in vertical_anatomy(records, 0.05, bands):
            lo, hi = row["band"]
            share = "–" if row["share"] is None else f"{100 * row['share']:5.1f}%"
            width = "–" if row["median_w"] is None else f"{row['median_w']:5.1f}px"
            print(f"  y {lo:>3}-{hi:>3}: {row['n']:>5} ({share})  "
                  f"median width {width}")
    print("\n  Rows above the pitch line are stands and advertising boards:\n"
          "  ball-shaped, wrong size, trivially maskable.")


if __name__ == "__main__":
    raise SystemExit(main())
