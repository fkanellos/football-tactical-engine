#!/usr/bin/env python3
"""Reduce a video or frame folder to per-frame colour histograms.

The decode half of shot-boundary detection. `pipeline/video/shots.py` finds cuts
but deliberately never touches a pixel — `pipeline/` is pure stdlib and its
whole suite runs in seconds, which an imaging dependency would end. So this
script owns decoding and emits the ``FrameSignature`` histograms the detector
consumes, exactly as `research/ball_probe.py` owns GPU inference and emits the
jsonl that `pipeline/ball/` consumes.

**No imaging library either.** Not numpy, PIL, or OpenCV: ffmpeg downscales to a
tiny RGB raster and writes raw bytes to stdout, and this reads them with
``bytes``. A 32x18 frame is 1 728 bytes, so histogramming a 100-minute match is
IO-bound on the decode and costs nothing in Python. The same code path handles
an .mp4 and a folder of extracted .jpg frames, which matters because the ball
work established the folder as the unit everything else joins on.

**Why so small a raster.** The question is "did the whole image change", and
detail is actively harmful: at full resolution a scoreboard animating in one
corner, or grass texture under a pan, moves enough fine-bin mass to look like a
cut. 32x18 keeps layout and colour and throws away everything else. Bins are
per-channel and coarse for the same reason.

Usage:

    # a folder of frames (the unit the ball probe and annotator also use)
    python research/frame_signatures.py --frames ofi_frames --out ofi.sig.jsonl

    # a video file directly, with its real frame rate
    python research/frame_signatures.py --video match.mp4 --out match.sig.jsonl

Output: one JSON object per line,
    {"index": 0, "timestamp_s": 0.0, "histogram": [...]}
in frame order, joinable by ``index`` with every other per-frame artifact in
this repo (probe dumps, ground-truth labels) because the ordering rule is the
same: sorted position in the folder, or decode order for a video.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

#: Raster the histogram is computed from. 32x18 preserves the 16:9 layout.
GRID_W, GRID_H = 32, 18
#: Bins per channel; 8 gives a 24-long signature, coarse enough that a pan is
#: not a cut and fine enough that grass-vs-crowd-vs-skin separate cleanly.
BINS = 8


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="folder of extracted frames")
    src.add_argument("--video", help="video file")
    p.add_argument("--out", required=True, help="output jsonl path")
    p.add_argument("--fps", type=float, default=25.0,
                   help="frame rate used for timestamps (and to resample a video)")
    p.add_argument("--grid", type=int, nargs=2, default=[GRID_W, GRID_H],
                   metavar=("W", "H"))
    p.add_argument("--bins", type=int, default=BINS)
    return p.parse_args()


def ffmpeg_raw_stream(source_args, width, height):
    """Yield one downscaled RGB frame at a time as raw bytes.

    Streams rather than buffering: a 100-minute match is 150 000 frames, and
    holding them costs 250 MB for no reason when the consumer is sequential.
    """
    cmd = [
        "ffmpeg", "-loglevel", "error", *source_args,
        "-vf", f"scale={width}:{height}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield buf
    finally:
        proc.stdout.close()
        proc.wait()


def histogram(frame: bytes, bins: int):
    """Per-channel histogram of an RGB raster, normalised to sum to 1.0.

    Concatenated rather than joint (a 3D colour cube) because a joint histogram
    of this raster is mostly empty and its distance is dominated by which single
    cell the grass landed in. Three marginals answer "how much of the image is
    green / bright / red" — the blunt question that separates a pitch view from
    a crowd shot from a close-up.
    """
    width = 256 // bins
    counts = [0] * (bins * 3)
    for i, value in enumerate(frame):
        counts[(i % 3) * bins + min(value // width, bins - 1)] += 1
    total = sum(counts)
    return [c / total for c in counts]


def main():
    args = parse_args()
    width, height = args.grid

    if args.frames:
        folder = Path(args.frames)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in FRAME_EXTS)
        if not files:
            raise SystemExit(f"no frames found in {folder}")
        # -framerate before -i makes the image sequence a video at a known rate,
        # so timestamps match every other artifact derived from this folder.
        pattern = files[0].name
        stem_digits = sum(c.isdigit() for c in files[0].stem)
        if stem_digits:
            pattern = f"%0{stem_digits}d{files[0].suffix}"
        source = ["-framerate", str(args.fps), "-i", str(folder / pattern)]
        expected = len(files)
    else:
        source = ["-i", args.video, "-r", str(args.fps)]
        expected = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for index, raw in enumerate(ffmpeg_raw_stream(source, width, height)):
            rec = {
                "index": index,
                "timestamp_s": index / args.fps,
                "histogram": [round(v, 6) for v in histogram(raw, args.bins)],
            }
            f.write(json.dumps(rec) + "\n")
            n += 1

    print(f"{n} frames -> {out_path}")
    if expected is not None and n != expected:
        # Loud, because every join in this repo is on frame index and a silent
        # off-by-N here would misalign signatures against probe dumps and labels.
        print(
            f"WARNING: folder has {expected} frames but {n} were decoded — "
            "the index join with probe dumps and labels is NOT safe",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
