#!/usr/bin/env python3
"""Flatten a TrackLab ``states.pklz`` into two plain jsonl files.

The bridge between the GPU side and this repo. `pipeline/` is pure stdlib and
its whole suite runs in seconds; reading `states.pklz` directly would need
pandas *and* whichever tracklab classes the pickles reference, which is a
dependency the analysis layer should never carry. Same split as everywhere
else here: `ball_probe.py` owns YOLO and emits jsonl, `frame_signatures.py`
owns ffmpeg and emits jsonl, and this owns pandas/pickle and emits jsonl.
`pipeline/patterns/tracking.py::load_tracklab_states` reads the result.

**Run it on the machine that produced the file** (the Colab VM, inside the
sn-gamestate venv), not locally: the pickles were written by that Python and
that pandas.

**It drops most of the file.** Of the 25 detection columns, ~19 are tracker
internals we never read — `embeddings` (a float vector per detection, and the
bulk of the bytes), Kalman predictions, match costs, ages, hit counts. The
kept set is exactly what the tactical layer consumes, which turns a
hundred-megabyte artifact into a few megabytes that can live in git next to
the measurements it feeds.

Usage (Colab, inside the venv):

    cd /content/sn-gamestate
    uv run --python .venv/bin/python \
        /content/football-tactical-engine/research/export_states.py \
        --states outputs/video-demo/2026-09-15/15-51-07/states.pklz \
        --out /content/drive/MyDrive/ofi

Writes ``<out>.detections.jsonl`` and ``<out>.frames.jsonl``.
"""

import argparse
import json
import math
import pickle
import zipfile
from pathlib import Path

#: Detection columns the tactical layer actually reads. Everything else in the
#: 25-column frame is tracker bookkeeping. `state` is kept because it carries
#: StrongSORT's confirmed/tentative flag, which is a quality signal we may want
#: to gate on; `bbox_conf` and `role_confidence` for the same reason.
DETECTION_COLUMNS = [
    "image_id",
    "bbox_ltwh",
    "bbox_conf",
    "track_id",
    "bbox_pitch",
    "role",
    "role_confidence",
    "team",
    "team_cluster",
    "jersey_number",
    "jersey_number_confidence",
    "state",
]

#: Frame columns. `keypoints`/`lines`/`parameters` are the calibration payload
#: (docs/calibration-design.md §9); `file_path` is what proves the index join
#: against probe dumps and hand labels.
FRAME_COLUMNS = [
    "id",
    "name",
    "frame",
    "nframes",
    "video_id",
    "file_path",
    "keypoints",
    "lines",
    "parameters",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--states", required=True, help="path to states.pklz")
    p.add_argument("--out", required=True, help="output prefix (two files written)")
    p.add_argument("--video-id", default=None,
                   help="which video's members to export; default: all found")
    return p.parse_args()


def clean(value):
    """Make a pandas/numpy value JSON-safe without importing numpy here.

    NaN is the one that matters: ``track_id`` is a float64 column with NaN for
    detections the tracker never associated, and ``jersey_number`` is NaN when
    OCR read nothing. Both must land in JSON as ``null`` rather than as the
    string "NaN", which is not valid JSON and which every downstream reader
    would silently treat as a present value.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    # numpy scalars and arrays, without importing numpy
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    return str(value)


def rows(df, columns, label):
    """Yield dicts for the requested columns, reporting any that are missing."""
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"  WARNING: {label} is missing expected columns: {missing}")
    for record in df[present].to_dict(orient="records"):
        yield {k: clean(v) for k, v in record.items()}


def main():
    args = parse_args()
    states = Path(args.states)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(states) as z:
        names = z.namelist()
        print(f"members: {names}")
        detection_members = sorted(
            n for n in names
            if n.endswith(".pkl") and not n.endswith("_image.pkl")
        )
        if args.video_id is not None:
            detection_members = [n for n in detection_members
                                 if n == f"{args.video_id}.pkl"]
        if not detection_members:
            raise SystemExit("no detection members found in the archive")

        n_det = n_frame = 0
        det_path = Path(f"{out}.detections.jsonl")
        frame_path = Path(f"{out}.frames.jsonl")
        with open(det_path, "w") as fd, open(frame_path, "w") as ff:
            for member in detection_members:
                video_id = member[:-len(".pkl")]
                print(f"\nvideo {video_id}:")
                with z.open(member) as f:
                    det = pickle.load(f)
                print(f"  detections {det.shape}")
                for rec in rows(det, DETECTION_COLUMNS, member):
                    rec["video_id"] = video_id
                    fd.write(json.dumps(rec) + "\n")
                    n_det += 1

                image_member = f"{video_id}_image.pkl"
                if image_member not in names:
                    print(f"  WARNING: no {image_member}; frames not exported")
                    continue
                with z.open(image_member) as f:
                    img = pickle.load(f)
                print(f"  frames     {img.shape}")
                for rec in rows(img, FRAME_COLUMNS, image_member):
                    ff.write(json.dumps(rec) + "\n")
                    n_frame += 1

    print(f"\n{n_det} detections -> {det_path} "
          f"({det_path.stat().st_size / 1e6:.1f} MB)")
    print(f"{n_frame} frames     -> {frame_path} "
          f"({frame_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
