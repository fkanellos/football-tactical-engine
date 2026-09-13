#!/usr/bin/env python3
"""Standalone ball-detection probe for a folder of extracted broadcast frames.

The next-GPU-session experiment from docs/ball-tracking-design.md §7: before
changing anything in the pipeline, measure whether an off-the-shelf detector
can see the ball in our footage at all, and at what confidence/resolution.
Runs a stock (COCO) Ultralytics YOLO checkpoint over the frames folder and
dumps EVERY 'sports ball' (class 32) candidate per frame — no selection, no
thresholding beyond a permissive floor — so precision/recall trade-offs are
explored offline by pipeline/ball/ingest.py + the measurement layer, not
re-run on GPU.

Deliberately independent of tracklab/sn-gamestate: it needs only ultralytics
(already installed in the sn-gamestate venv) and runs in ~2-6 min on a T4 for
858 frames. Person detections are counted per frame as a sanity signal (a
frame where persons drop to zero is a replay/close-up, not a pitch view).

Usage (Colab, after research/setup.sh):
    cd /content/sn-gamestate
    uv run --python .venv/bin/python \
        /content/football-tactical-engine/research/ball_probe.py \
        --frames /content/frames --out /content/ball_probe \
        --weights pretrained_models/yolo/yolo11m.pt --imgsz 640 1280

Output: one JSONL per imgsz, ball_probe_<imgsz>.jsonl, lines of
    {"index": i, "frame": "000001", "imgsz": 1280,
     "balls": [[l, t, r, b, conf], ...], "n_persons": 14}
in the frame's native pixel coordinates (ultralytics rescales boxes back).
Join with calibration_dump.jsonl via pipeline.ball.ingest.probe_to_samples.
"""

import argparse
import json
from pathlib import Path

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
COCO_SPORTS_BALL = 32
COCO_PERSON = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", required=True, help="folder of extracted frames")
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--weights", default="yolo11m.pt",
                   help="ultralytics checkpoint (COCO classes)")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280],
                   help="inference sizes to probe (one output file each)")
    p.add_argument("--conf", type=float, default=0.05,
                   help="permissive floor; real thresholding happens offline")
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO  # deferred: GPU env only

    frames = sorted(
        p for p in Path(args.frames).iterdir() if p.suffix.lower() in FRAME_EXTS
    )
    if not frames:
        raise SystemExit(f"no frames found in {args.frames}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    for imgsz in args.imgsz:
        out_path = out_dir / f"ball_probe_{imgsz}.jsonl"
        n_ball_frames = 0
        with open(out_path, "w") as f:
            for start in range(0, len(frames), args.batch):
                batch = frames[start:start + args.batch]
                results = model(
                    [str(p) for p in batch],
                    imgsz=imgsz,
                    conf=args.conf,
                    verbose=False,
                )
                for offset, res in enumerate(results):
                    balls, n_persons = [], 0
                    for box in res.boxes.cpu().numpy():
                        if int(box.cls[0]) == COCO_PERSON:
                            n_persons += 1
                        elif int(box.cls[0]) == COCO_SPORTS_BALL:
                            l, t, r, b = (float(v) for v in box.xyxy[0])
                            balls.append([l, t, r, b, float(box.conf[0])])
                    if balls:
                        n_ball_frames += 1
                    rec = {
                        "index": start + offset,
                        "frame": batch[offset].stem,
                        "imgsz": imgsz,
                        "balls": balls,
                        "n_persons": n_persons,
                    }
                    f.write(json.dumps(rec) + "\n")
        print(f"imgsz={imgsz}: {n_ball_frames}/{len(frames)} frames with >=1 "
              f"ball candidate at conf>={args.conf} -> {out_path}")


if __name__ == "__main__":
    main()
