"""Adapter: raw ball-probe detections + per-frame homographies -> BallSample stream.

The first real ball evidence arrives as two JSONL files from the GPU session
(design §7): ``ball_probe_<imgsz>.jsonl`` written by ``research/ball_probe.py``
(all COCO sports-ball candidates per frame, pixel bboxes at the clip's native
resolution) and ``calibration_dump.jsonl`` written by the existing
patch_sn_gamestate.py step 4 (per-frame image->pitch homography in the same
pixel frame — the fixed 1280x720 config, docs/calibration-design.md §3.4).
This module joins them into the pitch-coordinate ``BallSample`` stream every
metric in this package consumes.

Ground-plane caveat, stated where the code lives: projecting the bbox through a
plane-Z0 homography assumes the ball is ON the ground. An airborne ball lands
metres beyond its true ground point (displaced away from the camera along the
viewing ray) — the exact reason SoccerNet-GSR dropped the ball from its
annotations (Somers et al. 2024 §4.1). Positions produced here are therefore
"ground-track estimates", good for possession proximity and coverage
measurement, biased during flight; the bias is measurable later against
restart geometry (design §5.4).

Selection: per frame we keep the highest-confidence candidate at/above
``conf_floor`` and record how many candidates the frame carried —
``n_candidates`` is itself a false-positive-pressure measurement on footage
strewn with ball-coloured paper debris.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .model import BallSample, Point2

Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class ProbeFrame:
    """One frame of raw probe output: every candidate, in pixel space."""

    index: int
    boxes: Tuple[Tuple[float, float, float, float, float], ...]  # (l, t, r, b, conf)


def load_ball_probe(path: str) -> List[ProbeFrame]:
    """Read ball_probe JSONL: one line per frame, {"index", "balls": [[l,t,r,b,conf]..]}."""
    frames: List[ProbeFrame] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frames.append(
                ProbeFrame(
                    index=int(rec["index"]),
                    boxes=tuple(tuple(float(v) for v in b) for b in rec.get("balls", [])),
                )
            )
    frames.sort(key=lambda fr: fr.index)
    return frames


def load_homographies(path: str) -> Dict[int, Optional[Matrix]]:
    """Read calibration_dump.jsonl -> {frame index: image->pitch H or None}.

    Frame ids that do not parse as integers are enumerated in file order —
    the dump is written per processed frame in sequence, so order is identity.
    """
    out: Dict[int, Optional[Matrix]] = {}
    with open(path) as f:
        for order, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                idx = int(rec.get("frame"))
            except (TypeError, ValueError):
                idx = order
            h = rec.get("h_image_to_pitch")
            out[idx] = h if h is not None else None
    return out


def project_pixel(h_image_to_pitch: Matrix, uv: Point2) -> Optional[Point2]:
    """Apply an image->pitch homography to a pixel point."""
    u, v = uv
    m = h_image_to_pitch
    x = m[0][0] * u + m[0][1] * v + m[0][2]
    y = m[1][0] * u + m[1][1] * v + m[1][2]
    w = m[2][0] * u + m[2][1] * v + m[2][2]
    if abs(w) < 1e-12:
        return None
    return (x / w, y / w)


def probe_to_samples(
    probe_frames: Sequence[ProbeFrame],
    homographies: Dict[int, Optional[Matrix]],
    fps: float = 25.0,
    conf_floor: float = 0.1,
) -> List[BallSample]:
    """Join probe candidates with per-frame homographies into BallSamples.

    One sample per probe frame (missing when no candidate clears the floor or
    no homography exists — the two failure modes are deliberately merged into
    "no usable ball" because downstream metrics treat them identically; the
    split is recoverable from the raw files when it matters).

    The projected point is the bbox *bottom-centre* — the ground-contact point
    under the on-ground assumption (see module docstring for the airborne bias).
    """
    samples: List[BallSample] = []
    for fr in probe_frames:
        t = fr.index / fps
        candidates = [b for b in fr.boxes if b[4] >= conf_floor]
        h = homographies.get(fr.index)
        if not candidates or h is None:
            samples.append(BallSample(t=t, xy=None, confidence=0.0, n_candidates=len(candidates)))
            continue
        best = max(candidates, key=lambda b: b[4])
        l, top, r, bottom, conf = best
        xy = project_pixel(h, ((l + r) / 2.0, bottom))
        samples.append(
            BallSample(t=t, xy=xy, confidence=conf, n_candidates=len(candidates))
        )
    return samples
