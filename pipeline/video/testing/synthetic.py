"""Synthetic frame-signature sequences with known ground truth.

Same contract as the other `testing/synthetic.py` modules in this repo: build a
sequence whose answer is known by construction, then corrupt it in a named,
realistic way. Here the "answer" is the set of frame indices that start a new
shot, so every builder returns ``(signatures, true_boundaries)`` and a test
asserts against the second element rather than against a hand-copied list.

A "shot" is modelled as a stable base histogram plus small per-frame noise —
which is what a locked-off camera on a green pitch actually produces. Pans and
zooms are modelled as a *drift* of that base, because that is the property that
makes them dangerous: they are large cumulative change with small per-frame
change, and any detector that thresholds cumulative change calls them cuts.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from ..shots import FrameSignature

FPS = 25.0
N_BINS = 24


def _normalise(values: Sequence[float]) -> Tuple[float, ...]:
    total = sum(values)
    if total <= 0:
        raise ValueError("histogram sums to zero")
    return tuple(v / total for v in values)


def base_histogram(rng: random.Random, dominant: int = None) -> Tuple[float, ...]:
    """A plausible scene: most mass in a few bins, a thin floor elsewhere.

    Football frames are not uniform — grass dominates a pitch view, skin and
    shirts dominate a close-up, and a crowd shot is broad but dark. The
    ``dominant`` bin is what a cut moves.
    """
    if dominant is None:
        dominant = rng.randrange(N_BINS)
    values = [rng.uniform(0.01, 0.05) for _ in range(N_BINS)]
    values[dominant] += rng.uniform(2.0, 4.0)
    values[(dominant + 1) % N_BINS] += rng.uniform(0.5, 1.5)
    return _normalise(values)


def _jitter(hist: Sequence[float], rng: random.Random, scale: float) -> Tuple[float, ...]:
    return _normalise([max(1e-6, h + rng.uniform(-scale, scale) * h) for h in hist])


def single_shot(
    n_frames: int = 200, seed: int = 0, noise: float = 0.02
) -> Tuple[List[FrameSignature], List[int]]:
    """One continuous take — the OFI clip's structure, and the null case."""
    rng = random.Random(seed)
    base = base_histogram(rng)
    sigs = [
        FrameSignature(i, i / FPS, _jitter(base, rng, noise)) for i in range(n_frames)
    ]
    return sigs, []


def cut_sequence(
    shot_lengths: Sequence[int] = (60, 80, 50),
    seed: int = 0,
    noise: float = 0.02,
) -> Tuple[List[FrameSignature], List[int]]:
    """Hard cuts between visually distinct shots.

    Dominant bins are spaced around the histogram so successive shots never
    collide by chance — a cut between two shots that happen to look alike is a
    real phenomenon but belongs in its own named builder, not hiding inside the
    baseline case.
    """
    rng = random.Random(seed)
    sigs: List[FrameSignature] = []
    boundaries: List[int] = []
    index = 0
    for shot_no, length in enumerate(shot_lengths):
        base = base_histogram(rng, dominant=(shot_no * 7) % N_BINS)
        if shot_no > 0:
            boundaries.append(index)
        for _ in range(length):
            sigs.append(FrameSignature(index, index / FPS, _jitter(base, rng, noise)))
            index += 1
    return sigs, boundaries


def pan_sequence(
    n_frames: int = 200, seed: int = 0, drift_per_frame: float = 0.02
) -> Tuple[List[FrameSignature], List[int]]:
    """A long camera pan: the histogram walks a long way, one small step at a time.

    The adversarial case for any cumulative-change detector, and the reason the
    threshold reads frame-to-frame distance rather than distance from the shot's
    first frame. Ground truth is *no* boundaries however far the scene travels.
    """
    rng = random.Random(seed)
    hist = list(base_histogram(rng, dominant=0))
    sigs = []
    for i in range(n_frames):
        shift = [0.0] * N_BINS
        for b in range(N_BINS):
            shift[b] = hist[b] * (1.0 - drift_per_frame) + hist[
                (b - 1) % N_BINS
            ] * drift_per_frame
        hist = list(_normalise(shift))
        sigs.append(FrameSignature(i, i / FPS, _jitter(hist, rng, 0.01)))
    return sigs, []


def dissolve_sequence(
    shot_len: int = 80, dissolve_frames: int = 15, seed: int = 0
) -> Tuple[List[FrameSignature], List[int]]:
    """Two shots joined by a linear cross-fade.

    Ground truth here is a *range*, not a point: any boundary inside the
    dissolve is defensible. Returns the dissolve's first frame as the nominal
    answer, and tests assert containment in the window rather than equality —
    claiming frame-exact accuracy on a gradual transition would be measuring
    the fixture, not the detector.
    """
    rng = random.Random(seed)
    a = base_histogram(rng, dominant=0)
    b = base_histogram(rng, dominant=N_BINS // 2)
    sigs: List[FrameSignature] = []
    index = 0
    for _ in range(shot_len):
        sigs.append(FrameSignature(index, index / FPS, _jitter(a, rng, 0.02)))
        index += 1
    for step in range(dissolve_frames):
        w = (step + 1) / (dissolve_frames + 1)
        mixed = _normalise([(1 - w) * x + w * y for x, y in zip(a, b)])
        sigs.append(FrameSignature(index, index / FPS, _jitter(mixed, rng, 0.01)))
        index += 1
    for _ in range(shot_len):
        sigs.append(FrameSignature(index, index / FPS, _jitter(b, rng, 0.02)))
        index += 1
    return sigs, [shot_len]


def flash_sequence(
    n_frames: int = 150, flash_at: int = 75, seed: int = 0
) -> Tuple[List[FrameSignature], List[int]]:
    """One continuous shot with a single white-out frame (camera strobe, pyro).

    Two large distances one frame apart, into the flash and out of it. A
    detector without a minimum-shot rule reports two cuts and invents a
    one-frame segment; ground truth is zero cuts.
    """
    rng = random.Random(seed)
    base = base_histogram(rng, dominant=3)
    white = _normalise([1.0 if b == N_BINS - 1 else 0.01 for b in range(N_BINS)])
    sigs = []
    for i in range(n_frames):
        hist = white if i == flash_at else _jitter(base, rng, 0.02)
        sigs.append(FrameSignature(i, i / FPS, hist))
    return sigs, []
