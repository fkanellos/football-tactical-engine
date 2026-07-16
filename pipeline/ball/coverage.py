"""Detection coverage: how often the ball exists at all, and the shape of its absence.

The single most important unknown about real footage (design §5.1): the
*detection rate* alone decides which downstream consumers are viable, and the
*gap-length distribution* decides whether misses are interpolable (isolated
frames) or structural (multi-second blackouts). These are different worlds:
a 92% rate made of isolated misses supports pass detection after gap-bridging;
a 92% rate with three 20 s blackouts does not, no matter what tracker runs on top.

Gap *context* (where the ball was last seen, how fast it must have moved across
the gap, whether an endpoint hugs the pitch boundary) is the no-ground-truth
proxy for the spatial failure distribution: we cannot know where an undetected
ball was, but the bracketing detections say what kind of event the miss was —
a struck ball (high implied speed => motion blur), an exit (boundary endpoint),
or a quiet loss in traffic (low speed, mid-pitch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .model import BallSample, Point2, PITCH_LENGTH_M, PITCH_WIDTH_M, median


@dataclass(frozen=True)
class Gap:
    """A maximal run of frames with no ball.

    ``duration_s`` is the information horizon: time between the bracketing
    detections (or to the stream edge for leading/trailing gaps) — the span a
    bridging model would have to cover, which is why it is measured between
    detections rather than between missing samples.
    """

    start_index: int  # first missing frame
    end_index: int  # one past the last missing frame
    n_frames: int
    duration_s: float
    at_edge: bool  # leading/trailing gap: only one bracketing detection exists


@dataclass(frozen=True)
class GapContext:
    """What the detections bracketing a gap imply about it."""

    gap: Gap
    last_xy: Optional[Point2]
    next_xy: Optional[Point2]
    displacement_m: Optional[float]
    implied_speed_ms: Optional[float]  # displacement / duration; None at edges
    near_boundary: bool  # either endpoint within boundary_margin_m of a pitch line


@dataclass(frozen=True)
class CoverageReport:
    """The headline numbers. ``*_fraction`` denominators are stated per field."""

    n_frames: int
    n_detected: int
    detection_rate: float
    n_gaps: int
    median_gap_s: Optional[float]
    max_gap_s: Optional[float]
    #: fraction of *missing frames* that sit in gaps of <= isolated_max_frames
    isolated_miss_fraction: Optional[float]
    #: fraction of *missing frames* in gaps of duration <= interpolable_max_s
    interpolable_miss_fraction: Optional[float]
    #: fraction of *all frames* inside gaps of duration >= blackout_min_s
    blackout_frame_fraction: float
    gaps: Tuple[Gap, ...]


def find_gaps(samples: Sequence[BallSample]) -> List[Gap]:
    gaps: List[Gap] = []
    n = len(samples)
    i = 0
    while i < n:
        if samples[i].detected:
            i += 1
            continue
        start = i
        while i < n and not samples[i].detected:
            i += 1
        end = i
        prev_t = samples[start - 1].t if start > 0 else None
        next_t = samples[end].t if end < n else None
        at_edge = prev_t is None or next_t is None
        if prev_t is not None and next_t is not None:
            duration = next_t - prev_t
        elif next_t is not None:
            duration = next_t - samples[start].t
        elif prev_t is not None:
            duration = samples[end - 1].t - prev_t
        else:  # the whole stream is one gap
            duration = samples[end - 1].t - samples[start].t
        gaps.append(Gap(start, end, end - start, duration, at_edge))
    return gaps


def coverage_report(
    samples: Sequence[BallSample],
    isolated_max_frames: int = 2,
    interpolable_max_s: float = 1.0,
    blackout_min_s: float = 5.0,
) -> CoverageReport:
    n = len(samples)
    n_detected = sum(1 for s in samples if s.detected)
    gaps = find_gaps(samples)
    n_missing = n - n_detected

    durations = [g.duration_s for g in gaps]
    isolated = sum(g.n_frames for g in gaps if g.n_frames <= isolated_max_frames)
    interpolable = sum(g.n_frames for g in gaps if g.duration_s <= interpolable_max_s)
    blackout = sum(g.n_frames for g in gaps if g.duration_s >= blackout_min_s)

    return CoverageReport(
        n_frames=n,
        n_detected=n_detected,
        detection_rate=(n_detected / n) if n else 0.0,
        n_gaps=len(gaps),
        median_gap_s=median(durations) if durations else None,
        max_gap_s=max(durations) if durations else None,
        isolated_miss_fraction=(isolated / n_missing) if n_missing else None,
        interpolable_miss_fraction=(interpolable / n_missing) if n_missing else None,
        blackout_frame_fraction=(blackout / n) if n else 0.0,
        gaps=tuple(gaps),
    )


def gap_contexts(
    samples: Sequence[BallSample],
    boundary_margin_m: float = 3.0,
) -> List[GapContext]:
    contexts: List[GapContext] = []
    for gap in find_gaps(samples):
        last_xy = samples[gap.start_index - 1].xy if gap.start_index > 0 else None
        next_xy = samples[gap.end_index].xy if gap.end_index < len(samples) else None
        displacement: Optional[float] = None
        speed: Optional[float] = None
        if last_xy is not None and next_xy is not None:
            displacement = math.hypot(next_xy[0] - last_xy[0], next_xy[1] - last_xy[1])
            if gap.duration_s > 0.0:
                speed = displacement / gap.duration_s
        near_boundary = any(
            xy is not None and _boundary_distance_m(xy) <= boundary_margin_m
            for xy in (last_xy, next_xy)
        )
        contexts.append(GapContext(gap, last_xy, next_xy, displacement, speed, near_boundary))
    return contexts


def zone_counts(samples: Sequence[BallSample]) -> List[List[int]]:
    """3x3 grid of detection counts (x-thirds x y-thirds, low index = -x / -y).

    Only *detected* positions can be zoned — this shows where detections
    concentrate (and which regions never produce one), not where failures
    happen; read it together with ``gap_contexts``.
    """
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for s in samples:
        if s.xy is None:
            continue
        gx = _third(s.xy[0], PITCH_LENGTH_M)
        gy = _third(s.xy[1], PITCH_WIDTH_M)
        grid[gx][gy] += 1
    return grid


def _third(value: float, extent: float) -> int:
    edge = extent / 6.0  # thirds of [-extent/2, extent/2]
    if value < -edge:
        return 0
    if value > edge:
        return 2
    return 1


def _boundary_distance_m(xy: Point2) -> float:
    """Distance from a pitch point to the nearest touchline/goal line."""
    dx = PITCH_LENGTH_M / 2.0 - abs(xy[0])
    dy = PITCH_WIDTH_M / 2.0 - abs(xy[1])
    return max(0.0, min(dx, dy))
