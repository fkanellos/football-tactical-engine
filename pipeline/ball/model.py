"""Ball observation stream: the neutral data model every ball metric consumes.

One sample per pipeline frame, in absolute pitch coordinates (105 x 68 m, origin
at the centre spot — same convention as ``pipeline/patterns/tracking.py``).
``xy is None`` means "no ball this frame", which is a first-class state, not an
error: the whole measurement layer exists because we do not yet know how often
that state occurs on real footage (docs/ball-tracking-design.md §1).

The stream is deliberately simpler than ``TrackingFrame``: ball metrics need only
(t, position, confidence, candidate count), and keeping the type tiny lets the
ingest adapter (ingest.py), the synthetic builder (testing/synthetic.py), and a
future TrackLab-state adapter all produce it trivially.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

Point2 = Tuple[float, float]

#: Canonical pitch (matches pipeline/patterns/tracking.py and pipeline/calibration/pitch.py).
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


@dataclass(frozen=True)
class BallSample:
    """The ball as one frame knows it.

    ``n_candidates`` is the number of raw detector candidates the frame carried
    *before* selection (paper debris, heads, and other round white objects all
    produce candidates); it is a false-positive-pressure signal even when the
    selected position is right. ``interpolated`` marks samples synthesised by
    ``motion.interpolate_gaps`` — never by a detector — so downstream consumers
    can distinguish observed evidence from bridged evidence.
    """

    t: float
    xy: Optional[Point2]
    confidence: float = 1.0
    n_candidates: int = -1  # -1 = unknown/not recorded
    interpolated: bool = False

    @property
    def detected(self) -> bool:
        return self.xy is not None


def missing(t: float) -> BallSample:
    """A frame with no ball."""
    return BallSample(t=t, xy=None, confidence=0.0, n_candidates=0)


def with_position(sample: BallSample, xy: Optional[Point2]) -> BallSample:
    return replace(sample, xy=xy)


def detected_indices(samples: Sequence[BallSample]) -> List[int]:
    return [i for i, s in enumerate(samples) if s.detected]


def out_of_bounds_m(xy: Point2) -> float:
    """Distance (m) outside the pitch rectangle; 0.0 for any point on the pitch."""
    dx = max(0.0, abs(xy[0]) - PITCH_LENGTH_M / 2.0)
    dy = max(0.0, abs(xy[1]) - PITCH_WIDTH_M / 2.0)
    return (dx * dx + dy * dy) ** 0.5


def median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence (even length: mean of the middle pair)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
