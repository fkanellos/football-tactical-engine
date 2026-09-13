"""Frame-level signals computed before tracking.

Everything else in `pipeline/` consumes tracking output — players already
detected, identified and projected onto the pitch. This package sits upstream of
that: signals read off the raw frame sequence itself, which is where broadcast
structure lives.

Today that is shot-boundary detection, the producer of the segment gaps the
batch stack has always assumed and never had (red-team review §1.1/§3.1). The
live design's discontinuity gate (live-architecture-design.md §5.1) names three
more signals — calibration collapse, detection evaporation, whole-frame position
teleports — but all three read *tracking* output and so belong downstream, not
here. This package stays strictly pre-tracking.

Like the rest of `pipeline/`, nothing here decodes an image: shot detection
consumes ``FrameSignature`` histograms produced by `research/frame_signatures.py`,
keeping the package pure-stdlib and its tests free of binary fixtures.
"""

from .shots import (
    DEFAULT_MAD_K,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_MIN_SHOT_FRAMES,
    FrameSignature,
    Segment,
    ShotBoundary,
    ShotReport,
    adaptive_threshold,
    consecutive_distances,
    detect_shot_boundaries,
    histogram_distance,
    segment_for_frame,
    segments_from_boundaries,
    shot_report,
)

__all__ = [
    "DEFAULT_MAD_K",
    "DEFAULT_MIN_DISTANCE",
    "DEFAULT_MIN_SHOT_FRAMES",
    "FrameSignature",
    "Segment",
    "ShotBoundary",
    "ShotReport",
    "adaptive_threshold",
    "consecutive_distances",
    "detect_shot_boundaries",
    "histogram_distance",
    "segment_for_frame",
    "segments_from_boundaries",
    "shot_report",
]
