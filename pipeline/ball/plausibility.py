"""Physical plausibility of a ball track: is this trajectory a ball's?

No ground truth needed — the checks lean on physics the same way the
calibration layer leans on known pitch geometry (calibration-design.md §6.1):
a ball cannot exceed ~40 m/s off the boot, cannot teleport 30 m between
consecutive frames, and has no business 10 m outside the pitch. Violations are
almost always identity errors (the "ball" jumped to a piece of paper debris, a
white boot, a bald head) rather than kinematic ones, which is what makes them
diagnostic: they count false positives without anyone labelling a frame.

Speed here is *implied* speed — displacement between consecutive detections
over their time difference. Across a long gap a large displacement is
legitimate (the ball genuinely travelled); a teleport is an overspeed at
near-consecutive-frame dt, where no real ball flight fits between the samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .model import BallSample, median, out_of_bounds_m

#: Hardest recorded shots sit around 35-45 m/s (~130-160 km/h); 40 m/s flags
#: only what no boot produces. A named default, not a measured constant.
MAX_BALL_SPEED_MS = 40.0
#: dt at/below which an overspeed is a teleport (no unobserved flight fits):
#: covers consecutive frames at 25 fps (0.04 s) and the 5 Hz analysis grid (0.2 s).
TELEPORT_MAX_DT_S = 0.25
#: Balls legitimately leave the pitch (throw-ins, corners) by metres, not tens
#: of metres; beyond this the "ball" is likelier an advertising board artefact.
OUT_OF_BOUNDS_MARGIN_M = 5.0


@dataclass(frozen=True)
class KinematicFlag:
    """Plausibility verdict for one detected sample (relative to the previous one)."""

    index: int
    dt_s: Optional[float]  # time since previous detection; None for the first
    speed_ms: Optional[float]
    overspeed: bool
    teleport: bool
    out_of_bounds_by_m: float


@dataclass(frozen=True)
class PlausibilityReport:
    n_detected: int
    n_speed_pairs: int
    overspeed_count: int
    teleport_count: int
    out_of_bounds_count: int
    median_speed_ms: Optional[float]
    max_speed_ms: Optional[float]


def kinematic_flags(
    samples: Sequence[BallSample],
    max_speed_ms: float = MAX_BALL_SPEED_MS,
    teleport_max_dt_s: float = TELEPORT_MAX_DT_S,
    oob_margin_m: float = OUT_OF_BOUNDS_MARGIN_M,
) -> List[KinematicFlag]:
    """One flag per *detected* sample, in stream order."""
    flags: List[KinematicFlag] = []
    prev_t: Optional[float] = None
    prev_xy = None
    for i, s in enumerate(samples):
        if s.xy is None:
            continue
        dt: Optional[float] = None
        speed: Optional[float] = None
        if prev_xy is not None and prev_t is not None:
            dt = s.t - prev_t
            if dt > 0.0:
                speed = math.hypot(s.xy[0] - prev_xy[0], s.xy[1] - prev_xy[1]) / dt
        overspeed = speed is not None and speed > max_speed_ms
        teleport = overspeed and dt is not None and dt <= teleport_max_dt_s
        oob = out_of_bounds_m(s.xy)
        flags.append(
            KinematicFlag(
                index=i,
                dt_s=dt,
                speed_ms=speed,
                overspeed=overspeed,
                teleport=teleport,
                out_of_bounds_by_m=oob if oob > oob_margin_m else 0.0,
            )
        )
        prev_t, prev_xy = s.t, s.xy
    return flags


def plausibility_report(
    samples: Sequence[BallSample],
    max_speed_ms: float = MAX_BALL_SPEED_MS,
    teleport_max_dt_s: float = TELEPORT_MAX_DT_S,
    oob_margin_m: float = OUT_OF_BOUNDS_MARGIN_M,
) -> PlausibilityReport:
    flags = kinematic_flags(samples, max_speed_ms, teleport_max_dt_s, oob_margin_m)
    speeds = [f.speed_ms for f in flags if f.speed_ms is not None]
    return PlausibilityReport(
        n_detected=len(flags),
        n_speed_pairs=len(speeds),
        overspeed_count=sum(1 for f in flags if f.overspeed),
        teleport_count=sum(1 for f in flags if f.teleport),
        out_of_bounds_count=sum(1 for f in flags if f.out_of_bounds_by_m > 0.0),
        median_speed_ms=median(speeds) if speeds else None,
        max_speed_ms=max(speeds) if speeds else None,
    )
