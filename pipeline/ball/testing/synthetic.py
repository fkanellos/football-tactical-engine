"""Scripted ball trajectories with injectable detector pathologies.

Same philosophy as ``pipeline/calibration/testing/synthetic.py``: physics we
control exactly, so every metric can be asserted against known truth before any
real export exists. The trajectory pieces mirror what a real ball does in 2D
pitch coordinates:

- ``ground_pass``: struck, then linear deceleration (rolling friction);
- ``flight``: a kicked aerial ball — its *ground projection* moves at constant
  velocity (drag ignored), which is exactly what a plane-projected detector
  would report if it saw the ball's landing shadow;
- ``dribble``: low speed, small heading jitter around a carrier's path;
- ``hold``: dead ball / keeper's hands.

Corruptors model the failure modes ranked in the design doc: random dropout
(per-frame misses), span dropout (occlusion / out-of-frame blackouts), Gaussian
position noise (projection + bbox jitter), and static false positives that
replace or fill frames (the paper-debris signature: a plausible white object at
a fixed pitch location).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from ..model import BallSample, Point2

DEFAULT_HZ = 25.0


def _series(
    t0: float,
    duration_s: float,
    hz: float,
    position_at,
) -> List[BallSample]:
    n = max(1, round(duration_s * hz))
    out: List[BallSample] = []
    for i in range(n):
        rel = i / hz
        out.append(BallSample(t=t0 + rel, xy=position_at(rel), confidence=0.9, n_candidates=1))
    return out


def ground_pass(
    p0: Point2,
    direction_deg: float,
    speed_ms: float,
    duration_s: float,
    t0: float = 0.0,
    hz: float = DEFAULT_HZ,
    decel_ms2: float = 1.0,
) -> List[BallSample]:
    """A struck ground ball decelerating linearly (stops if it runs out of speed)."""
    ux, uy = math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))

    def pos(rel: float) -> Point2:
        t_stop = speed_ms / decel_ms2 if decel_ms2 > 0 else float("inf")
        tt = min(rel, t_stop)
        d = speed_ms * tt - 0.5 * decel_ms2 * tt * tt
        return (p0[0] + ux * d, p0[1] + uy * d)

    return _series(t0, duration_s, hz, pos)


def flight(
    p0: Point2,
    p1: Point2,
    duration_s: float,
    t0: float = 0.0,
    hz: float = DEFAULT_HZ,
) -> List[BallSample]:
    """Aerial ball p0 -> p1: constant-velocity ground projection."""

    def pos(rel: float) -> Point2:
        frac = rel / duration_s if duration_s > 0 else 1.0
        return (p0[0] + frac * (p1[0] - p0[0]), p0[1] + frac * (p1[1] - p0[1]))

    return _series(t0, duration_s, hz, pos)


def dribble(
    p0: Point2,
    direction_deg: float,
    speed_ms: float,
    duration_s: float,
    t0: float = 0.0,
    hz: float = DEFAULT_HZ,
    weave_amplitude_m: float = 0.4,
    weave_period_s: float = 1.2,
) -> List[BallSample]:
    """A carried ball: slow progress along a heading with lateral weave."""
    ux, uy = math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))
    nx, ny = -uy, ux

    def pos(rel: float) -> Point2:
        d = speed_ms * rel
        w = weave_amplitude_m * math.sin(2.0 * math.pi * rel / weave_period_s)
        return (p0[0] + ux * d + nx * w, p0[1] + uy * d + ny * w)

    return _series(t0, duration_s, hz, pos)


def hold(p: Point2, duration_s: float, t0: float = 0.0, hz: float = DEFAULT_HZ) -> List[BallSample]:
    """A dead/held ball."""
    return _series(t0, duration_s, hz, lambda rel: p)


def chain(*segments: Sequence[BallSample]) -> List[BallSample]:
    """Concatenate trajectory segments (each already carries its own t0)."""
    out: List[BallSample] = []
    for seg in segments:
        out.extend(seg)
    out.sort(key=lambda s: s.t)
    return out


def end_of(segment: Sequence[BallSample]) -> Tuple[float, Point2]:
    """(next t0, last position) for chaining segments without gaps."""
    last = segment[-1]
    assert last.xy is not None
    dt = segment[-1].t - segment[-2].t if len(segment) >= 2 else 1.0 / DEFAULT_HZ
    return (last.t + dt, last.xy)


# ---------------------------------------------------------------------------
# Corruptors
# ---------------------------------------------------------------------------

def drop_random(
    samples: Sequence[BallSample], miss_rate: float, rng: random.Random
) -> List[BallSample]:
    """Independent per-frame dropout (isolated misses)."""
    return [
        BallSample(t=s.t, xy=None, confidence=0.0, n_candidates=0)
        if s.detected and rng.random() < miss_rate
        else s
        for s in samples
    ]


def drop_span(
    samples: Sequence[BallSample], t_start: float, t_end: float
) -> List[BallSample]:
    """Blackout: everything in [t_start, t_end) goes missing (occlusion, replay)."""
    return [
        BallSample(t=s.t, xy=None, confidence=0.0, n_candidates=0)
        if t_start <= s.t < t_end
        else s
        for s in samples
    ]


def add_noise(
    samples: Sequence[BallSample], sigma_m: float, rng: random.Random
) -> List[BallSample]:
    """Gaussian position noise on every detection."""
    out: List[BallSample] = []
    for s in samples:
        if s.xy is None:
            out.append(s)
            continue
        out.append(
            BallSample(
                t=s.t,
                xy=(s.xy[0] + rng.gauss(0.0, sigma_m), s.xy[1] + rng.gauss(0.0, sigma_m)),
                confidence=s.confidence,
                n_candidates=s.n_candidates,
            )
        )
    return out


def inject_static_false_positive(
    samples: Sequence[BallSample],
    fp_xy: Point2,
    t_start: float,
    t_end: float,
    confidence: float = 0.4,
    only_when_missing: bool = True,
) -> List[BallSample]:
    """The paper-debris model: a fixed pitch location detected as the ball.

    With ``only_when_missing`` (the common real case: the true ball outscores
    the debris when visible) only missing frames in the window are filled;
    otherwise the FP *overrides* real detections — the identity-switch case.
    """
    out: List[BallSample] = []
    for s in samples:
        in_window = t_start <= s.t < t_end
        if in_window and (not only_when_missing or not s.detected):
            out.append(BallSample(t=s.t, xy=fp_xy, confidence=confidence, n_candidates=1))
        elif in_window and s.detected:
            out.append(
                BallSample(
                    t=s.t, xy=s.xy, confidence=s.confidence,
                    n_candidates=max(s.n_candidates, 1) + 1,
                )
            )
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Player context helpers (for consistency metrics)
# ---------------------------------------------------------------------------

def players_near_ball(
    samples: Sequence[BallSample],
    n_players: int = 4,
    ring_radius_m: float = 6.0,
    fallback_xy: Point2 = (0.0, 0.0),
) -> List[Optional[List[Point2]]]:
    """Players loosely surrounding the ball each frame (open-play picture).

    On missing-ball frames the ring stays where the ball last was — players do
    not vanish with a dropped detection.
    """
    frames: List[Optional[List[Point2]]] = []
    anchor = fallback_xy
    for s in samples:
        if s.xy is not None:
            anchor = s.xy
        ring: List[Point2] = []
        for k in range(n_players):
            ang = 2.0 * math.pi * k / n_players
            ring.append((anchor[0] + ring_radius_m * math.cos(ang),
                         anchor[1] + ring_radius_m * math.sin(ang)))
        frames.append(ring)
    return frames


def players_static(
    samples: Sequence[BallSample], positions: Sequence[Point2]
) -> List[Optional[List[Point2]]]:
    """The same fixed player set for every frame."""
    fixed = list(positions)
    return [fixed for _ in samples]
