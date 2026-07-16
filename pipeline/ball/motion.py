"""Ball motion model: gated smoothing and honest gap-bridging.

The ball is not a person, and the person tracker's assumptions are wrong for it
in both directions (design §2): appearance embeddings carry no information at
5-10 px, and a constant-velocity-with-person-gates Kalman filter rejects real
ball motion (0 to 30 m/s in one kick) while accepting false positives near the
last position. What the ball *does* obey, in 2D pitch coordinates:

- between contacts it is nearly constant-velocity (a flight's ground projection
  is exactly that without drag; a rolling ball decays gently);
- at contacts velocity changes instantly and arbitrarily (a kick is a step —
  fight it and you smear every pass; hence REANCHOR, not tighter gates);
- unobserved spans grow uncertainty fast: coast briefly, then admit LOST.

Same alpha-beta + innovation-gate + COAST/REANCHOR/LOST machinery as the camera
smoother (calibration-design.md §7), with dt-aware gates because ball streams
have real gaps, and a velocity cap because nothing on a pitch moves at 60 m/s.

``interpolate_gaps`` is the batch counterpart: fill only gaps short enough that
a ballistic/rolled ball couldn't have done anything interesting inside them
(default <= 1.0 s, matching the feature layer's max_gap_s), tag every filled
sample ``interpolated=True`` with decayed confidence, and never bridge across a
segment boundary — the "never interpolate through long gaps" contract of
phase4-5-design.md §2.2, now enforced in code rather than prose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .model import BallSample, Point2


class BallTrackStatus(Enum):
    MEASURED = "measured"
    COASTED = "coasted"
    REANCHORED = "reanchored"
    LOST = "lost"


@dataclass(frozen=True)
class BallMotionConfig:
    """Gains/gates named. Gates are dt-aware: innovation is compared against
    ``gate_base_m + gate_growth_ms * dt`` so a legitimate 0.5 s gap widens the
    acceptance region instead of guaranteeing rejection."""

    position_gain: float = 0.5
    velocity_gain_fraction: float = 0.3
    min_quality: float = 0.1
    gate_base_m: float = 2.0
    gate_growth_ms: float = 15.0
    max_speed_ms: float = 40.0  # velocity clamp after every update
    coast_speed_decay_per_s: float = 0.6  # exponential; rolling ball slows, lost ball shouldn't run away
    max_coast_s: float = 1.2
    reanchor_run: int = 3
    reanchor_agreement_m: float = 3.0


@dataclass(frozen=True)
class SmoothedBall:
    """One frame of smoother output; position/velocity None when LOST."""

    status: BallTrackStatus
    t: float
    xy: Optional[Point2]
    velocity: Optional[Point2]


class BallSmoother:
    """Causal (live-capable) 2D alpha-beta filter with kick-aware reanchoring.

    Feed one ``update(t, xy, quality)`` per frame; ``xy`` None coasts. Quality
    comes from ``quality.score_ball_series`` (or 1.0 to trust the detector).
    ``reset()`` at every segment boundary — nothing bridges a broadcast cut.
    """

    def __init__(self, config: BallMotionConfig = BallMotionConfig()):
        self.config = config
        self._t: Optional[float] = None
        self._pos: Optional[List[float]] = None
        self._vel: List[float] = [0.0, 0.0]
        self._coast_time = 0.0
        self._rejects: List[Tuple[float, Point2]] = []

    def reset(self) -> None:
        self._t = None
        self._pos = None
        self._vel = [0.0, 0.0]
        self._coast_time = 0.0
        self._rejects = []

    # -- internals ----------------------------------------------------------

    def _predict(self, dt: float) -> List[float]:
        assert self._pos is not None
        return [self._pos[0] + self._vel[0] * dt, self._pos[1] + self._vel[1] * dt]

    def _clamp_velocity(self) -> None:
        speed = math.hypot(*self._vel)
        cap = self.config.max_speed_ms
        if speed > cap:
            scale = cap / speed
            self._vel = [self._vel[0] * scale, self._vel[1] * scale]

    def _emit(self, status: BallTrackStatus, t: float) -> SmoothedBall:
        if self._pos is None or status is BallTrackStatus.LOST:
            return SmoothedBall(BallTrackStatus.LOST, t, None, None)
        return SmoothedBall(status, t, (self._pos[0], self._pos[1]), (self._vel[0], self._vel[1]))

    def _coast(self, t: float, dt: float) -> SmoothedBall:
        self._coast_time += dt
        if self._pos is None or self._coast_time > self.config.max_coast_s:
            self._pos = None
            return self._emit(BallTrackStatus.LOST, t)
        self._pos = self._predict(dt)
        decay = math.exp(-self.config.coast_speed_decay_per_s * dt)
        self._vel = [self._vel[0] * decay, self._vel[1] * decay]
        self._t = t
        return self._emit(BallTrackStatus.COASTED, t)

    def _consistent_rejections(self) -> bool:
        cfg = self.config
        if len(self._rejects) < cfg.reanchor_run:
            return False
        recent = [xy for (_, xy) in self._rejects[-cfg.reanchor_run:]]
        # A kicked ball moves between rejected frames, so consistency means
        # "consecutive steps compatible with one ball in motion", not "all
        # rejections in one place": each step may span agreement + one
        # ball-speed step, no more.
        for a, b in zip(recent, recent[1:]):
            step = math.hypot(b[0] - a[0], b[1] - a[1])
            if step > cfg.reanchor_agreement_m + cfg.max_speed_ms * self._reject_dt():
                return False
        return True

    def _reject_dt(self) -> float:
        ts = [t for (t, _) in self._rejects[-self.config.reanchor_run:]]
        if len(ts) < 2:
            return 0.0
        return (ts[-1] - ts[0]) / (len(ts) - 1)

    # -- public -------------------------------------------------------------

    def update(self, t: float, xy: Optional[Point2], quality: float = 1.0) -> SmoothedBall:
        cfg = self.config
        dt = 0.0 if self._t is None else max(0.0, t - self._t)

        if xy is None or quality < cfg.min_quality:
            if self._t is None:
                return SmoothedBall(BallTrackStatus.LOST, t, None, None)
            return self._coast(t, dt)

        if self._pos is None:
            self._t = t
            self._pos = [xy[0], xy[1]]
            self._vel = [0.0, 0.0]
            self._coast_time = 0.0
            self._rejects = []
            return self._emit(BallTrackStatus.MEASURED, t)

        pred = self._predict(dt)
        innovation = math.hypot(xy[0] - pred[0], xy[1] - pred[1])
        gate = cfg.gate_base_m + cfg.gate_growth_ms * dt
        if innovation > gate:
            self._rejects.append((t, xy))
            if self._consistent_rejections():
                # A kick (or a filter divergence): snap to the rejected
                # consensus and derive velocity from its trend.
                recent = self._rejects[-cfg.reanchor_run:]
                (t0, p0), (t1, p1) = recent[0], recent[-1]
                self._pos = [p1[0], p1[1]]
                span = t1 - t0
                self._vel = (
                    [(p1[0] - p0[0]) / span, (p1[1] - p0[1]) / span] if span > 0 else [0.0, 0.0]
                )
                self._clamp_velocity()
                self._t = t
                self._coast_time = 0.0
                self._rejects = []
                return self._emit(BallTrackStatus.REANCHORED, t)
            return self._coast(t, dt)

        self._rejects = []
        self._coast_time = 0.0
        weight = max(0.2, min(1.0, quality))
        alpha = cfg.position_gain * weight
        beta = cfg.velocity_gain_fraction * alpha
        self._pos = pred
        self._pos[0] += alpha * (xy[0] - pred[0])
        self._pos[1] += alpha * (xy[1] - pred[1])
        if dt > 0.0:
            self._vel[0] += beta * (xy[0] - pred[0]) / dt
            self._vel[1] += beta * (xy[1] - pred[1]) / dt
        self._clamp_velocity()
        self._t = t
        return self._emit(BallTrackStatus.MEASURED, t)


def smooth_series(
    samples: Sequence[BallSample],
    qualities: Optional[Sequence[float]] = None,
    config: BallMotionConfig = BallMotionConfig(),
) -> List[SmoothedBall]:
    """Causal pass over one continuous segment of samples."""
    smoother = BallSmoother(config)
    out: List[SmoothedBall] = []
    for i, s in enumerate(samples):
        q = qualities[i] if qualities is not None else (1.0 if s.detected else 0.0)
        out.append(smoother.update(s.t, s.xy, q))
    return out


def smooth_series_bidirectional(
    samples: Sequence[BallSample],
    qualities: Optional[Sequence[float]] = None,
    config: BallMotionConfig = BallMotionConfig(),
) -> List[SmoothedBall]:
    """Batch mode: forward and backward causal passes, positions averaged.

    Cancels the causal filter's phase lag to first order — the same
    batch-vs-live asymmetry the calibration smoother uses
    (live-architecture-design.md §2 allows exactly this)."""
    fwd = smooth_series(samples, qualities, config)
    rev_q = list(reversed(qualities)) if qualities is not None else None
    bwd = list(reversed(smooth_series(list(reversed(samples)), rev_q, config)))
    blended: List[SmoothedBall] = []
    for f, b in zip(fwd, bwd):
        if f.xy is None and b.xy is None:
            blended.append(f)
            continue
        if f.xy is None or b.xy is None:
            blended.append(f if f.xy is not None else b)
            continue
        xy = ((f.xy[0] + b.xy[0]) / 2.0, (f.xy[1] + b.xy[1]) / 2.0)
        # Backward-pass velocity is sign-flipped in real time.
        vel: Optional[Point2] = None
        if f.velocity is not None and b.velocity is not None:
            vel = (
                (f.velocity[0] - b.velocity[0]) / 2.0,
                (f.velocity[1] - b.velocity[1]) / 2.0,
            )
        status = (
            BallTrackStatus.MEASURED
            if BallTrackStatus.MEASURED in (f.status, b.status)
            else f.status
        )
        blended.append(SmoothedBall(status, f.t, xy, vel))
    return blended


def interpolate_gaps(
    samples: Sequence[BallSample],
    max_gap_s: float = 1.0,
    confidence_decay: float = 0.5,
) -> List[BallSample]:
    """Fill missing samples inside short gaps by linear interpolation.

    Only gaps whose bracketing detections are <= ``max_gap_s`` apart are
    bridged; longer gaps (and leading/trailing gaps) stay missing — that is the
    documented contract, not a limitation. Filled samples carry
    ``interpolated=True`` and confidence = ``confidence_decay`` x the bracketing
    minimum, so quality scoring caps them below observed evidence.
    """
    out = list(samples)
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
        if start == 0 or end == n:
            continue  # edge gap: nothing to anchor one side on
        prev_s, next_s = samples[start - 1], samples[end]
        span = next_s.t - prev_s.t
        if span <= 0.0 or span > max_gap_s:
            continue
        assert prev_s.xy is not None and next_s.xy is not None
        conf = confidence_decay * min(prev_s.confidence, next_s.confidence)
        for j in range(start, end):
            frac = (samples[j].t - prev_s.t) / span
            xy = (
                prev_s.xy[0] + frac * (next_s.xy[0] - prev_s.xy[0]),
                prev_s.xy[1] + frac * (next_s.xy[1] - prev_s.xy[1]),
            )
            out[j] = replace(samples[j], xy=xy, confidence=conf, interpolated=True)
    return out
