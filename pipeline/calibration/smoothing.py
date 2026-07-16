"""Temporal smoothing of per-frame calibrations, in camera-pose space.

Why this exists (calibration-design.md §7): nbjw_calib estimates every frame
independently — quantised keypoints and unstable line intersections make the
resulting homography sequence jitter even when the camera is physically still,
and single bad frames (hallucinated landmarks, degenerate geometry) produce
excursions no per-frame check can repair. A broadcast main camera, however, is a
fixed mount: position is constant to centimetres, and pan/tilt/zoom follow
smooth operator motions. That prior is expressible only in pose space — which is
why the smoother runs on (pan, tilt, roll, ln f, position), never on raw
homography entries (see camera.py module docstring for the argument).

Mechanics: per-parameter alpha-beta filtering with innovation gating.
- Accepted measurement -> state tracks it (gain scaled by the frame's
  calibration quality, so a shaky frame nudges rather than yanks).
- Gated-out measurement (innovation beyond the per-parameter gate) -> COAST on
  the constant-velocity prediction with decaying velocity.
- A run of mutually consistent rejections -> REANCHOR: the camera genuinely
  moved faster than the gates allow (whip pan) or the filter diverged; snap to
  the rejected consensus rather than fight it.
- Too long without an accepted measurement -> LOST (emit None; downstream treats
  the frame as uncalibrated — honesty over interpolation).

Segments: the smoother never bridges a broadcast cut. Callers reset() at segment
boundaries (the streaming design's cut detector already provides them).

Assumes uniformly spaced frames (velocities are per-frame); resample first if
feeding a variable-rate stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .camera import CameraPose, Matrix, Point2, compose_homography

_N_PARAMS = 7  # pan, tilt, roll, ln f, cx, cy, cz
_SMOOTH_DYNAMIC = (0, 1, 2, 3)  # parameters that get a velocity model
_POSITION = (4, 5, 6)  # fixed-mount prior: near-constant, tiny gain


class SmoothedFrameStatus(Enum):
    MEASURED = "measured"      # measurement accepted and fused
    COASTED = "coasted"        # prediction held (gated-out or missing measurement)
    REANCHORED = "reanchored"  # state snapped to a consistent run of rejections
    LOST = "lost"              # nothing trustworthy for too long: no output


@dataclass(frozen=True)
class SmootherConfig:
    """All gains/gates named, dataclass-per-config like detector configs.

    Gates are per-frame innovation limits at 25 fps: a real operator pan tops out
    around ~20 deg/s (~0.8 deg/frame), so a 3 deg jump between prediction and
    measurement is far outside camera physics — reject it. Position gates are
    generous because per-frame position estimates scatter widely even for good
    frames; the tiny position gain is what enforces the fixed-mount prior.
    """

    angle_gain: float = 0.35
    focal_gain: float = 0.35
    position_gain: float = 0.04
    velocity_gain_fraction: float = 0.25  # beta = fraction * alpha
    min_quality: float = 0.15  # below this a measurement is not even considered
    pan_gate_deg: float = 3.0
    tilt_gate_deg: float = 2.0
    roll_gate_deg: float = 3.0
    focal_gate_lnf: float = 0.15
    position_gate_m: float = 10.0
    velocity_decay: float = 0.85  # per coasted frame
    max_coast_frames: int = 25  # ~1 s at 25 fps, then LOST
    reanchor_run: int = 3  # consecutive consistent rejections that force a snap


@dataclass(frozen=True)
class SmoothedFrame:
    """One frame of smoother output; ``pose``/``h`` are None when LOST."""

    status: SmoothedFrameStatus
    pose: Optional[CameraPose]
    h: Optional[Matrix]


def _gates(config: SmootherConfig) -> List[float]:
    return [
        config.pan_gate_deg,
        config.tilt_gate_deg,
        config.roll_gate_deg,
        config.focal_gate_lnf,
        config.position_gate_m,
        config.position_gate_m,
        config.position_gate_m,
    ]


class CameraSmoother:
    """Causal (live-capable) smoother over decomposed camera poses.

    Feed one (pose, quality) per frame via ``update``; poses come from
    ``decompose_homography`` on the per-frame estimate, quality from
    ``quality.score_frame``. The same class serves batch use through
    ``smooth_sequence`` below (live architecture parity principle: same numbers,
    later — live-architecture-design.md §2).
    """

    def __init__(self, config: SmootherConfig = SmootherConfig(),
                 principal_point: Point2 = (640.0, 360.0)):
        self.config = config
        self.principal_point = principal_point
        self._state: Optional[List[float]] = None
        self._velocity: List[float] = [0.0] * _N_PARAMS
        self._coast_count = 0
        self._reject_buffer: List[List[float]] = []

    def reset(self) -> None:
        """Call at every segment boundary (broadcast cut): nothing bridges a cut."""
        self._state = None
        self._velocity = [0.0] * _N_PARAMS
        self._coast_count = 0
        self._reject_buffer = []

    # -- internals ----------------------------------------------------------

    def _predict(self) -> List[float]:
        assert self._state is not None
        pred = self._state[:]
        for i in _SMOOTH_DYNAMIC:
            pred[i] += self._velocity[i]
        return pred

    def _emit(self, status: SmoothedFrameStatus) -> SmoothedFrame:
        if self._state is None or status is SmoothedFrameStatus.LOST:
            return SmoothedFrame(status=SmoothedFrameStatus.LOST, pose=None, h=None)
        pose = CameraPose.from_vector(self._state, self.principal_point)
        return SmoothedFrame(status=status, pose=pose, h=compose_homography(pose))

    def _coast(self) -> SmoothedFrame:
        self._coast_count += 1
        if self._state is None or self._coast_count > self.config.max_coast_frames:
            return self._emit(SmoothedFrameStatus.LOST)
        self._state = self._predict()
        for i in _SMOOTH_DYNAMIC:
            self._velocity[i] *= self.config.velocity_decay
        return self._emit(SmoothedFrameStatus.COASTED)

    def _consistent_rejections(self) -> bool:
        if len(self._reject_buffer) < self.config.reanchor_run:
            return False
        recent = self._reject_buffer[-self.config.reanchor_run:]
        gates = _gates(self.config)
        for i in range(_N_PARAMS):
            vals = [m[i] for m in recent]
            if max(vals) - min(vals) > gates[i]:
                return False
        return True

    # -- public -------------------------------------------------------------

    def update(self, pose: Optional[CameraPose], quality: float) -> SmoothedFrame:
        cfg = self.config
        if pose is None or quality < cfg.min_quality:
            return self._coast()
        meas = pose.as_vector()

        if self._state is None:
            self._state = meas[:]
            self._velocity = [0.0] * _N_PARAMS
            self._coast_count = 0
            self._reject_buffer = []
            return self._emit(SmoothedFrameStatus.MEASURED)

        pred = self._predict()
        innovation = [meas[i] - pred[i] for i in range(_N_PARAMS)]
        gates = _gates(cfg)
        if any(abs(innovation[i]) > gates[i] for i in range(_N_PARAMS)):
            self._reject_buffer.append(meas)
            if self._consistent_rejections():
                recent = self._reject_buffer[-cfg.reanchor_run:]
                self._state = [
                    sorted(m[i] for m in recent)[len(recent) // 2]
                    for i in range(_N_PARAMS)
                ]
                self._velocity = [0.0] * _N_PARAMS
                self._coast_count = 0
                self._reject_buffer = []
                return self._emit(SmoothedFrameStatus.REANCHORED)
            return self._coast()

        self._reject_buffer = []
        self._coast_count = 0
        # Quality-weighted alpha-beta update; a low-quality frame nudges, a
        # high-quality frame corrects.
        weight = max(0.2, min(1.0, quality))
        self._state = pred
        for i in range(_N_PARAMS):
            alpha = cfg.position_gain if i in _POSITION else (
                cfg.focal_gain if i == 3 else cfg.angle_gain
            )
            alpha *= weight
            self._state[i] += alpha * innovation[i]
            if i in _SMOOTH_DYNAMIC:
                self._velocity[i] += cfg.velocity_gain_fraction * alpha * innovation[i]
        return self._emit(SmoothedFrameStatus.MEASURED)


def smooth_sequence(
    measurements: Sequence[Tuple[Optional[CameraPose], float]],
    config: SmootherConfig = SmootherConfig(),
    principal_point: Point2 = (640.0, 360.0),
) -> List[SmoothedFrame]:
    """Causal pass over one continuous segment of (pose, quality) measurements."""
    smoother = CameraSmoother(config, principal_point)
    return [smoother.update(pose, quality) for pose, quality in measurements]


def smooth_sequence_bidirectional(
    measurements: Sequence[Tuple[Optional[CameraPose], float]],
    config: SmootherConfig = SmootherConfig(),
    principal_point: Point2 = (640.0, 360.0),
) -> List[SmoothedFrame]:
    """Batch mode: forward and backward causal passes, states averaged.

    Removes the causal filter's phase lag during pans (forward lags behind,
    backward lags ahead; the mean cancels to first order). Live mode uses the
    plain causal pass — parity in the smoothing *idiom*, batch just sees the
    future too (live-architecture-design.md §2 allows exactly this asymmetry).
    """
    fwd = smooth_sequence(measurements, config, principal_point)
    bwd = list(reversed(smooth_sequence(list(reversed(measurements)), config, principal_point)))
    blended: List[SmoothedFrame] = []
    for f, b in zip(fwd, bwd):
        if f.pose is None and b.pose is None:
            blended.append(SmoothedFrame(SmoothedFrameStatus.LOST, None, None))
            continue
        if f.pose is None or b.pose is None:
            blended.append(f if f.pose is not None else b)
            continue
        fv, bv = f.pose.as_vector(), b.pose.as_vector()
        mv = [(x + y) / 2.0 for x, y in zip(fv, bv)]
        pose = CameraPose.from_vector(mv, principal_point)
        status = (
            SmoothedFrameStatus.MEASURED
            if SmoothedFrameStatus.MEASURED in (f.status, b.status)
            else f.status
        )
        blended.append(SmoothedFrame(status=status, pose=pose, h=compose_homography(pose)))
    return blended
