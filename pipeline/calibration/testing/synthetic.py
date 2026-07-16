"""Synthetic broadcast-camera trajectories and noisy keypoint observations.

The measurement layer must be testable before any real per-frame homography has
been exported (calibration-design.md §8): we script a physically plausible main
camera (fixed mount at the halfway line, pan/tilt/zoom only), derive exact
ground-truth homographies from it, then corrupt observations the same ways the
real detector does — pixel noise, grid quantisation (nbjw_calib extracts keypoint
coordinates on an integer half-resolution heatmap grid), dropout, and outright
hallucinated landmarks (frame 060 of the OFI run shows "Big rect. left main"
detected on empty grass). Refitting from those corrupted observations reproduces,
in miniature, the per-frame estimator whose output we need to score and smooth.

Determinism: every stochastic helper takes a seed or ``random.Random``; tests
must never consume global randomness.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..camera import (
    CameraPose,
    Correspondence,
    HomographyFit,
    Matrix,
    Point2,
    compose_homography,
    fit_homography,
    project,
)
from ..pitch import LANDMARKS, visible_landmarks

FRAME_WIDTH = 1280.0
FRAME_HEIGHT = 720.0

#: nbjw_calib reads keypoints off a 480x270 heatmap grid; at a 1280-wide frame
#: that is a ~2.67 px quantisation step. Scenarios use it as the realistic default.
NBJW_QUANTIZE_PX = FRAME_WIDTH / 480.0


def main_camera_pose(
    pan_deg: float = 0.0,
    tilt_deg: float = 12.0,
    roll_deg: float = 0.0,
    focal_px: float = 1400.0,
    position: Tuple[float, float, float] = (0.0, -55.0, 18.0),
    frame_width: float = FRAME_WIDTH,
    frame_height: float = FRAME_HEIGHT,
) -> CameraPose:
    """A plausible main broadcast camera: halfway line, ~20 m up, ~20 m behind the near touchline."""
    return CameraPose(
        pan_deg=pan_deg,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
        focal_px=focal_px,
        position=position,
        principal_point=(frame_width / 2.0, frame_height / 2.0),
    )


def near_goal_pose(focal_px: float = 4300.0, side: str = "right") -> CameraPose:
    """The failure framing from the OFI run: tight on one goal area.

    Same physical camera as ``main_camera_pose`` — only pan/tilt/zoom change —
    but the visible landmark set collapses to a handful of points hugging the
    goal line, ~20 m across at one pitch corner. Extrapolating that fit to the
    rest of the pitch is exactly the degenerate-geometry scenario (design §4, H2).
    """
    sign = 1.0 if side == "right" else -1.0
    return main_camera_pose(pan_deg=sign * 40.5, tilt_deg=10.2, focal_px=focal_px)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def _ease(t: float) -> float:
    """Smoothstep: cameramen accelerate and decelerate pans smoothly."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def static_trajectory(pose: CameraPose, n_frames: int) -> List[CameraPose]:
    return [pose] * n_frames


def pan_zoom_trajectory(
    n_frames: int,
    pan_start_deg: float,
    pan_end_deg: float,
    focal_start_px: float = 1400.0,
    focal_end_px: float = 1400.0,
    tilt_deg: float = 12.0,
) -> List[CameraPose]:
    """A single eased pan (optionally with a zoom ramp) on the fixed mount."""
    poses = []
    for i in range(n_frames):
        t = _ease(i / max(1, n_frames - 1))
        poses.append(
            main_camera_pose(
                pan_deg=pan_start_deg + (pan_end_deg - pan_start_deg) * t,
                tilt_deg=tilt_deg,
                focal_px=focal_start_px + (focal_end_px - focal_start_px) * t,
            )
        )
    return poses


def broadcast_trajectory(n_frames: int = 200) -> List[CameraPose]:
    """Hold at midfield -> pan+zoom to the right goal -> hold tight -> pan back.

    Covers the full difficulty range in one sequence: a static stretch (where any
    frame-to-frame wobble is pure estimation error), a smooth camera motion, and
    a near-goal hold where the landmark geometry is degenerate.
    """
    quarter = n_frames // 4
    segs: List[CameraPose] = []
    segs += static_trajectory(main_camera_pose(), quarter)
    segs += pan_zoom_trajectory(quarter, 0.0, 41.0, 1400.0, 3200.0, tilt_deg=11.7)
    segs += static_trajectory(near_goal_pose(), quarter)
    segs += pan_zoom_trajectory(n_frames - 3 * quarter, 41.0, 0.0, 3200.0, 1400.0, tilt_deg=11.7)
    return segs


# ---------------------------------------------------------------------------
# Observation corruption
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationNoise:
    """How the synthetic detector corrupts perfect projections.

    ``noise_px``: isotropic Gaussian localisation error.
    ``quantize_px``: snap to a grid of this pitch (0 = off) — models nbjw_calib's
    integer heatmap coordinates, a temporally *white* error source.
    ``dropout_rate``: probability a visible landmark is simply not detected.
    ``n_hallucinations``: landmarks NOT visible in the frame reported anyway at a
    uniform random in-frame position (the OOD failure mode seen on the OFI clip).
    """

    noise_px: float = 1.5
    quantize_px: float = 0.0
    dropout_rate: float = 0.0
    n_hallucinations: int = 0


CLEAN = ObservationNoise(noise_px=0.0)
REALISTIC = ObservationNoise(noise_px=1.5, quantize_px=NBJW_QUANTIZE_PX, dropout_rate=0.15)
DEGRADED = ObservationNoise(
    noise_px=4.0, quantize_px=NBJW_QUANTIZE_PX, dropout_rate=0.3, n_hallucinations=2
)


def observe_landmarks(
    pose: CameraPose,
    noise: ObservationNoise = CLEAN,
    rng: Optional[random.Random] = None,
    frame_width: float = FRAME_WIDTH,
    frame_height: float = FRAME_HEIGHT,
) -> List[Correspondence]:
    """Simulated per-frame keypoint detections: (world, observed pixel) pairs."""
    rng = rng or random.Random(0)
    h = compose_homography(pose)
    visible = visible_landmarks(h, frame_width, frame_height)
    observations: List[Correspondence] = []
    for name, wxy in sorted(visible.items()):
        if noise.dropout_rate > 0.0 and rng.random() < noise.dropout_rate:
            continue
        p = project(h, wxy)
        if p is None:
            continue
        u, v = p
        if noise.noise_px > 0.0:
            u += rng.gauss(0.0, noise.noise_px)
            v += rng.gauss(0.0, noise.noise_px)
        if noise.quantize_px > 0.0:
            u = round(u / noise.quantize_px) * noise.quantize_px
            v = round(v / noise.quantize_px) * noise.quantize_px
        observations.append((wxy, (u, v)))

    if noise.n_hallucinations > 0:
        hidden = sorted(set(LANDMARKS) - set(visible))
        rng.shuffle(hidden)
        for name in hidden[: noise.n_hallucinations]:
            observations.append(
                (
                    LANDMARKS[name],
                    (rng.uniform(0.0, frame_width), rng.uniform(0.0, frame_height)),
                )
            )
    return observations


def per_frame_fits(
    trajectory: Sequence[CameraPose],
    noise: ObservationNoise = REALISTIC,
    seed: int = 0,
) -> List[Optional[HomographyFit]]:
    """Refit a homography per frame from corrupted observations — the miniature
    stand-in for the real per-frame estimator whose output quality.py scores and
    smoothing.py stabilises."""
    return [fit for fit, _ in simulate_sequence(trajectory, noise, seed)]


def simulate_sequence(
    trajectory: Sequence[CameraPose],
    noise: ObservationNoise = REALISTIC,
    seed: int = 0,
) -> List[Tuple[Optional[HomographyFit], List[Correspondence]]]:
    """Per-frame (fit, observations) pairs — what a patched real export gives us.

    The observations travel with the fit because the quality score needs both:
    residual/support evidence comes from the correspondences, conditioning from
    the fit diagnostics.
    """
    rng = random.Random(seed)
    out: List[Tuple[Optional[HomographyFit], List[Correspondence]]] = []
    for pose in trajectory:
        obs = observe_landmarks(pose, noise, rng)
        out.append((fit_homography(obs) if len(obs) >= 4 else None, obs))
    return out


def true_homographies(trajectory: Sequence[CameraPose]) -> List[Matrix]:
    return [compose_homography(p) for p in trajectory]
