"""Calibration measurement and stabilisation layer.

Everything downstream of Phase 3 computes in metres through the per-frame
homography; this package makes that homography *measurable* (quality.py — the
no-ground-truth quality signals and the per-frame ``calibration_quality`` gate),
*stabilisable* (smoothing.py — pose-space temporal filtering), and *testable
before any real export exists* (testing/synthetic.py — scripted broadcast
cameras with injected detector pathologies). Design: /docs/calibration-design.md.
"""

from .camera import (
    CameraPose,
    Correspondence,
    HomographyFit,
    compose_homography,
    decompose_homography,
    fit_homography,
    project,
    rescale_homography,
    unproject,
)
from .pitch import (
    LANDMARKS,
    LINE_SEGMENTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    centre_circle_points,
    pitch_corners,
    visible_landmarks,
)
from .quality import (
    CalibrationQualityConfig,
    FrameCalibrationQuality,
    PlausibilityMetrics,
    ReprojectionMetrics,
    SupportMetrics,
    landmark_motion_px,
    plausibility_metrics,
    reprojection_metrics,
    score_frame,
    static_spans,
    support_metrics,
    teleport_count,
    world_jitter_m,
)
from .smoothing import (
    CameraSmoother,
    SmoothedFrame,
    SmoothedFrameStatus,
    SmootherConfig,
    smooth_sequence,
    smooth_sequence_bidirectional,
)

__all__ = [
    "CameraPose",
    "Correspondence",
    "HomographyFit",
    "compose_homography",
    "decompose_homography",
    "fit_homography",
    "project",
    "rescale_homography",
    "unproject",
    "LANDMARKS",
    "LINE_SEGMENTS",
    "PITCH_LENGTH_M",
    "PITCH_WIDTH_M",
    "centre_circle_points",
    "pitch_corners",
    "visible_landmarks",
    "CalibrationQualityConfig",
    "FrameCalibrationQuality",
    "PlausibilityMetrics",
    "ReprojectionMetrics",
    "SupportMetrics",
    "landmark_motion_px",
    "plausibility_metrics",
    "reprojection_metrics",
    "score_frame",
    "static_spans",
    "support_metrics",
    "teleport_count",
    "world_jitter_m",
    "CameraSmoother",
    "SmoothedFrame",
    "SmoothedFrameStatus",
    "SmootherConfig",
    "smooth_sequence",
    "smooth_sequence_bidirectional",
]
