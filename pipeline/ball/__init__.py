"""Ball measurement and motion layer.

The ball is the least reliable channel the pipeline will ever carry — and, as
of the 2026-07 diagnosis, it is not carried at all (the deployed detector keeps
only the person class; docs/ball-tracking-design.md §2). This package makes the
ball stream *measurable before it exists*: coverage (how often is there a ball,
and what shape are the gaps), physical plausibility (is this trajectory a
ball's), player consistency (is anyone chasing it), a per-frame ``ball_quality``
gate mirroring ``calibration_quality``, and a ball-appropriate motion model with
honest gap-bridging. Everything is validated against synthetic trajectories
(testing/synthetic.py) exactly as pipeline/calibration/ was.

The one signal that needs labels rather than self-consistency lives in
``evaluation.py``: hand-clicked ball centres (tools/annotator.html ball mode)
scored against probe detections in native pixel space, for the real
recall/precision the rest of this package deliberately does without.

Design: /docs/ball-tracking-design.md.
"""

from .consistency import (
    IsolationSpan,
    isolation_spans,
    nearest_player_distances,
)
from .coverage import (
    CoverageReport,
    Gap,
    GapContext,
    coverage_report,
    find_gaps,
    gap_contexts,
    zone_counts,
)
from .evaluation import (
    BallGroundTruth,
    DetectionEval,
    GroundTruthLabel,
    evaluate_detections,
    format_sweep,
    load_ground_truth,
    parse_ground_truth,
    sweep,
)
from .ingest import (
    ProbeFrame,
    load_ball_probe,
    load_homographies,
    probe_to_samples,
    project_pixel,
)
from .model import (
    BallSample,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    Point2,
    detected_indices,
    missing,
    out_of_bounds_m,
)
from .motion import (
    BallMotionConfig,
    BallSmoother,
    BallTrackStatus,
    SmoothedBall,
    interpolate_gaps,
    smooth_series,
    smooth_series_bidirectional,
)
from .plausibility import (
    KinematicFlag,
    PlausibilityReport,
    kinematic_flags,
    plausibility_report,
)
from .quality import (
    BallQualityConfig,
    FrameBallQuality,
    score_ball_series,
    temporal_support,
)
