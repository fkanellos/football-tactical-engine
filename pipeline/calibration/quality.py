"""Per-frame calibration quality: measurable signals, no ground truth required.

The measurement problem (calibration-design.md §6): we have no calibration ground
truth for our footage, so every signal here is either (a) internal consistency —
how well the fitted homography explains the detected landmarks it came from and
how well-conditioned that fit was; (b) physical plausibility — does the implied
camera exist and does the implied pitch make metric sense; or (c) temporal
consistency — does the estimate move like a camera on a fixed mount.

Cheap-and-diagnostic vs vanity (design §6.4): everything in this module is O(few
dozen small matrix ops) per frame; nothing needs pixels. The output of record is
``FrameCalibrationQuality.quality`` in [0, 1] — the gate the Phase 4 feature layer
already budgets for (phase4-5-design.md §2.4 data-quality signals): it multiplies
into detector confidence and hard-gates metre-sensitive features when low.

Score idiom matches the detectors: a product of factors in [0, 1], each a linear
ramp between a named "good" and "bad" threshold in the config dataclass — soft
enough to rank frames, transparent enough to debug from the component fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .camera import (
    Correspondence,
    Matrix,
    Point2,
    decompose_homography,
    project,
    unproject,
)
from .linalg import inv3, mat_vec
from .pitch import PITCH_LENGTH_M, PITCH_WIDTH_M, pitch_corners

#: Discount applied once when a frame carries no exported keypoints (=> neither
#: residual nor support evidence exists) — same philosophy as the detectors'
#: MISSING_FEATURE_FACTOR: missing evidence lowers confidence, it does not zero
#: it. A bare-homography frame can therefore never score above ~0.6.
MISSING_EVIDENCE_FACTOR = 0.7
#: Discount applied once when there is no temporal reference to check against
#: (first frame of a segment, or the frame after a cut). Milder: absence here is
#: structural, not suspicious.
MISSING_TEMPORAL_FACTOR = 0.85


def ramp(value: float, good: float, bad: float) -> float:
    """Linear score ramp: 1 at/beyond ``good``, 0 at/beyond ``bad``.

    Works in either direction (good < bad penalises large values, good > bad
    penalises small ones).
    """
    if good == bad:
        return 1.0 if value == good else 0.0
    t = (value - bad) / (good - bad)
    return max(0.0, min(1.0, t))


# ---------------------------------------------------------------------------
# Small polygon helpers (pure python, tiny inputs)
# ---------------------------------------------------------------------------

def convex_hull(points: Sequence[Point2]) -> List[Point2]:
    """Andrew's monotone chain; returns CCW hull without the repeated endpoint."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def half(iterable: Sequence[Point2]) -> List[Point2]:
        chain: List[Point2] = []
        for p in iterable:
            while len(chain) >= 2 and (
                (chain[-1][0] - chain[-2][0]) * (p[1] - chain[-2][1])
                - (chain[-1][1] - chain[-2][1]) * (p[0] - chain[-2][0])
            ) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    lower = half(pts)
    upper = half(list(reversed(pts)))
    return lower[:-1] + upper[:-1]


def polygon_area(polygon: Sequence[Point2]) -> float:
    if len(polygon) < 3:
        return 0.0
    s = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def clip_polygon_to_rect(
    polygon: Sequence[Point2], x_min: float, y_min: float, x_max: float, y_max: float
) -> List[Point2]:
    """Sutherland-Hodgman clip of a convex/simple polygon to an axis-aligned rect."""
    def clip_edge(poly: List[Point2], inside, intersect) -> List[Point2]:
        out: List[Point2] = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
        return out

    def x_cut(bound: float):
        def f(p: Point2, q: Point2) -> Point2:
            t = (bound - p[0]) / (q[0] - p[0])
            return (bound, p[1] + t * (q[1] - p[1]))
        return f

    def y_cut(bound: float):
        def f(p: Point2, q: Point2) -> Point2:
            t = (bound - p[1]) / (q[1] - p[1])
            return (p[0] + t * (q[0] - p[0]), bound)
        return f

    poly = list(polygon)
    for inside, intersect in (
        (lambda p: p[0] >= x_min, x_cut(x_min)),
        (lambda p: p[0] <= x_max, x_cut(x_max)),
        (lambda p: p[1] >= y_min, y_cut(y_min)),
        (lambda p: p[1] <= y_max, y_cut(y_max)),
    ):
        if not poly:
            return []
        poly = clip_edge(poly, inside, intersect)
    return poly


# ---------------------------------------------------------------------------
# Component metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReprojectionMetrics:
    """Residuals of a homography against detected landmark correspondences.

    Pixel residuals answer "does H explain what the detector saw"; metre
    residuals answer the question Phase 4 actually cares about — "by how many
    metres would this frame mislocate things at pitch scale". Both are biased
    low when computed on the same points the fit consumed; they remain sharply
    diagnostic for degeneracy because a degenerate H *cannot* keep residuals low
    at held-out or extrapolated landmarks (design §6.2).
    """

    n_points: int
    rms_px: float
    max_px: float
    rms_m: float
    max_m: float


def reprojection_metrics(
    h: Matrix, correspondences: Sequence[Correspondence]
) -> Optional[ReprojectionMetrics]:
    if not correspondences:
        return None
    hinv = inv3(h)
    sq_px, sq_m, max_px, max_m, n = 0.0, 0.0, 0.0, 0.0, 0
    for (wxy, uv) in correspondences:
        p = project(h, wxy)
        if p is None:
            continue
        d_px = math.hypot(p[0] - uv[0], p[1] - uv[1])
        w = unproject(hinv, uv, inverse_given=True) if hinv is not None else None
        d_m = math.hypot(w[0] - wxy[0], w[1] - wxy[1]) if w is not None else float("inf")
        sq_px += d_px * d_px
        sq_m += min(d_m, 1e6) ** 2
        max_px = max(max_px, d_px)
        max_m = max(max_m, d_m)
        n += 1
    if n == 0:
        return None
    return ReprojectionMetrics(
        n_points=n,
        rms_px=math.sqrt(sq_px / n),
        max_px=max_px,
        rms_m=math.sqrt(sq_m / n),
        max_m=max_m,
    )


@dataclass(frozen=True)
class SupportMetrics:
    """Geometry of the landmark set the fit stood on.

    A homography from 4+ points clustered in one penalty box has tiny residuals
    and garbage extrapolation — residuals alone cannot see that. ``spread_major/
    minor_m`` are the principal std deviations of the world points;
    ``collinearity`` = minor/major (0 = perfectly collinear = one-parameter
    family of homographies fits equally well); ``world_coverage`` = hull area /
    pitch area.
    """

    n_points: int
    spread_major_m: float
    spread_minor_m: float
    collinearity: float
    world_coverage: float


def support_metrics(world_points: Sequence[Point2]) -> Optional[SupportMetrics]:
    n = len(world_points)
    if n == 0:
        return None
    cx = sum(p[0] for p in world_points) / n
    cy = sum(p[1] for p in world_points) / n
    sxx = sum((p[0] - cx) ** 2 for p in world_points) / n
    syy = sum((p[1] - cy) ** 2 for p in world_points) / n
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in world_points) / n
    # Eigenvalues of the 2x2 covariance.
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, tr * tr / 4.0 - det))
    lam1, lam2 = tr / 2.0 + disc, max(0.0, tr / 2.0 - disc)
    major, minor = math.sqrt(lam1), math.sqrt(lam2)
    hull_area = polygon_area(convex_hull(world_points))
    return SupportMetrics(
        n_points=n,
        spread_major_m=major,
        spread_minor_m=minor,
        collinearity=(minor / major) if major > 1e-9 else 0.0,
        world_coverage=hull_area / (PITCH_LENGTH_M * PITCH_WIDTH_M),
    )


@dataclass(frozen=True)
class PlausibilityMetrics:
    """What the homography claims about the world, checked against physics.

    Computable from H alone — the only component that needs *no* exported
    keypoints, so it can gate even a bare per-frame homography stream.

    ``horizon_v_fraction``: vertical position of the ground plane's vanishing
    line at the frame's centre column, as a fraction of frame height (0 = top).
    A calibrated main-camera view keeps it in the top strip or above the frame
    (< ~0.3); a value near/below the frame middle means the estimate claims a
    near-ground camera looking flat — implausible for broadcast tracking shots.
    ``chirality_ok``: all four pitch corners project with a consistent
    homogeneous sign (no part of the pitch "behind the camera") — garbage fits
    routinely violate this.
    ``pitch_fraction_of_frame``: image area covered by the projected pitch
    rectangle / frame area — broadcast play framings keep the pitch dominant.
    """

    decomposable: bool
    camera_height_m: Optional[float]
    camera_distance_m: Optional[float]  # horizontal distance from pitch centre
    roll_deg: Optional[float]
    focal_px: Optional[float]
    metres_per_px_centre: Optional[float]
    horizon_v_fraction: Optional[float]
    chirality_ok: bool
    pitch_fraction_of_frame: Optional[float]
    visible_pitch_area_m2: Optional[float]


def plausibility_metrics(
    h: Matrix,
    frame_width: float,
    frame_height: float,
    principal_point: Optional[Point2] = None,
) -> PlausibilityMetrics:
    pp = principal_point or (frame_width / 2.0, frame_height / 2.0)
    pose = decompose_homography(h, pp)

    # Ground-plane scale at the frame centre.
    mpp: Optional[float] = None
    c0 = unproject(h, (frame_width / 2.0, frame_height / 2.0))
    c1 = unproject(h, (frame_width / 2.0 + 1.0, frame_height / 2.0))
    c2 = unproject(h, (frame_width / 2.0, frame_height / 2.0 + 1.0))
    if c0 and c1 and c2:
        mpp = max(
            math.hypot(c1[0] - c0[0], c1[1] - c0[1]),
            math.hypot(c2[0] - c0[0], c2[1] - c0[1]),
        )

    # Horizon: image of the ground plane's line at infinity = third row of H^-1,
    # evaluated as a v-coordinate at the centre column.
    horizon_v_fraction: Optional[float] = None
    hinv = inv3(h)
    if hinv is not None:
        a, b, c = hinv[2][0], hinv[2][1], hinv[2][2]
        if abs(b) > 1e-12 * max(abs(a), abs(c), 1e-300):
            v_h = -(a * frame_width / 2.0 + c) / b
            horizon_v_fraction = v_h / frame_height

    # Chirality: sample a world grid over the pitch; a homogeneous-sign flip is
    # only damning when it happens at a point whose projection lands in or near
    # the frame — a noisy-but-usable zoomed-view H routinely flips the *far*
    # pitch corners out beyond the horizon, and that alone must not zero a frame.
    centre_w = mat_vec(h, [0.0, 0.0, 1.0])[2]
    chirality_ok = abs(centre_w) > 1e-12
    if chirality_ok:
        margin_w, margin_h = 2.0 * frame_width, 2.0 * frame_height
        for gi in range(6):
            for gj in range(5):
                wx = -PITCH_LENGTH_M / 2.0 + PITCH_LENGTH_M * gi / 5.0
                wy = -PITCH_WIDTH_M / 2.0 + PITCH_WIDTH_M * gj / 4.0
                u, v, w = mat_vec(h, [wx, wy, 1.0])
                if abs(w) < 1e-12:
                    continue
                if w * centre_w > 0.0:
                    continue
                # flipped point: only fatal if it claims to be near the frame
                if -margin_w <= u / w <= frame_width + margin_w and \
                        -margin_h <= v / w <= frame_height + margin_h:
                    chirality_ok = False
                    break
            if not chirality_ok:
                break

    # Pitch rectangle projected forward into the image (corners whose w flips
    # sit far beyond the horizon; clamp them out by clipping).
    pitch_fraction: Optional[float] = None
    visible_area: Optional[float] = None
    image_corners: List[Point2] = []
    if chirality_ok:
        for (wx, wy) in pitch_corners():
            u, v, w = mat_vec(h, [wx, wy, 1.0])
            if abs(w) < 1e-12 or w * centre_w <= 0.0:
                image_corners = []
                break
            image_corners.append((u / w, v / w))
    if len(image_corners) == 4:
        clipped = clip_polygon_to_rect(image_corners, 0.0, 0.0, frame_width, frame_height)
        pitch_fraction = polygon_area(clipped) / (frame_width * frame_height)
        # World-side area of the visible pitch: back-project the clipped polygon
        # (its vertices lie on the pitch, so they are safely below the horizon).
        ground = [unproject(hinv, p, inverse_given=True) if hinv else None for p in clipped]
        if ground and all(g is not None for g in ground):
            visible_area = polygon_area([g for g in ground if g is not None])

    return PlausibilityMetrics(
        decomposable=pose is not None,
        camera_height_m=pose.position[2] if pose else None,
        camera_distance_m=math.hypot(pose.position[0], pose.position[1]) if pose else None,
        roll_deg=pose.roll_deg if pose else None,
        focal_px=pose.focal_px if pose else None,
        metres_per_px_centre=mpp,
        horizon_v_fraction=horizon_v_fraction,
        chirality_ok=chirality_ok,
        pitch_fraction_of_frame=pitch_fraction,
        visible_pitch_area_m2=visible_area,
    )


# ---------------------------------------------------------------------------
# Sequence metrics
# ---------------------------------------------------------------------------

def landmark_motion_px(
    prev_obs: Dict[str, Point2], obs: Dict[str, Point2]
) -> Optional[float]:
    """Median pixel motion of landmarks detected in both frames.

    The camera-static test: detected pitch landmarks are world-fixed, so their
    pixel motion IS camera motion (plus detector noise). Needs only the exported
    keypoints, not pixels.
    """
    shared = sorted(set(prev_obs) & set(obs))
    if not shared:
        return None
    motions = sorted(
        math.hypot(obs[k][0] - prev_obs[k][0], obs[k][1] - prev_obs[k][1])
        for k in shared
    )
    return motions[len(motions) // 2]


def static_spans(
    landmark_series: Sequence[Dict[str, Point2]],
    max_median_motion_px: float = 1.5,
    min_span_frames: int = 5,
) -> List[Tuple[int, int]]:
    """Index spans [start, end) where the camera is (near-)static."""
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i in range(1, len(landmark_series)):
        m = landmark_motion_px(landmark_series[i - 1], landmark_series[i])
        is_static = m is not None and m <= max_median_motion_px
        if is_static and start is None:
            start = i - 1
        elif not is_static and start is not None:
            if i - start >= min_span_frames:
                spans.append((start, i))
            start = None
    if start is not None and len(landmark_series) - start >= min_span_frames:
        spans.append((start, len(landmark_series)))
    return spans


def world_jitter_m(
    h_prev: Matrix,
    h_curr: Matrix,
    frame_width: float,
    frame_height: float,
    grid_n: int = 3,
) -> Optional[float]:
    """Median world displacement of a fixed pixel grid between two homographies.

    On a static-camera span this is pure calibration error — the pitch does not
    move (design §6.2, temporal-stability signal). During a pan it conflates
    camera motion with error, so only aggregate it over ``static_spans``.
    """
    displacements: List[float] = []
    for i in range(grid_n):
        for j in range(grid_n):
            u = frame_width * (i + 1) / (grid_n + 1)
            v = frame_height * (j + 1) / (grid_n + 1)
            a = unproject(h_prev, (u, v))
            b = unproject(h_curr, (u, v))
            if a is None or b is None:
                continue
            displacements.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    if not displacements:
        return None
    displacements.sort()
    return displacements[len(displacements) // 2]


def teleport_count(
    world_track: Sequence[Tuple[float, float, float]],
    max_speed_ms: float = 12.0,
) -> int:
    """Number of physically impossible jumps in a (t, x, y) pitch-coordinate track.

    12 m/s is comfortably above elite sprint speed (~10.3 m/s); consecutive
    samples implying more are calibration or tracking failures, not running.
    """
    count = 0
    for i in range(1, len(world_track)):
        t0, x0, y0 = world_track[i - 1]
        t1, x1, y1 = world_track[i]
        dt = t1 - t0
        if dt <= 0.0:
            continue
        if math.hypot(x1 - x0, y1 - y0) / dt > max_speed_ms:
            count += 1
    return count


# ---------------------------------------------------------------------------
# The combined per-frame score
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationQualityConfig:
    """Named thresholds for every ramp in the combined score.

    Values are engineering priors pinned by the synthetic tests, not measured
    constants — same status as detector defaults pre-validation (STATUS.md
    caveat). Each pair reads (good, bad): score 1 at/beyond good, 0 at/beyond bad.
    """

    # residual component (needs correspondences)
    rms_m_good: float = 0.5
    rms_m_bad: float = 3.0
    # support component (needs correspondences). Scales calibrated against the
    # synthetic regimes (see test_calibration_quality): a full wide-midfield view
    # fits n~10 / spread~12 m / gap~5e-2; the near-goal tight view n~7 /
    # spread~9 m / gap~2e-3.
    n_points_good: int = 7
    n_points_bad: int = 3
    collinearity_good: float = 0.2
    collinearity_bad: float = 0.02
    spread_major_good_m: float = 10.0
    spread_major_bad_m: float = 3.0
    nullspace_gap_good: float = 2e-2
    nullspace_gap_bad: float = 3e-4
    # plausibility component (needs only H)
    camera_height_good_m: Tuple[float, float] = (8.0, 45.0)
    camera_height_bad_m: Tuple[float, float] = (2.0, 90.0)
    roll_good_deg: float = 3.0
    roll_bad_deg: float = 12.0
    mpp_good: Tuple[float, float] = (0.01, 0.35)
    mpp_bad: Tuple[float, float] = (0.002, 1.2)
    horizon_frac_good: float = 0.30
    horizon_frac_bad: float = 0.55
    pitch_frame_fraction_good: float = 0.35
    pitch_frame_fraction_bad: float = 0.05
    undecomposable_factor: float = 0.3
    chirality_break_factor: float = 0.1
    # temporal component (needs a reference pose, e.g. the smoother's prediction)
    pan_dev_good_deg: float = 0.5
    pan_dev_bad_deg: float = 4.0
    tilt_dev_good_deg: float = 0.3
    tilt_dev_bad_deg: float = 2.5
    focal_dev_good: float = 0.03  # |Δ ln f|
    focal_dev_bad: float = 0.25


@dataclass(frozen=True)
class FrameCalibrationQuality:
    """The per-frame gate Phase 4 consumes, with its evidence kept visible.

    ``quality`` = product of the component scores; a missing *input* (no
    keypoints, no temporal reference) contributes its MISSING_* factor exactly
    once. Components stay separate so a rejected frame can be *explained* —
    "support 0.1: five collinear points in one box" beats an opaque 0.07.
    """

    quality: float
    residual_score: Optional[float]
    support_score: Optional[float]
    plausibility_score: float
    temporal_score: Optional[float]
    reprojection: Optional[ReprojectionMetrics]
    support: Optional[SupportMetrics]
    plausibility: Optional[PlausibilityMetrics]


def _band_ramp(value: float, good: Tuple[float, float], bad: Tuple[float, float]) -> float:
    """1 inside the good band, 0 outside the bad band, linear in between."""
    low = ramp(value, good[0], bad[0])
    high = ramp(value, good[1], bad[1])
    return min(low, high)


def score_frame(
    h: Optional[Matrix],
    frame_width: float,
    frame_height: float,
    correspondences: Optional[Sequence[Correspondence]] = None,
    nullspace_gap: Optional[float] = None,
    reference_pose_vector: Optional[Sequence[float]] = None,
    config: CalibrationQualityConfig = CalibrationQualityConfig(),
) -> FrameCalibrationQuality:
    """Score one frame's calibration; ``h`` None (no estimate) scores 0.

    ``reference_pose_vector`` is an optional CameraPose.as_vector() to judge
    temporal consistency against — in practice the smoother's one-step
    prediction; leave None for the first frame or across a cut.
    """
    if h is None:
        return FrameCalibrationQuality(
            quality=0.0, residual_score=None, support_score=None,
            plausibility_score=0.0, temporal_score=None,
            reprojection=None, support=None, plausibility=None,
        )

    reproj = reprojection_metrics(h, correspondences) if correspondences else None
    residual_score: Optional[float] = None
    if reproj is not None:
        residual_score = ramp(reproj.rms_m, config.rms_m_good, config.rms_m_bad)

    supp = support_metrics([c[0] for c in correspondences]) if correspondences else None
    support_score: Optional[float] = None
    if supp is not None:
        support_score = (
            ramp(float(supp.n_points), float(config.n_points_good), float(config.n_points_bad))
            * ramp(supp.collinearity, config.collinearity_good, config.collinearity_bad)
            * ramp(supp.spread_major_m, config.spread_major_good_m, config.spread_major_bad_m)
        )
        if nullspace_gap is not None:
            # Compare magnitudes on a log scale: the gap spans decades.
            log_gap = math.log10(max(nullspace_gap, 1e-300))
            support_score *= ramp(
                log_gap,
                math.log10(config.nullspace_gap_good),
                math.log10(config.nullspace_gap_bad),
            )

    plaus = plausibility_metrics(h, frame_width, frame_height)
    plausibility_score = 1.0
    if not plaus.decomposable:
        plausibility_score *= config.undecomposable_factor
    else:
        plausibility_score *= _band_ramp(
            plaus.camera_height_m, config.camera_height_good_m, config.camera_height_bad_m
        )
        plausibility_score *= ramp(
            abs(plaus.roll_deg), config.roll_good_deg, config.roll_bad_deg
        )
    if plaus.metres_per_px_centre is not None:
        plausibility_score *= _band_ramp(
            plaus.metres_per_px_centre, config.mpp_good, config.mpp_bad
        )
    if plaus.horizon_v_fraction is not None:
        plausibility_score *= ramp(
            plaus.horizon_v_fraction, config.horizon_frac_good, config.horizon_frac_bad
        )
    if not plaus.chirality_ok:
        plausibility_score *= config.chirality_break_factor
    if plaus.pitch_fraction_of_frame is not None:
        plausibility_score *= ramp(
            plaus.pitch_fraction_of_frame,
            config.pitch_frame_fraction_good,
            config.pitch_frame_fraction_bad,
        )

    temporal_score: Optional[float] = None
    if reference_pose_vector is not None and plaus.decomposable:
        pose = decompose_homography(h, (frame_width / 2.0, frame_height / 2.0))
        if pose is not None:
            vec = pose.as_vector()
            temporal_score = (
                ramp(abs(vec[0] - reference_pose_vector[0]), config.pan_dev_good_deg, config.pan_dev_bad_deg)
                * ramp(abs(vec[1] - reference_pose_vector[1]), config.tilt_dev_good_deg, config.tilt_dev_bad_deg)
                * ramp(abs(vec[3] - reference_pose_vector[3]), config.focal_dev_good, config.focal_dev_bad)
            )

    quality = plausibility_score
    if residual_score is None and support_score is None:
        quality *= MISSING_EVIDENCE_FACTOR
    else:
        quality *= residual_score if residual_score is not None else 1.0
        quality *= support_score if support_score is not None else 1.0
    if temporal_score is None:
        quality *= MISSING_TEMPORAL_FACTOR
    else:
        quality *= temporal_score

    return FrameCalibrationQuality(
        quality=quality,
        residual_score=residual_score,
        support_score=support_score,
        plausibility_score=plausibility_score,
        temporal_score=temporal_score,
        reprojection=reproj,
        support=supp,
        plausibility=plaus,
    )
