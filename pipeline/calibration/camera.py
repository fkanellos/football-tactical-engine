"""Homography fitting, projection, and the camera-pose parameterization.

Conventions (calibration-design.md §5.1):
- World: canonical pitch, origin at the centre spot, x along the touchlines
  (|x| <= 52.5), y along the halfway line (|y| <= 34), z up. Ground plane z=0.
- Image: pixels, origin top-left, u right, v down.
- A frame's calibration is the 3x3 homography H mapping ground-plane world points
  (X, Y, 1) to image points (u, v, 1) up to scale — same direction as the
  world->image homography sn-gamestate fits; invert for pixel->pitch.

The camera-pose parameterization is the load-bearing design choice: temporal
smoothing operates on (pan, tilt, roll, focal length, camera position), never on
raw homography entries. Raw entries mix translation-scale (~1e2) and perspective
(~1e-3) magnitudes, so any isotropic noise model or gain on them is meaningless —
this is exactly the mistake in upstream nbjw_calib's unused SequentialCalib
(utils_calib_seq.py regularises 9-vectors of H entries against pixel residuals).
Pose parameters, by contrast, are physically bounded (a broadcast main camera
sits on a fixed mount: position ~constant; |roll| small; pan/tilt/zoom smooth),
so priors and gates on them are honest.

Decomposition assumes square pixels, zero skew, and a known principal point
(defaults to the frame centre) — the same assumptions nbjw_calib bakes into its
``cv2.calibrateCamera`` flags, so we are not assuming anything upstream doesn't.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .linalg import (
    Matrix,
    cross,
    det3,
    inv3,
    jacobi_eigh,
    mat_mul,
    mat_transpose,
    mat_vec,
    nearest_rotation,
    vec_norm,
)

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]

#: Correspondence: ((world_x_m, world_y_m), (image_u_px, image_v_px)).
Correspondence = Tuple[Point2, Point2]


# ---------------------------------------------------------------------------
# Camera pose
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraPose:
    """Decomposed broadcast-camera state; the smoothing state vector.

    Angles in degrees. ``pan`` is rotation about the world up-axis (0 = looking
    along +y from the -y stand side, positive toward +x, i.e. panning right);
    ``tilt`` positive looks down; ``roll`` rotates about the optical axis.
    ``position`` is the camera centre in world metres, z up (so position[2] > 0
    for any physical camera).
    """

    pan_deg: float
    tilt_deg: float
    roll_deg: float
    focal_px: float
    position: Point3
    principal_point: Point2

    def as_vector(self) -> List[float]:
        """Smoothing-state vector: (pan, tilt, roll, ln f, cx, cy, cz).

        Focal length enters as ln f so that a zoom gate/gain is relative (a 5%
        zoom change is the same innovation at f=1500 as at f=3000).
        """
        return [
            self.pan_deg,
            self.tilt_deg,
            self.roll_deg,
            math.log(self.focal_px),
            self.position[0],
            self.position[1],
            self.position[2],
        ]

    @staticmethod
    def from_vector(vec: Sequence[float], principal_point: Point2) -> "CameraPose":
        return CameraPose(
            pan_deg=vec[0],
            tilt_deg=vec[1],
            roll_deg=vec[2],
            focal_px=math.exp(vec[3]),
            position=(vec[4], vec[5], vec[6]),
            principal_point=principal_point,
        )


def _rotation_world_to_cam(pan_deg: float, tilt_deg: float, roll_deg: float) -> Matrix:
    """R such that X_cam = R (X_world - C); see CameraPose for angle meanings."""
    phi = math.radians(pan_deg)
    tau = math.radians(tilt_deg)
    gam = math.radians(roll_deg)
    st, ct = math.sin(tau), math.cos(tau)
    sg, cg = math.sin(gam), math.cos(gam)
    sp, cp = math.sin(phi), math.cos(phi)
    # A = Rz(roll) @ Rx(tilt) @ R_base, with R_base mapping world (x, y, z) to
    # camera (right, down, forward) for a camera on the -y side looking at +y.
    a = [
        [cg, sg * st, sg * ct],
        [sg, -cg * st, -cg * ct],
        [0.0, ct, -st],
    ]
    rz_phi = [
        [cp, -sp, 0.0],
        [sp, cp, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return mat_mul(a, rz_phi)


def compose_homography(pose: CameraPose) -> Matrix:
    """World ground plane -> image homography for a camera pose."""
    r = _rotation_world_to_cam(pose.pan_deg, pose.tilt_deg, pose.roll_deg)
    c = pose.position
    t = [-sum(r[i][k] * c[k] for k in range(3)) for i in range(3)]
    f, (pu, pv) = pose.focal_px, pose.principal_point
    k = [[f, 0.0, pu], [0.0, f, pv], [0.0, 0.0, 1.0]]
    rt = [[r[0][0], r[0][1], t[0]], [r[1][0], r[1][1], t[1]], [r[2][0], r[2][1], t[2]]]
    h = mat_mul(k, rt)
    return _normalize_h(h)


def decompose_homography(
    h: Matrix, principal_point: Point2, min_focal_px: float = 50.0,
    max_focal_px: float = 100_000.0,
) -> Optional[CameraPose]:
    """Recover (pan, tilt, roll, f, position) from a ground-plane homography.

    Returns None when the homography admits no physically valid decomposition
    (non-positive f^2 solution, focal outside [min, max], camera below the pitch
    plane, or a reflected frame) — an important *signal*, not just a failure:
    garbage per-frame fits routinely fail here, so decomposability itself feeds
    the plausibility score (quality.py §6.3).
    """
    pu, pv = principal_point
    h1 = [h[0][0], h[1][0], h[2][0]]
    h2 = [h[0][1], h[1][1], h[2][1]]
    h3 = [h[0][2], h[1][2], h[2][2]]

    # omega = K^-T K^-1 = (1/f^2) M + e3 e3^T, with both calibration constraints
    # (r1 . r2 = 0, |r1| = |r2|) linear in x = 1/f^2.
    def m_form(a: List[float], b: List[float]) -> float:
        return (
            a[0] * b[0]
            + a[1] * b[1]
            - pu * (a[0] * b[2] + a[2] * b[0])
            - pv * (a[1] * b[2] + a[2] * b[1])
            + (pu * pu + pv * pv) * a[2] * b[2]
        )

    a1, b1 = m_form(h1, h2), h1[2] * h2[2]
    a2, b2 = m_form(h1, h1) - m_form(h2, h2), h1[2] ** 2 - h2[2] ** 2
    denom = a1 * a1 + a2 * a2
    if denom < 1e-30:
        return None
    x = -(a1 * b1 + a2 * b2) / denom
    if x <= 0.0:
        return None
    f = 1.0 / math.sqrt(x)
    if not (min_focal_px <= f <= max_focal_px):
        return None

    k_inv = [[1.0 / f, 0.0, -pu / f], [0.0, 1.0 / f, -pv / f], [0.0, 0.0, 1.0]]
    b = mat_mul(k_inv, h)
    b1c = [b[0][0], b[1][0], b[2][0]]
    b2c = [b[0][1], b[1][1], b[2][1]]
    b3c = [b[0][2], b[1][2], b[2][2]]
    n1, n2 = vec_norm(b1c), vec_norm(b2c)
    if n1 < 1e-12 or n2 < 1e-12:
        return None

    for lam in (2.0 / (n1 + n2), -2.0 / (n1 + n2)):
        r1 = [x_ * lam for x_ in b1c]
        r2 = [x_ * lam for x_ in b2c]
        r3 = cross(r1, r2)
        r_cols = [[r1[i], r2[i], r3[i]] for i in range(3)]
        r = nearest_rotation(r_cols)
        if r is None:
            continue
        t = [x_ * lam for x_ in b3c]
        rt = mat_transpose(r)
        c = [-sum(rt[i][k] * t[k] for k in range(3)) for i in range(3)]
        if c[2] <= 0.0:
            continue
        tilt = math.degrees(math.asin(max(-1.0, min(1.0, -r[2][2]))))
        roll = math.degrees(math.atan2(r[0][2], -r[1][2]))
        pan = math.degrees(math.atan2(r[2][0], r[2][1]))
        return CameraPose(
            pan_deg=pan,
            tilt_deg=tilt,
            roll_deg=roll,
            focal_px=f,
            position=(c[0], c[1], c[2]),
            principal_point=(pu, pv),
        )
    return None


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project(h: Matrix, world_xy: Point2) -> Optional[Point2]:
    """Ground-plane world point -> pixel; None at/beyond the horizon (w ~ 0)."""
    u, v, w = mat_vec(h, [world_xy[0], world_xy[1], 1.0])
    scale = max(abs(u), abs(v), 1.0)
    if abs(w) < 1e-9 * scale or abs(w) < 1e-12:
        return None
    return (u / w, v / w)


def unproject(h_or_hinv: Matrix, pixel: Point2, inverse_given: bool = False) -> Optional[Point2]:
    """Pixel -> ground-plane world point (metres).

    Pass the world->image H (default) and it is inverted internally, or pass an
    already-inverted matrix with ``inverse_given=True``.
    """
    hinv = h_or_hinv if inverse_given else inv3(h_or_hinv)
    if hinv is None:
        return None
    x, y, w = mat_vec(hinv, [pixel[0], pixel[1], 1.0])
    scale = max(abs(x), abs(y), 1.0)
    if abs(w) < 1e-9 * scale or abs(w) < 1e-12:
        return None
    return (x / w, y / w)


def rescale_homography(h: Matrix, from_size: Point2, to_size: Point2) -> Matrix:
    """Re-express a world->image homography for a different pixel frame size.

    This is the exact repair for the pixel-frame mismatch found in the OFI run
    (calibration-design.md §3.4): sn-gamestate's nbjw_calib module denormalises
    keypoints to a hardcoded 1920x1080 frame, so on a 1280x720 video the fitted H
    lives in a virtual 1080p frame while bboxes are 720p pixels. H_720 = S @ H_1080
    with S = diag(1280/1920, 720/1080, 1).
    """
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    scaled = [
        [h[0][0] * sx, h[0][1] * sx, h[0][2] * sx],
        [h[1][0] * sy, h[1][1] * sy, h[1][2] * sy],
        [h[2][0], h[2][1], h[2][2]],
    ]
    return _normalize_h(scaled)


def _normalize_h(h: Matrix) -> Matrix:
    """Scale so H[2][2] = 1 when safely possible, else unit Frobenius norm."""
    norm = math.sqrt(sum(x * x for row in h for x in row))
    if norm == 0.0:
        return [row[:] for row in h]
    if abs(h[2][2]) > 1e-9 * norm:
        s = 1.0 / h[2][2]
    else:
        s = 1.0 / norm
    return [[x * s for x in row] for row in h]


# ---------------------------------------------------------------------------
# DLT homography fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HomographyFit:
    """A fitted world->image homography plus its conditioning diagnostics.

    ``nullspace_gap`` is the ratio (second-smallest / largest) eigenvalue of the
    normalised DLT normal matrix: how uniquely the correspondences pin down H.
    Near-degenerate landmark sets (few points, near-collinear, tight cluster)
    drive it toward 0 even when the residual is tiny — it is the "the fit looks
    perfect but extrapolates garbage" detector (calibration-design.md §6.2).
    ``rms_residual_px`` is the RMS image-space reprojection residual over the
    correspondences actually used.
    """

    h: Matrix
    nullspace_gap: float
    rms_residual_px: float
    n_points: int


def _normalizing_transform(points: Sequence[Point2]) -> Optional[Matrix]:
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    mean_dist = sum(math.hypot(p[0] - cx, p[1] - cy) for p in points) / n
    if mean_dist < 1e-9:
        return None
    s = math.sqrt(2.0) / mean_dist
    return [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]]


def fit_homography(correspondences: Sequence[Correspondence]) -> Optional[HomographyFit]:
    """Hartley-normalised DLT over >= 4 world<->image correspondences.

    Returns None when the system is outright unsolvable (fewer than 4 points, a
    zero-spread point set, or a singular result). Degradation short of that is
    *reported, not hidden*: check ``nullspace_gap`` before trusting the fit.
    """
    if len(correspondences) < 4:
        return None
    world = [c[0] for c in correspondences]
    image = [c[1] for c in correspondences]
    tw = _normalizing_transform(world)
    ti = _normalizing_transform(image)
    if tw is None or ti is None:
        return None

    def apply(t: Matrix, p: Point2) -> Point2:
        q = mat_vec(t, [p[0], p[1], 1.0])
        return (q[0], q[1])

    wn = [apply(tw, p) for p in world]
    im = [apply(ti, p) for p in image]

    # Normal matrix A^T A accumulated directly (rows of A per correspondence).
    ata = [[0.0] * 9 for _ in range(9)]
    for (wx, wy), (u, v) in zip(wn, im):
        rows = (
            [-wx, -wy, -1.0, 0.0, 0.0, 0.0, u * wx, u * wy, u],
            [0.0, 0.0, 0.0, -wx, -wy, -1.0, v * wx, v * wy, v],
        )
        for row in rows:
            for i in range(9):
                ri = row[i]
                if ri == 0.0:
                    continue
                for j in range(i, 9):
                    ata[i][j] += ri * row[j]
    for i in range(9):
        for j in range(i):
            ata[i][j] = ata[j][i]

    eigvals, eigvecs = jacobi_eigh(ata)
    largest = max(abs(eigvals[0]), abs(eigvals[-1]), 1e-300)
    nullspace_gap = max(0.0, eigvals[1]) / largest
    hv = eigvecs[0]
    hn = [[hv[0], hv[1], hv[2]], [hv[3], hv[4], hv[5]], [hv[6], hv[7], hv[8]]]

    ti_inv = inv3(ti)
    if ti_inv is None:
        return None
    h = mat_mul(ti_inv, mat_mul(hn, tw))
    if abs(det3(h)) < 1e-18:
        return None
    h = _normalize_h(h)

    sq_sum = 0.0
    for (wxy, uv) in correspondences:
        p = project(h, wxy)
        if p is None:
            sq_sum += 1e6  # a used correspondence at the horizon: effectively broken
        else:
            sq_sum += (p[0] - uv[0]) ** 2 + (p[1] - uv[1]) ** 2
    rms = math.sqrt(sq_sum / len(correspondences))
    return HomographyFit(
        h=h, nullspace_gap=nullspace_gap, rms_residual_px=rms,
        n_points=len(correspondences),
    )
