"""Minimal pure-Python linear algebra for the calibration layer.

The whole repository is deliberately stdlib-only (no numpy), and every problem in
this package is tiny (3x3 matrices, 9x9 symmetric eigenproblems), so hand-rolled
routines are both fast enough and dependency-free. Everything here is plain
``list[list[float]]`` / ``list[float]``; no classes, no cleverness.

Numerical choices, briefly:
- The DLT nullspace (calibration-design.md §5.2) needs the eigenvector of the
  smallest eigenvalue of A^T A (9x9 symmetric PSD). Cyclic Jacobi is exact enough
  (machine precision after ~8 sweeps) and trivially robust at this size.
- Rotation re-orthogonalisation uses Higham's Newton iteration for the polar
  factor, which converges quadratically for any nonsingular input.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

Matrix = List[List[float]]
Vector = List[float]


# ---------------------------------------------------------------------------
# Basic matrix / vector operations
# ---------------------------------------------------------------------------

def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Matrix product a @ b for rectangular lists-of-rows."""
    rows, inner, cols = len(a), len(b), len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def mat_vec(a: Matrix, v: Vector) -> Vector:
    return [sum(a[i][k] * v[k] for k in range(len(v))) for i in range(len(a))]


def mat_transpose(a: Matrix) -> Matrix:
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def mat_scale(a: Matrix, s: float) -> Matrix:
    return [[x * s for x in row] for row in a]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def det3(a: Matrix) -> float:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def inv3(a: Matrix) -> Optional[Matrix]:
    """Inverse of a 3x3 via the adjugate; None when (near-)singular."""
    d = det3(a)
    scale = max(abs(x) for row in a for x in row)
    if scale == 0.0 or abs(d) < 1e-14 * scale**3:
        return None
    inv_d = 1.0 / d
    return [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_d,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_d,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_d,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_d,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_d,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_d,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_d,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_d,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_d,
        ],
    ]


def vec_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def vec_sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def vec_scale(v: Vector, s: float) -> Vector:
    return [x * s for x in v]


def cross(a: Vector, b: Vector) -> Vector:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition (cyclic Jacobi)
# ---------------------------------------------------------------------------

def jacobi_eigh(a: Matrix, sweeps: int = 12) -> Tuple[Vector, Matrix]:
    """Eigenvalues/eigenvectors of a symmetric matrix, ascending eigenvalue order.

    Returns ``(eigenvalues, eigenvectors)`` where ``eigenvectors[k]`` is the unit
    eigenvector (as a row) for ``eigenvalues[k]``. Plain cyclic Jacobi: adequate
    and numerically boring for the n<=9 matrices this package produces.
    """
    n = len(a)
    m = [row[:] for row in a]
    v = identity(n)

    for _ in range(sweeps):
        off = math.sqrt(sum(m[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        scale = math.sqrt(sum(m[i][j] ** 2 for i in range(n) for j in range(n)))
        if scale == 0.0 or off <= 1e-15 * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) <= 1e-300:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    eigvals = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigvals[i])
    values = [eigvals[i] for i in order]
    vectors = [[v[r][i] for r in range(n)] for i in order]
    return values, vectors


# ---------------------------------------------------------------------------
# Nearest rotation (polar decomposition, Higham iteration)
# ---------------------------------------------------------------------------

def nearest_rotation(a: Matrix, iterations: int = 30) -> Optional[Matrix]:
    """Orthogonal polar factor of a 3x3 matrix (the nearest rotation in Frobenius norm).

    Newton iteration X <- (X + X^-T)/2; None if the input is singular. A negative
    determinant input converges to a reflection, which is rejected (None) — a
    correspondence set that yields a reflected frame is geometrically invalid.
    """
    x = [row[:] for row in a]
    for _ in range(iterations):
        xin = inv3(x)
        if xin is None:
            return None
        xit = mat_transpose(xin)
        nxt = [[(x[i][j] + xit[i][j]) * 0.5 for j in range(3)] for i in range(3)]
        delta = max(abs(nxt[i][j] - x[i][j]) for i in range(3) for j in range(3))
        x = nxt
        if delta < 1e-13:
            break
    if det3(x) < 0.0:
        return None
    return x
