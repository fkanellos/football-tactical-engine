"""Per-frame ``ball_quality``: the gate the design docs already assume exists.

phase4-5-design.md §2.2 says "carry a validity flag"; §2.4 folds ball validity
into the per-frame data-quality score. This module generalises that boolean into
a [0, 1] signal with visible components, exactly as ``calibration_quality`` did
for the homography (calibration-design.md §6.3): a missing ball scores 0, a
detected ball scores the product of

- **confidence**: the detector's own score (ramped, not trusted raw);
- **kinematics**: implied speed vs what a ball can do — a teleporting
  detection is somebody else's white object;
- **temporal support**: detections around it — a lone blip inside a blackout
  is far likelier a false positive than the middle of a tracked flight;
- **consistency**: distance to the nearest visible player (missing player data
  discounts once, it does not zero — same MISSING_* philosophy as calibration);
- **candidate ambiguity**: frames where the detector offered several balls are
  discounted even when selection picked one.

Downstream contract (design §6): possession inference and every Tier 2 event
detector gate on this signal; pure-shape features (line height, compactness)
ignore it. Interpolated samples (motion.py) carry their own decayed confidence
and are additionally capped here, so bridged evidence can never outrank
observed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .consistency import PlayerFrame, nearest_player_distances
from .model import BallSample
from .plausibility import kinematic_flags

#: Discount when no player data exists to judge consistency against — absence
#: of evidence, not evidence of absence (mirrors calibration's MISSING_* idiom).
MISSING_PLAYERS_FACTOR = 0.9
#: Hard cap on the quality of an interpolated (never-observed) sample.
INTERPOLATED_CAP = 0.6


def ramp(value: float, good: float, bad: float) -> float:
    """Linear score ramp: 1 at/beyond ``good``, 0 at/beyond ``bad`` (either direction)."""
    if good == bad:
        return 1.0 if value == good else 0.0
    t = (value - bad) / (good - bad)
    return max(0.0, min(1.0, t))


@dataclass(frozen=True)
class BallQualityConfig:
    """Named thresholds for every ramp — engineering priors pinned by the
    synthetic tests, to be re-fit on the first real ball export (same epistemic
    status as every detector default in this repo; STATUS.md caveat applies)."""

    conf_good: float = 0.5
    conf_bad: float = 0.05
    speed_good_ms: float = 25.0
    speed_bad_ms: float = 45.0
    teleport_factor: float = 0.05
    support_window_s: float = 0.6
    support_good: float = 0.6
    support_bad: float = 0.0
    isolation_good_m: float = 15.0
    isolation_bad_m: float = 40.0
    multi_candidate_factor: float = 0.8


@dataclass(frozen=True)
class FrameBallQuality:
    """The per-frame gate, evidence kept visible so a low score is explainable."""

    quality: float
    detected: bool
    interpolated: bool
    confidence_score: Optional[float]
    kinematic_score: Optional[float]
    support_score: Optional[float]
    consistency_score: Optional[float]
    speed_ms: Optional[float]
    nearest_player_m: Optional[float]


_MISSING = FrameBallQuality(
    quality=0.0, detected=False, interpolated=False,
    confidence_score=None, kinematic_score=None, support_score=None,
    consistency_score=None, speed_ms=None, nearest_player_m=None,
)


def temporal_support(samples: Sequence[BallSample], index: int, window_s: float) -> float:
    """Fraction of *other* samples within ±window_s that carry a detection."""
    t0 = samples[index].t
    n_window = 0
    n_detected = 0
    for j, s in enumerate(samples):
        if j == index or abs(s.t - t0) > window_s:
            continue
        n_window += 1
        if s.detected:
            n_detected += 1
    if n_window == 0:
        return 1.0  # a one-sample stream has no neighbours to disagree with
    return n_detected / n_window


def score_ball_series(
    samples: Sequence[BallSample],
    player_frames: Optional[Sequence[Optional[PlayerFrame]]] = None,
    config: BallQualityConfig = BallQualityConfig(),
) -> List[FrameBallQuality]:
    """One ``FrameBallQuality`` per sample; missing frames score exactly 0."""
    flags_by_index = {f.index: f for f in kinematic_flags(samples)}
    distances: Sequence[Optional[float]]
    if player_frames is not None:
        distances = nearest_player_distances(samples, player_frames)
    else:
        distances = [None] * len(samples)

    out: List[FrameBallQuality] = []
    for i, s in enumerate(samples):
        if s.xy is None:
            out.append(_MISSING)
            continue

        confidence_score = ramp(s.confidence, config.conf_good, config.conf_bad)

        flag = flags_by_index.get(i)
        kinematic_score: Optional[float] = None
        speed = flag.speed_ms if flag else None
        if flag is not None and flag.speed_ms is not None:
            kinematic_score = ramp(flag.speed_ms, config.speed_good_ms, config.speed_bad_ms)
            if flag.teleport:
                kinematic_score *= config.teleport_factor

        support = temporal_support(samples, i, config.support_window_s)
        support_score = ramp(support, config.support_good, config.support_bad)

        nearest = distances[i]
        consistency_score: Optional[float] = None
        if nearest is not None:
            consistency_score = ramp(nearest, config.isolation_good_m, config.isolation_bad_m)

        quality = confidence_score * support_score
        quality *= kinematic_score if kinematic_score is not None else 1.0
        quality *= consistency_score if consistency_score is not None else MISSING_PLAYERS_FACTOR
        if s.n_candidates > 1:
            quality *= config.multi_candidate_factor
        if s.interpolated:
            quality = min(quality, INTERPOLATED_CAP)

        out.append(
            FrameBallQuality(
                quality=quality,
                detected=True,
                interpolated=s.interpolated,
                confidence_score=confidence_score,
                kinematic_score=kinematic_score,
                support_score=support_score,
                consistency_score=consistency_score,
                speed_ms=speed,
                nearest_player_m=nearest,
            )
        )
    return out
