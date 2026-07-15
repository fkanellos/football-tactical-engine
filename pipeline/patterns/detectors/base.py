"""Detector interface, event model, registry, and the shared detection idiom.

Every detector follows the same recipe (design doc §3.1):

1. per-frame candidate score in [0, 1] from soft thresholds (``soft_threshold``);
2. episode extraction with hysteresis (``episodes_from_scores``);
3. per-episode ``confidence`` (did it happen?) and ``intensity`` (how strongly?)
   via ``episode_confidence`` + detector-specific intensity normalisation.

Contract for implementations:
- Pure and deterministic: (FeatureSeries, config) -> events. No I/O, no state
  across calls. This is what makes detectors unit-testable with synthetic series.
- Segment-aware: never emit an episode spanning a broadcast-cut gap (iterate
  ``series.segments()``).
- Missing data lowers confidence or skips frames; it never raises. Convention:
  when an *optional* input feature is None on a frame, its factor in the score
  product is ``MISSING_FEATURE_FACTOR`` (a mild discount), and the frame counts
  as incomplete for the completeness penalty in ``episode_confidence``.
- Events for the same (pattern, team) must not overlap; different patterns MAY
  overlap freely (a high press flowing into a counter-attack is two events).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type

from ..features import FeatureSeries
from ..tracking import MatchMeta, TeamSide

#: Score factor used in place of a soft-threshold term when the underlying
#: feature is missing on a frame (e.g. back line off-camera): mild discount so
#: strong remaining evidence can still form an episode, per "discount, don't
#: drop" (design doc §3.2). The completeness penalty in ``episode_confidence``
#: then caps how confident such episodes can get.
MISSING_FEATURE_FACTOR = 0.6


@dataclass(frozen=True)
class PatternEvent:
    """One detected occurrence of a tactical motif.

    ``confidence``: probability-like belief the motif actually occurred, in [0, 1] —
    rule margin x data quality.
    ``intensity``: normalised strength of the motif in [0, 1] (a 6-man press beats a
    3-man press at equal confidence). The two are deliberately independent axes.

    ``metadata`` keys are detector-specific but drawn from a shared vocabulary where
    possible: "side" (attacking-perspective left/right), "zone", "trigger",
    "involved_track_ids", "turnover_location". The UI uses ``start_s``/``end_s`` to
    deep-link video clips, so spans should be generous enough to give a viewer 1-2 s
    of lead-in context.
    """

    pattern_id: str
    team: TeamSide  # the team EXHIBITING the pattern
    start_s: float
    end_s: float
    confidence: float
    intensity: float
    period: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class PatternDetector(ABC):
    """Strategy interface: one tactical motif, one class.

    Class attributes each subclass must define:
    - ``pattern_id``: stable snake_case identifier (also the rules-table key, so it
      is part of the Phase 4/5 contract — never rename casually).
    - ``display_name``: human-readable name for UI/logs.
    - ``required_features``: dotted feature paths this detector reads (see
      features.py module docstring). Prefix ``"team."`` means "both teams". The
      runner validates availability and skips-with-warning instead of crashing.
    """

    pattern_id: ClassVar[str]
    display_name: ClassVar[str]
    required_features: ClassVar[Tuple[str, ...]]

    @abstractmethod
    def detect(self, series: FeatureSeries, meta: MatchMeta) -> List[PatternEvent]:
        """Detect all occurrences of this motif for BOTH teams across the match."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Shared helpers — the concrete implementations of the detection idiom.
# --------------------------------------------------------------------------

def soft_threshold(
    value: Optional[float], threshold: float, softness: float, above: bool = True
) -> float:
    """Logistic score in [0, 1] for how far ``value`` sits past ``threshold``.

    ``softness`` is the transition width: |value - threshold| = softness gives
    ~0.73, exactly at threshold gives 0.5. ``above=False`` inverts the direction
    (score high when below threshold). ``None`` values return
    ``MISSING_FEATURE_FACTOR`` — callers should also record frame incompleteness.
    Soft thresholds are what make event confidences meaningful and are the seam
    where learned weights later replace hand-set ones (design doc §3.8).
    """
    if value is None:
        return MISSING_FEATURE_FACTOR
    if softness <= 0:
        raise ValueError("softness must be positive")
    z = (value - threshold) / softness
    if not above:
        z = -z
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def episodes_from_scores(
    scores: Sequence[Optional[float]],
    times: Sequence[float],
    enter_threshold: float,
    exit_threshold: float,
    min_duration_s: float,
    merge_gap_s: float,
) -> List[Tuple[int, int]]:
    """Turn a per-frame score series into episode index spans with hysteresis.

    Enter an episode when score >= ``enter_threshold``; leave when it drops below
    ``exit_threshold`` (< enter, so episodes don't flicker at the boundary). Then
    merge episodes separated by gaps < ``merge_gap_s`` and drop episodes shorter
    than ``min_duration_s``. ``None`` scores (missing data) end episodes but do
    not reset the merge window. Returns [start_index, end_index) pairs.
    """
    if len(scores) != len(times):
        raise ValueError("scores and times must be the same length")
    raw: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, s in enumerate(scores):
        if s is None or s < exit_threshold:
            if start is not None:
                raw.append((start, i))
                start = None
        elif start is None and s >= enter_threshold:
            start = i
    if start is not None:
        raw.append((start, len(scores)))

    merged: List[Tuple[int, int]] = []
    for span in raw:
        if merged and times[span[0]] - times[merged[-1][1] - 1] < merge_gap_s:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)

    return [
        (a, b) for a, b in merged if times[b - 1] - times[a] >= min_duration_s - 1e-9
    ]


def episode_confidence(
    scores: Sequence[float],
    qualities: Sequence[float],
    complete: Sequence[bool],
    n_components: int,
) -> float:
    """Standard confidence for one episode (design doc §3.1).

    (mean frame score) ** (1/k) recovers the geometric-mean per-component margin
    (each frame score is a product of k soft-threshold factors); multiplied by
    mean data quality and by a completeness penalty (0.4..1.0) so episodes built
    on missing-feature discounts can never look certain.
    """
    if not scores:
        return 0.0
    margin = (sum(scores) / len(scores)) ** (1.0 / max(1, n_components))
    quality = sum(qualities) / len(qualities) if qualities else 0.0
    completeness = (sum(1 for c in complete if c) / len(complete)) if complete else 0.0
    return max(0.0, min(1.0, margin * quality * (0.4 + 0.6 * completeness)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class DetectorRegistry:
    """Registry so the runner discovers detectors without hardcoded imports."""

    _detectors: ClassVar[Dict[str, Type[PatternDetector]]] = {}

    @classmethod
    def register(cls, detector_cls: Type[PatternDetector]) -> Type[PatternDetector]:
        """Class decorator: ``@DetectorRegistry.register`` above each detector."""
        cls._detectors[detector_cls.pattern_id] = detector_cls
        return detector_cls

    @classmethod
    def pattern_ids(cls) -> List[str]:
        return sorted(cls._detectors)

    @classmethod
    def build_all(cls, configs: Optional[Dict[str, Any]] = None) -> List[PatternDetector]:
        """Instantiate every registered detector, with optional per-pattern config
        overrides keyed by ``pattern_id``."""
        configs = configs or {}
        return [
            detector_cls(configs[pid]) if pid in configs else detector_cls()
            for pid, detector_cls in sorted(cls._detectors.items())
        ]
