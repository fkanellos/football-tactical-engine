"""EventDetector interface, staged registry, and shared evidence helpers.

The idiom mirrors ``patterns.detectors.base`` (soft thresholds, dataclass configs,
registry, synthetic-fixture tests) with ONE deliberate deviation (design doc §7.2):
event detectors run in STAGES and receive prior stages' events as ``context``,
because the evidence really is layered — shot outcomes need restarts (kickoff =>
goal), passes need shots (a launch is one or the other), set-pieces/stoppages need
restarts (skip explained dead spells). Any detector must still run correctly (more
conservatively) with an empty context.

Contract for implementations (otherwise identical to pattern detectors):
- Pure and deterministic: (KinematicSeries, context) -> events. No I/O, no state.
- Missing data lowers confidence or skips candidates; it never raises.
- Tier 1 restart detection is deliberately NOT segment-local (design doc §4.3) —
  restarts bridge broadcast cuts; every other detector stays within segments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type

# Shared soft-threshold idiom with the pattern layer — same function, not a copy
# (the §8 refactor promotes these into a shared module).
from ..patterns.detectors.base import clamp01, soft_threshold  # noqa: F401 (re-exported)
from .kinematics import KinematicSeries
from .model import EventType, MatchEvent


def combine_evidence(factors: Sequence[float], quality: float = 1.0) -> float:
    """Standard event confidence: geometric mean of the evidence factors x quality.

    The geometric mean keeps the result in the per-factor margin scale (a product
    would punish merely-having-many-factors), matching what ``episode_confidence``
    does for pattern episodes via its 1/k exponent.
    """
    if not factors:
        return 0.0
    product = 1.0
    for f in factors:
        product *= clamp01(f)
    return clamp01((product ** (1.0 / len(factors))) * clamp01(quality))


class EventDetector(ABC):
    """Strategy interface: one event family, one class.

    ``stage`` orders detectors in the runner (lower runs first and feeds
    ``context`` of the later stages); ``event_types`` declares what the detector
    may emit (runner sanity-checks it).
    """

    detector_id: ClassVar[str]
    display_name: ClassVar[str]
    event_types: ClassVar[Tuple[EventType, ...]]
    stage: ClassVar[int]

    @abstractmethod
    def detect(
        self, kin: KinematicSeries, context: Sequence[MatchEvent] = ()
    ) -> List[MatchEvent]:
        """Detect all events of this family across the match."""
        raise NotImplementedError


class EventDetectorRegistry:
    """Registry so the runner discovers detectors without hardcoded imports."""

    _detectors: ClassVar[Dict[str, Type[EventDetector]]] = {}

    @classmethod
    def register(cls, detector_cls: Type[EventDetector]) -> Type[EventDetector]:
        cls._detectors[detector_cls.detector_id] = detector_cls
        return detector_cls

    @classmethod
    def detector_ids(cls) -> List[str]:
        return sorted(cls._detectors)

    @classmethod
    def build_all(cls, configs: Optional[Dict[str, Any]] = None) -> List[EventDetector]:
        """Instantiate every registered detector in STAGE order (ties by id)."""
        configs = configs or {}
        ordered = sorted(cls._detectors.items(), key=lambda kv: (kv[1].stage, kv[0]))
        return [
            detector_cls(configs[did]) if did in configs else detector_cls()
            for did, detector_cls in ordered
        ]
