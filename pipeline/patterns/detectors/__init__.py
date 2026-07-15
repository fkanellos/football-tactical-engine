"""Tactical motif detectors (Strategy pattern).

One module per motif; each registers itself with the shared ``DetectorRegistry`` so
the runner discovers detectors without a hardcoded list. Adding a motif = adding a
module here, nothing else.
"""

from .base import DetectorRegistry, PatternDetector, PatternEvent
from .counter_attack import CounterAttackDetector
from .flank_overload import FlankOverloadDetector
from .high_press import HighPressDetector
from .low_block import LowBlockDetector
from .offside_trap import OffsideTrapDetector

__all__ = [
    "DetectorRegistry",
    "PatternDetector",
    "PatternEvent",
    "CounterAttackDetector",
    "FlankOverloadDetector",
    "HighPressDetector",
    "LowBlockDetector",
    "OffsideTrapDetector",
]
