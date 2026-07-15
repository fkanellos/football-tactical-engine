"""Live/streaming twin of the batch pattern pipeline.

Design: /docs/live-architecture-design.md. Guiding rule is the PARITY PRINCIPLE:
the streaming path computes the same numbers as the batch path, delayed —
never different numbers faster — so all offline tuning transfers to live.

    causal.py      delayed-causal smoothing/derivatives (bit-parity with batch)
    possession.py  causal possession machine; turnovers confirm with backdated starts
    features.py    StreamingFeatureExtractor (skeleton) + structural latency table
    detector.py    episode lifecycle machines (PROVISIONAL/CONFIRMED/CLOSED),
                   ConfidencePrior protocol, StreamingDetector base
    high_press.py  worked example wrapping the batch high-press scorer live
    events.py      LivePatternUpdate lifecycle events + WebSocket wire schema

Scouting priors (pipeline.scouting) plug into detector machines via the
ConfidencePrior protocol — see /docs/opponent-scouting-design.md §3.
"""

from .causal import CausalChain, StreamingDerivative, StreamingMovingAverage
from .detector import (
    AnchoredWindowConfig,
    AnchoredWindowMachine,
    ConfidencePrior,
    EpisodeStateMachine,
    StreamingDetector,
    StreamingEpisodeConfig,
)
from .events import LiveEventKind, LivePatternUpdate, StreamStatus, WS_SCHEMA_VERSION
from .features import FeatureLatency, StreamingFeatureExtractor, feature_latencies
from .high_press import StreamingHighPressDetector
from .possession import StreamingPossessionMachine, TurnoverConfirmation

__all__ = [
    "AnchoredWindowConfig",
    "AnchoredWindowMachine",
    "CausalChain",
    "ConfidencePrior",
    "EpisodeStateMachine",
    "FeatureLatency",
    "LiveEventKind",
    "LivePatternUpdate",
    "StreamStatus",
    "StreamingDerivative",
    "StreamingDetector",
    "StreamingEpisodeConfig",
    "StreamingFeatureExtractor",
    "StreamingHighPressDetector",
    "StreamingMovingAverage",
    "StreamingPossessionMachine",
    "TurnoverConfirmation",
    "WS_SCHEMA_VERSION",
    "feature_latencies",
]
