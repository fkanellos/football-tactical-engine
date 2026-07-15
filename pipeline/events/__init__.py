"""Event inference: discrete match events from tracking data alone.

Design: /docs/event-inference-design.md. This layer sits BELOW Phase 4 — it
consumes raw tracking (never FeatureSeries: possession inference becomes a
consumer of this stream, §8) and produces match_events.json, with an explicit
confidence hierarchy:

    tier 1  boundary restarts (throw-in, corner, goal kick, kickoff) — reliable
    tier 2  passes (+subtypes), shots (+outcome ladder), set-piece organization
    tier 3  coarse stoppage signal — segmentation only, no cause claims

    model.py       MatchEvent / EventType / MatchEventStream / match-events/v1
    kinematics.py  resampled+smoothed ball & player tracks, launch detection
    base.py        EventDetector ABC, staged registry, evidence helpers
    restarts.py    tier 1        shots.py, passes.py, setpieces.py  tier 2
    stoppages.py   tier 3        runner.py  staged orchestration
"""

from .base import EventDetector, EventDetectorRegistry, combine_evidence
from .kinematics import (
    BallLaunch,
    KinematicExtractor,
    KinematicSegment,
    KinematicSeries,
    KinematicsConfig,
    PlayerTrack,
    detect_launches,
)
from .model import EVENT_SCHEMA_VERSION, EVENT_TIERS, EventType, MatchEvent, MatchEventStream
from .passes import PassConfig, PassDetector
from .restarts import RestartConfig, RestartDetector
from .runner import EventInferencePipeline, write_events_json
from .setpieces import SetPieceConfig, SetPieceDetector
from .shots import ShotConfig, ShotDetector
from .stoppages import StoppageConfig, StoppageDetector

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EVENT_TIERS",
    "BallLaunch",
    "EventDetector",
    "EventDetectorRegistry",
    "EventInferencePipeline",
    "EventType",
    "KinematicExtractor",
    "KinematicSegment",
    "KinematicSeries",
    "KinematicsConfig",
    "MatchEvent",
    "MatchEventStream",
    "PassConfig",
    "PassDetector",
    "PlayerTrack",
    "RestartConfig",
    "RestartDetector",
    "SetPieceConfig",
    "SetPieceDetector",
    "ShotConfig",
    "ShotDetector",
    "StoppageConfig",
    "StoppageDetector",
    "combine_evidence",
    "detect_launches",
    "write_events_json",
]
