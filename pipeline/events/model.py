"""Match-event data model and the match-events/v1 JSON schema.

A ``MatchEvent`` is one inferred discrete game event (throw-in, pass, shot, ...).
Every event carries its confidence TIER (design doc §2) so consumers can filter by
honesty class without a lookup table: tier 1 = boundary-geometry restarts
(reliable), tier 2 = kinematic events (feasible with real uncertainty), tier 3 =
proxies (weak, explicitly conflating).

Not to be confused with ``pipeline.patterns.streaming.events`` — that module is the
live *wire* schema for pattern lifecycle messages; this one is *match* events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..patterns.tracking import TeamSide

#: Version tag written into match_events.json (pipeline<->consumer contract).
EVENT_SCHEMA_VERSION = "match-events/v1"


class EventType(Enum):
    # tier 1 — boundary-crossing restarts
    THROW_IN = "throw_in"
    CORNER = "corner"
    GOAL_KICK = "goal_kick"
    KICKOFF = "kickoff"
    # tier 2 — kinematic events
    PASS = "pass"
    SHOT = "shot"
    SET_PIECE_SETUP = "set_piece_setup"
    # tier 3 — coarse signals
    STOPPAGE = "stoppage"


#: Confidence tier per event type (design doc §2). Rides on every serialized event.
EVENT_TIERS: Dict[EventType, int] = {
    EventType.THROW_IN: 1,
    EventType.CORNER: 1,
    EventType.GOAL_KICK: 1,
    EventType.KICKOFF: 1,
    EventType.PASS: 2,
    EventType.SHOT: 2,
    EventType.SET_PIECE_SETUP: 2,
    EventType.STOPPAGE: 3,
}


@dataclass(frozen=True)
class MatchEvent:
    """One inferred match event.

    ``team`` is the event's protagonist (thrower, passer, shooter, restart taker);
    ``None`` when attribution failed or is not meaningful (stoppages).
    ``x``/``y`` are the event's defining location in ABSOLUTE pitch coordinates
    (crossing point, launch point, stoppage point); ``None`` when unknown.
    ``confidence`` means "this event occurred" — interpretation refinements (shot
    outcome, pass subtype) carry their own confidences inside ``metadata``.
    """

    event_type: EventType
    team: Optional[TeamSide]
    start_s: float
    end_s: float
    confidence: float
    period: int
    x: Optional[float] = None
    y: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tier(self) -> int:
        return EVENT_TIERS[self.event_type]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, t0: float, t1: float) -> bool:
        return self.start_s < t1 and self.end_s > t0

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "tier": self.tier,
            "team": self.team.value if self.team is not None else None,
            "period": self.period,
            "start": round(self.start_s, 2),
            "end": round(self.end_s, 2),
            "confidence": round(self.confidence, 3),
            "x": None if self.x is None else round(self.x, 2),
            "y": None if self.y is None else round(self.y, 2),
            "metadata": dict(self.metadata),
        }


class MatchEventStream:
    """Time-ordered inferred events for one match + the JSON contract."""

    def __init__(self, match_id: str, events: Iterable[MatchEvent]):
        self.match_id = match_id
        self.events: List[MatchEvent] = sorted(events, key=lambda e: (e.start_s, e.end_s))

    def __len__(self) -> int:
        return len(self.events)

    def of_type(self, *types: EventType) -> List[MatchEvent]:
        wanted = set(types)
        return [e for e in self.events if e.event_type in wanted]

    def in_window(self, t0: float, t1: float) -> List[MatchEvent]:
        return [e for e in self.events if e.overlaps(t0, t1)]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": EVENT_SCHEMA_VERSION,
            "match_id": self.match_id,
            "events": [e.to_json_dict() for e in self.events],
        }
