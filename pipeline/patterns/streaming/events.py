"""Live event model: the pattern-instance lifecycle and its WebSocket wire form.

Live detection cannot wait for an episode to end before saying anything, so a
live pattern *instance* goes through an explicit lifecycle (live-architecture
design doc §4) instead of appearing fully formed like a batch ``PatternEvent``:

    (internal candidate) -> PROVISIONAL -> [UPDATE ...] -> CONFIRMED
                                 |                             |
                                 v                             v
                             RETRACTED                      CLOSED

- PROVISIONAL: enough sustained evidence to be worth showing, not enough to be
  sure. UIs render these visually distinct (e.g. pulsing/outline) and MUST be
  prepared to remove them on RETRACTED.
- CONFIRMED: the instance has met the same duration/score bar a batch episode
  needs; it will not be retracted, only closed.
- CLOSED: the pattern ended; the update carries the final span and stats and is
  the live twin of the batch ``PatternEvent``.
- RETRACTED: a provisional that fizzled. ``reason`` says why ("score_faded",
  "broadcast_cut", "stream_end").

Sub-provisional candidates that fizzle are never emitted at all — the two-tier
false-start defence (candidates are free, provisionals are accountable).

The wire schema (``to_ws_dict``) is versioned and is the pipeline<->UI contract
for the Kotlin Compose client, sibling to the persisted events.json/profile.json
contract of the batch pipeline (phase 4-5 design doc §4.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..tracking import TeamSide

#: Bump when the wire shape changes incompatibly; the UI checks it on connect.
WS_SCHEMA_VERSION = 1


class LiveEventKind(Enum):
    PROVISIONAL = "provisional"
    UPDATE = "update"
    CONFIRMED = "confirmed"
    RETRACTED = "retracted"
    CLOSED = "closed"


@dataclass(frozen=True)
class LivePatternUpdate:
    """One lifecycle emission for one live pattern instance.

    ``instance_id`` is stable across the instance's lifetime — the UI keys on it
    to upgrade a provisional card in place rather than stacking duplicates.

    Two confidences travel together (opponent-scouting design doc §3):
    ``raw_confidence`` is what the detector saw in THIS match's frames (batch
    ``episode_confidence`` over the frames so far, capped while provisional);
    ``confidence`` additionally folds in the opponent scouting prior when one is
    active. Logging both is the safeguard that lets a post-match audit measure
    how much the prior shifted live output. Without a prior they are equal.

    ``maturity`` in [0, 1] is how much of the confirmation bar the instance has
    cleared (elapsed active span / min duration) — the honest "how partial is
    this partial episode" axis, deliberately separate from confidence.
    """

    kind: LiveEventKind
    instance_id: str
    pattern_id: str
    team: TeamSide
    period: int
    start_s: float
    last_s: float
    confidence: float
    raw_confidence: float
    maturity: float
    intensity: Optional[float] = None
    end_s: Optional[float] = None  # set on CLOSED
    reason: Optional[str] = None   # set on RETRACTED / non-natural CLOSED
    metadata: Dict[str, Any] = field(default_factory=dict)
    prior: Optional[Dict[str, Any]] = None  # PriorAdjuster.describe() when active

    def to_ws_dict(self, seq: int) -> Dict[str, Any]:
        """Wire form consumed by the Kotlin UI over WebSocket."""
        msg: Dict[str, Any] = {
            "v": WS_SCHEMA_VERSION,
            "seq": seq,
            "type": "pattern",
            "event": self.kind.value,
            "instance": {
                "id": self.instance_id,
                "pattern": self.pattern_id,
                "team": self.team.value,
                "period": self.period,
                "start_s": round(self.start_s, 2),
                "last_s": round(self.last_s, 2),
                "confidence": round(self.confidence, 3),
                "raw_confidence": round(self.raw_confidence, 3),
                "maturity": round(self.maturity, 3),
                "metadata": dict(self.metadata),
            },
        }
        inst = msg["instance"]
        if self.intensity is not None:
            inst["intensity"] = round(self.intensity, 3)
        if self.end_s is not None:
            inst["end_s"] = round(self.end_s, 2)
        if self.reason is not None:
            msg["reason"] = self.reason
        if self.prior is not None:
            inst["prior"] = dict(self.prior)
        return msg


@dataclass(frozen=True)
class StreamStatus:
    """Periodic stream-health message (segment state, data quality).

    ``state``: "live" (wide shot, tracking healthy), "broken" (cut/replay/
    close-up — pattern state was flushed), "recovering" (wide shot back,
    features warming up). The UI greys out live panels outside "live".
    """

    state: str
    segment_id: int
    match_clock_s: Optional[float]
    quality: float
    open_instances: int

    def to_ws_dict(self, seq: int) -> Dict[str, Any]:
        return {
            "v": WS_SCHEMA_VERSION,
            "seq": seq,
            "type": "stream_status",
            "state": self.state,
            "segment_id": self.segment_id,
            "match_clock_s": self.match_clock_s,
            "quality": round(self.quality, 3),
            "open_instances": self.open_instances,
        }
