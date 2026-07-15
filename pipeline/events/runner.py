"""Orchestration: tracking -> kinematics -> staged detectors -> MatchEventStream.

Stages run in dependency order (base.py: restarts -> shots -> passes ->
set-pieces -> stoppages); each stage sees the accumulated events of the earlier
ones as ``context``. Detector failures are contained, same policy as the pattern
runner: one broken detector logs and is skipped, the rest of the match survives.

``MatchEventStream.to_json_dict()`` is the persisted match_events.json contract
(schema ``match-events/v1``, design doc §7.3).
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from ..patterns.tracking import MatchTracking
from .base import EventDetector, EventDetectorRegistry
from .kinematics import KinematicExtractor
from .model import MatchEvent, MatchEventStream

logger = logging.getLogger(__name__)


class EventInferencePipeline:
    """End-to-end event inference for one match."""

    def __init__(
        self,
        detectors: Optional[List[EventDetector]] = None,
        extractor: Optional[KinematicExtractor] = None,
    ):
        # registry order IS stage order; an explicit list must keep it too
        self.detectors = detectors if detectors is not None else EventDetectorRegistry.build_all()
        self.extractor = extractor or KinematicExtractor()

    def run(self, match: MatchTracking) -> MatchEventStream:
        kin = self.extractor.extract(match)
        events: List[MatchEvent] = []
        for detector in self.detectors:
            try:
                found = detector.detect(kin, context=tuple(events))
            except Exception:
                logger.exception("event detector %s failed; skipping", detector.detector_id)
                continue
            stray = [e for e in found if e.event_type not in detector.event_types]
            if stray:
                logger.warning(
                    "detector %s emitted undeclared event types %s; dropping them",
                    detector.detector_id, sorted({e.event_type.value for e in stray}),
                )
                found = [e for e in found if e.event_type in detector.event_types]
            events.extend(found)
        return MatchEventStream(match.meta.match_id, events)


def write_events_json(stream: MatchEventStream, path: str) -> None:
    """Persist a stream as match_events.json (schema match-events/v1)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stream.to_json_dict(), f, indent=2)
