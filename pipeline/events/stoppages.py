"""Tier 3: the coarse stoppage signal. Design doc §6.

Scoped narrowly on purpose (review clarification): this channel answers exactly
one question — is play stopped right now? — so the event stream can be segmented
into in-play and dead-ball phases. It classifies NO cause and types NO restart
(Tier 1 and setpieces.py own typing). Coverage is deliberately uniform: stoppage
events are emitted for every dead phase, including those a higher-tier event
already explains, with ``explained_by`` cross-referencing that event's type.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import ClassVar, List, Optional, Sequence, Tuple

from .base import EventDetector, EventDetectorRegistry, combine_evidence, soft_threshold
from .kinematics import KinematicSegment, KinematicSeries
from .model import EventType, MatchEvent


@dataclass
class StoppageConfig:
    """Thresholds; stand-still levels are engine-original (design doc §10)."""

    stop_speed_ms: float = 0.7       # median visible-player speed below = standing
    ball_dead_speed_ms: float = 0.5  # ball at rest (or untracked) counts as dead
    min_stop_s: float = 2.5
    min_visible_players: int = 6
    min_confidence: float = 0.25


@EventDetectorRegistry.register
class StoppageDetector(EventDetector):
    """Collective-motion collapse + dead ball => a play-is-stopped span."""

    detector_id: ClassVar[str] = "stoppages"
    display_name: ClassVar[str] = "Stoppages"
    event_types: ClassVar[Tuple[EventType, ...]] = (EventType.STOPPAGE,)
    stage: ClassVar[int] = 5  # last: cross-references every earlier event

    def __init__(self, config: Optional[StoppageConfig] = None):
        self.config = config or StoppageConfig()

    def detect(
        self, kin: KinematicSeries, context: Sequence[MatchEvent] = ()
    ) -> List[MatchEvent]:
        cfg = self.config
        events: List[MatchEvent] = []
        for seg in kin.segments:
            stopped = [self._stopped_at(seg, i) for i in range(len(seg))]
            for start_i, end_i in self._runs(seg, stopped):
                event = self._build_event(kin, seg, start_i, end_i, stopped, context)
                if event is not None and event.confidence >= cfg.min_confidence:
                    events.append(event)
        return events

    def _stopped_at(self, seg: KinematicSegment, i: int) -> Optional[bool]:
        """True/False = play stopped/flowing; None = not enough players visible."""
        cfg = self.config
        speeds = [
            s for tr, _, _ in seg.visible_players(i)
            if (s := tr.speed(i)) is not None
        ]
        if len(speeds) < cfg.min_visible_players:
            return None
        ball_speed = seg.ball_speed(i)
        ball_dead = seg.ball_pos(i) is None or (
            ball_speed is not None and ball_speed < cfg.ball_dead_speed_ms
        )
        return median(speeds) < cfg.stop_speed_ms and ball_dead

    def _runs(
        self, seg: KinematicSegment, stopped: List[Optional[bool]]
    ) -> List[Tuple[int, int]]:
        cfg = self.config
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i, s in enumerate(stopped):
            if s and start is None:
                start = i
            elif not s and start is not None:  # False or None both end the run
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(seg) - 1))
        return [
            (a, b) for a, b in runs
            if seg.grid[b] - seg.grid[a] >= cfg.min_stop_s - 1e-9
        ]

    def _build_event(
        self, kin: KinematicSeries, seg: KinematicSegment,
        start_i: int, end_i: int, stopped: List[Optional[bool]],
        context: Sequence[MatchEvent],
    ) -> Optional[MatchEvent]:
        cfg = self.config
        t0, t1 = seg.grid[start_i], seg.grid[end_i]

        mid = (start_i + end_i) // 2
        speeds = [
            s for tr, _, _ in seg.visible_players(mid)
            if (s := tr.speed(mid)) is not None
        ]
        stillness = soft_threshold(median(speeds), cfg.stop_speed_ms, 0.25, above=False) \
            if speeds else 0.5

        valid = [i for i in range(start_i, end_i + 1) if seg.ball_pos(i) is not None]
        ball_observed = len(valid) / max(1, end_i - start_i + 1)
        ball_evidence = 0.6 + 0.4 * ball_observed  # an untracked ball is weak, not fatal

        # where play stopped: last resting ball position we saw during the span
        xy: Optional[Tuple[float, float]] = None
        for i in reversed(valid):
            xy = seg.ball_pos(i)
            break

        explained_by = next(
            (e.event_type.value for e in context if e.tier < 3 and e.overlaps(t0, t1)),
            None,
        )
        return MatchEvent(
            event_type=EventType.STOPPAGE,
            team=None,
            start_s=t0,
            end_s=t1,
            confidence=combine_evidence([stillness, ball_evidence]),
            period=seg.period,
            x=xy[0] if xy else None,
            y=xy[1] if xy else None,
            metadata={
                "explained_by": explained_by,
                "ball_observed_fraction": round(ball_observed, 2),
            },
        )
