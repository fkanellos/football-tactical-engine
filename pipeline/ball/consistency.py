"""Consistency of the ball with player behaviour.

The camera follows the ball and players chase it, so a detected "ball" that
stays tens of metres from every visible player during open play is suspect —
the paper-debris false-positive signature on our footage (design §3). The
per-frame signal is the distance to the nearest visible player; the *sustained*
version (isolation spans) is the strong one, because a genuine long pass puts
the ball 20+ m from everyone for a second or two, while nothing in football
leaves a live ball unattended for many seconds.

Player positions arrive as bare pitch points per frame, aligned by index with
the ball samples — deliberately decoupled from ``TrackingFrame`` so this module
works on the minimal export the first GPU session produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .model import BallSample, Point2

PlayerFrame = Sequence[Point2]  # visible players' pitch positions for one frame


def nearest_player_distances(
    samples: Sequence[BallSample],
    player_frames: Sequence[Optional[PlayerFrame]],
) -> List[Optional[float]]:
    """Per index: distance from the detected ball to the nearest visible player.

    None when the ball is missing, the frame has no player data, or no players
    are visible. ``player_frames`` must be index-aligned with ``samples``.
    """
    if len(player_frames) != len(samples):
        raise ValueError(
            f"player_frames ({len(player_frames)}) and samples ({len(samples)}) "
            "must be index-aligned"
        )
    out: List[Optional[float]] = []
    for s, players in zip(samples, player_frames):
        if s.xy is None or not players:
            out.append(None)
            continue
        bx, by = s.xy
        out.append(min(math.hypot(bx - px, by - py) for (px, py) in players))
    return out


@dataclass(frozen=True)
class IsolationSpan:
    """A sustained run of detections all far from every visible player."""

    start_index: int
    end_index: int  # one past the last isolated frame
    duration_s: float
    min_distance_m: float  # the closest any player came during the span


def isolation_spans(
    samples: Sequence[BallSample],
    player_frames: Sequence[Optional[PlayerFrame]],
    min_isolation_m: float = 25.0,
    min_duration_s: float = 2.0,
) -> List[IsolationSpan]:
    """Spans where a detected ball stayed > min_isolation_m from everyone.

    Frames without ball or player data break a span (absence of evidence is not
    isolation). A long pass in flight survives the default thresholds — 25 m of
    separation sustained for 2 s needs ~2 s of nobody closing on the ball, which
    open play essentially never produces.
    """
    distances = nearest_player_distances(samples, player_frames)
    spans: List[IsolationSpan] = []
    start: Optional[int] = None
    span_min = math.inf
    for i, d in enumerate(distances):
        isolated = d is not None and d > min_isolation_m
        if isolated:
            if start is None:
                start = i
                span_min = math.inf
            span_min = min(span_min, d)
            continue
        if start is not None:
            _close_span(samples, spans, start, i, span_min, min_duration_s)
            start = None
    if start is not None:
        _close_span(samples, spans, start, len(samples), span_min, min_duration_s)
    return spans


def _close_span(
    samples: Sequence[BallSample],
    spans: List[IsolationSpan],
    start: int,
    end: int,
    span_min: float,
    min_duration_s: float,
) -> None:
    duration = samples[end - 1].t - samples[start].t
    if duration >= min_duration_s:
        spans.append(IsolationSpan(start, end, duration, span_min))
