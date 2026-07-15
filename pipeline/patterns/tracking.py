"""Raw tracking data model and ingest adapter.

This is the only module that knows anything about the upstream sn-gamestate/TrackLab
output format. Everything downstream (features, detectors) consumes the neutral
dataclasses defined here, so a change of tracking provider touches exactly one file.

Coordinate conventions (see /docs/phase4-5-design.md §1.3):
- Canonical pitch 105 x 68 m, origin at the centre spot.
- ``x`` runs along the touchline direction, ``y`` along the halfway line.
- All positions in this module are *absolute* pitch coordinates; the team-relative
  frame (own goal at x'=0) is introduced by the feature layer, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


class TeamSide(Enum):
    """Stable team identity for a match (mapped from TrackLab's left/right labels)."""

    HOME = "home"
    AWAY = "away"

    def opponent(self) -> "TeamSide":
        return TeamSide.AWAY if self is TeamSide.HOME else TeamSide.HOME


class Role(Enum):
    GOALKEEPER = "goalkeeper"
    OUTFIELD = "outfield"
    REFEREE = "referee"
    OTHER = "other"


@dataclass(frozen=True)
class PlayerObservation:
    """One tracked person in one frame, projected onto the pitch.

    ``track_id`` is the upstream tracker identity — subject to ID switches; Phase 4
    logic must not assume long-term stability. ``jersey_number`` is best-effort OCR
    and frequently ``None``.
    """

    track_id: int
    team: Optional[TeamSide]  # None when team classification failed for this frame
    role: Role
    x: float
    y: float
    jersey_number: Optional[int] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class BallObservation:
    """Ball position for one frame. Absent entirely when the ball was not detected."""

    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class TrackingFrame:
    """Everything the upstream pipeline knows about one video frame.

    ``players`` contains only *visible* people (broadcast camera => typically 12-16 of
    22); partial observability is the norm, not an error state.
    """

    frame_index: int
    timestamp_s: float
    period: int  # 1 or 2 (extensible to extra time)
    players: Tuple[PlayerObservation, ...]
    ball: Optional[BallObservation]

    def outfielders(self, team: TeamSide) -> List[PlayerObservation]:
        """Visible outfield players of ``team`` in this frame."""
        return [p for p in self.players if p.team is team and p.role is Role.OUTFIELD]


@dataclass(frozen=True)
class MatchMeta:
    """Match-level facts needed to interpret frames.

    ``home_attacks_positive_x`` maps period -> whether HOME's attacking direction is
    +x in that period. This is the single source of truth the feature layer uses to
    build the team-relative frame; it must be established during ingest (e.g. from
    kickoff positions or provided manually).
    """

    match_id: str
    home_attacks_positive_x: Dict[int, bool] = field(default_factory=dict)
    pitch_length_m: float = PITCH_LENGTH_M
    pitch_width_m: float = PITCH_WIDTH_M

    def attack_direction(self, team: TeamSide, period: int) -> int:
        """Return +1 if ``team`` attacks toward +x in ``period``, else -1."""
        if period not in self.home_attacks_positive_x:
            raise ValueError(
                f"attack direction unknown for period {period} "
                f"(match {self.match_id}); ingest must populate home_attacks_positive_x"
            )
        home_positive = self.home_attacks_positive_x[period]
        if team is TeamSide.HOME:
            return 1 if home_positive else -1
        return -1 if home_positive else 1


@dataclass
class MatchTracking:
    """A match's full (gappy) tracking stream: metadata + time-ordered frames.

    Frames are NOT guaranteed contiguous — broadcast cuts leave gaps. Segmentation
    into continuous spans happens in the feature layer, which owns the definition of
    "continuous" (max_gap_s).
    """

    meta: MatchMeta
    frames: List[TrackingFrame]


def load_tracklab_states(path: str, match_id: str) -> MatchTracking:
    """Adapt a TrackLab states file (sn-gamestate output) into a ``MatchTracking``.

    Responsibilities (implementation pending real output from /research):
    - map TrackLab's per-detection rows (bbox_pitch, track_id, role, team left/right,
      jersey) into ``PlayerObservation``s grouped by frame;
    - convert TrackLab pitch coordinates into our canonical frame if they differ;
    - map left/right team labels to HOME/AWAY stably across the match;
    - infer ``home_attacks_positive_x`` per period (kickoff heuristic or sidecar
      config) and record it in ``MatchMeta``;
    - drop referee/other detections into ``Role.REFEREE``/``Role.OTHER`` (kept for
      debugging, ignored by features).

    The exact file schema is pinned down by the Phase 1-3 research notebooks; keep
    all knowledge of it inside this function.
    """
    raise NotImplementedError
