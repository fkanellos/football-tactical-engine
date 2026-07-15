"""Causal possession state machine for the live pipeline.

The batch possession machine (``FeatureExtractor._infer_possession``) confirms a
possession flip only after it persists for ``turnover_persistence_s``, then
BACKDATES the flip: frames between the first touch and the confirmation are
rewritten to the new side. Live we cannot rewrite frames that were already
emitted, so this machine makes the causal version of the same decision:

- frames during the pending window keep reporting the OLD holder (exactly what
  the batch machine's *provisional* pass says before rewriting);
- when the flip survives the persistence window, a ``TurnoverConfirmation`` is
  emitted carrying the BACKDATED start time — the single point where live and
  batch feature streams are allowed to differ, and the live analogue of the
  batch rewrite. Downstream consumers that anchor on turnovers (the streaming
  counter-attack detector) receive the anchor ``turnover_persistence_s`` late
  but with the true start time, and evaluate their window from that backdated
  anchor (live-architecture design doc §3.4).

Everything else mirrors the batch machine one-to-one (same branch structure, so
divergence can only come from the no-rewrite constraint, not from drift):

- DEAD (ball out of bounds) resets holder AND pending; the first clear holder
  afterwards re-establishes possession WITHOUT a turnover event (restarts never
  create counter-attack anchors);
- a missing ball / no clear holder persists the current state;
- CONTESTED frames keep a pending flip alive (a 50/50 in the middle of a
  steal does not reset the persistence clock).

The raw per-frame read (who is within possession radius of the ball) is pure
per-frame geometry with no history; the streaming feature extractor computes it
with the same code path as batch and feeds it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ..features import PossessionFeatures, PossessionState
from ..tracking import TeamSide

#: Per-frame possession read, as produced by the batch machine's first pass:
#: None (ball missing / no clear holder), "dead", "contested", or
#: (side_in_possession, holder_track_id).
RawPossessionRead = Union[None, str, Tuple[TeamSide, int]]


@dataclass(frozen=True)
class TurnoverConfirmation:
    """A possession flip that survived the persistence window.

    ``started_at_s`` is backdated to when the new holder first appeared —
    it equals the ``timestamp_s`` of the frame the batch machine would stamp
    ``turnover_won_by`` on. ``confirmed_at_s - started_at_s`` is (at least) the
    persistence window: the structural lag of every turnover-anchored live
    signal.
    """

    side: TeamSide
    started_at_s: float
    confirmed_at_s: float


class StreamingPossessionMachine:
    """push(t, read) -> (PossessionFeatures, Optional[TurnoverConfirmation])."""

    def __init__(self, turnover_persistence_s: float = 2.0):
        self.turnover_persistence_s = turnover_persistence_s
        self._current: Optional[TeamSide] = None
        self._last_flip_t: Optional[float] = None
        self._pending: Optional[Tuple[TeamSide, float]] = None  # (side, start_t)

    def segment_reset(self) -> None:
        """Broadcast cut / new segment: carrying a belief across is guesswork
        (mirrors the batch machine running per-segment)."""
        self._current = None
        self._last_flip_t = None
        self._pending = None

    def push(
        self, t: float, read: RawPossessionRead
    ) -> Tuple[PossessionFeatures, Optional[TurnoverConfirmation]]:
        out = PossessionFeatures()
        event: Optional[TurnoverConfirmation] = None

        if read == "dead":
            out.state = PossessionState.DEAD
            self._current = None  # restart re-establishes possession, no turnover
            self._pending = None
            return out, None

        if isinstance(read, tuple):
            side, holder = read
            out.holder_track_id = holder
            if self._current is None:
                self._current = side  # first establishment: no turnover event
                self._pending = None
            elif side is self._current:
                self._pending = None
            else:
                if self._pending is None or self._pending[0] is not side:
                    self._pending = (side, t)
                if t - self._pending[1] >= self.turnover_persistence_s - 1e-9:
                    started = self._pending[1]
                    self._current = side
                    self._last_flip_t = started
                    self._pending = None
                    event = TurnoverConfirmation(
                        side=side, started_at_s=started, confirmed_at_s=t
                    )
                    # batch stamps turnover_won_by on the backdated start frame;
                    # live can only stamp the confirmation frame — the backdated
                    # time travels in the event instead.
                    out.turnover_won_by = side
        elif read == "contested":
            out.state = PossessionState.CONTESTED
            out.time_since_turnover_s = (
                None if self._last_flip_t is None else t - self._last_flip_t
            )
            return out, None
        # read is None (ball missing / no holder): state persists; pending kept.

        if self._current is None:
            out.state = PossessionState.CONTESTED
        else:
            out.state = PossessionState.for_side(self._current)
            out.time_since_turnover_s = (
                None if self._last_flip_t is None else t - self._last_flip_t
            )
        return out, event
