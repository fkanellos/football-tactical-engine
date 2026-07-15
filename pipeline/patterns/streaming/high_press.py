"""Streaming high press: the worked example of wrapping a batch detector live.

Scoring is ``detectors.high_press.frame_score`` — the SAME function the batch
detector runs, so a live episode and the post-match episode over the same
frames get identical scores by construction. Lifecycle thresholds come straight
from ``HighPressConfig`` (enter/exit/min_duration/merge_gap); only the
live-specific knobs (``provisional_after_s`` etc.) are new.

Metadata differences vs batch, forced by causality:
- ``trigger`` ("restart"/"open_play") uses a rolling ring of recent possession
  states instead of looking back over the assembled segment — same 6 s
  definition, evaluated at episode onset (first-write-wins in the machine);
- ``is_counter_press`` is tagged from time-since-turnover at onset rather than
  the episode median — the live tag can therefore differ from the batch tag on
  episodes that start >5 s after a turnover but whose median falls under it (a
  documented, harmless divergence: the batch profile remains the audit record).

The other episode-shaped detectors (low block, flank overload, offside trap)
wrap the same way once their per-frame scoring is extracted to module level
like high_press's — deliberately not done wholesale here to keep the diff to
the batch layer reviewable. Counter-attack is anchored-window-shaped instead:
see ``AnchoredWindowMachine`` and live-architecture design doc §4.3.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar, Deque, Dict, Mapping, Optional, Tuple

from ..detectors.high_press import (
    HighPressConfig,
    HighPressDetector,
    frame_intensity as batch_frame_intensity,
    frame_score as batch_frame_score,
)
from ..features import FrameFeatures, PossessionState
from ..tracking import MatchMeta, TeamSide
from .detector import ConfidencePrior, StreamingDetector, StreamingEpisodeConfig


class StreamingHighPressDetector(StreamingDetector):
    pattern_id: ClassVar[str] = HighPressDetector.pattern_id
    display_name: ClassVar[str] = HighPressDetector.display_name
    required_features: ClassVar[Tuple[str, ...]] = HighPressDetector.required_features

    def __init__(
        self,
        config: Optional[HighPressConfig] = None,
        priors: Optional[Mapping[TeamSide, ConfidencePrior]] = None,
        provisional_after_s: float = 1.5,
    ):
        self.config = config or HighPressConfig()
        episode_config = StreamingEpisodeConfig(
            enter_threshold=self.config.score_enter,
            exit_threshold=self.config.score_exit,
            min_duration_s=self.config.min_duration_s,
            merge_gap_s=self.config.merge_gap_s,
            n_components=5,  # HighPressDetector._N_COMPONENTS
            provisional_after_s=provisional_after_s,
        )
        super().__init__(episode_config, priors=priors)
        # rolling window of (timestamp, possession state) for the restart tag
        self._recent_states: Deque[Tuple[float, PossessionState]] = deque()

    def push(self, frame: FrameFeatures, meta: MatchMeta):
        self._recent_states.append((frame.timestamp_s, frame.possession.state))
        while self._recent_states and frame.timestamp_s - self._recent_states[0][0] > 6.0:
            self._recent_states.popleft()
        return super().push(frame, meta)

    def segment_break(self):
        self._recent_states.clear()
        return super().segment_break()

    def frame_score(
        self, frame: FrameFeatures, meta: MatchMeta, team: TeamSide
    ) -> Tuple[Optional[float], bool, Optional[float], Optional[Dict[str, Any]]]:
        score, complete = batch_frame_score(frame, team, meta, frame.period, self.config)
        if score is None:
            return None, complete, None, None
        intensity = batch_frame_intensity(frame, team, self.config)
        recent_dead = any(
            state is PossessionState.DEAD for _, state in self._recent_states
        )
        tst = frame.possession.time_since_turnover_s
        metadata = {
            "trigger": "restart" if recent_dead else "open_play",
            "is_counter_press": tst is not None and tst < self.config.counter_press_window_s,
        }
        return score, complete, intensity, metadata
