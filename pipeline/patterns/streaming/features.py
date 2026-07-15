"""Streaming feature extraction: the causal twin of ``FeatureExtractor``.

Batch ``FeatureExtractor`` sees a whole segment at once; live we see one
tracking frame at a time. The redesign keeps every formula and threshold and
converts the non-causal steps into delayed-causal ones (live-architecture
design doc §3):

    stage                 batch (whole segment)          streaming (this module)
    -----------------------------------------------------------------------------
    resample to grid      interpolate across gaps        hold back a grid point
                          <= max_gap_s                   until the next raw obs
                                                         arrives (bounded by
                                                         max_gap_s) or emit from
                                                         the nearest obs
    smooth positions      centered moving average        StreamingMovingAverage:
                                                         same math, emitted
                                                         half a window late
    velocities            central difference             StreamingDerivative:
                                                         same math, 1 sample late
    possession            state machine with             StreamingPossessionMachine:
                          backdated flips                same machine, flips emit
                                                         TurnoverConfirmation with
                                                         the backdated start
    per-frame assembly    pure per-frame geometry        identical code path —
    (shape/pressure/lanes)                               nothing to change
    line velocity /       derivative then smoothed       CausalChain(derivative,
    closing speed         over assembled series          moving average)
    quality               per-frame                      identical

Because every stage is either per-frame pure or delayed-exact, a live
``FrameFeatures`` emitted for time t is IDENTICAL to the batch one for the same
frames — the parity principle. The price is a fixed, known emission delay
(``feature_latencies``), not different numbers.

STARTUP TRANSIENT: after a segment start there is no history. The causal
helpers emit edge-shrunk values immediately (exactly like batch does at a
segment's first frames), so features are never None *because of streaming* —
they are merely as noisy as batch's segment-edge values. The honest warm-up
figure in ``feature_latencies`` is the time until a feature's values match
mid-segment quality; detectors already survive noisy edges because their
episode gates (min_duration_s, persistence) exceed every warm-up below.

Status: ``feature_latencies`` is implemented (it is pure arithmetic and the
design doc's latency table is generated from it). The extractor class is a
skeleton pending the same real-tracking-data validation the batch extractor
gets first; its stages are all either shared with batch or implemented in
``causal.py``/``possession.py``, so filling it in is assembly work, not design
work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..features import FeatureExtractorConfig, FrameFeatures
from ..tracking import MatchMeta, TrackingFrame


@dataclass(frozen=True)
class FeatureLatency:
    """Structural timing of one feature channel in the live pipeline.

    ``delay_s``: how long after real time t the value FOR t is emitted
    (resample hold-back + smoothing look-ahead + derivative look-ahead). This
    is exact and unavoidable — it buys bit-parity with batch.

    ``warmup_s``: after a segment start, how long until values reach
    mid-segment quality (edge windows are shrunk, derivatives one-sided).
    Values ARE emitted during warm-up, flagged only by their position after a
    segment start; detectors' duration gates absorb this.
    """

    delay_s: float
    warmup_s: float


def feature_latencies(config: Optional[FeatureExtractorConfig] = None) -> Dict[str, FeatureLatency]:
    """Per-feature-family structural latency, derived from the extractor config.

    At the defaults (5 Hz, 1 s smoothing window): positions/shape 0.5 s,
    velocities 0.7 s, line velocity / closing speed 1.1 s. Turnover-anchored
    signals additionally wait ``turnover_persistence_s`` (2 s) for flip
    confirmation — that is an event lag, not a feature delay, and is listed
    under ``possession.turnover``.
    """
    cfg = config or FeatureExtractorConfig()
    dt = 1.0 / cfg.resample_hz
    w = max(1, int(round(cfg.smoothing_window_s * cfg.resample_hz)))
    if w % 2 == 0:
        w += 1
    half = w // 2

    hold_back = dt / 2  # nearest-obs resampling look-ahead (typical; a tracking
    # dropout stretches this up to max_gap_s while interpolation waits)
    smooth = half * dt
    deriv = dt

    position_delay = hold_back + smooth
    velocity_delay = position_delay + deriv
    # def_line_velocity / press_closing_speed: assembled series -> derivative
    # -> second smoothing pass (FeatureExtractor._post_derivatives)
    line_velocity_delay = position_delay + deriv + smooth

    return {
        "positions_shape": FeatureLatency(position_delay, warmup_s=smooth),
        "velocities": FeatureLatency(velocity_delay, warmup_s=smooth + deriv),
        "line_velocity_closing_speed": FeatureLatency(
            line_velocity_delay, warmup_s=2 * smooth + deriv
        ),
        "possession.state": FeatureLatency(position_delay, warmup_s=0.0),
        "possession.turnover": FeatureLatency(
            position_delay + cfg.turnover_persistence_s, warmup_s=0.0
        ),
        "quality": FeatureLatency(position_delay, warmup_s=0.0),
    }


class StreamingFeatureExtractor:
    """push(TrackingFrame) -> FrameFeatures emitted as they become final.

    Skeleton (see module docstring for status). Responsibilities when filled in:

    - maintain per-track and ball channel buffers; align raw observations onto
      the same uniform grid as batch (``_sample_channel`` semantics: nearest
      obs within dt/2, else interpolation once the closing observation arrives,
      never across more than ``max_gap_s``);
    - run positions through ``causal.StreamingMovingAverage`` and velocities
      through ``causal.StreamingDerivative`` (identical numbers to batch,
      half-window late);
    - compute the per-grid-point possession read (same geometry as
      ``FeatureExtractor._infer_possession``'s first pass) and feed
      ``possession.StreamingPossessionMachine``; surface returned
      ``TurnoverConfirmation``s to callers (counter-attack anchors);
    - assemble ``FrameFeatures`` with the batch per-frame geometry
      (``_fill_team_shape`` / ``_fill_pressure_and_occupancy`` refactored to be
      callable per frame — they already are per-frame pure);
    - run line height / mean-3-defender distance through
      ``causal.CausalChain(StreamingDerivative, StreamingMovingAverage)`` to
      produce ``def_line_velocity`` / ``press_closing_speed``;
    - on ``notice_break()``: flush delay lines (batch right-edge semantics),
      emit the tail frames, reset all channel state and the possession machine,
      and increment ``segment_id``.
    """

    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        self.config = config or FeatureExtractorConfig()
        self.segment_id = 0

    def push(self, frame: TrackingFrame, meta: MatchMeta) -> List[FrameFeatures]:
        """Ingest one raw tracking frame; return any feature frames that became
        final (0..n per push — n > 1 when a gap-closing observation lets several
        held-back grid points interpolate and emit at once)."""
        raise NotImplementedError

    def notice_break(self, reason: str = "broadcast_cut") -> List[FrameFeatures]:
        """Upstream declared a discontinuity (cut/replay/calibration loss):
        flush and reset. Returns the flushed tail frames of the dying segment."""
        raise NotImplementedError

    def flush(self) -> List[FrameFeatures]:
        """End of stream: drain all delay lines."""
        raise NotImplementedError
