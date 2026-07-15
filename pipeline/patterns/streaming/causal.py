"""Causal (streaming) counterparts of the batch smoothing/derivative helpers.

The live pipeline's cornerstone is the PARITY PRINCIPLE (live-architecture design
doc §2): every number the streaming path computes must be bit-identical to what
the batch path would compute over the same frames — otherwise thresholds tuned in
the offline validation loop are meaningless live.

Batch smoothing is a CENTERED moving average (``features._moving_average``) and
velocities use CENTRAL differences (``features._derivative``); both look into the
future. The streaming versions here keep the exact same math and buy causality
with DELAY instead of with a different filter: a value for sample ``i`` is
emitted only once sample ``i + lookahead`` has arrived (lookahead = half the
smoothing window, or 1 for the derivative). We deliberately do NOT use an
exponential/one-sided filter, which would be zero-delay but produce *different
numbers* than the batch pipeline and silently invalidate offline tuning.

Delay accounting lives in ``streaming.features.feature_latencies``; the values
are small (≈0.5–1.1 s at 5 Hz with a 1 s window) and are dwarfed by the GPU
stage latency anyway (design doc §1).

Implementation note: each emitted value is computed by running the *batch*
helper over a window-sized slice around the target index — the slice fully
determines the batch result for that index, so parity holds by construction
(asserted in tests) rather than by a re-implementation that could drift.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from ..features import _derivative, _moving_average


class StreamingMovingAverage:
    """Causal, delayed version of ``features._moving_average``.

    ``push(v)`` returns the smoothed values that became computable (0 or 1 in
    steady state; values come out ``half_window`` samples behind the input).
    ``flush()`` emits the tail after the last sample (edge windows shrink,
    exactly like the batch helper at series end). ``None`` inputs are legal and
    propagate exactly as in batch (smoothing never bridges a ``None``).
    """

    def __init__(self, window: int):
        if window < 1:
            raise ValueError("window must be >= 1")
        if window % 2 == 0:
            window += 1  # mirror FeatureExtractor's odd-window adjustment
        self.window = window
        self.half = window // 2
        self._buf: Deque[Optional[float]] = deque(maxlen=window)
        self._pushed = 0
        self._emitted = 0

    @property
    def delay_samples(self) -> int:
        return self.half

    def push(self, value: Optional[float]) -> List[Optional[float]]:
        self._buf.append(value)
        self._pushed += 1
        out: List[Optional[float]] = []
        while self._emitted <= self._pushed - 1 - self.half:
            out.append(self._value_at(self._emitted))
            self._emitted += 1
        return out

    def flush(self) -> List[Optional[float]]:
        """Emit the last ``half`` values (shrunken right-edge windows)."""
        out: List[Optional[float]] = []
        while self._emitted < self._pushed:
            out.append(self._value_at(self._emitted))
            self._emitted += 1
        return out

    def _value_at(self, i: int) -> Optional[float]:
        # The batch value at index i depends only on values[i-half : i+half+1];
        # everything needed is in the deque by the time we emit i.
        lo = self._pushed - len(self._buf)
        a = max(lo, i - self.half)
        window_slice = [self._buf[k - lo] for k in range(a, self._pushed)]
        return _moving_average(window_slice, self.window)[i - a]


class StreamingDerivative:
    """Causal, delayed version of ``features._derivative`` (delay: 1 sample).

    Central difference where both neighbours exist, one-sided at run edges —
    identical to batch because the value at index i depends only on
    values[i-1 : i+2].
    """

    def __init__(self, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self._buf: Deque[Optional[float]] = deque(maxlen=3)
        self._pushed = 0
        self._emitted = 0

    @property
    def delay_samples(self) -> int:
        return 1

    def push(self, value: Optional[float]) -> List[Optional[float]]:
        self._buf.append(value)
        self._pushed += 1
        out: List[Optional[float]] = []
        while self._emitted <= self._pushed - 2:
            out.append(self._value_at(self._emitted))
            self._emitted += 1
        return out

    def flush(self) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        while self._emitted < self._pushed:
            out.append(self._value_at(self._emitted))
            self._emitted += 1
        return out

    def _value_at(self, i: int) -> Optional[float]:
        lo = self._pushed - len(self._buf)
        a = max(lo, i - 1)
        window_slice = [self._buf[k - lo] for k in range(a, self._pushed)]
        return _derivative(window_slice, self.dt)[i - a]


class CausalChain:
    """Compose streaming stages; output of one feeds the next.

    Mirrors the batch compositions:
    - positions -> velocity: ``StreamingMovingAverage`` then ``StreamingDerivative``
      (FeatureExtractor stage 2);
    - line height -> line velocity: ``StreamingDerivative`` then
      ``StreamingMovingAverage`` (FeatureExtractor stage 5 smooths the derivative).

    ``flush()`` drains every stage in order, feeding tails downstream, so total
    output length always equals total input length.
    """

    def __init__(self, *stages):
        if not stages:
            raise ValueError("need at least one stage")
        self.stages = stages

    @property
    def delay_samples(self) -> int:
        return sum(s.delay_samples for s in self.stages)

    def push(self, value: Optional[float]) -> List[Optional[float]]:
        values = [value]
        for stage in self.stages:
            nxt: List[Optional[float]] = []
            for v in values:
                nxt.extend(stage.push(v))
            values = nxt
        return values

    def flush(self) -> List[Optional[float]]:
        values: List[Optional[float]] = []
        for stage in self.stages:
            drained: List[Optional[float]] = []
            for v in values:
                drained.extend(stage.push(v))
            drained.extend(stage.flush())
            values = drained
        return values
