"""Parity tests: streaming causal helpers vs the batch smoothing/derivatives.

The parity principle (live-architecture design doc §2) is asserted here in its
strongest form: for arbitrary inputs (including None runs and short series),
push+flush through the streaming helpers produces the EXACT sequence the batch
helper produces over the full series.
"""

from __future__ import annotations

import random
import unittest

from pipeline.patterns.features import _derivative, _moving_average
from pipeline.patterns.streaming.causal import (
    CausalChain,
    StreamingDerivative,
    StreamingMovingAverage,
)


def _random_series(seed: int, n: int = 200, none_prob: float = 0.15):
    rng = random.Random(seed)
    return [None if rng.random() < none_prob else round(rng.uniform(-30, 30), 3) for _ in range(n)]


def _stream_all(stage, values):
    out = []
    for v in values:
        out.extend(stage.push(v))
    out.extend(stage.flush())
    return out


class StreamingMovingAverageTest(unittest.TestCase):
    def test_parity_plain_ramp(self):
        values = [float(i) for i in range(30)]
        self.assertEqual(_stream_all(StreamingMovingAverage(5), values), _moving_average(values, 5))

    def test_parity_with_nones(self):
        for seed in (1, 2, 3):
            values = _random_series(seed)
            for window in (1, 3, 5, 9):
                with self.subTest(seed=seed, window=window):
                    self.assertEqual(
                        _stream_all(StreamingMovingAverage(window), values),
                        _moving_average(values, window),
                    )

    def test_parity_series_shorter_than_window(self):
        values = [1.0, 2.0, None]
        self.assertEqual(_stream_all(StreamingMovingAverage(9), values), _moving_average(values, 9))

    def test_even_window_bumped_to_odd_like_batch_config(self):
        ma = StreamingMovingAverage(4)
        self.assertEqual(ma.window, 5)

    def test_emission_delay(self):
        ma = StreamingMovingAverage(5)
        emitted = []
        for i in range(10):
            got = ma.push(float(i))
            if i < 2:
                self.assertEqual(got, [])  # nothing until half a window buffered
            emitted.extend(got)
        self.assertEqual(len(emitted), 8)  # 10 pushed - half window (2) held back


class StreamingDerivativeTest(unittest.TestCase):
    DT = 0.2

    def test_parity_plain(self):
        values = [float(i * i) for i in range(20)]
        self.assertEqual(
            _stream_all(StreamingDerivative(self.DT), values), _derivative(values, self.DT)
        )

    def test_parity_with_nones(self):
        for seed in (4, 5, 6):
            values = _random_series(seed)
            with self.subTest(seed=seed):
                self.assertEqual(
                    _stream_all(StreamingDerivative(self.DT), values),
                    _derivative(values, self.DT),
                )

    def test_single_sample(self):
        self.assertEqual(_stream_all(StreamingDerivative(self.DT), [3.0]), [None])


class CausalChainTest(unittest.TestCase):
    DT = 0.2

    def test_smooth_then_derivative_matches_batch_velocity_path(self):
        # FeatureExtractor stage 2: positions smoothed, then differentiated
        for seed in (7, 8):
            values = _random_series(seed)
            batch = _derivative(_moving_average(values, 5), self.DT)
            chain = CausalChain(StreamingMovingAverage(5), StreamingDerivative(self.DT))
            with self.subTest(seed=seed):
                self.assertEqual(_stream_all(chain, values), batch)

    def test_derivative_then_smooth_matches_batch_line_velocity_path(self):
        # FeatureExtractor stage 5: def_line_velocity = smoothed derivative
        for seed in (9, 10):
            values = _random_series(seed)
            batch = _moving_average(_derivative(values, self.DT), 5)
            chain = CausalChain(StreamingDerivative(self.DT), StreamingMovingAverage(5))
            with self.subTest(seed=seed):
                self.assertEqual(_stream_all(chain, values), batch)

    def test_total_delay(self):
        chain = CausalChain(StreamingMovingAverage(5), StreamingDerivative(self.DT))
        self.assertEqual(chain.delay_samples, 3)  # half window (2) + derivative (1)


if __name__ == "__main__":
    unittest.main()
