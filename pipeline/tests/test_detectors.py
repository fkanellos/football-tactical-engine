"""Detector semantics tests: fire on the textbook positive, stay silent on
negatives and edge cases. These pin what each heuristic MEANS (design doc §6)."""

from __future__ import annotations

import unittest
from functools import lru_cache
from typing import List

from pipeline.patterns.detectors import (
    CounterAttackDetector,
    FlankOverloadDetector,
    HighPressDetector,
    LowBlockDetector,
    OffsideTrapDetector,
)
from pipeline.patterns.detectors.base import PatternEvent
from pipeline.patterns.features import FeatureExtractor, FeatureSeries
from pipeline.patterns.testing import synthetic
from pipeline.patterns.tracking import TeamSide

_SCENARIOS = {
    "high_press": synthetic.high_press_scenario,
    "high_press_low_vis": synthetic.high_press_low_visibility_scenario,
    "passive_buildup": synthetic.passive_buildup_scenario,
    "low_block": synthetic.low_block_scenario,
    "low_block_short": lambda: synthetic.low_block_scenario(block_duration_s=8.0),
    "mid_block": synthetic.mid_block_scenario,
    "flank_overload": synthetic.flank_overload_scenario,
    "balanced_attack": synthetic.balanced_attack_scenario,
    "line_step": synthetic.line_step_scenario,
    "line_drop": lambda: synthetic.line_step_scenario(step=False),
    "counter": synthetic.counter_attack_scenario,
    "slow_transition": synthetic.slow_transition_scenario,
    "flicker": synthetic.flickering_possession_scenario,
}


@lru_cache(maxsize=None)
def features(scenario: str) -> FeatureSeries:
    match = _SCENARIOS[scenario]()
    return FeatureExtractor().extract(match), match.meta  # type: ignore[return-value]


def run(detector, scenario: str) -> List[PatternEvent]:
    series, meta = features(scenario)
    return detector.detect(series, meta)


def confident(events: List[PatternEvent], team: TeamSide = None) -> List[PatternEvent]:
    return [
        e for e in events if e.confidence >= 0.5 and (team is None or e.team is team)
    ]


def overlapping(events: List[PatternEvent], t0: float, t1: float) -> List[PatternEvent]:
    return [e for e in events if e.start_s < t1 and e.end_s > t0]


class HighPressDetectorTest(unittest.TestCase):
    def test_fires_on_textbook_press(self):
        events = confident(run(HighPressDetector(), "high_press"), TeamSide.HOME)
        self.assertTrue(events, "expected a confident HOME high_press event")
        self.assertTrue(overlapping(events, 9.0, 16.0))
        self.assertGreater(events[0].intensity, 0.2)
        self.assertFalse(confident(run(HighPressDetector(), "high_press"), TeamSide.AWAY))

    def test_silent_on_passive_buildup(self):
        self.assertEqual(run(HighPressDetector(), "passive_buildup"), [])

    def test_low_visibility_never_confident(self):
        events = run(HighPressDetector(), "high_press_low_vis")
        self.assertFalse(
            confident(events),
            f"close-up camera hides the press context; got {events}",
        )


class LowBlockDetectorTest(unittest.TestCase):
    def test_fires_on_sustained_deep_block(self):
        events = confident(run(LowBlockDetector(), "low_block"), TeamSide.HOME)
        self.assertTrue(events, "expected a confident HOME low_block event")
        event = events[0]
        self.assertGreaterEqual(event.duration_s, 12.0)
        self.assertTrue(overlapping([event], 15.0, 40.0))
        self.assertLess(event.metadata["mean_line_height_m"], 16.0)
        self.assertFalse(confident(run(LowBlockDetector(), "low_block"), TeamSide.AWAY))

    def test_silent_when_deep_spell_too_short(self):
        self.assertFalse(confident(run(LowBlockDetector(), "low_block_short")))

    def test_silent_on_mid_block(self):
        self.assertEqual(run(LowBlockDetector(), "mid_block"), [])


class FlankOverloadDetectorTest(unittest.TestCase):
    def test_fires_on_left_overload(self):
        events = confident(run(FlankOverloadDetector(), "flank_overload"), TeamSide.HOME)
        self.assertTrue(events, "expected a confident HOME flank_overload event")
        self.assertTrue(all(e.metadata["side"] == "left" for e in events))
        self.assertTrue(overlapping(events, 9.0, 15.5))
        self.assertGreaterEqual(events[0].metadata["max_attackers_in_zone"], 3)

    def test_no_right_side_or_away_events(self):
        events = run(FlankOverloadDetector(), "flank_overload")
        self.assertFalse([e for e in events if e.metadata["side"] == "right"])
        self.assertFalse([e for e in events if e.team is TeamSide.AWAY])

    def test_silent_on_balanced_attack(self):
        self.assertEqual(run(FlankOverloadDetector(), "balanced_attack"), [])


class OffsideTrapDetectorTest(unittest.TestCase):
    def test_fires_on_coordinated_step(self):
        events = [e for e in run(OffsideTrapDetector(), "line_step") if e.team is TeamSide.HOME]
        self.assertTrue(events, "expected a HOME offside_trap event")
        event = events[0]
        self.assertTrue(overlapping([event], 5.0, 9.5))
        self.assertGreaterEqual(event.confidence, 0.35)
        self.assertLessEqual(event.confidence, 0.7)  # honesty cap
        self.assertGreaterEqual(event.metadata["line_gain_m"], 2.0)

    def test_silent_when_line_drops(self):
        self.assertEqual(run(OffsideTrapDetector(), "line_drop"), [])


class CounterAttackDetectorTest(unittest.TestCase):
    def test_fires_on_fast_break(self):
        events = confident(run(CounterAttackDetector(), "counter"), TeamSide.HOME)
        self.assertTrue(events, "expected a confident HOME counter_attack event")
        event = events[0]
        self.assertAlmostEqual(event.start_s, 8.5, delta=1.5)
        self.assertEqual(event.metadata["turnover_location"], "deep")
        self.assertGreaterEqual(event.metadata["n_runners"], 2)
        self.assertGreaterEqual(event.metadata["gain_m"], 25.0)

    def test_silent_on_slow_transition(self):
        self.assertEqual(run(CounterAttackDetector(), "slow_transition"), [])

    def test_silent_without_confirmed_turnover(self):
        self.assertEqual(run(CounterAttackDetector(), "flicker"), [])


class CrossScenarioSilenceTest(unittest.TestCase):
    """No detector may fire confidently on another motif's scenario (unless the
    scenarios genuinely overlap, listed in ALLOWED)."""

    ALLOWED = {
        ("high_press", "high_press"),
        ("low_block", "low_block"),
        ("low_block", "low_block_short"),   # short spell: silent required anyway by
                                            # its own test; here we allow low-conf noise
        ("flank_overload", "flank_overload"),
        ("offside_trap", "line_step"),
        ("counter_attack", "counter"),
    }

    def test_matrix(self):
        detectors = [
            HighPressDetector(),
            LowBlockDetector(),
            FlankOverloadDetector(),
            OffsideTrapDetector(),
            CounterAttackDetector(),
        ]
        for scenario in _SCENARIOS:
            for det in detectors:
                if (det.pattern_id, scenario) in self.ALLOWED:
                    continue
                events = confident(run(det, scenario))
                self.assertFalse(
                    events,
                    f"{det.pattern_id} fired confidently on '{scenario}': {events}",
                )


if __name__ == "__main__":
    unittest.main()
