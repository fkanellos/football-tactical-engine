"""Semantics tests for the causal possession machine.

Mirrors the batch state machine's documented behaviour (features.py module
docstring): persistence-gated flips, DEAD resets without turnover events,
state persistence through missing-ball frames, contested frames keeping a
pending flip alive.
"""

from __future__ import annotations

import unittest

from pipeline.patterns.features import PossessionState
from pipeline.patterns.streaming.possession import StreamingPossessionMachine
from pipeline.patterns.tracking import TeamSide

DT = 0.2
HOME, AWAY = TeamSide.HOME, TeamSide.AWAY


def run(machine, reads, t0=0.0):
    """Push reads at DT spacing; return (features list, events list)."""
    feats, events = [], []
    for i, read in enumerate(reads):
        f, e = machine.push(t0 + i * DT, read)
        feats.append(f)
        if e is not None:
            events.append((t0 + i * DT, e))
    return feats, events


class PossessionMachineTest(unittest.TestCase):
    def machine(self):
        return StreamingPossessionMachine(turnover_persistence_s=2.0)

    def test_first_establishment_no_turnover(self):
        feats, events = run(self.machine(), [(HOME, 4)] * 5)
        self.assertEqual(events, [])
        self.assertTrue(all(f.state is PossessionState.HOME for f in feats))

    def test_confirmed_flip_is_backdated(self):
        reads = [(HOME, 4)] * 5 + [(AWAY, 17)] * 15
        feats, events = run(self.machine(), reads)
        self.assertEqual(len(events), 1)
        confirmed_at, event = events[0]
        self.assertIs(event.side, AWAY)
        self.assertAlmostEqual(event.started_at_s, 5 * DT)  # first AWAY touch
        self.assertAlmostEqual(event.confirmed_at_s, confirmed_at)
        self.assertAlmostEqual(event.confirmed_at_s - event.started_at_s, 2.0)
        # causal constraint: frames during the pending window still said HOME
        self.assertIs(feats[10].state, PossessionState.HOME)
        # after confirmation the clock counts from the BACKDATED start
        idx = int(round(confirmed_at / DT))
        self.assertIs(feats[idx].state, PossessionState.AWAY)
        self.assertAlmostEqual(feats[idx].time_since_turnover_s, 2.0)

    def test_fizzled_flip_never_confirms(self):
        # AWAY touches for 1s (< 2s persistence), HOME recovers
        reads = [(HOME, 4)] * 5 + [(AWAY, 17)] * 5 + [(HOME, 4)] * 10
        feats, events = run(self.machine(), reads)
        self.assertEqual(events, [])
        self.assertTrue(all(f.state is PossessionState.HOME for f in feats))

    def test_dead_resets_and_reestablishment_is_not_a_turnover(self):
        reads = [(HOME, 4)] * 5 + ["dead"] * 5 + [(AWAY, 17)] * 10
        feats, events = run(self.machine(), reads)
        self.assertEqual(events, [])  # throw-ins/goal kicks never anchor counters
        self.assertIs(feats[7].state, PossessionState.DEAD)
        self.assertIs(feats[12].state, PossessionState.AWAY)

    def test_missing_ball_persists_state(self):
        reads = [(HOME, 4)] * 5 + [None] * 10
        feats, events = run(self.machine(), reads)
        self.assertTrue(all(f.state is PossessionState.HOME for f in feats[5:]))

    def test_contested_keeps_pending_alive(self):
        # AWAY 0.8s, contested 0.4s, AWAY 0.8s: pending clock started at first
        # AWAY touch, so the flip confirms 2.0s after it despite the contest
        reads = [(HOME, 4)] * 5 + [(AWAY, 17)] * 4 + ["contested"] * 2 + [(AWAY, 17)] * 9
        feats, events = run(self.machine(), reads)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][1].started_at_s, 5 * DT)
        self.assertIs(feats[10].state, PossessionState.CONTESTED)

    def test_segment_reset_clears_belief(self):
        m = self.machine()
        run(m, [(HOME, 4)] * 10)
        m.segment_reset()
        feats, events = run(m, [(AWAY, 17)] * 10, t0=100.0)
        self.assertEqual(events, [])  # re-establishment, not a turnover
        self.assertTrue(all(f.state is PossessionState.AWAY for f in feats))

    def test_holder_reported_during_pending(self):
        reads = [(HOME, 4)] * 5 + [(AWAY, 17)] * 3
        feats, _ = run(self.machine(), reads)
        # batch reports the pending holder's track id even before confirmation
        self.assertEqual(feats[6].holder_track_id, 17)
        self.assertIs(feats[6].state, PossessionState.HOME)


if __name__ == "__main__":
    unittest.main()
