"""Tests for the live episode lifecycle machines.

The heart of this file is the PARITY suite: for arbitrary score series, the
set of CLOSED live instances must equal ``episodes_from_scores`` spans with
identical confidences — live detection is batch detection plus earliness,
never a different detector.
"""

from __future__ import annotations

import random
import unittest

from pipeline.patterns.detectors.base import episode_confidence, episodes_from_scores
from pipeline.patterns.features import (
    BallFeatures,
    FrameFeatures,
    PossessionFeatures,
    PossessionState,
    QualityFeatures,
    TeamFrameFeatures,
)
from pipeline.patterns.streaming.detector import (
    AnchoredWindowConfig,
    AnchoredWindowMachine,
    EpisodeStateMachine,
    StreamingEpisodeConfig,
)
from pipeline.patterns.streaming.events import LiveEventKind, LivePatternUpdate, StreamStatus
from pipeline.patterns.streaming.high_press import StreamingHighPressDetector
from pipeline.patterns.tracking import MatchMeta, TeamSide

DT = 0.2
K = LiveEventKind


def config(**overrides) -> StreamingEpisodeConfig:
    defaults = dict(
        enter_threshold=0.30,
        exit_threshold=0.15,
        min_duration_s=3.0,
        merge_gap_s=2.0,
        n_components=5,
        provisional_after_s=1.5,
        min_update_interval_s=1.0,
        provisional_confidence_cap=0.8,
    )
    defaults.update(overrides)
    return StreamingEpisodeConfig(**defaults)


def machine(**overrides) -> EpisodeStateMachine:
    return EpisodeStateMachine("high_press", TeamSide.HOME, config(**overrides))


def push_series(m, scores, qualities=None, t0=0.0):
    updates = []
    for i, s in enumerate(scores):
        q = qualities[i] if qualities else 1.0
        updates.extend(m.push(t0 + i * DT, s, q, complete=s is not None))
    return updates


def kinds(updates):
    return [u.kind for u in updates]


class EpisodeLifecycleTest(unittest.TestCase):
    def test_short_blip_is_silent(self):
        # 1.0s of strong score (< provisional_after_s): a candidate that
        # fizzles must never have alerted anyone
        m = machine()
        updates = push_series(m, [0.6] * 5 + [0.0] * 30)
        updates.extend(m.flush())
        self.assertEqual(updates, [])

    def test_provisional_then_retracted(self):
        # 2.0s of score: past provisional (1.5s), short of confirmation (3s)
        m = machine()
        updates = push_series(m, [0.6] * 10 + [0.0] * 25)
        updates.extend(m.flush())
        self.assertEqual(kinds(updates), [K.PROVISIONAL, K.RETRACTED])
        self.assertEqual(updates[1].reason, "score_faded")
        self.assertEqual(updates[0].instance_id, updates[1].instance_id)

    def test_full_lifecycle_order_and_span(self):
        m = machine()
        updates = push_series(m, [0.6] * 20 + [0.0] * 25)  # 4s active
        updates.extend(m.flush())
        ks = kinds(updates)
        self.assertEqual(ks[0], K.PROVISIONAL)
        self.assertIn(K.CONFIRMED, ks)
        self.assertEqual(ks[-1], K.CLOSED)
        self.assertLess(ks.index(K.PROVISIONAL), ks.index(K.CONFIRMED))
        closed = updates[-1]
        self.assertAlmostEqual(closed.start_s, 0.0)
        self.assertAlmostEqual(closed.end_s, 19 * DT)  # last active frame
        self.assertIsNone(closed.reason)  # natural fade = clean close
        # provisional fired at the configured earliness, confirm at the batch bar
        self.assertAlmostEqual(updates[0].last_s, 1.6, places=6)
        confirmed = updates[ks.index(K.CONFIRMED)]
        self.assertAlmostEqual(confirmed.last_s - confirmed.start_s, 3.0, places=6)

    def test_flicker_within_merge_gap_is_one_instance(self):
        # dip of 1s (< merge_gap 2s) must not close/retract/reopen
        scores = [0.6] * 10 + [0.0] * 5 + [0.6] * 10 + [0.0] * 25
        m = machine()
        updates = push_series(m, scores)
        updates.extend(m.flush())
        ids = {u.instance_id for u in updates}
        self.assertEqual(len(ids), 1)
        ks = kinds(updates)
        self.assertEqual(ks.count(K.PROVISIONAL), 1)
        self.assertEqual(ks.count(K.CONFIRMED), 1)
        self.assertNotIn(K.RETRACTED, ks)
        self.assertEqual(ks.count(K.CLOSED), 1)

    def test_gap_beyond_merge_gap_makes_two_instances(self):
        scores = [0.6] * 20 + [0.0] * 15 + [0.6] * 20 + [0.0] * 15
        m = machine()
        updates = push_series(m, scores)
        updates.extend(m.flush())
        closed = [u for u in updates if u.kind is K.CLOSED]
        self.assertEqual(len(closed), 2)
        self.assertNotEqual(closed[0].instance_id, closed[1].instance_id)

    def test_segment_break_retracts_provisional(self):
        m = machine()
        updates = push_series(m, [0.6] * 10)  # 2s: provisional, unconfirmed
        updates.extend(m.segment_break())
        self.assertEqual(kinds(updates), [K.PROVISIONAL, K.RETRACTED])
        self.assertEqual(updates[-1].reason, "broadcast_cut")

    def test_segment_break_closes_confirmed(self):
        m = machine()
        updates = push_series(m, [0.6] * 18)  # 3.4s: confirmed
        updates.extend(m.segment_break())
        self.assertEqual(kinds(updates)[-1], K.CLOSED)
        self.assertEqual(updates[-1].reason, "broadcast_cut")
        self.assertAlmostEqual(updates[-1].end_s, 17 * DT)

    def test_late_confirmation_at_flush(self):
        # provisional_after_s > min_duration_s: episode meets the batch bar
        # while still a candidate; flush must still count it (batch parity)
        m = machine(provisional_after_s=6.0)
        updates = push_series(m, [0.6] * 20)  # 3.8s active, never provisional
        updates.extend(m.flush())
        self.assertEqual(kinds(updates), [K.PROVISIONAL, K.CONFIRMED, K.CLOSED])
        self.assertEqual(updates[-1].reason, "stream_end")

    def test_confidence_capped_while_provisional_only(self):
        m = machine()
        updates = push_series(m, [0.999] * 40 + [0.0] * 25)
        updates.extend(m.flush())
        by_kind = {u.kind: u for u in updates}
        self.assertLessEqual(by_kind[K.PROVISIONAL].raw_confidence, 0.8)
        self.assertGreater(by_kind[K.CLOSED].raw_confidence, 0.9)  # cap lifted

    def test_maturity_grows_and_saturates(self):
        m = machine()
        updates = push_series(m, [0.6] * 40 + [0.0] * 25)
        updates.extend(m.flush())
        maturities = [u.maturity for u in updates]
        self.assertEqual(maturities, sorted(maturities))
        self.assertEqual(maturities[-1], 1.0)
        self.assertLess(updates[0].maturity, 1.0)  # provisional was honest about it

    def test_updates_are_throttled(self):
        m = machine()
        updates = push_series(m, [0.6] * 50)
        stamps = [u.last_s for u in updates if u.kind is K.UPDATE]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        self.assertTrue(all(g >= 1.0 - 1e-9 for g in gaps))
        self.assertGreater(len(stamps), 2)

    def test_metadata_first_write_wins(self):
        m = machine()
        m.push(0.0, 0.6, 1.0, metadata={"trigger": "restart"})
        for i in range(1, 20):
            m.push(i * DT, 0.6, 1.0, metadata={"trigger": "open_play"})
        updates = m.flush()
        self.assertEqual(updates[-1].metadata["trigger"], "restart")


class BatchParityTest(unittest.TestCase):
    """CLOSED live instances == batch episodes, spans and confidences alike."""

    ENTER, EXIT, MIN_DUR, MERGE, N_COMP = 0.30, 0.15, 3.0, 2.0, 5

    def _series(self, seed, n=600):
        rng = random.Random(seed)
        scores, qualities = [], []
        for _ in range(n):
            if rng.random() < 0.10:
                scores.append(None)
            else:
                # bursty: occasional strong stretches over a weak baseline
                scores.append(round(rng.uniform(0.0, 0.65), 3))
            qualities.append(round(rng.uniform(0.3, 1.0), 3))
        return scores, qualities

    def test_parity_across_seeds(self):
        for seed in range(1, 8):
            with self.subTest(seed=seed):
                self._assert_parity(seed)

    def _assert_parity(self, seed):
        scores, qualities = self._series(seed)
        times = [i * DT for i in range(len(scores))]
        completes = [s is not None for s in scores]

        spans = episodes_from_scores(
            scores, times, self.ENTER, self.EXIT, self.MIN_DUR, self.MERGE
        )
        expected = []
        for a, b in spans:
            conf = episode_confidence(
                [s for s in scores[a:b] if s is not None],
                qualities[a:b],
                completes[a:b],
                self.N_COMP,
            )
            expected.append((times[a], times[b - 1], conf))

        m = EpisodeStateMachine(
            "p", TeamSide.HOME,
            StreamingEpisodeConfig(
                enter_threshold=self.ENTER,
                exit_threshold=self.EXIT,
                min_duration_s=self.MIN_DUR,
                merge_gap_s=self.MERGE,
                n_components=self.N_COMP,
            ),
        )
        closed = []
        for i, s in enumerate(scores):
            for u in m.push(times[i], s, qualities[i], complete=completes[i]):
                if u.kind is K.CLOSED:
                    closed.append((u.start_s, u.end_s, u.raw_confidence))
        for u in m.flush():
            if u.kind is K.CLOSED:
                closed.append((u.start_s, u.end_s, u.raw_confidence))

        self.assertEqual(len(closed), len(expected), f"seed {seed}: {closed} vs {expected}")
        for got, want in zip(closed, expected):
            self.assertAlmostEqual(got[0], want[0], places=9)
            self.assertAlmostEqual(got[1], want[1], places=9)
            self.assertAlmostEqual(got[2], want[2], places=9)


class AnchoredWindowTest(unittest.TestCase):
    def cfg(self):
        return AnchoredWindowConfig(
            window_s=14.0, min_score=0.30, n_components=3, provisional_score=0.30
        )

    def awm(self):
        return AnchoredWindowMachine("counter_attack", TeamSide.HOME, self.cfg())

    def test_successful_counter(self):
        m = self.awm()
        m.open(anchor_t=100.0, period=2, metadata={"turnover_location": "middle"})
        updates = []
        for i, score in enumerate([0.05, 0.15, 0.35, 0.5, 0.6]):
            updates.extend(m.push(100.0 + (i + 1), score, quality=0.9))
        updates.extend(m.close(106.0))  # possession resolved it
        ks = kinds(updates)
        self.assertEqual(ks, [K.PROVISIONAL, K.UPDATE, K.UPDATE, K.CONFIRMED, K.CLOSED])
        closed = updates[-1]
        self.assertAlmostEqual(closed.start_s, 100.0)  # backdated turnover anchor
        self.assertEqual(closed.metadata["turnover_location"], "middle")
        self.assertAlmostEqual(
            closed.raw_confidence,
            episode_confidence([0.6], [0.9] * 5, [True] * 5, 3),
        )

    def test_fizzle_after_provisional_retracts(self):
        m = self.awm()
        m.open(0.0, period=1)
        updates = []
        updates.extend(m.push(1.0, 0.4, 1.0))   # provisional
        updates.extend(m.push(2.0, 0.1, 1.0))   # counter broke down (UPDATE shows the drop)
        updates.extend(m.close(3.0))
        self.assertEqual(kinds(updates), [K.PROVISIONAL, K.UPDATE, K.RETRACTED])
        self.assertLess(updates[1].raw_confidence, updates[0].raw_confidence)

    def test_quiet_window_is_silent(self):
        m = self.awm()
        m.open(0.0, period=1)
        updates = []
        for i in range(5):
            updates.extend(m.push(1.0 + i, 0.05, 1.0))
        updates.extend(m.close(7.0))
        self.assertEqual(updates, [])

    def test_window_expiry_resolves_on_inside_frames_only(self):
        m = self.awm()
        m.open(0.0, period=1)
        updates = []
        updates.extend(m.push(5.0, 0.5, 1.0))
        updates.extend(m.push(15.0, 0.9, 1.0))  # outside 14s window: not counted
        ks = kinds(updates)
        self.assertEqual(ks, [K.PROVISIONAL, K.CONFIRMED, K.CLOSED])
        self.assertAlmostEqual(updates[-1].end_s, 5.0)  # last in-window frame
        self.assertFalse(m.is_open)


class StreamingHighPressIntegrationTest(unittest.TestCase):
    """End-to-end: shared batch scorer -> lifecycle machine, on synthetic frames."""

    def _press_frame(self, t: float) -> FrameFeatures:
        home = TeamFrameFeatures(
            side=TeamSide.HOME,
            n_visible_outfield=9,
            def_line_height=50.0,
            defenders_within_15m=5,
            nearest_defender_dist=2.5,
            press_closing_speed=1.5,
        )
        return FrameFeatures(
            timestamp_s=t,
            period=1,
            segment_id=0,
            home=home,
            away=TeamFrameFeatures(side=TeamSide.AWAY, n_visible_outfield=8),
            ball=BallFeatures(valid=True, x=30.0, y=0.0),  # x_rel(HOME)=82.5: deep press zone
            possession=PossessionFeatures(state=PossessionState.AWAY),
            quality=QualityFeatures(n_visible_home=9, n_visible_away=8, ball_valid=True, score=0.9),
        )

    def test_press_lifecycle_for_pressing_team_only(self):
        meta = MatchMeta(match_id="m1", home_attacks_positive_x={1: True})
        det = StreamingHighPressDetector()
        updates = []
        for i in range(25):  # 5s of sustained press
            updates.extend(det.push(self._press_frame(i * DT), meta))
        updates.extend(det.flush())
        self.assertTrue(updates)
        self.assertTrue(all(u.team is TeamSide.HOME for u in updates))
        ks = kinds(updates)
        self.assertIn(K.PROVISIONAL, ks)
        self.assertIn(K.CONFIRMED, ks)
        self.assertEqual(ks[-1], K.CLOSED)
        closed = updates[-1]
        self.assertEqual(closed.metadata["trigger"], "open_play")
        self.assertFalse(closed.metadata["is_counter_press"])
        self.assertIsNotNone(closed.intensity)
        self.assertGreater(closed.raw_confidence, 0.3)


class WireSchemaTest(unittest.TestCase):
    def test_pattern_update_wire_form(self):
        u = LivePatternUpdate(
            kind=LiveEventKind.PROVISIONAL,
            instance_id="high_press:away:3",
            pattern_id="high_press",
            team=TeamSide.AWAY,
            period=2,
            start_s=2705.0,
            last_s=2712.4,
            confidence=0.512,
            raw_confidence=0.431,
            maturity=0.55,
            intensity=0.61,
            metadata={"trigger": "open_play"},
            prior={"applied": True, "shift": 0.42},
        )
        msg = u.to_ws_dict(seq=812)
        self.assertEqual(msg["v"], 1)
        self.assertEqual(msg["seq"], 812)
        self.assertEqual(msg["type"], "pattern")
        self.assertEqual(msg["event"], "provisional")
        inst = msg["instance"]
        self.assertEqual(inst["id"], "high_press:away:3")
        self.assertEqual(inst["team"], "away")
        self.assertEqual(inst["confidence"], 0.512)
        self.assertEqual(inst["raw_confidence"], 0.431)
        self.assertEqual(inst["prior"]["shift"], 0.42)
        self.assertNotIn("end_s", inst)  # only on CLOSED

    def test_stream_status_wire_form(self):
        msg = StreamStatus(
            state="recovering", segment_id=14, match_clock_s=2712.4,
            quality=0.8123, open_instances=2,
        ).to_ws_dict(seq=813)
        self.assertEqual(msg["type"], "stream_status")
        self.assertEqual(msg["state"], "recovering")
        self.assertEqual(msg["quality"], 0.812)


if __name__ == "__main__":
    unittest.main()
