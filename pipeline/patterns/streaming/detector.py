"""Streaming detection: episode lifecycle state machines + detector base class.

Two machine shapes cover all five batch detectors (live-architecture design doc
§4):

- ``EpisodeStateMachine`` — for hysteresis-episode detectors (high press, low
  block, flank overload, offside trap events): consumes the SAME per-frame
  scores the batch detector computes, and reproduces ``episodes_from_scores``
  semantics causally. Every batch-visible outcome is identical (parity-tested):
  a CLOSED live instance's (start, end, confidence) equals the batch episode's.
  What live adds is the early lifecycle: PROVISIONAL when evidence has been
  sustained for ``provisional_after_s``, CONFIRMED when the batch bar
  (``min_duration_s``) is met, RETRACTED when a provisional fizzles.

- ``AnchoredWindowMachine`` — for anchor-triggered window patterns (counter
  attack): opens on an external anchor (a confirmed turnover), scores a growing
  window, and resolves when the window expires or possession resolves it.

False-start handling is two-tier by design: a candidate must survive
``provisional_after_s`` before ANYTHING is emitted (sub-second pressure blips
never reach a human), and a provisional that dies emits an explicit RETRACTED
so the UI can undo — the retraction *rate* is a first-class live health metric
(too high => raise ``provisional_after_s``, trading alert earliness for trust).

Priors from opponent scouting profiles (``pipeline.scouting.priors``) plug in
via the ``ConfidencePrior`` protocol. Deliberately narrow contract:
- the prior NEVER touches per-frame scores or episode membership — live episode
  geometry stays identical to batch;
- it may shift the *reported* confidence (bounded, both raw and adjusted are
  always emitted) and scale ``provisional_after_s`` (earlier first alert for
  expected patterns, later for unexpected ones);
- confirmation always additionally requires the RAW confidence to clear an
  absolute floor — a prior can accelerate belief, it cannot fabricate evidence
  (opponent-scouting design doc §3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Protocol, Tuple

from ..features import FrameFeatures
from ..tracking import MatchMeta, TeamSide
from ..detectors.base import episode_confidence
from .events import LiveEventKind, LivePatternUpdate


class ConfidencePrior(Protocol):
    """What a scouting prior is allowed to do to live detection. See module doc."""

    def adjust_confidence(self, raw: float) -> float: ...

    def provisional_after_scale(self) -> float: ...

    def confirm_allowed(self, raw_confidence: float) -> bool: ...

    def describe(self) -> Dict[str, Any]: ...


@dataclass
class StreamingEpisodeConfig:
    """Lifecycle tunables. The first five mirror the batch detector's config
    verbatim (parity requires it); the rest are live-only.

    ``provisional_after_s`` is the earliness/trust dial: how long a candidate
    must sustain before the first (retractable) alert. ``provisional_confidence_cap``
    keeps a 1.5-second screaming-strong candidate from outranking confirmed
    events in the UI.
    """

    enter_threshold: float
    exit_threshold: float
    min_duration_s: float
    merge_gap_s: float
    n_components: int
    provisional_after_s: float = 1.5
    min_update_interval_s: float = 1.0
    provisional_confidence_cap: float = 0.8


class _Phase(Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"


@dataclass
class _OpenInstance:
    """Accumulators for the currently open instance.

    ``scores``/``qualities``/``completes`` hold exactly the frames the batch
    episode span would cover (start .. last active frame): scores drop None
    frames, qualities/completes keep every frame — mirroring how the batch
    detectors call ``episode_confidence`` over a span. Frames arriving while
    dormant sit in ``gap_frames`` and are committed only if the episode resumes
    (i.e. only if batch would have merged across the gap).
    """

    instance_id: str
    start_t: float
    last_active_t: float
    period: int
    provisional_after_s: float
    scores: List[float] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    completes: List[bool] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    gap_frames: List[Tuple[Optional[float], float, bool, Optional[float]]] = field(
        default_factory=list
    )
    dormant: bool = False
    last_emit_t: Optional[float] = None


class EpisodeStateMachine:
    """Causal ``episodes_from_scores`` + the provisional/confirmed lifecycle.

    One machine per (pattern, team). Feed it every frame's (time, score,
    quality, completeness) — score None means "pattern not applicable this
    frame", exactly as in batch — and it emits ``LivePatternUpdate``s.

    Parity contract (tested): for any score series, the set of CLOSED instances
    equals ``episodes_from_scores(...)`` spans, with identical confidence
    (when no prior is attached; a prior changes reported confidence only,
    never spans).
    """

    def __init__(
        self,
        pattern_id: str,
        team: TeamSide,
        config: StreamingEpisodeConfig,
        prior: Optional[ConfidencePrior] = None,
    ):
        self.pattern_id = pattern_id
        self.team = team
        self.config = config
        self.prior = prior
        self._phase = _Phase.IDLE
        self._open: Optional[_OpenInstance] = None
        self._instances_created = 0

    @property
    def phase(self) -> _Phase:
        return self._phase

    # -- public ---------------------------------------------------------------

    def push(
        self,
        t: float,
        score: Optional[float],
        quality: float,
        complete: bool = True,
        intensity: Optional[float] = None,
        period: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[LivePatternUpdate]:
        cfg = self.config
        updates: List[LivePatternUpdate] = []

        # 1. a dormant instance whose merge window has expired can never be
        #    resumed (batch merges only gaps STRICTLY shorter than merge_gap_s).
        if (
            self._open is not None
            and self._open.dormant
            and t - self._open.last_active_t >= cfg.merge_gap_s - 1e-9
        ):
            updates.extend(self._close(reason="score_faded"))

        inst = self._open
        if inst is None:
            if score is not None and score >= cfg.enter_threshold:
                inst = self._open_instance(t, period)
                self._commit_frame(inst, score, quality, complete, intensity, metadata)
                updates.extend(self._transitions(t))
            return updates

        # 2. an instance is open: classify this frame.
        if not inst.dormant:
            if score is not None and score >= cfg.exit_threshold:
                self._commit_frame(inst, score, quality, complete, intensity, metadata)
                inst.last_active_t = t
            else:
                inst.dormant = True
                inst.gap_frames.append((score, quality, complete, intensity))
        else:
            if score is not None and score >= cfg.enter_threshold:
                # resume: batch would merge, so the gap frames join the span
                for g_score, g_quality, g_complete, g_intensity in inst.gap_frames:
                    self._commit_frame(inst, g_score, g_quality, g_complete, g_intensity, None)
                inst.gap_frames.clear()
                inst.dormant = False
                self._commit_frame(inst, score, quality, complete, intensity, metadata)
                inst.last_active_t = t
            else:
                inst.gap_frames.append((score, quality, complete, intensity))

        if not inst.dormant:
            updates.extend(self._transitions(t))
        return updates

    def segment_break(self, reason: str = "broadcast_cut") -> List[LivePatternUpdate]:
        """Broadcast cut: episodes never span segments (batch parity)."""
        return self._close(reason=reason)

    def flush(self, reason: str = "stream_end") -> List[LivePatternUpdate]:
        """End of stream (or of match): resolve whatever is open."""
        return self._close(reason=reason)

    # -- internals --------------------------------------------------------------

    def _open_instance(self, t: float, period: int) -> _OpenInstance:
        scale = self.prior.provisional_after_scale() if self.prior else 1.0
        inst = _OpenInstance(
            instance_id=f"{self.pattern_id}:{self.team.value}:{self._instances_created}",
            start_t=t,
            last_active_t=t,
            period=period,
            provisional_after_s=self.config.provisional_after_s * scale,
        )
        self._instances_created += 1
        self._open = inst
        self._phase = _Phase.CANDIDATE
        return inst

    @staticmethod
    def _commit_frame(
        inst: _OpenInstance,
        score: Optional[float],
        quality: float,
        complete: bool,
        intensity: Optional[float],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if score is not None:
            inst.scores.append(score)
        inst.qualities.append(quality)
        inst.completes.append(complete)
        if intensity is not None:
            inst.intensities.append(intensity)
        if metadata:
            for key, value in metadata.items():
                # first write wins: episode context is defined at onset
                inst.metadata.setdefault(key, value)

    def _confidences(self) -> Tuple[float, float, float]:
        """(uncapped raw, capped raw, prior-adjusted) for the open instance."""
        inst = self._open
        assert inst is not None
        raw = episode_confidence(
            inst.scores, inst.qualities, inst.completes, self.config.n_components
        )
        capped = raw
        if self._phase is not _Phase.CONFIRMED:
            capped = min(raw, self.config.provisional_confidence_cap)
        adjusted = self.prior.adjust_confidence(capped) if self.prior else capped
        return raw, capped, adjusted

    def _transitions(self, t: float) -> List[LivePatternUpdate]:
        cfg = self.config
        inst = self._open
        assert inst is not None
        updates: List[LivePatternUpdate] = []

        if (
            self._phase is _Phase.CANDIDATE
            and t - inst.start_t >= inst.provisional_after_s - 1e-9
        ):
            self._phase = _Phase.PROVISIONAL
            updates.append(self._emit(LiveEventKind.PROVISIONAL, t))

        if self._phase is _Phase.PROVISIONAL and (
            inst.last_active_t - inst.start_t >= cfg.min_duration_s - 1e-9
        ):
            raw, _, _ = self._confidences()
            if self.prior is None or self.prior.confirm_allowed(raw):
                self._phase = _Phase.CONFIRMED
                updates.append(self._emit(LiveEventKind.CONFIRMED, t))

        if (
            not updates
            and self._phase in (_Phase.PROVISIONAL, _Phase.CONFIRMED)
            and (
                inst.last_emit_t is None
                or t - inst.last_emit_t >= cfg.min_update_interval_s - 1e-9
            )
        ):
            updates.append(self._emit(LiveEventKind.UPDATE, t))
        return updates

    def _close(self, reason: str) -> List[LivePatternUpdate]:
        inst = self._open
        if inst is None:
            return []
        updates: List[LivePatternUpdate] = []
        span = inst.last_active_t - inst.start_t
        raw, _, _ = self._confidences()

        # late-confirmation path: the batch bar was met but the lifecycle never
        # got an active frame after crossing it (e.g. cut mid-episode, or
        # provisional_after_s > min_duration_s configs) — batch would count this
        # episode, so live must too.
        if span >= self.config.min_duration_s - 1e-9 and self._phase in (
            _Phase.CANDIDATE,
            _Phase.PROVISIONAL,
        ):
            if self.prior is None or self.prior.confirm_allowed(raw):
                if self._phase is _Phase.CANDIDATE:
                    self._phase = _Phase.PROVISIONAL
                    updates.append(self._emit(LiveEventKind.PROVISIONAL, inst.last_active_t))
                self._phase = _Phase.CONFIRMED
                updates.append(self._emit(LiveEventKind.CONFIRMED, inst.last_active_t))

        if self._phase is _Phase.CONFIRMED:
            updates.append(
                self._emit(
                    LiveEventKind.CLOSED,
                    inst.last_active_t,
                    end_s=inst.last_active_t,
                    reason=None if reason == "score_faded" else reason,
                )
            )
        elif self._phase is _Phase.PROVISIONAL:
            updates.append(
                self._emit(LiveEventKind.RETRACTED, inst.last_active_t, reason=reason)
            )
        # CANDIDATE: silent discard — nobody was ever told.

        self._open = None
        self._phase = _Phase.IDLE
        return updates

    def _emit(
        self,
        kind: LiveEventKind,
        t: float,
        end_s: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> LivePatternUpdate:
        inst = self._open
        assert inst is not None
        _, capped, adjusted = self._confidences()
        span = inst.last_active_t - inst.start_t
        maturity = min(1.0, span / self.config.min_duration_s) if self.config.min_duration_s > 0 else 1.0
        intensity = (
            sum(inst.intensities) / len(inst.intensities) if inst.intensities else None
        )
        inst.last_emit_t = t
        return LivePatternUpdate(
            kind=kind,
            instance_id=inst.instance_id,
            pattern_id=self.pattern_id,
            team=self.team,
            period=inst.period,
            start_s=inst.start_t,
            last_s=inst.last_active_t,
            confidence=adjusted,
            raw_confidence=capped,
            maturity=maturity,
            intensity=intensity,
            end_s=end_s,
            reason=reason,
            metadata=dict(inst.metadata),
            prior=self.prior.describe() if self.prior else None,
        )


# ---------------------------------------------------------------------------
# Anchored-window lifecycle (counter-attack)
# ---------------------------------------------------------------------------


@dataclass
class AnchoredWindowConfig:
    """Mirrors CounterAttackConfig where parity demands it (window, score_min)."""

    window_s: float
    min_score: float
    n_components: int
    provisional_score: float = 0.30  # emit PROVISIONAL when cumulative score first clears this
    min_update_interval_s: float = 1.0


class AnchoredWindowMachine:
    """Lifecycle for anchor-triggered window patterns (counter-attack live).

    The batch detector sees the whole 14 s window at once and keeps the anchor
    iff the final score clears ``score_min``. Live, the window fills up frame by
    frame: the caller opens the machine on a confirmed turnover (backdated
    anchor from ``StreamingPossessionMachine``), pushes the CUMULATIVE
    window-so-far score each frame (peak progression so far, runners so far —
    computed by the streaming counter-attack detector with the same formulas as
    batch), and closes it when the window expires or possession resolves it.

    PROVISIONAL fires the first time the cumulative score clears
    ``provisional_score`` ("a counter is developing"); resolution at close:
    final score >= ``min_score`` -> CONFIRMED + CLOSED, else RETRACTED (or
    silence, if no provisional was ever emitted). There is no merge-gap concept:
    a counter-attack window is one shot.
    """

    def __init__(
        self,
        pattern_id: str,
        team: TeamSide,
        config: AnchoredWindowConfig,
        prior: Optional[ConfidencePrior] = None,
    ):
        self.pattern_id = pattern_id
        self.team = team
        self.config = config
        self.prior = prior
        self._open: Optional[_OpenInstance] = None
        self._provisional = False
        self._instances_created = 0
        self._last_score = 0.0

    @property
    def is_open(self) -> bool:
        return self._open is not None

    def open(self, anchor_t: float, period: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Anchor confirmed (turnover won by our team). No emission yet."""
        if self._open is not None:
            raise RuntimeError("previous window still open; close it first")
        self._open = _OpenInstance(
            instance_id=f"{self.pattern_id}:{self.team.value}:{self._instances_created}",
            start_t=anchor_t,
            last_active_t=anchor_t,
            period=period,
            provisional_after_s=0.0,
            metadata=dict(metadata or {}),
        )
        self._instances_created += 1
        self._provisional = False
        self._last_score = 0.0

    def push(
        self,
        t: float,
        cumulative_score: float,
        quality: float,
        intensity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[LivePatternUpdate]:
        inst = self._open
        if inst is None:
            return []
        cfg = self.config
        if t - inst.start_t > cfg.window_s:
            # this frame falls outside the window: resolve on what was seen
            # inside it (batch stops scanning at the window edge too)
            return self.close(t)

        updates: List[LivePatternUpdate] = []
        inst.qualities.append(quality)
        inst.completes.append(True)
        if intensity is not None:
            inst.intensities.append(intensity)
        if metadata:
            for key, value in metadata.items():
                inst.metadata[key] = value  # window context evolves; last wins here
        inst.last_active_t = t
        self._last_score = cumulative_score

        if not self._provisional and cumulative_score >= cfg.provisional_score:
            self._provisional = True
            updates.append(self._emit(LiveEventKind.PROVISIONAL, t))
        elif self._provisional and (
            inst.last_emit_t is None or t - inst.last_emit_t >= cfg.min_update_interval_s - 1e-9
        ):
            updates.append(self._emit(LiveEventKind.UPDATE, t))
        return updates

    def close(self, t: float, reason: Optional[str] = None) -> List[LivePatternUpdate]:
        """Window over (expired / possession lost / segment break): resolve."""
        inst = self._open
        if inst is None:
            return []
        cfg = self.config
        updates: List[LivePatternUpdate] = []
        raw = episode_confidence(
            [self._last_score], inst.qualities, inst.completes, cfg.n_components
        )
        succeeded = self._last_score >= cfg.min_score and (
            self.prior is None or self.prior.confirm_allowed(raw)
        )
        if succeeded:
            if not self._provisional:
                self._provisional = True
                updates.append(self._emit(LiveEventKind.PROVISIONAL, t))
            updates.append(self._emit(LiveEventKind.CONFIRMED, t))
            updates.append(
                self._emit(LiveEventKind.CLOSED, t, end_s=inst.last_active_t, reason=reason)
            )
        elif self._provisional:
            updates.append(
                self._emit(LiveEventKind.RETRACTED, t, reason=reason or "window_unfulfilled")
            )
        self._open = None
        self._provisional = False
        return updates

    def _emit(
        self,
        kind: LiveEventKind,
        t: float,
        end_s: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> LivePatternUpdate:
        inst = self._open
        assert inst is not None
        cfg = self.config
        raw = episode_confidence(
            [self._last_score], inst.qualities, inst.completes, cfg.n_components
        )
        adjusted = self.prior.adjust_confidence(raw) if self.prior else raw
        elapsed = inst.last_active_t - inst.start_t
        intensity = (
            sum(inst.intensities) / len(inst.intensities) if inst.intensities else None
        )
        inst.last_emit_t = t
        return LivePatternUpdate(
            kind=kind,
            instance_id=inst.instance_id,
            pattern_id=self.pattern_id,
            team=self.team,
            period=inst.period,
            start_s=inst.start_t,
            last_s=inst.last_active_t,
            confidence=adjusted,
            raw_confidence=raw,
            maturity=min(1.0, elapsed / cfg.window_s) if cfg.window_s > 0 else 1.0,
            intensity=intensity,
            end_s=end_s,
            reason=reason,
            metadata=dict(inst.metadata),
            prior=self.prior.describe() if self.prior else None,
        )


# ---------------------------------------------------------------------------
# Detector base class
# ---------------------------------------------------------------------------


class StreamingDetector(ABC):
    """Streaming twin of ``PatternDetector`` for hysteresis-episode motifs.

    Subclasses supply ``frame_score`` — which MUST be the same function the
    batch detector uses (extracted to module level, e.g.
    ``detectors.high_press.frame_score``), not a reimplementation. The base
    class owns the per-team lifecycle machines and the fan-out.

    Anchored-window motifs (counter-attack) do not fit this shape; their
    streaming detector composes ``StreamingPossessionMachine`` anchors with an
    ``AnchoredWindowMachine`` directly (live-architecture design doc §4.3).
    """

    pattern_id: ClassVar[str]
    display_name: ClassVar[str]
    required_features: ClassVar[Tuple[str, ...]]

    def __init__(
        self,
        episode_config: StreamingEpisodeConfig,
        priors: Optional[Mapping[TeamSide, ConfidencePrior]] = None,
    ):
        priors = priors or {}
        self.machines: Dict[TeamSide, EpisodeStateMachine] = {
            side: EpisodeStateMachine(
                self.pattern_id, side, episode_config, prior=priors.get(side)
            )
            for side in (TeamSide.HOME, TeamSide.AWAY)
        }

    @abstractmethod
    def frame_score(
        self, frame: FrameFeatures, meta: MatchMeta, team: TeamSide
    ) -> Tuple[Optional[float], bool, Optional[float], Optional[Dict[str, Any]]]:
        """(score, complete, intensity_hint, metadata) for one frame/team —
        score/complete via the batch detector's shared scoring function."""
        raise NotImplementedError

    def push(self, frame: FrameFeatures, meta: MatchMeta) -> List[LivePatternUpdate]:
        updates: List[LivePatternUpdate] = []
        for side in (TeamSide.HOME, TeamSide.AWAY):
            score, complete, intensity, metadata = self.frame_score(frame, meta, side)
            updates.extend(
                self.machines[side].push(
                    frame.timestamp_s,
                    score,
                    frame.quality.score,
                    complete=complete,
                    intensity=intensity,
                    period=frame.period,
                    metadata=metadata,
                )
            )
        return updates

    def segment_break(self) -> List[LivePatternUpdate]:
        updates: List[LivePatternUpdate] = []
        for machine in self.machines.values():
            updates.extend(machine.segment_break())
        return updates

    def flush(self) -> List[LivePatternUpdate]:
        updates: List[LivePatternUpdate] = []
        for machine in self.machines.values():
            updates.extend(machine.flush())
        return updates
