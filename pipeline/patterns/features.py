"""Feature extraction layer: raw tracking frames -> per-frame tactical features.

This layer is the contract between raw data and the detectors. It owns:

- resampling to a uniform time grid and smoothing (positions, then velocities);
- segmentation into continuous spans (broadcast cuts => gaps; period changes also
  split segments);
- the team-relative coordinate frame (own goal x'=0, opponent goal x'=105,
  y'>0 = attacking team's left);
- possession inference (the state machine below — the most consequential and most
  error-prone derived signal, see design doc §2.2);
- every named signal in the catalog (design doc §2).

Detectors consume ``FeatureSeries`` and nothing else. When detectors are later
replaced by learned models (design doc §3.8), this layer is unchanged: a
``FeatureSeries`` is already "a time-indexed feature matrix".

Naming convention for feature paths (used by ``PatternDetector.required_features``
and ``FeatureSeries.values``): dotted attribute paths on ``FrameFeatures``, e.g.
``"ball.x"``, ``"possession.state"``, ``"home.def_line_height"``. The pseudo-prefix
``"team."`` in a detector's declaration means "needed for both teams".

Implementation notes (pure stdlib on purpose — no numpy until /pipeline grows a
real dependency manifest; at 5 Hz x 90 min the volumes are trivial):

- Possession state is reset at segment boundaries: carrying a belief across a
  broadcast cut is guesswork, and detectors cannot span segments anyway.
- A DEAD spell (ball out of bounds) also resets possession; the first clear holder
  after a restart establishes possession WITHOUT a turnover event, so throw-ins
  and goal kicks do not create counter-attack anchors (documented simplification).
- When the ball is missing (not detected), possession state persists — we do not
  flip possession on frames where we cannot see the ball.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import median, pstdev
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .tracking import MatchMeta, MatchTracking, Role, TeamSide, TrackingFrame


class PossessionState(Enum):
    HOME = "home"
    AWAY = "away"
    CONTESTED = "contested"
    DEAD = "dead"  # restarts / ball out of play, to the extent we can infer it

    @staticmethod
    def for_side(side: TeamSide) -> "PossessionState":
        return PossessionState.HOME if side is TeamSide.HOME else PossessionState.AWAY

    def side(self) -> Optional[TeamSide]:
        if self is PossessionState.HOME:
            return TeamSide.HOME
        if self is PossessionState.AWAY:
            return TeamSide.AWAY
        return None


class Lane(Enum):
    """Five vertical lanes of 13.6 m, named from the attacking team's perspective."""

    WIDE_LEFT = "wide_left"
    HALFSPACE_LEFT = "halfspace_left"
    CENTRE = "centre"
    HALFSPACE_RIGHT = "halfspace_right"
    WIDE_RIGHT = "wide_right"


_LANE_HALFSPACE_EDGE = 6.8   # |y'| <= 6.8 -> centre lane
_LANE_WIDE_EDGE = 20.4       # |y'| > 20.4 -> wide lane


def lane_for_y_rel(y_rel: float) -> Lane:
    """Lane for a team-relative y' (positive = attacking team's left)."""
    if y_rel > _LANE_WIDE_EDGE:
        return Lane.WIDE_LEFT
    if y_rel > _LANE_HALFSPACE_EDGE:
        return Lane.HALFSPACE_LEFT
    if y_rel >= -_LANE_HALFSPACE_EDGE:
        return Lane.CENTRE
    if y_rel >= -_LANE_WIDE_EDGE:
        return Lane.HALFSPACE_RIGHT
    return Lane.WIDE_RIGHT


def to_team_relative(x: float, y: float, direction: int) -> Tuple[float, float]:
    """Absolute pitch coords -> team-relative (x' from own goal, y' to attacking left)."""
    return direction * x + 52.5, direction * y


@dataclass
class TeamFrameFeatures:
    """Shape/pressure signals for one team in one frame.

    All x-quantities are in the TEAM-RELATIVE frame: distance from this team's own
    goal line, in metres (0 = own goal, 105 = opponent goal). ``centroid_y`` is
    positive toward this team's attacking-perspective left.

    Everything is computed from VISIBLE outfield players only; ``n_visible_outfield``
    must always be consulted before trusting shape values. Fields are ``None`` when
    not computable in this frame (too few visible players, or not applicable —
    pressure fields are only populated for the out-of-possession team, and
    ``local_superiority_10m`` only for the team in possession).
    """

    side: TeamSide
    n_visible_outfield: int = 0

    # -- shape --------------------------------------------------------------
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None
    width: Optional[float] = None            # max-min y spread of visible outfielders
    depth: Optional[float] = None            # max-min x spread of visible outfielders
    hull_area: Optional[float] = None        # convex hull area, m^2
    stretch_index: Optional[float] = None    # mean distance to own centroid

    # -- defensive line -----------------------------------------------------
    def_line_height: Optional[float] = None      # 2nd-deepest outfielder, GK excluded
    def_line_flatness: Optional[float] = None    # std of x' of deepest 4 outfielders
    def_line_velocity: Optional[float] = None    # smoothed d/dt of def_line_height
    n_behind_ball: Optional[int] = None

    # -- movement -------------------------------------------------------------
    n_forward_runners: int = 0  # visible outfielders with team-relative forward
                                # velocity >= FeatureExtractorConfig.runner_speed_ms

    # -- pressure on the ball (out-of-possession team only) ------------------
    nearest_defender_dist: Optional[float] = None
    defenders_within_5m: Optional[int] = None
    defenders_within_15m: Optional[int] = None
    press_closing_speed: Optional[float] = None  # +ve = collapsing on the ball, m/s

    # -- occupancy ------------------------------------------------------------
    lane_occupancy: Dict[Lane, int] = field(default_factory=dict)
    local_superiority_10m: Optional[int] = None  # (teammates incl. holder - opponents)
                                                 # within 10m of ball; possessing team only


@dataclass
class BallFeatures:
    """Smoothed ball kinematics in ABSOLUTE pitch coordinates.

    ``valid`` is False when the ball was missing/unreliable around this frame; all
    other fields are then ``None`` and must not be interpolated through.
    """

    valid: bool = False
    x: Optional[float] = None
    y: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    speed: Optional[float] = None

    def x_rel(self, team: TeamSide, meta: MatchMeta, period: int) -> Optional[float]:
        """Ball distance from ``team``'s own goal line (team-relative x')."""
        if not self.valid or self.x is None:
            return None
        return meta.attack_direction(team, period) * self.x + 52.5

    def y_rel(self, team: TeamSide, meta: MatchMeta, period: int) -> Optional[float]:
        """Ball y' from ``team``'s attacking perspective (positive = their left)."""
        if not self.valid or self.y is None:
            return None
        return meta.attack_direction(team, period) * self.y

    def vx_rel(self, team: TeamSide, meta: MatchMeta, period: int) -> Optional[float]:
        """Ball velocity along ``team``'s attacking direction (+ = toward opponent)."""
        if not self.valid or self.vx is None:
            return None
        return meta.attack_direction(team, period) * self.vx

    def lane(self, team: TeamSide, meta: MatchMeta, period: int) -> Optional[Lane]:
        """Vertical lane the ball occupies, from ``team``'s attacking perspective."""
        y_rel = self.y_rel(team, meta, period)
        return None if y_rel is None else lane_for_y_rel(y_rel)


@dataclass
class PossessionFeatures:
    """Output of the possession state machine for one frame."""

    state: PossessionState = PossessionState.CONTESTED
    holder_track_id: Optional[int] = None
    time_since_turnover_s: Optional[float] = None
    turnover_won_by: Optional[TeamSide] = None  # set on the (backdated) flip frame only


@dataclass
class QualityFeatures:
    """Per-frame data-quality signals; multiply into every detector's confidence."""

    n_visible_home: int = 0
    n_visible_away: int = 0
    ball_valid: bool = False
    score: float = 0.0  # overall in [0, 1]


@dataclass
class FrameFeatures:
    """The full per-frame feature vector — the only thing detectors ever see."""

    timestamp_s: float
    period: int
    segment_id: int  # continuous-span index; episodes never cross segments
    home: TeamFrameFeatures = field(default_factory=lambda: TeamFrameFeatures(TeamSide.HOME))
    away: TeamFrameFeatures = field(default_factory=lambda: TeamFrameFeatures(TeamSide.AWAY))
    ball: BallFeatures = field(default_factory=BallFeatures)
    possession: PossessionFeatures = field(default_factory=PossessionFeatures)
    quality: QualityFeatures = field(default_factory=QualityFeatures)

    def team(self, side: TeamSide) -> TeamFrameFeatures:
        return self.home if side is TeamSide.HOME else self.away


class FeatureSeries:
    """Time-ordered sequence of ``FrameFeatures`` with windowing helpers.

    This is the "shared time-windowed feature representation" of the design doc:
    detectors slice it, iterate it, and pull columns out of it, but never mutate it.
    """

    def __init__(self, meta: MatchMeta, frames: Sequence[FrameFeatures]):
        self.meta = meta
        self.frames: List[FrameFeatures] = list(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def times(self) -> List[float]:
        """Timestamps of all frames, seconds."""
        return [f.timestamp_s for f in self.frames]

    def values(self, path: str) -> List[Optional[Any]]:
        """Extract one feature column by dotted path, e.g. ``"home.def_line_height"``.

        Enum-valued features are returned as the enum members. Missing values stay
        ``None`` — callers decide how to handle gaps.
        """
        parts = path.split(".")
        out: List[Optional[Any]] = []
        for frame in self.frames:
            node: Any = frame
            for part in parts:
                node = getattr(node, part, None)
                if node is None:
                    break
            out.append(node)
        return out

    def window(self, t0: float, t1: float) -> "FeatureSeries":
        """Sub-series with ``t0 <= timestamp_s < t1``."""
        return FeatureSeries(
            self.meta, [f for f in self.frames if t0 <= f.timestamp_s < t1]
        )

    def segments(self) -> Iterator["FeatureSeries"]:
        """Iterate the continuous spans (one sub-series per ``segment_id``)."""
        group: List[FrameFeatures] = []
        for frame in self.frames:
            if group and frame.segment_id != group[-1].segment_id:
                yield FeatureSeries(self.meta, group)
                group = []
            group.append(frame)
        if group:
            yield FeatureSeries(self.meta, group)

    def sliding_windows(self, length_s: float, stride_s: float) -> Iterator["FeatureSeries"]:
        """Sliding windows within segments (never spanning a segment boundary)."""
        for seg in self.segments():
            if not seg.frames:
                continue
            start = seg.frames[0].timestamp_s
            end = seg.frames[-1].timestamp_s
            t = start
            while t + length_s <= end + 1e-9:
                yield seg.window(t, t + length_s)
                t += stride_s

    def frame_interval_s(self) -> float:
        """Median inter-frame interval (the resample grid step)."""
        deltas = [
            b.timestamp_s - a.timestamp_s
            for a, b in zip(self.frames, self.frames[1:])
            if a.segment_id == b.segment_id
        ]
        return median(deltas) if deltas else 0.2

    def observed_seconds(self) -> float:
        """Total covered time (sum of segment durations), for per-90 normalisation."""
        dt = self.frame_interval_s()
        total = 0.0
        for seg in self.segments():
            if seg.frames:
                total += (seg.frames[-1].timestamp_s - seg.frames[0].timestamp_s) + dt
        return total


@dataclass
class FeatureExtractorConfig:
    """Tunables for normalisation and possession inference.

    Threshold provenance is tracked in the design doc's sourcing appendix; radii
    that define *signals* (5m/15m pressure counts, 10m superiority, runner speed)
    live here rather than in detector configs because they are part of the feature
    definitions themselves.
    """

    resample_hz: float = 5.0
    max_gap_s: float = 1.0          # gaps longer than this split segments
    smoothing_window_s: float = 1.0
    min_visible_for_shape: int = 5  # fewer visible outfielders => shape fields None

    # possession state machine
    possession_radius_m: float = 2.0    # holder = nearest player within this radius
    contested_radius_m: float = 2.0     # opponent also within => CONTESTED
    turnover_persistence_s: float = 2.0 # a flip must survive this long to count
    out_of_bounds_margin_m: float = 0.5

    # signal-definition constants
    runner_speed_ms: float = 4.0  # forward-run threshold for n_forward_runners


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Centered moving average over contiguous non-None runs (edges shrink)."""
    if window <= 1:
        return list(values)
    half = window // 2
    out: List[Optional[float]] = [None] * len(values)
    for i, v in enumerate(values):
        if v is None:
            continue
        acc, n = 0.0, 0
        for j in range(max(0, i - half), min(len(values), i + half + 1)):
            vj = values[j]
            if vj is None:
                # stay inside the contiguous run around i
                if j < i:
                    acc, n = 0.0, 0
                    continue
                break
            acc += vj
            n += 1
        out[i] = acc / n if n else None
    return out


def _derivative(values: List[Optional[float]], dt: float) -> List[Optional[float]]:
    """Central-difference derivative; one-sided at run edges; None propagates."""
    n = len(values)
    out: List[Optional[float]] = [None] * n
    for i in range(n):
        if values[i] is None:
            continue
        prev_v = values[i - 1] if i > 0 else None
        next_v = values[i + 1] if i < n - 1 else None
        if prev_v is not None and next_v is not None:
            out[i] = (next_v - prev_v) / (2 * dt)
        elif next_v is not None:
            out[i] = (next_v - values[i]) / dt
        elif prev_v is not None:
            out[i] = (values[i] - prev_v) / dt
    return out


def _convex_hull_area(points: List[Tuple[float, float]]) -> float:
    """Convex hull area via Andrew's monotone chain + shoelace. 0 for < 3 points."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


@dataclass
class _Track:
    """One track's positions resampled onto a segment's grid (internal)."""

    track_id: int
    team: Optional[TeamSide]
    role: Role
    xs: List[Optional[float]]
    ys: List[Optional[float]]
    vxs: List[Optional[float]] = field(default_factory=list)
    vys: List[Optional[float]] = field(default_factory=list)


@dataclass
class _SegmentData:
    """All resampled per-segment state handed between extraction stages."""

    segment_id: int
    period: int
    grid: List[float]
    tracks: List[_Track]
    ball_x: List[Optional[float]]
    ball_y: List[Optional[float]]
    ball_vx: List[Optional[float]] = field(default_factory=list)
    ball_vy: List[Optional[float]] = field(default_factory=list)


class FeatureExtractor:
    """Turns a ``MatchTracking`` into a ``FeatureSeries``. See module docstring."""

    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        self.config = config or FeatureExtractorConfig()

    # -- public ---------------------------------------------------------------

    def extract(self, match: MatchTracking) -> FeatureSeries:
        all_features: List[FrameFeatures] = []
        for seg in self._resample_and_segment(match):
            self._smooth_and_differentiate(seg)
            possession = self._infer_possession(seg, match.meta)
            frames = self._assemble_frames(seg, possession, match.meta)
            self._post_derivatives(seg, frames)
            self._quality(frames)
            all_features.extend(frames)
        return FeatureSeries(match.meta, all_features)

    # -- stage 1: resample + segment -------------------------------------------

    def _resample_and_segment(self, match: MatchTracking) -> List[_SegmentData]:
        frames = sorted(match.frames, key=lambda f: f.timestamp_s)
        if not frames:
            return []
        dt = 1.0 / self.config.resample_hz

        raw_segments: List[List[TrackingFrame]] = [[frames[0]]]
        for prev, cur in zip(frames, frames[1:]):
            if (cur.timestamp_s - prev.timestamp_s) > self.config.max_gap_s or cur.period != prev.period:
                raw_segments.append([])
            raw_segments[-1].append(cur)

        segments: List[_SegmentData] = []
        for seg_id, seg_frames in enumerate(raw_segments):
            t0, t1 = seg_frames[0].timestamp_s, seg_frames[-1].timestamp_s
            n_steps = max(1, int(round((t1 - t0) / dt)) + 1)
            grid = [t0 + i * dt for i in range(n_steps)]

            # gather per-track observations
            obs: Dict[int, List[Tuple[float, float, float]]] = {}
            traits: Dict[int, Tuple[Optional[TeamSide], Role]] = {}
            for fr in seg_frames:
                for p in fr.players:
                    obs.setdefault(p.track_id, []).append((fr.timestamp_s, p.x, p.y))
                    traits[p.track_id] = (p.team, p.role)  # last wins; fine for v1

            tracks = [
                _Track(
                    track_id=tid,
                    team=traits[tid][0],
                    role=traits[tid][1],
                    xs=self._sample_channel(grid, [(t, x) for t, x, _ in series]),
                    ys=self._sample_channel(grid, [(t, y) for t, _, y in series]),
                )
                for tid, series in obs.items()
            ]

            ball_obs = [
                (fr.timestamp_s, fr.ball.x, fr.ball.y) for fr in seg_frames if fr.ball is not None
            ]
            segments.append(
                _SegmentData(
                    segment_id=seg_id,
                    period=seg_frames[0].period,
                    grid=grid,
                    tracks=tracks,
                    ball_x=self._sample_channel(grid, [(t, x) for t, x, _ in ball_obs]),
                    ball_y=self._sample_channel(grid, [(t, y) for t, _, y in ball_obs]),
                )
            )
        return segments

    def _sample_channel(
        self, grid: List[float], obs: List[Tuple[float, float]]
    ) -> List[Optional[float]]:
        """Sample one scalar channel onto the grid: nearest obs within dt/2, else
        linear interpolation across gaps <= max_gap_s, else None."""
        dt = 1.0 / self.config.resample_hz
        out: List[Optional[float]] = [None] * len(grid)
        if not obs:
            return out
        obs = sorted(obs)
        times = [t for t, _ in obs]
        j = 0
        for i, g in enumerate(grid):
            while j < len(obs) - 1 and times[j + 1] <= g:
                j += 1
            # j = last obs with time <= g (or 0)
            candidates = []
            if times[j] <= g or j == 0:
                candidates.append(j)
            if j + 1 < len(obs):
                candidates.append(j + 1)
            nearest = min(candidates, key=lambda k: abs(times[k] - g))
            if abs(times[nearest] - g) <= dt / 2 + 1e-9:
                out[i] = obs[nearest][1]
                continue
            if times[j] <= g and j + 1 < len(obs):
                t_prev, v_prev = obs[j]
                t_next, v_next = obs[j + 1]
                if (t_next - t_prev) <= self.config.max_gap_s + 1e-9:
                    w = (g - t_prev) / (t_next - t_prev)
                    out[i] = v_prev + w * (v_next - v_prev)
        return out

    # -- stage 2: smooth + velocities -------------------------------------------

    def _smooth_and_differentiate(self, seg: _SegmentData) -> None:
        dt = 1.0 / self.config.resample_hz
        w = max(1, int(round(self.config.smoothing_window_s * self.config.resample_hz)))
        if w % 2 == 0:
            w += 1
        for tr in seg.tracks:
            tr.xs = _moving_average(tr.xs, w)
            tr.ys = _moving_average(tr.ys, w)
            tr.vxs = _derivative(tr.xs, dt)
            tr.vys = _derivative(tr.ys, dt)
        seg.ball_x = _moving_average(seg.ball_x, w)
        seg.ball_y = _moving_average(seg.ball_y, w)
        seg.ball_vx = _derivative(seg.ball_x, dt)
        seg.ball_vy = _derivative(seg.ball_y, dt)

    # -- stage 3: possession -----------------------------------------------------

    def _infer_possession(self, seg: _SegmentData, meta: MatchMeta) -> List[PossessionFeatures]:
        cfg = self.config
        n = len(seg.grid)
        out = [PossessionFeatures() for _ in range(n)]

        half_x = meta.pitch_length_m / 2 + cfg.out_of_bounds_margin_m
        half_y = meta.pitch_width_m / 2 + cfg.out_of_bounds_margin_m

        current: Optional[TeamSide] = None
        last_flip_t: Optional[float] = None
        pending: Optional[Tuple[TeamSide, int]] = None  # (side, start_index)

        # raw per-frame read: "dead" | "contested" | (side, holder_id) | None
        raws: List[Any] = []
        for i in range(n):
            bx, by = seg.ball_x[i], seg.ball_y[i]
            if bx is None or by is None:
                raws.append(None)
                continue
            if abs(bx) > half_x or abs(by) > half_y:
                raws.append("dead")
                continue
            best: Dict[TeamSide, Tuple[float, int]] = {}
            for tr in seg.tracks:
                if tr.team is None or tr.role not in (Role.OUTFIELD, Role.GOALKEEPER):
                    continue
                x, y = tr.xs[i], tr.ys[i]
                if x is None or y is None:
                    continue
                d = math.hypot(x - bx, y - by)
                if tr.team not in best or d < best[tr.team][0]:
                    best[tr.team] = (d, tr.track_id)
            home_d = best[TeamSide.HOME][0] if TeamSide.HOME in best else math.inf
            away_d = best[TeamSide.AWAY][0] if TeamSide.AWAY in best else math.inf
            home_holds = home_d <= cfg.possession_radius_m
            away_holds = away_d <= cfg.possession_radius_m
            if home_holds and away_holds:
                raws.append("contested")
            elif home_holds and away_d <= cfg.contested_radius_m:
                raws.append("contested")
            elif away_holds and home_d <= cfg.contested_radius_m:
                raws.append("contested")
            elif home_holds:
                raws.append((TeamSide.HOME, best[TeamSide.HOME][1]))
            elif away_holds:
                raws.append((TeamSide.AWAY, best[TeamSide.AWAY][1]))
            else:
                raws.append(None)

        for i, raw in enumerate(raws):
            t = seg.grid[i]
            if raw == "dead":
                out[i].state = PossessionState.DEAD
                current, pending = None, None  # restart re-establishes possession
                continue

            if isinstance(raw, tuple):
                side, holder = raw
                out[i].holder_track_id = holder
                if current is None:
                    current = side  # first establishment: no turnover event
                    pending = None
                elif side is current:
                    pending = None
                else:
                    if pending is None or pending[0] is not side:
                        pending = (side, i)
                    if t - seg.grid[pending[1]] >= cfg.turnover_persistence_s - 1e-9:
                        # confirmed flip: backdate to pending start
                        start = pending[1]
                        current = side
                        last_flip_t = seg.grid[start]
                        out[start].turnover_won_by = side
                        for k in range(start, i + 1):
                            if out[k].state is not PossessionState.DEAD:
                                out[k].state = PossessionState.for_side(side)
                                out[k].time_since_turnover_s = seg.grid[k] - last_flip_t
                        pending = None
            elif raw == "contested":
                out[i].state = PossessionState.CONTESTED
                out[i].time_since_turnover_s = (
                    None if last_flip_t is None else t - last_flip_t
                )
                continue
            # raw is None (ball missing / no holder): state persists; pending kept.

            if current is None:
                out[i].state = PossessionState.CONTESTED
            else:
                out[i].state = PossessionState.for_side(current)
                out[i].time_since_turnover_s = (
                    None if last_flip_t is None else t - last_flip_t
                )
        return out

    # -- stage 4: per-frame assembly ----------------------------------------------

    def _assemble_frames(
        self, seg: _SegmentData, possession: List[PossessionFeatures], meta: MatchMeta
    ) -> List[FrameFeatures]:
        cfg = self.config
        frames: List[FrameFeatures] = []
        for i, t in enumerate(seg.grid):
            bx, by = seg.ball_x[i], seg.ball_y[i]
            ball = BallFeatures(valid=(bx is not None and by is not None))
            if ball.valid:
                ball.x, ball.y = bx, by
                ball.vx, ball.vy = seg.ball_vx[i], seg.ball_vy[i]
                if ball.vx is not None and ball.vy is not None:
                    ball.speed = math.hypot(ball.vx, ball.vy)

            frame = FrameFeatures(
                timestamp_s=t,
                period=seg.period,
                segment_id=seg.segment_id,
                ball=ball,
                possession=possession[i],
            )
            for side in (TeamSide.HOME, TeamSide.AWAY):
                frame.team(side).side = side
                self._fill_team_shape(frame, seg, i, side, meta)
            self._fill_pressure_and_occupancy(frame, seg, i, meta)
            frames.append(frame)
        return frames

    def _visible_outfielders(
        self, seg: _SegmentData, i: int, side: TeamSide
    ) -> List[Tuple[_Track, float, float]]:
        out = []
        for tr in seg.tracks:
            if tr.team is side and tr.role is Role.OUTFIELD:
                x, y = tr.xs[i], tr.ys[i]
                if x is not None and y is not None:
                    out.append((tr, x, y))
        return out

    def _fill_team_shape(
        self, frame: FrameFeatures, seg: _SegmentData, i: int, side: TeamSide, meta: MatchMeta
    ) -> None:
        cfg = self.config
        tf = frame.team(side)
        direction = meta.attack_direction(side, seg.period)
        visible = self._visible_outfielders(seg, i, side)
        tf.n_visible_outfield = len(visible)

        rel = [to_team_relative(x, y, direction) for _, x, y in visible]

        # lane occupancy + forward runners are counts: computed at any visibility
        occ: Dict[Lane, int] = {lane: 0 for lane in Lane}
        for _, y_rel in rel:
            occ[lane_for_y_rel(y_rel)] += 1
        tf.lane_occupancy = occ
        tf.n_forward_runners = sum(
            1
            for tr, _, _ in visible
            if tr.vxs[i] is not None and direction * tr.vxs[i] >= cfg.runner_speed_ms
        )

        if len(visible) < cfg.min_visible_for_shape:
            return

        xs_rel = sorted(xr for xr, _ in rel)
        ys_rel = [yr for _, yr in rel]
        tf.centroid_x = sum(xr for xr, _ in rel) / len(rel)
        tf.centroid_y = sum(ys_rel) / len(ys_rel)
        tf.width = max(ys_rel) - min(ys_rel)
        tf.depth = xs_rel[-1] - xs_rel[0]
        tf.hull_area = _convex_hull_area(rel)
        tf.stretch_index = sum(
            math.hypot(xr - tf.centroid_x, yr - tf.centroid_y) for xr, yr in rel
        ) / len(rel)
        tf.def_line_height = xs_rel[1]  # 2nd deepest (deepest is often a straggler/error)
        if len(xs_rel) >= 4:
            tf.def_line_flatness = pstdev(xs_rel[:4])

        if frame.ball.valid:
            ball_x_rel = frame.ball.x_rel(side, meta, seg.period)
            if ball_x_rel is not None:
                tf.n_behind_ball = sum(1 for xr, _ in rel if xr < ball_x_rel)

    def _fill_pressure_and_occupancy(
        self, frame: FrameFeatures, seg: _SegmentData, i: int, meta: MatchMeta
    ) -> None:
        possession_side = frame.possession.state.side()
        if possession_side is None or not frame.ball.valid:
            return
        defending = possession_side.opponent()
        bx, by = frame.ball.x, frame.ball.y

        def dists(side: TeamSide) -> List[float]:
            return sorted(
                math.hypot(x - bx, y - by)
                for _, x, y in self._visible_outfielders(seg, i, side)
            )

        d_def = dists(defending)
        tf_def = frame.team(defending)
        if d_def:
            tf_def.nearest_defender_dist = d_def[0]
            tf_def.defenders_within_5m = sum(1 for d in d_def if d <= 5.0)
            tf_def.defenders_within_15m = sum(1 for d in d_def if d <= 15.0)

        d_att = dists(possession_side)
        frame.team(possession_side).local_superiority_10m = sum(
            1 for d in d_att if d <= 10.0
        ) - sum(1 for d in d_def if d <= 10.0)

    # -- stage 5: cross-frame derivatives ------------------------------------------

    def _post_derivatives(self, seg: _SegmentData, frames: List[FrameFeatures]) -> None:
        """def_line_velocity and press_closing_speed need the assembled time series."""
        dt = 1.0 / self.config.resample_hz
        w = max(1, int(round(self.config.smoothing_window_s * self.config.resample_hz)))
        if w % 2 == 0:
            w += 1
        for side in (TeamSide.HOME, TeamSide.AWAY):
            line = [f.team(side).def_line_height for f in frames]
            vel = _moving_average(_derivative(line, dt), w)
            for f, v in zip(frames, vel):
                f.team(side).def_line_velocity = v

            # mean distance of 3 nearest defenders to ball, where this side defends
            mean3: List[Optional[float]] = [None] * len(frames)
            for idx, f in enumerate(frames):
                if (
                    f.possession.state.side() is None
                    or f.possession.state.side() is side
                    or not f.ball.valid
                ):
                    continue
                ds = sorted(
                    math.hypot(x - f.ball.x, y - f.ball.y)
                    for _, x, y in self._visible_outfielders(seg, idx, side)
                )[:3]
                if ds:
                    mean3[idx] = sum(ds) / len(ds)
            closing = _moving_average(_derivative(mean3, dt), w)
            for f, c in zip(frames, closing):
                f.team(side).press_closing_speed = None if c is None else -c

    # -- stage 6: quality ------------------------------------------------------------

    def _quality(self, frames: List[FrameFeatures]) -> None:
        for f in frames:
            nh = f.home.n_visible_outfield
            na = f.away.n_visible_outfield
            f.quality.n_visible_home = nh
            f.quality.n_visible_away = na
            f.quality.ball_valid = f.ball.valid
            score = (min(nh, 8) / 8.0) * (min(na, 8) / 8.0)
            if not f.ball.valid:
                score *= 0.5
            f.quality.score = max(0.0, min(1.0, score))
