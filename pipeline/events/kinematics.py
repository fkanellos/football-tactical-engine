"""Kinematic layer for event inference: resampled, smoothed ball + player tracks.

This is features.py stages 1-2 WITHOUT stages 3+ (possession, shape, pressure):
event inference must not depend on possession inference, because the design makes
possession a downstream consumer of the event stream (design doc §3, §8). It keeps
per-player tracks exposed instead of collapsing them into team aggregates.

Grid alignment contract: ``resample_hz`` / ``max_gap_s`` default to the feature
layer's values so event timestamps land on the same grid as ``FrameFeatures``
timestamps. Ball smoothing is deliberately SHORTER than the feature layer's (kicks
are transients; trends are the other layer's business) — ball speeds here therefore
differ slightly from ``FrameFeatures.ball.speed``, by design.

The segmentation loop is re-implemented from features.py on purpose (the flagged §8
refactor extracts one shared resampling core); the smoothing/derivative primitives
are shared imports so the math cannot drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Shared smoothing math with the feature layer — same functions, not copies. The §8
# refactor promotes these to a shared module; until then this import is the seam.
from ..patterns.features import _derivative, _moving_average
from ..patterns.tracking import MatchMeta, MatchTracking, Role, TeamSide


@dataclass
class KinematicsConfig:
    """Tunables; provenance in design doc §10."""

    resample_hz: float = 5.0            # must match FeatureExtractorConfig (grid alignment)
    max_gap_s: float = 1.0              # frame gaps longer than this split segments
    ball_smoothing_window_s: float = 0.4    # short: event inference needs transients
    player_smoothing_window_s: float = 1.0  # matches the feature layer
    boundary_noise_m: float = 0.5       # 1-sigma position noise near lines (measure debt)

    # ball launch (kick) detection — design doc §3
    min_speed_jump_ms: float = 1.8      # per grid step; ~9 m/s^2 at 5 Hz (Link & Hoernig
                                        # use >= 4 m/s^2; ours higher: smoothing smears steps)
    min_launch_speed_ms: float = 4.5    # engine-original: below it, kick vs carry is noise
    redirect_angle_deg: float = 50.0    # velocity direction change that counts as a touch
    redirect_min_speed_ms: float = 4.0
    kicker_radius_m: float = 2.5        # attribution radius (possession radius + margin)
    kicker_lookback_s: float = 0.4
    launch_cooldown_frames: int = 2     # suppress duplicate launches from smearing


@dataclass
class PlayerTrack:
    """One track's smoothed positions/velocities on a segment's grid."""

    track_id: int
    team: Optional[TeamSide]
    role: Role
    xs: List[Optional[float]]
    ys: List[Optional[float]]
    vxs: List[Optional[float]] = field(default_factory=list)
    vys: List[Optional[float]] = field(default_factory=list)

    def pos(self, i: int) -> Optional[Tuple[float, float]]:
        x, y = self.xs[i], self.ys[i]
        return None if x is None or y is None else (x, y)

    def speed(self, i: int) -> Optional[float]:
        vx, vy = self.vxs[i], self.vys[i]
        return None if vx is None or vy is None else math.hypot(vx, vy)


@dataclass
class KinematicSegment:
    """One continuous tracking span, resampled and smoothed."""

    segment_id: int
    period: int
    grid: List[float]
    players: List[PlayerTrack]
    ball_x: List[Optional[float]]
    ball_y: List[Optional[float]]
    ball_vx: List[Optional[float]] = field(default_factory=list)
    ball_vy: List[Optional[float]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.grid)

    def ball_pos(self, i: int) -> Optional[Tuple[float, float]]:
        x, y = self.ball_x[i], self.ball_y[i]
        return None if x is None or y is None else (x, y)

    def ball_speed(self, i: int) -> Optional[float]:
        vx, vy = self.ball_vx[i], self.ball_vy[i]
        return None if vx is None or vy is None else math.hypot(vx, vy)

    def visible_players(
        self, i: int, team: Optional[TeamSide] = None,
        roles: Tuple[Role, ...] = (Role.OUTFIELD, Role.GOALKEEPER),
    ) -> List[Tuple[PlayerTrack, float, float]]:
        out = []
        for tr in self.players:
            if tr.role not in roles or tr.team is None:
                continue
            if team is not None and tr.team is not team:
                continue
            p = tr.pos(i)
            if p is not None:
                out.append((tr, p[0], p[1]))
        return out

    def nearest_player(
        self, i: int, x: float, y: float, team: Optional[TeamSide] = None,
        max_dist_m: Optional[float] = None,
    ) -> Optional[Tuple[PlayerTrack, float]]:
        best: Optional[Tuple[PlayerTrack, float]] = None
        for tr, px, py in self.visible_players(i, team):
            d = math.hypot(px - x, py - y)
            if best is None or d < best[1]:
                best = (tr, d)
        if best is None or (max_dist_m is not None and best[1] > max_dist_m):
            return None
        return best

    def players_within(
        self, i: int, x: float, y: float, radius_m: float,
        team: Optional[TeamSide] = None,
    ) -> List[Tuple[PlayerTrack, float]]:
        out = []
        for tr, px, py in self.visible_players(i, team):
            d = math.hypot(px - x, py - y)
            if d <= radius_m:
                out.append((tr, d))
        return out


class KinematicSeries:
    """All segments of one match, time-ordered, plus the match metadata."""

    def __init__(self, meta: MatchMeta, segments: Sequence[KinematicSegment],
                 config: Optional[KinematicsConfig] = None):
        self.meta = meta
        self.segments: List[KinematicSegment] = list(segments)
        self.config = config or KinematicsConfig()

    def frame_interval_s(self) -> float:
        return 1.0 / self.config.resample_hz


@dataclass(frozen=True)
class BallLaunch:
    """One detected ball launch (kick / redirect) — the shared Tier 2 primitive."""

    segment_id: int
    index: int
    t_s: float
    x: float
    y: float
    speed: float
    vx: float
    vy: float
    kind: str  # "speed_jump" | "redirect"
    kicker_track_id: Optional[int]
    kicker_team: Optional[TeamSide]
    kicker_dist_m: Optional[float]


def detect_launches(seg: KinematicSegment, config: KinematicsConfig) -> List[BallLaunch]:
    """All ball launches in one segment, in time order.

    A launch is a speed step >= ``min_speed_jump_ms`` landing at >=
    ``min_launch_speed_ms``, or a velocity redirect >= ``redirect_angle_deg`` at
    speed. Attribution: nearest player within ``kicker_radius_m`` across the
    ``kicker_lookback_s`` frames up to the launch (the ball has already left the
    foot by the launch frame at 5 Hz).
    """
    launches: List[BallLaunch] = []
    lookback = max(1, int(round(config.kicker_lookback_s * config.resample_hz)))
    skip_until = -1
    for i in range(1, len(seg)):
        if i <= skip_until:
            continue
        s_prev, s = seg.ball_speed(i - 1), seg.ball_speed(i)
        if s is None:
            continue
        jump = (
            s_prev is not None
            and (s - s_prev) >= config.min_speed_jump_ms
            and s >= config.min_launch_speed_ms
        )
        redirect = False
        if not jump and s_prev is not None and s_prev >= config.redirect_min_speed_ms \
                and s >= config.redirect_min_speed_ms:
            vx0, vy0 = seg.ball_vx[i - 1], seg.ball_vy[i - 1]
            vx1, vy1 = seg.ball_vx[i], seg.ball_vy[i]
            if None not in (vx0, vy0, vx1, vy1):
                dot = vx0 * vx1 + vy0 * vy1
                n0, n1 = math.hypot(vx0, vy0), math.hypot(vx1, vy1)
                if n0 > 0 and n1 > 0:
                    cos = max(-1.0, min(1.0, dot / (n0 * n1)))
                    redirect = math.degrees(math.acos(cos)) >= config.redirect_angle_deg
        if not jump and not redirect:
            continue
        pos = seg.ball_pos(i)
        vx, vy = seg.ball_vx[i], seg.ball_vy[i]
        if pos is None or vx is None or vy is None:
            continue
        kicker: Optional[Tuple[PlayerTrack, float]] = None
        for j in range(max(0, i - lookback), i + 1):
            bp = seg.ball_pos(j)
            if bp is None:
                continue
            near = seg.nearest_player(j, bp[0], bp[1], max_dist_m=config.kicker_radius_m)
            if near is not None and (kicker is None or near[1] < kicker[1]):
                kicker = near
        launches.append(
            BallLaunch(
                segment_id=seg.segment_id,
                index=i,
                t_s=seg.grid[i],
                x=pos[0],
                y=pos[1],
                speed=s,
                vx=vx,
                vy=vy,
                kind="speed_jump" if jump else "redirect",
                kicker_track_id=kicker[0].track_id if kicker else None,
                kicker_team=kicker[0].team if kicker else None,
                kicker_dist_m=kicker[1] if kicker else None,
            )
        )
        skip_until = i + config.launch_cooldown_frames
    return launches


class KinematicExtractor:
    """MatchTracking -> KinematicSeries (features.py stages 1-2, tracks exposed)."""

    def __init__(self, config: Optional[KinematicsConfig] = None):
        self.config = config or KinematicsConfig()

    def extract(self, match: MatchTracking) -> KinematicSeries:
        cfg = self.config
        frames = sorted(match.frames, key=lambda f: f.timestamp_s)
        if not frames:
            return KinematicSeries(match.meta, [], cfg)
        dt = 1.0 / cfg.resample_hz

        raw_segments = [[frames[0]]]
        for prev, cur in zip(frames, frames[1:]):
            if (cur.timestamp_s - prev.timestamp_s) > cfg.max_gap_s or cur.period != prev.period:
                raw_segments.append([])
            raw_segments[-1].append(cur)

        w_ball = self._odd_window(cfg.ball_smoothing_window_s)
        w_player = self._odd_window(cfg.player_smoothing_window_s)

        segments: List[KinematicSegment] = []
        for seg_id, seg_frames in enumerate(raw_segments):
            t0, t1 = seg_frames[0].timestamp_s, seg_frames[-1].timestamp_s
            n_steps = max(1, int(round((t1 - t0) / dt)) + 1)
            grid = [t0 + i * dt for i in range(n_steps)]

            obs: Dict[int, List[Tuple[float, float, float]]] = {}
            traits: Dict[int, Tuple[Optional[TeamSide], Role]] = {}
            for fr in seg_frames:
                for p in fr.players:
                    obs.setdefault(p.track_id, []).append((fr.timestamp_s, p.x, p.y))
                    traits[p.track_id] = (p.team, p.role)

            players = []
            for tid, series in obs.items():
                xs = self._sample_channel(grid, [(t, x) for t, x, _ in series])
                ys = self._sample_channel(grid, [(t, y) for t, _, y in series])
                xs = _moving_average(xs, w_player)
                ys = _moving_average(ys, w_player)
                players.append(
                    PlayerTrack(
                        track_id=tid,
                        team=traits[tid][0],
                        role=traits[tid][1],
                        xs=xs,
                        ys=ys,
                        vxs=_derivative(xs, dt),
                        vys=_derivative(ys, dt),
                    )
                )

            ball_obs = [(fr.timestamp_s, fr.ball.x, fr.ball.y) for fr in seg_frames if fr.ball]
            bx = self._sample_channel(grid, [(t, x) for t, x, _ in ball_obs])
            by = self._sample_channel(grid, [(t, y) for t, _, y in ball_obs])
            bx = _moving_average(bx, w_ball)
            by = _moving_average(by, w_ball)
            segments.append(
                KinematicSegment(
                    segment_id=seg_id,
                    period=seg_frames[0].period,
                    grid=grid,
                    players=players,
                    ball_x=bx,
                    ball_y=by,
                    ball_vx=_derivative(bx, dt),
                    ball_vy=_derivative(by, dt),
                )
            )
        return KinematicSeries(match.meta, segments, cfg)

    def _odd_window(self, window_s: float) -> int:
        w = max(1, int(round(window_s * self.config.resample_hz)))
        return w + 1 if w % 2 == 0 else w

    def _sample_channel(
        self, grid: List[float], obs: List[Tuple[float, float]]
    ) -> List[Optional[float]]:
        """Nearest obs within dt/2, else linear interpolation across gaps <=
        max_gap_s, else None. Same semantics as the feature layer's sampler."""
        cfg = self.config
        dt = 1.0 / cfg.resample_hz
        out: List[Optional[float]] = [None] * len(grid)
        if not obs:
            return out
        obs = sorted(obs)
        times = [t for t, _ in obs]
        j = 0
        for i, g in enumerate(grid):
            while j < len(obs) - 1 and times[j + 1] <= g:
                j += 1
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
                if (t_next - t_prev) <= cfg.max_gap_s + 1e-9:
                    w = (g - t_prev) / (t_next - t_prev)
                    out[i] = v_prev + w * (v_next - v_prev)
        return out
