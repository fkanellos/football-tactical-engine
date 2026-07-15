"""Synthetic tracking-data generator: textbook tactical scenarios as MatchTracking.

Each scenario function builds a full 22-player match snippet whose geometry and
kinematics encode ONE tactical story (plus deliberately-negative variants). These
pin the semantics of the detectors before any real tracking data exists (design
doc §6): if a detector fires on its positive scenario and stays silent on the
negatives, its heuristic means what the design doc says it means.

Conventions:
- HOME attacks +x in period 1 throughout (meta.home_attacks_positive_x = {1: True}).
- Players are scripted as (time, x', y') waypoints in their OWN team-relative
  frame (own goal x'=0, opponent goal x'=105, y'>0 = attacking left), linearly
  interpolated; the builder converts to absolute pitch coordinates.
- The ball is scripted in HOME-relative coordinates via ``hrel`` (converted to
  absolute), usually attached to whichever scripted player "has" it.
- Track ids: HOME GK=100, outfield 101-110; AWAY GK=200, outfield 201-210.
- Everything is deterministic — no randomness, so test failures are debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..tracking import (
    BallObservation,
    MatchMeta,
    MatchTracking,
    PlayerObservation,
    Role,
    TeamSide,
    TrackingFrame,
)

Waypoint = Tuple[float, float, float]  # (t_seconds, x_rel, y_rel)


@dataclass
class Script:
    """One player's scripted movement in their own team-relative frame."""

    track_id: int
    team: TeamSide
    role: Role
    waypoints: List[Waypoint]


def hrel(x_rel: float, y_rel: float) -> Tuple[float, float]:
    """HOME-relative -> absolute pitch coordinates (HOME attacks +x)."""
    return x_rel - 52.5, y_rel


def _to_abs(team: TeamSide, x_rel: float, y_rel: float) -> Tuple[float, float]:
    if team is TeamSide.HOME:
        return x_rel - 52.5, y_rel
    return 52.5 - x_rel, -y_rel


def _lerp(waypoints: Sequence[Waypoint], t: float) -> Tuple[float, float]:
    if t <= waypoints[0][0]:
        return waypoints[0][1], waypoints[0][2]
    for (t0, x0, y0), (t1, x1, y1) in zip(waypoints, waypoints[1:]):
        if t0 <= t <= t1:
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + w * (x1 - x0), y0 + w * (y1 - y0)
    return waypoints[-1][1], waypoints[-1][2]


def static(track_id: int, team: TeamSide, role: Role, x_rel: float, y_rel: float) -> Script:
    return Script(track_id, team, role, [(0.0, x_rel, y_rel)])


def formation_positions(center_x: float, width: float = 44.0, depth: float = 30.0) -> List[Tuple[float, float]]:
    """Ten outfield team-relative positions in a 4-4-2 around ``center_x``."""
    ys4 = [-width / 2, -width / 6, width / 6, width / 2]
    positions = [(center_x - depth / 2, y) for y in ys4]
    positions += [(center_x, y) for y in ys4]
    positions += [(center_x + depth / 2, y) for y in (-8.0, 8.0)]
    return positions


def team_442(
    team: TeamSide, center_x: float, width: float = 44.0, depth: float = 30.0
) -> List[Script]:
    """A full static team: GK + 4-4-2 outfield block centred at ``center_x``."""
    base = 100 if team is TeamSide.HOME else 200
    scripts = [static(base, team, Role.GOALKEEPER, 6.0, 0.0)]
    for i, (x, y) in enumerate(formation_positions(center_x, width, depth)):
        scripts.append(static(base + 1 + i, team, Role.OUTFIELD, x, y))
    return scripts


def build_match(
    match_id: str,
    duration_s: float,
    scripts: List[Script],
    ball_waypoints_abs: List[Waypoint],  # (t, x_abs, y_abs)
    hz: float = 5.0,
    hidden_tracks: Optional[Set[int]] = None,
) -> MatchTracking:
    """Sample all scripts at ``hz`` into a single-period MatchTracking.

    ``hidden_tracks`` simulates broadcast partial observability: those track ids
    are omitted from every frame (as if permanently off-camera).
    """
    hidden = hidden_tracks or set()
    dt = 1.0 / hz
    n = int(round(duration_s * hz)) + 1
    frames: List[TrackingFrame] = []
    for i in range(n):
        t = i * dt
        players = []
        for s in scripts:
            if s.track_id in hidden:
                continue
            x_rel, y_rel = _lerp(s.waypoints, t)
            x, y = _to_abs(s.team, x_rel, y_rel)
            players.append(
                PlayerObservation(track_id=s.track_id, team=s.team, role=s.role, x=x, y=y)
            )
        bx, by = _lerp([(t0, x, y) for t0, x, y in ball_waypoints_abs], t)
        frames.append(
            TrackingFrame(
                frame_index=i,
                timestamp_s=t,
                period=1,
                players=tuple(players),
                ball=BallObservation(x=bx, y=by),
            )
        )
    meta = MatchMeta(match_id=match_id, home_attacks_positive_x={1: True})
    return MatchTracking(meta=meta, frames=frames)


# ---------------------------------------------------------------------------
# High press
# ---------------------------------------------------------------------------

def high_press_scenario(hidden_tracks: Optional[Set[int]] = None) -> MatchTracking:
    """HOME presses AWAY's deep build-up, textbook.

    Story (20s): AWAY centre-back holds the ball 12m from AWAY's own goal
    (HOME-relative x'=93). HOME holds a high line (back four 48m from own goal).
    From t=5s, five HOME pressers converge on the ball from 12-16m out at
    ~1.1 m/s, the nearest settling ~2.8m off the ball (never inside the 2m
    possession radius, so possession stays cleanly AWAY).

    Expected: one high_press event for HOME, roughly t=10..16, confidence >= 0.5.
    With ``hidden_tracks`` hiding everything but four pressers (close-up camera),
    expected: nothing, or only low-confidence (< 0.5) events.
    """
    ball_abs = hrel(93.0, 0.0)

    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 8.0, 0.0)]
    # HOME high back four (defensive line 48m) + one holding mid
    for tid, y in zip((101, 102, 103, 104), (-18.0, -6.0, 6.0, 18.0)):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, 48.0, y))
    scripts.append(static(105, TeamSide.HOME, Role.OUTFIELD, 70.0, 0.0))
    # Five pressers: start on rings 12-16m from the ball, converge at ~1.1 m/s
    # from t=5s along straight lines toward the ball, stopping short of 2.6m.
    press_starts = [
        (106, 93.0 - 12.0, 0.0),     # straight up the middle, ends nearest (~2.8m)
        (107, 93.0 - 9.2, 9.0),      # ~12.9m out
        (108, 93.0 - 9.2, -9.0),
        (109, 93.0 - 4.0, 14.5),     # ~15m out
        (110, 93.0 - 4.0, -14.5),
    ]
    for tid, x0, y0 in press_starts:
        dx, dy = 93.0 - x0, 0.0 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        stop = 2.8 if tid == 106 else 4.5
        travel = dist - stop
        arrive = 5.0 + travel / 1.1
        frac = travel / dist
        scripts.append(
            Script(
                tid,
                TeamSide.HOME,
                Role.OUTFIELD,
                [(0.0, x0, y0), (5.0, x0, y0), (arrive, x0 + frac * dx, y0 + frac * dy),
                 (20.0, x0 + frac * dx, y0 + frac * dy)],
            )
        )
    # AWAY in build-up shape; CB 201 on the ball (AWAY-relative x'=12).
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 5.0, 0.0))
    scripts.append(static(201, TeamSide.AWAY, Role.OUTFIELD, 12.0, 0.0))
    for tid, (x, y) in zip(
        range(202, 211),
        [(12.0, -16.0), (12.0, 16.0), (16.0, -28.0), (16.0, 28.0),
         (30.0, -10.0), (30.0, 10.0), (45.0, -18.0), (45.0, 18.0), (55.0, 0.0)],
    ):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, x, y))

    return build_match(
        "synthetic-high-press",
        20.0,
        scripts,
        [(0.0, ball_abs[0], ball_abs[1])],
        hidden_tracks=hidden_tracks,
    )


def high_press_low_visibility_scenario() -> MatchTracking:
    """The press above, but the camera close-up hides HOME's line and most pressers:
    only four HOME outfielders are visible. Expected: no confident detection."""
    return high_press_scenario(hidden_tracks={100, 101, 102, 103, 104, 105, 106})


def passive_buildup_scenario() -> MatchTracking:
    """AWAY builds up deep but HOME sits off: same geometry as the press scenario
    minus any converging movement (all HOME players static, nearest 12m away).
    Expected: no high_press events."""
    ball_abs = hrel(93.0, 0.0)
    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 8.0, 0.0)]
    for tid, y in zip((101, 102, 103, 104), (-18.0, -6.0, 6.0, 18.0)):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, 48.0, y))
    scripts.append(static(105, TeamSide.HOME, Role.OUTFIELD, 70.0, 0.0))
    for tid, (x, y) in zip(
        (106, 107, 108, 109, 110),
        [(81.0, 0.0), (83.8, 9.0), (83.8, -9.0), (89.0, 14.5), (89.0, -14.5)],
    ):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, x, y))
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 5.0, 0.0))
    scripts.append(static(201, TeamSide.AWAY, Role.OUTFIELD, 12.0, 0.0))
    for tid, (x, y) in zip(
        range(202, 211),
        [(12.0, -16.0), (12.0, 16.0), (16.0, -28.0), (16.0, 28.0),
         (30.0, -10.0), (30.0, 10.0), (45.0, -18.0), (45.0, 18.0), (55.0, 0.0)],
    ):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, x, y))
    return build_match(
        "synthetic-passive-buildup", 20.0, scripts, [(0.0, ball_abs[0], ball_abs[1])]
    )


# ---------------------------------------------------------------------------
# Low block
# ---------------------------------------------------------------------------

def low_block_scenario(block_duration_s: float = 30.0) -> MatchTracking:
    """HOME drops into a deep, compact block while AWAY circulates in front of it.

    Story: 10s of mid shape (line 38m — should NOT read as a low block), then
    HOME sinks over 2s into a 32m-wide block with the back four 12m from goal
    and holds it for ``block_duration_s`` while the AWAY holder circulates
    laterally ~40m from the HOME goal.

    Expected (default 30s hold): one low_block event for HOME covering the deep
    phase. With ``block_duration_s=8``: no event (a low block is a state, not a
    moment — min duration).
    """
    t0, t1 = 10.0, 12.0
    end = t1 + block_duration_s

    def sink(tid: int, mid: Tuple[float, float], deep: Tuple[float, float]) -> Script:
        return Script(
            tid, TeamSide.HOME, Role.OUTFIELD,
            [(0.0, mid[0], mid[1]), (t0, mid[0], mid[1]), (t1, deep[0], deep[1]),
             (end, deep[0], deep[1])],
        )

    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 4.0, 0.0)]
    mids_y4 = (-16.0, -5.0, 5.0, 16.0)
    for tid, y in zip((101, 102, 103, 104), mids_y4):        # back four 38 -> 12
        scripts.append(sink(tid, (38.0, y * 1.25), (12.0, y)))
    for tid, y in zip((105, 106, 107, 108), mids_y4):        # midfield 50 -> 20
        scripts.append(sink(tid, (50.0, y * 1.25), (20.0, y)))
    for tid, y in zip((109, 110), (-7.0, 7.0)):              # strikers 62 -> 30
        scripts.append(sink(tid, (62.0, y), (30.0, y)))

    # AWAY in sustained possession: holder 210 carries the ball laterally across
    # the pitch ~40m from HOME's goal (AWAY-relative x'=65).
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 8.0, 0.0))
    for tid, (x, y) in zip(
        range(201, 210),
        [(35.0, -15.0), (35.0, 15.0), (48.0, -25.0), (48.0, 25.0), (48.0, 0.0),
         (58.0, -12.0), (58.0, 12.0), (68.0, -20.0), (68.0, 20.0)],
    ):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, x, y))
    holder_wps = []
    t = 0.0
    y_cycle = [8.0, -8.0]
    k = 0
    while t <= end + 6.0:
        holder_wps.append((t, 65.0, y_cycle[k % 2]))
        t += 6.0
        k += 1
    scripts.append(Script(210, TeamSide.AWAY, Role.OUTFIELD, holder_wps))

    ball_wps = [(t, *_to_abs(TeamSide.AWAY, x, y)) for t, x, y in holder_wps]
    return build_match(f"synthetic-low-block-{int(block_duration_s)}s", end, scripts, ball_wps)


def mid_block_scenario() -> MatchTracking:
    """HOME defends in a mid block (line ~38m) the whole time. Expected: no
    low_block events — deep is part of the definition."""
    end = 40.0
    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 4.0, 0.0)]
    mids_y4 = (-20.0, -6.5, 6.5, 20.0)
    for tid, y in zip((101, 102, 103, 104), mids_y4):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, 38.0, y))
    for tid, y in zip((105, 106, 107, 108), mids_y4):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, 50.0, y))
    for tid, y in zip((109, 110), (-7.0, 7.0)):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, 62.0, y))
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 8.0, 0.0))
    for tid, (x, y) in zip(
        range(201, 210),
        [(35.0, -15.0), (35.0, 15.0), (48.0, -25.0), (48.0, 25.0), (48.0, 0.0),
         (58.0, -12.0), (58.0, 12.0), (68.0, -20.0), (68.0, 20.0)],
    ):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, x, y))
    holder_wps = [(0.0, 40.0, 8.0), (6.0, 40.0, -8.0), (12.0, 40.0, 8.0),
                  (18.0, 40.0, -8.0), (24.0, 40.0, 8.0), (30.0, 40.0, -8.0),
                  (36.0, 40.0, 8.0), (42.0, 40.0, -8.0)]
    scripts.append(Script(210, TeamSide.AWAY, Role.OUTFIELD, holder_wps))
    ball_wps = [(t, *_to_abs(TeamSide.AWAY, x, y)) for t, x, y in holder_wps]
    return build_match("synthetic-mid-block", end, scripts, ball_wps)


# ---------------------------------------------------------------------------
# Flank overload
# ---------------------------------------------------------------------------

def flank_overload_scenario() -> MatchTracking:
    """HOME overloads its left flank in the final third.

    Story: 8s of balanced central possession (negative window), then the ball
    goes to the left winger high and wide (x'=75, y'=+19) and four HOME players
    occupy the left wide/half-space lanes against two AWAY defenders, with
    HOME's centroid leaning ~6m left. Held for 8s, then back to balanced.

    Expected: one flank_overload event for HOME with metadata side="left"
    covering roughly t=8..16, and no "right" events.
    """
    end = 22.0

    def phased(tid: int, bal: Tuple[float, float], over: Tuple[float, float]) -> Script:
        return Script(
            tid, TeamSide.HOME, Role.OUTFIELD,
            [(0.0, *bal), (7.0, *bal), (8.5, *over), (16.0, *over), (17.5, *bal), (end, *bal)],
        )

    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 6.0, 0.0)]
    # the left-side cluster: winger (holder in phase B), fullback, cm, striker
    scripts.append(phased(101, (62.0, 14.0), (75.0, 19.0)))   # LW
    scripts.append(phased(102, (48.0, 18.0), (70.0, 24.0)))   # LB overlapping
    scripts.append(phased(103, (55.0, 6.0), (70.0, 13.0)))    # CM joining
    scripts.append(phased(104, (68.0, 2.0), (78.0, 11.0)))    # ST drifting
    # the rest: mild right/centre balance
    for tid, (x, y) in zip(
        (105, 106, 107, 108, 109, 110),
        [(55.0, 2.0), (48.0, -3.0), (55.0, 3.0), (48.0, -6.0), (62.0, 0.0), (62.0, -2.0)],
    ):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, x, y))

    # AWAY defending: two defenders near the overload zone, rest spread
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 5.0, 0.0))
    away_positions_abs = [
        (20.0, 22.0), (17.0, 12.0),                       # the two near the flank
        (25.0, 0.0), (22.0, -10.0), (28.0, -18.0),
        (35.0, 8.0), (35.0, -8.0), (42.0, 15.0), (42.0, -15.0), (45.0, 0.0),
    ]
    for tid, (ax, ay) in zip(range(201, 211), away_positions_abs):
        # convert absolute -> AWAY-relative for scripting
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, 52.5 - ax, -ay))

    # ball: with CM 105-ish centrally, then with the LW during the overload
    ball_wps = [
        (0.0, *hrel(55.0, 2.0)), (7.0, *hrel(55.0, 2.0)),
        (8.5, *hrel(75.0, 19.0)), (16.0, *hrel(75.0, 19.0)),
        (17.5, *hrel(55.0, 2.0)), (end, *hrel(55.0, 2.0)),
    ]
    return build_match("synthetic-flank-overload", end, scripts, ball_wps)


def balanced_attack_scenario() -> MatchTracking:
    """HOME attacks through the centre with balanced occupation the whole time.
    Expected: no flank_overload events."""
    end = 20.0
    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 6.0, 0.0)]
    for tid, (x, y) in zip(
        range(101, 111),
        [(62.0, 14.0), (48.0, 18.0), (55.0, 6.0), (68.0, 2.0), (55.0, 2.0),
         (48.0, -3.0), (55.0, -6.0), (48.0, -18.0), (62.0, -14.0), (68.0, -2.0)],
    ):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, x, y))
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 5.0, 0.0))
    for tid, (ax, ay) in zip(
        range(201, 211),
        [(20.0, 22.0), (17.0, 12.0), (25.0, 0.0), (22.0, -10.0), (28.0, -18.0),
         (35.0, 8.0), (35.0, -8.0), (42.0, 15.0), (42.0, -15.0), (45.0, 0.0)],
    ):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, 52.5 - ax, -ay))
    ball_wps = [(0.0, *hrel(55.0, 2.0))]
    return build_match("synthetic-balanced-attack", end, scripts, ball_wps)


# ---------------------------------------------------------------------------
# Offside trap / line step-up
# ---------------------------------------------------------------------------

def line_step_scenario(step: bool = True) -> MatchTracking:
    """HOME's flat back four steps up in unison as AWAY prepares a through ball.

    Story (15s): AWAY holder 22.5m from HOME's line-side goal-... concretely: the
    AWAY attacking midfielder holds the ball 30m from HOME's goal. HOME's back
    four sits flat at 20m. At t=6s the four step to 24.5m over 1.5s (3 m/s, in
    unison, staying flat); at t=8s the through ball is struck in behind (14 m/s
    toward HOME's goal).

    Expected (``step=True``): one offside_trap event for HOME around t=6-8.
    With ``step=False`` the back four DROPS from 20m to 14m instead (retreat):
    expected no events.
    """
    end = 15.0
    target = 24.5 if step else 14.0

    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 5.0, 0.0)]
    for tid, y in zip((101, 102, 103, 104), (-12.0, -4.0, 4.0, 12.0)):
        scripts.append(
            Script(
                tid, TeamSide.HOME, Role.OUTFIELD,
                [(0.0, 20.0, y), (6.0, 20.0, y), (7.5, target, y), (end, target, y)],
            )
        )
    for tid, (x, y) in zip(
        (105, 106, 107, 108), [(38.0, -10.0), (38.0, 10.0), (45.0, -3.0), (45.0, 3.0)]
    ):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, x, y))
    scripts.append(static(109, TeamSide.HOME, Role.OUTFIELD, 55.0, -6.0))
    scripts.append(static(110, TeamSide.HOME, Role.OUTFIELD, 55.0, 6.0))

    # AWAY: holder (AM, track 208) at HOME-relative x'=30; a striker on the line.
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 6.0, 0.0))
    away_abs = [
        (-32.5 + 1.5, -8.0),   # striker hovering on the HOME line
        (-10.0, -15.0), (-10.0, 15.0), (0.0, -5.0), (0.0, 5.0),
        (10.0, -20.0), (10.0, 20.0), (-22.5, 0.0),  # 208 = holder
        (15.0, 0.0), (25.0, 0.0),
    ]
    for tid, (ax, ay) in zip(range(201, 211), away_abs):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, 52.5 - ax, -ay))

    ball_wps = [
        (0.0, -22.5, 0.0), (8.0, -22.5, 0.0), (9.5, -43.5, 0.0), (end, -43.5, 0.0)
    ]
    return build_match(
        "synthetic-line-step" if step else "synthetic-line-drop", end, scripts, ball_wps
    )


# ---------------------------------------------------------------------------
# Counter-attack
# ---------------------------------------------------------------------------

def counter_attack_scenario(fast_break: bool = True, flicker: bool = False) -> MatchTracking:
    """HOME regains deep and breaks at speed.

    Story (25s): AWAY attacks; their AM (208) holds the ball 25m from HOME's
    goal. At t=8s HOME's CM (105, standing 3m away) intercepts: the ball
    transfers to him, he secures it for 2s (confirming the turnover), then
    drives 40m upfield in 6s flanked by two more HOME runners at ~6-7 m/s.

    Expected (``fast_break=True``): one counter_attack event for HOME anchored
    ~t=8, confidence >= 0.5.
    ``fast_break=False``: after securing, HOME just circulates (+5m in 10s) —
    expected no events.
    ``flicker=True``: the "interception" lasts only 1s before AWAY recovers the
    ball — no confirmed turnover, expected no events (tests possession
    hysteresis end-to-end).
    """
    end = 25.0
    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 5.0, 0.0)]
    for tid, y in zip((101, 102, 103, 104), (-14.0, -5.0, 5.0, 14.0)):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, 15.0, y))

    if flicker:
        cm_wps = [(0.0, 25.0, 5.0), (end, 25.0, 5.0)]
        r1_wps = [(0.0, 30.0, -10.0), (end, 30.0, -10.0)]
        r2_wps = [(0.0, 26.0, 12.0), (end, 26.0, 12.0)]
    elif fast_break:
        cm_wps = [(0.0, 25.0, 5.0), (8.0, 25.0, 5.0), (10.0, 28.0, 4.0),
                  (16.0, 68.0, 0.0), (end, 68.0, 0.0)]
        r1_wps = [(0.0, 30.0, -10.0), (10.0, 30.0, -10.0), (16.0, 70.0, -12.0),
                  (end, 70.0, -12.0)]
        r2_wps = [(0.0, 26.0, 12.0), (10.0, 26.0, 12.0), (16.0, 62.0, 14.0),
                  (end, 62.0, 14.0)]
    else:
        cm_wps = [(0.0, 25.0, 5.0), (8.0, 25.0, 5.0), (10.0, 28.0, 4.0),
                  (20.0, 33.0, 2.0), (end, 33.0, 2.0)]
        r1_wps = [(0.0, 30.0, -10.0), (end, 30.0, -10.0)]
        r2_wps = [(0.0, 26.0, 12.0), (end, 26.0, 12.0)]

    scripts.append(Script(105, TeamSide.HOME, Role.OUTFIELD, cm_wps))
    scripts.append(Script(106, TeamSide.HOME, Role.OUTFIELD, r1_wps))
    scripts.append(Script(107, TeamSide.HOME, Role.OUTFIELD, r2_wps))
    scripts.append(static(108, TeamSide.HOME, Role.OUTFIELD, 20.0, -2.0))
    scripts.append(static(109, TeamSide.HOME, Role.OUTFIELD, 35.0, -18.0))
    scripts.append(static(110, TeamSide.HOME, Role.OUTFIELD, 45.0, 8.0))

    # AWAY committed forward (they were attacking when they lost it)
    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, 6.0, 0.0))
    away_abs = [
        (-20.0, -12.0), (-20.0, 12.0), (-15.0, 0.0), (-5.0, -18.0), (-5.0, 18.0),
        (0.0, -6.0), (0.0, 6.0), (-27.5, 8.0),  # 208 = the AM who loses it
        (12.0, 0.0), (20.0, 0.0),
    ]
    for tid, (ax, ay) in zip(range(201, 211), away_abs):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, 52.5 - ax, -ay))

    # ball: with AWAY 208 (abs -27.5, 8) until t=8, then with HOME 105.
    if flicker:
        ball_wps = [
            (0.0, -27.5, 8.0), (8.0, -27.5, 8.0),
            (8.2, -27.5, 5.0),                      # to HOME 105 (3m away)
            (9.2, -27.5, 5.0), (9.4, -27.5, 8.0),   # AWAY recovers after 1s
            (end, -27.5, 8.0),
        ]
    else:
        cm_ball = [(t, *_to_abs(TeamSide.HOME, x, y)) for t, x, y in cm_wps if t >= 8.0]
        ball_wps = [(0.0, -27.5, 8.0), (7.9, -27.5, 8.0)] + cm_ball
    name = "synthetic-counter" if fast_break else "synthetic-slow-transition"
    if flicker:
        name = "synthetic-flicker-possession"
    return build_match(name, end, scripts, ball_wps)


def slow_transition_scenario() -> MatchTracking:
    """Regain followed by patient circulation — expected: no counter_attack."""
    return counter_attack_scenario(fast_break=False)


def flickering_possession_scenario() -> MatchTracking:
    """A 1s 'interception' that AWAY immediately recovers — no confirmed
    turnover, so no counter window. Expected: no counter_attack events."""
    return counter_attack_scenario(flicker=True)
