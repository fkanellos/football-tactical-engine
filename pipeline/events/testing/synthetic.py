"""Synthetic tracking fixtures for event inference: one scenario per event story.

Same philosophy as the pattern fixtures (patterns/testing/synthetic.py, whose
Script/waypoint machinery this reuses): each scenario scripts a full 22-player
snippet whose ball kinematics and geometry encode ONE event story plus deliberate
negatives, so the detectors' semantics are pinned before real data exists.

Additional builder capability the event layer needs: ``ball_hidden_spans`` —
windows where the ball is absent from every frame, simulating the tracker losing
the ball (out of frame past the boundary, in the net, during dead time).

Conventions are inherited: HOME attacks +x in period 1; players scripted in their
own team-relative frame; ball scripted in ABSOLUTE coordinates; HOME tracks
100-110, AWAY 200-210; everything deterministic.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from ...patterns.testing.synthetic import (  # shared scenario machinery
    Script,
    Waypoint,
    _lerp,
    _to_abs,
    formation_positions,
    static,
    team_442,
)
from ...patterns.tracking import (
    BallObservation,
    MatchMeta,
    MatchTracking,
    PlayerObservation,
    Role,
    TeamSide,
    TrackingFrame,
)


def build_events_match(
    match_id: str,
    duration_s: float,
    scripts: List[Script],
    ball_waypoints_abs: List[Waypoint],  # (t, x_abs, y_abs)
    hz: float = 5.0,
    ball_hidden_spans: Sequence[Tuple[float, float]] = (),
    hidden_tracks: Optional[Set[int]] = None,
) -> MatchTracking:
    """Like patterns' build_match, plus windows where the ball goes untracked."""
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
        ball = None
        if not any(a <= t <= b for a, b in ball_hidden_spans):
            bx, by = _lerp([(t0, x, y) for t0, x, y in ball_waypoints_abs], t)
            ball = BallObservation(x=bx, y=by)
        frames.append(
            TrackingFrame(frame_index=i, timestamp_s=t, period=1,
                          players=tuple(players), ball=ball)
        )
    meta = MatchMeta(match_id=match_id, home_attacks_positive_x={1: True})
    return MatchTracking(meta=meta, frames=frames)


def replace(scripts: List[Script], *replacements: Script) -> List[Script]:
    """Swap scripted players by track id (customize a team_442 background cast)."""
    ids = {s.track_id for s in replacements}
    return [s for s in scripts if s.track_id not in ids] + list(replacements)


def phased_team(
    team: TeamSide, center_a: float, center_b: float,
    t0: float, t1: float, end: float,
) -> List[Script]:
    """A full team whose 4-4-2 block moves from ``center_a`` to ``center_b``."""
    base = 100 if team is TeamSide.HOME else 200
    scripts = [static(base, team, Role.GOALKEEPER, 6.0, 0.0)]
    for i, ((xa, ya), (xb, yb)) in enumerate(
        zip(formation_positions(center_a), formation_positions(center_b))
    ):
        scripts.append(
            Script(base + 1 + i, team, Role.OUTFIELD,
                   [(0.0, xa, ya), (t0, xa, ya), (t1, xb, yb), (end, xb, yb)])
        )
    return scripts


def jog(track_id: int, team: TeamSide, role: Role, x_rel: float, y_rel: float,
        until: float, end: float, amp: float = 2.0, half_period: float = 2.0) -> Script:
    """A player shuttling around a base point at ~2*amp/half_period m/s, then
    freezing at ``until`` (speeds collapse) and holding to ``end``."""
    waypoints: List[Waypoint] = []
    t, k = 0.0, 0
    while t < until:
        waypoints.append((t, x_rel, y_rel + (amp if k % 2 == 0 else -amp)))
        t += half_period
        k += 1
    waypoints.append((until, x_rel, y_rel))
    waypoints.append((end, x_rel, y_rel))
    return Script(track_id, team, role, waypoints)


# ---------------------------------------------------------------------------
# Tier 1: throw-in / corner / goal kick / kickoff
# ---------------------------------------------------------------------------

def throw_in_scenario(excursion_m: float = 1.5) -> MatchTracking:
    """AWAY's defender puts the ball out over HOME's attacking-left touchline;
    HOME throws in from the crossing point.

    Story (30s): the ball sits by AWAY 201 near the touchline; at t=5 it rolls
    out, ending ``excursion_m`` beyond y=+34; the tracker loses it at 6.6; it
    reappears at rest by the line at t=13, and HOME 105 (who walked over during
    the dead time) throws to HOME 106 at t=15.

    Expected: one THROW_IN for HOME, crossing ~(10.4, 34), resumption observed.
    ``excursion_m=0.3`` is the ambiguous variant: same morphology, exit within
    position noise — still a THROW_IN, at visibly lower confidence.
    """
    scripts = team_442(TeamSide.HOME, 45.0) + team_442(TeamSide.AWAY, 40.0)
    scripts = replace(
        scripts,
        static(201, TeamSide.AWAY, Role.OUTFIELD, 42.5, -32.5),   # abs (10, 32.5)
        Script(105, TeamSide.HOME, Role.OUTFIELD,                 # the thrower walks over
               [(0.0, 62.9, 26.0), (7.0, 62.9, 26.0), (12.0, 62.9, 33.2), (30.0, 62.9, 33.2)]),
        static(106, TeamSide.HOME, Role.OUTFIELD, 66.5, 26.0),    # abs (14, 26): receiver
    )
    out_y = 34.0 + excursion_m
    ball = [
        (0.0, 9.5, 32.0), (5.0, 9.5, 32.0), (6.2, 11.0, out_y), (6.6, 11.0, out_y),
        (13.0, 10.4, 33.6), (15.0, 10.4, 33.6), (16.4, 14.0, 26.0), (30.0, 14.0, 26.0),
    ]
    return build_events_match(
        f"synthetic-throw-in-{excursion_m}", 30.0, scripts, ball,
        ball_hidden_spans=[(6.7, 12.9)],
    )


def ball_stays_in_scenario() -> MatchTracking:
    """The ball brushes toward the touchline (max y=33.6) and play flows on.
    Expected: no restart events at all."""
    scripts = team_442(TeamSide.HOME, 45.0) + team_442(TeamSide.AWAY, 40.0)
    scripts = replace(scripts, static(201, TeamSide.AWAY, Role.OUTFIELD, 42.5, -32.5))
    ball = [
        (0.0, 9.5, 32.0), (5.0, 9.5, 32.0), (6.0, 10.5, 33.6), (7.0, 9.5, 32.5),
        (10.0, 5.0, 20.0), (16.0, -5.0, 5.0), (20.0, 6.0, -10.0),
        (25.0, -8.0, 0.0), (30.0, -4.0, 10.0),
    ]
    return build_events_match("synthetic-ball-stays-in", 30.0, scripts, ball)


def corner_scenario() -> MatchTracking:
    """AWAY's defender blocks the ball behind their own goal line; HOME takes the
    corner from the (+,+) corner arc.

    Expected: one CORNER for HOME, defender (AWAY 204) as last touch, resumption
    at the corner point, no attribution conflict.
    """
    scripts = team_442(TeamSide.HOME, 55.0) + team_442(TeamSide.AWAY, 30.0)
    scripts = replace(
        scripts,
        static(204, TeamSide.AWAY, Role.OUTFIELD, 5.0, -9.2),     # abs (47.5, 9.2): blocker
        Script(107, TeamSide.HOME, Role.OUTFIELD,                 # corner taker walks over
               [(0.0, 90.0, 20.0), (8.0, 90.0, 20.0), (14.0, 104.3, 33.0), (30.0, 104.3, 33.0)]),
        static(108, TeamSide.HOME, Role.OUTFIELD, 96.5, 22.0),    # abs (44, 22): receiver
    )
    ball = [
        (0.0, 47.0, 9.0), (5.0, 47.0, 9.0), (6.4, 53.9, 11.8), (7.0, 53.9, 11.8),
        (20.0, 52.2, 33.4), (24.0, 52.2, 33.4), (25.6, 44.0, 22.0), (30.0, 44.0, 22.0),
    ]
    return build_events_match(
        "synthetic-corner", 30.0, scripts, ball, ball_hidden_spans=[(7.1, 19.9)]
    )


def goal_kick_scenario(deflected: bool = False) -> MatchTracking:
    """HOME's striker puts the ball behind AWAY's goal line; AWAY's keeper
    restarts from the six-yard box.

    Expected: one GOAL_KICK for AWAY.
    ``deflected=True``: same last touch (HOME striker => attribution says goal
    kick) but play resumes from the corner arc — an unseen deflection. Expected:
    CORNER for HOME with ``attribution_conflict: true`` (geometry overrules touch,
    design doc §4.2).
    """
    scripts = team_442(TeamSide.HOME, 55.0) + team_442(TeamSide.AWAY, 30.0)
    scripts = replace(
        scripts,
        static(109, TeamSide.HOME, Role.OUTFIELD, 96.1, 8.8),     # abs (43.6, 8.8): last touch
    )
    ball = [(0.0, 44.0, 9.0), (5.0, 44.0, 9.0), (6.4, 53.9, 12.5), (7.0, 53.9, 12.5)]
    if deflected:
        scripts = replace(
            scripts,
            Script(107, TeamSide.HOME, Role.OUTFIELD,             # corner taker
                   [(0.0, 90.0, -20.0), (8.0, 90.0, -20.0), (16.0, 104.3, -33.0), (30.0, 104.3, -33.0)]),
            static(108, TeamSide.HOME, Role.OUTFIELD, 96.5, -22.0),
        )
        ball += [(18.0, 52.2, -33.4), (22.0, 52.2, -33.4), (23.6, 44.0, -22.0), (30.0, 44.0, -22.0)]
        hidden = [(7.1, 17.9)]
    else:
        scripts = replace(
            scripts,
            static(200, TeamSide.AWAY, Role.GOALKEEPER, 4.5, -2.0),  # abs (48.0, 2.0): taker
            static(209, TeamSide.AWAY, Role.OUTFIELD, 32.5, 5.0),    # abs (20, -5): receiver
        )
        ball += [(18.0, 48.2, 2.0), (21.0, 48.2, 2.0), (23.0, 20.0, -5.0), (30.0, 20.0, -5.0)]
        hidden = [(7.1, 17.9)]
    return build_events_match(
        "synthetic-goal-kick-deflected" if deflected else "synthetic-goal-kick",
        30.0, scripts, ball, ball_hidden_spans=hidden,
    )


def kickoff_scenario() -> MatchTracking:
    """Both teams in their own halves, ball at rest on the centre spot, HOME 110
    plays it backward. Expected: one KICKOFF for HOME, nothing else."""
    scripts = team_442(TeamSide.HOME, 30.0) + team_442(TeamSide.AWAY, 30.0)
    scripts = replace(
        scripts,
        static(110, TeamSide.HOME, Role.OUTFIELD, 51.7, -0.5),    # abs (-0.8, -0.5): taker
        static(108, TeamSide.HOME, Role.OUTFIELD, 44.5, 3.0),     # abs (-8, 3): receiver
    )
    ball = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (5.5, -8.0, 3.0), (20.0, -8.0, 3.0)]
    return build_events_match("synthetic-kickoff", 20.0, scripts, ball)


# ---------------------------------------------------------------------------
# Tier 2: passes
# ---------------------------------------------------------------------------

def pass_scenario(variant: str = "short") -> MatchTracking:
    """One scripted pass per variant, against a full 22-player background.

    Variants: "short" (12m to feet), "long" (35m diagonal), "through" (received
    beyond AWAY's back line at abs x=37.5), "cross" (wide right-of-centre... wide
    channel into the box), "cutback" (byline strip backward to the penalty-spot
    zone), "intercepted" (AWAY body on the short pass's line), "dribble" (ball
    carried 20m at 4 m/s — expected: NO pass event).
    """
    scripts = team_442(TeamSide.HOME, 40.0) + team_442(TeamSide.AWAY, 30.0)
    end = 20.0
    if variant in ("short", "intercepted"):
        scripts = replace(scripts, static(105, TeamSide.HOME, Role.OUTFIELD, 47.5, 2.0))
        if variant == "short":
            scripts = replace(scripts, static(106, TeamSide.HOME, Role.OUTFIELD, 59.5, 2.0))
        else:
            scripts = replace(scripts, static(208, TeamSide.AWAY, Role.OUTFIELD, 45.5, -2.0))
        ball = [(0.0, -4.6, 2.0), (5.0, -4.6, 2.0), (6.5, 7.0, 2.0), (end, 7.0, 2.0)]
    elif variant == "long":
        scripts = replace(
            scripts,
            static(105, TeamSide.HOME, Role.OUTFIELD, 47.5, 2.0),
            static(106, TeamSide.HOME, Role.OUTFIELD, 82.5, -6.0),  # abs (30, -6)
        )
        ball = [(0.0, -4.6, 2.0), (5.0, -4.6, 2.0), (7.0, 30.0, -6.0), (end, 30.0, -6.0)]
    elif variant == "through":
        scripts = replace(
            scripts,
            static(105, TeamSide.HOME, Role.OUTFIELD, 72.5, 5.0),   # abs (20, 5)
            Script(109, TeamSide.HOME, Role.OUTFIELD,               # runner beats the line
                   [(0.0, 85.5, 3.0), (5.0, 85.5, 3.0), (7.5, 99.0, 3.0), (end, 99.0, 3.0)]),
        )
        ball = [(0.0, 20.4, 5.0), (5.0, 20.4, 5.0), (7.0, 43.0, 4.0), (end, 43.0, 4.0)]
    elif variant == "cross":
        scripts = replace(
            scripts,
            static(102, TeamSide.HOME, Role.OUTFIELD, 74.5, 26.0),  # abs (22, 26): wide
            static(109, TeamSide.HOME, Role.OUTFIELD, 92.5, 2.0),   # abs (40, 2): in the box
        )
        ball = [(0.0, 22.3, 25.6), (5.0, 22.3, 25.6), (7.0, 40.0, 2.0), (end, 40.0, 2.0)]
    elif variant == "cutback":
        scripts = replace(
            scripts,
            static(102, TeamSide.HOME, Role.OUTFIELD, 98.5, 15.0),  # abs (46, 15): byline
            static(106, TeamSide.HOME, Role.OUTFIELD, 88.5, 1.0),   # abs (36, 1)
        )
        ball = [(0.0, 45.7, 14.8), (5.0, 45.7, 14.8), (6.7, 36.0, 1.0), (end, 36.0, 1.0)]
    elif variant == "dribble":
        scripts = replace(
            scripts,
            Script(105, TeamSide.HOME, Role.OUTFIELD,
                   [(0.0, 47.5, 2.0), (5.0, 47.5, 2.0), (10.0, 67.5, 2.0), (end, 67.5, 2.0)]),
        )
        ball = [(0.0, -4.7, 2.0), (5.0, -4.7, 2.0), (10.0, 15.3, 2.0), (end, 15.3, 2.0)]
    else:
        raise ValueError(f"unknown pass variant {variant!r}")
    return build_events_match(f"synthetic-pass-{variant}", end, scripts, ball)


# ---------------------------------------------------------------------------
# Tier 2: shots
# ---------------------------------------------------------------------------

def shot_scenario(outcome: str = "goal") -> MatchTracking:
    """HOME 109 shoots from ~19m. Variants pin the outcome honesty ladder:

    - "goal": crosses the line inside the mouth, tracker loses it in the net,
      and a kickoff follows (both teams walk back to their halves; AWAY restarts
      from the spot). Expected: SHOT outcome "goal" via kickoff corroboration.
    - "saved": the flight reverses off AWAY's keeper. Expected: outcome "saved".
    - "wide": misses the mouth, goes behind, AWAY goal kick follows. Expected:
      outcome "off_target" with restart_after "goal_kick".
    """
    end = 35.0
    if outcome == "goal":
        scripts = phased_team(TeamSide.HOME, 70.0, 30.0, 10.0, 22.0, end) \
            + team_442(TeamSide.AWAY, 25.0)
        scripts = replace(
            scripts,
            Script(109, TeamSide.HOME, Role.OUTFIELD,   # shooter, then retreats
                   [(0.0, 88.0, 2.2), (10.0, 88.0, 2.2), (22.0, 40.0, 2.2), (end, 40.0, 2.2)]),
            Script(210, TeamSide.AWAY, Role.OUTFIELD,   # kickoff taker walks to the spot
                   [(0.0, 40.0, 8.0), (10.0, 40.0, 8.0), (22.0, 51.6, 0.4), (end, 51.6, 0.4)]),
            static(209, TeamSide.AWAY, Role.OUTFIELD, 45.5, -1.5),  # abs (7, 1.5): receiver
        )
        ball = [
            (0.0, 35.3, 2.0), (8.0, 35.3, 2.0), (9.4, 54.5, 0.4),
            (24.0, 0.0, 0.0), (28.0, 0.0, 0.0), (29.4, 7.0, 1.5), (end, 7.0, 1.5),
        ]
        hidden = [(9.6, 23.9)]
    elif outcome == "saved":
        scripts = team_442(TeamSide.HOME, 70.0) + team_442(TeamSide.AWAY, 25.0)
        scripts = replace(
            scripts,
            static(109, TeamSide.HOME, Role.OUTFIELD, 88.0, 2.2),      # abs (35.5, 2.2)
            static(200, TeamSide.AWAY, Role.GOALKEEPER, 0.9, -0.4),    # abs (51.6, 0.4)
        )
        ball = [
            (0.0, 35.3, 2.0), (8.0, 35.3, 2.0), (9.2, 50.9, 0.55),
            (10.4, 44.0, 5.0), (end, 44.0, 5.0),
        ]
        hidden = []
    elif outcome == "wide":
        scripts = team_442(TeamSide.HOME, 70.0) + team_442(TeamSide.AWAY, 25.0)
        scripts = replace(
            scripts,
            static(109, TeamSide.HOME, Role.OUTFIELD, 88.0, 2.2),
            static(200, TeamSide.AWAY, Role.GOALKEEPER, 4.2, 1.5),     # abs (48.3, -1.5)
            static(209, TeamSide.AWAY, Role.OUTFIELD, 32.5, -5.0),     # abs (20, 5): receiver
        )
        ball = [
            (0.0, 35.3, 2.0), (8.0, 35.3, 2.0), (9.4, 54.4, 4.9),
            (18.0, 48.3, -1.5), (21.0, 48.3, -1.5), (23.0, 20.0, 5.0), (end, 20.0, 5.0),
        ]
        hidden = [(9.6, 17.9)]
    else:
        raise ValueError(f"unknown shot outcome {outcome!r}")
    return build_events_match(f"synthetic-shot-{outcome}", end, scripts, ball,
                              ball_hidden_spans=hidden)


# ---------------------------------------------------------------------------
# Tier 2: set-piece organization
# ---------------------------------------------------------------------------

def set_piece_scenario() -> MatchTracking:
    """Play stops at abs (30, 10) — a direct free kick range for HOME. AWAY
    builds a four-man wall on the 9.15m arc; both teams load the box; HOME 108
    delivers at t=20 while HOME 106 attacks the box.

    Expected: one SET_PIECE_SETUP, organization "wall", wall_team "away",
    taker team HOME, ~t=6..20, runners_into_box >= 1.
    """
    end = 25.0
    wall_targets = [(37.5, 4.2), (38.1, 5.6), (38.7, 6.9), (39.3, 8.3)]
    wall_starts = [(34.0, -2.0), (35.0, 10.0), (41.0, 3.0), (40.0, 12.0)]

    def away(x_abs: float, y_abs: float) -> Tuple[float, float]:
        return 52.5 - x_abs, -y_abs

    scripts: List[Script] = [static(100, TeamSide.HOME, Role.GOALKEEPER, 8.0, 0.0)]
    for tid, (x, y) in zip(
        (101, 102, 103, 104, 105),
        [(57.5, -15.0), (62.5, 15.0), (52.5, 0.0), (64.5, -2.0), (70.5, 20.0)],
    ):
        scripts.append(static(tid, TeamSide.HOME, Role.OUTFIELD, x, y))
    scripts.append(
        Script(106, TeamSide.HOME, Role.OUTFIELD,   # the box runner
               [(0.0, 85.5, -8.0), (20.0, 85.5, -8.0), (21.5, 95.5, -6.0), (end, 95.5, -6.0)])
    )
    scripts.append(static(107, TeamSide.HOME, Role.OUTFIELD, 95.5, 8.0))   # abs (43, 8)
    scripts.append(static(108, TeamSide.HOME, Role.OUTFIELD, 81.5, 10.5))  # abs (29, 10.5): taker
    scripts.append(static(109, TeamSide.HOME, Role.OUTFIELD, 96.5, -5.0))  # abs (44, -5)
    scripts.append(static(110, TeamSide.HOME, Role.OUTFIELD, 97.5, 3.0))   # abs (45, 3)

    scripts.append(static(200, TeamSide.AWAY, Role.GOALKEEPER, *away(51.5, 0.2)))
    scripts.append(static(201, TeamSide.AWAY, Role.OUTFIELD, *away(42.0, 20.0)))
    for tid, (sx, sy), (tx, ty) in zip((202, 203, 204, 205), wall_starts, wall_targets):
        sxr, syr = away(sx, sy)
        txr, tyr = away(tx, ty)
        scripts.append(
            Script(tid, TeamSide.AWAY, Role.OUTFIELD,
                   [(0.0, sxr, syr), (6.0, sxr, syr), (10.0, txr, tyr), (end, txr, tyr)])
        )
    for tid, (x, y) in zip(
        (206, 207, 208, 209, 210),
        [(46.0, -4.0), (46.0, 4.0), (44.0, 0.0), (30.0, -15.0), (25.0, 18.0)],
    ):
        scripts.append(static(tid, TeamSide.AWAY, Role.OUTFIELD, *away(x, y)))

    ball = [
        (0.0, 10.0, -5.0), (5.6, 28.9, 9.4), (6.0, 30.0, 10.0), (20.0, 30.0, 10.0),
        (21.4, 49.0, 1.5), (end, 49.0, 1.5),
    ]
    return build_events_match("synthetic-set-piece", end, scripts, ball)


# ---------------------------------------------------------------------------
# Tier 3: stoppage
# ---------------------------------------------------------------------------

def stoppage_scenario() -> MatchTracking:
    """Open play (everyone jogging, ball carried), then at t=8 everything stops:
    players freeze, the ball rests at (0, 5) and the tracker loses it at t=12.
    Expected: one STOPPAGE from ~t=8 to the end, explained_by None."""
    end = 25.0
    scripts: List[Script] = [
        static(100, TeamSide.HOME, Role.GOALKEEPER, 6.0, 0.0),
        static(200, TeamSide.AWAY, Role.GOALKEEPER, 6.0, 0.0),
    ]
    home_bases = [(30.0, -20.0), (30.0, -7.0), (30.0, 7.0), (30.0, 20.0),
                  (45.0, -20.0), (45.0, -7.0), (45.0, 7.0), (45.0, 20.0),
                  (58.0, -8.0), (58.0, 8.0)]
    for i, (x, y) in enumerate(home_bases):
        scripts.append(jog(101 + i, TeamSide.HOME, Role.OUTFIELD, x, y, until=8.0, end=end))
    away_bases = [(35.0, -18.0), (35.0, -6.0), (35.0, 6.0), (35.0, 18.0),
                  (50.0, -18.0), (50.0, -6.0), (50.0, 6.0), (50.0, 18.0),
                  (62.0, -9.0), (62.0, 9.0)]
    for i, (x, y) in enumerate(away_bases):
        scripts.append(jog(201 + i, TeamSide.AWAY, Role.OUTFIELD, x, y, until=8.0, end=end))
    ball = [(0.0, -10.0, 0.0), (8.0, 0.0, 5.0), (12.0, 0.0, 5.0), (end, 0.0, 5.0)]
    return build_events_match("synthetic-stoppage", end, scripts, ball,
                              ball_hidden_spans=[(12.1, end)])


def slow_circulation_scenario() -> MatchTracking:
    """Players stand near-still while the ball keeps circulating — static players
    are NOT a stoppage while the ball moves. Expected: no STOPPAGE events."""
    end = 20.0
    scripts = team_442(TeamSide.HOME, 45.0) + team_442(TeamSide.AWAY, 40.0)
    ball = [
        (0.0, 10.0, 10.0), (4.0, -10.0, 5.0), (8.0, 5.0, -12.0),
        (12.0, -12.0, -8.0), (16.0, 8.0, 12.0), (end, -6.0, 4.0),
    ]
    return build_events_match("synthetic-slow-circulation", end, scripts, ball)
