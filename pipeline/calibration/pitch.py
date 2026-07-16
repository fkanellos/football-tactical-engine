"""Canonical pitch geometry: the known ground truth every quality check leans on.

A calibration can be interrogated without any labelled data because the pitch
itself is a metric object we know exactly (calibration-design.md §6.1): 105 x 68 m,
boxes at fixed offsets, a 9.15 m centre circle. This module is the single source
for that geometry, in the engine's canonical frame (origin at the centre spot,
x along the touchlines, y along the halfway line — /docs/phase4-5-design.md §1.1).

Landmark names follow the SoccerNet line vocabulary where one exists ("Big rect.
left main" etc.) so real exported detections can be joined against this catalogue
without a mapping table. IFAB dimensions assumed exact; broadcast pitches deviate
by centimetres, far below the error scale this layer measures (metres).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .camera import Matrix, Point2, project

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

_HL = PITCH_LENGTH_M / 2.0  # 52.5
_HW = PITCH_WIDTH_M / 2.0  # 34.0

PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_HALF_WIDTH_M = 20.16
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16
PENALTY_SPOT_FROM_GOAL_M = 11.0
CENTRE_CIRCLE_RADIUS_M = 9.15
GOAL_HALF_WIDTH_M = 3.66

#: Ground-plane landmarks a keypoint detector can plausibly localise: line
#: intersections, penalty spots, the centre mark. Left = x < 0.
LANDMARKS: Dict[str, Point2] = {
    "corner_left_top": (-_HL, _HW),
    "corner_left_bottom": (-_HL, -_HW),
    "corner_right_top": (_HL, _HW),
    "corner_right_bottom": (_HL, -_HW),
    "halfway_top": (0.0, _HW),
    "halfway_bottom": (0.0, -_HW),
    "centre_mark": (0.0, 0.0),
    "big_rect_left_top_corner": (-_HL + PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M),
    "big_rect_left_bottom_corner": (-_HL + PENALTY_AREA_DEPTH_M, -PENALTY_AREA_HALF_WIDTH_M),
    "big_rect_left_top_goal_line": (-_HL, PENALTY_AREA_HALF_WIDTH_M),
    "big_rect_left_bottom_goal_line": (-_HL, -PENALTY_AREA_HALF_WIDTH_M),
    "big_rect_right_top_corner": (_HL - PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M),
    "big_rect_right_bottom_corner": (_HL - PENALTY_AREA_DEPTH_M, -PENALTY_AREA_HALF_WIDTH_M),
    "big_rect_right_top_goal_line": (_HL, PENALTY_AREA_HALF_WIDTH_M),
    "big_rect_right_bottom_goal_line": (_HL, -PENALTY_AREA_HALF_WIDTH_M),
    "small_rect_left_top_corner": (-_HL + GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M),
    "small_rect_left_bottom_corner": (-_HL + GOAL_AREA_DEPTH_M, -GOAL_AREA_HALF_WIDTH_M),
    "small_rect_left_top_goal_line": (-_HL, GOAL_AREA_HALF_WIDTH_M),
    "small_rect_left_bottom_goal_line": (-_HL, -GOAL_AREA_HALF_WIDTH_M),
    "small_rect_right_top_corner": (_HL - GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M),
    "small_rect_right_bottom_corner": (_HL - GOAL_AREA_DEPTH_M, -GOAL_AREA_HALF_WIDTH_M),
    "small_rect_right_top_goal_line": (_HL, GOAL_AREA_HALF_WIDTH_M),
    "small_rect_right_bottom_goal_line": (_HL, -GOAL_AREA_HALF_WIDTH_M),
    "penalty_spot_left": (-_HL + PENALTY_SPOT_FROM_GOAL_M, 0.0),
    "penalty_spot_right": (_HL - PENALTY_SPOT_FROM_GOAL_M, 0.0),
    "goal_left_post_top": (-_HL, GOAL_HALF_WIDTH_M),
    "goal_left_post_bottom": (-_HL, -GOAL_HALF_WIDTH_M),
    "goal_right_post_top": (_HL, GOAL_HALF_WIDTH_M),
    "goal_right_post_bottom": (_HL, -GOAL_HALF_WIDTH_M),
}

# Centre-circle samples (the real detector carries 12 "Circle central" keypoints)
# and penalty-arc landmarks (arc tip + arc/box-line intersections, keypoints
# 37-44 & 46/56 in the nbjw vocabulary). Without these the catalogue would
# under-represent exactly the views the OFI clip shows most.
for _i in range(8):
    _ang = math.pi * _i / 4.0
    LANDMARKS[f"centre_circle_{_i * 45}"] = (
        CENTRE_CIRCLE_RADIUS_M * math.cos(_ang),
        CENTRE_CIRCLE_RADIUS_M * math.sin(_ang),
    )
_ARC_DX = PENALTY_AREA_DEPTH_M - PENALTY_SPOT_FROM_GOAL_M  # 5.5 m spot->box line
_ARC_Y = math.sqrt(CENTRE_CIRCLE_RADIUS_M**2 - _ARC_DX**2)  # ~7.31 m
LANDMARKS.update(
    {
        "penalty_arc_left_tip": (-_HL + PENALTY_SPOT_FROM_GOAL_M + CENTRE_CIRCLE_RADIUS_M, 0.0),
        "penalty_arc_right_tip": (_HL - PENALTY_SPOT_FROM_GOAL_M - CENTRE_CIRCLE_RADIUS_M, 0.0),
        "penalty_arc_left_box_top": (-_HL + PENALTY_AREA_DEPTH_M, _ARC_Y),
        "penalty_arc_left_box_bottom": (-_HL + PENALTY_AREA_DEPTH_M, -_ARC_Y),
        "penalty_arc_right_box_top": (_HL - PENALTY_AREA_DEPTH_M, _ARC_Y),
        "penalty_arc_right_box_bottom": (_HL - PENALTY_AREA_DEPTH_M, -_ARC_Y),
    }
)

#: Straight painted line segments, for coverage / residual-vs-line checks.
LINE_SEGMENTS: Dict[str, Tuple[Point2, Point2]] = {
    "Side line top": ((-_HL, _HW), (_HL, _HW)),
    "Side line bottom": ((-_HL, -_HW), (_HL, -_HW)),
    "Side line left": ((-_HL, -_HW), (-_HL, _HW)),
    "Side line right": ((_HL, -_HW), (_HL, _HW)),
    "Middle line": ((0.0, -_HW), (0.0, _HW)),
    "Big rect. left main": (
        (-_HL + PENALTY_AREA_DEPTH_M, -PENALTY_AREA_HALF_WIDTH_M),
        (-_HL + PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M),
    ),
    "Big rect. left top": ((-_HL, PENALTY_AREA_HALF_WIDTH_M), (-_HL + PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M)),
    "Big rect. left bottom": ((-_HL, -PENALTY_AREA_HALF_WIDTH_M), (-_HL + PENALTY_AREA_DEPTH_M, -PENALTY_AREA_HALF_WIDTH_M)),
    "Big rect. right main": (
        (_HL - PENALTY_AREA_DEPTH_M, -PENALTY_AREA_HALF_WIDTH_M),
        (_HL - PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M),
    ),
    "Big rect. right top": ((_HL, PENALTY_AREA_HALF_WIDTH_M), (_HL - PENALTY_AREA_DEPTH_M, PENALTY_AREA_HALF_WIDTH_M)),
    "Big rect. right bottom": ((_HL, -PENALTY_AREA_HALF_WIDTH_M), (_HL - PENALTY_AREA_DEPTH_M, -PENALTY_AREA_HALF_WIDTH_M)),
    "Small rect. left main": (
        (-_HL + GOAL_AREA_DEPTH_M, -GOAL_AREA_HALF_WIDTH_M),
        (-_HL + GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M),
    ),
    "Small rect. left top": ((-_HL, GOAL_AREA_HALF_WIDTH_M), (-_HL + GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M)),
    "Small rect. left bottom": ((-_HL, -GOAL_AREA_HALF_WIDTH_M), (-_HL + GOAL_AREA_DEPTH_M, -GOAL_AREA_HALF_WIDTH_M)),
    "Small rect. right main": (
        (_HL - GOAL_AREA_DEPTH_M, -GOAL_AREA_HALF_WIDTH_M),
        (_HL - GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M),
    ),
    "Small rect. right top": ((_HL, GOAL_AREA_HALF_WIDTH_M), (_HL - GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M)),
    "Small rect. right bottom": ((_HL, -GOAL_AREA_HALF_WIDTH_M), (_HL - GOAL_AREA_DEPTH_M, -GOAL_AREA_HALF_WIDTH_M)),
}


def centre_circle_points(n: int = 16) -> List[Point2]:
    """n points on the centre circle — the known-conic invariant (design §6.1)."""
    return [
        (
            CENTRE_CIRCLE_RADIUS_M * math.cos(2.0 * math.pi * i / n),
            CENTRE_CIRCLE_RADIUS_M * math.sin(2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def pitch_corners() -> List[Point2]:
    return [(-_HL, -_HW), (_HL, -_HW), (_HL, _HW), (-_HL, _HW)]


def visible_landmarks(
    h: Matrix,
    frame_width: float,
    frame_height: float,
    margin_px: float = 0.0,
    landmarks: Optional[Dict[str, Point2]] = None,
) -> Dict[str, Point2]:
    """Landmarks whose projection under ``h`` falls inside the frame.

    The world-side view of "what could this camera possibly have seen" — the
    basis for both synthetic observation generation and support/coverage scoring.
    """
    result: Dict[str, Point2] = {}
    for name, wxy in (landmarks or LANDMARKS).items():
        p = project(h, wxy)
        if p is None:
            continue
        if -margin_px <= p[0] <= frame_width + margin_px and -margin_px <= p[1] <= frame_height + margin_px:
            result[name] = wxy
    return result
