"""Basic distance and vector utilities."""

from __future__ import annotations

import math

from threader.models import Player


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(x2 - x1, y2 - y1)


def player_distance(a: Player, b: Player) -> float:
    """Euclidean distance between two players."""
    return euclidean(a.x, a.y, b.x, b.y)


def point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Shortest distance from point (px, py) to the line segment (x1,y1)→(x2,y2).

    Uses vector projection: project the point onto the infinite line,
    then clamp the parameter t to [0, 1] to stay on the segment.
    """
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        # Degenerate segment (start == end)
        return euclidean(px, py, x1, y1)

    # Projection parameter t ∈ [0, 1]
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return euclidean(px, py, closest_x, closest_y)


def projection_parameter(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Compute the projection parameter t of point (px,py) onto line (x1,y1)→(x2,y2).

    t = 0 means at (x1,y1), t = 1 means at (x2,y2).
    Values outside [0,1] indicate the projection is beyond the segment.
    """
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        return 0.0

    return ((px - x1) * dx + (py - y1) * dy) / length_sq
