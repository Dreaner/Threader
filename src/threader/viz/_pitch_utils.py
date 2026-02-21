"""
Project: Threader
File Name: _pitch_utils.py
Description:
    Shared geometry utilities for pitch rendering.
    Used by both plotly_pitch.py (2D) and plotly_pitch_3d.py (3D).
"""

from __future__ import annotations

import math


def arc_points(
    cx: float,
    cy: float,
    r: float,
    theta1_deg: float,
    theta2_deg: float,
    n: int = 40,
) -> tuple[list[float], list[float]]:
    """Generate (x, y) points along a circular arc."""
    xs, ys = [], []
    for i in range(n + 1):
        angle = math.radians(theta1_deg + (theta2_deg - theta1_deg) * i / n)
        xs.append(cx + r * math.cos(angle))
        ys.append(cy + r * math.sin(angle))
    return xs, ys
