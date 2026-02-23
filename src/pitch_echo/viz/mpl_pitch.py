"""
Project: PitchEcho
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: pitch.py
Description: 
    Football pitch rendering with matplotlib.
    Draws a standard football pitch with accurate proportions.
    Supports PFF's center-origin coordinate system.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def draw_pitch(
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    *,
    center_origin: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 8),
    pitch_color: str = "#2d572c",
    line_color: str = "white",
    linewidth: float = 1.5,
) -> tuple[Figure, Axes]:
    """Draw a football pitch.

    Args:
        pitch_length: Length of the pitch in meters.
        pitch_width: Width of the pitch in meters.
        center_origin: If True, origin is at center (PFF coords).
            If False, origin is at bottom-left.
        ax: Optional existing axes to draw on.
        figsize: Figure size (width, height) in inches.
        pitch_color: Background color of the pitch.
        line_color: Color of the pitch markings.
        linewidth: Width of the pitch lines.

    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_facecolor(pitch_color)
    fig.set_facecolor(pitch_color)

    if center_origin:
        half_l = pitch_length / 2
        half_w = pitch_width / 2
        x_min, x_max = -half_l, half_l
        y_min, y_max = -half_w, half_w
    else:
        x_min, x_max = 0, pitch_length
        y_min, y_max = 0, pitch_width

    lw = linewidth
    lc = line_color

    # Pitch outline
    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color=lc, linewidth=lw,
    )

    # Halfway line
    mid_x = (x_min + x_max) / 2
    ax.plot([mid_x, mid_x], [y_min, y_max], color=lc, linewidth=lw)

    # Center circle (radius ~9.15m)
    mid_y = (y_min + y_max) / 2
    center_circle = mpatches.Circle(
        (mid_x, mid_y), 9.15,
        fill=False, color=lc, linewidth=lw,
    )
    ax.add_patch(center_circle)

    # Center spot
    ax.plot(mid_x, mid_y, "o", color=lc, markersize=3)

    # Penalty areas (16.5m from goal line, 40.32m wide)
    pa_w = 40.32 / 2  # half-width of penalty area
    pa_d = 16.5       # depth of penalty area

    # Left penalty area
    ax.plot(
        [x_min, x_min + pa_d, x_min + pa_d, x_min],
        [mid_y - pa_w, mid_y - pa_w, mid_y + pa_w, mid_y + pa_w],
        color=lc, linewidth=lw,
    )

    # Right penalty area
    ax.plot(
        [x_max, x_max - pa_d, x_max - pa_d, x_max],
        [mid_y - pa_w, mid_y - pa_w, mid_y + pa_w, mid_y + pa_w],
        color=lc, linewidth=lw,
    )

    # Goal areas (5.5m from goal line, 18.32m wide)
    ga_w = 18.32 / 2
    ga_d = 5.5

    # Left goal area
    ax.plot(
        [x_min, x_min + ga_d, x_min + ga_d, x_min],
        [mid_y - ga_w, mid_y - ga_w, mid_y + ga_w, mid_y + ga_w],
        color=lc, linewidth=lw,
    )

    # Right goal area
    ax.plot(
        [x_max, x_max - ga_d, x_max - ga_d, x_max],
        [mid_y - ga_w, mid_y - ga_w, mid_y + ga_w, mid_y + ga_w],
        color=lc, linewidth=lw,
    )

    # Penalty spots (11m from goal line)
    ax.plot(x_min + 11, mid_y, "o", color=lc, markersize=3)
    ax.plot(x_max - 11, mid_y, "o", color=lc, markersize=3)

    # Penalty arcs (9.15m from penalty spot, outside the box)
    left_arc = mpatches.Arc(
        (x_min + 11, mid_y), 2 * 9.15, 2 * 9.15,
        angle=0, theta1=-53, theta2=53,
        color=lc, linewidth=lw,
    )
    right_arc = mpatches.Arc(
        (x_max - 11, mid_y), 2 * 9.15, 2 * 9.15,
        angle=0, theta1=127, theta2=233,
        color=lc, linewidth=lw,
    )
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # Corner arcs (1m radius)
    for cx, cy in [
        (x_min, y_min), (x_min, y_max),
        (x_max, y_min), (x_max, y_max),
    ]:
        theta1 = 0
        if cx == x_min and cy == y_min:
            theta1, theta2 = 0, 90
        elif cx == x_min and cy == y_max:
            theta1, theta2 = 270, 360
        elif cx == x_max and cy == y_min:
            theta1, theta2 = 90, 180
        else:
            theta1, theta2 = 180, 270

        corner = mpatches.Arc(
            (cx, cy), 2, 2,
            angle=0, theta1=theta1, theta2=theta2,
            color=lc, linewidth=lw,
        )
        ax.add_patch(corner)

    # Set axis properties
    margin = 3
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax
