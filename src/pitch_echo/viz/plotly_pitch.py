"""
Project: Threader
File Created: 2026-02-16
Author: Xingnan Zhu
File Name: plotly_pitch.py
Description:
    Football pitch rendering with Plotly.
    Draws a standard football pitch using layout shapes for performance.
    Supports PFF's center-origin coordinate system.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go


# Pitch colors
PITCH_COLOR = "#1a472a"
LINE_COLOR = "rgba(255, 255, 255, 0.7)"
LINE_WIDTH = 1.5


def _arc_points(
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


def draw_pitch(
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    *,
    center_origin: bool = True,
) -> go.Figure:
    """Draw a football pitch as a Plotly figure using shapes and scatter arcs.

    Args:
        pitch_length: Length of the pitch in meters.
        pitch_width: Width of the pitch in meters.
        center_origin: If True, origin is at pitch center (PFF coords).

    Returns:
        A Plotly Figure with the pitch drawn.
    """
    if center_origin:
        half_l = pitch_length / 2
        half_w = pitch_width / 2
        x_min, x_max = -half_l, half_l
        y_min, y_max = -half_w, half_w
    else:
        x_min, x_max = 0.0, pitch_length
        y_min, y_max = 0.0, pitch_width

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2

    shapes: list[dict] = []
    line_props = dict(color=LINE_COLOR, width=LINE_WIDTH)

    # --- Pitch outline ---
    shapes.append(
        dict(
            type="rect",
            x0=x_min, y0=y_min, x1=x_max, y1=y_max,
            line=line_props,
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    )

    # --- Halfway line ---
    shapes.append(
        dict(
            type="line",
            x0=mid_x, y0=y_min, x1=mid_x, y1=y_max,
            line=line_props,
            layer="above",
        )
    )

    # --- Center circle (r = 9.15m) ---
    shapes.append(
        dict(
            type="circle",
            x0=mid_x - 9.15, y0=mid_y - 9.15,
            x1=mid_x + 9.15, y1=mid_y + 9.15,
            line=line_props,
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    )

    # --- Penalty areas (16.5m deep, 40.32m wide) ---
    pa_w = 40.32 / 2
    pa_d = 16.5

    # Left penalty area
    shapes.append(
        dict(
            type="rect",
            x0=x_min, y0=mid_y - pa_w,
            x1=x_min + pa_d, y1=mid_y + pa_w,
            line=line_props,
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    )
    # Right penalty area
    shapes.append(
        dict(
            type="rect",
            x0=x_max - pa_d, y0=mid_y - pa_w,
            x1=x_max, y1=mid_y + pa_w,
            line=line_props,
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    )

    # --- Goal areas (5.5m deep, 18.32m wide) ---
    ga_w = 18.32 / 2
    ga_d = 5.5

    # Left goal area
    shapes.append(
        dict(
            type="rect",
            x0=x_min, y0=mid_y - ga_w,
            x1=x_min + ga_d, y1=mid_y + ga_w,
            line=line_props,
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    )
    # Right goal area
    shapes.append(
        dict(
            type="rect",
            x0=x_max - ga_d, y0=mid_y - ga_w,
            x1=x_max, y1=mid_y + ga_w,
            line=line_props,
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    )

    # --- Goals (7.32m wide, drawn behind goal line) ---
    goal_w = 7.32 / 2
    goal_d = 2.0
    # Left goal
    shapes.append(
        dict(
            type="rect",
            x0=x_min - goal_d, y0=mid_y - goal_w,
            x1=x_min, y1=mid_y + goal_w,
            line=dict(color="rgba(255,255,255,0.5)", width=1.5),
            fillcolor="rgba(255,255,255,0.05)",
            layer="above",
        )
    )
    # Right goal
    shapes.append(
        dict(
            type="rect",
            x0=x_max, y0=mid_y - goal_w,
            x1=x_max + goal_d, y1=mid_y + goal_w,
            line=dict(color="rgba(255,255,255,0.5)", width=1.5),
            fillcolor="rgba(255,255,255,0.05)",
            layer="above",
        )
    )

    # --- Build figure ---
    fig = go.Figure()

    # Penalty arcs (9.15m from penalty spot, outside the box)
    # Left arc: penalty spot at (x_min + 11, mid_y)
    lx, ly = _arc_points(x_min + 11, mid_y, 9.15, -53, 53)
    fig.add_trace(
        go.Scatter(
            x=lx, y=ly,
            mode="lines",
            line=dict(color=LINE_COLOR, width=LINE_WIDTH),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Right arc: penalty spot at (x_max - 11, mid_y)
    rx, ry = _arc_points(x_max - 11, mid_y, 9.15, 127, 233)
    fig.add_trace(
        go.Scatter(
            x=rx, y=ry,
            mode="lines",
            line=dict(color=LINE_COLOR, width=LINE_WIDTH),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Corner arcs (1m radius)
    corners = [
        (x_min, y_min, 0, 90),
        (x_min, y_max, 270, 360),
        (x_max, y_min, 90, 180),
        (x_max, y_max, 180, 270),
    ]
    for cx, cy, t1, t2 in corners:
        cax, cay = _arc_points(cx, cy, 1.0, t1, t2, n=20)
        fig.add_trace(
            go.Scatter(
                x=cax, y=cay,
                mode="lines",
                line=dict(color=LINE_COLOR, width=LINE_WIDTH),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Penalty spots + center spot
    fig.add_trace(
        go.Scatter(
            x=[x_min + 11, x_max - 11, mid_x],
            y=[mid_y, mid_y, mid_y],
            mode="markers",
            marker=dict(size=4, color=LINE_COLOR),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --- Layout ---
    margin = 5
    fig.update_layout(
        plot_bgcolor=PITCH_COLOR,
        paper_bgcolor="#0f0f1a",
        xaxis=dict(
            range=[x_min - margin, x_max + margin],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[y_min - margin, y_max + margin],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=shapes,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        dragmode=False,
    )

    return fig
