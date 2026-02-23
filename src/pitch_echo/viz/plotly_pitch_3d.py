"""
Project: PitchEcho
File Name: plotly_pitch_3d.py
Description:
    3D football pitch rendering with Plotly.
    Draws a standard football pitch in 3D space (z=0 plane) using
    go.Scatter3d traces and a go.Mesh3d ground surface.
    Supports PFF's center-origin coordinate system.
"""

from __future__ import annotations

import plotly.graph_objects as go

from pitch_echo.viz._pitch_utils import arc_points


# Pitch colors (consistent with plotly_pitch.py)
PITCH_COLOR = "#1a472a"
LINE_COLOR = "rgba(255, 255, 255, 0.7)"
LINE_WIDTH = 2

# Ball height used to scale z-axis (meters)
Z_MAX = 25.0


def _rect_pts(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return a closed rectangle as (x, y, z) lists with a trailing None."""
    return (
        [x0, x1, x1, x0, x0, None],
        [y0, y0, y1, y1, y0, None],
        [0.0, 0.0, 0.0, 0.0, 0.0, None],
    )


def draw_pitch_3d(
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    *,
    center_origin: bool = True,
) -> go.Figure:
    """Draw a football pitch as a Plotly 3D figure.

    The pitch is placed entirely at z=0. A green Mesh3d surface forms the
    ground; all pitch markings are go.Scatter3d traces.

    Args:
        pitch_length: Length of the pitch in meters.
        pitch_width: Width of the pitch in meters.
        center_origin: If True, origin is at pitch center (PFF coords).

    Returns:
        A Plotly Figure with the 3D pitch drawn.
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

    fig = go.Figure()

    # ── Ground surface (green rectangle at z=0) ───────────────────────
    fig.add_trace(
        go.Mesh3d(
            x=[x_min, x_max, x_max, x_min],
            y=[y_min, y_min, y_max, y_max],
            z=[0.0, 0.0, 0.0, 0.0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=PITCH_COLOR,
            opacity=1.0,
            hoverinfo="skip",
            showlegend=False,
            name="pitch_surface",
        )
    )

    # ── All straight pitch lines (merged into one Scatter3d) ──────────
    # Pitch outline + halfway line + penalty areas + goal areas + goals
    pa_w = 40.32 / 2
    pa_d = 16.5
    ga_w = 18.32 / 2
    ga_d = 5.5
    goal_w = 7.32 / 2
    goal_d = 2.0

    segments = [
        # Pitch outline
        _rect_pts(x_min, y_min, x_max, y_max),
        # Halfway line (vertical)
        ([mid_x, mid_x, None], [y_min, y_max, None], [0.0, 0.0, None]),
        # Left penalty area
        _rect_pts(x_min, mid_y - pa_w, x_min + pa_d, mid_y + pa_w),
        # Right penalty area
        _rect_pts(x_max - pa_d, mid_y - pa_w, x_max, mid_y + pa_w),
        # Left goal area
        _rect_pts(x_min, mid_y - ga_w, x_min + ga_d, mid_y + ga_w),
        # Right goal area
        _rect_pts(x_max - ga_d, mid_y - ga_w, x_max, mid_y + ga_w),
        # Left goal
        _rect_pts(x_min - goal_d, mid_y - goal_w, x_min, mid_y + goal_w),
        # Right goal
        _rect_pts(x_max, mid_y - goal_w, x_max + goal_d, mid_y + goal_w),
    ]

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    for sx, sy, sz in segments:
        line_x.extend(sx)
        line_y.extend(sy)
        line_z.extend(sz)

    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(color=LINE_COLOR, width=LINE_WIDTH),
            hoverinfo="skip",
            showlegend=False,
            name="pitch_lines",
        )
    )

    # ── Penalty arcs ──────────────────────────────────────────────────
    # Left arc (penalty spot at x_min + 11)
    lx, ly = arc_points(x_min + 11, mid_y, 9.15, -53, 53)
    fig.add_trace(
        go.Scatter3d(
            x=lx, y=ly, z=[0.0] * len(lx),
            mode="lines",
            line=dict(color=LINE_COLOR, width=LINE_WIDTH),
            hoverinfo="skip",
            showlegend=False,
            name="left_arc",
        )
    )

    # Right arc (penalty spot at x_max - 11)
    rx, ry = arc_points(x_max - 11, mid_y, 9.15, 127, 233)
    fig.add_trace(
        go.Scatter3d(
            x=rx, y=ry, z=[0.0] * len(rx),
            mode="lines",
            line=dict(color=LINE_COLOR, width=LINE_WIDTH),
            hoverinfo="skip",
            showlegend=False,
            name="right_arc",
        )
    )

    # ── Center circle (r = 9.15m) ─────────────────────────────────────
    ccx, ccy = arc_points(mid_x, mid_y, 9.15, 0, 360, n=60)
    fig.add_trace(
        go.Scatter3d(
            x=ccx, y=ccy, z=[0.0] * len(ccx),
            mode="lines",
            line=dict(color=LINE_COLOR, width=LINE_WIDTH),
            hoverinfo="skip",
            showlegend=False,
            name="center_circle",
        )
    )

    # ── Corner arcs (r = 1m) ──────────────────────────────────────────
    for cx, cy, t1, t2 in [
        (x_min, y_min, 0, 90),
        (x_min, y_max, 270, 360),
        (x_max, y_min, 90, 180),
        (x_max, y_max, 180, 270),
    ]:
        cax, cay = arc_points(cx, cy, 1.0, t1, t2, n=20)
        fig.add_trace(
            go.Scatter3d(
                x=cax, y=cay, z=[0.0] * len(cax),
                mode="lines",
                line=dict(color=LINE_COLOR, width=LINE_WIDTH),
                hoverinfo="skip",
                showlegend=False,
                name="corner_arc",
            )
        )

    # ── Penalty spots + center spot ───────────────────────────────────
    fig.add_trace(
        go.Scatter3d(
            x=[x_min + 11, x_max - 11, mid_x],
            y=[mid_y, mid_y, mid_y],
            z=[0.0, 0.0, 0.0],
            mode="markers",
            marker=dict(size=3, color=LINE_COLOR),
            hoverinfo="skip",
            showlegend=False,
            name="spots",
        )
    )

    # ── 3D scene layout ───────────────────────────────────────────────
    margin = 5.0
    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        scene=dict(
            xaxis=dict(
                range=[x_min - margin, x_max + margin],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showbackground=False,
                title="",
            ),
            yaxis=dict(
                range=[y_min - margin, y_max + margin],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showbackground=False,
                title="",
            ),
            zaxis=dict(
                range=[0, Z_MAX],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showbackground=False,
                title="",
            ),
            bgcolor="#0f0f1a",
            aspectmode="manual",
            aspectratio=dict(
                x=pitch_length / pitch_width,  # ≈1.544 for 105×68
                y=1.0,
                z=0.35,  # Compress z so aerial balls don't dominate
            ),
            camera=dict(
                eye=dict(x=0.0, y=-0.3, z=1.5),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        dragmode="orbit",
    )

    return fig
