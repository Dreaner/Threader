"""
Project: Threader
File Name: plotly_animation_3d.py
Description:
    3D Plotly animation builder for pass replay.
    Creates a Plotly figure with native frames/slider/play-button
    that animates player and ball movement in 3D space.
    Ball z-coordinate (height) is taken directly from AnimationFrame.ball_z.
"""

from __future__ import annotations

import plotly.graph_objects as go

from threader.data.pff.tracking_frames import AnimationFrame
from threader.viz.plotly_pitch_3d import Z_MAX, draw_pitch_3d

# Colors (consistent with plotly_animation.py)
HOME_COLOR = "#3498db"
AWAY_COLOR = "#e74c3c"
TRAIL_COLOR = "rgba(255, 255, 255, 0.35)"

# Frame timing (ms)
FRAME_DURATION_MS = 100
TRANSITION_DURATION_MS = 80


def _ball_size_from_z(z: float) -> int:
    """Map ball height to marker size (perspective effect).

    z=0m → size 6, z=Z_MAX → size 4.
    """
    t = min(z / Z_MAX, 1.0)
    return int(6 - t * 2)


def _ball_color_from_z(z: float) -> str:
    """Map ball height to color (white at ground, orange at peak).

    Simulates sunlight / aerial perspective.
    """
    t = min(z / Z_MAX, 1.0)
    g = int(255 - t * 100)
    b = int(255 - t * 255)
    return f"rgb(255,{g},{b})"


def build_animation_figure_3d(
    frames: list[AnimationFrame],
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    title: str | None = None,
) -> go.Figure:
    """Build a Plotly 3D figure with native animation frames.

    The figure contains 6 animated data traces (in fixed order after the pitch):
      0: Home team players (blue markers + jersey text, z=0)
      1: Away team players (red markers + jersey text, z=0)
      2: Ball (3D marker at actual height ball_z)
      3: Ball trail (3D line, accumulated)
      4: Ball shadow (ground projection at z=0)
      5: Shadow line (vertical dotted line from ground to ball)

    Args:
        frames: List of AnimationFrame objects (chronological, ~10fps).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        title: Optional title for the figure.

    Returns:
        Plotly Figure with 3D animation controls.
    """
    if not frames:
        fig = draw_pitch_3d(pitch_length=pitch_length, pitch_width=pitch_width)
        fig.update_layout(title="No tracking data available")
        return fig

    # ── Build base figure (3D pitch) ──────────────────────────────────
    fig = draw_pitch_3d(pitch_length=pitch_length, pitch_width=pitch_width)
    pitch_trace_count = len(fig.data)

    first = frames[0]

    # Trace +0: Home team players (z=0 plane)
    fig.add_trace(
        go.Scatter3d(
            x=[p.x for p in first.home_players],
            y=[p.y for p in first.home_players],
            z=[0.0] * len(first.home_players),
            mode="markers+text",
            marker=dict(
                size=7,
                color=HOME_COLOR,
                line=dict(width=2, color="white"),
                opacity=0.9,
            ),
            text=[str(p.jersey_num) if p.jersey_num else "" for p in first.home_players],
            textfont=dict(size=8, color="white", family="Arial Black"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            name="Home",
        )
    )

    # Trace +1: Away team players (z=0 plane)
    fig.add_trace(
        go.Scatter3d(
            x=[p.x for p in first.away_players],
            y=[p.y for p in first.away_players],
            z=[0.0] * len(first.away_players),
            mode="markers+text",
            marker=dict(
                size=7,
                color=AWAY_COLOR,
                line=dict(width=2, color="white"),
                opacity=0.9,
            ),
            text=[str(p.jersey_num) if p.jersey_num else "" for p in first.away_players],
            textfont=dict(size=8, color="white", family="Arial Black"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            name="Away",
        )
    )

    # Trace +2: Ball (at actual height)
    fig.add_trace(
        go.Scatter3d(
            x=[first.ball_x],
            y=[first.ball_y],
            z=[first.ball_z],
            mode="markers",
            marker=dict(
                size=_ball_size_from_z(first.ball_z),
                color=_ball_color_from_z(first.ball_z),
                line=dict(width=1, color="#333333"),
                symbol="circle",
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Ball",
        )
    )

    # Trace +3: Ball trail (3D, starts with first point)
    fig.add_trace(
        go.Scatter3d(
            x=[first.ball_x],
            y=[first.ball_y],
            z=[first.ball_z],
            mode="lines",
            line=dict(color=TRAIL_COLOR, width=2),
            hoverinfo="skip",
            showlegend=False,
            name="Ball Trail",
        )
    )

    # Trace +4: Ball shadow (ground projection at z=0)
    fig.add_trace(
        go.Scatter3d(
            x=[first.ball_x],
            y=[first.ball_y],
            z=[0.0],
            mode="markers",
            marker=dict(
                size=6,
                color="rgba(255,255,255,0.25)",
                symbol="circle",
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Ball Shadow",
        )
    )

    # Trace +5: Shadow line (vertical dotted line from ground to ball)
    fig.add_trace(
        go.Scatter3d(
            x=[first.ball_x, first.ball_x],
            y=[first.ball_y, first.ball_y],
            z=[0.0, first.ball_z],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
            name="Shadow Line",
        )
    )

    # ── Build animation frames ────────────────────────────────────────
    plotly_frames = []
    slider_steps = []

    ball_trail_x: list[float] = []
    ball_trail_y: list[float] = []
    ball_trail_z: list[float] = []

    for i, af in enumerate(frames):
        ball_trail_x.append(af.ball_x)
        ball_trail_y.append(af.ball_y)
        ball_trail_z.append(af.ball_z)

        bsize = _ball_size_from_z(af.ball_z)
        bcolor = _ball_color_from_z(af.ball_z)

        frame_data = [
            # Home players (z=0)
            go.Scatter3d(
                x=[p.x for p in af.home_players],
                y=[p.y for p in af.home_players],
                z=[0.0] * len(af.home_players),
                text=[str(p.jersey_num) if p.jersey_num else "" for p in af.home_players],
            ),
            # Away players (z=0)
            go.Scatter3d(
                x=[p.x for p in af.away_players],
                y=[p.y for p in af.away_players],
                z=[0.0] * len(af.away_players),
                text=[str(p.jersey_num) if p.jersey_num else "" for p in af.away_players],
            ),
            # Ball (real height)
            go.Scatter3d(
                x=[af.ball_x],
                y=[af.ball_y],
                z=[af.ball_z],
                marker=dict(size=bsize, color=bcolor),
            ),
            # Ball trail (accumulated 3D path)
            go.Scatter3d(
                x=list(ball_trail_x),
                y=list(ball_trail_y),
                z=list(ball_trail_z),
            ),
            # Ball shadow (ground projection)
            go.Scatter3d(
                x=[af.ball_x],
                y=[af.ball_y],
                z=[0.0],
            ),
            # Shadow line (ground → ball)
            go.Scatter3d(
                x=[af.ball_x, af.ball_x],
                y=[af.ball_y, af.ball_y],
                z=[0.0, af.ball_z],
            ),
        ]

        t = af.relative_time
        time_label = "PASS" if t == 0 else (f"{t:.1f}s" if t < 0 else f"+{t:.1f}s")
        frame_name = f"f{i}"

        plotly_frames.append(
            go.Frame(
                data=frame_data,
                name=frame_name,
                traces=[
                    pitch_trace_count,      # Home players
                    pitch_trace_count + 1,  # Away players
                    pitch_trace_count + 2,  # Ball
                    pitch_trace_count + 3,  # Ball trail
                    pitch_trace_count + 4,  # Ball shadow
                    pitch_trace_count + 5,  # Shadow line
                ],
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"<b>{time_label}</b>",
                            x=0.02,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16, color="white", family="Arial Black"),
                            bgcolor="rgba(0,0,0,0.7)",
                            bordercolor="rgba(255,255,255,0.3)",
                            borderwidth=1,
                            borderpad=6,
                        )
                    ]
                ),
            )
        )

        slider_steps.append(
            dict(
                args=[
                    [frame_name],
                    dict(
                        frame=dict(duration=FRAME_DURATION_MS, redraw=True),
                        mode="immediate",
                        transition=dict(duration=TRANSITION_DURATION_MS),
                    ),
                ],
                label=time_label,
                method="animate",
            )
        )

    fig.frames = plotly_frames

    # ── Play / Pause buttons ──────────────────────────────────────────
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=-0.02,
                xanchor="left",
                yanchor="top",
                pad=dict(r=10, t=10),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=FRAME_DURATION_MS, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=TRANSITION_DURATION_MS),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
                font=dict(color="white", size=12),
                bgcolor="rgba(30, 30, 60, 0.85)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
            )
        ],
    )

    # ── Timeline slider ───────────────────────────────────────────────
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.15,
                len=0.85,
                xanchor="left",
                y=-0.02,
                yanchor="top",
                pad=dict(b=10, t=20),
                currentvalue=dict(
                    prefix="",
                    visible=True,
                    xanchor="center",
                    font=dict(size=13, color="white"),
                ),
                transition=dict(duration=TRANSITION_DURATION_MS),
                font=dict(color="rgba(255,255,255,0.6)", size=9),
                bgcolor="rgba(30, 30, 60, 0.5)",
                activebgcolor="rgba(255, 215, 0, 0.6)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
                ticklen=4,
                tickcolor="rgba(255,255,255,0.3)",
            )
        ],
    )

    # ── Initial time annotation ───────────────────────────────────────
    t0 = frames[0].relative_time
    label0 = "PASS" if t0 == 0 else f"{t0:.1f}s"
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{label0}</b>",
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="white", family="Arial Black"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1,
                borderpad=6,
            )
        ],
    )

    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color="white", family="Arial"),
                x=0.5,
                xanchor="center",
            )
        )

    fig.update_layout(margin=dict(b=80))

    return fig
