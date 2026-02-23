"""
Project: Threader
File Created: 2026-02-18
Author: Xingnan Zhu
File Name: plotly_animation.py
Description:
    Plotly animation builder for pass replay.
    Creates a Plotly figure with native frames/slider/play-button
    that animates player and ball movement from tracking data.
"""

from __future__ import annotations

import plotly.graph_objects as go

from threader.data.pff.tracking_frames import AnimationFrame
from threader.viz.plotly_pitch import draw_pitch

# Colors (consistent with plotly_passes.py)
HOME_COLOR = "#3498db"
AWAY_COLOR = "#e74c3c"
BALL_COLOR = "#ffffff"
TRAIL_COLOR = "rgba(255, 255, 255, 0.35)"

# Frame timing (ms)
FRAME_DURATION_MS = 100  # 10fps → 100ms per frame
TRANSITION_DURATION_MS = 80  # Smooth interpolation


def build_animation_figure(
    frames: list[AnimationFrame],
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    title: str | None = None,
) -> go.Figure:
    """Build a Plotly figure with native animation frames.

    The figure contains 4 data traces (in fixed order):
      0: Home team players (blue markers + jersey text)
      1: Away team players (red markers + jersey text)
      2: Ball (white marker)
      3: Ball trail (line trace, accumulated)
      4: Time indicator annotation (updated per frame)

    Each Plotly Frame updates traces 0-3 with that frame's positions.

    Args:
        frames: List of AnimationFrame objects (chronological, ~10fps).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        title: Optional title for the figure.

    Returns:
        Plotly Figure with animation controls.
    """
    if not frames:
        fig = draw_pitch(pitch_length=pitch_length, pitch_width=pitch_width, center_origin=True)
        fig.update_layout(title="No tracking data available")
        return fig

    # ── Build base figure (pitch + initial frame traces) ─────────────
    fig = draw_pitch(pitch_length=pitch_length, pitch_width=pitch_width, center_origin=True)

    # The pitch drawing adds several traces (arcs, spots, etc.)
    # We need to track where our animated traces start
    pitch_trace_count = len(fig.data)

    first = frames[0]

    # Trace pitch_trace_count+0: Home team players
    fig.add_trace(
        go.Scatter(
            x=[p.x for p in first.home_players],
            y=[p.y for p in first.home_players],
            mode="markers+text",
            marker=dict(size=22, color=HOME_COLOR, line=dict(width=2, color="white"), opacity=0.9),
            text=[str(p.jersey_num) if p.jersey_num else "" for p in first.home_players],
            textfont=dict(size=9, color="white", family="Arial Black"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            name="Home",
        )
    )

    # Trace pitch_trace_count+1: Away team players
    fig.add_trace(
        go.Scatter(
            x=[p.x for p in first.away_players],
            y=[p.y for p in first.away_players],
            mode="markers+text",
            marker=dict(size=22, color=AWAY_COLOR, line=dict(width=2, color="white"), opacity=0.9),
            text=[str(p.jersey_num) if p.jersey_num else "" for p in first.away_players],
            textfont=dict(size=9, color="white", family="Arial Black"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            name="Away",
        )
    )

    # Trace pitch_trace_count+2: Ball
    fig.add_trace(
        go.Scatter(
            x=[first.ball_x],
            y=[first.ball_y],
            mode="markers",
            marker=dict(
                size=12,
                color=BALL_COLOR,
                line=dict(width=2, color="#333333"),
                symbol="circle",
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Ball",
        )
    )

    # Trace pitch_trace_count+3: Ball trail (starts empty, accumulates)
    fig.add_trace(
        go.Scatter(
            x=[first.ball_x],
            y=[first.ball_y],
            mode="lines",
            line=dict(color=TRAIL_COLOR, width=2, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
            name="Ball Trail",
        )
    )

    # ── Build animation frames ───────────────────────────────────────
    plotly_frames = []
    slider_steps = []

    # Collect ball trail positions
    ball_trail_x = []
    ball_trail_y = []

    for i, af in enumerate(frames):
        ball_trail_x.append(af.ball_x)
        ball_trail_y.append(af.ball_y)

        frame_data = [
            # Home players
            go.Scatter(
                x=[p.x for p in af.home_players],
                y=[p.y for p in af.home_players],
                text=[str(p.jersey_num) if p.jersey_num else "" for p in af.home_players],
            ),
            # Away players
            go.Scatter(
                x=[p.x for p in af.away_players],
                y=[p.y for p in af.away_players],
                text=[str(p.jersey_num) if p.jersey_num else "" for p in af.away_players],
            ),
            # Ball
            go.Scatter(
                x=[af.ball_x],
                y=[af.ball_y],
            ),
            # Ball trail (accumulated)
            go.Scatter(
                x=list(ball_trail_x),
                y=list(ball_trail_y),
            ),
        ]

        # Time label for this frame
        t = af.relative_time
        if t < 0:
            time_label = f"{t:.1f}s"
        elif t == 0:
            time_label = "PASS"
        else:
            time_label = f"+{t:.1f}s"

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

    # ── Play / Pause buttons ─────────────────────────────────────────
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

    # ── Timeline slider ──────────────────────────────────────────────
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

    # ── Initial time annotation ──────────────────────────────────────
    t0 = frames[0].relative_time
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{t0:.1f}s</b>",
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

    # ── Layout adjustments ───────────────────────────────────────────
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color="white", family="Arial"),
                x=0.5,
                xanchor="center",
            )
        )

    # Need extra bottom margin for slider/buttons
    fig.update_layout(
        margin=dict(b=80),
    )

    return fig
