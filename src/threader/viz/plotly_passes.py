"""
Project: Threader
File Created: 2026-02-16
Author: Xingnan Zhu
File Name: plotly_passes.py
Description:
    Pass analysis visualization with Plotly.
    Renders players, ball, and pass option arrows on a Plotly pitch figure.
    Designed for interactive use within Dash.
"""

from __future__ import annotations

import plotly.graph_objects as go

from threader.core.models import Player, Snapshot
from threader.metrics.pass_value.models import AnalysisResult, PassOption
from threader.viz.plotly_pitch import draw_pitch


# Rank colors: gold → silver → bronze → gray
_RANK_COLORS = ["#FFD700", "#C0C0C0", "#CD7F32", "#888888", "#666666"]

# Team colors
HOME_COLOR = "#3498db"
AWAY_COLOR = "#e74c3c"
PASSER_COLOR = "#2ecc71"
BALL_COLOR = "#ffffff"


def _player_hover(player: Player, extra: str = "") -> str:
    """Build hover text for a player."""
    name = player.name or f"ID:{player.player_id}"
    num = f"#{player.jersey_num}" if player.jersey_num else ""
    pos = player.position or ""
    parts = [f"<b>{num} {name}</b>"]
    if pos:
        parts.append(f"Position: {pos}")
    parts.append(f"x: {player.x:.1f}, y: {player.y:.1f}")
    if extra:
        parts.append(extra)
    return "<br>".join(parts)


def plot_players(
    fig: go.Figure,
    snapshot: Snapshot,
    passer: Player | None = None,
) -> None:
    """Add player markers to the pitch figure.

    Args:
        fig: Plotly figure with pitch already drawn.
        snapshot: The snapshot containing all player positions.
        passer: Optional passer to highlight differently.
    """
    passer_id = passer.player_id if passer else None

    # Separate passer from home players if applicable
    home_regular = []
    passer_player = None

    for p in snapshot.home_players:
        if p.player_id == passer_id:
            passer_player = p
        else:
            home_regular.append(p)

    # Home team players
    if home_regular:
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in home_regular],
                y=[p.y for p in home_regular],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=HOME_COLOR,
                    line=dict(width=2, color="white"),
                    opacity=0.9,
                ),
                text=[str(p.jersey_num) if p.jersey_num else "" for p in home_regular],
                textfont=dict(size=9, color="white", family="Arial Black"),
                textposition="middle center",
                hovertext=[_player_hover(p) for p in home_regular],
                hoverinfo="text",
                showlegend=False,
                name="Home",
            )
        )

    # Away team players
    if snapshot.away_players:
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in snapshot.away_players],
                y=[p.y for p in snapshot.away_players],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=AWAY_COLOR,
                    line=dict(width=2, color="white"),
                    opacity=0.9,
                ),
                text=[
                    str(p.jersey_num) if p.jersey_num else ""
                    for p in snapshot.away_players
                ],
                textfont=dict(size=9, color="white", family="Arial Black"),
                textposition="middle center",
                hovertext=[_player_hover(p) for p in snapshot.away_players],
                hoverinfo="text",
                showlegend=False,
                name="Away",
            )
        )

    # Passer (highlighted)
    if passer_player:
        fig.add_trace(
            go.Scatter(
                x=[passer_player.x],
                y=[passer_player.y],
                mode="markers+text",
                marker=dict(
                    size=26,
                    color=PASSER_COLOR,
                    line=dict(width=3, color="#ffffff"),
                    opacity=1.0,
                    symbol="circle",
                ),
                text=[
                    str(passer_player.jersey_num)
                    if passer_player.jersey_num
                    else ""
                ],
                textfont=dict(size=10, color="white", family="Arial Black"),
                textposition="middle center",
                hovertext=[_player_hover(passer_player, "<b>⚽ PASSER</b>")],
                hoverinfo="text",
                showlegend=False,
                name="Passer",
            )
        )

    # Ball
    fig.add_trace(
        go.Scatter(
            x=[snapshot.ball.x],
            y=[snapshot.ball.y],
            mode="markers",
            marker=dict(
                size=10,
                color=BALL_COLOR,
                line=dict(width=1.5, color="#333333"),
                symbol="circle",
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Ball",
        )
    )


def plot_pass_options(
    fig: go.Figure,
    result: AnalysisResult,
    *,
    top_n: int = 3,
    show_all: bool = True,
    selected_idx: int | None = None,
) -> None:
    """Draw pass option arrows from passer to targets.

    Args:
        fig: Plotly figure.
        result: AnalysisResult with scored options.
        top_n: Number of top options to highlight.
        show_all: Show all options (dimmed for non-top).
        selected_idx: Index of the currently selected option (0-based rank).
    """
    ranked = result.ranked_options
    passer = result.passer
    options_to_draw = ranked if show_all else ranked[:top_n]

    for i, option in enumerate(options_to_draw):
        is_top = i < top_n
        is_selected = selected_idx is not None and i == selected_idx

        color = _RANK_COLORS[min(i, len(_RANK_COLORS) - 1)]
        opacity = 1.0 if is_selected else (0.85 if is_top else 0.25)
        width = 3.5 if is_selected else (2.5 if is_top else 1.0)

        target = option.target

        # Arrow annotation
        fig.add_annotation(
            x=target.x,
            y=target.y,
            ax=passer.x,
            ay=passer.y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=width,
            arrowcolor=color,
            opacity=opacity,
        )

        # Score label at midpoint (only for top-N)
        if is_top:
            mid_x = passer.x + (target.x - passer.x) * 0.5
            mid_y = passer.y + (target.y - passer.y) * 0.5

            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"<b>#{i + 1}</b> ({option.pass_score:.0f})",
                showarrow=False,
                font=dict(size=11, color=color, family="Arial Black"),
                bgcolor="rgba(0,0,0,0.75)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                xref="x",
                yref="y",
            )


def build_analysis_figure(
    result: AnalysisResult,
    *,
    top_n: int = 3,
    show_all: bool = True,
    title: str | None = None,
    selected_idx: int | None = None,
) -> go.Figure:
    """Build a complete Plotly figure for a pass analysis result.

    Args:
        result: AnalysisResult from the analyzer.
        top_n: Number of top options to highlight.
        show_all: Show all options.
        title: Optional title.
        selected_idx: Highlighted option index.

    Returns:
        A Plotly Figure ready for display.
    """
    snapshot = result.snapshot
    fig = draw_pitch(
        pitch_length=snapshot.pitch_length,
        pitch_width=snapshot.pitch_width,
        center_origin=True,
    )

    plot_players(fig, snapshot, passer=result.passer)
    plot_pass_options(
        fig,
        result,
        top_n=top_n,
        show_all=show_all,
        selected_idx=selected_idx,
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

    return fig
