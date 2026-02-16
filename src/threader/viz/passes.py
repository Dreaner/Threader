"""Pass options visualization — arrows, scores, and rankings on the pitch.

Renders the analysis result on top of a pitch drawing, showing:
  - Player positions (home/away colored)
  - Pass option arrows (color-coded by rank)
  - Pass Score labels
  - The passer highlighted
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from threader.models import AnalysisResult, Player, Snapshot
from threader.viz.pitch import draw_pitch


# Rank colors: gold → silver → bronze → gray
_RANK_COLORS = ["#FFD700", "#C0C0C0", "#CD7F32", "#888888", "#666666"]


def plot_players(
    ax: Axes,
    snapshot: Snapshot,
    passer: Player | None = None,
    *,
    home_color: str = "#3498db",
    away_color: str = "#e74c3c",
    passer_color: str = "#2ecc71",
    marker_size: float = 200,
) -> None:
    """Plot all players on the pitch.

    Args:
        ax: Matplotlib axes (with pitch already drawn).
        snapshot: The snapshot to visualize.
        passer: Optional passer to highlight differently.
        home_color: Color for home team markers.
        away_color: Color for away team markers.
        passer_color: Color for the passer marker.
        marker_size: Size of player markers.
    """
    for player in snapshot.home_players:
        color = passer_color if (passer and player.player_id == passer.player_id) else home_color
        ax.scatter(
            player.x, player.y,
            s=marker_size, c=color, edgecolors="white",
            linewidths=1.5, zorder=5,
        )
        label = str(player.jersey_num) if player.jersey_num else ""
        if label:
            ax.annotate(
                label,
                (player.x, player.y),
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                zorder=6,
            )

    for player in snapshot.away_players:
        ax.scatter(
            player.x, player.y,
            s=marker_size, c=away_color, edgecolors="white",
            linewidths=1.5, zorder=5,
        )
        label = str(player.jersey_num) if player.jersey_num else ""
        if label:
            ax.annotate(
                label,
                (player.x, player.y),
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                zorder=6,
            )

    # Ball
    ax.scatter(
        snapshot.ball.x, snapshot.ball.y,
        s=80, c="white", edgecolors="black",
        linewidths=1, zorder=7, marker="o",
    )


def plot_pass_options(
    ax: Axes,
    result: AnalysisResult,
    *,
    top_n: int = 3,
    show_all: bool = False,
    arrow_width: float = 2.0,
) -> None:
    """Draw pass option arrows from passer to targets.

    Args:
        ax: Matplotlib axes.
        result: AnalysisResult with scored options.
        top_n: Number of top options to highlight.
        show_all: If True, show all options (dimmed for non-top).
        arrow_width: Width of the arrow lines.
    """
    ranked = result.ranked_options
    passer = result.passer

    options_to_draw = ranked if show_all else ranked[:top_n]

    for i, option in enumerate(options_to_draw):
        rank = i + 1
        is_top = i < top_n

        color = _RANK_COLORS[min(i, len(_RANK_COLORS) - 1)]
        alpha = 0.9 if is_top else 0.3
        lw = arrow_width if is_top else arrow_width * 0.5

        target = option.target
        dx = target.x - passer.x
        dy = target.y - passer.y

        ax.annotate(
            "",
            xy=(target.x, target.y),
            xytext=(passer.x, passer.y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                mutation_scale=15,
            ),
            zorder=4,
        )

        if is_top:
            # Score label at midpoint of arrow
            mid_x = passer.x + dx * 0.5
            mid_y = passer.y + dy * 0.5
            ax.annotate(
                f"#{rank} ({option.pass_score:.0f})",
                (mid_x, mid_y),
                fontsize=9,
                fontweight="bold",
                color=color,
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    alpha=0.7,
                    edgecolor=color,
                ),
                zorder=8,
            )


def visualize_analysis(
    result: AnalysisResult,
    *,
    top_n: int = 3,
    show_all: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 9),
) -> tuple[Figure, Axes]:
    """Full visualization of a pass analysis result.

    Draws the pitch, players, and pass option arrows.

    Args:
        result: AnalysisResult from the analyzer.
        top_n: Number of top options to highlight.
        show_all: Show all pass options (dimmed non-top).
        title: Optional title for the plot.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    snapshot = result.snapshot
    fig, ax = draw_pitch(
        pitch_length=snapshot.pitch_length,
        pitch_width=snapshot.pitch_width,
        center_origin=True,
        figsize=figsize,
    )

    plot_players(ax, snapshot, passer=result.passer)
    plot_pass_options(ax, result, top_n=top_n, show_all=show_all)

    if title:
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)

    return fig, ax
