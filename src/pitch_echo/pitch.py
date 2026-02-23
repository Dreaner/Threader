"""
Composable Pitch class — mplsoccer-style API for pass analysis visualization.

Usage (matplotlib)::

    pitch = Pitch()
    fig, ax = pitch.draw()
    pitch.pass_options(result, top_n=3, ax=ax)

Usage (Plotly)::

    pitch = Pitch(backend="plotly")
    fig = pitch.draw()
    pitch.pass_options(result, top_n=3, fig=fig)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as MplFigure

    import plotly.graph_objects as go

    from pitch_echo.analysis.models import AnalysisResult
    from pitch_echo.network.models import NetworkMetrics, PassNetwork


class Pitch:
    """A composable football pitch for pass analysis visualization.

    Supports both matplotlib and Plotly rendering backends.
    All plotting methods use lazy imports so ``matplotlib`` / ``plotly``
    are only required when actually called.

    Args:
        pitch_length: Pitch length in meters (default 105.0).
        pitch_width: Pitch width in meters (default 68.0).
        center_origin: Use center-origin coordinates (PFF style).
        backend: ``"mpl"`` for matplotlib, ``"plotly"`` for Plotly.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        *,
        center_origin: bool = True,
        backend: Literal["mpl", "plotly"] = "mpl",
    ) -> None:
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.center_origin = center_origin
        self.backend = backend

    # ------------------------------------------------------------------
    # draw
    # ------------------------------------------------------------------

    def draw(
        self,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (12, 8),
        **kwargs: Any,
    ) -> tuple[MplFigure, Axes] | go.Figure:
        """Draw the pitch.

        Returns:
            matplotlib: ``(fig, ax)`` tuple.
            Plotly: ``go.Figure``.
        """
        if self.backend == "mpl":
            from pitch_echo.viz.mpl_pitch import draw_pitch

            return draw_pitch(
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width,
                center_origin=self.center_origin,
                ax=ax,
                figsize=figsize,
                **kwargs,
            )
        else:
            from pitch_echo.viz.plotly_pitch import draw_pitch

            return draw_pitch(
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width,
                center_origin=self.center_origin,
            )

    # ------------------------------------------------------------------
    # pass_options
    # ------------------------------------------------------------------

    def pass_options(
        self,
        result: AnalysisResult,
        *,
        top_n: int = 3,
        show_all: bool = True,
        ax: Axes | None = None,
        fig: go.Figure | None = None,
        title: str | None = None,
        selected_idx: int | None = None,
    ) -> tuple[MplFigure, Axes] | go.Figure:
        """Visualize pass analysis results on the pitch.

        For matplotlib pass *ax* (from :meth:`draw`).
        For Plotly pass *fig* (from :meth:`draw`).
        If neither is given a new pitch is drawn automatically.
        """
        if self.backend == "mpl":
            from pitch_echo.viz.mpl_passes import (
                plot_pass_options,
                plot_players,
                visualize_analysis,
            )

            if ax is not None:
                plot_players(ax, result.snapshot, passer=result.passer)
                plot_pass_options(ax, result, top_n=top_n, show_all=show_all)
                return ax.figure, ax
            return visualize_analysis(
                result, top_n=top_n, show_all=show_all, title=title,
            )
        else:
            from pitch_echo.viz.plotly_passes import (
                build_analysis_figure,
                plot_pass_options as plotly_pass_options,
                plot_players as plotly_plot_players,
            )

            if fig is not None:
                plotly_plot_players(fig, result.snapshot, passer=result.passer)
                plotly_pass_options(
                    fig, result, top_n=top_n, show_all=show_all,
                    selected_idx=selected_idx,
                )
                return fig
            return build_analysis_figure(
                result, top_n=top_n, show_all=show_all,
                title=title, selected_idx=selected_idx,
            )

    # ------------------------------------------------------------------
    # pass_network
    # ------------------------------------------------------------------

    def pass_network(
        self,
        network: PassNetwork,
        metrics: NetworkMetrics | None = None,
        *,
        min_edge_count: int = 2,
        title: str | None = None,
    ) -> go.Figure:
        """Visualize a pass network on the pitch (Plotly only).

        Args:
            network: The pass network to render.
            metrics: Optional graph metrics for metric-toggle buttons.
            min_edge_count: Minimum combined pass count for an edge to show.
            title: Optional figure title.
        """
        from pitch_echo.viz.plotly_network import build_network_figure

        return build_network_figure(
            network,
            metrics=metrics,
            min_edge_count=min_edge_count,
            title=title,
        )
