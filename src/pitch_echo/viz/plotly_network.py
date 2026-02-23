"""
Project: PitchEcho
Author: Xingnan Zhu
File Name: pass_network/viz/plotly_network.py
Description:
    Plotly-based pass network visualization.

    Renders a team's pass network on a football pitch:
      - Undirected edges: line width ∝ combined pass count (A→B + B→A)
      - Nodes: size ∝ total pass involvement (sent + received)
      - Interactive buttons (when metrics provided) to switch between:
          Plain / Degree centrality / Betweenness / PageRank

    Usage:
        fig = build_network_figure(network)           # plain only
        fig = build_network_figure(network, metrics)  # with metric buttons
        fig.show()
"""

from __future__ import annotations

import plotly.graph_objects as go

from pitch_echo.network.models import NetworkMetrics, PassNetwork
from pitch_echo.viz.plotly_pitch import draw_pitch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDGE_COLOR = "rgba(255, 255, 255, 0.45)"
_NODE_DEFAULT_COLOR = "#3498db"
_NODE_BORDER_COLOR = "rgba(255, 255, 255, 0.9)"
_NODE_BORDER_WIDTH = 2

# Dark blue → medium blue → gold
_METRIC_COLORSCALE = [
    [0.0, "#1a3a6b"],
    [0.5, "#4a90d9"],
    [1.0, "#e8b838"],
]

_METRIC_LABELS: dict[str, str] = {
    "degree": "Degree Centrality",
    "betweenness": "Betweenness",
    "pagerank": "PageRank",
}

_METRIC_ATTRS: dict[str, str] = {
    "degree": "degree_centrality",
    "betweenness": "betweenness_centrality",
    "pagerank": "pagerank",
}

# Node size range (pixels)
_NODE_MIN_SIZE = 18.0
_NODE_MAX_SIZE = 40.0

# Edge width range (pixels)
_EDGE_MIN_WIDTH = 1.5
_EDGE_MAX_WIDTH = 6.0

# Button bar styling
_BTN_BG = "rgba(20, 30, 50, 0.85)"
_BTN_BORDER = "rgba(255, 255, 255, 0.2)"


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def build_network_figure(
    network: PassNetwork,
    metrics: NetworkMetrics | None = None,
    *,
    min_edge_count: int = 2,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive Plotly figure showing the pass network.

    When metrics are provided, adds Plain / Degree / Betweenness / PageRank
    toggle buttons at the top of the figure.  The default view is Plain.

    Args:
        network: PassNetwork built by build_pass_network().
        metrics: Optional NetworkMetrics.  If None, only the plain view is
            rendered (no buttons).
        min_edge_count: Edges with combined count below this value are hidden.
        title: Optional figure title.

    Returns:
        A Plotly Figure ready for fig.show() or fig.write_html().
    """
    fig = draw_pitch()

    if not network.nodes:
        return fig

    # ------------------------------------------------------------------ #
    # Step 1 — Build undirected edge dict                                 #
    # ------------------------------------------------------------------ #
    undirected: dict[tuple[int, int], int] = {}
    for (src, dst), edge in network.edges.items():
        key = (min(src, dst), max(src, dst))
        undirected[key] = undirected.get(key, 0) + edge.count

    visible_edges = {k: v for k, v in undirected.items() if v >= min_edge_count}
    max_count = max(visible_edges.values()) if visible_edges else 1

    # ------------------------------------------------------------------ #
    # Step 2 — Draw edges (one trace per edge, below nodes)               #
    # ------------------------------------------------------------------ #
    for (pid_a, pid_b), count in visible_edges.items():
        node_a = network.nodes.get(pid_a)
        node_b = network.nodes.get(pid_b)
        if node_a is None or node_b is None:
            continue

        width = _EDGE_MIN_WIDTH + (count / max_count) * (_EDGE_MAX_WIDTH - _EDGE_MIN_WIDTH)

        fig.add_trace(
            go.Scatter(
                x=[node_a.avg_x, node_b.avg_x],
                y=[node_a.avg_y, node_b.avg_y],
                mode="lines",
                line=dict(color=_EDGE_COLOR, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Record how many traces exist before node traces are added.
    # These (pitch arcs + edges) must always remain visible.
    n_pre_node = len(fig.data)

    # ------------------------------------------------------------------ #
    # Step 3 — Build shared node arrays                                   #
    # ------------------------------------------------------------------ #
    node_ids = list(network.nodes.keys())
    nodes = [network.nodes[pid] for pid in node_ids]

    node_xs = [n.avg_x for n in nodes]
    node_ys = [n.avg_y for n in nodes]

    involvements = [n.pass_count + n.receive_count for n in nodes]
    max_inv = max(involvements) if involvements else 1
    node_sizes = [
        _NODE_MIN_SIZE + (inv / max_inv) * (_NODE_MAX_SIZE - _NODE_MIN_SIZE)
        for inv in involvements
    ]

    node_labels = [
        str(n.jersey_num) if n.jersey_num is not None else str(n.player_id)
        for n in nodes
    ]

    hover_texts = []
    for pid, node in zip(node_ids, nodes):
        name_part = f" {node.name}" if node.name else ""
        parts = [f"<b>#{node.jersey_num or pid}{name_part}</b>"]
        parts.append(f"Passes sent: {node.pass_count} | Received: {node.receive_count}")
        if metrics and pid in metrics.players:
            pm = metrics.players[pid]
            parts.append(f"Degree: {pm.degree_centrality:.3f}")
            parts.append(f"Betweenness: {pm.betweenness_centrality:.3f}")
            parts.append(f"PageRank: {pm.pagerank:.3f}")
        hover_texts.append("<br>".join(parts))

    # ------------------------------------------------------------------ #
    # Step 4 — Plain node trace (always present, visible by default)      #
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Scatter(
            x=node_xs,
            y=node_ys,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=_NODE_DEFAULT_COLOR,
                line=dict(color=_NODE_BORDER_COLOR, width=_NODE_BORDER_WIDTH),
                showscale=False,
            ),
            text=node_labels,
            textfont=dict(color="white", size=10),
            textposition="middle center",
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            visible=True,
        )
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Metric node traces (only when metrics are provided)        #
    # ------------------------------------------------------------------ #
    if metrics is None:
        # No buttons — minimal top margin for title only
        fig.update_layout(autosize=True, margin=dict(l=10, r=10, t=55, b=10))
        _apply_title(fig, title)
        return fig

    for metric_key in ("degree", "betweenness", "pagerank"):
        attr = _METRIC_ATTRS[metric_key]
        color_values = [
            getattr(metrics.players[pid], attr, 0.0)
            if pid in metrics.players else 0.0
            for pid in node_ids
        ]
        fig.add_trace(
            go.Scatter(
                x=node_xs,
                y=node_ys,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=color_values,
                    colorscale=_METRIC_COLORSCALE,
                    cmin=0.0,
                    cmax=1.0,
                    colorbar=dict(
                        title=dict(
                            text=_METRIC_LABELS[metric_key],
                            side="right",
                            font=dict(color="rgba(255,255,255,0.8)", size=11),
                        ),
                        tickfont=dict(color="rgba(255,255,255,0.7)", size=9),
                        thickness=12,
                        len=0.6,
                        x=1.02,
                    ),
                    line=dict(color=_NODE_BORDER_COLOR, width=_NODE_BORDER_WIDTH),
                    showscale=True,
                ),
                text=node_labels,
                textfont=dict(color="white", size=10),
                textposition="middle center",
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False,
                visible=False,
            )
        )

    # ------------------------------------------------------------------ #
    # Step 6 — Metric selector buttons                                    #
    # ------------------------------------------------------------------ #
    # Visibility array: [pitch+edge traces (always True)] + [4 node traces]
    pre = [True] * n_pre_node

    def _vis(plain: bool, degree: bool, betweenness: bool, pagerank: bool) -> list:
        return pre + [plain, degree, betweenness, pagerank]

    # Responsive layout: extra top margin for the button bar + title;
    # extra right margin for the metric colorbar.
    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=60, t=70, b=10),
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                xanchor="center",
                y=1.12,
                yanchor="top",
                pad=dict(r=10, t=5),
                buttons=[
                    dict(
                        label="Plain",
                        method="restyle",
                        args=[{"visible": _vis(True, False, False, False)}],
                    ),
                    dict(
                        label="Degree",
                        method="restyle",
                        args=[{"visible": _vis(False, True, False, False)}],
                    ),
                    dict(
                        label="Betweenness",
                        method="restyle",
                        args=[{"visible": _vis(False, False, True, False)}],
                    ),
                    dict(
                        label="PageRank",
                        method="restyle",
                        args=[{"visible": _vis(False, False, False, True)}],
                    ),
                ],
                active=0,
                showactive=True,
                bgcolor=_BTN_BG,
                font=dict(color="rgba(255,255,255,0.9)", size=11),
                bordercolor=_BTN_BORDER,
                borderwidth=1,
            )
        ]
    )

    _apply_title(fig, title)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_title(fig: go.Figure, title: str | None) -> None:
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color="rgba(255,255,255,0.9)", size=14),
                x=0.5,
                xanchor="center",
            )
        )
