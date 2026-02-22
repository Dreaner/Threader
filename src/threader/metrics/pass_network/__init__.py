"""
Pass Network metric — aggregated pass structure analysis.

Builds a directed weighted graph of a team's passing relationships
across a match or period.  Nodes are players positioned by their
average location; edges encode pass counts and completion rates.

compute_metrics() derives graph-theoretic indicators (density, degree
centrality, betweenness centrality, PageRank) from a built PassNetwork.
"""

from threader.metrics.pass_network.builder import build_pass_network
from threader.metrics.pass_network.metrics import compute_metrics
from threader.metrics.pass_network.models import (
    NetworkMetrics,
    PassEdge,
    PassNetwork,
    PlayerMetrics,
    PlayerNode,
)

__all__ = [
    "NetworkMetrics",
    "PassEdge",
    "PassNetwork",
    "PlayerMetrics",
    "PlayerNode",
    "build_pass_network",
    "compute_metrics",
]
