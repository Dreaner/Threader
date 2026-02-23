"""
Project: PitchEcho
Author: Xingnan Zhu
File Name: pass_network/metrics.py
Description:
    Graph-theoretic network metrics computed from a PassNetwork.

    Algorithms implemented (pure Python, no external graph library):

    1. Density          — |E| / (n × (n−1))
    2. Degree centrality — (out_edges + in_edges) / (2 × (n−1))
    3. Betweenness centrality — Brandes-style, Dijkstra weighted by 1/count
    4. PageRank         — power iteration, edge weight = count, damping 0.85

    All player-level metrics are normalized to [0, 1].
"""

from __future__ import annotations

import heapq
from collections import defaultdict

from pitch_echo.network.models import (
    NetworkMetrics,
    PassNetwork,
    PlayerMetrics,
)


def compute_metrics(network: PassNetwork) -> NetworkMetrics:
    """Compute all graph-theoretic metrics from a PassNetwork.

    Args:
        network: A PassNetwork built by build_pass_network().

    Returns:
        NetworkMetrics containing density and per-player indicators.
    """
    density = _density(network)
    degree = _degree_centrality(network)
    betweenness = _betweenness_centrality(network)
    pr = _pagerank(network)

    players = {
        pid: PlayerMetrics(
            player_id=pid,
            degree_centrality=degree[pid],
            betweenness_centrality=betweenness.get(pid, 0.0),
            pagerank=pr[pid],
        )
        for pid in network.nodes
    }

    return NetworkMetrics(
        game_id=network.game_id,
        team_id=network.team_id,
        period=network.period,
        density=density,
        players=players,
    )


# ---------------------------------------------------------------------------
# Internal algorithm implementations
# ---------------------------------------------------------------------------


def _density(network: PassNetwork) -> float:
    """Network density: actual edges / maximum possible directed edges."""
    n = len(network.nodes)
    if n <= 1:
        return 0.0
    return len(network.edges) / (n * (n - 1))


def _degree_centrality(network: PassNetwork) -> dict[int, float]:
    """Normalized degree centrality for each player.

    degree = (unique out-edges + unique in-edges) / (2 × (n − 1))
    """
    n = len(network.nodes)
    if n <= 1:
        return {pid: 0.0 for pid in network.nodes}

    out_deg: dict[int, int] = defaultdict(int)
    in_deg: dict[int, int] = defaultdict(int)

    for src, dst in network.edges:
        out_deg[src] += 1
        in_deg[dst] += 1

    denom = 2 * (n - 1)
    return {
        pid: (out_deg[pid] + in_deg[pid]) / denom
        for pid in network.nodes
    }


def _betweenness_centrality(network: PassNetwork) -> dict[int, float]:
    """Betweenness centrality using Dijkstra on weighted directed graph.

    Edge distance = 1 / edge.count  (more passes → shorter distance → stronger connection).
    For each source node, compute shortest paths to all reachable nodes,
    track predecessors, then back-propagate dependency scores (Brandes 2001).

    Normalized by (n−1)(n−2) so the theoretical maximum is 1.0.
    """
    nodes = list(network.nodes.keys())
    n = len(nodes)
    if n <= 2:
        return {pid: 0.0 for pid in nodes}

    # Build adjacency: adj[u] = list of (v, distance=1/count)
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (src, dst), edge in network.edges.items():
        if edge.count > 0:
            adj[src].append((dst, 1.0 / edge.count))

    betweenness: dict[int, float] = defaultdict(float)

    for s in nodes:
        # --- Dijkstra from source s ---
        dist: dict[int, float] = {v: float("inf") for v in nodes}
        dist[s] = 0.0

        # sigma[v] = number of shortest paths from s to v
        sigma: dict[int, float] = defaultdict(float)
        sigma[s] = 1.0

        # pred[v] = list of predecessors on shortest paths from s
        pred: dict[int, list[int]] = defaultdict(list)

        # visited stack (in order of non-decreasing distance) for back-prop
        stack: list[int] = []

        # min-heap: (distance, node)
        heap: list[tuple[float, int]] = [(0.0, s)]

        visited: set[int] = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            stack.append(u)

            for v, w in adj[u]:
                alt = dist[u] + w
                if alt < dist[v] - 1e-12:
                    dist[v] = alt
                    sigma[v] = sigma[u]
                    pred[v] = [u]
                    heapq.heappush(heap, (alt, v))
                elif abs(alt - dist[v]) < 1e-12:
                    # Equal-length path — accumulate sigma and add predecessor
                    sigma[v] += sigma[u]
                    pred[v].append(u)

        # --- Back-propagation (Brandes dependency accumulation) ---
        delta: dict[int, float] = defaultdict(float)

        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalize by (n−1)(n−2) for directed graph
    denom = (n - 1) * (n - 2)
    return {
        pid: betweenness.get(pid, 0.0) / denom
        for pid in nodes
    }


def _pagerank(
    network: PassNetwork,
    damping: float = 0.85,
    iterations: int = 100,
) -> dict[int, float]:
    """PageRank with edge weight = edge.count, normalized to [0, 1].

    rank[v] = (1−d)/n + d × Σ_u (rank[u] × count(u→v) / out_weight[u])

    After convergence the values are divided by the maximum so the
    top-ranked player always scores 1.0.
    """
    nodes = list(network.nodes.keys())
    n = len(nodes)
    if n == 0:
        return {}

    rank: dict[int, float] = {pid: 1.0 / n for pid in nodes}

    # Total outgoing weight per node
    out_weight: dict[int, float] = defaultdict(float)
    for (src, _), edge in network.edges.items():
        out_weight[src] += edge.count

    # Pre-build inbound adjacency for efficiency
    # in_adj[v] = list of (u, count(u→v))
    in_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (src, dst), edge in network.edges.items():
        if edge.count > 0:
            in_adj[dst].append((src, edge.count))

    base = (1.0 - damping) / n

    for _ in range(iterations):
        new_rank: dict[int, float] = {}
        for v in nodes:
            incoming = sum(
                rank[u] * cnt / out_weight[u]
                for u, cnt in in_adj[v]
                if out_weight[u] > 0
            )
            new_rank[v] = base + damping * incoming
        rank = new_rank

    # Normalize to [0, 1]
    max_rank = max(rank.values()) if rank else 1.0
    if max_rank == 0:
        return {pid: 0.0 for pid in nodes}
    return {pid: v / max_rank for pid, v in rank.items()}
