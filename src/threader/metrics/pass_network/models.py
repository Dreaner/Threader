"""
Project: Threader
Author: Xingnan Zhu
File Name: pass_network/models.py
Description:
    Data models for pass network analysis.
    A PassNetwork is a directed weighted graph: nodes are players,
    edges are pass relationships aggregated over a match or period.

    NetworkMetrics and PlayerMetrics hold computed graph-theoretic indicators
    derived from a PassNetwork by compute_metrics() in metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerNode:
    """A player in the pass network, positioned by their average location.

    avg_x / avg_y are computed by averaging all snapshot positions for this
    player across every pass event the network was built from.  This gives
    a stable estimate of the player's typical operating zone.

    Coordinates use PFF center-origin: x ∈ [-52.5, 52.5], y ∈ [-34, 34].
    """

    player_id: int
    team_id: int
    name: str | None
    jersey_num: int | None
    position: str | None  # e.g. "GK", "CB", "CF"

    avg_x: float
    avg_y: float

    pass_count: int    # outgoing pass attempts (all, regardless of completion)
    receive_count: int # times this player was the intended target


@dataclass(frozen=True)
class PassEdge:
    """A directed pass relationship between two players.

    Represents all passes from passer_id to receiver_id that matched the
    builder's filter (period, completed_only, etc.).
    """

    passer_id: int
    receiver_id: int  # intended target (target_id from PassEvent)

    count: int      # total passes matching the filter on this edge
    completed: int  # subset of count that were completed

    @property
    def completion_rate(self) -> float:
        """Fraction of passes on this edge that were completed."""
        return self.completed / self.count if self.count > 0 else 0.0


@dataclass(frozen=True)
class PassNetwork:
    """A directed weighted pass network for one team in one match (or period).

    nodes: player_id  → PlayerNode
    edges: (passer_id, receiver_id) → PassEdge
    """

    game_id: int
    team_id: int
    period: int | None  # None = full match

    nodes: dict[int, PlayerNode]                  # player_id → PlayerNode
    edges: dict[tuple[int, int], PassEdge]        # (passer_id, receiver_id) → PassEdge

    @property
    def total_passes(self) -> int:
        """Total passes matching the network's filter."""
        return sum(e.count for e in self.edges.values())

    @property
    def total_completed(self) -> int:
        return sum(e.completed for e in self.edges.values())

    @property
    def completion_rate(self) -> float:
        total = self.total_passes
        return self.total_completed / total if total > 0 else 0.0

    def top_combinations(self, top_n: int = 5) -> list[PassEdge]:
        """Most frequent pass combinations (by count), descending."""
        return sorted(self.edges.values(), key=lambda e: e.count, reverse=True)[:top_n]

    def most_involved(self, top_n: int = 5) -> list[PlayerNode]:
        """Players with most total pass involvements (sent + received)."""
        return sorted(
            self.nodes.values(),
            key=lambda n: n.pass_count + n.receive_count,
            reverse=True,
        )[:top_n]


# ---------------------------------------------------------------------------
# Network metrics output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerMetrics:
    """Graph-theoretic indicators for a single player in the pass network.

    All values are normalized to [0, 1] for easy cross-player comparison.
    """

    player_id: int

    degree_centrality: float
    """Fraction of possible pass partnerships this player has.

    degree = (unique_out_edges + unique_in_edges) / (2 × (n − 1))

    High → passes to / receives from many different teammates.
    Low  → very few unique passing relationships (e.g. isolated GK).
    """

    betweenness_centrality: float
    """Fraction of shortest passing paths that route through this player.

    Shortest paths are computed on the weighted graph where edge distance =
    1 / edge.count (more passes = shorter / stronger connection).

    Normalized by (n−1)(n−2) so the maximum possible value is 1.0.

    High → essential relay player; removing them would fragment flow.
    Low  → peripheral; bypassed by most main passing routes.
    """

    pagerank: float
    """Relative importance based on who passes to this player (0–1, normalized).

    Computed with damping factor 0.85, edge weight = edge.count.
    Normalized so the highest-ranked player = 1.0.

    High → receives passes from players who themselves are well-connected.
    Low  → receives from few or low-importance teammates.
    """


@dataclass(frozen=True)
class NetworkMetrics:
    """Graph-theoretic summary of a team's pass network.

    Produced by compute_metrics(network) in metrics.py.
    """

    game_id: int
    team_id: int
    period: int | None

    density: float
    """Ratio of actual edges to the maximum possible edges: |E| / (n × (n−1)).

    Range [0, 1].  Higher = passing distributed across more routes.
    Lower = passing concentrated through a few dominant paths.
    """

    players: dict[int, PlayerMetrics]  # player_id → PlayerMetrics

    def top_hubs(self, top_n: int = 3) -> list[PlayerMetrics]:
        """Players with the highest degree centrality (most unique partnerships)."""
        return sorted(
            self.players.values(),
            key=lambda p: p.degree_centrality,
            reverse=True,
        )[:top_n]

    def top_connectors(self, top_n: int = 3) -> list[PlayerMetrics]:
        """Players with the highest betweenness centrality (key relay nodes)."""
        return sorted(
            self.players.values(),
            key=lambda p: p.betweenness_centrality,
            reverse=True,
        )[:top_n]

    def top_receivers(self, top_n: int = 3) -> list[PlayerMetrics]:
        """Players with the highest PageRank (focal points of team's passing)."""
        return sorted(
            self.players.values(),
            key=lambda p: p.pagerank,
            reverse=True,
        )[:top_n]
