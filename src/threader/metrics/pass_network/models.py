"""
Project: Threader
Author: Xingnan Zhu
File Name: pass_network/models.py
Description:
    Data models for pass network analysis.
    A PassNetwork is a directed weighted graph: nodes are players,
    edges are pass relationships aggregated over a match or period.
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
