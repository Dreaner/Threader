"""
Project: PitchEcho
Author: Xingnan Zhu
File Name: pass_network/builder.py
Description:
    Builds a PassNetwork from a list of PFF PassEvent objects.

    Algorithm:
    1. Filter events: passer's team == team_id, optionally by period/outcome.
    2. Accumulate player positions from ALL team players in every filtered
       snapshot.  x is normalized to "team attacks positive-x" before averaging
       so that positions from Period 1 and Period 2 are comparable.
    3. Count outgoing (pass_count) and incoming (receive_count) per player.
    4. Build edges: (passer_id, target_id) → PassEdge with count + completed.
    5. Assemble and return PassNetwork.
"""

from __future__ import annotations

from collections import defaultdict

from pitch_echo.data.pff.events import PassEvent
from pitch_echo.network.models import PassEdge, PassNetwork, PlayerNode


def _passer_team(event: PassEvent) -> int | None:
    """Return the team_id of the passer by looking them up in the snapshot."""
    for player in event.snapshot.all_players:
        if player.player_id == event.passer_id:
            return player.team_id
    return None


def build_pass_network(
    events: list[PassEvent],
    team_id: int,
    *,
    period: int | None = None,
    completed_only: bool = True,
) -> PassNetwork:
    """Build a PassNetwork for one team from a list of pass events.

    Args:
        events: All pass events from a match (from extract_pass_events()).
        team_id: The team to analyze — filters events where the passer
            belongs to this team.
        period: If set, only include events from this period.
            None (default) = aggregate the full match.
        completed_only: If True (default), only edges for completed passes
            are counted in the network.  Node position and involvement
            counts (pass_count / receive_count) always include all attempts.

    Returns:
        PassNetwork for the specified team and filter settings.
    """
    game_id = events[0].game_id if events else 0

    # ------------------------------------------------------------------ #
    # Step 1 — Filter to this team's pass events in the requested period  #
    # ------------------------------------------------------------------ #
    team_events = [
        e for e in events
        if _passer_team(e) == team_id
        and (period is None or e.period == period)
    ]

    if not team_events:
        return PassNetwork(
            game_id=game_id,
            team_id=team_id,
            period=period,
            nodes={},
            edges={},
        )

    # ------------------------------------------------------------------ #
    # Step 2 — Accumulate player positions from all filtered snapshots    #
    # ------------------------------------------------------------------ #
    # pos_accum: player_id → [sum_x, sum_y, count]
    pos_accum: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    # Collect player metadata (name, jersey_num, position).
    # Snapshot Player objects have jersey_num/position but name is None in PFF;
    # we supplement from PassEvent.passer_name / target_name below.
    player_meta: dict[int, tuple[str | None, int | None, str | None]] = {}

    for event in team_events:
        team_players = [p for p in event.snapshot.all_players if p.team_id == team_id]
        if not team_players:
            continue
        # Normalize x to "team attacks positive-x" so that Period 1 and
        # Period 2 positions are comparable before averaging.
        # attack_direction returns +1.0 if the team currently attacks toward
        # positive-x, -1.0 if toward negative-x.
        attack_dir = event.snapshot.attack_direction(team_players[0])
        for p in team_players:
            acc = pos_accum[p.player_id]
            # PFF rotates the entire coordinate frame 180° in P2, so both x
            # and y flip.  Multiply both by attack_dir to get a consistent
            # "team attacks positive-x, left-flank positive-y" frame.
            acc[0] += p.x * attack_dir
            acc[1] += p.y * attack_dir
            acc[2] += 1.0
            # Keep the most-informative metadata seen so far
            existing = player_meta.get(p.player_id, (None, None, None))
            player_meta[p.player_id] = (
                existing[0],
                p.jersey_num if p.jersey_num is not None else existing[1],
                p.position if p.position is not None else existing[2],
            )

    # Supplement player names from event-level fields (more reliable)
    for event in team_events:
        for pid, name in (
            (event.passer_id, event.passer_name),
            (event.target_id, event.target_name),
        ):
            existing = player_meta.get(pid, (None, None, None))
            if existing[0] is None and name:
                player_meta[pid] = (name, existing[1], existing[2])

    # ------------------------------------------------------------------ #
    # Step 3 — Count pass involvements (always all attempts)              #
    # ------------------------------------------------------------------ #
    pass_count: dict[int, int] = defaultdict(int)
    receive_count: dict[int, int] = defaultdict(int)

    for event in team_events:
        pass_count[event.passer_id] += 1
        receive_count[event.target_id] += 1

    # ------------------------------------------------------------------ #
    # Step 4 — Build edges (respects completed_only filter)               #
    # ------------------------------------------------------------------ #
    edge_total: dict[tuple[int, int], int] = defaultdict(int)
    edge_done: dict[tuple[int, int], int] = defaultdict(int)

    for event in team_events:
        if completed_only and not event.is_complete:
            continue
        key = (event.passer_id, event.target_id)
        edge_total[key] += 1
        if event.is_complete:
            edge_done[key] += 1

    # ------------------------------------------------------------------ #
    # Step 5 — Assemble nodes and edges                                   #
    # ------------------------------------------------------------------ #
    nodes: dict[int, PlayerNode] = {}
    for pid, acc in pos_accum.items():
        cnt = acc[2]
        if cnt == 0:
            continue
        name, jersey_num, position = player_meta.get(pid, (None, None, None))
        nodes[pid] = PlayerNode(
            player_id=pid,
            team_id=team_id,
            name=name,
            jersey_num=jersey_num,
            position=position,
            avg_x=acc[0] / cnt,
            avg_y=acc[1] / cnt,
            pass_count=pass_count[pid],
            receive_count=receive_count[pid],
        )

    edges: dict[tuple[int, int], PassEdge] = {
        key: PassEdge(
            passer_id=key[0],
            receiver_id=key[1],
            count=total,
            completed=edge_done[key],
        )
        for key, total in edge_total.items()
    }

    return PassNetwork(
        game_id=game_id,
        team_id=team_id,
        period=period,
        nodes=nodes,
        edges=edges,
    )
