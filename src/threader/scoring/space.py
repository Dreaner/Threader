"""Space available — how much room does the receiver have?

Simply the distance (in meters) to the nearest opponent.
"""

from __future__ import annotations

from threader.geometry.distance import player_distance
from threader.models import Player


def space_available(
    receiver: Player,
    opponents: list[Player],
) -> float:
    """Distance to the nearest opponent (meters).

    Returns 0.0 if no opponents are given (shouldn't happen in practice).
    """
    if not opponents:
        return 0.0

    return min(player_distance(receiver, opp) for opp in opponents)
