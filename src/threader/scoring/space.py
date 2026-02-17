"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: space.py
Description: 
    Space available — how much room does the receiver have?
    Simply the distance (in meters) to the nearest opponent.
"""

from __future__ import annotations

from threader.geometry.distance import player_distance
from threader.models import Player


def space_available(
    receiver: Player,
    opponents: list[Player],
) -> float:
    """Distance to the nearest opponent (meters), capped at 15m.

    Values beyond 15m provide diminishing tactical benefit, so we cap
    to prevent excessive bonus for deep/isolated players (e.g. GK).

    Returns 0.0 if no opponents are given (shouldn't happen in practice).
    """
    if not opponents:
        return 0.0

    return min(15.0, min(player_distance(receiver, opp) for opp in opponents))
