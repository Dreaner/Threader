"""
Project: Threader
Author: Xingnan Zhu
File Name: completion.py
Description:
    Completion probability — can this pass reach its target?
    Factors:
      1. Distance decay (longer passes are harder)
      2. Defender blocking (passing lane obstruction)
"""

from __future__ import annotations

from threader.core.models import Player
from threader.geometry.distance import player_distance
from threader.geometry.passing_lane import passing_lane_blocking


def _base_probability(distance: float) -> float:
    """Distance-based completion probability.

    Short passes (~10m) succeed ~95% of the time.
    Long passes (30m+) drop to ~40-70%.
    """
    if distance < 10:
        return 0.95
    elif distance < 20:
        return 0.85
    elif distance < 30:
        return 0.70
    else:
        return max(0.40, 0.90 - distance * 0.015)


def completion_probability(
    passer: Player,
    receiver: Player,
    defenders: list[Player],
) -> float:
    """Calculate the probability that a pass from passer to receiver succeeds.

    Returns a value in [0, 1].
    """
    dist = player_distance(passer, receiver)
    base = _base_probability(dist)
    blocking = passing_lane_blocking(passer, receiver, defenders)

    return base * (1.0 - blocking)
