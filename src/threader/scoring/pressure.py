"""Receiving pressure — will the receiver be under pressure?

Calculated from the weighted distance of the 3 nearest defenders.
Range: 0 (no pressure) → 10 (extreme pressure).
"""

from __future__ import annotations

from threader.geometry.distance import player_distance
from threader.models import Player


def receiving_pressure(
    receiver: Player,
    defenders: list[Player],
) -> float:
    """Calculate the pressure on a receiver from nearby defenders.

    Uses the 3 nearest defenders with decreasing weights:
      1st nearest: weight 1.0
      2nd nearest: weight 0.5
      3rd nearest: weight 0.33

    Returns a value in [0, 10].
    """
    if not defenders:
        return 0.0

    distances = sorted(
        (player_distance(receiver, d) for d in defenders),
    )
    nearest_3 = distances[:3]

    pressure = 0.0
    for i, dist in enumerate(nearest_3):
        weight = 1.0 / (i + 1)

        if dist < 2:
            pressure += 5.0 * weight
        elif dist < 5:
            pressure += 3.0 * weight
        elif dist < 10:
            pressure += 1.0 * weight
        else:
            pressure += 0.3 * weight

    return min(10.0, pressure)
