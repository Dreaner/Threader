"""Penetration score — how much forward progress does this pass achieve?

Two sub-dimensions:
  A. Forward distance (x-gain capped at 20m → score 1.0)
  B. Defenders bypassed (each defender adds 0.15, capped at 0.5)

Final score capped at 1.0.
"""

from __future__ import annotations

from threader.models import Player


def penetration_score(
    passer: Player,
    receiver: Player,
    defenders: list[Player],
) -> float:
    """Calculate the penetration score for a pass.

    Returns a value in [0, 1].
    """
    # A. Forward distance
    x_gain = receiver.x - passer.x

    if x_gain <= 0:
        forward_score = 0.0
    else:
        forward_score = min(1.0, x_gain / 20.0)

    # B. Defenders bypassed
    defenders_passed = sum(
        1 for d in defenders if passer.x < d.x < receiver.x
    )
    penetration_bonus = min(0.5, defenders_passed * 0.15)

    return min(1.0, forward_score + penetration_bonus)
