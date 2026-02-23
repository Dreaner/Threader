"""
Project: PitchEcho
Author: Xingnan Zhu
File Name: penetration.py
Description:
    Penetration score — how much forward progress does this pass achieve?
    Two sub-dimensions:
      A. Forward distance (x-gain capped at 20m → score 1.0)
         Backward passes receive a mild negative drag (down to -0.3).
      B. Defenders bypassed (each defender adds 0.15, capped at 0.5)
    Final score range: [-0.3, 1.0].
"""

from __future__ import annotations

from pitch_echo.core.models import Player


def penetration_score(
    passer: Player,
    receiver: Player,
    defenders: list[Player],
    *,
    attack_direction: float = 1.0,
) -> float:
    """Calculate the penetration score for a pass.

    Args:
        attack_direction: +1.0 if the team attacks towards positive-x,
            -1.0 if towards negative-x.  All x-deltas are multiplied
            by this value so 'forward' always means towards the
            opponent's goal.

    Returns a value in [-0.3, 1.0].
    Forward passes earn positive scores (up to 1.0).
    Backward passes receive a mild negative drag (down to -0.3),
    proportional to backward distance.
    """
    # A. Forward distance (or backward drag)
    # Multiply by attack_direction so forward is always positive
    x_gain = (receiver.x - passer.x) * attack_direction

    if x_gain <= 0:
        # Mild backward drag: -0.3 at 40m back, ~-0.075 at 10m back
        forward_score = max(-0.3, x_gain / 40.0)
    else:
        forward_score = min(1.0, x_gain / 20.0)

    # B. Defenders bypassed
    # Use attack_direction to orient the comparison
    if attack_direction >= 0:
        defenders_passed = sum(
            1 for d in defenders if passer.x < d.x < receiver.x
        )
    else:
        defenders_passed = sum(
            1 for d in defenders if receiver.x < d.x < passer.x
        )
    penetration_bonus = min(0.5, defenders_passed * 0.15)

    # Clamp to [-0.3, 1.0] — backward passes keep their negative drag
    return max(-0.3, min(1.0, forward_score + penetration_bonus))
