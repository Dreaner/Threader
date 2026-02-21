"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: pass_score.py
Description: 
    Pass Score — the master scoring formula combining all 5 dimensions.
    Pass Score = (
        completion_probability × zone_value × 4.27 +  # Expected value (amplified)
        penetration_score × 1.00 +                    # Penetration bonus/drag
        space_available × 0.0001                      # Space bonus (capped at 15m)
    ) × (1.0 − pressure/10 × 0.01)                   # Pressure as multiplier
      × 100
    Range: 0–100
    
    Key changes (v1.1):
    - zone_value amplified ×1.5 to widen gap between attacking/defensive positions
    - Pressure switched from additive penalty to multiplicative scaling
    - Penetration now returns negative for backward passes (mild drag)
    - Space capped at 15m upstream

    Key changes (v1.2 — Spearman-optimized weights):
    - Weights optimized via scipy.optimize.differential_evolution
      directly maximizing Spearman ρ(Pass Score, ΔxT) on 64-match dataset
    - zone_amplifier 1.5 → 4.27 (+185%): zone value is the dominant signal
    - penetration_weight 0.20 → 1.00 (+400%): forward progress matters more
    - space_weight 0.001 → 0.0001 (−90%): space is a minor tiebreaker
    - pressure_scaling 0.20 → 0.01 (−95%): pressure has minimal effect on ΔxT
    - Test Spearman ρ: 0.6318 → 0.6540 (+0.022)
    - Test AUC (lines-broken): 0.7533 → 0.7675 (+0.014)

    Key changes (v1.3 — team-context zone value):
    - Added relative_zone_weight (α): amplifies zone value for players above
      team mean xT, reduces for players below; formula:
      adj_zone = abs_xT + α * (abs_xT - team_mean_xT)
    - GK excluded from team mean calculation
    - All 5 weights re-optimized jointly on 64-match dataset
    - zone_amplifier 4.27 → 3.01 (−30%): relative component takes over part of signal
    - penetration_weight 1.00 → 0.46 (−54%): team context captures forward advance
    - relative_zone_weight: 0 → 1.19 (new parameter)
    - Test Spearman ρ: 0.6406 → 0.6795 (+0.039)
    - Test AUC (lines-broken): 0.7517 → 0.7766 (+0.025)
"""

from __future__ import annotations

from dataclasses import dataclass

from threader.models import PassOption, Player
from threader.scoring.completion import completion_probability
from threader.scoring.penetration import penetration_score
from threader.scoring.pressure import receiving_pressure
from threader.scoring.space import space_available
from threader.scoring.zone_value import zone_value


@dataclass(frozen=True)
class ScoringWeights:
    """Adjustable weights for the Pass Score formula.

    Used by sensitivity analysis to sweep parameters.
    Default values are Spearman-optimized (v1.2) via
    ``scripts/optimize_weights.py``.
    """

    zone_amplifier: float = 3.005181
    penetration_weight: float = 0.459953
    space_weight: float = 0.000101
    pressure_scaling: float = 0.010001
    relative_zone_weight: float = 1.194279  # α: team-context zone adjustment


# Default weights — used when no overrides are supplied.
DEFAULT_WEIGHTS = ScoringWeights()


def _adjusted_zone(abs_zone: float, team_mean_xT: float | None, alpha: float) -> float:
    """Apply team-context scaling to an absolute xT value.

    Formula: abs_zone + alpha * (abs_zone - team_mean_xT)
    Equivalent to: abs_zone * (1 + alpha) - alpha * team_mean_xT

    When the receiver is above the team average, the zone value is amplified.
    When below, it is reduced. alpha=0 returns abs_zone unchanged.
    Result is clamped to 0.0 to prevent negative zone values.
    """
    if team_mean_xT is None or alpha == 0.0:
        return abs_zone
    return max(0.0, abs_zone + alpha * (abs_zone - team_mean_xT))


def compute_pass_score(
    comp: float,
    zone: float,
    pressure: float,
    space: float,
    penetration: float,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> float:
    """Compute the final Pass Score (0–100) from pre-calculated dimensions.

    This is the pure formula, separated for reuse in sensitivity analysis
    without re-computing the individual dimension scores.
    """
    base_score = (
        comp * zone * weights.zone_amplifier
        + penetration * weights.penetration_weight
        + space * weights.space_weight
    )
    pressure_factor = 1.0 - (pressure / 10.0) * weights.pressure_scaling
    return max(0.0, min(100.0, base_score * pressure_factor * 100.0))


def score_pass_option(
    passer: Player,
    target: Player,
    defenders: list[Player],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    attack_direction: float = 1.0,
    weights: ScoringWeights | None = None,
    team_mean_xT: float | None = None,
) -> PassOption:
    """Score a single pass option from passer to target.

    Args:
        attack_direction: +1.0 if the passer's team attacks towards
            positive-x, -1.0 if towards negative-x.  Used to orient
            zone_value and penetration_score correctly.
        weights: Optional ScoringWeights override for sensitivity analysis.
            When None, uses DEFAULT_WEIGHTS.
        team_mean_xT: Mean xT of the passer's outfield teammates (GK excluded).
            When provided and ``weights.relative_zone_weight > 0``, the zone
            value is adjusted by how much the target exceeds the team average.
            When None, zone_value is used as-is (original behaviour).

    Returns a PassOption with all 5 dimension scores and the
    final Pass Score (0–100).  ``zone_value`` in the returned PassOption
    reflects the team-context-adjusted value when ``team_mean_xT`` is supplied.
    """
    w = weights or DEFAULT_WEIGHTS

    comp = completion_probability(passer, target, defenders)
    zone = zone_value(
        target.x, target.y, pitch_length, pitch_width,
        attack_direction=attack_direction,
    )
    adj_zone = _adjusted_zone(zone, team_mean_xT, w.relative_zone_weight)
    pressure = receiving_pressure(target, defenders)
    space = space_available(target, defenders)
    penetration = penetration_score(
        passer, target, defenders,
        attack_direction=attack_direction,
    )

    final_score = compute_pass_score(comp, adj_zone, pressure, space, penetration, w)

    return PassOption(
        target=target,
        pass_score=round(final_score, 1),
        completion_probability=round(comp, 3),
        zone_value=round(adj_zone, 4),
        receiving_pressure=round(pressure, 1),
        space_available=round(space, 1),
        penetration_score=round(penetration, 2),
    )
