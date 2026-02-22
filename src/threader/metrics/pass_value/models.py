"""
Project: Threader
Author: Xingnan Zhu
File Name: metrics/pass_value/models.py
Description:
    Data models specific to the Pass Value metric:
    PassOption, AnalysisResult, and ScoringWeights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from threader.core.models import Player, Snapshot


@dataclass(frozen=True)
class PassOption:
    """A scored pass option — one possible target for the ball carrier."""

    target: Player
    pass_score: float  # 0–100
    completion_probability: float  # 0–1
    zone_value: float  # xT value at target position
    receiving_pressure: float  # 0–10
    space_available: float  # meters
    penetration_score: float  # 0–1

    @property
    def rank_key(self) -> float:
        """Key for sorting (higher is better)."""
        return self.pass_score


@dataclass
class AnalysisResult:
    """Complete analysis of a passing moment.

    Contains the passer, the snapshot, and all scored pass options
    ranked from best to worst.
    """

    passer: Player
    snapshot: Snapshot
    options: list[PassOption] = field(default_factory=list)
    team_mean_xT: float | None = None  # Mean xT of outfield teammates (GK excluded)

    @property
    def ranked_options(self) -> list[PassOption]:
        """Options sorted by Pass Score, highest first."""
        return sorted(self.options, key=lambda o: o.rank_key, reverse=True)

    @property
    def best_option(self) -> PassOption | None:
        ranked = self.ranked_options
        return ranked[0] if ranked else None


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
