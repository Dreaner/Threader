"""
Project: Threader
Author: Xingnan Zhu
File Name: analyzer.py
Description:
    Main analysis entry point for the Pass Value metric.
    Takes a snapshot and a passer, evaluates all possible pass targets,
    and returns ranked pass options.
"""

from __future__ import annotations

from threader.core.models import Player, Snapshot
from threader.metrics.pass_value.models import AnalysisResult
from threader.metrics.pass_value.scoring.pass_score import score_pass_option
from threader.metrics.pass_value.scoring.zone_value import zone_value


def analyze_snapshot(
    snapshot: Snapshot,
    passer: Player,
) -> AnalysisResult:
    """Analyze all pass options for a given passer in a snapshot.

    Evaluates every teammate as a potential pass target, scores each
    option, and returns them ranked by Pass Score (highest first).

    Args:
        snapshot: The frozen moment with all player positions.
        passer: The ball carrier.

    Returns:
        AnalysisResult with all scored and ranked pass options.
    """
    teammates = snapshot.teammates(passer)
    defenders = snapshot.opponents(passer)
    attack_dir = snapshot.attack_direction(passer)

    team_mean_xT = _compute_team_mean_xT(
        teammates,
        pitch_length=snapshot.pitch_length,
        pitch_width=snapshot.pitch_width,
        attack_direction=attack_dir,
    )

    options = [
        score_pass_option(
            passer=passer,
            target=teammate,
            defenders=defenders,
            pitch_length=snapshot.pitch_length,
            pitch_width=snapshot.pitch_width,
            attack_direction=attack_dir,
            team_mean_xT=team_mean_xT,
        )
        for teammate in teammates
    ]

    return AnalysisResult(
        passer=passer,
        snapshot=snapshot,
        options=options,
        team_mean_xT=team_mean_xT,
    )


def _compute_team_mean_xT(
    teammates: list[Player],
    pitch_length: float,
    pitch_width: float,
    attack_direction: float,
) -> float | None:
    """Compute the mean xT of outfield teammates (GK excluded).

    Returns None if there are no eligible players, which causes
    ``score_pass_option`` to skip the team-context adjustment.
    """
    eligible = [p for p in teammates if (p.position or "").upper() != "GK"]
    if not eligible:
        return None
    total = sum(
        zone_value(
            p.x, p.y, pitch_length, pitch_width,
            attack_direction=attack_direction,
        )
        for p in eligible
    )
    return total / len(eligible)


def analyze_pass_event(pass_event) -> AnalysisResult:
    """Analyze a pass event — find what the optimal pass should have been.

    Locates the passer in the snapshot, then evaluates all teammates.

    Args:
        pass_event: A pass event with ``snapshot``, ``passer_id``, and
            ``passer_name`` attributes.

    Returns:
        AnalysisResult with ranked pass options.

    Raises:
        ValueError: If the passer cannot be found in the snapshot.
    """
    snapshot = pass_event.snapshot
    passer = _find_player(snapshot, pass_event.passer_id)

    if passer is None:
        raise ValueError(
            f"Passer {pass_event.passer_name} (ID {pass_event.passer_id}) "
            f"not found in snapshot"
        )

    return analyze_snapshot(snapshot, passer)


def _find_player(snapshot: Snapshot, player_id: int) -> Player | None:
    """Find a player in the snapshot by ID."""
    for player in snapshot.all_players:
        if player.player_id == player_id:
            return player
    return None
