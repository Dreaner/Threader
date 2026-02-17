"""
Project: Threader
File Created: 2026-02-16 23:11:04
Author: Xingnan Zhu
File Name: analyzer.py
Description: 
    Main analysis entry point.
    Takes a snapshot and a passer, evaluates all possible pass targets,
    and returns ranked pass options.
"""

from __future__ import annotations

from threader.models import AnalysisResult, PassEvent, Player, Snapshot
from threader.scoring.pass_score import score_pass_option


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

    options = [
        score_pass_option(
            passer=passer,
            target=teammate,
            defenders=defenders,
            pitch_length=snapshot.pitch_length,
            pitch_width=snapshot.pitch_width,
            attack_direction=attack_dir,
        )
        for teammate in teammates
    ]

    return AnalysisResult(
        passer=passer,
        snapshot=snapshot,
        options=options,
    )


def analyze_pass_event(pass_event: PassEvent) -> AnalysisResult:
    """Analyze a PFF pass event — find what the optimal pass should have been.

    Locates the passer in the snapshot, then evaluates all teammates.

    Args:
        pass_event: A PassEvent extracted from PFF event data.

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
