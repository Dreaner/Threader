"""
Baseline models for comparison against the full Pass Score.

Each baseline assigns scores to pass options from a ValidatedPass,
producing a ranking that can be compared to ground truth.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from threader.geometry.distance import euclidean
from threader.metrics.pass_value.models import DEFAULT_WEIGHTS, ScoringWeights
from threader.metrics.pass_value.scoring.pass_score import _adjusted_zone, compute_pass_score

if TYPE_CHECKING:
    from threader.metrics.pass_value.models import PassOption
    from threader.validation.collector import ValidatedPass


# ---------------------------------------------------------------------------
# Scoring functions — each returns a list of (player_id, score) tuples
# sorted by score descending (best first).
# ---------------------------------------------------------------------------


def _rank(scored: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Sort (player_id, score) pairs descending by score."""
    return sorted(scored, key=lambda x: x[1], reverse=True)


def rank_random(vp: ValidatedPass) -> list[tuple[int, float]]:
    """Random ranking — shuffle teammates."""
    options = [(opt.target.player_id, random.random()) for opt in vp.ranked_options]
    return _rank(options)


def rank_distance(vp: ValidatedPass) -> list[tuple[int, float]]:
    """Distance-only — closest teammate gets highest score.

    Score = negative distance so that closer = higher.
    """
    options = [
        (opt.target.player_id, -euclidean(vp.passer_x, vp.passer_y, opt.target.x, opt.target.y))
        for opt in vp.ranked_options
    ]
    return _rank(options)


def rank_zone_value(vp: ValidatedPass) -> list[tuple[int, float]]:
    """Zone-value-only — rank by xT of target position."""
    return _rank([(opt.target.player_id, opt.zone_value) for opt in vp.ranked_options])


def rank_completion(vp: ValidatedPass) -> list[tuple[int, float]]:
    """Completion-only — rank by completion probability."""
    return _rank(
        [(opt.target.player_id, opt.completion_probability) for opt in vp.ranked_options]
    )


def rank_expected_value(vp: ValidatedPass) -> list[tuple[int, float]]:
    """Expected-value-only — completion × zone_value (no penetration/space/pressure)."""
    return _rank([
        (opt.target.player_id, opt.completion_probability * opt.zone_value)
        for opt in vp.ranked_options
    ])


def rank_pass_score(vp: ValidatedPass) -> list[tuple[int, float]]:
    """Full Pass Score — current production formula."""
    return _rank([(opt.target.player_id, opt.pass_score) for opt in vp.ranked_options])


def rank_with_weights(
    vp: ValidatedPass,
    weights: ScoringWeights,
) -> list[tuple[int, float]]:
    """Re-score all options with custom weights and rank.

    Uses the raw dimension values stored in each PassOption to recompute
    the Pass Score with different weights (avoids re-running geometry).

    When ``weights.relative_zone_weight > 0``, the stored ``zone_value``
    (raw xT, collected with alpha=0) is adjusted using the team context:
    ``adj_zone = raw_xT + alpha * (raw_xT - team_mean_xT)``.
    This requires the ValidatedPass to have ``team_mean_xT`` populated
    (available in caches collected after v3).
    """
    alpha = weights.relative_zone_weight
    team_mean = getattr(vp, "team_mean_xT", None)

    scored: list[tuple[int, float]] = []
    for opt in vp.ranked_options:
        adj_zone = _adjusted_zone(opt.zone_value, team_mean, alpha)
        score = compute_pass_score(
            comp=opt.completion_probability,
            zone=adj_zone,
            pressure=opt.receiving_pressure,
            space=opt.space_available,
            penetration=opt.penetration_score,
            weights=weights,
        )
        scored.append((opt.target.player_id, score))
    return _rank(scored)


# ---------------------------------------------------------------------------
# Named baselines registry
# ---------------------------------------------------------------------------

BASELINES: dict[str, str] = {
    "Random": "rank_random",
    "Distance-only": "rank_distance",
    "Zone-value-only": "rank_zone_value",
    "Completion-only": "rank_completion",
    "Expected-value": "rank_expected_value",
    "Pass Score (full)": "rank_pass_score",
}

_BASELINE_FUNCS = {
    "rank_random": rank_random,
    "rank_distance": rank_distance,
    "rank_zone_value": rank_zone_value,
    "rank_completion": rank_completion,
    "rank_expected_value": rank_expected_value,
    "rank_pass_score": rank_pass_score,
}


def get_baseline_func(name: str):
    """Return the ranking function for a named baseline."""
    func_name = BASELINES.get(name, name)
    return _BASELINE_FUNCS[func_name]


# ---------------------------------------------------------------------------
# Metrics computed against each baseline
# ---------------------------------------------------------------------------


def evaluate_baseline(
    records: list[ValidatedPass],
    rank_func,
    *,
    n_random_trials: int = 50,
    is_random: bool = False,
) -> dict[str, float]:
    """Evaluate a baseline ranking function on the validated passes.

    Returns:
        auc_roc: AUC-ROC for predicting pass completion (actual target's score).
        concordance: % of betterOption-annotated passes where the model
            ranks PFF's suggestion above the actual target.
        mrr: Mean Reciprocal Rank of PFF's betterOption player.
    """
    from scipy.stats import mannwhitneyu  # noqa: E402

    # For random baseline, average over multiple trials
    trials = n_random_trials if is_random else 1

    auc_sum = 0.0
    concordance_sum = 0.0
    mrr_sum = 0.0

    for _ in range(trials):
        # --- AUC-ROC for completion prediction ---
        completed_scores: list[float] = []
        defended_scores: list[float] = []

        # --- BetterOption concordance & MRR ---
        concord_total = 0
        concord_hits = 0
        rr_sum = 0.0

        for vp in records:
            ranking = rank_func(vp)
            # Map player_id → rank (1-based) and score
            id_to_rank = {pid: i + 1 for i, (pid, _) in enumerate(ranking)}
            id_to_score = {pid: s for pid, s in ranking}

            # Actual target's score in this ranking
            target_score = id_to_score.get(vp.actual_target_id, 0.0)
            if vp.actual_outcome == "C":
                completed_scores.append(target_score)
            else:
                defended_scores.append(target_score)

            # BetterOption analysis
            if vp.pff_better_option_player_id and vp.pff_better_option_player_id in id_to_rank:
                concord_total += 1
                bo_rank = id_to_rank[vp.pff_better_option_player_id]
                at_rank = id_to_rank.get(vp.actual_target_id, len(ranking))
                if bo_rank < at_rank:
                    concord_hits += 1
                rr_sum += 1.0 / bo_rank

        # AUC via Mann-Whitney U
        if completed_scores and defended_scores:
            try:
                u_stat, _ = mannwhitneyu(
                    completed_scores, defended_scores, alternative="greater"
                )
                n1, n2 = len(completed_scores), len(defended_scores)
                auc = u_stat / (n1 * n2)
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5

        concordance = concord_hits / concord_total if concord_total > 0 else 0.0
        mrr = rr_sum / concord_total if concord_total > 0 else 0.0

        auc_sum += auc
        concordance_sum += concordance
        mrr_sum += mrr

    return {
        "auc_roc": round(auc_sum / trials, 4),
        "concordance": round(concordance_sum / trials * 100, 1),
        "mrr": round(mrr_sum / trials, 4),
    }


def run_all_baselines(records: list[ValidatedPass]) -> dict[str, dict[str, float]]:
    """Evaluate all named baselines and return results keyed by name."""
    results: dict[str, dict[str, float]] = {}
    for name, func_name in BASELINES.items():
        func = _BASELINE_FUNCS[func_name]
        is_rnd = func_name == "rank_random"
        results[name] = evaluate_baseline(records, func, is_random=is_rnd)
    return results
