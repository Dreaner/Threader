"""
Sensitivity analysis — weight perturbation, dimension ablation,
and pressure-specific deep dive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from threader.scoring.pass_score import DEFAULT_WEIGHTS, ScoringWeights, compute_pass_score
from threader.validation.baselines import rank_with_weights

if TYPE_CHECKING:
    from threader.validation.collector import ValidatedPass


# ── Helpers ──────────────────────────────────────────────────────────────────


def _auc_from_records(
    records: list[ValidatedPass],
    rank_func,
) -> float:
    """Compute AUC-ROC for completion prediction using a ranking function."""
    completed_scores: list[float] = []
    defended_scores: list[float] = []

    for vp in records:
        ranking = rank_func(vp)
        id_to_score = {pid: s for pid, s in ranking}
        score = id_to_score.get(vp.actual_target_id, 0.0)
        if vp.actual_outcome == "C":
            completed_scores.append(score)
        else:
            defended_scores.append(score)

    if not completed_scores or not defended_scores:
        return 0.5

    try:
        u, _ = stats.mannwhitneyu(completed_scores, defended_scores, alternative="greater")
        return u / (len(completed_scores) * len(defended_scores))
    except ValueError:
        return 0.5


def _concordance_from_records(
    records: list[ValidatedPass],
    rank_func,
) -> float:
    """Compute PFF betterOption concordance rate."""
    total = 0
    hits = 0
    for vp in records:
        if not vp.pff_better_option_player_id:
            continue
        ranking = rank_func(vp)
        id_to_rank = {pid: i + 1 for i, (pid, _) in enumerate(ranking)}
        if vp.pff_better_option_player_id not in id_to_rank:
            continue
        total += 1
        bo_rank = id_to_rank[vp.pff_better_option_player_id]
        at_rank = id_to_rank.get(vp.actual_target_id, len(ranking))
        if bo_rank < at_rank:
            hits += 1
    return hits / total if total > 0 else 0.0


def _rank1_ids(
    records: list[ValidatedPass],
    rank_func,
) -> list[int]:
    """Get the #1-ranked player_id for each pass."""
    return [rank_func(vp)[0][0] for vp in records]


# ── A. Weight Perturbation ──────────────────────────────────────────────────


def weight_sweep(
    records: list[ValidatedPass],
    steps: int = 11,
) -> dict[str, list[dict]]:
    """Sweep each weight ±50% and measure AUC, concordance, rank-1 flip rate.

    Returns a dict keyed by weight name, each containing a list of
    {value, auc, concordance, flip_rate} dicts.
    """
    base_weights = DEFAULT_WEIGHTS
    weight_names = ["zone_amplifier", "penetration_weight", "space_weight", "pressure_scaling"]

    # Get baseline rank-1 IDs for flip rate computation
    def make_rank_func(w: ScoringWeights):
        return lambda vp: rank_with_weights(vp, w)

    baseline_rank1 = _rank1_ids(records, make_rank_func(base_weights))

    results: dict[str, list[dict]] = {}

    for wname in weight_names:
        default_val = getattr(base_weights, wname)
        lo = default_val * 0.5
        hi = default_val * 1.5
        sweep_values = np.linspace(lo, hi, steps)

        sweep_row: list[dict] = []
        for val in sweep_values:
            w = ScoringWeights(**{**{f: getattr(base_weights, f) for f in weight_names}, wname: val})
            rfunc = make_rank_func(w)

            auc = _auc_from_records(records, rfunc)
            concordance = _concordance_from_records(records, rfunc)
            trial_rank1 = _rank1_ids(records, rfunc)
            n_flips = sum(1 for a, b in zip(baseline_rank1, trial_rank1) if a != b)
            flip_rate = n_flips / len(records) if records else 0.0

            sweep_row.append({
                "value": round(float(val), 6),
                "auc": round(auc, 4),
                "concordance": round(concordance * 100, 1),
                "flip_rate": round(flip_rate * 100, 1),
            })

        results[wname] = sweep_row

    return results


# ── B. Dimension Ablation ────────────────────────────────────────────────────


def dimension_ablation(records: list[ValidatedPass]) -> dict[str, dict]:
    """Remove each dimension one at a time and measure impact.

    Ablation is done via extreme weight settings:
    - zone_amplifier → 0 (removes zone value from expected value)
    - penetration_weight → 0
    - space_weight → 0
    - pressure_scaling → 0 (removes pressure dampening)
    Also tests completion ablation by setting zone_amplifier=0) which
    effectively zeroes the expected-value term.

    Returns dict keyed by ablated dimension name.
    """
    base = DEFAULT_WEIGHTS
    weight_names = ["zone_amplifier", "penetration_weight", "space_weight", "pressure_scaling"]

    def make_rank_func(w: ScoringWeights):
        return lambda vp: rank_with_weights(vp, w)

    # Baseline metrics
    baseline_func = make_rank_func(base)
    base_auc = _auc_from_records(records, baseline_func)
    base_concord = _concordance_from_records(records, baseline_func)
    baseline_rank1 = _rank1_ids(records, baseline_func)

    results: dict[str, dict] = {
        "baseline": {
            "auc": round(base_auc, 4),
            "concordance": round(base_concord * 100, 1),
        }
    }

    # Map dimension name → which weight to zero
    ablation_map = {
        "zone_value": "zone_amplifier",
        "penetration": "penetration_weight",
        "space": "space_weight",
        "pressure": "pressure_scaling",
    }

    for dim_name, wname in ablation_map.items():
        w = ScoringWeights(**{**{f: getattr(base, f) for f in weight_names}, wname: 0.0})
        rfunc = make_rank_func(w)
        auc = _auc_from_records(records, rfunc)
        concordance = _concordance_from_records(records, rfunc)
        trial_rank1 = _rank1_ids(records, rfunc)
        n_flips = sum(1 for a, b in zip(baseline_rank1, trial_rank1) if a != b)
        flip_rate = n_flips / len(records) if records else 0.0

        results[dim_name] = {
            "auc": round(auc, 4),
            "auc_delta": round(auc - base_auc, 4),
            "concordance": round(concordance * 100, 1),
            "concordance_delta": round((concordance - base_concord) * 100, 1),
            "flip_rate": round(flip_rate * 100, 1),
        }

    return results


# ── C. Pressure-Specific Deep Dive ──────────────────────────────────────────


def pressure_deep_dive(records: list[ValidatedPass]) -> dict:
    """Explore pressure parameter space for optimal configuration.

    Varies:
    1. pressure_scaling k in (1 - pressure/10 × k): 0.00 → 0.40
    2. Compare AUC improvement vs no-pressure model

    Returns sweep results.
    """
    weight_names = ["zone_amplifier", "penetration_weight", "space_weight", "pressure_scaling"]
    base = DEFAULT_WEIGHTS

    def make_rank_func(w: ScoringWeights):
        return lambda vp: rank_with_weights(vp, w)

    # Baseline (no pressure)
    no_pressure = ScoringWeights(
        zone_amplifier=base.zone_amplifier,
        penetration_weight=base.penetration_weight,
        space_weight=base.space_weight,
        pressure_scaling=0.0,
    )
    no_pressure_auc = _auc_from_records(records, make_rank_func(no_pressure))
    no_pressure_concord = _concordance_from_records(records, make_rank_func(no_pressure))

    # Sweep pressure_scaling from 0 to 0.40
    k_values = np.linspace(0.0, 0.40, 21)
    sweep: list[dict] = []

    for k in k_values:
        w = ScoringWeights(
            zone_amplifier=base.zone_amplifier,
            penetration_weight=base.penetration_weight,
            space_weight=base.space_weight,
            pressure_scaling=float(k),
        )
        rfunc = make_rank_func(w)
        auc = _auc_from_records(records, rfunc)
        concordance = _concordance_from_records(records, rfunc)
        sweep.append({
            "k": round(float(k), 3),
            "auc": round(auc, 4),
            "auc_delta_vs_no_pressure": round(auc - no_pressure_auc, 4),
            "concordance": round(concordance * 100, 1),
        })

    # Find best k
    best_by_auc = max(sweep, key=lambda x: x["auc"])

    # Also: check if pressure metric itself discriminates PFF pressureType annotations
    pressed_records = [r for r in records if r.pff_pressure_type == "P"]
    not_pressed_records = [r for r in records if r.pff_pressure_type == "N"]

    pressure_discrimination: dict = {}
    if pressed_records and not_pressed_records:
        p_scores = np.array([r.actual_target_pressure for r in pressed_records])
        n_scores = np.array([r.actual_target_pressure for r in not_pressed_records])
        try:
            u, pval = stats.mannwhitneyu(p_scores, n_scores, alternative="greater")
            pressure_discrimination = {
                "auc": round(u / (len(p_scores) * len(n_scores)), 4),
                "p_value": float(pval),
                "mean_pressed": round(float(np.mean(p_scores)), 2),
                "mean_not_pressed": round(float(np.mean(n_scores)), 2),
            }
        except ValueError:
            pressure_discrimination = {"error": "cannot compute"}

    return {
        "no_pressure_baseline": {
            "auc": round(no_pressure_auc, 4),
            "concordance": round(no_pressure_concord * 100, 1),
        },
        "k_sweep": sweep,
        "best_k_by_auc": best_by_auc,
        "pressure_discrimination": pressure_discrimination,
    }


# ── Run all sensitivity analyses ────────────────────────────────────────────


def run_sensitivity(records: list[ValidatedPass]) -> dict:
    """Run all sensitivity analyses."""
    return {
        "weight_sweep": weight_sweep(records),
        "ablation": dimension_ablation(records),
        "pressure_deep_dive": pressure_deep_dive(records),
    }
