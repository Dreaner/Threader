#!/usr/bin/env python3
"""
Threader Weight Optimizer — directly maximize Spearman ρ via scipy.optimize.

Instead of fitting a regression model and heuristically mapping coefficients
back to formula weights, this script treats Spearman ρ (Pass Score vs ΔxT)
as the objective and uses differential evolution to find the weight
combination that maximizes it.

Pipeline:
  1. Collect / load validated passes (same cache as train_scoring_model.py)
  2. Split by match into train / test sets
  3. Define objective: −Spearman ρ(Pass Score(weights), ΔxT) on train set
  4. Run scipy.optimize.differential_evolution over 4 weight parameters
  5. Evaluate optimized weights on held-out test set vs DEFAULT_WEIGHTS

Usage:
    python scripts/optimize_weights.py                        # full run
    python scripts/optimize_weights.py --matches 5            # quick test
    python scripts/optimize_weights.py --metric auc           # optimize AUC
    python scripts/optimize_weights.py --metric combined      # weighted combo
    python scripts/optimize_weights.py --maxiter 500          # more iterations
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize, stats

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from threader.learning.dataset import train_test_split_by_match
from threader.learning.evaluate import evaluate_weights, rescore_with_weights
from threader.scoring.pass_score import DEFAULT_WEIGHTS, ScoringWeights
from threader.validation.collector import (
    ValidatedPass,
    collect_validated_passes,
    data_summary,
    load_cache,
    save_cache,
)

# ── Parameter names & bounds ─────────────────────────────────────────────
PARAM_NAMES = ["zone_amplifier", "penetration_weight", "space_weight", "pressure_scaling"]

PARAM_BOUNDS = [
    (0.5, 5.0),    # zone_amplifier
    (0.01, 1.0),   # penetration_weight
    (0.0001, 0.05),  # space_weight
    (0.01, 0.50),  # pressure_scaling
]


# ── Helpers ──────────────────────────────────────────────────────────────

def _header(title: str) -> str:
    width = 60
    return f"\n{'═' * width}\n  {title}\n{'═' * width}"


def _subheader(title: str) -> str:
    return f"\n  ── {title} {'─' * max(1, 50 - len(title))}"


def _kv(key: str, value: object, indent: int = 4) -> str:
    pad = " " * indent
    return f"{pad}{key}: {value}"


def _params_to_weights(params: np.ndarray) -> ScoringWeights:
    """Convert a 4-element parameter vector to ScoringWeights."""
    return ScoringWeights(
        zone_amplifier=float(params[0]),
        penetration_weight=float(params[1]),
        space_weight=float(params[2]),
        pressure_scaling=float(params[3]),
    )


def _weights_to_params(w: ScoringWeights) -> np.ndarray:
    """Convert ScoringWeights to a 4-element parameter vector."""
    return np.array([w.zone_amplifier, w.penetration_weight, w.space_weight, w.pressure_scaling])


# ── Objective functions ──────────────────────────────────────────────────

def _negative_spearman(
    params: np.ndarray,
    records: list[ValidatedPass],
    delta_xts: np.ndarray,
) -> float:
    """Objective: −Spearman ρ between Pass Score and ΔxT.

    We minimize this (== maximize Spearman ρ).
    """
    weights = _params_to_weights(params)
    scores = rescore_with_weights(records, weights)

    # Guard against constant scores (all identical → ρ undefined)
    if np.std(scores) < 1e-12:
        return 0.0  # ρ=0, not useful

    rho, _ = stats.spearmanr(scores, delta_xts)
    return -float(rho)


def _negative_auc(
    params: np.ndarray,
    records: list[ValidatedPass],
    has_lines_broken: np.ndarray,
) -> float:
    """Objective: −AUC-ROC for lines-broken ≥ 1 prediction."""
    from sklearn.metrics import roc_auc_score

    weights = _params_to_weights(params)
    scores = rescore_with_weights(records, weights)

    if np.std(scores) < 1e-12:
        return -0.5

    try:
        auc = roc_auc_score(has_lines_broken, scores)
    except ValueError:
        auc = 0.5
    return -float(auc)


def _negative_combined(
    params: np.ndarray,
    records: list[ValidatedPass],
    delta_xts: np.ndarray,
    has_lines_broken: np.ndarray,
    spearman_weight: float = 0.7,
) -> float:
    """Objective: −(w × Spearman ρ + (1−w) × AUC)."""
    from sklearn.metrics import roc_auc_score

    weights = _params_to_weights(params)
    scores = rescore_with_weights(records, weights)

    if np.std(scores) < 1e-12:
        return 0.0

    rho, _ = stats.spearmanr(scores, delta_xts)
    try:
        auc = roc_auc_score(has_lines_broken, scores)
    except ValueError:
        auc = 0.5

    combined = spearman_weight * float(rho) + (1.0 - spearman_weight) * float(auc)
    return -combined


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threader Weight Optimizer — maximize Spearman ρ directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/FIFA_World_Cup_2022",
        help="Root data directory",
    )
    parser.add_argument(
        "--matches", type=int, default=None,
        help="Max matches to process (default: all)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=".validation_cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip cache — always re-collect",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of matches for test set (default: 0.2)",
    )
    parser.add_argument(
        "--metric", type=str, default="spearman",
        choices=["spearman", "auc", "combined"],
        help="Optimization target (default: spearman)",
    )
    parser.add_argument(
        "--maxiter", type=int, default=200,
        help="Max generations for differential evolution (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--popsize", type=int, default=15,
        help="Population size multiplier for DE (default: 15)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for DE (default: 1)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Collect data ──────────────────────────────────────────────────
    cache_dir = Path(args.cache_dir)
    cache_tag = f"validated_passes_n{args.matches or 'all'}.pkl"
    cache_path = cache_dir / cache_tag

    records: list[ValidatedPass] | None = None
    if not args.no_cache:
        records = load_cache(cache_path)
        if records:
            print(f"\n  Loaded {len(records)} cached records from {cache_path}", file=sys.stderr)
            if not hasattr(records[0], "delta_xt"):
                print("  ⚠ Cached records lack delta_xt field. Re-collecting...", file=sys.stderr)
                records = None

    if records is None:
        print("\n  Collecting pass data...\n", file=sys.stderr)
        records = collect_validated_passes(
            args.data_dir,
            max_matches=args.matches,
            verbose=True,
        )
        save_cache(records, cache_path)
        print(f"  Cached to {cache_path}\n", file=sys.stderr)

    if not records:
        print("ERROR: No validated passes collected.", file=sys.stderr)
        sys.exit(1)

    summary = data_summary(records)

    # ── Print header ─────────────────────────────────────────────────────
    print(_header("THREADER WEIGHT OPTIMIZER"))
    print(_kv("Matches", summary.get("matches", "?")))
    print(_kv("Total passes", summary.get("total", 0)))
    print(_kv("Metric", args.metric))
    print(_kv("Max iterations", args.maxiter))
    print(_kv("Population size", f"{args.popsize}× (= {args.popsize * len(PARAM_NAMES)} individuals)"))
    print(_kv("Test fraction", args.test_fraction))
    print(_kv("Seed", args.seed))

    # ── 2. Split data ────────────────────────────────────────────────────
    train_recs, test_recs = train_test_split_by_match(
        records, test_fraction=args.test_fraction, seed=args.seed,
    )
    print(f"\n  Train: {len(train_recs)} passes  |  Test: {len(test_recs)} passes")

    # Pre-compute target arrays for the train set (avoid recomputing every eval)
    train_delta_xts = np.array([r.delta_xt for r in train_recs])
    train_has_lb = np.array([1.0 if r.pff_lines_broken_count >= 1 else 0.0 for r in train_recs])

    # Quick stats
    print(_kv("Train ΔxT: mean", f"{train_delta_xts.mean():.5f}"))
    print(_kv("Train ΔxT: std", f"{train_delta_xts.std():.5f}"))
    print(_kv("Train lines-broken ≥1", f"{train_has_lb.sum():.0f} ({train_has_lb.mean()*100:.1f}%)"))

    # Baseline: default weights on train set
    default_scores_train = rescore_with_weights(train_recs, DEFAULT_WEIGHTS)
    rho_default_train, _ = stats.spearmanr(default_scores_train, train_delta_xts)
    print(f"\n  Baseline (default weights) — Train Spearman ρ: {rho_default_train:.4f}")

    # ── 3. Build objective ───────────────────────────────────────────────
    print(_header("OPTIMIZATION"))

    n_evals = [0]  # mutable counter for callback
    best_val = [0.0]

    if args.metric == "spearman":
        def objective(params: np.ndarray) -> float:
            return _negative_spearman(params, train_recs, train_delta_xts)
    elif args.metric == "auc":
        def objective(params: np.ndarray) -> float:
            return _negative_auc(params, train_recs, train_has_lb)
    elif args.metric == "combined":
        def objective(params: np.ndarray) -> float:
            return _negative_combined(params, train_recs, train_delta_xts, train_has_lb)
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    def callback(xk: np.ndarray, convergence: float = 0.0) -> None:
        n_evals[0] += 1
        val = objective(xk)
        if n_evals[0] == 1 or val < best_val[0]:
            best_val[0] = val
        metric_name = {"spearman": "ρ", "auc": "AUC", "combined": "combined"}[args.metric]
        print(
            f"  Gen {n_evals[0]:>4d}  |  best {metric_name} = {-best_val[0]:.6f}  |  "
            f"convergence = {convergence:.6f}  |  "
            f"params = [{', '.join(f'{p:.4f}' for p in xk)}]"
        )

    print(f"  Running differential_evolution (maxiter={args.maxiter}, popsize={args.popsize})...")
    print(f"  Parameter bounds:")
    for name, (lo, hi) in zip(PARAM_NAMES, PARAM_BOUNDS):
        current = getattr(DEFAULT_WEIGHTS, name)
        print(f"    {name:<22}  [{lo}, {hi}]  (current: {current})")
    print()

    result = optimize.differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        strategy="best1bin",
        maxiter=args.maxiter,
        tol=1e-6,
        seed=args.seed,
        popsize=args.popsize,
        polish=True,  # Refine with L-BFGS-B after DE converges
        callback=callback,
        workers=args.workers,
        disp=False,
    )

    # ── 4. Results ───────────────────────────────────────────────────────
    optimized_weights = _params_to_weights(result.x)
    opt_metric_value = -result.fun

    print(_header("OPTIMIZATION RESULT"))
    print(_kv("Success", result.success))
    print(_kv("Message", result.message))
    print(_kv("Generations", result.nit))
    print(_kv("Function evaluations", result.nfev))
    metric_label = {"spearman": "Spearman ρ", "auc": "AUC-ROC", "combined": "Combined"}[args.metric]
    print(_kv(f"Optimized {metric_label}", f"{opt_metric_value:.6f}"))

    # ── 5. Weights comparison table ──────────────────────────────────────
    print(_subheader("Weights: Default → Optimized"))

    try:
        from tabulate import tabulate
        rows = []
        for name in PARAM_NAMES:
            c = getattr(DEFAULT_WEIGHTS, name)
            o = getattr(optimized_weights, name)
            delta = o - c
            pct = (delta / c * 100) if c != 0 else float("inf")
            rows.append([name, f"{c:.6f}", f"{o:.6f}", f"{delta:+.6f}", f"{pct:+.1f}%"])
        print(tabulate(rows, headers=["Weight", "Default", "Optimized", "Δ", "Δ%"],
                        tablefmt="simple", stralign="right"))
    except ImportError:
        for name in PARAM_NAMES:
            c = getattr(DEFAULT_WEIGHTS, name)
            o = getattr(optimized_weights, name)
            print(f"    {name:<22}  default={c:.6f}  optimized={o:.6f}  Δ={o-c:+.6f}")

    # ── 6. Evaluation on test set ────────────────────────────────────────
    print(_header("EVALUATION: TEST SET"))

    # Test set — optimized
    eval_opt = evaluate_weights(test_recs, optimized_weights, label="optimized")

    # Train set metrics for overfitting check
    opt_scores_train = rescore_with_weights(train_recs, optimized_weights)
    rho_opt_train, _ = stats.spearmanr(opt_scores_train, train_delta_xts)

    print(_subheader("Spearman ρ (Pass Score vs ΔxT)"))
    print(_kv("Train — default weights", f"{rho_default_train:.4f}"))
    print(_kv("Train — optimized weights", f"{rho_opt_train:.4f}"))
    print(_kv("Train Δ", f"{rho_opt_train - rho_default_train:+.4f}"))
    print()
    print(_kv("Test  — default weights", eval_opt["vs_default"]["delta_xt_spearman_default"]))
    print(_kv("Test  — optimized weights", eval_opt["delta_xt_spearman"]))
    print(_kv("Test  Δ", f"{eval_opt['vs_default']['delta_xt_spearman_improvement']:+.4f}"))

    # Check for overfitting
    train_test_gap = abs(rho_opt_train - eval_opt["delta_xt_spearman"])
    if train_test_gap > 0.05:
        print(f"\n  ⚠ Train-test gap = {train_test_gap:.4f} — possible overfitting")
    else:
        print(f"\n  ✓ Train-test gap = {train_test_gap:.4f} — no overfitting concern")

    print(_subheader("Lines-Broken AUC"))
    print(_kv("Test  — default weights", eval_opt["vs_default"]["lines_broken_auc_default"]))
    print(_kv("Test  — optimized weights", eval_opt["lines_broken_auc"]))
    print(_kv("Test  Δ", f"{eval_opt['vs_default']['lines_broken_auc_improvement']:+.4f}"))

    # ── 7. Copy-paste snippet ────────────────────────────────────────────
    print(_subheader("Copy-Paste: New DEFAULT_WEIGHTS"))
    print(f"""
    DEFAULT_WEIGHTS = ScoringWeights(
        zone_amplifier={optimized_weights.zone_amplifier:.6f},
        penetration_weight={optimized_weights.penetration_weight:.6f},
        space_weight={optimized_weights.space_weight:.6f},
        pressure_scaling={optimized_weights.pressure_scaling:.6f},
    )
""")

    elapsed = time.time() - t0
    print(f"{'═' * 60}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
