#!/usr/bin/env python3
"""
Threader Weight Learning — train regression models to optimize Pass Score weights.

Pipeline:
  1. Collect / load validated passes (same as validation framework)
  2. Build feature matrix (5 dimensions) + target (ΔxT / lines_broken / composite)
  3. Train 3 model layers: Linear → Linear+Interactions → XGBoost
  4. Interpret coefficients → suggest new ScoringWeights
  5. Evaluate suggested weights vs current defaults

Usage:
    python scripts/train_scoring_model.py                      # full run
    python scripts/train_scoring_model.py --matches 5          # quick test
    python scripts/train_scoring_model.py --target delta_xt    # regression on ΔxT
    python scripts/train_scoring_model.py --target lines_broken  # classification
    python scripts/train_scoring_model.py --target offensive_value  # composite (default)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _header(title: str) -> str:
    width = 60
    return f"\n{'═' * width}\n  {title}\n{'═' * width}"


def _subheader(title: str) -> str:
    return f"\n  ── {title} {'─' * max(1, 50 - len(title))}"


def _kv(key: str, value, indent: int = 4) -> str:
    pad = " " * indent
    return f"{pad}{key}: {value}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threader Weight Learning — train models to optimize Pass Score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/FIFA_World_Cup_2022",
        help="Root data directory",
    )
    parser.add_argument(
        "--matches", type=int, default=None,
        help="Max matches to process",
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
        "--target", type=str, default="offensive_value",
        choices=["delta_xt", "lines_broken", "offensive_value"],
        help="Target variable for regression (default: offensive_value)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of matches for test set (default: 0.2)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Collect data ──────────────────────────────────────────────────
    from threader.validation.collector import (
        collect_validated_passes,
        data_summary,
        load_cache,
        save_cache,
    )

    cache_dir = Path(args.cache_dir)
    cache_tag = f"validated_passes_n{args.matches or 'all'}.pkl"
    cache_path = cache_dir / cache_tag

    records = None
    if not args.no_cache:
        records = load_cache(cache_path)
        if records:
            print(f"\n  Loaded {len(records)} cached records from {cache_path}", file=sys.stderr)
            # Check if records have the new fields
            if not hasattr(records[0], "delta_xt"):
                print("  ⚠ Cached records lack new fields (delta_xt). Re-collecting...", file=sys.stderr)
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
    print(_header("THREADER WEIGHT LEARNING"))
    print(_kv("Matches", summary.get("matches", "?")))
    print(_kv("Total passes", summary.get("total", 0)))
    print(_kv("Target variable", args.target))
    print(_kv("Test fraction", args.test_fraction))

    # Quick stats on new fields
    dxts = [r.delta_xt for r in records]
    lbs = [r.pff_lines_broken_count for r in records]
    print(_kv("ΔxT: mean", f"{sum(dxts)/len(dxts):.5f}"))
    print(_kv("ΔxT: min/max", f"{min(dxts):.5f} / {max(dxts):.5f}"))
    print(_kv("Lines broken ≥1", f"{sum(1 for x in lbs if x >= 1)} ({sum(1 for x in lbs if x >= 1)/len(lbs)*100:.1f}%)"))

    # ── 2. Split data ────────────────────────────────────────────────────
    from threader.learning.dataset import build_dataset, train_test_split_by_match

    train_recs, test_recs = train_test_split_by_match(
        records, test_fraction=args.test_fraction, seed=42,
    )
    print(f"\n  Train: {len(train_recs)} passes  |  Test: {len(test_recs)} passes")

    is_binary = args.target == "lines_broken"

    X_train, y_train, feat_names = build_dataset(train_recs, target=args.target)
    X_test, y_test, _ = build_dataset(test_recs, target=args.target)

    print(f"  Features: {feat_names}")
    print(f"  y_train: mean={y_train.mean():.5f} std={y_train.std():.5f}")
    print(f"  y_test:  mean={y_test.mean():.5f} std={y_test.std():.5f}")

    # ── 3. Train models ──────────────────────────────────────────────────
    print(_header("MODEL TRAINING"))

    from threader.learning.models import train_all_models

    results = train_all_models(X_train, y_train, X_test, y_test, feat_names, is_binary)

    for r in results:
        print(_subheader(r.name))
        print(_kv("Train R²", r.train_r2))
        print(_kv("Test R²", r.test_r2))
        print(_kv("Train RMSE", r.train_rmse))
        print(_kv("Test RMSE", r.test_rmse))
        if r.train_auc is not None:
            print(_kv("Train AUC", r.train_auc))
            print(_kv("Test AUC", r.test_auc))

        if r.coefficients is not None:
            print(f"\n    Coefficients:")
            for name, val in sorted(r.coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"      {name:<30} {val:>12.6f}")

        if r.feature_importance is not None and "error" not in r.feature_importance:
            print(f"\n    Feature importance:")
            for name, val in sorted(r.feature_importance.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(val * 50)
                print(f"      {name:<15} {val:.4f} {bar}")

    # ── 4. Interpret & suggest weights ───────────────────────────────────
    print(_header("WEIGHT SUGGESTION"))

    from threader.learning.interpret import (
        interpret_interactions,
        interpret_linear_coefficients,
        interpret_xgboost,
        suggest_weights,
    )

    linear_interp = interpret_linear_coefficients(results[0])
    print(_subheader("Linear Model Interpretation"))
    if "error" not in linear_interp:
        print(f"    Relative importance:")
        for k, v in sorted(linear_interp["relative_importance"].items(), key=lambda x: x[1], reverse=True):
            print(f"      {k:<15} {v:.4f} ({linear_interp['signs'][k]})")

    interaction_interp = interpret_interactions(results[1])
    print(_subheader("Top Interaction Terms"))
    if "error" not in interaction_interp:
        for k, v in interaction_interp.get("top_interactions", {}).items():
            print(f"      {k:<35} {v:>12.6f}")

    xgb_interp = interpret_xgboost(results[2])
    print(_subheader("XGBoost Feature Importance"))
    if "error" not in xgb_interp:
        for k, v in xgb_interp.get("feature_importance", {}).items():
            bar = "█" * int(v * 50)
            print(f"      {k:<15} {v:.4f} {bar}")

    suggestion = suggest_weights(results[0], results[1], results[2])

    print(_subheader("Suggested Weights"))
    if "error" not in suggestion:
        current = suggestion["current_weights"]
        suggested = suggestion["suggested_weights"]

        try:
            from tabulate import tabulate
            rows = []
            for k in current:
                c = current[k]
                s = suggested[k]
                delta = s - c
                rows.append([k, f"{c:.4f}", f"{s:.4f}", f"{delta:+.4f}"])
            print(tabulate(rows, headers=["Weight", "Current", "Suggested", "Δ"],
                            tablefmt="simple", stralign="right"))
        except ImportError:
            for k in current:
                print(f"      {k:<22} current={current[k]:.4f}  suggested={suggested[k]:.4f}  Δ={suggested[k]-current[k]:+.4f}")
    else:
        print(f"    {suggestion.get('error')}")

    # ── 5. Evaluate suggested weights ────────────────────────────────────
    print(_header("EVALUATION: SUGGESTED vs DEFAULT"))

    from threader.learning.evaluate import evaluate_weights
    from threader.metrics.pass_value.models import ScoringWeights

    if "error" not in suggestion:
        sw = suggestion["suggested_scoring_weights"]
        eval_result = evaluate_weights(test_recs, sw, label="suggested")

        print(_kv("ΔxT Spearman ρ (suggested)", eval_result["delta_xt_spearman"]))
        print(_kv("ΔxT Spearman ρ (default)", eval_result["vs_default"]["delta_xt_spearman_default"]))
        print(_kv("ΔxT improvement", f"{eval_result['vs_default']['delta_xt_spearman_improvement']:+.4f}"))
        print()
        print(_kv("Lines-broken AUC (suggested)", eval_result["lines_broken_auc"]))
        print(_kv("Lines-broken AUC (default)", eval_result["vs_default"]["lines_broken_auc_default"]))
        print(_kv("Lines-broken improvement", f"{eval_result['vs_default']['lines_broken_auc_improvement']:+.4f}"))

        # Also show code snippet for easy copy-paste
        print(_subheader("Copy-Paste: New DEFAULT_WEIGHTS"))
        print(f"""
    DEFAULT_WEIGHTS = ScoringWeights(
        zone_amplifier={sw.zone_amplifier},
        penetration_weight={sw.penetration_weight},
        space_weight={sw.space_weight},
        pressure_scaling={sw.pressure_scaling},
    )
""")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
