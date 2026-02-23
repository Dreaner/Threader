"""
Dataset preparation for weight learning.

Converts ValidatedPass records into feature matrices (X) and
offensive-value target vectors (y) for scikit-learn / XGBoost models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from threader.validation.collector import ValidatedPass

# Feature names (order matches columns in X)
FEATURE_NAMES = [
    "completion",
    "zone_value",
    "pressure",
    "space",
    "penetration",
]


def build_dataset(
    records: list[ValidatedPass],
    target: str = "delta_xt",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y) arrays from validated passes.

    Args:
        records: List of ValidatedPass records.
        target: Target variable —
            "delta_xt"         → continuous ΔxT (regression)
            "lines_broken"     → binary label: ≥1 line broken (classification)
            "offensive_value"  → composite: ΔxT_norm + lines_broken_binary

    Returns:
        (X, y, feature_names) where X is (n, 5) and y is (n,).
    """
    n = len(records)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)

    for i, r in enumerate(records):
        X[i, 0] = r.actual_target_completion
        X[i, 1] = r.actual_target_zone
        X[i, 2] = r.actual_target_pressure
        X[i, 3] = r.actual_target_space
        X[i, 4] = r.actual_target_penetration

        if target == "delta_xt":
            y[i] = r.delta_xt
        elif target == "lines_broken":
            y[i] = 1.0 if r.pff_lines_broken_count >= 1 else 0.0
        elif target == "offensive_value":
            # Composite: normalize ΔxT to [0, 1] range (roughly),
            # then add binary lines-broken
            # ΔxT typically in [-0.45, 0.45], so scale by ~2.2 to get [-1, 1]
            dxt_norm = np.clip(r.delta_xt * 2.2, -1.0, 1.0)
            lb_binary = 1.0 if r.pff_lines_broken_count >= 1 else 0.0
            y[i] = 0.6 * dxt_norm + 0.4 * lb_binary
        else:
            msg = f"Unknown target: {target!r}"
            raise ValueError(msg)

    return X, y, list(FEATURE_NAMES)


def train_test_split_by_match(
    records: list[ValidatedPass],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[ValidatedPass], list[ValidatedPass]]:
    """Split records into train/test by match_id (no leakage).

    Ensures all passes from a match are in the same set.
    """
    rng = np.random.default_rng(seed)
    match_ids = sorted({r.match_id for r in records})
    rng.shuffle(match_ids)  # type: ignore[arg-type]

    n_test = max(1, int(len(match_ids) * test_fraction))
    test_ids = set(match_ids[:n_test])

    train = [r for r in records if r.match_id not in test_ids]
    test = [r for r in records if r.match_id in test_ids]

    return train, test
