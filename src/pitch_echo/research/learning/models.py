"""
Model training for weight learning.

Three layers of increasing complexity:
  1. Linear regression (directly interpretable coefficients)
  2. Linear + interaction terms (captures comp × zone synergy)
  3. XGBoost (non-linear, interpreted via SHAP)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass
class ModelResult:
    """Container for a trained model's results."""

    name: str
    model: object
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    # For classification targets
    train_auc: float | None = None
    test_auc: float | None = None
    coefficients: dict[str, float] | None = None
    feature_importance: dict[str, float] | None = None


def _eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_binary: bool = False,
) -> tuple[float, float, float | None]:
    """Compute R², RMSE, and optionally AUC."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    auc = None
    if is_binary:
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = None
    return r2, rmse, auc


def train_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    is_binary: bool = False,
) -> ModelResult:
    """Layer 1: Plain linear regression."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    tr_r2, tr_rmse, tr_auc = _eval_metrics(y_train, y_pred_train, is_binary)
    te_r2, te_rmse, te_auc = _eval_metrics(y_test, y_pred_test, is_binary)

    coeffs = dict(zip(feature_names, model.coef_))
    coeffs["intercept"] = float(model.intercept_)

    return ModelResult(
        name="Linear Regression",
        model=model,
        train_r2=round(tr_r2, 4),
        test_r2=round(te_r2, 4),
        train_rmse=round(tr_rmse, 6),
        test_rmse=round(te_rmse, 6),
        train_auc=round(tr_auc, 4) if tr_auc is not None else None,
        test_auc=round(te_auc, 4) if te_auc is not None else None,
        coefficients=coeffs,
    )


def train_linear_interactions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    is_binary: bool = False,
) -> ModelResult:
    """Layer 2: Linear regression with interaction terms (degree=2)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    tr_r2, tr_rmse, tr_auc = _eval_metrics(y_train, y_pred_train, is_binary)
    te_r2, te_rmse, te_auc = _eval_metrics(y_test, y_pred_test, is_binary)

    # Extract interaction feature names
    poly = pipe.named_steps["poly"]
    poly_names = poly.get_feature_names_out(feature_names)
    ridge = pipe.named_steps["ridge"]
    coeffs = dict(zip(poly_names, ridge.coef_))
    coeffs["intercept"] = float(ridge.intercept_)

    return ModelResult(
        name="Linear + Interactions (Ridge)",
        model=pipe,
        train_r2=round(tr_r2, 4),
        test_r2=round(te_r2, 4),
        train_rmse=round(tr_rmse, 6),
        test_rmse=round(te_rmse, 6),
        train_auc=round(tr_auc, 4) if tr_auc is not None else None,
        test_auc=round(te_auc, 4) if te_auc is not None else None,
        coefficients=coeffs,
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    is_binary: bool = False,
) -> ModelResult:
    """Layer 3: XGBoost gradient boosting."""
    try:
        import xgboost as xgb
    except ImportError:
        return ModelResult(
            name="XGBoost",
            model=None,
            train_r2=0.0,
            test_r2=0.0,
            train_rmse=0.0,
            test_rmse=0.0,
            feature_importance={"error": "xgboost not installed"},
        )

    objective = "binary:logistic" if is_binary else "reg:squarederror"
    eval_metric = "auc" if is_binary else "rmse"

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        eval_metric=eval_metric,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    tr_r2, tr_rmse, tr_auc = _eval_metrics(y_train, y_pred_train, is_binary)
    te_r2, te_rmse, te_auc = _eval_metrics(y_test, y_pred_test, is_binary)

    importance = dict(zip(feature_names, model.feature_importances_))

    return ModelResult(
        name="XGBoost",
        model=model,
        train_r2=round(tr_r2, 4),
        test_r2=round(te_r2, 4),
        train_rmse=round(tr_rmse, 6),
        test_rmse=round(te_rmse, 6),
        train_auc=round(tr_auc, 4) if tr_auc is not None else None,
        test_auc=round(te_auc, 4) if te_auc is not None else None,
        feature_importance=importance,
    )


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    is_binary: bool = False,
) -> list[ModelResult]:
    """Train all three model layers and return results."""
    results = [
        train_linear(X_train, y_train, X_test, y_test, feature_names, is_binary),
        train_linear_interactions(X_train, y_train, X_test, y_test, feature_names, is_binary),
        train_xgboost(X_train, y_train, X_test, y_test, feature_names, is_binary),
    ]
    return results
