"""Baseline model factories and training helpers."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(random_state: int = 42) -> Pipeline:
    """Logistic regression baseline with median imputation and scaling."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    C=0.5,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_tree(random_state: int = 42) -> HistGradientBoostingClassifier:
    """Tree-based sklearn baseline that supports NaN natively."""
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
        random_state=random_state,
    )


def make_xgb(random_state: int = 42, scale_pos_weight: float = 1.0):
    """XGBoost baseline configured for probability outputs."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError(
            "xgboost could not be loaded (often missing OpenMP/libomp on macOS)."
        ) from exc

    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
    )


def fit_predict(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """Fit model and return positive-class probabilities on eval data."""
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_eval)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_eval)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise ValueError(f"Model {type(model).__name__} does not expose probabilities.")
    return probabilities.astype(float)
