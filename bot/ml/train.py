"""Walk-forward ML training with an embargo gap (no lookahead, no leakage).

Protocol
--------
1. Build bar-close features and a forward-return label
   (label at row t uses returns AFTER t — that is the thing being predicted).
2. Split history into ``folds`` contiguous chunks. For each fold: train on
   everything BEFORE the test chunk, leave ``embargo_bars`` untouched rows
   between train and test (kills overlap leakage from forward labels), then
   evaluate on the test chunk.
3. Fit the final model on all data minus the embargo tail, with the same
   hyperparameters, for live/inference use. The OUT-OF-SAMPLE metrics from
   step 2 — not the in-sample fit — are the honest performance estimate.

This is intentionally simple and hard to fool. It is not a guarantee of
future performance; treat the metrics as descriptive of the past only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TrainingResult:
    n_rows: int
    n_features: int
    folds: list[dict] = field(default_factory=list)
    oos_accuracy: float = 0.0
    oos_precision: float = 0.0
    oos_positive_rate: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    model_path: str = ""


def build_dataset(df: pd.DataFrame, horizon: int = 8) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Features at bar close + binary label = future return over ``horizon``
    bars > 0. Label row t is ONLY used as the prediction target for t."""
    from bot.strategies.ml_strategy import MLClassifierStrategy

    feats = MLClassifierStrategy.build_features(df)
    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    y = (future_ret > 0).astype(int)
    y[future_ret.isna()] = np.nan  # last ``horizon`` rows have no label

    feature_names = list(feats.columns)
    return feats, y, feature_names


def train_classifier(df: pd.DataFrame, *, horizon: int = 8, folds: int = 5,
                     embargo_bars: int = 24, model_dir: str | Path = "data/models",
                     model_kind: str = "hist_gb") -> TrainingResult:
    """Train a long-direction classifier with walk-forward evaluation."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score
    import joblib

    feats, y, feature_names = build_dataset(df, horizon)
    data = feats.copy()
    data["__label__"] = y
    data = data.dropna()
    if len(data) < 500:
        raise ValueError(
            f"not enough labeled rows to train ({len(data)}); download more history")

    X_all = data[feature_names]
    y_all = data["__label__"].astype(int)

    def _new_model():
        if model_kind == "hist_gb":
            return HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, learning_rate=0.06,
                l2_regularization=1.0, random_state=42)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    result = TrainingResult(
        n_rows=len(data), n_features=len(feature_names),
        feature_names=feature_names)

    # ---- walk-forward folds -------------------------------------------------
    indices = np.arange(len(data))
    fold_bounds = np.array_split(indices, folds)
    for k, test_idx in enumerate(fold_bounds):
        if len(test_idx) < 50:
            continue
        train_end = test_idx[0] - embargo_bars
        if train_end < 200:
            logger.warning("fold {}: not enough training rows before test — skipped", k)
            continue
        train_idx = indices[:train_end]

        model = _new_model()
        model.fit(X_all.iloc[train_idx], y_all.iloc[train_idx])
        proba = model.predict_proba(X_all.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        truth = y_all.iloc[test_idx]

        fold_metrics = {
            "fold": k,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "accuracy": float(accuracy_score(truth, pred)),
            "precision": float(precision_score(truth, pred, zero_division=0)),
        }
        result.folds.append(fold_metrics)
        logger.info("walk-forward fold {}: {}", k, fold_metrics)

    if result.folds:
        result.oos_accuracy = float(np.mean([f["accuracy"] for f in result.folds]))
        result.oos_precision = float(np.mean([f["precision"] for f in result.folds]))

    # ---- final model for inference ------------------------------------------
    final_model = _new_model()
    final_model.fit(X_all, y_all)

    positive_rate = float(y_all.mean())
    result.oos_positive_rate = positive_rate
    if result.oos_accuracy and result.oos_accuracy < max(0.52, positive_rate + 0.02):
        logger.warning(
            "walk-forward accuracy {:.3f} is barely better than the base rate "
            "{:.3f} — this model likely has NO real edge", result.oos_accuracy,
            positive_rate)

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ml_classifier.joblib"
    joblib.dump({
        "model": final_model,
        "feature_names": feature_names,
        "horizon": horizon,
        "trained_rows": len(data),
        "oos_accuracy": result.oos_accuracy,
    }, path)
    result.model_path = str(path)
    logger.info("model saved to {} (oos_accuracy={:.3f}, oos_precision={:.3f})",
                path, result.oos_accuracy, result.oos_precision)
    return result
