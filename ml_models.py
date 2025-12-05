# ml_models.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import config
from preprocessing import train_test_split_stratified, fit_transform_train_test
from data_loader import get_features_and_target


def build_models() -> Dict[str, Any]:
    """
    Construct the six classifiers used in the paper.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=config.RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=config.RANDOM_STATE
        ),
        "LinearSVM": SVC(
            kernel="linear",
            probability=True,
            random_state=config.RANDOM_STATE,
        ),
        "RBFSVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=config.RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    return models


def compute_sensitivity_specificity(y_true, y_pred) -> tuple[float, float]:
    """
    Compute sensitivity (recall for positive class) and specificity.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sensitivity, specificity


def train_and_evaluate_all_models(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str | None = None
) -> dict:
    """
    Training all models, evaluating on a held-out test set, and returning metrics.
    Also compute ROC curves and feature importance for RandomForest.
    """
    if target_col is None:
        target_col = config.TARGET_SLIP_BIN

    X, y = get_features_and_target(df, target_col, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # preprocess: impute + scale
    X_train_proc, X_test_proc, preproc_pipe = fit_transform_train_test(X_train, X_test)

    models = build_models()
    metrics_records = []
    roc_curves = {}
    rf_feature_importance = None
    rf_feature_names = list(X.columns)

    for name, model in models.items():
        model.fit(X_train_proc, y_train)

        # predictions
        y_pred = model.predict(X_test_proc)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        else:
            # For some models without predict_proba (should not happen here)
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        sens, spec = compute_sensitivity_specificity(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        metrics_records.append(
            {
                "model": name,
                "accuracy": acc,
                "AUC": auc,
                "sensitivity": sens,
                "specificity": spec,
            }
        )

        # ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

        # RF feature importance
        if name == "RandomForest":
            rf_feature_importance = model.feature_importances_

    metrics_df = pd.DataFrame.from_records(metrics_records)
    metrics_df.to_csv(
        config.RESULTS_DIR / "ml_metrics_and_importance.csv", index=False
    )

    if rf_feature_importance is not None:
        rf_importance_df = pd.DataFrame(
            {"feature": rf_feature_names, "importance": rf_feature_importance}
        ).sort_values("importance", ascending=False)
    else:
        rf_importance_df = None

    return {
        "metrics": metrics_df,
        "roc_curves": roc_curves,
        "rf_importance": rf_importance_df,
    }
