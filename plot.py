# plotting.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config


def plot_correlations_bar(corr_df: pd.DataFrame, top_n: int = 20):
    """
    Fig2-like: horizontal bar plot of Spearman's rho with slip_grade.
    """
    df = corr_df.sort_values("spearman_rho", ascending=True).tail(top_n)

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(df))
    plt.barh(y_pos, df["spearman_rho"])
    plt.yticks(y_pos, df["variable"])
    plt.xlabel("Spearman's rho with slip grade")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig2_correlations.png", dpi=300)
    plt.close()


def plot_quartile_results(q_df: pd.DataFrame):
    """
    Fig3-like: for each variable, plot prevalence and mean slip grade by quartile.
    """
    vars_unique = q_df["variable"].unique()

    for v in vars_unique:
        sub = q_df[q_df["variable"] == v].sort_values("quartile")

        fig, ax1 = plt.subplots(figsize=(6, 4))
        x = np.arange(len(sub))
        width = 0.35

        # prevalence bar
        ax1.bar(x - width/2, sub["prevalence_%"], width)
        ax1.set_ylabel("Prevalence of spondylolisthesis (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sub["quartile"])
        ax1.set_xlabel(f"{v} quartile")

        ax2 = ax1.twinx()
        ax2.plot(x + width/2, sub["mean_slip_grade"], marker="o")
        ax2.set_ylabel("Mean slip grade")

        fig.tight_layout()
        fig.savefig(
            config.FIGURES_DIR / f"fig3_quartiles_{v}.png",
            dpi=300
        )
        plt.close(fig)


def plot_roc_curves(roc_curves: dict):
    """
    Fig4A-like: ROC curves of all models.
    """
    plt.figure(figsize=(6, 6))
    for name, d in roc_curves.items():
        plt.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves for ML models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig4A_ROC.png", dpi=300)
    plt.close()


def plot_ml_metrics(metrics_df: pd.DataFrame):
    """
    Fig4B-like: bar plot of accuracy, sensitivity, specificity for each model.
    """
    df = metrics_df.copy()
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, df["accuracy"], width, label="Accuracy")
    plt.bar(x, df["sensitivity"], width, label="Sensitivity")
    plt.bar(x + width, df["specificity"], width, label="Specificity")

    plt.xticks(x, models, rotation=30)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig4B_metrics.png", dpi=300)
    plt.close()


def plot_rf_feature_importance(rf_importance_df: pd.DataFrame, top_n: int = 20):
    """
    Fig4C-like: feature importance of RandomForest.
    """
    if rf_importance_df is None or rf_importance_df.empty:
        return

    df = rf_importance_df.sort_values("importance", ascending=True).tail(top_n)
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(df))
    plt.barh(y_pos, df["importance"])
    plt.yticks(y_pos, df["feature"])
    plt.xlabel("RandomForest feature importance")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig4C_rf_importance.png", dpi=300)
    plt.close()
