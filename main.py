# main.py
import config
from data_loader import load_dataset
from stats_analysis import run_all_stats
from ml_models import train_and_evaluate_all_models
from plotting import (
    plot_correlations_bar,
    plot_quartile_results,
    plot_roc_curves,
    plot_ml_metrics,
    plot_rf_feature_importance,
)


def main():
    # 1. 读取数据
    df = load_dataset()

    # 2. 统计分析
    stats_results = run_all_stats(df)
    corr_df = stats_results["correlations"]
    quart_df = stats_results["quartiles"]

    # 3. 机器学习建模与评估
    ml_results = train_and_evaluate_all_models(df)
    metrics_df = ml_results["metrics"]
    roc_curves = ml_results["roc_curves"]
    rf_importance_df = ml_results["rf_importance"]

    # 4. 画图（Fig2–4）
    plot_correlations_bar(corr_df)
    plot_quartile_results(quart_df)
    plot_roc_curves(roc_curves)
    plot_ml_metrics(metrics_df)
    if rf_importance_df is not None:
        plot_rf_feature_importance(rf_importance_df)

    print("All analyses finished. Check 'results/' and 'figures/' folders.")


if __name__ == "__main__":
    main()
