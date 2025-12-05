# stats_analysis.py
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import statsmodels.api as sm
import config


def describe_prevalence_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Table 1 style stats:
    - prevalence of spondylolisthesis
    - mean slip grade
    grouped by sex, age group, and pars_defect.
    """
    df = df.copy()

    # 年龄分组：<55, 55-64, >=65
    bins = [0, 55, 65, 200]
    labels = ["<55", "55-64", ">=65"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    records = []

    # ---- 按性别 ----
    for sex_val, group in df.groupby("sex"):
        name = "Female" if sex_val == 1 else "Male"
        n = len(group)
        prevalence = 100 * group[config.TARGET_SLIP_BIN].mean()
        mean_grade = group[config.TARGET_SLIP_GRADE].mean()
        records.append(
            {
                "grouping": "Sex",
                "category": name,
                "n": n,
                "prevalence_%": prevalence,
                "mean_slip_grade": mean_grade,
            }
        )

    # ---- 按年龄组 ----
    for ag, group in df.groupby("age_group"):
        n = len(group)
        prevalence = 100 * group[config.TARGET_SLIP_BIN].mean()
        mean_grade = group[config.TARGET_SLIP_GRADE].mean()
        records.append(
            {
                "grouping": "Age_group",
                "category": str(ag),
                "n": n,
                "prevalence_%": prevalence,
                "mean_slip_grade": mean_grade,
            }
        )

    # ---- 按 pars 缺损 ----
    for pd_val, group in df.groupby("pars_defect"):
        name = "Present" if pd_val == 1 else "Absent"
        n = len(group)
        prevalence = 100 * group[config.TARGET_SLIP_BIN].mean()
        mean_grade = group[config.TARGET_SLIP_GRADE].mean()
        records.append(
            {
                "grouping": "Pars_defect",
                "category": name,
                "n": n,
                "prevalence_%": prevalence,
                "mean_slip_grade": mean_grade,
            }
        )

    table = pd.DataFrame.from_records(records)
    table.to_csv(config.RESULTS_DIR / "table1_prevalence_by_group.csv", index=False)
    return table


def correlation_with_slip_grade(
    df: pd.DataFrame,
    variables: list[str] | None = None
) -> pd.DataFrame:
    """
    Compute Spearman's rho and Kendall's tau between slip_grade and each variable.
    """
    if variables is None:
        variables = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

    records = []
    y = df[config.TARGET_SLIP_GRADE]

    for var in variables:
        if var not in df.columns:
            continue
        x = df[var]
        # Spearman
        rho, p_rho = spearmanr(x, y, nan_policy="omit")
        # Kendall
        tau, p_tau = kendalltau(x, y, nan_policy="omit")

        records.append(
            {
                "variable": var,
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "kendall_tau": tau,
                "kendall_p": p_tau,
            }
        )

    corr_df = pd.DataFrame.from_records(records)
    corr_df.to_csv(
        config.RESULTS_DIR / "correlations_spearman_kendall.csv",
        index=False
    )
    return corr_df


def partial_correlation_residual(
    df: pd.DataFrame,
    var_x: str,
    var_y: str,
    covariates: list[str]
) -> tuple[float, float]:
    """
    Partial correlation between var_x and var_y controlling for covariates.
    Implementation via residuals + Pearson correlation.
    """
    # drop NA rows used in this subset
    cols = [var_x, var_y] + covariates
    sub = df[cols].dropna().copy()

    # X residuals
    X_cov = sm.add_constant(sub[covariates])
    model_x = sm.OLS(sub[var_x], X_cov).fit()
    res_x = model_x.resid

    # Y residuals
    model_y = sm.OLS(sub[var_y], X_cov).fit()
    res_y = model_y.resid

    r, p = pearsonr(res_x, res_y)
    return r, p


def partial_correlations_key_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute partial correlations between slip_grade and key variables,
    adjusted for age and sex (as in the manuscript).
    """
    covariates = ["age", "sex"]
    var_y = config.TARGET_SLIP_GRADE
    records = []

    for var_x in config.PARTIAL_CORR_VARS:
        if var_x not in df.columns:
            continue
        r, p = partial_correlation_residual(df, var_x, var_y, covariates)
        records.append(
            {"variable": var_x, "partial_r": r, "p_value": p}
        )

    pc_df = pd.DataFrame.from_records(records)
    pc_df.to_csv(config.RESULTS_DIR / "partial_correlations.csv", index=False)
    return pc_df


def quartile_stratification(
    df: pd.DataFrame,
    vars_quartile: list[str] | None = None
) -> pd.DataFrame:
    """
    For L3_IMATI, PV_fat_area, PMFI, compute quartiles,
    and for each quartile compute prevalence and mean slip grade.
    """
    if vars_quartile is None:
        vars_quartile = ["L3_IMATI", "PV_fat_area", "PMFI"]

    records = []

    for v in vars_quartile:
        if v not in df.columns:
            continue
        # 按四分位分组
        try:
            q = pd.qcut(df[v], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except ValueError:
            # 若重复值太多导致 qcut 失败，改用 rank-based
            q = pd.qcut(df[v].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])

        df_tmp = df.copy()
        df_tmp[f"{v}_quartile"] = q

        for q_label, group in df_tmp.groupby(f"{v}_quartile"):
            prevalence = 100 * group[config.TARGET_SLIP_BIN].mean()
            mean_grade = group[config.TARGET_SLIP_GRADE].mean()
            records.append(
                {
                    "variable": v,
                    "quartile": q_label,
                    "n": len(group),
                    "prevalence_%": prevalence,
                    "mean_slip_grade": mean_grade,
                }
            )

    q_df = pd.DataFrame.from_records(records)
    q_df.to_csv(config.RESULTS_DIR / "quartile_summary.csv", index=False)
    return q_df


def run_all_stats(df: pd.DataFrame):
    """
    Convenience function to run all statistical analyses.
    """
    table1 = describe_prevalence_by_group(df)
    corr_df = correlation_with_slip_grade(df)
    partial_df = partial_correlations_key_vars(df)
    quart_df = quartile_stratification(df)

    return {
        "table1": table1,
        "correlations": corr_df,
        "partial_correlations": partial_df,
        "quartiles": quart_df,
    }
