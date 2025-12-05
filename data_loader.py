# data_loader.py
import pandas as pd
from typing import Tuple
import config


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """
    Load the lumbar spondylolisthesis dataset from CSV.
    """
    csv_path = config.DEFAULT_DATA_FILE if path is None else path
    df = pd.read_csv(csv_path)

    # --- 基本类型清洗 ---
    # 性别：统一转成 0/1 （0=male, 1=female）
    if "sex" in df.columns:
        if df["sex"].dtype == "O":
            df["sex"] = df["sex"].str.upper().map({"M": 0, "F": 1})
        df["sex"] = df["sex"].astype("Int64")

    # pars 缺损：确保是 0/1
    if "pars_defect" in df.columns:
        df["pars_defect"] = df["pars_defect"].astype("Int64")

    # slip_bin：确保是 0/1
    if "slip_bin" in df.columns:
        df["slip_bin"] = df["slip_bin"].astype(int)

    # slip_grade / slip_level：整数
    for col in ["slip_grade", "slip_level"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def get_features_and_target(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features X and target y from the dataframe.
    """
    if feature_cols is None:
        feature_cols = config.ML_FEATURES

    # 过滤不存在的特征（防止列名不完全匹配时报错）
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y
