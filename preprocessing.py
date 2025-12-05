# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import config


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None
):
    """
    Stratified split by target y (for slip_bin).
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def make_numeric_pipeline() -> Pipeline:
    """
    Impute missing values and scale features (for most ML models).
    """
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    return pipe


def fit_transform_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """
    Fit preprocessing pipeline on training data, transform train and test.
    """
    pipe = make_numeric_pipeline()
    X_train_proc = pipe.fit_transform(X_train)
    X_test_proc = pipe.transform(X_test)
    return X_train_proc, X_test_proc, pipe
