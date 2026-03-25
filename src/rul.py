from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RUL_FEATURE_DROP = ("timestamp", "system_id", "fault_type", "rul")


@dataclass
class RULArtifacts:
    model: object
    scaler: StandardScaler
    feature_names: list[str]
    X_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    mae: float
    rmse: float
    r2: float
    train_seconds: float


def build_rul_labels(df: pd.DataFrame, group_col: str = "system_id") -> pd.Series:
    if group_col not in df.columns:
        n = len(df)
        return pd.Series(np.arange(n - 1, -1, -1), index=df.index, name="rul")

    parts = []
    for _, g in df.groupby(group_col, sort=False):
        parts.append(pd.Series(np.arange(len(g) - 1, -1, -1), index=g.index))
    return pd.concat(parts).reindex(df.index).rename("rul")


def prepare_rul_dataset(
    fused_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    df = fused_df.copy()
    df["rul"] = build_rul_labels(df)

    feature_cols = [c for c in df.columns if c not in RUL_FEATURE_DROP]
    X = df[feature_cols].values
    y = df["rul"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, feature_cols, scaler


def train_rul_model(
    fused_df: pd.DataFrame,
    model_kind: str = "random_forest",
    random_state: int = 42,
) -> RULArtifacts:
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_rul_dataset(fused_df)

    if model_kind == "gbr":
        model = GradientBoostingRegressor(random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return RULArtifacts(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        mae=mae,
        rmse=rmse,
        r2=r2,
        train_seconds=train_seconds,
    )
