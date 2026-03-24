from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

MIN_SAMPLES_TO_TRAIN = 10
NON_FEATURE_COLUMNS = ("timestamp", "system_id", "fault_type")


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    scaler: StandardScaler
    label_encoder: LabelEncoder


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "fault_type" in df.columns:
        df = df.dropna(subset=["fault_type"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def select_feature_matrix(
    df: pd.DataFrame,
    drop_cols: tuple[str, ...] = NON_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, list[str]]:
    candidate = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    numeric_df = candidate.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping metadata.")
    return numeric_df, list(numeric_df.columns)


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    df = clean_dataframe(df)
    if "fault_type" not in df.columns:
        raise ValueError("DataFrame must contain a 'fault_type' column for training.")
    if len(df) < MIN_SAMPLES_TO_TRAIN:
        raise ValueError(
            f"Need at least {MIN_SAMPLES_TO_TRAIN} samples to train; got {len(df)}."
        )

    X_df, feature_names = select_feature_matrix(df)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["fault_type"].astype(str))

    class_counts = pd.Series(y).value_counts()
    stratify = y if (class_counts >= 2).all() else None
    if stratify is None and (class_counts < 1).any():
        raise ValueError("Empty class encountered; cannot train.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        class_names=list(label_encoder.classes_),
        scaler=scaler,
        label_encoder=label_encoder,
    )
