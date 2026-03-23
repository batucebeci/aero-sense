from __future__ import annotations

import numpy as np
import pandas as pd

DERIVED_FEATURE_COLUMNS = [
    "battery_drop_rate",
    "temperature_rise_rate",
    "vibration_magnitude",
    "signal_drop_rate",
    "gps_stability_score",
    "power_fluctuation_score",
    "navigation_consistency_score",
    "acceleration_magnitude",
    "speed_change_rate",
    "altitude_change_rate",
]

TS_FEATURE_SENSORS = [
    "motor_temperature",
    "battery_level",
    "voltage",
    "current",
    "vibration_magnitude",
    "gps_accuracy",
    "signal_strength",
    "speed",
]
TS_FEATURE_STATS = ("mean", "std", "range", "energy")


def timeseries_feature_columns(window: int = 10) -> list[str]:
    return [f"{s}_{stat}_w{window}" for s in TS_FEATURE_SENSORS for stat in TS_FEATURE_STATS]


def _rate_of_change(series: pd.Series) -> pd.Series:
    return series.diff().fillna(0.0)


def _magnitude(x: pd.Series, y: pd.Series, z: pd.Series) -> pd.Series:
    return np.sqrt(x.pow(2) + y.pow(2) + z.pow(2))


def _rolling_std(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std().fillna(0.0)


MAX_GROUPS_FOR_TS = 500


def add_derived_features(df: pd.DataFrame, group_col: str | None = "system_id") -> pd.DataFrame:
    df = df.copy()
    if group_col and group_col in df.columns and df[group_col].nunique() > MAX_GROUPS_FOR_TS:
        group_col = None
    if group_col and group_col in df.columns:
        df = df.sort_values([group_col, "timestamp"] if "timestamp" in df.columns else [group_col])
        grouped = df.groupby(group_col, sort=False)

        df["battery_drop_rate"] = grouped["battery_level"].transform(lambda s: -_rate_of_change(s))
        df["temperature_rise_rate"] = grouped["motor_temperature"].transform(_rate_of_change)
        df["signal_drop_rate"] = grouped["signal_strength"].transform(lambda s: -_rate_of_change(s))
        df["gps_stability_score"] = grouped["gps_accuracy"].transform(
            lambda s: 1.0 / (1.0 + _rolling_std(s))
        )
        df["power_fluctuation_score"] = grouped["voltage"].transform(_rolling_std) + grouped[
            "current"
        ].transform(_rolling_std)
        df["speed_change_rate"] = grouped["speed"].transform(_rate_of_change).abs()
        df["altitude_change_rate"] = grouped["altitude"].transform(_rate_of_change).abs()
    else:
        df["battery_drop_rate"] = -_rate_of_change(df["battery_level"])
        df["temperature_rise_rate"] = _rate_of_change(df["motor_temperature"])
        df["signal_drop_rate"] = -_rate_of_change(df["signal_strength"])
        df["gps_stability_score"] = 1.0 / (1.0 + _rolling_std(df["gps_accuracy"]))
        df["power_fluctuation_score"] = _rolling_std(df["voltage"]) + _rolling_std(df["current"])
        df["speed_change_rate"] = _rate_of_change(df["speed"]).abs()
        df["altitude_change_rate"] = _rate_of_change(df["altitude"]).abs()

    df["vibration_magnitude"] = _magnitude(df["vibration_x"], df["vibration_y"], df["vibration_z"])
    df["acceleration_magnitude"] = _magnitude(
        df["acceleration_x"], df["acceleration_y"], df["acceleration_z"]
    )

    speed_z = (df["speed"] - df["speed"].mean()) / (df["speed"].std() + 1e-6)
    alt_z = (df["altitude"] - df["altitude"].mean()) / (df["altitude"].std() + 1e-6)
    df["navigation_consistency_score"] = 1.0 / (1.0 + speed_z.abs() + alt_z.abs())

    return df


def add_timeseries_features(
    df: pd.DataFrame,
    window: int = 10,
    group_col: str | None = "system_id",
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    sensors = sensors or TS_FEATURE_SENSORS
    available = [s for s in sensors if s in df.columns]

    if group_col and group_col in df.columns and df[group_col].nunique() > MAX_GROUPS_FOR_TS:
        group_col = None

    if group_col and group_col in df.columns:
        df = df.sort_values(
            [group_col, "timestamp"] if "timestamp" in df.columns else [group_col]
        )

        def _roll(series: pd.Series, op: str) -> pd.Series:
            roll = series.rolling(window=window, min_periods=1)
            if op == "mean":
                return roll.mean()
            if op == "std":
                return roll.std().fillna(0.0)
            if op == "range":
                return (roll.max() - roll.min()).fillna(0.0)
            if op == "energy":
                return series.pow(2).rolling(window=window, min_periods=1).sum()
            raise ValueError(op)

        grouped = df.groupby(group_col, sort=False)
        for sensor in available:
            for stat in TS_FEATURE_STATS:
                df[f"{sensor}_{stat}_w{window}"] = grouped[sensor].transform(
                    lambda s, _op=stat: _roll(s, _op)
                )
    else:
        for sensor in available:
            roll = df[sensor].rolling(window=window, min_periods=1)
            df[f"{sensor}_mean_w{window}"] = roll.mean()
            df[f"{sensor}_std_w{window}"] = roll.std().fillna(0.0)
            df[f"{sensor}_range_w{window}"] = (roll.max() - roll.min()).fillna(0.0)
            df[f"{sensor}_energy_w{window}"] = (
                df[sensor].pow(2).rolling(window=window, min_periods=1).sum()
            )

    return df
