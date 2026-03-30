from __future__ import annotations

import pandas as pd

from .feature_engineering import DERIVED_FEATURE_COLUMNS
from .synthetic_sensor_generator import RAW_SENSOR_COLUMNS

META_COLUMNS = ("timestamp", "system_id", "fault_type")
CROSS_DOMAIN_COLUMNS = [
    "thermal_power_index",
    "mechanical_stress_index",
    "comm_nav_health_index",
]
FUSED_FEATURE_COLUMNS = RAW_SENSOR_COLUMNS + DERIVED_FEATURE_COLUMNS + CROSS_DOMAIN_COLUMNS


def fuse_sensors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["thermal_power_index"] = (
        (df["motor_temperature"] - 70).clip(lower=0) / 30.0
        + df["current"].abs() / 20.0
    )
    df["mechanical_stress_index"] = (
        df["vibration_magnitude"] + df["acceleration_magnitude"].abs() / 10.0
    )
    df["comm_nav_health_index"] = (
        df["signal_strength"].clip(upper=-40) / -100.0
        + df["gps_satellite_count"] / 15.0
        + df["gps_stability_score"]
    )

    meta = [c for c in META_COLUMNS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta]
    return df[meta + feature_cols]
