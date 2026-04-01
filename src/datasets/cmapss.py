from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import DATA_DIR

CMAPSS_DIR = DATA_DIR / "cmapss"

CMAPSS_COLUMNS = (
    ["unit_id", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_MAP = {
    "motor_temperature": "sensor_2",
    "battery_level": "sensor_9",
    "voltage": "sensor_14",
    "current": "sensor_3",
    "vibration_x": "sensor_11",
    "vibration_y": "sensor_12",
    "vibration_z": "sensor_13",
    "acceleration_x": "op_setting_1",
    "acceleration_y": "op_setting_2",
    "acceleration_z": "op_setting_3",
    "gps_accuracy": "sensor_7",
    "gps_satellite_count": "sensor_4",
    "signal_strength": "sensor_8",
    "altitude": "sensor_15",
    "speed": "sensor_17",
    "heading": "sensor_20",
}


def _label_fault_type(rul: pd.Series, healthy_cutoff: int = 50) -> pd.Series:
    labels = np.where(rul <= healthy_cutoff, "Motor Overheating", "Normal")
    return pd.Series(labels, index=rul.index, name="fault_type")


def load_cmapss(
    file_name: str = "train_FD001.txt",
    base_dir: Path | None = None,
    sample_interval_seconds: float = 1.0,
) -> pd.DataFrame:
    base_dir = Path(base_dir) if base_dir else CMAPSS_DIR
    path = base_dir / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"CMAPSS file not found: {path}. "
            "Download from NASA's Prognostics Center and place under data/cmapss/."
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)

    max_cycles = df.groupby("unit_id")["time_in_cycles"].transform("max")
    rul = max_cycles - df["time_in_cycles"]
    df["fault_type"] = _label_fault_type(rul)

    out = pd.DataFrame()
    base_time = datetime(2025, 1, 1)
    out["timestamp"] = [
        base_time + timedelta(seconds=int(t) * sample_interval_seconds)
        for t in df["time_in_cycles"].values
    ]
    out["system_id"] = "CMAPSS-" + df["unit_id"].astype(str).str.zfill(3)
    for target, source in SENSOR_MAP.items():
        out[target] = df[source] if source in df.columns else 0.0
    out["fault_type"] = df["fault_type"]
    return out
