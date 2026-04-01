from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import DATA_DIR

AI4I_DIR = DATA_DIR / "ai4i"

FAULT_MAPPING = {
    "HDF": "Motor Overheating",
    "PWF": "Power Instability",
    "OSF": "Vibration Fault",
    "TWF": "Sensor Drift",
    "RNF": "Communication Fault",
}


def _label_row(row: pd.Series) -> str:
    if row.get("Machine failure", 0) == 0:
        return "Normal"
    for code, name in FAULT_MAPPING.items():
        if row.get(code, 0) == 1:
            return name
    return "Normal"


def load_ai4i(
    file_name: str = "ai4i2020.csv",
    base_dir: Path | None = None,
    sample_interval_seconds: float = 60.0,
) -> pd.DataFrame:
    base_dir = Path(base_dir) if base_dir else AI4I_DIR
    path = base_dir / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"AI4I file not found: {path}. "
            "Download from UCI: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset"
        )

    src = pd.read_csv(path)
    src.columns = [c.strip().lstrip("﻿") for c in src.columns]

    n = len(src)
    rng = np.random.default_rng(0)
    base_time = datetime(2025, 1, 1)

    air_t_c = src["Air temperature [K]"] - 273.15
    proc_t_c = src["Process temperature [K]"] - 273.15
    rpm = src["Rotational speed [rpm]"]
    torque = src["Torque [Nm]"]
    wear = src["Tool wear [min]"]

    out = pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=i * sample_interval_seconds) for i in range(n)],
        "system_id": "AI4I-TYPE-" + src["Type"].astype(str),
        "motor_temperature": proc_t_c.astype(float),
        "battery_level": np.clip(100 - wear * 0.4, 20, 100),
        "voltage": 23.5 + (rpm - rpm.mean()) / rpm.std() * 0.4,
        "current": 11.0 + (torque - torque.mean()) / max(torque.std(), 1e-6) * 1.5,
        "vibration_x": np.abs(rng.normal(0.15, 0.05, n) + (torque - torque.mean()) / max(torque.std(), 1e-6) * 0.05),
        "vibration_y": np.abs(rng.normal(0.15, 0.05, n)),
        "vibration_z": np.abs(rng.normal(0.15, 0.05, n)),
        "acceleration_x": rng.normal(0.0, 0.2, n),
        "acceleration_y": rng.normal(0.0, 0.2, n),
        "acceleration_z": rng.normal(9.81, 0.25, n),
        "gps_accuracy": rng.normal(2.0, 0.4, n),
        "gps_satellite_count": rng.integers(9, 15, n).astype(float),
        "signal_strength": rng.normal(-60, 4, n),
        "altitude": air_t_c.astype(float),
        "speed": rpm.astype(float) / 100.0,
        "heading": rng.uniform(0, 360, n),
        "fault_type": src.apply(_label_row, axis=1),
    })
    return out
