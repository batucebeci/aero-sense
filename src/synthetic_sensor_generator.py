from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

FAULT_CLASSES = [
    "Normal",
    "Battery Fault",
    "Motor Overheating",
    "GPS Fault",
    "Communication Fault",
    "Sensor Drift",
    "Vibration Fault",
    "Power Instability",
    "Navigation Instability",
]

RAW_SENSOR_COLUMNS = [
    "motor_temperature",
    "battery_level",
    "voltage",
    "current",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gps_accuracy",
    "gps_satellite_count",
    "signal_strength",
    "altitude",
    "speed",
    "heading",
]


def _base_normal(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "motor_temperature": rng.normal(70, 4, n),
        "battery_level": np.clip(rng.normal(85, 6, n), 30, 100),
        "voltage": rng.normal(23.5, 0.4, n),
        "current": rng.normal(11, 1.2, n),
        "vibration_x": rng.normal(0.15, 0.05, n),
        "vibration_y": rng.normal(0.15, 0.05, n),
        "vibration_z": rng.normal(0.15, 0.05, n),
        "acceleration_x": rng.normal(0.0, 0.2, n),
        "acceleration_y": rng.normal(0.0, 0.2, n),
        "acceleration_z": rng.normal(9.81, 0.25, n),
        "gps_accuracy": rng.normal(2.0, 0.4, n),
        "gps_satellite_count": rng.integers(9, 15, n).astype(float),
        "signal_strength": rng.normal(-60, 4, n),
        "altitude": rng.normal(120, 15, n),
        "speed": rng.normal(15, 3, n),
        "heading": rng.uniform(0, 360, n),
    }


def _inject_fault(data: dict[str, np.ndarray], fault: str, rng: np.random.Generator) -> None:
    n = len(data["motor_temperature"])

    if fault == "Normal":
        return

    if fault == "Battery Fault":
        data["battery_level"] = np.clip(rng.normal(35, 8, n), 5, 60)
        data["voltage"] = rng.normal(20.5, 0.6, n)
        data["current"] = rng.normal(16, 2.5, n)
        return

    if fault == "Motor Overheating":
        data["motor_temperature"] = rng.normal(98, 5, n)
        data["vibration_x"] += rng.normal(0.25, 0.08, n)
        data["vibration_y"] += rng.normal(0.25, 0.08, n)
        data["current"] = rng.normal(14, 1.5, n)
        return

    if fault == "GPS Fault":
        data["gps_accuracy"] = rng.normal(8.5, 2.0, n)
        data["gps_satellite_count"] = rng.integers(2, 6, n).astype(float)
        data["altitude"] += rng.normal(0, 25, n)
        return

    if fault == "Communication Fault":
        data["signal_strength"] = rng.normal(-92, 6, n)
        data["gps_satellite_count"] = rng.integers(4, 9, n).astype(float)
        return

    if fault == "Sensor Drift":
        drift = np.linspace(0, rng.uniform(10, 18), n)
        data["motor_temperature"] += drift
        data["acceleration_z"] += np.linspace(0, rng.uniform(1.0, 2.0), n)
        return

    if fault == "Vibration Fault":
        data["vibration_x"] = rng.normal(1.1, 0.3, n)
        data["vibration_y"] = rng.normal(1.0, 0.3, n)
        data["vibration_z"] = rng.normal(1.2, 0.3, n)
        data["acceleration_x"] = rng.normal(0.0, 0.8, n)
        data["acceleration_y"] = rng.normal(0.0, 0.8, n)
        return

    if fault == "Power Instability":
        data["voltage"] = rng.normal(23.5, 1.8, n)
        data["current"] = rng.normal(11, 4.5, n)
        data["battery_level"] = np.clip(rng.normal(70, 12, n), 10, 100)
        return

    if fault == "Navigation Instability":
        data["speed"] = rng.normal(15, 9, n)
        data["altitude"] = rng.normal(120, 60, n)
        data["heading"] = (data["heading"] + rng.normal(0, 80, n)) % 360
        data["gps_accuracy"] = rng.normal(4.5, 1.2, n)
        return

    raise ValueError(f"Unknown fault class: {fault}")


NOISE_REFERENCE_STD = {
    "motor_temperature": 14.0,
    "battery_level": 25.0,
    "voltage": 1.5,
    "current": 2.5,
    "vibration_x": 0.45,
    "vibration_y": 0.45,
    "vibration_z": 0.5,
    "acceleration_x": 0.3,
    "acceleration_y": 0.3,
    "acceleration_z": 0.6,
    "gps_accuracy": 3.0,
    "signal_strength": 15.0,
    "altitude": 25.0,
    "speed": 5.0,
}


def _inject_noise(
    data: dict[str, np.ndarray], rng: np.random.Generator, noise_scale: float
) -> None:
    if noise_scale <= 0:
        return
    for col, ref_std in NOISE_REFERENCE_STD.items():
        if col not in data:
            continue
        arr = np.asarray(data[col], dtype=float)
        data[col] = arr + rng.normal(0.0, ref_std * noise_scale, size=arr.shape)


def generate_sensor_data(
    samples_per_class: int = 400,
    fault_classes: Sequence[str] | None = None,
    seed: int = 42,
    start_time: datetime | None = None,
    sample_interval_seconds: float = 1.0,
    noise_scale: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = list(fault_classes) if fault_classes else list(FAULT_CLASSES)
    start_time = start_time or datetime(2025, 1, 1, 0, 0, 0)

    frames: list[pd.DataFrame] = []
    cursor = start_time
    for system_id, fault in enumerate(classes, start=1):
        data = _base_normal(samples_per_class, rng)
        _inject_fault(data, fault, rng)
        _inject_noise(data, rng, noise_scale)

        timestamps = [
            cursor + timedelta(seconds=i * sample_interval_seconds)
            for i in range(samples_per_class)
        ]
        cursor = timestamps[-1] + timedelta(seconds=sample_interval_seconds)

        df = pd.DataFrame(data)
        df.insert(0, "timestamp", timestamps)
        df.insert(1, "system_id", f"SYS-{system_id:02d}")
        df["fault_type"] = fault
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    return full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
