from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from .synthetic_sensor_generator import RAW_SENSOR_COLUMNS

REQUIRED_COLUMNS = set(RAW_SENSOR_COLUMNS) | {"timestamp", "fault_type"}

PathLike = str | Path | IO


def load_sensor_csv(source: PathLike) -> pd.DataFrame:
    try:
        df = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"Could not parse CSV: {e}") from e
    if df.shape[1] < 2 or df.empty:
        raise ValueError("CSV must contain at least 2 columns and 1 row of data.")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = sorted(c for c in REQUIRED_COLUMNS if c not in df.columns and c != "fault_type")
    return (len(missing) == 0), missing


def basic_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": len(df),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
    }
    if "fault_type" in df.columns:
        summary["class_distribution"] = df["fault_type"].value_counts().to_dict()
    return summary
