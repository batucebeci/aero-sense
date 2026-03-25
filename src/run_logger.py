from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from .utils import OUTPUTS_DIR

RUN_LOG_PATH = OUTPUTS_DIR / "runs" / "run_log.jsonl"


def _ensure_dir() -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _coerce(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value
    return str(value)


def log_run(kind: str, params: dict[str, Any], metrics: dict[str, Any], extra: dict | None = None) -> str:
    _ensure_dir()
    run_id = uuid.uuid4().hex[:12]
    entry = {
        "run_id": run_id,
        "kind": kind,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "params": {k: _coerce(v) for k, v in params.items()},
        "metrics": {k: _coerce(v) for k, v in metrics.items()},
        "extra": {k: _coerce(v) for k, v in (extra or {}).items()},
    }
    with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return run_id


def load_runs() -> pd.DataFrame:
    if not RUN_LOG_PATH.exists():
        return pd.DataFrame(columns=["run_id", "kind", "timestamp"])
    rows = []
    with RUN_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame(columns=["run_id", "kind", "timestamp"])

    flat = []
    for row in rows:
        base = {
            "run_id": row["run_id"],
            "kind": row["kind"],
            "timestamp": row["timestamp"],
        }
        for k, v in row.get("metrics", {}).items():
            base[f"metric.{k}"] = v
        for k, v in row.get("params", {}).items():
            base[f"param.{k}"] = v
        flat.append(base)
    return pd.DataFrame(flat).sort_values("timestamp", ascending=False).reset_index(drop=True)
