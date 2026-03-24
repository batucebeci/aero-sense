from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from .utils import MODELS_DIR

REGISTRY_DIR = MODELS_DIR / "registry"
REGISTRY_INDEX = REGISTRY_DIR / "index.jsonl"
DEFAULT_MAX_VERSIONS = 10


@dataclass
class ModelEntry:
    version_id: str
    model_name: str
    metric_f1: float
    metric_accuracy: float
    n_features: int
    created_at: str
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "metric_f1": self.metric_f1,
            "metric_accuracy": self.metric_accuracy,
            "n_features": self.n_features,
            "created_at": self.created_at,
            "path": self.path,
        }


def _ensure_dir() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _version_id() -> str:
    return datetime.utcnow().strftime("v%Y%m%d-%H%M%S")


def register_model(
    payload: dict,
    metric_f1: float,
    metric_accuracy: float,
    model_name: str,
    max_versions: int = DEFAULT_MAX_VERSIONS,
) -> ModelEntry:
    _ensure_dir()
    version = _version_id()
    filename = f"{model_name.lower().replace(' ', '_')}_{version}.pkl"
    path = REGISTRY_DIR / filename
    joblib.dump(payload, path)

    entry = ModelEntry(
        version_id=version,
        model_name=model_name,
        metric_f1=float(metric_f1),
        metric_accuracy=float(metric_accuracy),
        n_features=len(payload.get("feature_names", [])),
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        path=str(path),
    )
    with REGISTRY_INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.as_dict()) + "\n")
    _prune(max_versions)
    return entry


def _prune(max_versions: int) -> None:
    if max_versions <= 0:
        return
    entries = list_models()
    if len(entries) <= max_versions:
        return
    keep, drop = entries[:max_versions], entries[max_versions:]
    for old in drop:
        try:
            Path(old.path).unlink(missing_ok=True)
        except OSError:
            pass
    with REGISTRY_INDEX.open("w", encoding="utf-8") as f:
        for e in keep:
            f.write(json.dumps(e.as_dict()) + "\n")


def list_models() -> list[ModelEntry]:
    if not REGISTRY_INDEX.exists():
        return []
    entries: list[ModelEntry] = []
    with REGISTRY_INDEX.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(ModelEntry(**d))
            except (json.JSONDecodeError, TypeError):
                continue
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def load_model(version_id: str) -> dict | None:
    for entry in list_models():
        if entry.version_id == version_id:
            return joblib.load(entry.path)
    return None
