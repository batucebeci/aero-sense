from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import PreparedData
from .utils import MODELS_DIR


def _build_lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _build_xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


MODEL_REGISTRY = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "SVM": lambda: SVC(
        kernel="rbf", probability=True, random_state=42, class_weight="balanced"
    ),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, random_state=42, class_weight="balanced"
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        random_state=42, max_depth=12, class_weight="balanced"
    ),
    "LightGBM": _build_lightgbm,
    "XGBoost": _build_xgboost,
}


def build_model(name: str, params: dict | None = None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    model = MODEL_REGISTRY[name]()
    if params:
        model.set_params(**params)
    return model


@dataclass
class TrainedModel:
    name: str
    model: object
    train_seconds: float
    predict_seconds: float
    y_pred: np.ndarray
    y_proba: np.ndarray | None


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def train_single_model(name: str, data: PreparedData) -> TrainedModel:
    factory = MODEL_REGISTRY[name]
    model = factory()

    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    train_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(data.X_test)
    predict_seconds = time.perf_counter() - t0

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(data.X_test)
    else:
        y_proba = None

    return TrainedModel(
        name=name,
        model=model,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        y_pred=y_pred,
        y_proba=y_proba,
    )


def train_all_models(data: PreparedData) -> dict[str, TrainedModel]:
    return {name: train_single_model(name, data) for name in MODEL_REGISTRY}


def save_model(trained: TrainedModel, data: PreparedData, models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{_slugify(trained.name)}_fault_model.pkl"
    payload = {
        "model": trained.model,
        "scaler": data.scaler,
        "label_encoder": data.label_encoder,
        "feature_names": data.feature_names,
        "class_names": data.class_names,
        "model_name": trained.name,
    }
    joblib.dump(payload, path)
    return path


def load_model(path: Path) -> dict:
    return joblib.load(path)
