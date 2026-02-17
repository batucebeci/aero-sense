from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pandas as pd

from src.anomaly_detection import AnomalyResult
from src.hyperparameter_tuning import TuningResult
from src.model_evaluation import EvaluationResult
from src.model_training import TrainedModel
from src.preprocessing import PreparedData
from src.rul import RULArtifacts


@dataclass
class AppState:
    raw_df: pd.DataFrame | None = None
    fused_df: pd.DataFrame | None = None
    prepared: PreparedData | None = None
    trained_models: dict[str, TrainedModel] = field(default_factory=dict)
    evaluation_results: dict[str, EvaluationResult] = field(default_factory=dict)
    best_model_name: str | None = None
    last_prediction_df: pd.DataFrame | None = None

    anomaly_model: object | None = None
    anomaly_result: AnomalyResult | None = None
    rul_artifacts: RULArtifacts | None = None
    tuning_results: dict[str, TuningResult] = field(default_factory=dict)
    reference_df: pd.DataFrame | None = None
    drift_report: pd.DataFrame | None = None
    drift_summary: dict | None = None
    live_buffer: list = field(default_factory=list)
    per_class_thresholds: dict[str, float] | None = None

    flash_message: tuple[str, str] | None = None

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def reset_models(self) -> None:
        with self._lock:
            self.trained_models = {}
            self.evaluation_results = {}
            self.best_model_name = None
            self.last_prediction_df = None
            self.anomaly_model = None
            self.anomaly_result = None
            self.rul_artifacts = None
            self.tuning_results = {}
            self.drift_report = None
            self.drift_summary = None
            self.per_class_thresholds = None

    def set_flash(self, level: str, text: str) -> None:
        with self._lock:
            self.flash_message = (level, text)

    def pop_flash(self) -> tuple[str, str] | None:
        with self._lock:
            flash = self.flash_message
            self.flash_message = None
            return flash

    def update_after_train(
        self,
        trained: dict[str, TrainedModel],
        results: dict[str, EvaluationResult],
        best_name: str,
    ) -> None:
        with self._lock:
            self.trained_models = trained
            self.evaluation_results = results
            self.best_model_name = best_name
            self.last_prediction_df = None
            self.tuning_results = {}

    def set_anomaly(self, model: object, result: AnomalyResult) -> None:
        with self._lock:
            self.anomaly_model = model
            self.anomaly_result = result

    def set_rul(self, artifacts: RULArtifacts) -> None:
        with self._lock:
            self.rul_artifacts = artifacts

    def set_tuning(self, model_name: str, result: TuningResult) -> None:
        with self._lock:
            self.tuning_results = {**self.tuning_results, model_name: result}

    def set_prediction(self, df: pd.DataFrame) -> None:
        with self._lock:
            self.last_prediction_df = df

    def set_drift(self, report: pd.DataFrame, summary: dict) -> None:
        with self._lock:
            self.drift_report = report
            self.drift_summary = summary

    def update_model(self, name: str, trained: TrainedModel, result: EvaluationResult) -> None:
        with self._lock:
            self.trained_models = {**self.trained_models, name: trained}
            self.evaluation_results = {**self.evaluation_results, name: result}

    def append_live(self, item: dict, max_size: int = 120) -> None:
        with self._lock:
            self.live_buffer.append(item)
            if len(self.live_buffer) > max_size:
                self.live_buffer = self.live_buffer[-max_size:]

    def live_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.live_buffer)

    def set_thresholds(self, thresholds: dict[str, float] | None) -> None:
        with self._lock:
            self.per_class_thresholds = dict(thresholds) if thresholds else None


_STATE = AppState()


def get_state() -> AppState:
    return _STATE
