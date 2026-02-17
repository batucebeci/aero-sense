from __future__ import annotations

import io

import numpy as np
import pandas as pd

from src.alarms import get_alarm_log
from src.anomaly_detection import anomaly_breakdown, fit_isolation_forest, score_anomalies
from src.data_loader import load_sensor_csv, validate_columns
from src.drift_detection import feature_drift_report, overall_drift_summary
from src.feature_engineering import add_derived_features, add_timeseries_features
from src.hyperparameter_tuning import tune_model
from src.model_evaluation import best_model_name, evaluate_all
from src.model_registry import register_model
from src.model_training import build_model, save_model, train_all_models
from src.preprocessing import PreparedData, prepare_training_data
from src.rul import train_rul_model
from src.run_logger import log_run
from src.sensor_fusion import fuse_sensors
from src.synthetic_sensor_generator import generate_sensor_data
from src.threshold_tuning import (
    apply_thresholds,
    optimize_thresholds_for_f1,
    optimize_thresholds_for_recall,
)
from src.utils import MODELS_DIR, risk_level_for

from .state import AppState


def generate_synthetic(state: AppState, samples_per_class: int, seed: int) -> pd.DataFrame:
    df = generate_sensor_data(samples_per_class=samples_per_class, seed=seed)
    _set_dataset(state, df)
    return df


def load_uploaded_csv(state: AppState, file_bytes: bytes) -> pd.DataFrame:
    df = load_sensor_csv(io.BytesIO(file_bytes))
    ok, missing = validate_columns(df)
    if not ok:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    _set_dataset(state, df)
    return df


def _set_dataset(state: AppState, df: pd.DataFrame) -> None:
    fused = fuse_sensors(add_timeseries_features(add_derived_features(df)))
    with state.lock:
        state.raw_df = df
        state.fused_df = fused
        if state.reference_df is None:
            state.reference_df = fused.copy()
    state.reset_models()


def train_models(state: AppState) -> dict:
    if state.fused_df is None:
        raise RuntimeError("No dataset loaded. Generate or upload data first.")

    prepared: PreparedData = prepare_training_data(state.fused_df)
    with state.lock:
        state.prepared = prepared

    trained = train_all_models(prepared)
    results = evaluate_all(trained, prepared)
    best_name = best_model_name(results)

    state.update_after_train(trained, results, best_name)
    save_model(trained[best_name], prepared, models_dir=MODELS_DIR)

    best_result = results[best_name]
    register_model(
        payload={
            "model": trained[best_name].model,
            "scaler": prepared.scaler,
            "label_encoder": prepared.label_encoder,
            "feature_names": prepared.feature_names,
            "class_names": prepared.class_names,
            "model_name": best_name,
        },
        metric_f1=best_result.f1,
        metric_accuracy=best_result.accuracy,
        model_name=best_name,
    )
    log_run(
        kind="train",
        params={"models": list(trained.keys()), "n_features": len(prepared.feature_names)},
        metrics={
            "best_model": best_name,
            "accuracy": best_result.accuracy,
            "precision_macro": best_result.precision,
            "recall_macro": best_result.recall,
            "f1_macro": best_result.f1,
        },
        extra={"all_f1": {name: r.f1 for name, r in results.items()}},
    )

    return {
        "best_model": best_name,
        "metrics": best_result,
        "n_train": int(prepared.X_train.shape[0]),
        "n_test": int(prepared.X_test.shape[0]),
    }


def retrain_with_tuned_params(state: AppState, model_name: str, params: dict) -> dict:
    if state.prepared is None:
        raise RuntimeError("Train baseline models first so that PreparedData is available.")

    prepared = state.prepared
    custom = build_model(model_name, params)

    import time
    t0 = time.perf_counter()
    custom.fit(prepared.X_train, prepared.y_train)
    train_seconds = time.perf_counter() - t0

    y_pred = custom.predict(prepared.X_test)
    y_proba = custom.predict_proba(prepared.X_test) if hasattr(custom, "predict_proba") else None

    from src.model_evaluation import evaluate
    from src.model_training import TrainedModel

    trained = TrainedModel(
        name=model_name,
        model=custom,
        train_seconds=train_seconds,
        predict_seconds=0.0,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    result = evaluate(trained, prepared)
    state.update_model(model_name, trained, result)
    with state.lock:
        state.best_model_name = best_model_name(state.evaluation_results)
    return {"model": model_name, "f1_macro": result.f1, "params": params}


def fit_anomaly_layer(state: AppState, contamination: float = 0.08) -> dict:
    if state.prepared is None:
        raise RuntimeError("Train models first so that PreparedData is available.")

    model = fit_isolation_forest(state.prepared, contamination=contamination)
    result = score_anomalies(model, state.prepared.X_test)
    state.set_anomaly(model, result)

    breakdown = anomaly_breakdown(
        result.anomaly_flags, state.prepared.y_test, state.prepared.class_names
    )
    log_run(
        kind="anomaly",
        params={"contamination": contamination},
        metrics={
            "n_anomalies": result.n_anomalies,
            "threshold": result.threshold,
        },
        extra={"breakdown": breakdown.to_dict("records")},
    )
    return {
        "n_anomalies": result.n_anomalies,
        "threshold": result.threshold,
        "breakdown": breakdown,
    }


def train_rul(state: AppState, model_kind: str = "random_forest") -> dict:
    if state.fused_df is None:
        raise RuntimeError("No dataset loaded.")
    artifacts = train_rul_model(state.fused_df, model_kind=model_kind)
    state.set_rul(artifacts)
    log_run(
        kind="rul",
        params={"model_kind": model_kind, "n_features": len(artifacts.feature_names)},
        metrics={"mae": artifacts.mae, "rmse": artifacts.rmse, "r2": artifacts.r2},
    )
    return {"mae": artifacts.mae, "rmse": artifacts.rmse, "r2": artifacts.r2}


def run_tuning(
    state: AppState,
    model_name: str,
    n_trials: int = 20,
    timeout_seconds: float | None = 120,
) -> dict:
    if state.prepared is None:
        raise RuntimeError("Train models first so that PreparedData is available.")
    result = tune_model(
        model_name, state.prepared, n_trials=n_trials, timeout_seconds=timeout_seconds
    )
    state.set_tuning(model_name, result)
    log_run(
        kind="tune",
        params={"model": model_name, "n_trials": n_trials, "best_params": result.best_params},
        metrics={"best_f1_macro": result.best_score, "elapsed_seconds": result.elapsed_seconds},
    )
    return {
        "best_score": result.best_score,
        "best_params": result.best_params,
        "elapsed_seconds": result.elapsed_seconds,
        "n_trials": result.n_trials,
    }


def compute_drift(state: AppState) -> dict:
    if state.reference_df is None or state.fused_df is None:
        raise RuntimeError("Need both a reference and a current dataset to compute drift.")
    report = feature_drift_report(state.reference_df, state.fused_df)
    summary = overall_drift_summary(report)
    state.set_drift(report, summary)
    log_run(
        kind="drift",
        params={"n_features": len(report)},
        metrics=summary,
    )
    return {"report": report, "summary": summary}


def tune_thresholds(
    state: AppState,
    strategy: str = "recall",
    target_recall: float = 0.9,
    model_name: str | None = None,
) -> dict:
    if state.prepared is None or not state.trained_models:
        raise RuntimeError("Train models before tuning thresholds.")
    name = model_name or state.best_model_name
    if name not in state.trained_models:
        raise ValueError(f"Unknown model: {name}")
    trained = state.trained_models[name]
    if trained.y_proba is None:
        raise RuntimeError(f"Model {name} does not expose predict_proba.")

    prepared = state.prepared
    if strategy == "f1":
        thresholds = optimize_thresholds_for_f1(
            prepared.y_test, trained.y_proba, prepared.class_names
        )
    else:
        thresholds = optimize_thresholds_for_recall(
            prepared.y_test, trained.y_proba, prepared.class_names, target_recall=target_recall
        )
    state.set_thresholds(thresholds)
    log_run(
        kind="threshold",
        params={"strategy": strategy, "target_recall": target_recall, "model": name},
        metrics={"thresholds": thresholds},
    )
    return {"strategy": strategy, "target_recall": target_recall, "thresholds": thresholds}


def emit_alarms_for_predictions(records: pd.DataFrame) -> int:
    alarm_log = get_alarm_log()
    count = 0
    for _, row in records.iterrows():
        risk = risk_level_for(row.get("predicted_fault", ""))
        alarm = alarm_log.add_from_prediction(
            fault=row.get("predicted_fault", "Unknown"),
            risk=risk,
            confidence=row.get("confidence"),
            recommendation=row.get("recommendation", ""),
            system_id=row.get("system_id"),
        )
        if alarm is not None:
            count += 1
    return count


def predict_on_current_dataset(state: AppState, model_name: str | None = None) -> pd.DataFrame:
    if state.prepared is None or not state.trained_models:
        raise RuntimeError("Train models before running predictions.")

    name = model_name or state.best_model_name
    if name not in state.trained_models:
        raise ValueError(f"Unknown model: {name}")

    trained = state.trained_models[name]
    prepared = state.prepared

    from src.report_generator import build_prediction_records

    y_proba = trained.y_proba
    if state.per_class_thresholds and y_proba is not None:
        y_pred = apply_thresholds(y_proba, prepared.class_names, state.per_class_thresholds)
    else:
        y_pred = trained.y_pred

    test_size = prepared.X_test.shape[0]
    raw = state.raw_df.reset_index(drop=True) if state.raw_df is not None else None
    timestamps = raw["timestamp"].iloc[:test_size] if raw is not None and "timestamp" in raw.columns else None
    system_ids = raw["system_id"].iloc[:test_size] if raw is not None and "system_id" in raw.columns else None

    records = build_prediction_records(
        y_pred,
        prepared.class_names,
        y_proba,
        timestamps=timestamps,
        system_ids=system_ids,
    )

    actual = np.array([prepared.class_names[int(i)] for i in prepared.y_test])
    records.insert(2, "actual_fault", actual)
    records["correct"] = records["actual_fault"] == records["predicted_fault"]

    state.set_prediction(records)
    emit_alarms_for_predictions(records)
    log_run(
        kind="predict",
        params={"model": name},
        metrics={
            "n_samples": len(records),
            "average_confidence": float(records["confidence"].mean())
            if records["confidence"].notna().any()
            else None,
        },
    )
    return records
