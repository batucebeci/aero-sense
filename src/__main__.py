from __future__ import annotations

import json
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd

from .anomaly_detection import fit_isolation_forest, score_anomalies
from .feature_engineering import add_derived_features, add_timeseries_features
from .model_evaluation import comparison_table, evaluate_all
from .model_training import train_all_models
from .preprocessing import prepare_training_data
from .rul import train_rul_model
from .sensor_fusion import fuse_sensors
from .synthetic_sensor_generator import generate_sensor_data


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--samples-per-class", default=300, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def generate(samples_per_class: int, seed: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sensor_data(samples_per_class=samples_per_class, seed=seed)
    df.to_csv(output, index=False)
    click.echo(f"Wrote {len(df)} rows to {output}")


@cli.command()
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def train(data_path: Path, output_path: Path) -> None:
    raw = _load_dataset(data_path)
    fused = fuse_sensors(add_timeseries_features(add_derived_features(raw)))
    prepared = prepare_training_data(fused)
    trained = train_all_models(prepared)
    results = evaluate_all(trained, prepared)
    table = comparison_table(results)
    click.echo(table.to_string(index=False))

    best_name = table.iloc[0]["Model"]
    best = trained[best_name]
    payload = {
        "model": best.model,
        "scaler": prepared.scaler,
        "label_encoder": prepared.label_encoder,
        "feature_names": prepared.feature_names,
        "class_names": prepared.class_names,
        "model_name": best_name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    click.echo(f"Best model: {best_name}")
    click.echo(f"Wrote model to {output_path}")
    click.echo(f"Wrote metrics to {metrics_path}")


@cli.command()
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def predict(data_path: Path, model_path: Path, output_path: Path) -> None:
    raw = _load_dataset(data_path)
    fused = fuse_sensors(add_timeseries_features(add_derived_features(raw)))
    payload = joblib.load(model_path)
    feature_names = payload["feature_names"]
    for c in feature_names:
        if c not in fused.columns:
            fused[c] = 0.0
    X = payload["scaler"].transform(fused[feature_names].values)
    preds = payload["model"].predict(X)
    labels = np.array(payload["class_names"])[preds]
    out = pd.DataFrame(
        {
            "timestamp": raw.get("timestamp"),
            "system_id": raw.get("system_id"),
            "predicted_fault": labels,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    click.echo(f"Wrote {len(out)} predictions to {output_path}")


@cli.command()
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--contamination", default=0.08, show_default=True)
def anomaly(data_path: Path, contamination: float) -> None:
    raw = _load_dataset(data_path)
    fused = fuse_sensors(add_timeseries_features(add_derived_features(raw)))
    prepared = prepare_training_data(fused)
    model = fit_isolation_forest(prepared, contamination=contamination)
    result = score_anomalies(model, prepared.X_test)
    click.echo(
        json.dumps(
            {
                "n_anomalies": result.n_anomalies,
                "threshold": result.threshold,
                "contamination": result.contamination,
            },
            indent=2,
        )
    )


@cli.command()
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-kind", default="random_forest", show_default=True)
def rul(data_path: Path, model_kind: str) -> None:
    raw = _load_dataset(data_path)
    fused = fuse_sensors(add_timeseries_features(add_derived_features(raw)))
    artifacts = train_rul_model(fused, model_kind=model_kind)
    click.echo(
        json.dumps(
            {"mae": artifacts.mae, "rmse": artifacts.rmse, "r2": artifacts.r2},
            indent=2,
        )
    )


if __name__ == "__main__":
    cli()
