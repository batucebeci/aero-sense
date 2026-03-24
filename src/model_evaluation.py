from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .model_training import TrainedModel
from .preprocessing import PreparedData


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_seconds: float
    predict_seconds: float
    confusion: np.ndarray
    per_class_f1: dict[str, float]
    classification_report: str


def evaluate(trained: TrainedModel, data: PreparedData) -> EvaluationResult:
    y_true = data.y_test
    y_pred = trained.y_pred
    class_names = data.class_names

    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {class_names[i]: float(per_class[i]) for i in range(len(class_names))}

    return EvaluationResult(
        model_name=trained.name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        train_seconds=trained.train_seconds,
        predict_seconds=trained.predict_seconds,
        confusion=confusion_matrix(y_true, y_pred),
        per_class_f1=per_class_f1,
        classification_report=classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    )


def evaluate_all(
    trained_models: dict[str, TrainedModel], data: PreparedData
) -> dict[str, EvaluationResult]:
    return {name: evaluate(tm, data) for name, tm in trained_models.items()}


def comparison_table(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": r.accuracy,
                "Precision (macro)": r.precision,
                "Recall (macro)": r.recall,
                "F1 (macro)": r.f1,
                "Train time (s)": r.train_seconds,
                "Predict time (s)": r.predict_seconds,
            }
        )
    df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return df


def best_model_name(results: dict[str, EvaluationResult]) -> str:
    return max(results.items(), key=lambda kv: kv[1].f1)[0]
