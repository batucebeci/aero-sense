from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .preprocessing import PreparedData


@dataclass
class AnomalyResult:
    scores: np.ndarray
    anomaly_flags: np.ndarray
    threshold: float
    contamination: float
    n_anomalies: int


def fit_isolation_forest(
    prepared: PreparedData,
    contamination: float = 0.08,
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(prepared.X_train)
    return model


def score_anomalies(model: IsolationForest, X: np.ndarray) -> AnomalyResult:
    raw_scores = model.score_samples(X)
    threshold = float(np.percentile(raw_scores, 100 * model.contamination))
    flags = raw_scores < threshold
    return AnomalyResult(
        scores=raw_scores,
        anomaly_flags=flags,
        threshold=threshold,
        contamination=float(model.contamination),
        n_anomalies=int(flags.sum()),
    )


def anomaly_breakdown(
    flags: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    out = []
    for idx, name in enumerate(class_names):
        mask = labels == idx
        total = int(mask.sum())
        if total == 0:
            continue
        anomalies = int(flags[mask].sum())
        out.append(
            {
                "fault_type": name,
                "samples": total,
                "anomalies": anomalies,
                "anomaly_rate": anomalies / total,
            }
        )
    return pd.DataFrame(out).sort_values("anomaly_rate", ascending=False).reset_index(drop=True)
