from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_BUCKETS = 10
PSI_THRESHOLD_MODERATE = 0.1
PSI_THRESHOLD_SEVERE = 0.25


def _psi(reference: np.ndarray, current: np.ndarray, buckets: int = PSI_BUCKETS) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = np.quantile(reference, np.linspace(0, 1, buckets + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _classify_psi(value: float) -> str:
    if value != value:
        return "unknown"
    if value < PSI_THRESHOLD_MODERATE:
        return "stable"
    if value < PSI_THRESHOLD_SEVERE:
        return "moderate"
    return "severe"


def feature_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    features = features or [
        c
        for c in reference.columns
        if c in current.columns and np.issubdtype(reference[c].dtype, np.number)
    ]

    rows = []
    for feature in features:
        ref = reference[feature].dropna().values
        cur = current[feature].dropna().values
        if len(ref) < 5 or len(cur) < 5:
            continue
        psi = _psi(ref, cur)
        ks_stat, p_value = ks_2samp(ref, cur)
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "psi_level": _classify_psi(psi),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(p_value),
                "reference_mean": float(np.mean(ref)),
                "current_mean": float(np.mean(cur)),
                "mean_shift": float(np.mean(cur) - np.mean(ref)),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def overall_drift_summary(report: pd.DataFrame) -> dict:
    if report.empty:
        return {"overall": "unknown", "n_severe": 0, "n_moderate": 0, "n_stable": 0}
    counts = report["psi_level"].value_counts().to_dict()
    n_severe = int(counts.get("severe", 0))
    n_moderate = int(counts.get("moderate", 0))
    n_stable = int(counts.get("stable", 0))
    if n_severe >= 3:
        overall = "severe"
    elif n_severe + n_moderate >= 5:
        overall = "moderate"
    else:
        overall = "stable"
    return {
        "overall": overall,
        "n_severe": n_severe,
        "n_moderate": n_moderate,
        "n_stable": n_stable,
    }
