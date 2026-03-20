from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize


def per_class_roc(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]
) -> dict[str, dict]:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    out: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        out[name] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    return out


def per_class_pr(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]
) -> dict[str, dict]:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    out: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        out[name] = {
            "precision": precision,
            "recall": recall,
            "auc": float(auc(recall, precision)),
        }
    return out


def calibration_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    rows = []
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        try:
            frac, mean_pred = calibration_curve(
                y_bin[:, i], y_proba[:, i], n_bins=n_bins, strategy="quantile"
            )
        except ValueError:
            continue
        for mp, fp in zip(mean_pred, frac):
            rows.append(
                {"class": name, "mean_predicted": float(mp), "fraction_positive": float(fp)}
            )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    pred_idx = y_proba.argmax(axis=1)
    confidence = y_proba.max(axis=1)
    correct = (pred_idx == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.sum() == 0:
            continue
        bucket_conf = confidence[mask].mean()
        bucket_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bucket_conf - bucket_acc)
    return float(ece)
