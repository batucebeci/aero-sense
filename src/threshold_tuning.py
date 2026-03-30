from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize


@dataclass
class ClassThresholdInfo:
    class_name: str
    n_positive: int
    default_threshold: float
    max_f1_threshold: float
    max_f1: float
    selected_threshold: float
    precision_at_selected: float
    recall_at_selected: float


def _binary_pr(y_true_bin: np.ndarray, y_proba_class: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return precision_recall_curve(y_true_bin, y_proba_class)


def threshold_for_target_recall(
    y_true_bin: np.ndarray, y_proba_class: np.ndarray, target_recall: float
) -> tuple[float, float, float]:
    precisions, recalls, thresholds = _binary_pr(y_true_bin, y_proba_class)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0
    valid = recalls[:-1] >= target_recall
    if not valid.any():
        idx = int(np.argmax(recalls[:-1]))
    else:
        idx = int(np.where(valid)[0][-1])
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])


def threshold_for_max_f1(
    y_true_bin: np.ndarray, y_proba_class: np.ndarray
) -> tuple[float, float, float, float]:
    precisions, recalls, thresholds = _binary_pr(y_true_bin, y_proba_class)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx]), float(precisions[idx]), float(recalls[idx])


def optimize_thresholds_for_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    target_recall: float = 0.9,
) -> dict[str, float]:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)
    out: dict[str, float] = {}
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            out[name] = 0.5
            continue
        thr, _, _ = threshold_for_target_recall(y_bin[:, i], y_proba[:, i], target_recall)
        out[name] = thr
    return out


def optimize_thresholds_for_f1(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)
    out: dict[str, float] = {}
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            out[name] = 0.5
            continue
        thr, _, _, _ = threshold_for_max_f1(y_bin[:, i], y_proba[:, i])
        out[name] = thr
    return out


def apply_thresholds(
    y_proba: np.ndarray, class_names: list[str], thresholds: dict[str, float]
) -> np.ndarray:
    n_samples, n_classes = y_proba.shape
    thresh = np.array(
        [thresholds.get(class_names[i], 0.5) for i in range(n_classes)], dtype=float
    )
    fires = y_proba >= thresh
    relative = y_proba - thresh
    preds = np.empty(n_samples, dtype=int)
    for row in range(n_samples):
        if fires[row].any():
            masked = np.where(fires[row], relative[row], -np.inf)
            preds[row] = int(np.argmax(masked))
        else:
            preds[row] = int(np.argmax(y_proba[row]))
    return preds


def threshold_overview(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    selected_thresholds: dict[str, float],
) -> list[ClassThresholdInfo]:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)
    rows = []
    for i, name in enumerate(class_names):
        positive = int(y_bin[:, i].sum())
        if positive == 0:
            rows.append(
                ClassThresholdInfo(
                    class_name=name,
                    n_positive=0,
                    default_threshold=0.5,
                    max_f1_threshold=0.5,
                    max_f1=0.0,
                    selected_threshold=selected_thresholds.get(name, 0.5),
                    precision_at_selected=0.0,
                    recall_at_selected=0.0,
                )
            )
            continue
        max_thr, max_f1, _, _ = threshold_for_max_f1(y_bin[:, i], y_proba[:, i])
        sel_thr = selected_thresholds.get(name, max_thr)
        prec_sel = float(((y_proba[:, i] >= sel_thr) & (y_bin[:, i] == 1)).sum()
                         / max(1, (y_proba[:, i] >= sel_thr).sum()))
        rec_sel = float(((y_proba[:, i] >= sel_thr) & (y_bin[:, i] == 1)).sum()
                        / max(1, positive))
        rows.append(
            ClassThresholdInfo(
                class_name=name,
                n_positive=positive,
                default_threshold=0.5,
                max_f1_threshold=max_thr,
                max_f1=max_f1,
                selected_threshold=sel_thr,
                precision_at_selected=prec_sel,
                recall_at_selected=rec_sel,
            )
        )
    return rows


def per_class_metrics_at_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    thresholds: dict[str, float] | None,
) -> pd.DataFrame:
    if thresholds is None:
        preds = np.argmax(y_proba, axis=1)
    else:
        preds = apply_thresholds(y_proba, class_names, thresholds)

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)

    rows = []
    for i, name in enumerate(class_names):
        positive = int(y_bin[:, i].sum())
        if positive == 0:
            continue
        f1 = float(f1_score(y_true == i, preds == i, zero_division=0))
        tp = int(((preds == i) & (y_true == i)).sum())
        fp = int(((preds == i) & (y_true != i)).sum())
        fn = int(((preds != i) & (y_true == i)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        rows.append(
            {
                "class": name,
                "support": positive,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "threshold": round(thresholds.get(name, 0.5) if thresholds else 0.5, 4),
            }
        )
    return pd.DataFrame(rows)
