import numpy as np

from src.threshold_tuning import (
    apply_thresholds,
    optimize_thresholds_for_f1,
    optimize_thresholds_for_recall,
    per_class_metrics_at_thresholds,
    threshold_for_max_f1,
    threshold_for_target_recall,
)


def _imbalanced_proba_setup():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 200 + [1] * 5 + [2] * 5)
    proba = np.zeros((len(y_true), 3))
    proba[y_true == 0] = np.stack([rng.uniform(0.6, 0.95, 200), rng.uniform(0.02, 0.2, 200), rng.uniform(0.02, 0.2, 200)], axis=1)
    proba[y_true == 1] = np.stack([rng.uniform(0.2, 0.5, 5), rng.uniform(0.4, 0.7, 5), rng.uniform(0.05, 0.2, 5)], axis=1)
    proba[y_true == 2] = np.stack([rng.uniform(0.2, 0.5, 5), rng.uniform(0.05, 0.2, 5), rng.uniform(0.4, 0.7, 5)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return y_true, proba, ["Normal", "Rare-A", "Rare-B"]


def test_threshold_for_target_recall_meets_target():
    y_true, proba, _ = _imbalanced_proba_setup()
    y_bin = (y_true == 1).astype(int)
    thr, prec, recall = threshold_for_target_recall(y_bin, proba[:, 1], target_recall=0.8)
    assert 0.0 <= thr <= 1.0
    assert recall >= 0.8 - 1e-6


def test_threshold_for_max_f1_returns_finite():
    y_true, proba, _ = _imbalanced_proba_setup()
    y_bin = (y_true == 1).astype(int)
    thr, f1, p, r = threshold_for_max_f1(y_bin, proba[:, 1])
    assert 0.0 <= thr <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_apply_thresholds_falls_back_to_argmax():
    y_true, proba, names = _imbalanced_proba_setup()
    thresholds = {name: 0.99 for name in names}
    preds = apply_thresholds(proba, names, thresholds)
    assert (preds == proba.argmax(axis=1)).all()


def test_optimize_for_recall_raises_minority_recall():
    y_true, proba, names = _imbalanced_proba_setup()
    default_preds = proba.argmax(axis=1)
    default_rec_rareA = ((default_preds == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())

    thresholds = optimize_thresholds_for_recall(y_true, proba, names, target_recall=0.9)
    tuned_preds = apply_thresholds(proba, names, thresholds)
    tuned_rec_rareA = ((tuned_preds == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
    assert tuned_rec_rareA >= default_rec_rareA


def test_optimize_for_f1_improves_macro_f1():
    y_true, proba, names = _imbalanced_proba_setup()
    before = per_class_metrics_at_thresholds(y_true, proba, names, None)
    thresholds = optimize_thresholds_for_f1(y_true, proba, names)
    after = per_class_metrics_at_thresholds(y_true, proba, names, thresholds)
    assert after["f1"].mean() >= before["f1"].mean() - 1e-6
