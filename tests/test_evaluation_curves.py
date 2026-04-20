import numpy as np

from src.evaluation_curves import (
    calibration_table,
    expected_calibration_error,
    per_class_pr,
    per_class_roc,
)
from src.model_training import train_single_model


def test_roc_pr_calibration(prepared_dataset):
    trained = train_single_model("Logistic Regression", prepared_dataset)
    proba = trained.y_proba
    assert proba is not None

    roc = per_class_roc(prepared_dataset.y_test, proba, prepared_dataset.class_names)
    pr = per_class_pr(prepared_dataset.y_test, proba, prepared_dataset.class_names)
    cal_df = calibration_table(prepared_dataset.y_test, proba, prepared_dataset.class_names)

    for name in roc:
        assert 0.0 <= roc[name]["auc"] <= 1.0
        assert 0.0 <= pr[name]["auc"] <= 1.0
    assert {"class", "mean_predicted", "fraction_positive"}.issubset(cal_df.columns)


def test_ece_bounded(prepared_dataset):
    trained = train_single_model("Logistic Regression", prepared_dataset)
    ece = expected_calibration_error(prepared_dataset.y_test, trained.y_proba)
    assert 0.0 <= ece <= 1.0
