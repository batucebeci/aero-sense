import numpy as np
import pandas as pd

from src.drift_detection import feature_drift_report, overall_drift_summary


def test_no_drift_when_identical():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    report = feature_drift_report(df, df.copy())
    assert (report["psi"].abs() < 1e-6).all()
    summary = overall_drift_summary(report)
    assert summary["overall"] == "stable"


def test_severe_drift_detected():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=400)})
    cur = pd.DataFrame({"a": rng.normal(loc=5, size=400)})
    report = feature_drift_report(ref, cur)
    assert report["psi"].iloc[0] > 0.25
    assert report["psi_level"].iloc[0] == "severe"
