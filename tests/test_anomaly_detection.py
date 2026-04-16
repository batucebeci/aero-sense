from src.anomaly_detection import (
    anomaly_breakdown,
    fit_isolation_forest,
    score_anomalies,
)


def test_anomaly_pipeline(prepared_dataset):
    model = fit_isolation_forest(prepared_dataset, contamination=0.1)
    result = score_anomalies(model, prepared_dataset.X_test)
    assert result.scores.shape[0] == prepared_dataset.X_test.shape[0]
    assert result.anomaly_flags.dtype == bool
    assert 0 <= result.n_anomalies <= prepared_dataset.X_test.shape[0]


def test_anomaly_breakdown_columns(prepared_dataset):
    model = fit_isolation_forest(prepared_dataset, contamination=0.1)
    result = score_anomalies(model, prepared_dataset.X_test)
    breakdown = anomaly_breakdown(
        result.anomaly_flags, prepared_dataset.y_test, prepared_dataset.class_names
    )
    assert {"fault_type", "samples", "anomalies", "anomaly_rate"}.issubset(breakdown.columns)
    assert (breakdown["anomaly_rate"] >= 0).all()
    assert (breakdown["anomaly_rate"] <= 1).all()
