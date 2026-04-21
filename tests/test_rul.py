from src.rul import build_rul_labels, train_rul_model


def test_rul_labels_monotonically_decreasing(small_dataset):
    labels = build_rul_labels(small_dataset)
    for sid, sub in small_dataset.groupby("system_id"):
        vals = labels.loc[sub.index].tolist()
        assert vals == sorted(vals, reverse=True)


def test_rul_model_trains(fused_dataset):
    artifacts = train_rul_model(fused_dataset)
    assert artifacts.mae >= 0
    assert artifacts.rmse >= 0
    assert -1.0 <= artifacts.r2 <= 1.0
    assert artifacts.X_test.shape[1] == len(artifacts.feature_names)
