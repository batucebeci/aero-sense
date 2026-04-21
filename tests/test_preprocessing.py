import numpy as np
import pandas as pd
import pytest

from src.preprocessing import prepare_training_data, select_feature_matrix
from src.synthetic_sensor_generator import generate_sensor_data


def test_preprocessing_shapes(prepared_dataset):
    p = prepared_dataset
    assert p.X_train.shape[1] == p.X_test.shape[1] == len(p.feature_names)
    assert p.X_train.shape[0] > p.X_test.shape[0]


def test_labels_cover_all_classes(prepared_dataset):
    p = prepared_dataset
    assert set(np.unique(p.y_train)).union(set(np.unique(p.y_test))) == set(
        range(len(p.class_names))
    )


def test_scaler_fitted(prepared_dataset):
    p = prepared_dataset
    mean = p.X_train.mean(axis=0)
    assert np.allclose(mean, 0, atol=1e-6)


def test_select_feature_matrix_drops_non_numeric():
    df = generate_sensor_data(samples_per_class=20, seed=1)
    df["extra_text"] = "junk"
    X, names = select_feature_matrix(df)
    assert "extra_text" not in names
    assert all(np.issubdtype(X[c].dtype, np.number) for c in X.columns)


def test_prepare_training_data_handles_extra_string_column():
    df = generate_sensor_data(samples_per_class=20, seed=2)
    df["extra_text"] = "junk"
    prepared = prepare_training_data(df)
    assert "extra_text" not in prepared.feature_names


def test_prepare_training_data_rejects_tiny_dataset():
    df = generate_sensor_data(samples_per_class=1, seed=3)
    with pytest.raises(ValueError, match="at least"):
        prepare_training_data(df.head(5))


def test_prepare_training_data_falls_back_when_class_under_2():
    df = generate_sensor_data(samples_per_class=20, seed=4)
    rare = df[df["fault_type"] == "Normal"].head(1)
    others = df[df["fault_type"] != "Normal"]
    mixed = pd.concat([rare, others], ignore_index=True)
    prepared = prepare_training_data(mixed)
    assert prepared.X_train.shape[0] > 0
