from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import add_derived_features, add_timeseries_features
from src.preprocessing import prepare_training_data
from src.sensor_fusion import fuse_sensors
from src.synthetic_sensor_generator import generate_sensor_data


@pytest.fixture(scope="session")
def small_dataset():
    return generate_sensor_data(samples_per_class=60, seed=7)


@pytest.fixture(scope="session")
def fused_dataset(small_dataset):
    return fuse_sensors(add_timeseries_features(add_derived_features(small_dataset)))


@pytest.fixture(scope="session")
def prepared_dataset(fused_dataset):
    return prepare_training_data(fused_dataset)
