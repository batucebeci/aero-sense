from src.feature_engineering import add_derived_features
from src.sensor_fusion import META_COLUMNS, fuse_sensors


def test_fused_includes_cross_domain_indices(small_dataset):
    fused = fuse_sensors(add_derived_features(small_dataset))
    for col in ("thermal_power_index", "mechanical_stress_index", "comm_nav_health_index"):
        assert col in fused.columns


def test_meta_columns_at_front(small_dataset):
    fused = fuse_sensors(add_derived_features(small_dataset))
    expected_meta = [c for c in META_COLUMNS if c in fused.columns]
    assert list(fused.columns[: len(expected_meta)]) == expected_meta
