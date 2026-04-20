from src.feature_engineering import (
    DERIVED_FEATURE_COLUMNS,
    add_derived_features,
    add_timeseries_features,
    timeseries_feature_columns,
)


def test_derived_features_present(small_dataset):
    enriched = add_derived_features(small_dataset)
    for col in DERIVED_FEATURE_COLUMNS:
        assert col in enriched.columns, f"missing derived feature {col}"


def test_timeseries_features_present(small_dataset):
    enriched = add_timeseries_features(add_derived_features(small_dataset), window=8)
    expected = timeseries_feature_columns(window=8)
    for col in expected:
        if col.startswith("vibration_magnitude") and "vibration_magnitude" not in enriched.columns:
            continue
        assert col in enriched.columns, f"missing TS feature {col}"


def test_vibration_magnitude_is_nonnegative(small_dataset):
    enriched = add_derived_features(small_dataset)
    assert (enriched["vibration_magnitude"] >= 0).all()


def test_no_nans_after_pipeline(small_dataset):
    enriched = add_timeseries_features(add_derived_features(small_dataset))
    assert enriched.isna().sum().sum() == 0
