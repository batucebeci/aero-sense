from src.calibration import calibrate_classifier
from src.model_training import train_single_model


def test_calibration_runs(prepared_dataset):
    trained = train_single_model("Logistic Regression", prepared_dataset)
    result = calibrate_classifier(trained.model, prepared_dataset, method="isotonic")
    assert result.ece_before >= 0
    assert result.ece_after >= 0
    assert result.y_proba_calibrated.shape == trained.y_proba.shape
