import numpy as np

from src.counterfactual import search_counterfactual
from src.model_training import train_single_model


def test_counterfactual_runs(prepared_dataset):
    trained = train_single_model("Logistic Regression", prepared_dataset)
    sample = prepared_dataset.X_test[0]
    target = (int(prepared_dataset.y_test[0]) + 1) % len(prepared_dataset.class_names)
    result = search_counterfactual(trained.model, sample, target_class=target, max_steps=15, step=0.4)
    assert result.counterfactual.shape == sample.shape
    assert isinstance(result.success, bool)
    if result.success:
        assert result.counterfactual_prediction == target
