from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CounterfactualResult:
    success: bool
    original_prediction: int
    target_prediction: int
    steps: int
    perturbation: np.ndarray
    counterfactual: np.ndarray
    counterfactual_prediction: int
    counterfactual_proba: np.ndarray


def search_counterfactual(
    model,
    sample: np.ndarray,
    target_class: int,
    feature_indices: list[int] | None = None,
    step: float = 0.25,
    max_steps: int = 60,
) -> CounterfactualResult:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for counterfactual search.")

    sample = sample.astype(float).copy()
    n_features = sample.shape[0]
    feature_indices = list(feature_indices) if feature_indices else list(range(n_features))

    original_proba = model.predict_proba(sample.reshape(1, -1))[0]
    original_pred = int(np.argmax(original_proba))

    if original_pred == target_class:
        return CounterfactualResult(
            success=True,
            original_prediction=original_pred,
            target_prediction=target_class,
            steps=0,
            perturbation=np.zeros_like(sample),
            counterfactual=sample.copy(),
            counterfactual_prediction=original_pred,
            counterfactual_proba=original_proba,
        )

    current = sample.copy()
    best_proba = original_proba
    for iteration in range(max_steps):
        best_gain = 0.0
        best_change = None
        for idx in feature_indices:
            for direction in (-1.0, 1.0):
                trial = current.copy()
                trial[idx] = trial[idx] + direction * step
                proba = model.predict_proba(trial.reshape(1, -1))[0]
                gain = proba[target_class] - best_proba[target_class]
                if gain > best_gain:
                    best_gain = gain
                    best_change = (idx, direction, proba, trial)

        if best_change is None:
            break
        _, _, proba, trial = best_change
        current = trial
        best_proba = proba
        if int(np.argmax(proba)) == target_class:
            return CounterfactualResult(
                success=True,
                original_prediction=original_pred,
                target_prediction=target_class,
                steps=iteration + 1,
                perturbation=current - sample,
                counterfactual=current,
                counterfactual_prediction=target_class,
                counterfactual_proba=proba,
            )

    return CounterfactualResult(
        success=False,
        original_prediction=original_pred,
        target_prediction=target_class,
        steps=max_steps,
        perturbation=current - sample,
        counterfactual=current,
        counterfactual_prediction=int(np.argmax(best_proba)),
        counterfactual_proba=best_proba,
    )
