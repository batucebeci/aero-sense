from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ShapResult:
    feature_names: list[str]
    class_names: list[str]
    shap_values: np.ndarray
    X: np.ndarray
    base_values: np.ndarray | None


def _is_tree_model(model: ClassifierMixin) -> bool:
    return isinstance(model, RandomForestClassifier | DecisionTreeClassifier)


def build_explainer(model: ClassifierMixin, X_background: np.ndarray):
    if _is_tree_model(model):
        return shap.TreeExplainer(model)
    background = shap.sample(X_background, min(50, len(X_background)), random_state=0)
    return shap.KernelExplainer(model.predict_proba, background)


def compute_shap_values(
    model: ClassifierMixin,
    X: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    max_samples: int = 80,
) -> ShapResult:
    if len(X) > max_samples:
        idx = np.random.default_rng(0).choice(len(X), size=max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = build_explainer(model, X)
    raw = explainer.shap_values(X_sample)

    if isinstance(raw, list):
        shap_array = np.stack(raw, axis=-1)
    else:
        shap_array = np.asarray(raw)

    base_values = getattr(explainer, "expected_value", None)
    if base_values is not None:
        base_values = np.atleast_1d(base_values)

    return ShapResult(
        feature_names=feature_names,
        class_names=class_names,
        shap_values=shap_array,
        X=X_sample,
        base_values=base_values,
    )


def global_feature_importance(result: ShapResult) -> pd.DataFrame:
    vals = np.abs(result.shap_values)
    if vals.ndim == 3:
        vals = vals.mean(axis=2)
    importance = vals.mean(axis=0)
    return (
        pd.DataFrame({"feature": result.feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def waterfall_data(
    model: ClassifierMixin,
    sample: np.ndarray,
    feature_names: list[str],
    predicted_class: int,
    top_k: int = 8,
) -> pd.DataFrame:
    explainer = build_explainer(model, sample.reshape(1, -1))
    raw = explainer.shap_values(sample.reshape(1, -1))

    if isinstance(raw, list):
        shap_array = np.stack(raw, axis=-1)
    else:
        shap_array = np.asarray(raw)

    if shap_array.ndim == 3:
        contributions = shap_array[0, :, predicted_class]
    elif shap_array.ndim == 2:
        contributions = shap_array[0]
    else:
        contributions = shap_array

    base_value = getattr(explainer, "expected_value", 0.0)
    if isinstance(base_value, list | np.ndarray):
        base_value = float(np.asarray(base_value).reshape(-1)[predicted_class])
    else:
        base_value = float(base_value)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": sample,
            "shap": contributions,
            "abs_shap": np.abs(contributions),
        }
    ).sort_values("abs_shap", ascending=False)

    top = df.head(top_k).copy()
    remaining_sum = float(df["shap"].iloc[top_k:].sum()) if len(df) > top_k else 0.0
    if remaining_sum != 0.0:
        top = pd.concat(
            [
                top,
                pd.DataFrame(
                    [
                        {
                            "feature": f"Other ({len(df) - top_k})",
                            "value": np.nan,
                            "shap": remaining_sum,
                            "abs_shap": abs(remaining_sum),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    top = top.iloc[::-1].reset_index(drop=True)
    top.insert(0, "base_value", base_value)
    top["cumulative"] = base_value + top["shap"].cumsum()
    return top


def top_features_for_prediction(
    model: ClassifierMixin,
    sample: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    predicted_class: int,
    top_k: int = 5,
) -> pd.DataFrame:
    explainer = build_explainer(model, sample.reshape(1, -1))
    raw = explainer.shap_values(sample.reshape(1, -1))

    if isinstance(raw, list):
        shap_array = np.stack(raw, axis=-1)
    else:
        shap_array = np.asarray(raw)

    if shap_array.ndim == 3:
        contributions = shap_array[0, :, predicted_class]
    elif shap_array.ndim == 2:
        contributions = shap_array[0]
    else:
        contributions = shap_array

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": sample,
            "shap_contribution": contributions,
        }
    )
    df["abs_contribution"] = df["shap_contribution"].abs()
    return df.sort_values("abs_contribution", ascending=False).head(top_k).reset_index(drop=True)
