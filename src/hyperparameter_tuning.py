from __future__ import annotations

import time
from dataclasses import dataclass

import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import PreparedData


@dataclass
class TuningResult:
    model_name: str
    best_params: dict
    best_score: float
    n_trials: int
    elapsed_seconds: float


def _suggest_random_forest(trial: optuna.Trial) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 80, 400, step=40),
        max_depth=trial.suggest_int("max_depth", 4, 24),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def _suggest_svm(trial: optuna.Trial) -> SVC:
    return SVC(
        C=trial.suggest_float("C", 0.1, 20.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        kernel=trial.suggest_categorical("kernel", ["rbf", "poly"]),
        class_weight="balanced",
        probability=True,
        random_state=42,
    )


def _suggest_knn(trial: optuna.Trial) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=trial.suggest_int("n_neighbors", 3, 25),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        p=trial.suggest_categorical("p", [1, 2]),
    )


def _suggest_logistic(trial: optuna.Trial) -> LogisticRegression:
    return LogisticRegression(
        C=trial.suggest_float("C", 0.01, 30.0, log=True),
        max_iter=trial.suggest_int("max_iter", 800, 4000, step=200),
        class_weight="balanced",
        random_state=42,
    )


def _suggest_decision_tree(trial: optuna.Trial) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight="balanced",
        random_state=42,
    )


def _suggest_lightgbm(trial: optuna.Trial):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _suggest_xgboost(trial: optuna.Trial):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


SUGGESTERS = {
    "Random Forest": _suggest_random_forest,
    "SVM": _suggest_svm,
    "KNN": _suggest_knn,
    "Logistic Regression": _suggest_logistic,
    "Decision Tree": _suggest_decision_tree,
    "LightGBM": _suggest_lightgbm,
    "XGBoost": _suggest_xgboost,
}


def tune_model(
    model_name: str,
    data: PreparedData,
    n_trials: int = 25,
    cv_folds: int = 3,
    timeout_seconds: float | None = 120,
) -> TuningResult:
    if model_name not in SUGGESTERS:
        raise ValueError(f"No suggester registered for {model_name}")

    suggester = SUGGESTERS[model_name]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        clf = suggester(trial)
        scores = cross_val_score(
            clf, data.X_train, data.y_train, cv=cv_folds, scoring="f1_macro", n_jobs=-1,
        )
        return float(scores.mean())

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    return TuningResult(
        model_name=model_name,
        best_params=study.best_params,
        best_score=float(study.best_value),
        n_trials=len(study.trials),
        elapsed_seconds=elapsed,
    )
