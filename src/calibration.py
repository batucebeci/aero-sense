from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .evaluation_curves import expected_calibration_error
from .preprocessing import PreparedData


@dataclass
class CalibrationResult:
    method: str
    calibrated_model: object
    ece_before: float
    ece_after: float
    y_proba_calibrated: np.ndarray


def calibrate_classifier(
    base_estimator,
    data: PreparedData,
    method: str = "isotonic",
) -> CalibrationResult:
    if not hasattr(base_estimator, "predict_proba"):
        raise ValueError("Base estimator must support predict_proba.")

    proba_before = base_estimator.predict_proba(data.X_test)
    ece_before = expected_calibration_error(data.y_test, proba_before)

    calibrator = CalibratedClassifierCV(FrozenEstimator(base_estimator), method=method, cv=5)
    calibrator.fit(data.X_train, data.y_train)

    proba_after = calibrator.predict_proba(data.X_test)
    ece_after = expected_calibration_error(data.y_test, proba_after)

    return CalibrationResult(
        method=method,
        calibrated_model=calibrator,
        ece_before=ece_before,
        ece_after=ece_after,
        y_proba_calibrated=proba_after,
    )
