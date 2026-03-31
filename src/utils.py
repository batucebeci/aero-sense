from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"
RUNS_DIR = OUTPUTS_DIR / "runs"

RISK_LEVELS = {
    "Normal": "Low",
    "Sensor Drift": "Medium",
    "GPS Fault": "Medium-High",
    "Communication Fault": "Medium-High",
    "Navigation Instability": "High",
    "Vibration Fault": "High",
    "Power Instability": "High",
    "Battery Fault": "High",
    "Motor Overheating": "Critical",
}


def ensure_directories() -> None:
    for directory in (
        DATA_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        EXPLANATIONS_DIR,
        RUNS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def risk_level_for(fault: str) -> str:
    return RISK_LEVELS.get(fault, "Unknown")
