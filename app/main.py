from __future__ import annotations

import warnings
from contextlib import asynccontextmanager
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r'Field "model_.*" in .* has conflict with protected namespace "model_".*',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names.*",
    category=UserWarning,
)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.alarms import get_alarm_log
from src.utils import ensure_directories
from src.webhook import install_webhook_if_configured

from . import auth, observability
from .routes import (
    alarms,
    anomaly,
    api,
    batch,
    compare,
    counterfactual,
    data,
    drift,
    evaluation,
    explain,
    features,
    home,
    live,
    predict,
    registry,
    report,
    rul,
    runs,
    samples,
    threshold,
    trace,
    train,
    tuning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    app.state.templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
    app.state.templates.env.globals["nav_links"] = [
        ("/", "Home"),
        ("/data", "Dataset"),
        ("/features", "Features"),
        ("/compare", "Compare"),
        ("/trace", "Trace"),
        ("/train", "Train"),
        ("/tuning", "Tune"),
        ("/evaluation", "Evaluate"),
        ("/threshold", "Threshold"),
        ("/predict", "Predict"),
        ("/anomaly", "Anomaly"),
        ("/rul", "RUL"),
        ("/explain", "Explain"),
        ("/counterfactual", "Counterfactual"),
        ("/drift", "Drift"),
        ("/live", "Live"),
        ("/alarms", "Alarms"),
        ("/registry", "Registry"),
        ("/runs", "Runs"),
        ("/report", "Report"),
    ]

    webhook_installed = install_webhook_if_configured(get_alarm_log())
    app.state.webhook_installed = webhook_installed
    yield


app = FastAPI(
    title="Aero-Sense — Sensor Fusion & Fault Diagnosis",
    description=(
        "Multi-sensor fusion fault diagnosis platform with explainable ML, "
        "anomaly detection, RUL regression, drift monitoring, alarm log, "
        "live streaming, model registry, ROC/PR/calibration evaluation, "
        "batch prediction, and optional API key auth + rate limiting."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

observability.install(app)
auth.install(app)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

for module in (
    home,
    data,
    samples,
    features,
    compare,
    trace,
    train,
    tuning,
    evaluation,
    threshold,
    predict,
    batch,
    anomaly,
    rul,
    explain,
    counterfactual,
    drift,
    live,
    alarms,
    registry,
    runs,
    report,
    api,
):
    app.include_router(module.router)
