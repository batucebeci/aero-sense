from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.alarms import get_alarm_log
from src.feature_engineering import add_derived_features, add_timeseries_features
from src.sensor_fusion import fuse_sensors
from src.synthetic_sensor_generator import FAULT_CLASSES, generate_sensor_data
from src.utils import risk_level_for

from ..state import get_state

router = APIRouter()


@router.get("/live", response_class=HTMLResponse)
async def live_page(request: Request):
    state = get_state()
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "live.html",
        {
            "has_models": bool(state.trained_models),
            "best_model": state.best_model_name,
            "flash": state.pop_flash(),
        },
    )


def _build_live_sample(rng: random.Random) -> tuple[dict, str]:
    fault = rng.choice(FAULT_CLASSES)
    np_seed = rng.randint(0, 10_000_000)
    df = generate_sensor_data(
        samples_per_class=12, fault_classes=[fault], seed=np_seed
    )
    derived = add_derived_features(df)
    fused = fuse_sensors(add_timeseries_features(derived))
    last_row = fused.iloc[[-1]].copy()
    last_raw = df.iloc[[-1]].copy()
    return {
        "fused_row": last_row,
        "raw_row": last_raw,
        "actual_fault": fault,
    }, fault


def _predict_with_state(fused_row: pd.DataFrame, raw_row: pd.DataFrame) -> dict:
    state = get_state()
    if not state.trained_models or state.prepared is None:
        return {"predicted_fault": None, "confidence": None, "risk_level": None}

    feature_names = state.prepared.feature_names
    missing = [c for c in feature_names if c not in fused_row.columns]
    for c in missing:
        fused_row[c] = 0.0
    X = state.prepared.scaler.transform(fused_row[feature_names].values)

    trained = state.trained_models[state.best_model_name]
    pred_idx = int(trained.model.predict(X)[0])
    label = state.prepared.class_names[pred_idx]

    confidence = None
    if hasattr(trained.model, "predict_proba"):
        confidence = float(trained.model.predict_proba(X)[0, pred_idx])

    risk = risk_level_for(label)
    alarm_log = get_alarm_log()
    alarm_log.add_from_prediction(
        fault=label,
        risk=risk,
        confidence=confidence,
        recommendation="",
        system_id=str(raw_row["system_id"].iloc[0]) if "system_id" in raw_row.columns else None,
        source="live",
    )
    return {"predicted_fault": label, "confidence": confidence, "risk_level": risk}


def _payload(
    fused_row: pd.DataFrame, raw_row: pd.DataFrame, actual_fault: str, prediction: dict
) -> dict:
    raw = raw_row.iloc[0]
    return {
        "t": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "actual_fault": actual_fault,
        "predicted_fault": prediction["predicted_fault"],
        "confidence": prediction["confidence"],
        "risk_level": prediction["risk_level"],
        "sensors": {
            "motor_temperature": float(raw["motor_temperature"]),
            "battery_level": float(raw["battery_level"]),
            "voltage": float(raw["voltage"]),
            "current": float(raw["current"]),
            "vibration_x": float(raw["vibration_x"]),
            "vibration_y": float(raw["vibration_y"]),
            "vibration_z": float(raw["vibration_z"]),
            "gps_accuracy": float(raw["gps_accuracy"]),
            "signal_strength": float(raw["signal_strength"]),
            "speed": float(raw["speed"]),
        },
    }


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    rng = random.Random(np.random.SeedSequence().entropy)
    state = get_state()
    try:
        while True:
            ctx, fault = await asyncio.to_thread(_build_live_sample, rng)
            prediction = await asyncio.to_thread(
                _predict_with_state, ctx["fused_row"], ctx["raw_row"]
            )
            payload = _payload(ctx["fused_row"], ctx["raw_row"], fault, prediction)

            state.append_live({"t": payload["t"], **payload["sensors"]})

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.7)
    except WebSocketDisconnect:
        return
