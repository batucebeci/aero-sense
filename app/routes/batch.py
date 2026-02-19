from __future__ import annotations

import io
from datetime import datetime

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from src.data_loader import load_sensor_csv
from src.feature_engineering import add_derived_features, add_timeseries_features
from src.report_generator import build_prediction_records
from src.sensor_fusion import fuse_sensors

from ..services import emit_alarms_for_predictions
from ..state import get_state

router = APIRouter()

MAX_BATCH_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...), model_name: str = Form("")):
    state = get_state()
    if not state.trained_models or state.prepared is None:
        return Response(
            content="Train models first before calling /api/predict/batch.",
            status_code=400,
            media_type="text/plain",
        )

    contents = bytearray()
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_BATCH_UPLOAD_BYTES:
            return Response(
                content=f"File too large; limit is {MAX_BATCH_UPLOAD_BYTES // 1024 // 1024} MB.",
                status_code=413,
                media_type="text/plain",
            )
    raw = load_sensor_csv(io.BytesIO(bytes(contents)))
    enriched = add_timeseries_features(add_derived_features(raw))
    fused = fuse_sensors(enriched)

    feature_names = state.prepared.feature_names
    for col in feature_names:
        if col not in fused.columns:
            fused[col] = 0.0
    X = state.prepared.scaler.transform(fused[feature_names].values)

    target = model_name or state.best_model_name
    trained = state.trained_models[target]
    y_pred = trained.model.predict(X)
    y_proba = trained.model.predict_proba(X) if hasattr(trained.model, "predict_proba") else None

    records = build_prediction_records(
        np.asarray(y_pred),
        state.prepared.class_names,
        y_proba,
        timestamps=raw["timestamp"] if "timestamp" in raw.columns else None,
        system_ids=raw["system_id"] if "system_id" in raw.columns else None,
    )
    emit_alarms_for_predictions(records)

    buf = io.StringIO()
    records.to_csv(buf, index=False)
    filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
