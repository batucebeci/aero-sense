from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from src.sample_datasets import get_definitions_with_paths

from .. import charts, services
from ..state import get_state

router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

PREVIEW_SENSORS = [
    "motor_temperature",
    "battery_level",
    "voltage",
    "current",
    "vibration_x",
    "gps_accuracy",
    "signal_strength",
    "speed",
]


@router.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_data": state.raw_df is not None,
        "flash": state.pop_flash(),
        "samples": get_definitions_with_paths(),
    }

    if state.raw_df is not None:
        df = state.raw_df
        context.update(
            {
                "rows": len(df),
                "columns": int(df.shape[1]),
                "missing_values": int(df.isna().sum().sum()),
                "preview_html": df.head(15).to_html(
                    classes="data-table", index=False, border=0, na_rep="—"
                ),
                "class_balance_chart": charts.class_balance_chart(df),
                "sensor_dist_chart": charts.sensor_distribution_chart(
                    df, [c for c in PREVIEW_SENSORS if c in df.columns]
                ),
                "fault_counts": df["fault_type"].value_counts().to_dict()
                if "fault_type" in df.columns
                else {},
            }
        )

    return templates.TemplateResponse(request, "data.html", context)


@router.post("/data/generate")
def generate(
    samples_per_class: int = Form(300),
    seed: int = Form(42),
    noise_scale: float = Form(0.9),
):
    state = get_state()
    samples_per_class = max(50, min(samples_per_class, 1500))
    noise_scale = max(0.0, min(noise_scale, 1.5))
    services.generate_synthetic(
        state, samples_per_class=samples_per_class, seed=seed, noise_scale=noise_scale
    )
    state.set_flash(
        "success",
        f"Generated {samples_per_class * 9} synthetic samples (9 classes, noise={noise_scale:.2f}).",
    )
    return RedirectResponse(url="/data", status_code=303)


@router.post("/data/upload")
async def upload(file: UploadFile = File(...)):
    state = get_state()
    contents = bytearray()
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_BYTES:
            state.set_flash(
                "error", f"File too large; limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
            )
            return RedirectResponse(url="/data", status_code=303)
    try:
        df = services.load_uploaded_csv(state, bytes(contents))
    except ValueError as e:
        state.set_flash("error", str(e))
        return RedirectResponse(url="/data", status_code=303)
    state.set_flash("success", f"Loaded {len(df)} rows from {file.filename}.")
    return RedirectResponse(url="/data", status_code=303)
