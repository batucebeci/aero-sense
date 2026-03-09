from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from src.synthetic_sensor_generator import RAW_SENSOR_COLUMNS

from .. import charts
from ..state import get_state

router = APIRouter()


@router.get("/trace", response_class=HTMLResponse)
def trace_page(
    request: Request,
    system_id: str | None = Query(None),
    sensor: str = Query("motor_temperature"),
):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_data": state.raw_df is not None,
        "flash": state.pop_flash(),
    }

    if state.raw_df is not None and "system_id" in state.raw_df.columns:
        df = state.raw_df
        systems = sorted(df["system_id"].unique().tolist())
        sensors = [c for c in RAW_SENSOR_COLUMNS if c in df.columns]
        if system_id not in systems:
            system_id = systems[0] if systems else None
        if sensor not in sensors:
            sensor = sensors[0] if sensors else None
        context.update(
            {
                "systems": systems,
                "sensors": sensors,
                "selected_system": system_id,
                "selected_sensor": sensor,
                "trace_chart": charts.timeseries_trace_chart(df, system_id, sensor)
                if system_id and sensor
                else "",
            }
        )

    return templates.TemplateResponse(request, "trace.html", context)
