from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from src.synthetic_sensor_generator import RAW_SENSOR_COLUMNS

from .. import charts
from ..state import get_state

router = APIRouter()


@router.get("/compare", response_class=HTMLResponse)
def compare_page(
    request: Request,
    feature: str = Query("motor_temperature"),
):
    state = get_state()
    templates = request.app.state.templates

    context = {"has_data": state.raw_df is not None, "flash": state.pop_flash()}

    if state.raw_df is not None and "system_id" in state.raw_df.columns:
        df = state.raw_df
        sensors = [c for c in RAW_SENSOR_COLUMNS if c in df.columns]
        if feature not in sensors:
            feature = sensors[0] if sensors else None

        per_system_stats = (
            df.groupby("system_id")[sensors]
            .agg(["mean", "std"])
            .round(2)
        )
        per_system_stats.columns = ["_".join(c) for c in per_system_stats.columns]
        per_system_stats = per_system_stats.reset_index()

        context.update(
            {
                "available_sensors": sensors,
                "selected_feature": feature,
                "system_comparison_chart": charts.system_comparison_chart(df, feature)
                if feature
                else "",
                "fault_heatmap": charts.system_fault_heatmap(df),
                "per_system_html": per_system_stats.head(50).to_html(
                    classes="data-table", index=False, border=0
                ),
                "n_systems": int(df["system_id"].nunique()),
            }
        )

    return templates.TemplateResponse(request, "compare.html", context)
