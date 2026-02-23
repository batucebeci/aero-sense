from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from src.feature_engineering import DERIVED_FEATURE_COLUMNS

from .. import charts
from ..state import get_state

router = APIRouter()


@router.get("/features", response_class=HTMLResponse)
def features_page(
    request: Request,
    feature: str = Query("battery_drop_rate"),
):
    state = get_state()
    templates = request.app.state.templates

    if state.fused_df is None:
        return templates.TemplateResponse(
            request, "features.html", {"has_data": False, "flash": state.pop_flash()}
        )

    df = state.fused_df
    available = [c for c in DERIVED_FEATURE_COLUMNS if c in df.columns]
    if feature not in available:
        feature = available[0] if available else None

    stats_df = df[available].describe().round(3).T.reset_index().rename(columns={"index": "feature"})

    context = {
        "has_data": True,
        "flash": state.pop_flash(),
        "available": available,
        "selected": feature,
        "stats_html": stats_df.to_html(classes="data-table", index=False, border=0),
        "feature_chart": charts.feature_distribution_chart(df, feature) if feature else "",
    }
    return templates.TemplateResponse(request, "features.html", context)
