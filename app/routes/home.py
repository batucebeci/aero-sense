from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..state import get_state

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    state = get_state()
    templates = request.app.state.templates

    summary = {
        "has_data": state.raw_df is not None,
        "rows": len(state.raw_df) if state.raw_df is not None else 0,
        "fused_features": (
            int(state.fused_df.shape[1] - sum(c in state.fused_df.columns for c in ("timestamp", "system_id", "fault_type")))
            if state.fused_df is not None
            else 0
        ),
        "n_models": len(state.trained_models),
        "best_model": state.best_model_name,
    }

    return templates.TemplateResponse(
        request, "home.html", {"summary": summary, "flash": state.pop_flash()}
    )
