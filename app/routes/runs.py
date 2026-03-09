from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.run_logger import load_runs

from ..state import get_state

router = APIRouter()


@router.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    state = get_state()
    templates = request.app.state.templates
    runs = load_runs()
    context = {
        "n_runs": len(runs),
        "runs_html": runs.head(100).to_html(classes="data-table", index=False, border=0, na_rep="—")
        if not runs.empty
        else None,
        "flash": state.pop_flash(),
    }
    return templates.TemplateResponse(request, "runs.html", context)
