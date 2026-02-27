from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.model_registry import list_models

from ..state import get_state

router = APIRouter()


@router.get("/registry", response_class=HTMLResponse)
def registry_page(request: Request):
    state = get_state()
    templates = request.app.state.templates
    entries = list_models()
    return templates.TemplateResponse(
        request,
        "registry.html",
        {
            "entries": [e.as_dict() for e in entries],
            "n_entries": len(entries),
            "flash": state.pop_flash(),
        },
    )
