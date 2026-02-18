from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.alarms import get_alarm_log

from ..state import get_state

router = APIRouter()


@router.get("/alarms", response_class=HTMLResponse)
def alarms_page(request: Request):
    state = get_state()
    templates = request.app.state.templates
    log = get_alarm_log()
    alarms = log.snapshot()
    by_risk: dict[str, int] = {}
    for a in alarms:
        by_risk[a.risk] = by_risk.get(a.risk, 0) + 1
    context = {
        "alarms": [a.as_dict() for a in alarms],
        "n_total": len(alarms),
        "by_risk": by_risk,
        "flash": state.pop_flash(),
    }
    return templates.TemplateResponse(request, "alarms.html", context)


@router.post("/alarms/clear")
def alarms_clear():
    get_alarm_log().clear()
    get_state().set_flash("success", "Alarm log cleared.")
    return RedirectResponse(url="/alarms", status_code=303)
