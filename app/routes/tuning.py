from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import services
from ..state import get_state

router = APIRouter()


@router.get("/tuning", response_class=HTMLResponse)
def tuning_page(request: Request):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_models": bool(state.trained_models),
        "model_names": list(state.trained_models.keys()),
        "tuning_results": state.tuning_results,
        "flash": state.pop_flash(),
    }
    return templates.TemplateResponse(request, "tuning.html", context)


@router.post("/tuning/run")
def tuning_run(
    model_name: str = Form(...),
    n_trials: int = Form(20),
    timeout_seconds: float = Form(120),
    apply: bool = Form(False),
):
    state = get_state()
    try:
        result = services.run_tuning(
            state, model_name=model_name, n_trials=n_trials, timeout_seconds=timeout_seconds
        )
        msg = (
            f"Tuned {model_name}: best F1 {result['best_score']:.4f} after "
            f"{result['n_trials']} trials in {result['elapsed_seconds']:.1f}s."
        )
        if apply:
            services.retrain_with_tuned_params(state, model_name, result["best_params"])
            msg += " Tuned params applied to active model."
        state.set_flash("success", msg)
    except (RuntimeError, ValueError) as e:
        state.set_flash("error", str(e))
    return RedirectResponse(url="/tuning", status_code=303)
