from __future__ import annotations

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.calibration import calibrate_classifier
from src.evaluation_curves import (
    calibration_table,
    expected_calibration_error,
    per_class_pr,
    per_class_roc,
)

from .. import charts
from ..state import get_state

router = APIRouter()


@router.get("/evaluation", response_class=HTMLResponse)
def evaluation_page(request: Request, model: str | None = Query(None)):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_models": bool(state.trained_models),
        "model_names": list(state.trained_models.keys()),
        "flash": state.pop_flash(),
    }

    if not state.trained_models or state.prepared is None:
        return templates.TemplateResponse(request, "evaluation.html", context)

    target = model if model in state.trained_models else state.best_model_name
    trained = state.trained_models[target]
    y_proba = trained.y_proba
    prepared = state.prepared

    if y_proba is None:
        context.update({"target_model": target, "no_proba": True})
        return templates.TemplateResponse(request, "evaluation.html", context)

    roc = per_class_roc(prepared.y_test, y_proba, prepared.class_names)
    pr = per_class_pr(prepared.y_test, y_proba, prepared.class_names)
    cal_df = calibration_table(prepared.y_test, y_proba, prepared.class_names)
    ece = expected_calibration_error(prepared.y_test, y_proba)

    context.update(
        {
            "target_model": target,
            "ece": ece,
            "roc_chart": charts.roc_chart(roc),
            "pr_chart": charts.pr_chart(pr),
            "calibration_chart": charts.calibration_chart(cal_df),
        }
    )
    return templates.TemplateResponse(request, "evaluation.html", context)


@router.post("/evaluation/calibrate")
def calibrate(model_name: str = Form(...), method: str = Form("isotonic")):
    state = get_state()
    if model_name not in state.trained_models or state.prepared is None:
        state.set_flash("error", "Model not found or not ready.")
        return RedirectResponse(url="/evaluation", status_code=303)

    trained = state.trained_models[model_name]
    try:
        result = calibrate_classifier(trained.model, state.prepared, method=method)
    except (ValueError, RuntimeError) as e:
        state.set_flash("error", str(e))
        return RedirectResponse(url="/evaluation", status_code=303)

    state.set_flash(
        "success",
        f"Calibrated {model_name} ({method}). ECE {result.ece_before:.4f} -> {result.ece_after:.4f}.",
    )
    return RedirectResponse(url=f"/evaluation?model={model_name}", status_code=303)
