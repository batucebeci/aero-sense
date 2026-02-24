from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import charts, services
from ..state import get_state

router = APIRouter()


@router.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_models": bool(state.trained_models),
        "flash": state.pop_flash(),
        "model_names": list(state.trained_models.keys()),
        "best_model": state.best_model_name,
        "has_predictions": state.last_prediction_df is not None,
    }

    if state.last_prediction_df is not None:
        records = state.last_prediction_df
        context.update(
            {
                "preview_html": records.head(50)
                .assign(
                    confidence=lambda d: d["confidence"].map(
                        lambda v: "—" if v is None or v != v else f"{v:.1%}"
                    )
                )
                .to_html(classes="data-table", index=False, border=0, na_rep="—"),
                "distribution_chart": charts.prediction_distribution_chart(records),
                "total": len(records),
                "correct": int(records["correct"].sum()) if "correct" in records.columns else None,
            }
        )

    return templates.TemplateResponse(request, "predict.html", context)


@router.post("/predict/run")
def predict_run(model_name: str = Form("")):
    state = get_state()
    target = model_name or None
    try:
        services.predict_on_current_dataset(state, model_name=target)
        chosen = target or state.best_model_name
        state.set_flash("success", f"Generated predictions using {chosen}.")
    except (RuntimeError, ValueError) as e:
        state.set_flash("error", str(e))
    return RedirectResponse(url="/predict", status_code=303)
