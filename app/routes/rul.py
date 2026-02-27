from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import charts, services
from ..state import get_state

router = APIRouter()


@router.get("/rul", response_class=HTMLResponse)
def rul_page(request: Request):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_data": state.fused_df is not None,
        "has_rul": state.rul_artifacts is not None,
        "flash": state.pop_flash(),
    }

    if state.rul_artifacts is not None:
        rul = state.rul_artifacts
        context.update(
            {
                "mae": rul.mae,
                "rmse": rul.rmse,
                "r2": rul.r2,
                "train_seconds": rul.train_seconds,
                "scatter_chart": charts.rul_scatter_chart(rul.y_test, rul.y_pred),
                "residual_chart": charts.rul_residual_chart(rul.y_test, rul.y_pred),
            }
        )

    return templates.TemplateResponse(request, "rul.html", context)


@router.post("/rul/train")
def rul_train(model_kind: str = Form("random_forest")):
    state = get_state()
    try:
        outcome = services.train_rul(state, model_kind=model_kind)
        state.set_flash(
            "success",
            f"RUL model trained — MAE {outcome['mae']:.2f}, R² {outcome['r2']:.3f}.",
        )
    except RuntimeError as e:
        state.set_flash("error", str(e))
    return RedirectResponse(url="/rul", status_code=303)
