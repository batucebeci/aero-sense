from __future__ import annotations

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.threshold_tuning import per_class_metrics_at_thresholds, threshold_overview

from .. import charts, services
from ..state import get_state

router = APIRouter()


@router.get("/threshold", response_class=HTMLResponse)
def threshold_page(request: Request, model: str | None = Query(None)):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_models": bool(state.trained_models),
        "model_names": list(state.trained_models.keys()),
        "current_thresholds": state.per_class_thresholds,
        "flash": state.pop_flash(),
    }

    if not state.trained_models or state.prepared is None:
        return templates.TemplateResponse(request, "threshold.html", context)

    target = model if model in state.trained_models else state.best_model_name
    trained = state.trained_models[target]
    prepared = state.prepared

    context["target_model"] = target
    if trained.y_proba is None:
        context["no_proba"] = True
        return templates.TemplateResponse(request, "threshold.html", context)

    selected = state.per_class_thresholds or {}
    rows = threshold_overview(prepared.y_test, trained.y_proba, prepared.class_names, selected)
    overview_data = [
        {
            "class_name": r.class_name,
            "n_positive": r.n_positive,
            "max_f1_threshold": round(r.max_f1_threshold, 4),
            "max_f1": round(r.max_f1, 4),
            "selected_threshold": round(r.selected_threshold, 4),
            "precision_at_selected": round(r.precision_at_selected, 4),
            "recall_at_selected": round(r.recall_at_selected, 4),
        }
        for r in rows
    ]

    before_df = per_class_metrics_at_thresholds(
        prepared.y_test, trained.y_proba, prepared.class_names, None
    )
    after_df = per_class_metrics_at_thresholds(
        prepared.y_test,
        trained.y_proba,
        prepared.class_names,
        state.per_class_thresholds,
    )

    context.update(
        {
            "overview": overview_data,
            "before_html": before_df.to_html(classes="data-table", index=False, border=0),
            "after_html": after_df.to_html(classes="data-table", index=False, border=0),
            "pr_chart": charts.per_class_pr_with_thresholds_chart(
                prepared.y_test,
                trained.y_proba,
                prepared.class_names,
                state.per_class_thresholds or {},
            ),
            "macro_f1_before": round(before_df["f1"].mean(), 4) if not before_df.empty else None,
            "macro_f1_after": round(after_df["f1"].mean(), 4) if not after_df.empty else None,
        }
    )
    return templates.TemplateResponse(request, "threshold.html", context)


@router.post("/threshold/optimize")
def threshold_optimize(
    strategy: str = Form("recall"),
    target_recall: float = Form(0.9),
):
    state = get_state()
    try:
        outcome = services.tune_thresholds(state, strategy=strategy, target_recall=target_recall)
    except (RuntimeError, ValueError) as e:
        state.set_flash("error", str(e))
        return RedirectResponse(url="/threshold", status_code=303)
    msg = (
        f"Thresholds set via {outcome['strategy']}"
        + (f" (target recall {outcome['target_recall']:.2f})" if outcome['strategy'] == 'recall' else "")
        + "."
    )
    state.set_flash("success", msg)
    return RedirectResponse(url="/threshold", status_code=303)


@router.post("/threshold/reset")
def threshold_reset():
    state = get_state()
    state.set_thresholds(None)
    state.set_flash("success", "Thresholds reset to default (argmax).")
    return RedirectResponse(url="/threshold", status_code=303)
