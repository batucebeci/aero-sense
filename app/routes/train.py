from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.model_evaluation import comparison_table

from .. import charts, services
from ..state import get_state

router = APIRouter()


@router.get("/train", response_class=HTMLResponse)
def train_page(request: Request, model: str | None = Query(None)):
    state = get_state()
    templates = request.app.state.templates

    has_data = state.fused_df is not None
    has_models = bool(state.trained_models)

    context = {
        "has_data": has_data,
        "has_models": has_models,
        "flash": state.pop_flash(),
        "model_names": list(state.trained_models.keys()),
        "best_model": state.best_model_name,
    }

    if has_models:
        table = comparison_table(state.evaluation_results)
        focus_model = model if model in state.trained_models else state.best_model_name
        result = state.evaluation_results[focus_model]
        context.update(
            {
                "comparison_table_html": table.style.format(
                    {
                        "Accuracy": "{:.3f}",
                        "Precision (macro)": "{:.3f}",
                        "Recall (macro)": "{:.3f}",
                        "F1 (macro)": "{:.3f}",
                        "Train time (s)": "{:.2f}",
                        "Predict time (s)": "{:.3f}",
                    }
                ).hide(axis="index").to_html(),
                "comparison_chart": charts.model_comparison_chart(table),
                "focus_model": focus_model,
                "confusion_chart": charts.confusion_matrix_chart(
                    result.confusion, state.prepared.class_names, focus_model
                ),
                "f1_chart": charts.per_class_f1_chart(result.per_class_f1, focus_model),
                "classification_report": result.classification_report,
            }
        )

    return templates.TemplateResponse(request, "train.html", context)


@router.post("/train/run")
def train_run():
    state = get_state()
    try:
        outcome = services.train_models(state)
        state.set_flash(
            "success",
            f"Trained {len(state.trained_models)} models. Best: {outcome['best_model']} "
            f"(F1={outcome['metrics'].f1:.3f}).",
        )
    except RuntimeError as e:
        state.set_flash("error", str(e))
    return RedirectResponse(url="/train", status_code=303)
