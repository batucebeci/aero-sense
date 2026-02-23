from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from src.explainability import (
    compute_shap_values,
    global_feature_importance,
    top_features_for_prediction,
    waterfall_data,
)

from .. import charts
from ..state import get_state

router = APIRouter()


@router.get("/explain", response_class=HTMLResponse)
def explain_page(
    request: Request,
    model: str | None = Query(None),
    sample_index: int = Query(0),
):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_models": bool(state.trained_models),
        "flash": state.pop_flash(),
        "model_names": list(state.trained_models.keys()),
    }

    if not state.trained_models or state.prepared is None:
        return templates.TemplateResponse(request, "explain.html", context)

    target = model if model in state.trained_models else state.best_model_name
    trained = state.trained_models[target]
    prepared = state.prepared

    shap_result = compute_shap_values(
        trained.model,
        prepared.X_test,
        prepared.feature_names,
        prepared.class_names,
        max_samples=60,
    )
    importance_df = global_feature_importance(shap_result)

    n_test = prepared.X_test.shape[0]
    sample_index = max(0, min(sample_index, n_test - 1))
    sample = prepared.X_test[sample_index]
    predicted_class_idx = int(trained.y_pred[sample_index])
    predicted_label = prepared.class_names[predicted_class_idx]
    actual_label = prepared.class_names[int(prepared.y_test[sample_index])]

    top_features = top_features_for_prediction(
        trained.model,
        sample,
        prepared.feature_names,
        prepared.class_names,
        predicted_class_idx,
        top_k=8,
    )
    waterfall_df = waterfall_data(
        trained.model,
        sample,
        prepared.feature_names,
        predicted_class_idx,
        top_k=8,
    )

    context.update(
        {
            "target_model": target,
            "importance_html": importance_df.head(15).to_html(
                classes="data-table", index=False, border=0
            ),
            "importance_chart": charts.shap_importance_chart(importance_df),
            "sample_index": sample_index,
            "max_sample_index": n_test - 1,
            "predicted_label": predicted_label,
            "actual_label": actual_label,
            "is_correct": predicted_label == actual_label,
            "top_features_html": top_features.assign(
                value=lambda d: d["value"].round(3),
                shap_contribution=lambda d: d["shap_contribution"].round(4),
                abs_contribution=lambda d: d["abs_contribution"].round(4),
            ).to_html(classes="data-table", index=False, border=0),
            "per_prediction_chart": charts.per_prediction_chart(top_features, predicted_label),
            "waterfall_chart": charts.waterfall_chart(waterfall_df, predicted_label),
        }
    )
    return templates.TemplateResponse(request, "explain.html", context)
