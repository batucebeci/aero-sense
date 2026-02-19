from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from src.counterfactual import search_counterfactual

from ..state import get_state

router = APIRouter()


@router.get("/counterfactual", response_class=HTMLResponse)
def counterfactual_page(
    request: Request,
    sample_index: int = Query(0),
    target: str | None = Query(None),
):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_models": bool(state.trained_models),
        "flash": state.pop_flash(),
    }

    if not state.trained_models or state.prepared is None:
        return templates.TemplateResponse(request, "counterfactual.html", context)

    prepared = state.prepared
    trained = state.trained_models[state.best_model_name]

    n_test = prepared.X_test.shape[0]
    sample_index = max(0, min(sample_index, n_test - 1))
    sample = prepared.X_test[sample_index]
    actual_label = prepared.class_names[int(prepared.y_test[sample_index])]
    pred_label = prepared.class_names[int(trained.y_pred[sample_index])]
    default_target = target or "Normal"

    if default_target not in prepared.class_names:
        default_target = prepared.class_names[0]

    target_idx = prepared.class_names.index(default_target)
    result = search_counterfactual(
        trained.model,
        sample,
        target_class=target_idx,
        step=0.3,
        max_steps=50,
    )

    diff = result.counterfactual - sample
    nonzero = np.abs(diff) > 1e-6
    impacted = [
        {
            "feature": prepared.feature_names[i],
            "original_scaled": float(sample[i]),
            "counterfactual_scaled": float(result.counterfactual[i]),
            "delta_scaled": float(diff[i]),
        }
        for i in range(len(diff))
        if nonzero[i]
    ]
    impacted.sort(key=lambda r: abs(r["delta_scaled"]), reverse=True)

    context.update(
        {
            "class_names": prepared.class_names,
            "actual_label": actual_label,
            "pred_label": pred_label,
            "target_label": default_target,
            "sample_index": sample_index,
            "max_sample_index": n_test - 1,
            "success": result.success,
            "steps": result.steps,
            "counterfactual_pred": prepared.class_names[result.counterfactual_prediction],
            "impacted": impacted[:12],
        }
    )
    return templates.TemplateResponse(request, "counterfactual.html", context)
