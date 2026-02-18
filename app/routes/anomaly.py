from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import charts, services
from ..state import get_state

router = APIRouter()


@router.get("/anomaly", response_class=HTMLResponse)
def anomaly_page(request: Request):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_models": bool(state.trained_models),
        "has_anomaly": state.anomaly_result is not None,
        "flash": state.pop_flash(),
    }

    if state.anomaly_result is not None and state.prepared is not None:
        result = state.anomaly_result
        from src.anomaly_detection import anomaly_breakdown

        breakdown = anomaly_breakdown(
            result.anomaly_flags, state.prepared.y_test, state.prepared.class_names
        )
        context.update(
            {
                "n_anomalies": result.n_anomalies,
                "threshold": result.threshold,
                "contamination": result.contamination,
                "score_chart": charts.anomaly_score_chart(result.scores, result.threshold),
                "breakdown_chart": charts.anomaly_breakdown_chart(breakdown),
                "breakdown_html": breakdown.assign(
                    anomaly_rate=lambda d: (d["anomaly_rate"] * 100).round(1).astype(str) + "%"
                ).to_html(classes="data-table", index=False, border=0),
            }
        )

    return templates.TemplateResponse(request, "anomaly.html", context)


@router.post("/anomaly/fit")
def anomaly_fit(contamination: float = Form(0.08)):
    state = get_state()
    try:
        outcome = services.fit_anomaly_layer(state, contamination=contamination)
        state.set_flash(
            "success",
            f"Isolation Forest fit — {outcome['n_anomalies']} anomalies flagged at threshold "
            f"{outcome['threshold']:.4f}.",
        )
    except RuntimeError as e:
        state.set_flash("error", str(e))
    return RedirectResponse(url="/anomaly", status_code=303)
