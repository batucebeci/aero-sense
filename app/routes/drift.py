from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import charts, services
from ..state import get_state

router = APIRouter()


@router.get("/drift", response_class=HTMLResponse)
def drift_page(request: Request):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_data": state.fused_df is not None,
        "has_reference": state.reference_df is not None,
        "has_report": state.drift_report is not None,
        "flash": state.pop_flash(),
    }

    if state.drift_report is not None:
        report = state.drift_report
        context.update(
            {
                "summary": state.drift_summary,
                "drift_chart": charts.drift_chart(report),
                "drift_table_html": report.head(25)
                .assign(
                    psi=lambda d: d["psi"].round(4),
                    ks_statistic=lambda d: d["ks_statistic"].round(4),
                    ks_pvalue=lambda d: d["ks_pvalue"].map(lambda v: f"{v:.2e}"),
                    reference_mean=lambda d: d["reference_mean"].round(3),
                    current_mean=lambda d: d["current_mean"].round(3),
                    mean_shift=lambda d: d["mean_shift"].round(3),
                )
                .to_html(classes="data-table", index=False, border=0),
            }
        )

    return templates.TemplateResponse(request, "drift.html", context)


@router.post("/drift/compute")
def drift_compute():
    state = get_state()
    try:
        outcome = services.compute_drift(state)
        summary = outcome["summary"]
        state.set_flash(
            "success",
            f"Drift computed — overall: {summary['overall']} "
            f"(severe={summary['n_severe']}, moderate={summary['n_moderate']}).",
        )
    except RuntimeError as e:
        state.set_flash("error", str(e))
    return RedirectResponse(url="/drift", status_code=303)


@router.post("/drift/reset_reference")
def reset_reference():
    state = get_state()
    if state.fused_df is None:
        state.set_flash("error", "No dataset loaded.")
    else:
        state.reference_df = state.fused_df.copy()
        state.drift_report = None
        state.drift_summary = None
        state.set_flash("success", "Reference snapshot updated to current dataset.")
    return RedirectResponse(url="/drift", status_code=303)
