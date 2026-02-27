from __future__ import annotations

from datetime import datetime
from io import StringIO

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from src.pdf_report import build_pdf_report
from src.report_generator import render_html_report, summarize_predictions
from src.utils import REPORTS_DIR, ensure_directories

from ..state import get_state

router = APIRouter()


@router.get("/report", response_class=HTMLResponse)
def report_page(request: Request):
    state = get_state()
    templates = request.app.state.templates

    context = {
        "has_predictions": state.last_prediction_df is not None,
        "flash": state.pop_flash(),
    }

    if state.last_prediction_df is not None:
        records = state.last_prediction_df
        summary = summarize_predictions(records)
        context.update(
            {
                "summary": summary,
                "model_name": state.best_model_name,
                "records_html": records.head(100)
                .assign(
                    confidence=lambda d: d["confidence"].map(
                        lambda v: "—" if v is None or v != v else f"{v:.1%}"
                    )
                )
                .to_html(classes="data-table", index=False, border=0, na_rep="—"),
            }
        )

    return templates.TemplateResponse(request, "report.html", context)


@router.get("/report/csv")
def report_csv():
    state = get_state()
    if state.last_prediction_df is None:
        return Response(content="No predictions available.", media_type="text/plain", status_code=400)
    buf = StringIO()
    state.last_prediction_df.to_csv(buf, index=False)
    filename = f"fault_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/report/html")
def report_html():
    state = get_state()
    if state.last_prediction_df is None:
        return Response(content="No predictions available.", media_type="text/plain", status_code=400)
    ensure_directories()
    records = state.last_prediction_df
    summary = summarize_predictions(records)
    filename = f"fault_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    path = REPORTS_DIR / filename
    render_html_report(records, state.best_model_name or "model", summary, path)
    content = path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/report/pdf")
def report_pdf():
    state = get_state()
    if state.last_prediction_df is None:
        return Response(content="No predictions available.", media_type="text/plain", status_code=400)
    ensure_directories()
    records = state.last_prediction_df
    summary = summarize_predictions(records)
    filename = f"fault_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = REPORTS_DIR / filename
    build_pdf_report(records, summary, state.best_model_name or "model", path)
    content = path.read_bytes()
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
