from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .utils import RISK_LEVELS


def _build_summary_table(summary: dict, model_name: str) -> Table:
    fault_counts = summary.get("fault_counts", {})
    avg_conf = summary.get("average_confidence")
    rows = [
        ["Model", model_name],
        ["Total samples", str(summary.get("total_samples", 0))],
        ["Distinct faults", str(len(fault_counts))],
        ["Average confidence", f"{avg_conf:.1%}" if avg_conf is not None else "—"],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    table = Table(rows, colWidths=[5 * cm, 10 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
            ]
        )
    )
    return table


def _build_fault_table(summary: dict) -> Table:
    fault_counts = summary.get("fault_counts", {})
    rows = [["Fault", "Count", "Risk"]]
    for fault, count in fault_counts.items():
        rows.append([fault, str(count), RISK_LEVELS.get(fault, "—")])
    table = Table(rows, colWidths=[7 * cm, 3 * cm, 5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5e7eb")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table


def _build_records_table(records: pd.DataFrame, limit: int = 25) -> Table:
    cols = ["timestamp", "system_id", "predicted_fault", "risk_level", "confidence"]
    cols = [c for c in cols if c in records.columns]
    df = records.head(limit).copy()
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].map(
            lambda v: "—" if v is None or v != v else f"{v:.1%}"
        )
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str).str.slice(0, 19)

    data = [cols] + df[cols].astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ]
        )
    )
    return table


def build_pdf_report(
    records: pd.DataFrame,
    summary: dict,
    model_name: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Aero-Sense — Sensor Fusion &amp; Fault Diagnosis Report</b>", styles["Title"]),
        Spacer(1, 0.4 * cm),
        Paragraph("Run summary", styles["Heading2"]),
        _build_summary_table(summary, model_name),
        Spacer(1, 0.6 * cm),
        Paragraph("Fault distribution", styles["Heading2"]),
        _build_fault_table(summary),
        Spacer(1, 0.6 * cm),
        Paragraph("Sample predictions (first 25)", styles["Heading2"]),
        _build_records_table(records),
    ]
    doc.build(story)
    return output_path
