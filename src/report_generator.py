from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RISK_LEVELS, risk_level_for

RECOMMENDATIONS = {
    "Normal": "System is within normal range. Continue routine monitoring.",
    "Battery Fault": "Inspect battery cells; review charge cycles and voltage profile.",
    "Motor Overheating": "Stop the motor and check the cooling system and load profile.",
    "GPS Fault": "Check the GPS antenna and sky view; switch to backup positioning if needed.",
    "Communication Fault": "Check link range, antenna alignment and telemetry status.",
    "Sensor Drift": "Recalibrate the affected sensor and cross-check against a redundant sensor.",
    "Vibration Fault": "Inspect mechanical mounts, propeller balance and fastener torque.",
    "Power Instability": "Check the power regulator and wiring; monitor current draw.",
    "Navigation Instability": "Verify the IMU/GPS fusion and review the control loop tuning.",
}


def _confidence_from_probabilities(proba_row: np.ndarray | None) -> float | None:
    if proba_row is None:
        return None
    return float(np.max(proba_row))


def build_prediction_records(
    predictions: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray | None,
    timestamps: pd.Series | None = None,
    system_ids: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for i, pred_idx in enumerate(predictions):
        fault = class_names[int(pred_idx)]
        proba_row = probabilities[i] if probabilities is not None else None
        confidence = _confidence_from_probabilities(proba_row)
        rows.append(
            {
                "timestamp": timestamps.iloc[i] if timestamps is not None else None,
                "system_id": system_ids.iloc[i] if system_ids is not None else None,
                "predicted_fault": fault,
                "confidence": confidence,
                "risk_level": risk_level_for(fault),
                "recommendation": RECOMMENDATIONS.get(fault, ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(records: pd.DataFrame) -> dict:
    summary = {
        "total_samples": len(records),
        "fault_counts": records["predicted_fault"].value_counts().to_dict(),
        "risk_counts": records["risk_level"].value_counts().to_dict(),
    }
    if "confidence" in records.columns and records["confidence"].notna().any():
        summary["average_confidence"] = float(records["confidence"].mean())
    return summary


def render_html_report(
    records: pd.DataFrame,
    model_name: str,
    summary: dict,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    table_html = records.head(50).to_html(index=False, classes="records", border=0, na_rep="-")
    fault_counts_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{v}</td><td>{escape(RISK_LEVELS.get(k, '-'))}</td></tr>"
        for k, v in summary.get("fault_counts", {}).items()
    )

    avg_conf = summary.get("average_confidence")
    avg_conf_text = f"{avg_conf:.2%}" if avg_conf is not None else "—"

    html = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<title>Aero-Sense — Fault Diagnosis Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 32px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #666; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; margin-bottom: 24px; }}
  th, td {{ padding: 6px 12px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f3f3f3; }}
  .summary-grid {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: #f8f9fb; padding: 16px 20px; border-radius: 8px; min-width: 180px; }}
  .card h3 {{ margin: 0; font-size: 14px; color: #555; }}
  .card p {{ margin: 4px 0 0 0; font-size: 22px; font-weight: 600; }}
</style></head>
<body>
<h1>Aero-Sense — Sensor Fusion &amp; Fault Diagnosis Report</h1>
<div class="meta">Model: {escape(model_name)} · Generated: {generated_at}</div>

<div class="summary-grid">
  <div class="card"><h3>Total samples</h3><p>{summary.get("total_samples", 0)}</p></div>
  <div class="card"><h3>Average confidence</h3><p>{avg_conf_text}</p></div>
  <div class="card"><h3>Distinct faults</h3><p>{len(summary.get("fault_counts", {}))}</p></div>
</div>

<h2>Fault distribution</h2>
<table>
  <thead><tr><th>Fault</th><th>Count</th><th>Risk</th></tr></thead>
  <tbody>{fault_counts_rows}</tbody>
</table>

<h2>Prediction details (first 50)</h2>
{table_html}
</body></html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
