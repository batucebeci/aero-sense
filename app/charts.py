from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_TEMPLATE = "plotly_white"


def _to_html(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False, default_height="380px")


def class_balance_chart(df: pd.DataFrame) -> str:
    counts = df["fault_type"].value_counts().reset_index()
    counts.columns = ["fault_type", "count"]
    fig = px.bar(
        counts,
        x="fault_type",
        y="count",
        color="fault_type",
        template=DEFAULT_TEMPLATE,
        title="Class balance",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(t=40, b=20))
    return _to_html(fig)


def sensor_distribution_chart(df: pd.DataFrame, sensors: list[str]) -> str:
    melted = df[sensors].melt(var_name="sensor", value_name="value")
    fig = px.box(
        melted,
        x="sensor",
        y="value",
        template=DEFAULT_TEMPLATE,
        title="Sensor value distributions",
    )
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def feature_distribution_chart(df: pd.DataFrame, feature: str) -> str:
    fig = px.box(
        df,
        x="fault_type",
        y=feature,
        color="fault_type",
        template=DEFAULT_TEMPLATE,
        title=f"{feature} per fault class",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(t=40, b=20))
    return _to_html(fig)


def model_comparison_chart(table: pd.DataFrame) -> str:
    fig = go.Figure()
    for metric in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]:
        fig.add_trace(go.Bar(name=metric, x=table["Model"], y=table[metric]))
    fig.update_layout(
        barmode="group",
        template=DEFAULT_TEMPLATE,
        title="Model performance comparison",
        yaxis_range=[0, 1.05],
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def confusion_matrix_chart(matrix: np.ndarray, class_names: list[str], model_name: str) -> str:
    fig = px.imshow(
        matrix,
        x=class_names,
        y=class_names,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True,
        color_continuous_scale="Blues",
        template=DEFAULT_TEMPLATE,
        title=f"Confusion matrix — {model_name}",
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def per_class_f1_chart(per_class: dict[str, float], model_name: str) -> str:
    items = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)
    fig = px.bar(
        x=[name for name, _ in items],
        y=[score for _, score in items],
        labels={"x": "Fault class", "y": "F1 score"},
        template=DEFAULT_TEMPLATE,
        title=f"Per-class F1 — {model_name}",
    )
    fig.update_layout(margin=dict(t=40, b=20), yaxis_range=[0, 1.05], xaxis_tickangle=-30)
    return _to_html(fig)


def shap_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> str:
    top = importance_df.head(top_n).iloc[::-1]
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        template=DEFAULT_TEMPLATE,
        title="Global SHAP feature importance",
    )
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def per_prediction_chart(top_features_df: pd.DataFrame, predicted_label: str) -> str:
    df = top_features_df.iloc[::-1].copy()
    df["direction"] = np.where(df["shap_contribution"] >= 0, "Toward", "Against")
    fig = px.bar(
        df,
        x="shap_contribution",
        y="feature",
        color="direction",
        color_discrete_map={"Toward": "#1f77b4", "Against": "#d62728"},
        orientation="h",
        template=DEFAULT_TEMPLATE,
        title=f"SHAP contributions — prediction: {predicted_label}",
    )
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def prediction_distribution_chart(records: pd.DataFrame) -> str:
    counts = records["predicted_fault"].value_counts().reset_index()
    counts.columns = ["predicted_fault", "count"]
    fig = px.bar(
        counts,
        x="predicted_fault",
        y="count",
        color="predicted_fault",
        template=DEFAULT_TEMPLATE,
        title="Predicted fault distribution",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(t=40, b=20))
    return _to_html(fig)


def anomaly_score_chart(scores: np.ndarray, threshold: float) -> str:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores, nbinsx=40, name="Score distribution"))
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        annotation_position="top right",
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title="Isolation Forest score distribution",
        xaxis_title="Anomaly score (higher = more normal)",
        yaxis_title="Count",
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def anomaly_breakdown_chart(breakdown: pd.DataFrame) -> str:
    fig = px.bar(
        breakdown,
        x="fault_type",
        y="anomaly_rate",
        color="fault_type",
        template=DEFAULT_TEMPLATE,
        title="Anomaly rate per fault class",
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-30,
        yaxis_tickformat=".0%",
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def rul_scatter_chart(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(size=5, opacity=0.55, color="#2563eb"),
            name="Predictions",
        )
    )
    max_v = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_v],
            y=[0, max_v],
            mode="lines",
            line=dict(color="#ef4444", dash="dash"),
            name="Ideal",
        )
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title="RUL — predicted vs actual",
        xaxis_title="Actual RUL",
        yaxis_title="Predicted RUL",
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def rul_residual_chart(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    residuals = y_pred - y_true
    fig = px.histogram(
        x=residuals,
        nbins=30,
        template=DEFAULT_TEMPLATE,
        title="RUL residual distribution (predicted − actual)",
    )
    fig.update_layout(margin=dict(t=40, b=20), xaxis_title="Residual", yaxis_title="Count")
    return _to_html(fig)


def drift_chart(report: pd.DataFrame, top_n: int = 15) -> str:
    df = report.head(top_n).iloc[::-1]
    color_map = {"stable": "#10b981", "moderate": "#f59e0b", "severe": "#ef4444", "unknown": "#94a3b8"}
    fig = px.bar(
        df,
        x="psi",
        y="feature",
        color="psi_level",
        color_discrete_map=color_map,
        orientation="h",
        template=DEFAULT_TEMPLATE,
        title="Top drifted features (PSI)",
    )
    fig.add_vline(x=0.1, line_dash="dot", line_color="#f59e0b")
    fig.add_vline(x=0.25, line_dash="dot", line_color="#ef4444")
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def waterfall_chart(waterfall_df: pd.DataFrame, predicted_label: str) -> str:
    rows = waterfall_df.copy()
    base_value = float(rows["base_value"].iloc[0]) if not rows.empty else 0.0
    measure = ["absolute"] + ["relative"] * len(rows)
    x_labels = ["Base value"] + rows["feature"].tolist()
    y_values = [base_value] + rows["shap"].tolist()

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=x_labels,
            y=y_values,
            connector={"line": {"color": "rgb(180,180,180)"}},
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#2563eb"}},
            totals={"marker": {"color": "#0f172a"}},
        )
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title=f"SHAP waterfall — {predicted_label}",
        xaxis_tickangle=-30,
        margin=dict(t=50, b=20),
    )
    return _to_html(fig)


def system_comparison_chart(df: pd.DataFrame, feature: str) -> str:
    if "system_id" not in df.columns or "timestamp" not in df.columns:
        return ""
    plot_df = df.sort_values(["system_id", "timestamp"]).copy()
    fig = px.line(
        plot_df,
        x="timestamp",
        y=feature,
        color="system_id",
        template=DEFAULT_TEMPLATE,
        title=f"{feature} over time by system",
    )
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def system_fault_heatmap(df: pd.DataFrame) -> str:
    if "system_id" not in df.columns or "fault_type" not in df.columns:
        return ""
    pivot = (
        df.pivot_table(
            index="system_id",
            columns="fault_type",
            values=df.columns[0],
            aggfunc="count",
            fill_value=0,
        )
    )
    fig = px.imshow(
        pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        labels=dict(x="Fault", y="System", color="Samples"),
        color_continuous_scale="Blues",
        text_auto=True,
        template=DEFAULT_TEMPLATE,
        title="Fault counts by system",
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def per_class_pr_with_thresholds_chart(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    thresholds: dict[str, float],
) -> str:
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import label_binarize

    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)

    fig = go.Figure()
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, thr = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=name, hovertemplate=name + "<br>R=%{x:.3f}, P=%{y:.3f}")
        )
        sel = thresholds.get(name)
        if sel is not None and len(thr) > 0:
            best_idx = int(np.argmin(np.abs(thr - sel)))
            fig.add_trace(
                go.Scatter(
                    x=[recall[best_idx]],
                    y=[precision[best_idx]],
                    mode="markers",
                    marker=dict(size=10, symbol="x", line=dict(width=2)),
                    name=f"{name} @ {sel:.2f}",
                    showlegend=False,
                )
            )

    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title="Per-class precision-recall with selected thresholds",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def live_sensor_chart(buffer: list[dict], sensor: str = "motor_temperature") -> str:
    if not buffer:
        return ""
    df = pd.DataFrame(buffer)
    if sensor not in df.columns:
        return ""
    fig = px.line(
        df,
        x="t",
        y=sensor,
        template=DEFAULT_TEMPLATE,
        title=f"Live sensor — {sensor}",
    )
    fig.update_layout(margin=dict(t=40, b=20))
    return _to_html(fig)


def roc_chart(curves: dict) -> str:
    fig = go.Figure()
    for name, data in curves.items():
        fig.add_trace(
            go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f"{name} (AUC={data['auc']:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8"), name="chance")
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title="One-vs-rest ROC curves",
        xaxis_title="FPR",
        yaxis_title="TPR",
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def pr_chart(curves: dict) -> str:
    fig = go.Figure()
    for name, data in curves.items():
        fig.add_trace(
            go.Scatter(
                x=data["recall"],
                y=data["precision"],
                mode="lines",
                name=f"{name} (AP={data['auc']:.3f})",
            )
        )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title="One-vs-rest Precision-Recall curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def calibration_chart(calibration_df: pd.DataFrame) -> str:
    if calibration_df.empty:
        return ""
    fig = px.line(
        calibration_df,
        x="mean_predicted",
        y="fraction_positive",
        color="class",
        markers=True,
        template=DEFAULT_TEMPLATE,
        title="Reliability diagram (per class)",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8"),
            name="perfect",
        )
    )
    fig.update_layout(
        margin=dict(t=40, b=20),
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
    )
    return _to_html(fig)


TRACE_MAX_POINTS = 2000
TRACE_MAX_FAULT_BANDS = 80


def timeseries_trace_chart(
    df: pd.DataFrame,
    system_id: str,
    sensor: str,
) -> str:
    sub = df[df["system_id"] == system_id].sort_values("timestamp").copy()
    if sub.empty or sensor not in sub.columns:
        return ""

    downsampled = False
    if len(sub) > TRACE_MAX_POINTS:
        step = max(1, len(sub) // TRACE_MAX_POINTS)
        sub = sub.iloc[::step].copy()
        downsampled = True

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["timestamp"],
            y=sub[sensor],
            mode="lines",
            line=dict(color="#2563eb"),
            name=sensor,
        )
    )

    if "fault_type" in sub.columns:
        labels = sub["fault_type"].values
        times = sub["timestamp"].values
        bands: list[tuple] = []
        if len(labels) > 0:
            start = 0
            for i in range(1, len(labels) + 1):
                if i == len(labels) or labels[i] != labels[start]:
                    current = labels[start]
                    if current != "Normal":
                        bands.append((times[start], times[i - 1], str(current)))
                    start = i

        if len(bands) > TRACE_MAX_FAULT_BANDS:
            keep = sorted(
                bands,
                key=lambda b: (b[1] - b[0]).total_seconds() if hasattr(b[1] - b[0], "total_seconds") else 0,
                reverse=True,
            )[:TRACE_MAX_FAULT_BANDS]
            bands = sorted(keep, key=lambda b: b[0])

        for x0, x1, label in bands:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(239,68,68,0.12)",
                line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation=dict(font=dict(size=10)),
            )

    title = f"{sensor} — system {system_id} (fault windows highlighted)"
    if downsampled:
        title += " · downsampled for display"
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title=title,
        margin=dict(t=40, b=20),
    )
    return _to_html(fig)


def calibration_compare_chart(before_ece: float, after_ece: float) -> str:
    fig = go.Figure(
        data=[
            go.Bar(name="ECE", x=["Before", "After calibration"], y=[before_ece, after_ece])
        ]
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title="Expected Calibration Error (lower is better)",
        margin=dict(t=40, b=20),
        yaxis_range=[0, max(before_ece, after_ece) * 1.2 + 0.01],
    )
    return _to_html(fig)
