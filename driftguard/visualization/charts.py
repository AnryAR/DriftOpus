from __future__ import annotations

from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..distances.core import entropy
from ..models import ColumnReport, DatasetReport


def _clean_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()


def build_numeric_overview_figure(reference: pd.Series, current: pd.Series, column: str) -> go.Figure:
    ref = _clean_numeric(reference)
    cur = _clean_numeric(current)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Histogram", "Boxplot", "ECDF", "QQ Plot"))
    fig.add_trace(go.Histogram(x=ref, name="reference", opacity=0.6, histnorm="probability density"), row=1, col=1)
    fig.add_trace(go.Histogram(x=cur, name="current", opacity=0.6, histnorm="probability density"), row=1, col=1)
    fig.add_trace(go.Box(y=ref, name="reference", boxmean=True), row=1, col=2)
    fig.add_trace(go.Box(y=cur, name="current", boxmean=True), row=1, col=2)
    if ref.size:
        ref_ecdf_x = np.sort(ref)
        ref_ecdf_y = np.arange(1, ref.size + 1) / ref.size
        fig.add_trace(go.Scatter(x=ref_ecdf_x, y=ref_ecdf_y, mode="lines", name="reference"), row=2, col=1)
    if cur.size:
        cur_ecdf_x = np.sort(cur)
        cur_ecdf_y = np.arange(1, cur.size + 1) / cur.size
        fig.add_trace(go.Scatter(x=cur_ecdf_x, y=cur_ecdf_y, mode="lines", name="current"), row=2, col=1)
    if ref.size and cur.size:
        min_size = min(ref.size, cur.size)
        quantiles = np.linspace(0.01, 0.99, num=min(100, min_size))
        ref_q = np.quantile(ref, quantiles)
        cur_q = np.quantile(cur, quantiles)
        fig.add_trace(go.Scatter(x=ref_q, y=cur_q, mode="markers", name="qq"), row=2, col=2)
    fig.update_layout(title=f"Numeric drift overview: {column}", height=800, legend_orientation="h")
    return fig


def build_categorical_overview_figure(reference: pd.Series, current: pd.Series, column: str) -> go.Figure:
    ref = reference.astype(str).fillna("__MISSING__")
    cur = current.astype(str).fillna("__MISSING__")
    ref_counts = ref.value_counts(normalize=True)
    cur_counts = cur.value_counts(normalize=True)
    categories = list(pd.Index(ref_counts.index).union(cur_counts.index))
    ref_aligned = ref_counts.reindex(categories, fill_value=0.0)
    cur_aligned = cur_counts.reindex(categories, fill_value=0.0)
    top_categories = categories[: min(len(categories), 20)]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Frequency", "Stacked", "Entropy", "Rare Categories"))
    fig.add_trace(go.Bar(x=top_categories, y=ref_aligned.loc[top_categories], name="reference"), row=1, col=1)
    fig.add_trace(go.Bar(x=top_categories, y=cur_aligned.loc[top_categories], name="current"), row=1, col=1)
    fig.add_trace(go.Bar(x=top_categories, y=ref_aligned.loc[top_categories], name="reference", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=top_categories, y=cur_aligned.loc[top_categories], name="current", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=["reference", "current"], y=[float(entropy(ref_aligned)), float(entropy(cur_aligned))]), row=2, col=1)
    rare = [category for category, freq in cur_counts.items() if freq < 0.05]
    if rare:
        fig.add_trace(go.Bar(x=rare, y=[cur_counts[category] for category in rare], name="rare"), row=2, col=2)
    fig.update_layout(title=f"Categorical drift overview: {column}", height=800, barmode="group", legend_orientation="h")
    return fig


def build_drift_heatmap_figure(report: DatasetReport) -> go.Figure:
    columns = list(report.columns.keys())
    scores = [report.columns[column].score for column in columns]
    fig = go.Figure(data=go.Heatmap(z=[scores], x=columns, y=["drift score"], colorscale="RdYlGn_r"))
    fig.update_layout(title="Column drift heatmap", height=300, xaxis_tickangle=-45)
    return fig


def draw_numeric_matplotlib_figure(
    reference: pd.Series,
    current: pd.Series,
    column: str,
    column_report: ColumnReport,
) -> plt.Figure:
    ref = _clean_numeric(reference)
    cur = _clean_numeric(current)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.ravel()
    axes[0].hist(ref, bins=20, alpha=0.6, density=True, label="reference")
    axes[0].hist(cur, bins=20, alpha=0.6, density=True, label="current")
    axes[0].set_title(f"{column} histograms")
    axes[0].legend()
    axes[1].boxplot([ref, cur], labels=["reference", "current"])
    axes[1].set_title("Boxplot")
    if ref.size:
        ref_ecdf_x = np.sort(ref)
        ref_ecdf_y = np.arange(1, ref.size + 1) / ref.size
        axes[2].plot(ref_ecdf_x, ref_ecdf_y, label="reference")
    if cur.size:
        cur_ecdf_x = np.sort(cur)
        cur_ecdf_y = np.arange(1, cur.size + 1) / cur.size
        axes[2].plot(cur_ecdf_x, cur_ecdf_y, label="current")
    axes[2].set_title("ECDF")
    axes[2].legend()
    if ref.size and cur.size:
        quantiles = np.linspace(0.01, 0.99, num=min(100, min(ref.size, cur.size)))
        axes[3].scatter(np.quantile(ref, quantiles), np.quantile(cur, quantiles), s=12)
    axes[3].set_title("QQ plot")
    fig.suptitle(f"{column} | {column_report.severity.value} | score={column_report.score:.3f}")
    fig.tight_layout()
    return fig


def draw_categorical_matplotlib_figure(
    reference: pd.Series,
    current: pd.Series,
    column: str,
    column_report: ColumnReport,
) -> plt.Figure:
    ref = reference.astype(str).fillna("__MISSING__")
    cur = current.astype(str).fillna("__MISSING__")
    ref_counts = ref.value_counts(normalize=True)
    cur_counts = cur.value_counts(normalize=True)
    categories = list(pd.Index(ref_counts.index).union(cur_counts.index))[:20]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.ravel()
    axes[0].bar(np.arange(len(categories)) - 0.2, [ref_counts.get(category, 0.0) for category in categories], width=0.4, label="reference")
    axes[0].bar(np.arange(len(categories)) + 0.2, [cur_counts.get(category, 0.0) for category in categories], width=0.4, label="current")
    axes[0].set_xticks(np.arange(len(categories)))
    axes[0].set_xticklabels(categories, rotation=45, ha="right")
    axes[0].set_title("Category frequencies")
    axes[0].legend()
    axes[1].bar(["reference", "current"], [float(entropy(ref_counts)), float(entropy(cur_counts))])
    axes[1].set_title("Entropy")
    axes[2].bar(["reference", "current"], [len(ref_counts), len(cur_counts)])
    axes[2].set_title("Cardinality")
    rare = [category for category, freq in cur_counts.items() if freq < 0.05]
    if rare:
        axes[3].bar(rare, [cur_counts[category] for category in rare])
        axes[3].set_xticklabels(rare, rotation=45, ha="right")
    axes[3].set_title("Rare categories")
    fig.suptitle(f"{column} | {column_report.severity.value} | score={column_report.score:.3f}")
    fig.tight_layout()
    return fig
