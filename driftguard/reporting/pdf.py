from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from ..models import DatasetReport
from ..visualization.charts import draw_categorical_matplotlib_figure, draw_numeric_matplotlib_figure


def _summary_page(report: DatasetReport) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    lines = [
        f"{report.title}",
        "",
        f"Generated at: {report.generated_at}",
        f"Overall score: {report.overall_score:.3f}",
        f"Severity: {report.severity.value}",
        f"Analyzed columns: {len(report.columns)}",
        "",
        "Schema issues:",
    ]
    if report.schema_issues:
        lines.extend([f"- {issue}" for issue in report.schema_issues])
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Excluded columns:")
    if report.excluded_columns:
        lines.extend([f"- {column}: {reason}" for column, reason in report.excluded_columns.items()])
    else:
        lines.append("- None")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    return fig


def generate_pdf_report(
    report: DatasetReport,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    with PdfPages(output_path) as pdf:
        pdf.savefig(_summary_page(report), bbox_inches="tight")
        plt.close("all")
        for column, column_report in report.columns.items():
            if column_report.role == "numeric":
                fig = draw_numeric_matplotlib_figure(reference_df[column], current_df[column], column, column_report)
            else:
                fig = draw_categorical_matplotlib_figure(reference_df[column], current_df[column], column, column_report)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return str(output_path)

