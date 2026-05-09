from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from plotly.offline import get_plotlyjs

from ..models import ColumnReport, DatasetReport, DriftSeverity
from ..utils.serialization import to_native
from ..visualization.charts import build_categorical_overview_figure, build_drift_heatmap_figure, build_numeric_overview_figure


_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 16px; margin-bottom: 18px; box-shadow: 0 8px 30px rgba(0,0,0,.2); }
    h1, h2, h3 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; background: #334155; }
    .sev-NO_DRIFT { background: #14532d; }
    .sev-LOW_DRIFT { background: #166534; }
    .sev-MEDIUM_DRIFT { background: #854d0e; }
    .sev-HIGH_DRIFT { background: #b45309; }
    .sev-CRITICAL_DRIFT { background: #991b1b; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #1f2937; padding: 8px; text-align: left; vertical-align: top; }
    .muted { color: #94a3b8; }
    .chart { margin-top: 16px; }
    .small { font-size: 13px; }
  </style>
  <script>{{ plotly_js|safe }}</script>
</head>
<body>
  <div class="card">
    <h1>{{ title }}</h1>
    <p class="muted">Generated at {{ generated_at }}</p>
    <div class="grid">
      <div class="card"><h3>Overall Score</h3><div class="badge sev-{{ severity }}">{{ "%.3f"|format(overall_score) }}</div></div>
      <div class="card"><h3>Severity</h3><div class="badge sev-{{ severity }}">{{ severity }}</div></div>
      <div class="card"><h3>Analyzed Columns</h3><div class="badge">{{ analyzed_columns }}</div></div>
    </div>
  </div>

  {% if schema_issues %}
  <div class="card">
    <h2>Schema Issues</h2>
    <ul>
      {% for issue in schema_issues %}
      <li>{{ issue }}</li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}

  {% if excluded_columns %}
  <div class="card">
    <h2>Excluded Columns</h2>
    <table>
      <thead><tr><th>Column</th><th>Reason</th></tr></thead>
      <tbody>
      {% for column, reason in excluded_columns.items() %}
        <tr><td>{{ column }}</td><td>{{ reason }}</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  {% if heatmap_html %}
  <div class="card chart">
    <h2>Drift Heatmap</h2>
    {{ heatmap_html | safe }}
  </div>
  {% endif %}

  <div class="card">
    <h2>Column Analysis</h2>
    {% for column in columns %}
      <div class="card">
        <h3>{{ column.column }} <span class="badge sev-{{ column.severity }}">{{ column.severity }}</span></h3>
        <p class="small">{{ column.summary }}</p>
        <p class="small"><strong>Recommendation:</strong> {{ column.recommendation }}</p>
        <table>
          <thead><tr><th>Test</th><th>Statistic</th><th>P-Value</th><th>Score</th><th>Triggered</th><th>Interpretation</th></tr></thead>
          <tbody>
            {% for test in column.tests.values() %}
            <tr>
              <td>{{ test.test_name }}</td>
              <td>{{ test.statistic }}</td>
              <td>{{ test.p_value }}</td>
              <td>{{ "%.3f"|format(test.score) }}</td>
              <td>{{ test.triggered }}</td>
              <td>{{ test.interpretation }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        <div class="chart">
          {{ column_charts[column.column] | safe }}
        </div>
      </div>
    {% endfor %}
  </div>
</body>
</html>
"""
)


def _figure_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def generate_html_report(
    report: DatasetReport,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> str:
    heatmap_html = ""
    column_charts: Dict[str, str] = {}
    if report.columns:
        heatmap = build_drift_heatmap_figure(report)
        heatmap_html = _figure_html(heatmap)
    for column, column_report in report.columns.items():
        if column_report.role == "numeric":
            fig = build_numeric_overview_figure(reference_df[column], current_df[column], column)
        else:
            fig = build_categorical_overview_figure(reference_df[column], current_df[column], column)
        column_charts[column] = _figure_html(fig)

    html = _TEMPLATE.render(
        title=report.title,
        generated_at=report.generated_at,
        overall_score=report.overall_score,
        severity=report.severity.value,
        analyzed_columns=len(report.columns),
        schema_issues=report.schema_issues,
        excluded_columns=report.excluded_columns,
        columns=list(report.columns.values()),
        column_charts=column_charts,
        heatmap_html=heatmap_html,
        plotly_js=get_plotlyjs(),
    )
    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")
    return html

