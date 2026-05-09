from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from driftguard import DriftAnalyzer


def test_html_and_pdf_exports(reference_df, current_df, tmp_path) -> None:
    html_path = tmp_path / "drift_report.html"
    pdf_path = tmp_path / "drift_report.pdf"
    analyzer = DriftAnalyzer(reference_df, current_df, bins=8, max_samples_per_column=100)
    analyzer.run(report_mode="html", output_path=html_path)
    analyzer.run(report_mode="pdf", output_path=pdf_path)
    html_text = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert pdf_path.exists()
    assert "<html" in html_text.lower()
    assert "plotly" in html_text.lower()
    assert pdf_path.stat().st_size > 0


def test_cli_analyze_csv(reference_df, current_df, tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    output_path = tmp_path / "cli_report.json"
    reference_df.to_csv(reference_path, index=False)
    current_df.to_csv(current_path, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "driftguard",
            "analyze",
            "--reference",
            str(reference_path),
            "--current",
            str(current_path),
            "--report",
            "json",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "overall_score" in payload
    assert "severity" in payload
    assert "overall_score=" in result.stdout

