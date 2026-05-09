from __future__ import annotations

import pytest

from driftguard import DriftAnalyzer
from driftguard.models import DriftSeverity


def test_analyzer_builds_report_and_excludes_datetimes(reference_df, current_df) -> None:
    report = DriftAnalyzer(reference_df, current_df, bins=8, max_samples_per_column=100).run()
    assert report.overall_score > 0.0
    assert report.severity in {
        DriftSeverity.NO_DRIFT,
        DriftSeverity.LOW_DRIFT,
        DriftSeverity.MEDIUM_DRIFT,
        DriftSeverity.HIGH_DRIFT,
        DriftSeverity.CRITICAL_DRIFT,
    }
    assert "signup_date" in report.excluded_columns
    assert "datetime" in report.excluded_columns["signup_date"]
    assert "correlation_drift" in report.dataset_metrics
    assert "covariance_drift" in report.dataset_metrics
    assert "pca_latent_drift" in report.dataset_metrics
    assert "data_quality_score" in report.dataset_metrics
    assert "salary" in report.columns
    assert report.columns["salary"].score > 0.0
    assert report.columns["salary"].tests["ks_test"].score > 0.0
    summary_df = report.to_dataframe()
    assert not summary_df.empty
    assert set(summary_df.columns) == {"column", "dtype", "role", "score", "severity", "summary", "recommendation"}


def test_analyzer_raises_on_null_values(reference_df, null_current_df) -> None:
    with pytest.raises(ValueError, match=r"current_df\['salary'\] -> 1 nulls found"):
        DriftAnalyzer(reference_df, null_current_df).run()


def test_analyzer_column_subset(reference_df, current_df) -> None:
    report = DriftAnalyzer(reference_df, current_df, bins=8, max_samples_per_column=100).run(columns=["age", "city"])
    assert set(report.columns) == {"age", "city"}


def test_report_serialization_helpers(reference_df, current_df, tmp_path) -> None:
    report = DriftAnalyzer(reference_df, current_df, bins=8, max_samples_per_column=100).run()
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    html_path = tmp_path / "report.html"
    json_text = report.to_json(json_path)
    csv_text = report.to_csv(csv_path)
    html_text = report.to_html(html_path)
    assert json_path.exists()
    assert csv_path.exists()
    assert html_path.exists()
    assert "\"overall_score\"" in json_text
    assert "column" in csv_text
    assert "<table" in html_text.lower()
