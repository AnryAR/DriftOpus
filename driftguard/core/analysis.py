from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..configs.defaults import DriftAnalyzerConfig
from ..models import ColumnReport, DatasetReport, DriftSeverity
from ..statistics import combine_scores, run_categorical_tests, run_numeric_tests, severity_from_score
from ..utils.logging import get_logger
from ..validators import check_nulls, column_role, resolve_analysis_columns, validate_dataframes
from ..version import __version__


def _safe_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return numeric.dropna(axis=1, how="all")


class DriftAnalyzer:
    def __init__(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        config: DriftAnalyzerConfig | None = None,
        **overrides,
    ) -> None:
        self.reference_df = reference_df
        self.current_df = current_df
        if config is not None:
            merged_config = config.model_dump()
            merged_config.update(overrides)
            self.config = DriftAnalyzerConfig(**merged_config)
        else:
            self.config = DriftAnalyzerConfig(**overrides)
        self.logger = get_logger(self.__class__.__name__)

    def run(
        self,
        columns: Sequence[str] | None = None,
        report_mode: str | None = None,
        output_path: str | Path | None = None,
    ) -> DatasetReport:
        validate_dataframes(self.reference_df, self.current_df)
        selected_columns, excluded_columns, schema_issues = resolve_analysis_columns(
            self.reference_df,
            self.current_df,
            columns=columns,
        )

        if selected_columns:
            check_nulls(self.reference_df, selected_columns, "reference_df")
            check_nulls(self.current_df, selected_columns, "current_df")

        column_reports = self._analyze_columns(selected_columns)
        overall_score = self._overall_score(column_reports)
        severity = severity_from_score(overall_score, self.config.severity_thresholds)
        dataset_metrics = self._dataset_metrics(selected_columns)
        quality_score = self._quality_score(selected_columns, excluded_columns, schema_issues)
        dataset_metrics["data_quality_score"] = quality_score

        report = DatasetReport(
            title=self.config.report_title,
            overall_score=overall_score,
            severity=severity,
            columns={report.column: report for report in column_reports},
            excluded_columns=excluded_columns,
            schema_issues=schema_issues,
            dataset_metrics=dataset_metrics,
            config=self.config.model_dump(),
        )
        self.logger.info("Drift analysis completed: score=%s severity=%s", report.overall_score, report.severity.value)
        if report_mode:
            self._export_report(report, report_mode, output_path)
        return report

    def _analyze_columns(self, columns: Sequence[str]) -> List[ColumnReport]:
        if not columns:
            return []
        if self.config.n_jobs > 1 and len(columns) > 1:
            reports = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(self._analyze_single_column)(column) for column in columns
            )
            return list(reports)
        return [self._analyze_single_column(column) for column in columns]

    def _analyze_single_column(self, column: str) -> ColumnReport:
        reference_series = self.reference_df[column]
        current_series = self.current_df[column]
        role = column_role(reference_series)
        if role == "numeric":
            tests = run_numeric_tests(reference_series, current_series, self.config)
        else:
            tests = run_categorical_tests(reference_series, current_series, self.config)
        scores = [result.score for result in tests.values()]
        score = combine_scores(scores)
        severity = severity_from_score(score, self.config.severity_thresholds)
        summary = self._build_summary(column, role, tests, score)
        recommendation = self._build_recommendation(role, tests, severity)
        metadata = {
            "reference_constant": bool(reference_series.dropna().nunique() <= 1),
            "current_constant": bool(current_series.dropna().nunique() <= 1),
            "reference_dtype": str(reference_series.dtype),
            "current_dtype": str(current_series.dtype),
        }
        return ColumnReport(
            column=column,
            dtype=str(reference_series.dtype),
            role=role,
            score=score,
            severity=severity,
            tests=tests,
            summary=summary,
            recommendation=recommendation,
            metadata=metadata,
        )

    def _build_summary(self, column: str, role: str, tests: Dict[str, object], score: float) -> str:
        if not tests:
            return f"No tests executed for {column}."
        primary = max(tests.values(), key=lambda result: getattr(result, "score", 0.0))
        if getattr(primary, "interpretation", ""):
            return primary.interpretation
        return f"{column} analysed as {role} with score {score:.3f}."

    def _build_recommendation(self, role: str, tests: Dict[str, object], severity: DriftSeverity) -> str:
        triggered = [result for result in tests.values() if getattr(result, "triggered", False)]
        if not triggered:
            return "No material drift detected."
        top = max(triggered, key=lambda result: getattr(result, "score", 0.0))
        if top.recommendation:
            return top.recommendation
        if role == "numeric":
            return "Review numerical feature scaling, binning, and upstream data pipelines."
        return "Review categorical mappings, labels, and lookup tables."

    def _overall_score(self, reports: Sequence[ColumnReport]) -> float:
        if not reports:
            return 0.0
        weights = [self.config.feature_importance.get(report.column, 1.0) for report in reports]
        scores = [report.score for report in reports]
        return combine_scores(scores, weights)

    def _dataset_metrics(self, columns: Sequence[str]) -> Dict[str, float]:
        numeric_columns = [
            column
            for column in columns
            if column_role(self.reference_df[column]) == "numeric" and column_role(self.current_df[column]) == "numeric"
        ]
        metrics: Dict[str, float] = {}
        if not numeric_columns:
            return metrics
        reference_numeric = _safe_numeric_frame(self.reference_df, numeric_columns)
        current_numeric = _safe_numeric_frame(self.current_df, numeric_columns)
        if not reference_numeric.empty and not current_numeric.empty:
            ref_corr = reference_numeric.corr(numeric_only=True).fillna(0.0).to_numpy()
            cur_corr = current_numeric.corr(numeric_only=True).fillna(0.0).to_numpy()
            ref_cov = reference_numeric.cov(numeric_only=True).fillna(0.0).to_numpy()
            cur_cov = current_numeric.cov(numeric_only=True).fillna(0.0).to_numpy()
            metrics["correlation_drift"] = float(np.mean(np.abs(ref_corr - cur_corr)))
            metrics["covariance_drift"] = float(np.mean(np.abs(ref_cov - cur_cov)))
            metrics["quality_numeric_columns"] = float(len(reference_numeric.columns))
            if self.config.include_pca and len(reference_numeric.columns) >= 2:
                scaler = StandardScaler()
                ref_scaled = scaler.fit_transform(reference_numeric)
                cur_scaled = scaler.transform(current_numeric)
                n_components = min(5, ref_scaled.shape[1], ref_scaled.shape[0], cur_scaled.shape[0])
                if n_components >= 1:
                    pca = PCA(n_components=n_components, random_state=self.config.random_state)
                    ref_latent = pca.fit_transform(ref_scaled)
                    cur_latent = pca.transform(cur_scaled)
                    latent_distance = np.linalg.norm(ref_latent.mean(axis=0) - cur_latent.mean(axis=0))
                    metrics["pca_latent_drift"] = float(latent_distance)
        return metrics

    def _quality_score(self, selected_columns: Sequence[str], excluded_columns: Dict[str, str], schema_issues: List[str]) -> float:
        total = len(selected_columns) + len(excluded_columns) + len(schema_issues)
        if total == 0:
            return 1.0
        penalty = len(excluded_columns) + len(schema_issues)
        return float(np.clip(1.0 - (penalty / total), 0.0, 1.0))

    def _export_report(self, report: DatasetReport, report_mode: str, output_path: str | Path | None) -> None:
        report_mode = report_mode.lower()
        if report_mode == "json":
            report.to_json(output_path or "driftguard_report.json")
            return
        if report_mode == "csv":
            report.to_csv(output_path or "driftguard_report.csv")
            return
        if report_mode == "html":
            from ..reporting.html import generate_html_report

            generate_html_report(report, self.reference_df, self.current_df, output_path or "driftguard_report.html")
            return
        if report_mode == "pdf":
            from ..reporting.pdf import generate_pdf_report

            generate_pdf_report(report, self.reference_df, self.current_df, output_path or "driftguard_report.pdf")
            return
        if report_mode == "window":
            from ..reporting.window import launch_window_report

            launch_window_report(report, self.reference_df, self.current_df, output_path)
            return
        raise ValueError(f"Unsupported report_mode: {report_mode}")
