from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils.serialization import to_native


class DriftSeverity(str, Enum):
    NO_DRIFT = "NO_DRIFT"
    LOW_DRIFT = "LOW_DRIFT"
    MEDIUM_DRIFT = "MEDIUM_DRIFT"
    HIGH_DRIFT = "HIGH_DRIFT"
    CRITICAL_DRIFT = "CRITICAL_DRIFT"


@dataclass
class DriftTestResult:
    test_name: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    score: float = 0.0
    triggered: bool = False
    effect_size: Optional[float] = None
    interpretation: str = ""
    recommendation: str = ""
    reference_summary: Dict[str, Any] = field(default_factory=dict)
    current_summary: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_native(self)


@dataclass
class ColumnReport:
    column: str
    dtype: str
    role: str
    score: float
    severity: DriftSeverity
    tests: Dict[str, DriftTestResult] = field(default_factory=dict)
    summary: str = ""
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_native(self)


@dataclass
class DatasetReport:
    title: str
    overall_score: float
    severity: DriftSeverity
    columns: Dict[str, ColumnReport] = field(default_factory=dict)
    excluded_columns: Dict[str, str] = field(default_factory=dict)
    schema_issues: List[str] = field(default_factory=list)
    dataset_metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return to_native(self)

    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for column, report in self.columns.items():
            rows.append(
                {
                    "column": column,
                    "dtype": report.dtype,
                    "role": report.role,
                    "score": report.score,
                    "severity": report.severity.value,
                    "summary": report.summary,
                    "recommendation": report.recommendation,
                }
            )
        return pd.DataFrame(rows)

    def to_json(self, path: Optional[Path | str] = None) -> str:
        import json

        data = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(data, encoding="utf-8")
        return data

    def to_csv(self, path: Optional[Path | str] = None) -> str:
        csv_text = self.to_dataframe().to_csv(index=False)
        if path is not None:
            Path(path).write_text(csv_text, encoding="utf-8")
        return csv_text

    def to_html(self, path: Optional[Path | str] = None) -> str:
        df = self.to_dataframe()
        html = df.to_html(index=False, escape=False)
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

