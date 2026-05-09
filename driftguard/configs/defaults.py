from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SeverityThresholds(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    no_drift_max: float = 0.10
    low_drift_max: float = 0.30
    medium_drift_max: float = 0.60
    high_drift_max: float = 0.80

    @field_validator(
        "no_drift_max",
        "low_drift_max",
        "medium_drift_max",
        "high_drift_max",
    )
    @classmethod
    def _bounded(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Severity thresholds must be between 0 and 1.")
        return value

    @field_validator("low_drift_max")
    @classmethod
    def _order_low(cls, value: float, info: ValidationInfo) -> float:
        no_drift_max = info.data.get("no_drift_max")
        if no_drift_max is not None and value < no_drift_max:
            raise ValueError("low_drift_max must be >= no_drift_max.")
        return value

    @field_validator("medium_drift_max")
    @classmethod
    def _order_medium(cls, value: float, info: ValidationInfo) -> float:
        low_drift_max = info.data.get("low_drift_max")
        if low_drift_max is not None and value < low_drift_max:
            raise ValueError("medium_drift_max must be >= low_drift_max.")
        return value

    @field_validator("high_drift_max")
    @classmethod
    def _order_high(cls, value: float, info: ValidationInfo) -> float:
        medium_drift_max = info.data.get("medium_drift_max")
        if medium_drift_max is not None and value < medium_drift_max:
            raise ValueError("high_drift_max must be >= medium_drift_max.")
        return value


class DriftAnalyzerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ks_threshold: float = 0.05
    psi_threshold: float = 0.20
    js_threshold: float = 0.10
    wasserstein_threshold: float = 0.10
    p_value_alpha: float = 0.05
    bins: int = 10
    max_samples_per_column: int = 100_000
    bootstrap_iterations: int = 200
    categorical_rare_threshold: float = 0.01
    categorical_top_k: int = 25
    n_jobs: int = 1
    random_state: int = 42
    include_pca: bool = True
    severity_thresholds: SeverityThresholds = Field(default_factory=SeverityThresholds)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    report_title: str = "DriftGuard Report"

    @field_validator(
        "ks_threshold",
        "psi_threshold",
        "js_threshold",
        "wasserstein_threshold",
        "p_value_alpha",
        "categorical_rare_threshold",
    )
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("Thresholds must be non-negative.")
        return value

    @field_validator(
        "bins",
        "max_samples_per_column",
        "bootstrap_iterations",
        "categorical_top_k",
        "n_jobs",
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Integer configuration values must be positive.")
        return value

    @field_validator("feature_importance")
    @classmethod
    def _importance_values(cls, value: Dict[str, float]) -> Dict[str, float]:
        for key, weight in value.items():
            if weight < 0.0:
                raise ValueError(f"Feature importance for {key!r} must be non-negative.")
        return value
