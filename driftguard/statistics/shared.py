from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Iterable as TypingIterable, List, Optional, Sequence

import numpy as np

from ..configs.defaults import SeverityThresholds
from ..models import DriftSeverity, DriftTestResult

EPS = 1e-12


def score_from_pvalue(p_value: float | None, alpha: float) -> float:
    if p_value is None:
        return 0.0
    if p_value >= alpha:
        return 0.0
    return float(np.clip(1.0 - (p_value / max(alpha, EPS)), 0.0, 1.0))


def score_from_distance(distance: float | None, threshold: float) -> float:
    if distance is None:
        return 0.0
    if distance <= 0:
        return 0.0
    return float(np.clip(distance / (distance + max(threshold, EPS)), 0.0, 1.0))


def score_from_ratio(ratio: float | None) -> float:
    if ratio is None:
        return 0.0
    ratio = abs(ratio)
    return float(np.clip(ratio / (1.0 + ratio), 0.0, 1.0))


def combine_scores(scores: Sequence[float], weights: Sequence[float] | None = None) -> float:
    if not scores:
        return 0.0
    values = np.asarray(scores, dtype=float)
    if weights is None:
        return float(np.clip(values.mean(), 0.0, 1.0))
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.sum() <= 0:
        return float(np.clip(values.mean(), 0.0, 1.0))
    return float(np.clip(np.average(values, weights=weight_array), 0.0, 1.0))


def severity_from_score(score: float, thresholds: SeverityThresholds) -> DriftSeverity:
    if score <= thresholds.no_drift_max:
        return DriftSeverity.NO_DRIFT
    if score <= thresholds.low_drift_max:
        return DriftSeverity.LOW_DRIFT
    if score <= thresholds.medium_drift_max:
        return DriftSeverity.MEDIUM_DRIFT
    if score <= thresholds.high_drift_max:
        return DriftSeverity.HIGH_DRIFT
    return DriftSeverity.CRITICAL_DRIFT


def make_result(
    test_name: str,
    *,
    statistic: float | None = None,
    p_value: float | None = None,
    score: float = 0.0,
    triggered: bool = False,
    effect_size: float | None = None,
    interpretation: str = "",
    recommendation: str = "",
    reference_summary: Dict[str, float] | None = None,
    current_summary: Dict[str, float] | None = None,
    details: Dict[str, object] | None = None,
) -> DriftTestResult:
    return DriftTestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        score=float(np.clip(score, 0.0, 1.0)),
        triggered=triggered,
        effect_size=effect_size,
        interpretation=interpretation,
        recommendation=recommendation,
        reference_summary=reference_summary or {},
        current_summary=current_summary or {},
        details=details or {},
    )

