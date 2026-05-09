from __future__ import annotations

import pandas as pd

from ..models import DriftTestResult
from ..statistics.pure import mutual_information
from ..statistics.shared import make_result, score_from_distance


def _prepare_feature_labels(series: pd.Series, bins: int) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return clean.astype(str)
    if pd.api.types.is_numeric_dtype(clean.dtype):
        try:
            bucket_count = min(bins, max(2, clean.nunique()))
            return pd.qcut(clean, q=bucket_count, duplicates="drop").astype(str)
        except Exception:
            return clean.astype(str)
    return clean.astype(str)


def target_leakage_drift_check(
    feature_reference: pd.Series,
    feature_current: pd.Series,
    target_reference: pd.Series | None = None,
    target_current: pd.Series | None = None,
    bins: int = 10,
) -> DriftTestResult:
    if target_reference is None or target_current is None:
        return make_result(
            "target_leakage_drift_check",
            score=0.0,
            interpretation="Target data was not supplied for leakage monitoring.",
            recommendation="Provide reference and current target series to enable leakage drift checks.",
        )

    reference_frame = pd.DataFrame({"feature": feature_reference, "target": target_reference}).dropna()
    current_frame = pd.DataFrame({"feature": feature_current, "target": target_current}).dropna()
    reference_features = _prepare_feature_labels(reference_frame["feature"], bins=bins)
    current_features = _prepare_feature_labels(current_frame["feature"], bins=bins)
    reference_targets = reference_frame.loc[reference_features.index, "target"].astype(str)
    current_targets = current_frame.loc[current_features.index, "target"].astype(str)
    reference_mi = float(mutual_information(reference_features.astype(str).to_numpy(), reference_targets.to_numpy()))
    current_mi = float(mutual_information(current_features.astype(str).to_numpy(), current_targets.to_numpy()))
    statistic = abs(reference_mi - current_mi)
    score = score_from_distance(statistic, 0.10)
    return make_result(
        "target_leakage_drift_check",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Target leakage mutual information changed from {reference_mi:.4f} to {current_mi:.4f}.",
        recommendation="Large changes may indicate that a feature is becoming more or less predictive of the target.",
        details={"reference_mi": reference_mi, "current_mi": current_mi},
    )
