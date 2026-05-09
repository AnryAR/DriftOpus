from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ..configs.defaults import DriftAnalyzerConfig
from ..core.registry import get_custom_tests
from ..distances.core import categorical_probability_vectors, js_divergence, normalized_entropy, psi_from_probabilities
from ..models import DriftTestResult
from ..statistics.pure import (
    category_counts,
    chi_square_statistic_from_labels,
    g_test_statistic_from_labels,
    mutual_information,
    permutation_p_value,
)
from ..utils.sampling import series_to_categorical_array
from .shared import make_result, score_from_distance, score_from_pvalue


def _summary(series: pd.Series) -> Dict[str, float]:
    return {
        "count": float(series.dropna().shape[0]),
        "unique": float(series.dropna().nunique()),
        "cardinality": float(series.dropna().nunique()),
    }


def _contingency(reference: np.ndarray, current: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray, List[str]]:
    _, _, labels = categorical_probability_vectors(reference, current, top_k=top_k)
    ref_counts = category_counts(reference, labels)
    cur_counts = category_counts(current, labels)
    contingency = np.vstack([ref_counts, cur_counts]).astype(float)
    return contingency, np.vstack([ref_counts, cur_counts]).astype(float), labels


def _expected_from_contingency(contingency: np.ndarray) -> np.ndarray:
    total = contingency.sum()
    if total <= 0.0:
        return np.zeros_like(contingency, dtype=float)
    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    return row_sums @ col_sums / total


def chi_square_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    contingency, observed, labels = _contingency(reference, current, config.categorical_top_k)
    try:
        statistic = chi_square_statistic_from_labels(reference, current, labels)
        p_value, _ = permutation_p_value(
            reference,
            current,
            lambda ref, cur: chi_square_statistic_from_labels(ref, cur, labels),
            iterations=config.bootstrap_iterations,
            random_state=config.random_state,
        )
    except Exception as exc:
        return make_result(
            "chi_square_test",
            score=0.0,
            interpretation="Chi-square test could not be computed.",
            recommendation="The category support may be too sparse; inspect manually.",
            details={"error": str(exc), "categories": labels},
        )
    expected = _expected_from_contingency(contingency)
    score = score_from_pvalue(float(p_value), config.p_value_alpha)
    return make_result(
        "chi_square_test",
        statistic=float(statistic),
        p_value=float(p_value),
        score=score,
        triggered=bool(p_value < config.p_value_alpha),
        interpretation=f"Chi-square statistic={float(statistic):.4f}, p-value={float(p_value):.6g}.",
        recommendation="Category frequency changes can affect encoding stability and downstream models.",
        details={"categories": labels, "expected": expected.tolist(), "observed": observed.tolist()},
    )


def g_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    contingency, observed, labels = _contingency(reference, current, config.categorical_top_k)
    try:
        statistic = g_test_statistic_from_labels(reference, current, labels)
        p_value, _ = permutation_p_value(
            reference,
            current,
            lambda ref, cur: g_test_statistic_from_labels(ref, cur, labels),
            iterations=config.bootstrap_iterations,
            random_state=config.random_state,
        )
    except Exception as exc:
        return make_result(
            "g_test",
            score=0.0,
            interpretation="G-test could not be computed.",
            recommendation="Sparse categories may require more smoothing or grouping.",
            details={"error": str(exc), "categories": labels},
        )
    expected = _expected_from_contingency(contingency)
    score = score_from_pvalue(float(p_value), config.p_value_alpha)
    return make_result(
        "g_test",
        statistic=float(statistic),
        p_value=float(p_value),
        score=score,
        triggered=bool(p_value < config.p_value_alpha),
        interpretation=f"G statistic={float(statistic):.4f}, p-value={float(p_value):.6g}.",
        recommendation="A large likelihood-ratio shift suggests category mix drift.",
        details={"categories": labels, "expected": expected.tolist(), "observed": observed.tolist()},
    )


def entropy_shift(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, labels = categorical_probability_vectors(reference, current, top_k=config.categorical_top_k)
    ref_entropy = normalized_entropy(ref_prob)
    cur_entropy = normalized_entropy(cur_prob)
    statistic = abs(ref_entropy - cur_entropy)
    score = score_from_distance(statistic, 0.10)
    return make_result(
        "entropy_shift",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Normalized entropy changed from {ref_entropy:.4f} to {cur_entropy:.4f}.",
        recommendation="Entropy shifts often appear when category diversity changes abruptly.",
        details={"categories": labels, "reference_entropy": ref_entropy, "current_entropy": cur_entropy},
    )


def js_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, labels = categorical_probability_vectors(reference, current, top_k=config.categorical_top_k)
    statistic = js_divergence(ref_prob, cur_prob)
    score = score_from_distance(statistic, config.js_threshold)
    return make_result(
        "jensen_shannon_divergence",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Jensen-Shannon divergence={statistic:.4f}.",
        recommendation="This quantifies symmetric divergence between the categorical distributions.",
        details={"categories": labels},
    )


def psi_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, labels = categorical_probability_vectors(reference, current, top_k=config.categorical_top_k)
    statistic = psi_from_probabilities(ref_prob, cur_prob)
    score = score_from_distance(statistic, config.psi_threshold)
    return make_result(
        "psi",
        statistic=statistic,
        score=score,
        triggered=statistic >= config.psi_threshold,
        interpretation=f"PSI={statistic:.4f}.",
        recommendation="A PSI above threshold suggests categorical drift is operationally meaningful.",
        details={"categories": labels},
    )


def frequency_drift(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, labels = categorical_probability_vectors(reference, current, top_k=config.categorical_top_k)
    statistic = float(0.5 * np.abs(ref_prob - cur_prob).sum())
    score = score_from_distance(statistic, 0.10)
    return make_result(
        "category_frequency_drift",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Total variation distance={statistic:.4f}.",
        recommendation="The category frequencies have shifted. Check encoders and upstream lookup tables.",
        details={"categories": labels},
    )


def rare_category_emergence(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_series = pd.Series(reference, dtype="object")
    cur_series = pd.Series(current, dtype="object")
    ref_freq = ref_series.value_counts(normalize=True)
    cur_freq = cur_series.value_counts(normalize=True)
    emergent = [
        category
        for category, freq in cur_freq.items()
        if (category not in ref_freq.index and freq >= config.categorical_rare_threshold)
        or (category in ref_freq.index and ref_freq[category] < config.categorical_rare_threshold and freq >= config.categorical_rare_threshold)
    ]
    score = float(min(1.0, sum(cur_freq.get(category, 0.0) for category in emergent)))
    return make_result(
        "rare_category_emergence",
        score=score,
        triggered=bool(emergent),
        interpretation=f"Emergent rare categories: {emergent}" if emergent else "No new rare categories detected.",
        recommendation="Investigate unseen labels, mapping changes, or upstream data quality issues.",
        details={"emergent_categories": emergent},
    )


def unseen_category_detection(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_series = pd.Series(reference, dtype="object")
    cur_series = pd.Series(current, dtype="object")
    unseen = sorted(set(cur_series.dropna().unique()) - set(ref_series.dropna().unique()))
    if unseen:
        mass = float(cur_series.isin(unseen).mean())
    else:
        mass = 0.0
    score = float(np.clip(mass, 0.0, 1.0))
    return make_result(
        "new_unseen_category_detection",
        score=score,
        triggered=bool(unseen),
        interpretation=f"Unseen categories detected: {unseen}" if unseen else "No unseen categories detected.",
        recommendation="Map or validate new labels before they enter the model pipeline.",
        details={"unseen_categories": unseen, "mass": mass},
    )


def categorical_mutual_information_shift(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    labels = np.concatenate([np.asarray(reference, dtype="object"), np.asarray(current, dtype="object")])
    source = np.concatenate([np.zeros(len(reference), dtype=int), np.ones(len(current), dtype=int)])
    statistic = float(mutual_information(labels, source))
    score = score_from_distance(statistic, 0.10)
    return make_result(
        "categorical_mutual_information_shift",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Mutual information with source label={statistic:.4f}.",
        recommendation="High mutual information means the category is informative of the period/source.",
    )


def cardinality_drift(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_unique = len(pd.Series(reference).dropna().unique())
    cur_unique = len(pd.Series(current).dropna().unique())
    if ref_unique == 0:
        statistic = float(cur_unique)
    else:
        statistic = float(abs(cur_unique - ref_unique) / ref_unique)
    score = score_from_distance(statistic, 0.10)
    return make_result(
        "cardinality_drift",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Cardinality changed from {ref_unique} to {cur_unique}.",
        recommendation="Cardinality changes can break encodings and feature stores.",
        details={"reference_unique": ref_unique, "current_unique": cur_unique},
    )


CATEGORICAL_TESTS: List[Callable[[np.ndarray, np.ndarray, DriftAnalyzerConfig], DriftTestResult]] = [
    g_test,
    chi_square_test,
    entropy_shift,
    js_test,
    psi_test,
    frequency_drift,
    rare_category_emergence,
    unseen_category_detection,
    categorical_mutual_information_shift,
    cardinality_drift,
]


def _normalize_custom_result(test_name: str, result: DriftTestResult | Dict[str, object]) -> DriftTestResult:
    if isinstance(result, DriftTestResult):
        return result
    payload = dict(result)
    payload.setdefault("test_name", test_name)
    return DriftTestResult(**payload)


def run_categorical_tests(reference: pd.Series, current: pd.Series, config: DriftAnalyzerConfig) -> Dict[str, DriftTestResult]:
    reference_values = series_to_categorical_array(reference, config.max_samples_per_column, config.random_state)
    current_values = series_to_categorical_array(current, config.max_samples_per_column, config.random_state)
    results: Dict[str, DriftTestResult] = {}
    for test in CATEGORICAL_TESTS:
        try:
            result = test(reference_values, current_values, config)
        except Exception as exc:
            result = make_result(
                test.__name__,
                score=0.0,
                interpretation=f"{test.__name__} failed to compute.",
                recommendation="Inspect the feature manually; sparse support may be the issue.",
                details={"error": str(exc)},
            )
        results[result.test_name] = result
    for custom_test in get_custom_tests("categorical"):
        try:
            custom_result = custom_test.run(reference, current, config)
            result = _normalize_custom_result(custom_test.name, custom_result)
            results[result.test_name] = result
        except Exception as exc:
            results[custom_test.name] = make_result(
                custom_test.name,
                score=0.0,
                interpretation=f"Custom categorical test {custom_test.name} failed.",
                recommendation="Review the custom plugin implementation.",
                details={"error": str(exc)},
            )
    return results
