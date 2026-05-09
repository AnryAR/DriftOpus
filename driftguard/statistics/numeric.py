from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ..configs.defaults import DriftAnalyzerConfig
from ..core.registry import get_custom_tests
from ..distances.core import (
    align_numeric_distributions,
    bhattacharyya_distance,
    bootstrap_ks_confidence_interval,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    mmd_rbf,
    psi_from_probabilities,
    wasserstein_distance,
)
from ..models import DriftTestResult
from ..statistics.pure import (
    anderson_darling_statistic,
    chi_square_statistic_from_histograms,
    cramervonmises_statistic,
    energy_distance,
    ks_pvalue,
    ks_statistic,
    mann_whitney_pvalue,
    mann_whitney_u_statistic,
    normal_sf,
    permutation_p_value,
    variance_ratio,
    variance_ratio_statistic,
)
from ..utils.sampling import series_to_numeric_array
from .shared import make_result, score_from_distance, score_from_pvalue


def _summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0.0}
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _histogram_contingency(
    reference: np.ndarray,
    current: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    contingency = np.vstack([ref_counts.astype(float), cur_counts.astype(float)])
    total = contingency.sum()
    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / max(total, 1.0)
    return contingency, expected


def ks_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    statistic = ks_statistic(reference, current)
    p_value = ks_pvalue(statistic, reference.size, current.size)
    lower, upper, observed = bootstrap_ks_confidence_interval(
        reference,
        current,
        iterations=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    score = max(
        score_from_pvalue(float(p_value), config.p_value_alpha),
        score_from_distance(float(statistic), config.ks_threshold),
    )
    triggered = bool(p_value < config.p_value_alpha)
    return make_result(
        "ks_test",
        statistic=float(statistic),
        p_value=float(p_value),
        score=score,
        triggered=triggered,
        interpretation=f"KS statistic={float(statistic):.4f}, p-value={float(p_value):.6g}.",
        recommendation="Inspect distributional changes and consider retraining if the feature is important.",
        reference_summary=_summary(reference),
        current_summary=_summary(current),
        details={"bootstrap_ci": {"lower": lower, "upper": upper, "observed": observed}},
    )


def anderson_darling_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    try:
        p_value, observed = permutation_p_value(
            reference,
            current,
            anderson_darling_statistic,
            iterations=config.bootstrap_iterations,
            random_state=config.random_state,
        )
        score = score_from_pvalue(p_value, config.p_value_alpha)
        return make_result(
            "anderson_darling_test",
            statistic=float(observed),
            p_value=float(p_value),
            score=score,
            triggered=bool(p_value < config.p_value_alpha),
            interpretation=f"Anderson-Darling statistic={float(observed):.4f}, p-value={float(p_value):.6g}.",
            recommendation="Check for tail shifts and consider a fresh baseline.",
            reference_summary=_summary(reference),
            current_summary=_summary(current),
            details={"permutation_statistic": float(observed), "iterations": config.bootstrap_iterations},
        )
    except Exception as exc:
        return make_result(
            "anderson_darling_test",
            score=0.0,
            interpretation="Anderson-Darling test could not be computed for this feature.",
            recommendation="The distributions may be degenerate; inspect manually.",
            details={"error": str(exc)},
        )


def wasserstein_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    statistic = wasserstein_distance(reference, current)
    score = score_from_distance(statistic, config.wasserstein_threshold)
    return make_result(
        "wasserstein_distance",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        effect_size=statistic,
        interpretation=f"Wasserstein distance={statistic:.4f}.",
        recommendation="A larger transport cost indicates a shift in the feature distribution.",
        reference_summary=_summary(reference),
        current_summary=_summary(current),
    )


def kl_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, _ = align_numeric_distributions(reference, current, config.bins)
    statistic = kl_divergence(ref_prob, cur_prob)
    score = score_from_distance(statistic, config.psi_threshold)
    return make_result(
        "kl_divergence",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"KL divergence={statistic:.4f}.",
        recommendation="Large divergence suggests a change in the overall probability mass.",
        details={"reference_probabilities": ref_prob.tolist(), "current_probabilities": cur_prob.tolist()},
    )


def psi_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, _ = align_numeric_distributions(reference, current, config.bins)
    statistic = psi_from_probabilities(ref_prob, cur_prob)
    score = score_from_distance(statistic, config.psi_threshold)
    return make_result(
        "psi",
        statistic=statistic,
        score=score,
        triggered=statistic >= config.psi_threshold,
        interpretation=f"PSI={statistic:.4f}.",
        recommendation="A PSI above the threshold indicates practical drift.",
        details={"reference_probabilities": ref_prob.tolist(), "current_probabilities": cur_prob.tolist()},
    )


def js_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, _ = align_numeric_distributions(reference, current, config.bins)
    statistic = js_divergence(ref_prob, cur_prob)
    score = score_from_distance(statistic, config.js_threshold)
    return make_result(
        "jensen_shannon_divergence",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Jensen-Shannon divergence={statistic:.4f}.",
        recommendation="Divergent histograms often mean upstream feature engineering drift.",
        details={"reference_probabilities": ref_prob.tolist(), "current_probabilities": cur_prob.tolist()},
    )


def hellinger_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, _ = align_numeric_distributions(reference, current, config.bins)
    statistic = hellinger_distance(ref_prob, cur_prob)
    score = score_from_distance(statistic, config.js_threshold)
    return make_result(
        "hellinger_distance",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Hellinger distance={statistic:.4f}.",
        recommendation="Hellinger distance captures shape changes between distributions.",
    )


def bhattacharyya_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, _ = align_numeric_distributions(reference, current, config.bins)
    statistic = bhattacharyya_distance(ref_prob, cur_prob)
    score = score_from_distance(statistic, config.js_threshold)
    return make_result(
        "bhattacharyya_distance",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Bhattacharyya distance={statistic:.4f}.",
        recommendation="Bhattacharyya distance grows as overlap between distributions shrinks.",
    )


def cramer_von_mises_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    try:
        p_value, observed = permutation_p_value(
            reference,
            current,
            cramervonmises_statistic,
            iterations=config.bootstrap_iterations,
            random_state=config.random_state,
        )
        score = score_from_pvalue(float(p_value), config.p_value_alpha)
        return make_result(
            "cramer_von_mises_test",
            statistic=float(observed),
            p_value=float(p_value),
            score=score,
            triggered=bool(p_value < config.p_value_alpha),
            interpretation=f"Cramer-von Mises statistic={float(observed):.4f}, p-value={float(p_value):.6g}.",
            recommendation="Inspect the distribution tails and central tendency together.",
            details={"permutation_statistic": float(observed), "iterations": config.bootstrap_iterations},
        )
    except Exception as exc:
        return make_result(
            "cramer_von_mises_test",
            score=0.0,
            interpretation="Cramer-von Mises test could not be computed for this feature.",
            recommendation="The distributions may be degenerate; inspect manually.",
            details={"error": str(exc)},
        )


def mann_whitney_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    statistic = mann_whitney_u_statistic(reference, current)
    p_value = mann_whitney_pvalue(reference, current)
    score = score_from_pvalue(float(p_value), config.p_value_alpha)
    return make_result(
        "mann_whitney_u_test",
        statistic=float(statistic),
        p_value=float(p_value),
        score=score,
        triggered=bool(p_value < config.p_value_alpha),
        interpretation=f"Mann-Whitney U statistic={float(statistic):.4f}, p-value={float(p_value):.6g}.",
        recommendation="A rank shift can signal changes in the feature median or skew.",
    )


def energy_distance_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    statistic = float(energy_distance(reference, current, max_points=config.max_samples_per_column))
    score = score_from_distance(statistic, config.wasserstein_threshold)
    return make_result(
        "energy_distance",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Energy distance={statistic:.4f}.",
        recommendation="Energy distance rising across time usually warrants investigation.",
    )


def z_test_mean_shift(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_mean = float(np.mean(reference))
    cur_mean = float(np.mean(current))
    ref_var = float(np.var(reference, ddof=1)) if reference.size > 1 else 0.0
    cur_var = float(np.var(current, ddof=1)) if current.size > 1 else 0.0
    standard_error = np.sqrt((ref_var / max(reference.size, 1)) + (cur_var / max(current.size, 1)))
    if standard_error <= 0:
        z_score = 0.0 if ref_mean == cur_mean else float("inf")
        p_value = 0.0 if ref_mean != cur_mean else 1.0
    else:
        z_score = (ref_mean - cur_mean) / standard_error
        p_value = float(2.0 * normal_sf(abs(z_score)))
    score = score_from_pvalue(p_value, config.p_value_alpha)
    return make_result(
        "z_test_mean_shift",
        statistic=float(z_score),
        p_value=p_value,
        score=score,
        triggered=bool(p_value < config.p_value_alpha),
        interpretation=f"Z-score={float(z_score):.4f}, p-value={p_value:.6g}.",
        recommendation="Mean shifts can indicate calibration or population changes.",
        details={"reference_mean": ref_mean, "current_mean": cur_mean},
    )


def f_test_variance_ratio(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_var = float(np.var(reference, ddof=1)) if reference.size > 1 else 0.0
    cur_var = float(np.var(current, ddof=1)) if current.size > 1 else 0.0
    ratio = variance_ratio(reference, current)
    try:
        p_value, observed = permutation_p_value(
            reference,
            current,
            variance_ratio_statistic,
            iterations=config.bootstrap_iterations,
            random_state=config.random_state,
        )
    except Exception:
        observed = variance_ratio_statistic(reference, current)
        p_value = 1.0
    score = score_from_pvalue(float(p_value), config.p_value_alpha)
    return make_result(
        "variance_ratio_f_test",
        statistic=float(ratio),
        p_value=float(p_value),
        score=score,
        triggered=bool(p_value < config.p_value_alpha),
        effect_size=float(observed),
        interpretation=f"Variance ratio={float(ratio):.4f}, p-value={float(p_value):.6g}.",
        recommendation="A variance change may affect model uncertainty and score stability.",
        details={"reference_variance": ref_var, "current_variance": cur_var, "log_variance_ratio": float(observed)},
    )


def chi_square_binning_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    ref_prob, cur_prob, edges = align_numeric_distributions(reference, current, config.bins)
    contingency, expected = _histogram_contingency(reference, current, edges)
    statistic = chi_square_statistic_from_histograms(reference, current, edges)
    try:
        p_value, _ = permutation_p_value(
            reference,
            current,
            lambda ref, cur: chi_square_statistic_from_histograms(ref, cur, edges),
            iterations=config.bootstrap_iterations,
            random_state=config.random_state,
        )
    except Exception:
        p_value = 1.0
    score = score_from_pvalue(float(p_value), config.p_value_alpha)
    return make_result(
        "chi_square_binning_test",
        statistic=float(statistic),
        p_value=float(p_value),
        score=score,
        triggered=bool(p_value < config.p_value_alpha),
        interpretation=f"Chi-square statistic={float(statistic):.4f}, p-value={float(p_value):.6g}.",
        recommendation="Binned frequency drift can expose coarse distribution changes.",
        details={
            "bin_edges": edges.tolist(),
            "expected": expected.tolist(),
            "observed": contingency.tolist(),
            "reference_probabilities": ref_prob.tolist(),
            "current_probabilities": cur_prob.tolist(),
        },
    )


def bootstrap_ks_ci_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    lower, upper, observed = bootstrap_ks_confidence_interval(
        reference,
        current,
        iterations=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    score = score_from_distance(observed, config.ks_threshold)
    triggered = bool(lower > 0.0 or observed > config.ks_threshold)
    return make_result(
        "ks_bootstrap_ci",
        statistic=observed,
        score=score,
        triggered=triggered,
        interpretation=f"Bootstrap KS CI: [{lower:.4f}, {upper:.4f}] around statistic={observed:.4f}.",
        recommendation="If the bootstrap interval is consistently high, the shift is stable.",
        details={"lower": lower, "upper": upper},
    )


def mmd_test(reference: np.ndarray, current: np.ndarray, config: DriftAnalyzerConfig) -> DriftTestResult:
    statistic = mmd_rbf(reference, current)
    score = score_from_distance(statistic, config.wasserstein_threshold)
    return make_result(
        "mmd_rbf",
        statistic=statistic,
        score=score,
        triggered=score > 0.0,
        interpretation=f"Maximum Mean Discrepancy={statistic:.4f}.",
        recommendation="MMD is sensitive to higher-order distribution changes.",
    )


NUMERIC_TESTS: List[Callable[[np.ndarray, np.ndarray, DriftAnalyzerConfig], DriftTestResult]] = [
    ks_test,
    anderson_darling_test,
    wasserstein_test,
    kl_test,
    psi_test,
    js_test,
    hellinger_test,
    bhattacharyya_test,
    cramer_von_mises_test,
    mann_whitney_test,
    energy_distance_test,
    z_test_mean_shift,
    f_test_variance_ratio,
    chi_square_binning_test,
    bootstrap_ks_ci_test,
    mmd_test,
]


def _normalize_custom_result(test_name: str, result: DriftTestResult | Dict[str, object]) -> DriftTestResult:
    if isinstance(result, DriftTestResult):
        return result
    payload = dict(result)
    payload.setdefault("test_name", test_name)
    return DriftTestResult(**payload)


def run_numeric_tests(reference: pd.Series, current: pd.Series, config: DriftAnalyzerConfig) -> Dict[str, DriftTestResult]:
    reference_values = series_to_numeric_array(reference, config.max_samples_per_column, config.random_state)
    current_values = series_to_numeric_array(current, config.max_samples_per_column, config.random_state)
    results: Dict[str, DriftTestResult] = {}
    for test in NUMERIC_TESTS:
        try:
            result = test(reference_values, current_values, config)
        except Exception as exc:
            result = make_result(
                test.__name__,
                score=0.0,
                interpretation=f"{test.__name__} failed to compute.",
                recommendation="Inspect the feature manually; the distribution may be degenerate.",
                details={"error": str(exc)},
            )
        results[result.test_name] = result
    for custom_test in get_custom_tests("numeric"):
        try:
            custom_result = custom_test.run(reference, current, config)
            result = _normalize_custom_result(custom_test.name, custom_result)
            results[result.test_name] = result
        except Exception as exc:
            results[custom_test.name] = make_result(
                custom_test.name,
                score=0.0,
                interpretation=f"Custom numeric test {custom_test.name} failed.",
                recommendation="Review the custom plugin implementation.",
                details={"error": str(exc)},
            )
    return results
