from __future__ import annotations

from math import erfc, exp, log, sqrt
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_SAMPLE_SIZE = 1000


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def sample_array(values: Sequence[float] | np.ndarray, max_samples: int, random_state: int) -> np.ndarray:
    array = np.asarray(values)
    if array.size <= max_samples:
        return array
    rng = np.random.default_rng(random_state)
    indices = rng.choice(array.size, size=max_samples, replace=False)
    return array[indices]


def entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, EPS, None)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def kl_divergence(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    reference = np.clip(np.asarray(reference_probabilities, dtype=float), EPS, None)
    current = np.clip(np.asarray(current_probabilities, dtype=float), EPS, None)
    reference = reference / reference.sum()
    current = current / current.sum()
    return float(np.sum(reference * np.log(reference / current)))


def js_divergence(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    reference = np.asarray(reference_probabilities, dtype=float)
    current = np.asarray(current_probabilities, dtype=float)
    mixture = 0.5 * (reference + current)
    return 0.5 * kl_divergence(reference, mixture) + 0.5 * kl_divergence(current, mixture)


def hellinger_distance(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    reference = np.clip(np.asarray(reference_probabilities, dtype=float), EPS, None)
    current = np.clip(np.asarray(current_probabilities, dtype=float), EPS, None)
    reference = reference / reference.sum()
    current = current / current.sum()
    return float(np.sqrt(0.5 * np.square(np.sqrt(reference) - np.sqrt(current)).sum()))


def bhattacharyya_distance(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    reference = np.clip(np.asarray(reference_probabilities, dtype=float), EPS, None)
    current = np.clip(np.asarray(current_probabilities, dtype=float), EPS, None)
    reference = reference / reference.sum()
    current = current / current.sum()
    coefficient = float(np.sum(np.sqrt(reference * current)))
    return float(-np.log(max(coefficient, EPS)))


def psi_from_probabilities(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    reference = np.clip(np.asarray(reference_probabilities, dtype=float), EPS, None)
    current = np.clip(np.asarray(current_probabilities, dtype=float), EPS, None)
    reference = reference / reference.sum()
    current = current / current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def normalized_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, EPS, None)
    probs = probs / probs.sum()
    if probs.size <= 1:
        return 0.0
    return float(entropy(probs) / np.log(probs.size))


def _sorted_support(reference: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.sort(_as_float_array(reference))
    current = np.sort(_as_float_array(current))
    if reference.size == 0 or current.size == 0:
        return np.array([]), np.array([]), np.array([])
    support = np.unique(np.concatenate([reference, current]))
    if support.size < 2:
        return support, np.array([]), np.array([])
    reference_cdf = np.searchsorted(reference, support, side="right") / reference.size
    current_cdf = np.searchsorted(current, support, side="right") / current.size
    return support, reference_cdf, current_cdf


def ks_statistic(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    support, reference_cdf, current_cdf = _sorted_support(np.asarray(reference), np.asarray(current))
    if support.size < 2:
        return 0.0
    return float(np.max(np.abs(reference_cdf - current_cdf)))


def ks_pvalue(statistic: float, n_reference: int, n_current: int) -> float:
    if n_reference <= 0 or n_current <= 0:
        return 1.0
    if statistic <= 0.0:
        return 1.0
    effective_n = sqrt((n_reference * n_current) / (n_reference + n_current))
    if effective_n <= 0.0:
        return 1.0
    lambda_value = (effective_n + 0.12 + 0.11 / effective_n) * statistic
    total = 0.0
    for term_index in range(1, 256):
        term = (-1) ** (term_index - 1) * exp(-2.0 * (term_index**2) * (lambda_value**2))
        total += term
        if abs(term) < 1e-12:
            break
    return float(np.clip(2.0 * total, 0.0, 1.0))


def wasserstein_distance(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    support, reference_cdf, current_cdf = _sorted_support(np.asarray(reference), np.asarray(current))
    if support.size < 2:
        return 0.0
    deltas = np.diff(support)
    return float(np.sum(deltas * np.abs(reference_cdf[:-1] - current_cdf[:-1])))


def cramervonmises_statistic(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    support, reference_cdf, current_cdf = _sorted_support(np.asarray(reference), np.asarray(current))
    if support.size < 2:
        return 0.0
    deltas = np.diff(support)
    difference = reference_cdf[:-1] - current_cdf[:-1]
    range_width = max(float(support[-1] - support[0]), EPS)
    return float(np.sum(deltas * np.square(difference)) / range_width)


def anderson_darling_statistic(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    support, reference_cdf, current_cdf = _sorted_support(np.asarray(reference), np.asarray(current))
    if support.size < 2:
        return 0.0
    deltas = np.diff(support)
    difference = reference_cdf[:-1] - current_cdf[:-1]
    pooled = 0.5 * (reference_cdf[:-1] + current_cdf[:-1])
    weights = 1.0 / (pooled * (1.0 - pooled) + EPS)
    range_width = max(float(support[-1] - support[0]), EPS)
    return float(np.sum(deltas * np.square(difference) * weights) / range_width)


def _sample_if_needed(values: np.ndarray, max_points: int, random_state: int) -> np.ndarray:
    if values.size <= max_points:
        return values
    rng = np.random.default_rng(random_state)
    indices = rng.choice(values.size, size=max_points, replace=False)
    return values[indices]


def _mean_absolute_difference_same(values: np.ndarray) -> float:
    values = np.sort(_as_float_array(values))
    n = values.size
    if n < 2:
        return 0.0
    coefficients = 2.0 * np.arange(1, n + 1) - n - 1
    total = float(np.sum(coefficients * values))
    return float(2.0 * total / (n * (n - 1)))


def _mean_absolute_difference_between(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.sort(_as_float_array(reference))
    current = np.sort(_as_float_array(current))
    n_reference = reference.size
    n_current = current.size
    if n_reference == 0 or n_current == 0:
        return 0.0
    prefix_current = np.concatenate([[0.0], np.cumsum(current)])
    count_le = np.searchsorted(current, reference, side="right")
    sum_le = prefix_current[count_le]
    sum_total = float(prefix_current[-1])
    count_gt = n_current - count_le
    sum_gt = sum_total - sum_le
    total = float(np.sum(reference * count_le - sum_le + sum_gt - reference * count_gt))
    return float(total / (n_reference * n_current))


def energy_distance(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray, max_points: int = 1000) -> float:
    reference_array = _sample_if_needed(_as_float_array(reference), max_points, 42)
    current_array = _sample_if_needed(_as_float_array(current), max_points, 43)
    if reference_array.size == 0 or current_array.size == 0:
        return 0.0
    between = _mean_absolute_difference_between(reference_array, current_array)
    within_reference = _mean_absolute_difference_same(reference_array)
    within_current = _mean_absolute_difference_same(current_array)
    return float(max(0.0, 2.0 * between - within_reference - within_current))


def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    counts = np.clip(counts, 0.0, None) + EPS
    return counts / counts.sum()


def align_numeric_distributions(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_array = _as_float_array(reference)
    current_array = _as_float_array(current)
    combined = np.concatenate([reference_array, current_array])
    if combined.size == 0:
        return np.array([1.0]), np.array([1.0]), np.array([0.0, 1.0])
    if np.all(combined == combined[0]):
        value = float(combined[0])
        return np.array([1.0]), np.array([1.0]), np.array([value - 0.5, value + 0.5])
    bin_count = min(bins, max(2, int(np.sqrt(combined.size))))
    edges = np.histogram_bin_edges(combined, bins=bin_count)
    if len(np.unique(edges)) < 2:
        edges = np.linspace(combined.min(), combined.max(), num=3)
    ref_counts, _ = np.histogram(reference_array, bins=edges)
    cur_counts, _ = np.histogram(current_array, bins=edges)
    return _normalize_counts(ref_counts), _normalize_counts(cur_counts), edges


def _histogram_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(float)


def chi_square_statistic_from_counts(observed: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0:
        return 0.0
    total = observed.sum()
    if total <= 0.0:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    mask = expected > 0.0
    return float(np.sum(((observed - expected) ** 2)[mask] / expected[mask]))


def g_test_statistic_from_counts(observed: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0:
        return 0.0
    total = observed.sum()
    if total <= 0.0:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    mask = (observed > 0.0) & (expected > 0.0)
    return float(2.0 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask])))


def chi_square_statistic_from_histograms(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    edges: np.ndarray,
) -> float:
    reference_counts = _histogram_counts(_as_float_array(reference), edges)
    current_counts = _histogram_counts(_as_float_array(current), edges)
    return chi_square_statistic_from_counts(np.vstack([reference_counts, current_counts]))


def rankdata(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=float)
    index = 0
    while index < sorted_values.size:
        end = index + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[index]:
            end += 1
        average_rank = 0.5 * (index + 1 + end)
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def mann_whitney_u_statistic(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    reference_array = _as_float_array(reference)
    current_array = _as_float_array(current)
    if reference_array.size == 0 or current_array.size == 0:
        return 0.0
    combined = np.concatenate([reference_array, current_array])
    ranks = rankdata(combined)
    u_value = np.sum(ranks[: reference_array.size]) - reference_array.size * (reference_array.size + 1) / 2.0
    return float(u_value)


def mann_whitney_pvalue(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    reference_array = _as_float_array(reference)
    current_array = _as_float_array(current)
    n_reference = reference_array.size
    n_current = current_array.size
    if n_reference == 0 or n_current == 0:
        return 1.0
    combined = np.concatenate([reference_array, current_array])
    u_value = mann_whitney_u_statistic(reference_array, current_array)
    mean_u = n_reference * n_current / 2.0
    _, counts = np.unique(combined, return_counts=True)
    tie_correction = np.sum(counts**3 - counts)
    total = n_reference + n_current
    variance = n_reference * n_current / 12.0 * (total + 1.0 - tie_correction / (total * (total - 1.0))) if total > 1 else 0.0
    if variance <= 0.0:
        return 1.0
    z_score = (u_value - mean_u - 0.5 * np.sign(u_value - mean_u)) / sqrt(variance)
    return float(np.clip(2.0 * normal_sf(abs(z_score)), 0.0, 1.0))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def normal_sf(value: float) -> float:
    return 0.5 * erfc(value / sqrt(2.0))


def erf(value: float) -> float:
    # math.erf exists, but keeping the wrapper centralizes numeric behavior.
    from math import erf as _erf

    return _erf(value)


def permutation_p_value(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    iterations: int,
    random_state: int,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> Tuple[float, float]:
    reference_array = sample_array(np.asarray(reference), min(sample_size, np.asarray(reference).size), random_state)
    current_array = sample_array(np.asarray(current), min(sample_size, np.asarray(current).size), random_state + 1)
    if reference_array.size == 0 or current_array.size == 0:
        return 1.0, 0.0
    observed = float(statistic_fn(reference_array, current_array))
    combined = np.concatenate([reference_array, current_array])
    n_reference = reference_array.size
    rng = np.random.default_rng(random_state)
    extreme = 0
    total_iterations = max(1, int(iterations))
    for _ in range(total_iterations):
        permuted = rng.permutation(combined)
        statistic = float(statistic_fn(permuted[:n_reference], permuted[n_reference:]))
        if statistic >= observed - EPS:
            extreme += 1
    p_value = (extreme + 1.0) / (total_iterations + 1.0)
    return float(np.clip(p_value, 0.0, 1.0)), observed


def category_counts(values: Sequence[object] | np.ndarray, categories: Sequence[object]) -> np.ndarray:
    values_array = np.asarray(values, dtype=object)
    if "__OTHER__" in categories:
        known = set(categories)
        known.discard("__OTHER__")
        mapped = np.array([value if value in known else "__OTHER__" for value in values_array], dtype=object)
    else:
        mapped = values_array
    categorical = pd.Categorical(mapped, categories=categories)
    codes = categorical.codes
    codes = codes[codes >= 0]
    if len(categories) == 0:
        return np.array([], dtype=float)
    return np.bincount(codes, minlength=len(categories)).astype(float)


def chi_square_statistic_from_labels(
    reference: Sequence[object] | np.ndarray,
    current: Sequence[object] | np.ndarray,
    categories: Sequence[object],
) -> float:
    reference_counts = category_counts(reference, categories)
    current_counts = category_counts(current, categories)
    return chi_square_statistic_from_counts(np.vstack([reference_counts, current_counts]))


def g_test_statistic_from_labels(
    reference: Sequence[object] | np.ndarray,
    current: Sequence[object] | np.ndarray,
    categories: Sequence[object],
) -> float:
    reference_counts = category_counts(reference, categories)
    current_counts = category_counts(current, categories)
    return g_test_statistic_from_counts(np.vstack([reference_counts, current_counts]))


def mutual_information(x: Sequence[object] | np.ndarray, y: Sequence[object] | np.ndarray) -> float:
    x_array = np.asarray(x, dtype=object)
    y_array = np.asarray(y, dtype=object)
    if x_array.size == 0 or y_array.size == 0:
        return 0.0
    if x_array.size != y_array.size:
        raise ValueError("Mutual information requires paired sequences of equal length.")
    _, x_codes = np.unique(x_array, return_inverse=True)
    _, y_codes = np.unique(y_array, return_inverse=True)
    joint = np.zeros((x_codes.max() + 1, y_codes.max() + 1), dtype=float)
    np.add.at(joint, (x_codes, y_codes), 1.0)
    joint = joint / joint.sum()
    x_marginal = joint.sum(axis=1, keepdims=True)
    y_marginal = joint.sum(axis=0, keepdims=True)
    expected = x_marginal @ y_marginal
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / np.clip(expected[mask], EPS, None))))


def variance_ratio(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    reference_array = _as_float_array(reference)
    current_array = _as_float_array(current)
    if reference_array.size < 2 and current_array.size < 2:
        return 1.0
    reference_variance = float(np.var(reference_array, ddof=1)) if reference_array.size > 1 else 0.0
    current_variance = float(np.var(current_array, ddof=1)) if current_array.size > 1 else 0.0
    if reference_variance == 0.0 and current_variance == 0.0:
        return 1.0
    if current_variance == 0.0:
        return float("inf")
    return float(reference_variance / current_variance)


def variance_ratio_statistic(reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray) -> float:
    ratio = variance_ratio(reference, current)
    if ratio <= 0.0 or not np.isfinite(ratio):
        return float("inf")
    return float(abs(log(ratio)))
