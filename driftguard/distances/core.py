from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..statistics.pure import (
    anderson_darling_statistic,
    bhattacharyya_distance as _bhattacharyya_distance,
    cramervonmises_statistic,
    entropy as _entropy,
    hellinger_distance as _hellinger_distance,
    js_divergence as _js_divergence,
    kl_divergence as _kl_divergence,
    ks_statistic,
    normalized_entropy as _normalized_entropy,
    psi_from_probabilities as _psi_from_probabilities,
    sample_array,
    wasserstein_distance as _wasserstein_distance,
)

EPS = 1e-12


def _normalize(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    counts = np.clip(counts, 0.0, None) + EPS
    return counts / counts.sum()


def categorical_probability_vectors(
    reference: Sequence[str],
    current: Sequence[str],
    top_k: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ref_series = pd.Series(reference, dtype="object")
    cur_series = pd.Series(current, dtype="object")
    ref_counts = ref_series.value_counts()
    cur_counts = cur_series.value_counts()
    combined = ref_counts.add(cur_counts, fill_value=0).sort_values(ascending=False)
    if top_k is not None and len(combined) > top_k:
        keep = list(combined.index[:top_k])
    else:
        keep = list(combined.index)
    ref_filtered = ref_series.where(ref_series.isin(keep), "__OTHER__")
    cur_filtered = cur_series.where(cur_series.isin(keep), "__OTHER__")
    categories = list(pd.Index(ref_filtered.unique()).union(cur_filtered.unique()))
    ref_probs = ref_filtered.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    cur_probs = cur_filtered.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    return _normalize(ref_probs), _normalize(cur_probs), categories


def align_numeric_distributions(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)
    reference_array = reference_array[np.isfinite(reference_array)]
    current_array = current_array[np.isfinite(current_array)]
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
    return _normalize(ref_counts), _normalize(cur_counts), edges


def kl_divergence(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    return _kl_divergence(reference_probabilities, current_probabilities)


def js_divergence(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    return _js_divergence(reference_probabilities, current_probabilities)


def hellinger_distance(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    return _hellinger_distance(reference_probabilities, current_probabilities)


def bhattacharyya_distance(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    return _bhattacharyya_distance(reference_probabilities, current_probabilities)


def psi_from_probabilities(reference_probabilities: np.ndarray, current_probabilities: np.ndarray) -> float:
    return _psi_from_probabilities(reference_probabilities, current_probabilities)


def normalized_entropy(probabilities: np.ndarray) -> float:
    return _normalized_entropy(probabilities)


def entropy(probabilities: np.ndarray) -> float:
    return _entropy(probabilities)


def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
    return _wasserstein_distance(reference, current)


def mmd_rbf(reference: np.ndarray, current: np.ndarray, gamma: float | None = None, max_points: int = 500) -> float:
    reference_array = sample_array(np.asarray(reference, dtype=float), min(max_points, np.asarray(reference, dtype=float).size), 42)
    current_array = sample_array(np.asarray(current, dtype=float), min(max_points, np.asarray(current, dtype=float).size), 43)
    reference_array = reference_array[np.isfinite(reference_array)]
    current_array = current_array[np.isfinite(current_array)]
    if reference_array.size == 0 or current_array.size == 0:
        return 0.0
    reference_array = reference_array.reshape(-1, 1)
    current_array = current_array.reshape(-1, 1)
    if gamma is None:
        combined = np.vstack([reference_array, current_array]).ravel()
        distances = np.abs(combined[:, None] - combined[None, :])
        positive = distances[distances > 0.0]
        median = np.median(positive) if positive.size else 1.0
        gamma = 1.0 if not np.isfinite(median) or median <= 0.0 else 1.0 / (2.0 * median**2)

    def _kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * np.square(x - y.T))

    k_xx = _kernel(reference_array, reference_array)
    k_yy = _kernel(current_array, current_array)
    k_xy = _kernel(reference_array, current_array)
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def bootstrap_ks_confidence_interval(
    reference: np.ndarray,
    current: np.ndarray,
    iterations: int,
    random_state: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)
    reference_array = reference_array[np.isfinite(reference_array)]
    current_array = current_array[np.isfinite(current_array)]
    observed = float(ks_statistic(reference_array, current_array))
    if reference_array.size == 0 or current_array.size == 0:
        return 0.0, 0.0, observed
    rng = np.random.default_rng(random_state)
    bootstrapped: List[float] = []
    for _ in range(max(1, iterations)):
        ref_sample = rng.choice(reference_array, size=reference_array.size, replace=True)
        cur_sample = rng.choice(current_array, size=current_array.size, replace=True)
        bootstrapped.append(float(ks_statistic(ref_sample, cur_sample)))
    alpha = 1.0 - confidence
    lower = float(np.quantile(bootstrapped, alpha / 2.0))
    upper = float(np.quantile(bootstrapped, 1.0 - alpha / 2.0))
    return lower, upper, observed


__all__ = [
    "align_numeric_distributions",
    "bhattacharyya_distance",
    "bootstrap_ks_confidence_interval",
    "categorical_probability_vectors",
    "entropy",
    "hellinger_distance",
    "js_divergence",
    "kl_divergence",
    "mmd_rbf",
    "normalized_entropy",
    "psi_from_probabilities",
    "wasserstein_distance",
]
