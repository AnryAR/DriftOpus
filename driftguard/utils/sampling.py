from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def sample_series(series: pd.Series, max_samples: int, random_state: int) -> pd.Series:
    if len(series) <= max_samples:
        return series
    return series.sample(n=max_samples, random_state=random_state, replace=False)


def series_to_numeric_array(series: pd.Series, max_samples: int, random_state: int) -> np.ndarray:
    sampled = sample_series(series, max_samples, random_state)
    numeric = pd.to_numeric(sampled, errors="coerce").dropna()
    return numeric.astype(float).to_numpy()


def series_to_categorical_array(series: pd.Series, max_samples: int, random_state: int) -> np.ndarray:
    sampled = sample_series(series, max_samples, random_state).dropna()
    return sampled.astype(str).to_numpy()

