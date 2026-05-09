from __future__ import annotations

import pandas as pd

from driftguard.configs.defaults import DriftAnalyzerConfig
from driftguard.models import DriftTestResult
from driftguard.statistics.categorical import run_categorical_tests


def test_categorical_tests_detect_unseen_categories() -> None:
    reference = pd.Series(["a", "a", "b", "b", "c", "c"])
    current = pd.Series(["a", "b", "c", "d", "d", "d"])
    results = run_categorical_tests(reference, current, DriftAnalyzerConfig(max_samples_per_column=100))
    assert "chi_square_test" in results
    assert "g_test" in results
    assert "new_unseen_category_detection" in results
    assert all(isinstance(result, DriftTestResult) for result in results.values())
    assert results["new_unseen_category_detection"].triggered is True
    assert results["new_unseen_category_detection"].details["unseen_categories"] == ["d"]

