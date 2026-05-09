from __future__ import annotations

import pandas as pd

from driftguard.configs.defaults import DriftAnalyzerConfig
from driftguard.distances.advanced import target_leakage_drift_check
from driftguard.core.registry import CustomDriftTest, register_test
from driftguard.models import DriftTestResult
from driftguard.statistics.numeric import run_numeric_tests


def test_numeric_tests_return_expected_results() -> None:
    reference = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    current = pd.Series([1, 2, 2, 3, 8, 9, 10, 11])
    results = run_numeric_tests(reference, current, DriftAnalyzerConfig(max_samples_per_column=100, bootstrap_iterations=32))
    assert "ks_test" in results
    assert "psi" in results
    assert "mmd_rbf" in results
    assert "ks_bootstrap_ci" in results
    assert "hellinger_distance" in results
    assert "bhattacharyya_distance" in results
    assert all(isinstance(result, DriftTestResult) for result in results.values())
    assert 0.0 <= results["ks_test"].score <= 1.0
    assert "bootstrap_ci" in results["ks_test"].details


def test_custom_numeric_test_registration_is_included() -> None:
    def custom_numeric_test(reference: pd.Series, current: pd.Series, config: DriftAnalyzerConfig) -> DriftTestResult:
        return DriftTestResult(
            test_name="custom_numeric_test",
            statistic=1.0,
            score=0.9,
            triggered=True,
            interpretation="Custom numeric drift test triggered.",
        )

    register_test(CustomDriftTest(name="custom_numeric_test", kind="numeric", run=custom_numeric_test))
    results = run_numeric_tests(
        pd.Series([1, 2, 3, 4]),
        pd.Series([1, 1, 1, 1]),
        DriftAnalyzerConfig(max_samples_per_column=50, bootstrap_iterations=16),
    )
    assert "custom_numeric_test" in results
    assert results["custom_numeric_test"].triggered is True


def test_target_leakage_drift_check() -> None:
    feature_reference = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
    feature_current = pd.Series([1, 2, 2, 3, 3, 4, 4, 4])
    target_reference = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    target_current = pd.Series([0, 1, 1, 1, 1, 1, 0, 0])
    result = target_leakage_drift_check(feature_reference, feature_current, target_reference, target_current)
    assert result.test_name == "target_leakage_drift_check"
    assert result.score >= 0.0
    assert "reference_mi" in result.details
