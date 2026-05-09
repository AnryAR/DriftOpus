from __future__ import annotations

import pandas as pd
import pytest

from driftguard.validators import check_nulls, column_role, find_duplicate_columns, is_datetime_series, resolve_analysis_columns, validate_dataframes


def test_resolve_analysis_columns_ignores_datetime_and_reports_exclusions(reference_df, current_df) -> None:
    selected, excluded, issues = resolve_analysis_columns(reference_df, current_df, columns=["age", "signup_date", "city"])
    assert "age" in selected
    assert "city" in selected
    assert "signup_date" not in selected
    assert "signup_date" in excluded
    assert "datetime" in excluded["signup_date"]
    assert issues == []


def test_column_role_and_datetime_detection(reference_df) -> None:
    assert column_role(reference_df["age"]) == "numeric"
    assert column_role(reference_df["city"]) == "categorical"
    assert column_role(reference_df["is_vip"]) == "categorical"
    assert is_datetime_series(reference_df["signup_date"]) is True


def test_find_duplicate_columns_and_validation_errors() -> None:
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    assert find_duplicate_columns(df) == ["a"]
    with pytest.raises(ValueError, match="Duplicate columns detected in reference_df"):
        validate_dataframes(df, pd.DataFrame({"a": [1]}))


def test_check_nulls_raises_clear_message(null_current_df) -> None:
    with pytest.raises(ValueError, match=r"current_df\['salary'\] -> 1 nulls found"):
        check_nulls(null_current_df, ["salary"], "current_df")

