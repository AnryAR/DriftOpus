from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def validate_dataframes(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> None:
    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")
    if not isinstance(current_df, pd.DataFrame):
        raise TypeError("current_df must be a pandas DataFrame.")
    if reference_df.empty:
        raise ValueError("reference_df must not be empty.")
    if current_df.empty:
        raise ValueError("current_df must not be empty.")
    duplicate_reference = find_duplicate_columns(reference_df)
    duplicate_current = find_duplicate_columns(current_df)
    if duplicate_reference:
        raise ValueError(f"Duplicate columns detected in reference_df: {duplicate_reference}")
    if duplicate_current:
        raise ValueError(f"Duplicate columns detected in current_df: {duplicate_current}")


def find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    return list(df.columns[df.columns.duplicated()].unique())


def is_datetime_series(series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series.dtype) or is_datetime64tz_dtype(series.dtype):
        return True
    if is_object_dtype(series.dtype):
        sample = series.dropna().head(10)
        if not sample.empty and all(isinstance(value, (pd.Timestamp, datetime, np.datetime64)) for value in sample):
            return True
    return False


def column_role(series: pd.Series) -> str:
    if is_datetime_series(series):
        return "datetime"
    if is_bool_dtype(series.dtype):
        return "categorical"
    if is_numeric_dtype(series.dtype):
        return "numeric"
    if is_categorical_dtype(series.dtype) or is_string_dtype(series.dtype) or is_object_dtype(series.dtype):
        return "categorical" if not _looks_unsupported(series) else "unsupported"
    return "unsupported"


def _looks_unsupported(series: pd.Series) -> bool:
    sample = series.dropna().head(20)
    for value in sample:
        if isinstance(value, (dict, list, set, frozenset)):
            return True
        try:
            hash(value)
        except Exception:
            return True
    return False


def resolve_analysis_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> Tuple[List[str], Dict[str, str], List[str]]:
    reference_columns = list(reference_df.columns)
    current_columns = list(current_df.columns)
    excluded: Dict[str, str] = {}
    schema_issues: List[str] = []
    reference_datetime = [column for column in reference_columns if is_datetime_series(reference_df[column])]
    current_datetime = [column for column in current_columns if is_datetime_series(current_df[column])]

    for column in reference_datetime:
        excluded[column] = "datetime column in reference_df"
    for column in current_datetime:
        excluded.setdefault(column, "datetime column in current_df")

    if columns is None:
        candidate = [column for column in reference_columns if column in current_columns]
        extra_reference = [column for column in reference_columns if column not in current_columns]
        extra_current = [column for column in current_columns if column not in reference_columns]
        if extra_reference:
            schema_issues.append(
                f"reference_df has columns not present in current_df: {extra_reference}"
            )
        if extra_current:
            schema_issues.append(
                f"current_df has columns not present in reference_df: {extra_current}"
            )
    else:
        candidate = list(dict.fromkeys(columns))
        missing = [column for column in candidate if column not in reference_columns or column not in current_columns]
        if missing:
            raise ValueError(
                f"Requested columns missing from one or both dataframes: {missing}"
            )

    selected: List[str] = []
    for column in candidate:
        if column in excluded:
            continue
        role_reference = column_role(reference_df[column])
        role_current = column_role(current_df[column])
        if "unsupported" in {role_reference, role_current}:
            excluded[column] = "unsupported datatype"
            continue
        selected.append(column)
    return selected, excluded, schema_issues


def check_nulls(df: pd.DataFrame, columns: Sequence[str], df_name: str) -> None:
    null_issues: List[str] = []
    for column in columns:
        null_count = int(df[column].isna().sum())
        if null_count > 0:
            null_issues.append(
                f"{df_name}['{column}'] -> {null_count} nulls found."
            )
    if null_issues:
        message = [
            "Null values detected:",
            *null_issues,
            "Please clean data before running drift analysis.",
        ]
        raise ValueError("\n".join(message))
