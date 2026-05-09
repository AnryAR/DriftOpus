from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.core.registry import clear_registered_tests


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registered_tests()
    yield
    clear_registered_tests()


@pytest.fixture()
def reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25],
            "salary": [100, 110, 120, 130, 140, 150],
            "city": ["a", "a", "b", "b", "c", "c"],
            "is_vip": [True, False, False, True, False, False],
            "signup_date": pd.date_range("2024-01-01", periods=6, freq="D"),
        }
    )


@pytest.fixture()
def current_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 25, 26, 27],
            "salary": [100, 200, 300, 150, 180, 190],
            "city": ["a", "b", "c", "d", "d", "d"],
            "is_vip": [False, False, True, True, True, False],
            "signup_date": pd.date_range("2024-02-01", periods=6, freq="D"),
        }
    )


@pytest.fixture()
def null_current_df(current_df: pd.DataFrame) -> pd.DataFrame:
    df = current_df.copy()
    df.loc[0, "salary"] = None
    return df
