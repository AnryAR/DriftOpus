from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Sequence

import pandas as pd

from ..configs.defaults import DriftAnalyzerConfig
from ..models import DriftTestResult

TestKind = Literal["numeric", "categorical", "dataset", "all"]


@dataclass
class CustomDriftTest:
    name: str
    kind: TestKind
    run: Callable[[pd.Series, pd.Series, DriftAnalyzerConfig], DriftTestResult | dict]
    description: str = ""
    weight: float = 1.0


_REGISTERED_TESTS: List[CustomDriftTest] = []


def register_test(custom_test: CustomDriftTest | Callable[[pd.Series, pd.Series, DriftAnalyzerConfig], DriftTestResult | dict]) -> CustomDriftTest:
    if isinstance(custom_test, CustomDriftTest):
        _REGISTERED_TESTS[:] = [test for test in _REGISTERED_TESTS if test.name != custom_test.name]
        _REGISTERED_TESTS.append(custom_test)
        return custom_test
    wrapped = CustomDriftTest(
        name=getattr(custom_test, "__name__", "custom_test"),
        kind="all",
        run=custom_test,
        description=getattr(custom_test, "__doc__", "") or "",
    )
    _REGISTERED_TESTS[:] = [test for test in _REGISTERED_TESTS if test.name != wrapped.name]
    _REGISTERED_TESTS.append(wrapped)
    return wrapped


def get_custom_tests(kind: TestKind) -> List[CustomDriftTest]:
    return [test for test in _REGISTERED_TESTS if test.kind in {kind, "all"}]


def clear_registered_tests() -> None:
    _REGISTERED_TESTS.clear()
