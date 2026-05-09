from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def to_native(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_native(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value

