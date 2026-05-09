from __future__ import annotations

import time

import numpy as np
import pandas as pd

from driftguard import DriftAnalyzer


def _make_frame(rows: int, shift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(42 if not shift else 43)
    base = rng.normal(loc=0.0 if not shift else 0.6, scale=1.0, size=rows)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=rows),
            "salary": (base * 12_000 + 80_000).round(2),
            "city": rng.choice(["a", "b", "c", "d"], size=rows, p=[0.4, 0.3, 0.2, 0.1] if not shift else [0.2, 0.2, 0.2, 0.4]),
            "is_vip": rng.choice([True, False], size=rows, p=[0.1, 0.9] if not shift else [0.3, 0.7]),
            "signup_date": pd.date_range("2024-01-01", periods=rows, freq="min"),
        }
    )
    return frame


def main() -> None:
    reference_df = _make_frame(100_000, shift=False)
    current_df = _make_frame(100_000, shift=True)
    analyzer = DriftAnalyzer(reference_df, current_df, bins=12, max_samples_per_column=50_000, n_jobs=2)
    start = time.perf_counter()
    report = analyzer.run()
    elapsed = time.perf_counter() - start
    print(f"rows={len(reference_df)}")
    print(f"overall_score={report.overall_score:.4f}")
    print(f"severity={report.severity.value}")
    print(f"elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()

