from __future__ import annotations

from pathlib import Path

import pandas as pd

from driftguard import DriftAnalyzer


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "sample_data"
    reference_df = pd.read_csv(base_dir / "reference.csv", parse_dates=["signup_date"])
    current_df = pd.read_csv(base_dir / "current.csv", parse_dates=["signup_date"])
    analyzer = DriftAnalyzer(reference_df, current_df, bins=8, max_samples_per_column=100)
    report = analyzer.run(report_mode="html", output_path=Path(__file__).with_name("driftguard_report.html"))
    print(f"overall_score={report.overall_score:.4f}")
    print(f"severity={report.severity.value}")


if __name__ == "__main__":
    main()

