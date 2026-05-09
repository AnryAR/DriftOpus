from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .analyzer import DriftAnalyzer


def _infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        series = converted[column]
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            parsed = pd.to_datetime(series, errors="coerce", utc=False)
            non_null = series.notna().sum()
            if non_null > 0 and parsed.notna().sum() / non_null >= 0.9:
                converted[column] = parsed
    return converted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driftguard", description="DriftGuard dataset drift analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run a drift analysis on two CSV files")
    analyze.add_argument("--reference", required=True, help="Path to the reference CSV")
    analyze.add_argument("--current", required=True, help="Path to the current CSV")
    analyze.add_argument("--columns", default="", help="Comma-separated list of columns to analyze")
    analyze.add_argument("--report", default="json", choices=["json", "csv", "html", "pdf", "window"], help="Report mode")
    analyze.add_argument("--output", default="", help="Output file path")
    analyze.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    analyze.add_argument("--bins", type=int, default=10, help="Numeric bin count")
    analyze.add_argument("--max-samples", type=int, default=100000, help="Maximum samples per column")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "analyze":
        parser.error("Only the analyze command is supported.")

    reference_df = _infer_datetime_columns(pd.read_csv(args.reference))
    current_df = _infer_datetime_columns(pd.read_csv(args.current))
    columns = [column.strip() for column in args.columns.split(",") if column.strip()] or None
    analyzer = DriftAnalyzer(
        reference_df=reference_df,
        current_df=current_df,
        bins=args.bins,
        max_samples_per_column=args.max_samples,
        n_jobs=args.n_jobs,
    )
    output_path = Path(args.output) if args.output else None
    report = analyzer.run(columns=columns, report_mode=args.report, output_path=output_path)
    print(f"overall_score={report.overall_score:.4f}")
    print(f"severity={report.severity.value}")
    if output_path is not None:
        print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
