# DriftGuard

DriftGuard is a Python library for dataset drift detection, statistical validation,
distribution comparison, and reporting.

## Features

- Numeric and categorical drift tests
- Schema validation and null checking
- Dataset and column-level scoring
- HTML, PDF, JSON, CSV, and interactive window reports
- CLI support
- Custom test registration
- Advanced helpers for target leakage drift, MMD, Hellinger, Bhattacharyya, and PCA/correlation monitoring

## Core API

```python
from driftguard import DriftAnalyzer

analyzer = DriftAnalyzer(reference_df=train_df, current_df=prod_df)
report = analyzer.run()
```

Column subsets are supported:

```python
report = analyzer.run(columns=["age", "salary", "city"])
```

Report modes:

- `html` writes a standalone interactive report
- `pdf` writes a multi-page PDF
- `json` writes the structured result object
- `csv` writes a compact summary
- `window` opens the HTML report in a desktop window

## Custom Tests

```python
from driftguard import CustomDriftTest, DriftTestResult, register_test

def my_test(reference, current, config):
    return DriftTestResult(
        test_name="my_test",
        score=0.8,
        triggered=True,
        interpretation="Custom drift signal detected.",
    )

register_test(CustomDriftTest(name="my_test", kind="numeric", run=my_test))
```

## Quick Start

```python
import pandas as pd
from driftguard import DriftAnalyzer

reference_df = pd.read_csv("train.csv")
current_df = pd.read_csv("prod.csv")

analyzer = DriftAnalyzer(reference_df=reference_df, current_df=current_df)
report = analyzer.run()
print(report.overall_score, report.severity)
```

## CLI

```bash
driftguard analyze --reference train.csv --current prod.csv --report pdf --output report.pdf
```

If the input CSV has date-like columns, the CLI will infer them before analysis so
datetime features are excluded consistently with the in-memory API.

## Benchmark

```bash
python benchmarks/benchmark_driftguard.py
```

## Report Modes

- `html`: Standalone interactive HTML
- `window`: Interactive desktop window using PyQt5 (`pip install driftguard[window]`)
- `pdf`: Multi-page PDF export
- `json`: JSON output
- `csv`: CSV summary
