from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

import pandas as pd

from ..models import DatasetReport
from .html import generate_html_report


def launch_window_report(
    report: DatasetReport,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> str:
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        output_path = Path(temp_file.name)
        temp_file.close()
    html_path = Path(output_path)
    generate_html_report(report, reference_df, current_df, html_path)
    try:
        from PyQt5.QtCore import QUrl
        from PyQt5.QtWidgets import QApplication, QMainWindow
        from PyQt5.QtWebEngineWidgets import QWebEngineView

        app = QApplication.instance() or QApplication([])
        window = QMainWindow()
        view = QWebEngineView()
        view.load(QUrl.fromLocalFile(str(html_path.resolve())))
        window.setCentralWidget(view)
        window.setWindowTitle(report.title)
        window.resize(1400, 900)
        window.show()
        app.exec_()
    except Exception:
        webbrowser.open(html_path.as_uri())
    return str(html_path)
