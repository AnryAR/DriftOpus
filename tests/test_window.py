from __future__ import annotations

import builtins
from pathlib import Path

from driftguard import DriftAnalyzer
from driftguard.reporting.window import launch_window_report


def test_window_report_falls_back_to_browser(reference_df, current_df, tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_open(url: str) -> bool:
        captured["url"] = url
        return True

    monkeypatch.setattr("webbrowser.open", fake_open)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if str(name).startswith("PyQt5"):
            raise ImportError("PyQt5 disabled for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = DriftAnalyzer(reference_df, current_df).run()
    html_path = launch_window_report(report, reference_df, current_df, tmp_path / "window_report.html")
    assert Path(html_path).exists()
    assert captured["url"].startswith("file:")

