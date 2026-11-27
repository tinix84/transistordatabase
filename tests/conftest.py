"""Pytest configuration for GUI tests."""
import pytest
import os
from importlib.util import find_spec

# PyQt5 is an optional extra; skip the legacy Qt GUI suites when it is absent.
if find_spec("PyQt5") is None:
    collect_ignore_glob = [
        "test_gui_*_mixin.py", "test_gui_imports.py", "test_gui_mainwindow.py",
        "test_gui_memory.py", "test_gui_performance.py",
    ]


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for the entire test session."""
    QApplication = pytest.importorskip("PyQt5.QtWidgets").QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()
