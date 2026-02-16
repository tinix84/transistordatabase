"""Pytest configuration for GUI tests."""
import pytest
import os
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()
