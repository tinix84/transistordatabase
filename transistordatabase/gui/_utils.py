"""Shared utility functions for the GUI package."""
import os
import sys


def resource_path(relative_path):
    """Bugfix method, fixing some old issue with the filepath."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
