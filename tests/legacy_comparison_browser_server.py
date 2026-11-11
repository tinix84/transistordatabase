"""Serve the production app with read-only in-memory algebraic browser fixtures.

Run with uvicorn legacy_comparison_browser_server:app --app-dir tests --port 8031.
No repository files or stored devices are changed by this fixture server.
"""
from types import SimpleNamespace

from test_legacy_comparison_closure import device

from transistordatabase.gui_web.backend import main

devices = {"bounded-a": device("bounded-a"), "bounded-b": device("bounded-b")}
main._repo = SimpleNamespace(list_all=lambda: list(devices), get_by_name=devices.get)
app = main.app
