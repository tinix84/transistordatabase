"""Serve compiled production UI with private, read-only native export fixtures.

Run uvicorn native_export_browser_server:app --app-dir tests --port 8003.
The fixtures do not create or change any stored device records.
"""
from copy import deepcopy
from types import SimpleNamespace

from test_public_export_independent_review import part
from transistordatabase.gui_web.backend import main

first = part()
first.metadata.name = "Browser export 雪"
first.switch.metadata["source_id"] = 2**60 + 1
second = deepcopy(first)
second.metadata.name = "Browser export second"
second.thermal_properties.r_th_cs = .2
devices = {device.metadata.name: device for device in (first, second)}
main._repo = SimpleNamespace(list_all=lambda: list(devices), get_by_name=devices.get)
app = main.app
