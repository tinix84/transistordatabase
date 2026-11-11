"""Serve compiled UI with private, read-only Rg-energy repository fixtures.

Run uvicorn rg_energy_browser_server:app --app-dir tests --port 8003.
This process does not create or change stored device records.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from test_rg_energy_access import part
from transistordatabase.gui_web.backend import main

device = part()
duplicate = deepcopy(device)
duplicate.metadata.name = "Browser Rg duplicate"
duplicate.switch.e_on_data.append(deepcopy(duplicate.switch.e_on_data[0]))
hole = deepcopy(device)
hole.metadata.name = "Browser Rg current hole"
hole.switch.e_on_data.append(replace(
    hole.switch.e_on_data[1], r_g=5,
    graph_i_e=np.array([[15., 20.], [.006, .010]])))
invalid = deepcopy(device)
invalid.metadata.name = "Browser Rg invalid other temperature"
invalid.switch.e_on_data.append(replace(
    invalid.switch.e_on_data[0], t_j=100, graph_r_e=None))
devices = {item.metadata.name: item for item in (device, duplicate, hole, invalid)}
main._repo = SimpleNamespace(list_all=lambda: list(devices), get_by_name=devices.get)
app = main.app
