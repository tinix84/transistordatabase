"""Headless MATLAB MAT5 and exactly conditioned GeckoCIRCUITS SCL exports.

These are data-format exports. Simulator execution is outside this contract.
The SCL grammar follows the existing exporter in transistor.py. Its fabricated
zero losses, mirrored channel branches, and nearest conditions are excluded.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import savemat

from transistordatabase.channel_models import channel_records, real_graph
from transistordatabase.condition_tables import bounded_interpolate, samples
from transistordatabase.core.adapters import core_to_legacy_dicts
from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.static_losses import _energy_record, finite


def _name(transistor):
    """Use one safe MATLAB variable and file stem without interpreting paths."""
    name = transistor.metadata.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A device name is required")
    stem = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not stem[0].isalpha():
        stem = "device_" + stem
    if len(stem) > 63:
        raise ValueError("The normalized device name exceeds 63 MATLAB identifier characters")
    if stem.lower() in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                        *(f"lpt{i}" for i in range(1, 10)), "core_json"}:
        stem = "device_" + stem
    return stem


def _mat_value(value):
    """Use the historical MATLAB unknown=NaN convention in the numeric view."""
    if value is None:
        return np.nan
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _mat_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mat_value(item) for item in value]
    return value


def render_matlab(transistor):
    """Return a MAT5 numeric legacy view and complete versioned core JSON.

    This is the transistor database dictionary format, not the SC scData format
    or a Simulink loss model. core_json preserves nulls, array shapes and evidence.
    """
    document = core_to_json_document(transistor)
    top, switch, diode = core_to_legacy_dicts(transistor)
    # The adapter requires constructor placeholders. They are not source evidence.
    top["r_g_int"] = None
    for component, view in ((transistor.switch, switch), (transistor.diode, diode)):
        if component.thermal_foster is None:
            view["thermal_foster"]["r_th_total"] = None
    diode["t_j_max"] = transistor.diode.metadata.get("t_j_max")
    top.update(switch=switch, diode=diode)
    stream = BytesIO()
    stem = _name(transistor)
    savemat(stream, {stem: _mat_value(top),
                     "core_json": json.dumps(document, allow_nan=False, ensure_ascii=True)},
            long_field_names=True, do_compression=True)
    return {f"{stem}_Matlab.mat": stream.getvalue()}


def _graph(graph, label):
    return real_graph(graph, label)


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def render_gecko(transistor, selection):
    """Render one measured switch slice with explicit gate/Rg/Tj/supply conditions.

    v_g_on is the recorded on-gate voltage for both energy events. v_g_off is
    their separate recorded off-gate voltage, as in the core static-loss engine.
    No forward-recovery evidence exists in the core diode model for SCL export.
    """
    selection = dict(selection)
    required = {"component", "t_j", "v_supply", "v_g_on", "v_g_off", "r_g_on", "r_g_off"}
    allowed = required | {"output_path", "recheck"}
    if set(selection) - allowed or not required <= selection.keys():
        raise ValueError("Declare component, t_j, v_supply, v_g_on, v_g_off, r_g_on and r_g_off")
    if selection.get("recheck", False) is not False:
        raise ValueError("Nearest-condition recheck is unsupported. Select exact recorded conditions")
    if selection["component"] != "switch":
        raise ValueError("Gecko export supports switch slices only. Diode forward-recovery energy is unavailable")
    t_j = finite(selection["t_j"], "t_j")
    supply = finite(selection["v_supply"], "v_supply")
    selection["t_j"], selection["v_supply"] = t_j, supply
    if supply <= 0:
        raise ValueError("v_supply must be positive")
    for key in ("v_g_on", "v_g_off", "r_g_on", "r_g_off"):
        if selection[key] is not None:
            selection[key] = finite(selection[key], key)
    for key in ("r_g_on", "r_g_off"):
        if selection[key] is not None and selection[key] < 0:
            raise ValueError("Gate resistances must be nonnegative or explicitly unknown")
    channels = channel_records(transistor.switch.channel_data, t_j, selection["v_g_on"])
    if len(channels) != 1:
        raise ValueError("Need one unambiguous channel record at exact Tj and gate voltage")
    graph = _graph(channels[0].graph_v_i, "Channel V(I)")
    current, voltage = samples(graph[1], graph[0], "Channel V(I)")
    if np.any((current < 0) & (voltage > 0)) or np.any((current > 0) & (voltage < 0)):
        raise ValueError("Gecko channel samples must describe passive conduction")
    energies = []
    for records, rg in ((transistor.switch.e_on_data, "r_g_on"),
                        (transistor.switch.e_off_data, "r_g_off")):
        conditions = {"t_j": t_j, "v_supply": supply, "v_g": selection["v_g_on"],
                      "v_g_off": selection["v_g_off"], "r_g": selection[rg], "i_channel": 0}
        (record, _), _, _ = _energy_record(records, conditions)
        if record.dataset_type != "graph_i_e":
            raise ValueError("Gecko export requires measured energy-current curves, not single points")
        graph = _graph(record.graph_i_e, "Switching E(I)")
        x, y = samples(graph[0], graph[1], "Switching E(I)")
        if np.any(x < 0) or np.any(y < 0):
            raise ValueError("Gecko switching currents and energies must be nonnegative")
        energies.append((x, y))
    lower = max(x[0] for x, _ in energies)
    upper = min(x[-1] for x, _ in energies)
    if lower >= upper:
        raise ValueError("Turn-on and turn-off curves need a shared nonzero current interval")
    grid = np.unique(np.concatenate(([lower, upper], *(x[(x >= lower) & (x <= upper)] for x, _ in energies))))
    on, off = [[bounded_interpolate(x, y, point, "Switching E(I)") for point in grid]
               for x, y in energies]
    text = ("anzMesskurvenPvCOND 1\n<LeitverlusteMesskurve>\n"
            f"data[][] 2 {len(current)} {_numbers(voltage)} {_numbers(current)}\n"
            f"tj {t_j:.17g}\n<\\LeitverlusteMesskurve>\n"
            "anzMesskurvenPvSWITCH 1\n<SchaltverlusteMesskurve>\n"
            f"data[][] 3 {len(grid)} {_numbers(grid)} {_numbers(on)} {_numbers(off)}\n"
            f"tj {t_j:.17g}\nuBlock {supply:.17g}\n<\\SchaltverlusteMesskurve>\n")
    stem = _name(transistor)
    filename = f"{stem}_Switch.scl"
    manifest = {"schema": "transistordatabase-gecko-export", "version": 1,
                "file": filename, "selection": {key: selection[key] for key in sorted(required)},
                "channel_current_coverage_a": [float(current[0]), float(current[-1])],
                "switching_current_coverage_a": [float(lower), float(upper)],
                "scope": "Recorded switch slice. No mirrored conduction or inferred losses. Simulator acceptance is not verified.",
                "gate_convention": "v_g_on matches core v_g for both events; v_g_off matches the separate core v_g_off",
                "core_document": core_to_json_document(transistor)}
    return {filename: text.encode("utf-8"),
            f"{stem}_Switch.manifest.json": json.dumps(manifest, allow_nan=False, ensure_ascii=True, indent=2).encode("utf-8")}


def write_export_files(output_path, files):
    """Write pre-rendered bytes at explicit paths. Refuse every existing target."""
    directory = Path(output_path).resolve()
    targets = [(directory / name, data) for name, data in files.items()]
    if any(path.parent != directory for path, _ in targets):
        raise ValueError("Export names must be plain filenames")
    for path, _ in targets:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Export target already exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    # Reserve all names before writing. Exclusive mode also protects racing callers.
    # Files remain recoverable if filesystem I/O fails. No deletion or replacement occurs.
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = [(stack.enter_context(path.open("xb")), data) for path, data in targets]
        for stream, data in streams:
            stream.write(data)
    return directory
