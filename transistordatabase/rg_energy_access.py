"""Read-only repository access to bounded gate-resistance energy helpers."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re

import numpy as np

from transistordatabase.channel_models import real_graph
from transistordatabase.condition_tables import samples
from transistordatabase.rg_formula import (
    find_rg_for_target_energy, interpolate_energy_at_rg,
)
from transistordatabase.static_losses import finite

NUMERIC_CONDITIONS = (
    "v_supply", "v_g", "v_g_off", "load_inductance", "commutation_inductance",
)
TEXT_CONDITIONS = ("commutation_device", "convention", "event", "method")
MAX_CURVES = 1000
MAX_SAMPLES = 200000
MAX_CURVE_SAMPLES = 20000
MAX_OUTPUT_BYTES = 24 * 1024 * 1024
PREVIEW_SAMPLES = 200


def _number(value, name, nonnegative=False):
    """Reject boolean, missing, non-finite and invalid physical magnitudes."""
    number = finite(value, name)
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _text(value, name):
    """Preserve exact source text, including the literal method unknown."""
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"{name} needs nonblank text of at most 2000 characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} needs valid UTF-8 scalar text") from exc
    return value


def _conditions(value):
    """Require all fixed conditions. Null selects recorded unknown only."""
    names = set(NUMERIC_CONDITIONS + TEXT_CONDITIONS)
    if not isinstance(value, Mapping) or set(value) != names:
        raise ValueError("Declare all nine switching energy conditions. Use null for recorded unknown values")
    result = {}
    for name in NUMERIC_CONDITIONS:
        item = value[name]
        result[name] = (None if item is None else _number(
            item, name, name in {"load_inductance", "commutation_inductance"}))
    for name in TEXT_CONDITIONS:
        item = value[name]
        result[name] = None if item is None else _text(item, name)
    return result


def _records(device, branch):
    """Return the complete branch without suppressing competing records."""
    if not isinstance(branch, str) or branch not in {"e_on", "e_off", "e_rr"}:
        raise ValueError("Select e_on, e_off or e_rr as the switching branch")
    component = device.diode if branch == "e_rr" else device.switch
    return getattr(component, f"{branch}_data")


def _bounded_json(value):
    """Check actual encoded response size and JSON numerical validity."""
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("Rg-energy output exceeds the 24 MiB limit")
    return value


def _source(device_id, branch, index, record, snapshot):
    """Describe repository coordinates and available source evidence."""
    representation = record.dataset_type
    graph = getattr(record, representation)
    graph = real_graph(graph, representation)
    axis, energy = samples(graph[0], graph[1], representation)
    if np.any(axis < 0) or np.any(energy < 0):
        raise ValueError("Resistance, current and switching energy must be nonnegative")
    evidence = record.evidence
    conditions = {name: getattr(record, name) for name in NUMERIC_CONDITIONS}
    conditions["commutation_device"] = record.commutation_device
    conditions.update({name: getattr(evidence, name, None)
                       for name in ("convention", "event", "method")})
    conditions = _conditions(conditions)
    temperature = _number(record.t_j, "recorded t_j")
    unavailable = []
    diagnostics = []
    def optional_scalar(name, required):
        value = getattr(record, name)
        try:
            if value is None and not required:
                return None
            return _number(value, f"recorded {name}", True)
        except ValueError as exc:
            if required:
                raise
            unavailable.append(name)
            diagnostics.append(str(exc))
            return None

    current = optional_scalar("i_x", representation == "graph_r_e")
    resistance = optional_scalar("r_g", representation == "graph_i_e")
    source_filename = getattr(evidence, "filename", None)
    if source_filename is not None:
        try:
            _text(source_filename, "source filename")
        except ValueError as exc:
            source_filename = None
            unavailable.append("source_filename")
            diagnostics.append(str(exc))
    source_sha = getattr(evidence, "sha256", None)
    if source_sha is not None and (not isinstance(source_sha, str)
                                  or re.fullmatch(r"[0-9a-fA-F]{64}", source_sha) is None):
        source_sha = None
        unavailable.append("source_sha256")
        diagnostics.append("A recorded source SHA-256 must have 64 hexadecimal characters")
    dependency = {"representation": representation, "t_j": temperature,
                  "i_x": current, "r_g": resistance, "conditions": conditions,
                  "axis": axis.tolist(), "energy_j": energy.tolist()}
    digest = hashlib.sha256(json.dumps(
        dependency, sort_keys=True, allow_nan=False, ensure_ascii=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    positions = np.unique(np.linspace(0, len(axis) - 1,
                                     min(len(axis), PREVIEW_SAMPLES), dtype=int))
    component = "diode" if branch == "e_rr" else "switch"
    source = {
        "source_id": f"{device_id}:{component}.{branch}_data[{index}]",
        "source_kind": ("measurement-evidence" if source_filename or source_sha
                        else "repository-locator"),
        "branch": branch, "index": index, "representation": representation,
        "t_j": temperature, "i_x": current, "r_g": resistance,
        "conditions": conditions, "source_filename": source_filename,
        "source_sha256": source_sha, "snapshot_sha256": digest,
        "status": "available", "diagnostic": "; ".join(diagnostics) or None,
        "unavailable_fields": unavailable,
        "coverage": {"axis": "r_g" if representation == "graph_r_e" else "i_channel",
                     "axis_unit": "ohm" if representation == "graph_r_e" else "A",
                     "lower": float(axis[0]), "upper": float(axis[-1]),
                     "energy_min": float(np.min(energy)),
                     "energy_max": float(np.max(energy)), "sample_count": len(axis)},
        "preview": {"axis": axis[positions].tolist(),
                    "energy_j": energy[positions].tolist(),
                    "truncated": len(axis) > PREVIEW_SAMPLES},
    }
    if snapshot:
        source["snapshot"] = dependency
    return source


def _unavailable_source(device_id, branch, index, record, diagnostic, snapshot):
    """Retain an unusable row without turning invalid evidence into unknown data."""
    unavailable = ["coverage", "preview", "snapshot_sha256", "snapshot"]
    def scalar(name):
        value = getattr(record, name, None)
        if value is None:
            return None
        try:
            return _number(value, f"recorded {name}", name != "t_j")
        except ValueError:
            unavailable.append(name)
            return None

    evidence = record.evidence
    conditions = {name: getattr(record, name) for name in NUMERIC_CONDITIONS}
    conditions["commutation_device"] = record.commutation_device
    conditions.update({name: getattr(evidence, name, None)
                       for name in ("convention", "event", "method")})
    try:
        conditions = _conditions(conditions)
    except ValueError:
        conditions = None
        unavailable.append("conditions")
    filename = getattr(evidence, "filename", None)
    if filename is not None:
        try:
            _text(filename, "source filename")
        except ValueError:
            filename = None
            unavailable.append("source_filename")
    source_sha = getattr(evidence, "sha256", None)
    if source_sha is not None and (not isinstance(source_sha, str)
                                  or re.fullmatch(r"[0-9a-fA-F]{64}", source_sha) is None):
        source_sha = None
        unavailable.append("source_sha256")
    component = "diode" if branch == "e_rr" else "switch"
    source = {
        "source_id": f"{device_id}:{component}.{branch}_data[{index}]",
        "source_kind": "measurement-evidence" if filename or source_sha else "repository-locator",
        "branch": branch, "index": index, "representation": record.dataset_type,
        "t_j": scalar("t_j"), "i_x": scalar("i_x"), "r_g": scalar("r_g"),
        "conditions": conditions, "source_filename": filename,
        "source_sha256": source_sha, "snapshot_sha256": None,
        "coverage": None, "preview": None, "status": "unavailable",
        "diagnostic": diagnostic, "unavailable_fields": unavailable,
    }
    if snapshot:
        source["snapshot"] = None
    return source


def _sample_size(graph):
    """Bound malformed sequence traversal without coercing ragged source data."""
    pending = [graph]
    count = 0
    visited = 0
    while pending:
        value = pending.pop()
        visited += 1
        if isinstance(value, np.ndarray):
            count += value.size
        elif isinstance(value, (list, tuple)):
            if (len(value) > 2 * MAX_CURVE_SAMPLES
                    or len(pending) + len(value) + visited > 4 * MAX_CURVE_SAMPLES):
                return 2 * MAX_CURVE_SAMPLES + 1
            pending.extend(value)
        else:
            count += 1
        if count > 2 * MAX_CURVE_SAMPLES or visited > 4 * MAX_CURVE_SAMPLES:
            return 2 * MAX_CURVE_SAMPLES + 1
    return count


def _source_records(device, branches):
    """Apply resource bounds before the helper scans the full selected branch."""
    selected = []
    count = 0
    for branch in branches:
        for index, record in enumerate(_records(device, branch)):
            if record.dataset_type not in {"graph_r_e", "graph_i_e"}:
                continue
            size = _sample_size(getattr(record, record.dataset_type))
            if size > 2 * MAX_CURVE_SAMPLES:
                raise ValueError("An Rg-energy curve exceeds the 20000 sample limit")
            count += size
            selected.append((branch, index, record))
            if len(selected) > MAX_CURVES or count > 2 * MAX_SAMPLES:
                raise ValueError("Rg-energy sources exceed 1000 curves or 200000 samples")
    return selected


def _sources(device, selected, snapshot=False):
    """Return available dependencies and attributable diagnostics for unusable rows."""
    device_id = _text(device.metadata.name, "device name")
    result = []
    for branch, index, record in selected:
        try:
            source = _source(device_id, branch, index, record, snapshot)
        except (ValueError, TypeError, OverflowError) as exc:
            source = _unavailable_source(device_id, branch, index, record, str(exc), snapshot)
        result.append(source)
    return result


def rg_energy_inventory(device):
    """List source curves without choosing a physical context for the caller."""
    return _bounded_json({"device_id": _text(device.metadata.name, "device name"),
                          "curves": _sources(device, _source_records(
                              device, ("e_on", "e_off", "e_rr")))})


def calculate_rg_energy(device, request):
    """Validate an independent headless request and call unchanged public math."""
    if not isinstance(request, Mapping):
        raise ValueError("Supply an Rg-energy request object")
    operation = request.get("operation")
    target = ({"energy-at-rg": "r_g", "rg-at-energy": "e_target"}.get(operation)
              if isinstance(operation, str) else None)
    required = {"operation", "branch", "representation", "i_channel", "t_j", "conditions"}
    if target is None or set(request) != required | {target}:
        raise ValueError("Select energy-at-rg with r_g or rg-at-energy with e_target")
    representation = request["representation"]
    if not isinstance(representation, str) or representation not in {"graph_r_e", "graph_i_e"}:
        raise ValueError("Select graph_r_e or graph_i_e as the representation")
    query = {"operation": operation, "branch": request["branch"],
             "representation": representation,
             "i_channel": _number(request["i_channel"], "i_channel", True),
             "t_j": _number(request["t_j"], "t_j"),
             "conditions": _conditions(request["conditions"]),
             target: _number(request[target], target, True)}
    records = _records(device, query["branch"])
    selected = _source_records(device, (query["branch"],))
    helper = (interpolate_energy_at_rg if operation == "energy-at-rg"
              else find_rg_for_target_energy)
    value = helper(records, query[target], query["i_channel"], query["t_j"],
                   conditions=query["conditions"], representation=representation)
    sources = _sources(device, selected, snapshot=True)
    return _bounded_json({
        "device_id": _text(device.metadata.name, "device name"),
        "operation": operation, "value": value,
        "unit": "J" if operation == "energy-at-rg" else "ohm",
        "evidence_tag": "bounded-interpolation-estimate", "query": query,
        "conditions": query["conditions"], "source_scope": "considered-branch",
        "sources": sources, "method": "piecewise-linear; no extrapolation",
    })
