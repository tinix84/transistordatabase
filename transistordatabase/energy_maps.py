"""Source-preserving energy-current slices and bounded I/V estimates."""
from __future__ import annotations

import bisect
import json
import math

from transistordatabase.condition_tables import MeasuredTable, bounded_interpolate, require_conditions
from transistordatabase.grid_analysis import (
    AnalysisResourceLimitError, MAX_RESPONSE_BYTES, input_hash, snapshot,
)

NUMERIC_CONDITIONS = ("t_j_c", "v_g_on_v", "v_g_off_v", "r_g_on_ohm", "r_g_off_ohm")
TEXT_CONDITIONS = ("commutation_device", "commutation_source", "energy_convention")
METHOD_VERSION = "source-energy-map-v1"


def _conditions(value):
    if set(value) != set(NUMERIC_CONDITIONS + TEXT_CONDITIONS):
        raise ValueError("Declare every fixed condition, with null for unknown values")
    numeric = {name: value[name] for name in NUMERIC_CONDITIONS}
    for name, item in numeric.items():
        if item is not None and (isinstance(item, bool) or
                                 not isinstance(item, (int, float)) or not math.isfinite(item)):
            raise ValueError(f"{name} must be finite or explicitly unknown")
        if item is not None and name.startswith("r_g") and item < 0:
            raise ValueError("Gate resistance must be nonnegative")
    require_conditions(numeric, numeric)
    for name in TEXT_CONDITIONS:
        item = value[name]
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f"{name} must be nonblank or explicitly unknown")
    if value["energy_convention"] is None:
        raise ValueError("Declare a nonblank energy convention")


def _matching(left, right):
    try:
        require_conditions({k: left[k] for k in NUMERIC_CONDITIONS},
                           {k: right[k] for k in NUMERIC_CONDITIONS})
    except ValueError:
        return False
    return all(left[k] == right[k] for k in TEXT_CONDITIONS)


def _axis(values, label, *, positive=False):
    if not values or any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                         not math.isfinite(v) or (v <= 0 if positive else v < 0)
                         for v in values):
        raise ValueError(f"{label} must contain finite {'positive' if positive else 'nonnegative'} values")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique values")
    return sorted(values)


def build_energy_map(request):
    """Keep actual points, exclusions, missing cells, and ambiguity visible.

    Current interpolation uses the common bounded helper. Voltage interpolation
    requires explicit selection and four available displayed grid corners.
    """
    request = snapshot(request)
    slices = request["slices"]
    input_bytes = len(json.dumps(request, allow_nan=False).encode())
    if input_bytes > 4_000_000:
        raise AnalysisResourceLimitError("Energy-map input exceeds 4 MB")
    if (len(slices) > 128 or len(request["currents_a"]) > 256 or
            len(request["voltages_v"]) > 128 or
            len(request["currents_a"]) * len(request["voltages_v"]) > 16384 or
            sum(len(s["current_a"]) for s in slices) > 100000):
        raise AnalysisResourceLimitError("Energy-map source or displayed grid exceeds its size limit")
    if not slices:
        raise ValueError("Declare at least one energy slice")
    kind = request["energy_kind"]
    if kind not in {"on", "off", "rr"}:
        raise ValueError("Select on, off, or rr energy")
    _conditions(request["conditions"])
    currents = _axis(request["currents_a"], "Displayed currents")
    voltages = _axis(request["voltages_v"], "Displayed voltages", positive=True)
    ids, selected, diagnostics, tables = set(), [], [], {}
    for source in slices:
        for field in ("id", "source"):
            if not isinstance(source[field], str) or not source[field].strip():
                raise ValueError(f"Each source slice needs a nonblank {field}")
        if source["id"] in ids:
            raise ValueError("Energy slice IDs must be unique")
        ids.add(source["id"])
        _conditions(source["conditions"])
        axis = _axis(source["current_a"], "Source currents")
        energies = source["energy_j"]
        if len(axis) < 2 or len(axis) != len(energies):
            raise ValueError("Each energy slice needs at least two aligned current-energy pairs")
        if any(isinstance(e, bool) or not isinstance(e, (float, int)) or
               not math.isfinite(e) or e < 0 for e in energies):
            raise ValueError("Source energies must be finite and nonnegative")
        _axis([source["voltage_v"]], "Source voltage", positive=True)
        if source["source_kind"] not in {"measured", "datasheet", "simulated", "manual", "unknown"}:
            raise ValueError("Declare the source kind")
        if source["energy_kind"] not in {"on", "off", "rr"}:
            raise ValueError("Declare the source energy kind")
        reason = ("energy-kind-mismatch" if source["energy_kind"] != kind else
                  "fixed-condition-mismatch" if not _matching(source["conditions"], request["conditions"])
                  else None)
        diagnostics.append({"slice_id": source["id"], "status": "excluded" if reason else "selected",
                            "reason": reason})
        if reason is None:
            selected.append(source)
            # This bounded table container does not change the declared source kind.
            tables[source["id"]] = MeasuredTable(
                "energy", source["current_a"], source["energy_j"],
                {name: source["conditions"][name] for name in NUMERIC_CONDITIONS}, source["source"])

    projected_bytes = (2 * input_bytes + len(currents) * len(voltages) * 600 +
                       sum((len(currents) + 4) * (len(json.dumps({
                           "slice_id": s["id"], "source": s["source"],
                           "source_kind": s["source_kind"]}).encode()) + 600)
                           for s in selected if s["voltage_v"] in voltages))
    if projected_bytes > MAX_RESPONSE_BYTES:
        raise AnalysisResourceLimitError("Conservative energy-map response bound exceeds 24 MB")

    def cell(current, voltage):
        candidates, unsupported = [], []
        for source in selected:
            if source["voltage_v"] != voltage:
                continue
            try:
                table = tables[source["id"]]
                energy = table.evaluate(current, table.conditions)
            except ValueError as exc:
                unsupported.append({"slice_id": source["id"], "reason": str(exc)})
                continue
            mode = "recorded-point" if current in source["current_a"] else "bounded-current-linear"
            candidates.append({"slice_id": source["id"], "source": source["source"],
                               "source_kind": source["source_kind"], "energy_j": energy, "mode": mode})
        status = "available" if len(candidates) == 1 else "ambiguous" if candidates else "missing"
        return {"current_a": current, "voltage_v": voltage, "status": status,
                "energy_j": candidates[0]["energy_j"] if len(candidates) == 1 else None,
                "mode": candidates[0]["mode"] if len(candidates) == 1 else None,
                "candidates": candidates, "unsupported_slices": unsupported}

    cells = [[cell(i, v) for i in currents] for v in voltages]
    estimate = None
    query = request.get("estimate")
    if query is not None:
        estimate = {"status": "unsupported", "energy_j": None,
                    "method": "bounded-bilinear-estimate", "query": query, "corners": []}
        try:
            if query.get("selected") is not True:
                raise ValueError("Select the bilinear estimate explicitly")
            i, v = query["current_a"], query["voltage_v"]
            _axis([i], "Query current")
            _axis([v], "Query voltage", positive=True)
            if len(currents) < 2 or len(voltages) < 2:
                raise ValueError("A bilinear estimate needs two displayed values on each axis")
            if not currents[0] <= i <= currents[-1] or not voltages[0] <= v <= voltages[-1]:
                raise ValueError("The estimate lies outside the displayed grid")
            ix = min(max(bisect.bisect_right(currents, i) - 1, 0), len(currents) - 2)
            vx = min(max(bisect.bisect_right(voltages, v) - 1, 0), len(voltages) - 2)
            corners = [cells[y][x] for y in (vx, vx + 1) for x in (ix, ix + 1)]
            estimate["corners"] = corners
            if any(c["status"] != "available" for c in corners):
                raise ValueError("All four grid corners must have one available source value")
            lower = bounded_interpolate(currents[ix:ix + 2],
                                        [c["energy_j"] for c in corners[:2]], i)
            upper = bounded_interpolate(currents[ix:ix + 2],
                                        [c["energy_j"] for c in corners[2:]], i)
            energy = bounded_interpolate(voltages[vx:vx + 2], [lower, upper], v)
            estimate.update(status="estimated", energy_j=energy, error=None)
        except ValueError as exc:
            estimate["error"] = str(exc)
    result = {"method_version": METHOD_VERSION, "energy_kind": kind,
            "conditions": request["conditions"], "currents_a": currents, "voltages_v": voltages,
            "units": {"current": "A", "voltage": "V", "energy": "J"},
            "cells": cells, "estimate": estimate, "source_slices": slices,
            "slice_diagnostics": diagnostics, "input_sha256": input_hash(request),
            "assumptions": ["Exact fixed conditions, including unknown values",
                            "No voltage scaling or nearest-condition selection",
                            "Bilinear interpolation is an estimate, not a measured surface",
                            "Bilinear interpolation uses displayed corners. Interior source knots are not inserted."]}
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RESPONSE_BYTES:
        raise AnalysisResourceLimitError("Energy-map response exceeds 24 MB")
    return result
