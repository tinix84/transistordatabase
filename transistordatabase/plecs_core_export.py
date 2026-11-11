"""Exact, explicitly selected core curves rendered as PLECS table subsets.

No gate slice, temperature, voltage, omitted mechanism or interpolation is chosen
implicitly. Fixed gate conditions are retained in an adapter-owned JSON Comment
record, since the standard PLECS numeric table axes are current/temperature/voltage.
This is a data-format export; simulator acceptance is a separate validation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
from xml.etree import ElementTree as ET

import numpy as np

from transistordatabase.plecs_document import (
    MAX_NUMBERS, NAMESPACE, parse_plecs_document, render_plecs_document,
)
from transistordatabase.thermal import foster_network


def _finite(value, label, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, (bool, str)) or value is None or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number" + (" or explicit unknown null" if nullable else ""))
    return float(value)


def _curve(graph, label):
    array = np.asarray(graph, dtype=float)
    if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] < 1:
        raise ValueError(f"{label}: requires an actual 2×N curve")
    if array.size > MAX_NUMBERS * 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: finite bounded data required")
    if np.any(np.diff(array[0]) <= 0):
        raise ValueError(f"{label}: current axis must be increasing and unique")
    return array


def _text(parent, name, values):
    child = ET.SubElement(parent, f"{{{NAMESPACE}}}{name}")
    child.text = " ".join(format(float(value), ".17g") for value in values)
    return child


def _element(parent, name, attributes=None):
    return ET.SubElement(parent, f"{{{NAMESPACE}}}{name}", attributes or {})


def core_selection_document(transistor, selection: dict):
    """Create an immutable report and PLECS document from exact fixed conditions.

    Per-kind null selections are deliberate omissions and require readable reasons.
    A selected null gate/Rg matches an actual unknown, never any arbitrary slice.
    """
    selected = deepcopy(selection)
    allowed = {"component", "conduction", "turn_on", "turn_off", "include_thermal",
               "omission_reasons", "recovery_voltage_convention"}
    if set(selected) != allowed:
        raise ValueError(f"Declare exactly these core selection fields: {sorted(allowed)}")
    component_name = selected["component"]
    if component_name not in ("switch", "diode"):
        raise ValueError("Select exactly one switch or diode component")
    component = getattr(transistor, component_name)
    if not isinstance(selected["include_thermal"], bool):
        raise ValueError("include_thermal must be explicit boolean")
    conventions = ("preserve-signed", "positive-bus-to-negative-plecs")
    if selected["recovery_voltage_convention"] not in conventions:
        raise ValueError("Declare an explicit supported recovery-voltage convention")
    reasons = selected["omission_reasons"]
    if not isinstance(reasons, dict) or not set(reasons).issubset({"conduction", "turn_on", "turn_off", "thermal"}):
        raise ValueError("Omission reasons must name known mechanisms")
    omitted = []
    for key in ("conduction", "turn_on", "turn_off", "thermal"):
        if (key == "thermal" and not selected["include_thermal"]) or (key != "thermal" and selected[key] is None):
            if not isinstance(reasons.get(key), str) or not reasons[key].strip():
                raise ValueError(f"Omitting {key} requires an explicit nonblank reason")
            omitted.append(key)
        elif key in reasons:
            raise ValueError(f"{key} is included but also has an omission reason")
    if all(selected[k] is None for k in ("conduction", "turn_on", "turn_off")):
        raise ValueError("Select at least one actual semiconductor table")
    if not all(isinstance(v, str) and v.strip() for v in (
        transistor.metadata.name, transistor.metadata.manufacturer, transistor.metadata.type
    )):
        raise ValueError("Export requires actual device identity")
    root = ET.Element(f"{{{NAMESPACE}}}SemiconductorLibrary", {"version": "1.1"})
    package = _element(root, "Package", {"class": transistor.metadata.type if component_name == "switch" else "Diode",
        "vendor": transistor.metadata.manufacturer, "partnumber": transistor.metadata.name})
    block = _element(package, "SemiconductorData", {"type": transistor.metadata.type if component_name == "switch" else "Diode"})
    evidence = []
    cell_count = 0
    for key, xml_name in (("conduction", "ConductionLoss"), ("turn_on", "TurnOnLoss"), ("turn_off", "TurnOffLoss")):
        context = selected[key]
        if context is None:
            continue
        context_fields = {"gate_voltage_v"} if key == "conduction" else {"gate_voltage_v", "gate_resistance_ohm"}
        if not isinstance(context, dict) or set(context) != context_fields:
            raise ValueError(f"{key}: exact selectors require {sorted(context_fields)}")
        gate = _finite(context["gate_voltage_v"], f"{key} gate", nullable=True)
        resistance = None if key == "conduction" else _finite(context["gate_resistance_ohm"], f"{key} Rg", nullable=True)
        if resistance is not None and resistance < 0:
            raise ValueError("Gate resistance must be nonnegative")
        def matches(record, kind=key, selected_gate=gate, selected_resistance=resistance):
            actual_gate = _finite(record.v_g, f"{kind} recorded gate", nullable=True)
            if kind == "conduction":
                return actual_gate == selected_gate
            actual_resistance = _finite(record.r_g, f"{kind} recorded Rg", nullable=True)
            if actual_resistance is not None and actual_resistance < 0:
                raise ValueError("Recorded gate resistance must be nonnegative")
            return actual_gate == selected_gate and actual_resistance == selected_resistance

        if key == "conduction":
            records = [record for record in component.channel_data if matches(record)]
        elif component_name == "diode":
            if key == "turn_on":
                raise ValueError("Core diode has no turn-on model; omit it explicitly")
            records = [record for record in component.e_rr_data if matches(record)]
        else:
            records = [record for record in getattr(component, "e_on_data" if key == "turn_on" else "e_off_data")
                       if matches(record)]
        if not records:
            raise ValueError(f"{key}: no actual curves at the exact declared gate/Rg conditions")
        additional_conditions = None
        if key != "conduction":
            contexts = {(record.v_g_off, record.commutation_device,
                         record.load_inductance, record.commutation_inductance)
                        for record in records}
            if len(contexts) != 1:
                raise ValueError(f"{key}: mixed gate-off/commutation conditions require a narrower supported selection")
            off, commutation, load, parasitic = next(iter(contexts))
            additional_conditions = {
                "gate_off_voltage_v": _finite(off, "gate-off voltage", nullable=True),
                "commutation_device": commutation,
                "load_inductance_h": _finite(load, "load inductance", nullable=True),
                "commutation_inductance_h": _finite(parasitic, "commutation inductance", nullable=True),
            }
        grids, currents, temperatures, voltages = {}, None, set(), set()
        for record in records:
            temperature = _finite(record.t_j, f"{key} Tj")
            if key == "conduction":
                source_graph = np.asarray(record.graph_v_i, dtype=float)
                if source_graph.ndim != 2 or source_graph.shape[0] != 2:
                    raise ValueError("Conduction requires an actual voltage/current 2×N curve")
                graph = _curve(source_graph[[1, 0]], key)
                voltage = None
            else:
                if record.dataset_type != "graph_i_e" or record.graph_i_e is None:
                    raise ValueError(f"{key}: selected context includes unsupported non-current energy data")
                graph = _curve(record.graph_i_e, key)
                voltage = _finite(record.v_supply, f"{key} supply")
                if np.any(graph[1] < 0):
                    raise ValueError(f"{key}: negative energy requires a resolved source convention")
                if component_name == "diode" and selected["recovery_voltage_convention"] == "positive-bus-to-negative-plecs":
                    if voltage < 0:
                        raise ValueError("Positive-bus conversion requires nonnegative core recovery voltages")
                    voltage = -voltage
            if currents is None:
                currents = graph[0]
            elif not np.array_equal(currents, graph[0]):
                raise ValueError(f"{key}: current axes differ; no interpolation is implicit")
            pair = (temperature, voltage)
            if pair in grids:
                raise ValueError(f"{key}: duplicate temperature/voltage conditions are ambiguous")
            grids[pair] = graph[1]
            temperatures.add(temperature)
            voltages.add(voltage)
            cell_count += graph.shape[1]
            if cell_count > MAX_NUMBERS:
                raise ValueError(f"Core selection exceeds {MAX_NUMBERS} numeric/plot cells")
        temperatures = sorted(temperatures)
        voltages = [None] if key == "conduction" else sorted(voltages)
        if len(grids) != len(temperatures) * len(voltages):
            raise ValueError(f"{key}: incomplete temperature × voltage rectangle")
        loss = _element(block, xml_name)
        _element(loss, "ComputationMethod").text = "Table only"
        _text(loss, "CurrentAxis", currents)
        if key != "conduction":
            _text(loss, "VoltageAxis", voltages)
        _text(loss, "TemperatureAxis", temperatures)
        values = _element(loss, "VoltageDrop" if key == "conduction" else "Energy", {"scale": "1"})
        for temperature in temperatures:
            row = _element(values, "Temperature")
            if key == "conduction":
                row.text = " ".join(format(float(y), ".17g") for y in grids[(temperature, None)])
            else:
                for voltage in voltages:
                    _text(row, "Voltage", grids[(temperature, voltage)])
        evidence.append({"kind": key, "conditions": context, "temperatures_c": temperatures,
                         "additional_conditions": additional_conditions,
                         "voltages_v": None if key == "conduction" else voltages,
                         "current_a": currents.tolist(), "curve_count": len(records)})
    if selected["include_thermal"]:
        if component.thermal_foster is None:
            raise ValueError("Selected component has no Foster model")
        resistance, tau = foster_network(asdict(component.thermal_foster))
        network = _element(_element(package, "ThermalModel"), "Branch", {"type": "Foster"})
        for r, t in zip(resistance, tau, strict=True):
            if not math.isfinite(float(t)) or float(t) <= 0:
                raise ValueError("Foster tau is not representable in exported SI float precision")
            _element(network, "RTauElement", {"R": format(float(r), ".17g"), "Tau": format(float(t), ".17g")})
    fixed_conditions = {"schema_version": "tdb-fixed-conditions-1", "selection": selected,
                        "electrical_ratings": asdict(transistor.electrical_ratings),
                        "evidence": evidence,
                        "scope": "exact selected tables; omitted mechanisms are unknown, not zero"}
    _element(_element(package, "Comment"), "Line").text = "TDB_FIXED_CONDITIONS_JSON=" + json.dumps(fixed_conditions, allow_nan=False, sort_keys=True)
    document = parse_plecs_document(ET.tostring(root, encoding="utf-8", xml_declaration=True),
                                    "core:" + transistor.metadata.name)
    return document, {"status": "selected-subset", "selection": selected,
                       "omitted_mechanisms": omitted, "evidence": evidence,
                       "simulator_validation": "not-performed",
                       "fixed_conditions_storage": "TDB_FIXED_CONDITIONS_JSON Comment metadata; not an additional PLECS table axis"}


def render_core_selection(transistor, selection: dict) -> bytes:
    """Render only the exact supported core subset established by the report."""
    document, _ = core_selection_document(transistor, selection)
    return render_plecs_document(document)
