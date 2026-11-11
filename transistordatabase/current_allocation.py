"""Allocate signed current samples by an explicitly selected gate-state assumption."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

from transistordatabase.gate_states import hysteretic_gate_states


def _utf8(value):
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Supply UTF-8 scalar text without isolated Unicode surrogates") from exc


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain actual finite numbers")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds the finite numerical range") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must contain actual finite numbers")
    if number == 0 and value != 0:
        raise ValueError(f"{name} is too small for a nonzero numeric value")
    return number


def _vector(values, name):
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 10000:
        raise ValueError(f"{name} needs 2 to 10000 aligned samples")
    return [_number(value, name) for value in values]


def allocate_gate_current(*, time_s, current_a, gate, source, conditions, selected,
                          provenance=None):
    """Return complementary sample allocations without a physical-path inference.

    Binary samples accept actual booleans or numeric 0/1. A declared voltage
    threshold uses strict greater-than. No state exists between sample times.
    """
    if selected is not True:
        raise ValueError("Explicitly select the gate-state allocation assumption")
    if not isinstance(source, str) or not source.strip() or len(source) > 2000:
        raise ValueError("Supply a nonblank allocation source of at most 2000 characters")
    _utf8(source)
    if not isinstance(conditions, dict) or any(
            not isinstance(key, str) or not key.strip() for key in conditions):
        raise ValueError("Declare conditions with nonblank string names")
    for key in conditions:
        _utf8(key)
    conditions = {key: None if value is None else _number(value, "conditions")
                  for key, value in conditions.items()}
    time = _vector(time_s, "time_s")
    current = _vector(current_a, "current_a")
    if len(time) != len(current):
        raise ValueError("Time and current samples must have equal lengths")
    if any(right <= left for left, right in zip(time, time[1:], strict=False)):
        raise ValueError("Time samples must increase strictly")
    if not isinstance(gate, dict):
        raise ValueError("Supply an explicit gate policy")
    digitization = None
    if gate.get("mode") == "binary":
        if set(gate) != {"mode", "state"}:
            raise ValueError("Binary gate policy requires only mode and state")
        values = gate["state"]
        if not isinstance(values, (list, tuple)) or len(values) != len(time):
            raise ValueError("Gate state needs one value per current sample")
        states = []
        for value in values:
            if not isinstance(value, bool) and (not isinstance(value, Real) or value not in (0, 1)):
                raise ValueError("Binary gate state accepts only actual booleans or numeric 0/1")
            states.append(bool(value))
        policy = {"mode": "binary", "state": states}
    elif gate.get("mode") == "threshold":
        if set(gate) != {"mode", "gate_v", "threshold_v"}:
            raise ValueError("Threshold gate policy requires mode, gate_v and threshold_v")
        values = _vector(gate["gate_v"], "gate_v")
        threshold = _number(gate["threshold_v"], "threshold_v")
        if len(values) != len(time):
            raise ValueError("Gate voltage needs one value per current sample")
        states = [value > threshold for value in values]
        policy = {"mode": "threshold", "gate_v": values, "threshold_v": threshold}
    elif gate.get("mode") == "hysteresis":
        if set(gate) != {"mode", "gate_v", "digitizer"}:
            raise ValueError("Hysteresis gate policy requires only mode, gate_v and digitizer")
        values = _vector(gate["gate_v"], "gate_v")
        if len(values) != len(time):
            raise ValueError("Gate voltage needs one value per current sample")
        digitization = hysteretic_gate_states(values, gate["digitizer"])
        states = digitization["evidence"]["states"]
        policy = {"mode": "hysteresis", "gate_v": values, "digitizer": digitization["policy"]}
    else:
        raise ValueError("Select binary, threshold or hysteresis gate policy")
    if provenance is not None:
        if (not isinstance(provenance, dict)
                or set(provenance) - {"source", "calibration_notes", "imported_filename", "edited_after_import"}
                or any(value is not None and not isinstance(value, str)
                       for key, value in provenance.items() if key != "edited_after_import")
                or ("edited_after_import" in provenance
                    and type(provenance["edited_after_import"]) is not bool)):
            raise ValueError("Supply valid waveform provenance")
        for value in provenance.values():
            if isinstance(value, str):
                _utf8(value)
        provenance = {"source": None, "calibration_notes": None,
                      "imported_filename": None, "edited_after_import": False} | provenance
    inputs = {"time_s": time, "current_a": current, "gate": policy,
              "source": source.strip(), "conditions": conditions,
              "selected": True, "provenance": provenance}
    rows = [{"time_s": t, "current_a": value, "gate_on": state,
             "gate_on_current_a": value if state else 0.0,
             "gate_off_current_a": 0.0 if state else value}
            for t, value, state in zip(time, current, states, strict=True)]
    result = {"samples": rows, "sample_count": len(time),
              "semantics": "Sample-only mask. No state or interpolation is defined between samples. Threshold equality is low.",
              "scope": "Gate-on channel allocation and gate-off diode allocation are command-state assumptions. They do not identify physical current paths, simultaneous sharing, losses or energy.",
              "units": {"time_s": "s", "current_a": "A", "gate_on_current_a": "A",
                        "gate_off_current_a": "A", "gate_on": "boolean"}}
    if digitization is not None:
        result["gate_digitization"] = digitization["evidence"]
        result["semantics"] = "Sample-only hysteretic mask. Strict crossings change state. Equality and deadband hold the preceding sampled state. No switching time is inferred between samples."
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    return {"inputs": inputs, "result": result,
            "evidence": {"method": "gate-state-allocation-v1", "input_sha256": digest(inputs),
                         "result_sha256": digest(result)}}
