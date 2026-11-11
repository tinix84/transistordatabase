"""Explicit held-current PWM estimates through the accepted bounded data engines.

This is a semiconductor estimate, not a converter efficiency or waveform solver.
Every condition must be supplied, including None for deliberately unknown values.
"""
from __future__ import annotations

import math
from numbers import Real

from transistordatabase.condition_tables import MeasuredTable, require_conditions
from transistordatabase.waveform_losses import (
    evaluate_switching_energy, prepare_switching_models,
)


def finite(value, label):
    """Reject missing, boolean and nonfinite scientific inputs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} needs an explicit finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} needs an explicit finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} needs an explicit finite number")
    return value


def _energy_record(records, conditions):
    """Select one unambiguous exactly conditioned energy record."""
    required = {"t_j", "v_supply", "v_g", "r_g", "v_g_off", "i_channel"}
    if not required <= conditions.keys():
        raise ValueError("Energy lookup requires t_j, v_supply, v_g, r_g, "
                         "v_g_off and i_channel; use None for explicit unknowns")
    current = finite(conditions["i_channel"], "i_channel")
    if current < 0:
        raise ValueError("This legacy adapter supports nonnegative current only; "
                         "use signed directional waveform models")
    names = ("t_j", "v_supply", "v_g", "r_g", "v_g_off")
    requested = {name: conditions[name] for name in names}
    for name, value in requested.items():
        if value is not None:
            finite(value, name)
    matching = []
    for record in records:
        recorded = {name: getattr(record, name) for name in names}
        try:
            require_conditions(recorded, requested)
        except ValueError:
            continue
        if record.dataset_type in {"graph_i_e", "single"}:
            matching.append((record, recorded))
    if len(matching) != 1:
        raise ValueError("Need one unambiguous switching energy record at the exact "
                         "requested Tj, supply, gate voltages and gate resistance")
    return matching[0], requested, current


def _energy_model(records, conditions):
    (record, recorded), requested, _ = _energy_record(records, conditions)
    if record.dataset_type != "graph_i_e" or record.graph_i_e is None:
        raise ValueError("Static-duty estimate requires a bounded current-energy curve; "
                         "single-point evidence is not a curve")
    return {"mode": "recorded", "source": record.comment or "Core switching record",
            "current_a": record.graph_i_e[0], "energy_j": record.graph_i_e[1],
            "recorded_conditions": recorded, "requested_conditions": requested}


def energy_at(records, conditions):
    """Bounded current lookup, or an exact single-point measurement."""
    (record, _), _, current = _energy_record(records, conditions)
    if record.dataset_type == "single":
        require_conditions({"i_channel": record.i_x}, {"i_channel": current})
        energy = finite(record.e_x, "single-point energy")
        if energy < 0:
            raise ValueError("Passive switching heat must be nonnegative")
        return energy
    model = _energy_model(records, conditions)
    return evaluate_switching_energy(prepare_switching_models({"positive": model}), current)["energy_j"]


def static_duty_losses(transistor, operating_point):
    """Estimate one device with held current and explicit PWM duty/frequency.

    Exactly one on/off event per PWM period is selected by this operation for
    0 < duty < 1. Endpoint duty or zero frequency has no switching events.
    """
    required = {"t_j", "v_g", "v_g_off", "i_channel", "f_sw", "duty",
                "v_supply", "r_g_on", "r_g_off"}
    missing = sorted(required - operating_point.keys())
    if missing:
        raise ValueError("Static-duty estimate requires explicit " + ", ".join(missing)
                         + "; use waveform analysis for measured switching events")
    op = dict(operating_point)
    for name in required - {"v_g", "v_g_off", "r_g_on", "r_g_off"}:
        op[name] = finite(op[name], name)
    for name in ("v_g", "v_g_off", "r_g_on", "r_g_off"):
        if op[name] is not None:
            op[name] = finite(op[name], name)
    if min(op["i_channel"], op["f_sw"]) < 0 or op["v_supply"] <= 0:
        raise ValueError("Current/frequency must be nonnegative and supply positive")
    if not 0 <= op["duty"] <= 1:
        raise ValueError("duty must lie in [0, 1]")
    if any(op[name] is not None and op[name] < 0 for name in ("r_g_on", "r_g_off")):
        raise ValueError("Gate resistances must be nonnegative or explicitly unknown")
    from transistordatabase.converter_losses import analyze_converter

    models = {"source": "Core recorded data; explicitly selected held-current PWM estimate",
              "conduction": {}, "switching": {}}
    if op["duty"] > 0 and op["i_channel"] > 0:
        requested = {"t_j_c": op["t_j"], "v_g_v": op["v_g"]}
        matching = []
        for record in transistor.switch.channel_data:
            recorded = {"t_j_c": record.t_j, "v_g_v": record.v_g}
            try:
                require_conditions(recorded, requested)
            except ValueError:
                continue
            matching.append((record, recorded))
        if len(matching) != 1:
            raise ValueError("Need one unambiguous channel record at exact Tj and gate voltage")
        record, recorded = matching[0]
        table = MeasuredTable("channel", record.graph_v_i[1], record.graph_v_i[0],
                              recorded, "Core channel record")
        models["conduction"]["positive"] = {"mode": "recorded", "source": table.source,
            "current_a": table.x, "drop_v": table.y,
            "recorded_conditions": recorded, "requested_conditions": requested}
    elif op["i_channel"] == 0:
        models["conduction"]["positive"] = {
            "mode": "linear-drop-estimate", "selected": True,
            "source": "Zero-current identity P = V * 0; no finite-current model implied",
            "offset_v": 0, "resistance_ohm": 0, "conditions": {}}
    switching_active = op["f_sw"] > 0 and 0 < op["duty"] < 1
    if switching_active:
        conditions = {name: op[name] for name in ("t_j", "v_g", "v_g_off", "v_supply", "i_channel")}
        for name, records, rg in (("turn_on", transistor.switch.e_on_data, "r_g_on"),
                                  ("turn_off", transistor.switch.e_off_data, "r_g_off")):
            models["switching"][name] = {"positive": _energy_model(records, dict(conditions, r_g=op[rg]))}
    binding = {"role": "device", "members": ["device"], "multiplicity": 1,
        "conduction_current": {"value": op["i_channel"]},
        "duty": {"value": op["duty"]}, "frequency_hz": {"value": op["f_sw"]}}
    result = analyze_converter({"preset": "custom", "operation": "held-envelope-estimate",
        "time_s": [0, 1], "signals": {}, "bindings": [binding],
        "role_models": {"device": models}, "assumptions": {"envelope_selected": True}})
    components = result["branches"][0]["components"]
    return {"conduction_losses": components["conduction"]["power_w"],
            "switching_losses_on": components["turn_on"]["power_w"],
            "switching_losses_off": components["turn_off"]["power_w"],
            "total_losses": result["totals"]["total_power_w"],
            "method": "held-current-static-duty-estimate", "conditions": op,
            "switching_status": "computed" if switching_active else "inactive-explicit-duty-or-frequency",
            "evidence": result["branches"][0]["evidence"],
            "assumptions": ["Held positive per-device current; supplied duty applies to conduction",
                            "One turn-on and turn-off per PWM period only for 0 < duty < 1",
                            "Exact recorded conditions; bounded piecewise-linear curves"]}
