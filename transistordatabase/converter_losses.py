"""Declarative converter aggregation of supplied SI waveforms or held envelopes.

This module does not solve converter operating points or synthesize PWM. Preset
data describes physical roles; waveform_losses remains the scientific authority.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np

from transistordatabase.waveform_losses import (
    WaveformResourceLimitError, analyze_waveform, evaluate_switching_energy,
    prepare_switching_models,
)

REVISION = "1"
COMPONENTS = ("conduction", "turn_on", "turn_off")
MAX_ROLE_INTERVALS = 10000
MAX_ROLES = 32
MAX_INPUT_BYTES = 4_000_000
MAX_RESPONSE_BYTES = 24_000_000


def _ref(signal=None, scale=1, *, value=None, complement=False):
    if signal is None:
        return {"value": value}
    return {"signal": signal, "scale": scale, **({"complement": True} if complement else {})}


def _role(role, members, current, *, commutation=None, gate=None, bus=None,
          duty=None, frequency=None, omitted=False, activation=None):
    result = {"role": role, "members": members.split(","),
              "multiplicity": len(members.split(",")), "conduction_current": current}
    for name, value in (("switching_current", commutation), ("gate_signal", gate),
                        ("bus_voltage", bus), ("duty", duty), ("frequency_hz", frequency),
                        ("activation", activation)):
        if value is not None:
            result[name] = value
    if omitted:
        result["source_switching_omission"] = True
    return result


def _presets():
    waveform = "waveform"
    envelope = "held-envelope-estimate"
    presets = []

    def add(identifier, label, operation, bindings, source):
        presets.append({"id": identifier, "label": label, "operation": operation,
                        "bindings": bindings, "source": source, "revision": REVISION})

    dab = [_role(role, members, _ref(cond), commutation=_ref(comm, sign), gate=gate,
                 bus=_ref(bus)) for role, members, cond, comm, sign, gate, bus in [
                     ("s12", "s1,s2", "is2", "itp", -1, "vgs2", "vp"),
                     ("s34", "s3,s4", "is4", "itp", 1, "vgs4", "vp"),
                     ("s56", "s5,s6", "is6", "its", 1, "vgs6", "vs"),
                     ("s78", "s7,s8", "is8", "its", -1, "vgs8", "vs")]]
    add("sc-dab", "SC DAB: eight physical devices", waveform, dab,
        "SC calc_DAB_Psemi.m:16–48; corrected shared pre-edge/independent Eoff reference")
    roles = ("inverter_hb1", "inverter_hb2", "rectifier_hb1", "rectifier_hb2")
    zcs = [_role(role, members, _ref(current), commutation=_ref("itp") if i == 0 else None,
                 gate="vgs1" if i == 0 else None, bus=_ref("vp") if i == 0 else None,
                 omitted=i != 0) for i, (role, members, current) in enumerate(zip(
                     roles, ("s1,s2", "d3,d4", "d5,d6", "d7,d8"),
                     ("is1", "idp1", "ids1", "ids3"), strict=True))]
    add("sc-src-zcs", "SC SRC-ZCS: secondary switching omitted", waveform, zcs,
        "SC calc_SRCZCS_Psemi.m:17–49")
    zvs = [_role(role, members, _ref(current),
                 commutation=_ref("itp", -1 if i == 0 else 1) if i < 2 else None,
                 gate=("vgs2", "vgs4")[i] if i < 2 else None,
                 bus=_ref("vp") if i < 2 else None, omitted=i >= 2)
           for i, (role, members, current) in enumerate(zip(
               roles, ("s1,s2", "s3,s4", "s5,s6", "s7,s8"),
               ("is2", "is4", "is6", "is8"), strict=True))]
    for name in ("src-zvs", "llc"):
        add(f"sc-{name}", f"SC {name.upper()}: secondary switching omitted", waveform,
            deepcopy(zvs), "SC calc_SRCZVS_Psemi.m:17–50 / calc_llc_Psemi.m:17–48")
    bridge = [
        _role("mosfet", "s1,s2", _ref("iac"), bus=_ref("vdc"),
              duty=_ref("dn1"), frequency=_ref("fs")),
        _role("pfc_diode", "idpfc1,idpfc2", _ref("iac", -1), bus=_ref("vdc"),
              duty=_ref("dp1"), frequency=_ref("fs")),
        _role("reverse_igbt", "isrev1,isrev2", _ref("iac", -1), bus=_ref("vdc"),
              duty=_ref("dp1"), frequency=_ref(value=0), activation="negative-power"),
    ]
    add("sc-bridgeless", "SC bridgeless PFC", envelope, bridge,
        "SC calc_bridgeless_semiconductor_loss.m:25–94")
    six = [
        _role("low_mosfets", "sna,snb,snc", _ref("ia"), bus=_ref("vdc"),
              duty=_ref("dna"), frequency=_ref("fs")),
        _role("high_mosfets", "spa,spb,spc", _ref("ia", -1), bus=_ref("vdc"),
              duty=_ref("dpa"), frequency=_ref("fs")),
        _role("neutral_igbts", "spn,snn", _ref("ia", 3), bus=_ref("vdc"),
              duty=_ref("dpn"), frequency=_ref("fl")),
    ]
    add("sc-six-switch", "SC three-phase totem-pole PFC", envelope, six,
        "SC three_phase_totempole_pfc_semiconductor_loss.m:29–92")
    for name, phase, neutral in (("vr", 6, 0), ("vrn", 6, 6), ("ivr", 12, 0),
                                 ("ivrn", 12, 6), ("tvr", 0, 0)):
        bindings = [
            _role("pfc_mosfet", ",".join(f"s{i}" for i in range(1, 7)), _ref("ipfc"),
                  bus=_ref("vdcp"), duty=_ref("dpfc"), frequency=_ref("fs")),
            _role("pfc_diode", ",".join(f"dp{i}" for i in range(1, 7)), _ref("ipfc", -1),
                  bus=_ref("vdcp"), duty=_ref("dpfc", complement=True), frequency=_ref("fs")),
        ]
        for role, count, signal in (("phase_rectifier", phase, "idrl"),
                                     ("neutral_rectifier", neutral, "idrn")):
            bindings.append({"role": role, "members": [f"{role}_{i}" for i in range(count)],
                             "multiplicity": count, "conduction_current": _ref(signal),
                             "duty": _ref(value=1), "frequency_hz": _ref(value=0),
                             **({"inactive_reason": "No physical positions in selected topology"}
                                if count == 0 else {})})
        add(f"sc-{name}", f"SC {name.upper()} PFC", envelope, bindings,
            "SC calc_vr_semiconductor_loss.m:27–53,95–133")
    inverter = []
    for phase in "abc":
        for position, sign, duty in (("upper", 1, f"dp{phase}"), ("lower", -1, f"dn{phase}")):
            name = f"{phase}_{position}"
            inverter.append(_role(name, name, _ref(f"i{phase}", sign), bus=_ref("vdc"),
                                  duty=_ref(duty), frequency=_ref("fs")))
    add("inverter-six", "Two-level inverter: six explicit positions", envelope, inverter,
        "SC EXAMPLE_TwoLevelInverter.m:31–79; supplied independent phase signals")
    return presets


def converter_presets():
    """Return independent declarative preset snapshots, never executable expressions."""
    return {"revision": REVISION, "presets": _presets()}


def _finite(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, not boolean")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Converter result/input must remain finite")


def _signal(reference, signals, count, label, broadcasts):
    if not isinstance(reference, dict):
        raise ValueError(f"{label} needs a declarative signal or value binding")
    if set(reference) - {"signal", "value", "scale", "complement"}:
        raise ValueError(f"Unsupported {label} binding fields")
    if ("signal" in reference) == ("value" in reference):
        raise ValueError(f"{label} needs exactly one signal or constant value")
    raw = signals[reference["signal"]] if "signal" in reference else reference["value"]
    if np.ndim(raw) == 0:
        result = np.full(count, _finite(raw, label))
        broadcasts.append({"binding": label, "value": float(result[0]),
                           "representation": "Explicit scalar broadcast"})
    else:
        result = np.asarray(raw, dtype=float)
        if result.ndim != 1 or len(result) != count or not np.isfinite(result).all():
            raise ValueError(f"{label} must contain {count} aligned finite values")
    scale = _finite(reference.get("scale", 1), f"{label} scale")
    result = result * scale
    if reference.get("complement", False):
        result = 1 - result
    if not np.isfinite(result).all():
        raise ValueError(f"{label} binding overflows")
    return result


def _omitted(reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit nonblank source omission reason is required")
    return {"omit": True, "omission_reason": reason.strip()}


def _summary(components, duration):
    known = math.fsum(item["energy_j"] for item in components.values()
                      if item["status"] == "computed")
    complete = all(item["status"] == "computed" for item in components.values())
    return {"known_component_energy_j": known, "known_component_power_w": known / duration,
            "total_energy_j": known if complete else None,
            "total_power_w": known / duration if complete else None,
            "total_status": "computed" if complete else "partial-explicit-omissions"}


def _zero(reason):
    return {name: {"status": "computed", "energy_j": 0.0, "power_w": 0.0,
                   "inactive_reason": reason} for name in COMPONENTS}


def converter_evidence_budget(request, bindings):
    """Bound retained input and repeated evidence before branch evaluation.

    The estimate includes source conditions repeated in events and held intervals.
    Numerical rows have a fixed reserve. No waveform or curve is evaluated here.
    """
    encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_INPUT_BYTES:
        raise WaveformResourceLimitError("Converter input exceeds 4000000 bytes")
    projected = 4 * len(encoded) + 4096
    intervals = len(request["time_s"]) - 1
    for binding in bindings:
        if binding["multiplicity"] == 0:
            reason = binding.get("inactive_reason", "")
            projected += 6 * len(json.dumps(reason).encode("utf-8"))
            continue
        model = request["role_models"].get(binding["role"]) or {}
        conduction = model.get("conduction") or {}
        if request["operation"] == "held-envelope-estimate":
            # Held evidence has one row per interval and no polynomial plots.
            projected += intervals * 768
        elif not conduction.get("omit"):
            drops = [conduction.get(direction) or {} for direction in ("positive", "negative")]
            source_bytes = max(len(json.dumps(drop.get("source", "")).encode("utf-8")) for drop in drops)
            knots = sum(len(drop.get("current_a", [])) for drop in drops)
            segments = min(10000, intervals * (2 + knots))
            # One exact segment has at most nine fixed-shape plot records.
            projected += segments * (source_bytes + 2048)
        switching = model.get("switching") or {}
        if binding.get("source_switching_omission"):
            reason = request.get("assumptions", {}).get("source_omission_reason", "")
            # Each affected role retains on/off components and scaled copies.
            projected += 4 * len(json.dumps(reason).encode("utf-8"))
        for name in ("turn_on", "turn_off"):
            spec = switching.get(name) or {}
            if not spec or spec.get("omit") or binding.get("source_switching_omission"):
                continue
            metadata_bytes = max(len(json.dumps(
                {key: (spec.get(direction) or {}).get(key) for key in
                 ("mode", "source", "recorded_conditions", "requested_conditions")},
                allow_nan=False, separators=(",", ":"),
            ).encode("utf-8")) for direction in ("positive", "negative"))
            projected += intervals * (metadata_bytes + 1024)
    if projected > MAX_RESPONSE_BYTES:
        raise WaveformResourceLimitError("Converter evidence expansion exceeds 24000000 bytes")
    return projected


def _waveform_branch(binding, model, request, time, broadcasts):
    signals, n = request["signals"], len(time)
    current = _signal(binding["conduction_current"], signals, n, "conduction_current", broadcasts)
    conduction = deepcopy(model["conduction"])
    if "duty" in binding:
        conduction["duty"] = _signal(binding["duty"], signals, n, "duty", broadcasts).tolist()
    omission = _omitted("Internal component selection; not a physical omission")
    cond_request = {"waveform": {"time_s": time.tolist(), "current_a": current.tolist()},
                    "conduction": conduction,
                    "events": {"mode": "omitted", "omission_reason": omission["omission_reason"]},
                    "switching": {"turn_on": omission, "turn_off": omission}}
    cond = analyze_waveform(cond_request)
    switching = deepcopy(model.get("switching"))
    if binding.get("source_switching_omission"):
        if switching is not None and any(not switching[name].get("omit") for name in ("turn_on", "turn_off")):
            raise ValueError("Use a custom binding to add switching absent from the source preset")
        omit = _omitted(request["assumptions"].get("source_omission_reason"))
        switching = {"turn_on": omit, "turn_off": omit}
    if switching is None:
        raise ValueError(f"Missing switching scope for {binding['role']}")
    both_omitted = all(switching[name].get("omit") for name in ("turn_on", "turn_off"))
    commutation = current if both_omitted else _signal(
        binding["switching_current"], signals, n, "switching_current", broadcasts)
    waveform = {"time_s": time.tolist(), "current_a": commutation.tolist()}
    event_policy = ({"mode": "omitted", "omission_reason": "All physical switching components explicitly omitted"}
                    if both_omitted else model["events"])
    if not both_omitted and "gate_signal" in binding and (
            event_policy.get("mode") == "gate" or binding["gate_signal"] in signals):
        waveform["gate_v"] = _signal(_ref(binding["gate_signal"]), signals, n, "gate_v", broadcasts).tolist()
    if not both_omitted and "bus_voltage" in binding:
        waveform["bus_voltage_v"] = _signal(binding["bus_voltage"], signals, n, "bus_voltage_v", broadcasts).tolist()
    sw = analyze_waveform({"waveform": waveform, "conduction": omission,
                           "switching": switching, "events": event_policy})
    components = {"conduction": cond["components"]["conduction"],
                  **{name: sw["components"][name] for name in ("turn_on", "turn_off")}}
    evidence = {"conduction_signals": cond["signals"],
                "switching_signals": {} if both_omitted else sw["signals"],
                "switching_signals_status": "not-evaluated-explicit-omissions" if both_omitted else "supplied",
                "power_segments": cond["power_segments"], "conduction_plot": cond["conduction_plot"],
                "events": sw["events"], "event_policy": sw["event_policy"],
                "switching_cumulative_plot": [] if both_omitted else sw["switching_cumulative_plot"]}
    return components, evidence


def _envelope_branch(binding, model, request, time, broadcasts):
    signals, n = request["signals"], len(time) - 1
    current = _signal(binding["conduction_current"], signals, n, "conduction_current", broadcasts)
    commutation = (_signal(binding["switching_current"], signals, n, "switching_current", broadcasts)
                   if "switching_current" in binding else current)
    duty = _signal(binding["duty"], signals, n, "duty", broadcasts)
    frequency = _signal(binding["frequency_hz"], signals, n, "frequency_hz", broadcasts)
    bus = (_signal(binding["bus_voltage"], signals, n, "bus_voltage_v", broadcasts)
           if "bus_voltage" in binding else None)
    if np.any((duty < 0) | (duty > 1)) or np.any(frequency < 0):
        raise ValueError("Envelope duty must lie in [0,1] and frequency must be nonnegative")
    if bus is not None and np.any(bus < 0):
        raise ValueError("Switching bus voltage must be nonnegative")
    switching_active = (duty > 0) & (duty < 1) & (frequency > 0)
    omission = _omitted("Internal component selection; not a physical omission")
    components, prepared = {}, {}
    for name in ("turn_on", "turn_off"):
        spec = model.get("switching", {}).get(name)
        if spec is not None and spec.get("omit"):
            reason = _omitted(spec.get("omission_reason"))["omission_reason"]
            components[name] = {"status": "omitted", "reason": reason, "power_w": None, "energy_j": None}
        elif not switching_active.any():
            components[name] = {"status": "computed", "energy_j": 0.0, "power_w": 0.0,
                                "inactive_reason": "D=0, D=1 or fs=0 on every held interval"}
        elif spec is None:
            raise ValueError(f"Missing active envelope {name} models")
        else:
            prepared[name] = prepare_switching_models(spec)
            components[name] = {"status": "computed", "energy_j": 0.0, "power_w": 0.0}
    intervals, cond_energy = [], 0.0
    conduction_omission = model.get("conduction", {}).get("omit", False)
    for i, (start, end) in enumerate(zip(time[:-1], time[1:], strict=True)):
        cond_spec = deepcopy(model.get("conduction", {}))
        cond_spec["duty"] = float(duty[i])
        # Exact held conduction uses the existing signed piecewise-linear engine.
        if duty[i] == 0:
            pc = {"status": "computed", "power_w": 0.0, "energy_j": 0.0,
                  "inactive_reason": "Explicit D=0 held interval"}
        else:
            cond = analyze_waveform({"waveform": {"time_s": [float(start), float(end)],
                                                  "current_a": [float(current[i])] * 2},
                                     "conduction": cond_spec,
                                     "events": {"mode": "omitted", "omission_reason": omission["omission_reason"]},
                                     "switching": {"turn_on": omission, "turn_off": omission}})
            pc = cond["components"]["conduction"]
        if pc["energy_j"] is not None:
            cond_energy += pc["energy_j"]
        row = {"start_s": float(start), "end_s": float(end), "current_a": float(current[i]),
               "switching_current_a": float(commutation[i]), "duty": float(duty[i]),
               "frequency_hz": float(frequency[i]),
               "bus_voltage_v": None if bus is None else float(bus[i]),
               "conduction_power_w": pc["power_w"], "energy_evidence": {}}
        for name in ("turn_on", "turn_off"):
            item = components[name]
            power = None if item["status"] == "omitted" else 0.0
            if name in prepared and switching_active[i]:
                evidence = evaluate_switching_energy(prepared[name], commutation[i],
                                                     None if bus is None else bus[i])
                power = evidence["energy_j"] * frequency[i]
                item["energy_j"] += power * (end - start)
                row["energy_evidence"][name] = evidence
            row[f"{name}_power_w"] = None if power is None else float(power)
        intervals.append(row)
    duration = float(time[-1] - time[0])
    if conduction_omission:
        reason = _omitted(model["conduction"].get("omission_reason"))["omission_reason"]
        components["conduction"] = {"status": "omitted", "reason": reason, "energy_j": None, "power_w": None}
    else:
        components["conduction"] = {"status": "computed", "energy_j": cond_energy,
                                    "power_w": cond_energy / duration}
    for item in components.values():
        if item["status"] == "computed":
            item["energy_j"] = float(item["energy_j"])
            item["power_w"] = item["energy_j"] / duration
    return components, {"intervals": intervals, "representation": "Interval-held I, duty, bus and frequency",
                        "switching_interpretation": "Selected hard-switching frequency envelope estimate; no resolved PWM events or event counts"}


def analyze_converter(request: dict) -> dict:
    """Aggregate same-interval physical positions without inventing missing loss data."""
    request = _plain(request)
    _finite_tree(request)
    time = np.asarray(request["time_s"], dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time_s needs at least two strictly increasing finite edges/samples")
    duration = _finite(time[-1] - time[0], "duration_s")
    if duration <= 0:
        raise ValueError("Observation duration must be positive")
    operation, preset_id = request["operation"], request["preset"]
    if operation not in {"waveform", "held-envelope-estimate"}:
        raise ValueError("Unknown converter operation")
    assumptions = request.get("assumptions", {})
    request["assumptions"] = assumptions
    if operation == "held-envelope-estimate" and assumptions.get("envelope_selected") is not True:
        raise ValueError("Select the held hard-switching envelope estimate explicitly")
    if preset_id == "custom":
        bindings = deepcopy(request["bindings"])
        source = "Caller-defined physical-role mapping; no preset topology equivalence claimed"
    else:
        matching = [item for item in _presets() if item["id"] == preset_id]
        if not matching:
            raise ValueError("Unknown converter preset")
        preset = matching[0]
        if "bindings" in request and request["bindings"] is not None:
            raise ValueError("Preset mappings are immutable; choose custom for edited bindings")
        if preset["operation"] != operation:
            raise ValueError("Preset operation does not match supplied representation")
        bindings, source = preset["bindings"], preset["source"]
    if not bindings or len(bindings) > MAX_ROLES:
        raise WaveformResourceLimitError("Converter requires 1–32 explicit physical roles")
    if len(bindings) * (len(time) - 1) > MAX_ROLE_INTERVALS:
        raise WaveformResourceLimitError("Resource limit: at most 10000 total role intervals; reduce supplied resolution explicitly")
    converter_evidence_budget(request, bindings)
    names, members = set(), set()
    branches, expanded_segments = [], 0
    for binding in bindings:
        role, count, physical = binding["role"], binding["multiplicity"], binding["members"]
        if not isinstance(role, str) or not role.strip() or role in names:
            raise ValueError("Role identifiers must be nonblank and unique")
        names.add(role)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count != len(physical):
            raise ValueError("Multiplicity must be a nonnegative integer equal to member count")
        if any(not isinstance(name, str) or not name.strip() or name in members for name in physical) or len(set(physical)) != len(physical):
            raise ValueError("Physical member identifiers must be nonblank and globally unique")
        members.update(physical)
        if count > 1 and assumptions.get("symmetry_selected") is not True:
            raise ValueError("Select identical-behavior symmetry explicitly for multiplicity greater than one")
        inactive = binding.get("inactive_reason") if count == 0 else None
        if count == 0 and (not isinstance(inactive, str) or not inactive.strip()):
            raise ValueError("A zero-count role requires an explicit inactive reason")
        if binding.get("activation") is not None:
            if binding["activation"] != "negative-power":
                raise ValueError("Unsupported activation condition")
            power = _finite(assumptions["operating_power_w"], "operating_power_w")
            if power >= 0:
                inactive = "Reverse IGBTs inactive by selected nonnegative-power source assumption"
        broadcasts = []
        model = deepcopy(request["role_models"].get(role))
        if inactive:
            components, evidence = _zero(inactive), {"inactive_reason": inactive}
        elif model is None:
            raise ValueError(f"Missing required model binding for role {role}")
        else:
            calculate = _waveform_branch if operation == "waveform" else _envelope_branch
            components, evidence = calculate(binding, model, request, time, broadcasts)
        expanded_segments += len(evidence.get("power_segments", evidence.get("intervals", [])))
        if expanded_segments > 10000:
            raise WaveformResourceLimitError("Resource limit: converter exact expansion exceeds 10000 total segments")
        scaled = {name: {**item, "energy_j": None if item["energy_j"] is None else item["energy_j"] * count,
                         "power_w": None if item["power_w"] is None else item["power_w"] * count,
                         **({"event_count": item["event_count"] * count,
                             "event_rate_hz": item["event_rate_hz"] * count}
                            if "event_count" in item else {})} for name, item in components.items()}
        branches.append({"role": role, "members": physical, "multiplicity": count,
                         "components": components, "scaled_components": scaled,
                         **_summary(scaled, duration), "evidence": evidence,
                         "broadcasts": broadcasts, "binding": binding, "model_bindings": model})
    if set(request["role_models"]) - names:
        raise ValueError("Model bindings include unknown roles")
    totals = {}
    for name in COMPONENTS:
        items = [branch["scaled_components"][name] for branch in branches]
        known = math.fsum(item["energy_j"] for item in items if item["status"] == "computed")
        complete = all(item["status"] == "computed" for item in items)
        totals[name] = {"status": "computed" if complete else "partial-explicit-omissions",
                        "energy_j": known if complete else None,
                        "power_w": known / duration if complete else None,
                        "known_energy_j": known, "known_power_w": known / duration}
    known = math.fsum(item["known_energy_j"] for item in totals.values())
    complete = all(item["status"] == "computed" for item in totals.values())
    result = {"preset": preset_id, "preset_revision": REVISION, "operation": operation,
              "duration_s": duration, "bindings": bindings, "branches": branches,
              "physical_member_count": len(members), "source": source,
              "totals": {"components": totals, "known_component_energy_j": known,
                         "known_component_power_w": known / duration,
                         "total_energy_j": known if complete else None,
                         "total_power_w": known / duration if complete else None,
                         "total_status": "computed" if complete else "partial-explicit-omissions"},
              "units": {"power": "W", "energy": "J", "time": "s", "current": "A", "voltage": "V"},
              "request": request, "assumptions": deepcopy(assumptions),
              "interpretation": "Supplied-data aggregation; no operating-point solver, physical ZVS proof or implicit parallel-die sharing"}
    _finite_tree(result)
    if len(json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise WaveformResourceLimitError("Converter response exceeds 24000000 bytes")
    return result
