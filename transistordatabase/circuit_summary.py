"""Physical heat ownership around one authoritative converter calculation.

The boundary is an ideal fixed sink. Each declared junction has its own path.
This module does not solve operating points or update electrical models with Tj.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from transistordatabase import converter_losses
from transistordatabase.grouped_series import _csv_rows
from transistordatabase.thermal import steady_junction
from transistordatabase.waveform_losses import WaveformResourceLimitError

MAX_INPUT_BYTES = 4_000_000
MAX_RESPONSE_BYTES = 24_000_000
MAX_MEMBERS = 128
COMPONENTS = converter_losses.COMPONENTS
DIRECTIONS = ("positive", "negative", "zero")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonblank text")
    return value


def _number(value, label, nonnegative=False):
    if isinstance(value, (bool, str)) or value is None:
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or nonnegative and result < 0:
        raise ValueError(f"{label} is outside its finite physical domain")
    return result


def _conditions(value):
    if not isinstance(value, dict):
        raise ValueError("Source conditions must be an explicit object")
    for key, item in value.items():
        _text(key, "Condition name")
        if item is not None and not isinstance(item, str):
            _number(item, "Source condition")


def _preflight(request):
    """Bound physical expansion and metadata before electrical evaluation."""
    encoded = _json(request).encode("utf-8")
    if len(encoded) > MAX_INPUT_BYTES:
        raise WaveformResourceLimitError("Circuit summary input exceeds 4000000 bytes")
    _text(request["configuration_id"], "Configuration identifier")
    _text(request["scenario_id"], "Scenario identifier")
    if not isinstance(request.get("acknowledge_known_only", False), bool):
        raise ValueError("Known-only acknowledgement must be boolean")
    provenance = request.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("Provenance must be an explicit object or null")
        digest = provenance.get("source_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in "0123456789abcdefABCDEF" for c in digest)):
            raise ValueError("A source SHA-256 must contain exactly 64 hexadecimal characters")
    boundary = request["boundary"]
    if boundary["kind"] != "ideal-fixed-sink":
        raise ValueError("Only an explicitly ideal fixed sink is supported")
    _number(boundary["temperature_c"], "Sink temperature")
    _text(boundary["source"], "Sink source")
    _conditions(boundary["conditions"])
    converter = request["converter"]
    if converter["preset"] == "custom":
        bindings = converter["bindings"]
    else:
        presets = converter_losses.converter_presets()["presets"]
        selected = next((p for p in presets if p["id"] == converter["preset"]), None)
        if selected is None:
            raise ValueError("Unknown converter preset")
        if converter.get("bindings") is not None:
            raise ValueError("Preset bindings cannot be replaced")
        bindings = selected["bindings"]
    expected = {}
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 32:
        raise WaveformResourceLimitError("Circuit summary requires 1 through 32 roles")
    roles = set()
    for binding in bindings:
        role = _text(binding["role"], "Role")
        if role in roles:
            raise ValueError("Role identifiers must be unique")
        roles.add(role)
        members = binding["members"]
        count = binding["multiplicity"]
        if isinstance(count, bool) or not isinstance(count, int) or count != len(members) or count < 0:
            raise ValueError("Multiplicity must equal the physical member count")
        if len(expected) + count > MAX_MEMBERS:
            raise WaveformResourceLimitError("Circuit summary permits at most 128 physical members")
        for member in members:
            _text(member, "Physical member identifier")
            if member in expected:
                raise ValueError("Physical member identifiers must be globally unique")
            expected[member] = role
    members = request["members"]
    if not isinstance(members, list) or len(members) > MAX_MEMBERS:
        raise WaveformResourceLimitError("Circuit summary permits at most 128 physical members")
    supplied, junction_ids = {}, set()
    for member in members:
        identifier = _text(member["id"], "Member identifier")
        if identifier in supplied or expected.get(identifier) != member["role"]:
            raise ValueError("Members must match the converter's physical identifiers and roles")
        supplied[identifier] = member["role"]
        if not isinstance(member["layout_complete"], bool):
            raise ValueError("Layout completeness must be boolean")
        junctions = member["junctions"]
        if not isinstance(junctions, list) or len(junctions) > 128:
            raise WaveformResourceLimitError("A member permits at most 128 junctions")
        local = set()
        for junction in junctions:
            jid = _text(junction["id"], "Junction identifier")
            if jid in junction_ids:
                raise ValueError("Junction identifiers must be globally unique")
            junction_ids.add(jid)
            local.add(jid)
            resistance, source = junction["rtheta_k_w"], junction["source"]
            if resistance is not None:
                _number(resistance, "Junction resistance", nonnegative=True)
            if source is not None:
                _text(source, "Junction source")
            _conditions(junction["conditions"])
        owners = member["loss_owners"]
        if set(owners) != set(COMPONENTS):
            raise ValueError("Declare ownership for all three loss components")
        for component in COMPONENTS:
            if set(owners[component]) != set(DIRECTIONS):
                raise ValueError("Declare positive, negative and zero ownership explicitly")
            for owner in owners[component].values():
                if set(owner) != {"die", "junction_id"}:
                    raise ValueError("Ownership requires explicit die and junction_id fields")
                if owner["die"] not in ("channel", "diode", None):
                    raise ValueError("Die ownership must be channel, diode or null")
                if owner["junction_id"] is not None and owner["junction_id"] not in local:
                    raise ValueError("A heat owner must reference a junction of its own member")
    if supplied != expected:
        raise ValueError("Members must exactly cover the converter's physical positions")
    # Model/evidence metadata is stored once per role. Only small component rows
    # expand per member. Bound that expansion before the calculator allocates it.
    expansion = sum(len(_json(m).encode("utf-8")) + 4096 for m in members)
    projected = 3 * (len(encoded) + expansion
                     + converter_losses.converter_evidence_budget(converter, bindings))
    for member in members:
        repeated_csv = _json([request["configuration_id"], request["scenario_id"],
                              member["id"], member["role"]])
        projected += 3 * max(1, len(member["junctions"])) * (len(repeated_csv.encode("utf-8")) + 512)
    for binding in bindings:
        if binding["multiplicity"] == 0:
            continue
        model = converter["role_models"].get(binding["role"], {})
        conduction = model.get("conduction") or {}
        reasons = []
        for name in COMPONENTS:
            spec = conduction if name == "conduction" else (model.get("switching") or {}).get(name, {})
            if spec.get("omit"):
                reasons.append(spec.get("omission_reason", ""))
        if binding.get("source_switching_omission"):
            reasons.append(converter.get("assumptions", {}).get("source_omission_reason", ""))
        reason_bytes = sum(len(_json(reason).encode("utf-8")) + 80 for reason in reasons)
        role_members = [m for m in members if m["role"] == binding["role"]]
        projected += 3 * (sum(len(m["junctions"]) for m in role_members)
                          + 4 * len(role_members)) * reason_bytes
    if projected > MAX_RESPONSE_BYTES:
        raise WaveformResourceLimitError("Circuit metadata expansion exceeds the response budget")
    return encoded


def _electrical(components, duration):
    computed = [c for c in components.values() if c["status"] == "computed"]
    known = math.fsum(c["energy_j"] for c in computed) if computed else None
    complete = len(computed) == len(COMPONENTS)
    return {"known_component_energy_j": known,
            "known_component_power_w": None if known is None else known / duration,
            "total_energy_j": known if complete else None,
            "total_power_w": known / duration if complete else None,
            "total_status": "computed" if complete else "partial-explicit-omissions"}


def _direction(current):
    return "positive" if current > 0 else "negative" if current < 0 else "zero"


def _allocation(branch, duration):
    """Use accepted directional energies, events or held interval powers only."""
    evidence, components = branch["evidence"], branch["components"]
    energy = {name: {d: 0.0 for d in DIRECTIONS} for name in COMPONENTS}
    if "intervals" in evidence:
        for interval in evidence["intervals"]:
            width = interval["end_s"] - interval["start_s"]
            for name in COMPONENTS:
                power = interval[f"{name}_power_w"]
                if power is not None:
                    current = interval["current_a" if name == "conduction" else "switching_current_a"]
                    energy[name][_direction(current)] += power * width
    else:
        directional = components["conduction"].get("directional_energy_j", {})
        energy["conduction"].update(directional)
        for event in evidence.get("events", []):
            energy[event["event"]][_direction(event["current_a"])] += event["energy_j"]
    result, residuals = {}, {}
    for name, item in components.items():
        if item["status"] != "computed":
            result[name] = None
            residuals[name] = None
            continue
        total = math.fsum(energy[name].values())
        authoritative = item["energy_j"]
        residual = authoritative - total
        relative_residual = abs(residual) / max(total, authoritative) if max(total, authoritative) else 0
        if ((total == 0) != (authoritative == 0)
                or relative_residual > 1e-12):
            raise ValueError("Accepted loss evidence does not account for its component energy")
        # Preserve directional evidence. A floating residual has no physical owner.
        residuals[name] = {"energy_j": residual, "power_w": residual / duration,
                           "relative_energy_residual": relative_residual}
        result[name] = energy[name]
    return result, residuals


def _power(energies, duration):
    """Group physical-owner energies before the single duration division."""
    energy = math.fsum(energies)
    power = energy / duration
    if not math.isfinite(power) or energy > 0 and power == 0:
        raise ValueError("Allocated heat exceeds finite power representation")
    return power


def _maximum(junctions, complete, acknowledged):
    full = [j for j in junctions if j["tj_c"] is not None]
    known = [j for j in junctions if j["known_heat_tj_c"] is not None]
    value = max((j["tj_c"] for j in full), default=None) if complete else None
    partial = max((j["known_heat_tj_c"] for j in known), default=None) if acknowledged else None
    return {"tj_c": value,
            "junction_ids": [j["id"] for j in full if j["tj_c"] == value] if value is not None else [],
            "known_heat_tj_c": partial,
            "known_heat_junction_ids": [j["id"] for j in known if j["known_heat_tj_c"] == partial] if partial is not None else [],
            "status": "complete" if value is not None else "incomplete"}


def _member(member, branch, allocation, duration, boundary, acknowledged):
    owners = member["loss_owners"]
    junctions = {j["id"]: {**deepcopy(j), "assigned": [], "omissions": []}
                 for j in member["junctions"]}
    die_values = {"channel": [], "diode": []}
    die_unknown = {"channel": False, "diode": False}
    unallocated, die_unallocated, omissions = [], [], []
    for name in COMPONENTS:
        values = allocation[name]
        if values is None:
            reason = f"{name}: {branch['components'][name].get('reason', 'required component omitted')}"
            omissions.append(reason)
            for owner in owners[name].values():
                for die in die_unknown:
                    if owner["die"] in (die, None):
                        die_unknown[die] = True
                targets = junctions.values() if owner["junction_id"] is None else [junctions[owner["junction_id"]]]
                for target in targets:
                    if reason not in target["omissions"]:
                        target["omissions"].append(reason)
            continue
        for direction, energy in values.items():
            if energy == 0:
                continue
            owner = owners[name][direction]
            die = owner["die"]
            if die is None:
                die_unknown = dict.fromkeys(die_unknown, True)
                die_unallocated.append(energy)
            else:
                die_values[die].append(energy)
            jid = owner["junction_id"]
            if jid is None:
                unallocated.append(energy)
                reason = f"{name}/{direction}: known heat has no junction owner"
                omissions.append(reason)
                for target in junctions.values():
                    target["omissions"].append(reason)
            else:
                junctions[jid]["assigned"].append(energy)
    rows = []
    inventory_complete = member["layout_complete"] and bool(junctions)
    any_computed = any(v is not None for v in allocation.values())
    for junction in junctions.values():
        assigned = junction.pop("assigned")
        known = _power(assigned, duration) if assigned else (0.0 if any_computed and not unallocated else None)
        missing = junction["omissions"]
        if not inventory_complete:
            missing.append("Physical junction inventory is incomplete")
        heat = known if not missing else None
        if junction["rtheta_k_w"] is None or junction["source"] is None:
            missing.append("A sourced thermal resistance is missing")
        usable = junction["rtheta_k_w"] is not None and junction["source"] is not None
        tj = steady_junction(heat, junction["rtheta_k_w"], boundary["temperature_c"], junction["source"]) if heat is not None and usable else None
        known_tj = steady_junction(known, junction["rtheta_k_w"], boundary["temperature_c"], junction["source"]) if acknowledged and known is not None and usable else None
        rows.append({**junction, "heat_w": heat, "known_heat_w": known,
                     "tj_c": tj, "known_heat_tj_c": known_tj,
                     "status": "complete" if tj is not None else "incomplete"})
    for junction in rows:
        for reason in junction["omissions"]:
            if reason not in omissions:
                omissions.append(reason)
    if not junctions and "Physical junction inventory is incomplete" not in omissions:
        omissions.append("Physical junction inventory is incomplete")
    dies = {}
    for die, values in die_values.items():
        known = _power(values, duration) if values else (0.0 if any_computed and not die_unallocated else None)
        dies[die] = {"power_w": known if not die_unknown[die] else None,
                     "known_power_w": known,
                     "status": "incomplete" if die_unknown[die] or known is None else "complete"}
    dies["unallocated"] = {"known_power_w": _power(die_unallocated, duration) if die_unallocated else (0.0 if any_computed else None),
                           "status": "unallocated" if die_unallocated else "none"}
    complete = inventory_complete and not unallocated and all(j["tj_c"] is not None for j in rows)
    return {"id": member["id"], "role": member["role"],
            "layout_complete": member["layout_complete"], "loss_owners": deepcopy(owners),
            "components": deepcopy(branch["components"]),
            "electrical": _electrical(branch["components"], duration),
            "die_losses": dies, "junctions": rows, "omissions": omissions,
            "unallocated_junction_heat_w": _power(unallocated, duration) if unallocated else (0.0 if any_computed else None),
            "maximum": _maximum(rows, complete, acknowledged)}


def _csv(result):
    columns = ["configuration_id", "scenario_id", "input_sha256", "member_id", "role", "junction_id",
               "member_total_power_w", "member_known_power_w", "channel_power_w", "diode_power_w",
               "unallocated_die_power_w", "unallocated_junction_heat_w", "heat_w", "known_heat_w",
               "rtheta_k_w", "source", "conditions", "sink_temperature_c",
               "tj_c", "known_heat_tj_c", "status", "omissions"]
    rows = []
    for member in result["members"]:
        for junction in member["junctions"] or [{"id": None, "status": "incomplete", "omissions": ["No declared junction inventory"]}]:
            rows.append({"configuration_id": result["configuration_id"], "scenario_id": result["scenario_id"],
                         "input_sha256": result["input_sha256"], "member_id": member["id"], "role": member["role"],
                         "junction_id": junction["id"], "member_total_power_w": member["electrical"]["total_power_w"],
                         "member_known_power_w": member["electrical"]["known_component_power_w"],
                         "channel_power_w": member["die_losses"]["channel"]["power_w"],
                         "diode_power_w": member["die_losses"]["diode"]["power_w"],
                         "unallocated_die_power_w": member["die_losses"]["unallocated"]["known_power_w"],
                         "unallocated_junction_heat_w": member["unallocated_junction_heat_w"],
                         **{key: junction.get(key) for key in ("heat_w", "known_heat_w", "rtheta_k_w", "source", "tj_c", "known_heat_tj_c")},
                         "conditions": _json(junction["conditions"]) if "conditions" in junction else None,
                         "sink_temperature_c": result["boundary"]["temperature_c"],
                         "status": junction["status"], "omissions": " | ".join(junction["omissions"])})
    return _csv_rows(rows, columns)


def analyze_circuit_summary(request: dict) -> dict:
    """Calculate once, then audit physical heat ownership and fixed-sink paths."""
    encoded = _preflight(request)
    request = deepcopy(request)
    converter = converter_losses.analyze_converter(request["converter"])
    duration = converter["duration_s"]
    branches = {b["role"]: b for b in converter["branches"]}
    allocation = {role: _allocation(branch, duration) for role, branch in branches.items()}
    acknowledged = request.get("acknowledge_known_only", False)
    members = [_member(m, branches[m["role"]], allocation[m["role"]][0], duration,
                       request["boundary"], acknowledged) for m in request["members"]]
    electrical = deepcopy(converter["totals"])
    for name in COMPONENTS:
        computed = [b["scaled_components"][name] for b in branches.values()
                    if b["multiplicity"] > 0 and b["scaled_components"][name]["status"] == "computed"]
        if not computed:
            electrical["components"][name]["known_energy_j"] = None
            electrical["components"][name]["known_power_w"] = None
    computed_any = any(b["multiplicity"] > 0 and any(c["status"] == "computed" for c in b["components"].values()) for b in branches.values())
    if not computed_any:
        electrical["known_component_energy_j"] = None
        electrical["known_component_power_w"] = None
    junctions = [j for m in members for j in m["junctions"]]
    complete = bool(members) and all(m["maximum"]["tj_c"] is not None for m in members)
    digest = hashlib.sha256(encoded).hexdigest()
    provenance = request.get("provenance") or {}
    result = {"method": "explicit-physical-ownership-fixed-sink-v1",
              "input_sha256": digest,
              "input_identity": {"canonical_request_sha256": digest,
                                 "converter_request_sha256": hashlib.sha256(_json(request["converter"]).encode("utf-8")).hexdigest(),
                                 "upstream_source_sha256": provenance.get("source_sha256")},
              "configuration_id": request["configuration_id"], "scenario_id": request["scenario_id"],
              "duration_s": duration, "boundary": request["boundary"], "members": members,
              "electrical": electrical, "maximum": _maximum(junctions, complete, acknowledged),
              "evidence_by_role": {role: {"evidence": b["evidence"], "binding": b["binding"],
                                           "allocation_residuals": allocation[role][1],
                                           "model_bindings": b["model_bindings"], "broadcasts": b["broadcasts"]}
                                   for role, b in branches.items()},
              "coverage": {"member_count": len(members),
                           "layout_complete_members": sum(m["layout_complete"] for m in members),
                           "unallocated_known_heat_w": math.fsum(m["unallocated_junction_heat_w"] or 0 for m in members) if computed_any else None,
                           "omissions": [f"{m['id']}: {reason}" for m in members for reason in m["omissions"]]},
              "acknowledge_known_only": acknowledged, "provenance": request.get("provenance"),
              "request": request, "iteration_status": "not-required-constant-heat",
              "null_csv": "NULL",
              "csv_metadata_scope": "Each row references the canonical input SHA-256. Full boundary and model sources remain in the core JSON.",
              "interpretation": "Constant accepted electrical heat; independent junction paths to one ideal fixed sink. No electrothermal feedback."}
    exported = _json(result)
    csv_text = _csv(result)
    # JSON strings escape quotes/backslashes on the wire. Check duplication
    # before constructing the final response or its export strings.
    export_size = len(_json(exported).encode("utf-8"))
    if len(exported.encode("utf-8")) + export_size + len(_json(csv_text).encode("utf-8")) + 64 > MAX_RESPONSE_BYTES:
        raise WaveformResourceLimitError("Circuit summary response and exports exceed 24000000 bytes")
    return {**result, "export_json": exported, "export_csv": csv_text}
