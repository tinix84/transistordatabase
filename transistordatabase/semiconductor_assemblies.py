"""Pure bounded semiconductor assembly transformations; all quantities use SI.

These operations estimate ideal equal sharing or select signed branches. They do
not solve balancing, simultaneous channel/diode sharing or thermal geometry.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np

from .condition_tables import MeasuredTable, require_conditions
from .thermal import foster_network

SCALARS = {
    "blocking_voltage_v", "current_rating_a", "area_m2", "cost", "weight_kg",
    "internal_resistance_k_per_w", "shared_sink_resistance_k_per_w",
}
CURVES = ("conduction", "turn_on", "turn_off")


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _table(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a recorded table")
    if not isinstance(value.get("source"), str) or not value["source"].strip():
        raise ValueError(f"{name} needs a nonblank source")
    expected = "channel" if name == "conduction" else "energy"
    if value.get("kind") not in ({"channel", "diode"} if name == "conduction" else {expected}):
        raise ValueError(f"{name} has the wrong table kind")
    if len(value.get("x", [])) > 10000 or len(value.get("y", [])) > 10000:
        raise ValueError("Assembly tables support at most 10000 samples per curve")
    table = MeasuredTable(**value)
    return {"kind": table.kind, "x": table.x.tolist(), "y": table.y.tolist(),
            "conditions": dict(table.conditions), "source": table.source}


def _device(device):
    if not isinstance(device, dict) or set(device) - (SCALARS | set(CURVES) | {"source", "currency", "foster"}):
        raise ValueError("Device contains unsupported fields")
    if not isinstance(device.get("source"), str) or not device["source"].strip():
        raise ValueError("Device needs a nonblank source")
    result = deepcopy(device)
    for name in SCALARS:
        result[name] = None if device.get(name) is None else _finite(device[name], name)
    currency = device.get("currency")
    if currency is not None and (not isinstance(currency, str) or not currency.strip()):
        raise ValueError("Currency must be a nonblank identifier or unknown")
    result["currency"] = currency
    for name in CURVES:
        result[name] = None if device.get(name) is None else _table(device[name], name)
    foster = device.get("foster")
    if foster is not None:
        if set(foster) - {"resistance_k_per_w", "tau_s", "capacity_j_per_k"}:
            raise ValueError("Unsupported Foster fields")
        r, tau = foster_network({"r_th_vector": foster["resistance_k_per_w"],
                                 "tau_vector": foster["tau_s"],
                                 "c_th_vector": foster.get("capacity_j_per_k")})
        r, tau = [float(v) for v in r], [float(v) for v in tau]
        c = foster.get("capacity_j_per_k")
        if c is not None:
            c = [_finite(v, "Foster capacity", positive=True) for v in c]
        result["foster"] = {"resistance_k_per_w": r, "tau_s": tau, "capacity_j_per_k": c}
    else:
        result["foster"] = None
    return result


def _scaled(values, factor, name):
    result = [float(v * factor) for v in values]
    if not np.isfinite(result).all():
        raise ValueError(f"{name} scaling overflow")
    if any(original != 0 and scaled == 0 for original, scaled in zip(values, result, strict=True)):
        raise ValueError(f"{name} scaling underflow")
    return result


def _interpolate(x, y, query):
    """Reject nonrepresentable derived values rather than leaking NaN into JSON."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = float(np.interp(query, x, y))
    if not np.isfinite(value):
        raise ValueError("Assembly interpolation is not representable")
    return value


def parallel_bank(device: dict, count: int) -> dict:
    """Return an independent ideal equal-sharing bank; unknowns remain None."""
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("Parallel count must be an integer at least one")
    result = _device(device)
    try:
        n = float(count)
    except OverflowError as exc:
        raise ValueError("Parallel count overflow") from exc
    if not np.isfinite(n):
        raise ValueError("Parallel count overflow")
    for name in ("current_rating_a", "area_m2", "cost", "weight_kg"):
        if result[name] is not None:
            result[name] = _scaled([result[name]], n, name)[0]
    if result["internal_resistance_k_per_w"] is not None:
        original = result["internal_resistance_k_per_w"]
        result["internal_resistance_k_per_w"] = original / n
        if original > 0 and result["internal_resistance_k_per_w"] == 0:
            raise ValueError("Internal thermal resistance scaling underflow")
    for name in CURVES:
        table = result[name]
        if table is not None:
            table["x"] = _scaled(table["x"], n, "current axis")
            if name != "conduction":
                table["y"] = _scaled(table["y"], n, "energy")
            _table(table, name)  # reject scaling that collapses distinct knots
    if result["foster"] is not None:
        poles = result["foster"]
        poles["resistance_k_per_w"] = [r / n for r in poles["resistance_k_per_w"]]
        if poles["capacity_j_per_k"] is not None:
            poles["capacity_j_per_k"] = _scaled(poles["capacity_j_per_k"], n, "Foster capacity")
        _device(result)  # includes derived-pole finite/underflow checks
    return {"operation": "parallel-bank", "device": result, "count": int(count),
            "assumptions": ["Ideal equal current sharing; no balancing solver",
                            "Independent internal paths scale; shared external sink is unchanged"],
            "source": device["source"]}


def signed_composite(positive: dict, negative: dict, common_resistance_k_per_w: float,
                     voltage_estimate: dict | None = None) -> dict:
    """Keep independent signed branches and their event-specific conditions."""
    branches = {"positive": _device(positive), "negative": _device(negative)}
    common = _finite(common_resistance_k_per_w, "Common thermal resistance")
    estimates = []
    if voltage_estimate is not None:
        if set(voltage_estimate) != {"selected", "target_voltage_v"} or voltage_estimate["selected"] is not True:
            raise ValueError("Select the named linear-voltage estimate explicitly")
        target = _finite(voltage_estimate["target_voltage_v"], "Target voltage", positive=True)
    else:
        target = None
    for event in ("turn_on", "turn_off"):
        tables = [(direction, branch[event]) for direction, branch in branches.items()
                  if branch[event] is not None]
        if target is None and len(tables) == 2:
            require_conditions(tables[0][1]["conditions"], tables[1][1]["conditions"])
        if target is not None:
            other = []
            for direction, table in tables:
                recorded = deepcopy(table["conditions"])
                voltage = recorded.get("v_supply_v")
                if voltage is None:
                    raise ValueError("Linear-voltage estimate needs a recorded positive v_supply_v")
                voltage = _finite(voltage, "Recorded supply voltage", positive=True)
                conditions = {k: v for k, v in recorded.items() if k != "v_supply_v"}
                if other:
                    require_conditions(other[0], conditions)
                other.append(conditions)
                factor = target / voltage
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Voltage scaling factor is not representable")
                table["y"] = _scaled(table["y"], factor, "energy")
                table["conditions"]["v_supply_v"] = target
                estimates.append({"branch": direction, "event": event,
                                  "mode": "linear-voltage-estimate", "factor": factor,
                                  "recorded_conditions": recorded,
                                  "requested_conditions": deepcopy(table["conditions"]),
                                  "source": table["source"]})
    summary = {}
    for name in ("cost", "weight_kg"):
        values = [branch[name] for branch in branches.values()]
        currencies = [branch["currency"] for branch in branches.values()]
        compatible = name != "cost" or currencies[0] == currencies[1] is not None
        known = sum(v for v in values if v is not None) if compatible else None
        if known is not None and not np.isfinite(known):
            raise ValueError(f"Composite {name} overflow")
        summary[name] = known if all(v is not None for v in values) else None
        summary[f"known_{name}_sum"] = known
    ratings = [branch["blocking_voltage_v"] for branch in branches.values()]
    summary["blocking_voltage_v"] = min(ratings) if all(v is not None for v in ratings) else None
    summary["currency"] = positive.get("currency") if positive.get("currency") == negative.get("currency") else None
    return {"operation": "signed-composite", "branches": branches, "summary": summary,
            "common_resistance_k_per_w": common, "estimates": estimates,
            "assumptions": ["Positive IGBT and negative diode branches remain separate at zero",
                            "No nonlinear sharing solution; common thermal path is explicitly supplied"]}


def reverse_envelope(channel: dict, diode: dict) -> dict:
    """Exact ideal lower-drop piecewise-linear envelope on overlapping domains."""
    channel, diode = _table(channel, "conduction"), _table(diode, "conduction")
    # Gate biases may differ by construction, but every other named condition must agree.
    conditions = [{k: v for k, v in table["conditions"].items() if k != "v_g_v"}
                  for table in (channel, diode)]
    require_conditions(*conditions)
    low = max(channel["x"][0], diode["x"][0])
    high = min(channel["x"][-1], diode["x"][-1])
    if not low < high:
        raise ValueError("Reverse envelope needs a nonempty overlapping current interval")
    knots = sorted({low, high, *(x for table in (channel, diode)
                                for x in table["x"] if low < x < high)})
    cx, cy, dx, dy = channel["x"], channel["y"], diode["x"], diode["y"]
    crossings = []
    for left, right in zip(knots, knots[1:], strict=False):
        dl = _interpolate(cx, cy, left) - _interpolate(dx, dy, left)
        dr = _interpolate(cx, cy, right) - _interpolate(dx, dy, right)
        if (dl < 0 < dr) or (dr < 0 < dl):
            # Scale before dividing: opposite huge drops must not overflow dl-dr.
            scale = max(abs(dl), abs(dr))
            fraction = (dl / scale) / (dl / scale - dr / scale)
            crossing = left * (1 - fraction) + right * fraction
            if not np.isfinite(crossing) or not left < crossing < right:
                raise ValueError("Reverse-envelope crossing is not representable")
            crossings.append(float(crossing))
    knots = sorted(set(knots + crossings))
    drops = [min(_interpolate(cx, cy, x), _interpolate(dx, dy, x)) for x in knots]
    negative_x, negative_y = [-x for x in reversed(knots)], [-v for v in reversed(drops)]
    return {"operation": "reverse-envelope", "mode": "ideal-lower-drop-envelope-estimate",
            "current_a": negative_x, "voltage_v": negative_y,
            "crossings_magnitude_a": crossings, "overlap_magnitude_a": [low, high],
            "sources": {"channel": channel, "diode": diode},
            "assumptions": ["Selected ideal lower-drop envelope; not simultaneous channel/diode sharing",
                            "Recorded overlap only; no extrapolation; independently declared gate biases"]}


def analyze_assembly(request: dict) -> dict:
    """Canonical pure entry point used by HTTP, browser and future integrations."""
    request = deepcopy(request)
    operation = request["operation"]
    if operation == "parallel-bank":
        result = parallel_bank(request["device"], request["count"])
    elif operation == "signed-composite":
        result = signed_composite(request["positive"], request["negative"],
                                  request["common_resistance_k_per_w"], request.get("voltage_estimate"))
    elif operation == "reverse-envelope":
        if request.get("selected") is not True:
            raise ValueError("Select the ideal lower-drop envelope estimate explicitly")
        result = reverse_envelope(request["channel"], request["diode"])
    else:
        raise ValueError("Unsupported assembly operation")
    result["request"] = request
    return result
