"""Solve passive reverse branches at a common voltage within recorded domains."""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
import math

from transistordatabase.current_allocation import _number, _utf8


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"Supply nonblank {name} of at most 2000 characters")
    _utf8(value)
    return value.strip()


def _conditions(value):
    if not isinstance(value, dict):
        raise ValueError("Supply recorded and requested branch conditions")
    result = {}
    for key, number in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Condition names must be nonblank strings")
        _utf8(key)
        result[key] = None if number is None else _number(number, "conditions")
    if result.get("junction_temperature_c") is None or "gate_bias_v" not in result:
        raise ValueError("Declare a known branch temperature and an explicit gate bias condition")
    return result


def _branch(value, kind, temperature):
    keys = {"kind", "current_a", "voltage_v", "source", "isolation_evidence",
            "recorded_conditions", "requested_conditions"}
    if not isinstance(value, dict) or set(value) != keys or value["kind"] != kind:
        raise ValueError(f"Supply an independent {kind} branch with all source evidence")
    result = {"kind": kind, "source": _text(value["source"], "branch source"),
              "isolation_evidence": _text(value["isolation_evidence"], "branch isolation evidence")}
    for key in ("recorded_conditions", "requested_conditions"):
        result[key] = _conditions(value[key])
    if result["recorded_conditions"] != result["requested_conditions"]:
        raise ValueError("Recorded and requested branch conditions must match exactly")
    if result["recorded_conditions"]["junction_temperature_c"] != temperature:
        raise ValueError("Both branch temperatures must match the common junction temperature")
    for key in ("current_a", "voltage_v"):
        values = value[key]
        if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 10000:
            raise ValueError("Each branch requires 2 to 10000 aligned current and voltage points")
        result[key] = [_number(number, key) for number in values]
        if any(number < 0 for number in result[key]):
            raise ValueError("Branch current and voltage magnitudes must be nonnegative")
    if len(result["current_a"]) != len(result["voltage_v"]):
        raise ValueError("Branch current and voltage points must have equal lengths")
    if any(b <= a for a, b in zip(result["current_a"], result["current_a"][1:], strict=False)):
        raise ValueError("Branch current magnitudes must increase strictly")
    if any(b < a for a, b in zip(result["voltage_v"], result["voltage_v"][1:], strict=False)):
        raise ValueError("Passive branch voltage magnitudes must not decrease")
    return result


def _rational(value):
    """Use exact arithmetic for the canonical decimal values without zero tolerances."""
    return Fraction(str(value))


def _float(value):
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("A derived result exceeds the finite numerical range") from exc
    if not math.isfinite(result) or (result == 0 and value != 0):
        raise ValueError("A derived result exceeds the finite numerical range")
    return result


def _interpolator(branch):
    currents = [_rational(value) for value in branch["current_a"]]
    voltages = [_rational(value) for value in branch["voltage_v"]]

    def voltage(current):
        if not currents[0] <= current <= currents[-1]:
            raise ValueError("The current exceeds the recorded branch domain")
        index = min(bisect_right(currents, current) - 1, len(currents) - 2)
        return (voltages[index] + (current - currents[index])
                * (voltages[index + 1] - voltages[index])
                / (currents[index + 1] - currents[index]))
    return currents, voltage


def solve_reverse_current_sharing(*, selected, current_a, junction_temperature_c,
                                  compatible_operating_state, channel, diode,
                                  mode="strict-recorded"):
    """Return a unique recorded root or an explicitly selected blocking estimate.

    Branch data must isolate channel and diode currents in compatible operating
    states. Terminal gate-on curves can contain both paths and are not suitable.
    No zero endpoint, current domain, or channel blocking rule is inferred.
    """
    if selected is not True or compatible_operating_state is not True:
        raise ValueError("Explicitly select the model and confirm compatible independent branch operating states")
    if mode not in ("strict-recorded", "ideal-diode-blocking"):
        raise ValueError("Select strict-recorded or ideal-diode-blocking mode")
    current = _number(current_a, "current_a")
    if current > 0:
        raise ValueError("The signed reverse current must be nonpositive")
    temperature = _number(junction_temperature_c, "junction_temperature_c")
    channel = _branch(channel, "channel-only", temperature)
    diode = _branch(diode, "diode-only", temperature)
    inputs = {"selected": True, "mode": mode, "current_a": current,
              "junction_temperature_c": temperature, "compatible_operating_state": True,
              "channel": channel, "diode": diode}
    c, vc = _interpolator(channel)
    d, vd = _interpolator(diode)
    total = -_rational(current)
    lower, upper = max(c[0], total - d[-1]), min(c[-1], total - d[0])
    bounds = ({"lower_a": _float(lower), "upper_a": _float(upper)}
              if lower <= upper else None)
    knots = sorted({lower, upper} | {x for x in c if lower <= x <= upper}
                   | {total - x for x in d if lower <= total - x <= upper}) if bounds else []
    values = [vc(x) - vd(total - x) for x in knots]
    roots = {x for x, value in zip(knots, values, strict=True) if value == 0}
    flats = []
    for a, b, fa, fb in zip(knots, knots[1:], values, values[1:], strict=False):
        if fa == fb == 0:
            if flats and flats[-1][1] == a:
                flats[-1][1] = b
            else:
                flats.append([a, b])
        elif fa * fb < 0:
            roots.add(a - fa * (b - a) / (fb - fa))
    # The passive voltage difference is nondecreasing. Its zero set is either
    # one isolated point or one interval, so a flat root excludes isolated roots.
    isolated = [] if flats else sorted(roots)
    blocking = None
    if mode == "ideal-diode-blocking" and d[0] == 0 and c[0] <= total <= c[-1]:
        drop = vc(total)
        if 0 <= drop <= vd(Fraction(0)):
            blocking = total
    # A threshold root and its identical blocking endpoint count only once.
    candidates = set(isolated)
    if blocking is not None:
        candidates.add(blocking)
    coordinate_error = None
    try:
        flat_coordinates = [[_float(x) for x in row] for row in flats]
        root_coordinates = [_float(x) for x in sorted(candidates)]
    except ValueError as exc:
        flat_coordinates, root_coordinates = [], []
        coordinate_error = str(exc)
    result = {
        "status": "unsupported", "support": "none",
        "diagnostic": "No common-voltage solution exists within the recorded branch domains",
        **dict.fromkeys(("channel_current_a", "diode_current_a", "current_sum_a",
                         "current_residual_a", "voltage_v", "channel_static_power_w",
                         "diode_static_power_w", "total_static_power_w", "blocking_margin_v")),
        "feasible_channel_current_magnitude_a": bounds,
        "root_evidence": {"isolated_root_count": len(isolated),
                          "flat_intervals_a": flat_coordinates,
                          "candidate_channel_current_magnitude_a": root_coordinates,
                          "evaluated_knot_count": len(knots), "voltage_residual_v": None},
        "equations": ["J = -I_device", "I_channel_magnitude + I_diode_magnitude = J",
                      "Recorded root: V_channel(I_channel_magnitude) = V_diode(I_diode_magnitude)",
                      "Selected ideal-diode blocking: I_diode = 0 and 0 <= V_channel(J) <= V_diode(0)",
                      "Vds = -voltage_drop", "P_static_branch = Vds * I_branch"],
        "scope": "Calculated static operating point only. Independent isolated branches and compatible operating states are declared inputs. No dynamic loss or accuracy claim.",
        "units": {"current": "A", "voltage": "V", "static_power": "W", "junction_temperature": "degC"},
    }
    if coordinate_error:
        result["diagnostic"] = coordinate_error
    elif flats or len(candidates) > 1:
        result.update(status="ambiguous", diagnostic="The branch data permit a continuum or multiple distinct solutions")
    elif candidates:
        root = next(iter(candidates))
        is_blocking = root == blocking and root not in isolated
        drop = vc(root)
        try:
            ic, id_ = _float(-root), _float(-(total - root))
            voltage = _float(-drop)
            # Powers and conservation evidence describe the actual returned floats.
            pc = _float(Fraction(voltage) * Fraction(ic))
            pd = _float(Fraction(voltage) * Fraction(id_))
            current_sum = _float(Fraction(ic) + Fraction(id_))
            residual = _float(Fraction(ic) + Fraction(id_) - Fraction(current))
            power = _float(Fraction(pc) + Fraction(pd))
            margin = _float(vd(Fraction(0)) - drop) if is_blocking else None
            voltage_residual = None if is_blocking else _float(vc(-_rational(ic)) - vd(-_rational(id_)))
            result.update(status="supported", support="estimate" if is_blocking else "recorded",
                          diagnostic=("Explicit ideal-diode blocking estimate" if is_blocking else "Unique isolated common-voltage root within recorded domains"),
                          channel_current_a=ic, diode_current_a=id_, current_sum_a=current_sum,
                          current_residual_a=residual, voltage_v=voltage, channel_static_power_w=pc,
                          diode_static_power_w=pd, total_static_power_w=power, blocking_margin_v=margin)
            result["root_evidence"]["voltage_residual_v"] = voltage_residual
        except ValueError as exc:
            result["diagnostic"] = str(exc)

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    return {"inputs": inputs, "result": result,
            "evidence": {"method": "bounded-reverse-sharing-v1", "input_sha256": digest(inputs),
                         "result_sha256": digest(result)}}
