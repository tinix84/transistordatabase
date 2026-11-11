"""Sampled gate hysteresis with explicit thresholds and initial-state evidence."""
from __future__ import annotations

import math
from numbers import Real
from fractions import Fraction


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be an actual finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} exceeds the finite numeric range") from error
    if not math.isfinite(result) or (result == 0 and value != 0):
        raise ValueError(f"{label} exceeds the finite numeric range")
    return result


def _convex(minimum, maximum, fraction):
    # Exact binary64 endpoints keep cancellation distinct from underflow.
    weight = Fraction.from_float(fraction)
    exact = (1 - weight) * Fraction.from_float(minimum) + weight * Fraction.from_float(maximum)
    try:
        result = float(exact)
    except OverflowError as error:
        raise ValueError("Resolved hysteresis threshold exceeds the finite range") from error
    if not math.isfinite(result) or (result == 0 and exact != 0):
        raise ValueError("Resolved hysteresis threshold exceeds the nonzero finite numeric range")
    return result


def hysteretic_gate_states(gate_v, digitizer):
    """Return original-sample states and pre-edge brackets without time inference.

    Strict crossings change state. Equality and the deadband retain the prior
    sampled state. A declared initial state describes sample zero, not a
    hypothetical event before the trace. No source implementation is copied.
    """
    if not isinstance(gate_v, (list, tuple)) or not 2 <= len(gate_v) <= 10000:
        raise ValueError("Hysteresis needs 2 to 10000 gate voltage samples")
    values = [_number(value, "gate_v") for value in gate_v]
    if not isinstance(digitizer, dict) or set(digitizer) - {"thresholds", "initial_state"}:
        raise ValueError("Declare hysteresis thresholds and optional initial_state only")
    thresholds = digitizer.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Declare absolute or relative hysteresis thresholds")
    scale = thresholds.get("scale")
    minimum = maximum = None
    if scale == "absolute":
        if set(thresholds) != {"scale", "low_v", "high_v"}:
            raise ValueError("Absolute hysteresis requires only low_v and high_v")
        low = _number(thresholds["low_v"], "low_v")
        high = _number(thresholds["high_v"], "high_v")
        canonical = {"scale": scale, "low_v": low, "high_v": high}
    elif scale == "relative":
        if set(thresholds) != {"scale", "low_fraction", "high_fraction", "minimum_v", "maximum_v"}:
            raise ValueError("Relative hysteresis requires fractions and explicit calibrated voltage bounds")
        lower = _number(thresholds["low_fraction"], "low_fraction")
        upper = _number(thresholds["high_fraction"], "high_fraction")
        if not 0 <= lower < upper <= 1:
            raise ValueError("Relative fractions must satisfy 0 <= low < high <= 1")
        minimum = _number(thresholds["minimum_v"], "minimum_v")
        maximum = _number(thresholds["maximum_v"], "maximum_v")
        if not minimum < maximum:
            raise ValueError("Calibrated hysteresis bounds must satisfy minimum_v < maximum_v")
        low, high = (_convex(minimum, maximum, value) for value in (lower, upper))
        canonical = {"scale": scale, "low_fraction": lower, "high_fraction": upper,
                     "minimum_v": minimum, "maximum_v": maximum}
    else:
        raise ValueError("Select absolute or relative hysteresis thresholds")
    if not low < high:
        raise ValueError("Resolved hysteresis thresholds must satisfy low_v < high_v")
    initial = digitizer.get("initial_state")
    if initial is not None and type(initial) is not bool:
        raise ValueError("initial_state must be an actual boolean or omitted")
    inferred = False if values[0] < low else True if values[0] > high else None
    if initial is None:
        if inferred is None:
            raise ValueError("First gate sample is inside or on the hysteresis band; declare initial_state")
        state = inferred
        initial_source = "first-sample-below-low" if state is False else "first-sample-above-high"
    else:
        if inferred is not None and inferred != initial:
            raise ValueError("Declared initial_state conflicts with the first gate sample outside the band")
        state = initial
        initial_source = "declared"
    states = [state]
    for value in values[1:]:
        if value > high:
            state = True
        elif value < low:
            state = False
        states.append(state)
    on = [i for i in range(len(states) - 1) if not states[i] and states[i + 1]]
    off = [i for i in range(len(states) - 1) if states[i] and not states[i + 1]]
    return {"policy": {"thresholds": canonical, "initial_state": initial},
            "evidence": {"method": "strict-hysteresis-v1", "threshold_scale": scale,
                "low_v": low, "high_v": high, "calibration_min_v": minimum,
                "calibration_max_v": maximum, "initial_state": states[0],
                "initial_state_source": initial_source, "states": states,
                "turn_on_brackets": on, "turn_off_brackets": off,
                "semantics": "Strict > high and < low crossings. Equality and deadband hold the previous sampled state. No pre-trace event or interpolated switching time.",
                "units": {"low_v": "V", "high_v": "V", "calibration_min_v": "V",
                          "calibration_max_v": "V", "states": "boolean", "brackets": "pre-edge sample index"}}}
