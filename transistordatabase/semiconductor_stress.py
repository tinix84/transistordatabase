"""Independent semiconductor stress proxies with explicit waveform semantics.

These indices do not calculate physical losses, volume, or price. The formulas
follow the four MOSFET index contracts recorded in the ENCO inventory. This
module does not import ENCO or infer gate transitions.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
from numbers import Real


MAX_SAMPLES = 20000
MAX_TRANSITIONS = 20000


def _number(value, label: str) -> float:
    """Accept finite real numbers without string or Boolean conversion."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return value


def _product(*values: float) -> float:
    """Reject arithmetic overflow and positive-magnitude underflow."""
    try:
        result = math.prod(values)
    except OverflowError as exc:
        raise ValueError("The result exceeds the finite numerical range") from exc
    if not math.isfinite(result) or (result == 0 and all(v != 0 for v in values)):
        raise ValueError("The result exceeds the finite numerical range")
    return result


def _digest(value: dict) -> str:
    """Hash canonical JSON evidence without a dependency on file state."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Source and condition names must contain valid Unicode") from exc
    return hashlib.sha256(encoded).hexdigest()


def _metric(value: float | None, unit: str, reason: str | None = None) -> dict:
    return {"value": value, "unit": unit,
            "status": "unknown" if value is None else "calculated", "reason": reason}


def calculate_stress_indices(
    time_s, current_a, voltage_v, *, source: str, conditions: dict,
    rds_on_ohm: float | None = None, transitions: list[dict] | None = None,
) -> dict:
    """Calculate proxies over the complete recorded piecewise-linear interval.

    Current and voltage share one strictly increasing time vector. Voltage keeps
    its sign. The peak is max(V), not max(abs(V)). Each transition is an explicit
    frequency-weighted record. Its I and V values use bounded linear sampling.
    An empty transition list means a declared absence of transitions. None means
    unknown transitions. No boundary wrap, extrapolation, or dead-time rule applies.
    """
    if not isinstance(source, str) or not source.strip() or len(source) > 2000:
        raise ValueError("Supply a nonblank source with at most 2000 characters")
    if not isinstance(conditions, dict) or len(conditions) > 100:
        raise ValueError("Supply a conditions object with at most 100 entries")
    normalized_conditions = {}
    for key, value in conditions.items():
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise ValueError("Condition names must contain 1 to 200 nonblank characters")
        normalized_conditions[key] = None if value is None else _number(value, key)
    vectors = []
    for label, data in (("time_s", time_s), ("current_a", current_a),
                        ("voltage_v", voltage_v)):
        if not isinstance(data, (list, tuple)) or not 2 <= len(data) <= MAX_SAMPLES:
            raise ValueError(f"{label} must contain 2 to {MAX_SAMPLES} samples")
        vectors.append([_number(v, label) for v in data])
    time, current, voltage = vectors
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Time, current, and voltage must have matching sample counts")
    if any(b <= a for a, b in zip(time[:-1], time[1:], strict=True)):
        raise ValueError("Time must increase strictly")
    duration = time[-1] - time[0]
    if not math.isfinite(duration):
        raise ValueError("The time interval exceeds the finite numerical range")
    # Integrate the square of each linear segment analytically. Scaling avoids
    # squaring the input before the normalized second moment is available.
    scale = max(abs(v) for v in current)
    if scale:
        scaled = [v / scale for v in current]
        if any(x == 0 and original != 0
               for x, original in zip(scaled, current, strict=True)):
            raise ValueError("The current ratio exceeds the finite numerical range")
        terms = []
        for a, b, x, y in zip(time[:-1], time[1:], scaled[:-1], scaled[1:],
                              strict=True):
            term = ((b - a) / duration) * (x*x + x*y + y*y) / 3
            if term == 0 and (x != 0 or y != 0):
                raise ValueError("The weighted current exceeds the finite numerical range")
            terms.append(term)
        moment = math.fsum(terms)
        mean_square = _product(scale, scale, moment)
        rms = _product(scale, math.sqrt(moment))
    else:
        mean_square = rms = 0.0
    peak = max(voltage)
    volume = _product(mean_square, peak, peak)
    if rds_on_ohm is not None:
        rds_on_ohm = _number(rds_on_ohm, "rds_on_ohm")
        if rds_on_ohm <= 0:
            raise ValueError("rds_on_ohm must be positive")
    cost = None
    cost_reason = "The on-state resistance is unknown"
    if rds_on_ohm is not None:
        if peak == 0:
            cost_reason = "The signed peak voltage is zero. The cost proxy is undefined"
        else:
            cost = _product(1 / _product(rds_on_ohm, peak, peak))
            cost_reason = None
    records = None
    samples = []
    switching = None
    if transitions is not None:
        if not isinstance(transitions, list) or len(transitions) > MAX_TRANSITIONS:
            raise ValueError(f"Supply at most {MAX_TRANSITIONS} transition records")
        records = []
        for record in transitions:
            if not isinstance(record, dict) or set(record) != {
                "kind", "time_s", "frequency_hz"
            }:
                raise ValueError("Each transition requires kind, time_s, and frequency_hz")
            if record["kind"] not in ("turn_on", "turn_off"):
                raise ValueError("Transition kind must be turn_on or turn_off")
            stamp = _number(record["time_s"], "Transition time_s")
            frequency = _number(record["frequency_hz"], "Transition frequency_hz")
            if not time[0] <= stamp <= time[-1]:
                raise ValueError("Transition time must be within the recorded interval")
            if frequency < 0:
                raise ValueError("Transition frequency must be nonnegative")
            index = min(bisect_right(time, stamp) - 1, len(time) - 2)
            interval = time[index + 1] - time[index]
            left_weight = (time[index + 1] - stamp) / interval
            right_weight = (stamp - time[index]) / interval
            if ((left_weight == 0 and stamp != time[index + 1])
                    or (right_weight == 0 and stamp != time[index])):
                raise ValueError("The transition fraction exceeds the finite numerical range")
            # Independent endpoint distances preserve a small contribution near
            # either endpoint. Computing one weight as 1-other loses that term.
            i = (_product(left_weight, current[index])
                 + _product(right_weight, current[index + 1]))
            v = (_product(left_weight, voltage[index])
                 + _product(right_weight, voltage[index + 1]))
            contribution = _product(frequency, i, i, v)
            normalized = {"kind": record["kind"], "time_s": stamp,
                          "frequency_hz": frequency}
            records.append(normalized)
            samples.append({**normalized, "current_a": i, "voltage_v": v,
                            "contribution": contribution})
        try:
            switching = math.fsum(row["contribution"] for row in samples)
        except OverflowError as exc:
            raise ValueError("The switching sum exceeds the finite numerical range") from exc
        if not math.isfinite(switching):
            raise ValueError("The switching sum exceeds the finite numerical range")
    inputs = {"source": source.strip(), "conditions": normalized_conditions,
              "time_s": time, "current_a": current, "voltage_v": voltage,
              "rds_on_ohm": rds_on_ohm, "transitions": records}
    result = {
        "current_rms_a": rms, "current_mean_square_a2": mean_square,
        "peak_voltage_v": peak, "duration_s": duration, "sample_count": len(time),
        "metrics": {
            "conduction": _metric(mean_square, "A^2"),
            "volume": _metric(volume, "A^2 V^2"),
            "switching": _metric(switching, "Hz A^2 V",
                                  "The transition records are unknown"
                                  if switching is None else None),
            "cost": _metric(cost, "1/(ohm V^2)", cost_reason),
        },
        "transition_samples": samples,
        "definitions": {
            "current": "Exact integral of piecewise-linear I(t)^2 divided by the full recorded duration",
            "peak": "Maximum signed voltage sample, equal to the piecewise-linear maximum. No absolute-value conversion",
            "conduction": "Mean-square current",
            "volume": "Mean-square current times signed peak voltage squared",
            "switching": "Sum of frequency_hz times sampled current squared times signed sampled voltage",
            "cost": "1/(rds_on_ohm * signed_peak_voltage^2)",
            "sampling": "Bounded piecewise-linear interpolation at each explicit transition time, including endpoints",
            "frequency": "Each record supplies its own weight in Hz. No inferred event rate or cycle normalization",
        },
        "scope": "Stress proxies only. These indices are not physical loss in W, volume in m^3, or currency. Signed switching values can be negative. No ENCO transition-detector equivalence is claimed",
    }
    return {"inputs": inputs, "result": result,
            "evidence": {"method": "semiconductor-stress-v1",
                         "input_sha256": _digest(inputs),
                         "result_sha256": _digest(result)}}
