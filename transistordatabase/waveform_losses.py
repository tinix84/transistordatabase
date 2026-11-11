"""Validated signed waveform losses, exact conduction segments and event evidence.

All calculations use SI units. Current, optional duty and waveform signals are
piecewise linear between supplied samples. Gate evidence detects events only;
it does not silently mask conduction. No curve is clamped or extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from math import frexp, ldexp
from typing import Optional

import numpy as np

from transistordatabase.gate_states import hysteretic_gate_states
import numpy.typing as npt

from transistordatabase.condition_tables import MeasuredTable, require_conditions
from transistordatabase.core.models import (
    ChannelCharacteristics, LinearizedModel, SwitchingLossData,
)


class WaveformResourceLimitError(ValueError):
    """Exact output expansion exceeds the declared workbench resource budget."""


@dataclass
class WaveformData:
    """Compatibility waveform; optional voltage is measured device voltage."""

    time: npt.NDArray[np.float64]
    current: npt.NDArray[np.float64]
    voltage: Optional[npt.NDArray[np.float64]] = None
    gate: Optional[npt.NDArray[np.float64]] = None


@dataclass
class LossResult:
    """Compatibility result; cycle count is deprecated/unknown, with separate event counts."""

    p_cond: float
    p_sw_on: float
    p_sw_off: float
    p_total: float
    n_cycles: Optional[int]
    switching_instants: npt.NDArray[np.float64]
    n_on: int = 0
    n_off: int = 0
    n_cycles_definition: str = "Deprecated: unknown; event counts do not establish complete periods"


def _finite(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _vector(value, label, size=None):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError(f"{label} must contain at least two finite 1D samples")
    if size is not None and len(value) != size:
        raise ValueError(f"{label} must align with time_s")
    return value.copy()


def _aligned(value, label, size, *, nonnegative=False):
    if np.ndim(value) == 0:
        result = np.full(size, _finite(value, label))
    else:
        result = _vector(value, label, size)
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{label} must be nonnegative")
    return result


def _waveform(request):
    time = _vector(request["time_s"], "time_s")
    current = _vector(request["current_a"], "current_a", len(time))
    duration = _finite(time[-1] - time[0], "duration_s")
    if duration <= 0 or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must increase strictly with positive full duration")
    signals = {"time_s": time, "current_a": current}
    for name in ("gate_v", "device_voltage_v", "bus_voltage_v"):
        if request.get(name) is not None:
            signals[name] = (_aligned(request[name], name, len(time), nonnegative=True)
                             if name == "bus_voltage_v" else
                             _vector(request[name], name, len(time)))
    return signals, duration


def _source(model):
    source = model.get("source", "")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Every model needs a nonblank source identifier")
    return source.strip()


def _prepare_model(model, *, energy=False):
    """Validate one model once; all table lookup policy stays in MeasuredTable."""
    if model is None:
        return None
    source = _source(model)
    mode = model.get("mode")
    if mode == "linear-drop-estimate" and not energy:
        if model.get("selected") is not True:
            raise ValueError("Select the linear-drop estimate explicitly")
        offset = _finite(model["offset_v"], "offset_v")
        resistance = _finite(model["resistance_ohm"], "resistance_ohm")
        if min(offset, resistance) < 0:
            raise ValueError("Voltage offset and resistance must be nonnegative")
        conditions = model.get("conditions", {})
        require_conditions(conditions, conditions)
        return {"mode": mode, "source": source, "offset": offset,
                "resistance": resistance, "conditions": conditions}
    if mode not in ({"recorded", "voltage-scaling-estimate"} if energy else {"recorded"}):
        raise ValueError("Unsupported model mode")
    table = MeasuredTable("energy" if energy else "channel", model["current_a"],
                          model["energy_j" if energy else "drop_v"],
                          model["recorded_conditions"], source)
    requested = dict(model["requested_conditions"])
    if mode == "voltage-scaling-estimate":
        if model.get("selected") is not True:
            raise ValueError("Select the voltage-scaling estimate explicitly")
        recorded_voltage = table.conditions.get("v_supply_v")
        requested_voltage = requested.get("v_supply_v")
        if recorded_voltage is None or requested_voltage is None:
            raise ValueError("Voltage scaling requires known recorded/requested v_supply_v")
        if min(recorded_voltage, requested_voltage) <= 0:
            raise ValueError("Voltage scaling requires positive supply voltages")
        comparison = dict(requested, v_supply_v=recorded_voltage)
        require_conditions(table.conditions, comparison)
    else:
        require_conditions(table.conditions, requested)
    return {"mode": mode, "source": source, "table": table, "requested": requested}


def _drop(model, magnitude):
    if model is None:
        raise ValueError("Missing conduction model for a current direction")
    if model["mode"] == "linear-drop-estimate":
        return model["offset"] + model["resistance"] * magnitude
    return model["table"].evaluate(magnitude, model["requested"])


def _omission(spec, label):
    if not spec.get("omit", False):
        return None
    reason = spec.get("omission_reason", "")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"Explicit omission of {label} requires a reason")
    return {"status": "omitted", "reason": reason.strip(), "energy_j": None,
            "power_w": None}


def _component(energy, duration, **extra):
    return {"status": "computed", "energy_j": float(energy),
            "power_w": float(energy / duration), **extra}


def _conduction(signals, duration, spec):
    omitted = _omission(spec, "conduction")
    if omitted is not None:
        return omitted, [], []
    models = {direction: _prepare_model(spec.get(direction))
              for direction in ("positive", "negative")}
    if not any(models.values()):
        raise ValueError("Conduction requires a model or explicit omission")
    time, current = signals["time_s"], signals["current_a"]
    duty = _aligned(spec.get("duty", 1.0), "duty", len(time))
    if np.any((duty < 0) | (duty > 1)):
        raise ValueError("Conduction duty must lie in [0, 1]")
    segments, plot = [], []
    directional = {"positive": 0.0, "negative": 0.0, "zero": 0.0}
    cumulative = 0.0
    for index, (left, right) in enumerate(zip(time[:-1], time[1:], strict=True)):
        ia, ib = current[index:index + 2]
        cuts = [0.0, 1.0]
        if ia != ib:
            if (ia < 0 < ib) or (ib < 0 < ia):
                cuts.append(float(-ia / (ib - ia)))
            for direction, model in models.items():
                if model is None or "table" not in model:
                    continue
                polarity = 1 if direction == "positive" else -1
                knots = model["table"].x
                low, high = sorted((polarity * ia, polarity * ib))
                for knot in knots[np.searchsorted(knots, low):np.searchsorted(knots, high, side="right")]:
                    fraction = (polarity * knot - ia) / (ib - ia)
                    if 0 < fraction < 1:
                        cuts.append(float(fraction))
        cuts = sorted(set(cuts))
        if len(segments) + len(cuts) - 1 > 10000:
            raise WaveformResourceLimitError(
                "Resource limit: at most 10000 exact segments / 90000 plot records; "
                "reduce input size explicitly (no interval was truncated)")
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            start, end = left + (right - left) * a, left + (right - left) * b
            ja, jb = ia + (ib - ia) * a, ia + (ib - ia) * b
            midpoint = (ja + jb) / 2
            direction = "positive" if midpoint > 0 else "negative" if midpoint < 0 else "zero"
            da, db = duty[index] + (duty[index + 1] - duty[index]) * a, duty[index] + (duty[index + 1] - duty[index]) * b
            if direction == "zero" or (da == 0 and db == 0):
                coefficients, source = np.zeros(4), "Zero current or explicitly zero duty"
            else:
                model = models[direction]
                magnitude_a, magnitude_b = abs(ja), abs(jb)
                va, vb = _drop(model, magnitude_a), _drop(model, magnitude_b)
                coefficients = np.polynomial.polynomial.polymul(
                    np.polynomial.polynomial.polymul(
                        [magnitude_a, magnitude_b - magnitude_a], [va, vb - va]),
                    [da, db - da])
                coefficients = np.pad(coefficients, (0, 4 - len(coefficients)))
                source = model["source"]
            energy = float((end - start) * np.sum(coefficients / np.arange(1, 5)))
            segment = {"start_s": float(start), "end_s": float(end),
                       "sample_interval_index": index, "direction": direction,
                       "power_coefficients_w": coefficients.tolist(), "energy_j": energy,
                       "cumulative_start_j": cumulative,
                       "cumulative_end_j": cumulative + energy, "source": source}
            segments.append(segment)
            for u in np.linspace(0, 1, 9):
                plot.append({"time_s": float(start + (end - start) * u),
                             "power_w": float(np.polynomial.polynomial.polyval(u, coefficients)),
                             "cumulative_j": float(cumulative + (end - start) * np.sum(
                                 coefficients * u ** np.arange(1, 5) / np.arange(1, 5)))})
            cumulative += energy
            directional[direction] += energy
    return _component(cumulative, duration, directional_energy_j=directional), segments, plot


def _event_indices(signals, policy, *, gate_values=None):
    mode = policy.get("mode")
    if mode != "hysteresis" and policy.get("digitizer") is not None:
        raise ValueError("Hysteresis settings require the hysteresis event mode")
    if mode == "hysteresis":
        if any(policy.get(name) is not None for name in ("threshold_v", "turn_on", "turn_off")):
            raise ValueError("Hysteresis and threshold or mask event detection are mutually exclusive")
        if "gate_v" not in signals:
            raise ValueError("Gate signal required for hysteresis event detection")
        values = signals["gate_v"].tolist() if gate_values is None else gate_values
        digitization = hysteretic_gate_states(values, policy.get("digitizer"))
        evidence = digitization["evidence"]
        return {"turn_on": np.asarray(evidence["turn_on_brackets"], dtype=int),
                "turn_off": np.asarray(evidence["turn_off_brackets"], dtype=int),
                "digitization": evidence}
    if mode == "gate":
        if any(policy.get(name) is not None for name in ("turn_on", "turn_off")):
            raise ValueError("Gate detection and explicit event masks are mutually exclusive")
        if "gate_v" not in signals:
            raise ValueError("Gate signal required for gate event detection")
        threshold = _finite(policy["threshold_v"], "gate threshold")
        high = signals["gate_v"] > threshold
        return {"turn_on": np.flatnonzero(~high[:-1] & high[1:]),
                "turn_off": np.flatnonzero(high[:-1] & ~high[1:])}
    if mode == "masks":
        if policy.get("threshold_v") is not None:
            raise ValueError("Explicit event masks and a gate threshold are mutually exclusive")
        result = {}
        for name in ("turn_on", "turn_off"):
            mask = np.asarray(policy[name])
            if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(signals["time_s"]) - 1:
                raise ValueError("Event masks must be boolean arrays of length N-1")
            result[name] = np.flatnonzero(mask)
        if np.intersect1d(result["turn_on"], result["turn_off"]).size:
            raise ValueError("Turn-on and turn-off masks cannot overlap")
        return result
    if mode == "omitted":
        if any(policy.get(name) is not None for name in ("turn_on", "turn_off", "threshold_v")):
            raise ValueError("Omitted event detection cannot also request gate or mask detection")
        reason = policy.get("omission_reason", "")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Omitted event detection requires a reason")
        return None
    raise ValueError("Choose gate events, explicit masks or explicit event omission")


def prepare_switching_models(component):
    """Prepare independently conditioned directional energy data for repeated use."""
    return {direction: _prepare_model(component.get(direction), energy=True)
            for direction in ("positive", "negative")}


def _positive_product_ratio(value, numerator, denominator):
    """Evaluate finite nonnegative a*b/c without avoidable intermediate overflow."""
    if value == 0 or numerator == 0:
        return 0.0
    a, ea = frexp(value)
    b, eb = frexp(numerator)
    c, ec = frexp(denominator)
    result = ldexp(a * b / c, ea + eb - ec)
    if not np.isfinite(result) or result == 0:
        raise ValueError("Energy scaling is outside representable positive SI precision")
    return result


def evaluate_switching_energy(models, current_a, bus_voltage_v=None):
    """Shared bounded signed energy lookup for events and held PWM estimates.

    Prepared models come from :func:`prepare_switching_models`. No event or
    switching frequency is inferred here. Finite E(0) uses the positive branch.
    """
    current = _finite(current_a, "switching current_a")
    direction = "negative" if current < 0 else "positive"
    model = models[direction]
    if model is None:
        raise ValueError(f"Missing {direction} switching energy model")
    table, requested = model["table"], dict(model["requested"])
    if bus_voltage_v is not None:
        voltage = _finite(bus_voltage_v, "bus_voltage_v")
        if voltage < 0:
            raise ValueError("bus_voltage_v must be nonnegative")
        requested["v_supply_v"] = voltage
    if model["mode"] == "voltage-scaling-estimate":
        reference = table.conditions["v_supply_v"]
        query_voltage = requested.get("v_supply_v")
        if query_voltage is None or query_voltage <= 0:
            raise ValueError("Voltage scaling requires positive bus voltage")
        require_conditions(table.conditions, dict(requested, v_supply_v=reference))
        energy = _positive_product_ratio(
            table.evaluate(abs(current), table.conditions), query_voltage, reference)
    else:
        energy = table.evaluate(abs(current), requested)
    result = {"current_a": current, "direction": direction, "energy_j": float(energy),
              "source": model["source"], "mode": model["mode"],
              "recorded_conditions": dict(table.conditions),
              "requested_conditions": requested}
    _assert_finite_tree(result)
    return result


def _switching(signals, duration, spec, indices):
    components, events = {}, []
    for name in ("turn_on", "turn_off"):
        component = spec[name]
        omitted = _omission(component, name)
        if omitted is not None:
            components[name] = omitted
            continue
        if indices is None:
            raise ValueError("Requested switching losses require event evidence")
        models = prepare_switching_models(component)
        if not any(models.values()):
            raise ValueError(f"{name} requires an energy model or explicit omission")
        total = 0.0
        for index in indices[name]:
            lookup = evaluate_switching_energy(
                models, signals["current_a"][index],
                signals["bus_voltage_v"][index] if "bus_voltage_v" in signals else None)
            total += lookup["energy_j"]
            events.append({"event": name, "sample_index": int(index),
                           "time_s": float(signals["time_s"][index]),
                           "bracket_s": signals["time_s"][index:index + 2].tolist(),
                           **lookup,
                           "sampling_convention": "pre-edge sample i for bracket [i,i+1]"})
        count = len(indices[name])
        components[name] = _component(total, duration, event_count=count,
                                      event_rate_hz=float(count / duration))
    return components, sorted(events, key=lambda event: (event["sample_index"], event["event"]))


def waveform_scalar_statistics(time_s, values, *, include_samples=True):
    """Return scalar sample statistics and exact piecewise-linear mean and RMS."""
    time, values = _vector(time_s, "time_s"), _vector(values, "values")
    if len(time) != len(values) or np.any(np.diff(time) <= 0):
        raise ValueError("Statistics require aligned strictly increasing samples")
    dt, duration = np.diff(time), _finite(time[-1] - time[0], "duration")
    scale = float(np.max(abs(values)))
    normalized = values / scale if scale else values
    a, b = normalized[:-1], normalized[1:]
    mean = scale * np.sum((dt / duration) * (a + b) / 2)
    rms = scale * np.sqrt(np.sum((dt / duration) * (a * a + a * b + b * b) / 3))
    if not np.isfinite([mean, rms]).all():
        raise ValueError("Statistics overflow")
    result = {"mean": float(mean), "rms": float(rms),
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "sample_count": len(values)}
    if include_samples:
        sample_mean = float(np.mean(normalized))
        result.update(arithmetic_mean=float(scale * sample_mean),
            sample_standard_deviation=float(scale * np.sqrt(
                np.sum((normalized - sample_mean) ** 2) / (len(values) - 1))),
            discrete_rms=float(scale * np.sqrt(np.mean(normalized ** 2))))
    _assert_finite_tree(result)
    return result


def waveform_statistics(time_s, values):
    """Exact time-weighted mean and RMS of piecewise-linear samples."""
    time, values = _vector(time_s, "time_s"), _vector(values, "values")
    scalar = waveform_scalar_statistics(time, values, include_samples=False)
    result = {"mean": scalar["mean"], "rms": scalar["rms"],
            "minimum": scalar["minimum"], "maximum": scalar["maximum"],
            "interval_derivative": ((values[1:] - values[:-1]) / np.diff(time)).tolist(),
            "interval_start_s": time[:-1].tolist(), "interval_end_s": time[1:].tolist(),
            "method": "Exact time-weighted piecewise-linear integrals"}
    _assert_finite_tree(result)
    return result


def series_spectrum(time_s, values, *, window="rectangular", detrend="none",
                    relative_threshold=0.1):
    """Uniform-sample rFFT with explicit window/normalization and Parseval evidence.

    The discrete sequence period is N*dt, not the N-1 interval waveform duration.
    Amplitude uses coherent-gain correction; Parseval uses raw windowed samples.
    No resampling or continuous-time RMS equivalence is implied.
    """
    if (isinstance(relative_threshold, (bool, np.bool_)) or
            not isinstance(relative_threshold, (int, float, np.integer, np.floating)) or
            not np.isfinite(relative_threshold) or not 0 <= relative_threshold <= 1):
        raise ValueError("FFT relative_threshold must be finite and within [0, 1]")
    time, values = _vector(time_s, "time_s"), _vector(values, "values")
    if len(time) != len(values) or np.any(np.diff(time) <= 0):
        raise ValueError("Spectrum requires aligned strictly increasing samples")
    spacing = np.diff(time)
    dt = _finite(spacing[0], "sample_interval_s")
    if not np.allclose(spacing, dt, rtol=1e-9, atol=0):
        raise ValueError("FFT requires uniform samples; no implicit resampling")
    if window not in {"rectangular", "hann"} or detrend not in {"none", "mean"}:
        raise ValueError("Unsupported FFT window or detrending")
    scale = float(np.max(abs(values)))
    values = values / scale if scale else values.copy()
    values = values - np.mean(values) if detrend == "mean" else values
    weights = np.ones(len(values)) if window == "rectangular" else np.hanning(len(values))
    gain = float(np.mean(weights))
    if gain <= 0:
        raise ValueError("FFT window has zero coherent gain; use more samples")
    transformed = np.fft.rfft(values * weights)
    multiplicity = np.full(len(transformed), 2.0)
    multiplicity[0] = 1
    if len(values) % 2 == 0:
        multiplicity[-1] = 1
    amplitude = (abs(transformed) * multiplicity / (len(values) * gain)) * scale
    phasor = (transformed * (multiplicity / (len(values) * gain))) * scale
    phase = [None if value == 0 else float(np.arctan2(value.imag, value.real))
             for value in phasor]
    peak = float(np.max(amplitude))
    selected = (np.flatnonzero(amplitude > peak * relative_threshold)[:10].tolist()
                if peak else [])
    bin_square = abs(transformed) ** 2 * multiplicity / len(values) ** 2
    result = {"frequency_hz": np.fft.rfftfreq(len(values), dt).tolist(),
            "magnitude": amplitude.tolist(),
            "real": phasor.real.tolist(), "imaginary": phasor.imag.tolist(),
            "phase_rad": phase, "phase_reference": "first supplied sample",
            "relative_threshold": float(relative_threshold),
            "selected_bin_indices": selected,
            "windowed_bin_mean_square_normalized": bin_square.tolist(),
            "mean_square_amplitude_scale": scale,
            "mean_square_normalization": "Multiply normalized bins by amplitude scale squared; RMS is evaluated before squaring that scale",
            "discrete_input_rms": float(np.sqrt(np.mean(values ** 2)) * scale),
            "discrete_input_rms_definition": "Discrete samples after selected detrending, before windowing; separate from continuous time-weighted RMS",
            "windowed_discrete_rms": float(np.sqrt(np.mean((values * weights) ** 2)) * scale),
            "parseval_rms": float(np.sqrt(np.sum(bin_square)) * scale),
            "n": len(values), "sample_interval_s": dt, "sequence_period_s": len(values) * dt,
            "window": window, "detrend": detrend, "coherent_gain": gain,
            "normalization": "One-sided peak amplitude; double interior bins only",
            "resampling": "none"}
    _assert_finite_tree(result)
    return result


def waveform_spectrum(time_s, values, *, window="rectangular", detrend="none",
                      relative_threshold=0.1):
    """Expose the generic FFT with the established current-specific SI names."""
    result = series_spectrum(time_s, values, window=window, detrend=detrend,
                             relative_threshold=relative_threshold)
    for generic, current in (("magnitude", "amplitude_a"), ("real", "real_a"),
                             ("imaginary", "imaginary_a"),
                             ("mean_square_amplitude_scale", "mean_square_amplitude_scale_a"),
                             ("discrete_input_rms", "discrete_input_rms_a"),
                             ("windowed_discrete_rms", "windowed_discrete_rms_a"),
                             ("parseval_rms", "parseval_rms_a")):
        result[current] = result.pop(generic)
    result["mean_square_normalization"] = (
        "Multiply normalized bins by amplitude_scale_a squared; RMS is evaluated before squaring that SI scale")
    return result


def _assert_finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_tree(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("Waveform result overflow; use supported finite magnitudes")


def _plain_snapshot(value):
    """Copy caller evidence into an independent plain SI serialization snapshot."""
    if isinstance(value, np.ndarray):
        return _plain_snapshot(value.tolist())
    if isinstance(value, dict):
        return {key: _plain_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_snapshot(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def analyze_waveform(request: dict) -> dict:
    """Calculate exact signed loss components and auditable event/statistical views.

    Missing requested models reject. Explicit omissions remain null, distinct
    from a supported computed zero. The full interval always sets average power.
    """
    request = _plain_snapshot(request)
    signals, duration = _waveform(request["waveform"])
    indices = _event_indices(signals, request["events"], gate_values=request["waveform"].get("gate_v"))
    conduction, segments, plot = _conduction(signals, duration, request["conduction"])
    switching, events = _switching(signals, duration, request["switching"], indices)
    components = {"conduction": conduction, **switching}
    known_energy = sum(component["energy_j"] for component in components.values()
                       if component["status"] == "computed")
    complete = all(component["status"] == "computed" for component in components.values())
    switching_cumulative = [{"time_s": float(signals["time_s"][0]), "energy_j": 0.0}]
    switching_energy = 0.0
    for event in events:
        switching_cumulative.append({"time_s": event["time_s"], "energy_j": switching_energy})
        switching_energy += event["energy_j"]
        switching_cumulative.append({"time_s": event["time_s"], "energy_j": switching_energy})
    switching_cumulative.append({"time_s": float(signals["time_s"][-1]), "energy_j": switching_energy})
    if not any(component["status"] == "computed" for component in switching.values()):
        switching_cumulative = []
    result = {"duration_s": duration, "components": components,
              "known_component_energy_j": known_energy,
              "known_component_power_w": known_energy / duration,
              "total_energy_j": known_energy if complete else None,
              "total_power_w": known_energy / duration if complete else None,
              "total_status": "computed" if complete else "partial-explicit-omissions",
              "power_segments": segments, "conduction_plot": plot, "events": events,
              "switching_cumulative_plot": switching_cumulative,
              "signals": {name: value.tolist() for name, value in signals.items()},
              "current_statistics": waveform_statistics(signals["time_s"], signals["current_a"]),
              "event_policy": request["events"], "request": request,
              "waveform_provenance": request.get("provenance") or {
                  "source": None, "calibration_notes": None, "imported_filename": None},
              "assumptions": ["Piecewise-linear current, voltage-drop curves and duty",
                              "Exact cubic conduction integration; gate does not mask conduction",
                              "Pre-edge event sample i for bracket [i,i+1]; no initial or wrapped event",
                              "Switching energies are event impulses, not finite power pulses",
                              "Power averages use the complete supplied time interval",
                              "Measured device voltage is distinct from switching bus voltage"]}
    if indices is not None and "digitization" in indices:
        result["gate_digitization"] = indices["digitization"]
    if request.get("spectrum") is not None:
        result["spectrum"] = waveform_spectrum(signals["time_s"], signals["current_a"], **request["spectrum"])
    _assert_finite_tree(result)
    return result


def _legacy_signals(waveform):
    return {"time_s": waveform.time, "current_a": waveform.current,
            "gate_v": waveform.gate, "device_voltage_v": waveform.voltage}


def calc_conduction_loss(waveform: WaveformData, linearized: LinearizedModel) -> float:
    """Exact forward-only compatibility estimate; reverse flow needs explicit model."""
    signals, duration = _waveform(_legacy_signals(waveform))
    model = {"mode": "linear-drop-estimate", "selected": True,
             "offset_v": linearized.v0_channel, "resistance_ohm": linearized.r_channel,
             "source": "Caller-supplied LinearizedModel", "conditions": {"t_j_c": linearized.t_j}}
    return _finite(_conduction(signals, duration, {"positive": model})[0]["power_w"],
                   "conduction power_w")


def calc_conduction_loss_from_channel(waveform: WaveformData, channel: ChannelCharacteristics) -> float:
    """Exact bounded forward V-I integration at the caller's recorded condition."""
    signals, duration = _waveform(_legacy_signals(waveform))
    model = {"mode": "recorded", "current_a": channel.graph_v_i[1],
             "drop_v": channel.graph_v_i[0], "recorded_conditions": {"t_j_c": channel.t_j, "v_g_v": channel.v_g},
             "requested_conditions": {"t_j_c": channel.t_j, "v_g_v": channel.v_g},
             "source": "Caller-supplied ChannelCharacteristics"}
    return _finite(_conduction(signals, duration, {"positive": model})[0]["power_w"],
                   "conduction power_w")


def detect_switching_edges(waveform: WaveformData, threshold: float = 0.5):
    """Pre-edge gate sample times, with strict finite/aligned validation."""
    signals, _ = _waveform(_legacy_signals(waveform))
    indices = _event_indices(signals, {"mode": "gate", "threshold_v": threshold})
    return tuple(signals["time_s"][indices[name]] for name in ("turn_on", "turn_off"))


def detect_zvs(waveform: WaveformData, v_threshold_fraction: float = 0.1):
    """Pre-edge voltage-threshold classification only, not physical ZVS proof."""
    signals, _ = _waveform(_legacy_signals(waveform))
    if "device_voltage_v" not in signals or "gate_v" not in signals:
        raise ValueError("Voltage and gate signals required for ZVS detection")
    fraction = _finite(v_threshold_fraction, "v_threshold_fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("Voltage threshold fraction must lie in [0,1]")
    indices = _event_indices(signals, {"mode": "gate", "threshold_v": 0.5})["turn_on"]
    voltage = abs(signals["device_voltage_v"])
    low = voltage[indices] <= np.max(voltage) * fraction
    return signals["time_s"][indices[low]], signals["time_s"][indices[~low]]


def _legacy_energy_model(data, t_j):
    matching = [record for record in data if record.t_j == t_j]
    if len(matching) != 1:
        raise ValueError("Need one unambiguous energy record at exactly requested temperature")
    record = matching[0]
    if record.dataset_type != "graph_i_e" or record.graph_i_e is None:
        raise ValueError("Single-point energy scaling is unsupported; select an explicit estimate")
    conditions = {"t_j_c": record.t_j, "v_supply_v": record.v_supply,
                  "v_g_v": record.v_g, "r_g_ohm": record.r_g, "v_g_off_v": record.v_g_off}
    return {"mode": "recorded", "current_a": record.graph_i_e[0], "energy_j": record.graph_i_e[1],
            "recorded_conditions": conditions, "requested_conditions": conditions,
            "source": record.comment or "Caller-supplied SwitchingLossData at recorded conditions"}


def calc_switching_loss(waveform: WaveformData, e_on_data: list[SwitchingLossData],
                        e_off_data: list[SwitchingLossData], t_j: float = 25.0,
                        zvs_reduction: float = 0.0) -> tuple[float, float]:
    """Bounded compatibility lookup at exact Tj and the unique recorded conditions.

    No bus-voltage adaptation is implied by device-voltage samples. The historical
    arbitrary ZVS reduction is rejected; select a supported explicit energy model.
    """
    if zvs_reduction != 0:
        raise ValueError("Arbitrary ZVS energy reduction is unsupported")
    signals, duration = _waveform(_legacy_signals(waveform))
    indices = _event_indices(signals, {"mode": "gate", "threshold_v": 0.5})
    spec = {"turn_on": {"positive": _legacy_energy_model(e_on_data, t_j)},
            "turn_off": {"positive": _legacy_energy_model(e_off_data, t_j)}}
    components, _ = _switching(signals, duration, spec, indices)
    return (_finite(components["turn_on"]["power_w"], "turn_on power_w"),
            _finite(components["turn_off"]["power_w"], "turn_off power_w"))


def calc_total_loss(waveform: WaveformData, linearized: LinearizedModel,
                   e_on_data: list[SwitchingLossData], e_off_data: list[SwitchingLossData],
                   t_j: float = 25.0) -> LossResult:
    """Compatibility total: missing gate/data reject instead of fabricating zero."""
    conduction = calc_conduction_loss(waveform, linearized)
    on, off = calc_switching_loss(waveform, e_on_data, e_off_data, t_j)
    on_times, off_times = detect_switching_edges(waveform)
    total = _finite(conduction + on + off, "total power_w")
    return LossResult(conduction, on, off, total,
                      None,
                      np.sort(np.concatenate([on_times, off_times])), len(on_times), len(off_times))
