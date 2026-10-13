"""Headless averaged operating-point calculations for PWM converters.

The functions in this module operate on SI-valued inputs and read core TDB
curve objects without changing them. They support ideal CCM/DCM operating
points and reject curve extrapolation. Duplicate channel-current samples use
the maximum recorded voltage at that current for a conservative loss screen.
The estimate model is not a device-loss estimate.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from transistordatabase.core.models import (
    ChannelCharacteristics,
    SwitchingLossData,
    Transistor,
)


SUPPORTED_TOPOLOGIES = ("buck", "boost", "buck-boost", "flyback")
SUPPORTED_MODES = ("estimate", "conduction-only", "detailed")
_SWEEP_AXES = {
    "v_in": ("v_in", "V"),
    "v_out": ("v_out", "V"),
    "p_out": ("p_out", "W"),
    "f_sw": ("f_sw", "Hz"),
    "l_value": ("l_value", "H"),
    "c_out": ("c_out", "F"),
}
_SWEEP_METRICS = {
    "duty_cycle": ("calculated", "duty_cycle", "1"),
    "i_in_avg_a": ("calculated", "i_in_avg_a", "A"),
    "i_out_avg_a": ("calculated", "i_out_avg_a", "A"),
    "i_l_peak_a": ("calculated", "i_l_peak_a", "A"),
    "i_l_ripple_pp_a": ("calculated", "i_l_ripple_pp_a", "A"),
    "v_out_ripple_pp_v": ("calculated", "v_out_ripple_pp_v", "V"),
    "p_loss_total_w": ("losses", "total_w", "W"),
    "efficiency_percent": ("performance", "efficiency_percent", "%"),
    "t_switch_c": ("temperatures", "switch_c", "deg C"),
    "t_diode_c": ("temperatures", "diode_c", "deg C"),
}


def _finite_positive(parameters: Mapping[str, Any], name: str) -> float:
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _optional_finite(parameters: Mapping[str, Any], name: str) -> float | None:
    value = parameters.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _valid_sweep_point_count(value: Any) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return 2 <= value <= 200


def _same_condition(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-9))


def _curve_arrays(curve: Any, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted samples, using the upper-voltage envelope at duplicate I."""
    graph = getattr(curve, "graph_v_i", None)
    values = np.asarray(graph, dtype=float) if graph is not None else np.empty((0, 0))
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] < 2:
        raise ValueError(f"{label} curve must contain voltage and current samples")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} curve contains non-finite samples")
    if np.any(values[1] < 0) or np.any(values[0] < 0):
        raise ValueError(f"{label} curve must use non-negative current and voltage")
    currents, inverse = np.unique(values[1], return_inverse=True)
    voltages = np.full(currents.shape, -np.inf, dtype=float)
    np.maximum.at(voltages, inverse, values[0])
    if currents.size < 2:
        raise ValueError(f"{label} curve must contain at least two distinct currents")
    return currents, voltages


def _interpolate_curve(curve: Any, current: float, label: str) -> float:
    currents, voltages = _curve_arrays(curve, label)
    if current < currents[0] or current > currents[-1]:
        raise ValueError(
            f"{label} curve supports {currents[0]:g} to {currents[-1]:g} A; "
            f"requested {current:g} A"
        )
    return float(np.interp(current, currents, voltages))


def _channel_curve(
    curves: list[ChannelCharacteristics],
    *,
    t_j: float,
    v_g: float | None,
    label: str,
) -> ChannelCharacteristics:
    temperature_matches = [curve for curve in curves if _same_condition(curve.t_j, t_j)]
    if label == "diode" and v_g is None:
        unconditioned_matches = [
            curve for curve in temperature_matches if curve.v_g is None
        ]
        available_gate_voltages = sorted(
            {float(curve.v_g) for curve in temperature_matches if curve.v_g is not None}
        )
        if not unconditioned_matches and available_gate_voltages:
            available = ", ".join(f"{gate:g} V" for gate in available_gate_voltages)
            raise ValueError(
                f"Diode channel curves at t_j={t_j:g} C are gate-conditioned; "
                f"provide diode_v_g (available: {available})"
            )
    matches = [
        curve for curve in temperature_matches if _same_condition(curve.v_g, v_g)
    ]
    if not matches:
        condition = f"t_j={t_j:g} C"
        if v_g is not None:
            condition += f", v_g={v_g:g} V"
        raise ValueError(f"Missing required {label} channel curve at {condition}")
    if len(matches) != 1:
        raise ValueError(
            f"Multiple {label} channel curves match t_j={t_j:g} C; "
            "the data model does not identify which curve to select"
        )
    return matches[0]


def _switching_curve(
    curves: list[SwitchingLossData],
    *,
    label: str,
    t_j: float,
    v_supply: float,
    v_g: float,
    v_g_off: float | None,
    r_g: float | None,
) -> SwitchingLossData:
    base_matches = []
    for curve in curves:
        if curve.dataset_type != "graph_i_e":
            continue
        if not _same_condition(curve.t_j, t_j):
            continue
        if not _same_condition(curve.v_supply, v_supply):
            continue
        if r_g is not None and not _same_condition(curve.r_g, r_g):
            continue
        base_matches.append(curve)

    def matches_turn_on_gate(curve: SwitchingLossData) -> bool:
        if not _same_condition(curve.v_g, v_g):
            return False
        if v_g_off is None or curve.v_g_off is None:
            return True
        return _same_condition(curve.v_g_off, v_g_off)

    def matches_turn_off_gate(curve: SwitchingLossData) -> bool:
        if curve.v_g_off is not None:
            if not _same_condition(curve.v_g, v_g):
                return False
            if v_g_off is None:
                return True
            return _same_condition(curve.v_g_off, v_g_off)
        if v_g_off is None:
            return True
        return _same_condition(curve.v_g, v_g_off)

    if label == "e_on":
        # Eon.v_g is the turn-on gate voltage. Older TDB records may not
        # carry v_g_off on the Eon curve, even when the caller specifies it.
        matches = [curve for curve in base_matches if matches_turn_on_gate(curve)]
    else:
        # Modern Eoff records pair the turn-on and turn-off gates explicitly.
        # Legacy records store the turn-off gate in v_g and leave v_g_off empty.
        matches = [curve for curve in base_matches if matches_turn_off_gate(curve)]

    def available_turn_off_gate(curve: SwitchingLossData) -> bool:
        if label != "e_off":
            return False
        return curve.v_g_off is None or _same_condition(curve.v_g, v_g)

    def format_available_turn_off_gates(
        curves_to_format: list[SwitchingLossData],
    ) -> str:
        available_off_gates = sorted(
            {
                float(curve.v_g_off if curve.v_g_off is not None else curve.v_g)
                for curve in curves_to_format
                if available_turn_off_gate(curve)
            }
        )
        if not available_off_gates or label != "e_off":
            return ""
        available_gate_text = ", ".join(
            f"{gate:g} V" for gate in available_off_gates
        )
        return f"; available turn-off gate voltages: {available_gate_text}"

    if not matches:
        available = format_available_turn_off_gates(base_matches)
        raise ValueError(
            f"Missing required {label} graph_i_e curve matching t_j={t_j:g} C, "
            f"v_supply={v_supply:g} V, v_g={v_g:g} V{available}"
        )
    if len(matches) != 1:
        available = format_available_turn_off_gates(matches)
        raise ValueError(
            f"Multiple {label} graph_i_e curves match the requested conditions; "
            "specify gate-resistance and turn-off-voltage conditions to select one"
            f"{available}"
        )
    return matches[0]


def _energy_at_current(curve: SwitchingLossData, current: float, label: str) -> float:
    graph = curve.graph_i_e
    values = np.asarray(graph, dtype=float) if graph is not None else np.empty((0, 0))
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] < 2:
        raise ValueError(
            f"{label} graph_i_e curve has no usable current-energy samples"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{label} graph_i_e curve contains non-finite samples")
    order = np.argsort(values[0], kind="stable")
    currents = values[0, order]
    energies = values[1, order]
    if np.any(np.diff(currents) <= 0) or np.any(energies < 0):
        raise ValueError(
            f"{label} graph_i_e curve must have unique currents and non-negative energy"
        )
    if current < currents[0] or current > currents[-1]:
        raise ValueError(
            f"{label} switching curve supports {currents[0]:g} to {currents[-1]:g} A; "
            f"requested {current:g} A"
        )
    return float(np.interp(current, currents, energies))


def _mean_conduction_power(
    curve: ChannelCharacteristics, valley: float, peak: float, label: str
) -> float:
    if np.isclose(valley, peak):
        voltage = _interpolate_curve(curve, valley, label)
        return voltage * valley
    currents = np.linspace(valley, peak, 65)
    voltages = np.array(
        [_interpolate_curve(curve, current, label) for current in currents]
    )
    power = voltages * currents
    integral = np.sum((power[:-1] + power[1:]) * 0.5 * np.diff(currents))
    return float(integral / (peak - valley))


def _dcm_output_ripple(
    topology: str,
    duty: float,
    diode_fraction: float,
    peak_current: float,
    output_current: float,
    frequency: float,
    capacitance: float,
    diode_current_scale: float = 1.0,
) -> float:
    """Estimate DCM capacitor ripple by integrating its piecewise current."""
    phase = np.linspace(0.0, 1.0, 20_001)
    magnetizing_current = np.zeros_like(phase)
    on = phase < duty
    diode = (phase >= duty) & (phase < duty + diode_fraction)
    magnetizing_current[on] = peak_current * phase[on] / duty
    magnetizing_current[diode] = peak_current * (
        1.0 - (phase[diode] - duty) / diode_fraction
    )
    if topology == "buck":
        output_branch_current = magnetizing_current
    else:
        output_branch_current = np.where(
            diode, magnetizing_current * diode_current_scale, 0.0
        )
    capacitor_current = output_branch_current - output_current
    phase_step = np.diff(phase)
    charge_increments = (
        (capacitor_current[:-1] + capacitor_current[1:]) * 0.5 * phase_step
    ) / frequency
    charge = np.concatenate(([0.0], np.cumsum(charge_increments)))
    return float(np.ptp(charge) / capacitance)


def _ccm_output_ripple(
    topology: str,
    duty: float,
    valley_current: float,
    peak_current: float,
    output_current: float,
    frequency: float,
    capacitance: float,
    diode_current_scale: float = 1.0,
) -> float:
    """Integrate CCM capacitor charge exactly across linear current segments."""
    if topology == "buck":
        segments = (
            (duty, valley_current - output_current, peak_current - output_current),
            (
                1.0 - duty,
                peak_current - output_current,
                valley_current - output_current,
            ),
        )
    else:
        segments = (
            (duty, -output_current, -output_current),
            (
                1.0 - duty,
                diode_current_scale * peak_current - output_current,
                diode_current_scale * valley_current - output_current,
            ),
        )

    charge = 0.0
    extrema = [charge]
    for fraction, current_start, current_end in segments:
        duration = fraction / frequency
        delta_current = current_end - current_start
        if delta_current != 0.0:
            zero_fraction = -current_start / delta_current
            if 0.0 < zero_fraction < 1.0:
                zero_time = duration * zero_fraction
                zero_charge = charge + 0.5 * current_start * zero_time
                extrema.append(zero_charge)
        charge += 0.5 * (current_start + current_end) * duration
        extrema.append(charge)
    return float((max(extrema) - min(extrema)) / capacitance)


def _temperature_result(
    switch_transistor: Transistor,
    diode_transistor: Transistor,
    switch_loss_per_device: float,
    diode_loss_per_device: float,
    t_case: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"switch_c": None, "diode_c": None, "t_case_c": t_case}
    if t_case is None:
        result["unavailable_reason"] = (
            "t_case must be supplied for a steady-state temperature estimate"
        )
        return result
    for label, transistor, component, r_cs_name, loss in (
        (
            "switch",
            switch_transistor,
            switch_transistor.switch,
            "r_th_switch_cs",
            switch_loss_per_device,
        ),
        (
            "diode",
            diode_transistor,
            diode_transistor.diode,
            "r_th_diode_cs",
            diode_loss_per_device,
        ),
    ):
        thermal = getattr(component, "thermal_foster", None)
        r_jc = getattr(thermal, "r_th_total", None)
        r_cs = getattr(transistor.thermal_properties, r_cs_name, None)
        if r_jc is None or r_cs is None:
            result[f"{label}_unavailable_reason"] = (
                f"{label} junction-to-case and case-to-sink thermal resistances "
                "are required"
            )
            continue
        try:
            r_jc = float(r_jc)
            r_cs = float(r_cs)
        except (TypeError, ValueError):
            r_jc = r_cs = float("nan")
        if not np.isfinite([r_jc, r_cs]).all() or r_jc <= 0 or r_cs <= 0:
            result[f"{label}_unavailable_reason"] = (
                f"{label} junction-to-case and case-to-sink thermal resistances "
                "must be finite and greater than zero; nonpositive TDB values "
                "are treated as unavailable"
            )
            continue
        result[f"{label}_c"] = float(t_case + loss * (r_jc + r_cs))
    return result


def calculate_topology_operating_point(
    topology: str,
    parameters: Mapping[str, Any],
    switch_transistor: Transistor | None = None,
    diode_transistor: Transistor | None = None,
    *,
    include_waveforms: bool = True,
) -> dict[str, Any]:
    """Calculate ideal estimates or TDB-channel-backed partial losses.

    ``v_out`` is an output-voltage magnitude. Buck-boost denotes the inverting
    topology and reports ``output_polarity=-1``. Flyback uses a primary-to-
    secondary turns ratio ``turns_ratio_nps``. Curve-backed modes use exact TDB
    curve conditions and never extrapolate.
    """
    if topology == "buck_boost":
        topology = "buck-boost"
    if topology not in SUPPORTED_TOPOLOGIES:
        supported = ", ".join(SUPPORTED_TOPOLOGIES)
        raise ValueError(f"Unsupported topology '{topology}'. Supported: {supported}")

    mode = parameters.get("mode")
    if mode not in SUPPORTED_MODES:
        supported_modes = ", ".join(f"'{item}'" for item in SUPPORTED_MODES)
        raise ValueError(f"mode must be explicitly set to one of {supported_modes}")
    conduction_regime = parameters.get("conduction_regime", "CCM")
    if conduction_regime not in ("CCM", "DCM"):
        raise ValueError("conduction_regime must be 'CCM' or 'DCM'")

    v_in = _finite_positive(parameters, "v_in")
    v_out = _finite_positive(parameters, "v_out")
    p_out = _finite_positive(parameters, "p_out")
    f_sw = _finite_positive(parameters, "f_sw")
    l_value = _finite_positive(parameters, "l_value")
    c_out = _finite_positive(parameters, "c_out")
    t_j = _optional_finite(parameters, "t_j")
    v_g = _optional_finite(parameters, "v_g")
    diode_v_g = _optional_finite(parameters, "diode_v_g")
    v_g_off = _optional_finite(parameters, "v_g_off")
    r_g_on = _optional_finite(parameters, "r_g_on")
    r_g_off = _optional_finite(parameters, "r_g_off")
    t_case = _optional_finite(parameters, "t_case")
    if topology == "flyback" and parameters.get("turns_ratio_nps") is None:
        raise ValueError("Flyback requires a positive turns_ratio_nps (Np/Ns)")
    turns_ratio_nps = (
        _finite_positive(parameters, "turns_ratio_nps")
        if topology == "flyback"
        else None
    )
    if topology != "flyback" and parameters.get("turns_ratio_nps") is not None:
        raise ValueError("turns_ratio_nps is only supported for Flyback")
    parallel_switches = parameters.get("parallel_switches", 1)
    parallel_diodes = parameters.get("parallel_diodes", 1)
    if isinstance(parallel_switches, bool) or not isinstance(parallel_switches, int):
        raise ValueError("parallel_switches must be an integer of at least 1")
    if parallel_switches < 1:
        raise ValueError("parallel_switches must be an integer of at least 1")
    if isinstance(parallel_diodes, bool) or not isinstance(parallel_diodes, int):
        raise ValueError("parallel_diodes must be an integer of at least 1")
    if parallel_diodes < 1:
        raise ValueError("parallel_diodes must be an integer of at least 1")

    output_current_scale = 1.0
    if topology == "buck":
        if not v_out < v_in:
            raise ValueError(
                f"Buck requires 0 < v_out < v_in; got {v_out:g} V and {v_in:g} V"
            )
        duty = v_out / v_in
        i_in = p_out / v_in
        i_out = p_out / v_out
        i_l_average = i_out
        v_l_on = v_in - v_out
        v_block = v_in
    elif topology == "boost":
        if not v_out > v_in:
            raise ValueError(
                f"Boost requires v_out > v_in > 0; got {v_out:g} V and {v_in:g} V"
            )
        duty = 1.0 - v_in / v_out
        i_in = p_out / v_in
        i_out = p_out / v_out
        i_l_average = i_in
        v_l_on = v_in
        v_block = v_out
    elif topology == "buck-boost":
        duty = v_out / (v_in + v_out)
        i_in = p_out / v_in
        i_out = p_out / v_out
        i_l_average = i_in / duty
        v_l_on = v_in
        v_block = v_in + v_out
    else:
        # The Flyback equations below use primary-referred magnetizing current
        # and the ideal turns ratio supplied by the caller.
        if turns_ratio_nps is None:
            raise ValueError("Flyback requires a positive turns_ratio_nps (Np/Ns)")
        duty = (turns_ratio_nps * v_out) / (v_in + turns_ratio_nps * v_out)
        i_in = p_out / v_in
        i_out = p_out / v_out
        i_l_average = i_in / duty
        v_l_on = v_in
        v_block = v_in + turns_ratio_nps * v_out
        output_current_scale = turns_ratio_nps

    if conduction_regime == "CCM":
        diode_fraction = 1.0 - duty
        idle_fraction = 0.0
        i_ripple = v_l_on * duty / (l_value * f_sw)
        i_l_valley = i_l_average - i_ripple / 2.0
        i_l_peak = i_l_average + i_ripple / 2.0
        if i_l_valley <= 0:
            minimum_inductance = v_l_on * duty / (2 * f_sw * i_l_average)
            raise ValueError(
                "Requested CCM point reaches zero inductor current; use DCM or "
                f"increase L above the CCM boundary of {minimum_inductance:g} H"
            )
        if topology == "flyback":
            i_diode_average = output_current_scale * i_l_average * diode_fraction
            i_in = duty * i_l_average
        else:
            i_diode_average = i_l_average * diode_fraction
    else:
        l_f = l_value * f_sw
        if topology == "buck":
            duty = np.sqrt(2.0 * l_f * p_out / (v_in * (v_in - v_out)))
            diode_fraction = duty * (v_in - v_out) / v_out
            i_l_peak = (v_in - v_out) * duty / l_f
            i_in = 0.5 * i_l_peak * duty
            i_l_average = 0.5 * i_l_peak * (duty + diode_fraction)
            v_l_on = v_in - v_out
        elif topology == "boost":
            duty = np.sqrt(2.0 * l_f * p_out * (v_out - v_in) / (v_out * v_in**2))
            diode_fraction = v_in * duty / (v_out - v_in)
            i_l_peak = v_in * duty / l_f
            i_in = 0.5 * i_l_peak * (duty + diode_fraction)
            i_l_average = i_in
            v_l_on = v_in
        elif topology == "buck-boost":
            duty = np.sqrt(2.0 * l_f * p_out) / v_in
            diode_fraction = v_in * duty / v_out
            i_l_peak = v_in * duty / l_f
            i_in = 0.5 * i_l_peak * duty
            i_l_average = 0.5 * i_l_peak * (duty + diode_fraction)
            v_l_on = v_in
        else:
            duty = np.sqrt(2.0 * l_f * p_out) / v_in
            diode_fraction = v_in * duty / (turns_ratio_nps * v_out)
            i_l_peak = v_in * duty / l_f
            i_in = 0.5 * i_l_peak * duty
            i_l_average = 0.5 * i_l_peak * (duty + diode_fraction)
            v_l_on = v_in
        occupied_fraction = duty + diode_fraction
        if not 0 < duty < 1 or not 0 < diode_fraction < 1 or occupied_fraction >= 1:
            raise ValueError(
                "Requested DCM point is at or above the CCM boundary: "
                f"switch duty + diode conduction fraction = {occupied_fraction:g}; "
                "select CCM for this operating point"
            )
        idle_fraction = 1.0 - occupied_fraction
        i_l_valley = 0.0
        i_ripple = i_l_peak
        i_diode_average = 0.5 * output_current_scale * i_l_peak * diode_fraction

    i_diode_valley = output_current_scale * i_l_valley
    i_diode_peak = output_current_scale * i_l_peak
    i_switch_rms = (
        np.sqrt(duty * (i_l_average**2 + i_ripple**2 / 12))
        if conduction_regime == "CCM"
        else np.sqrt(duty * i_l_peak**2 / 3)
    )
    diode_rms_scale = output_current_scale
    i_diode_rms = diode_rms_scale * (
        np.sqrt(diode_fraction * (i_l_average**2 + i_ripple**2 / 12))
        if conduction_regime == "CCM"
        else np.sqrt(diode_fraction * i_l_peak**2 / 3)
    )
    if conduction_regime == "DCM":
        v_ripple = _dcm_output_ripple(
            topology,
            duty,
            diode_fraction,
            i_l_peak,
            i_out,
            f_sw,
            c_out,
            diode_current_scale=output_current_scale,
        )
    else:
        v_ripple = _ccm_output_ripple(
            topology,
            duty,
            i_l_valley,
            i_l_peak,
            i_out,
            f_sw,
            c_out,
            diode_current_scale=output_current_scale,
        )
    output_polarity = -1 if topology == "buck-boost" else 1
    diode_block = v_out + v_in / turns_ratio_nps if topology == "flyback" else v_block

    result: dict[str, Any] = {
        "topology": topology,
        "model": {
            "mode": mode,
            "conduction_regime": conduction_regime,
            "output_polarity": output_polarity,
            "switch_transistor": switch_transistor.metadata.name
            if switch_transistor
            else None,
            "diode_transistor": diode_transistor.metadata.name
            if diode_transistor
            else None,
            "parallel_switches": parallel_switches,
            "parallel_diodes": parallel_diodes,
            "assumptions": [
                "Ideal averaged conversion ratio and lossless current balance.",
                (
                    "Triangular inductor current in continuous conduction mode."
                    if conduction_regime == "CCM"
                    else (
                        "Inductor current starts at zero, ramps to a peak, returns "
                        "to zero, then idles."
                    )
                ),
                (
                    "Output ripple is a calculated peak-to-peak estimate; "
                    "capacitor ESR and switching ripple are ignored."
                ),
                (
                    "Buck-boost is inverting; v_out and i_out are reported "
                    "as magnitudes."
                    if topology == "buck-boost"
                    else "v_out and i_out are positive output magnitudes."
                ),
            ],
        },
        "parameters": {
            "v_in": v_in,
            "v_out": v_out,
            "p_out": p_out,
            "f_sw": f_sw,
            "l_value": l_value,
            "c_out": c_out,
            "conduction_regime": conduction_regime,
            "turns_ratio_nps": turns_ratio_nps,
        },
        "calculated": {
            "duty_cycle": duty,
            "i_in_avg_a": i_in,
            "i_out_avg_a": i_out,
            "i_l_avg_a": i_l_average,
            "i_l_valley_a": i_l_valley,
            "i_l_peak_a": i_l_peak,
            "i_l_ripple_pp_a": i_ripple,
            "v_out_ripple_pp_v": v_ripple,
            "i_switch_rms_a": i_switch_rms,
            "i_diode_rms_a": i_diode_rms,
            "i_diode_avg_a": i_diode_average,
            "i_diode_peak_a": i_diode_peak,
            "diode_conduction_fraction": diode_fraction,
            "idle_fraction": idle_fraction,
            "v_switch_block_v": v_block,
            "v_diode_block_v": diode_block,
            "v_out_signed_v": output_polarity * v_out,
        },
        "losses": None,
        "temperatures": None,
        "performance": {"p_out_w": p_out, "p_in_w": None, "efficiency_percent": None},
    }

    if mode == "estimate":
        result["model"]["assumptions"].append(
            "Estimate mode does not calculate device, gate-drive, magnetic, "
            "or thermal losses."
        )
        if topology == "flyback":
            result["model"]["assumptions"].append(
                "Flyback assumes an ideal transformer with no leakage, winding "
                "copper, core, or clamp losses."
            )
    else:
        if switch_transistor is None:
            raise ValueError(
                "Curve-backed mode requires a switch transistor with TDB curves"
            )
        if diode_transistor is None:
            raise ValueError(
                "Curve-backed mode requires a diode transistor with TDB curves"
            )
        if t_j is None:
            raise ValueError(
                "Curve-backed mode requires t_j to select exact device curves"
            )
        if v_g is None:
            raise ValueError(
                "Curve-backed mode requires v_g to select exact switch curves"
            )

        switch_channel = _channel_curve(
            switch_transistor.switch.channel_data, t_j=t_j, v_g=v_g, label="switch"
        )
        diode_channel = _channel_curve(
            diode_transistor.diode.channel_data,
            t_j=t_j,
            v_g=diode_v_g,
            label="diode",
        )
        switch_valley = i_l_valley / parallel_switches
        switch_peak = i_l_peak / parallel_switches
        diode_valley = i_diode_valley / parallel_diodes
        diode_peak = i_diode_peak / parallel_diodes
        p_cond_switch_per_device = duty * _mean_conduction_power(
            switch_channel, switch_valley, switch_peak, "switch channel"
        )
        p_cond_diode_per_device = diode_fraction * _mean_conduction_power(
            diode_channel, diode_valley, diode_peak, "diode channel"
        )
        p_cond_switch = p_cond_switch_per_device * parallel_switches
        p_cond_diode = p_cond_diode_per_device * parallel_diodes
        p_sw_on = 0.0
        p_sw_off = 0.0
        e_on_curve = None
        e_off_curve = None
        if mode == "detailed":
            e_on_curve = _switching_curve(
                switch_transistor.switch.e_on_data,
                label="e_on",
                t_j=t_j,
                v_supply=v_block,
                v_g=v_g,
                v_g_off=v_g_off,
                r_g=r_g_on,
            )
            e_off_curve = _switching_curve(
                switch_transistor.switch.e_off_data,
                label="e_off",
                t_j=t_j,
                v_supply=v_block,
                v_g=v_g,
                v_g_off=v_g_off,
                r_g=r_g_off,
            )
            e_on = _energy_at_current(e_on_curve, switch_valley, "e_on")
            e_off = _energy_at_current(e_off_curve, switch_peak, "e_off")
            p_sw_on = e_on * f_sw * parallel_switches
            p_sw_off = e_off * f_sw * parallel_switches
        p_total = p_cond_switch + p_cond_diode + p_sw_on + p_sw_off
        p_in = p_out + p_total

        result["model"]["assumptions"].extend(
            [
                (
                    "TDB channel losses use exact-temperature/gate curves with "
                    "linear interpolation only inside recorded current ranges."
                ),
                (
                    "Duplicate TDB channel-current samples use the maximum "
                    "recorded voltage at each current as a conservative "
                    "upper-voltage loss envelope."
                ),
                (
                    "Curve-backed currents and ripple are calculated at the "
                    "ideal lossless operating point; they are not recalculated "
                    "from device losses."
                ),
                (
                    "Parallel devices are identical and share current evenly; "
                    "each branch uses the original per-device TDB curves "
                    "without modifying curve data."
                ),
            ]
        )
        if topology == "flyback":
            result["model"]["assumptions"].append(
                "Flyback assumes an ideal transformer with no leakage, winding "
                "copper, core, or clamp losses."
            )
        if mode == "detailed":
            result["model"]["assumptions"].extend(
                [
                    (
                        "Detailed switching mode uses exact-voltage TDB energy "
                        "curves with linear interpolation only inside recorded "
                        "current ranges."
                    ),
                    (
                        "Switching energies use valley current at turn-on and "
                        "peak current at turn-off."
                    ),
                    (
                        "Losses include switch/diode channel conduction and "
                        "switch turn-on/turn-off only; reverse recovery, gate "
                        "drive, dead time, passive losses, and parasitics are "
                        "excluded."
                    ),
                    (
                        "Commutation-device metadata is recorded provenance "
                        "and is not validated against the selected diode "
                        "identity; confirm measurement compatibility "
                        "separately."
                    ),
                ]
            )
        else:
            result["model"]["assumptions"].extend(
                [
                    (
                        "Conduction-only mode reports TDB switch and diode "
                        "channel conduction losses; switching, reverse-recovery, "
                        "gate-drive, passive, and magnetic losses are excluded."
                    ),
                    (
                        "Conduction-only input power and efficiency are "
                        "partial-loss estimates based on the ideal lossless "
                        "operating point."
                    ),
                ]
            )

        def commutation_device_status(curve: SwitchingLossData) -> str:
            recorded = curve.commutation_device
            if not isinstance(recorded, str) or not recorded.strip():
                return "unknown"
            recorded_name = recorded.strip().casefold()
            selected_diode_name = diode_transistor.metadata.name.strip().casefold()
            if recorded_name == selected_diode_name:
                return "matches_selected_diode_name"
            return "unverified"

        result["model"]["curve_conditions"] = {
            "switch_channel": {"t_j": switch_channel.t_j, "v_g": switch_channel.v_g},
            "diode_channel": {"t_j": diode_channel.t_j, "v_g": diode_channel.v_g},
        }
        if mode == "detailed":
            assert e_on_curve is not None and e_off_curve is not None
            result["model"]["curve_conditions"].update(
                {
                    "e_on": {
                        "t_j": e_on_curve.t_j,
                        "v_supply": e_on_curve.v_supply,
                        "v_g": e_on_curve.v_g,
                        "v_g_off": e_on_curve.v_g_off,
                        "r_g": e_on_curve.r_g,
                        "commutation_device": e_on_curve.commutation_device,
                        "commutation_device_status": commutation_device_status(
                            e_on_curve
                        ),
                        "measurement_testbench": e_on_curve.measurement_testbench,
                    },
                    "e_off": {
                        "t_j": e_off_curve.t_j,
                        "v_supply": e_off_curve.v_supply,
                        "v_g": e_off_curve.v_g,
                        "v_g_off": e_off_curve.v_g_off,
                        "r_g": e_off_curve.r_g,
                        "commutation_device": e_off_curve.commutation_device,
                        "commutation_device_status": commutation_device_status(
                            e_off_curve
                        ),
                        "measurement_testbench": e_off_curve.measurement_testbench,
                    },
                }
            )
        per_switch_device_loss = sum((
            p_cond_switch_per_device,
            p_sw_on / parallel_switches,
            p_sw_off / parallel_switches,
        ))
        result["losses"] = {
            "conduction_switch_w": p_cond_switch,
            "conduction_diode_w": p_cond_diode,
            "switching_on_w": p_sw_on,
            "switching_off_w": p_sw_off,
            "total_w": p_total,
            "scope": (
                "included channel conduction and switching losses only; "
                "excludes reverse recovery, gate drive, passive, and magnetic "
                "losses"
                if mode == "detailed"
                else (
                    "included switch and diode channel conduction only; "
                    "excludes switching, reverse recovery, gate drive, passive, "
                    "and magnetic losses"
                )
            ),
            "loss_scope": (
                "channel_conduction_and_switch_turn_on_turn_off"
                if mode == "detailed"
                else "channel_conduction_only"
            ),
            "switching_losses_included": mode == "detailed",
            "reverse_recovery_losses_included": False,
            "gate_drive_losses_included": False,
            "per_switch_device_w": per_switch_device_loss,
            "per_diode_device_w": p_cond_diode_per_device,
        }
        result["temperatures"] = _temperature_result(
            switch_transistor,
            diode_transistor,
            result["losses"]["per_switch_device_w"],
            p_cond_diode_per_device,
            t_case,
        )
        result["performance"] = {
            "p_out_w": p_out,
            "p_in_w": p_in,
            "efficiency_percent": 100 * p_out / p_in,
            "input_power_scope": (
                "ideal operating point plus included partial device losses; "
                "loss feedback is not iterated"
            ),
            "efficiency_scope": (
                "partial-loss estimate based on the ideal lossless operating "
                "point"
            ),
        }

    if include_waveforms:
        phase = np.unique(
            np.concatenate(
                (
                    np.linspace(0.0, 1.0, 201),
                    np.array([duty, duty + diode_fraction]),
                )
            )
        )
        if conduction_regime == "CCM":
            current = np.where(
                phase <= duty,
                i_l_valley + (i_l_peak - i_l_valley) * phase / duty,
                i_l_peak - (i_l_peak - i_l_valley) * (phase - duty) / diode_fraction,
            )
        else:
            current = np.zeros_like(phase)
            on = phase < duty
            diode = (phase >= duty) & (phase < duty + diode_fraction)
            current[on] = i_l_peak * phase[on] / duty
            current[diode] = i_l_peak * (1.0 - (phase[diode] - duty) / diode_fraction)
        result["waveforms"] = {
            "time_s": (phase / f_sw).tolist(),
            "inductor_current_a": current.tolist(),
            "switch_state": (phase < duty).astype(int).tolist(),
            "diode_state": (
                ((phase >= duty) & (phase < duty + diode_fraction)).astype(int).tolist()
            ),
        }
    return result


def calculate_topology_sweep(
    topology: str,
    parameters: Mapping[str, Any],
    sweep: Mapping[str, Any],
    switch_transistor: Transistor | None = None,
    diode_transistor: Transistor | None = None,
) -> dict[str, Any]:
    """Generate a Python-side line or contour sweep of one supported metric."""
    x_axis = sweep.get("x_axis")
    y_axis = sweep.get("y_axis")
    metric = sweep.get("metric")
    if x_axis not in _SWEEP_AXES:
        raise ValueError(f"Unsupported sweep x_axis '{x_axis}'")
    if y_axis is not None and (y_axis not in _SWEEP_AXES or y_axis == x_axis):
        raise ValueError("Sweep y_axis must be a different supported axis")
    if metric not in _SWEEP_METRICS:
        raise ValueError(f"Unsupported sweep metric '{metric}'")

    # For an unambiguous request schema, ranges are named x_min/y_min.
    x_min = _finite_positive(sweep, "x_min")
    x_max = _finite_positive(sweep, "x_max")
    x_points = sweep.get("x_points", 0)
    if x_min >= x_max or not _valid_sweep_point_count(x_points):
        raise ValueError("x sweep requires x_min < x_max and 2 to 200 points")
    x_values = np.linspace(x_min, x_max, x_points)
    y_values = None
    if y_axis is not None:
        y_min = _finite_positive(sweep, "y_min")
        y_max = _finite_positive(sweep, "y_max")
        y_points = sweep.get("y_points", 0)
        if y_min >= y_max or not _valid_sweep_point_count(y_points):
            raise ValueError("y sweep requires y_min < y_max and 2 to 200 points")
        if x_points * y_points > 2500:
            raise ValueError("Contour sweeps are limited to 2500 operating points")
        y_values = np.linspace(y_min, y_max, y_points)

    def evaluate(x_value: float, y_value: float | None) -> float:
        changed = dict(parameters)
        changed[_SWEEP_AXES[x_axis][0]] = float(x_value)
        if y_axis is not None and y_value is not None:
            changed[_SWEEP_AXES[y_axis][0]] = float(y_value)
        try:
            point = calculate_topology_operating_point(
                topology,
                changed,
                switch_transistor,
                diode_transistor,
                include_waveforms=False,
            )
        except ValueError as exc:
            location = f"x={x_value:g} {_SWEEP_AXES[x_axis][1]}"
            if y_value is not None and y_axis is not None:
                location += f", y={y_value:g} {_SWEEP_AXES[y_axis][1]}"
            raise ValueError(f"Sweep point ({location}) is unsupported: {exc}") from exc
        section, key, _ = _SWEEP_METRICS[metric]
        value = point[section]
        if value is None or value.get(key) is None:
            raise ValueError(
                f"Sweep metric '{metric}' is unavailable in "
                f"{parameters.get('mode')} mode"
            )
        return float(value[key])

    if y_values is None:
        values: Any = [evaluate(float(x), None) for x in x_values]
    else:
        values = [[evaluate(float(x), float(y)) for x in x_values] for y in y_values]
    return {
        "kind": "line" if y_values is None else "contour",
        "x_axis": x_axis,
        "x_unit": _SWEEP_AXES[x_axis][1],
        "x_values": x_values.tolist(),
        "y_axis": y_axis,
        "y_unit": _SWEEP_AXES[y_axis][1] if y_axis else None,
        "y_values": y_values.tolist() if y_values is not None else None,
        "metric": metric,
        "metric_unit": _SWEEP_METRICS[metric][2],
        "values": values,
    }
