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
SUPPORTED_MODES = ("estimate", "conduction-only", "detailed", "loss-balanced")
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


class _CurveRangeError(ValueError):
    """A requested current falls below or above a measured curve's domain."""

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        requested: float,
        side: str,
    ) -> None:
        self.side = side
        super().__init__(
            f"{label} curve supports {minimum:g} to {maximum:g} A; "
            f"requested {requested:g} A"
        )


class _CcmBoundaryError(ValueError):
    """An otherwise valid loss-balanced trial falls outside the CCM domain."""


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
    endpoint_tolerance = 1e-12 * max(1.0, abs(currents[0]), abs(currents[-1]))
    if currents[0] - endpoint_tolerance <= current < currents[0]:
        current = float(currents[0])
    elif currents[-1] < current <= currents[-1] + endpoint_tolerance:
        current = float(currents[-1])
    if current < currents[0] or current > currents[-1]:
        raise _CurveRangeError(
            label,
            float(currents[0]),
            float(currents[-1]),
            current,
            "below" if current < currents[0] else "above",
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


def _mean_channel_voltage(
    curve: ChannelCharacteristics, valley: float, peak: float, label: str
) -> float:
    """Average channel drop over a linear inductor-current ramp."""
    if np.isclose(valley, peak):
        return _interpolate_curve(curve, valley, label)
    currents = np.linspace(valley, peak, 65)
    voltages = np.array(
        [_interpolate_curve(curve, current, label) for current in currents]
    )
    increments = 0.5 * (voltages[:-1] + voltages[1:]) * np.diff(currents)
    return float(np.sum(increments) / (peak - valley))


def _loss_balanced_ccm_operating_point(
    topology: str,
    parameters: Mapping[str, Any],
    switch_curve: ChannelCharacteristics,
    diode_curve: ChannelCharacteristics,
    *,
    ideal_duty: float,
    output_current_scale: float,
    parallel_switches: int,
    parallel_diodes: int,
) -> dict[str, float | int]:
    """Solve a regulated CCM point with measured channel conduction losses.

    Duty is root-solved against the input/output power balance. Inductor ripple
    is triangular and refined against the switch-channel drop. This is an
    averaged channel model, not a switching or parasitic simulation.
    """
    v_in = _finite_positive(parameters, "v_in")
    v_out = _finite_positive(parameters, "v_out")
    p_out = _finite_positive(parameters, "p_out")
    f_sw = _finite_positive(parameters, "f_sw")
    l_value = _finite_positive(parameters, "l_value")
    if not 0.0 < ideal_duty < 1.0:
        raise ValueError("Loss-balanced duty must be between 0 and 1")

    def evaluate(duty: float) -> dict[str, float]:
        if topology == "buck":
            average_current = p_out / v_out
        elif topology == "boost" or topology == "buck-boost":
            average_current = p_out / (v_out * (1.0 - duty))
        else:
            average_current = p_out / (
                v_out * output_current_scale * (1.0 - duty)
            )
        input_current = average_current if topology == "boost" else duty * average_current
        switch_drop = _interpolate_curve(
            switch_curve,
            average_current / parallel_switches,
            "switch channel",
        )
        on_voltage = (
            v_in - v_out - switch_drop
            if topology == "buck"
            else v_in - switch_drop
        )
        ripple = on_voltage * duty / (l_value * f_sw)
        for _ in range(50):
            valley = average_current - ripple / 2.0
            peak = average_current + ripple / 2.0
            if valley <= 0:
                raise _CcmBoundaryError(
                    "Loss-balanced point reaches zero inductor current; "
                    "increase L or use an operating point inside the CCM region"
                )
            switch_drop = _mean_channel_voltage(
                switch_curve,
                valley / parallel_switches,
                peak / parallel_switches,
                "switch channel",
            )
            on_voltage = (
                v_in - v_out - switch_drop
                if topology == "buck"
                else v_in - switch_drop
            )
            updated_ripple = on_voltage * duty / (l_value * f_sw)
            if updated_ripple <= 0:
                raise ValueError(
                    "Switch channel drop leaves no positive inductor-current ramp"
                )
            if abs(updated_ripple - ripple) <= 1e-10 * max(1.0, ripple):
                ripple = updated_ripple
                break
            ripple = 0.5 * (ripple + updated_ripple)
        else:
            raise ValueError("Loss-balanced inductor ripple calculation did not converge")

        valley = average_current - ripple / 2.0
        peak = average_current + ripple / 2.0
        p_switch_per_device = duty * _mean_conduction_power(
            switch_curve,
            valley / parallel_switches,
            peak / parallel_switches,
            "switch channel",
        )
        p_diode_per_device = (1.0 - duty) * _mean_conduction_power(
            diode_curve,
            output_current_scale * valley / parallel_diodes,
            output_current_scale * peak / parallel_diodes,
            "diode channel",
        )
        switch_voltage = _mean_channel_voltage(
            switch_curve,
            valley / parallel_switches,
            peak / parallel_switches,
            "switch channel",
        )
        diode_voltage = _mean_channel_voltage(
            diode_curve,
            output_current_scale * valley / parallel_diodes,
            output_current_scale * peak / parallel_diodes,
            "diode channel",
        )
        total_loss = (
            parallel_switches * p_switch_per_device
            + parallel_diodes * p_diode_per_device
        )
        residual = v_in * input_current - p_out - total_loss
        if topology == "buck":
            volt_second_residual = (
                duty * (v_in - switch_voltage - v_out)
                + (1.0 - duty) * (-v_out - diode_voltage)
            )
        elif topology == "boost":
            volt_second_residual = (
                duty * (v_in - switch_voltage)
                + (1.0 - duty) * (v_in - v_out - diode_voltage)
            )
        elif topology == "buck-boost":
            volt_second_residual = (
                duty * (v_in - switch_voltage)
                + (1.0 - duty) * (-v_out - diode_voltage)
            )
        else:
            turns_ratio_nps = _finite_positive(parameters, "turns_ratio_nps")
            volt_second_residual = (
                duty * (v_in - switch_voltage)
                + (1.0 - duty)
                * (-turns_ratio_nps * (v_out + diode_voltage))
            )
        return {
            "average_current": average_current,
            "input_current": input_current,
            "ripple": ripple,
            "valley": valley,
            "peak": peak,
            "switch_loss_per_device": p_switch_per_device,
            "diode_loss_per_device": p_diode_per_device,
            "total_loss": total_loss,
            "residual": residual,
            "volt_second_residual": volt_second_residual,
        }

    def no_supported_solution(cause: Exception) -> ValueError:
        return ValueError(
            "No loss-balanced solution was found inside the available CCM "
            f"curve ranges: {cause}"
        )

    def solve_bracket(
        low_duty: float,
        low_result: dict[str, float],
        high_duty: float,
        high_result: dict[str, float],
    ) -> dict[str, float | int]:
        for iteration in range(1, 81):
            duty = (low_duty + high_duty) / 2.0
            state = evaluate(duty)
            if abs(state["residual"]) <= 1e-9 * max(p_out, 1.0):
                return {
                    **state,
                    "duty": duty,
                    "iterations": iteration,
                    "power_balance_residual": state["residual"],
                }
            if state["residual"] < 0:
                low_duty, low_result = duty, state
            else:
                high_duty, high_result = duty, state
        raise ValueError("Loss-balanced power-balance calculation did not converge")

    def refine_supported_lower_boundary(
        invalid_duty: float,
        valid_duty: float,
        valid_state: dict[str, float],
    ) -> dict[str, float | int]:
        """Search between a lower-domain failure and the first valid trial."""
        tolerance = 1e-9 * max(p_out, 1.0)
        boundary_error: Exception = ValueError(
            "power residual is positive at the first supported duty"
        )
        for iteration in range(1, 81):
            duty = (invalid_duty + valid_duty) / 2.0
            if duty <= invalid_duty or duty >= valid_duty:
                break
            try:
                state = evaluate(duty)
            except (_CurveRangeError, _CcmBoundaryError) as exc:
                invalid_duty = duty
                boundary_error = exc
                continue
            if abs(state["residual"]) <= tolerance:
                return {
                    **state,
                    "duty": duty,
                    "iterations": iteration,
                    "power_balance_residual": state["residual"],
                }
            if state["residual"] < 0:
                return solve_bracket(duty, state, valid_duty, valid_state)
            valid_duty, valid_state = duty, state
        raise no_supported_solution(boundary_error) from boundary_error

    low = ideal_duty
    try:
        low_state = evaluate(low)
    except (_CurveRangeError, _CcmBoundaryError) as exc:
        if isinstance(exc, _CurveRangeError) and exc.side == "above":
            raise no_supported_solution(exc) from exc
        # Increase duty only across known lower-domain failures. This searches
        # for the first supported CCM point without evaluating outside a curve.
        step = max(1e-6, (1.0 - low) * 1e-4)
        last_error: Exception = exc
        invalid_low = low
        probe_duty = low
        for _ in range(60):
            candidate = min(probe_duty + step, 1.0 - 1e-9)
            if candidate <= probe_duty:
                raise no_supported_solution(last_error) from exc
            try:
                low_state = evaluate(candidate)
                low = candidate
                if low_state["residual"] > 0:
                    return refine_supported_lower_boundary(
                        invalid_low, low, low_state
                    )
                break
            except _CcmBoundaryError as candidate_error:
                invalid_low = probe_duty = candidate
                last_error = candidate_error
                step *= 2.0
            except _CurveRangeError as candidate_error:
                if candidate_error.side == "below":
                    invalid_low = probe_duty = candidate
                    last_error = candidate_error
                    step *= 2.0
                    continue
                raise no_supported_solution(candidate_error) from candidate_error
        else:
            raise no_supported_solution(last_error) from exc
    if abs(low_state["residual"]) <= 1e-10 * max(p_out, 1.0):
        return {**low_state, "duty": low, "iterations": 0,
                "power_balance_residual": low_state["residual"]}
    if low_state["residual"] > 0:
        raise ValueError(
            "No loss-balanced solution exists inside the supported CCM region; "
            "power residual is already positive at its first supported duty"
        )

    high = low + max(1e-6, (1.0 - low) * 0.01)
    step = high - low
    high_state: dict[str, float] | None = None
    upper_error: Exception | None = None
    for _ in range(60):
        high = min(high, 1.0 - 1e-9)
        if high <= low:
            break
        try:
            high_state = evaluate(high)
        except (_CurveRangeError, _CcmBoundaryError) as exc:
            upper_error = exc
            break
        if high_state["residual"] >= 0:
            break
        low, low_state = high, high_state
        step *= 2.0
        high = min(1.0 - 1e-9, low + step)
    if high <= low and upper_error is None:
        raise no_supported_solution(
            ValueError("power residual remains negative at the upper duty limit")
        )
    if high_state is not None and high_state["residual"] >= 0:
        return solve_bracket(low, low_state, high, high_state)
    if upper_error is None:
        upper_error = ValueError("duty reached the upper CCM limit")

    # A coarse expansion may step past the last supported current point.
    # Bisect that interval, looking for either a supported power sign change
    # or the boundary of the exact curve/CCM domain.
    invalid_high = high
    boundary_error = upper_error
    for _ in range(80):
        candidate = (low + invalid_high) / 2.0
        try:
            candidate_state = evaluate(candidate)
        except (_CurveRangeError, _CcmBoundaryError) as exc:
            invalid_high = candidate
            boundary_error = exc
            continue
        if candidate_state["residual"] >= 0:
            return solve_bracket(low, low_state, candidate, candidate_state)
        low, low_state = candidate, candidate_state
    raise no_supported_solution(boundary_error) from boundary_error


def _loss_balanced_channel_dcm_operating_point(
    topology: str,
    parameters: Mapping[str, Any],
    switch_curve: ChannelCharacteristics,
    diode_curve: ChannelCharacteristics,
    *,
    parallel_switches: int,
    parallel_diodes: int,
) -> dict[str, Any]:
    """Solve supported DCM topologies from exact channel curves.

    Integrate each current ramp with its measured current-dependent inductor
    voltage. Flyback current is primary-referred; its diode current and
    reflected off-state voltage use the explicit turns ratio. The peak current
    is solved against output charge per period; input charge and channel
    energy use the same trajectory, so both balances are independently reported.
    """
    v_in = _finite_positive(parameters, "v_in")
    v_out = _finite_positive(parameters, "v_out")
    p_out = _finite_positive(parameters, "p_out")
    f_sw = _finite_positive(parameters, "f_sw")
    l_value = _finite_positive(parameters, "l_value")
    turns_ratio_nps = (
        _finite_positive(parameters, "turns_ratio_nps")
        if topology == "flyback"
        else 1.0
    )
    if topology == "buck":
        if v_out >= v_in:
            raise ValueError("Buck DCM requires 0 < v_out < v_in")
        on_voltage_base = v_in - v_out
    elif topology == "boost":
        if v_out <= v_in:
            raise ValueError("Boost DCM requires v_out > v_in > 0")
        on_voltage_base = v_in
    elif topology == "buck-boost":
        on_voltage_base = v_in
    elif topology == "flyback":
        on_voltage_base = v_in
    else:
        raise ValueError(
            "Channel-feedback DCM supports scalar Buck, Boost, Buck-Boost, "
            "and Flyback operating points only"
        )

    switch_i, switch_v = _curve_arrays(switch_curve, "switch channel")
    diode_i, diode_v = _curve_arrays(diode_curve, "diode channel")
    for label, currents in (
        ("switch channel", switch_i), ("diode channel", diode_i)
    ):
        if currents[0] > 1e-12:
            raise _CurveRangeError(
                label, float(currents[0]), float(currents[-1]), 0.0, "below"
            )
    switch_positive_limit = float(switch_i[-1])
    invalid_switch = np.flatnonzero(switch_v >= on_voltage_base)
    if invalid_switch.size:
        first_invalid = int(invalid_switch[0])
        if first_invalid == 0:
            raise ValueError(
                f"{topology.title()} DCM switch channel leaves no positive "
                "inductor-current "
                "ramp at zero current"
            )
        feasible_current = float(switch_i[first_invalid - 1])
        invalid_current = float(switch_i[first_invalid])
        for _ in range(80):
            midpoint = (feasible_current + invalid_current) / 2.0
            if np.interp(midpoint, switch_i, switch_v) < on_voltage_base:
                feasible_current = midpoint
            else:
                invalid_current = midpoint
        switch_positive_limit = feasible_current

    current_limit = min(
        float(switch_i[-1] * parallel_switches),
        float(diode_i[-1] * parallel_diodes / turns_ratio_nps),
        switch_positive_limit * parallel_switches,
    )
    if current_limit <= 0:
        raise ValueError(
            f"{topology.title()} DCM channel curves have no positive current range"
        )
    period = 1.0 / f_sw
    target_i_out = p_out / v_out

    def integrate(values: np.ndarray, currents: np.ndarray) -> float:
        return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(currents)))

    def cumulative(values: np.ndarray, currents: np.ndarray) -> np.ndarray:
        increments = 0.5 * (values[:-1] + values[1:]) * np.diff(currents)
        return np.concatenate(([0.0], np.cumsum(increments)))

    def evaluate(peak: float) -> dict[str, Any]:
        if peak <= 0:
            return {"peak": 0.0, "output_current": 0.0, "occupied": 0.0}
        switch_branch_peak = peak / parallel_switches
        diode_current_scale = turns_ratio_nps if topology == "flyback" else 1.0
        diode_branch_peak = diode_current_scale * peak / parallel_diodes
        if switch_branch_peak > switch_i[-1]:
            raise _CurveRangeError(
                "switch channel", float(switch_i[0]), float(switch_i[-1]),
                switch_branch_peak, "above",
            )
        if diode_branch_peak > diode_i[-1]:
            raise _CurveRangeError(
                "diode channel", float(diode_i[0]), float(diode_i[-1]),
                diode_branch_peak, "above",
            )
        current = np.linspace(0.0, peak, 257)
        switch_drop = np.interp(current / parallel_switches, switch_i, switch_v)
        diode_drop = np.interp(
            diode_current_scale * current / parallel_diodes, diode_i, diode_v
        )
        on_voltage = on_voltage_base - switch_drop
        off_voltage = (
            v_out + diode_drop
            if topology == "buck"
            else (
                v_out - v_in + diode_drop
                if topology == "boost"
                else (
                    turns_ratio_nps * (v_out + diode_drop)
                    if topology == "flyback"
                    else v_out + diode_drop
                )
            )
        )
        if np.any(on_voltage <= 0):
            bad = float(current[np.flatnonzero(on_voltage <= 0)[0]])
            raise ValueError(
                f"{topology.title()} DCM switch channel leaves no positive "
                "inductor-current "
                f"ramp at {bad:g} A"
            )
        inverse_on = 1.0 / on_voltage
        inverse_off = 1.0 / off_voltage
        on_time = l_value * integrate(inverse_on, current)
        diode_time = l_value * integrate(inverse_off, current)
        charge_on = l_value * integrate(current * inverse_on, current)
        charge_diode = l_value * integrate(current * inverse_off, current)
        switch_energy = l_value * integrate(
            current * switch_drop * inverse_on, current
        )
        diode_energy = l_value * integrate(
            diode_current_scale * current * diode_drop * inverse_off, current
        )
        switch_voltage_time = l_value * integrate(
            switch_drop * inverse_on, current
        )
        diode_voltage_time = l_value * integrate(
            diode_drop * inverse_off, current
        )
        duty = on_time / period
        diode_fraction = diode_time / period
        occupied = duty + diode_fraction
        input_current = (
            (charge_on + charge_diode) / period
            if topology == "boost"
            else charge_on / period
        )
        diode_current = diode_current_scale * charge_diode / period
        output_current = (
            (charge_on + charge_diode) / period
            if topology == "buck"
            else diode_current
        )
        inductor_average = (
            (charge_on + charge_diode) / period
            if topology in ("buck", "buck-boost", "flyback")
            else input_current
        )
        switch_loss = switch_energy / period
        diode_loss = diode_energy / period
        switch_mean = switch_voltage_time / on_time
        diode_mean = diode_voltage_time / diode_time
        if topology == "buck":
            volt_second_residual = (
                duty * (v_in - v_out - switch_mean)
                + diode_fraction * (-v_out - diode_mean)
            )
        elif topology == "boost":
            volt_second_residual = (
                duty * (v_in - switch_mean)
                + diode_fraction * (v_in - v_out - diode_mean)
            )
        elif topology == "flyback":
            volt_second_residual = (
                duty * (v_in - switch_mean)
                - diode_fraction * turns_ratio_nps * (v_out + diode_mean)
            )
        else:
            volt_second_residual = (
                duty * (v_in - switch_mean)
                + diode_fraction * (-v_out - diode_mean)
            )
        return {
            "peak": peak,
            "duty": duty,
            "diode_fraction": diode_fraction,
            "idle_fraction": 1.0 - occupied,
            "occupied": occupied,
            "average_current": inductor_average,
            "input_current": input_current,
            "output_current": output_current,
            "diode_current": diode_current,
            "ripple": peak,
            "valley": 0.0,
            "switch_loss_per_device": switch_loss / parallel_switches,
            "diode_loss_per_device": diode_loss / parallel_diodes,
            "switch_loss": switch_loss,
            "diode_loss": diode_loss,
            "total_loss": switch_loss + diode_loss,
            "power_balance_residual": (
                v_in * input_current - p_out - switch_loss - diode_loss
            ),
            "volt_second_residual": volt_second_residual,
            "switch_rms": np.sqrt(
                l_value * integrate(current**2 * inverse_on, current) / period
            ),
            "diode_rms": diode_current_scale * np.sqrt(
                l_value * integrate(current**2 * inverse_off, current) / period
            ),
            "waveform_on_time_s": l_value * cumulative(inverse_on, current),
            "waveform_diode_time_s": l_value * cumulative(inverse_off, current),
            "waveform_current_a": current,
        }

    low, high = 0.0, current_limit
    high_state = evaluate(high)
    if high_state["occupied"] >= 1.0:
        boundary_low, boundary_high = 0.0, high
        for _ in range(80):
            mid = (boundary_low + boundary_high) / 2.0
            state = evaluate(mid)
            if state["occupied"] < 1.0:
                boundary_low = mid
            else:
                boundary_high = mid
        high = boundary_low
        high_state = evaluate(high)
        boundary_tolerance = 1e-10 * max(target_i_out, 1.0)
        if target_i_out >= high_state["output_current"] - boundary_tolerance:
            raise ValueError(
                f"Requested {topology.title()} DCM point is at or above the "
                "CCM boundary: "
                f"required output current {target_i_out:g} A reaches or exceeds "
                f"supported DCM boundary near {high_state['output_current']:g} A"
            )
    elif high_state["output_current"] < target_i_out * (1.0 - 1e-10):
        raise ValueError(
            f"No supported {topology.title()} DCM solution inside the "
            "exact channel-curve "
            f"range; peak current is limited to {current_limit:g} A"
        )

    state: dict[str, Any] = high_state
    for _iteration in range(1, 81):
        peak = (low + high) / 2.0
        state = evaluate(peak)
        if state["occupied"] >= 1.0:
            high = peak
            continue
        current_error = state["output_current"] - target_i_out
        if abs(current_error) <= 1e-10 * max(target_i_out, 1.0):
            break
        if current_error < 0:
            low = peak
        else:
            high = peak
    else:
        raise ValueError(
            f"{topology.title()} DCM loss-balanced current solve did not converge"
        )

    if state["occupied"] >= 1.0:
        raise ValueError(
            f"Requested {topology.title()} DCM point is at or above the "
            "CCM boundary; "
            "select CCM for this operating point"
        )
    if abs(state["power_balance_residual"]) > 1e-8 * max(p_out, 1.0):
        raise ValueError(
            f"{topology.title()} DCM quadrature did not close "
            "input/output plus channel-loss "
            f"power balance (residual {state['power_balance_residual']:g} W)"
        )
    if abs(state["volt_second_residual"]) > 1e-8 * max(v_in, v_out):
        raise ValueError(
            f"{topology.title()} DCM quadrature did not close the averaged "
            "inductor "
            f"volt-second balance (residual {state['volt_second_residual']:g} V)"
        )
    state["iterations"] = _iteration
    state["power_balance_residual_w"] = state["power_balance_residual"]
    diode_times = state["waveform_diode_time_s"]
    state["waveform_phase"] = np.concatenate(
        (
            state["waveform_on_time_s"] / period,
            state["duty"] + (diode_times[-1] - diode_times[-2::-1]) / period,
            np.array([1.0]),
        )
    )
    state["waveform_current"] = np.concatenate(
        (
            state["waveform_current_a"],
            state["waveform_current_a"][-2::-1],
            np.array([0.0]),
        )
    )
    return state


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


def _trajectory_output_ripple(
    phase: np.ndarray,
    current: np.ndarray,
    *,
    topology: str,
    duty: float,
    diode_fraction: float,
    output_current: float,
    frequency: float,
    capacitance: float,
    diode_current_scale: float = 1.0,
) -> float:
    """Integrate output-capacitor charge from a solved switching trajectory."""
    cycle_time = 1.0 / frequency
    diode_end = duty + diode_fraction
    charge = 0.0
    extrema = [charge]
    for index in range(len(phase) - 1):
        phase_start = float(phase[index])
        phase_end = float(phase[index + 1])
        midpoint = (phase_start + phase_end) / 2.0
        carries_output_current = (
            midpoint < diode_end
            if topology == "buck"
            else duty < midpoint < diode_end
        )
        if carries_output_current:
            current_start = (
                float(current[index]) * diode_current_scale - output_current
            )
            current_end = (
                float(current[index + 1]) * diode_current_scale - output_current
            )
        else:
            current_start = current_end = -output_current
        duration = (phase_end - phase_start) * cycle_time
        delta_current = current_end - current_start
        if delta_current != 0.0:
            zero_fraction = -current_start / delta_current
            if 0.0 < zero_fraction < 1.0:
                zero_time = duration * zero_fraction
                extrema.append(charge + 0.5 * current_start * zero_time)
        charge += 0.5 * (current_start + current_end) * duration
        extrema.append(charge)
    return float((max(extrema) - min(extrema)) / capacitance)


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
    if (
        mode == "loss-balanced"
        and conduction_regime == "DCM"
        and topology not in ("buck", "boost", "buck-boost", "flyback")
    ):
        raise ValueError(f"loss-balanced DCM does not support topology '{topology}'")

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
        if i_l_valley <= 0 and mode != "loss-balanced":
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
        loss_balanced_channel_dcm = (
            mode == "loss-balanced"
            and topology in ("buck", "boost", "buck-boost", "flyback")
            and conduction_regime == "DCM"
        )
        if (
            not loss_balanced_channel_dcm
            and (
                not 0 < duty < 1
                or not 0 < diode_fraction < 1
                or occupied_fraction >= 1
            )
        ):
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
        v_ripple = (
            0.0
            if mode == "loss-balanced"
            and topology in ("buck", "boost", "buck-boost", "flyback")
            else _dcm_output_ripple(
                topology,
                duty,
                diode_fraction,
                i_l_peak,
                i_out,
                f_sw,
                c_out,
                diode_current_scale=output_current_scale,
            )
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
                (
                    f"{topology.title()} DCM current ramps and duty are solved "
                    "from "
                    "the exact channel curves and requested output current."
                    if mode == "loss-balanced"
                    and conduction_regime == "DCM"
                    else (
                        "CCM duty is adjusted to balance the requested output "
                        "power and included channel-conduction losses."
                        if mode == "loss-balanced"
                        else "Ideal averaged conversion ratio and lossless current balance."
                    )
                ),
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
        if mode == "loss-balanced":
            if conduction_regime == "DCM":
                solved = _loss_balanced_channel_dcm_operating_point(
                    topology,
                    parameters,
                    switch_channel,
                    diode_channel,
                    parallel_switches=parallel_switches,
                    parallel_diodes=parallel_diodes,
                )
            else:
                solved = _loss_balanced_ccm_operating_point(
                    topology,
                    parameters,
                    switch_channel,
                    diode_channel,
                    ideal_duty=duty,
                    output_current_scale=output_current_scale,
                    parallel_switches=parallel_switches,
                    parallel_diodes=parallel_diodes,
                )
            duty = float(solved["duty"])
            i_l_average = float(solved["average_current"])
            i_in = float(solved["input_current"])
            i_l_ripple = float(solved["ripple"])
            i_l_valley = float(solved["valley"])
            i_l_peak = float(solved["peak"])
            if conduction_regime == "DCM":
                diode_fraction = float(solved["diode_fraction"])
                idle_fraction = float(solved["idle_fraction"])
                i_l_valley = 0.0
                i_l_peak = float(solved["peak"])
                i_l_ripple = i_l_peak
                i_l_average = float(solved["average_current"])
                i_in = float(solved["input_current"])
                i_diode_valley = 0.0
                i_diode_peak = output_current_scale * i_l_peak
                i_diode_average = float(solved["diode_current"])
                i_switch_rms = float(solved["switch_rms"])
                i_diode_rms = float(solved["diode_rms"])
                v_ripple = _trajectory_output_ripple(
                    solved["waveform_phase"],
                    solved["waveform_current"],
                    topology=topology,
                    duty=duty,
                    diode_fraction=diode_fraction,
                    output_current=i_out,
                    frequency=f_sw,
                    capacitance=c_out,
                    diode_current_scale=output_current_scale,
                )
            else:
                diode_fraction = 1.0 - duty
                idle_fraction = 0.0
                i_diode_valley = output_current_scale * i_l_valley
                i_diode_peak = output_current_scale * i_l_peak
                i_diode_average = output_current_scale * diode_fraction * i_l_average
                i_switch_rms = np.sqrt(
                    duty * (i_l_average**2 + i_l_ripple**2 / 12)
                )
                i_diode_rms = output_current_scale * np.sqrt(
                    diode_fraction * (i_l_average**2 + i_l_ripple**2 / 12)
                )
                v_ripple = _ccm_output_ripple(
                    topology, duty, i_l_valley, i_l_peak, i_out, f_sw, c_out,
                    diode_current_scale=output_current_scale,
                )
            switch_valley = i_l_valley / parallel_switches
            switch_peak = i_l_peak / parallel_switches
            diode_valley = i_diode_valley / parallel_diodes
            diode_peak = i_diode_peak / parallel_diodes
            p_cond_switch_per_device = float(solved["switch_loss_per_device"])
            p_cond_diode_per_device = float(solved["diode_loss_per_device"])
            result["calculated"].update(
                {
                    "duty_cycle": duty,
                    "i_in_avg_a": i_in,
                    "i_l_avg_a": i_l_average,
                    "i_l_valley_a": i_l_valley,
                    "i_l_peak_a": i_l_peak,
                    "i_l_ripple_pp_a": i_l_ripple,
                    "v_out_ripple_pp_v": v_ripple,
                    "i_switch_rms_a": i_switch_rms,
                    "i_diode_rms_a": i_diode_rms,
                    "i_diode_avg_a": i_diode_average,
                    "i_diode_peak_a": i_diode_peak,
                    "diode_conduction_fraction": diode_fraction,
                    "idle_fraction": idle_fraction,
                    "inductor_volt_second_residual_v": float(
                        solved["volt_second_residual"]
                    ),
                }
            )
            if conduction_regime == "DCM":
                result["calculated"]["switch_conduction_fraction"] = duty
                result["calculated"]["power_balance_residual_w"] = float(
                    solved["power_balance_residual"]
                )
        else:
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

        if mode == "loss-balanced":
            result["model"]["assumptions"].extend(
                [
                    (
                        f"{topology.title()} DCM integrates current-dependent "
                        "switch and "
                        "diode channel curves over both ramps and solves output "
                        "charge with separate input/output power and volt-second "
                        "residuals."
                        if conduction_regime == "DCM"
                        else "Loss-balanced mode solves CCM duty/current so input "
                        "power balances output power plus included channel losses."
                    ),
                    (
                        f"{topology.title()} DCM reports the zero-current idle "
                        "fraction after "
                        "switch and diode demagnetization intervals."
                        if conduction_regime == "DCM"
                        else "Inductor ripple is triangular and refined using "
                        "measured switch-channel voltage at the selected exact "
                        "curve conditions."
                    ),
                    (
                        "CCM power and averaged volt-second residuals can differ "
                        "because the CCM solver uses a triangular-ripple "
                        "approximation with nonlinear channel curves."
                        if conduction_regime == "CCM"
                        else "The DCM residuals describe the measured-curve "
                        "averaged ramp model; parasitic and switching effects "
                        "are not modeled."
                    ),
                    "Only switch and diode channel conduction losses are included; switching, reverse recovery, gate drive, passive, and magnetic losses remain excluded.",
                    "Thermal output is a one-pass estimate at the selected curve temperature, not an electrothermal iteration.",
                    "Parallel devices are identical and share current evenly; each branch uses the original per-device TDB curves without modifying curve data.",
                ]
            )
        else:
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
        elif mode == "conduction-only":
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
        if mode == "loss-balanced":
            temperatures = result["temperatures"]
            switch_temperature = temperatures.get("switch_c")
            diode_temperature = temperatures.get("diode_c")
            if switch_temperature is None or diode_temperature is None:
                result["model"]["thermal_closure"] = "unavailable"
                temperatures["thermal_closure"] = "unavailable"
                temperatures["thermal_closure_reason"] = (
                    temperatures.get("switch_unavailable_reason")
                    or temperatures.get("diode_unavailable_reason")
                    or temperatures.get("unavailable_reason")
                )
            elif (
                abs(switch_temperature - t_j) <= 1e-6
                and abs(diode_temperature - t_j) <= 1e-6
            ):
                result["model"]["thermal_closure"] = (
                    "closed_at_selected_curve_temperature"
                )
                temperatures["thermal_closure"] = (
                    "closed_at_selected_curve_temperature"
                )
            else:
                result["model"]["thermal_closure"] = "not_closed"
                temperatures["thermal_closure"] = "not_closed"
                temperatures["thermal_closure_reason"] = (
                    "Calculated junction temperature differs from the selected "
                    "exact curve temperature; temperature-dependent curve "
                    "interpolation is not implemented."
                )
        result["performance"] = {
            "p_out_w": p_out,
            "p_in_w": v_in * i_in if mode == "loss-balanced" else p_in,
            "efficiency_percent": 100 * p_out / p_in,
            "input_power_scope": (
                f"loss-balanced {conduction_regime} channel-conduction power balance"
                if mode == "loss-balanced"
                else "ideal operating point plus included partial device losses; "
                "loss feedback is not iterated"
            ),
            "efficiency_scope": (
                f"loss-balanced {conduction_regime} channel-conduction estimate"
                if mode == "loss-balanced"
                else "partial-loss estimate based on the ideal lossless operating point"
            ),
            "power_balance_residual_w": (
                float(solved["power_balance_residual"])
                if mode == "loss-balanced"
                else None
            ),
        }

    if include_waveforms:
        if mode == "loss-balanced" and conduction_regime == "DCM":
            phase = solved["waveform_phase"]
            current = solved["waveform_current"]
        else:
            phase = np.unique(
                np.concatenate(
                    (
                        np.linspace(0.0, 1.0, 201),
                        np.array([duty, duty + diode_fraction]),
                    )
                )
            )
        if mode == "loss-balanced" and conduction_regime == "DCM":
            pass
        elif conduction_regime == "CCM":
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
    loss_balanced = parameters.get("mode") == "loss-balanced"
    conduction_regime = parameters.get("conduction_regime", "CCM")
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

    def coordinate_label(x_value: float, y_value: float | None) -> str:
        location = f"x={x_value:g} {_SWEEP_AXES[x_axis][1]}"
        if y_value is not None and y_axis is not None:
            location += f", y={y_value:g} {_SWEEP_AXES[y_axis][1]}"
        return location

    def evaluate(
        x_value: float, y_value: float | None
    ) -> tuple[float, float | None, float | None]:
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
                f"Sweep point ({coordinate_label(x_value, y_value)}) has no "
                f"'{metric}' value in "
                f"{parameters.get('mode')} mode; choose a metric available "
                "for every operating point"
            )
        residual = None
        power_residual = None
        if loss_balanced:
            residual = point["calculated"].get("inductor_volt_second_residual_v")
            if residual is None:
                raise ValueError(
                    f"Sweep point ({coordinate_label(x_value, y_value)}) has no "
                    "inductor volt-second residual"
                )
            if conduction_regime == "DCM":
                power_residual = point["performance"].get(
                    "power_balance_residual_w"
                )
                if power_residual is None:
                    raise ValueError(
                        f"Sweep point ({coordinate_label(x_value, y_value)}) "
                        "has no power-balance residual"
                    )
        return (
            float(value[key]),
            float(residual) if residual is not None else None,
            float(power_residual) if power_residual is not None else None,
        )

    if y_values is None:
        evaluated = [evaluate(float(x), None) for x in x_values]
        values: Any = [metric_value for metric_value, _, _ in evaluated]
        residual_values = (
            [residual for _, residual, _ in evaluated] if loss_balanced else None
        )
        power_residual_values = (
            [power for _, _, power in evaluated]
            if loss_balanced and conduction_regime == "DCM" else None
        )
    else:
        evaluated = [
            [evaluate(float(x), float(y)) for x in x_values] for y in y_values
        ]
        values = [
            [metric_value for metric_value, _, _ in row] for row in evaluated
        ]
        residual_values = (
            [[residual for _, residual, _ in row] for row in evaluated]
            if loss_balanced else None
        )
        power_residual_values = (
            [[power for _, _, power in row] for row in evaluated]
            if loss_balanced and conduction_regime == "DCM" else None
        )
    result = {
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
    if loss_balanced:
        result.update(
            {
                "mode": "loss-balanced",
                "conduction_regime": conduction_regime,
                "loss_scope": "channel_conduction_only",
                "voltage_law_residual_values_v": residual_values,
                "voltage_law_note": (
                    "Each point reuses the scalar DCM ramp and charge solve. "
                    "Exact-condition channel curves must cover zero through "
                    "the solved branch-current peaks, and points at or above "
                    "the CCM boundary are rejected. Power-balance and "
                    "averaged inductor volt-second residuals are independently "
                    "checked at every point."
                    if conduction_regime == "DCM" else
                    "Each point solves included channel-loss power balance and "
                    "reports its averaged inductor volt-second residual in "
                    "voltage_law_residual_values_v; nonlinear curve points may "
                    "not close both balances exactly."
                ),
            }
        )
        if power_residual_values is not None:
            result["power_balance_residual_values_w"] = power_residual_values
    return result
