"""Inverse constant-capacitance gate-loop resistance estimate in SI units."""

from decimal import Decimal, localcontext
import math
from numbers import Real

import numpy as np


def gate_resistance_for_slew(
    target_dv_dt_v_s, c_rss_min_f, v_drive_high_v, v_drive_low_v,
    v_plateau_low_v, v_plateau_high_v, *, selected=False,
):
    """Estimate required total series resistance over the Miller plateau interval.

    This inverts the gate-current/voltage relation in ``signed_gate_slew``.
    The maximum per-plateau requirements reproduce SC ``mfiles/calcRg.m``.
    Each selected value is the minimum total resistance that keeps slew at
    or below the target throughout the supplied plateau interval under this model.
    Greater resistance reduces the predicted slew magnitude.
    """
    if selected is not True:
        raise ValueError("Select the constant-capacitance gate resistance estimate explicitly")
    inputs = (target_dv_dt_v_s, c_rss_min_f, v_drive_high_v, v_drive_low_v,
              v_plateau_low_v, v_plateau_high_v)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           for value in inputs):
        raise ValueError("Every input must be a finite real numeric scalar in SI units")
    try:
        values = tuple(float(value) for value in inputs)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Every input must fit the finite numeric range") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Every input must be a finite real numeric scalar in SI units")
    slew, capacitance, high, low, plateau_low, plateau_high = values
    if slew <= 0 or capacitance <= 0:
        raise ValueError("Target slew magnitude and minimum Crss must be positive")
    if not low < plateau_low <= plateau_high < high:
        raise ValueError("Require driver low < plateau low <= plateau high < driver high")

    # Decimal intermediates prevent avoidable overflow in finite voltage differences.
    with localcontext() as context:
        context.prec = 40
        slew, capacitance, high, low, plateau_low, plateau_high = map(
            Decimal.from_float, values)
        gate_current = capacitance * slew
        result = {
            "gate_current_magnitude_a": float(gate_current),
            "r_on_min_ohm": float((high - plateau_high) / gate_current),
            "r_on_max_ohm": float((high - plateau_low) / gate_current),
            "r_off_min_ohm": float((plateau_low - low) / gate_current),
            "r_off_max_ohm": float((plateau_high - low) / gate_current),
        }
    if not all(math.isfinite(value) and value > 0 for value in result.values()):
        raise ValueError("Calculation exceeds the positive finite numeric range")
    result.update(
        r_on_ohm=result["r_on_max_ohm"], r_off_ohm=result["r_off_max_ohm"],
        mode="constant-capacitance-gate-resistance-estimate", unit="ohm",
        scope="Total gate-loop series resistance, including driver, internal and external resistance",
        sizing_policy="The maximum per-plateau requirement is the minimum total resistance that keeps slew at or below the target over the supplied interval",
        model_source="SC mfiles/calcRg.m; sc-92afe9737e47b855",
        equations=["Ig=Crss_min*target_abs_dV_dt",
                   "R_on=(V_driver_high-V_plateau)/Ig",
                   "R_off=(V_plateau-V_driver_low)/Ig"],
        assumptions=[
            "Supplied Crss_min is a valid positive minimum across the declared switching conditions",
            "Constant Miller capacitance, fixed driver rails and supplied plateau bounds",
            "No parasitic transient or calibrated switching-loss prediction",
            "The min/max fields are extrema of the per-plateau required resistance, not allowable resistance limits",
            "Selected values describe minimum total series resistance, not an external resistor value",
            "Device accuracy requires independent measurement",
        ],
    )
    return result
