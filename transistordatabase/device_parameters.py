"""Headless device parameter calculations; all inputs and outputs use SI units."""
from __future__ import annotations

import numpy as np

from transistordatabase.condition_tables import bounded_interpolate, samples


def _finite(**values):
    if any(isinstance(v, bool) or not np.isfinite(v) for v in values.values()):
        raise ValueError("Every parameter must be a finite number")


def _result(**values):
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Calculation exceeds finite numeric range")
    return {key: float(val) for key, val in values.items()}


def fit_linear_vi(current_a, voltage_v, lower_a: float, upper_a: float) -> dict:
    """Fit V = V0 + R I within strict open current bounds, with residuals."""
    _finite(lower_a=lower_a, upper_a=upper_a)
    if lower_a >= upper_a:
        raise ValueError("Fit lower bound must be below upper bound")
    current, voltage = samples(current_a, voltage_v, "V-I fit")
    chosen = (current > lower_a) & (current < upper_a)
    x, y = current[chosen], voltage[chosen]
    if len(x) < 2:
        raise ValueError("At least two distinct current samples inside strict fit bounds are required")
    xc, yc = x - np.mean(x), y - np.mean(y)
    resistance = np.dot(xc, yc) / np.dot(xc, xc)
    offset = np.mean(y) - resistance * np.mean(x)
    if resistance < 0 or offset < 0:
        raise ValueError("Fit gives a negative threshold or resistance; passive linear model unsupported")
    predicted = offset + resistance * x
    residuals = y - predicted
    sse = np.dot(residuals, residuals)
    total = np.dot(yc, yc)
    result = _result(v0_v=offset, resistance_ohm=resistance,
                     rmse_v=np.sqrt(sse / len(x)), max_residual_v=max(abs(residuals)))
    result.update({"r_squared": float(1 - sse / total) if total > 0 else None,
                   "sample_count": len(x), "degrees_of_freedom": len(x) - 2,
                   "current_a": x.tolist(), "voltage_v": y.tolist(),
                   "fitted_voltage_v": predicted.tolist(),
                   "residual_v": residuals.tolist(), "bounds_a": [lower_a, upper_a],
                   "scope": "Threshold-plus-resistance fit inside strict open bounds"})
    return result


def decompose_capacitances(c_iss_f: float, c_oss_f: float, c_rss_f: float) -> dict:
    """Cgd=Crss, Cgs=Ciss−Crss, Cds=Coss−Crss at one common condition."""
    _finite(c_iss_f=c_iss_f, c_oss_f=c_oss_f, c_rss_f=c_rss_f)
    if min(c_iss_f, c_oss_f, c_rss_f) < 0 or c_rss_f > min(c_iss_f, c_oss_f):
        raise ValueError("Require 0 ≤ Crss ≤ min(Ciss, Coss) at the same condition")
    return _result(c_gs_f=c_iss_f - c_rss_f, c_ds_f=c_oss_f - c_rss_f, c_gd_f=c_rss_f)


def integrate_capacitance(voltage_v, capacitance_f, start_v: float, end_v: float) -> dict:
    """Integrate piecewise-linear C(V) for charge and stored energy ∫V C(V)dV."""
    x, y = samples(voltage_v, capacitance_f, "C(V)")
    _finite(start_v=start_v, end_v=end_v)
    if np.any(x < 0) or np.any(y < 0) or not 0 <= start_v < end_v:
        raise ValueError("Capacitance/voltage must be nonnegative; require start < end")
    a, b = bounded_interpolate(x, y, start_v), bounded_interpolate(x, y, end_v)
    interior = (x > start_v) & (x < end_v)
    xx = np.r_[start_v, x[interior], end_v]
    yy = np.r_[a, y[interior], b]
    dx = np.diff(xx)
    q = (yy[:-1] + yy[1:]) * dx / 2
    # Exact integral of V times a linear C, including interpolated endpoints.
    e = dx / 6 * ((2 * xx[:-1] + xx[1:]) * yy[:-1]
                 + (xx[:-1] + 2 * xx[1:]) * yy[1:])
    result = _result(charge_c=np.sum(q), energy_j=np.sum(e))
    result.update(voltage_v=xx.tolist(), capacitance_f=yy.tolist(),
                  q_c=np.r_[0, np.cumsum(q)].tolist(), e_j=np.r_[0, np.cumsum(e)].tolist(),
                  scope="Piecewise-linear measured C(V); charge/energy relative to start voltage")
    return result


def signed_gate_slew(v_th_v, current_a, gm_s, r_g_ohm, c_ies_f, c_res_f, v_drive_v):
    """Constant-gm/capacitance estimate with collector-voltage polarity explicit."""
    _finite(v_th_v=v_th_v, current_a=current_a, gm_s=gm_s, r_g_ohm=r_g_ohm,
            c_ies_f=c_ies_f, c_res_f=c_res_f, v_drive_v=v_drive_v)
    if min(gm_s, r_g_ohm, c_ies_f, c_res_f) <= 0 or min(v_th_v, current_a) < 0:
        raise ValueError("Require positive gm/Rg/Cies/Cres and nonnegative threshold/current")
    if c_res_f > c_ies_f:
        raise ValueError("Require Cres ≤ Cies so implied gate-emitter capacitance is nonnegative")
    plateau = v_th_v + current_a / gm_s
    gate = (v_drive_v - plateau) / r_g_ohm
    return _result(v_plateau_v=plateau, gate_current_a=gate,
                   di_dt_a_s=gm_s * gate / c_ies_f, dv_ce_dt_v_s=-gate / c_res_f)


def energy_current_coefficient(energy_j, reference_current_a, conditions, source):
    """Preserve workbook E/I as J/A, never label this coefficient event energy."""
    _finite(energy_j=energy_j, reference_current_a=reference_current_a)
    if energy_j < 0 or reference_current_a <= 0 or not source.strip():
        raise ValueError("Require nonnegative energy, positive test current and source")
    for value in conditions.values():
        if value is not None:
            _finite(condition=value)
    return {**_result(coefficient_j_per_a=energy_j / reference_current_a),
            "reference_energy_j": energy_j, "reference_current_a": reference_current_a,
            "conditions": dict(conditions), "source": source, "unit": "J/A",
            "scope": "Energy-per-current coefficient; not a measured E(I) curve"}


def resistance_temperature_estimate(t1_c, r1_ohm, t2_c, r2_ohm, target_c,
                                    *, selected=False, max_typ_ratio=None):
    """Explicit linear temperature estimate, including disclosed extrapolation."""
    if not selected:
        raise ValueError("Select the linear resistance-temperature estimate explicitly")
    _finite(t1_c=t1_c, r1_ohm=r1_ohm, t2_c=t2_c, r2_ohm=r2_ohm, target_c=target_c)
    if min(r1_ohm, r2_ohm) <= 0 or t1_c == t2_c:
        raise ValueError("Require positive resistance and distinct reference temperatures")
    resistance = r1_ohm + (target_c - t1_c) * (r2_ohm - r1_ohm) / (t2_c - t1_c)
    if resistance <= 0:
        raise ValueError("Estimated resistance is nonpositive")
    result = _result(resistance_ohm=resistance)
    if max_typ_ratio is not None:
        _finite(max_typ_ratio=max_typ_ratio)
        if max_typ_ratio < 1:
            raise ValueError("Maximum/typical ratio must be at least one")
        result.update(_result(estimated_max_ohm=resistance * max_typ_ratio))
    assumptions = ["Linear resistance versus temperature at unchanged gate bias"]
    if max_typ_ratio is not None:
        assumptions.append("Selected max/typical ratio is transferred unchanged")
    result.update(mode="linear-resistance-temperature-estimate",
                  extrapolated=not min(t1_c, t2_c) <= target_c <= max(t1_c, t2_c),
                  assumptions=assumptions,
                  reference_points=[[t1_c, r1_ohm], [t2_c, r2_ohm]])
    return result
