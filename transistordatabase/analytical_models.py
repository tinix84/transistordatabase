"""Analytical switching loss and device models.

Provides physics-based models for estimating switching losses when
empirical data is incomplete or unavailable. Based on:

- Biela switching loss model (inductive switching)
- Gate charge model (switching time estimation)
- IGBT tail current model (turn-off with minority carrier tail)

References:
    [1] J. Biela, "Optimierung des elektromagnetisch integrierten
        Serien-Parallel-Resonanzkonverters", ETH Zurich, 2005.
    [2] B. J. Baliga, "Fundamentals of Power Semiconductor Devices",
        Springer, 2008.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    from scipy.optimize import brentq, curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from transistordatabase.core.models import ChannelCharacteristics, Transistor


# Package parasitic inductance lookup table (l_s, l_d) in H
PACKAGE_INDUCTANCE: dict[str, tuple[float, float]] = {
    "TO-247": (4e-9, 10e-9),
    "TO-220": (5e-9, 12e-9),
    "TO-263": (3e-9, 8e-9),
    "D2PAK": (3e-9, 8e-9),
    "SOT-227": (2e-9, 5e-9),
    "QFN": (0.5e-9, 1e-9),
    "DFN": (0.5e-9, 1e-9),
    "Module": (5e-9, 15e-9),
}


def get_package_inductance(housing_type: str) -> tuple[float, float]:
    """Look up parasitic inductances for a housing type.

    :param housing_type: Housing type string (e.g., "TO-247").
    :return: Tuple of (l_s, l_d) in H. Defaults to (4e-9, 10e-9) if not found.
    """
    return PACKAGE_INDUCTANCE.get(housing_type, (4e-9, 10e-9))


@dataclass
class GateChargeModelParams:
    """Parameters for gate charge switching time model.

    :param q_gs: Gate-source charge in C.
    :param q_gd: Gate-drain (Miller) charge in C.
    :param q_g: Total gate charge in C.
    :param v_th: Threshold voltage in V.
    :param v_plateau: Plateau voltage in V.
    :param r_g: External gate resistance in Ohm.
    :param r_g_int: Internal gate resistance in Ohm.
    """

    q_gs: float
    q_gd: float
    q_g: float
    v_th: float
    v_plateau: float
    r_g: float
    r_g_int: float = 0.0


class GateChargeModel:
    """Gate charge based switching time estimation model.

    Estimates switching transition times from gate charge characteristics.
    Useful for predicting di/dt and dv/dt during switching.
    """

    def __init__(self, params: GateChargeModelParams) -> None:
        self.params = params

    def calc_turn_on_times(
        self, v_driver: float = 15.0
    ) -> dict[str, float]:
        """Calculate turn-on transition times.

        :param v_driver: Gate driver supply voltage in V.
        :return: Dict with 't_delay', 't_rise', 't_fall_v', 't_total' in s.
        """
        p = self.params
        r_total = p.r_g + p.r_g_int

        # Delay time: charge gate to threshold
        t_delay = r_total * p.q_gs * np.log(
            v_driver / (v_driver - p.v_th)
        ) if v_driver > p.v_th else 0.0

        # Current rise time: charge from threshold to plateau
        q_rise = p.q_gs - (p.q_gs * p.v_th / p.v_plateau)
        t_rise = r_total * q_rise / (v_driver - p.v_plateau) if v_driver > p.v_plateau else 0.0

        # Voltage fall time: Miller plateau (constant Vgs = V_plateau)
        t_fall_v = r_total * p.q_gd / (v_driver - p.v_plateau) if v_driver > p.v_plateau else 0.0

        return {
            't_delay': t_delay,
            't_rise': t_rise,
            't_fall_v': t_fall_v,
            't_total': t_delay + t_rise + t_fall_v,
        }

    def calc_turn_off_times(
        self, v_driver: float = 15.0, v_off: float = 0.0
    ) -> dict[str, float]:
        """Calculate turn-off transition times.

        :param v_driver: Gate driver on-state voltage in V.
        :param v_off: Gate driver off-state voltage in V (0 or negative).
        :return: Dict with 't_delay', 't_rise_v', 't_fall_i', 't_total' in s.
        """
        p = self.params
        r_total = p.r_g + p.r_g_int
        v_swing = v_driver - v_off

        # Delay: discharge from v_driver to plateau
        t_delay = r_total * (p.q_g - p.q_gd - p.q_gs) / v_swing if v_swing > 0 else 0.0

        # Voltage rise time: Miller plateau discharge
        t_rise_v = r_total * p.q_gd / (p.v_plateau - v_off) if p.v_plateau > v_off else 0.0

        # Current fall time: discharge below plateau to threshold
        t_fall_i = r_total * p.q_gs / (p.v_plateau - v_off) if p.v_plateau > v_off else 0.0

        return {
            't_delay': t_delay,
            't_rise_v': t_rise_v,
            't_fall_i': t_fall_i,
            't_total': t_delay + t_rise_v + t_fall_i,
        }


@dataclass
class IgbtModelParams:
    """Parameters for IGBT turn-off tail current model.

    :param v_ce_sat: Collector-emitter saturation voltage in V.
    :param i_tail_factor: Tail current as fraction of load current.
    :param tau_tail: Tail current time constant in s.
    :param t_fall: Current fall time in s.
    """

    v_ce_sat: float = 1.5
    i_tail_factor: float = 0.1
    tau_tail: float = 1e-6
    t_fall: float = 200e-9


class IgbtModel:
    """IGBT-specific model accounting for minority carrier tail current.

    IGBTs have a tail current during turn-off due to stored minority
    carriers in the drift region. This model estimates the additional
    energy loss from the tail current.
    """

    def __init__(self, params: IgbtModelParams) -> None:
        self.params = params

    def calc_tail_energy(self, v_dc: float, i_load: float) -> float:
        """Calculate energy loss from IGBT tail current.

        :param v_dc: DC bus voltage in V.
        :param i_load: Load current in A.
        :return: Tail current energy loss in J.
        """
        p = self.params
        i_tail = i_load * p.i_tail_factor
        # E_tail = V_dc * I_tail * tau_tail (exponential decay integral)
        return v_dc * i_tail * p.tau_tail

    def calc_e_off_total(
        self, v_dc: float, i_load: float,
        e_off_mosfet: Optional[float] = None,
    ) -> float:
        """Calculate total IGBT turn-off energy including tail current.

        :param v_dc: DC bus voltage in V.
        :param i_load: Load current in A.
        :param e_off_mosfet: Base turn-off energy (overlap) in J.
        :return: Total turn-off energy in J.
        """
        p = self.params

        if e_off_mosfet is None:
            # Simple estimate: overlap during current fall
            e_off_mosfet = 0.5 * v_dc * i_load * p.t_fall

        e_tail = self.calc_tail_energy(v_dc, i_load)
        return e_off_mosfet + e_tail

    def calc_conduction_loss(
        self, i_avg: float, _i_rms: float
    ) -> float:
        """Calculate IGBT conduction loss using V_CE(sat) model.

        :param i_avg: Average current in A.
        :param _i_rms: RMS current (unused in this model)
        :param i_rms: RMS current in A.
        :return: Conduction power loss in W.
        """
        p = self.params
        # P_cond = V_CE0 * I_avg + r_CE * I_rms^2
        # Simplified: use V_CE_sat as constant
        return p.v_ce_sat * i_avg


@dataclass
class HalfBridgeParams:
    """Operating conditions and circuit parameters for half-bridge switching.

    :param v_0: DC bus voltage in V.
    :param i_0: Load current in A.
    :param v_g_on: Turn-on gate voltage in V (e.g., +20).
    :param v_g_off: Turn-off gate voltage in V (e.g., -5 or 0).
    :param r_g: Total gate resistance (external + internal) in Ohm.
    :param l_s: Source parasitic inductance in H.
    :param l_d: Drain parasitic inductance in H.
    """

    v_0: float
    i_0: float
    v_g_on: float
    v_g_off: float
    r_g: float
    l_s: float = 4e-9
    l_d: float = 10e-9


@dataclass
class TransconductanceParams:
    """Transconductance model parameters from transfer characteristic fit.

    Transfer characteristic: i_ch = k_1 * (v_gs - v_th)^x + k_2
    Transconductance: g_m(i_ch) = (k_1 * i_ch^x / (i_ch - k_2))^(1/x)

    :param k_1: Coefficient in transfer characteristic.
    :param k_2: Offset in transfer characteristic.
    :param x: Exponent in transfer characteristic.
    :param v_th: Threshold voltage in V.
    """

    k_1: float
    k_2: float
    x: float
    v_th: float

    def calc_gm(self, i_ch: float) -> float:
        """Calculate transconductance at given channel current.

        :param i_ch: Channel current in A.
        :return: Transconductance in S.
        """
        if i_ch <= 0:
            return 0.01  # Minimal gm
        if i_ch <= self.k_2:
            return 100.0  # Large default if out of range
        return (self.k_1 * i_ch**self.x / (i_ch - self.k_2))**(1/self.x)


@dataclass
class ReverseRecoveryParams:
    """Body diode reverse recovery time constants.

    :param tau_rr: Recovery time constant in s.
    :param tau_c: Effective carrier lifetime in s.
    :param t_m: Drift region transit time in s.
    :param q_rr_star: Corrected Q_rr (= Q_rr_datasheet - Q_oss) in C.
    """

    tau_rr: float
    tau_c: float
    t_m: float
    q_rr_star: float


@dataclass
class SwitchingEnergyResult:
    """Detailed switching energy breakdown from Christen-Biela model.

    All energies in J, times in s, currents in A, voltages in V.

    :param e_on: Turn-on switching energy in J.
    :param e_off: Turn-off switching energy in J.
    :param e_total: Total switching energy in J.
    :param i_0_zvs: Zero voltage switching threshold current in A.
    :param i_oss_off: C_oss charging current during turn-off in A.
    :param v_mil_off: Miller plateau voltage during turn-off in V.
    :param t_rv: Voltage rise time during turn-off in s.
    :param t_fi: Current fall time during turn-off in s.
    :param v_ld_off: Voltage drop across parasitic inductance during turn-off in V.
    :param gm_off: Transconductance during turn-off in S.
    :param i_oss_on: C_oss discharging current during turn-on in A.
    :param v_mil_on: Miller plateau voltage during turn-on in V.
    :param t_ri: Current rise time during turn-on in s.
    :param t_fv: Voltage fall time during turn-on in s.
    :param v_ld_on: Voltage drop across parasitic inductance during turn-on in V.
    :param gm_on_1b: Transconductance in phase 1b during turn-on in S.
    :param gm_on_3b: Transconductance in phase 3b during turn-on in S.
    :param i_rr_calc: Calculated reverse recovery current in A.
    :param t_rs: Reverse recovery storage time in s.
    :param q_rs: Reverse recovery charge in C.
    :param e_rs: Reverse recovery storage energy in J.
    :param e_rf: Reverse recovery fall energy in J.
    :param e_rr_s2: Total reverse recovery energy in J.
    """

    # Summary
    e_on: float
    e_off: float
    e_total: float
    i_0_zvs: float

    # Turn-off breakdown
    i_oss_off: float
    v_mil_off: float
    t_rv: float
    t_fi: float
    v_ld_off: float
    gm_off: float

    # Turn-on breakdown
    i_oss_on: float
    v_mil_on: float
    t_ri: float
    t_fv: float
    v_ld_on: float
    gm_on_1b: float
    gm_on_3b: float

    # Reverse recovery
    i_rr_calc: float
    t_rs: float
    q_rs: float
    e_rs: float
    e_rf: float
    e_rr_s2: float

    def to_dict(self) -> dict[str, float]:
        """Return summary dict with main energies.

        :return: Dict with e_on, e_off, e_total, i_0_zvs.
        """
        return {
            "e_on": self.e_on,
            "e_off": self.e_off,
            "e_total": self.e_total,
            "i_0_zvs": self.i_0_zvs,
        }


def calc_charge_equivalent_capacitance(
    v_axis: npt.NDArray[np.float64],
    c_axis: npt.NDArray[np.float64],
    v_0: float,
) -> float:
    """Calculate charge-equivalent capacitance from C(V) curve.

    Implements eq(32): C_eq = (1/V_0) * integral_0^V_0 C(v) dv

    :param v_axis: Voltage points in V.
    :param c_axis: Capacitance points in F.
    :param v_0: Operating voltage in V.
    :return: Charge-equivalent capacitance in F.
    """
    # Clip to [0, v_0] range
    mask = (v_axis >= 0) & (v_axis <= v_0)
    v_clipped = v_axis[mask]
    c_clipped = c_axis[mask]

    if len(v_clipped) == 0:
        # Extrapolate with constant
        return float(c_axis[-1]) if len(c_axis) > 0 else 0.0

    # Add endpoint if needed
    if v_clipped[-1] < v_0:
        v_clipped = np.append(v_clipped, v_0)
        c_clipped = np.append(c_clipped, c_clipped[-1])

    # Integrate
    charge = float(np.trapezoid(c_clipped, v_clipped))
    return charge / v_0


def calc_charge_stored(
    v_axis: npt.NDArray[np.float64],
    c_axis: npt.NDArray[np.float64],
    v_0: float,
) -> float:
    """Calculate charge stored in capacitance.

    Q = integral_0^V_0 C(v) dv (charge stored in C_oss of ONE switch)

    :param v_axis: Voltage points in V.
    :param c_axis: Capacitance points in F.
    :param v_0: Operating voltage in V.
    :return: Stored charge in C.
    """
    # Same clipping as above
    mask = (v_axis >= 0) & (v_axis <= v_0)
    v_clipped = v_axis[mask]
    c_clipped = c_axis[mask]

    if len(v_clipped) == 0:
        return 0.0

    # Add endpoint if needed
    if v_clipped[-1] < v_0:
        v_clipped = np.append(v_clipped, v_0)
        c_clipped = np.append(c_clipped, c_clipped[-1])

    # Integrate
    return float(np.trapezoid(c_clipped, v_clipped))


def calc_device_capacitances(
    c_iss_eq: float, c_oss_eq: float, c_rss_eq: float
) -> tuple[float, float, float]:
    """Calculate device internal capacitances from charge-equivalent values.

    Implements eqs (33)-(35):
    C_gs = C_iss_eq - C_rss_eq
    C_ds = C_oss_eq - C_rss_eq
    C_gd = C_rss_eq

    :param c_iss_eq: Input capacitance (charge-equivalent) in F.
    :param c_oss_eq: Output capacitance (charge-equivalent) in F.
    :param c_rss_eq: Reverse transfer capacitance (charge-equivalent) in F.
    :return: Tuple of (c_gs, c_ds, c_gd) in F.
    """
    c_gs = c_iss_eq - c_rss_eq
    c_ds = c_oss_eq - c_rss_eq
    c_gd = c_rss_eq

    # Guard against negative values
    if c_gs < 0:
        warnings.warn(f"Negative C_gs={c_gs:.2e} clamped to 0", stacklevel=2)
        c_gs = 0.0
    if c_ds < 0:
        warnings.warn(f"Negative C_ds={c_ds:.2e} clamped to 0", stacklevel=2)
        c_ds = 0.0
    if c_gd < 0:
        warnings.warn(f"Negative C_gd={c_gd:.2e} clamped to 0", stacklevel=2)
        c_gd = 0.0

    return (c_gs, c_ds, c_gd)


def extract_reverse_recovery_params(
    q_rr: float,
    i_rr: float,
    di_dt: float,
    q_oss: float,
    t_rr: float | None = None,
) -> ReverseRecoveryParams:
    """Extract reverse recovery time constants from datasheet parameters.

    Implements Christen/Biela Section III-A.2 (eqs 38-41) with supplement corrections.

    Step I: Q_rr_star = Q_rr - Q_oss (supplement Mistake 6)
    Step II: Q_rf, tau_rr (eqs 38-39)
    Step III: tau_c via numerical solve (eq 40)
    Step IV: T_m (CORRECTED eq 41: 1/T_m = 1/tau_rr - 1/tau_c)

    :param q_rr: Reverse recovery charge from datasheet in C.
    :param i_rr: Peak reverse recovery current from datasheet in A.
    :param di_dt: Current slope during reverse recovery in A/s (absolute value).
    :param q_oss: Charge stored in C_oss at test voltage in C.
    :param t_rr: Reverse recovery time from datasheet in s (optional, unused).
    :return: ReverseRecoveryParams with tau_rr, tau_c, t_m, q_rr_star.
    """
    # Step I: Correct Q_rr
    q_rr_star = q_rr - q_oss

    if q_rr_star <= 0:
        # Edge case: negligible reverse recovery
        return ReverseRecoveryParams(
            tau_rr=1e-12,
            tau_c=2e-12,
            t_m=2e-12,
            q_rr_star=max(q_rr_star, 0.0),
        )

    # Step II: Calculate Q_rf and tau_rr
    di_dt_abs = abs(di_dt)
    q_rf = q_rr_star - (i_rr**2) / (2 * di_dt_abs)  # eq 38

    if q_rf <= 0:
        q_rf = q_rr_star * 0.01  # Minimal recovery charge

    tau_rr = q_rf / i_rr  # eq 39 (assuming T_2 -> +inf)

    # Step III: Find tau_c by solving eq 40
    # I_rr = |di_dt| * (tau_c - tau_rr) * (1 - exp(-T_1/tau_c))
    # where T_1 = I_rr / |di_dt|

    t_1 = i_rr / di_dt_abs

    def equation_40(tau_c_candidate: float) -> float:
        """Equation (40) to solve for tau_c.

        :param tau_c_candidate: Candidate value for tau_c.
        :return: Residual of equation 40.
        """
        if tau_c_candidate <= tau_rr:
            return 1e10  # Invalid region
        return (
            di_dt_abs * (tau_c_candidate - tau_rr) * (1 - np.exp(-t_1 / tau_c_candidate))
            - i_rr
        )

    try:
        if not HAS_SCIPY:
            raise ImportError("scipy not available")
        tau_c = float(brentq(equation_40, tau_rr * 1.01, tau_rr * 100, maxiter=100))
    except Exception:
        # Fallback: rough estimate
        tau_c = 2.0 * tau_rr

    # Step IV: Calculate T_m (CORRECTED eq 41)
    # 1/T_m = 1/tau_rr - 1/tau_c
    if tau_c <= tau_rr or (1/tau_rr - 1/tau_c) <= 0:
        t_m = tau_rr * 10  # Very long transit time (edge case)
    else:
        t_m = 1.0 / (1.0/tau_rr - 1.0/tau_c)

    return ReverseRecoveryParams(
        tau_rr=float(tau_rr),
        tau_c=float(tau_c),
        t_m=float(t_m),
        q_rr_star=float(q_rr_star),
    )


def fit_transconductance(
    channel_data_list: list[ChannelCharacteristics],
) -> TransconductanceParams:
    """Fit transconductance model from channel characteristics at multiple V_gs.

    Extracts transfer characteristic (V_gs vs I_d) from output characteristics
    (V_ds vs I_d at various V_gs) and fits power-law model:
    i_ch = k_1 * (v_gs - V_th)^x + k_2

    Reference: Christen/Biela eq(36)-(37).

    :param channel_data_list: List of ChannelCharacteristics at different V_gs values.
    :return: Fitted TransconductanceParams.
    :raises ValueError: If channel_data_list is empty or all entries missing v_g.
    """
    if not channel_data_list:
        raise ValueError("channel_data_list cannot be empty")

    # Extract transfer characteristic: V_gs -> I_d at a reference V_ds
    vgs_points = []
    id_points = []

    for ch_data in channel_data_list:
        if ch_data.v_g is None:
            continue

        # Extract current at median V_ds point (saturation region)
        voltages = ch_data.graph_v_i[0]
        currents = ch_data.graph_v_i[1]

        if len(voltages) == 0:
            continue

        # Use median V_ds (typically in saturation region)
        v_ds_ref = float(np.median(voltages))
        i_d_ref = float(np.interp(v_ds_ref, voltages, currents))

        vgs_points.append(ch_data.v_g)
        id_points.append(i_d_ref)

    if len(vgs_points) < 3:
        raise ValueError(
            f"Need at least 3 V_gs points for fitting, got {len(vgs_points)}"
        )

    vgs_array = np.array(vgs_points)
    id_array = np.array(id_points)

    # Estimate V_th as the V_gs where current starts (> 0.1A threshold)
    v_th_est = float(vgs_array[id_array > 0.1].min()) if any(id_array > 0.1) else float(vgs_array[0])

    # Fit function: i = k1*(vgs - vth)^x + k2
    def transfer_func(vgs, k1, k2, x, vth):
        return k1 * np.maximum(vgs - vth, 0)**x + k2

    # Initial guess and bounds
    p0 = [0.1, 0.0, 2.0, v_th_est]
    bounds = ([1e-6, -10, 1.0, 0], [100, 10, 6.0, 10])

    try:
        popt, _ = curve_fit(transfer_func, vgs_array, id_array, p0=p0, bounds=bounds, maxfev=5000)
        k1_fit, k2_fit, x_fit, vth_fit = popt
    except Exception:
        # Fallback to linear approximation
        # V_th = min V_gs with current > 0
        vth_fit = v_th_est
        # Linear gm = delta_I / delta_V_gs
        if len(vgs_array) > 1:
            gm_linear = (id_array[-1] - id_array[0]) / (vgs_array[-1] - vgs_array[0])
        else:
            gm_linear = 1.0
        k1_fit = float(gm_linear)
        k2_fit = 0.0
        x_fit = 1.0

    return TransconductanceParams(
        k_1=float(k1_fit),
        k_2=float(k2_fit),
        x=float(x_fit),
        v_th=float(vth_fit),
    )


class ChristenBielaModel:
    """Full Christen-Biela analytical switching loss model for half-bridge MOSFETs.

    Reference: Christen, Biela, IEEE TPEL 2019, DOI: 10.1109/TPEL.2018.2851068
    All equations use supplement errata corrections.

    :param c_gs: Gate-source capacitance (charge-equivalent) in F.
    :param c_ds: Drain-source capacitance (charge-equivalent) in F.
    :param c_gd: Gate-drain capacitance (charge-equivalent) in F.
    :param q_oss: Charge stored in C_oss of ONE switch at V_0 in C.
    :param gm_params: Transconductance model parameters.
    :param rr_params: Reverse recovery parameters (None = skip reverse recovery).
    """

    def __init__(
        self,
        c_gs: float,
        c_ds: float,
        c_gd: float,
        q_oss: float,
        gm_params: TransconductanceParams,
        rr_params: ReverseRecoveryParams | None = None,
    ) -> None:
        """Initialize Christen-Biela model with device parameters.

        :param c_gs: Gate-source capacitance in F.
        :param c_ds: Drain-source capacitance in F.
        :param c_gd: Gate-drain capacitance in F.
        :param q_oss: Charge stored in C_oss at V_0 in C.
        :param gm_params: Transconductance parameters.
        :param rr_params: Reverse recovery parameters (optional).
        """
        self.c_gs = c_gs
        self.c_ds = c_ds
        self.c_gd = c_gd
        self.c_oss = c_gd + c_ds
        self.q_oss = q_oss
        self.gm_params = gm_params
        self.rr_params = rr_params

    def _solve_i_oss(
        self,
        i_0: float,
        v_g: float,
        r_g: float,
        l_s: float,
        gm_initial: float | None = None,
        tol: float = 0.01,
        max_iter: int = 20,
    ) -> tuple[float, float, float]:
        """Solve iterative quadratic for I_oss with gm convergence.

        Implements eq(10) with CORRECTED sign (supplement confirms eq(10) form):
            0 = (2*L_s / (Q_oss*R_g)) * I_oss^2
              + (2/(g_m*R_g) + C_gd/(C_gd+C_ds)) * I_oss
              + (1/R_g) * (V_g - V_th - I_0/g_m)

        ITERATION (from supplement corrected flowchart):
        - For turn-off: g_m = f(I_0) initially, then g_m = f(I_0 - 2*I_oss)
        - Repeat until |g_m_new - g_m_old| / g_m_new < tol (1%)

        :param i_0: Load current in A.
        :param v_g: Gate voltage in V (V_g,off for turn-off, V_g,on for turn-on).
        :param r_g: Gate resistance in Ohm.
        :param l_s: Source inductance in H.
        :param gm_initial: Initial gm guess in S (if None, use gm(i_0)).
        :param tol: Convergence tolerance (relative).
        :param max_iter: Maximum iterations.
        :return: Tuple of (i_oss, gm_converged, v_mil).
        """
        if i_0 == 0:
            return (0.0, 0.01, self.gm_params.v_th)

        # Initial gm estimate
        if gm_initial is None:
            gm = self.gm_params.calc_gm(i_0)
        else:
            gm = gm_initial

        i_oss = 0.0

        for _iteration in range(max_iter):
            # Quadratic coefficients (eq 10)
            a = (2 * l_s) / (self.q_oss * r_g)
            b = (2 / (gm * r_g)) + (self.c_gd / (self.c_gd + self.c_ds))
            c = (1 / r_g) * (v_g - self.gm_params.v_th - i_0 / gm)

            # Solve quadratic: a*I_oss^2 + b*I_oss + c = 0
            discriminant = b**2 - 4*a*c

            if discriminant < 0:
                # No real solution: ZVS condition
                i_oss_new = 0.0
            else:
                # Two roots: select physically meaningful one
                root1 = (-b + np.sqrt(discriminant)) / (2*a)
                root2 = (-b - np.sqrt(discriminant)) / (2*a)

                # For turn-off: I_oss should be positive and < I_0/2
                # For turn-on: I_oss will be negative (select more negative root)
                if v_g < 0 or v_g < self.gm_params.v_th:
                    # Turn-off: select root closer to 0
                    i_oss_new = min(abs(root1), abs(root2))
                else:
                    # Turn-on: select more negative root
                    i_oss_new = min(root1, root2)

            # Update gm based on channel current: i_ch = I_0 - 2*I_oss
            i_ch = i_0 - 2 * i_oss_new
            if i_ch <= 0:
                i_ch = 0.1  # Minimal current for gm calculation

            gm_new = self.gm_params.calc_gm(i_ch)

            # Check convergence
            if abs(gm_new - gm) / max(gm_new, 1e-6) < tol:
                i_oss = i_oss_new
                gm = gm_new
                break

            gm = gm_new
            i_oss = i_oss_new

        # Calculate Miller voltage
        i_ch_final = i_0 - 2 * i_oss
        v_mil = self.gm_params.v_th + i_ch_final / max(gm, 1e-6)

        return (float(i_oss), float(gm), float(v_mil))

    def calc_turn_off_energy(self, params: HalfBridgeParams) -> dict[str, float]:
        """Calculate turn-off switching energy E_T,off.

        Intervals:
        - 0a: gate drops to V_g,off (lossless)
        - 1a: voltage rises, eq(10) for I_oss, eq(13) for t_rv
        - 2a: current falls, CORRECTED eq(15) for t_fi, eq(16) for V_Ld

        CORRECTED eq(15):
            t_fi = -ln((V_th - V_g) / (V_mil - V_g)) * (C_gs*R_g + L_s*g_m)
            Where V_g = V_g,off (the TURN-OFF gate supply voltage)

        eq(17):
            E_T,off = 0.5*t_rv*V_0*(I_0 - 2*I_oss)
                    + 0.5*t_fi*(V_0 + V_Ld)*(I_0 - 2*I_oss)

        :param params: Half-bridge operating parameters.
        :return: Dict with e_off, i_oss, v_mil, t_rv, t_fi, v_ld, gm.
        """
        if params.i_0 == 0:
            return {
                "e_off": 0.0, "i_oss": 0.0, "v_mil": self.gm_params.v_th,
                "t_rv": 0.0, "t_fi": 0.0, "v_ld": 0.0, "gm": 0.01
            }

        # Solve for I_oss (interval 1a)
        i_oss, gm, v_mil = self._solve_i_oss(
            params.i_0, params.v_g_off, params.r_g, params.l_s
        )

        # Check ZVS condition
        if params.i_0 < 2 * i_oss:
            # Near-ZVS: minimal turn-off loss
            return {
                "e_off": 0.0, "i_oss": float(i_oss), "v_mil": float(v_mil),
                "t_rv": 0.0, "t_fi": 0.0, "v_ld": 0.0, "gm": float(gm)
            }

        # Voltage rise time (eq 13)
        t_rv = self.q_oss / max(i_oss, 1e-9)

        # Current fall time (CORRECTED eq 15)
        # t_fi = -ln((V_th - V_g,off) / (V_mil - V_g,off)) * (C_gs*R_g + L_s*gm)
        numerator = self.gm_params.v_th - params.v_g_off
        denominator = v_mil - params.v_g_off

        if denominator <= 0 or numerator <= 0:
            t_fi = 0.0  # Channel already off
        else:
            t_fi = -np.log(numerator / denominator) * (
                self.c_gs * params.r_g + params.l_s * gm
            )
            t_fi = max(t_fi, 0.0)  # Ensure positive

        # Overvoltage from drain inductance (eq 16)
        i_ch = params.i_0 - 2 * i_oss
        v_ld = params.l_d * i_ch / max(t_fi, 1e-12) if t_fi > 0 else 0.0

        # Turn-off energy (eq 17)
        e_interval_1a = 0.5 * t_rv * params.v_0 * i_ch
        e_interval_2a = 0.5 * t_fi * (params.v_0 + v_ld) * i_ch
        e_off = e_interval_1a + e_interval_2a

        return {
            "e_off": float(e_off),
            "i_oss": float(i_oss),
            "v_mil": float(v_mil),
            "t_rv": float(t_rv),
            "t_fi": float(t_fi),
            "v_ld": float(v_ld),
            "gm": float(gm),
        }

    def calc_i_0_zvs(self, params: HalfBridgeParams) -> float:
        """Calculate maximum current for lossless turn-off (ZVS boundary).

        Implements eq(14):
            I_0,zvs = V_0/(2*L_s) * (-R_g*C_gd
                      + sqrt((R_g*C_gd)^2 - 8*(V_g-V_th)*L_s*(C_gd+C_ds)/V_0))

        Where V_g = V_g,off.

        :param params: Half-bridge operating parameters.
        :return: Maximum ZVS current in A (0 if no ZVS possible).
        """
        v_g = params.v_g_off

        # Calculate discriminant
        term1 = (params.r_g * self.c_gd)**2
        term2 = (
            8 * (v_g - self.gm_params.v_th) * params.l_s
            * (self.c_gd + self.c_ds) / params.v_0
        )
        discriminant = term1 - term2

        if discriminant < 0:
            return 0.0  # No ZVS possible

        i_0_zvs = (params.v_0 / (2 * params.l_s)) * (
            -params.r_g * self.c_gd + np.sqrt(discriminant)
        )

        return max(float(i_0_zvs), 0.0)

    def _calc_reverse_recovery(
        self,
        params: HalfBridgeParams,
        i_oss_on: float,
        gm_1b: float,
        t_ri: float,
    ) -> dict[str, float]:
        """Calculate reverse recovery energy contribution during turn-on.

        Implements interval 2b (reverse recovery) from Christen/Biela Section II-B.
        Uses supplement corrections for time constants and energy calculations.

        Energy components:
        - E_rs: Storage phase energy (triangular current rise)
        - E_rf: Fall phase energy (exponential current decay)
        - E_rr,s2 = E_rs + E_rf: Total reverse recovery energy

        Equations from supplement (corrected):
        - t_rs: Storage time from eq(30) with tau_c correction
        - Q_rs: Recovery charge from eq(29)
        - I_rr,calc: Peak recovery current (calculated, not from datasheet)

        :param params: Half-bridge operating parameters.
        :param i_oss_on: C_oss discharge current during turn-on in A.
        :param gm_1b: Transconductance in phase 1b in S.
        :param t_ri: Current rise time in s.
        :return: Dict with e_rr_s2, e_rs, e_rf, i_rr_calc, t_rs, q_rs.
        """
        if self.rr_params is None:
            # No reverse recovery model: return zeros
            return {
                "e_rr_s2": 0.0, "e_rs": 0.0, "e_rf": 0.0,
                "i_rr_calc": 0.0, "t_rs": 0.0, "q_rs": 0.0,
            }

        if params.i_0 == 0:
            return {
                "e_rr_s2": 0.0, "e_rs": 0.0, "e_rf": 0.0,
                "i_rr_calc": 0.0, "t_rs": 0.0, "q_rs": 0.0,
            }

        rr = self.rr_params

        # Calculate di/dt during current rise (interval 1b)
        # di/dt = I_0 / t_ri
        di_dt = params.i_0 / max(t_ri, 1e-12)

        # Storage time t_rs (eq 30 with tau_c correction from supplement)
        # t_rs = tau_c * ln(1 + di_dt * tau_c / I_0)
        t_rs = rr.tau_c * np.log(1 + di_dt * rr.tau_c / params.i_0)

        # Recovery charge Q_rs (eq 29)
        # Q_rs = (I_0 * t_rs) - 0.5 * di_dt * t_rs^2
        q_rs = params.i_0 * t_rs - 0.5 * di_dt * t_rs**2

        # Peak recovery current I_rr,calc (calculated from model)
        # I_rr,calc = di_dt * t_rs
        i_rr_calc = di_dt * t_rs

        # Storage phase energy E_rs (triangular rise)
        # E_rs = 0.5 * V_0 * I_rr,calc * t_rs
        e_rs = 0.5 * params.v_0 * i_rr_calc * t_rs

        # Fall phase energy E_rf (exponential decay)
        # E_rf = V_0 * I_rr,calc * tau_rr
        e_rf = params.v_0 * i_rr_calc * rr.tau_rr

        # Total reverse recovery energy
        e_rr_s2 = e_rs + e_rf

        return {
            "e_rr_s2": float(e_rr_s2),
            "e_rs": float(e_rs),
            "e_rf": float(e_rf),
            "i_rr_calc": float(i_rr_calc),
            "t_rs": float(t_rs),
            "q_rs": float(q_rs),
        }

    def calc_turn_on_energy(self, params: HalfBridgeParams) -> dict[str, float]:
        """Calculate turn-on switching energy E_T,on.

        Implements Section II-B intervals 0b, 1b, 2b, 3b with supplement corrections.

        Intervals:
        - 0b: Gate rises to V_th (lossless)
        - 1b: Current rises, solving for I_oss via eq(10), time via CORRECTED eq(20)
        - 2b: Reverse recovery (if rr_params provided)
        - 3b: Voltage falls, time via eq(26), energy via eq(28)

        CORRECTED eq(20):
            t_ri = ln((V_g,on - V_th) / (V_g,on - V_mil,1b))
                   * (C_gs*R_g + L_s*g_m,1b)

        CORRECTED eq(22):
            I_oss,on solved from eq(10) with V_g = V_g,on

        eq(26):
            t_fv = Q_oss / (I_0 + 2*I_oss,on)

        eq(28):
            E_T,on = 0.5*V_0*(I_0 - 2*I_oss,on)*t_ri + E_rr,s2
                   + 0.5*(V_0 - V_Ld)*(I_0 + 2*I_oss,on)*t_fv

        :param params: Half-bridge operating parameters.
        :return: Dict with e_on, i_oss_on, v_mil_on, t_ri, t_fv, v_ld_on,
                 gm_1b, gm_3b, plus reverse recovery breakdown.
        """
        if params.i_0 == 0:
            return {
                "e_on": 0.0, "i_oss_on": 0.0, "v_mil_on": self.gm_params.v_th,
                "t_ri": 0.0, "t_fv": 0.0, "v_ld_on": 0.0,
                "gm_1b": 0.01, "gm_3b": 0.01,
                "e_rr_s2": 0.0, "e_rs": 0.0, "e_rf": 0.0,
                "i_rr_calc": 0.0, "t_rs": 0.0, "q_rs": 0.0,
            }

        # --- Interval 1b: Current rise ---
        # Solve for I_oss,on using eq(10) with V_g = V_g,on
        # During turn-on, I_oss is typically negative (discharging C_oss)
        i_oss_on, gm_1b, v_mil_1b = self._solve_i_oss(
            params.i_0, params.v_g_on, params.r_g, params.l_s
        )

        # Current rise time (CORRECTED eq 20)
        # t_ri = ln((V_g,on - V_th) / (V_g,on - V_mil,1b)) * (C_gs*R_g + L_s*g_m,1b)
        numerator = params.v_g_on - self.gm_params.v_th
        denominator = params.v_g_on - v_mil_1b

        if denominator <= 0 or numerator <= 0:
            t_ri = 0.0  # Channel already fully on
        else:
            t_ri = np.log(numerator / denominator) * (
                self.c_gs * params.r_g + params.l_s * gm_1b
            )
            t_ri = max(t_ri, 0.0)

        # --- Interval 2b: Reverse recovery ---
        rr_result = self._calc_reverse_recovery(params, i_oss_on, gm_1b, t_ri)

        # --- Interval 3b: Voltage fall ---
        # Voltage fall time (eq 26)
        # t_fv = Q_oss / (I_0 + 2*I_oss,on)
        i_cap = params.i_0 + 2 * i_oss_on
        t_fv = self.q_oss / max(abs(i_cap), 1e-9)

        # Undervoltage from drain inductance during voltage fall
        # V_Ld = L_d * (I_0 + 2*I_oss,on) / t_fv
        v_ld_on = params.l_d * abs(i_cap) / max(t_fv, 1e-12) if t_fv > 0 else 0.0

        # Transconductance in phase 3b (at I_0 + 2*I_oss,on)
        i_ch_3b = params.i_0 + 2 * i_oss_on
        gm_3b = self.gm_params.calc_gm(abs(i_ch_3b))

        # --- Turn-on energy (eq 28) ---
        # E_T,on = 0.5*V_0*(I_0 - 2*I_oss,on)*t_ri + E_rr,s2
        #        + 0.5*(V_0 - V_Ld)*(I_0 + 2*I_oss,on)*t_fv
        e_interval_1b = 0.5 * params.v_0 * (params.i_0 - 2 * i_oss_on) * t_ri
        e_interval_2b = rr_result["e_rr_s2"]
        e_interval_3b = 0.5 * (params.v_0 - v_ld_on) * abs(i_cap) * t_fv

        e_on = e_interval_1b + e_interval_2b + e_interval_3b

        return {
            "e_on": float(e_on),
            "i_oss_on": float(i_oss_on),
            "v_mil_on": float(v_mil_1b),
            "t_ri": float(t_ri),
            "t_fv": float(t_fv),
            "v_ld_on": float(v_ld_on),
            "gm_1b": float(gm_1b),
            "gm_3b": float(gm_3b),
            **rr_result,  # Include all reverse recovery details
        }

    def calc_switching_energy(
        self, params: HalfBridgeParams
    ) -> SwitchingEnergyResult:
        """Calculate complete switching energy with full breakdown.

        Combines turn-on and turn-off energy calculations with detailed
        intermediate results for analysis and debugging.

        :param params: Half-bridge operating parameters.
        :return: SwitchingEnergyResult with all energy components
                 and intermediate values.
        """
        # Calculate turn-off energy
        off_result = self.calc_turn_off_energy(params)

        # Calculate turn-on energy
        on_result = self.calc_turn_on_energy(params)

        # Calculate ZVS threshold
        i_0_zvs = self.calc_i_0_zvs(params)

        # Combine into comprehensive result
        return SwitchingEnergyResult(
            # Summary
            e_on=on_result["e_on"],
            e_off=off_result["e_off"],
            e_total=on_result["e_on"] + off_result["e_off"],
            i_0_zvs=i_0_zvs,
            # Turn-off breakdown
            i_oss_off=off_result["i_oss"],
            v_mil_off=off_result["v_mil"],
            t_rv=off_result["t_rv"],
            t_fi=off_result["t_fi"],
            v_ld_off=off_result["v_ld"],
            gm_off=off_result["gm"],
            # Turn-on breakdown
            i_oss_on=on_result["i_oss_on"],
            v_mil_on=on_result["v_mil_on"],
            t_ri=on_result["t_ri"],
            t_fv=on_result["t_fv"],
            v_ld_on=on_result["v_ld_on"],
            gm_on_1b=on_result["gm_1b"],
            gm_on_3b=on_result["gm_3b"],
            # Reverse recovery
            i_rr_calc=on_result["i_rr_calc"],
            t_rs=on_result["t_rs"],
            q_rs=on_result["q_rs"],
            e_rs=on_result["e_rs"],
            e_rf=on_result["e_rf"],
            e_rr_s2=on_result["e_rr_s2"],
        )

    @classmethod
    def from_transistor(
        cls, transistor: Transistor, v_0: float, t_j: float = 25.0
    ) -> ChristenBielaModel:
        """Create model from a TDB Transistor object.

        Extracts all parameters automatically:
        1. Charge-equivalent capacitances from C(V) curves at operating voltage v_0
        2. Transconductance fit from channel data
        3. Reverse recovery params from diode fields (if available)

        :param transistor: Transistor object with capacitance and channel data.
        :param v_0: Operating DC bus voltage in V.
        :param t_j: Junction temperature in deg C.
        :return: ChristenBielaModel instance.
        :raises ValueError: If required data (C_oss, channel data) is missing.
        """
        # 1. Extract capacitances
        if not transistor.c_oss or not transistor.c_iss or not transistor.c_rss:
            raise ValueError(
                "Transistor must have c_oss, c_iss, and c_rss data for Christen-Biela model"
            )

        # Find capacitance data closest to t_j
        def find_closest_temp(cap_list, target_t_j):
            if not cap_list:
                return None
            return min(cap_list, key=lambda c: abs(c.t_j - target_t_j))

        c_oss_data = find_closest_temp(transistor.c_oss, t_j)
        c_iss_data = find_closest_temp(transistor.c_iss, t_j)
        c_rss_data = find_closest_temp(transistor.c_rss, t_j)

        if not c_oss_data or not c_iss_data or not c_rss_data:
            raise ValueError("Capacitance data not found at specified temperature")

        # Calculate charge-equivalent capacitances
        v_axis_oss = c_oss_data.graph_v_c[0]
        c_axis_oss = c_oss_data.graph_v_c[1]
        c_oss_eq = calc_charge_equivalent_capacitance(v_axis_oss, c_axis_oss, v_0)
        q_oss = calc_charge_stored(v_axis_oss, c_axis_oss, v_0)

        v_axis_iss = c_iss_data.graph_v_c[0]
        c_axis_iss = c_iss_data.graph_v_c[1]
        c_iss_eq = calc_charge_equivalent_capacitance(v_axis_iss, c_axis_iss, v_0)

        v_axis_rss = c_rss_data.graph_v_c[0]
        c_axis_rss = c_rss_data.graph_v_c[1]
        c_rss_eq = calc_charge_equivalent_capacitance(v_axis_rss, c_axis_rss, v_0)

        c_gs, c_ds, c_gd = calc_device_capacitances(c_iss_eq, c_oss_eq, c_rss_eq)

        # 2. Fit transconductance
        if not transistor.switch.channel_data:
            raise ValueError("Transistor must have switch channel data")

        # Find channel data at t_j (or nearest)
        channel_at_tj = [
            ch for ch in transistor.switch.channel_data
            if abs(ch.t_j - t_j) < 30  # Within 30C tolerance
        ]

        if not channel_at_tj:
            raise ValueError(f"No channel data found near t_j={t_j}C")

        gm_params = fit_transconductance(channel_at_tj)

        # 3. Extract reverse recovery params (if available)
        rr_params = None
        if (transistor.diode.q_rr is not None and
            transistor.diode.i_rr is not None and
            transistor.diode.di_dt_rr is not None):
            try:
                rr_params = extract_reverse_recovery_params(
                    q_rr=transistor.diode.q_rr,
                    i_rr=transistor.diode.i_rr,
                    di_dt=transistor.diode.di_dt_rr,
                    q_oss=q_oss,
                    t_rr=transistor.diode.t_rr,
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to extract reverse recovery params: {e}. Skipping reverse recovery.",
                    stacklevel=2
                )

        return cls(c_gs, c_ds, c_gd, q_oss, gm_params, rr_params)

    def calc_switching_loss_curve(
        self,
        v_0: float,
        currents: npt.NDArray[np.float64],
        v_g_on: float = 15.0,
        v_g_off: float = -5.0,
        r_g: float = 10.0,
        l_s: float = 4e-9,
        l_d: float = 10e-9,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate switching loss curves over a range of currents.

        :param v_0: DC bus voltage in V.
        :param currents: Array of current values in A.
        :param v_g_on: Turn-on gate voltage in V.
        :param v_g_off: Turn-off gate voltage in V.
        :param r_g: Gate resistance in Ohm.
        :param l_s: Source inductance in H.
        :param l_d: Drain inductance in H.
        :return: Tuple of (e_on_array, e_off_array, e_total_array) in J.
        """
        e_on_list = []
        e_off_list = []

        for i_0 in currents:
            params = HalfBridgeParams(
                v_0=v_0, i_0=float(i_0),
                v_g_on=v_g_on, v_g_off=v_g_off,
                r_g=r_g, l_s=l_s, l_d=l_d
            )
            result = self.calc_switching_energy(params)
            e_on_list.append(result.e_on)
            e_off_list.append(result.e_off)

        e_on = np.array(e_on_list)
        e_off = np.array(e_off_list)
        return (e_on, e_off, e_on + e_off)
