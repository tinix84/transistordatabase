"""Bounded Christen–Biela half-bridge estimate (2018 manuscript + errata).

SI throughout. The model's gm is I/(Vgs-Vth), a secant conductance, not dI/dVgs.
No measured accuracy or temperature scaling is inferred from supplied constants.
See docs/api/christen-biela.md for equation provenance and supported domains.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

from transistordatabase.core.models import ChannelCharacteristics, Transistor


class BielaConvergenceError(ValueError):
    """The numerical method failed within a potentially supported estimate domain."""


def _finite(**values: float) -> None:
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError('All model inputs and derived quantities must be finite')


def _checked_result(result: dict) -> dict:
    """Reject nonfinite phase results at public single-event boundaries too."""
    for value in result.values():
        if isinstance(value, dict):
            _checked_result(value)
        elif isinstance(value, (float, int, np.number)):
            _finite(result=value)
    return result


@dataclass
class HalfBridgeParams:
    """Bus/load/driver conditions; total Rg in ohms and inductances in H."""

    v_0: float
    i_0: float
    v_g_on: float
    v_g_off: float
    r_g: float
    l_s: float = 4e-9
    l_d: float = 10e-9
    t_j: float | None = None


@dataclass
class TransconductanceParams:
    """I = k1*(Vgs-Vth)**x + k2; optional measured transfer-current bounds."""

    k_1: float
    k_2: float
    x: float
    v_th: float
    current_min: float | None = None
    current_max: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def calc_gm(self, i_ch: float) -> float:
        """Return manuscript secant gm in S; reject unsupported channel currents."""
        _finite(k_1=self.k_1, k_2=self.k_2, x=self.x, v_th=self.v_th, i_ch=i_ch)
        if self.k_1 <= 0 or self.x <= 0 or self.v_th <= 0:
            raise ValueError('Require k1 > 0, exponent x > 0 and Vth > 0')
        bounds = [v for v in (self.current_min, self.current_max) if v is not None]
        if any(not np.isfinite(v) or v <= 0 for v in bounds):
            raise ValueError('Transfer-current coverage bounds must be finite and positive')
        if len(bounds) == 2 and bounds[0] >= bounds[1]:
            raise ValueError('Transfer-current coverage minimum must be below maximum')
        if i_ch <= 0 or i_ch <= self.k_2:
            raise ValueError('Channel current must be positive and greater than k2')
        if ((self.current_min is not None and i_ch < self.current_min)
                or (self.current_max is not None and i_ch > self.current_max)):
            raise ValueError('Channel current is outside the fitted transfer-current coverage')
        with np.errstate(over='ignore', invalid='ignore'):
            gm = i_ch * (self.k_1 / (i_ch - self.k_2)) ** (1 / self.x)
        _finite(gm=gm)
        if gm <= 0:
            raise ValueError('Secant transconductance must be positive')
        return float(gm)


@dataclass
class ReverseRecoveryParams:
    """Charge-control time constants, all seconds; corrected charge in C."""

    tau_rr: float
    tau_c: float
    t_m: float
    q_rr_star: float

    def validate(self) -> None:
        """Reject nonpositive or mutually inconsistent charge-control constants."""
        _finite(**asdict(self))
        if min(self.tau_rr, self.tau_c, self.t_m, self.q_rr_star) <= 0:
            raise ValueError('Recovery constants and corrected charge must be positive')
        if self.tau_c <= self.tau_rr:
            raise ValueError('Require tau_c > tau_rr for a positive transit time')
        expected = 1 / self.tau_rr - 1 / self.tau_c
        if not np.isclose(1 / self.t_m, expected, rtol=1e-5, atol=0):
            raise ValueError('Recovery constants must satisfy 1/Tm = 1/tau_rr - 1/tau_c')


@dataclass
class SwitchingEnergyResult:
    """Detailed SI result; diagnostics retain convergence and energy partitions."""

    e_on: float
    e_off: float
    e_total: float
    i_0_zvs: float
    i_oss_off: float
    v_mil_off: float
    t_rv: float
    t_fi: float
    v_ld_off: float
    gm_off: float
    i_oss_on: float
    v_mil_on: float
    t_ri: float
    t_fv: float
    v_ld_on: float
    gm_on_1b: float
    gm_on_3b: float
    i_rr_calc: float
    t_rs: float
    q_rs: float
    e_rs: float
    e_rf: float
    e_rr_s2: float
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return all SI values and diagnostics, not only a four-field summary."""
        return asdict(self)


def calc_charge_stored(v_axis, c_axis, v_0: float) -> float:
    """Integrate single-device C(V) only over recorded coverage [0,V0]."""
    from transistordatabase.device_parameters import integrate_capacitance
    return integrate_capacitance(v_axis, c_axis, 0, v_0)['charge_c']


def calc_charge_equivalent_capacitance(v_axis, c_axis, v_0: float) -> float:
    """Charge-equivalent C = integral(C dV)/V0; no endpoint extrapolation."""
    _finite(v_0=v_0)
    if v_0 <= 0:
        raise ValueError('V0 must be positive')
    return calc_charge_stored(v_axis, c_axis, v_0) / v_0


def calc_device_capacitances(c_iss_eq, c_oss_eq, c_rss_eq) -> tuple[float, float, float]:
    """Decompose valid terminal capacitances without silently clamping data."""
    from transistordatabase.device_parameters import decompose_capacitances
    values = decompose_capacitances(c_iss_eq, c_oss_eq, c_rss_eq)
    return values['c_gs_f'], values['c_ds_f'], values['c_gd_f']


def extract_reverse_recovery_params(
    q_rr: float, i_rr: float, di_dt: float, q_oss: float,
    t_rr: float | None = None, *, i_forward_test: float | None = None,
) -> ReverseRecoveryParams:
    """Extract eqs 38–41 using the required test forward current and errata.

    t_rr is retained for compatibility, but the model integrates the exponential
    tail to infinity and does not use the datasheet threshold-defined duration.
    """
    if i_forward_test is None:
        raise ValueError('Recovery extraction requires i_forward_test at the same test condition')
    _finite(q_rr=q_rr, i_rr=i_rr, di_dt=di_dt, q_oss=q_oss,
            i_forward_test=i_forward_test)
    if min(q_rr, i_rr, di_dt, i_forward_test) <= 0 or q_oss < 0:
        raise ValueError('Require positive Qrr, Irr, slope and forward current; Qoss >= 0')
    q_star = q_rr - q_oss
    q_rf = q_star - i_rr ** 2 / (2 * di_dt)
    if q_star <= 0 or q_rf <= 0:
        raise ValueError('Corrected Qrr and recovery fall charge Qrf must be positive')
    tau_rr = q_rf / i_rr
    t_1 = (i_forward_test + i_rr) / di_dt
    # Dimensionless solve avoids a seconds-scale xtol masking a failed root.
    def residual(ratio):
        return di_dt * tau_rr * (ratio - 1) * (-np.expm1(-t_1 / (tau_rr * ratio))) - i_rr
    upper = 2.0
    while residual(upper) < 0 and upper < 1e12:
        upper *= 2
    if residual(upper) <= 0:
        raise ValueError('Recovery extraction has no bounded physical root')
    ratio = brentq(residual, 1, upper, xtol=1e-12, rtol=1e-12)
    result = ReverseRecoveryParams(tau_rr, tau_rr * ratio,
                                  tau_rr * ratio / (ratio - 1), q_star)
    result.validate()
    return result


def fit_transconductance(
    channel_data_list: list[ChannelCharacteristics], *,
    v_ds: float | None = None, t_j: float | None = None, v_th: float | None = None,
) -> TransconductanceParams:
    """Fit three transfer coefficients at explicit common Vds/Tj and known Vth.

    At least four distinct gate voltages, full-rank Jacobian and bounded current
    coverage are required. A failed nonlinear fit never becomes a linear model.
    """
    from transistordatabase.condition_tables import bounded_interpolate
    if v_ds is None or t_j is None or v_th is None:
        raise ValueError('Transfer fitting requires explicit common v_ds, t_j and known v_th')
    _finite(v_ds=v_ds, t_j=t_j, v_th=v_th)
    if v_ds <= 0 or v_th <= 0:
        raise ValueError('Transfer Vds and Vth must be positive')
    points = []
    for curve in channel_data_list:
        if curve.t_j != t_j:
            continue
        if curve.v_g is None or curve.v_g <= v_th:
            raise ValueError('Every selected transfer gate voltage must be known and above Vth')
        graph = np.asarray(curve.graph_v_i, dtype=float)
        if graph.ndim != 2 or graph.shape[0] != 2:
            raise ValueError('Channel graph must have voltage and current rows')
        if graph.shape[1] == 1:
            if not np.all(np.isfinite(graph)) or graph[0, 0] != v_ds:
                raise ValueError('A single transfer sample must record the exact selected Vds')
            current = float(graph[1, 0])
        else:
            current = bounded_interpolate(graph[0], graph[1], v_ds, label='transfer Vds')
        if current <= 0:
            raise ValueError('Selected transfer currents must be positive')
        points.append((curve.v_g, current))
    if len(points) < 4 or len({p[0] for p in points}) != len(points):
        raise ValueError('Need at least four distinct gate voltages at the exact temperature')
    points.sort()
    gate, current = np.array(points).T
    if np.any(np.diff(current) <= 0):
        raise ValueError('Transfer current must increase strictly with gate voltage')
    scale = float(max(current))
    overdrive = gate - v_th
    def residual(p):
        return (np.exp(p[0]) * overdrive ** p[2] + p[1] - current) / scale
    fit = least_squares(residual, [np.log(scale / max(overdrive) ** 2), 0, 2],
                        bounds=([-100, -10 * scale, 0.1], [100, min(current), 10]),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if (not fit.success or np.linalg.matrix_rank(fit.jac) < 3
            or singular[-1] / singular[0] < 1e-8):
        raise ValueError('Transfer fit failed or has insufficient numerical rank')
    prediction = np.exp(fit.x[0]) * overdrive ** fit.x[2] + fit.x[1]
    diagnostics = {'rmse_a': float(np.sqrt(np.mean((prediction - current) ** 2))),
                   'max_abs_residual_a': float(max(abs(prediction - current))),
                   'rank': 3, 'jacobian_condition': float(singular[0] / singular[-1]),
                   'points': len(points), 'v_ds_v': v_ds, 't_j_c': t_j,
                   'v_g_min_v': float(min(gate)), 'v_g_max_v': float(max(gate)),
                   'scope': 'Nonlinear transfer fit; accuracy outside current coverage is unsupported'}
    return TransconductanceParams(float(np.exp(fit.x[0])), float(fit.x[1]),
                                 float(fit.x[2]), v_th,
                                 float(min(current)), float(max(current)), diagnostics)


class ChristenBielaModel:
    """Explicit half-bridge estimate with corrected manuscript phase equations.

    Constant charge-equivalent capacitances and single-device Qoss must describe
    the specified bus voltage and temperature. Missing RR is an explicit omission.
    """

    def __init__(self, c_gs, c_ds, c_gd, q_oss, gm_params, rr_params=None, *,
                 v_ref=None, t_j=None, source=None):
        _finite(c_gs=c_gs, c_ds=c_ds, c_gd=c_gd, q_oss=q_oss)
        if min(c_gs, c_ds, c_gd, q_oss) <= 0:
            raise ValueError('Require positive Cgs, Cds, Cgd and single-device Qoss')
        self.c_gs, self.c_ds, self.c_gd, self.q_oss = c_gs, c_ds, c_gd, q_oss
        self.c_oss = c_ds + c_gd
        _finite(c_oss=self.c_oss)
        self.gm_params, self.rr_params = gm_params, rr_params
        conditions = {'v_ref_v': v_ref, 't_j_c': t_j, 'source': source}
        _finite(**{k: v for k, v in conditions.items() if k != 'source' and v is not None})
        if v_ref is not None and v_ref <= 0:
            raise ValueError('Reference voltage must be positive')
        self.reference_conditions = conditions
        if rr_params is not None:
            rr_params.validate()

    def _validate(self, p):
        _finite(**{k: v for k, v in asdict(p).items() if v is not None})
        if self.reference_conditions['v_ref_v'] is not None and p.v_0 != self.reference_conditions['v_ref_v']:
            raise ValueError('Model was extracted at a fixed reference bus voltage; re-extract for this V0')
        if (p.t_j is not None and self.reference_conditions['t_j_c'] is not None
                and p.t_j != self.reference_conditions['t_j_c']):
            raise ValueError('Model parameter temperature does not match requested Tj')
        if min(p.v_0, p.i_0, p.r_g) <= 0 or min(p.l_s, p.l_d) < 0:
            raise ValueError('Hard-switching estimate requires V0, I0, Rg > 0 and Ls, Ld >= 0')
        if not p.v_g_off < self.gm_params.v_th < p.v_g_on:
            raise ValueError('Require Voff < Vth < Von')
        expected_charge = self.c_oss * p.v_0
        _finite(charge_equivalent_qoss=expected_charge)
        if not np.isclose(self.q_oss, expected_charge, rtol=.01, atol=0):
            raise ValueError('Charge-equivalent Qoss must match (Cgd+Cds)*V0 within 1% source rounding')
        self.gm_params.calc_gm(p.i_0)

    def _solve_miller(self, p, gate, *, tol=1e-9, max_iter=200):
        gm = self.gm_params.calc_gm(p.i_0)
        for iteration in range(1, max_iter + 1):
            a = 2 * p.l_s / (self.q_oss * p.r_g)
            b = 2 / (gm * p.r_g) + self.c_gd / self.c_oss
            c = (gate - self.gm_params.v_th - p.i_0 / gm) / p.r_g
            disc = b * b - 4 * a * c
            if not np.isfinite(disc):
                raise ValueError('Miller quadratic exceeds the finite numerical range')
            if disc < 0:
                raise BielaConvergenceError(
                    'Miller iteration quadratic has negative discriminant; '
                    'no converged estimate established, and existence of a self-consistent root is unknown')
            # Stable branch continuous from the zero-inductance limit, both events.
            i_oss = -2 * c / (b + np.sqrt(disc))
            i_ch = p.i_0 - 2 * i_oss
            if gate < self.gm_params.v_th and i_oss <= 0:
                raise ValueError('Turn-off Miller charging current must be positive')
            if gate > self.gm_params.v_th and i_oss >= 0:
                raise ValueError('Turn-on Miller discharging current must be negative')
            try:
                updated = self.gm_params.calc_gm(i_ch)
            except ValueError as exc:
                raise BielaConvergenceError(
                    'Miller iteration left transfer-current coverage/domain; '
                    'no converged estimate solution was established: ' + str(exc)) from exc
            gm_residual = abs(updated - gm) / updated
            polynomial = a * i_oss ** 2 + (2 / (updated * p.r_g)
                         + self.c_gd / self.c_oss) * i_oss + (
                             gate - self.gm_params.v_th - p.i_0 / updated) / p.r_g
            if gm_residual < tol and abs(polynomial) < tol * max(1, abs(c)):
                return {'i_oss': float(i_oss), 'gm': updated,
                        'v_mil': self.gm_params.v_th + i_ch / updated,
                        'diagnostics': {'converged': True, 'iterations': iteration,
                                        'gm_relative_residual': float(gm_residual),
                                        'equation_residual_a': float(polynomial)}}
            gm = updated
        raise BielaConvergenceError('Miller fixed-point solve did not converge')

    def _solve_i_oss(self, i_0, v_g, r_g, l_s, gm_initial=None, tol=1e-9, max_iter=200):
        """Compatibility tuple interface; convergence failures are explicit."""
        p = HalfBridgeParams(1, i_0, self.gm_params.v_th + 1,
                             self.gm_params.v_th - 1, r_g, l_s, 0)
        result = self._solve_miller(p, v_g, tol=tol, max_iter=max_iter)
        return result['i_oss'], result['gm'], result['v_mil']

    def calc_i_0_zvs(self, params):
        """Equation10 boundary, equal to eq14 for exact charge-equivalent inputs."""
        self._validate(params)
        drive = self.gm_params.v_th - params.v_g_off
        b = params.r_g * self.c_gd / self.c_oss
        root = np.sqrt(b * b + 8 * drive * params.l_s / self.q_oss)
        # Use the same Qoss as eq10 even inside the disclosed rounding tolerance.
        result = 4 * drive / (b + root)
        _finite(zvs=result)
        return float(result)

    def calc_turn_off_energy(self, params):
        """Calculate corrected voltage-rise and current-fall energy partitions."""
        self._validate(params)
        if params.i_0 <= self.calc_i_0_zvs(params):
            raise ValueError('Current is at or below the turn-off ZVS boundary; hard-switching phases unsupported')
        m = self._solve_miller(params, params.v_g_off)
        i_ch = params.i_0 - 2 * m['i_oss']
        t_rv = self.q_oss / m['i_oss']
        t_fi = np.log((m['v_mil'] - params.v_g_off)
                      / (self.gm_params.v_th - params.v_g_off)) * (
                          self.c_gs * params.r_g + params.l_s * m['gm'])
        if t_fi <= 0:
            raise ValueError('Turn-off current-fall interval must be positive')
        v_ld = params.l_d * i_ch / t_fi
        first = 0.5 * t_rv * params.v_0 * i_ch
        second = 0.5 * t_fi * (params.v_0 + v_ld) * i_ch
        return _checked_result({'e_off': first + second, 'i_oss': m['i_oss'], 'gm': m['gm'],
                'v_mil': m['v_mil'], 't_rv': float(t_rv), 't_fi': float(t_fi),
                'v_ld': float(v_ld), 'diagnostics': m['diagnostics'],
                'energy_partitions_j': {'voltage_rise': first, 'current_fall': second}})

    def _calc_reverse_recovery(self, params, i_oss_on, gm_1b, t_ri, v_ds0=None, t_fv=None):
        names = ('e_rr_s2', 'e_rs', 'e_rf', 'i_rr_calc', 't_rs', 'q_rs',
                 'storage_load_energy_j', 'body_diode_energy_j', 'storage_residual_a')
        if self.rr_params is None:
            return dict.fromkeys(names, 0.0)
        rr = self.rr_params
        slope = params.i_0 / t_ri
        t0 = t_ri
        def charge(t):
            return slope * rr.tau_c * (t0 - t - rr.tau_c * np.expm1(-t / rr.tau_c))
        def residual(dt):
            t = t0 + dt * rr.tau_c
            return (charge(t) + rr.t_m * (params.i_0 - slope * t)) / (slope * rr.tau_c ** 2)
        upper = 1.0
        while residual(upper) > 0 and upper < 1e8:
            upper *= 2
        if residual(upper) >= 0:
            raise ValueError('Reverse-recovery storage root cannot be bracketed')
        dt = brentq(residual, 0, upper, xtol=1e-12, rtol=1e-12)
        trs = dt * rr.tau_c
        irr = slope * trs
        qrs = 0.5 * trs * irr
        # Manuscript eq 27: integrate linear Vds fall times exponential recovery.
        v_ds0 = params.v_0 if v_ds0 is None else v_ds0
        if t_fv is None or t_fv <= 0:
            raise ValueError('Recovery energy requires a positive voltage-fall interval')
        ratio = t_fv / rr.tau_rr
        decay = np.exp(-ratio)
        if ratio < 1e-4:
            # Analytic limits avoid cancellation when the recovery tail is long.
            switch_fraction = ratio / 2 - ratio ** 2 / 6 + ratio ** 3 / 24 - ratio ** 4 / 120
            diode_fraction = ratio / 2 - ratio ** 2 / 3 + ratio ** 3 / 8 - ratio ** 4 / 30
        else:
            switch_fraction = 1 + np.expm1(-ratio) / ratio
            diode_fraction = -np.expm1(-ratio) / ratio - decay
        erf = v_ds0 * irr * rr.tau_rr * switch_fraction
        # Eq 28: diode voltage rises to Vds0 during tfv, then is V0 in the
        # remaining exponential tail. Vds0 differs from V0 when Ld is nonzero.
        body = (irr * v_ds0 * rr.tau_rr * diode_fraction
                + rr.tau_rr * irr * params.v_0 * decay)
        return {'e_rr_s2': qrs * v_ds0 + erf, 'e_rs': qrs * v_ds0, 'e_rf': erf,
                'i_rr_calc': irr, 't_rs': trs, 'q_rs': qrs,
                'storage_load_energy_j': trs * v_ds0 * params.i_0,
                'body_diode_energy_j': body,
                'storage_residual_a': residual(dt) * slope * rr.tau_c}

    def calc_turn_on_energy(self, params):
        """Calculate current rise, storage/recovery and voltage-fall partitions."""
        self._validate(params)
        gm1 = self.gm_params.calc_gm(params.i_0)
        ratio = params.i_0 / (gm1 * (params.v_g_on - self.gm_params.v_th))
        if ratio >= 1:
            raise ValueError('Turn-on drive cannot support the load-current plateau')
        tri = -np.log1p(-ratio) * (self.c_gs * params.r_g + params.l_s * gm1)
        if not np.isfinite(tri) or tri <= 0:
            raise ValueError('Current-rise interval exceeds the finite positive numerical range')
        v_ld = params.l_d * params.i_0 / tri
        v_ds0 = params.v_0 - v_ld
        if v_ds0 <= 0:
            raise ValueError('Drain-inductance drop reaches bus voltage; phase approximation unsupported')
        m = self._solve_miller(params, params.v_g_on)
        tfv = -self.q_oss / m['i_oss']
        ich = params.i_0 - 2 * m['i_oss']
        rr = self._calc_reverse_recovery(params, m['i_oss'], gm1, tri, v_ds0, tfv)
        first = 0.5 * tri * v_ds0 * params.i_0
        third = 0.5 * tfv * ich * v_ds0
        return _checked_result({'e_on': first + third + rr['storage_load_energy_j'] + rr['e_rr_s2'],
                'i_oss_on': m['i_oss'], 'v_mil_on': m['v_mil'],
                't_ri': float(tri), 't_fv': float(tfv), 'v_ld_on': float(v_ld),
                'gm_1b': gm1, 'gm_3b': m['gm'], **rr,
                'diagnostics': m['diagnostics'],
                'energy_partitions_j': {'current_rise': float(first), 'voltage_fall': float(third),
                                        'storage_load': rr['storage_load_energy_j'],
                                        'recovery_storage': rr['e_rs'], 'recovery_fall': rr['e_rf']}})

    def calc_switching_energy(self, params):
        """Return finite SI energies, phase values and solver evidence."""
        off, on = self.calc_turn_off_energy(params), self.calc_turn_on_energy(params)
        values = dict(e_on=on['e_on'], e_off=off['e_off'], e_total=on['e_on'] + off['e_off'],
                      i_0_zvs=self.calc_i_0_zvs(params), i_oss_off=off['i_oss'],
                      v_mil_off=off['v_mil'], t_rv=off['t_rv'], t_fi=off['t_fi'],
                      v_ld_off=off['v_ld'], gm_off=off['gm'],
                      i_oss_on=on['i_oss_on'], v_mil_on=on['v_mil_on'],
                      t_ri=on['t_ri'], t_fv=on['t_fv'], v_ld_on=on['v_ld_on'],
                      gm_on_1b=on['gm_1b'], gm_on_3b=on['gm_3b'],
                      **{k: on[k] for k in ('i_rr_calc', 't_rs', 'q_rs', 'e_rs', 'e_rf', 'e_rr_s2')})
        _finite(**values)
        if min(values['e_on'], values['e_off']) < 0:
            raise ValueError('Negative switching energy is outside this estimate domain')
        return SwitchingEnergyResult(**values, diagnostics={
            'turn_on': on['diagnostics'], 'turn_off': off['diagnostics'],
            'turn_on_energy_partitions_j': on['energy_partitions_j'],
            'turn_off_energy_partitions_j': off['energy_partitions_j'],
            'recovery_storage_residual_a': on['storage_residual_a'],
            'body_diode_energy_j': on['body_diode_energy_j'],
            'parameter_reference': self.reference_conditions.copy(),
            'transfer_fit': self.gm_params.diagnostics.copy(),
            'charge_consistency': {
                'expected_qoss_c': self.c_oss * params.v_0,
                'relative_difference': abs(self.q_oss / (self.c_oss * params.v_0) - 1),
                'source_rounding_relative_tolerance': .01,
                'boundary': 'Derived from the same equation10 Qoss; eq14 when charge equality is exact',
            },
            'recovery': 'omitted by explicit model selection' if self.rr_params is None else 'charge-control estimate',
        })

    def calc_switching_loss_curve(self, v_0, currents, v_g_on=15, v_g_off=-5,
                                 r_g=10, l_s=4e-9, l_d=10e-9):
        """Compatibility arrays; fail rather than silently replacing unsupported points."""
        results = [self.calc_switching_energy(HalfBridgeParams(
            v_0, float(i), v_g_on, v_g_off, r_g, l_s, l_d)) for i in currents]
        return tuple(np.array([getattr(r, key) for r in results])
                     for key in ('e_on', 'e_off', 'e_total'))

    @classmethod
    def from_transistor(cls, transistor: Transistor, v_0, t_j=25, *, v_ds_fit=None,
                        v_th=None, recovery_mode=None, i_forward_test=None):
        """Extract from exact-condition curves with explicit threshold/recovery scope.

        Legacy diode scalar fields carry no test temperature/voltage; automatic
        recovery extraction is unsupported. Pass a validated RR parameter object
        or explicitly select 'omit', with provenance supplied by the caller.
        """
        if recovery_mode != 'omit' and not isinstance(recovery_mode, ReverseRecoveryParams):
            raise ValueError("Select recovery_mode='omit' or supply condition-verified ReverseRecoveryParams")
        caps = []
        for name in ('c_iss', 'c_oss', 'c_rss'):
            matches = [c for c in getattr(transistor, name) if c.t_j == t_j]
            if len(matches) != 1 or matches[0].graph_v_c is None:
                raise ValueError(f'Require one unambiguous {name} curve at exact Tj={t_j}')
            graph = np.asarray(matches[0].graph_v_c)
            caps.append(calc_charge_equivalent_capacitance(graph[0], graph[1], v_0))
        gm = fit_transconductance(transistor.switch.channel_data, v_ds=v_ds_fit, t_j=t_j, v_th=v_th)
        cgs, cds, cgd = calc_device_capacitances(*caps)
        return cls(cgs, cds, cgd, caps[1] * v_0, gm,
                   None if recovery_mode == 'omit' else recovery_mode,
                   v_ref=v_0, t_j=t_j,
                   source=f'Transistor {transistor.metadata.name}; exact-temperature C(V) and output curves')
