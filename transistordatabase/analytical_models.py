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
    """Piecewise constant-capacitance RC/Miller timing estimate in SI units.

    Qgs is charge from zero gate voltage to the plateau; Qg is measured at
    v_driver. Scalar charges cannot determine physical drain waveforms or energy.
    """

    def __init__(self, params: GateChargeModelParams) -> None:
        self.params = params

    def _intervals(self, v_driver: float, v_off: float) -> dict:
        p = self.params
        values = vars(p) | {'v_driver': v_driver, 'v_off': v_off}
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if p.q_gs <= 0 or p.q_gd < 0 or p.q_g < p.q_gs + p.q_gd:
            raise ValueError('Require Qgs > 0, Qgd >= 0 and Qg >= Qgs + Qgd')
        if not 0 < p.v_th < p.v_plateau:
            raise ValueError('Require 0 < Vth < Vplateau')
        if not v_off < p.v_th < p.v_plateau < v_driver:
            raise ValueError('Require Voff < Vth < Vplateau < Von; increase drive voltage')
        r = p.r_g + p.r_g_int
        if p.r_g < 0 or p.r_g_int < 0 or r <= 0 or not np.isfinite(r):
            raise ValueError('Gate resistances must be nonnegative with finite total R > 0')
        c_pre = p.q_gs / p.v_plateau
        c_post = (p.q_g - (p.q_gs + p.q_gd)) / (v_driver - p.v_plateau)
        # Exact RC integration between finite gate-voltage boundaries.
        on = {
            't_delay': r * c_pre * np.log((v_driver - v_off) / (v_driver - p.v_th)),
            't_rise': r * c_pre * np.log((v_driver - p.v_th) / (v_driver - p.v_plateau)),
            't_fall_v': r * p.q_gd / (v_driver - p.v_plateau),
        }
        off = {
            't_delay': r * c_post * np.log((v_driver - v_off) / (p.v_plateau - v_off)),
            't_rise_v': r * p.q_gd / (p.v_plateau - v_off),
            't_fall_i': r * c_pre * np.log((p.v_plateau - v_off) / (p.v_th - v_off)),
        }
        for times in (on, off):
            times.update({key: float(value) for key, value in times.items()})
            times['t_total'] = sum(times.values())
            if not all(np.isfinite(t) and t >= 0 for t in times.values()):
                raise ValueError('Parameters exceed the finite numerical range of this estimate')
            if times['t_total'] <= 0:
                raise ValueError('Parameters produce an unresolvable zero switching interval')
        return {'turn_on': on, 'turn_off': off}

    def calc_turn_on_times(
        self, v_driver: float = 15.0, v_off: float = 0.0
    ) -> dict[str, float]:
        """Return delay, current-rise, Miller and total times in seconds.

        Ends at the Miller plateau; post-plateau charging is excluded.
        """
        return self._intervals(v_driver, v_off)['turn_on']

    def calc_turn_off_times(
        self, v_driver: float = 15.0, v_off: float = 0.0
    ) -> dict[str, float]:
        """Return post-charge delay, Miller, current-fall and total times in s.

        Ends at Vth, without extending to the asymptotic off rail.
        """
        return self._intervals(v_driver, v_off)['turn_off']

    def calc_switching_estimate(
        self, v_driver: float = 15.0, v_off: float = 0.0
    ) -> dict:
        """Return SI timings, phase boundaries and normalized drain illustrations.

        Gate voltage follows RC intervals, gate current is (Vrail - Vg)/R.
        Drain current uses a linear-gm normalization; drain voltage traverses
        1 to 0 (or 0 to 1) linearly during Miller charge. Neither is a physical
        drain waveform. Zero-charge stages are explicit instantaneous jumps.
        """
        intervals = self._intervals(v_driver, v_off)
        p = self.params
        r = p.r_g + p.r_g_int
        result = {
            'mode': 'estimate',
            'assumptions': [
                'Qgs spans zero gate voltage to Vplateau; Qg is specified at Von.',
                'Constant Cpre = Qgs/Vplateau and Cpost = (Qg-Qgs-Qgd)/(Von-Vplateau).',
                'Ideal resistive driver and constant-voltage Miller plateau; no parasitic inductance.',
                'Normalized Id uses linear transconductance; normalized Vds is linear during Miller.',
                'Drain traces illustrate intervals only; no physical drain amplitude or energy is inferred.',
                'Turn-on stops at Miller end (Vg=Vplateau); turn-off stops at Vth.',
                'Negative Voff extends constant Cpre below zero as an estimate, not measured behavior.',
            ],
        }
        for direction, times in intervals.items():
            is_on = direction == 'turn_on'
            rail = v_driver if is_on else v_off
            boundaries = ([v_off, p.v_th, p.v_plateau, p.v_plateau] if is_on
                          else [v_driver, p.v_plateau, p.v_plateau, p.v_th])
            phases, time, vg, ids, vds = [], [], [], [], []
            start = 0.0
            for index, (key, duration) in enumerate(list(times.items())[:3]):
                a, b = boundaries[index:index + 2]
                local = np.linspace(0, duration, 33 if duration > 0 else 2)
                if a == b:
                    voltage = np.full(local.shape, a)
                elif duration == 0:
                    voltage = np.array([a, b])
                else:
                    # Use endpoints to parameterize the exact exponential, avoiding
                    # a separate capacitance path for plotted and scalar results.
                    voltage = rail + (a - rail) * np.exp(
                        np.log((b - rail) / (a - rail)) * local / duration
                    )
                voltage[0], voltage[-1] = a, b
                current = np.clip((voltage - p.v_th) / (p.v_plateau - p.v_th), 0, 1)
                miller = (is_on and index == 2) or (not is_on and index == 1)
                if miller:
                    drain = np.linspace(1, 0, len(local)) if is_on else np.linspace(0, 1, len(local))
                else:
                    level = int((is_on and index < 2) or (not is_on and index == 2))
                    drain = np.full(local.shape, level)
                phases.append({'name': key, 'start_s': start, 'end_s': start + duration,
                               'instantaneous': duration == 0})
                time.extend((start + local).tolist())
                vg.extend(voltage.tolist())
                ids.extend(current.tolist())
                vds.extend(drain.tolist())
                start += duration
            result[direction] = {
                'times_s': times, 'phases': phases,
                'time_s': time, 'v_gate_v': vg,
                'i_gate_a': ((rail - np.array(vg)) / r).tolist(),
                'id_normalized': ids, 'vds_normalized': vds,
            }
            for key in ('time_s', 'v_gate_v', 'i_gate_a', 'id_normalized', 'vds_normalized'):
                if not np.all(np.isfinite(result[direction][key])):
                    raise ValueError('Parameters exceed the finite numerical range of the traces')
        return result


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


# Public compatibility exports; corrected Biela science lives in one module.
from transistordatabase.biela import (  # noqa: E402, F401
    HalfBridgeParams,
    TransconductanceParams,
    ReverseRecoveryParams,
    SwitchingEnergyResult,
    calc_charge_equivalent_capacitance,
    calc_charge_stored,
    calc_device_capacitances,
    extract_reverse_recovery_params,
    fit_transconductance,
    ChristenBielaModel,
)
