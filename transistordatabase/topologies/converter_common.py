"""Common converter loss calculation functions shared across boost, buck, and buck-boost topologies.

All topology-specific behavior is injected through the ``duty_cycle_*`` and ``blocking_voltage``
abstract methods, which each concrete subclass must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ConverterBase(ABC):
    """Abstract base for half-bridge DC-DC converter loss calculations.

    Subclasses must implement the five abstract methods that encode topology-
    specific duty cycle formulas and blocking-voltage references.
    """

    # ------------------------------------------------------------------
    # Abstract methods – topology-specific formulas
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2):
        """Return CCM duty cycle for mesh (numpy broadcast) mode."""

    @staticmethod
    @abstractmethod
    def duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2):
        """Return DCM primary-switch duty cycle for mesh mode."""

    @staticmethod
    @abstractmethod
    def duty_cycle_dcm2_mesh(duty_cycle_dcm1, v_in, v_out, v_channel1, v_channel2):
        """Return DCM secondary-switch duty cycle for mesh mode."""

    @staticmethod
    @abstractmethod
    def inductor_voltage(v_in, v_out, v_channel1):
        """Return the voltage across the inductor during the on-phase (v_L_on)."""

    @staticmethod
    @abstractmethod
    def blocking_voltage(v_in, v_out):
        """Return the switch blocking voltage for energy scaling."""

    # ------------------------------------------------------------------
    # Mesh helpers – identical across topologies
    # ------------------------------------------------------------------

    @staticmethod
    def f_m_calc_channel(m_i, v_g_on1, transistor1, transistor2):
        """Calculate all channel data for transistor1 and transistor2 in mesh for a given current."""
        vec_i_channel = np.linspace(1, 1000, 1000)
        v_channel1_switch = np.zeros_like(vec_i_channel)
        r_channel1_switch = np.zeros_like(vec_i_channel)
        v_channel2_diode = np.zeros_like(vec_i_channel)
        r_channel2_diode = np.zeros_like(vec_i_channel)
        v_channel1 = np.zeros_like(vec_i_channel)
        v_channel2 = np.zeros_like(vec_i_channel)

        i = 0
        while i < 1000:
            if vec_i_channel[i] <= transistor1.i_abs_max:
                v_channel1_switch[i], r_channel1_switch[i] = transistor1.calc_lin_channel(
                    t_j=max([channel.t_j for channel in transistor1.switch.channel]),
                    v_g=v_g_on1,
                    i_channel=vec_i_channel[i],
                    switch_or_diode="switch")
            else:
                v_channel1_switch[i] = v_channel1_switch[i - 1]
                r_channel1_switch[i] = r_channel1_switch[i - 1]

            if vec_i_channel[i] <= transistor2.i_abs_max:
                v_channel2_diode[i], r_channel2_diode[i] = transistor2.calc_lin_channel(
                    t_j=max([channel.t_j for channel in transistor2.diode.channel]),
                    v_g=0,
                    i_channel=vec_i_channel[i],
                    switch_or_diode="diode")
            else:
                v_channel2_diode[i] = v_channel2_diode[i - 1]
                r_channel2_diode[i] = r_channel2_diode[i - 1]

            v_channel1[i] = r_channel1_switch[i] * vec_i_channel[i] + v_channel1_switch[i]
            v_channel2[i] = v_channel2_diode[i]
            i = i + 1

        m_v_channel1 = np.zeros_like(m_i)
        m_v_channel2 = np.zeros_like(m_i)
        m_r_channel1_switch = np.zeros_like(m_i)
        m_v_channel1_switch = np.zeros_like(m_i)
        m_v_channel2_diode = np.zeros_like(m_i)
        m_r_channel2_diode = np.zeros_like(m_i)

        m_v_channel1[~np.isnan(m_i)] = np.interp((m_i[~np.isnan(m_i)]), vec_i_channel, v_channel1)
        m_v_channel2[~np.isnan(m_i)] = np.interp((m_i[~np.isnan(m_i)]), vec_i_channel, v_channel2)
        m_r_channel1_switch[~np.isnan(m_i)] = np.interp((m_i[~np.isnan(m_i)]), vec_i_channel, r_channel1_switch)
        m_v_channel1_switch[~np.isnan(m_i)] = np.interp((m_i[~np.isnan(m_i)]), vec_i_channel, v_channel1_switch)
        m_v_channel2_diode[~np.isnan(m_i)] = np.interp((m_i[~np.isnan(m_i)]), vec_i_channel, v_channel2_diode)
        m_r_channel2_diode[~np.isnan(m_i)] = np.interp((m_i[~np.isnan(m_i)]), vec_i_channel, r_channel2_diode)

        return [m_v_channel1, m_v_channel2, m_r_channel1_switch, m_v_channel1_switch, m_v_channel2_diode, m_r_channel2_diode]

    # ------------------------------------------------------------------
    # Mesh topology-specific current calculations
    # ------------------------------------------------------------------

    def f_m_i_peak(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate peak current in mesh."""
        v_channel1 = np.zeros_like(zeta)
        v_channel2 = np.zeros_like(zeta)
        n = 0
        while n < 2:
            duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
            duty_cycle_dcm1 = self.duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2)
            v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

            m_i_max_ccm = (p_out / v_out) + (v_l_on * duty_cycle_ccm / (2 * zeta))
            m_i_max_ccm[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

            m_i_peak_dcm = (v_l_on * duty_cycle_dcm1) / zeta
            m_i_peak_dcm[p_out > (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

            m_i_peak = np.full_like(m_i_max_ccm, np.nan)
            m_i_peak[~np.isnan(m_i_max_ccm)] = m_i_max_ccm[~np.isnan(m_i_max_ccm)]
            m_i_peak[~np.isnan(m_i_peak_dcm)] = m_i_peak_dcm[~np.isnan(m_i_peak_dcm)]

            channel = self.f_m_calc_channel(m_i_peak, v_g_on1, transistor1, transistor2)
            v_channel1 = channel[0]
            v_channel2 = channel[1]

            n = n + 1

        return m_i_peak

    def f_m_i1_rms(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate RMS current for transistor1 in mesh."""
        i_peak = self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        channel = self.f_m_calc_channel(i_peak, v_g_on1, transistor1, transistor2)

        v_channel1 = channel[0]
        v_channel2 = channel[1]

        duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
        duty_cycle_dcm1 = self.duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2)
        v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

        i_min_ccm = (p_out / v_out) - (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_max_ccm = (p_out / v_out) + (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_peak_dcm = (v_l_on * duty_cycle_dcm1) / zeta

        m_i1_rms_ccm = np.sqrt((duty_cycle_ccm * (i_min_ccm ** 2 + i_max_ccm * i_min_ccm + i_max_ccm ** 2)) / 3)
        m_i1_rms_ccm[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i1_rms_dcm = np.sqrt((duty_cycle_dcm1 * i_peak_dcm ** 2) / 3)
        m_i1_rms_dcm[p_out > (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i1_rms = np.full_like(m_i1_rms_ccm, np.nan)
        m_i1_rms[~np.isnan(m_i1_rms_ccm)] = m_i1_rms_ccm[~np.isnan(m_i1_rms_ccm)]
        m_i1_rms[~np.isnan(m_i1_rms_dcm)] = m_i1_rms_dcm[~np.isnan(m_i1_rms_dcm)]

        return m_i1_rms

    def f_m_i1_mean(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate mean current for transistor1 in mesh."""
        channel = self.f_m_calc_channel(
            self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
        duty_cycle_dcm1 = self.duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2)
        v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

        i_min_ccm = (p_out / v_out) - (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_max_ccm = (p_out / v_out) + (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_peak_dcm = (v_l_on * duty_cycle_dcm1) / zeta

        m_i1_mean_ccm = (duty_cycle_ccm * (i_min_ccm + i_max_ccm)) / 2
        m_i1_mean_ccm[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i1_mean_dcm = (duty_cycle_dcm1 * i_peak_dcm) / 2
        m_i1_mean_dcm[p_out > (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i1_mean = np.full_like(m_i1_mean_ccm, np.nan)
        m_i1_mean[~np.isnan(m_i1_mean_ccm)] = m_i1_mean_ccm[~np.isnan(m_i1_mean_ccm)]
        m_i1_mean[~np.isnan(m_i1_mean_dcm)] = m_i1_mean_dcm[~np.isnan(m_i1_mean_dcm)]

        return m_i1_mean

    def f_m_i2_rms(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate RMS current for transistor2 in mesh."""
        channel = self.f_m_calc_channel(
            self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
        duty_cycle_dcm1 = self.duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2)
        duty_cycle_dcm2 = self.duty_cycle_dcm2_mesh(duty_cycle_dcm1, v_in, v_out, v_channel1, v_channel2)
        v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

        i_min_ccm = (p_out / v_out) - (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_max_ccm = (p_out / v_out) + (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_peak_dcm = (v_l_on * duty_cycle_dcm1) / zeta

        m_i2_rms_ccm = np.sqrt(((1 - duty_cycle_ccm) * (i_min_ccm ** 2 + i_max_ccm * i_min_ccm + i_max_ccm ** 2)) / 3)
        m_i2_rms_ccm[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i2_rms_dcm = np.sqrt((duty_cycle_dcm2 * i_peak_dcm ** 2) / 3)
        m_i2_rms_dcm[p_out > (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i2_rms = np.full_like(m_i2_rms_ccm, np.nan)
        m_i2_rms[~np.isnan(m_i2_rms_ccm)] = m_i2_rms_ccm[~np.isnan(m_i2_rms_ccm)]
        m_i2_rms[~np.isnan(m_i2_rms_dcm)] = m_i2_rms_dcm[~np.isnan(m_i2_rms_dcm)]

        return m_i2_rms

    def f_m_i2_mean(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate mean current for transistor2 in mesh."""
        channel = self.f_m_calc_channel(
            self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
        duty_cycle_dcm1 = self.duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2)
        duty_cycle_dcm2 = self.duty_cycle_dcm2_mesh(duty_cycle_dcm1, v_in, v_out, v_channel1, v_channel2)
        v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

        i_min_ccm = (p_out / v_out) - (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_max_ccm = (p_out / v_out) + (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_peak_dcm = (v_l_on * duty_cycle_dcm1) / zeta

        m_i2_mean_ccm = ((1 - duty_cycle_ccm) * (i_min_ccm + i_max_ccm)) / 2
        m_i2_mean_ccm[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i2_mean_dcm = (duty_cycle_dcm2 * i_peak_dcm) / 2
        m_i2_mean_dcm[p_out > (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i2_mean = np.full_like(m_i2_mean_ccm, np.nan)
        m_i2_mean[~np.isnan(m_i2_mean_ccm)] = m_i2_mean_ccm[~np.isnan(m_i2_mean_ccm)]
        m_i2_mean[~np.isnan(m_i2_mean_dcm)] = m_i2_mean_dcm[~np.isnan(m_i2_mean_dcm)]

        return m_i2_mean

    def f_m_i_l_rms(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate inductor RMS current in mesh."""
        channel = self.f_m_calc_channel(
            self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
        duty_cycle_dcm1 = self.duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2)
        duty_cycle_dcm2 = self.duty_cycle_dcm2_mesh(duty_cycle_dcm1, v_in, v_out, v_channel1, v_channel2)
        v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

        i_min_ccm = (p_out / v_out) - (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_max_ccm = (p_out / v_out) + (v_l_on * duty_cycle_ccm / (2 * zeta))
        i_peak_dcm = (v_l_on * duty_cycle_dcm1) / zeta

        triangle = i_min_ccm ** 2 + i_max_ccm * i_min_ccm + i_max_ccm ** 2
        m_i_l_rms_ccm = np.sqrt((duty_cycle_ccm * triangle) / 3 + -((duty_cycle_ccm - 1) * triangle) / 3)
        m_i_l_rms_ccm[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i_l_rms_dcm = np.sqrt((duty_cycle_dcm1 * i_peak_dcm ** 2) / 3 + (duty_cycle_dcm2 * i_peak_dcm ** 2) / 3)
        m_i_l_rms_dcm[p_out > (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = np.nan

        m_i_l_rms = np.full_like(m_i_l_rms_ccm, np.nan)
        m_i_l_rms[~np.isnan(m_i_l_rms_ccm)] = m_i_l_rms_ccm[~np.isnan(m_i_l_rms_ccm)]
        m_i_l_rms[~np.isnan(m_i_l_rms_dcm)] = m_i_l_rms_dcm[~np.isnan(m_i_l_rms_dcm)]

        return m_i_l_rms

    @staticmethod
    def f_m_i_l_mean(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate mean inductor current in mesh."""
        return p_out / v_out

    # ------------------------------------------------------------------
    # Mesh conduction / aggregation wrappers
    # ------------------------------------------------------------------

    def f_m_conduction_losses1(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate conduction losses for transistor1."""
        m_i1_rms = self.f_m_i1_rms(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        m_i1_mean = self.f_m_i1_mean(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        m_i_peak = self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        channel1 = self.f_m_calc_channel(m_i_peak, v_g_on1, transistor1, transistor2)
        r_channel1_switch = channel1[2]
        v_channel1_switch = channel1[3]
        return (m_i1_rms ** 2) * r_channel1_switch + m_i1_mean * v_channel1_switch

    def f_m_conduction_losses2(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate conduction losses for transistor2."""
        m_i2_rms = self.f_m_i2_rms(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        m_i_peak = self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        channel2 = self.f_m_calc_channel(m_i_peak, v_g_on1, transistor1, transistor2)
        v_channel2_diode = channel2[4]
        return m_i2_rms * v_channel2_diode

    # ------------------------------------------------------------------
    # Mesh switching loss calculations
    # ------------------------------------------------------------------

    def f_m_p_on1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, frequency, transistor1, transistor2):
        """Calculate turn-on switching losses for transistor1."""
        channel = self.f_m_calc_channel(
            self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
        v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

        i_min_ccm = (p_out / v_out) - (v_l_on * duty_cycle_ccm / (2 * zeta))

        m_i_on_ccm1 = i_min_ccm
        m_i_on_ccm1[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = 0
        m_i_on1 = m_i_on_ccm1

        v_supply_chosen1 = max([i for i in [e_on.v_supply for e_on in transistor1.switch.e_on] if i is not None])
        try:
            transistor1.wp.e_on = transistor1.calc_object_i_e(
                e_on_off_rr="e_on",
                t_j=max([i for i in [e_on.t_j for e_on in transistor1.switch.e_on] if i is not None]),
                v_supply=v_supply_chosen1, r_g=r_g_on1, normalize_t_to_v=10)
        except Exception:
            transistor1.wp.e_on = transistor1.get_object_i_e(
                e_on_off_rr="e_on",
                t_j=max([i for i in [e_on.t_j for e_on in transistor1.switch.e_on] if i is not None]),
                v_supply=v_supply_chosen1,
                r_g=max([i for i in [e_on.r_g for e_on in transistor1.switch.e_on] if i is not None]),
                v_g=max([i for i in [e_on.v_g for e_on in transistor1.switch.e_on] if i is not None]))

        m_e_on1 = np.full_like(m_i_on1, np.nan)
        m_e_on1[~np.isnan(m_i_on1)] = np.interp(
            (m_i_on1[~np.isnan(m_i_on1)]), transistor1.wp.e_on.graph_i_e[0], transistor1.wp.e_on.graph_i_e[1])
        v_block = self.blocking_voltage(v_in, v_out)
        return m_e_on1 * frequency * 1000 * v_block / v_supply_chosen1

    def f_m_p_off1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate turn-off switching losses for transistor1."""
        m_i_off1 = self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)

        v_supply_chosen1 = max([i for i in [e_off.v_supply for e_off in transistor1.switch.e_off] if i is not None])
        try:
            transistor1.wp.e_off = transistor1.calc_object_i_e(
                e_on_off_rr="e_off",
                t_j=max([i for i in [e_off.t_j for e_off in transistor1.switch.e_off] if i is not None]),
                v_supply=v_supply_chosen1, r_g=r_g_off1, normalize_t_to_v=10)
        except Exception:
            transistor1.wp.e_off = transistor1.get_object_i_e(
                e_on_off_rr="e_off",
                t_j=max([i for i in [e_off.t_j for e_off in transistor1.switch.e_off] if i is not None]),
                v_supply=v_supply_chosen1,
                r_g=max([i for i in [e_off.r_g for e_off in transistor1.switch.e_off] if i is not None]),
                v_g=min([i for i in [e_off.v_g for e_off in transistor1.switch.e_off] if i is not None]))

        m_e_off1 = np.full_like(m_i_off1, np.nan)
        m_e_off1[~np.isnan(m_i_off1)] = np.interp(
            (m_i_off1[~np.isnan(m_i_off1)]), transistor1.wp.e_off.graph_i_e[0], transistor1.wp.e_off.graph_i_e[1])
        v_block = self.blocking_voltage(v_in, v_out)
        return m_e_off1 * frequency * 1000 * v_block / v_supply_chosen1

    def f_m_p_rr2(self, zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2):
        """Calculate reverse-recovery losses for transistor2."""
        channel = self.f_m_calc_channel(
            self.f_m_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        duty_cycle_ccm = self.duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2)
        v_l_on = self.inductor_voltage(v_in, v_out, v_channel1)

        i_min_ccm = (p_out / v_out) - (v_l_on * duty_cycle_ccm / (2 * zeta))
        m_i_off_ccm2 = i_min_ccm
        m_i_off_ccm2[p_out < (v_out * duty_cycle_ccm * (v_l_on / (2 * zeta)))] = 0
        m_i_off2 = m_i_off_ccm2

        v_supply_chosen2 = max([i for i in [e_rr.v_supply for e_rr in transistor2.diode.e_rr] if i is not None])
        try:
            transistor2.wp.e_rr = transistor2.calc_object_i_e(
                e_on_off_rr="e_rr",
                t_j=max([i for i in [e_rr.t_j for e_rr in transistor2.diode.e_rr] if i is not None]),
                v_supply=v_supply_chosen2, r_g=0, normalize_t_to_v=10)
        except Exception:
            transistor2.wp.e_rr = transistor2.get_object_i_e(
                e_on_off_rr="e_rr",
                t_j=max([i for i in [e_rr.t_j for e_rr in transistor2.diode.e_rr] if i is not None]),
                v_supply=v_supply_chosen2,
                r_g=max([i for i in [e_rr.r_g for e_rr in transistor2.diode.e_rr] if i is not None]),
                v_g=min([i for i in [e_rr.v_g for e_rr in transistor2.diode.e_rr] if i is not None]))

        m_e_rr2 = np.full_like(m_i_off2, np.nan)
        m_e_rr2[~np.isnan(m_i_off2)] = np.interp(
            (m_i_off2[~np.isnan(m_i_off2)]), transistor2.wp.e_rr.graph_i_e[0], transistor2.wp.e_rr.graph_i_e[1])
        v_block = self.blocking_voltage(v_in, v_out)
        return m_e_rr2 * frequency * 1000 * v_block / v_supply_chosen2

    # ------------------------------------------------------------------
    # Mesh aggregation wrappers
    # ------------------------------------------------------------------

    def f_m_conduction_losses(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate total conduction losses for transistor1 + transistor2."""
        return (self.f_m_conduction_losses1(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
                + self.f_m_conduction_losses2(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2))

    def f_m_p_on_off1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate total switching losses for transistor1."""
        return (self.f_m_p_on1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, frequency, transistor1, transistor2)
                + self.f_m_p_off1(zeta, v_in, v_out, p_out, v_g_on1, r_g_off1, frequency, transistor1, transistor2))

    def f_m_p_on_off_rr_1_2(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate total switching losses for transistor1 + transistor2."""
        return (self.f_m_p_on_off1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2)
                + self.f_m_p_rr2(zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2))

    def f_m_p1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate total power losses for transistor1."""
        return (self.f_m_conduction_losses1(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
                + self.f_m_p_on_off1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2))

    def f_m_p2(self, zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2):
        """Calculate total power losses for transistor2."""
        return (self.f_m_conduction_losses2(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
                + self.f_m_p_rr2(zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2))

    def f_m_t_switch1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, t_heatsink, r_th_heatsink, frequency, transistor1, transistor2):
        """Calculate switch temperature for transistor1."""
        m_p1 = self.f_m_p1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2)
        r_th_switch_jc = transistor1.switch.thermal_foster.r_th_total
        r_th_switch_cs = transistor1.r_th_switch_cs
        r_th_cs = transistor1.r_th_cs
        return t_heatsink + m_p1 * (r_th_switch_jc + r_th_switch_cs + r_th_cs + r_th_heatsink)

    def f_m_t_diode2(self, zeta, v_in, v_out, p_out, v_g_on1, t_heatsink, r_th_heatsink, frequency, transistor1, transistor2):
        """Calculate diode temperature for transistor2."""
        m_p2 = self.f_m_p2(zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2)
        r_th_diode_jc = transistor2.diode.thermal_foster.r_th_total
        r_th_diode_cs = transistor2.r_th_diode_cs
        r_th_cs = transistor2.r_th_cs
        return t_heatsink + m_p2 * (r_th_diode_jc + r_th_diode_cs + r_th_cs + r_th_heatsink)

    # ==================================================================
    # Vector (element-wise loop) functions
    # ==================================================================

    @staticmethod
    def f_vec_calc_channel(vec_i, v_g_on1, transistor1, transistor2):
        """Calculate all channel data for transistor1 and transistor2 in list for a given current in list."""
        vec_i_channel = np.linspace(1, 1000, 1000)
        v_channel1_switch = np.zeros_like(vec_i_channel)
        r_channel1_switch = np.zeros_like(vec_i_channel)
        v_channel2_diode = np.zeros_like(vec_i_channel)
        r_channel2_diode = np.zeros_like(vec_i_channel)
        v_channel1 = np.zeros_like(vec_i_channel)
        v_channel2 = np.zeros_like(vec_i_channel)

        i = 0
        while i < 1000:
            if vec_i_channel[i] <= transistor1.i_abs_max:
                v_channel1_switch[i], r_channel1_switch[i] = transistor1.calc_lin_channel(
                    t_j=max([channel.t_j for channel in transistor1.switch.channel]),
                    v_g=v_g_on1,
                    i_channel=vec_i_channel[i],
                    switch_or_diode="switch")
            else:
                v_channel1_switch[i] = v_channel1_switch[i - 1]
                r_channel1_switch[i] = r_channel1_switch[i - 1]

            if vec_i_channel[i] <= transistor2.i_abs_max:
                v_channel2_diode[i], r_channel2_diode[i] = transistor2.calc_lin_channel(
                    t_j=max([channel.t_j for channel in transistor2.diode.channel]),
                    v_g=0,
                    i_channel=vec_i_channel[i],
                    switch_or_diode="diode")
            else:
                v_channel2_diode[i] = v_channel2_diode[i - 1]
                r_channel2_diode[i] = r_channel2_diode[i - 1]

            v_channel1[i] = r_channel1_switch[i] * vec_i_channel[i] + v_channel1_switch[i]
            v_channel2[i] = v_channel2_diode[i]
            i = i + 1

        vec_v_channel1 = np.zeros_like(vec_i)
        vec_v_channel2 = np.zeros_like(vec_i)
        vec_r_channel1_switch = np.zeros_like(vec_i)
        vec_v_channel1_switch = np.zeros_like(vec_i)
        vec_v_channel2_diode = np.zeros_like(vec_i)

        i = 0
        while i < np.size(vec_i):
            vec_v_channel1[i] = np.interp((vec_i[i]), vec_i_channel, v_channel1)
            vec_v_channel2[i] = np.interp(vec_i[i], vec_i_channel, v_channel2)
            vec_r_channel1_switch[i] = np.interp(vec_i[i], vec_i_channel, r_channel1_switch)
            vec_v_channel1_switch[i] = np.interp(vec_i[i], vec_i_channel, v_channel1_switch)
            vec_v_channel2_diode[i] = np.interp(vec_i[i], vec_i_channel, v_channel2_diode)
            i = i + 1

        return [vec_v_channel1, vec_v_channel2, vec_r_channel1_switch, vec_v_channel1_switch, vec_v_channel2_diode]

    # ------------------------------------------------------------------
    # Vector topology-specific current calculations
    # ------------------------------------------------------------------

    def f_vec_i_peak(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate peak current in list."""
        duty_cycle_ccm = np.zeros_like(zeta)
        duty_cycle_dcm1 = np.zeros_like(zeta)
        vec_i_peak = np.zeros_like(zeta)
        v_channel1 = np.zeros_like(zeta)
        v_channel2 = np.zeros_like(zeta)

        n = 0
        while n < 2:
            i = 0
            while i < np.size(zeta):
                duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
                duty_cycle_dcm1[i] = self.duty_cycle_dcm1_mesh(zeta[i], v_in[i], v_out[i], p_out[i], v_channel1[i], v_channel2[i])
                v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

                if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                    vec_i_peak[i] = (p_out[i] / v_out[i]) + (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
                else:
                    vec_i_peak[i] = (v_l_on * duty_cycle_dcm1[i]) / zeta[i]
                i = i + 1

            channel = self.f_vec_calc_channel(vec_i_peak, v_g_on1, transistor1, transistor2)
            v_channel1 = channel[0]
            v_channel2 = channel[1]
            n = n + 1

        return vec_i_peak

    def f_vec_i1_rms(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate RMS current for transistor1 in list."""
        duty_cycle_ccm = np.zeros_like(zeta)
        duty_cycle_dcm1 = np.zeros_like(zeta)
        i_min_ccm = np.zeros_like(zeta)
        i_max_ccm = np.zeros_like(zeta)
        i_peak_dcm = np.zeros_like(zeta)
        vec_i1_rms = np.zeros_like(zeta)

        channel = self.f_vec_calc_channel(
            self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        i = 0
        while i < np.size(zeta):
            duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm1[i] = self.duty_cycle_dcm1_mesh(zeta[i], v_in[i], v_out[i], p_out[i], v_channel1[i], v_channel2[i])
            v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

            i_min_ccm[i] = (p_out[i] / v_out[i]) - (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_max_ccm[i] = (p_out[i] / v_out[i]) + (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_peak_dcm[i] = (v_l_on * duty_cycle_dcm1[i]) / zeta[i]

            if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                vec_i1_rms[i] = np.sqrt(
                    (duty_cycle_ccm[i] * (i_min_ccm[i] ** 2 + i_max_ccm[i] * i_min_ccm[i] + i_max_ccm[i] ** 2)) / 3)
            else:
                vec_i1_rms[i] = np.sqrt((duty_cycle_dcm1[i] * i_peak_dcm[i] ** 2) / 3)
            i = i + 1

        return vec_i1_rms

    def f_vec_i1_mean(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate mean current for transistor1 in list."""
        duty_cycle_ccm = np.zeros_like(zeta)
        duty_cycle_dcm1 = np.zeros_like(zeta)
        i_min_ccm = np.zeros_like(zeta)
        i_max_ccm = np.zeros_like(zeta)
        i_peak_dcm = np.zeros_like(zeta)
        vec_i1_mean = np.zeros_like(zeta)

        channel = self.f_vec_calc_channel(
            self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        i = 0
        while i < np.size(zeta):
            duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm1[i] = self.duty_cycle_dcm1_mesh(zeta[i], v_in[i], v_out[i], p_out[i], v_channel1[i], v_channel2[i])
            v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

            i_min_ccm[i] = (p_out[i] / v_out[i]) - (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_max_ccm[i] = (p_out[i] / v_out[i]) + (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_peak_dcm[i] = (v_l_on * duty_cycle_dcm1[i]) / zeta[i]

            if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                vec_i1_mean[i] = (duty_cycle_ccm[i] * (i_min_ccm[i] + i_max_ccm[i])) / 2
            else:
                vec_i1_mean[i] = (duty_cycle_dcm1[i] * i_peak_dcm[i]) / 2
            i = i + 1

        return vec_i1_mean

    def f_vec_i2_rms(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate RMS current for transistor2 in list."""
        duty_cycle_ccm = np.zeros_like(zeta)
        duty_cycle_dcm1 = np.zeros_like(zeta)
        duty_cycle_dcm2 = np.zeros_like(zeta)
        i_min_ccm = np.zeros_like(zeta)
        i_max_ccm = np.zeros_like(zeta)
        i_peak_dcm = np.zeros_like(zeta)
        vec_i2_rms = np.zeros_like(zeta)

        channel = self.f_vec_calc_channel(
            self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        i = 0
        while i < np.size(zeta):
            duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm1[i] = self.duty_cycle_dcm1_mesh(zeta[i], v_in[i], v_out[i], p_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm2[i] = self.duty_cycle_dcm2_mesh(duty_cycle_dcm1[i], v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

            i_min_ccm[i] = (p_out[i] / v_out[i]) - (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_max_ccm[i] = (p_out[i] / v_out[i]) + (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_peak_dcm[i] = (v_l_on * duty_cycle_dcm1[i]) / zeta[i]

            if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                vec_i2_rms[i] = np.sqrt(-((duty_cycle_ccm[i] - 1) * (i_min_ccm[i] ** 2 + i_max_ccm[i] * i_min_ccm[i] + i_max_ccm[i] ** 2)) / 3)
            else:
                vec_i2_rms[i] = np.sqrt((duty_cycle_dcm2[i] * i_peak_dcm[i] ** 2) / 3)
            i = i + 1

        return vec_i2_rms

    def f_vec_i2_mean(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate mean current for transistor2 in list."""
        duty_cycle_ccm = np.zeros_like(zeta)
        duty_cycle_dcm1 = np.zeros_like(zeta)
        duty_cycle_dcm2 = np.zeros_like(zeta)
        i_min_ccm = np.zeros_like(zeta)
        i_max_ccm = np.zeros_like(zeta)
        i_peak_dcm = np.zeros_like(zeta)
        vec_i2_mean = np.zeros_like(zeta)

        channel = self.f_vec_calc_channel(
            self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        i = 0
        while i < np.size(zeta):
            duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm1[i] = self.duty_cycle_dcm1_mesh(zeta[i], v_in[i], v_out[i], p_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm2[i] = self.duty_cycle_dcm2_mesh(duty_cycle_dcm1[i], v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

            i_min_ccm[i] = (p_out[i] / v_out[i]) - (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_max_ccm[i] = (p_out[i] / v_out[i]) + (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_peak_dcm[i] = (v_l_on * duty_cycle_dcm1[i]) / zeta[i]

            if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                vec_i2_mean[i] = ((1 - duty_cycle_ccm[i]) * (i_min_ccm[i] + i_max_ccm[i])) / 2
            else:
                vec_i2_mean[i] = (duty_cycle_dcm2[i] * i_peak_dcm[i]) / 2
            i = i + 1

        return vec_i2_mean

    def f_vec_i_l_rms(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate inductor RMS current in list."""
        duty_cycle_ccm = np.zeros_like(zeta)
        duty_cycle_dcm1 = np.zeros_like(zeta)
        duty_cycle_dcm2 = np.zeros_like(zeta)
        i_min_ccm = np.zeros_like(zeta)
        i_max_ccm = np.zeros_like(zeta)
        i_peak_dcm = np.zeros_like(zeta)
        vec_i_l_rms = np.zeros_like(zeta)

        channel = self.f_vec_calc_channel(
            self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        i = 0
        while i < np.size(zeta):
            duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm1[i] = self.duty_cycle_dcm1_mesh(zeta[i], v_in[i], v_out[i], p_out[i], v_channel1[i], v_channel2[i])
            duty_cycle_dcm2[i] = self.duty_cycle_dcm2_mesh(duty_cycle_dcm1[i], v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

            i_min_ccm[i] = (p_out[i] / v_out[i]) - (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_max_ccm[i] = (p_out[i] / v_out[i]) + (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))
            i_peak_dcm[i] = (v_l_on * duty_cycle_dcm1[i]) / zeta[i]

            if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                vec_i_l_rms[i] = np.sqrt(
                    (duty_cycle_ccm[i] * (i_min_ccm[i] ** 2 + i_max_ccm[i] * i_min_ccm[i] + i_max_ccm[i] ** 2)) / 3
                    + -((duty_cycle_ccm[i] - 1) * (i_min_ccm[i] ** 2 + i_max_ccm[i] * i_min_ccm[i] + i_max_ccm[i] ** 2)) / 3)
            else:
                vec_i_l_rms[i] = np.sqrt((duty_cycle_dcm1[i] * i_peak_dcm[i] ** 2) / 3 + (duty_cycle_dcm2[i] * i_peak_dcm[i] ** 2) / 3)
            i = i + 1

        return vec_i_l_rms

    @staticmethod
    def f_vec_i_l_mean(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate mean inductor current in list."""
        vec_i_l_mean = np.zeros_like(zeta)
        i = 0
        while i < np.size(zeta):
            vec_i_l_mean[i] = p_out[i] / v_out[i]
            i = i + 1
        return vec_i_l_mean

    # ------------------------------------------------------------------
    # Vector conduction / aggregation wrappers
    # ------------------------------------------------------------------

    def f_vec_conduction_losses1(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate conduction losses for transistor1."""
        vec_i1_rms = self.f_vec_i1_rms(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        vec_i1_mean = self.f_vec_i1_mean(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        vec_i_peak = self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        channel1 = self.f_vec_calc_channel(vec_i_peak, v_g_on1, transistor1, transistor2)
        vec_conduction_losses1 = np.zeros_like(zeta)
        r_channel1_switch = channel1[2]
        v_channel1_switch = channel1[3]

        i = 0
        while i < np.size(zeta):
            vec_conduction_losses1[i] = (vec_i1_rms[i] ** 2) * r_channel1_switch[i] + vec_i1_mean[i] * v_channel1_switch[i]
            i = i + 1

        return vec_conduction_losses1

    def f_vec_conduction_losses2(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate conduction losses for transistor2."""
        vec_i2_rms = self.f_vec_i2_rms(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        vec_i_peak = self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
        vec_conduction_losses2 = np.zeros_like(zeta)
        channel2 = self.f_vec_calc_channel(vec_i_peak, v_g_on1, transistor1, transistor2)
        v_channel2_diode = channel2[4]

        i = 0
        while i < np.size(zeta):
            vec_conduction_losses2[i] = vec_i2_rms[i] * v_channel2_diode[i]
            i = i + 1

        return vec_conduction_losses2

    # ------------------------------------------------------------------
    # Vector switching loss calculations
    # ------------------------------------------------------------------

    def f_vec_p_on1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, frequency, transistor1, transistor2):
        """Calculate turn-on switching losses for transistor1."""
        duty_cycle_ccm = np.zeros_like(zeta)
        vec_i_min_ccm = np.zeros_like(zeta)
        vec_i_on1 = np.zeros_like(zeta)
        vec_e_on1 = np.zeros_like(zeta)
        vec_p_on1 = np.zeros_like(zeta)

        channel = self.f_vec_calc_channel(
            self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        v_supply_chosen1 = max([i for i in [e_on.v_supply for e_on in transistor1.switch.e_on] if i is not None])
        try:
            transistor1.wp.e_on = transistor1.calc_object_i_e(
                e_on_off_rr="e_on",
                t_j=max([i for i in [e_on.t_j for e_on in transistor1.switch.e_on] if i is not None]),
                v_supply=v_supply_chosen1, r_g=r_g_on1, normalize_t_to_v=10)
        except Exception:
            transistor1.wp.e_on = transistor1.get_object_i_e(
                e_on_off_rr="e_on",
                t_j=max([i for i in [e_on.t_j for e_on in transistor1.switch.e_on] if i is not None]),
                v_supply=v_supply_chosen1,
                r_g=max([i for i in [e_on.r_g for e_on in transistor1.switch.e_on] if i is not None]),
                v_g=max([i for i in [e_on.v_g for e_on in transistor1.switch.e_on] if i is not None]))

        i = 0
        while i < np.size(zeta):
            duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

            if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                vec_i_min_ccm[i] = (p_out[i] / v_out[i]) - (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))

            vec_i_on1[i] = vec_i_min_ccm[i]

            vec_e_on1[i] = np.interp((vec_i_on1[i]), transistor1.wp.e_on.graph_i_e[0], transistor1.wp.e_on.graph_i_e[1])
            v_block = self.blocking_voltage(v_in[i], v_out[i])
            vec_p_on1[i] = vec_e_on1[i] * frequency[i] * 1000 * v_block / v_supply_chosen1

            i = i + 1

        return vec_p_on1

    def f_vec_p_off1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate turn-off switching losses for transistor1."""
        vec_e_off1 = np.zeros_like(zeta)
        vec_p_off1 = np.zeros_like(zeta)

        vec_i_off1 = self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)

        v_supply_chosen1 = max([i for i in [e_off.v_supply for e_off in transistor1.switch.e_off] if i is not None])
        try:
            transistor1.wp.e_off = transistor1.calc_object_i_e(
                e_on_off_rr="e_off",
                t_j=max([i for i in [e_off.t_j for e_off in transistor1.switch.e_off] if i is not None]),
                v_supply=v_supply_chosen1, r_g=r_g_off1, normalize_t_to_v=10)
        except Exception:
            transistor1.wp.e_off = transistor1.get_object_i_e(
                e_on_off_rr="e_off",
                t_j=max([i for i in [e_off.t_j for e_off in transistor1.switch.e_off] if i is not None]),
                v_supply=v_supply_chosen1,
                r_g=max([i for i in [e_off.r_g for e_off in transistor1.switch.e_off] if i is not None]),
                v_g=max([i for i in [e_off.v_g for e_off in transistor1.switch.e_off] if i is not None]))

        i = 0
        while i < np.size(zeta):
            vec_e_off1[i] = np.interp((vec_i_off1[i]), transistor1.wp.e_off.graph_i_e[0], transistor1.wp.e_off.graph_i_e[1])
            v_block = self.blocking_voltage(v_in[i], v_out[i])
            vec_p_off1[i] = vec_e_off1[i] * frequency[i] * 1000 * v_block / v_supply_chosen1
            i = i + 1

        return vec_p_off1

    def f_vec_p_rr2(self, zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2):
        """Calculate reverse-recovery losses for transistor2."""
        duty_cycle_ccm = np.zeros_like(zeta)
        vec_i_min_ccm = np.zeros_like(zeta)
        vec_i_rr2 = np.zeros_like(zeta)
        vec_e_rr2 = np.zeros_like(zeta)
        vec_p_rr2 = np.zeros_like(zeta)

        channel = self.f_vec_calc_channel(
            self.f_vec_i_peak(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2),
            v_g_on1, transistor1, transistor2)
        v_channel1 = channel[0]
        v_channel2 = channel[1]

        v_supply_chosen2 = max([i for i in [e_rr.v_supply for e_rr in transistor2.diode.e_rr] if i is not None])
        try:
            transistor2.wp.e_rr = transistor2.calc_object_i_e(
                e_on_off_rr="e_rr",
                t_j=max([i for i in [e_rr.t_j for e_rr in transistor2.diode.e_rr] if i is not None]),
                v_supply=v_supply_chosen2, r_g=0, normalize_t_to_v=10)
        except Exception:
            transistor2.wp.e_rr = transistor2.get_object_i_e(
                e_on_off_rr="e_rr",
                t_j=max([i for i in [e_rr.t_j for e_rr in transistor2.diode.e_rr] if i is not None]),
                v_supply=v_supply_chosen2,
                r_g=max([i for i in [e_rr.r_g for e_rr in transistor2.diode.e_rr] if i is not None]),
                v_g=min([i for i in [e_rr.v_g for e_rr in transistor2.diode.e_rr] if i is not None]))

        i = 0
        while i < np.size(zeta):
            duty_cycle_ccm[i] = self.duty_cycle_ccm_mesh(v_in[i], v_out[i], v_channel1[i], v_channel2[i])
            v_l_on = self.inductor_voltage(v_in[i], v_out[i], v_channel1[i])

            if p_out[i] > (v_out[i] * duty_cycle_ccm[i] * (v_l_on / (2 * zeta[i]))):
                vec_i_min_ccm[i] = (p_out[i] / v_out[i]) - (v_l_on * duty_cycle_ccm[i] / (2 * zeta[i]))

            vec_i_rr2[i] = vec_i_min_ccm[i]

            vec_e_rr2[i] = np.interp((vec_i_rr2[i]), transistor2.wp.e_rr.graph_i_e[0], transistor2.wp.e_rr.graph_i_e[1])
            v_block = self.blocking_voltage(v_in[i], v_out[i])
            vec_p_rr2[i] = vec_e_rr2[i] * frequency[i] * 1000 * v_block / v_supply_chosen2

            i = i + 1

        return vec_p_rr2

    # ------------------------------------------------------------------
    # Vector aggregation wrappers
    # ------------------------------------------------------------------

    def f_vec_conduction_losses(self, zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2):
        """Calculate total conduction losses for transistor1 + transistor2."""
        return (self.f_vec_conduction_losses1(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
                + self.f_vec_conduction_losses2(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2))

    def f_vec_p_on_off1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate total switching losses for transistor1."""
        return (self.f_vec_p_on1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, frequency, transistor1, transistor2)
                + self.f_vec_p_off1(zeta, v_in, v_out, p_out, v_g_on1, r_g_off1, frequency, transistor1, transistor2))

    def f_vec_p_on_off_rr_1_2(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate total switching losses for transistor1 + transistor2."""
        return (self.f_vec_p_on_off1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2)
                + self.f_vec_p_rr2(zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2))

    def f_vec_p1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2):
        """Calculate total power losses for transistor1."""
        return (self.f_vec_conduction_losses1(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
                + self.f_vec_p_on_off1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2))

    def f_vec_p2(self, zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2):
        """Calculate total power losses for transistor2."""
        return (self.f_vec_conduction_losses2(zeta, v_in, v_out, p_out, v_g_on1, transistor1, transistor2)
                + self.f_vec_p_rr2(zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2))

    def f_vec_t_switch1(self, zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, t_heatsink, r_th_heatsink, frequency,
                        transistor1, transistor2):
        """Calculate switch temperature for transistor1."""
        vec_p1 = self.f_vec_p1(zeta, v_in, v_out, p_out, v_g_on1, r_g_on1, r_g_off1, frequency, transistor1, transistor2)
        r_th_switch_jc = transistor1.switch.thermal_foster.r_th_total
        r_th_switch_cs = transistor1.r_th_switch_cs
        r_th_cs = transistor1.r_th_cs
        return t_heatsink + vec_p1 * (r_th_switch_jc + r_th_switch_cs + r_th_cs + r_th_heatsink)

    def f_vec_t_diode2(self, zeta, v_in, v_out, p_out, v_g_on1, t_heatsink, r_th_heatsink, frequency,
                       transistor1, transistor2):
        """Calculate diode temperature for transistor2."""
        vec_p2 = self.f_vec_p2(zeta, v_in, v_out, p_out, v_g_on1, frequency, transistor1, transistor2)
        r_th_diode_jc = transistor2.diode.thermal_foster.r_th_total
        r_th_diode_cs = transistor2.r_th_diode_cs
        r_th_cs = transistor2.r_th_cs
        return t_heatsink + vec_p2 * (r_th_diode_jc + r_th_diode_cs + r_th_cs + r_th_heatsink)


# ======================================================================
# Concrete topology implementations
# ======================================================================


class BoostConverter(ConverterBase):
    """Boost converter: V_out > V_in, blocking voltage = V_out."""

    @staticmethod
    def duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2):
        """Return boost CCM duty cycle: D = (V_out - V_in + V_ch2) / (V_out - V_ch1 + V_ch2)."""
        return (-v_in + v_out + v_channel2) / (v_out - v_channel1 + v_channel2)

    @staticmethod
    def duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2):
        """Return boost DCM primary duty cycle."""
        return np.sqrt((2 * zeta * p_out) / (
            v_out * (v_in - v_channel1) * (1 + ((v_in - v_channel1) / (-v_in + v_out + v_channel2)))))

    @staticmethod
    def duty_cycle_dcm2_mesh(duty_cycle_dcm1, v_in, v_out, v_channel1, v_channel2):
        """Return boost DCM secondary duty cycle."""
        return duty_cycle_dcm1 * ((v_in - v_channel1) / (-v_in + v_out + v_channel2))

    @staticmethod
    def inductor_voltage(v_in, v_out, v_channel1):
        """Return boost inductor on-phase voltage: V_in - V_ch1."""
        return v_in - v_channel1

    @staticmethod
    def blocking_voltage(v_in, v_out):
        """Return boost blocking voltage: V_out."""
        return v_out


class BuckConverter(ConverterBase):
    """Buck converter: V_out < V_in, blocking voltage = V_in."""

    @staticmethod
    def duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2):
        """Return buck CCM duty cycle: D = (V_out + V_ch2) / (V_in - V_ch1 + V_ch2)."""
        return (v_channel2 + v_out) / (v_in - v_channel1 + v_channel2)

    @staticmethod
    def duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2):
        """Return buck DCM primary duty cycle."""
        return np.sqrt((2 * zeta * p_out) / (
            v_out * (v_in - v_out - v_channel1) * (1 + ((v_in - v_out - v_channel1) / (v_out + v_channel2)))))

    @staticmethod
    def duty_cycle_dcm2_mesh(duty_cycle_dcm1, v_in, v_out, v_channel1, v_channel2):
        """Return buck DCM secondary duty cycle."""
        return duty_cycle_dcm1 * ((v_in - v_out - v_channel1) / (v_out + v_channel2))

    @staticmethod
    def inductor_voltage(v_in, v_out, v_channel1):
        """Return buck inductor on-phase voltage: V_in - V_out - V_ch1."""
        return v_in - v_out - v_channel1

    @staticmethod
    def blocking_voltage(v_in, v_out):
        """Return buck blocking voltage: V_in."""
        return v_in


class BuckBoostConverter(ConverterBase):
    """Buck-boost converter: blocking voltage = V_in + V_out."""

    @staticmethod
    def duty_cycle_ccm_mesh(v_in, v_out, v_channel1, v_channel2):
        """Return buck-boost CCM duty cycle: D = (V_out + V_ch2) / (V_in + V_out - V_ch1 + V_ch2)."""
        return (v_out + v_channel2) / (v_in + v_out - v_channel1 + v_channel2)

    @staticmethod
    def duty_cycle_dcm1_mesh(zeta, v_in, v_out, p_out, v_channel1, v_channel2):
        """Return buck-boost DCM primary duty cycle."""
        return np.sqrt((2 * zeta * p_out) / (
            v_out * (v_in - v_channel1) * (1 + ((v_in - v_channel1) / (v_out + v_channel2)))))

    @staticmethod
    def duty_cycle_dcm2_mesh(duty_cycle_dcm1, v_in, v_out, v_channel1, v_channel2):
        """Return buck-boost DCM secondary duty cycle."""
        return duty_cycle_dcm1 * ((v_in - v_channel1) / (v_out + v_channel2))

    @staticmethod
    def inductor_voltage(v_in, v_out, v_channel1):
        """Return buck-boost inductor on-phase voltage: V_in - V_ch1."""
        return v_in - v_channel1

    @staticmethod
    def blocking_voltage(v_in, v_out):
        """Return buck-boost blocking voltage: V_in + V_out."""
        return v_in + v_out
