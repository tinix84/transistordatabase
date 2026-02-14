"""GUI buck-boost converter functions.

Thin wrapper that delegates all calculations to the shared
:class:`~transistordatabase.topologies.converter_common.BuckBoostConverter` base class.
Module-level functions are provided for backward compatibility with the GUI.
"""
from transistordatabase.topologies.converter_common import BuckBoostConverter

_conv = BuckBoostConverter()

# -- Mesh functions --------------------------------------------------------
f_m_calc_channel = _conv.f_m_calc_channel
f_m_i_peak = _conv.f_m_i_peak
f_m_i1_rms = _conv.f_m_i1_rms
f_m_i1_mean = _conv.f_m_i1_mean
f_m_i2_rms = _conv.f_m_i2_rms
f_m_i2_mean = _conv.f_m_i2_mean
f_m_i_l_rms = _conv.f_m_i_l_rms
f_m_i_l_mean = _conv.f_m_i_l_mean
f_m_conduction_losses1 = _conv.f_m_conduction_losses1
f_m_conduction_losses2 = _conv.f_m_conduction_losses2
f_m_p_on1 = _conv.f_m_p_on1
f_m_p_off1 = _conv.f_m_p_off1
f_m_p_rr2 = _conv.f_m_p_rr2
f_m_conduction_losses = _conv.f_m_conduction_losses
f_m_p_on_off1 = _conv.f_m_p_on_off1
f_m_p_on_off_rr_1_2 = _conv.f_m_p_on_off_rr_1_2
f_m_p1 = _conv.f_m_p1
f_m_p2 = _conv.f_m_p2
f_m_t_switch1 = _conv.f_m_t_switch1
f_m_t_diode2 = _conv.f_m_t_diode2

# -- Vector functions ------------------------------------------------------
f_vec_calc_channel = _conv.f_vec_calc_channel
f_vec_i_peak = _conv.f_vec_i_peak
f_vec_i1_rms = _conv.f_vec_i1_rms
f_vec_i1_mean = _conv.f_vec_i1_mean
f_vec_i2_rms = _conv.f_vec_i2_rms
f_vec_i2_mean = _conv.f_vec_i2_mean
f_vec_i_l_rms = _conv.f_vec_i_l_rms
f_vec_i_l_mean = _conv.f_vec_i_l_mean
f_vec_conduction_losses1 = _conv.f_vec_conduction_losses1
f_vec_conduction_losses2 = _conv.f_vec_conduction_losses2
f_vec_p_on1 = _conv.f_vec_p_on1
f_vec_p_off1 = _conv.f_vec_p_off1
f_vec_p_rr2 = _conv.f_vec_p_rr2
f_vec_conduction_losses = _conv.f_vec_conduction_losses
f_vec_p_on_off1 = _conv.f_vec_p_on_off1
f_vec_p_on_off_rr_1_2 = _conv.f_vec_p_on_off_rr_1_2
f_vec_p1 = _conv.f_vec_p1
f_vec_p2 = _conv.f_vec_p2
f_vec_t_switch1 = _conv.f_vec_t_switch1
f_vec_t_diode2 = _conv.f_vec_t_diode2
