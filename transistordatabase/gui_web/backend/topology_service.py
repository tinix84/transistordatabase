"""
Topology calculator service for web API.

Provides simplified interface to topology converters for REST API usage.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from transistordatabase.core.models import Transistor
from transistordatabase.topologies.converter_common import (
    BuckConverter,
    BoostConverter,
    BuckBoostConverter,
)


class TopologyCalculatorService:
    """Service for DC-DC converter topology calculations."""

    SUPPORTED_TOPOLOGIES = ['buck', 'boost', 'buck-boost']

    @staticmethod
    def calculate(
        topology: str,
        parameters: Dict[str, Any],
        transistor: Optional[Transistor] = None
    ) -> Dict[str, Any]:
        """Calculate converter performance for given topology and parameters.

        :param topology: Topology type ('buck', 'boost', 'buck-boost')
        :param parameters: Operating parameters dict
        :param transistor: Optional transistor for accurate loss calculation
        :return: Calculation results dict
        """
        if topology not in TopologyCalculatorService.SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"Unsupported topology '{topology}'. "
                f"Supported: {', '.join(TopologyCalculatorService.SUPPORTED_TOPOLOGIES)}"
            )

        # Extract parameters
        v_in = parameters.get('v_in', 400)
        v_out = parameters.get('v_out', 48)
        p_out = parameters.get('p_out', 1000)
        f_sw = parameters.get('f_sw', 100000)  # Hz
        t_j = parameters.get('t_j', 25)
        v_g = parameters.get('v_g', 15)
        r_g_on = parameters.get('r_g_on', 10)
        r_g_off = parameters.get('r_g_off', 10)

        # Calculate basic converter parameters
        if topology == 'buck':
            duty_cycle = v_out / v_in
            i_in_avg = p_out / v_in
            i_out_avg = p_out / v_out
            i_sw_rms = i_out_avg * np.sqrt(duty_cycle)
            i_diode_rms = i_out_avg * np.sqrt(1 - duty_cycle)
            v_sw_block = v_in
            v_diode_block = v_in

        elif topology == 'boost':
            duty_cycle = 1 - (v_in / v_out)
            i_in_avg = p_out / v_in / (1 - duty_cycle)
            i_out_avg = p_out / v_out
            i_sw_rms = i_in_avg * np.sqrt(duty_cycle)
            i_diode_rms = i_in_avg * np.sqrt(1 - duty_cycle)
            v_sw_block = v_out
            v_diode_block = v_out

        else:  # buck-boost
            duty_cycle = v_out / (v_in + v_out)
            i_in_avg = p_out * duty_cycle / (v_in * (1 - duty_cycle))
            i_out_avg = p_out / v_out
            i_sw_rms = i_in_avg * np.sqrt(duty_cycle)
            i_diode_rms = i_in_avg * np.sqrt(1 - duty_cycle)
            v_sw_block = v_in + v_out
            v_diode_block = v_in + v_out

        # Estimate losses (simplified if no transistor provided)
        if transistor:
            # Use real transistor data for conduction loss
            try:
                closest_ch = min(
                    transistor.switch.channel_data,
                    key=lambda ch: abs(ch.t_j - t_j)
                )
                r_ds_on = closest_ch.get_resistance_at_current(i_sw_rms)
            except Exception:
                r_ds_on = 0.1  # Fallback

            # Estimate switching losses
            e_on = 0
            e_off = 0
            if transistor.switch.e_on_data:
                try:
                    for loss in transistor.switch.e_on_data:
                        if abs(loss.t_j - t_j) < 25 and loss.graph_i_e is not None:
                            e_on = float(np.interp(
                                i_sw_rms,
                                loss.graph_i_e[0],
                                loss.graph_i_e[1]
                            ))
                            break
                except Exception:
                    pass

            if transistor.switch.e_off_data:
                try:
                    for loss in transistor.switch.e_off_data:
                        if abs(loss.t_j - t_j) < 25 and loss.graph_i_e is not None:
                            e_off = float(np.interp(
                                i_sw_rms,
                                loss.graph_i_e[0],
                                loss.graph_i_e[1]
                            ))
                            break
                except Exception:
                    pass

        else:
            # Simplified estimates
            r_ds_on = 0.1
            e_on = 1e-4  # 100 µJ
            e_off = 1e-4

        # Calculate losses
        p_cond_switch = r_ds_on * (i_sw_rms ** 2)
        p_sw_on = e_on * f_sw
        p_sw_off = e_off * f_sw
        p_sw_total = p_sw_on + p_sw_off

        # Diode losses (simplified)
        v_f_diode = 1.0  # Forward voltage drop
        p_cond_diode = v_f_diode * i_diode_rms

        p_loss_total = p_cond_switch + p_sw_total + p_cond_diode
        p_in = p_out + p_loss_total
        efficiency = (p_out / p_in * 100) if p_in > 0 else 0

        # Generate waveform data (time-domain)
        t_period = 1 / f_sw
        time_points = np.linspace(0, t_period, 1000)
        t_on = duty_cycle * t_period

        # Current waveform (triangular approximation)
        i_waveform = []
        for t in time_points:
            if t < t_on:
                # Linear rise during on-time
                i_waveform.append(i_sw_rms * 1.5 * (t / t_on))
            else:
                # Linear fall during off-time
                i_waveform.append(i_sw_rms * 1.5 * (1 - (t - t_on) / (t_period - t_on)))

        # Voltage waveform (switch node)
        v_sw_waveform = [v_in if t < t_on else 0 for t in time_points]

        return {
            'topology': topology,
            'parameters': {
                'v_in': v_in,
                'v_out': v_out,
                'p_out': p_out,
                'f_sw': f_sw,
                't_j': t_j,
            },
            'calculated': {
                'duty_cycle': duty_cycle,
                'i_in_avg': i_in_avg,
                'i_out_avg': i_out_avg,
                'i_sw_rms': i_sw_rms,
                'i_diode_rms': i_diode_rms,
                'v_sw_block': v_sw_block,
                'v_diode_block': v_diode_block,
            },
            'losses': {
                'conduction_switch': p_cond_switch,
                'switching_on': p_sw_on,
                'switching_off': p_sw_off,
                'switching_total': p_sw_total,
                'conduction_diode': p_cond_diode,
                'total': p_loss_total,
            },
            'performance': {
                'p_in': p_in,
                'p_out': p_out,
                'efficiency': efficiency,
            },
            'stress': {
                'switch_voltage_max': v_sw_block,
                'switch_current_rms': i_sw_rms,
                'diode_voltage_max': v_diode_block,
                'diode_current_rms': i_diode_rms,
                'utilization_switch': (i_sw_rms / transistor.electrical_ratings.i_abs_max * 100)
                    if transistor else 0,
                'utilization_voltage': (v_sw_block / transistor.electrical_ratings.v_abs_max * 100)
                    if transistor else 0,
            },
            'waveforms': {
                'time': time_points.tolist(),
                'current': i_waveform,
                'voltage_switch': v_sw_waveform,
            }
        }

    @staticmethod
    def compare_topologies(
        parameters: Dict[str, Any],
        transistor: Optional[Transistor] = None
    ) -> Dict[str, Any]:
        """Compare all three topologies for given parameters.

        :param parameters: Operating parameters
        :param transistor: Optional transistor
        :return: Comparison results
        """
        results = {}

        for topology in TopologyCalculatorService.SUPPORTED_TOPOLOGIES:
            try:
                results[topology] = TopologyCalculatorService.calculate(
                    topology,
                    parameters,
                    transistor
                )
            except Exception as e:
                results[topology] = {
                    'error': str(e)
                }

        # Add comparison summary
        efficiencies = {
            topo: res.get('performance', {}).get('efficiency', 0)
            for topo, res in results.items()
            if 'error' not in res
        }

        if efficiencies:
            best_topology = max(efficiencies, key=efficiencies.get)
            results['comparison'] = {
                'efficiencies': efficiencies,
                'best_topology': best_topology,
                'best_efficiency': efficiencies[best_topology],
            }

        return results
