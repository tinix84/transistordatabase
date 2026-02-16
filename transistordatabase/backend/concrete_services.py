"""
Concrete implementations of backend services.

All ABC interfaces are imported from core.services. No duplicate interface
definitions exist in the backend layer.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from transistordatabase.core.adapters import core_to_legacy_dicts
from transistordatabase.core.models import (
    FosterThermalModel,
    Transistor,
)
from transistordatabase.core.services import (
    ICalculationService,
    IComparisonService,
    IExportService,
    IPlottingService,
    IValidationService,
)

logger = logging.getLogger(__name__)


def _load_data_file(filename: str) -> list[str]:
    """Load lines from a data file in the transistordatabase/data/ directory."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / filename
    items: list[str] = []
    with open(file_path) as f:
        for line in f.read().splitlines():
            if line.startswith("#") or line.isspace() or not line:
                continue
            items.append(str(line))
    return items


def _build_legacy_transistor(transistor: Transistor):
    """Convert a core Transistor to a legacy Transistor via the adapter.

    :param transistor: Core Transistor object.
    :return: Legacy ``transistordatabase.transistor.Transistor`` object.
    """
    from transistordatabase.transistor import Transistor as LegacyTransistor

    t_args, sw_args, di_args = core_to_legacy_dicts(transistor)
    housing_types = _load_data_file("housing_types.txt")
    manufacturers = _load_data_file("module_manufacturers.txt")
    return LegacyTransistor(
        t_args, sw_args, di_args,
        possible_housing_types=housing_types,
        possible_module_manufacturers=manufacturers,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
class PlottingService(IPlottingService):
    """Matplotlib-based plotting service returning structured plot data."""

    def __init__(self, backend: str = "matplotlib") -> None:
        self.backend = backend

    def plot_channel_characteristics(
        self, transistor: Transistor,
        component: str = "switch",
        temperatures: list[float] | None = None,
        gate_voltages: list[float] | None = None,
    ) -> Dict[str, Any]:
        """Plot channel characteristics (I-V curves)."""
        if component == "switch":
            channel_data = transistor.switch.channel_data
        elif component == "diode":
            channel_data = transistor.diode.channel_data
        else:
            raise ValueError(f"Invalid component: {component}")

        if not channel_data:
            return {"error": f"No channel data available for {component}"}

        plot_data: Dict[str, Any] = {
            'curves': [],
            'metadata': {
                'title': (
                    f'{transistor.metadata.name} - '
                    f'{component.title()} Channel Characteristics'
                ),
                'xlabel': 'Voltage [V]',
                'ylabel': 'Current [A]',
                'component': component,
                'transistor_name': transistor.metadata.name,
            },
        }

        for ch in channel_data:
            if temperatures and ch.t_j not in temperatures:
                continue
            if gate_voltages and ch.v_g not in gate_voltages:
                continue

            x = ch.graph_v_i[0]
            y = ch.graph_v_i[1]
            plot_data['curves'].append({
                'x_data': x.tolist() if hasattr(x, 'tolist') else list(x),
                'y_data': y.tolist() if hasattr(y, 'tolist') else list(y),
                'label': f'Vg = {ch.v_g}V, Tj = {ch.t_j} deg C',
                'metadata': {'t_j': ch.t_j, 'v_g': ch.v_g},
            })

        return plot_data

    def plot_switching_losses(
        self, transistor: Transistor,
        loss_type: str = "e_on",
        plot_type: str = "i_e",
    ) -> Dict[str, Any]:
        """Plot switching losses."""
        valid_loss = ("e_on", "e_off", "e_rr")
        valid_plot = ("i_e", "r_e", "t_e")
        if loss_type not in valid_loss:
            return {"error": f"Invalid loss_type: {loss_type}"}
        if plot_type not in valid_plot:
            return {"error": f"Invalid plot_type: {plot_type}"}

        if loss_type == "e_on":
            loss_data = transistor.switch.e_on_data
        elif loss_type == "e_off":
            loss_data = transistor.switch.e_off_data
        else:
            loss_data = transistor.diode.e_rr_data

        if not loss_data:
            return {"error": f"No {loss_type} data available"}

        plot_data: Dict[str, Any] = {
            'curves': [],
            'metadata': {
                'title': f'{transistor.metadata.name} - {loss_type.upper()}',
                'xlabel': {
                    'i_e': 'Current [A]',
                    'r_e': 'Gate Resistance [Ohm]',
                    't_e': 'Temperature [deg C]',
                }[plot_type],
                'ylabel': 'Energy [J]',
                'loss_type': loss_type,
                'plot_type': plot_type,
            },
        }

        dataset_attr = f'graph_{plot_type}'
        for loss in loss_data:
            if loss.dataset_type == dataset_attr:
                graph = getattr(loss, dataset_attr, None)
                if graph is not None:
                    x = graph[0]
                    y = graph[1]
                    plot_data['curves'].append({
                        'x_data': x.tolist() if hasattr(x, 'tolist') else list(x),
                        'y_data': y.tolist() if hasattr(y, 'tolist') else list(y),
                        'label': (
                            f'Vsupply={loss.v_supply}V, Vg={loss.v_g}V, '
                            f'Tj={loss.t_j} deg C, Rg={loss.r_g}Ohm'
                        ),
                        'metadata': {
                            'v_supply': loss.v_supply,
                            'v_g': loss.v_g,
                            't_j': loss.t_j,
                            'r_g': loss.r_g,
                        },
                    })

        return plot_data

    def plot_safe_operating_area(self, transistor: Transistor) -> Dict[str, Any]:
        """Plot safe operating area."""
        if not transistor.switch.soa:
            return {"error": "No SOA data available"}

        plot_data: Dict[str, Any] = {
            'curves': [],
            'metadata': {
                'title': f'{transistor.metadata.name} - Safe Operating Area',
                'xlabel': 'Voltage [V]',
                'ylabel': 'Current [A]',
                'transistor_name': transistor.metadata.name,
            },
        }

        for soa in transistor.switch.soa:
            if soa.graph_i_v is not None and soa.graph_i_v.size > 0:
                x = soa.graph_i_v[0]
                y = soa.graph_i_v[1]
                plot_data['curves'].append({
                    'x_data': x.tolist() if hasattr(x, 'tolist') else list(x),
                    'y_data': y.tolist() if hasattr(y, 'tolist') else list(y),
                    'label': (
                        f't_pulse = {soa.time_pulse}s, '
                        f'Tc = {soa.t_c} deg C'
                    ),
                    'metadata': {
                        'time_pulse': soa.time_pulse,
                        't_c': soa.t_c,
                    },
                })

        return plot_data

    def plot_thermal_impedance(self, transistor: Transistor) -> Dict[str, Any]:
        """Plot thermal impedance characteristics."""
        thermal_data: FosterThermalModel | None = None

        if transistor.switch.thermal_foster is not None:
            thermal_data = transistor.switch.thermal_foster
        elif transistor.diode.thermal_foster is not None:
            thermal_data = transistor.diode.thermal_foster

        if thermal_data is None or thermal_data.graph_t_rthjc is None:
            return {"error": "No thermal impedance data available"}

        x = thermal_data.graph_t_rthjc[0]
        y = thermal_data.graph_t_rthjc[1]
        plot_data: Dict[str, Any] = {
            'curves': [{
                'x_data': x.tolist() if hasattr(x, 'tolist') else list(x),
                'y_data': y.tolist() if hasattr(y, 'tolist') else list(y),
                'label': 'Thermal Impedance',
                'metadata': {'r_th_total': thermal_data.r_th_total},
            }],
            'metadata': {
                'title': f'{transistor.metadata.name} - Thermal Impedance',
                'xlabel': 'Time [s]',
                'ylabel': 'Thermal Impedance [K/W]',
                'transistor_name': transistor.metadata.name,
            },
        }
        return plot_data

    def plot_gate_charge(self, transistor: Transistor) -> Dict[str, Any]:
        """Plot gate charge characteristics."""
        if not transistor.switch.gate_charge_curves:
            return {"error": "No gate charge data available"}

        plot_data: Dict[str, Any] = {
            'curves': [],
            'metadata': {
                'title': f'{transistor.metadata.name} - Gate Charge',
                'xlabel': 'Gate Charge [C]',
                'ylabel': 'Gate Voltage [V]',
                'transistor_name': transistor.metadata.name,
            },
        }

        for gc in transistor.switch.gate_charge_curves:
            if gc.graph_q_v is not None:
                x = gc.graph_q_v[0]
                y = gc.graph_q_v[1]
                plot_data['curves'].append({
                    'x_data': x.tolist() if hasattr(x, 'tolist') else list(x),
                    'y_data': y.tolist() if hasattr(y, 'tolist') else list(y),
                    'label': (
                        f'Vsupply={gc.v_supply}V, Tj={gc.t_j} deg C, '
                        f'Ich={gc.i_channel}A'
                    ),
                    'metadata': {
                        'v_supply': gc.v_supply,
                        't_j': gc.t_j,
                        'i_channel': gc.i_channel,
                    },
                })

        return plot_data


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------
class CalculationService(ICalculationService):
    """Transistor calculation and analysis service."""

    def calculate_channel_resistance(
        self, channel_data: Any, current: float
    ) -> float:
        """Calculate channel resistance at specific current."""
        return channel_data.get_resistance_at_current(current)

    def interpolate_switching_losses(
        self, loss_data: list, conditions: Dict[str, float]
    ) -> float:
        """Interpolate switching losses for given conditions.

        Uses linear interpolation between available data points matching
        the closest temperature and supply voltage.
        """
        t_j = conditions.get('t_j', 25.0)
        i_channel = conditions.get('i_channel', 0.0)

        # Find datasets matching temperature
        matching = [d for d in loss_data if d.t_j == t_j]
        if not matching:
            # Find closest temperature
            temps = [d.t_j for d in loss_data]
            if not temps:
                return 0.0
            closest_t = min(temps, key=lambda x: abs(x - t_j))
            matching = [d for d in loss_data if d.t_j == closest_t]

        for data in matching:
            if data.dataset_type == 'graph_i_e' and data.graph_i_e is not None:
                currents = data.graph_i_e[0]
                energies = data.graph_i_e[1]
                return float(np.interp(i_channel, currents, energies))
            if data.dataset_type == 'single' and data.e_x is not None:
                return data.e_x

        return 0.0

    def calculate_thermal_impedance(
        self, thermal_model: Dict[str, Any],
        time_points: np.ndarray,
    ) -> np.ndarray:
        """Calculate thermal impedance Z_th(t) using Foster model."""
        r_th_vector = thermal_model.get('r_th_vector', [])
        tau_vector = thermal_model.get('tau_vector', [])

        if not r_th_vector or not tau_vector:
            r_th_total = thermal_model.get('r_th_total', 0.0)
            return np.full_like(time_points, r_th_total, dtype=np.float64)

        z_th = np.zeros_like(time_points, dtype=np.float64)
        for r_th, tau in zip(r_th_vector, tau_vector, strict=True):
            z_th += r_th * (1.0 - np.exp(-time_points / tau))
        return z_th

    def calculate_losses(
        self, transistor: Transistor, operating_point: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate transistor losses at given operating point."""
        t_j = operating_point.get('t_j', 25.0)
        v_g = operating_point.get('v_g', 15.0)
        i_channel = operating_point.get('i_channel', 0.0)
        f_sw = operating_point.get('f_sw', 10000.0)

        # Conduction losses from linearized model
        cond_loss = 0.0
        try:
            lm = transistor.switch.linearize_at_operating_point(
                t_j, v_g, i_channel
            )
            cond_loss = (lm.v0_channel * i_channel
                         + lm.r_channel * i_channel ** 2)
        except ValueError:
            pass

        # Switching losses
        e_on = self.interpolate_switching_losses(
            transistor.switch.e_on_data,
            {'t_j': t_j, 'i_channel': i_channel},
        )
        e_off = self.interpolate_switching_losses(
            transistor.switch.e_off_data,
            {'t_j': t_j, 'i_channel': i_channel},
        )
        sw_loss = (e_on + e_off) * f_sw

        return {
            'conduction_losses': cond_loss,
            'switching_losses_on': e_on * f_sw,
            'switching_losses_off': e_off * f_sw,
            'total_losses': cond_loss + sw_loss,
        }

    def find_optimal_working_point(
        self, transistor: Transistor, constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Find optimal working point based on constraints."""
        return {
            't_j': constraints.get('t_j_max', 125.0),
            'v_g': constraints.get('v_g', 15.0),
            'i_channel': constraints.get('i_max', 100.0),
        }

    def calculate_thermal_resistance(
        self, transistor: Transistor, power_profile: List[float]
    ) -> Dict[str, float]:
        """Calculate thermal resistance and junction temperature."""
        r_th = transistor.thermal_properties.r_th_cs or 0.0
        avg_power = sum(power_profile) / len(power_profile) if power_profile else 0.0
        t_j_rise = avg_power * r_th

        return {
            'r_th_jc': r_th,
            't_j_rise': t_j_rise,
            't_j_avg': 25.0 + t_j_rise,
        }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
class ExportService(IExportService):
    """Service for exporting transistor data to various formats."""

    def export_to_json(self, transistor: Transistor, file_path: Path) -> bool:
        """Export transistor to JSON format."""
        try:
            data = {
                'metadata': {
                    'name': transistor.metadata.name,
                    'type': transistor.metadata.type,
                    'manufacturer': transistor.metadata.manufacturer,
                    'housing_type': transistor.metadata.housing_type,
                },
                'electrical': {
                    'v_abs_max': transistor.electrical_ratings.v_abs_max,
                    'i_abs_max': transistor.electrical_ratings.i_abs_max,
                    'i_cont': transistor.electrical_ratings.i_cont,
                },
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            logger.exception("Error exporting to JSON")
            return False

    def export_to_csv(self, transistors: List[Transistor], file_path: Path) -> bool:
        """Export transistors to CSV format."""
        try:
            if not transistors:
                return False
            headers = [
                'name', 'type', 'manufacturer', 'v_abs_max',
                'i_abs_max', 'i_cont', 'housing_type',
            ]
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for t in transistors:
                    writer.writerow({
                        'name': t.metadata.name,
                        'type': t.metadata.type,
                        'manufacturer': t.metadata.manufacturer,
                        'v_abs_max': t.electrical_ratings.v_abs_max,
                        'i_abs_max': t.electrical_ratings.i_abs_max,
                        'i_cont': t.electrical_ratings.i_cont,
                        'housing_type': t.metadata.housing_type,
                    })
            return True
        except Exception:
            logger.exception("Error exporting to CSV")
            return False

    def export_to_spice(self, transistor: Transistor, file_path: Path) -> bool:
        """Export transistor model to SPICE format."""
        try:
            name = transistor.metadata.name.replace(' ', '_')
            lines = [
                f"* SPICE model for {transistor.metadata.name}",
                f"* Manufacturer: {transistor.metadata.manufacturer}",
                f"* Type: {transistor.metadata.type}",
                f".model {name} NPN (",
                f"+ BV={transistor.electrical_ratings.v_abs_max}",
                "+ )",
            ]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True
        except Exception:
            logger.exception("Error exporting to SPICE")
            return False

    def export_to_plecs(
        self, transistor: Transistor, output_path: Path,
        template_config: dict[str, Any] | None = None,
    ) -> None:
        """Export transistor to PLECS XML format.

        Bridges through the adapter to use the legacy PLECS exporter which
        relies on Jinja2 templates.

        :param transistor: Core Transistor object.
        :param output_path: Directory to write the XML files into.
        :param template_config: Optional config with ``recheck`` (bool) and
            ``gate_voltages`` (list of 4 floats: v_g_on, v_g_off, v_d_on,
            v_d_off).
        """
        legacy = _build_legacy_transistor(transistor)
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(out)
            recheck = True
            gate_voltages = None
            if template_config:
                recheck = template_config.get('recheck', True)
                gate_voltages = template_config.get('gate_voltages')
            legacy.export_plecs(recheck=recheck, gate_voltages=gate_voltages)
        finally:
            os.chdir(cwd)

    def export_to_matlab(self, transistor: Transistor, output_path: Path) -> None:
        """Export transistor data to MATLAB .mat format.

        :param transistor: Core Transistor object.
        :param output_path: Directory to write the .mat file into.
        """
        legacy = _build_legacy_transistor(transistor)
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(out)
            legacy.export_matlab()
        finally:
            os.chdir(cwd)

    def export_to_ltspice(self, transistor: Transistor, output_path: Path) -> None:
        """Export transistor to LTSpice DPT netlist.

        Uses the ``LTspiceDPT`` utility from ``utils/ltspice_dpt.py`` which
        works directly with core models.

        :param transistor: Core Transistor object.
        :param output_path: Directory or file path for the netlist.
        """
        from transistordatabase.utils.ltspice_dpt import DPTConfig, LTspiceDPT

        config = DPTConfig(
            v_dc=transistor.electrical_ratings.v_abs_max / 2,
            i_target=transistor.electrical_ratings.i_cont,
        )
        dpt = LTspiceDPT(config)
        dpt.generate_netlist(output_path, transistor=transistor)

    def export_to_gecko_circuits(
        self, transistor: Transistor, export_params: Dict[str, Any]
    ) -> Path:
        """Export transistor to GeckoCIRCUITS .scl format.

        :param transistor: Core Transistor object.
        :param export_params: Dict with optional keys: ``recheck`` (bool),
            ``v_supply``, ``v_g_on``, ``v_g_off``, ``r_g_on``, ``r_g_off``
            (all float), and ``output_path`` (str or Path).
        :return: Path to the output directory.
        """
        legacy = _build_legacy_transistor(transistor)
        out_dir = Path(export_params.get('output_path', '.'))
        out_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(out_dir)
            legacy.export_geckocircuits(
                recheck=export_params.get('recheck', True),
                v_supply=export_params.get('v_supply'),
                v_g_on=export_params.get('v_g_on'),
                v_g_off=export_params.get('v_g_off'),
                r_g_on=export_params.get('r_g_on'),
                r_g_off=export_params.get('r_g_off'),
            )
        finally:
            os.chdir(cwd)
        return out_dir


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
class ComparisonService(IComparisonService):
    """Service for comparing transistors with advanced plot generation."""

    def __init__(self) -> None:
        """Initialize comparison service with plotting support."""
        self.plotting_service = PlottingService()

    def compare_characteristics(
        self, transistors: List[Transistor], _comparison_type: str
    ) -> Dict[str, Any]:
        """Compare characteristics of multiple transistors.

        :param _comparison_type: Comparison type (reserved for future use)
        """
        return self.compare_transistors(transistors)

    def rank_transistors(
        self, transistors: List[Transistor], _criteria: Dict[str, float]
    ) -> list[tuple[Transistor, float]]:
        """Rank transistors based on given criteria.

        :param _criteria: Ranking criteria (reserved for future implementation)
        """
        return [(t, 1.0) for t in transistors]

    def compare_transistors(
        self, transistors: List[Transistor]
    ) -> Dict[str, Any]:
        """Compare multiple transistors (basic comparison)."""
        if len(transistors) < 2:
            return {'error': 'At least 2 transistors required for comparison'}

        v_values = [t.electrical_ratings.v_abs_max for t in transistors]
        i_values = [t.electrical_ratings.i_abs_max for t in transistors]

        return {
            'transistor_count': len(transistors),
            'transistors': [t.metadata.name for t in transistors],
            'electrical_comparison': {
                'voltage_max': {
                    'max': max(v_values), 'min': min(v_values),
                    'avg': sum(v_values) / len(v_values),
                },
                'current_max': {
                    'max': max(i_values), 'min': min(i_values),
                    'avg': sum(i_values) / len(i_values),
                },
            },
        }

    def generate_advanced_comparison(
        self,
        transistors: List[Transistor],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate advanced comparison with 9 plot types and configurations.

        :param transistors: 2-3 transistors to compare
        :param config: Configuration dict with keys:
            - t_j: Junction temperature (default 25°C)
            - v_supply: List of supply voltages per transistor
            - r_g_on: List of gate resistances for turn-on per transistor
            - r_g_off: List of gate resistances for turn-off per transistor
            - parallel_count: List of parallel transistor counts per transistor
            - i_channel: Channel current for comparison
        :return: Dict with 9 plot types
        """
        if not 2 <= len(transistors) <= 3:
            return {'error': 'Comparison requires 2-3 transistors'}

        t_j = config.get('t_j', 25.0)
        v_supply = config.get('v_supply', [600] * len(transistors))
        r_g_on = config.get('r_g_on', [10] * len(transistors))
        r_g_off = config.get('r_g_off', [10] * len(transistors))
        parallel_count = config.get('parallel_count', [1] * len(transistors))
        i_channel = config.get('i_channel', 50.0)

        return {
            'channel_characteristics': self._plot_channel_comparison(
                transistors, t_j
            ),
            'switching_losses_eon': self._plot_switching_comparison(
                transistors, 'e_on', t_j, v_supply, r_g_on
            ),
            'switching_losses_eoff': self._plot_switching_comparison(
                transistors, 'e_off', t_j, v_supply, r_g_off
            ),
            'gate_charge': self._plot_gate_charge_comparison(transistors, t_j),
            'soa': self._plot_soa_comparison(transistors),
            'thermal_impedance': self._plot_thermal_comparison(transistors),
            'capacitances': self._plot_capacitance_comparison(transistors, t_j),
            'loss_breakdown': self._plot_loss_breakdown(
                transistors, t_j, v_supply, r_g_on, r_g_off, i_channel
            ),
            'efficiency': self._plot_efficiency_comparison(
                transistors, t_j, v_supply, r_g_on, r_g_off, parallel_count
            ),
            'config': {
                't_j': t_j,
                'v_supply': v_supply,
                'r_g_on': r_g_on,
                'r_g_off': r_g_off,
                'parallel_count': parallel_count,
            }
        }

    def _plot_channel_comparison(
        self, transistors: List[Transistor], t_j: float
    ) -> Dict[str, Any]:
        """Generate channel characteristic comparison plot data."""
        curves = []

        for t in transistors:
            # Get channel data closest to target t_j
            channel_data = t.switch.channel_data
            if not channel_data:
                continue

            closest = min(
                channel_data,
                key=lambda ch: abs(ch.t_j - t_j),
                default=None
            )

            if closest and closest.graph_v_i is not None:
                v_data = closest.graph_v_i[0].tolist()
                i_data = closest.graph_v_i[1].tolist()
                curves.append({
                    'name': t.metadata.name,
                    'v_data': v_data,
                    'i_data': i_data,
                    't_j': closest.t_j,
                    'v_g': closest.v_g,
                })

        return {
            'title': f'Channel Characteristics Comparison (T_j ≈ {t_j}°C)',
            'xlabel': 'Voltage [V]',
            'ylabel': 'Current [A]',
            'curves': curves,
        }

    def _plot_switching_comparison(
        self,
        transistors: List[Transistor],
        loss_type: str,
        t_j: float,
        v_supply: List[float],
        r_g: List[float]
    ) -> Dict[str, Any]:
        """Generate switching loss comparison plot data."""
        curves = []

        for idx, t in enumerate(transistors):
            if loss_type == 'e_on':
                loss_data = t.switch.e_on_data
            elif loss_type == 'e_off':
                loss_data = t.switch.e_off_data
            else:
                continue

            if not loss_data:
                continue

            # Find data matching conditions
            target_v = v_supply[idx] if idx < len(v_supply) else 600
            target_r_g = r_g[idx] if idx < len(r_g) else 10

            for loss in loss_data:
                if abs(loss.t_j - t_j) > 25:
                    continue
                if abs(loss.v_supply - target_v) > 100:
                    continue
                if loss.r_g and abs(loss.r_g - target_r_g) > 5:
                    continue

                if loss.graph_i_e is not None:
                    curves.append({
                        'name': f"{t.metadata.name} (R_g={loss.r_g}Ω)",
                        'i_data': loss.graph_i_e[0].tolist(),
                        'e_data': loss.graph_i_e[1].tolist(),
                        't_j': loss.t_j,
                        'v_supply': loss.v_supply,
                        'r_g': loss.r_g,
                    })
                    break

        return {
            'title': f'{loss_type.upper()} Switching Loss Comparison',
            'xlabel': 'Current [A]',
            'ylabel': 'Energy [J]',
            'curves': curves,
        }

    def _plot_gate_charge_comparison(
        self, transistors: List[Transistor], t_j: float
    ) -> Dict[str, Any]:
        """Generate gate charge comparison plot data."""
        curves = []

        for t in transistors:
            gc_curves = t.switch.gate_charge_curves
            if not gc_curves:
                continue

            closest = min(
                gc_curves,
                key=lambda gc: abs(gc.t_j - t_j),
                default=None
            )

            if closest and closest.graph_q_v is not None:
                curves.append({
                    'name': t.metadata.name,
                    'q_data': closest.graph_q_v[0].tolist(),
                    'v_data': closest.graph_q_v[1].tolist(),
                    't_j': closest.t_j,
                    'v_supply': closest.v_supply,
                })

        return {
            'title': f'Gate Charge Comparison (T_j ≈ {t_j}°C)',
            'xlabel': 'Gate Charge [C]',
            'ylabel': 'Gate Voltage [V]',
            'curves': curves,
        }

    def _plot_soa_comparison(
        self, transistors: List[Transistor]
    ) -> Dict[str, Any]:
        """Generate SOA comparison plot data."""
        curves = []

        for t in transistors:
            soa_curves = t.switch.soa
            if not soa_curves:
                continue

            for soa in soa_curves:
                if soa.graph_i_v is not None and len(soa.graph_i_v) == 2:
                    curves.append({
                        'name': f"{t.metadata.name} (T_c={soa.t_c}°C, t={soa.time_pulse}s)",
                        'v_data': soa.graph_i_v[0].tolist(),
                        'i_data': soa.graph_i_v[1].tolist(),
                        't_c': soa.t_c,
                        'time_pulse': soa.time_pulse,
                    })

        return {
            'title': 'Safe Operating Area Comparison',
            'xlabel': 'Voltage [V]',
            'ylabel': 'Current [A]',
            'curves': curves,
        }

    def _plot_thermal_comparison(
        self, transistors: List[Transistor]
    ) -> Dict[str, Any]:
        """Generate thermal impedance comparison plot data."""
        curves = []

        time_points = np.logspace(-6, 1, 100)  # 1µs to 10s

        for t in transistors:
            foster = t.switch.thermal_foster
            if not foster:
                continue

            try:
                z_th = [foster.get_thermal_impedance(tp) for tp in time_points]
                curves.append({
                    'name': t.metadata.name,
                    'time_data': time_points.tolist(),
                    'z_th_data': z_th,
                })
            except (ValueError, AttributeError):
                continue

        return {
            'title': 'Thermal Impedance Comparison',
            'xlabel': 'Time [s]',
            'ylabel': 'Thermal Impedance [K/W]',
            'curves': curves,
            'log_scale_x': True,
        }

    def _plot_capacitance_comparison(
        self, transistors: List[Transistor], t_j: float
    ) -> Dict[str, Any]:
        """Generate capacitance comparison plot data."""
        cap_types = {'c_oss': 'C_oss', 'c_iss': 'C_iss', 'c_rss': 'C_rss'}
        all_curves = {}

        for cap_type, label in cap_types.items():
            curves = []
            for t in transistors:
                cap_list = getattr(t, cap_type, [])
                if not cap_list:
                    continue

                closest = min(
                    cap_list,
                    key=lambda c: abs(c.t_j - t_j),
                    default=None
                )

                if closest and closest.graph_v_c is not None:
                    curves.append({
                        'name': f"{t.metadata.name} - {label}",
                        'v_data': closest.graph_v_c[0].tolist(),
                        'c_data': closest.graph_v_c[1].tolist(),
                        't_j': closest.t_j,
                    })

            all_curves[cap_type] = curves

        return {
            'title': f'Capacitance Comparison (T_j ≈ {t_j}°C)',
            'xlabel': 'Voltage [V]',
            'ylabel': 'Capacitance [F]',
            'c_oss': all_curves.get('c_oss', []),
            'c_iss': all_curves.get('c_iss', []),
            'c_rss': all_curves.get('c_rss', []),
        }

    def _plot_loss_breakdown(
        self,
        transistors: List[Transistor],
        t_j: float,
        v_supply: List[float],
        r_g_on: List[float],
        r_g_off: List[float],
        i_channel: float
    ) -> Dict[str, Any]:
        """Generate loss breakdown comparison (conduction + switching)."""
        breakdown = []

        for idx, t in enumerate(transistors):
            target_v = v_supply[idx] if idx < len(v_supply) else 600

            # Estimate conduction loss
            p_cond = 0.0
            if t.switch.channel_data:
                try:
                    closest_ch = min(
                        t.switch.channel_data,
                        key=lambda ch: abs(ch.t_j - t_j)
                    )
                    r_ds = closest_ch.get_resistance_at_current(i_channel)
                    p_cond = r_ds * (i_channel ** 2)
                except (ValueError, AttributeError):
                    pass

            # Estimate switching losses
            p_sw_on = 0.0
            p_sw_off = 0.0

            if t.switch.e_on_data:
                for loss in t.switch.e_on_data:
                    if abs(loss.t_j - t_j) < 25 and abs(loss.v_supply - target_v) < 100:
                        if loss.graph_i_e is not None:
                            try:
                                e_on = float(np.interp(
                                    i_channel,
                                    loss.graph_i_e[0],
                                    loss.graph_i_e[1]
                                ))
                                p_sw_on = e_on * 100000  # Assume 100kHz
                            except Exception:
                                pass
                        break

            if t.switch.e_off_data:
                for loss in t.switch.e_off_data:
                    if abs(loss.t_j - t_j) < 25 and abs(loss.v_supply - target_v) < 100:
                        if loss.graph_i_e is not None:
                            try:
                                e_off = float(np.interp(
                                    i_channel,
                                    loss.graph_i_e[0],
                                    loss.graph_i_e[1]
                                ))
                                p_sw_off = e_off * 100000  # Assume 100kHz
                            except Exception:
                                pass
                        break

            p_total = p_cond + p_sw_on + p_sw_off

            breakdown.append({
                'name': t.metadata.name,
                'conduction': p_cond,
                'switching_on': p_sw_on,
                'switching_off': p_sw_off,
                'total': p_total,
            })

        return {
            'title': 'Loss Breakdown Comparison',
            'breakdown': breakdown,
            'conditions': {
                't_j': t_j,
                'i_channel': i_channel,
                'f_sw': 100000,
            }
        }

    def _plot_efficiency_comparison(
        self,
        transistors: List[Transistor],
        t_j: float,
        v_supply: List[float],
        r_g_on: List[float],
        r_g_off: List[float],
        parallel_count: List[int]
    ) -> Dict[str, Any]:
        """Generate efficiency vs current comparison."""
        curves = []

        current_points = np.linspace(1, 100, 50)

        for idx, t in enumerate(transistors):
            efficiencies = []
            target_v = v_supply[idx] if idx < len(v_supply) else 600
            n_parallel = parallel_count[idx] if idx < len(parallel_count) else 1

            for i_load in current_points:
                i_per_device = i_load / n_parallel

                # Calculate losses per device
                p_cond = 0.0
                if t.switch.channel_data:
                    try:
                        closest_ch = min(
                            t.switch.channel_data,
                            key=lambda ch: abs(ch.t_j - t_j)
                        )
                        r_ds = closest_ch.get_resistance_at_current(i_per_device)
                        p_cond = r_ds * (i_per_device ** 2)
                    except Exception:
                        p_cond = 0.01 * i_per_device  # Fallback

                p_sw = 0.0001 * i_per_device  # Simplified switching loss

                p_loss_total = (p_cond + p_sw) * n_parallel
                p_out = target_v * i_load
                p_in = p_out + p_loss_total

                efficiency = (p_out / p_in * 100) if p_in > 0 else 0
                efficiencies.append(efficiency)

            curves.append({
                'name': f"{t.metadata.name} (x{n_parallel})",
                'current_data': current_points.tolist(),
                'efficiency_data': efficiencies,
            })

        return {
            'title': 'Efficiency vs Load Current',
            'xlabel': 'Load Current [A]',
            'ylabel': 'Efficiency [%]',
            'curves': curves,
        }

    def find_similar_transistors(
        self, target: Transistor, candidates: List[Transistor],
        tolerance: float = 0.1,
    ) -> List[Transistor]:
        """Find transistors similar to target within tolerance."""
        target_v = target.electrical_ratings.v_abs_max
        target_i = target.electrical_ratings.i_abs_max
        similar = []

        for c in candidates:
            if c.metadata.name == target.metadata.name:
                continue
            v_diff = abs(c.electrical_ratings.v_abs_max - target_v) / max(target_v, 1)
            i_diff = abs(c.electrical_ratings.i_abs_max - target_i) / max(target_i, 1)
            if v_diff <= tolerance and i_diff <= tolerance:
                similar.append(c)

        similar.sort(key=lambda t: (
            abs(t.electrical_ratings.v_abs_max - target_v) / max(target_v, 1)
            + abs(t.electrical_ratings.i_abs_max - target_i) / max(target_i, 1)
        ))
        return similar


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationService(IValidationService):
    """Service for validating transistor data."""

    def validate_transistor(self, transistor: Transistor) -> Dict[str, List[str]]:
        """Validate complete transistor data."""
        errors: list[str] = []
        warnings: list[str] = []

        # Metadata
        if not transistor.metadata.name or not transistor.metadata.name.strip():
            errors.append("Transistor name is required")
        valid_types = ('IGBT', 'MOSFET', 'SiC-MOSFET', 'GaN-Transistor', 'Diode')
        if transistor.metadata.type not in valid_types:
            warnings.append(f"Unusual transistor type: {transistor.metadata.type}")
        if not transistor.metadata.manufacturer:
            warnings.append("Manufacturer is not specified")

        # Electrical
        if transistor.electrical_ratings.v_abs_max <= 0:
            errors.append("Maximum voltage must be positive")
        if transistor.electrical_ratings.i_abs_max <= 0:
            errors.append("Maximum current must be positive")
        if transistor.electrical_ratings.i_cont > transistor.electrical_ratings.i_abs_max:
            errors.append("Continuous current cannot exceed maximum current")

        # Thermal
        if (transistor.thermal_properties.r_th_cs is not None
                and transistor.thermal_properties.r_th_cs < 0):
            errors.append("Thermal resistance cannot be negative")

        return {'errors': errors, 'warnings': warnings}

    def validate_channel_data(self, channel_data: list) -> List[str]:
        """Validate channel characteristics data."""
        errors: list[str] = []
        for i, ch in enumerate(channel_data):
            if ch.graph_v_i is None or ch.graph_v_i.size == 0:
                errors.append(f"Channel data [{i}]: empty graph_v_i")
            if ch.graph_v_i is not None and ch.graph_v_i.ndim != 2:
                errors.append(f"Channel data [{i}]: graph_v_i must be 2D")
        return errors

    def validate_switching_data(self, switching_data: list) -> List[str]:
        """Validate switching loss data."""
        errors: list[str] = []
        for i, sd in enumerate(switching_data):
            if sd.dataset_type not in ('single', 'graph_i_e', 'graph_r_e', 'graph_t_e'):
                errors.append(f"Switching data [{i}]: invalid dataset_type")
        return errors

    def check_data_consistency(self, transistor: Transistor) -> List[str]:
        """Check internal data consistency."""
        issues: list[str] = []
        if (transistor.metadata.type == 'Diode'
                and transistor.electrical_ratings.i_cont
                > transistor.electrical_ratings.i_abs_max * 0.8):
            issues.append(
                "Continuous current high relative to maximum for a diode"
            )
        return issues

    def check_data_completeness(self, transistor: Transistor) -> Dict[str, Any]:
        """Check how complete the transistor data is."""
        meta_fields = ['name', 'type', 'manufacturer', 'housing_type', 'author']
        meta_filled = sum(
            1 for f in meta_fields if getattr(transistor.metadata, f, None)
        )
        meta_pct = meta_filled / len(meta_fields) * 100

        elec_fields = ['v_abs_max', 'i_abs_max', 'i_cont']
        elec_filled = sum(
            1 for f in elec_fields
            if getattr(transistor.electrical_ratings, f, None) is not None
        )
        elec_pct = elec_filled / len(elec_fields) * 100

        switch_pct = 100.0 if transistor.switch.channel_data else 0.0
        diode_pct = 100.0 if transistor.diode.channel_data else 0.0

        overall = (meta_pct + elec_pct + switch_pct + diode_pct) / 4.0

        return {
            'metadata': meta_pct,
            'electrical': elec_pct,
            'switch_data': switch_pct,
            'diode_data': diode_pct,
            'overall': overall,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class ConcreteServiceFactory:
    """Factory for creating concrete service implementations."""

    @staticmethod
    def create_plotting_service() -> IPlottingService:
        """Create plotting service."""
        return PlottingService()

    @staticmethod
    def create_calculation_service() -> ICalculationService:
        """Create calculation service."""
        return CalculationService()

    @staticmethod
    def create_export_service() -> IExportService:
        """Create export service."""
        return ExportService()

    @staticmethod
    def create_comparison_service() -> IComparisonService:
        """Create comparison service."""
        return ComparisonService()

    @staticmethod
    def create_validation_service() -> IValidationService:
        """Create validation service."""
        return ValidationService()

    @staticmethod
    def create_all_services() -> Dict[str, Any]:
        """Create all services."""
        return {
            'plotting': ConcreteServiceFactory.create_plotting_service(),
            'calculation': ConcreteServiceFactory.create_calculation_service(),
            'export': ConcreteServiceFactory.create_export_service(),
            'comparison': ConcreteServiceFactory.create_comparison_service(),
            'validation': ConcreteServiceFactory.create_validation_service(),
        }
