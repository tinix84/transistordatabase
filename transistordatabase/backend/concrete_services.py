"""
Concrete implementations of backend services.

All ABC interfaces are imported from core.services. No duplicate interface
definitions exist in the backend layer.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
        """Bounded lookup at fully specified recorded conditions; never nearest/clamped."""
        from transistordatabase.static_losses import energy_at

        return energy_at(loss_data, conditions)

    def calculate_thermal_impedance(
        self, thermal_model: Dict[str, Any],
        time_points: np.ndarray,
    ) -> np.ndarray:
        """Calculate thermal impedance Z_th(t) using Foster model."""
        from transistordatabase.thermal import thermal_impedance

        return thermal_impedance(thermal_model, time_points)

    def calculate_losses(
        self, transistor: Transistor, operating_point: Dict[str, float]
    ) -> Dict[str, float]:
        """Explicit held-current static-duty estimate; see static_losses contract."""
        from transistordatabase.static_losses import static_duty_losses

        return static_duty_losses(transistor, operating_point)

    def find_optimal_working_point(
        self, transistor: Transistor, constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Reject the former placeholder, which never evaluated an objective."""
        raise NotImplementedError(
            "No optimizer is implemented by this signature. Declare finite candidates "
            "and a named physical objective; constraint maxima are not an optimum."
        )

    def calculate_thermal_resistance(
        self, transistor: Transistor, power_profile: List[float]
    ) -> Dict[str, float]:
        """Reject the legacy signature's missing reference/path/time evidence."""
        raise ValueError(
            "Thermal calculation needs an explicit case/sink reference, physical "
            "path resistances and a time-weighted heat profile. Use thermal.steady_paths "
            "or thermal.simulate_foster; r_th_cs is not junction-to-case resistance."
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
class ExportService(IExportService):
    """Service for exporting transistor data to various formats."""

    def export_to_json(self, transistor: Transistor, file_path: Path) -> bool:
        """Export complete reloadable native core data, including source provenance."""
        try:
            from transistordatabase.core.repository import JsonTransistorLoader

            JsonTransistorLoader().save_to_json(transistor, file_path)
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
        """Reject an uncalibrated device model instead of emitting an arbitrary NPN."""
        raise NotImplementedError(
            "Core ratings/curves do not establish a calibrated SPICE model. "
            "Use the DPT workbench with an explicitly selected generic template "
            "or a supplied manufacturer model."
        )

    def export_to_plecs(
        self, transistor: Transistor, output_path: Path,
        template_config: dict[str, Any] | None = None,
    ) -> None:
        """Write a validated source-backed PLECS model without process cwd changes.

        Unsupported generic core selections fail explicitly. Gate defaults and
        nearest-condition legacy templates are not scientific export policies.
        """
        from transistordatabase.plecs_document import render_core_plecs
        from transistordatabase.plecs_core_export import render_core_selection
        import re

        if template_config and set(template_config) != {"selection"}:
            raise ValueError("Legacy PLECS template options are unsupported; use explicit document mapping")
        content = (render_core_selection(transistor, template_config["selection"])
                   if template_config else render_core_plecs(transistor))
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', transistor.metadata.name).strip('.')
        if not safe_name:
            raise ValueError("PLECS export requires a valid device name")
        destination = out / (safe_name + '.xml')
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing PLECS export: {destination}")
        with destination.open('xb') as target:
            target.write(content)

    def export_to_matlab(self, transistor: Transistor, output_path: Path) -> None:
        """Export MAT5 with the numeric legacy view and complete native core evidence."""
        from transistordatabase.native_export import render_matlab, write_export_files

        write_export_files(output_path, render_matlab(transistor))

    def export_to_ltspice(self, transistor: Transistor, output_path: Path) -> None:
        """Reject the signature's missing DPT conditions and template scope."""
        raise NotImplementedError(
            "Use /api/dpt/generate with explicit bus, current, gate/timing inputs "
            "and generic-template selection. Ratings do not choose a DPT operating point."
        )

    def export_to_gecko_circuits(
        self, transistor: Transistor, export_params: Dict[str, Any]
    ) -> Path:
        """Export one exact switch slice and its native evidence manifest.

        Declare component, t_j, v_supply, v_g_on, v_g_off, r_g_on and r_g_off.
        Unknown gate conditions require None. See native_export.render_gecko.
        """
        from transistordatabase.native_export import render_gecko, write_export_files

        if not export_params.get("output_path"):
            raise ValueError("Declare an explicit output_path for Gecko export")
        files = render_gecko(transistor, export_params)
        return write_export_files(export_params["output_path"], files)


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
        """Reject the former equal-score placeholder until a physical objective is named."""
        raise NotImplementedError(
            "Ranking requires a named physical metric, direction and complete "
            "comparable evaluation evidence; arbitrary equal scores are unsupported."
        )

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
        self, transistors: List[Transistor], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare recorded observations and explicit static-duty estimates separately.

        Observations retain every source slice and its actual conditions. Requested
        settings apply only to the estimate panel, never to selection by proximity.
        """
        from transistordatabase.static_losses import finite, static_duty_losses

        if not 2 <= len(transistors) <= 3:
            raise ValueError("Comparison requires 2-3 transistors")
        if not isinstance(config, dict):
            raise ValueError("Comparison config must be an object")
        count = len(transistors)
        per_device = ("v_supply", "v_g", "v_g_off", "r_g_on", "r_g_off", "parallel_count")
        for key in per_device:
            if key in config and (not isinstance(config[key], list) or len(config[key]) != count):
                raise ValueError(f"{key} must contain one explicit value per selected transistor")

        def panel(title, xlabel, ylabel):
            return {"title": title, "xlabel": xlabel, "ylabel": ylabel,
                    "status": "recorded-observations", "curves": [], "diagnostics": []}

        result = {
            "channel_characteristics": panel("Recorded channel slices", "Voltage [V]", "Current [A]"),
            "switching_losses_eon": panel("Recorded turn-on slices", "Current [A]", "Energy [J]"),
            "switching_losses_eoff": panel("Recorded turn-off slices", "Current [A]", "Energy [J]"),
            "gate_charge": panel("Recorded gate-charge slices", "Gate charge [C]", "Gate voltage [V]"),
            "soa": panel("Recorded safe operating area", "Voltage [V]", "Current [A]"),
            "thermal_impedance": panel("Foster transient impedance", "Time [s]", "Impedance [K/W]"),
            "capacitances": panel("Recorded capacitance slices", "Voltage [V]", "Capacitance [F]"),
        }
        result["thermal_impedance"]["log_scale_x"] = True

        def observation(key, transistor, index, graph, x_name, y_name, conditions, suffix=""):
            if graph is None:
                return
            graph = np.asarray(graph, dtype=float)
            if (graph.ndim != 2 or graph.shape[0] != 2 or graph.shape[1] < 2
                    or not np.isfinite(graph).all()):
                result[key]["diagnostics"].append({"name": transistor.metadata.name,
                    "record_index": index, "status": "invalid-data", "reason": "Need finite paired recorded samples"})
                return
            try:
                for name, value in conditions.items():
                    if value is not None:
                        finite(value, name)
            except ValueError as exc:
                result[key]["diagnostics"].append({"name": transistor.metadata.name,
                    "record_index": index, "status": "invalid-data", "reason": str(exc)})
                return
            label = ", ".join(f"{k}={v if v is not None else 'unknown'}" for k, v in conditions.items())
            result[key]["curves"].append({"name": f"{transistor.metadata.name}{suffix} [{index}] ({label})",
                "device": transistor.metadata.name, "record_index": index,
                x_name: graph[0].tolist(), y_name: graph[1].tolist(),
                "conditions": conditions, **conditions})

        result["thermal_impedance"].update(self._plot_thermal_comparison(transistors))

        breakdown, efficiency = [], []
        for index, transistor in enumerate(transistors):
            for j, record in enumerate(transistor.switch.channel_data):
                observation("channel_characteristics", transistor, j, record.graph_v_i,
                            "v_data", "i_data", {"t_j": record.t_j, "v_g": record.v_g})
            for key, records in (("switching_losses_eon", transistor.switch.e_on_data),
                                 ("switching_losses_eoff", transistor.switch.e_off_data)):
                for j, record in enumerate(records):
                    if record.dataset_type != "graph_i_e":
                        result[key]["diagnostics"].append({"name": transistor.metadata.name,
                            "record_index": j, "status": "unsupported-plot", "reason": f"{record.dataset_type} is not a current-energy curve"})
                        continue
                    observation(key, transistor, j, record.graph_i_e, "i_data", "e_data",
                        {"t_j": record.t_j, "v_supply": record.v_supply, "v_g": record.v_g,
                         "r_g": record.r_g, "v_g_off": record.v_g_off})
            for j, record in enumerate(transistor.switch.gate_charge_curves):
                observation("gate_charge", transistor, j, record.graph_q_v, "q_data", "v_data",
                            {"t_j": record.t_j, "v_supply": record.v_supply,
                             "i_channel": record.i_channel, "i_g": record.i_g})
            for j, record in enumerate(transistor.switch.soa):
                observation("soa", transistor, j, record.graph_i_v, "v_data", "i_data",
                            {"t_c": record.t_c, "time_pulse": record.time_pulse})
            for kind in ("c_oss", "c_iss", "c_rss"):
                for j, record in enumerate(getattr(transistor, kind, [])):
                    observation("capacitances", transistor, j, record.graph_v_c, "v_data", "c_data",
                                {"t_j": record.t_j}, " " + kind)
            row = {"name": transistor.metadata.name, "status": "unsupported", "total": None,
                   "conduction": None, "switching_on": None, "switching_off": None}
            try:
                required = {"t_j", "i_channel", "f_sw", "duty", *per_device}
                missing = sorted(required - config.keys())
                if missing:
                    raise ValueError("Explicit static-duty estimate needs " + ", ".join(missing))
                n = finite(config["parallel_count"][index], "parallel_count")
                if n < 1 or n != int(n):
                    raise ValueError("parallel_count must be a positive integer")
                op = {key: config[key] for key in ("t_j", "f_sw", "duty")}
                op.update({key: config[key][index] for key in per_device if key != "parallel_count"})
                total_current = finite(config["i_channel"], "total shared current")
                op["i_channel"] = total_current / n
                if total_current > 0 and op["i_channel"] == 0:
                    raise ValueError("Per-device current is below representable positive SI precision")
                calculation = static_duty_losses(transistor, op)
                values = {"conduction": calculation["conduction_losses"] * n,
                          "switching_on": calculation["switching_losses_on"] * n,
                          "switching_off": calculation["switching_losses_off"] * n,
                          "total": calculation["total_losses"] * n}
                for key, value in values.items():
                    finite(value, key)
                row.update(values, status="computed", calculation=calculation, parallel_count=int(n),
                           assumptions=["Selected ideal equal current sharing across identical devices"])
            except (ValueError, OverflowError) as exc:
                row["reason"] = str(exc)
            breakdown.append(row)
            eta = {"name": transistor.metadata.name, "status": "unsupported", "efficiency_percent": None}
            try:
                if row["status"] != "computed":
                    raise ValueError(row["reason"])
                output = finite(config.get("output_power_w"), "output_power_w")
                if output <= 0:
                    raise ValueError("output_power_w must be positive")
                denominator = finite(output + row["total"], "input power estimate")
                eta.update(status="computed", efficiency_percent=(output / denominator) * 100,
                           output_power_w=output, semiconductor_loss_w=row["total"])
            except ValueError as exc:
                eta["reason"] = str(exc)
            efficiency.append(eta)
        for value in result.values():
            if not value["curves"]:
                value["status"] = "missing-data"
                value["reason"] = "No supported recorded curves; missing data is not zero"
        result["loss_breakdown"] = {"title": "Explicit held-current PWM loss estimate", "rows": breakdown,
            "status": "computed" if all(row["status"] == "computed" for row in breakdown) else "partial-or-unsupported"}
        result["efficiency"] = {"title": "Semiconductor-only efficiency estimate", "rows": efficiency,
            "status": "computed" if all(row["status"] == "computed" for row in efficiency) else "partial-or-unsupported",
            "scope": "Pout / (Pout + semiconductor loss). Excludes magnetics, gate drive, auxiliaries and other converter losses."}
        result["config"] = json.loads(json.dumps(config, allow_nan=False))
        result["observation_policy"] = "All recorded slices shown at their actual conditions, independent of requested estimate settings"
        return result

    def _plot_thermal_comparison(
        self, transistors: List[Transistor]
    ) -> Dict[str, Any]:
        """Retain measured samples and validate each component's Foster response."""
        from transistordatabase.thermal import thermal_impedance

        curves, diagnostics = [], []
        time_points = np.logspace(-6, 1, 100)
        for transistor in transistors:
            for component_name, component in (
                ("switch", transistor.switch), ("diode", transistor.diode)
            ):
                foster = component.thermal_foster
                if foster is None:
                    continue
                identity = {"name": transistor.metadata.name, "component": component_name}
                if foster.graph_t_rthjc is not None:
                    try:
                        measured = np.asarray(foster.graph_t_rthjc)
                        if (measured.ndim != 2 or measured.shape[0] != 2
                                or measured.shape[1] == 0
                                or not np.issubdtype(measured.dtype, np.number)
                                or np.issubdtype(measured.dtype, np.complexfloating)
                                or not np.isfinite(measured).all()
                                or (measured < 0).any()
                                or (np.diff(measured[0]) <= 0).any()):
                            raise ValueError("Measured impedance needs finite nonnegative paired samples at increasing times")
                        curves.append({
                            "name": f"{transistor.metadata.name} / {component_name} - measured",
                            "time_data": measured[0].tolist(),
                            "z_th_data": measured[1].tolist(),
                        })
                    except (TypeError, ValueError, OverflowError) as error:
                        diagnostics.append({**identity, "status": "invalid-data", "reason": str(error)})
                if foster.r_th_vector is None:
                    continue
                try:
                    model = {key: getattr(foster, key) for key in (
                        "r_th_vector", "tau_vector", "c_th_vector", "r_th_total"
                    )}
                    impedance = thermal_impedance(model, time_points)
                    curves.append({
                        "name": f"{transistor.metadata.name} / {component_name} - Foster model",
                        "time_data": time_points.tolist(),
                        "z_th_data": impedance.tolist(),
                    })
                except (TypeError, ValueError, OverflowError) as error:
                    diagnostics.append({**identity, "status": "unsupported", "reason": str(error)})
        return {
            "title": "Thermal Impedance Comparison", "xlabel": "Time [s]",
            "ylabel": "Thermal Impedance [K/W]", "curves": curves,
            "log_scale_x": True, "diagnostics": diagnostics,
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
