"""Derive transparent switching-energy points from stored DPT event samples."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

MAX_DPT_ENERGY_POINTS = 50
_REQUIRED_CONDITIONS = ("t_j", "v_supply", "v_g", "v_g_off")
_CONTEXT_FIELDS = (
    "t_j",
    "v_supply",
    "v_g",
    "v_g_off",
    "r_g",
    "r_g_off",
    "source_r_g",
    "source_r_g_off",
    "load_inductance",
    "commutation_inductance",
    "commutation_device",
    "measurement_testbench",
)


class DptEnergyDerivationError(ValueError):
    """A user-correctable error while deriving DPT switching energy."""


def normalize_raw_measurements(raw_measurements: Any) -> list[dict[str, Any]]:
    """Accept the legacy singleton and list raw-measurement shapes."""
    if raw_measurements is None:
        return []
    if isinstance(raw_measurements, dict):
        return [raw_measurements]
    if isinstance(raw_measurements, list) and all(
        isinstance(record, dict) for record in raw_measurements
    ):
        return raw_measurements
    raise DptEnergyDerivationError(
        "Stored raw_measurement_data is not a measurement record or a list of "
        "measurement records."
    )


def source_token(
    raw_measurements: Sequence[dict[str, Any]], measurement_indices: Sequence[int]
) -> str:
    """Hash selected complete source records for stale-preview detection."""
    selected = [[index, raw_measurements[index]] for index in sorted(set(measurement_indices))]
    encoded = json.dumps(
        selected,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_token(request: dict[str, Any]) -> str:
    """Hash the reviewed derivation inputs, excluding server-issued tokens."""
    reviewed = {
        "loss_type": request["loss_type"],
        "points": request["points"],
    }
    for field in (
        "save_action",
        "existing_curve_index",
        "target_curve_count",
        "target_edit_token",
    ):
        if field in request:
            reviewed[field] = request[field]
    encoded = json.dumps(
        reviewed, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_dpt_energy(
    raw_measurements: Sequence[dict[str, Any]],
    loss_type: str,
    points: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Integrate signed power over exact stored samples for one loss curve."""
    if loss_type not in ("e_on", "e_off"):
        raise DptEnergyDerivationError("loss_type must be e_on or e_off.")
    if not 2 <= len(points) <= MAX_DPT_ENERGY_POINTS:
        raise DptEnergyDerivationError(
            f"Select between 2 and {MAX_DPT_ENERGY_POINTS} event windows to make a "
            "current-energy curve."
        )

    event_name = "turn_on" if loss_type == "e_on" else "turn_off"
    vds_field = "dpt_on_vds" if loss_type == "e_on" else "dpt_off_vds"
    id_field = "dpt_on_id" if loss_type == "e_on" else "dpt_off_id"
    resistance_field = "r_g" if loss_type == "e_on" else "r_g_off"
    acquisition_keys: set[tuple[int, int]] = set()
    calculated_points: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []

    for index, point in enumerate(points, start=1):
        measurement_index = point["measurement_index"]
        acquisition_index = point["acquisition_index"]
        identity = (measurement_index, acquisition_index)
        if identity in acquisition_keys:
            raise DptEnergyDerivationError(
                f"Point {index} selects the same {event_name} acquisition as an "
                "earlier point. Select each stored event window once."
            )
        acquisition_keys.add(identity)
        if not 0 <= measurement_index < len(raw_measurements):
            raise DptEnergyDerivationError(
                f"Point {index} refers to a stored measurement that no longer exists."
            )
        if isinstance(acquisition_index, bool) or acquisition_index < 0:
            raise DptEnergyDerivationError(
                f"Point {index} has an invalid acquisition index."
            )

        measurement = raw_measurements[measurement_index]
        context = _measurement_context(measurement, loss_type, index)
        contexts.append(context)
        vds_acquisitions = measurement.get(vds_field)
        id_acquisitions = measurement.get(id_field)
        if not isinstance(vds_acquisitions, list) or not isinstance(id_acquisitions, list):
            raise DptEnergyDerivationError(
                f"Point {index} is missing its stored {event_name} Vds or Id samples."
            )
        if acquisition_index >= len(vds_acquisitions) or acquisition_index >= len(id_acquisitions):
            raise DptEnergyDerivationError(
                f"Point {index} acquisition {acquisition_index + 1} has no paired "
                f"{event_name} Vds and Id data."
            )

        times_v, vds = _sample_array(vds_acquisitions[acquisition_index], "Vds", index)
        times_i, current = _sample_array(id_acquisitions[acquisition_index], "Id", index)
        if len(times_v) != len(times_i) or not np.array_equal(times_v, times_i):
            raise DptEnergyDerivationError(
                f"Point {index} Vds and Id samples are misaligned: their stored "
                "time coordinates must match exactly and in the same order."
            )
        if np.any(np.diff(times_v) < 0):
            raise DptEnergyDerivationError(
                f"Point {index} stored time coordinates are nonmonotonic."
            )
        if len(times_v) < 2:
            raise DptEnergyDerivationError(
                f"Point {index} needs at least two stored event samples to integrate."
            )

        i_x = float(point["i_x"])
        vds_sign = point["vds_sign"]
        id_sign = point["id_sign"]
        if not math.isfinite(i_x) or i_x <= 0:
            raise DptEnergyDerivationError(
                f"Point {index} needs an explicit positive current-axis value i_x."
            )
        if (
            isinstance(vds_sign, bool)
            or isinstance(id_sign, bool)
            or vds_sign not in (-1, 1)
            or id_sign not in (-1, 1)
        ):
            raise DptEnergyDerivationError(
                f"Point {index} must choose +1 or -1 for both Vds and Id polarity."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            signed_power = (vds_sign * vds) * (id_sign * current)
            intervals = np.diff(times_v)
            trapezoids = 0.5 * (signed_power[:-1] + signed_power[1:]) * intervals
            energy = float(np.sum(trapezoids, dtype=np.float64))
        if not np.isfinite(signed_power).all() or not math.isfinite(energy):
            raise DptEnergyDerivationError(
                f"Point {index} produces non-finite signed power or energy."
            )
        if energy <= 0:
            raise DptEnergyDerivationError(
                f"Point {index} has nonpositive signed raw-window energy "
                f"({energy:.9g} J). Review the selected Vds/Id polarity or event window; "
                "energy is never made positive with an absolute value."
            )

        calculated_points.append({
            "measurement_index": measurement_index,
            "acquisition_index": acquisition_index,
            "event": event_name,
            "i_x": i_x,
            "energy_j": energy,
            "start_s": float(times_v[0]),
            "end_s": float(times_v[-1]),
            "sample_count": len(times_v),
            "vds_sign": vds_sign,
            "id_sign": id_sign,
        })

    _require_compatible_contexts(contexts)
    currents = np.asarray([point["i_x"] for point in calculated_points])
    if len(np.unique(currents)) != len(currents):
        raise DptEnergyDerivationError(
            "Every event needs a distinct i_x value; graph_i_e curves require unique "
            "current-axis samples."
        )
    calculated_points.sort(key=lambda point: point["i_x"])
    graph_i_e = [
        [point["i_x"] for point in calculated_points],
        [point["energy_j"] for point in calculated_points],
    ]
    return {
        "loss_type": loss_type,
        "event": event_name,
        "points": calculated_points,
        "graph_i_e": graph_i_e,
        "conditions": contexts[0],
        "integration": {
            "quantity": "raw-window switching energy",
            "formula": "E = sum(0.5 * (p[n] + p[n+1]) * (t[n+1] - t[n]))",
            "power_formula": "p(t) = (Vds_sign * Vds(t)) * (Id_sign * Id(t))",
            "interval": "exact stored first-to-last event sample time",
            "boundary_detection": "none; user-selected imported windows are used as stored",
            "positive_energy_required": True,
        },
    }


def derivation_comment(
    result: dict[str, Any],
    source_sha256: str,
) -> str:
    """Format source intervals and explicit sign choices into curve provenance."""
    source_lines = [
        (
            f"measurement {point['measurement_index'] + 1}, acquisition "
            f"{point['acquisition_index'] + 1} [{point['start_s']:.17g}, "
            f"{point['end_s']:.17g}] s, i_x={point['i_x']:.17g} A, "
            f"Vds sign {point['vds_sign']:+d}, Id sign {point['id_sign']:+d}, "
            f"{point['sample_count']} stored samples"
        )
        for point in result["points"]
    ]
    human_note = (
        "Raw-window DPT energy; signed trapezoidal integration of Vds*Id over exact "
        "stored samples; not standards-normalized. "
        f"Source SHA-256: {source_sha256}. Sources: "
        + "; ".join(source_lines)
    )
    provenance = {
        "version": 1,
        "method": "raw_window_signed_trapezoid",
        "source_sha256": source_sha256,
        "loss_type": result["loss_type"],
        "conditions": result["conditions"],
        "points": result["points"],
    }
    encoded = json.dumps(
        provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return f"{human_note}\nDPT-ENERGY-PROVENANCE: {encoded}"


def _measurement_context(
    measurement: dict[str, Any], loss_type: str, point_number: int
) -> dict[str, Any]:
    """Read required conditions and known importer-unit metadata."""
    if measurement.get("dataset_type") not in ("dpt_u_i", "dpt_u_i_r"):
        raise DptEnergyDerivationError(
            f"Point {point_number} is not a stored dpt_u_i measurement."
        )
    context: dict[str, Any] = {}
    for field in _REQUIRED_CONDITIONS:
        value = _condition_number(measurement, field, point_number, required=True)
        context[field] = value
    for field in ("r_g", "r_g_off"):
        context[field] = _condition_number(measurement, field, point_number)
    if context["v_supply"] <= 0:
        raise DptEnergyDerivationError(
            f"Point {point_number} needs a positive v_supply condition."
        )
    for field in ("r_g", "r_g_off"):
        value = context[field]
        if value is not None and value < 0:
            raise DptEnergyDerivationError(
                f"Point {point_number} has a negative {field} condition."
            )
    resistance_field = "r_g" if loss_type == "e_on" else "r_g_off"
    if context[resistance_field] is None:
        raise DptEnergyDerivationError(
            f"Point {point_number} is missing {resistance_field}; enter the "
            "side-specific gate resistance when importing this DPT measurement."
        )

    provenance = measurement.get("import_provenance")
    has_inductance = any(
        measurement.get(field) is not None
        for field in ("load_inductance", "commutation_inductance")
    )
    verified_raw_import = (
        isinstance(provenance, dict)
        and provenance.get("parser") == "PyLTSpice.RawRead"
    )
    if has_inductance and not verified_raw_import:
        raise DptEnergyDerivationError(
            f"Point {point_number} has load or commutation inductance without "
            "verified PyLTSpice.RawRead importer provenance. Its stored unit cannot "
            "be established, so energy derivation stops rather than guessing."
        )
    # The imported raw-measurement contract stores these fields in µH, while
    # SwitchingLossData stores them in H.
    context["load_inductance"] = _condition_number(
        measurement, "load_inductance", point_number
    )
    context["commutation_inductance"] = _condition_number(
        measurement, "commutation_inductance", point_number
    )
    if verified_raw_import:
        for field in ("load_inductance", "commutation_inductance"):
            if context[field] is not None:
                context[field] *= 1e-6
    for field in ("commutation_device", "measurement_testbench"):
        value = measurement.get(field)
        if value is not None and not isinstance(value, str):
            raise DptEnergyDerivationError(
                f"Point {point_number} has an invalid {field} condition."
            )
        context[field] = value
    context["r_g"] = context[resistance_field]
    context["source_r_g"] = _condition_number(measurement, "r_g", point_number)
    context["source_r_g_off"] = _condition_number(
        measurement, "r_g_off", point_number
    )
    return context


def _condition_number(
    measurement: dict[str, Any],
    field: str,
    point_number: int,
    *,
    required: bool = False,
) -> float | None:
    value = measurement.get(field)
    if value is None:
        if required:
            raise DptEnergyDerivationError(
                f"Point {point_number} is missing required {field} condition metadata."
            )
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DptEnergyDerivationError(
            f"Point {point_number} has ambiguous {field} metadata; enter one "
            "finite scalar condition per stored measurement."
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DptEnergyDerivationError(
            f"Point {point_number} has non-finite {field} metadata."
        )
    return numeric


def _sample_array(
    samples: Any, signal: str, point_number: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        array = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DptEnergyDerivationError(
            f"Point {point_number} stored {signal} data is not numeric."
        ) from error
    if array.ndim != 2 or array.shape[1] != 2:
        raise DptEnergyDerivationError(
            f"Point {point_number} stored {signal} data must contain [time, value] "
            "sample pairs."
        )
    if not np.isfinite(array).all():
        raise DptEnergyDerivationError(
            f"Point {point_number} stored {signal} data contains non-finite time or "
            "value samples."
        )
    return array[:, 0], array[:, 1]


def _require_compatible_contexts(contexts: Sequence[dict[str, Any]]) -> None:
    baseline = contexts[0]
    mismatch_fields = [
        field
        for field in _CONTEXT_FIELDS
        if any(context[field] != baseline[field] for context in contexts[1:])
    ]
    if mismatch_fields:
        raise DptEnergyDerivationError(
            "Selected event windows have incompatible test conditions: "
            + ", ".join(mismatch_fields)
            + ". Keep each voltage, temperature, gate, resistance, test-bench, and "
            "commutation context in a separate curve."
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
