"""Evaluate supplied per-device waveforms with explicitly selected native curves.

Metadata limits apply to case provenance and the selected evidence whitelist. They do
not count the bounded numeric waveform/curve arrays or duplicated result snapshots.
Source comments are hashed in chunks and previewed at 500 characters, with a 4 MiB
source-comment bound.
Canonical hashes use sorted compact JSON encoded as UTF-8 with ``ensure_ascii=False``.
All public operations are local, deterministic and independent of GUI/API code.
"""
from __future__ import annotations

from datetime import date, datetime
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Any

import numpy as np

from transistordatabase.condition_tables import require_conditions, samples
from transistordatabase.core.models import Transistor
from transistordatabase.gate_states import hysteretic_gate_states
from transistordatabase.waveform_losses import (
    WaveformResourceLimitError,
    analyze_waveform,
)


MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_SAMPLES = 10_000
MAX_CURVE_POINTS = 10_000
MAX_SELECTED_POINTS = 60_000
MAX_SELECTED_RECORDS = 6
MAX_SEGMENTS = 10_000
MAX_PLOT_RECORDS = 90_000
MAX_METADATA_DEPTH = 8
MAX_METADATA_SCALARS = 128
MAX_METADATA_STRING = 2_000
MAX_COMMENT_PREVIEW = 500
MAX_SOURCE_COMMENT_BYTES = 4 * 1024 * 1024

OPERATION = "native-waveform-v1"
EVALUATOR_VERSION = "native-waveform-v1.0.0"
_DIRECTIONS = ("positive", "negative")
_SLOTS = (
    "conduction.positive",
    "conduction.negative",
    "turn_on.positive",
    "turn_on.negative",
    "turn_off.positive",
    "turn_off.negative",
)
_CHANNEL_CONDITION_KEYS = {"t_j_c", "v_g_v"}
_ENERGY_CONDITION_KEYS = {
    "t_j_c", "v_supply_v", "v_g_v", "v_g_off_v", "r_g_ohm",
    "load_inductance_h", "commutation_inductance_h",
}
_ENERGY_CONTEXT_KEYS = {
    "commutation_device", "convention", "event", "method",
    "measurement_testbench",
}
_WAVEFORM_KEYS = {
    "time_s", "current_a", "gate_v", "device_voltage_v", "bus_voltage_v",
}
_CASE_KEYS = {
    "schema_version", "case_id", "candidate_id", "operating_point_id",
    "position_id", "waveform", "events", "provenance",
}
_SELECTION_KEYS = {
    "schema_version", "input_temperature_K", "conduction", "turn_on", "turn_off",
}
_PROVENANCE_KEYS = {
    "origin_kind", "source_id", "source_sha256", "producer", "producer_version",
    "calibration_notes",
}
_EXCLUDED_PAYLOAD_FIELDS = [
    "original_base64", "companion_log_base64", "dpt_on_vds", "dpt_on_id",
    "dpt_off_vds", "dpt_off_id", "time_s", "voltage_v", "current_a", "gate_v",
    "trace_mapping", "step_values", "expression",
]
_LOSS_SCOPE = {
    "id": "device-conduction-turn-on-turn-off",
    "version": 1,
    "terms": ["conduction", "turn_on", "turn_off"],
}
_CONDITION_MATCHING_POLICY = {
    "numeric_atol": 1e-9,
    "numeric_rtol": 0.0,
    "text": "exact",
    "null": "matches-only-null",
}


class NativeWaveformError(ValueError):
    """A stable, transport-neutral error from native waveform preparation."""

    def __init__(self, code: str, message: str):
        if code not in {
            "invalid-input", "condition-mismatch", "missing-model",
            "unsupported-domain", "resource-limit",
        }:
            raise ValueError("Unknown native waveform error code")
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise NativeWaveformError(code, message)


def _object(value: Any, name: str, required: set[str], optional: set[str] = ()) -> dict:
    if type(value) is not dict:
        _fail("invalid-input", f"{name} must be a JSON object")
    if len(value) > len(required | set(optional)):
        _fail("invalid-input", f"{name} has too many fields")
    if any(not isinstance(key, str) for key in value):
        _fail("invalid-input", f"{name} field names must be text")
    keys = set(value)
    missing = required - keys
    extra = keys - required - set(optional)
    if missing:
        _fail("invalid-input", f"{name} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        _fail("invalid-input", f"{name} has unsupported fields: {', '.join(sorted(extra))}")
    return value


def _text(value: Any, name: str, *, maximum: int = MAX_METADATA_STRING,
          nonblank: bool = True) -> str:
    if not isinstance(value, str):
        _fail("invalid-input", f"{name} must be text")
    if len(value) > maximum:
        _fail("resource-limit", f"{name} exceeds {maximum} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail("invalid-input", f"{name} must contain Unicode scalar values")
    if nonblank and not value.strip():
        _fail("invalid-input", f"{name} must not be blank")
    return value


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str):
        _fail("invalid-input", f"{name} must be text")
    if not 1 <= len(value) <= 200:
        _fail("invalid-input", f"{name} must contain 1 to 200 Unicode scalar values")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail("invalid-input", f"{name} must contain Unicode scalar values")
    return value


def _number(value: Any, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.integer, np.floating)):
        _fail("invalid-input", f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        _fail("invalid-input", f"{name} is outside the finite numeric range")
    if not math.isfinite(result) or (result == 0.0 and value != 0):
        _fail("invalid-input", f"{name} must be finite and representable")
    if positive and result <= 0:
        _fail("invalid-input", f"{name} must be positive")
    if nonnegative and result < 0:
        _fail("invalid-input", f"{name} must be nonnegative")
    return result


def _sequence(value: Any, name: str, *, min_size: int, max_size: int,
              expected_size: int | None = None, boolean: bool = False) -> list:
    """Validate bounded built-in sequences or arrays without consuming iterators."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            _fail("invalid-input", f"{name} must be a one-dimensional vector")
        count = value.shape[0]
        if not boolean and value.dtype.kind not in "iuf":
            _fail("invalid-input", f"{name} must contain real numeric samples")
        raw = value
    elif type(value) in (list, tuple):
        count = len(value)
        raw = value
    else:
        _fail("invalid-input", f"{name} must be a list, tuple, or NumPy vector")
    if count > max_size:
        _fail("resource-limit", f"{name} exceeds {max_size} samples")
    if count < min_size:
        _fail("invalid-input", f"{name} must contain at least {min_size} samples")
    if expected_size is not None and count != expected_size:
        _fail("invalid-input", f"{name} must align with time_s")
    if boolean:
        if isinstance(value, np.ndarray):
            if value.dtype.kind != "b":
                _fail("invalid-input", f"{name} must contain actual booleans")
            return value.tolist()
        if any(type(item) is not bool for item in value):
            _fail("invalid-input", f"{name} must contain actual booleans")
        return list(value)
    if type(value) in (list, tuple):
        for item in value:
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (Real, np.integer, np.floating)
            ):
                _fail("invalid-input", f"{name} must contain real numeric samples")
    try:
        converted = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        _fail("invalid-input", f"{name} has values outside the float64 range")
    if not np.isfinite(converted).all():
        _fail("invalid-input", f"{name} must contain finite values")
    original = np.asarray(raw)
    if np.any((original != 0) & (converted == 0)):
        _fail("invalid-input", f"{name} contains nonzero values below float64 precision")
    return converted.tolist()


def _graph(value: Any, label: str, *, x_row: int, y_row: int,
           selected_points: list[int]) -> tuple[list[float], list[float]]:
    """Validate a two-row native graph before converting it to float64."""
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[0] != 2:
            _fail("unsupported-domain", f"{label} must have two aligned rows")
        points = value.shape[1]
        if points > MAX_CURVE_POINTS:
            _fail("resource-limit", f"{label} exceeds {MAX_CURVE_POINTS} points")
        if points < 2:
            _fail("unsupported-domain", f"{label} needs at least two points")
        if points > MAX_SELECTED_POINTS - selected_points[0]:
            _fail("resource-limit", f"Selected curves exceed {MAX_SELECTED_POINTS} points")
        if value.dtype.kind not in "iuf":
            _fail("unsupported-domain", f"{label} must contain real numeric values")
        raw = value
    elif type(value) in (list, tuple) and len(value) == 2:
        first, second = value
        if type(first) not in (list, tuple, np.ndarray) or type(second) not in (
            list, tuple, np.ndarray
        ):
            _fail("unsupported-domain", f"{label} must contain two numeric rows")
        if isinstance(first, np.ndarray) and first.ndim != 1:
            _fail("unsupported-domain", f"{label} rows must be one-dimensional")
        if isinstance(second, np.ndarray) and second.ndim != 1:
            _fail("unsupported-domain", f"{label} rows must be one-dimensional")
        if len(first) != len(second):
            _fail("unsupported-domain", f"{label} rows must have equal length")
        points = len(first)
        if points > MAX_CURVE_POINTS:
            _fail("resource-limit", f"{label} exceeds {MAX_CURVE_POINTS} points")
        if points < 2:
            _fail("unsupported-domain", f"{label} needs at least two points")
        if points > MAX_SELECTED_POINTS - selected_points[0]:
            _fail("resource-limit", f"Selected curves exceed {MAX_SELECTED_POINTS} points")
        raw = [first, second]
        for item in (first, second):
            if isinstance(item, np.ndarray):
                if item.dtype.kind not in "iuf":
                    _fail("unsupported-domain", f"{label} must contain real numeric values")
            elif any(isinstance(v, (bool, np.bool_)) or not isinstance(
                v, (Real, np.integer, np.floating)
            ) for v in item):
                _fail("unsupported-domain", f"{label} must contain real numeric values")
    else:
        _fail("unsupported-domain", f"{label} must be a two-row numeric graph")
    try:
        converted = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        _fail("unsupported-domain", f"{label} has values outside the float64 range")
    if converted.shape != (2, points) or not np.isfinite(converted).all():
        _fail("unsupported-domain", f"{label} must contain finite aligned samples")
    original = np.asarray(raw)
    if np.any((original != 0) & (converted == 0)):
        _fail("unsupported-domain", f"{label} contains nonzero values below float64 precision")
    if np.any(converted < 0):
        _fail("unsupported-domain", f"{label} values must be nonnegative")
    try:
        x, y = samples(converted[x_row], converted[y_row], label)
    except ValueError as exc:
        _fail("unsupported-domain", str(exc))
    return x.tolist(), y.tolist()


def _canonical_hash(value: Any, limit: int) -> str:
    """Hash canonical UTF-8 JSON while enforcing a byte budget incrementally."""
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    digest = sha256()
    size = 0
    try:
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            size += len(encoded)
            if size > limit:
                _fail("resource-limit", f"Encoded data exceeds the {limit}-byte limit")
            digest.update(encoded)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        if isinstance(exc, NativeWaveformError):
            raise
        _fail("invalid-input", f"Data cannot be encoded as finite JSON: {exc}")
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return _canonical_hash(value, MAX_OUTPUT_BYTES)


def _check_metadata(value: Any, *, name: str, depth: int = 0,
                    counter: list[int] | None = None) -> int:
    if counter is None:
        counter = [0]
    if depth > MAX_METADATA_DEPTH:
        _fail("resource-limit", f"{name} exceeds metadata depth {MAX_METADATA_DEPTH}")
    if isinstance(value, dict):
        for key, item in value.items():
            _text(key, f"{name} key", nonblank=False)
            _check_metadata(item, name=name, depth=depth + 1, counter=counter)
    elif type(value) in (list, tuple):
        for item in value:
            _check_metadata(item, name=name, depth=depth + 1, counter=counter)
    elif value is None or isinstance(value, (str, bool, int, float, np.generic)):
        if isinstance(value, str):
            _text(value, name, nonblank=False)
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            _fail("invalid-input", f"{name} must contain finite numbers")
        counter[0] += 1
        if counter[0] > MAX_METADATA_SCALARS:
            _fail("resource-limit", f"Metadata exceeds {MAX_METADATA_SCALARS} scalar fields")
    else:
        _fail("invalid-input", f"{name} contains unsupported metadata type {type(value).__name__}")
    return counter[0]


def _hash_source_comment(comment: Any, name: str) -> tuple[str | None, str | None]:
    if comment is None:
        return None, None
    if not isinstance(comment, str):
        _fail("invalid-input", f"{name} must be text")
    digest = sha256()
    byte_count = 0
    try:
        for offset in range(0, len(comment), 256):
            encoded = comment[offset:offset + 256].encode("utf-8")
            byte_count += len(encoded)
            if byte_count > MAX_SOURCE_COMMENT_BYTES:
                _fail("resource-limit", f"{name} exceeds {MAX_SOURCE_COMMENT_BYTES} bytes")
            digest.update(encoded)
    except UnicodeEncodeError:
        _fail("invalid-input", f"{name} must contain Unicode scalar values")
    return digest.hexdigest(), comment[:MAX_COMMENT_PREVIEW]


def _record_context(record: Any) -> dict[str, str | None]:
    evidence = getattr(record, "evidence", None)
    result = {
        "commutation_device": getattr(record, "commutation_device", None),
        "convention": getattr(evidence, "convention", None) if evidence is not None else None,
        "event": getattr(evidence, "event", None) if evidence is not None else None,
        "method": getattr(evidence, "method", None) if evidence is not None else None,
        "measurement_testbench": getattr(record, "measurement_testbench", None),
    }
    for key, value in result.items():
        if value is not None:
            result[key] = _text(value, f"recorded context {key}")
    return result


def _record_evidence(record: Any) -> dict[str, Any]:
    evidence = getattr(record, "evidence", None)
    comment_hash, comment_preview = _hash_source_comment(
        getattr(record, "comment", None), "source comment",
    )
    result: dict[str, Any] = {
        "filename": None,
        "sha256": None,
        "parser": None,
        "parser_version": None,
        "convention": None,
        "event": None,
        "method": None,
        "measurement_date": None,
        "measurement_testbench": None,
        "commutation_device": None,
        "load_inductance_h": None,
        "commutation_inductance_h": None,
        "comment_sha256": comment_hash,
        "comment_preview": comment_preview,
    }
    if evidence is not None:
        for key in ("filename", "sha256", "parser", "parser_version", "convention", "event", "method"):
            value = getattr(evidence, key, None)
            if value is not None:
                result[key] = _text(value, f"evidence {key}")
    value = getattr(record, "measurement_date", None)
    if value is not None:
        if not isinstance(value, (datetime, date)):
            _fail("invalid-input", "record measurement_date must be a date")
        result["measurement_date"] = value.isoformat()
    for key, attr in (
        ("measurement_testbench", "measurement_testbench"),
        ("commutation_device", "commutation_device"),
    ):
        value = getattr(record, attr, None)
        if value is not None:
            result[key] = _text(value, f"record {attr}")
    for key, attr in (
        ("load_inductance_h", "load_inductance"),
        ("commutation_inductance_h", "commutation_inductance"),
    ):
        value = getattr(record, attr, None)
        if value is not None:
            result[key] = _number(value, f"record {attr}")
    return result


def _native_record(device: Transistor, component: str, family: str, index: int) -> Any:
    if not isinstance(component, str) or component not in {"switch", "diode"}:
        _fail("invalid-input", "component must be switch or diode")
    if family == "channel_data":
        record_list = getattr(getattr(device, component), family, None)
    elif family in {"e_on_data", "e_off_data"} and component == "switch":
        record_list = getattr(device.switch, family, None)
    else:
        _fail("unsupported-domain", f"{component}.{family} is outside the supported loss boundary")
    if type(record_list) not in (list, tuple):
        _fail("invalid-input", f"{component}.{family} must be a stored record list")
    if not 0 <= index < len(record_list):
        _fail("missing-model", f"The selected {component}.{family} record does not exist")
    return record_list[index]


def _numeric_conditions(record: Any, *, energy: bool) -> dict[str, float | None]:
    if energy:
        raw = {
            "t_j_c": getattr(record, "t_j", None),
            "v_supply_v": getattr(record, "v_supply", None),
            "v_g_v": getattr(record, "v_g", None),
            "v_g_off_v": getattr(record, "v_g_off", None),
            "r_g_ohm": getattr(record, "r_g", None),
            "load_inductance_h": getattr(record, "load_inductance", None),
            "commutation_inductance_h": getattr(record, "commutation_inductance", None),
        }
    else:
        raw = {
            "t_j_c": getattr(record, "t_j", None),
            "v_g_v": getattr(record, "v_g", None),
        }
    conditions: dict[str, float | None] = {}
    for key, value in raw.items():
        if value is None:
            conditions[key] = None
        else:
            conditions[key] = _number(value, f"recorded condition {key}")
    return conditions


def _requested_conditions(selector: dict, *, energy: bool,
                          imposed_c: float) -> dict[str, float | None]:
    expected_keys = _ENERGY_CONDITION_KEYS if energy else _CHANNEL_CONDITION_KEYS
    raw = _object(selector.get("requested_conditions"), "requested_conditions", expected_keys)
    conditions: dict[str, float | None] = {}
    for key in expected_keys:
        value = raw[key]
        if value is None:
            conditions[key] = None
        else:
            conditions[key] = _number(value, f"requested_conditions.{key}")
    if conditions["t_j_c"] is None:
        _fail("condition-mismatch", "Requested curve temperature must be known")
    if not math.isclose(conditions["t_j_c"], imposed_c, rel_tol=0.0, abs_tol=1e-9):
        _fail("condition-mismatch", "Requested curve temperature does not match imposed temperature")
    return conditions


def _requested_context(selector: dict) -> dict[str, str | None]:
    raw = _object(selector.get("requested_context"), "requested_context", _ENERGY_CONTEXT_KEYS)
    result = {}
    for key in _ENERGY_CONTEXT_KEYS:
        value = raw[key]
        result[key] = None if value is None else _text(value, f"requested_context.{key}")
    return result


def _match_context(recorded: dict, requested: dict) -> None:
    if recorded != requested:
        _fail("condition-mismatch", "Requested energy context does not match the native record")


def _selector(device: Transistor, selector: Any, *, term: str, direction: str,
              imposed_c: float, selected_points: list[int]) -> tuple[dict, dict, dict]:
    is_energy = term in {"turn_on", "turn_off"}
    required = {"component", "record_index", "requested_conditions"}
    optional = {"requested_context"} if is_energy else set()
    item = _object(selector, f"{term}.{direction} selector", required, optional)
    component = item["component"]
    allowed = {"switch", "diode"} if term == "conduction" else {"switch"}
    if not isinstance(component, str) or component not in allowed:
        _fail("unsupported-domain", f"{term} does not support component {component!r}")
    index_value = item["record_index"]
    if type(index_value) is not int or index_value < 0:
        _fail("invalid-input", f"{term}.{direction}.record_index must be a nonnegative integer")
    family = "channel_data" if term == "conduction" else ("e_on_data" if term == "turn_on" else "e_off_data")
    record = _native_record(device, component, family, index_value)
    recorded = _numeric_conditions(record, energy=is_energy)
    requested = _requested_conditions(item, energy=is_energy, imposed_c=imposed_c)
    if is_energy:
        requested_context = _requested_context(item)
        recorded_context = _record_context(record)
        _match_context(recorded_context, requested_context)
    else:
        requested_context = recorded_context = None
    try:
        require_conditions(recorded, requested)
    except ValueError:
        _fail("condition-mismatch", f"Requested {term}.{direction} conditions do not match the native record")

    if is_energy:
        representation = getattr(record, "dataset_type", None)
        if representation != "graph_i_e":
            _fail("unsupported-domain", f"{family} supports graph_i_e only, got {representation!r}")
        graph = getattr(record, "graph_i_e", None)
        if graph is None:
            _fail("missing-model", f"{component}.{family}[{index_value}] has no graph_i_e data")
        current, energy = _graph(
            graph, f"{component}.{family}[{index_value}].graph_i_e", x_row=0, y_row=1,
            selected_points=selected_points,
        )
        table = {"current_a": current, "energy_j": energy}
    else:
        representation = "graph_v_i"
        graph = getattr(record, "graph_v_i", None)
        if graph is None:
            _fail("missing-model", f"{component}.{family}[{index_value}] has no graph_v_i data")
        current, drop = _graph(
            graph, f"{component}.{family}[{index_value}].graph_v_i", x_row=1, y_row=0,
            selected_points=selected_points,
        )
        table = {"current_a": current, "drop_v": drop}

    native_snapshot = {
        "component": component,
        "family": family,
        "record_index": index_value,
        "representation": representation,
        "numeric_snapshot": table,
        "recorded_conditions": recorded,
        "recorded_context": recorded_context,
    }
    snapshot_sha = _digest(native_snapshot)
    evidence = _record_evidence(record)
    source_record = {
        **native_snapshot,
        "slot": f"{term}.{direction}",
        "requested_conditions": requested,
        "requested_context": requested_context,
        "available_evidence": evidence,
        "excluded_payload_fields": list(_EXCLUDED_PAYLOAD_FIELDS),
        "snapshot_sha256": snapshot_sha,
    }
    source_id = f"{component}.{family}[{index_value}]#{snapshot_sha}"
    engine_model = {
        "mode": "recorded",
        **table,
        "recorded_conditions": recorded,
        "requested_conditions": requested,
        "source": source_id,
    }
    normalized_selector = {
        "component": component,
        "record_index": index_value,
        "requested_conditions": requested,
    }
    if is_energy:
        normalized_selector["requested_context"] = requested_context
    selected_points[0] += len(current)
    return engine_model, source_record, normalized_selector


def _term(device: Transistor, value: Any, *, term: str, imposed_c: float,
          selected_points: list[int], selected_records: list[int]) -> tuple[dict, dict, dict]:
    item = _object(value, term, {"mode"}, {"reason", "positive", "negative"})
    mode = item["mode"]
    if mode == "omitted":
        if set(item) != {"mode", "reason"}:
            _fail("invalid-input", f"omitted {term} must contain only mode and reason")
        reason = _text(item["reason"], f"{term}.reason")
        return {"omit": True, "omission_reason": reason}, {
            "mode": "omitted", "reason": reason,
        }, {direction: None for direction in _DIRECTIONS}
    if mode != "recorded" or set(item) != {"mode", "positive", "negative"}:
        _fail("invalid-input", f"{term} must be a recorded term or an explicit omission")
    if term == "conduction" and item["positive"] is None and item["negative"] is None:
        _fail("missing-model", "Computed conduction needs at least one selected direction")
    if term != "conduction" and item["positive"] is None and item["negative"] is None:
        _fail("missing-model", f"Computed {term} needs at least one selected direction")
    engine: dict[str, Any] = {}
    selection: dict[str, Any] = {"mode": "recorded"}
    sources: dict[str, Any] = {}
    for direction in _DIRECTIONS:
        branch = item[direction]
        if branch is None:
            engine[direction] = None
            selection[direction] = None
            sources[direction] = None
            continue
        selected_records[0] += 1
        if selected_records[0] > MAX_SELECTED_RECORDS:
            _fail("resource-limit", f"At most {MAX_SELECTED_RECORDS} directional records are allowed")
        model, source_record, normalized_selector = _selector(
            device, branch, term=term, direction=direction, imposed_c=imposed_c,
            selected_points=selected_points,
        )
        engine[direction] = model
        selection[direction] = normalized_selector
        sources[direction] = source_record
    return engine, selection, sources


def _plain_copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if type(value) is dict:
        return {key: _plain_copy(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain_copy(item) for item in value]
    return value


def _case_and_selection(device: Any, selection: Any, case: Any) -> dict:
    if not isinstance(device, Transistor):
        _fail("invalid-input", "device must be an already loaded core Transistor")
    device_name = _text(device.metadata.name, "device.metadata.name")
    case_obj = _object(
        case, "case", _CASE_KEYS,
    )
    selection_obj = _object(selection, "selection", _SELECTION_KEYS)
    for name in ("case_id", "candidate_id", "operating_point_id", "position_id"):
        _identity(case_obj[name], name)
    if type(case_obj["schema_version"]) is not int or case_obj["schema_version"] != 1:
        _fail("invalid-input", "case.schema_version must be integer 1")
    if type(selection_obj["schema_version"]) is not int or selection_obj["schema_version"] != 1:
        _fail("invalid-input", "selection.schema_version must be integer 1")
    provenance = _object(
        case_obj["provenance"], "provenance", {"origin_kind", "source_id"},
        _PROVENANCE_KEYS - {"origin_kind", "source_id"},
    )
    origin = provenance["origin_kind"]
    if not isinstance(origin, str) or origin not in {"simulated", "measured", "declared"}:
        _fail("invalid-input", "provenance.origin_kind must be simulated, measured, or declared")
    _identity(provenance["source_id"], "provenance.source_id")
    normalized_provenance = {key: None for key in _PROVENANCE_KEYS}
    normalized_provenance.update(provenance)
    for key in ("producer", "producer_version", "calibration_notes"):
        if normalized_provenance[key] is not None:
            _text(normalized_provenance[key], f"provenance.{key}", nonblank=False)
    source_sha = normalized_provenance["source_sha256"]
    if source_sha is not None and (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in source_sha)
    ):
        _fail("invalid-input", "provenance.source_sha256 must contain 64 hexadecimal characters")
    if source_sha is not None:
        normalized_provenance["source_sha256"] = source_sha.lower()

    wave = _object(
        case_obj["waveform"], "waveform", {"time_s", "current_a"},
        _WAVEFORM_KEYS - {"time_s", "current_a"},
    )
    time = _sequence(wave["time_s"], "time_s", min_size=2, max_size=MAX_SAMPLES)
    current = _sequence(
        wave["current_a"], "current_a", min_size=2, max_size=MAX_SAMPLES,
        expected_size=len(time),
    )
    if any(right <= left for left, right in zip(time, time[1:], strict=False)):
        _fail("invalid-input", "time_s must increase strictly")
    normalized_wave = {
        "time_s": time,
        "current_a": current,
    }
    for key in ("gate_v", "device_voltage_v"):
        value = wave.get(key)
        normalized_wave[key] = None if value is None else _sequence(
            value, key, min_size=2, max_size=MAX_SAMPLES, expected_size=len(time),
        )
    bus = wave.get("bus_voltage_v")
    if bus is None:
        normalized_wave["bus_voltage_v"] = None
    elif isinstance(bus, (bool, np.bool_)):
        _fail("invalid-input", "bus_voltage_v must be a real nonnegative number or vector")
    elif isinstance(bus, (Real, np.integer, np.floating)):
        scalar_bus = _number(bus, "bus_voltage_v", nonnegative=True)
        normalized_wave["bus_voltage_v"] = [scalar_bus] * len(time)
    else:
        normalized_wave["bus_voltage_v"] = _sequence(
            bus, "bus_voltage_v", min_size=2, max_size=MAX_SAMPLES,
            expected_size=len(time),
        )
        if any(value < 0 for value in normalized_wave["bus_voltage_v"]):
            _fail("invalid-input", "bus_voltage_v must be nonnegative")

    events = _normalize_events(case_obj["events"], normalized_wave)
    normalized_case = {
        "schema_version": 1,
        "case_id": case_obj["case_id"],
        "candidate_id": case_obj["candidate_id"],
        "operating_point_id": case_obj["operating_point_id"],
        "position_id": case_obj["position_id"],
        "waveform": normalized_wave,
        "events": events,
        "provenance": normalized_provenance,
    }
    input_temp_k = _number(selection_obj["input_temperature_K"], "input_temperature_K", positive=True)
    imposed_c = input_temp_k - 273.15
    if not math.isfinite(imposed_c):
        _fail("invalid-input", "input_temperature_K cannot be converted to degrees Celsius")
    selected_points = [0]
    selected_records = [0]
    source_slots: dict[str, Any] = {}
    engine_terms: dict[str, Any] = {}
    normalized_terms: dict[str, Any] = {}
    for term in ("conduction", "turn_on", "turn_off"):
        engine_term, normalized_term, sources = _term(
            device, selection_obj[term], term=term, imposed_c=imposed_c,
            selected_points=selected_points, selected_records=selected_records,
        )
        engine_terms[term] = engine_term
        normalized_terms[term] = normalized_term
        for direction in _DIRECTIONS:
            source_slots[f"{term}.{direction}"] = sources[direction]
    if events["mode"] == "omitted" and any(
        normalized_terms[term]["mode"] != "omitted" for term in ("turn_on", "turn_off")
    ):
        _fail("invalid-input", "Omitted event detection requires both switching terms omitted")
    normalized_selection = {
        "schema_version": 1,
        "input_temperature_K": input_temp_k,
        **normalized_terms,
    }

    evidence_values = [normalized_provenance]
    for source in source_slots.values():
        if source is not None:
            evidence_values.append(source["available_evidence"])
    counter = [0]
    for value in evidence_values:
        _check_metadata(value, name="metadata", counter=counter)

    engine_request = {
        "waveform": _plain_copy(normalized_wave),
        "events": _plain_copy(events),
        "conduction": engine_terms["conduction"],
        "switching": {
            "turn_on": engine_terms["turn_on"],
            "turn_off": engine_terms["turn_off"],
        },
        "provenance": _plain_copy(normalized_provenance),
    }
    native_snapshot = {"case": normalized_case, "selection": normalized_selection}
    request_sha = _canonical_hash(native_snapshot, MAX_REQUEST_BYTES)
    return {
        "engine_request": engine_request,
        "normalized_case": normalized_case,
        "normalized_selection": normalized_selection,
        "native_request_snapshot": native_snapshot,
        "request_sha256": request_sha,
        "source_slots": source_slots,
        "imposed_c": imposed_c,
        "input_temp_k": input_temp_k,
        "device_name": device_name,
    }


def _normalize_events(value: Any, waveform: dict) -> dict:
    if type(value) is not dict or "mode" not in value:
        _fail("invalid-input", "events must be an object with a mode")
    mode = value["mode"]
    count = len(waveform["time_s"]) - 1
    if mode == "masks":
        item = _object(value, "events", {"mode", "turn_on", "turn_off"})
        on = _sequence(item["turn_on"], "events.turn_on", min_size=1,
                       max_size=MAX_SAMPLES - 1, expected_size=count, boolean=True)
        off = _sequence(item["turn_off"], "events.turn_off", min_size=1,
                        max_size=MAX_SAMPLES - 1, expected_size=count, boolean=True)
        if any(a and b for a, b in zip(on, off, strict=True)):
            _fail("invalid-input", "Turn-on and turn-off event masks cannot overlap")
        return {"mode": "masks", "turn_on": on, "turn_off": off}
    if mode == "gate":
        item = _object(value, "events", {"mode", "threshold_v"})
        if waveform["gate_v"] is None:
            _fail("invalid-input", "Gate event detection requires waveform.gate_v")
        return {"mode": "gate", "threshold_v": _number(item["threshold_v"], "threshold_v")}
    if mode == "hysteresis":
        item = _object(value, "events", {"mode", "digitizer"})
        if waveform["gate_v"] is None:
            _fail("invalid-input", "Hysteresis event detection requires waveform.gate_v")
        digitizer = _object(item["digitizer"], "events.digitizer", {"thresholds"}, {"initial_state"})
        thresholds = digitizer["thresholds"]
        if type(thresholds) is not dict:
            _fail("invalid-input", "events.digitizer.thresholds must be an object")
        scale = thresholds.get("scale")
        if scale == "absolute":
            thresholds = _object(thresholds, "hysteresis thresholds", {"scale", "low_v", "high_v"})
            normalized_thresholds = {
                "scale": scale,
                "low_v": _number(thresholds["low_v"], "low_v"),
                "high_v": _number(thresholds["high_v"], "high_v"),
            }
        elif scale == "relative":
            thresholds = _object(
                thresholds, "hysteresis thresholds",
                {"scale", "low_fraction", "high_fraction", "minimum_v", "maximum_v"},
            )
            normalized_thresholds = {"scale": scale}
            for name in ("low_fraction", "high_fraction", "minimum_v", "maximum_v"):
                normalized_thresholds[name] = _number(thresholds[name], name)
        else:
            _fail("invalid-input", "Hysteresis scale must be absolute or relative")
        if "initial_state" in digitizer and digitizer["initial_state"] is not None and type(digitizer["initial_state"]) is not bool:
            _fail("invalid-input", "initial_state must be an actual boolean or null")
        normalized_digitizer = {"thresholds": normalized_thresholds}
        normalized_digitizer["initial_state"] = digitizer.get("initial_state")
        try:
            hysteretic_gate_states(waveform["gate_v"], normalized_digitizer)
        except ValueError as exc:
            _fail("invalid-input", str(exc))
        return {"mode": "hysteresis", "digitizer": normalized_digitizer}
    if mode == "omitted":
        item = _object(value, "events", {"mode", "omission_reason"})
        return {"mode": "omitted", "omission_reason": _text(item["omission_reason"], "omission_reason")}
    _fail("invalid-input", "Choose masks, gate, hysteresis, or omitted event detection")


def prepare_native_waveform_request(device: Transistor, selection: dict, case: dict) -> dict:
    """Return the normalized request for :func:`analyze_waveform`.

    The returned dictionary is an independent plain-data snapshot. It uses the
    analyzer's existing piecewise-linear and pre-edge event conventions.
    """
    return _case_and_selection(device, selection, case)["engine_request"]


def _error_from_engine(exc: Exception) -> NativeWaveformError:
    if isinstance(exc, WaveformResourceLimitError):
        return NativeWaveformError("resource-limit", str(exc))
    message = str(exc)
    lowered = message.lower()
    if "requested conditions do not match" in lowered:
        code = "condition-mismatch"
    elif "missing" in lowered and "model" in lowered:
        code = "missing-model"
    elif "supports" in lowered or "outside data range" in lowered or "out of range" in lowered:
        code = "unsupported-domain"
    else:
        code = "invalid-input"
    return NativeWaveformError(code, message)


def evaluate_native_waveform(device: Transistor, selection: dict, case: dict) -> dict:
    """Evaluate one supplied waveform and return a reproducible native record."""
    plan = _case_and_selection(device, selection, case)
    request_sha = plan["request_sha256"]
    device_name = plan["device_name"]
    waveform_sha = _canonical_hash({
        "waveform": plan["normalized_case"]["waveform"],
        "interval_convention": "duration=last_time-first_time; no periodic closure",
    }, MAX_OUTPUT_BYTES)
    event_sha = _canonical_hash({
        "events": plan["normalized_case"]["events"],
        "sampling_convention": "pre-edge sample i for bracket [i,i+1]; no initial or wrapped event",
    }, MAX_OUTPUT_BYTES)
    source_records = []
    for slot_name in _SLOTS:
        source = plan["source_slots"][slot_name]
        if source is None:
            source_records.append(None)
        else:
            source_records.append(source)
    model_sha = _digest(source_records)
    try:
        analysis = analyze_waveform(plan["engine_request"])
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise _error_from_engine(exc) from exc
    _assert_result_bounded(analysis)
    result = {
        "schema_version": 1,
        "operation": OPERATION,
        "evaluator_version": EVALUATOR_VERSION,
        "case_id": plan["normalized_case"]["case_id"],
        "candidate_id": plan["normalized_case"]["candidate_id"],
        "operating_point_id": plan["normalized_case"]["operating_point_id"],
        "position_id": plan["normalized_case"]["position_id"],
        "device_name": device_name,
        "request_sha256": request_sha,
        "waveform_sha256": waveform_sha,
        "event_policy_sha256": event_sha,
        "selected_model_sha256": model_sha,
        "condition_matching_policy": _plain_copy(_CONDITION_MATCHING_POLICY),
        "evaluation_record_sha256": None,
        "evaluation_id": None,
        "temperature": {
            "input_K": plan["input_temp_k"],
            "imposed_c": plan["imposed_c"],
            "status": "imposed",
        },
        "loss_scope": _plain_copy(_LOSS_SCOPE),
        "native_request_snapshot": plan["native_request_snapshot"],
        "source_records": source_records,
        "excluded_payload_fields": list(_EXCLUDED_PAYLOAD_FIELDS),
        "analysis": analysis,
    }
    digest_payload = {
        key: value for key, value in result.items()
        if key not in {"evaluation_record_sha256", "evaluation_id"}
    }
    result_sha = _digest(digest_payload)
    result["evaluation_record_sha256"] = result_sha
    result["evaluation_id"] = f"{OPERATION}:{result_sha}"
    _canonical_hash(result, MAX_OUTPUT_BYTES)
    return result


def _assert_result_bounded(value: Any) -> None:
    if not isinstance(value, dict):
        _fail("invalid-input", "Waveform analyzer returned a non-object result")
    segments = value.get("power_segments", [])
    plot = value.get("conduction_plot", [])
    events = value.get("events", [])
    if len(segments) > MAX_SEGMENTS:
        _fail("resource-limit", f"Conduction exceeds {MAX_SEGMENTS} exact segments")
    if len(plot) > MAX_PLOT_RECORDS:
        _fail("resource-limit", f"Conduction exceeds {MAX_PLOT_RECORDS} plot records")
    if len(events) > MAX_SAMPLES - 1:
        _fail("resource-limit", "Event count exceeds the waveform bracket limit")

