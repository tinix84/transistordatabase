"""Versioned, lossless JSON values for the existing headless core models.

Only explicitly registered core aggregates and dataclasses can be reconstructed;
serialized names never import modules or execute caller-selected constructors.
Legacy flat JSON remains the repository loader's separate compatibility branch.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
import math

import numpy as np

from . import models

SCHEMA = "transistordatabase-core"
VERSION = 2
_DATACLASSES = {name: value for name, value in vars(models).items()
                if isinstance(value, type) and is_dataclass(value)
                and value.__module__ == models.__name__}
_AGGREGATES = {"Transistor": models.Transistor, "Switch": models.Switch, "Diode": models.Diode}


def _encode(value, depth=0):
    if depth > 128:
        raise ValueError("Core JSON exceeds 128 nested values or contains a cycle")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Core JSON cannot contain nonfinite values")
        return value
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biuf" or not np.isfinite(value):
            raise ValueError("Core JSON supports only finite numeric numpy scalars")
        # Keep extended precision without a lossy intermediate Python float.
        return {"kind": "numpy-scalar", "dtype": value.dtype.str, "value": str(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError("Core JSON supports only finite numeric numpy arrays")
        return {"kind": "ndarray", "dtype": value.dtype.str,
                "shape": list(value.shape), "values": _encode(value.tolist(), depth + 1)}
    if isinstance(value, datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if is_dataclass(value) and type(value).__name__ in _DATACLASSES and type(value).__module__ == models.__name__:
        return {"kind": "dataclass", "name": type(value).__name__,
                "fields": {f.name: _encode(getattr(value, f.name), depth + 1) for f in fields(value)}}
    if type(value).__name__ in _AGGREGATES and type(value) is _AGGREGATES[type(value).__name__]:
        template = (models.Transistor(value.metadata, value.electrical_ratings, value.thermal_properties)
                    if isinstance(value, models.Transistor) else type(value)())
        if set(vars(value)) != set(vars(template)):
            raise ValueError("Core aggregate fields do not match this schema version")
        return {"kind": "aggregate", "name": type(value).__name__,
                "fields": {key: _encode(item, depth + 1) for key, item in vars(value).items()}}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Core JSON mapping keys must be strings")
        return {"kind": "mapping", "values": {key: _encode(item, depth + 1) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "values": [_encode(item, depth + 1) for item in value]}
    raise ValueError(f"Unsupported core JSON value type: {type(value).__name__}")


def _decode(value, depth=0):
    if depth > 128:
        raise ValueError("Core JSON exceeds 128 nested values")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Core JSON cannot contain nonfinite values")
        return value
    if not isinstance(value, dict) or "kind" not in value:
        raise ValueError("Malformed tagged core JSON value")
    kind = value["kind"]
    layouts = {"datetime": {"kind", "value"}, "numpy-scalar": {"kind", "dtype", "value"},
               "ndarray": {"kind", "dtype", "shape", "values"},
               "mapping": {"kind", "values"}, "tuple": {"kind", "values"},
               "list": {"kind", "values"}, "dataclass": {"kind", "name", "fields"},
               "aggregate": {"kind", "name", "fields"}}
    if kind not in layouts or set(value) != layouts[kind]:
        raise ValueError("Unknown or malformed core JSON tag")
    if kind == "datetime":
        return datetime.fromisoformat(value["value"])
    if kind in ("numpy-scalar", "ndarray"):
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Unsupported nonnumeric core JSON dtype")
        if kind == "numpy-scalar":
            if dtype.kind == "b":
                if value["value"] not in ("True", "False"):
                    raise ValueError("Invalid numpy boolean scalar")
                result = np.bool_(value["value"] == "True")
            else:
                result = dtype.type(value["value"])
        else:
            shape = value["shape"]
            if (not isinstance(shape, list) or len(shape) > 32
                    or any(type(size) is not int or size < 0 or size > 10_000_000
                           for size in shape)):
                raise ValueError("Core JSON array shape requires bounded nonnegative integer dimensions")
            numbers = _decode(value["values"], depth + 1)
            pending = [numbers]
            while pending:
                number = pending.pop()
                if isinstance(number, list):
                    pending.extend(number)
                    continue
                if dtype.kind == "b":
                    valid = isinstance(number, (bool, np.bool_))
                elif dtype.kind in "iu":
                    valid = isinstance(number, (int, np.integer)) and not isinstance(number, (bool, np.bool_))
                else:
                    valid = isinstance(number, (float, int, np.floating, np.integer)) and not isinstance(number, (bool, np.bool_))
                if not valid:
                    raise ValueError("Core JSON array values do not match their numeric dtype")
            result = np.asarray(numbers, dtype=dtype)
            if result.size == 0 and 0 in shape:
                # Nested empty lists retain dimensions only through the first zero.
                if list(result.shape) != shape[:shape.index(0) + 1]:
                    raise ValueError("Core JSON empty array shape does not match actual values")
                result = result.reshape(shape)
            elif list(result.shape) != shape:
                raise ValueError("Core JSON array shape does not match actual values")
        if not np.all(np.isfinite(result)):
            raise ValueError("Core JSON numeric values must be finite")
        return result
    if kind == "mapping":
        if not isinstance(value["values"], dict):
            raise ValueError("Core JSON mapping requires an object")
        return {key: _decode(item, depth + 1) for key, item in value["values"].items()}
    if kind in ("tuple", "list"):
        if not isinstance(value["values"], list):
            raise ValueError("Core JSON sequence requires an array")
        decoded = [_decode(item, depth + 1) for item in value["values"]]
        return tuple(decoded) if kind == "tuple" else decoded
    name, attributes = value["name"], value["fields"]
    if not isinstance(attributes, dict):
        raise ValueError("Core JSON class fields require an object")
    if kind == "dataclass":
        cls = _DATACLASSES.get(name)
        expected = {f.name for f in fields(cls)} if cls else set()
        if cls is None or set(attributes) != expected:
            raise ValueError("Unknown core dataclass or incompatible field set")
        return cls(**{key: _decode(item, depth + 1) for key, item in attributes.items()})
    if name not in _AGGREGATES:
        raise ValueError("Unknown core aggregate")
    decoded = {key: _decode(item, depth + 1) for key, item in attributes.items()}
    if name == "Transistor":
        required = {"metadata": models.TransistorMetadata, "electrical_ratings": models.ElectricalRatings,
                    "thermal_properties": models.ThermalProperties, "switch": models.Switch, "diode": models.Diode}
        if any(not isinstance(decoded.get(key), cls) for key, cls in required.items()):
            raise ValueError("Core transistor contains incompatible component types")
        result = models.Transistor(decoded["metadata"], decoded["electrical_ratings"], decoded["thermal_properties"])
    else:
        result = _AGGREGATES[name]()
    if set(decoded) != set(vars(result)):
        raise ValueError("Core aggregate fields do not match this schema version")
    vars(result).update(decoded)
    return result


def core_to_json_document(transistor: models.Transistor) -> dict:
    """Serialize all existing core fields, conditions, dates and provenance."""
    if not isinstance(transistor, models.Transistor):
        raise ValueError("Core JSON requires a Transistor aggregate")
    return {"schema": SCHEMA, "version": VERSION, "transistor": _encode(transistor)}


def core_from_json_document(document: dict) -> models.Transistor:
    """Decode only the supported version and explicitly registered core types."""
    if not isinstance(document, dict) or set(document) != {"schema", "version", "transistor"}:
        raise ValueError("Malformed native core JSON document")
    if document["schema"] != SCHEMA or type(document["version"]) is not int or document["version"] not in (1, VERSION):
        raise ValueError("Unsupported native core JSON schema/version")
    try:
        encoded = document["transistor"]
        if document["version"] == 1:
            from copy import deepcopy
            encoded = deepcopy(encoded)
            pending = [encoded]
            additive = {"RawMeasurementData": {"time_s", "voltage_v", "current_a", "gate_v", "evidence"},
                        "SwitchingLossData": {"evidence"}}
            while pending:
                item = pending.pop()
                if isinstance(item, dict):
                    if item.get("kind") == "dataclass" and item.get("name") in additive:
                        for name in additive[item["name"]]:
                            item["fields"].setdefault(name, None)
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
        result = _decode(encoded)
    except (TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed or unsupported native core JSON values") from exc
    if not isinstance(result, models.Transistor):
        raise ValueError("Core JSON root must be a Transistor")
    return result
