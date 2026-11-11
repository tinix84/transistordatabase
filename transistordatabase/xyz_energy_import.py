"""Import declared energy triples without interpolation or duplicate reduction."""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import json
import math
import re

from transistordatabase.energy_maps import NUMERIC_CONDITIONS, _conditions

MAX_RAW_BYTES = 4_000_000
MAX_ROWS = 100_000
MAX_CURRENT_AXIS = 256
MAX_VOLTAGE_AXIS = 128
MAX_CELLS = 16_384
MAX_RESPONSE_BYTES = 24_000_000
METHOD_VERSION = "xyz-energy-map-v1"
LIMITS = dict(raw_bytes=MAX_RAW_BYTES, source_rows=MAX_ROWS,
              current_axis=MAX_CURRENT_AXIS, voltage_axis=MAX_VOLTAGE_AXIS,
              cells=MAX_CELLS, response_bytes=MAX_RESPONSE_BYTES,
              numeric_token_characters=128, decimal_digits=80,
              decimal_exponent_magnitude=400)
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
POWERS = {"current": {"A": 0, "mA": -3},
          "voltage": {"V": 0, "kV": 3},
          "energy": {"J": 0, "mJ": -3, "µJ": -6, "uJ": -6}}


class XyzEnergyLimitError(ValueError):
    """A source or result exceeds its declared resource budget."""


def _encoded(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _text(value, name, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"Declare a nonblank {name} within {maximum} characters")
    value.encode("utf-8")


def _validate_options(options):
    required = {"source_name", "input_kind", "source", "source_kind",
                "energy_kind", "conditions", "options"}
    if not isinstance(options, dict) or set(options) != required:
        raise ValueError("Declare source metadata, conditions, and import options")
    for name, maximum in (("source_name", 500), ("source", 2000)):
        _text(options[name], name, maximum)
    for name, choices in (
        ("input_kind", {"file", "paste"}),
        ("source_kind", {"measured", "datasheet", "simulated", "manual", "unknown"}),
        ("energy_kind", {"e_on", "e_off", "e_rr"}),
    ):
        if not isinstance(options[name], str) or options[name] not in choices:
            raise ValueError(f"Declare a supported {name}")
    if not isinstance(options["conditions"], dict):
        raise ValueError("Declare every fixed condition in an object")
    _conditions(options["conditions"])
    for name in ("commutation_device", "commutation_source", "energy_convention"):
        if options["conditions"][name] is not None:
            _text(options["conditions"][name], name,
                  1000 if name == "commutation_device" else 2000)
    syntax = options["options"]
    if not isinstance(syntax, dict) or set(syntax) != {
            "delimiter", "decimal", "has_header", "columns", "units",
            "missing_energy_tokens"}:
        raise ValueError("Declare every import syntax option")
    if (not isinstance(syntax["delimiter"], str)
            or syntax["delimiter"] not in {"csv", "tsv", "whitespace"}):
        raise ValueError("Select csv, tsv, or whitespace delimiters")
    if syntax["decimal"] != "." or type(syntax["has_header"]) is not bool:
        raise ValueError("Use a decimal point and an explicit header selection")
    columns = syntax["columns"]
    if (not isinstance(columns, dict) or set(columns) != set(POWERS)
            or any(type(v) is not int for v in columns.values())
            or set(columns.values()) != {0, 1, 2}):
        raise ValueError("Map current, voltage, and energy to three different columns")
    units = syntax["units"]
    if (not isinstance(units, dict) or set(units) != set(POWERS)
            or any(not isinstance(units[name], str) or units[name] not in POWERS[name]
                   for name in POWERS)):
        raise ValueError("Select explicit supported units for all three columns")
    missing = syntax["missing_energy_tokens"]
    if (not isinstance(missing, list) or len(missing) > 20
            or any(not isinstance(v, str) or not v.strip() or len(v) > 128
                   or v != v.strip() for v in missing)
            or len(set(missing)) != len(missing)):
        raise ValueError("Declare at most 20 unique nonblank missing-energy tokens")
    for token in missing:
        if NUMBER.fullmatch(token):
            raise ValueError("A numeric energy token cannot mean missing energy")
    # Snapshot validated values. The caller's source and options stay unchanged.
    metadata = json.loads(_encoded(options))
    for name in NUMERIC_CONDITIONS:
        value = metadata["conditions"][name]
        if value is not None:
            metadata["conditions"][name] = 0.0 if value == 0 else float(value)
    return metadata


def _number(token, power, name, line):
    token = token.strip()
    if len(token) > 128 or not NUMBER.fullmatch(token):
        raise ValueError(f"Line {line}: {name} needs a decimal number within 128 characters")
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(f"Line {line}: invalid {name}") from exc
    parts = value.as_tuple()
    if len(parts.digits) > 80 or abs(parts.exponent) > 400:
        raise ValueError(f"Line {line}: {name} exceeds decimal precision or exponent limits")
    if value < 0:
        raise ValueError(f"Line {line}: {name} must be a nonnegative magnitude")
    with localcontext() as context:
        context.prec = 100
        converted = value.scaleb(power)
        key = "0" if converted == 0 else str(converted.normalize())
    floating = float(converted)
    if not math.isfinite(floating) or (floating == 0 and converted != 0):
        raise ValueError(f"Line {line}: {name} exceeds the finite numerical range")
    return converted, key, floating


def _records(text, syntax):
    for line, record in enumerate(io.StringIO(text, newline=None), 1):
        record = record.rstrip("\r\n")
        if not record.strip():
            continue
        if syntax["delimiter"] == "whitespace":
            fields = record.split()
        else:
            try:
                fields = next(csv.reader([record], delimiter=(
                    "," if syntax["delimiter"] == "csv" else "\t"), strict=True))
            except csv.Error as exc:
                raise ValueError(f"Line {line}: invalid quoted record") from exc
        if len(fields) != 3:
            raise ValueError(f"Line {line}: supply exactly three columns")
        if any(len(token) > 128 for token in fields):
            raise ValueError(f"Line {line}: a field exceeds 128 characters")
        yield line, record, fields


def import_xyz_energy_map(raw_bytes, options):
    """Keep original row evidence and explicit unknown, missing, and ambiguous cells."""
    if not isinstance(raw_bytes, bytes):
        raise TypeError("Supply original source bytes")
    if len(raw_bytes) > MAX_RAW_BYTES:
        raise XyzEnergyLimitError("XYZ source exceeds the 4 MB limit")
    metadata = _validate_options(options)
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Supply a strictly encoded UTF-8 source") from exc
    if "\x00" in text:
        raise ValueError("The source contains a null character")
    syntax = metadata["options"]
    records = _records(text, syntax)
    header = None
    if syntax["has_header"]:
        first = next(records, None)
        if first is None:
            raise ValueError("The source has no header or data rows")
        line, record, fields = first
        header = dict(line_number=line, raw_record=record, lexemes=fields)
    axes = {"current": {}, "voltage": {}}
    float_keys = {"current": {}, "voltage": {}}
    rows, groups = [], {}
    # Reserve cell and metadata evidence before building the result matrix.
    projected = len(_encoded(metadata)) + len(_encoded(header)) + 5000
    for line, record, fields in records:
        if len(rows) >= MAX_ROWS:
            raise XyzEnergyLimitError("XYZ source exceeds 100000 data rows")
        converted = {}
        for name in ("current", "voltage"):
            decimal, key, value = _number(fields[syntax["columns"][name]],
                                         POWERS[name][syntax["units"][name]], name, line)
            if value in float_keys[name] and float_keys[name][value] != decimal:
                raise ValueError(f"Line {line}: distinct {name} coordinates collapse to one float")
            if decimal not in axes[name]:
                projected += len(_encoded(key)) + len(str(value)) + 32
            float_keys[name][value] = decimal
            axes[name][decimal] = (key, value)
            converted[name] = decimal, key, value
        ni, nv = len(axes["current"]), len(axes["voltage"])
        if ni > MAX_CURRENT_AXIS or nv > MAX_VOLTAGE_AXIS or ni * nv > MAX_CELLS:
            raise XyzEnergyLimitError("XYZ axis count or grid product exceeds its limit")
        energy_token = fields[syntax["columns"]["energy"]].strip()
        unknown = energy_token == "" or energy_token in syntax["missing_energy_tokens"]
        energy = None if unknown else _number(
            energy_token, POWERS["energy"][syntax["units"]["energy"]], "energy", line)[2]
        row = dict(id=f"line-{line}", line_number=line, raw_record=record,
                   lexemes=fields, current_decimal_key=converted["current"][1],
                   voltage_decimal_key=converted["voltage"][1],
                   current_a=converted["current"][2], voltage_v=converted["voltage"][2],
                   energy_j=energy, status="unknown" if unknown else "available",
                   cell_status="unknown" if unknown else "available",
                   reason="missing-energy" if unknown else None)
        projected += len(_encoded(row)) + len(row["id"]) + 16
        if projected + ni * nv * 350 > MAX_RESPONSE_BYTES:
            raise XyzEnergyLimitError("XYZ response evidence exceeds the 24 MB bound")
        rows.append(row)
        coordinate = converted["current"][0], converted["voltage"][0]
        groups.setdefault(coordinate, []).append(row)
    if not rows:
        raise ValueError("Supply at least one XYZ data row")
    currents, voltages = sorted(axes["current"]), sorted(axes["voltage"])
    census = dict(source_rows=len(rows), available_rows=sum(
        row["status"] == "available" for row in rows),
        unknown_rows=sum(row["status"] == "unknown" for row in rows),
        available_cells=0, missing_cells=0, unknown_cells=0, ambiguous_cells=0,
        duplicate_coordinates=0, total_cells=len(currents) * len(voltages))
    cells = []
    for voltage in voltages:
        cell_row = []
        for current in currents:
            candidates = groups.get((current, voltage), [])
            status = ("missing" if not candidates else "ambiguous" if len(candidates) > 1
                      else candidates[0]["status"])
            reason = {"missing": "no-source-row", "ambiguous": "duplicate-coordinate",
                      "unknown": "missing-energy", "available": None}[status]
            for row in candidates:
                row["cell_status"] = status
            census[f"{status}_cells"] += 1
            census["duplicate_coordinates"] += status == "ambiguous"
            cell_row.append(dict(current_a=axes["current"][current][1],
                                 voltage_v=axes["voltage"][voltage][1], status=status,
                                 energy_j=candidates[0]["energy_j"] if status == "available" else None,
                                 row_ids=[row["id"] for row in candidates], reason=reason))
        cells.append(cell_row)
    source = dict(name=metadata["source_name"], source=metadata["source"],
                  source_kind=metadata["source_kind"], input_kind=metadata["input_kind"],
                  sha256=hashlib.sha256(raw_bytes).hexdigest(), byte_count=len(raw_bytes),
                  hash_scope=("original-file-bytes" if metadata["input_kind"] == "file"
                              else "utf8-paste-bytes"), encoding="utf-8")
    request_identity = dict(source_sha256=source["sha256"], metadata=metadata)
    result = dict(method_version=METHOD_VERSION, energy_kind=metadata["energy_kind"],
                  conditions=metadata["conditions"], source=source, options=syntax,
                  request_sha256=hashlib.sha256(json.dumps(
                      request_identity, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
                  currents_a=[axes["current"][c][1] for c in currents],
                  voltages_v=[axes["voltage"][v][1] for v in voltages],
                  current_decimal_keys=[axes["current"][c][0] for c in currents],
                  voltage_decimal_keys=[axes["voltage"][v][0] for v in voltages],
                  units=dict(current="A", voltage="V", energy="J"), cells=cells,
                  rows=rows, header=header, census=census, limits=LIMITS.copy(),
                  assumptions=["All coordinates and energies use nonnegative magnitudes",
                               "Zero current, voltage, and energy are valid observations",
                               "Every duplicate coordinate remains ambiguous",
                               "No interpolation, scaling, or source mutation",
                               "Raw records preserve row text. Keep the original source bytes for exact byte replay."])
    if len(_encoded(result)) > MAX_RESPONSE_BYTES:
        raise XyzEnergyLimitError("XYZ result exceeds the 24 MB limit")
    return result
