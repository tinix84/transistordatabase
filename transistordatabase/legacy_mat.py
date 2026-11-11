"""Bounded, nonexecuting legacy SC data migration.

Original bytes remain the archive authority. Preview validity does not imply
that a source curve supports passive loss calculations.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import re
import struct
import zlib

import numpy as np
from scipy.io import loadmat, savemat

MAX_SOURCE_BYTES = 24 * 1024 * 1024
MAX_EXPANDED_BYTES = 96 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 8_000_000
MAX_TOTAL_ELEMENTS = 24_000_000
MAX_FIELDS = 4096
MAX_VARIABLES = 128
MAX_DEPTH = 16
MAX_TEXT_BYTES = 1_000_000
MAX_INLINE_ELEMENTS = 200_000
MAX_CALIBRATION_POINTS = 20_000
MAX_BLOB_BYTES = 24 * 1024 * 1024
MAX_JSON_INTEGER = 2**53 - 1
SCHEMA = "transistordatabase-legacy-mat"
VERSION = 1
KNOWN_FIELDS = (
    "name", "blockingVoltage", "cost", "weight", "onStateVoltage",
    "onStateCurrent", "turnOnEnergy", "turnOnCurrent", "turnOnVoltage",
    "turnOffEnergy", "turnOffCurrent", "turnOffVoltage",
    "forwardThermalResistance", "reverseThermalResistance",
    "commonThermalResistance", "forwardConductionTrace", "reverseConductionTrace",
    "forwardTurnOnTrace", "forwardTurnOffTrace", "reverseTurnOnTrace",
    "reverseTurnOffTrace", "forwardThermalInterfaceArea",
    "reverseThermalInterfaceArea", "commonThermalInterfaceArea", "fileFormatVersion",
)
UNIT_MAP = {"onStateVoltage": "V", "onStateCurrent": "A",
            "turnOnEnergy": "J", "turnOffEnergy": "J", "turnOnCurrent": "A",
            "turnOffCurrent": "A", "turnOnVoltage": "V", "turnOffVoltage": "V",
            **{f"{p}ThermalResistance": "K/W" for p in ("forward", "reverse", "common")},
            **{f"{p}ThermalInterfaceArea": "m²" for p in ("forward", "reverse", "common")}}


class MatFormatError(ValueError):
    """An invalid or unsupported data container."""


@dataclass
class _Budget:
    elements: int = 0
    fields: int = 0
    expanded: int = 0
    text_bytes: int = 0
    containers: int = 0

    def shape(self, shape):
        if not 1 <= len(shape) <= 8 or any(n < 0 for n in shape):
            raise MatFormatError("Unsupported array dimensions")
        prefix, containers = 1, 0
        for index, dimension in enumerate(shape):
            prefix *= dimension
            if dimension > MAX_ARRAY_ELEMENTS or prefix > MAX_ARRAY_ELEMENTS:
                raise MatFormatError("MAT dimension or prefix container budget exceeded")
            if index < len(shape) - 1:
                containers += prefix
        self.containers += containers
        if self.containers > MAX_TOTAL_ELEMENTS:
            raise MatFormatError("MAT aggregate container budget exceeded")
        count = math.prod(shape)
        self.elements += count
        if count > MAX_ARRAY_ELEMENTS or self.elements > MAX_TOTAL_ELEMENTS:
            raise MatFormatError("MAT array element budget exceeded")
        return count

    def field(self, depth):
        self.fields += 1
        if depth > MAX_DEPTH or self.fields > MAX_FIELDS:
            raise MatFormatError("MAT nesting or field budget exceeded")

    def text(self, size):
        self.text_bytes += size
        if size > MAX_TEXT_BYTES or self.text_bytes > 2 * MAX_TEXT_BYTES:
            raise MatFormatError("MAT string byte budget exceeded")


def _tag(data, offset, endian):
    if offset + 8 > len(data):
        raise MatFormatError("Incomplete MAT data tag")
    first, second = struct.unpack_from(endian + "II", data, offset)
    if first > 18:
        short_type, short_size = struct.unpack_from(endian + "HH", data, offset)
        if short_size < 1 or short_size > 4 or short_type not in range(1, 19):
            raise MatFormatError("Invalid MAT small data tag")
        return short_type, data[offset + 4:offset + 4 + short_size], offset + 8
    end = offset + 8 + second
    if end > len(data):
        raise MatFormatError("MAT data size exceeds source bytes")
    # Compressed MAT elements do not have eight-byte trailing padding.
    return first, data[offset + 8:end], end if first == 15 else end + (-second % 8)


def _mat_preflight(raw):
    """Check expanded tags and nested dimensions before the SciPy allocation."""
    if len(raw) < 128 or raw[126:128] not in (b"IM", b"MI"):
        raise MatFormatError("Unsupported MAT header. HDF5 v7.3 is not supported")
    endian = "<" if raw[126:128] == b"IM" else ">"
    if struct.unpack_from(endian + "H", raw, 124)[0] != 0x100:
        raise MatFormatError("Unsupported MAT version. HDF5 v7.3 is not supported")
    budget = _Budget(expanded=len(raw))

    def matrix(data, depth):
        budget.field(depth)
        kind, flags, pos = _tag(data, 0, endian)
        if kind != 6 or len(flags) != 8:
            raise MatFormatError("Invalid MAT array flags")
        flag = struct.unpack_from(endian + "I", flags)[0]
        cls = flag & 255
        if flag & 0x800 or cls not in (2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            raise MatFormatError("Unsupported MATLAB object, function, cell, sparse, opaque or complex array")
        kind, dims, pos = _tag(data, pos, endian)
        if kind != 5 or len(dims) % 4 or not 2 <= len(dims) // 4 <= 8:
            raise MatFormatError("Invalid MAT dimensions")
        shape = struct.unpack(endian + "i" * (len(dims) // 4), dims)
        count = budget.shape(shape)
        kind, name, pos = _tag(data, pos, endian)
        if kind != 1 or len(name) > 1024:
            raise MatFormatError("Invalid MAT variable name")
        if cls == 2:
            if count != 1:
                raise MatFormatError("Only scalar MATLAB structs are supported")
            kind, width_raw, pos = _tag(data, pos, endian)
            if kind != 5 or len(width_raw) != 4:
                raise MatFormatError("Invalid MAT field width")
            width = struct.unpack(endian + "i", width_raw)[0]
            kind, names, pos = _tag(data, pos, endian)
            if kind != 1 or width <= 0 or width > 1024 or len(names) % width:
                raise MatFormatError("Invalid MAT struct field names")
            if len(names) // width > MAX_FIELDS - budget.fields:
                raise MatFormatError("MAT struct field budget exceeded")
            fields = [names[i:i + width].split(b"\0", 1)[0] for i in range(0, len(names), width)]
            if len(fields) != len(set(fields)) or any(not n for n in fields):
                raise MatFormatError("Duplicate or empty MAT struct field name")
            for _ in fields:
                kind, child, pos = _tag(data, pos, endian)
                if kind != 14:
                    raise MatFormatError("Unsupported MAT struct field encoding")
                matrix(child, depth + 1)
        else:
            kind, values, pos = _tag(data, pos, endian)
            sizes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 9: 8, 12: 8, 13: 8, 16: 1, 17: 2, 18: 4}
            if kind not in sizes or (cls != 4 and len(values) != count * sizes[kind]):
                raise MatFormatError("MAT numeric payload does not match dimensions")
            if cls == 4:
                budget.text(len(values))
        if pos != len(data):
            raise MatFormatError("Unexpected MAT array payload")
        return name

    variable_names = set()
    def elements(data, depth=0):
        if depth > 1:
            raise MatFormatError("Nested MAT compression is unsupported")
        pos = 0
        while pos < len(data):
            kind, payload, pos = _tag(data, pos, endian)
            if kind == 15:
                if depth:
                    raise MatFormatError("Nested MAT compression is unsupported")
                decoder = zlib.decompressobj()
                available = MAX_EXPANDED_BYTES - budget.expanded
                try:
                    expanded = decoder.decompress(payload, available + 1)
                except zlib.error as exc:
                    raise MatFormatError("Invalid MAT compressed stream") from exc
                budget.expanded += len(expanded)
                if budget.expanded > MAX_EXPANDED_BYTES or not decoder.eof or decoder.unused_data:
                    raise MatFormatError("MAT compressed expansion budget or stream validation failed")
                elements(expanded, depth + 1)
            elif kind == 14:
                name = matrix(payload, 0)
                if name in (b"__header__", b"__version__", b"__globals__"):
                    raise MatFormatError("Reserved MATLAB reader metadata names are unsupported")
                if name in variable_names or len(variable_names) >= MAX_VARIABLES:
                    raise MatFormatError("Duplicate MAT variable or variable budget exceeded")
                variable_names.add(name)
            else:
                raise MatFormatError("Unsupported MAT top-level element")
        if pos != len(data):
            raise MatFormatError("Incomplete MAT element padding")

    elements(raw[128:])


_NUMBER = re.compile(rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _octave_read(raw):
    """Read only literal numeric arrays, strings and scalar structs."""
    stream = BytesIO(raw)
    budget = _Budget()

    def line():
        return stream.readline().rstrip(b"\r\n")

    def content():
        while True:
            value = line()
            if value.strip() or stream.tell() == len(raw):
                return value

    def metadata(key):
        value = content()
        prefix = b"# " + key.encode() + b": "
        if not value.startswith(prefix):
            raise MatFormatError(f"Expected Octave {key} metadata")
        return value[len(prefix):]

    def integer(key):
        value = metadata(key)
        if not value.isdigit() or len(value) > 10:
            raise MatFormatError("Invalid Octave dimension or count")
        return int(value)

    def variable(depth=0):
        budget.field(depth)
        name = metadata("name").decode("utf-8")
        if not name or len(name) > 1024:
            raise MatFormatError("Invalid Octave variable name")
        kind = metadata("type")
        if kind == b"scalar struct":
            if integer("ndims") != 2 or content().split() != [b"1", b"1"]:
                raise MatFormatError("Only scalar Octave structs are supported")
            count = integer("length")
            if count > MAX_FIELDS:
                raise MatFormatError("Octave field budget exceeded")
            result = {}
            for _ in range(count):
                child, value = variable(depth + 1)
                if child in result:
                    raise MatFormatError("Duplicate Octave struct field")
                result[child] = value
            return name, result
        if kind == b"sq_string":
            if integer("elements") != 1:
                raise MatFormatError("Only single Octave strings are supported")
            length = integer("length")
            budget.text(length)
            text = stream.read(length)
            if len(text) != length or stream.read(1) not in (b"\n", b"\r"):
                raise MatFormatError("Octave string length does not match bytes")
            return name, np.array(list(text.decode("utf-8")), dtype="U1").reshape(1, -1)
        if kind == b"scalar":
            value = content().strip()
            if not _NUMBER.fullmatch(value):
                raise MatFormatError("Unsupported Octave scalar literal")
            budget.shape((1, 1))
            return name, np.array([[float(value)]])
        if kind not in (b"matrix", b"uint8 matrix"):
            raise MatFormatError("Unsupported Octave type. Only numeric data and strings are supported")
        offset = stream.tell()
        marker = content()
        if marker.startswith(b"# ndims: "):
            if marker != b"# ndims: 3":
                raise MatFormatError("Only three-dimensional Octave images are supported")
            tokens = content().split()
            if len(tokens) != 3 or any(not t.isdigit() for t in tokens):
                raise MatFormatError("Invalid Octave image dimensions")
            shape = tuple(int(t) for t in tokens)
            order = "F"
        else:
            stream.seek(offset)
            shape = (integer("rows"), integer("columns"))
            order = "C"
        count = budget.shape(shape)
        start = stream.tell()
        end = raw.find(b"#", start)
        end = len(raw) if end < 0 else end
        payload = raw[start:end]
        actual_count, actual_rows = 0, 0
        for row in (BytesIO(payload) if order == "C" else (payload,)):
            row_count = 0
            for token in re.finditer(rb"\S+", row):
                actual_count += 1
                row_count += 1
                if actual_count > count or not _NUMBER.fullmatch(token.group()):
                    raise MatFormatError("Octave numeric payload does not match dimensions")
            if row_count:
                actual_rows += 1
                if order == "C" and (row_count != shape[1] or actual_rows > shape[0]):
                    raise MatFormatError("Octave numeric row does not match the declared columns")
        if actual_count != count:
            raise MatFormatError("Octave numeric payload does not match dimensions")
        if order == "C" and count and actual_rows != shape[0]:
            raise MatFormatError("Octave numeric row count does not match dimensions")
        values = np.fromstring(payload.decode("ascii"), sep=" ") if count else np.empty(0)
        if kind == b"uint8 matrix":
            if np.any((values < 0) | (values > 255) | (values != np.floor(values))):
                raise MatFormatError("Octave image pixels must be integers from 0 to 255")
            values = values.astype(np.uint8)
        stream.seek(end)
        return name, values.reshape(shape, order=order)

    creator = content()
    if not creator.startswith(b"# Created by Octave "):
        raise MatFormatError("Unsupported Octave creator header")
    result = {}
    while stream.tell() < len(raw):
        pos = stream.tell()
        if not content():
            break
        stream.seek(pos)
        name, value = variable()
        if name in result or len(result) >= MAX_VARIABLES:
            raise MatFormatError("Duplicate Octave variable")
        result[name] = value
    return result


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.dtype.names:
        if value.shape != (1, 1):
            raise MatFormatError("Only scalar MATLAB structs are supported")
        return {name: _normalize(value[name][0, 0]) for name in value.dtype.names}
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "biufU":
        raise MatFormatError("Unsupported MATLAB object or value type")
    return value


def _read(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_SOURCE_BYTES or not raw:
        raise MatFormatError("Provide nonempty MAT bytes within the 24 MiB source budget")
    if raw.startswith(b"# Created by Octave "):
        return _octave_read(raw), "octave-text-subset"
    _mat_preflight(raw)
    try:
        values = loadmat(BytesIO(raw), chars_as_strings=False, mat_dtype=True, verify_compressed_data_integrity=True)
        return {k: _normalize(v) for k, v in values.items() if k not in ("__header__", "__version__", "__globals__")}, "mat5"
    except (ValueError, TypeError, OSError) as exc:
        raise MatFormatError(f"Invalid or unsupported MAT5 data: {exc}") from exc


def _text(value):
    return "".join(value.reshape(-1).tolist()) if isinstance(value, np.ndarray) and value.dtype.kind == "U" else None


def _number(value):
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf" and value.size == 1:
        return value.item()
    return None


def _source_version(sc):
    if "fileFormatVersion" not in sc:
        return "0.0"
    value = sc["fileFormatVersion"]
    if not isinstance(value, np.ndarray) or value.dtype.kind != "U" or value.ndim != 2 or value.shape[0] != 1:
        return None
    return _text(value)


def _descriptor(value, path, inline_budget, segments):
    if isinstance(value, dict):
        return {"kind": "struct", "shape": [1, 1], "fields": {
            key: _descriptor(item, path + "." + key, inline_budget, segments + [key]) for key, item in value.items()}}
    descriptor = {"kind": "array", "dtype": value.dtype.str, "itemsize": value.dtype.itemsize,
                  "shape": list(value.shape), "path": path}
    prefix, json_values = 1, value.size
    for dimension in value.shape[:-1]:
        prefix *= dimension
        json_values += prefix
    safe_integers = not (value.size and value.dtype.kind in "iu" and
                         (int(value.min()) < -MAX_JSON_INTEGER or int(value.max()) > MAX_JSON_INTEGER))
    inline = json_values <= 4096 and json_values <= inline_budget[0] and safe_integers
    if inline and (value.dtype.kind == "U" or np.all(np.isfinite(value))):
        descriptor["values"] = value.tolist()
        inline_budget[0] -= json_values
    else:
        binary = value.tobytes(order="C")
        descriptor.update(encoding="source-array-reference", byte_length=len(binary),
                          sha256=sha256(binary).hexdigest(), source_array_path=segments)
    if value.size and value.dtype.kind in "f" and not np.all(np.isfinite(value)):
        descriptor["nonfinite"] = True
    if path.endswith(".image"):
        descriptor["role"] = "original-image"
        descriptor["sha256"] = sha256(value.tobytes(order="C")).hexdigest()
        descriptor["byte_length"] = value.nbytes
        descriptor["source_array_path"] = segments
    return descriptor


def _diagnostics(values):
    diagnostics = []

    def add(path, code, message):
        diagnostics.append({"path": path, "code": code, "message": message})

    sc = values.get("scData")
    if not isinstance(sc, dict):
        add("$", "preview-only-workspace", "This workspace is preserved. Only scData supports explicit core finalization")
        return diagnostics
    if _source_version(sc) not in ("0.0", "0.1", "0.2"):
        add("scData.fileFormatVersion", "unsupported-source-version", "The original version is invalid or unsupported. Only a missing version means 0.0")
    physical_arrays = ("onStateCurrent", "onStateVoltage", "turnOnCurrent", "turnOnEnergy",
                       "turnOffCurrent", "turnOffEnergy", "turnOnVoltage", "turnOffVoltage")
    for key in physical_arrays:
        if key in sc and (not isinstance(sc[key], np.ndarray) or sc[key].dtype.kind not in "iuf"):
            add("scData." + key, "invalid-physical-field", "The source field is not a numeric physical array. Its original value remains preserved")
    for stem in ("turnOn", "turnOff"):
        current, energy = sc.get(stem + "Current"), sc.get(stem + "Energy")
        if isinstance(current, np.ndarray) and isinstance(energy, np.ndarray) and current.dtype.kind in "iuf" and energy.dtype.kind in "iuf":
            x, y = current.reshape(-1), energy.reshape(-1)
            if x.size != y.size:
                add("scData." + stem, "shape-conflict", "Current and energy counts differ")
                continue
            traces = [sc.get(branch + stem[0].upper() + stem[1:] + "Trace") for branch in ("forward", "reverse")]
            empty = all(not isinstance(t, dict) or not np.asarray(t.get("xValues", [])).size for t in traces)
            placeholder = x.size == 3 and np.array_equal(x, [-1e100, 0, 1e100]) and np.all(y == 0) and empty
            if placeholder:
                add("scData." + stem, "synthetic-placeholder", "The ±1e100 zero-energy signature has no original branch samples. It is not measured zero loss")
            elif x.size and np.all(y == 0) and empty:
                add("scData." + stem, "source-zero-unverified", "Original branch traces are absent. The stored zero curve has no established measurement evidence")
            if np.any(y < 0):
                add("scData." + stem, "negative-energy", "Negative source energy is preserved. Passive heat finalization is unsupported")
            refs = [_number(t.get("testVoltage")) for t in traces if isinstance(t, dict)]
            combined = _number(sc.get(stem + "Voltage"))
            refs = [v for v in refs if v is not None]
            if len(set(refs + ([combined] if combined is not None else []))) > 1:
                add("scData." + stem, "conflicting-reference-voltages", f"Original trace references {refs} differ from the combined reference {combined}")
    for stem in ("onState", "turnOn", "turnOff"):
        x = sc.get(stem + "Current")
        if isinstance(x, np.ndarray) and x.dtype.kind in "iuf" and np.any(np.abs(x) == 1e-6):
            add("scData." + stem, "synthetic-near-zero-candidate", "Combined ±1e-6 A points match an editor extrapolation location. Prefer original branch traces")
    for key in ("name", "blockingVoltage", "cost", "weight"):
        item = sc.get(key)
        if isinstance(item, np.ndarray) and not item.size:
            add("scData." + key, "unknown-empty", "The original empty value remains unknown")
    return diagnostics


def legacy_calibration_report(trace: dict) -> dict:
    """Compare recorded values with the historical calibration without editing them."""
    report = {"coordinate_convention": "x/image-height,y/image-width", "axes": {}}
    image = trace.get("image")
    for axis in ("x", "y"):
        try:
            low = _number(trace.get(axis + "LowerLeftCornerLocation"))
            high = _number(trace.get(axis + "UpperRightCornerLocation"))
            a = _number(trace.get(axis + "LowerLeftCornerValue"))
            b = _number(trace.get(axis + "UpperRightCornerValue"))
            scale = _text(trace.get(axis + "Scale"))
            if any(v is None or not math.isfinite(v) for v in (low, high, a, b)) or low == high:
                raise ValueError("Calibration requires distinct finite location anchors and finite value anchors")
            if not math.isfinite(high - low) or (scale == "lin" and not math.isfinite(b - a)):
                raise ValueError("Calibration anchor differences exceed finite numeric range")
            locations = np.asarray(trace.get(axis + "Locations"), dtype=float).reshape(-1)
            recorded = np.asarray(trace.get(axis + "Values"), dtype=float).reshape(-1)
            if locations.size > 20000 or locations.size != recorded.size or not np.all(np.isfinite(locations)) or not np.all(np.isfinite(recorded)):
                raise ValueError("Calibration requires matching bounded finite locations and recorded values")
            q = (locations - low) / (high - low)
            with np.errstate(over="ignore", invalid="ignore"):
                if scale == "lin":
                    reconstructed = a + q * (b - a)
                elif scale == "log" and a > 0 and b > 0:
                    reconstructed = 10 ** (math.log10(a) + q * (math.log10(b) - math.log10(a)))
                else:
                    raise ValueError("Calibration requires lin or positive-anchor log scale")
                residual = reconstructed - recorded
            if not np.all(np.isfinite(reconstructed)) or not np.all(np.isfinite(residual)):
                raise ValueError("Calibration results exceed finite numeric range")
            result = {"status": "reconstructed-for-comparison", "scale": scale,
                      "label": _text(trace.get(axis + "Label")), "reconstructed_values": reconstructed.tolist(),
                      "residuals": residual.tolist(), "outside_calibration_indices": np.flatnonzero((q < 0) | (q > 1)).tolist(),
                      "max_absolute_residual": float(np.max(np.abs(residual))) if residual.size else None}
            if isinstance(image, np.ndarray) and image.ndim >= 2:
                dimension = image.shape[0 if axis == "x" else 1]
                result["legacy_pixel_locations"] = (locations * dimension).tolist()
            report["axes"][axis] = result
        except (ValueError, TypeError, OverflowError) as exc:
            report["axes"][axis] = {"status": "unsupported-calibration", "message": str(exc)}
    return report


def preview_mat(raw: bytes, source_name: str | None = None) -> dict:
    """Return a versioned archive and bounded typed preview without execution."""
    if source_name is not None and (not isinstance(source_name, str) or len(source_name) > 2000):
        raise ValueError("Source name must be bounded text")
    values, format_name = _read(raw)
    sc = values.get("scData")
    original_version = _source_version(sc) if isinstance(sc, dict) and "fileFormatVersion" in sc else None
    history = []
    if isinstance(sc, dict):
        version = _source_version(sc)
        if version == "0.0":
            history.append({"from": "0.0", "to": "0.1", "operation": "Add three empty thermal-interface areas"})
            version = "0.1"
        if version == "0.1":
            history.append({"from": "0.1", "to": "0.2", "operation": "Change the format version and binary save format only"})
    calibrations = {}
    calibration_points = [0]
    def collect(item, path):
        if isinstance(item, dict):
            if "xLocations" in item or "yLocations" in item:
                calibration_points[0] += sum(v.size for k, v in item.items() if k in ("xLocations", "yLocations") and isinstance(v, np.ndarray))
                calibrations[path] = (legacy_calibration_report(item) if calibration_points[0] <= MAX_CALIBRATION_POINTS else
                                      {"status": "source-preserved-comparison-budget-exceeded"})
            for key, child in item.items():
                collect(child, path + "." + key)
    for key, item in values.items():
        collect(item, key)
    inline_budget = [MAX_INLINE_ELEMENTS]
    workspace_schema = ("scData" if isinstance(sc, dict) else "hsData" if "hsData" in values else
                        "digitizer-traces" if any(k.endswith("Trace") for k in values) else "numeric-workspace")
    proposed_units = (UNIT_MAP if isinstance(sc, dict) else
                      {key: "K·m²/W" for key in values.get("hsData", {})} if isinstance(values.get("hsData"), dict) else {})
    return {"schema": SCHEMA, "version": VERSION, "parser_version": 1, "workspace_schema": workspace_schema,
            "source": {"name": source_name, "sha256": sha256(raw).hexdigest(), "byte_length": len(raw), "format": format_name},
            "source_base64": base64.b64encode(raw).decode("ascii"),
            "source_version": original_version, "conversion_history": history,
            "variables": {k: _descriptor(v, k, inline_budget, [k]) for k, v in values.items()},
            "known_fields": [k for k in KNOWN_FIELDS if isinstance(sc, dict) and k in sc],
            "unknown_fields": [k for k in sc if k not in KNOWN_FIELDS] if isinstance(sc, dict) else [],
            "proposed_units": {"values": proposed_units, "provenance": "Legacy SC editor labels and thermal-area use. Explicit finalization must confirm units"},
            "legacy_coordinates": "x/image-height,y/image-width",
            "calibrations": calibrations,
            "diagnostics": _diagnostics(values), "status": "preview-only-requires-explicit-finalization"}


def mat_array_blob(raw: bytes, array_path: list[str]) -> dict:
    """Resolve a bounded source-array reference without opening a local path."""
    if not isinstance(array_path, list) or not 1 <= len(array_path) <= MAX_DEPTH + 1 or any(not isinstance(k, str) or not k or len(k) > 1024 for k in array_path):
        raise ValueError("Declare a bounded source array field path")
    value, _ = _read(raw)
    for key in array_path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("Source array field path does not exist")
        value = value[key]
    if not isinstance(value, np.ndarray) or value.nbytes > MAX_BLOB_BYTES:
        raise ValueError("Source array requires a numeric or character array within the 24 MiB blob budget")
    binary = value.tobytes(order="C")
    return {"source_sha256": sha256(raw).hexdigest(), "array_path": array_path,
            "dtype": value.dtype.str, "itemsize": value.dtype.itemsize,
            "shape": list(value.shape), "encoding": "base64-c-order",
            "byte_length": len(binary), "sha256": sha256(binary).hexdigest(),
            "data_base64": base64.b64encode(binary).decode("ascii")}


def archive_from_json(document: dict) -> dict:
    """Reparse authoritative bytes instead of trusting caller-modified preview values."""
    if document.get("schema") != SCHEMA or type(document.get("version")) is not int or document["version"] != VERSION:
        raise MatFormatError("Unsupported MAT archive schema or version")
    encoded = document.get("source_base64")
    if not isinstance(encoded, str) or len(encoded) > (MAX_SOURCE_BYTES + 2) // 3 * 4:
        raise MatFormatError("MAT archive source exceeds the byte budget")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise MatFormatError("Invalid MAT archive base64") from exc
    checked = preview_mat(raw, document.get("source", {}).get("name"))
    if checked != document:
        raise MatFormatError("MAT archive preview does not match authoritative source bytes")
    return checked


def _mat_value(value, path="$"):
    if isinstance(value, dict):
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", key) for key in value):
            raise MatFormatError("MAT5 export cannot preserve an unsupported field name. Use the source archive")
        return {key: _mat_value(item, path + "." + key) for key, item in value.items()}
    if value.dtype.kind == "U":
        if value.ndim != 2:
            raise MatFormatError("MAT5 export supports two-dimensional character arrays")
        if not value.size and value.shape != (0, 0):
            raise MatFormatError(f"MAT5 export cannot preserve empty character shape at {path}. Use the original source archive")
        return np.array(["".join(row) for row in value.tolist()], dtype=f"U{max(1, value.shape[1])}")
    return value


def _verify_mat_values(left, right, path="$"):
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise MatFormatError(f"MAT5 export changes source fields at {path}. Use the original source archive")
        for key in left:
            _verify_mat_values(left[key], right[key], path + "." + key)
    elif (not isinstance(right, np.ndarray) or left.shape != right.shape or
          left.dtype.kind != right.dtype.kind or left.dtype.itemsize != right.dtype.itemsize or
          not np.array_equal(left, right, equal_nan=left.dtype.kind in "f")):
        raise MatFormatError(f"MAT5 export changes source shape, dtype or values at {path}. Use the original source archive")


def export_mat(raw: bytes, *, mode: str) -> bytes:
    """Return original bytes or a semantic numeric MAT5 conversion."""
    values, _ = _read(raw)
    if mode == "source-preserved":
        return raw
    if mode != "legacy-mat5":
        raise MatFormatError("Select source-preserved or legacy-mat5 export")
    sc = values.get("scData")
    if not isinstance(sc, dict):
        raise MatFormatError("Legacy MAT5 migration requires a scalar scData record")
    version = _source_version(sc)
    if version not in ("0.0", "0.1", "0.2"):
        raise MatFormatError("Unsupported SC file format version")
    if version == "0.0":
        for branch in ("forward", "reverse", "common"):
            sc.setdefault(branch + "ThermalInterfaceArea", np.empty((0, 0)))
    sc["fileFormatVersion"] = np.array(list("0.2")).reshape(1, -1)
    output = BytesIO()
    savemat(output, _mat_value(values), long_field_names=True, do_compression=True)
    result = output.getvalue()
    converted, _ = _read(result)
    _verify_mat_values(values, converted)
    return result


def _finite(value, label, *, nullable=False):
    if nullable and value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Declare finite numeric {label}")
    return float(value)


def _curve_numbers(value):
    """Convert to the existing float64 core without integer wrap or hidden rounding."""
    if value.dtype.kind in "iu" and value.size and (
        int(value.min()) < -MAX_JSON_INTEGER or int(value.max()) > MAX_JSON_INTEGER
    ):
        raise ValueError("Integer curve samples exceed exact float64 core coverage. Preserve them in the source archive")
    converted = value.astype(np.float64).reshape(-1)
    if not np.all(np.isfinite(converted)):
        raise ValueError("Curve samples exceed finite float64 core coverage")
    return converted


def _outside_original_calibration(trace, count, selected_indices):
    outside = set()
    for axis in ("x", "y"):
        low = _number(trace.get(axis + "LowerLeftCornerLocation"))
        high = _number(trace.get(axis + "UpperRightCornerLocation"))
        locations = trace.get(axis + "Locations")
        if low is None or high is None or not math.isfinite(low) or not math.isfinite(high) or low == high:
            continue
        if not isinstance(locations, np.ndarray) or not locations.size:
            continue
        if locations.size != count or locations.dtype.kind not in "iuf":
            raise ValueError("Original calibration locations do not match the finite source sample indices")
        indices = sorted(selected_indices)
        selected = locations.reshape(-1)[indices]
        if not np.all(np.isfinite(selected)):
            raise ValueError("Selected original calibration locations must be finite")
        outside.update(indices[i] for i in np.flatnonzero((selected < min(low, high)) | (selected > max(low, high))))
    return outside


def _curve_source(sc, choice, diagnostics):
    kind, branch, source = choice["kind"], choice["branch"], choice["source"]
    stems = {"conduction": "Conduction", "turn-on": "TurnOn", "turn-off": "TurnOff"}
    combined = {"conduction": "onState", "turn-on": "turnOn", "turn-off": "turnOff"}[kind]
    reference = None
    if source == "original-trace":
        path = branch + stems[kind] + "Trace"
        trace = sc.get(path)
        if not isinstance(trace, dict):
            raise ValueError(f"Original trace {path} is absent")
        labels = [_text(trace.get(axis + "Label")) for axis in ("x", "y")]
        labels = [label.strip() if label else label for label in labels]
        required = ["Voltage (V)", "Current (A)"] if kind == "conduction" else ["Current (A)", "Energy (J)"]
        if labels == required:
            x, y = trace.get("xValues"), trace.get("yValues")
        elif labels == required[::-1]:
            x, y = trace.get("yValues"), trace.get("xValues")
        else:
            raise ValueError("Original trace labels require an explicit supported SI axis mapping")
        if kind == "conduction":
            x, y = y, x  # The internal pair is current, voltage or energy.
        reference = _number(trace.get("testVoltage"))
    else:
        path = combined
        if choice["coordinate_convention"] != "signed-passive":
            raise ValueError("Combined curves require the explicit signed-passive convention")
        codes = {d["code"] for d in diagnostics if d["path"] == "scData." + combined}
        if "synthetic-placeholder" in codes:
            raise ValueError("Synthetic placeholder energy cannot supply a passive loss curve")
        if "conflicting-reference-voltages" in codes:
            raise ValueError("Conflicting combined reference voltages require original branch traces")
        x = sc.get(combined + "Current")
        y = sc.get(combined + ("Voltage" if kind == "conduction" else "Energy"))
        reference = _number(sc.get(combined + "Voltage")) if kind != "conduction" else None
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or x.dtype.kind not in "iuf" or y.dtype.kind not in "iuf":
        raise ValueError("Curve samples must be numeric arrays")
    x, y = x.reshape(-1), y.reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("Curve requires matching finite sample arrays with at least two points")
    original_count = x.size
    indices = choice["sample_indices"]
    if indices is None and x.size > 20000:
        raise ValueError("Select at most 20000 recorded curve samples explicitly")
    if indices is not None:
        if not isinstance(indices, list) or not 2 <= len(indices) <= 20000 or any(type(i) is not int or i < 0 or i >= len(x) for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("Sample indices must select at least two distinct recorded points")
        x, y = x[indices], y[indices]
    selected_indices = set(range(x.size)) if indices is None else set(indices)
    x, y = _curve_numbers(x), _curve_numbers(y)
    policy = choice.get("extrapolation_policy", "reject")
    if policy not in ("reject", "explicit-estimate"):
        raise ValueError("Select reject or explicit-estimate for original calibration extrapolation")
    if source == "original-trace" and policy != "explicit-estimate" and _outside_original_calibration(trace, original_count, selected_indices):
        raise ValueError("Exclude original points outside calibration or explicitly select extrapolation as an estimate")
    if source == "combined":
        keep = x >= 0 if branch == "forward" else x <= 0
        x, y = x[keep], y[keep]
        if np.any(np.abs(x) == 1e-6):
            raise ValueError("Exclude the combined synthetic near-zero candidate points explicitly")
        if branch == "reverse":
            x = -x
            if kind == "conduction":
                y = -y
    elif choice["coordinate_convention"] == "signed-passive" and branch == "reverse":
        x = -x
        if kind == "conduction":
            y = -y
    elif choice["coordinate_convention"] not in ("trace-positive-magnitudes", "signed-passive"):
        raise ValueError("Select a supported coordinate convention")
    if x.size < 2 or np.any(x < 0) or np.any(y < 0):
        raise ValueError("Passive finalization requires nonnegative current and voltage or energy. Negative source energy remains unsupported")
    data_status = choice.get("data_status", "recorded-source-values")
    if data_status not in ("recorded-source-values", "source-zero-unverified", "declared-ideal-model"):
        raise ValueError("Select recorded source values or an explicit ideal model status")
    if data_status == "declared-ideal-model":
        if not isinstance(choice.get("ideal_model_scope"), str) or not choice["ideal_model_scope"].strip():
            raise ValueError("Declare the ideal model scope explicitly")
    elif choice.get("ideal_model_scope") is not None:
        raise ValueError("An ideal model scope requires the explicit ideal model status")
    if data_status == "source-zero-unverified":
        if not isinstance(choice.get("source_zero_acknowledgement"), str) or not choice["source_zero_acknowledgement"].strip():
            raise ValueError("Acknowledge that the recorded zero curve has no established measurement evidence")
        if kind == "conduction" or not np.all(y == 0):
            raise ValueError("The source-zero-unverified status requires an all-zero energy curve")
    elif choice.get("source_zero_acknowledgement") is not None:
        raise ValueError("A source zero acknowledgement requires source-zero-unverified status")
    if kind != "conduction" and source == "combined" and np.all(y == 0) and data_status not in ("source-zero-unverified", "declared-ideal-model"):
        raise ValueError("All-zero combined energy requires explicit source-zero-unverified acknowledgement, an original trace, or a scoped ideal model declaration")
    if np.any(x == 0) and choice["zero_policy"] != "recorded-branch-value":
        raise ValueError("Select the recorded branch zero convention explicitly")
    if choice["sample_order"] == "sort-distinct-current":
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
    elif choice["sample_order"] != "recorded-order":
        raise ValueError("Select a supported sample order")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Branch current samples must be distinct and strictly increasing")
    return x, y, reference, path


def finalize_mat(raw: bytes, *, identity: dict, ratings: dict, thermal_properties: dict,
                 curves: list[dict], omission_reasons: dict[str, str], units: str,
                 declaration_source: str, source_name: str | None = None):
    """Map an explicit supported branch subset into the existing core aggregate.

    Separate component records prevent interpolation across a signed-current gap.
    All original arrays, extras, images and declarations remain in provenance.
    """
    from transistordatabase.core.models import (
        ChannelCharacteristics, ElectricalRatings, SwitchingLossData,
        ThermalProperties, Transistor, TransistorMetadata,
    )
    values, _ = _read(raw)
    if source_name is not None and (not isinstance(source_name, str) or len(source_name) > 2000):
        raise ValueError("Source name must be bounded text")
    sc = values.get("scData")
    if not isinstance(sc, dict) or _source_version(sc) not in ("0.0", "0.1", "0.2"):
        raise ValueError("Finalization requires a supported scalar scData version")
    if units != "SC-editor-SI-confirmed":
        raise ValueError("Confirm SC editor SI units before finalization")
    if not isinstance(declaration_source, str) or not declaration_source.strip() or len(declaration_source) > 2000:
        raise ValueError("Declare a nonblank source citation for the new metadata")
    identity_keys = {"name", "type", "author", "manufacturer", "housing_type"}
    if set(identity) != identity_keys or any(not isinstance(v, str) or not v.strip() for v in identity.values()):
        raise ValueError("Declare the complete core identity separately from the source filename")
    if set(ratings) != {"v_abs_max", "i_abs_max", "i_cont", "t_j_max"}:
        raise ValueError("Declare all core electrical ratings")
    for key, value in ratings.items():
        _finite(value, key)
    if any(ratings[k] <= 0 for k in ("v_abs_max", "i_abs_max", "i_cont")) or ratings["i_cont"] > ratings["i_abs_max"]:
        raise ValueError("Ratings require positive limits with continuous current within the absolute limit")
    allowed_thermal = {"housing_area", "cooling_area", "r_th_cs", "r_th_switch_cs", "r_th_diode_cs", "t_c_max"}
    if not {"housing_area", "cooling_area"} <= set(thermal_properties) <= allowed_thermal:
        raise ValueError("Declare core housing and cooling areas explicitly")
    for key, value in thermal_properties.items():
        value = _finite(value, key, nullable=key not in ("housing_area", "cooling_area"))
        if value is not None and ((key.endswith("area") and value <= 0) or (key.startswith("r_th") and value < 0)):
            raise ValueError("Areas must be positive. Thermal resistances must be nonnegative")
    if not isinstance(curves, list) or not 1 <= len(curves) <= 6:
        raise ValueError("Select one to six explicit branch curves")
    mechanisms = {branch + kind for branch in ("forward", "reverse") for kind in ("ConductionTrace", "TurnOnTrace", "TurnOffTrace")}
    used, destinations = set(), set()
    checked = []
    diagnostics = _diagnostics(values)
    for choice in curves:
        required_choice = {"kind", "branch", "target", "source", "coordinate_convention", "sample_indices", "sample_order", "zero_policy", "conditions", "evidence_reference"}
        if not isinstance(choice, dict) or not required_choice <= set(choice) <= required_choice | {"data_status", "ideal_model_scope", "source_zero_acknowledgement", "extrapolation_policy"}:
            raise ValueError("Declare the exact supported curve mapping fields")
        if not isinstance(choice["evidence_reference"], str) or not choice["evidence_reference"].strip() or len(choice["evidence_reference"]) > 2000:
            raise ValueError("Declare a nonblank evidence reference for each selected curve")
        kind, branch, target = choice["kind"], choice["branch"], choice["target"]
        if kind not in ("conduction", "turn-on", "turn-off") or branch not in ("forward", "reverse") or target not in ("switch", "diode"):
            raise ValueError("Select supported curve kind, branch and component target")
        if choice["source"] not in ("original-trace", "combined"):
            raise ValueError("Select original-trace or combined curve source")
        mechanism = branch + {"conduction": "ConductionTrace", "turn-on": "TurnOnTrace", "turn-off": "TurnOffTrace"}[kind]
        if mechanism in used or (target, kind) in destinations:
            raise ValueError("Each source branch and target mechanism requires a distinct explicit mapping")
        if target == "diode" and kind == "turn-on":
            raise ValueError("The existing diode core has no turn-on representation. Preserve this branch in the archive")
        if choice["zero_policy"] not in ("reject-zero", "recorded-branch-value"):
            raise ValueError("Select a supported zero policy")
        x, y, reference, path = _curve_source(sc, choice, diagnostics)
        conditions = choice["conditions"]
        required = {"t_j_c", "gate_voltage_v", "gate_off_voltage_v", "gate_resistance_ohm", "supply_voltage_v", "commutation_device"}
        if set(conditions) != required:
            raise ValueError("Declare independent curve conditions, including explicit unknown null values")
        _finite(conditions["t_j_c"], "junction temperature")
        for key in required - {"commutation_device", "t_j_c"}:
            _finite(conditions[key], key, nullable=True)
        if conditions["commutation_device"] is not None and not isinstance(conditions["commutation_device"], str):
            raise ValueError("Commutation device must be text or unknown null")
        if target == "switch" and conditions["gate_voltage_v"] is None:
            raise ValueError("Declare the switch gate voltage explicitly")
        if kind != "conduction":
            supply, resistance = conditions["supply_voltage_v"], conditions["gate_resistance_ohm"]
            if supply is None or supply <= 0 or resistance is None or resistance < 0 or conditions["gate_voltage_v"] is None:
                raise ValueError("Energy curves require positive supply voltage, known gate voltage and nonnegative gate resistance")
            if reference is not None and reference != supply:
                raise ValueError("Declared supply voltage conflicts with this original curve reference")
        used.add(mechanism)
        destinations.add((target, kind))
        checked.append((choice, x, y, path))
    if set(omission_reasons) != mechanisms - used or any(not isinstance(v, str) or not v.strip() for v in omission_reasons.values()):
        raise ValueError("Give a reason for every omitted branch mechanism")
    declarations = {"identity": identity, "ratings": ratings, "thermal_properties": thermal_properties,
                    "curves": curves, "omission_reasons": omission_reasons, "units": units,
                    "declaration_source": declaration_source}
    # Original bytes preserve large images and unknown fields without duplicating
    # every decoded array in the native core document.
    provenance = {"schema": SCHEMA, "version": VERSION, "source_name": source_name,
                  "source_sha256": sha256(raw).hexdigest(),
                  "source_base64": base64.b64encode(raw).decode("ascii"),
                  "declarations": json.loads(json.dumps(declarations, allow_nan=False))}
    result = Transistor(TransistorMetadata(**identity), ElectricalRatings(**ratings), ThermalProperties(**thermal_properties))
    result.switch.metadata["legacy_mat_archive"] = provenance
    for choice, x, y, path in checked:
        target = getattr(result, choice["target"])
        conditions = choice["conditions"]
        if choice["kind"] == "conduction":
            target.channel_data.append(ChannelCharacteristics(conditions["t_j_c"], np.array([y, x]), conditions["gate_voltage_v"]))
        else:
            record = SwitchingLossData(
                "graph_i_e", conditions["t_j_c"], conditions["supply_voltage_v"], conditions["gate_voltage_v"],
                r_g=conditions["gate_resistance_ohm"], v_g_off=conditions["gate_off_voltage_v"],
                commutation_device=conditions["commutation_device"], graph_i_e=np.array([x, y]),
                comment=json.dumps({"source_path": path, "branch": choice["branch"], "mapping": choice["coordinate_convention"]}))
            destination = target.e_rr_data if choice["target"] == "diode" else target.e_on_data if choice["kind"] == "turn-on" else target.e_off_data
            destination.append(record)
    return result
