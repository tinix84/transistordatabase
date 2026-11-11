"""Bounded, source-preserving waveform and supplied-integral import operations."""
from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import math
import platform
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np

from transistordatabase.core.models import MeasurementEvidence, RawMeasurementData
from transistordatabase.core.models import SwitchingLossData
from transistordatabase.core.serialization import core_from_json_document, core_to_json_document
from transistordatabase.utils.ltspice_dpt import DPTConfig, LTspiceDPT

MAX_BYTES = 8_000_000
MAX_POINTS = 100_000
MAX_RECORDS = 10_000
MAX_CELLS = 500_000
MAX_OUTPUT_BYTES = 64_000_000
UNITS = {"s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
         "V": 1., "mV": 1e-3, "kV": 1e3, "A": 1., "mA": 1e-3,
         "kA": 1e3, "J": 1., "mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9}
DIMENSIONS = {"time": {"s", "ms", "us", "ns"},
              "voltage": {"V", "mV", "kV"}, "gate": {"V", "mV", "kV"},
              "current": {"A", "mA", "kA"}, "energy": {"J", "mJ", "uJ", "nJ"}}


class MeasurementLimitError(ValueError):
    """The input or output exceeds the bounded import size."""


def _bytes(encoded):
    if len(encoded) > 4 * ((MAX_BYTES + 2) // 3):
        raise MeasurementLimitError("Source exceeds the 8 MB import limit")
    try:
        result = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Source requires valid base64 bytes") from exc
    if not result or len(result) > MAX_BYTES:
        raise MeasurementLimitError("Source must contain 1 through 8000000 bytes")
    return result


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean values are not numeric measurements")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Measurement value must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError("Measurement value must be a finite real number")
    return number


def _scale(unit, dimension):
    if unit not in DIMENSIONS[dimension]:
        raise ValueError(f"Unsupported {dimension} unit: {unit}")
    return UNITS[unit]


def _step_values(values):
    """Preserve scalar step metadata without nonfinite JSON values or implicit casts."""
    for value in values.values():
        if isinstance(value, (int, float, np.number)):
            _number(value)
        elif not isinstance(value, str):
            raise ValueError("Step metadata requires finite real numbers or original text")
    return values


def _staged(data, suffix, companion=None):
    # Parser APIs require paths. Preserve each unique staging directory for audit.
    folder = Path(tempfile.gettempdir()) / ("tdb-measurement-" + uuid.uuid4().hex)
    folder.mkdir(exist_ok=False)
    path = folder / ("source" + suffix)
    with path.open("xb") as stream:
        stream.write(data)
    if companion is not None:
        with path.with_suffix(".log").open("xb") as stream:
            stream.write(companion)
    return path


def _waveform(columns):
    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()
              if values is not None}
    count = len(arrays["time_s"])
    if count < 2 or count > MAX_POINTS:
        raise MeasurementLimitError("Select 2 through 100000 samples")
    if any(a.ndim != 1 or len(a) != count or not np.all(np.isfinite(a))
           for a in arrays.values()):
        raise ValueError("Waveforms require aligned finite one-dimensional samples")
    if np.any(np.diff(arrays["time_s"]) <= 0):
        raise ValueError("Time must increase strictly within the selected step")
    return {name: value.tolist() if value is not None else None
            for name, value in {**columns, **arrays}.items()}


def _csv(data, request):
    options, mappings = request["csv"], request["mappings"]
    if options["delimiter"] == options["decimal"]:
        raise ValueError("CSV delimiter and decimal separator must differ")
    try:
        text = data.decode(options["encoding"])
        reader = csv.reader(io.StringIO(text, newline=""),
                            delimiter=options["delimiter"], strict=True)
        headers = next(reader)
        if not headers or any(not name.strip() for name in headers) or len(set(headers)) != len(headers):
            raise ValueError("CSV headers must be nonempty and unique")
        rows = []
        for row in reader:
            if len(row) != len(headers):
                raise ValueError("CSV row lengths must match the header")
            rows.append(row)
            if len(rows) * len(headers) > MAX_CELLS or len(rows) > MAX_POINTS:
                raise MeasurementLimitError("CSV sample or expanded-cell limit exceeded")
    except (UnicodeError, LookupError, csv.Error, StopIteration) as exc:
        raise ValueError("Cannot parse CSV with the declared encoding and delimiter") from exc
    step_name = options["step_column"]
    steps = []
    if step_name is not None:
        if step_name not in headers:
            raise ValueError("Mapped CSV step column is absent")
        index = headers.index(step_name)
        steps = list(dict.fromkeys(row[index] for row in rows))
        if options["step_value"] is None:
            return {"traces": headers, "steps": steps, "measurements": []}, None, {}
        if options["step_value"] not in steps:
            raise ValueError("Selected CSV step value is absent")
        rows = [row for row in rows if row[index] == options["step_value"]]
    available = {"traces": headers, "steps": steps, "measurements": []}
    if all(mapping is None for mapping in mappings.values()):
        return available, None, {}
    if any(mappings[key] is None for key in ("time", "voltage", "current")):
        raise ValueError("Map time, voltage, and current explicitly")
    columns = {}
    for key, output in (("time", "time_s"), ("voltage", "voltage_v"),
                        ("current", "current_a"), ("gate", "gate_v")):
        mapping = mappings[key]
        if mapping is None:
            columns[output] = None
            continue
        if mapping["name"] not in headers:
            raise ValueError(f"Mapped CSV column is absent: {mapping['name']}")
        factor = _scale(mapping["unit"], key) * mapping["polarity"]
        index = headers.index(mapping["name"])
        values = [row[index] for row in rows]
        if options["decimal"] == "," and any("." in value for value in values):
            raise ValueError("CSV numeric values conflict with the declared decimal separator")
        columns[output] = [_number(value.replace(options["decimal"], ".")) * factor
                           for value in values]
    return available, _waveform(columns), ({step_name: options["step_value"]} if step_name else {})


def _raw(data, companion, request):
    try:
        from spicelib.raw.raw_read import RawRead
    except ImportError as exc:
        raise ValueError("RAW import requires the optional spicelib parser in this server") from exc
    # Inspect only the bounded textual header before the path-based library reads arrays.
    encoding = "utf-16-le" if data[:2] in (b"T\x00", b"\xff\xfe") else "utf-8-sig"
    header = data[:min(len(data), 100_000)].decode(encoding, errors="replace")
    header = re.split(r"(?im)^(?:Binary|Values):", header, maxsplit=1)[0]
    def declared(label):
        match = re.search(r"(?im)^" + label + r":\s*(\d+)\s*$", header)
        if match is None:
            raise ValueError("RAW header lacks a declared sample or variable count")
        return int(match[1])
    points, traces = declared(r"No\. Points"), declared(r"No\. Variables")
    encoded_label = "No. Points:".encode("utf-16-le" if encoding == "utf-16-le" else "ascii")
    if data.count(encoded_label) != 1:
        raise ValueError("Select a single transient RAW plot")
    if points < 2 or traces < 2 or points * traces > MAX_CELLS:
        raise MeasurementLimitError("RAW expanded points times traces exceed the import limit")
    stepped = bool(re.search(r"(?im)^Flags:.*\bstepped\b", header))
    if stepped and companion is None:
        raise ValueError("Stepped RAW requires its matching companion log")
    path = _staged(data, ".raw", companion)
    mappings = request["mappings"]
    names = [mapping["name"] for mapping in mappings.values() if mapping is not None]
    try:
        reader = RawRead(path, traces_to_read=names or None, dialect="ltspice", verbose=False)
        available = {"traces": reader.get_trace_names(), "steps": list(reader.get_steps()),
                     "measurements": [], "staged_path": str(path)}
        if len(available["steps"]) > MAX_RECORDS:
            raise MeasurementLimitError("RAW step count exceeds 10000 records")
        step_values = {}
        if stepped or companion is not None:
            from spicelib.log.ltsteps import LTSpiceLogReader
            log = LTSpiceLogReader(str(path.with_suffix(".log")), read_measures=False)
            if stepped and (not list(log.get_step_vars()) or log.step_count != len(available["steps"])):
                raise ValueError("Stepped RAW requires matching explicit companion step records")
        step = request["step_index"]
        if not names:
            return available, None, {}, []
        if step is None and len(available["steps"]) == 1:
            step = 0
        if step not in available["steps"]:
            raise ValueError("Select an explicit available RAW step")
        if any(mappings[key] is None for key in ("time", "voltage", "current")):
            raise ValueError("Map RAW time, voltage, and current explicitly")
        columns = {}
        normalizations = []
        available["trace_types"] = {}
        for key, output in (("time", "time_s"), ("voltage", "voltage_v"),
                            ("current", "current_a"), ("gate", "gate_v")):
            mapping = mappings[key]
            if mapping is None:
                columns[output] = None
                continue
            if mapping["name"] not in available["traces"]:
                raise ValueError(f"Mapped RAW trace is absent: {mapping['name']}")
            trace = reader.get_trace(mapping["name"])
            available["trace_types"][mapping["name"]] = trace.whattype
            raw_values = trace.get_wave(step)
            if np.iscomplexobj(raw_values):
                raise ValueError("DPT import requires real transient traces")
            if key == "time":
                if mapping["name"] != reader.get_trace_names()[0] or trace.whattype != "time":
                    raise ValueError("Map the RAW time axis explicitly")
                values = reader.get_time_axis(step)
                normalizations.append("spicelib applies absolute time for LTspice transient compression before step validation")
            else:
                expected = "current" if key == "current" else "voltage"
                accepted = {"current", "device_current"} if key == "current" else {"voltage"}
                if trace.whattype not in accepted:
                    raise ValueError(f"RAW trace type must be {expected} for {key}")
                values = raw_values
            # LTspice stores trace samples in SI. Declare SI units to avoid double conversion.
            expected_unit = {"time": "s", "voltage": "V", "current": "A", "gate": "V"}[key]
            if mapping["unit"] != expected_unit:
                raise ValueError(f"RAW {key} units must declare stored SI unit {expected_unit}")
            columns[output] = np.asarray(values) * mapping["polarity"]
        if companion is not None:
            step_values = _step_values({name: log.stepset[name][step] for name in log.get_step_vars()})
        return available, _waveform(columns), step_values, normalizations
    except MeasurementLimitError:
        raise
    except Exception as exc:
        raise ValueError(f"Unsupported or corrupt RAW data: {exc}") from exc


def _log(data, request):
    if data.count(b"\n") > MAX_CELLS or data.count(b"\t") > MAX_CELLS:
        raise MeasurementLimitError("Log declared rows exceed the import limit")
    try:
        from spicelib.log.ltsteps import LTSpiceLogReader
    except ImportError as exc:
        raise ValueError("Log import requires the optional spicelib parser in this server") from exc
    path = _staged(data, ".log")
    try:
        reader = LTSpiceLogReader(str(path))
        names, step_names = list(reader.get_measure_names()), list(reader.get_step_vars())
        count = max([len(values) for values in reader.dataset.values()] + [reader.step_count, 1])
        if count > MAX_RECORDS or sum(len(values) for values in reader.dataset.values()) > MAX_CELLS:
            raise MeasurementLimitError("Log expanded measurements exceed the import limit")
        available = {"traces": [], "steps": list(range(count)), "measurements": names,
                     "step_variables": step_names, "staged_path": str(path)}
        selected = request["step_index"]
        if selected is not None and selected >= count:
            raise ValueError("Selected log step is absent")
        mapping = request["log"]
        # Expression text is metadata. The installed parser alone supplies numeric results.
        text = data.decode(reader.encoding)
        expressions = {}
        active = None
        ordinals = {}
        for line in text.splitlines():
            single = re.match(r"^\s*(\w+):[ \t]*(.+)=[^=]+$", line)
            if single and single[1].lower() in reader.dataset:
                expressions[single[1].lower()] = single[2].strip()
            if line.lower().startswith("measurement: "):
                active = line.split(":", 1)[1].strip().lower()
                ordinals[active] = []
            elif active and "\t" in line and line.split("\t", 1)[0].strip().isdigit():
                ordinals[active].append(int(line.split("\t", 1)[0]))
            elif active and "\t" in line:
                tokens = line.split("\t")
                if len(tokens) > 1:
                    expressions[active] = tokens[1]
        for name, numbers in ordinals.items():
            if numbers != list(range(1, len(numbers) + 1)):
                raise ValueError(f"Log measurement step ordinals must be unique and sequential: {name}")
        if count * sum(mapping["energy_" + event] is not None for event in ("on", "off")) > MAX_RECORDS:
            raise MeasurementLimitError("Log output event count exceeds 10000 records")
        def value(name, index, dimension, unit):
            if name is None:
                return None
            lookup = name.lower()
            values = reader.stepset.get(lookup, reader.dataset.get(lookup))
            if values is None or len(values) != count:
                raise ValueError(f"Mapped log field is absent or partial: {name}")
            return _number(_number(values[index]) * _scale(unit, dimension))
        records = []
        for event, name in (("turn_on", mapping["energy_on"]), ("turn_off", mapping["energy_off"])):
            if name is None:
                continue
            for index in range(count):
                record = {"event": event, "step_index": index, "status": "supplied-integral",
                          "expression": expressions.get(name.lower()),
                          "step_values": _step_values({key: reader.stepset[key][index] for key in step_names})}
                try:
                    record["energy_j"] = value(name, index, "energy", mapping["energy_unit"])
                    record["current_a"] = value(mapping["current"], index, "current", mapping["current_unit"])
                    record["bus_voltage_v"] = value(mapping["bus"], index, "voltage", mapping["bus_unit"])
                    start = reader.dataset.get(name.lower() + "_from")
                    stop = reader.dataset.get(name.lower() + "_to")
                    record["window_s"] = ([_number(start[index]), _number(stop[index])]
                                          if start is not None and stop is not None else None)
                    if record["window_s"] and record["window_s"][1] <= record["window_s"][0]:
                        raise ValueError("Log integral bounds must increase strictly")
                except (ValueError, IndexError, TypeError) as exc:
                    record.update(status="failed", energy_j=None, current_a=None,
                                  bus_voltage_v=None, window_s=None, error=str(exc))
                records.append(record)
        return available, records
    except MeasurementLimitError:
        raise
    except Exception as exc:
        raise ValueError(f"Unsupported or corrupt log data: {exc}") from exc


def preview_measurement(request):
    """Decode only the selected source and mappings, without simulator execution."""
    data = _bytes(request["content_base64"])
    companion = (_bytes(request["companion_log_base64"])
                 if request["companion_log_base64"] is not None else None)
    if request["source_kind"] == "simulation" and not (request["disclosure"] or "").strip():
        raise ValueError("Simulation sources require a model scope disclosure")
    fmt = request["format"]
    if fmt in ("raw", "log") and request["source_kind"] not in ("simulation", "unknown"):
        raise ValueError("LTspice RAW and log sources must be declared simulation or unknown")
    try:
        parser_version = version("spicelib") if fmt != "csv" else platform.python_version()
    except PackageNotFoundError as exc:
        raise ValueError("Install the declared spicelib dependency to import RAW or log files") from exc
    source = MeasurementEvidence(source_kind=request["source_kind"], filename=request["filename"],
        sha256=hashlib.sha256(data).hexdigest(), parser="python.csv" if fmt == "csv" else "spicelib",
        parser_version=parser_version,
        original_base64=request["content_base64"], companion_log_base64=request["companion_log_base64"],
        companion_log_filename=request.get("companion_log_filename"),
        companion_log_sha256=hashlib.sha256(companion).hexdigest() if companion else None,
        step_index=request["step_index"], trace_mapping={"signals": request["mappings"],
            "csv": request["csv"] if fmt == "csv" else None,
            "log": request["log"] if fmt == "log" else None},
        disclosure=request["disclosure"])
    wave, records = None, []
    if fmt == "csv":
        available, wave, source.step_values = _csv(data, request)
    elif fmt == "raw":
        available, wave, source.step_values, source.parser_normalizations = _raw(data, companion, request)
        source.trace_mapping["stored_trace_types"] = available.get("trace_types", {})
        if wave is not None and source.step_index is None:
            source.step_index = 0
    else:
        available, records = _log(data, request)
    from transistordatabase.waveform_losses import waveform_statistics
    statistics = ({name: waveform_statistics(wave["time_s"], values)
                   for name, values in wave.items() if name != "time_s" and values is not None}
                  if wave else None)
    return {"source": asdict(source), "available": available, "waveform": wave,
            "statistics": statistics, "log_records": records, "warnings": []}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def extract_measurement(request, conditions, device_document=None):
    """Create reviewable raw data and event points, with optional native archive attachment."""
    original_size = len(request["content_base64"]) + len(request["companion_log_base64"] or "")
    native_size = len(json.dumps(device_document)) if device_document is not None else 0
    if native_size > 40_000_000:
        raise MeasurementLimitError("Native archive input exceeds 40 MB")
    if original_size * (24 if device_document is not None else 10) + native_size * 6 + MAX_CELLS * 64 > MAX_OUTPUT_BYTES:
        raise MeasurementLimitError("Extraction evidence and native archive exceed the 64 MB output budget")
    preview = preview_measurement(request)
    source = MeasurementEvidence(**preview["source"])
    source.t_j = conditions["t_j"]
    source.bus_voltage_v = conditions["bus_voltage_v"]
    source.v_g_on = conditions["v_g_on"]
    source.v_g_off = conditions["v_g_off"]
    source.r_g_on = conditions["r_g_on"]
    source.r_g_off = conditions["r_g_off"]
    source.commutation_device = conditions["commutation_device"]
    source.manual_override = conditions.get("manual_override")
    points = {"turn_on": [], "turn_off": []}
    analysis, raw = None, None
    wave = preview["waveform"]
    if request["format"] != "log":
        if wave is None or wave["gate_v"] is None:
            raise ValueError("DPT extraction requires explicitly mapped time, voltage, current, and gate traces")
        cfg = DPTConfig(v_dc=conditions["bus_voltage_v"],
                       v_g_on=conditions["v_g_on"], v_g_off=conditions["v_g_off"],
                       r_g_on=conditions["r_g_on"], r_g_off=conditions["r_g_off"])
        dpt = LTspiceDPT(cfg)
        result = dpt.analyze_waveforms(wave["time_s"], wave["voltage_v"],
                                       wave["current_a"], wave["gate_v"])
        analysis = asdict(result)
        source.method = "waveform-local-edge"
        on, off = dpt.extract_switching_loss_data(result, t_j=conditions["t_j"], evidence=source)
        points = {"turn_on": [on], "turn_off": [off]}
        raw = RawMeasurementData(dataset_type="dpt-waveform", t_j=source.t_j,
            v_supply=source.bus_voltage_v, v_g=source.v_g_on, v_g_off=source.v_g_off,
            r_g=[source.r_g_on] if source.r_g_on is not None else None,
            r_g_off=[source.r_g_off] if source.r_g_off is not None else None,
            commutation_device=source.commutation_device,
            time_s=np.array(wave["time_s"]), voltage_v=np.array(wave["voltage_v"]),
            current_a=np.array(wave["current_a"]), gate_v=np.array(wave["gate_v"]),
            evidence=replace(source, method="waveform-local-edge",
                             event="paired-dpt-events", thresholds=result.event_evidence,
                             convention=result.energy_window_convention))
    else:
        source.method = "supplied-log-integral"
        if request["step_index"] is None:
            raise ValueError("Select an explicit log step before deriving a supplied-integral point")
        for record in preview["log_records"]:
            if record["step_index"] != request["step_index"] or record["status"] == "failed":
                continue
            event = record["event"]
            evidence = replace(source, step_values=record["step_values"], event=event,
                method="supplied-log-integral", window_s=tuple(record["window_s"]) if record["window_s"] else None,
                convention="Original simulator-supplied integral; no waveform local-edge extraction",
                expression=record["expression"], local_current_a=record["current_a"],
                bus_voltage_v=record["bus_voltage_v"])
            point = SwitchingLossData(dataset_type="single", t_j=source.t_j,
                v_supply=record["bus_voltage_v"], v_g=source.v_g_on, v_g_off=source.v_g_off,
                r_g=source.r_g_on if event == "turn_on" else source.r_g_off,
                i_x=record["current_a"], e_x=record["energy_j"], evidence=evidence,
                commutation_device=source.commutation_device)
            points[event].append(point)
        if not any(points.values()):
            raise ValueError("Selected log step has no valid mapped supplied integrals")
    native = None
    if device_document is not None:
        transistor = core_from_json_document(device_document)
        if raw:
            transistor.switch.raw_measurement_data.append(raw)
        transistor.switch.e_on_data.extend(points["turn_on"])
        transistor.switch.e_off_data.extend(points["turn_off"])
        native = core_to_json_document(transistor)
    response = {**preview, "source": asdict(source), "status": "review-not-persisted",
            "analysis": analysis, "raw_record": _plain(asdict(raw)) if raw else None,
            "points": {key: [_plain(asdict(point)) for point in values] for key, values in points.items()},
            "core_json_document": native,
            "core_json_text": json.dumps(native, ensure_ascii=False, allow_nan=False) if native else None}
    review_text = json.dumps(response, ensure_ascii=False, allow_nan=False)
    if len(review_text.encode("utf-8")) * 3 > MAX_OUTPUT_BYTES:
        raise MeasurementLimitError("Expanded review JSON exceeds the 64 MB output budget")
    response["review_json_text"] = review_text
    return response
