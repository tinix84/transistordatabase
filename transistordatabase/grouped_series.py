"""Bounded projections of accepted grid rows for physical-series exports."""
from __future__ import annotations

import csv
import io
import json
import re

from transistordatabase.waveform_losses import (
    series_spectrum, waveform_scalar_statistics,
)

METHOD_VERSION = "grouped-physical-series-v1"
SERIES_UNITS = {"current_a": "A", "gate_v": "V", "device_voltage_v": "V",
                "bus_voltage_v": "V"}
MAX_TIME_ROWS = 1024
MAX_FREQUENCY_ROWS = 10240
TIME_FIELDS = ("sample_count", "minimum", "maximum", "arithmetic_mean",
               "sample_standard_deviation", "discrete_rms", "time_weighted_mean",
               "time_weighted_rms")
FREQUENCY_FIELDS = ("bin_index", "frequency_hz", "magnitude", "real", "imaginary",
                    "phase_rad", "retained_bin_count")


def _csv_cell(value):
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value if re.match(r"^\s*[=+\-@\t\r]", value) else value
    return str(value)


def _csv_rows(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_cell(row.get(column)) for column in columns])
    return stream.getvalue()


def _encoded_size(value):
    return len(json.dumps(value, allow_nan=False).encode("utf-8"))


def project_grouped_series(rows, grid_input_sha256, available_bytes, limit_error):
    """Project one accepted snapshot without replaying its electrical calculation.

    The preflight counts escaped metadata in both JSON rows and CSV strings.
    Numeric and diagnostic reserves bound expansion before row materialization.
    """
    descriptors, projected_bytes = [], 4096
    for row in rows:
        evidence = row["inputs"]
        point, config = evidence["operating_point"], evidence["configuration"]
        waveform = evidence["waveform_request"]["waveform"]
        settings = evidence["waveform_request"].get("spectrum")
        for series, unit in SERIES_UNITS.items():
            if waveform.get(series) is None:
                continue
            metadata = {
                "configuration_id": row["configuration_id"],
                "configuration_source": config["source"],
                "operating_point_id": row["operating_point_id"],
                "operating_point_source": point["source"], "weight": row["weight"],
                "series": series, "unit": unit, "status": row["status"],
                "pair_status": row["status"],
                "error": row["error"], "input_sha256": row["input_sha256"],
                "method_version": METHOD_VERSION, "grid_input_sha256": grid_input_sha256,
                "window": settings.get("window") if settings else None,
                "detrend": settings.get("detrend") if settings else None,
                "relative_threshold": settings.get("relative_threshold", .1) if settings else None,
                "phase_reference": "first supplied sample" if settings else None,
            }
            # A finite IEEE-754 scalar needs fewer than 32 printable characters.
            # Reserve a 512-character diagnostic for independent series failures.
            template = dict(metadata, error=(row["error"] or "") + "x" * 512,
                            status="not_requested")
            frequency_count = 1
            if settings is not None and row["status"] == "computed":
                frequency_count = min(10, len(waveform["time_s"]) // 2 + 1)
                if settings.get("relative_threshold", .1) == 1:
                    frequency_count = 1
                if series == "current_a" and row["result"].get("spectrum") is not None:
                    frequency_count = max(1, len(row["result"]["spectrum"]["selected_bin_indices"]))
            for fields, count in ((TIME_FIELDS, 1), (FREQUENCY_FIELDS, frequency_count)):
                trial = {**template, **dict.fromkeys(fields, "9" * 32)}
                csv_text = _csv_rows([trial], list(trial))
                json_text = json.dumps(trial, allow_nan=False)
                projected_bytes += count * (
                    _encoded_size(trial) + _encoded_size(json_text) + _encoded_size(csv_text) + 8)
            if projected_bytes > available_bytes:
                raise limit_error("Grouped series metadata exceeds the complete response byte budget")
            descriptors.append((row, waveform, settings, series, metadata))
    if (len(descriptors) > MAX_TIME_ROWS or
            len(descriptors) * 10 > MAX_FREQUENCY_ROWS):
        raise limit_error("Grouped series exceeds 1024 time rows or 10240 frequency rows")

    time_rows, frequency_rows = [], []
    for row, _waveform, settings, series, metadata in descriptors:
        time_row = {**metadata, **dict.fromkeys(TIME_FIELDS)}
        frequency_base = {**metadata, **dict.fromkeys(FREQUENCY_FIELDS)}
        if row["status"] != "computed":
            time_rows.append(time_row)
            if settings is None:
                frequency_base.update(status="not_requested")
            frequency_rows.append(frequency_base)
            continue
        # Successful rows supply the accepted, scalar-aligned physical samples.
        signals = row["result"]["signals"]
        try:
            stats = waveform_scalar_statistics(signals["time_s"], signals[series])
            time_row.update({field: stats[field] for field in TIME_FIELDS
                             if field not in {"time_weighted_mean", "time_weighted_rms"}})
            time_row.update(time_weighted_mean=stats["mean"], time_weighted_rms=stats["rms"])
        except (ValueError, OverflowError, FloatingPointError) as exc:
            time_row.update(status="unsupported", error=str(exc))
        time_rows.append(time_row)
        if settings is None:
            frequency_base.update(status="not_requested", error=None)
            frequency_rows.append(frequency_base)
            continue
        try:
            if series == "current_a" and row["result"].get("spectrum") is not None:
                spectrum = row["result"]["spectrum"]
                keys = ("amplitude_a", "real_a", "imaginary_a")
            else:
                spectrum = series_spectrum(signals["time_s"], signals[series], **settings)
                keys = ("magnitude", "real", "imaginary")
            frequency_base["retained_bin_count"] = len(spectrum["selected_bin_indices"])
            for index in spectrum["selected_bin_indices"]:
                frequency_rows.append({**frequency_base, "bin_index": index,
                    "frequency_hz": spectrum["frequency_hz"][index],
                    "magnitude": spectrum[keys[0]][index], "real": spectrum[keys[1]][index],
                    "imaginary": spectrum[keys[2]][index], "phase_rad": spectrum["phase_rad"][index]})
            if not spectrum["selected_bin_indices"]:
                frequency_rows.append(frequency_base)
        except (ValueError, OverflowError, FloatingPointError) as exc:
            frequency_rows.append(dict(frequency_base, status="unsupported", error=str(exc)))
    metadata_columns = list(descriptors[0][-1]) if descriptors else []
    output = {"method_version": METHOD_VERSION,
              "scope": "All supplied physical series; point FFT settings apply to every series",
              "time_rows": time_rows, "frequency_rows": frequency_rows,
              "time_json": json.dumps(time_rows, allow_nan=False, separators=(",", ":")),
              "frequency_json": json.dumps(frequency_rows, allow_nan=False, separators=(",", ":")),
              "time_csv": _csv_rows(time_rows, metadata_columns + list(TIME_FIELDS)),
              "frequency_csv": _csv_rows(frequency_rows, metadata_columns + list(FREQUENCY_FIELDS)),
              "null_csv": "NULL"}
    if _encoded_size(output) > available_bytes:
        raise limit_error("Grouped series exceeds the complete response byte budget")
    return output
