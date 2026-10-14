"""Read the explicitly selected traces from LTspice transient RAW uploads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
import re
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np

MAX_RAW_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_RAW_REQUEST_BODY_BYTES = MAX_RAW_UPLOAD_BYTES + 1024 * 1024
MAX_RAW_POINTS = 1_000_000
MAX_RAW_PLOTS = 16
MAX_RAW_STEPS_PER_PLOT = 100
MAX_SELECTED_STEPS = 50
MAX_EVENT_SAMPLES = 250_000
MAX_TOTAL_EVENT_SAMPLES = 250_000
MAX_PREVIEW_POINTS = 800
MAX_TRACE_COUNT = 128
MAX_RAW_ARRAY_BYTES = 32 * 1024 * 1024
MAX_ASCII_VALUE_RECORDS = 4_000_000


class DptRawImportError(ValueError):
    """A supported-format, user-correctable DPT RAW import error."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ParsedRaw:
    """Header and optional selected waveform arrays from one RAW file."""

    filename: str
    plot_name: str
    trace_names: list[str]
    sample_count: int
    time_values: np.ndarray | None = None
    trace_values: dict[str, np.ndarray] | None = None
    plot_index: int | None = None
    step_ranges: list[dict[str, Any]] | None = None
    plots: list[dict[str, Any]] | None = None
    is_stepped: bool = False
    plot_headers: list[dict[str, Any]] | None = None


def _safe_filename(filename: str | None) -> str:
    """Keep only the uploaded basename for validation and provenance."""
    name = PureWindowsPath(filename or "").name
    name = PurePosixPath(name).name
    if not name:
        raise DptRawImportError("Choose an LTspice .raw file to upload.")
    return name


def _raw_encoding(content: bytes) -> str:
    """Detect the two header encodings emitted by LTspice/RawWrite."""
    return "utf_16_le" if len(content) > 1 and content[1] == 0 else "utf-8"


def _header_fields(header: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().casefold()] = value.strip()
    return fields


def _find_data_marker(
    content: bytes,
    start: int,
    encoding: str,
) -> tuple[int, str, int]:
    """Find the earliest line-start Binary/Values marker after a plot offset."""
    newline = "\n".encode(encoding)
    candidates: list[tuple[int, str, int]] = []
    for marker_name in ("Binary:", "Values:"):
        marker = marker_name.encode(encoding)
        cursor = start
        while True:
            marker_index = content.find(marker, cursor)
            if marker_index < 0:
                break
            line_start = marker_index == start or (
                marker_index >= start + len(newline)
                and content[marker_index - len(newline) : marker_index] == newline
            )
            if line_start:
                line_end = content.find(newline, marker_index + len(marker))
                if line_end >= 0:
                    candidates.append(
                        (
                            marker_index,
                            marker_name[:-1].casefold(),
                            line_end + len(newline),
                        )
                    )
                    break
            cursor = marker_index + len(marker)
    if not candidates:
        raise DptRawImportError(
            "The upload is not a valid LTspice .raw file with a complete Binary or "
            "Values data header. Re-export the LTspice result and try again."
        )
    return min(candidates, key=lambda item: item[0])


def _ascii_line(
    content: bytes,
    cursor: int,
    encoding: str,
) -> tuple[bytes, str, int]:
    """Read one bounded encoded text line without allocating its whole plot."""
    newline = "\n".encode(encoding)
    end = content.find(newline, cursor)
    if end < 0:
        if cursor >= len(content):
            raise DptRawImportError(
                "An ASCII RAW Values section ends before its declared samples. "
                "Re-export the complete LTspice result."
            )
        raw_line = content[cursor:]
        line_bytes = raw_line
        next_cursor = len(content)
    else:
        raw_line = content[cursor : end + len(newline)]
        line_bytes = raw_line[: end - cursor]
        next_cursor = end + len(newline)
    try:
        line = line_bytes.decode(encoding).strip()
    except UnicodeDecodeError as error:
        raise DptRawImportError(
            "An ASCII RAW value line has unsupported text encoding."
        ) from error
    return raw_line, line, next_cursor


def _validate_ascii_value(value: str, *, complex_value: bool) -> None:
    """Validate one PyLTSpice-compatible scalar without retaining it."""
    try:
        if complex_value:
            parts = value.split(",")
            if len(parts) != 2:
                raise ValueError("complex value must contain real and imaginary parts")
            float(parts[0])
            float(parts[1])
        else:
            float(value)
    except ValueError as error:
        raise DptRawImportError(
            "An ASCII RAW Values section contains a malformed numeric value. "
            "Re-export the LTspice result."
        ) from error


def _scan_raw_headers(content: bytes) -> list[dict[str, Any]]:
    """Validate plot declarations and binary payload bounds before PyLTSpice."""
    encoding = _raw_encoding(content)
    newline = "\n".encode(encoding)
    plots: list[dict[str, Any]] = []
    cursor = 0
    total_points = 0
    total_traces = 0
    total_ascii_array_bytes = 0
    total_ascii_value_records = 0
    while cursor < len(content):
        while content.startswith(newline, cursor):
            cursor += len(newline)
        if cursor >= len(content):
            break
        if len(plots) >= MAX_RAW_PLOTS:
            raise DptRawImportError(
                f"The file contains more than {MAX_RAW_PLOTS} result plots; export "
                "a smaller LTspice result."
            )
        header_start = cursor
        marker_index, data_kind, data_start = _find_data_marker(
            content, header_start, encoding
        )
        try:
            header_text = content[header_start:marker_index].decode(encoding)
        except UnicodeDecodeError as error:
            raise DptRawImportError(
                "The RAW plot header encoding is unsupported; re-export the file "
                "from LTspice using its native format."
            ) from error
        fields = _header_fields(header_text)
        try:
            sample_count = int(fields["no. points"])
            variable_count = int(fields["no. variables"])
        except (KeyError, ValueError) as error:
            raise DptRawImportError(
                "The RAW plot has malformed point or trace counts. Re-export the "
                "file and try again."
            ) from error
        if sample_count < 1 or variable_count < 1:
            raise DptRawImportError(
                "A RAW plot has an empty point or trace declaration."
            )
        if sample_count > MAX_RAW_POINTS - total_points:
            declared_total = total_points + sample_count
            if total_points == 0:
                detail = (
                    f"The .raw file contains {declared_total} samples; the limit is "
                    f"{MAX_RAW_POINTS}. Reduce the LTspice output."
                )
            else:
                detail = (
                    f"The file contains {declared_total:,} samples across its plots; "
                    f"the aggregate limit is {MAX_RAW_POINTS}. Export a smaller "
                    "LTspice result."
                )
            raise DptRawImportError(
                detail,
                status_code=413,
            )
        if variable_count > MAX_TRACE_COUNT - total_traces:
            raise DptRawImportError(
                f"The file declares more than {MAX_TRACE_COUNT} aggregate traces. "
                "Export fewer traces.",
                status_code=413,
            )
        total_points += sample_count
        total_traces += variable_count

        lines = header_text.splitlines()
        variables_line = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().casefold() == "variables:"
            ),
            None,
        )
        if variables_line is None:
            raise DptRawImportError(
                "A RAW plot has no Variables section. Re-export the LTspice result."
            )
        trace_names: list[str] = []
        for expected_index, line in enumerate(lines[variables_line + 1 :]):
            parts = line.lstrip().split("\t")
            if len(parts) < 3:
                raise DptRawImportError(
                    "A RAW plot has a malformed Variables entry. Re-export the "
                    "LTspice result."
                )
            try:
                actual_index = int(parts[0])
            except ValueError as error:
                raise DptRawImportError(
                    "A RAW plot has a malformed trace index. Re-export the result."
                ) from error
            if actual_index != expected_index or not parts[1]:
                raise DptRawImportError(
                    "A RAW plot has unordered or missing trace declarations. "
                    "Re-export the LTspice result."
                )
            trace_names.append(parts[1])
        if len(trace_names) != variable_count:
            raise DptRawImportError(
                "A RAW plot's trace declarations do not match No. Variables. "
                "Re-export the LTspice result."
            )

        plot_name = fields.get("plotname", "")
        flags = fields.get("flags", "").casefold().split()
        command = fields.get("command", "").casefold()
        if "ltspice" not in command:
            raise DptRawImportError(
                "This RAW plot is not identified as native LTspice output. "
                "Only LTspice .raw files are supported."
            )
        is_transient = plot_name.casefold() == "transient analysis"
        is_complex = "complex" in flags or plot_name.casefold() == "ac analysis"
        if is_transient and is_complex:
            raise DptRawImportError(
                "Complex-valued transient plots are unsupported for DPT import. "
                "Choose a real-valued transient plot; imaginary samples cannot be "
                "discarded safely."
            )
        summary: dict[str, Any] = {
            "plot_index": len(plots),
            "plot_name": plot_name,
            "sample_count": sample_count,
            "trace_names": trace_names,
            "is_transient": is_transient,
            "is_stepped": "stepped" in flags,
            "steps": [],
            "_header_start": header_start,
            "_marker_index": marker_index,
            "_data_start": data_start,
            "_data_kind": data_kind,
            "_fields": fields,
        }
        if is_complex:
            bytes_per_point = variable_count * 16
            trace_item_sizes = [16] * variable_count
        else:
            value_width = 8 if "double" in flags else 4
            bytes_per_point = 8 + (variable_count - 1) * value_width
            trace_item_sizes = [8] + [value_width] * (variable_count - 1)
        if data_kind == "values":
            array_bytes = sample_count * sum(trace_item_sizes)
            value_records = sample_count * variable_count
            if array_bytes > MAX_RAW_ARRAY_BYTES - total_ascii_array_bytes:
                raise DptRawImportError(
                    "ASCII RAW arrays exceed the 32 MiB aggregate decoded-array "
                    "budget. Export binary RAW data or a smaller result.",
                    status_code=413,
                )
            if value_records > MAX_ASCII_VALUE_RECORDS - total_ascii_value_records:
                raise DptRawImportError(
                    "The ASCII RAW file contains too many value records to validate "
                    "within the web import budget. Export binary RAW data or a "
                    "smaller result.",
                    status_code=413,
                )
            total_ascii_array_bytes += array_bytes
            total_ascii_value_records += value_records
            time_values = (
                np.empty(sample_count, dtype=np.float64) if is_transient else None
            )
            value_cursor = data_start
            for point_index in range(sample_count):
                for trace_index in range(variable_count):
                    while True:
                        raw_line, line, next_cursor = _ascii_line(
                            content, value_cursor, encoding
                        )
                        value_cursor = next_cursor
                        if line:
                            break
                    value = line
                    if trace_index == 0:
                        point_text, separator, value = line.partition("\t")
                        if not separator:
                            raise DptRawImportError(
                                "An ASCII RAW axis row has no point index. "
                                "Re-export the LTspice result."
                            )
                        try:
                            actual_point = int(point_text)
                        except ValueError as error:
                            raise DptRawImportError(
                                "An ASCII RAW axis row has a malformed point index."
                            ) from error
                        if actual_point != point_index:
                            raise DptRawImportError(
                                "ASCII RAW point indexes are not sequential; the "
                                "Values section is malformed."
                            )
                    _validate_ascii_value(value, complex_value=is_complex)
                    if trace_index == 0 and time_values is not None:
                        try:
                            axis_value = float(value)
                            # Match PyLTSpice's Axis.get_wave() workaround for
                            # LTspice 2nd-order-compression negative time markers.
                            time_values[point_index] = (
                                abs(axis_value) if trace_names[0] == "time" else axis_value
                            )
                        except ValueError as error:
                            raise DptRawImportError(
                                "The transient time axis contains a malformed value."
                            ) from error
            while value_cursor < len(content):
                _, trailing_line, next_cursor = _ascii_line(
                    content, value_cursor, encoding
                )
                if trailing_line:
                    break
                value_cursor = next_cursor
            summary["_time_values"] = time_values
            summary["_payload_end"] = value_cursor
            summary["_trace_item_sizes"] = trace_item_sizes
            plots.append(summary)
            cursor = value_cursor
            continue
        payload_end = data_start + sample_count * bytes_per_point
        if payload_end > len(content):
            raise DptRawImportError(
                f"RAW plot {len(plots) + 1} ends before its declared binary samples. "
                "Re-export the complete LTspice result."
            )
        summary["_payload_end"] = payload_end
        summary["_bytes_per_point"] = bytes_per_point
        summary["_trace_item_sizes"] = trace_item_sizes
        if is_transient:
            stride = (
                trace_item_sizes[0]
                if "fastaccess" in flags
                else bytes_per_point
            )
            try:
                raw_time_values = np.ndarray(
                    shape=(sample_count,),
                    dtype="<f8",
                    buffer=content,
                    offset=data_start,
                    strides=(stride,),
                )
                # PyLTSpice exposes abs(time) for LTspice's negative markers
                # used with 2nd-order compression; mirror that decoded axis.
                summary["_time_values"] = (
                    np.abs(raw_time_values)
                    if trace_names[0] == "time"
                    else raw_time_values
                )
            except (TypeError, ValueError) as error:
                raise DptRawImportError(
                    "The transient time axis cannot be safely addressed from its "
                    "binary payload. Re-export the LTspice result."
                ) from error
        plots.append(summary)
        cursor = payload_end
    if not plots:
        raise DptRawImportError("The .raw file contains no complete result plots.")
    for summary in plots:
        if not summary["is_transient"]:
            continue
        time_values = summary.get("_time_values")
        if time_values is None or len(time_values) != summary["sample_count"]:
            raise DptRawImportError(
                f"Transient plot {summary['plot_index'] + 1} has no valid time axis."
            )
        summary["steps"] = _step_ranges(
            time_values, stepped=summary["is_stepped"]
        )
    return plots


def _selected_ranges(
    ranges: Sequence[dict[str, Any]],
    step_indices: Sequence[int] | None,
    *,
    is_stepped: bool,
) -> list[dict[str, Any]]:
    """Validate selected source steps while retaining the user's output order."""
    if step_indices is None:
        if is_stepped:
            raise DptRawImportError(
                "Select at least one source step before previewing or importing."
            )
        step_indices = [0]
    selected = list(step_indices)
    if not selected or len(selected) > MAX_SELECTED_STEPS:
        raise DptRawImportError(
            f"Select between 1 and {MAX_SELECTED_STEPS} unique source steps."
        )
    if len(set(selected)) != len(selected):
        raise DptRawImportError("Select each source step only once.")
    by_index = {item["step_index"]: item for item in ranges}
    if any(index not in by_index for index in selected):
        raise DptRawImportError(
            "A selected source step is not present in this plot. Reload the file "
            "preview and choose again."
        )
    return [by_index[index] for index in selected]


def _selected_plot_bytes(
    content: bytes,
    summary: dict[str, Any],
    selected_ranges: Sequence[dict[str, Any]],
) -> bytes:
    """Build one bounded temporary plot containing only selected step samples."""
    ordered_ranges = sorted(selected_ranges, key=lambda item: item["start_index"])
    selected_count = sum(item["sample_count"] for item in ordered_ranges)
    selected_mask = np.zeros(summary["sample_count"], dtype=np.bool_)
    for step in ordered_ranges:
        selected_mask[step["start_index"] : step["end_index"]] = True

    encoding = _raw_encoding(content)
    header_start = summary["_header_start"]
    marker_index = summary["_marker_index"]
    header_text = content[header_start:marker_index].decode(encoding)
    lines = header_text.splitlines(keepends=True)
    points_lines = 0
    for index, line in enumerate(lines):
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip().casefold() == "no. points":
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            lines[index] = f"{key}:{selected_count}{ending}"
            points_lines += 1
        elif key.strip().casefold() == "flags":
            lines[index] = re.sub(
                r"\bstepped\b",
                lambda match: " " * len(match.group(0)),
                line,
                flags=re.IGNORECASE,
            )
    if points_lines != 1:
        raise DptRawImportError(
            "The RAW plot has an ambiguous No. Points header. Re-export the result."
        )
    output_encoding = (
        "utf-8"
        if summary["_data_kind"] == "values" and encoding == "utf_16_le"
        else encoding
    )
    selected_header = "".join(lines).encode(output_encoding)
    marker_and_line = content[marker_index : summary["_data_start"]]
    if output_encoding != encoding:
        marker_and_line = marker_and_line.decode(encoding).encode(output_encoding)
    data_start = summary["_data_start"]

    if summary["_data_kind"] == "binary":
        payload = bytearray()
        trace_item_sizes = summary["_trace_item_sizes"]
        flags = summary["_fields"].get("flags", "").casefold().split()
        if "fastaccess" in flags:
            trace_offset = data_start
            for item_size in trace_item_sizes:
                for step in ordered_ranges:
                    start = trace_offset + step["start_index"] * item_size
                    end = trace_offset + step["end_index"] * item_size
                    payload.extend(content[start:end])
                trace_offset += summary["sample_count"] * item_size
        else:
            record_size = summary["_bytes_per_point"]
            for step in ordered_ranges:
                start = data_start + step["start_index"] * record_size
                end = data_start + step["end_index"] * record_size
                payload.extend(content[start:end])
    else:
        payload = bytearray()
        cursor = data_start
        variable_count = len(summary["trace_names"])
        selected_point_index = 0
        for point_index in range(summary["sample_count"]):
            point_lines: list[bytes] = []
            axis_value: str | None = None
            line_ending = "\n"
            for _trace_index in range(variable_count):
                while True:
                    raw_line, line, next_cursor = _ascii_line(
                        content, cursor, encoding
                    )
                    cursor = next_cursor
                    if line:
                        break
                if _trace_index == 0:
                    _, separator, axis_value = line.partition("\t")
                    if not separator:
                        raise DptRawImportError(
                            "An ASCII RAW axis row has no point index. "
                            "Re-export the LTspice result."
                        )
                    decoded_raw_line = raw_line.decode(encoding)
                    if decoded_raw_line.endswith("\r\n"):
                        line_ending = "\r\n"
                point_lines.append(
                    raw_line
                    if output_encoding == encoding
                    else raw_line.decode(encoding).encode(output_encoding)
                )
            if selected_mask[point_index]:
                if axis_value is None:
                    raise DptRawImportError(
                        "An ASCII RAW axis row has no time value. Re-export the "
                        "LTspice result."
                    )
                payload.extend(
                    f"{selected_point_index}\t{axis_value}{line_ending}".encode(
                        output_encoding
                    )
                )
                payload.extend(b"".join(point_lines[1:]))
                selected_point_index += 1
    return selected_header + marker_and_line + bytes(payload)


def _step_ranges(
    time_values: np.ndarray,
    *,
    stepped: bool,
) -> list[dict[str, Any]]:
    """Split a stepped axis only at repeated first-time samples."""
    if len(time_values) < 2 or not np.isfinite(time_values).all():
        raise DptRawImportError(
            "The transient time axis must contain at least two finite samples."
        )
    if not stepped:
        if np.any(np.diff(time_values) < 0):
            raise DptRawImportError(
                "The selected time axis decreases between samples. Upload a "
                "transient file with samples in time order."
            )
        starts = [0]
    else:
        starts = [
            0,
            *(np.flatnonzero(time_values[1:] == time_values[0]) + 1).tolist(),
        ]
        if len(starts) < 2:
            raise DptRawImportError(
                "This plot is marked stepped, but its time axis has no repeated "
                "time-origin boundaries. Step indexes cannot be established safely."
            )
        if len(starts) > MAX_RAW_STEPS_PER_PLOT:
            raise DptRawImportError(
                f"The plot contains more than {MAX_RAW_STEPS_PER_PLOT} detectable "
                "steps; reduce the LTspice step count."
            )

    ranges: list[dict[str, Any]] = []
    for step_index, start in enumerate(starts):
        end = starts[step_index + 1] if step_index + 1 < len(starts) else len(time_values)
        if end - start < 2:
            raise DptRawImportError(
                f"Step {step_index} has fewer than two samples; its time boundary "
                "is ambiguous. Re-export the plot with complete time samples."
            )
        step_times = time_values[start:end]
        if stepped and np.any(np.diff(step_times) <= 0):
            raise DptRawImportError(
                f"Step {step_index} does not have a strictly increasing time axis "
                "between its repeated-origin boundary samples."
            )
        if not stepped and np.any(np.diff(step_times) < 0):
            raise DptRawImportError(
                "The selected time axis decreases between samples. Upload a "
                "transient file with samples in time order."
            )
        ranges.append(
            {
                "step_index": step_index,
                "start_index": int(start),
                "end_index": int(end),
                "sample_count": int(end - start),
                "time_range_s": [float(step_times[0]), float(step_times[-1])],
            }
        )
    return ranges


def parse_ltspice_raw(
    content: bytes | bytearray | memoryview,
    filename: str | None,
    requested_traces: Sequence[str] = (),
    *,
    plot_index: int | None = 0,
    step_indices: Sequence[int] | None = None,
) -> ParsedRaw:
    """Preflight all plots, then read signals for selected source steps only."""
    safe_filename = _safe_filename(filename)
    if PurePosixPath(safe_filename).suffix.casefold() != ".raw":
        raise DptRawImportError(
            "Unsupported file type. Choose an LTspice .raw file.", status_code=415
        )
    if not isinstance(content, (bytes, bytearray, memoryview)) or not content:
        raise DptRawImportError("The uploaded .raw file is empty.")
    if len(content) > MAX_RAW_UPLOAD_BYTES:
        size_limit = MAX_RAW_UPLOAD_BYTES // (1024 * 1024)
        raise DptRawImportError(
            f"The .raw file is {len(content) / (1024 * 1024):.1f} MiB; "
            f"the upload limit is {size_limit} MiB. Reduce the LTspice output "
            "or export a smaller transient window.",
            status_code=413,
        )

    selected_names = tuple(requested_traces)
    if selected_names and (
        len(selected_names) != 3
        or any(not isinstance(name, str) or not name for name in selected_names)
        or len(set(selected_names)) != 3
    ):
        raise DptRawImportError(
            "Choose three distinct traces for time, Vds, and Id."
        )

    try:
        plot_headers = _scan_raw_headers(bytes(content))
        public_plots = [
            {
                key: summary[key]
                for key in (
                    "plot_index",
                    "plot_name",
                    "sample_count",
                    "trace_names",
                    "is_transient",
                    "is_stepped",
                    "steps",
                )
            }
            for summary in plot_headers
        ]
        if plot_index is not None and (plot_index < 0 or plot_index >= len(public_plots)):
            raise DptRawImportError(
                "The selected plot index is not present in the uploaded file. "
                "Reload the plot list and choose again."
            )
        if plot_index is None:
            first = public_plots[0]
            return ParsedRaw(
                filename=safe_filename,
                plot_name=first["plot_name"],
                trace_names=first["trace_names"],
                sample_count=first["sample_count"],
                plots=public_plots,
                plot_headers=plot_headers,
            )

        selected_plot = plot_headers[plot_index]
        if not selected_plot["is_transient"]:
            raise DptRawImportError(
                f"Plot {plot_index + 1} contains '{selected_plot['plot_name']}', "
                "not a transient analysis. Select a transient plot."
            )
        trace_names = selected_plot["trace_names"]
        missing = [name for name in selected_names if name not in trace_names]
        if missing:
            raise DptRawImportError(
                "A selected trace is not present in the selected plot. Reload "
                "the trace list and choose again."
            )
        if selected_names and selected_names[0] != trace_names[0]:
            raise DptRawImportError(f"Choose '{trace_names[0]}' as the time axis.")

        source_time_values = selected_plot["_time_values"]
        if not selected_names:
            return ParsedRaw(
                filename=safe_filename,
                plot_name=selected_plot["plot_name"],
                trace_names=trace_names,
                sample_count=selected_plot["sample_count"],
                time_values=source_time_values,
                plot_index=plot_index,
                step_ranges=selected_plot["steps"],
                plots=public_plots,
                is_stepped=selected_plot["is_stepped"],
                plot_headers=plot_headers,
            )

        source_ranges = _selected_ranges(
            selected_plot["steps"],
            step_indices,
            is_stepped=selected_plot["is_stepped"],
        )
        # Construct a temporary single-plot source containing only the selected
        # step sample blocks. PyLTSpice then sees no unselected plot or step data.
        selected_content = _selected_plot_bytes(
            bytes(content), selected_plot, source_ranges
        )
        total_selected_samples = sum(item["sample_count"] for item in source_ranges)
        ordered_source_ranges = sorted(
            source_ranges, key=lambda item: item["start_index"]
        )
        selected_time_values = np.concatenate(
            [
                source_time_values[item["start_index"] : item["end_index"]]
                for item in ordered_source_ranges
            ]
        )
        step_ranges_by_index: dict[int, dict[str, Any]] = {}
        local_offset = 0
        for item in ordered_source_ranges:
            local_end = local_offset + item["sample_count"]
            step_ranges_by_index[item["step_index"]] = {
                **item,
                "start_index": local_offset,
                "end_index": local_end,
            }
            local_offset = local_end
        selected_step_ranges = [
            step_ranges_by_index[item["step_index"]] for item in source_ranges
        ]

        try:
            from PyLTSpice import RawRead
        except ImportError as error:
            raise DptRawImportError(
                "LTspice RAW import needs PyLTSpice. Install the project requirements "
                "and restart the local web server."
            ) from error
        with TemporaryDirectory(prefix="tdb-dpt-raw-selected-") as temporary_directory:
            from pathlib import Path

            selected_path = Path(temporary_directory) / "selected.raw"
            selected_path.write_bytes(selected_content)
            reader = RawRead(selected_path, traces_to_read=None, verbose=False)
            if reader.dialect != "ltspice" or reader.get_nr_plots() != 1:
                raise DptRawImportError(
                    "PyLTSpice could not validate the bounded selected RAW plot. "
                    "Re-export the transient result from LTspice."
                )
            selected_reader_plot = reader.plots[0]
            try:
                selected_reader_plot.read_trace_data(list(selected_names[1:]))
                trace_values = {
                    name: np.asarray(
                        selected_reader_plot.get_trace(name).get_wave(0),
                        dtype=np.float64,
                    )
                    for name in selected_names[1:]
                }
            except (IndexError, RuntimeError, TypeError, ValueError) as error:
                raise DptRawImportError(
                    "Could not read the selected traces from this RAW plot. Check "
                    "the trace choices and re-export the transient plot."
                ) from error
        if any(len(values) != total_selected_samples for values in trace_values.values()):
            raise DptRawImportError(
                "The selected traces do not share the selected source sample count. "
                "Re-export the transient result."
            )
        return ParsedRaw(
            filename=safe_filename,
            plot_name=selected_plot["plot_name"],
            trace_names=trace_names,
            sample_count=total_selected_samples,
            time_values=selected_time_values,
            trace_values=trace_values,
            plot_index=plot_index,
            step_ranges=selected_step_ranges,
            plots=public_plots,
            is_stepped=selected_plot["is_stepped"],
            plot_headers=plot_headers,
        )
    except DptRawImportError:
        raise
    except Exception as error:
        raise DptRawImportError(
            "The LTspice RAW headers or selected data are malformed. Re-export "
            "the transient result from LTspice and try again."
        ) from error


def _preview_indices(sample_count: int, limit: int) -> np.ndarray:
    """Return ordered preview indices while preserving first and last points."""
    if sample_count <= limit:
        return np.arange(sample_count)
    return np.linspace(0, sample_count - 1, num=limit, dtype=np.int64)


def preview_samples(values: np.ndarray, limit: int = MAX_PREVIEW_POINTS) -> np.ndarray:
    """Return a display-only, ordered subset while preserving first and last points."""
    return values[_preview_indices(len(values), limit)]


def event_window_bounds(
    time_values: np.ndarray,
    start_s: float,
    end_s: float,
) -> tuple[int, int]:
    """Return inclusive bounds in a validated nondecreasing time axis."""
    if not np.isfinite([start_s, end_s]).all() or end_s < start_s:
        raise DptRawImportError(
            "Each event window needs finite start/end seconds with end at or after start."
        )
    start_index = int(np.searchsorted(time_values, start_s, side="left"))
    end_index = int(np.searchsorted(time_values, end_s, side="right"))
    sample_count = end_index - start_index
    if sample_count == 0:
        raise DptRawImportError(
            "The selected event window contains no source samples. Widen the time window."
        )
    if sample_count > MAX_EVENT_SAMPLES:
        raise DptRawImportError(
            f"The selected event window contains {sample_count:,} samples; the limit "
            f"is {MAX_EVENT_SAMPLES:,}. Narrow the time window."
        )
    return start_index, end_index


def select_event_samples(
    time_values: np.ndarray,
    vds_values: np.ndarray,
    id_values: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    bounds: tuple[int, int] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Select inclusive event-window samples without sorting or interpolation."""
    start_index, end_index = bounds or event_window_bounds(
        time_values, start_s, end_s
    )
    vds_window = vds_values[start_index:end_index]
    id_window = id_values[start_index:end_index]
    if not np.isfinite(vds_window).all() or not np.isfinite(id_window).all():
        raise DptRawImportError(
            "The selected event window contains non-finite Vds or Id samples. "
            "Choose a clean window or export finite traces from LTspice."
        )
    vds: list[list[float]] = []
    ids: list[list[float]] = []
    for index in range(start_index, end_index):
        time_value = float(time_values[index])
        vds.append([time_value, float(vds_values[index])])
        ids.append([time_value, float(id_values[index])])
    return vds, ids


def selected_step_ranges(
    parsed: ParsedRaw,
    step_indices: Sequence[int] | None,
) -> list[dict[str, Any]]:
    """Validate unique explicit step indexes and return their exact source spans."""
    return _selected_ranges(
        parsed.step_ranges or [],
        step_indices,
        is_stepped=parsed.is_stepped,
    )


def make_trace_preview(
    parsed: ParsedRaw,
    selected_traces: Sequence[str],
    step_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a small display projection for the selected source traces."""
    if parsed.time_values is None or parsed.trace_values is None:
        raise DptRawImportError("Select traces before requesting a waveform preview.")
    step_ranges = selected_step_ranges(parsed, step_indices)
    previews: list[dict[str, Any]] = []
    for step in step_ranges:
        start = step["start_index"]
        end = step["end_index"]
        times = parsed.time_values[start:end]
        indices = _preview_indices(len(times), MAX_PREVIEW_POINTS)
        preview_traces: dict[str, list[list[float | None]]] = {}
        non_finite_counts: dict[str, int] = {}
        for name in selected_traces[1:]:
            values = parsed.trace_values[name][start:end]
            preview_values: list[list[float | None]] = []
            non_finite_count = 0
            for index in indices:
                value = float(values[index])
                if not np.isfinite(value):
                    value = None
                    non_finite_count += 1
                preview_values.append([float(times[index]), value])
            preview_traces[name] = preview_values
            non_finite_counts[name] = non_finite_count
        previews.append(
            {
                **step,
                "preview_sample_count": len(indices),
                "traces": preview_traces,
                "non_finite_preview_sample_counts": non_finite_counts,
            }
        )
    first = previews[0]
    return {
        "sample_count": sum(item["sample_count"] for item in previews),
        "time_range_s": first["time_range_s"] if len(previews) == 1 else None,
        "preview_sample_count": sum(item["preview_sample_count"] for item in previews),
        "traces": first["traces"] if len(previews) == 1 else {},
        "non_finite_preview_sample_counts": (
            first["non_finite_preview_sample_counts"] if len(previews) == 1 else {}
        ),
        "steps": previews,
    }
