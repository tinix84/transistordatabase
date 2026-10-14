"""Read the explicitly selected traces from LTspice transient RAW uploads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np

MAX_RAW_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_RAW_REQUEST_BODY_BYTES = MAX_RAW_UPLOAD_BYTES + 1024 * 1024
MAX_RAW_POINTS = 1_000_000
MAX_EVENT_SAMPLES = 250_000
MAX_TOTAL_EVENT_SAMPLES = 250_000
MAX_PREVIEW_POINTS = 800
MAX_TRACE_COUNT = 128


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


def _safe_filename(filename: str | None) -> str:
    """Keep only the uploaded basename for validation and provenance."""
    name = PureWindowsPath(filename or "").name
    name = PurePosixPath(name).name
    if not name:
        raise DptRawImportError("Choose an LTspice .raw file to upload.")
    return name


def parse_ltspice_raw(
    content: bytes | bytearray | memoryview,
    filename: str | None,
    requested_traces: Sequence[str] = (),
) -> ParsedRaw:
    """Validate one transient RAW file and optionally read selected traces.

    The uploaded bytes are written only to a request-scoped temporary directory.
    The directory is removed when parsing completes; no source file is retained.
    """
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

    try:
        from PyLTSpice import RawRead
    except ImportError as error:
        raise DptRawImportError(
            "LTspice RAW import needs PyLTSpice. Install the project requirements "
            "and restart the local web server."
        ) from error

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
        with TemporaryDirectory(prefix="tdb-dpt-raw-") as temporary_directory:
            # Use pathlib.Path on the active platform after basename validation.
            from pathlib import Path

            local_raw_path = Path(temporary_directory) / "upload.raw"
            local_raw_path.write_bytes(content)
            # RawRead with traces_to_read=None parses the header without loading
            # waveform arrays. Enforce all size and format limits before opening
            # the file again with user-selected traces.
            header = RawRead(local_raw_path, traces_to_read=None, verbose=False)
            if header.dialect != "ltspice":
                raise DptRawImportError(
                    "This RAW file is not detected as native LTspice output. "
                    "Only LTspice transient .raw files are supported."
                )
            if header.get_nr_plots() != 1:
                raise DptRawImportError(
                    "This file contains multiple result plots. Export one LTspice "
                    "transient result per .raw file."
                )
            plot_name = str(header.get_plot_name())
            if plot_name.casefold() != "transient analysis":
                raise DptRawImportError(
                    f"The .raw file contains '{plot_name}', not a transient "
                    "analysis. Export a transient LTspice simulation."
                )
            if not header.has_axis:
                raise DptRawImportError(
                    "The transient .raw file has no time axis. Export a transient "
                    "analysis with saved time samples."
                )
            flags = (
                str(header.get_raw_properties().get("Flags", ""))
                .casefold()
                .split()
            )
            if "stepped" in flags:
                raise DptRawImportError(
                    "Stepped LTspice .raw files are not supported in this import. "
                    "Export one parameter step at a time."
                )
            if len(header.get_steps() or []) > 1:
                raise DptRawImportError(
                    "This .raw file contains multiple parameter steps. Export one "
                    "parameter step at a time."
                )

            sample_count = int(header.nPoints)
            if sample_count < 2:
                raise DptRawImportError(
                    "The transient .raw file must contain at least two samples."
                )
            if sample_count > MAX_RAW_POINTS:
                raise DptRawImportError(
                    f"The .raw file contains {sample_count:,} samples; the limit is "
                    f"{MAX_RAW_POINTS:,}. Reduce the LTspice output or export a "
                    "smaller transient window.",
                    status_code=413,
                )

            trace_names = [str(name) for name in header.get_trace_names()]
            if len(trace_names) > MAX_TRACE_COUNT:
                raise DptRawImportError(
                    f"The .raw file contains {len(trace_names)} traces; the limit "
                    f"is {MAX_TRACE_COUNT}. Export only the needed LTspice traces."
                )

            if not selected_names:
                return ParsedRaw(
                    filename=safe_filename,
                    plot_name=plot_name,
                    trace_names=trace_names,
                    sample_count=sample_count,
                )

            missing = [name for name in selected_names if name not in trace_names]
            if missing:
                raise DptRawImportError(
                    "A selected trace is not present in the uploaded file. "
                    "Reload the trace list and choose again."
                )
            if selected_names[0] != trace_names[0]:
                raise DptRawImportError(
                    f"Choose '{trace_names[0]}' as the file's time axis."
                )

            # Re-open only after header preflight; selected arrays are now bounded
            # by MAX_RAW_POINTS and limited to the explicitly chosen traces.
            reader = RawRead(
                local_raw_path,
                traces_to_read=list(selected_names),
                verbose=False,
            )
            if (
                reader.dialect != "ltspice"
                or reader.get_nr_plots() != 1
                or int(reader.nPoints) != sample_count
                or [str(name) for name in reader.get_trace_names()] != trace_names
            ):
                raise DptRawImportError(
                    "The RAW file changed while it was being inspected. Upload it "
                    "again and retry the trace selection."
                )
            try:
                time_values = np.asarray(reader.get_axis(0), dtype=np.float64)
                trace_values = {
                    name: np.asarray(
                        reader.get_trace(name).get_wave(0), dtype=np.float64
                    )
                    for name in selected_names[1:]
                }
            except (IndexError, RuntimeError, TypeError, ValueError) as error:
                raise DptRawImportError(
                    "Could not read the selected time, Vds, and Id traces from this "
                    ".raw file. Check the trace choices and upload a transient file."
                ) from error

            if len(time_values) != sample_count or any(
                len(values) != sample_count for values in trace_values.values()
            ):
                raise DptRawImportError(
                    "The selected traces do not share the same sample count. "
                    "Choose traces from one transient result."
                )
            if not np.isfinite(time_values).all():
                raise DptRawImportError(
                    "The selected time trace contains non-finite samples. "
                    "Choose a finite time axis."
                )
            if np.any(np.diff(time_values) < 0):
                raise DptRawImportError(
                    "The selected time axis decreases between samples. Upload a "
                    "transient file with samples in time order; equal timestamps "
                    "are preserved in their source order."
                )

            return ParsedRaw(
                filename=safe_filename,
                plot_name=plot_name,
                trace_names=trace_names,
                sample_count=sample_count,
                time_values=time_values,
                trace_values=trace_values,
            )
    except DptRawImportError:
        raise
    except Exception as error:
        raise DptRawImportError(
            "PyLTSpice could not read this file as a valid LTspice .raw file. "
            "Re-export the transient result from LTspice and try again."
        ) from error


def preview_samples(values: np.ndarray, limit: int = MAX_PREVIEW_POINTS) -> np.ndarray:
    """Return a display-only, ordered subset while preserving first and last points."""
    if len(values) <= limit:
        return values.copy()
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=np.int64)
    return values[indices]


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


def make_trace_preview(parsed: ParsedRaw, selected_traces: Sequence[str]) -> dict[str, Any]:
    """Build a small display projection for the selected source traces."""
    if parsed.time_values is None or parsed.trace_values is None:
        raise DptRawImportError("Select traces before requesting a waveform preview.")
    times = parsed.time_values
    preview_times = preview_samples(times)
    indices = (
        np.arange(len(times))
        if len(times) <= MAX_PREVIEW_POINTS
        else np.linspace(0, len(times) - 1, MAX_PREVIEW_POINTS, dtype=np.int64)
    )
    preview_traces: dict[str, list[list[float | None]]] = {}
    non_finite_counts: dict[str, int] = {}
    for name in selected_traces[1:]:
        values = parsed.trace_values[name]
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
    return {
        "sample_count": parsed.sample_count,
        "time_range_s": [float(np.min(times)), float(np.max(times))],
        "preview_sample_count": len(preview_times),
        "traces": preview_traces,
        "non_finite_preview_sample_counts": non_finite_counts,
    }
