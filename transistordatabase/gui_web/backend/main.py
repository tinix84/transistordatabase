"""
FastAPI backend for Transistor Database Web Application.

Uses the real core services and JsonTransistorRepository for persistent
storage instead of in-memory placeholder dicts.
"""
from __future__ import annotations

import csv
import asyncio
import copy
import ctypes
import errno
import hashlib
import io
import importlib.metadata
import json
import logging
import math
import os
import secrets
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path, PureWindowsPath
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from transistordatabase.backend.concrete_services import (
    ConcreteServiceFactory,
    ExportService,
    PlottingService,
    ValidationService,
    ComparisonService,
)
from transistordatabase.core.models import (
    Transistor,
    TransistorMetadata,
    ElectricalRatings,
    ThermalProperties,
    ChannelCharacteristics,
    SwitchingLossData,
    GateChargeCurve,
    SOA,
    VoltageDependentCapacitance,
)
from transistordatabase.core.adapters import core_to_legacy_dicts, legacy_to_core
from transistordatabase.core.pair_repository import JsonPairRepository
from transistordatabase.core.repository import (
    JsonTransistorRepository,
    _load_data_file,
    _json_dict_to_legacy_transistor,
)
from transistordatabase.transistor import Transistor as LegacyTransistor
from transistordatabase.gui_web.backend.schemas import (
    ChannelCurveCreate,
    SwitchingLossCreate,
    GateChargeCurveCreate,
    SOACreate,
    CapacitanceCurveCreate,
    CurveUpdateRequest,
    CurveValidationResult,
    PLECSExportOptions,
    MATLABExportOptions,
    BatchExportRequest,
    TopologyCalculationRequest,
    AdvancedSearchRequest,
    UserSettings,
    BielaModelRequest,
    GateChargeModelRequest,
    DPTConfigRequest,
    DptRawImportRequest,
    DptRawTraceSelectionRequest,
    DptEnergyDerivationRequest,
    DptEnergySaveRequest,
    PLECSImportResponse,
)
from transistordatabase.gui_web.backend.dpt_raw_import import (
    DptRawImportError,
    MAX_RAW_REQUEST_BODY_BYTES,
    MAX_RAW_UPLOAD_BYTES,
    MAX_TOTAL_EVENT_SAMPLES,
    event_window_bounds,
    make_trace_preview,
    parse_ltspice_raw,
    selected_step_ranges,
    select_event_samples,
)
from transistordatabase.gui_web.backend.dpt_energy import (
    DptEnergyDerivationError,
    derive_dpt_energy,
    derivation_comment,
    normalize_raw_measurements,
    request_token as dpt_energy_request_token,
    source_token as dpt_energy_source_token,
)
from transistordatabase.gui_web.backend.topology_service import TopologyCalculatorService

logger = logging.getLogger(__name__)
_DATABASE_BACKUP_FORMAT = "transistordatabase.database-backup"
_DATABASE_BACKUP_VERSION = 1
_WINDOWS_RESERVED_DEVICE_NAMES = {
    "aux",
    "clock$",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
}

# In-memory settings storage (use database in production)
_user_settings: Dict[str, UserSettings] = {}

app = FastAPI(
    title="Transistor Database API",
    description="REST API for managing transistor data",
    version="1.0.0",
)

# Bound the legacy JSON parse required for raw DPT persistence and waveform reads.
MAX_DPT_LEGACY_RECORD_FILE_BYTES = 64 * 1024 * 1024
MAX_DPT_RAW_MEASUREMENT_BYTES = 32 * 1024 * 1024
DPT_RAW_REVIEW_TOKEN_TTL_SECONDS = 15 * 60
MAX_DPT_RAW_REVIEW_TOKENS = 1024
_dpt_raw_review_tokens: dict[str, tuple[float, str]] = {}
_dpt_raw_review_tokens_lock = threading.Lock()


class _DptRawRequestBodyTooLarge(Exception):
    """Signal that a streamed DPT RAW multipart body exceeded its cap."""


class DptRawRequestBodyLimitMiddleware:
    """Bound import request bodies before Starlette parses multipart uploads."""

    def __init__(self, application: Callable[..., Any]) -> None:
        self.application = application

    @staticmethod
    def _is_dpt_raw_import(scope: dict[str, Any]) -> bool:
        path = scope.get("path", "")
        return (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and path.startswith("/api/transistors/")
            and path.endswith((
                "/dpt_raw_import",
                "/dpt_raw_import/preview",
                "/dpt_raw_import/trace_preview",
            ))
        )

    @staticmethod
    async def _send_too_large(send: Callable[..., Any]) -> None:
        limit_mib = MAX_RAW_REQUEST_BODY_BYTES / (1024 * 1024)
        body = json.dumps({
            "detail": (
                f"The DPT RAW request body exceeds the {limit_mib:g} MiB limit, "
                "including multipart overhead. Reduce the .raw file or export a "
                "smaller transient window."
            )
        }).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        """Enforce the import body cap before forwarding request chunks."""
        if not self._is_dpt_raw_import(scope):
            await self.application(scope, receive, send)
            return

        content_lengths = [
            value
            for key, value in scope.get("headers", [])
            if key.lower() == b"content-length"
        ]
        for raw_length in content_lengths:
            try:
                if int(raw_length) > MAX_RAW_REQUEST_BODY_BYTES:
                    await self._send_too_large(send)
                    return
            except ValueError:
                # The streaming count below remains authoritative for malformed
                # or absent Content-Length headers.
                continue

        received_bytes = 0
        response_started = False

        async def bounded_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > MAX_RAW_REQUEST_BODY_BYTES:
                    raise _DptRawRequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.application(scope, bounded_receive, tracked_send)
        except _DptRawRequestBodyTooLarge:
            # Multipart parsing consumes the body before a route can start its
            # response. If another layer already replied, do not emit a second.
            if not response_started:
                await self._send_too_large(send)


# Register the cap inside CORS so early 413 responses retain normal CORS headers.
app.add_middleware(DptRawRequestBodyLimitMiddleware)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Service and repository wiring
# ---------------------------------------------------------------------------

# Default JSON database directory (sibling of the package root)
_default_db_dir = Path(__file__).resolve().parent.parent.parent.parent / "database"


class _LazyRepository:
    """Construct the persistent repository only when a request uses it."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._instance: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self._instance is None:
            self._instance = self._factory()
        return getattr(self._instance, name)


_repo = _LazyRepository(lambda: JsonTransistorRepository(_default_db_dir))

# Default switching pairs directory
_default_pairs_dir = Path(__file__).resolve().parent.parent.parent.parent / "switching_pairs"
_pair_repo = _LazyRepository(lambda: JsonPairRepository(_default_pairs_dir))

_services = ConcreteServiceFactory.create_all_services()
_validation: ValidationService = _services['validation']
_comparison: ComparisonService = _services['comparison']
_export: ExportService = _services['export']
_plotting: PlottingService = _services['plotting']
_topology_calc = TopologyCalculatorService()
_record_mutation_lock = threading.Lock()


class _WindowsOverlapped(ctypes.Structure):
    """Windows OVERLAPPED structure used for blocking file-region locks."""

    _fields_ = [
        ("internal", ctypes.c_void_p),
        ("internal_high", ctypes.c_void_p),
        ("offset", ctypes.c_uint32),
        ("offset_high", ctypes.c_uint32),
        ("event", ctypes.c_void_p),
    ]


def _lock_record_file(lock_file: Any) -> Optional[_WindowsOverlapped]:
    """Acquire the cross-process mutation lock; call from a worker thread."""
    if os.name == "nt":
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        lock_file_ex = kernel32.LockFileEx
        lock_file_ex.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_WindowsOverlapped),
        ]
        lock_file_ex.restype = wintypes.BOOL
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(lock_file.fileno()))
        overlapped = _WindowsOverlapped()
        if not lock_file_ex(handle, 0x00000002, 0, 1, 0, ctypes.byref(overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())
        return overlapped

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return None


def _unlock_record_file(
    lock_file: Any,
    overlapped: Optional[_WindowsOverlapped],
) -> None:
    """Release the cross-process mutation lock; call from a worker thread."""
    if os.name == "nt":
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        unlock_file_ex = kernel32.UnlockFileEx
        unlock_file_ex.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_WindowsOverlapped),
        ]
        unlock_file_ex.restype = wintypes.BOOL
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(lock_file.fileno()))
        if not unlock_file_ex(handle, 0, 1, 0, ctypes.byref(overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


async def _acquire_blocking_lock(
    acquire: Callable[[], Any],
    release: Callable[[Any], None],
) -> Any:
    """Acquire a blocking lock off-loop and release it if cancelled while waiting."""
    acquire_task = asyncio.create_task(asyncio.to_thread(acquire))
    try:
        return await asyncio.shield(acquire_task)
    except asyncio.CancelledError:
        try:
            acquired_value = await acquire_task
        except Exception:
            raise
        await asyncio.to_thread(release, acquired_value)
        raise


@asynccontextmanager
async def _record_mutation() -> AsyncIterator[None]:
    """Serialize record mutations without blocking an async event loop."""
    await _acquire_blocking_lock(
        _record_mutation_lock.acquire,
        lambda _acquired: _record_mutation_lock.release(),
    )
    try:
        lock_file = (Path(_repo.data_directory) / ".web-record-mutations.lock").open("a+b")
    except BaseException:
        _record_mutation_lock.release()
        raise
    overlapped = None
    try:
        overlapped = await _acquire_blocking_lock(
            lambda: _lock_record_file(lock_file),
            lambda token: _unlock_record_file(lock_file, token),
        )
        try:
            yield
        finally:
            await asyncio.to_thread(_unlock_record_file, lock_file, overlapped)
    finally:
        await asyncio.to_thread(lock_file.close)
        _record_mutation_lock.release()


def _record_mutation_endpoint(endpoint: Callable[..., Any]) -> Callable[..., Any]:
    """Hold the database mutation lock across a route's read-modify-write flow."""
    @wraps(endpoint)
    async def locked_endpoint(*args: Any, **kwargs: Any) -> Any:
        async with _record_mutation():
            return await endpoint(*args, **kwargs)

    return locked_endpoint


def _get_transistor(name: str) -> Transistor:
    """Get transistor by name, raise 404 if not found."""
    t = _repo.get_by_name(name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found")
    return t


def _validate_web_record_name(name: str) -> None:
    """Reject identities unsafe as a filename on either supported platform."""
    if not isinstance(name, str):
        raise HTTPException(status_code=400, detail="Device names must be strings.")
    windows_name = PureWindowsPath(name)
    windows_stem = name.split(".", 1)[0].rstrip(" .").casefold()
    reserved_windows_device = windows_stem in _WINDOWS_RESERVED_DEVICE_NAMES or (
        len(windows_stem) == 4
        and windows_stem[:3] in {"com", "lpt"}
        and windows_stem[3] in "123456789¹²³"
    )
    if (
        not name
        or not name.strip()
        or name in {".", ".."}
        or name.endswith((".", " "))
        or "/" in name
        or "\\" in name
        or any(character in name for character in '<>"|?*')
        # A drive-relative Windows path (for example ``C:escape``) can
        # override the base directory when joined with a Windows Path.
        or bool(windows_name.drive)
        or bool(windows_name.root)
        # Colons also address NTFS alternate data streams, even on Linux/WSL.
        or ":" in name
        or reserved_windows_device
        or "\0" in name
        or any(ord(character) < 32 for character in name)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Device names must be valid portable filenames: path separators, reserved Windows device names, "
                "invalid filename characters, and trailing dots or spaces are not allowed."
            ),
        )


def _ensure_record_path_is_contained(record_path: Path, data_directory: Path) -> None:
    """Reject paths that escape the configured repository on Windows or locally."""
    try:
        windows_root = PureWindowsPath(str(data_directory))
        windows_record = PureWindowsPath(str(record_path))
        windows_record.relative_to(windows_root)

        resolved_root = data_directory.resolve()
        resolved_record = record_path.resolve()
        resolved_record.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="The device record path must remain inside the configured database directory.",
        ) from error


def _legacy_record_path(name: str) -> Path:
    """Resolve a repository record path without interpreting a name as a path."""
    _validate_web_record_name(name)
    data_directory = getattr(_repo, "data_directory", None)
    if data_directory is None:
        raise HTTPException(
            status_code=500,
            detail="The configured repository does not expose its JSON record directory.",
        )
    directory = Path(data_directory)
    record_path = directory / f"{name}.json"
    _ensure_record_path_is_contained(record_path, directory)
    return record_path


def _record_identity_key(name: str) -> str:
    """Normalize an identity according to the destination filesystem semantics."""
    return os.path.normcase(name)


def _existing_record_name_keys_unlocked() -> set[str]:
    """List existing JSON identities using the current filesystem semantics."""
    data_directory = Path(_repo.data_directory)
    try:
        children = data_directory.iterdir()
    except FileNotFoundError:
        return set()
    return {
        _record_identity_key(child.stem)
        for child in children
        if child.suffix.casefold() == ".json"
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish a completed record atomically without replacing an existing one."""
    if os.name == "nt":
        # Windows rename refuses to replace an existing destination.
        os.rename(source, destination)
        return

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable on this Linux system")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise OSError(error_number, os.strerror(error_number), destination)

    raise OSError(
        errno.ENOTSUP,
        "Atomic no-overwrite record publication is supported only on Windows and Linux",
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently retaining the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject JavaScript-style non-finite numbers that are outside JSON."""
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    """Reject JSON number literals that cannot be represented as finite floats."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number is outside the finite float range: {value}")
    significand = value.lower().split("e", 1)[0]
    if parsed == 0.0 and any(character in "123456789" for character in significand):
        raise ValueError(f"JSON number underflows the finite float range: {value}")
    return parsed


def _strict_json_loads(content: str | bytes) -> Any:
    """Parse strict JSON without collapsing duplicate object keys."""
    return json.loads(
        content,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
    )


def _validate_raw_legacy_record(record: Any) -> Transistor:
    """Validate one raw JSON record without changing the object to be stored."""
    if not isinstance(record, dict):
        raise ValueError("Each database record must be a JSON object.")
    name = record.get("name")
    if not isinstance(name, str):
        raise ValueError("Record identity must be a string in the 'name' field.")
    try:
        _validate_web_record_name(name)
    except HTTPException as error:
        raise ValueError(str(error.detail)) from error

    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(f"Record cannot be safely serialized as UTF-8 JSON: {error!s}") from error

    try:
        transistor = legacy_to_core(_json_dict_to_legacy_transistor(copy.deepcopy(record)))
    except Exception as error:
        raise ValueError(f"Legacy transistor validation failed: {error!s}") from error
    if transistor.metadata.name != name:
        raise ValueError("Legacy validation changed the exact record identity.")
    return transistor


def _parse_backup_document(content: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse and fully validate a versioned raw-record backup document."""
    try:
        document = _strict_json_loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        return [], [{"file": "backup", "error": f"Invalid JSON: {error!s}"}]

    if not isinstance(document, dict):
        return [], [{"file": "backup", "error": "Backup root must be a JSON object."}]
    if document.get("format") != _DATABASE_BACKUP_FORMAT:
        return [], [{"file": "backup", "error": "Unsupported database backup format."}]
    if type(document.get("version")) is not int or document["version"] != _DATABASE_BACKUP_VERSION:
        return [], [{"file": "backup", "error": "Unsupported database backup version."}]
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        return [], [{"file": "backup", "error": "Backup 'records' must be a JSON array."}]

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        name = raw_record.get("name") if isinstance(raw_record, dict) else None
        label = name if isinstance(name, str) else f"record {index + 1}"
        if isinstance(name, str):
            identity_key = _record_identity_key(name)
            if identity_key in seen_names:
                errors.append({
                    "index": index,
                    "name": name,
                    "error": "Duplicate record identity under current filesystem path semantics.",
                })
                continue
            seen_names.add(identity_key)
        try:
            _validate_raw_legacy_record(raw_record)
        except Exception as error:
            errors.append({"index": index, "name": label, "error": str(error)})
            continue
        records.append(raw_record)
    return records, errors


def _read_database_raw_records_unlocked() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Read every JSON record and report any corrupt or mismatched member."""
    data_directory = Path(_repo.data_directory)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_identity_keys: set[str] = set()
    for record_path in sorted(data_directory.glob("*.json"), key=lambda path: path.name.casefold()):
        try:
            _validate_web_record_name(record_path.stem)
            expected_record_path = _legacy_record_path(record_path.stem)
            if record_path != expected_record_path:
                raise ValueError("Record path does not match a safe repository filename identity.")
            _ensure_record_path_is_contained(record_path, data_directory)
            identity_key = _record_identity_key(record_path.stem)
            if identity_key in seen_identity_keys:
                raise ValueError("Duplicate record identity under current filesystem path semantics.")
            seen_identity_keys.add(identity_key)
            record = _strict_json_loads(record_path.read_bytes())
            if not isinstance(record, dict):
                raise ValueError("Record must be a JSON object.")
            if record.get("name") != record_path.stem:
                raise ValueError("Record name does not exactly match its filename identity.")
            _validate_raw_legacy_record(record)
            records.append(record)
        except Exception as error:
            errors.append({"file": record_path.name, "error": str(error)})
    return records, errors


def _write_raw_record_create_unlocked(record: dict[str, Any]) -> None:
    """Atomically publish a raw record without replacing an existing identity."""
    record_path = _legacy_record_path(record["name"])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{record_path.name}.", suffix=".tmp", dir=record_path.parent
    )
    temporary_path = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as record_file:
        json.dump(record, record_file, indent=2, ensure_ascii=False, allow_nan=False)
        record_file.flush()
        os.fsync(record_file.fileno())
    _rename_noreplace(temporary_path, record_path)


def _patch_legacy_record_unlocked(name: str, transistor_data: Dict[str, Any]) -> None:
    """Patch only general-form fields in the stored legacy JSON record.

    Saving a web-created core model through the core-to-legacy adapter drops
    legacy-only values. The web form owns only these general fields, so retain
    the original JSON document and replace just those values.
    """
    try:
        updated = dict_to_transistor(transistor_data)
        if updated.metadata.name != name:
            raise HTTPException(
                status_code=400,
                detail="Transistor names cannot be changed through the edit form.",
            )

        record_path = _legacy_record_path(name)
        with record_path.open("r", encoding="utf-8") as record_file:
            record = json.load(record_file)
        if not isinstance(record, dict) or record.get("name") != name:
            raise HTTPException(
                status_code=409,
                detail="The stored record name does not match the requested device name.",
            )

        # Copy exactly the general fields represented by the web form. Curves,
        # measured data, recommendations, and unknown legacy keys are retained.
        record.update(
            {
                "type": updated.metadata.type,
                "manufacturer": updated.metadata.manufacturer,
                "housing_type": updated.metadata.housing_type,
                "author": updated.metadata.author,
                "comment": updated.metadata.comment,
                "v_abs_max": updated.electrical_ratings.v_abs_max,
                "i_abs_max": updated.electrical_ratings.i_abs_max,
                "i_cont": updated.electrical_ratings.i_cont,
                "r_th_cs": updated.thermal_properties.r_th_cs,
                "r_th_switch_cs": updated.thermal_properties.r_th_switch_cs,
                "r_th_diode_cs": updated.thermal_properties.r_th_diode_cs,
                "housing_area": updated.thermal_properties.housing_area,
                "cooling_area": updated.thermal_properties.cooling_area,
            }
        )
        switch_record = record.get("switch")
        if not isinstance(switch_record, dict):
            raise HTTPException(
                status_code=409,
                detail="The stored record has no switch section for its junction temperature limit.",
            )
        switch_record["t_j_max"] = updated.electrical_ratings.t_j_max

        # Run the same legacy constructor used by JsonTransistorLoader before
        # persisting. It enforces manufacturer/housing catalogs and legacy
        # record constraints while the deep copy prevents array conversion
        # from changing the raw JSON object we intend to write.
        validated_legacy = _json_dict_to_legacy_transistor(copy.deepcopy(record))
        record["manufacturer"] = validated_legacy.manufacturer
        record["housing_type"] = validated_legacy.housing_type
        # Replace the populated record atomically so an interrupted write or a
        # concurrent reader never observes a truncated JSON file.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{record_path.name}.", suffix=".tmp", dir=record_path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as record_file:
            json.dump(record, record_file, indent=2)
            record_file.flush()
            os.fsync(record_file.fileno())
        os.replace(temporary_path, record_path)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found")
    except (OSError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update the stored record for '{name}': {e!s}",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid transistor data: {e!s}")


async def _patch_legacy_record(name: str, transistor_data: Dict[str, Any]) -> None:
    """Serialize web record mutations across threads and server workers."""
    async with _record_mutation():
        _patch_legacy_record_unlocked(name, transistor_data)


def _legacy_curve_list(
    record: dict[str, Any],
    path: tuple[str | None, str],
    *,
    create: bool = False,
) -> list[Any]:
    """Return one existing legacy curve list, optionally creating it."""
    section_name, field = path
    if section_name is None:
        section = record
    else:
        section = record.get(section_name)
        if not isinstance(section, dict):
            raise HTTPException(
                status_code=409,
                detail=f"The stored record has no {section_name} section for curve updates.",
            )

    curves = section.get(field)
    if curves is None and create:
        curves = []
        section[field] = curves
    if not isinstance(curves, list):
        raise HTTPException(
            status_code=409,
            detail=f"The stored record has an invalid {field} curve list.",
        )
    return curves


_CURVE_STORAGE_PATHS: dict[tuple[str, str], tuple[str | None, str]] = {
    ("switch", "channel"): ("switch", "channel"),
    ("diode", "channel"): ("diode", "channel"),
    ("switch", "e_on"): ("switch", "e_on"),
    ("switch", "e_off"): ("switch", "e_off"),
    ("diode", "e_rr"): ("diode", "e_rr"),
    ("switch", "gate_charge"): ("switch", "charge_curve"),
    ("switch", "soa"): ("switch", "soa"),
    ("capacitance", "c_oss"): (None, "c_oss"),
    ("capacitance", "c_iss"): (None, "c_iss"),
    ("capacitance", "c_rss"): (None, "c_rss"),
}

_SWITCHING_ENERGY_PATCH_FIELDS = frozenset({
    "dataset_type", "t_j", "v_supply", "v_g", "v_g_off", "r_g", "i_x", "e_x",
    "graph_i_e", "graph_r_e", "graph_t_e", "comment", "measurement_date",
    "measurement_testbench", "commutation_device", "load_inductance", "commutation_inductance",
})

_CURVE_PATCH_FIELDS: dict[tuple[str, str], set[str] | frozenset[str]] = {
    ("switch", "channel"): {"t_j", "v_g", "graph_v_i"},
    ("diode", "channel"): {"t_j", "graph_v_i"},
    ("switch", "e_on"): _SWITCHING_ENERGY_PATCH_FIELDS,
    ("switch", "e_off"): _SWITCHING_ENERGY_PATCH_FIELDS,
    ("diode", "e_rr"): _SWITCHING_ENERGY_PATCH_FIELDS,
    ("switch", "gate_charge"): {"v_supply", "t_j", "i_channel", "i_g", "graph_q_v"},
    ("switch", "soa"): {"t_c", "time_pulse", "graph_i_v"},
    ("capacitance", "c_oss"): {"t_j", "graph_v_c"},
    ("capacitance", "c_iss"): {"t_j", "graph_v_c"},
    ("capacitance", "c_rss"): {"t_j", "graph_v_c"},
}


def _curve_storage_path(component: str, curve_type: str) -> tuple[str | None, str]:
    """Resolve a browser curve category to its legacy JSON list."""
    try:
        return _CURVE_STORAGE_PATHS[(component, curve_type)]
    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported curve category '{component}.{curve_type}'.",
        ) from error


def _curve_edit_token(curve: dict[str, Any]) -> str:
    """Return a stable precondition token for one raw curve entry."""
    # Legacy records can contain Infinity sentinels in curve values; hash the
    # same Python JSON representation accepted by JsonTransistorLoader.
    encoded = json.dumps(curve, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_legacy_record_unlocked(
    name: str,
    *,
    max_file_bytes: int | None = None,
) -> dict[str, Any]:
    """Read a named raw record while the caller holds the mutation lock."""
    record_path = _legacy_record_path(name)
    try:
        with record_path.open("rb") as record_file:
            raw_record_bytes = record_file.read(
                max_file_bytes + 1 if max_file_bytes is not None else -1
            )
        if max_file_bytes is not None and len(raw_record_bytes) > max_file_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The stored device JSON exceeds the "
                    f"{_format_byte_limit(max_file_bytes)} DPT "
                    "access limit. DPT reads and imports stop before parsing an "
                    "oversized file; no stored data was changed."
                ),
            )
        # The legacy loader accepts Python's non-finite JSON constants (used by
        # some checked-in DPT samples), so preserve them in raw records. Keep
        # duplicate-key rejection while matching its numeric parsing behavior.
        record = json.loads(
            raw_record_bytes,
            object_pairs_hook=_unique_json_object,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found") from error
    if not isinstance(record, dict) or record.get("name") != name:
        raise HTTPException(
            status_code=409,
            detail="The stored record name does not match the requested device name.",
        )
    return record


def _with_curve_edit_tokens(
    curves: list[dict[str, Any]],
    record: dict[str, Any],
    path: tuple[str | None, str],
) -> list[dict[str, Any]]:
    """Pair API curve projections with tokens for their exact raw entries."""
    raw_curves = _legacy_curve_list(record, path, create=True)
    if len(curves) != len(raw_curves) or any(not isinstance(curve, dict) for curve in raw_curves):
        raise HTTPException(
            status_code=409,
            detail="The stored curve list cannot be mapped safely to the current reader.",
        )
    return [
        {**curve, "_edit_token": _curve_edit_token(raw_curve)}
        for curve, raw_curve in zip(curves, raw_curves)
    ]


def _update_legacy_curve_unlocked(
    name: str,
    component: str,
    curve_type: str,
    index: int,
    request: CurveUpdateRequest,
) -> None:
    """Patch and validate one curve while preserving the raw enclosing record."""
    path = _curve_storage_path(component, curve_type)
    patch = request.curve
    allowed_fields = _CURVE_PATCH_FIELDS[(component, curve_type)]
    unexpected_fields = set(patch) - allowed_fields
    if unexpected_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fields for {component}.{curve_type}: {', '.join(sorted(unexpected_fields))}.",
        )
    try:
        json.dumps(patch, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="Curve patches must contain only finite JSON numbers.",
        ) from error

    record_path = _legacy_record_path(name)
    record = _read_legacy_record_unlocked(name)
    curves = _legacy_curve_list(record, path)
    if len(curves) != request.expected_count or not 0 <= index < len(curves):
        raise HTTPException(
            status_code=409,
            detail="The stored curve list changed; reload the device before editing this curve.",
        )
    original_curve = curves[index]
    if not isinstance(original_curve, dict) or _curve_edit_token(original_curve) != request.expected_token:
        raise HTTPException(
            status_code=409,
            detail="This curve changed since it was loaded; reload the device before editing it.",
        )

    updated_curve = dict(original_curve)
    updated_curve.update(patch)
    curves[index] = updated_curve
    try:
        # Keep the whole legacy reader as the final authority for record and
        # curve constraints, but serialize only the raw candidate copy.
        # Unrelated legacy fields may contain non-finite sentinel values that
        # the legacy reader accepts. Patch values were checked separately.
        json.dumps(record, allow_nan=True)
        _json_dict_to_legacy_transistor(copy.deepcopy(record))
        _write_legacy_record_unlocked(record_path, record)
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found") from error
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update {component}.{curve_type} curve: {error!s}",
        ) from error


def _legacy_json_default(value: Any) -> Any:
    """Serialize arrays and timestamps in the legacy JSON record format."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _encoded_json_size_up_to(
    value: Any,
    byte_limit: int,
    *,
    indent: int | None = None,
) -> int:
    """Count compact JSON UTF-8 bytes without building a duplicate JSON string."""
    encoder = json.JSONEncoder(
        allow_nan=True,
        default=_legacy_json_default,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )
    byte_count = 0
    for chunk in encoder.iterencode(value):
        byte_count += len(chunk.encode("utf-8"))
        if byte_count > byte_limit:
            return byte_count
    return byte_count


def _format_byte_limit(byte_limit: int) -> str:
    """Format a byte bound with MiB where that is an exact unit."""
    mib = 1024 * 1024
    if byte_limit % mib == 0:
        return f"{byte_limit // mib} MiB ({byte_limit:,} bytes)"
    return f"{byte_limit:,} bytes"


def _write_legacy_record_unlocked(
    record_path: Path,
    record: Dict[str, Any],
    *,
    newline: str | None = None,
) -> None:
    """Atomically replace a validated legacy record."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{record_path.name}.", suffix=".tmp", dir=record_path.parent
    )
    with os.fdopen(
        descriptor,
        "w",
        encoding="utf-8",
        newline=newline,
    ) as record_file:
        json.dump(
            record,
            record_file,
            indent=2,
            default=_legacy_json_default,
        )
        record_file.flush()
        os.fsync(record_file.fileno())
    os.replace(temporary_name, record_path)


def _append_legacy_curve_unlocked(
    name: str,
    path: tuple[Optional[str], str],
    curve: Dict[str, Any],
    expected_count: int,
) -> int:
    """Append one curve while retaining every other raw legacy field."""
    try:
        record_path = _legacy_record_path(name)
        with record_path.open("r", encoding="utf-8") as record_file:
            record = json.load(record_file)
        if not isinstance(record, dict) or record.get("name") != name:
            raise HTTPException(
                status_code=409,
                detail="The stored record name does not match the requested device name.",
            )
        curves = _legacy_curve_list(record, path, create=True)
        if len(curves) != expected_count:
            raise HTTPException(
                status_code=409,
                detail="The stored curve list changed or cannot be represented safely.",
            )
        curves.append(curve)
        _json_dict_to_legacy_transistor(copy.deepcopy(record))
        _write_legacy_record_unlocked(record_path, record)
        return len(curves)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found")
    except (OSError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update the stored curves for '{name}': {e!s}",
        )


def _delete_legacy_curve_unlocked(
    name: str,
    path: tuple[Optional[str], str],
    index: int,
    expected_count: int,
) -> int:
    """Delete one curve while retaining every other raw legacy field."""
    try:
        record_path = _legacy_record_path(name)
        with record_path.open("r", encoding="utf-8") as record_file:
            record = json.load(record_file)
        if not isinstance(record, dict) or record.get("name") != name:
            raise HTTPException(
                status_code=409,
                detail="The stored record name does not match the requested device name.",
            )
        curves = _legacy_curve_list(record, path)
        if len(curves) != expected_count or not 0 <= index < len(curves):
            raise HTTPException(
                status_code=409,
                detail="The stored curve list changed or cannot be represented safely.",
            )
        del curves[index]
        _json_dict_to_legacy_transistor(copy.deepcopy(record))
        _write_legacy_record_unlocked(record_path, record)
        return len(curves)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found")
    except (OSError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update the stored curves for '{name}': {e!s}",
        )


def _serialize_latest_curve(transistor: Transistor, curve_key: str) -> Dict[str, Any]:
    """Serialize only the newly added curve through the established core adapter."""
    transistor_args, switch_args, diode_args = core_to_legacy_dicts(transistor)
    curve_lists = {
        "switch.channel": switch_args["channel"],
        "diode.channel": diode_args["channel"],
        "switch.e_on": switch_args["e_on"],
        "switch.e_off": switch_args["e_off"],
        "diode.e_rr": diode_args["e_rr"],
        "switch.charge_curve": switch_args["charge_curve"],
        "switch.soa": switch_args["soa"],
        "c_oss": transistor_args["c_oss"],
        "c_iss": transistor_args["c_iss"],
        "c_rss": transistor_args["c_rss"],
    }
    curves = curve_lists[curve_key]
    if not curves:
        raise HTTPException(status_code=500, detail="The new curve could not be serialized.")
    return curves[-1]


def _core_curve_list(transistor: Transistor, curve_key: str) -> List[Any]:
    """Resolve the core curve list corresponding to a legacy storage key."""
    core_lists = {
        "switch.channel": transistor.switch.channel_data,
        "diode.channel": transistor.diode.channel_data,
        "switch.e_on": transistor.switch.e_on_data,
        "switch.e_off": transistor.switch.e_off_data,
        "diode.e_rr": transistor.diode.e_rr_data,
        "switch.charge_curve": transistor.switch.gate_charge_curves,
        "switch.soa": transistor.switch.soa,
        "c_oss": transistor.c_oss,
        "c_iss": transistor.c_iss,
        "c_rss": transistor.c_rss,
    }
    return core_lists[curve_key]


def _persist_curve_mutation(
    transistor_id: str,
    transistor: Transistor,
    curve_key: str,
    path: tuple[Optional[str], str],
) -> None:
    """Persist one added curve without rewriting unrelated legacy fields."""
    curve = _serialize_latest_curve(transistor, curve_key)
    _append_legacy_curve_unlocked(
        transistor_id,
        path,
        curve,
        expected_count=len(_core_curve_list(transistor, curve_key)) - 1,
    )


def _serialize_new_transistor(transistor: Transistor) -> str:
    """Serialize a new core record through the repository's legacy loader path."""
    transistor_args, switch_args, diode_args = core_to_legacy_dicts(transistor)
    legacy_transistor = LegacyTransistor(
        transistor_args,
        switch_args,
        diode_args,
        _load_data_file("housing_types.txt"),
        _load_data_file("module_manufacturers.txt"),
    )
    legacy_record = legacy_transistor.convert_to_dict()
    legacy_record.pop("_id", None)
    return json.dumps(legacy_record, indent=2)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def transistor_to_dict(transistor: Transistor) -> Dict[str, Any]:
    """Convert Transistor object to dictionary for API responses."""
    result: Dict[str, Any] = {
        "metadata": {
            "name": transistor.metadata.name,
            "type": transistor.metadata.type,
            "manufacturer": transistor.metadata.manufacturer,
            "housing_type": transistor.metadata.housing_type,
            "author": transistor.metadata.author,
            "comment": transistor.metadata.comment,
        },
        "electrical": {
            "v_abs_max": transistor.electrical_ratings.v_abs_max,
            "i_abs_max": transistor.electrical_ratings.i_abs_max,
            "i_cont": transistor.electrical_ratings.i_cont,
            "t_j_max": transistor.electrical_ratings.t_j_max,
        },
        "thermal": {
            "r_th_cs": transistor.thermal_properties.r_th_cs,
            "r_th_switch_cs": transistor.thermal_properties.r_th_switch_cs,
            "r_th_diode_cs": transistor.thermal_properties.r_th_diode_cs,
            "housing_area": transistor.thermal_properties.housing_area,
            "cooling_area": transistor.thermal_properties.cooling_area,
        },
        "switch": {
            "channel_count": len(transistor.switch.channel_data),
            "e_on_count": len(transistor.switch.e_on_data),
            "e_off_count": len(transistor.switch.e_off_data),
            "has_foster": transistor.switch.thermal_foster is not None,
            "gate_charge_count": len(transistor.switch.gate_charge_curves),
            "soa_count": len(transistor.switch.soa),
        },
        "diode": {
            "channel_count": len(transistor.diode.channel_data),
            "e_rr_count": len(transistor.diode.e_rr_data),
            "has_foster": transistor.diode.thermal_foster is not None,
        },
        "capacitances": {
            "c_oss_count": len(transistor.c_oss),
            "c_iss_count": len(transistor.c_iss),
            "c_rss_count": len(transistor.c_rss),
        },
    }
    return result


def dict_to_transistor(data: Dict[str, Any]) -> Transistor:
    """Convert dictionary to Transistor object.

    Handles both nested format (from web interface) and flat format (TDB files).
    """
    if "metadata" in data and "electrical" in data and "thermal" in data:
        metadata = TransistorMetadata(
            name=data["metadata"]["name"],
            type=data["metadata"]["type"],
            manufacturer=data["metadata"]["manufacturer"],
            housing_type=data["metadata"]["housing_type"],
            author=data["metadata"].get("author", ""),
            comment=data["metadata"].get("comment", ""),
        )
        electrical = ElectricalRatings(
            v_abs_max=data["electrical"]["v_abs_max"],
            i_abs_max=data["electrical"]["i_abs_max"],
            i_cont=data["electrical"]["i_cont"],
            t_j_max=data["electrical"]["t_j_max"],
        )
        thermal = ThermalProperties(
            r_th_cs=data["thermal"]["r_th_cs"],
            r_th_switch_cs=data["thermal"].get("r_th_switch_cs"),
            r_th_diode_cs=data["thermal"].get("r_th_diode_cs"),
            housing_area=data["thermal"]["housing_area"],
            cooling_area=data["thermal"]["cooling_area"],
        )
    else:
        metadata = TransistorMetadata(
            name=data.get("name", "Unknown"),
            type=data.get("type", "Unknown"),
            manufacturer=data.get("manufacturer", "Unknown"),
            housing_type=data.get("housing_type", "Unknown"),
            author=data.get("author", ""),
            comment=data.get("comment", ""),
        )
        t_j_max = data.get("t_j_max", data.get("t_c_max", 150))
        if t_j_max is None:
            t_j_max = 150
        electrical = ElectricalRatings(
            v_abs_max=data.get("v_abs_max", 0),
            i_abs_max=data.get("i_abs_max", 0),
            i_cont=data.get("i_cont", 0),
            t_j_max=t_j_max,
        )
        thermal = ThermalProperties(
            r_th_cs=data.get("r_th_cs", 0),
            r_th_switch_cs=data.get("r_th_switch_cs"),
            r_th_diode_cs=data.get("r_th_diode_cs"),
            housing_area=data.get("housing_area", 0),
            cooling_area=data.get("cooling_area", 0),
        )

    return Transistor(metadata=metadata, electrical=electrical, thermal=thermal)


# ---------------------------------------------------------------------------
# Core CRUD endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Transistor Database API", "version": "1.0.0"}


@app.get("/api/transistors")
async def get_transistors() -> List[Dict[str, Any]]:
    """Get all transistors from the repository."""
    names = _repo.list_all()
    result = []
    for name in names:
        t = _repo.get_by_name(name)
        if t is not None:
            result.append(transistor_to_dict(t))
    return result


@app.get("/api/transistors/{transistor_id}")
async def get_transistor(transistor_id: str) -> Dict[str, Any]:
    """Get a specific transistor by name."""
    return transistor_to_dict(_get_transistor(transistor_id))


@app.get("/api/transistor-record")
async def get_transistor_record(name: str = Query(...)) -> Dict[str, Any]:
    """Get a transistor by its exact, query-encoded repository name."""
    _validate_web_record_name(name)
    return transistor_to_dict(_get_transistor(name))


@app.post("/api/transistors")
async def create_transistor(transistor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new transistor."""
    async with _record_mutation():
        try:
            metadata = transistor_data.get("metadata")
            requested_name = (
                metadata.get("name")
                if isinstance(metadata, dict)
                else transistor_data.get("name")
            )
            if isinstance(requested_name, str) and requested_name:
                _validate_web_record_name(requested_name)
                requested_path = _legacy_record_path(requested_name)
                if requested_path.exists():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Transistor '{requested_name}' already exists.",
                    )

            transistor = dict_to_transistor(transistor_data)
            _validate_web_record_name(transistor.metadata.name)
            record_path = _legacy_record_path(transistor.metadata.name)
            serialized_record = _serialize_new_transistor(transistor)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{record_path.name}.",
                suffix=".tmp",
                dir=record_path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as record_file:
                record_file.write(serialized_record)
                record_file.flush()
                os.fsync(record_file.fileno())
            try:
                # Publish only a complete record and refuse to replace a name
                # created by another worker or process during serialization.
                _rename_noreplace(temporary_path, record_path)
            except FileExistsError:
                raise HTTPException(
                    status_code=409,
                    detail=f"Transistor '{transistor.metadata.name}' already exists.",
                )
            return {
                "id": transistor.metadata.name,
                "message": "Transistor created successfully",
            }
        except HTTPException:
            raise
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Cannot safely create the transistor record: {e!s}",
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid transistor data: {e!s}")


@app.put("/api/transistors/{transistor_id}")
async def update_transistor(transistor_id: str, transistor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing transistor by its path-safe legacy identity."""
    await _patch_legacy_record(transistor_id, transistor_data)
    return {"message": "Transistor updated successfully"}


@app.put("/api/transistor-record")
async def update_transistor_record(
    transistor_data: Dict[str, Any],
    name: str = Query(...),
) -> Dict[str, Any]:
    """Update by exact query name so URL-reserved characters stay in identity."""
    await _patch_legacy_record(name, transistor_data)
    return {"message": "Transistor updated successfully"}


@app.delete("/api/transistors/{transistor_id}")
async def delete_transistor(transistor_id: str) -> Dict[str, Any]:
    """Delete a transistor."""
    _validate_web_record_name(transistor_id)
    async with _record_mutation():
        if not _repo.delete(transistor_id):
            raise HTTPException(status_code=404, detail="Transistor not found")
    return {"message": "Transistor deleted successfully"}


@app.delete("/api/transistor-record")
async def delete_transistor_record(name: str = Query(...)) -> Dict[str, Any]:
    """Delete by exact query name; slash identities are rejected explicitly."""
    _validate_web_record_name(name)
    async with _record_mutation():
        if not _repo.delete(name):
            raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found")
    return {"message": "Transistor deleted successfully"}


# ---------------------------------------------------------------------------
# Validation & Comparison
# ---------------------------------------------------------------------------

@app.post("/api/transistors/{transistor_id}/validate")
async def validate_transistor(transistor_id: str) -> Dict[str, Any]:
    """Validate a transistor."""
    transistor = _get_transistor(transistor_id)
    return _validation.validate_transistor(transistor)


@app.post("/api/transistors/compare")
async def compare_transistors(transistor_ids: List[str]) -> Dict[str, Any]:
    """Compare multiple transistors (basic comparison)."""
    if len(transistor_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 transistors required")
    transistors = [_get_transistor(tid) for tid in transistor_ids]
    return _comparison.compare_transistors(transistors)


@app.post("/api/comparison/advanced")
async def advanced_comparison(request: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced comparison with 9 plot types and configuration.

    Request body:
    {
        "transistor_ids": ["id1", "id2", "id3"],
        "config": {
            "t_j": 25,
            "v_supply": [600, 600, 600],
            "r_g_on": [10, 15, 20],
            "r_g_off": [10, 15, 20],
            "parallel_count": [1, 2, 1],
            "i_channel": 50
        }
    }
    """
    transistor_ids = request.get('transistor_ids', [])
    config = request.get('config', {})

    if not 2 <= len(transistor_ids) <= 3:
        raise HTTPException(
            status_code=400,
            detail="Advanced comparison requires 2-3 transistors"
        )

    transistors = [_get_transistor(tid) for tid in transistor_ids]
    return _comparison.generate_advanced_comparison(transistors, config)


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------

@app.post("/api/transistors/{transistor_id}/export/{format}")
async def export_transistor(transistor_id: str, format: str) -> FileResponse:
    """Export a transistor in the specified format.

    Supported formats: json, csv, spice, plecs, matlab, gecko, ltspice.
    """
    transistor = _get_transistor(transistor_id)
    supported = ("json", "csv", "spice", "plecs", "matlab", "gecko", "ltspice")
    if format not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Supported: {', '.join(supported)}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="tdb_export_"))
    try:
        if format == "json":
            out = tmp_dir / f"{transistor_id}.json"
            _export.export_to_json(transistor, out)
            return FileResponse(out, media_type="application/json", filename=out.name)

        if format == "csv":
            out = tmp_dir / f"{transistor_id}.csv"
            _export.export_to_csv([transistor], out)
            return FileResponse(out, media_type="text/csv", filename=out.name)

        if format == "spice":
            out = tmp_dir / f"{transistor_id}.spice"
            _export.export_to_spice(transistor, out)
            return FileResponse(out, media_type="text/plain", filename=out.name)

        if format == "plecs":
            _export.export_to_plecs(transistor, tmp_dir)
            xml_files = list(tmp_dir.glob("*.xml"))
            if not xml_files:
                raise HTTPException(status_code=500, detail="PLECS export produced no files")
            return FileResponse(xml_files[0], media_type="application/xml", filename=xml_files[0].name)

        if format == "matlab":
            _export.export_to_matlab(transistor, tmp_dir)
            mat_files = list(tmp_dir.glob("*.mat"))
            if not mat_files:
                raise HTTPException(status_code=500, detail="MATLAB export produced no files")
            return FileResponse(mat_files[0], media_type="application/octet-stream", filename=mat_files[0].name)

        if format == "gecko":
            _export.export_to_gecko_circuits(transistor, {"output_path": str(tmp_dir)})
            scl_files = list(tmp_dir.glob("*.scl"))
            if not scl_files:
                raise HTTPException(status_code=500, detail="GeckoCIRCUITS export produced no files")
            return FileResponse(scl_files[0], media_type="text/plain", filename=scl_files[0].name)

        if format == "ltspice":
            out = tmp_dir / f"{transistor_id}.asc"
            _export.export_to_ltspice(transistor, out)
            return FileResponse(out, media_type="text/plain", filename=out.name)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e!s}")


@app.get("/api/transistors/{transistor_id}/export/{format}/preview")
async def export_preview(transistor_id: str, format: str) -> Dict[str, Any]:
    """Preview export content (first 100 lines for text formats).

    Returns preview text and metadata about the export.
    """
    transistor = _get_transistor(transistor_id)
    supported = ("json", "csv", "spice", "plecs", "ltspice")
    if format not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Preview not supported for '{format}'. Supported: {', '.join(supported)}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="tdb_preview_"))
    try:
        preview_lines = []
        file_size = 0
        file_name = ""

        if format == "json":
            out = tmp_dir / f"{transistor_id}.json"
            _export.export_to_json(transistor, out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        elif format == "csv":
            out = tmp_dir / f"{transistor_id}.csv"
            _export.export_to_csv([transistor], out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        elif format == "spice":
            out = tmp_dir / f"{transistor_id}.spice"
            _export.export_to_spice(transistor, out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        elif format == "plecs":
            _export.export_to_plecs(transistor, tmp_dir)
            xml_files = list(tmp_dir.glob("*.xml"))
            if not xml_files:
                raise HTTPException(status_code=500, detail="PLECS export produced no files")
            content = xml_files[0].read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = xml_files[0].name

        elif format == "ltspice":
            out = tmp_dir / f"{transistor_id}.asc"
            _export.export_to_ltspice(transistor, out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        return {
            "preview": '\n'.join(preview_lines),
            "total_lines": len(lines) if 'lines' in locals() else 0,
            "preview_lines": len(preview_lines),
            "file_size": file_size,
            "file_name": file_name,
            "truncated": len(lines) > 100 if 'lines' in locals() else False,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {e!s}")


@app.post("/api/transistors/batch_export")
async def batch_export(request: BatchExportRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Export multiple transistors in a ZIP archive.

    Request body:
    {
        "transistor_ids": ["id1", "id2", "id3"],
        "format": "json",
        "options": {}
    }

    Returns a ZIP file containing all exports.
    """
    if not request.transistor_ids:
        raise HTTPException(status_code=400, detail="No transistor IDs provided")

    if request.format not in ("json", "csv", "spice", "plecs", "matlab", "gecko", "ltspice"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            tmp_dir = Path(tempfile.mkdtemp(prefix="tdb_batch_"))

            for transistor_id in request.transistor_ids:
                try:
                    transistor = _get_transistor(transistor_id)
                    export_dir = tmp_dir / transistor_id

                    if request.format == "json":
                        out = export_dir / f"{transistor_id}.json"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_json(transistor, out)
                        zip_file.write(out, arcname=f"{transistor_id}.json")

                    elif request.format == "csv":
                        out = export_dir / f"{transistor_id}.csv"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_csv([transistor], out)
                        zip_file.write(out, arcname=f"{transistor_id}.csv")

                    elif request.format == "spice":
                        out = export_dir / f"{transistor_id}.spice"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_spice(transistor, out)
                        zip_file.write(out, arcname=f"{transistor_id}.spice")

                    elif request.format == "plecs":
                        export_dir.mkdir(parents=True, exist_ok=True)
                        _export.export_to_plecs(transistor, export_dir)
                        for xml_file in export_dir.glob("*.xml"):
                            zip_file.write(xml_file, arcname=f"{transistor_id}/{xml_file.name}")

                    elif request.format == "matlab":
                        export_dir.mkdir(parents=True, exist_ok=True)
                        _export.export_to_matlab(transistor, export_dir)
                        for mat_file in export_dir.glob("*.mat"):
                            zip_file.write(mat_file, arcname=f"{transistor_id}/{mat_file.name}")

                    elif request.format == "gecko":
                        export_dir.mkdir(parents=True, exist_ok=True)
                        _export.export_to_gecko_circuits(transistor, {"output_path": str(export_dir)})
                        for scl_file in export_dir.glob("*.scl"):
                            zip_file.write(scl_file, arcname=f"{transistor_id}/{scl_file.name}")

                    elif request.format == "ltspice":
                        out = export_dir / f"{transistor_id}.asc"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_ltspice(transistor, out)
                        zip_file.write(out, arcname=f"{transistor_id}.asc")

                except Exception as e:
                    # Add error file to ZIP
                    error_content = f"Export failed for {transistor_id}: {e!s}"
                    zip_file.writestr(f"{transistor_id}_ERROR.txt", error_content)

        # Seek to beginning of buffer
        zip_buffer.seek(0)

        # Return ZIP file
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=transistor_exports_{request.format}.zip"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch export failed: {e!s}")


# ---------------------------------------------------------------------------
# Upload / Import
# ---------------------------------------------------------------------------

@app.get("/api/database/backup")
async def export_database_backup() -> StreamingResponse:
    """Download all validated raw JSON records in a versioned backup file."""
    async with _record_mutation():
        records, errors = _read_database_raw_records_unlocked()
    if errors:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Database backup was not created because records are invalid.",
                "records": errors,
            },
        )

    document = {
        "format": _DATABASE_BACKUP_FORMAT,
        "version": _DATABASE_BACKUP_VERSION,
        "records": records,
    }
    content = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="transistordatabase-backup.json"'},
    )


@app.post("/api/database/backup/validate")
async def validate_database_backup(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Validate a complete backup and preview existing-name conflicts."""
    if not file.filename or Path(file.filename).suffix.casefold() != ".json":
        raise HTTPException(status_code=400, detail="Select a .json database backup file.")
    records, errors = _parse_backup_document(await file.read())
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Backup validation failed; no records can be restored.", "errors": errors},
        )

    async with _record_mutation():
        record_paths = [_legacy_record_path(record["name"]) for record in records]
        existing_name_keys = _existing_record_name_keys_unlocked()
        conflicts = []
        for record, record_path in zip(records, record_paths):
            if _record_identity_key(record["name"]) in existing_name_keys or record_path.exists():
                conflicts.append(record["name"])
    conflict_keys = {_record_identity_key(name) for name in conflicts}
    restore_names = [
        record["name"]
        for record in records
        if _record_identity_key(record["name"]) not in conflict_keys
    ]
    return {
        "valid": True,
        "format": _DATABASE_BACKUP_FORMAT,
        "version": _DATABASE_BACKUP_VERSION,
        "record_count": len(records),
        "restore_count": len(restore_names),
        "restore_names": restore_names,
        "conflicts": conflicts,
    }


@app.post("/api/database/backup/restore")
async def restore_database_backup(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Add validated records while leaving existing database files untouched."""
    if not file.filename or Path(file.filename).suffix.casefold() != ".json":
        raise HTTPException(status_code=400, detail="Select a .json database backup file.")
    records, errors = _parse_backup_document(await file.read())
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Backup validation failed; no records were restored.", "errors": errors},
        )

    restored: list[str] = []
    conflicts: list[str] = []
    errors: list[dict[str, str]] = []
    async with _record_mutation():
        record_paths = [_legacy_record_path(record["name"]) for record in records]
        existing_name_keys = _existing_record_name_keys_unlocked()
        for record, record_path in zip(records, record_paths):
            name = record["name"]
            name_key = _record_identity_key(name)
            if name_key in existing_name_keys or record_path.exists():
                conflicts.append(name)
                continue
            try:
                _write_raw_record_create_unlocked(record)
                restored.append(name)
                existing_name_keys.add(name_key)
            except FileExistsError:
                conflicts.append(name)
                existing_name_keys.add(name_key)
            except OSError as error:
                errors.append({"name": name, "error": str(error)})

    return {
        "valid": True,
        "restored": restored,
        "conflicts": conflicts,
        "errors": errors,
    }


@app.post("/api/transistors/upload")
async def upload_transistor(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Create one transistor from raw JSON without overwriting an existing name."""
    if not file.filename or Path(file.filename).suffix.casefold() != ".json":
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        record = _strict_json_loads(await file.read())
        transistor = _validate_raw_legacy_record(record)
        async with _record_mutation():
            record_path = _legacy_record_path(transistor.metadata.name)
            if (
                _record_identity_key(transistor.metadata.name) in _existing_record_name_keys_unlocked()
                or record_path.exists()
            ):
                raise HTTPException(
                    status_code=409,
                    detail=f"Transistor '{transistor.metadata.name}' already exists.",
                )
            try:
                _write_raw_record_create_unlocked(record)
            except FileExistsError:
                raise HTTPException(
                    status_code=409,
                    detail=f"Transistor '{transistor.metadata.name}' already exists.",
                )
        return {
            "id": transistor.metadata.name,
            "message": "Transistor uploaded successfully",
            "name": transistor.metadata.name,
            "manufacturer": transistor.metadata.manufacturer,
            "type": transistor.metadata.type,
        }
    except HTTPException:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e!s}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Cannot safely create record: {e!s}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {e!s}")


# ---------------------------------------------------------------------------
# Plot data endpoints
# ---------------------------------------------------------------------------

@app.get("/api/plots/channel/{transistor_id}")
async def plot_channel(
    transistor_id: str,
    component: str = Query("switch", pattern="^(switch|diode)$"),
) -> Dict[str, Any]:
    """Get channel characteristic plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_channel_characteristics(transistor, component=component)


@app.get("/api/plots/switching/{transistor_id}")
async def plot_switching(
    transistor_id: str,
    loss_type: str = Query("e_on", pattern="^(e_on|e_off|e_rr)$"),
    plot_type: str = Query("i_e", pattern="^(i_e|r_e|t_e)$"),
) -> Dict[str, Any]:
    """Get switching loss plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_switching_losses(transistor, loss_type=loss_type, plot_type=plot_type)


@app.get("/api/plots/soa/{transistor_id}")
async def plot_soa(transistor_id: str) -> Dict[str, Any]:
    """Get safe operating area plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_safe_operating_area(transistor)


@app.get("/api/plots/thermal/{transistor_id}")
async def plot_thermal(transistor_id: str) -> Dict[str, Any]:
    """Get thermal impedance plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_thermal_impedance(transistor)


@app.get("/api/plots/gate_charge/{transistor_id}")
async def plot_gate_charge(transistor_id: str) -> Dict[str, Any]:
    """Get gate charge plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_gate_charge(transistor)


# ---------------------------------------------------------------------------
# Curve management endpoints (Phase 1)
# ---------------------------------------------------------------------------

@app.get("/api/transistors/{transistor_id}/curves")
async def get_all_curves(transistor_id: str) -> Dict[str, Any]:
    """Get all curves for a transistor."""
    async with _record_mutation():
        transistor = _get_transistor(transistor_id)
        raw_record = _read_legacy_record_unlocked(transistor_id)

    def array_to_list(arr):
        """Convert numpy array to nested list for JSON."""
        if arr is None:
            return None
        return arr.tolist() if hasattr(arr, 'tolist') else arr

    result = {
        "switch": {
            "channel_data": [
                {
                    "t_j": ch.t_j,
                    "v_g": ch.v_g,
                    "graph_v_i": array_to_list(ch.graph_v_i),
                }
                for ch in transistor.switch.channel_data
            ],
            "e_on_data": [
                {
                    "dataset_type": e.dataset_type,
                    "t_j": e.t_j,
                    "v_supply": e.v_supply,
                    "v_g": e.v_g,
                    "v_g_off": e.v_g_off,
                    "r_g": e.r_g,
                    "i_x": e.i_x,
                    "e_x": e.e_x,
                    "graph_i_e": array_to_list(e.graph_i_e),
                    "graph_r_e": array_to_list(e.graph_r_e),
                    "graph_t_e": array_to_list(e.graph_t_e),
                }
                for e in transistor.switch.e_on_data
            ],
            "e_off_data": [
                {
                    "dataset_type": e.dataset_type,
                    "t_j": e.t_j,
                    "v_supply": e.v_supply,
                    "v_g": e.v_g,
                    "v_g_off": e.v_g_off,
                    "r_g": e.r_g,
                    "i_x": e.i_x,
                    "e_x": e.e_x,
                    "graph_i_e": array_to_list(e.graph_i_e),
                    "graph_r_e": array_to_list(e.graph_r_e),
                    "graph_t_e": array_to_list(e.graph_t_e),
                }
                for e in transistor.switch.e_off_data
            ],
            "gate_charge_curves": [
                {
                    "v_supply": gc.v_supply,
                    "t_j": gc.t_j,
                    "i_channel": gc.i_channel,
                    "i_g": gc.i_g,
                    "graph_q_v": array_to_list(gc.graph_q_v),
                }
                for gc in transistor.switch.gate_charge_curves
            ],
            "soa": [
                {
                    "t_c": soa.t_c,
                    "time_pulse": soa.time_pulse,
                    "graph_i_v": array_to_list(soa.graph_i_v),
                }
                for soa in transistor.switch.soa
            ],
        },
        "diode": {
            "channel_data": [
                {
                    "t_j": ch.t_j,
                    "graph_v_i": array_to_list(ch.graph_v_i),
                }
                for ch in transistor.diode.channel_data
            ],
            "e_rr_data": [
                {
                    "dataset_type": e.dataset_type,
                    "t_j": e.t_j,
                    "v_supply": e.v_supply,
                    "v_g": e.v_g,
                    "v_g_off": e.v_g_off,
                    "r_g": e.r_g,
                    "i_x": e.i_x,
                    "e_x": e.e_x,
                    "graph_i_e": array_to_list(e.graph_i_e),
                    "graph_r_e": array_to_list(e.graph_r_e),
                    "graph_t_e": array_to_list(e.graph_t_e),
                }
                for e in transistor.diode.e_rr_data
            ],
        },
        "capacitances": {
            "c_oss": [
                {
                    "t_j": cap.t_j,
                    "graph_v_c": array_to_list(cap.graph_v_c),
                }
                for cap in transistor.c_oss
            ],
            "c_iss": [
                {
                    "t_j": cap.t_j,
                    "graph_v_c": array_to_list(cap.graph_v_c),
                }
                for cap in transistor.c_iss
            ],
            "c_rss": [
                {
                    "t_j": cap.t_j,
                    "graph_v_c": array_to_list(cap.graph_v_c),
                }
                for cap in transistor.c_rss
            ],
        },
    }
    result["switch"]["channel_data"] = _with_curve_edit_tokens(
        result["switch"]["channel_data"], raw_record, ("switch", "channel")
    )
    result["switch"]["e_on_data"] = _with_curve_edit_tokens(
        result["switch"]["e_on_data"], raw_record, ("switch", "e_on")
    )
    result["switch"]["e_off_data"] = _with_curve_edit_tokens(
        result["switch"]["e_off_data"], raw_record, ("switch", "e_off")
    )
    result["switch"]["gate_charge_curves"] = _with_curve_edit_tokens(
        result["switch"]["gate_charge_curves"], raw_record, ("switch", "charge_curve")
    )
    result["switch"]["soa"] = _with_curve_edit_tokens(
        result["switch"]["soa"], raw_record, ("switch", "soa")
    )
    result["diode"]["channel_data"] = _with_curve_edit_tokens(
        result["diode"]["channel_data"], raw_record, ("diode", "channel")
    )
    result["diode"]["e_rr_data"] = _with_curve_edit_tokens(
        result["diode"]["e_rr_data"], raw_record, ("diode", "e_rr")
    )
    for curve_type in ("c_oss", "c_iss", "c_rss"):
        result["capacitances"][curve_type] = _with_curve_edit_tokens(
            result["capacitances"][curve_type], raw_record, (None, curve_type)
        )
    return result


@app.post("/api/transistors/{transistor_id}/curves/channel")
@_record_mutation_endpoint
async def add_channel_curve(
    transistor_id: str,
    curve_data: ChannelCurveCreate,
    component: str = Query("switch", pattern="^(switch|diode)$"),
) -> Dict[str, Any]:
    """Add a channel characteristic curve to switch or diode."""
    transistor = _get_transistor(transistor_id)

    # Convert lists to numpy 2D array
    graph_v_i = np.array([curve_data.v_data, curve_data.i_data], dtype=np.float64)

    # Create ChannelCharacteristics object
    new_curve = ChannelCharacteristics(
        t_j=curve_data.t_j,
        v_g=curve_data.v_g,
        graph_v_i=graph_v_i,
    )

    # Add to appropriate component
    if component == "switch":
        if new_curve.v_g is None:
            raise HTTPException(
                status_code=400,
                detail="v_g is mandatory for switch channel curves"
            )
        transistor.switch.channel_data.append(new_curve)
    else:  # diode
        transistor.diode.channel_data.append(new_curve)

    # Save transistor
    curve_key = f"{component}.channel"
    _persist_curve_mutation(transistor_id, transistor, curve_key, (component, "channel"))

    return {
        "message": f"Channel curve added to {component}",
        "curve_count": len(
            transistor.switch.channel_data if component == "switch"
            else transistor.diode.channel_data
        ),
    }


@app.post("/api/transistors/{transistor_id}/curves/switching/{loss_type}")
@_record_mutation_endpoint
async def add_switching_loss_curve(
    transistor_id: str,
    loss_type: str,
    curve_data: SwitchingLossCreate,
) -> Dict[str, Any]:
    """Add switching loss data (e_on, e_off, e_rr)."""
    transistor = _get_transistor(transistor_id)

    if loss_type not in ("e_on", "e_off", "e_rr"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid loss_type '{loss_type}'. Must be 'e_on', 'e_off', or 'e_rr'"
        )

    # Convert graph data to numpy arrays
    graph_i_e = None
    graph_r_e = None
    graph_t_e = None

    if curve_data.graph_i_e:
        graph_i_e = np.array(curve_data.graph_i_e, dtype=np.float64)
    if curve_data.graph_r_e:
        graph_r_e = np.array(curve_data.graph_r_e, dtype=np.float64)
    if curve_data.graph_t_e:
        graph_t_e = np.array(curve_data.graph_t_e, dtype=np.float64)

    # Create SwitchingLossData object
    new_loss = SwitchingLossData(
        dataset_type=curve_data.dataset_type,
        t_j=curve_data.t_j,
        v_supply=curve_data.v_supply,
        v_g=curve_data.v_g,
        v_g_off=curve_data.v_g_off,
        r_g=curve_data.r_g,
        i_x=curve_data.i_x,
        e_x=curve_data.e_x,
        graph_i_e=graph_i_e,
        graph_r_e=graph_r_e,
        graph_t_e=graph_t_e,
        comment=curve_data.comment,
        measurement_date=curve_data.measurement_date,
        measurement_testbench=curve_data.measurement_testbench,
        commutation_device=curve_data.commutation_device,
        load_inductance=curve_data.load_inductance,
        commutation_inductance=curve_data.commutation_inductance,
    )

    # Add to appropriate list
    if loss_type == "e_on":
        transistor.switch.e_on_data.append(new_loss)
    elif loss_type == "e_off":
        transistor.switch.e_off_data.append(new_loss)
    else:  # e_rr
        transistor.diode.e_rr_data.append(new_loss)

    # Save transistor
    curve_key = "diode.e_rr" if loss_type == "e_rr" else f"switch.{loss_type}"
    legacy_section = "diode" if loss_type == "e_rr" else "switch"
    legacy_field = "e_rr" if loss_type == "e_rr" else loss_type
    _persist_curve_mutation(
        transistor_id, transistor, curve_key, (legacy_section, legacy_field)
    )

    return {
        "message": f"Switching loss curve ({loss_type}) added",
        "curve_count": len(
            transistor.switch.e_on_data if loss_type == "e_on"
            else transistor.switch.e_off_data if loss_type == "e_off"
            else transistor.diode.e_rr_data
        ),
    }


@app.post("/api/transistors/{transistor_id}/curves/gate_charge")
@_record_mutation_endpoint
async def add_gate_charge_curve(
    transistor_id: str,
    curve_data: GateChargeCurveCreate,
) -> Dict[str, Any]:
    """Add a gate charge curve."""
    transistor = _get_transistor(transistor_id)

    # Convert lists to numpy 2D array [charge, voltage]
    graph_q_v = np.array([curve_data.q_data, curve_data.v_data], dtype=np.float64)

    # Create GateChargeCurve object
    new_curve = GateChargeCurve(
        v_supply=curve_data.v_supply,
        t_j=curve_data.t_j,
        i_channel=curve_data.i_channel,
        i_g=curve_data.i_g,
        graph_q_v=graph_q_v,
    )

    transistor.switch.gate_charge_curves.append(new_curve)
    _persist_curve_mutation(
        transistor_id, transistor, "switch.charge_curve", ("switch", "charge_curve")
    )

    return {
        "message": "Gate charge curve added",
        "curve_count": len(transistor.switch.gate_charge_curves),
    }


@app.post("/api/transistors/{transistor_id}/curves/soa")
@_record_mutation_endpoint
async def add_soa_curve(
    transistor_id: str,
    curve_data: SOACreate,
) -> Dict[str, Any]:
    """Add a Safe Operating Area curve."""
    transistor = _get_transistor(transistor_id)

    # Convert lists to numpy 2D array [voltage, current]
    graph_i_v = np.array([curve_data.v_data, curve_data.i_data], dtype=np.float64)

    # Create SOA object
    new_soa = SOA(
        t_c=curve_data.t_c,
        time_pulse=curve_data.time_pulse,
        graph_i_v=graph_i_v,
    )

    transistor.switch.soa.append(new_soa)
    _persist_curve_mutation(transistor_id, transistor, "switch.soa", ("switch", "soa"))

    return {
        "message": "SOA curve added",
        "curve_count": len(transistor.switch.soa),
    }


@app.post("/api/transistors/{transistor_id}/curves/capacitance/{cap_type}")
@_record_mutation_endpoint
async def add_capacitance_curve(
    transistor_id: str,
    cap_type: str,
    curve_data: CapacitanceCurveCreate,
) -> Dict[str, Any]:
    """Add a voltage-dependent capacitance curve (C_oss, C_iss, C_rss)."""
    transistor = _get_transistor(transistor_id)

    if cap_type not in ("c_oss", "c_iss", "c_rss"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cap_type '{cap_type}'. Must be 'c_oss', 'c_iss', or 'c_rss'"
        )

    # Convert lists to numpy 2D array [voltage, capacitance]
    graph_v_c = np.array([curve_data.v_data, curve_data.c_data], dtype=np.float64)

    # Create VoltageDependentCapacitance object
    new_cap = VoltageDependentCapacitance(
        t_j=curve_data.t_j,
        graph_v_c=graph_v_c,
    )

    # Add to appropriate list
    if cap_type == "c_oss":
        transistor.c_oss.append(new_cap)
    elif cap_type == "c_iss":
        transistor.c_iss.append(new_cap)
    else:  # c_rss
        transistor.c_rss.append(new_cap)

    _persist_curve_mutation(transistor_id, transistor, cap_type, (None, cap_type))

    return {
        "message": f"{cap_type.upper()} curve added",
        "curve_count": len(
            transistor.c_oss if cap_type == "c_oss"
            else transistor.c_iss if cap_type == "c_iss"
            else transistor.c_rss
        ),
    }


@app.put("/api/transistors/{transistor_id}/curves/{component}/{curve_type}/{index}")
@_record_mutation_endpoint
async def update_curve(
    transistor_id: str,
    component: str,
    curve_type: str,
    index: int,
    request: CurveUpdateRequest,
) -> dict[str, Any]:
    """Update one existing raw curve without rewriting unrelated curve data."""
    _update_legacy_curve_unlocked(
        transistor_id,
        component,
        curve_type,
        index,
        request,
    )
    return {
        "message": f"Curve updated: {component}.{curve_type}[{index}]",
        "updated_index": index,
        "curve_count": request.expected_count,
    }


@app.delete("/api/transistors/{transistor_id}/curves/{component}/{curve_type}/{index}")
@_record_mutation_endpoint
async def delete_curve(
    transistor_id: str,
    component: str,
    curve_type: str,
    index: int,
) -> Dict[str, Any]:
    """Delete a specific curve by component, type, and index."""
    transistor = _get_transistor(transistor_id)

    # Map component and curve_type to the appropriate list
    curve_map = {
        "switch": {
            "channel": transistor.switch.channel_data,
            "e_on": transistor.switch.e_on_data,
            "e_off": transistor.switch.e_off_data,
            "gate_charge": transistor.switch.gate_charge_curves,
            "soa": transistor.switch.soa,
        },
        "diode": {
            "channel": transistor.diode.channel_data,
            "e_rr": transistor.diode.e_rr_data,
        },
        "capacitance": {
            "c_oss": transistor.c_oss,
            "c_iss": transistor.c_iss,
            "c_rss": transistor.c_rss,
        },
    }

    if component not in curve_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid component '{component}'"
        )

    if curve_type not in curve_map[component]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid curve_type '{curve_type}' for component '{component}'"
        )

    curve_list = curve_map[component][curve_type]

    if not 0 <= index < len(curve_list):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid index {index}. Curve list has {len(curve_list)} items"
        )

    # Delete the curve
    if component == "capacitance":
        legacy_path = (None, curve_type)
    elif component == "switch" and curve_type == "gate_charge":
        legacy_path = ("switch", "charge_curve")
    else:
        legacy_path = (component, curve_type)

    _delete_legacy_curve_unlocked(
        transistor_id,
        legacy_path,
        index,
        expected_count=len(curve_list),
    )
    del curve_list[index]

    return {
        "message": f"Curve deleted: {component}.{curve_type}[{index}]",
        "remaining_count": len(curve_list),
    }


@app.post("/api/transistors/{transistor_id}/curves/validate")
async def validate_curves(transistor_id: str) -> CurveValidationResult:
    """Validate all curves for a transistor."""
    transistor = _get_transistor(transistor_id)

    errors = []
    warnings = []

    # Validate switch channel curves
    for i, ch in enumerate(transistor.switch.channel_data):
        if ch.v_g is None:
            errors.append(f"Switch channel curve {i}: v_g is None")
        if ch.graph_v_i.shape[0] != 2:
            errors.append(f"Switch channel curve {i}: invalid graph shape")

    # Validate switching losses
    for i, loss in enumerate(transistor.switch.e_on_data):
        if loss.dataset_type == "single" and loss.e_x is None:
            errors.append(f"E_on curve {i}: single type but e_x is None")

    # Add more validation rules...

    statistics = {
        "switch_channel_curves": len(transistor.switch.channel_data),
        "diode_channel_curves": len(transistor.diode.channel_data),
        "e_on_curves": len(transistor.switch.e_on_data),
        "e_off_curves": len(transistor.switch.e_off_data),
        "e_rr_curves": len(transistor.diode.e_rr_data),
        "gate_charge_curves": len(transistor.switch.gate_charge_curves),
        "soa_curves": len(transistor.switch.soa),
    }

    return CurveValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        statistics=statistics,
    )


# ---------------------------------------------------------------------------
# Topology Calculator (Phase 4)
# ---------------------------------------------------------------------------

@app.post("/api/topology/calculate")
def calculate_topology(request: TopologyCalculationRequest) -> Dict[str, Any]:
    """Calculate a topology point and optional sweep through reusable Python code."""
    try:
        switch_id = request.switch_transistor_id or request.transistor_id
        diode_id = request.diode_transistor_id or request.transistor_id
        switch_transistor = _get_transistor(switch_id) if switch_id else None
        diode_transistor = _get_transistor(diode_id) if diode_id else None
        parameters = request.model_dump(exclude={
            "transistor_id", "switch_transistor_id", "diode_transistor_id",
        })
        result = _topology_calc.calculate(
            request.topology,
            parameters,
            switch_transistor=switch_transistor,
            diode_transistor=diode_transistor,
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation failed: {e!s}")


@app.post("/api/topology/compare")
def compare_topologies(request: Dict[str, Any]) -> Dict[str, Any]:
    """Compare all supported topology formulas for the supplied SI inputs."""
    try:
        parameters = dict(request.get("parameters", request))
        switch_id = request.get("switch_transistor_id") or request.get("transistor_id")
        diode_id = request.get("diode_transistor_id") or request.get("transistor_id")
        switch_transistor = _get_transistor(switch_id) if switch_id else None
        diode_transistor = _get_transistor(diode_id) if diode_id else None
        result = _topology_calc.compare_topologies(
            parameters,
            switch_transistor=switch_transistor,
            diode_transistor=diode_transistor,
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e!s}")


@app.get("/api/topology/recommendations")
async def topology_recommendations(
    topology: str = Query(..., description="Topology type"),
    v_in: float = Query(..., description="Input voltage (V)"),
    v_out: float = Query(..., description="Output voltage (V)"),
    p_out: float = Query(..., description="Output power (W)")
) -> Dict[str, Any]:
    """Get transistor recommendations for a topology."""
    if topology == "buck":
        v_required = v_in * 1.2
        i_required = (p_out / v_out) * 1.3
    elif topology == "boost":
        v_required = v_out * 1.2
        i_required = (p_out / v_in) * 1.3
    else:
        v_required = (v_in + v_out) * 1.2
        i_required = max(p_out / v_in, p_out / v_out) * 1.3

    all_names = _repo.list_all()
    suitable = []

    for name in all_names:
        t = _repo.get_by_name(name)
        if t is None:
            continue

        if (t.electrical_ratings.v_abs_max >= v_required and
            t.electrical_ratings.i_abs_max >= i_required):

            v_util = v_required / t.electrical_ratings.v_abs_max
            i_util = i_required / t.electrical_ratings.i_abs_max
            util_score = (v_util + i_util) / 2

            suitable.append({
                'name': t.metadata.name,
                'type': t.metadata.type,
                'manufacturer': t.metadata.manufacturer,
                'v_abs_max': t.electrical_ratings.v_abs_max,
                'i_abs_max': t.electrical_ratings.i_abs_max,
                'utilization': util_score * 100,
                'voltage_margin': (t.electrical_ratings.v_abs_max - v_required) / v_required * 100,
                'current_margin': (t.electrical_ratings.i_abs_max - i_required) / i_required * 100,
            })

    suitable.sort(key=lambda x: abs(x['utilization'] - 75))

    return {
        'topology': topology,
        'requirements': {
            'v_required': v_required,
            'i_required': i_required,
            'p_out': p_out,
        },
        'recommendations': suitable[:10],
        'total_suitable': len(suitable),
    }


# ---------------------------------------------------------------------------
# Settings & Advanced Features (Phase 5)
# ---------------------------------------------------------------------------

@app.get("/api/settings/user")
async def get_user_settings(user_id: str = "default") -> UserSettings:
    """Get user settings."""
    if user_id not in _user_settings:
        _user_settings[user_id] = UserSettings()
    return _user_settings[user_id]


@app.put("/api/settings/user")
async def update_user_settings(settings: UserSettings, user_id: str = "default") -> Dict[str, str]:
    """Update user settings."""
    _user_settings[user_id] = settings
    return {"message": "Settings updated successfully"}


@app.post("/api/settings/user/reset")
async def reset_user_settings(user_id: str = "default") -> Dict[str, str]:
    """Reset user settings to defaults."""
    _user_settings[user_id] = UserSettings()
    return {"message": "Settings reset to defaults"}


@app.post("/api/transistors/search/advanced")
async def advanced_search(request: AdvancedSearchRequest) -> Dict[str, Any]:
    """Advanced transistor search with filters, sorting, and pagination."""
    all_names = _repo.list_all()
    results = []

    for name in all_names:
        t = _repo.get_by_name(name)
        if t is None:
            continue

        # Apply filters
        filters = request.filters
        if filters.get('type') and t.metadata.type not in filters['type']:
            continue
        if 'v_abs_max' in filters:
            v_range = filters['v_abs_max']
            if 'min' in v_range and t.electrical_ratings.v_abs_max < v_range['min']:
                continue
            if 'max' in v_range and t.electrical_ratings.v_abs_max > v_range['max']:
                continue
        if 'i_abs_max' in filters:
            i_range = filters['i_abs_max']
            if 'min' in i_range and t.electrical_ratings.i_abs_max < i_range['min']:
                continue
            if 'max' in i_range and t.electrical_ratings.i_abs_max > i_range['max']:
                continue
        if filters.get('has_gate_charge') and not t.switch.gate_charge_curves:
            continue

        results.append({
            'name': t.metadata.name,
            'type': t.metadata.type,
            'manufacturer': t.metadata.manufacturer,
            'housing_type': t.metadata.housing_type,
            'v_abs_max': t.electrical_ratings.v_abs_max,
            'i_abs_max': t.electrical_ratings.i_abs_max,
            'i_cont': t.electrical_ratings.i_cont,
            'has_gate_charge': len(t.switch.gate_charge_curves) > 0,
            'has_switching_losses': len(t.switch.e_on_data) > 0 or len(t.switch.e_off_data) > 0,
        })

    # Sort results
    if request.sort_field:
        reverse = request.sort_order == 'desc'
        results.sort(key=lambda x: x.get(request.sort_field, 0), reverse=reverse)

    # Paginate
    total = len(results)
    start = (request.page - 1) * request.per_page
    end = start + request.per_page
    page_results = results[start:end]

    return {
        'results': page_results,
        'pagination': {
            'page': request.page,
            'per_page': request.per_page,
            'total': total,
            'pages': (total + request.per_page - 1) // request.per_page,
        }
    }


@app.post("/api/favorites/add")
async def add_favorite(transistor_id: str, user_id: str = "default") -> Dict[str, str]:
    """Add transistor to favorites."""
    settings = _user_settings.get(user_id, UserSettings())
    if transistor_id not in settings.favorite_transistors:
        settings.favorite_transistors.append(transistor_id)
        _user_settings[user_id] = settings
    return {"message": "Added to favorites"}


@app.post("/api/favorites/remove")
async def remove_favorite(transistor_id: str, user_id: str = "default") -> Dict[str, str]:
    """Remove transistor from favorites."""
    settings = _user_settings.get(user_id, UserSettings())
    if transistor_id in settings.favorite_transistors:
        settings.favorite_transistors.remove(transistor_id)
        _user_settings[user_id] = settings
    return {"message": "Removed from favorites"}


@app.get("/api/favorites/list")
async def list_favorites(user_id: str = "default") -> List[str]:
    """Get list of favorite transistors."""
    settings = _user_settings.get(user_id, UserSettings())
    return settings.favorite_transistors


# ==================== Phase 6: Archive Integration Endpoints ====================


@app.post("/api/import/plecs", response_model=PLECSImportResponse)
async def import_plecs(file: UploadFile) -> PLECSImportResponse:
    """Import transistors from PLECS XML semiconductor library file.

    :param file: Uploaded PLECS XML file
    :return: Import results with transistor IDs
    """
    from transistordatabase.plecs_importer import import_plecs_xml
    errors = []
    warnings = []
    transistor_ids = []

    try:
        content = await file.read()
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp:
            tmp.write(content)
            tmp.flush()

            # Import transistors
            transistors = import_plecs_xml(tmp.name)

            # Save to repository
            async with _record_mutation():
                for transistor in transistors:
                    try:
                        _repo.save(transistor)
                        transistor_ids.append(transistor.metadata.name)
                    except Exception as e:
                        errors.append(f"Failed to save {transistor.metadata.name}: {str(e)}")

            return PLECSImportResponse(
                success=len(transistor_ids) > 0,
                transistors_imported=len(transistor_ids),
                transistor_ids=transistor_ids,
                errors=errors,
                warnings=warnings
            )

    except Exception as e:
        return PLECSImportResponse(
            success=False,
            transistors_imported=0,
            transistor_ids=[],
            errors=[str(e)]
        )


@app.post("/api/analytical/biela")
async def calculate_biela(request: BielaModelRequest) -> Dict[str, Any]:
    """Calculate switching losses using Biela analytical model.

    :param request: Biela model parameters and operating conditions
    :return: Calculated energies and optional curve data
    """
    from transistordatabase.analytical_models import BielaModel, BielaModelParams

    # Create model
    params = BielaModelParams(
        c_oss=request.c_oss,
        q_g=request.q_g,
        v_plateau=request.v_plateau,
        r_g_on=request.r_g_on,
        r_g_off=request.r_g_off,
        v_driver_on=request.v_driver_on,
        v_driver_off=request.v_driver_off
    )
    model = BielaModel(params)

    # Calculate at single operating point
    e_on = model.calc_e_on(request.v_dc, request.i_load, request.t_j)
    e_off = model.calc_e_off(request.v_dc, request.i_load, request.t_j)

    result = {
        'operating_point': {
            'v_dc': request.v_dc,
            'i_load': request.i_load,
            't_j': request.t_j,
        },
        'results': {
            'e_on': e_on,
            'e_off': e_off,
            'e_total': e_on + e_off,
        }
    }

    # If curve requested, calculate over current range
    if request.i_min is not None and request.i_max is not None:
        currents = np.linspace(request.i_min, request.i_max, request.i_points)
        e_on_curve, e_off_curve, e_total_curve = model.calc_switching_loss_curve(
            request.v_dc, currents, request.t_j
        )

        result['curve'] = {
            'currents': currents.tolist(),
            'e_on': e_on_curve.tolist(),
            'e_off': e_off_curve.tolist(),
            'e_total': e_total_curve.tolist(),
        }

    return result


@app.post("/api/analytical/gate_charge")
async def calculate_gate_charge(request: GateChargeModelRequest) -> Dict[str, Any]:
    """Calculate switching times using gate charge model.

    :param request: Gate charge parameters
    :return: Turn-on and turn-off timing results
    """
    from transistordatabase.analytical_models import GateChargeModel, GateChargeModelParams

    # Create model
    params = GateChargeModelParams(
        q_gs=request.q_gs,
        q_gd=request.q_gd,
        q_g=request.q_g,
        v_th=request.v_th,
        v_plateau=request.v_plateau,
        r_g=request.r_g,
        r_g_int=request.r_g_int
    )
    model = GateChargeModel(params)

    # Calculate turn-on times
    turn_on = model.calc_turn_on_times(request.v_driver)

    # Calculate turn-off times
    turn_off = model.calc_turn_off_times(request.v_driver, request.v_off)

    return {
        'parameters': {
            'q_gs': request.q_gs,
            'q_gd': request.q_gd,
            'q_g': request.q_g,
            'v_th': request.v_th,
            'v_plateau': request.v_plateau,
            'r_g_total': request.r_g + request.r_g_int,
        },
        'turn_on_times': {
            't_delay': turn_on['t_delay'] * 1e9,  # Convert to ns
            't_rise': turn_on['t_rise'] * 1e9,
            't_fall_v': turn_on['t_fall_v'] * 1e9,
            't_total': turn_on['t_total'] * 1e9,
        },
        'turn_off_times': {
            't_delay': turn_off['t_delay'] * 1e9,
            't_rise_v': turn_off['t_rise_v'] * 1e9,
            't_fall_i': turn_off['t_fall_i'] * 1e9,
            't_total': turn_off['t_total'] * 1e9,
        }
    }


@app.post("/api/dpt/generate_netlist")
async def generate_dpt_netlist(
    transistor_id: str,
    config: DPTConfigRequest
) -> Dict[str, str]:
    """Generate LTspice Double Pulse Test netlist for a transistor.

    :param transistor_id: ID of transistor to test
    :param config: DPT configuration parameters
    :return: Generated netlist as text
    """
    from transistordatabase.utils.ltspice_dpt import LTspiceDPT, DPTConfig

    transistor = _get_transistor(transistor_id)

    # Create DPT configuration
    dpt_config = DPTConfig(
        v_dc=config.v_dc,
        i_target=config.i_target,
        l_load=config.l_load,
        r_g_on=config.r_g_on,
        r_g_off=config.r_g_off,
        v_g_on=config.v_g_on,
        v_g_off=config.v_g_off,
        t_dead=config.t_dead,
        t_pulse2=config.t_pulse2
    )

    # Generate netlist
    dpt = LTspiceDPT()
    netlist = dpt.generate_netlist(transistor, dpt_config)

    return {
        'netlist': netlist,
        'transistor_name': transistor.metadata.name,
        'config': {
            'v_dc': config.v_dc,
            'i_target': config.i_target,
            'r_g_on': config.r_g_on,
            'r_g_off': config.r_g_off,
        }
    }


_DPT_RAW_WAVEFORM_FIELDS = (
    "dpt_on_vds",
    "dpt_on_id",
    "dpt_off_vds",
    "dpt_off_id",
)
MAX_DPT_WAVEFORM_PREVIEW_POINTS = 1000
MAX_DPT_WAVEFORM_PREVIEW_TOTAL_POINTS = 40000


async def _read_dpt_raw_upload(file: UploadFile) -> bytearray:
    """Read a bounded upload in chunks without retaining a source file."""
    filename = PureWindowsPath(file.filename or "").name
    if not filename.casefold().endswith(".raw"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Choose an LTspice .raw file.",
        )
    content = bytearray()
    while chunk := await file.read(64 * 1024):
        content.extend(chunk)
        if len(content) > MAX_RAW_UPLOAD_BYTES:
            limit_mib = MAX_RAW_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The .raw upload exceeds the {limit_mib} MiB limit. Reduce the "
                    "LTspice output or export a smaller transient window."
                ),
            )
    return content


def _dpt_raw_upload_error(error: DptRawImportError) -> HTTPException:
    """Map a parser or waveform selection error to an actionable API response."""
    return HTTPException(status_code=error.status_code, detail=str(error))


def _dpt_raw_trace_selection(
    request: DptRawTraceSelectionRequest,
) -> tuple[str, str, str]:
    """Return the three explicit trace choices in parser order."""
    return request.time_trace, request.vds_trace, request.id_trace


def _dpt_raw_review_required(plots: list[dict[str, Any]], plot_index: int) -> bool:
    """Require reviewed server binding for stepped or selected multi-plot data."""
    return len(plots) > 1 or bool(plots[plot_index]["is_stepped"])


def _dpt_raw_review_signature(
    transistor_id: str,
    content: bytes | bytearray,
    selection: DptRawImportRequest,
) -> str:
    """Bind the import token to the bytes and canonical complete user selection."""
    selection_data = selection.model_dump(
        mode="json", exclude={"review_token"}, exclude_none=True
    )
    request_body = json.dumps(
        {
            "transistor_id": transistor_id,
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "selection": selection_data,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(request_body).hexdigest()


def _issue_dpt_raw_review_token(signature: str) -> str:
    """Create a short-lived one-time token for one reviewed source selection."""
    now = time.monotonic()
    token = secrets.token_urlsafe(32)
    with _dpt_raw_review_tokens_lock:
        expired = [
            value
            for value, (expires_at, _signature) in _dpt_raw_review_tokens.items()
            if expires_at <= now
        ]
        for value in expired:
            _dpt_raw_review_tokens.pop(value, None)
        if len(_dpt_raw_review_tokens) >= MAX_DPT_RAW_REVIEW_TOKENS:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The local RAW review session limit is full. Wait for an existing "
                    "review to expire, then preview this import again."
                ),
            )
        _dpt_raw_review_tokens[token] = (
            now + DPT_RAW_REVIEW_TOKEN_TTL_SECONDS,
            signature,
        )
    return token


def _consume_dpt_raw_review_token(token: str | None, signature: str) -> None:
    """Consume one reviewed token and reject missing, stale, or altered requests."""
    if not token:
        raise HTTPException(
            status_code=409,
            detail="Review this exact RAW file, plot, trace mapping, windows, and conditions before importing.",
        )
    now = time.monotonic()
    with _dpt_raw_review_tokens_lock:
        stored = _dpt_raw_review_tokens.pop(token, None)
    if stored is None or stored[0] <= now or stored[1] != signature:
        raise HTTPException(
            status_code=409,
            detail="The RAW import review is missing, expired, already used, or no longer matches the submitted file and selections. Review it again.",
        )


def _require_dpt_raw_common_context(selection: DptRawImportRequest) -> None:
    """Require explicit common conditions for all grouped source-step imports."""
    if selection.step_selections is None:
        raise DptRawImportError(
            "Select each source step and enter its current and event windows."
        )
    required_conditions = (
        "t_j",
        "v_supply",
        "v_g",
        "v_g_off",
        "r_g",
        "r_g_off",
        "load_inductance",
        "commutation_inductance",
        "measurement_testbench",
        "commutation_device",
    )
    missing_conditions = [
        name
        for name in required_conditions
        if getattr(selection.metadata, name) in (None, "")
    ]
    if missing_conditions:
        raise DptRawImportError(
            "A grouped source-step record requires explicit common values for: "
            + ", ".join(missing_conditions)
            + ". These values are not read from RAW or .log data."
        )
    if not selection.same_context_confirmed:
        raise DptRawImportError(
            "Confirm that selected steps share the entered conditions; only current may vary."
        )


@app.post("/api/transistors/{transistor_id}/dpt_raw_import/preview")
async def preview_dpt_raw_import(
    transistor_id: str,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """List bounded plot metadata and step ranges without loading signal traces."""
    _validate_web_record_name(transistor_id)
    content = await _read_dpt_raw_upload(file)
    try:
        parsed = parse_ltspice_raw(content, file.filename, plot_index=None)
    except DptRawImportError as error:
        raise _dpt_raw_upload_error(error) from error
    plots = parsed.plots or []
    first_transient = next((plot for plot in plots if plot["is_transient"]), None)
    if first_transient is None:
        raise HTTPException(
            status_code=400,
            detail="The .raw file contains no transient analysis plot.",
        )
    response = {
        "filename": parsed.filename,
        "plots": plots,
        "total_sample_count": sum(plot["sample_count"] for plot in plots),
    }
    if first_transient is not None:
        response.update(
            {
                "plot_name": first_transient["plot_name"],
                "sample_count": first_transient["sample_count"],
                "time_trace_name": first_transient["trace_names"][0],
                "trace_names": first_transient["trace_names"],
            }
        )
    return response


@app.post("/api/transistors/{transistor_id}/dpt_raw_import/trace_preview")
async def preview_dpt_raw_traces(
    transistor_id: str,
    file: UploadFile = File(...),
    selection_json: str = Form(...),
) -> Dict[str, Any]:
    """Return a display-only sample of the selected time, Vds, and Id traces."""
    _validate_web_record_name(transistor_id)
    try:
        selection = DptRawTraceSelectionRequest.model_validate_json(selection_json)
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid DPT trace selection: {error!s}",
        ) from error
    content = await _read_dpt_raw_upload(file)
    try:
        header = parse_ltspice_raw(content, file.filename, plot_index=None)
        plots = header.plots or []
        if len(plots) > 1 and selection.plot_index is None:
            raise DptRawImportError(
                "Select a transient plot from the file preview before loading traces."
            )
        plot_index = selection.plot_index if selection.plot_index is not None else 0
        if plot_index >= len(plots) or not plots[plot_index]["is_transient"]:
            raise DptRawImportError(
                "The selected plot is not a transient analysis. Select a transient plot."
            )
        if plots[plot_index]["is_stepped"] and not selection.step_indices:
            raise DptRawImportError(
                "Select at least one source step before previewing traces."
            )
        parsed = parse_ltspice_raw(
            content,
            file.filename,
            _dpt_raw_trace_selection(selection),
            plot_index=plot_index,
            step_indices=selection.step_indices or None,
        )
        preview = make_trace_preview(
            parsed,
            _dpt_raw_trace_selection(selection),
            selection.step_indices or None,
        )
        review_required = _dpt_raw_review_required(plots, plot_index)
        review_token = None
        if review_required and selection.step_selections is not None:
            reviewed_selection = DptRawImportRequest.model_validate(
                selection.model_dump(mode="python", exclude={"review_token"})
            )
            _require_dpt_raw_common_context(reviewed_selection)
            range_by_index = {
                item["step_index"]: item for item in (parsed.step_ranges or [])
            }
            total_event_samples = 0
            for step_selection in reviewed_selection.step_selections or []:
                selected_range = range_by_index.get(step_selection.step_index)
                if selected_range is None:
                    raise DptRawImportError(
                        f"Source step {step_selection.step_index} is not present in "
                        "the selected plot. Reload the file preview."
                    )
                step_times = parsed.time_values[
                    selected_range["start_index"] : selected_range["end_index"]
                ] if parsed.time_values is not None else np.asarray([])
                on_bounds = event_window_bounds(
                    step_times,
                    step_selection.turn_on.start_s,
                    step_selection.turn_on.end_s,
                )
                off_bounds = event_window_bounds(
                    step_times,
                    step_selection.turn_off.start_s,
                    step_selection.turn_off.end_s,
                )
                total_event_samples += (
                    on_bounds[1] - on_bounds[0] + off_bounds[1] - off_bounds[0]
                )
                if total_event_samples > MAX_TOTAL_EVENT_SAMPLES:
                    raise DptRawImportError(
                        f"The selected event windows contain {total_event_samples:,} "
                        f"samples combined; the limit is {MAX_TOTAL_EVENT_SAMPLES:,}. "
                        "Narrow windows or select fewer steps.",
                        status_code=413,
                    )
            review_token = _issue_dpt_raw_review_token(
                _dpt_raw_review_signature(
                    transistor_id, content, reviewed_selection
                )
            )
    except DptRawImportError as error:
        raise _dpt_raw_upload_error(error) from error
    return {
        "filename": parsed.filename,
        "time_trace": selection.time_trace,
        "vds_trace": selection.vds_trace,
        "id_trace": selection.id_trace,
        "plot_index": parsed.plot_index,
        "step_indices": [item["step_index"] for item in preview["steps"]],
        "review_required": review_required,
        "review_token": review_token,
        **preview,
    }


@app.post("/api/transistors/{transistor_id}/dpt_raw_import")
async def import_dpt_raw_measurement(
    transistor_id: str,
    file: UploadFile = File(...),
    selection_json: str = Form(...),
) -> Dict[str, Any]:
    """Append explicitly reviewed LTspice acquisitions to the legacy record."""
    _validate_web_record_name(transistor_id)
    try:
        selection = DptRawImportRequest.model_validate_json(selection_json)
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid DPT import selection: {error!s}",
        ) from error
    content = await _read_dpt_raw_upload(file)
    trace_selection = _dpt_raw_trace_selection(selection)
    try:
        header = parse_ltspice_raw(content, file.filename, plot_index=None)
        plots = header.plots or []
        if len(plots) > 1 and selection.plot_index is None:
            raise DptRawImportError(
                "Select a transient plot before confirming the import."
            )
        plot_index = selection.plot_index if selection.plot_index is not None else 0
        if plot_index >= len(plots) or not plots[plot_index]["is_transient"]:
            raise DptRawImportError(
                "The selected plot is not a transient analysis. Select a transient plot."
            )
        summary = plots[plot_index]
        review_required = _dpt_raw_review_required(plots, plot_index)
        if review_required:
            _require_dpt_raw_common_context(selection)
        elif selection.step_selections is not None:
            raise DptRawImportError(
                "Explicit source-step acquisition review is only required for a "
                "stepped plot or a selected plot from a multi-plot file."
            )

        axis_only = parse_ltspice_raw(
            content,
            file.filename,
            plot_index=plot_index,
        )
        assert axis_only.time_values is not None
        time_values = axis_only.time_values
        on_acquisitions: list[list[list[float]]] = []
        on_id_acquisitions: list[list[list[float]]] = []
        off_acquisitions: list[list[list[float]]] = []
        off_id_acquisitions: list[list[list[float]]] = []
        event_windows_by_step: dict[str, dict[str, list[float]]] = {}
        current_by_step: dict[str, float] = {}
        total_event_samples = 0
        if selection.step_selections is None:
            assert selection.turn_on is not None and selection.turn_off is not None
            selected_ranges = selected_step_ranges(axis_only, [0])
            windows = [(selected_ranges[0], selection.turn_on, selection.turn_off, None)]
        else:
            step_by_index = {
                item["step_index"]: item
                for item in (axis_only.step_ranges or [])
            }
            windows = []
            for step_selection in selection.step_selections:
                step = step_by_index.get(step_selection.step_index)
                if step is None:
                    raise DptRawImportError(
                        f"Source step {step_selection.step_index} is not present in "
                        "the selected plot. Reload the file preview."
                    )
                windows.append(
                    (
                        step,
                        step_selection.turn_on,
                        step_selection.turn_off,
                        step_selection.i_x,
                    )
                )

        preflight_windows = []
        for step, turn_on, turn_off, i_x in windows:
            start = step["start_index"]
            end = step["end_index"]
            step_time = time_values[start:end]
            on_bounds = event_window_bounds(
                step_time, turn_on.start_s, turn_on.end_s
            )
            off_bounds = event_window_bounds(
                step_time, turn_off.start_s, turn_off.end_s
            )
            total_event_samples += (
                on_bounds[1] - on_bounds[0] + off_bounds[1] - off_bounds[0]
            )
            if total_event_samples > MAX_TOTAL_EVENT_SAMPLES:
                raise DptRawImportError(
                    f"The selected event windows contain {total_event_samples:,} "
                    f"samples combined; the limit is {MAX_TOTAL_EVENT_SAMPLES:,}. "
                    "Narrow windows or select fewer steps.",
                    status_code=413,
                )
            preflight_windows.append(
                (step, turn_on, turn_off, i_x, on_bounds, off_bounds)
            )

        # Load selected signals only after every source step, time window and the
        # aggregate persisted-sample budget have passed preflight.
        parsed = parse_ltspice_raw(
            content,
            file.filename,
            trace_selection,
            plot_index=plot_index,
            step_indices=(
                [item.step_index for item in selection.step_selections]
                if selection.step_selections is not None
                else [0]
            ),
        )
        assert parsed.time_values is not None
        assert parsed.trace_values is not None
        expected_loaded_times = np.concatenate(
            [
                time_values[item["start_index"] : item["end_index"]]
                for item in sorted(
                    [window[0] for window in windows],
                    key=lambda item: item["start_index"],
                )
            ]
        )
        if not np.array_equal(parsed.time_values, expected_loaded_times):
            raise DptRawImportError(
                "The RAW time axis changed between preflight and trace loading. "
                "Upload the source again."
            )
        vds_values = parsed.trace_values[selection.vds_trace]
        id_values = parsed.trace_values[selection.id_trace]
        loaded_step_by_index = {
            item["step_index"]: item
            for item in (parsed.step_ranges or [])
        }
        for step, turn_on, turn_off, i_x, on_bounds, off_bounds in preflight_windows:
            loaded_step = loaded_step_by_index[step["step_index"]]
            start = loaded_step["start_index"]
            end = loaded_step["end_index"]
            step_time = parsed.time_values[start:end]
            step_vds = vds_values[start:end]
            step_id = id_values[start:end]
            on_vds, on_id = select_event_samples(
                step_time,
                step_vds,
                step_id,
                turn_on.start_s,
                turn_on.end_s,
                bounds=on_bounds,
            )
            off_vds, off_id = select_event_samples(
                step_time,
                step_vds,
                step_id,
                turn_off.start_s,
                turn_off.end_s,
                bounds=off_bounds,
            )
            on_acquisitions.append(on_vds)
            on_id_acquisitions.append(on_id)
            off_acquisitions.append(off_vds)
            off_id_acquisitions.append(off_id)
            step_key = str(step["step_index"])
            event_windows_by_step[step_key] = {
                "turn_on": [turn_on.start_s, turn_on.end_s],
                "turn_off": [turn_off.start_s, turn_off.end_s],
            }
            if i_x is not None:
                current_by_step[step_key] = i_x
    except DptRawImportError as error:
        raise _dpt_raw_upload_error(error) from error

    try:
        parser_version = importlib.metadata.version("PyLTSpice")
    except importlib.metadata.PackageNotFoundError:
        parser_version = "unknown"
    imported_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "source_filename": parsed.filename,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "parser": "PyLTSpice.RawRead",
        "parser_version": parser_version,
        "imported_at": imported_at,
        "source_plot_index": plot_index,
        "trace_mapping": {
            "time": selection.time_trace,
            "vds": selection.vds_trace,
            "id": selection.id_trace,
        },
        "event_windows_s": (
            {
                "turn_on": [
                    selection.turn_on.start_s,
                    selection.turn_on.end_s,
                ],
                "turn_off": [
                    selection.turn_off.start_s,
                    selection.turn_off.end_s,
                ],
            }
            if selection.step_selections is None
            else None
        ),
    }
    if selection.step_selections is not None:
        provenance.update(
            {
                "source_step_indices": [
                    item["step_index"] for item in selected_step_ranges(
                        parsed,
                        [item.step_index for item in selection.step_selections],
                    )
                ],
                "current_axis_i_x_by_source_step_a": current_by_step,
                "same_context_confirmed_by_user": selection.same_context_confirmed,
                "conditions_source": "explicit user-entered common metadata",
                "step_parameters_imported": False,
                "event_windows_are_step_local_seconds": True,
                "event_windows_by_source_step_s": event_windows_by_step,
            }
        )
    new_measurement = {
        "dataset_type": "dpt_u_i",
        **selection.metadata.model_dump(mode="json", exclude_none=True),
        "dpt_on_vds": on_acquisitions,
        "dpt_on_id": on_id_acquisitions,
        "dpt_off_vds": off_acquisitions,
        "dpt_off_id": off_id_acquisitions,
        "import_provenance": provenance,
    }

    async with _record_mutation():
        if review_required:
            _consume_dpt_raw_review_token(
                selection.review_token,
                _dpt_raw_review_signature(transistor_id, content, selection),
            )
        record_path = _legacy_record_path(transistor_id)
        record = _read_legacy_record_unlocked(
            transistor_id,
            max_file_bytes=MAX_DPT_LEGACY_RECORD_FILE_BYTES,
        )
        existing = record.get("raw_measurement_data", [])
        if existing is None:
            existing_records: list[dict[str, Any]] = []
        elif isinstance(existing, dict):
            existing_records = [existing]
        elif isinstance(existing, list) and all(
            isinstance(measurement, dict) for measurement in existing
        ):
            existing_records = existing
        else:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Stored raw_measurement_data is not a list of measurement "
                    "records; it was left unchanged."
                ),
            )

        candidate_raw_records = [*existing_records, new_measurement]
        raw_payload_size = _encoded_json_size_up_to(
            candidate_raw_records,
            MAX_DPT_RAW_MEASUREMENT_BYTES,
        )
        if raw_payload_size > MAX_DPT_RAW_MEASUREMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Appending this measurement would exceed the cumulative "
                    f"raw DPT payload limit of "
                    f"{_format_byte_limit(MAX_DPT_RAW_MEASUREMENT_BYTES)}. "
                    "Shorten the event windows or reduce existing raw DPT data; "
                    "the existing record was left unchanged."
                ),
            )

        candidate = {**record, "raw_measurement_data": candidate_raw_records}
        candidate_file_size = _encoded_json_size_up_to(
            candidate,
            MAX_DPT_LEGACY_RECORD_FILE_BYTES,
            indent=2,
        )
        if candidate_file_size > MAX_DPT_LEGACY_RECORD_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Appending this measurement would make the stored device JSON "
                    f"exceed the DPT file limit of "
                    f"{_format_byte_limit(MAX_DPT_LEGACY_RECORD_FILE_BYTES)}. "
                    "The existing record was left unchanged."
                ),
            )
        # Validate a deep copy of the model payload only. The legacy converter
        # mutates arrays in several nested branches, while copying raw waveform
        # records here would duplicate the largest part of the request memory.
        model_candidate = copy.deepcopy({
            key: value
            for key, value in candidate.items()
            if key != "raw_measurement_data"
        })
        try:
            # Validate the legacy transistor while the raw measurement payload
            # stays on its established top-level JSON persistence path.
            _json_dict_to_legacy_transistor(model_candidate)
            _write_legacy_record_unlocked(
                record_path,
                candidate,
                newline="\n",
            )
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"Could not append this DPT measurement: {error!s}",
            ) from error

    return {
        "success": True,
        "transistor_id": transistor_id,
        "raw_measurement_count": len(candidate["raw_measurement_data"]),
        "on_sample_count": sum(map(len, on_acquisitions)),
        "off_sample_count": sum(map(len, off_acquisitions)),
        "acquisition_count": len(on_acquisitions),
        "provenance": provenance,
    }


def _json_safe_dpt_value(value: Any) -> Any:
    """Convert raw DPT values into strict JSON values without mutating them."""
    if isinstance(value, np.ndarray):
        return _json_safe_dpt_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe_dpt_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe_dpt_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_dpt_value(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _is_dpt_sample_pair(value: Any) -> bool:
    """Return whether a sequence looks like one ``[time, value]`` sample."""
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        return False
    return all(
        not isinstance(item, (list, tuple, np.ndarray, dict))
        for item in value
    )


def _dpt_waveform_acquisitions(value: Any) -> list[Any]:
    """Return the stored acquisition sequences without copying their samples."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return [value]
        if value.ndim == 3:
            return list(value)
        return []
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(_is_dpt_sample_pair(sample) for sample in value):
            return [value]
        return list(value)
    return []


def _allocate_dpt_waveform_preview_limits(
    records: list[dict[str, Any]], requested_limit: int
) -> list[dict[str, list[int]]]:
    """Allocate a fair per-acquisition cap under one response point budget."""
    descriptors: list[tuple[int, str, int, int]] = []
    for record_index, record in enumerate(records):
        for field in _DPT_RAW_WAVEFORM_FIELDS:
            for acquisition_index, acquisition in enumerate(
                _dpt_waveform_acquisitions(record.get(field))
            ):
                try:
                    sample_count = len(acquisition)
                except TypeError:
                    sample_count = 0
                descriptors.append((record_index, field, acquisition_index, sample_count))

    minimum_points = sum(min(sample_count, 2) for *_, sample_count in descriptors)
    if minimum_points > MAX_DPT_WAVEFORM_PREVIEW_TOTAL_POINTS:
        raise HTTPException(
            status_code=413,
            detail=(
                "This record contains too many non-empty waveform acquisitions "
                "to retain their endpoints within the 40,000-point preview budget. "
                "Open a smaller set of measurements or reduce stored acquisitions."
            ),
        )

    counts = [sample_count for *_, sample_count in descriptors]
    low, high = 2, requested_limit
    while low <= high:
        candidate = (low + high) // 2
        if sum(min(sample_count, candidate) for sample_count in counts) <= (
            MAX_DPT_WAVEFORM_PREVIEW_TOTAL_POINTS
        ):
            low = candidate + 1
        else:
            high = candidate - 1

    limits = [min(sample_count, max(2, high)) for sample_count in counts]
    remaining = MAX_DPT_WAVEFORM_PREVIEW_TOTAL_POINTS - sum(limits)
    for index, sample_count in enumerate(counts):
        if remaining <= 0:
            break
        if limits[index] < min(sample_count, requested_limit):
            limits[index] += 1
            remaining -= 1

    allocated: list[dict[str, list[int]]] = [
        {field: [] for field in _DPT_RAW_WAVEFORM_FIELDS} for _ in records
    ]
    for (record_index, field, _acquisition_index, _sample_count), limit in zip(
        descriptors, limits, strict=True
    ):
        allocated[record_index][field].append(limit)
    return allocated


def _serialize_dpt_waveform_acquisitions(
    value: Any,
    point_limit: int | None = None,
    acquisition_limits: list[int] | None = None,
    projection_stats: dict[str, int] | None = None,
) -> list[Any]:
    """Serialize acquisitions, optionally keeping bounded endpoint samples."""
    acquisitions = _dpt_waveform_acquisitions(value)
    serialized: list[Any] = []
    for index, acquisition in enumerate(acquisitions):
        try:
            sample_count = len(acquisition)
        except TypeError:
            sample_count = 0
        limit = (
            acquisition_limits[index]
            if acquisition_limits is not None and index < len(acquisition_limits)
            else point_limit
        )
        if limit is not None and sample_count > limit:
            indices = np.linspace(
                0, sample_count - 1, num=limit, dtype=np.int64
            )
            if isinstance(acquisition, np.ndarray):
                selected = acquisition[indices]
            else:
                selected = [acquisition[int(index)] for index in indices]
            if projection_stats is not None:
                projection_stats["downsampled_acquisition_count"] = (
                    projection_stats.get("downsampled_acquisition_count", 0) + 1
                )
                projection_stats["returned_points"] = (
                    projection_stats.get("returned_points", 0) + len(indices)
                )
            serialized.append(_json_safe_dpt_value(selected))
        else:
            if projection_stats is not None:
                projection_stats["returned_points"] = (
                    projection_stats.get("returned_points", 0) + sample_count
                )
            serialized.append(_json_safe_dpt_value(acquisition))
    return serialized


def _serialize_dpt_raw_measurement(
    record: Any,
    point_limit: int | None = None,
    acquisition_limits: dict[str, list[int]] | None = None,
    projection_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Project one raw DPT record into a JSON-safe read-only response."""
    if isinstance(record, dict):
        source = record
    else:
        source = vars(record)
    metadata = {
        key: _json_safe_dpt_value(source[key])
        for key in source
        if key not in _DPT_RAW_WAVEFORM_FIELDS
    }
    return {
        "metadata": metadata,
        **{
            field: _serialize_dpt_waveform_acquisitions(
                source.get(field),
                point_limit,
                acquisition_limits.get(field) if acquisition_limits else None,
                projection_stats,
            )
            for field in _DPT_RAW_WAVEFORM_FIELDS
        },
    }


@app.get("/api/transistors/{transistor_id}/dpt_data")
async def get_dpt_data(
    transistor_id: str,
    waveform_preview_points: Optional[int] = Query(
        default=None, ge=2, le=MAX_DPT_WAVEFORM_PREVIEW_POINTS
    ),
) -> Dict[str, Any]:
    """Get DPT measurement data for a transistor.

    :param transistor_id: Transistor ID
    :param waveform_preview_points: Optional per-acquisition point cap for web previews
    :return: DPT data if available
    """
    preview_requested = waveform_preview_points is not None
    async with _record_mutation():
        raw_record = _read_legacy_record_unlocked(
            transistor_id,
            max_file_bytes=(
                MAX_DPT_LEGACY_RECORD_FILE_BYTES if preview_requested else None
            ),
        )
        raw_measurements = raw_record.get("raw_measurement_data")
        if preview_requested:
            raw_payload_size = _encoded_json_size_up_to(
                raw_measurements,
                MAX_DPT_RAW_MEASUREMENT_BYTES,
            )
            if raw_payload_size > MAX_DPT_RAW_MEASUREMENT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "The stored raw DPT payload exceeds the cumulative preview "
                        f"limit of "
                        f"{_format_byte_limit(MAX_DPT_RAW_MEASUREMENT_BYTES)}. "
                        "The opt-in bounded waveform preview stops before model "
                        "conversion or serialization; the stored record was left "
                        "unchanged."
                    ),
                )
        if isinstance(raw_measurements, dict) or preview_requested:
            # The legacy loader expects an iterable of measurement objects,
            # while older records can store one object directly. Raw waveform
            # arrays do not contribute to the DPT summary, so keep them out of
            # this model-only validation copy and avoid loading them twice.
            model_record = copy.deepcopy({
                key: value
                for key, value in raw_record.items()
                if key != "raw_measurement_data"
            })
            transistor = legacy_to_core(
                _json_dict_to_legacy_transistor(model_record)
            )
        else:
            transistor = _get_transistor(transistor_id)

    # Extract DPT data from switching loss data
    dpt_data = {
        'has_e_on': len(transistor.switch.e_on_data) > 0,
        'has_e_off': len(transistor.switch.e_off_data) > 0,
        'has_e_rr': len(transistor.diode.e_rr_data) > 0 if transistor.diode else False,
    }

    # Get measurement metadata from first dataset if available
    if transistor.switch.e_on_data:
        first = transistor.switch.e_on_data[0]
        dpt_data['metadata'] = {
            't_j': first.t_j,
            'v_supply': first.v_supply,
            'v_g': first.v_g,
            'r_g': first.r_g,
            'comment': first.comment or '',
        }

    raw_measurements = raw_record.get("raw_measurement_data")
    if isinstance(raw_measurements, dict):
        raw_measurements = [raw_measurements]
    raw_records = [
        record for record in raw_measurements if isinstance(record, dict)
    ] if isinstance(raw_measurements, list) else []
    projection_stats = {
        "downsampled_acquisition_count": 0,
        "returned_points": 0,
    }
    allocated_limits = (
        _allocate_dpt_waveform_preview_limits(
            raw_records, waveform_preview_points
        )
        if waveform_preview_points is not None
        else None
    )
    dpt_data["raw_measurement_data"] = [
        _serialize_dpt_raw_measurement(
            record,
            waveform_preview_points,
            allocated_limits[index] if allocated_limits is not None else None,
            projection_stats,
        )
        for index, record in enumerate(raw_records)
    ]
    if waveform_preview_points is not None:
        downsampled_count = projection_stats["downsampled_acquisition_count"]
        dpt_data["waveform_preview"] = {
            "max_points_per_acquisition": waveform_preview_points,
            "max_total_points": MAX_DPT_WAVEFORM_PREVIEW_TOTAL_POINTS,
            "returned_points": projection_stats["returned_points"],
            "sampling": "uniform_with_endpoints",
            "downsampled": downsampled_count > 0,
            "downsampled_acquisition_count": downsampled_count,
        }

    return dpt_data


def _read_dpt_energy_sources_unlocked(transistor_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read bounded full-fidelity raw DPT data while the caller holds the lock."""
    raw_record = _read_legacy_record_unlocked(
        transistor_id,
        max_file_bytes=MAX_DPT_LEGACY_RECORD_FILE_BYTES,
    )
    raw_measurements_value = raw_record.get("raw_measurement_data")
    raw_payload_size = _encoded_json_size_up_to(
        raw_measurements_value,
        MAX_DPT_RAW_MEASUREMENT_BYTES,
    )
    if raw_payload_size > MAX_DPT_RAW_MEASUREMENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "The stored raw DPT payload exceeds the full-fidelity energy "
                f"derivation limit of {_format_byte_limit(MAX_DPT_RAW_MEASUREMENT_BYTES)}. "
                "Shorten the imported event windows or reduce stored raw data."
            ),
        )
    try:
        raw_measurements = normalize_raw_measurements(raw_measurements_value)
    except DptEnergyDerivationError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return raw_record, raw_measurements


def _dpt_energy_preview_result(
    raw_measurements: list[dict[str, Any]],
    request: DptEnergyDerivationRequest,
) -> tuple[dict[str, Any], str]:
    """Derive exact event energy and source digest."""
    request_data = request.model_dump(mode="json")
    try:
        result = derive_dpt_energy(
            raw_measurements,
            request.loss_type,
            request_data["points"],
        )
    except DptEnergyDerivationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    source_sha256 = dpt_energy_source_token(
        raw_measurements,
        [point["measurement_index"] for point in request_data["points"]],
    )
    return result, source_sha256


def _same_switching_condition(left: float | None, right: float | None) -> bool:
    """Match the 1e-9 absolute condition tolerance used by TDB topology lookup."""
    if left is None or right is None:
        return left is right
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-9))


def _switching_curve_matches_dpt_context(
    curve: SwitchingLossData,
    loss_type: str,
    conditions: dict[str, Any],
) -> bool:
    """Mirror topology lookup conditions for one derived E_on/E_off curve."""
    if curve.dataset_type != "graph_i_e":
        return False
    if not _same_switching_condition(curve.t_j, conditions["t_j"]):
        return False
    if not _same_switching_condition(curve.v_supply, conditions["v_supply"]):
        return False
    if not _same_switching_condition(curve.r_g, conditions["r_g"]):
        return False
    if loss_type == "e_on":
        return _same_switching_condition(curve.v_g, conditions["v_g"]) and (
            curve.v_g_off is None
            or _same_switching_condition(curve.v_g_off, conditions["v_g_off"])
        )
    if curve.v_g_off is not None:
        return (
            _same_switching_condition(curve.v_g, conditions["v_g"])
            and _same_switching_condition(curve.v_g_off, conditions["v_g_off"])
        )
    return _same_switching_condition(curve.v_g, conditions["v_g_off"])


def _dpt_energy_provenances_from_comment(comment: Any) -> list[dict[str, Any]] | None:
    marker = "DPT-ENERGY-PROVENANCE: "
    if not isinstance(comment, str):
        return None
    values: list[dict[str, Any]] = []
    for line in reversed(comment.splitlines()):
        if line.startswith(marker):
            try:
                value = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return None
            if not isinstance(value, dict):
                return None
            values.append(value)
    return values or None


def _dpt_energy_source_acquisitions(
    provenances: list[dict[str, Any]],
) -> set[tuple[int, int]] | None:
    """Return every source acquisition already represented by provenance."""
    identities: set[tuple[int, int]] = set()
    for provenance in provenances:
        points = provenance.get("points")
        if not isinstance(points, list) or not points:
            return None
        for point in points:
            if not isinstance(point, dict):
                return None
            measurement_index = point.get("measurement_index")
            acquisition_index = point.get("acquisition_index")
            if (
                isinstance(measurement_index, bool)
                or not isinstance(measurement_index, int)
                or measurement_index < 0
                or isinstance(acquisition_index, bool)
                or not isinstance(acquisition_index, int)
                or acquisition_index < 0
            ):
                return None
            identity = (measurement_index, acquisition_index)
            if identity in identities:
                return None
            identities.add(identity)
    return identities


def _dpt_energy_graph_matches_provenance(
    provenances: list[dict[str, Any]],
    graph: Any,
) -> bool:
    """Require every persisted point to match a source point in the markers."""
    expected_points: list[tuple[float, float]] = []
    try:
        for provenance in provenances:
            for point in provenance["points"]:
                current = point["i_x"]
                energy = point["energy_j"]
                if (
                    isinstance(current, bool)
                    or not isinstance(current, (int, float))
                    or isinstance(energy, bool)
                    or not isinstance(energy, (int, float))
                    or not math.isfinite(float(current))
                    or not math.isfinite(float(energy))
                    or current <= 0
                    or energy <= 0
                ):
                    return False
                expected_points.append((float(current), float(energy)))
        expected = np.asarray(expected_points, dtype=np.float64).T
        actual = np.asarray(graph, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if (
        expected.ndim != 2
        or expected.shape[0] != 2
        or actual.ndim != 2
        or actual.shape[0] != 2
        or actual.shape[1] != expected.shape[1]
        or not np.isfinite(actual).all()
    ):
        return False
    expected = expected[:, np.argsort(expected[0], kind="stable")]
    actual = actual[:, np.argsort(actual[0], kind="stable")]
    return bool(np.array_equal(actual, expected))


def _dpt_energy_curve_conditions_match(
    curve: SwitchingLossData,
    conditions: dict[str, Any],
) -> bool:
    """Check stored curve fields against the provenance condition snapshot."""
    numeric_fields = (
        "t_j",
        "v_supply",
        "v_g",
        "v_g_off",
        "r_g",
        "load_inductance",
        "commutation_inductance",
    )
    if any(getattr(curve, field) != conditions.get(field) for field in numeric_fields):
        return False
    return all(
        getattr(curve, field) == conditions.get(field)
        for field in ("commutation_device", "measurement_testbench")
    )


def _find_dpt_energy_curve_target(
    transistor: Transistor,
    loss_type: str,
    result: dict[str, Any],
) -> int | None:
    """Find one safe same-context graph to extend, or reject ambiguity."""
    curves = (
        transistor.switch.e_on_data if loss_type == "e_on"
        else transistor.switch.e_off_data
    )
    matching_indices = [
        index
        for index, curve in enumerate(curves)
        if _switching_curve_matches_dpt_context(
            curve, loss_type, result["conditions"]
        )
    ]
    if len(matching_indices) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Multiple existing {loss_type} graph_i_e curves match the TDB "
                "temperature, voltage, gate, and resistance lookup conditions. "
                "Resolve that ambiguity before saving a derived curve."
            ),
        )
    if not matching_indices:
        return None

    curve_index = matching_indices[0]
    existing = curves[curve_index]
    provenance = _dpt_energy_provenances_from_comment(existing.comment)
    if provenance is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"An existing {loss_type} graph_i_e curve matches the TDB lookup "
                "conditions, but its commutation/test-bench context cannot be "
                "verified. It was left unchanged to avoid an ambiguous curve."
            ),
        )
    used_acquisitions = _dpt_energy_source_acquisitions(provenance)
    if (
        used_acquisitions is None
        or any(
            entry.get("version") != 1
            or entry.get("method") != "raw_window_signed_trapezoid"
            or entry.get("loss_type") != loss_type
            or entry.get("conditions") != result["conditions"]
            for entry in provenance
        )
        or not _dpt_energy_curve_conditions_match(existing, result["conditions"])
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"An existing {loss_type} curve matches the TDB lookup conditions, "
                "but its saved condition fields diverge from or cannot be verified "
                "against the DPT provenance. Restore the recorded conditions or "
                "create a separate curve."
            ),
        )
    if not _dpt_energy_graph_matches_provenance(provenance, existing.graph_i_e):
        raise HTTPException(
            status_code=409,
            detail=(
                f"The existing {loss_type} graph samples do not exactly match the "
                "i_x and energy values in its DPT provenance. Restore the recorded "
                "graph or create a separate curve."
            ),
        )
    requested_acquisitions = {
        (point["measurement_index"], point["acquisition_index"])
        for point in result["points"]
    }
    if used_acquisitions & requested_acquisitions:
        raise HTTPException(
            status_code=409,
            detail=(
                "A selected DPT acquisition is already represented in the matching "
                "curve. Each stored event window can be used only once per curve."
            ),
        )
    _curve_graph_for_dpt_merge(existing, result["graph_i_e"])
    return curve_index


def _curve_graph_for_dpt_merge(
    curve: SwitchingLossData,
    new_graph: list[list[float]],
) -> list[list[float]]:
    """Merge unique new current points into an exact-context graph."""
    try:
        existing_graph = np.asarray(curve.graph_i_e, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=409,
            detail="The existing same-context graph_i_e curve is not numeric.",
        ) from error
    if (
        existing_graph.ndim != 2
        or existing_graph.shape[0] != 2
        or existing_graph.shape[1] < 2
        or not np.isfinite(existing_graph).all()
        or np.any(np.diff(np.sort(existing_graph[0])) <= 0)
        or np.any(existing_graph[0] <= 0)
        or np.any(existing_graph[1] <= 0)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "The existing same-context graph_i_e curve has invalid or "
                "ambiguous current-energy samples and was left unchanged."
            ),
        )
    added_graph = np.asarray(new_graph, dtype=np.float64)
    if np.intersect1d(existing_graph[0], added_graph[0]).size:
        raise HTTPException(
            status_code=409,
            detail=(
                "A derived i_x duplicates a current already in the matching curve. "
                "Choose new distinct measured current values."
            ),
        )
    merged = np.concatenate((existing_graph, added_graph), axis=1)
    if merged.shape[1] > 50:
        raise HTTPException(
            status_code=409,
            detail=(
                "The merged switching-energy curve would exceed the 50-window "
                "maximum. Save this context in separate curves."
            ),
        )
    merged = merged[:, np.argsort(merged[0], kind="stable")]
    return merged.tolist()


def _dpt_energy_target_snapshot(
    raw_record: dict[str, Any],
    loss_type: str,
    curve_index: int | None,
) -> dict[str, Any]:
    """Capture the reviewed add/extend target and its optimistic edit token."""
    curves = _legacy_curve_list(raw_record, ("switch", loss_type))
    if curve_index is None:
        return {
            "save_action": "add",
            "existing_curve_index": None,
            "target_curve_count": len(curves),
            "target_edit_token": None,
        }
    if not 0 <= curve_index < len(curves) or not isinstance(curves[curve_index], dict):
        raise HTTPException(
            status_code=409,
            detail="The matching switching curve changed; reload and preview again.",
        )
    return {
        "save_action": "extend",
        "existing_curve_index": curve_index,
        "target_curve_count": len(curves),
        "target_edit_token": _curve_edit_token(curves[curve_index]),
    }


@app.post("/api/transistors/{transistor_id}/dpt_energy/preview")
async def preview_dpt_energy_curve(
    transistor_id: str,
    request: DptEnergyDerivationRequest,
) -> dict[str, Any]:
    """Preview a raw-window E_on/E_off curve without changing stored data."""
    async with _record_mutation():
        raw_record, raw_measurements = _read_dpt_energy_sources_unlocked(transistor_id)
        result, source_sha256 = _dpt_energy_preview_result(
            raw_measurements, request
        )
        transistor = _get_transistor(transistor_id)
        curve_index = _find_dpt_energy_curve_target(
            transistor, request.loss_type, result
        )
        target_snapshot = _dpt_energy_target_snapshot(
            raw_record, request.loss_type, curve_index
        )
        reviewed_request_sha256 = dpt_energy_request_token({
            **request.model_dump(mode="json"),
            **target_snapshot,
        })
    result["source_token"] = source_sha256
    result["request_token"] = reviewed_request_sha256
    result.update(target_snapshot)
    return result


@app.post("/api/transistors/{transistor_id}/dpt_energy/save")
@_record_mutation_endpoint
async def save_dpt_energy_curve(
    transistor_id: str,
    request: DptEnergySaveRequest,
) -> dict[str, Any]:
    """Save a confirmed, current preview while preserving all raw DPT records."""
    raw_record, raw_measurements = _read_dpt_energy_sources_unlocked(transistor_id)
    request_data = request.model_dump(mode="json")
    reviewed_request_sha256 = dpt_energy_request_token(request_data)
    if reviewed_request_sha256 != request.request_token:
        raise HTTPException(
            status_code=409,
            detail=(
                "The derivation inputs changed after preview. Preview the current "
                "event selections, current-axis values, and polarity choices again."
            ),
        )
    measurement_indices = [
        point["measurement_index"] for point in request_data["points"]
    ]
    if any(index >= len(raw_measurements) for index in measurement_indices):
        raise HTTPException(
            status_code=400,
            detail="A selected stored DPT measurement no longer exists.",
        )
    source_sha256 = dpt_energy_source_token(
        raw_measurements,
        measurement_indices,
    )
    if source_sha256 != request.source_token:
        raise HTTPException(
            status_code=409,
            detail=(
                "The selected raw DPT source data changed after preview. Review the "
                "current stored waveforms and preview the derivation again."
            ),
        )
    try:
        result = derive_dpt_energy(
            raw_measurements,
            request.loss_type,
            request_data["points"],
        )
    except DptEnergyDerivationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    transistor = _get_transistor(transistor_id)
    curve_index = _find_dpt_energy_curve_target(
        transistor, request.loss_type, result
    )
    current_target_snapshot = _dpt_energy_target_snapshot(
        raw_record, request.loss_type, curve_index
    )
    requested_target_snapshot = {
        "save_action": request.save_action,
        "existing_curve_index": request.existing_curve_index,
        "target_curve_count": request.target_curve_count,
        "target_edit_token": request.target_edit_token,
    }
    if current_target_snapshot != requested_target_snapshot:
        raise HTTPException(
            status_code=409,
            detail=(
                "The switching curve target changed after preview. Reload the curve "
                "list and preview the derivation again."
            ),
        )
    curve_comment = derivation_comment(result, source_sha256)
    curve_list = _core_curve_list(transistor, f"switch.{request.loss_type}")
    path = ("switch", request.loss_type)
    if curve_index is None:
        conditions = result["conditions"]
        new_curve = SwitchingLossData(
            dataset_type="graph_i_e",
            t_j=conditions["t_j"],
            v_supply=conditions["v_supply"],
            v_g=conditions["v_g"],
            v_g_off=conditions["v_g_off"],
            r_g=conditions["r_g"],
            graph_i_e=np.asarray(result["graph_i_e"], dtype=np.float64),
            comment=curve_comment,
            measurement_testbench=conditions["measurement_testbench"],
            commutation_device=conditions["commutation_device"],
            load_inductance=conditions["load_inductance"],
            commutation_inductance=conditions["commutation_inductance"],
        )
        curve_list.append(new_curve)
        _persist_curve_mutation(
            transistor_id,
            transistor,
            f"switch.{request.loss_type}",
            path,
        )
        saved_index = len(curve_list) - 1
        action = "added"
    else:
        if len(curve_list) != len(
            _legacy_curve_list(raw_record, path)
        ):
            raise HTTPException(
                status_code=409,
                detail="The stored switching curve list changed; reload and preview again.",
            )
        raw_curves = _legacy_curve_list(raw_record, path)
        if not 0 <= curve_index < len(raw_curves):
            raise HTTPException(
                status_code=409,
                detail="The matching switching curve changed; reload and preview again.",
            )
        existing_comment = raw_curves[curve_index].get("comment") or ""
        updated_comment = (
            f"{existing_comment}\n\n{curve_comment}" if existing_comment else curve_comment
        )
        merged_graph = _curve_graph_for_dpt_merge(
            curve_list[curve_index], result["graph_i_e"]
        )
        _update_legacy_curve_unlocked(
            transistor_id,
            "switch",
            request.loss_type,
            curve_index,
            CurveUpdateRequest(
                expected_count=len(raw_curves),
                expected_token=_curve_edit_token(raw_curves[curve_index]),
                curve={"graph_i_e": merged_graph, "comment": updated_comment},
            ),
        )
        saved_index = curve_index
        action = "extended"

    saved_curve_count = len(
        transistor.switch.e_on_data if request.loss_type == "e_on"
        else transistor.switch.e_off_data
    )
    return {
        "success": True,
        "loss_type": request.loss_type,
        "action": action,
        "curve_index": saved_index,
        "curve_count": saved_curve_count,
        "point_count": len(result["graph_i_e"][0]),
        "graph_i_e": result["graph_i_e"],
    }


@app.post("/api/transistors/{transistor_id}/dpt_validate")
async def validate_dpt_data(transistor_id: str) -> Dict[str, Any]:
    """Validate DPT measurement data against datasheet specifications.

    :param transistor_id: Transistor ID
    :return: Validation results
    """
    transistor = _get_transistor(transistor_id)

    errors = []
    warnings = []

    # Check if transistor has switching loss data
    if not transistor.switch.e_on_data and not transistor.switch.e_off_data:
        errors.append("No switching loss data available for validation")
        return {'valid': False, 'errors': errors, 'warnings': warnings}

    # Check voltage consistency
    v_supplies = set()
    for loss_data in transistor.switch.e_on_data + transistor.switch.e_off_data:
        v_supplies.add(loss_data.v_supply)

    if transistor.electrical_ratings.v_abs_max > 0:
        for v_supply in v_supplies:
            if v_supply > transistor.electrical_ratings.v_abs_max:
                errors.append(
                    f"Supply voltage {v_supply}V exceeds rated maximum "
                    f"{transistor.electrical_ratings.v_abs_max}V"
                )

    # Check current levels
    max_currents = []
    for loss_data in transistor.switch.e_on_data:
        if loss_data.graph_i_e and len(loss_data.graph_i_e[0]) > 0:
            max_currents.append(max(loss_data.graph_i_e[0]))

    if max_currents and transistor.electrical_ratings.i_abs_max > 0:
        max_test_current = max(max_currents)
        if max_test_current > transistor.electrical_ratings.i_abs_max:
            warnings.append(
                f"Test current {max_test_current}A exceeds rated maximum "
                f"{transistor.electrical_ratings.i_abs_max}A"
            )

    # Check temperature range
    temperatures = set()
    for loss_data in transistor.switch.e_on_data + transistor.switch.e_off_data:
        temperatures.add(loss_data.t_j)

    if temperatures:
        if min(temperatures) < -55:
            warnings.append(f"Test temperature {min(temperatures)}°C is very low")
        if max(temperatures) > transistor.electrical_ratings.t_j_max:
            errors.append(
                f"Test temperature {max(temperatures)}°C exceeds rated maximum "
                f"{transistor.electrical_ratings.t_j_max}°C"
            )

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'statistics': {
            'v_supply_range': [min(v_supplies), max(v_supplies)] if v_supplies else [0, 0],
            'i_max_tested': max(max_currents) if max_currents else 0,
            't_j_range': [min(temperatures), max(temperatures)] if temperatures else [0, 0],
        }
    }


# ==================== Switching-Pair Endpoints ====================


@app.get("/api/pairs/")
async def list_pairs(
    pair_type: str | None = None,
    device: str | None = None,
    manufacturer: str | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    source: str | None = None,
    page: int = 1,
    per_page: int = 20,
) -> Dict[str, Any]:
    """List switching pairs with filtering and pagination.

    Query parameters:
    - pair_type: Filter by pair type (mosfet_self, mosfet_plus_diode, etc.)
    - device: Filter by device ID
    - manufacturer: Filter by manufacturer
    - v_min: Minimum blocking voltage (V)
    - v_max: Maximum blocking voltage (V)
    - source: Filter by source (datasheet, dpt_measurement, etc.)
    - page: Page number (default: 1)
    - per_page: Results per page (default: 20)

    Returns: Dict with total count, pagination info, and list of pair summaries.
    """
    filters = {}
    if pair_type:
        filters['pair_type'] = pair_type
    if device:
        filters['device'] = device
    if manufacturer:
        filters['manufacturer'] = manufacturer
    if v_min is not None:
        filters['v_min'] = v_min
    if v_max is not None:
        filters['v_max'] = v_max
    if source:
        filters['source'] = source

    all_pairs = _pair_repo.list_pairs(filters=filters if filters else None)

    # Pagination
    total = len(all_pairs)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = all_pairs[start:end]

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pairs": paginated,
    }


@app.get("/api/pairs/{pair_id}")
async def get_pair(pair_id: str) -> Dict[str, Any]:
    """Get full switching pair data by ID.

    Returns: Complete SwitchingPair object with all metadata, electrical ratings,
    switching data, conduction curves, and thermal properties.
    """
    try:
        pair = _pair_repo.load_pair(pair_id)
        return pair.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Switching pair not found: {pair_id}")


@app.get("/api/pairs/{pair_id}/validate")
async def validate_pair(pair_id: str) -> Dict[str, Any]:
    """Validate a switching pair and return quality metrics.

    Returns: Validation result with errors, warnings, and quality_score (0-100).
    """
    try:
        pair = _pair_repo.load_pair(pair_id)
        return pair.validate()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Switching pair not found: {pair_id}")


@app.get("/api/devices/")
async def list_unique_devices(
    type: str | None = None,
    manufacturer: str | None = None,
) -> Dict[str, Any]:
    """List unique devices derived from all switching pairs.

    Query parameters:
    - type: Filter by device type (MOSFET, IGBT, Diode, etc.)
    - manufacturer: Filter by manufacturer

    Returns: Dict with total count and list of unique devices.
    """
    devices = _pair_repo.get_unique_devices()

    if type:
        devices = [d for d in devices if d.get('type', '').lower() == type.lower()]
    if manufacturer:
        devices = [d for d in devices if d.get('manufacturer', '').lower() == manufacturer.lower()]

    return {"total": len(devices), "devices": devices}


@app.get("/api/devices/{device_id}/pairs")
async def get_device_pairs(device_id: str) -> Dict[str, Any]:
    """Get all switching pairs containing a specific device.

    Returns: Dict with device_id, total count, and list of pair summaries.
    """
    results = _pair_repo.search_by_device(device_id)
    return {"device_id": device_id, "total": len(results), "pairs": results}


@app.get("/api/pairs/export/csv")
async def export_pairs_csv(
    pair_type: str | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
) -> StreamingResponse:
    """Export switching pairs as ScDataGeneric CSV for ntbee2.

    Query parameters:
    - pair_type: Filter by pair type
    - v_min: Minimum voltage (V)
    - v_max: Maximum voltage (V)

    Returns: CSV file with columns: name, blockingVoltage, pair_type, devices,
    thermal resistance, cost, weight, source, quality_score.
    """
    filters = {}
    if pair_type:
        filters['pair_type'] = pair_type
    if v_min is not None:
        filters['v_min'] = v_min
    if v_max is not None:
        filters['v_max'] = v_max

    pairs = _pair_repo.list_pairs(filters=filters if filters else None)

    # CSV columns matching ScDataGeneric format
    columns = [
        'name', 'blockingVoltage', 'pair_type', 'high_side_device', 'low_side_device',
        'high_side_count', 'low_side_count',
        'forwardThermalResistance', 'reverseThermalResistance',
        'cost', 'weight', 'source', 'quality_score',
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    for pair_summary in pairs:
        try:
            pair = _pair_repo.load_pair(pair_summary['pair_id'])
            pair_dict = pair.to_dict()

            hs = pair_dict.get('high_side', {})
            ls = pair_dict.get('low_side', {})

            row = {
                'name': pair_dict.get('pair_id', ''),
                'blockingVoltage': hs.get('electrical_ratings', {}).get('v_abs_max', 0),
                'pair_type': pair_dict.get('pair_type', ''),
                'high_side_device': hs.get('device_id', ''),
                'low_side_device': ls.get('device_id', ''),
                'high_side_count': hs.get('count', 1),
                'low_side_count': ls.get('count', 1),
                'forwardThermalResistance': hs.get('thermal_properties', {}).get('r_th_jc', 0),
                'reverseThermalResistance': hs.get('thermal_properties', {}).get('r_th_cs', 0),
                'cost': hs.get('metadata', {}).get('cost', 0) or 0,
                'weight': hs.get('metadata', {}).get('weight', 0) or 0,
                'source': pair_dict.get('switching_data', {}).get('source', ''),
                'quality_score': pair_dict.get('migration', {}).get('quality_score', 0),
            }
            writer.writerow(row)
        except Exception as e:
            logger.warning(f"Error exporting pair {pair_summary.get('pair_id')}: {e}")

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=switching_pairs.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


# ---------------------------------------------------------------------------
# Static file serving for the Vue 3 frontend (webgui flavour)
# ---------------------------------------------------------------------------

def _frontend_dist() -> Path | None:
    """Locate the Vue 3 dist/ directory (works in-source and PyInstaller).

    :return: Path to the dist/ directory, or None if not found.
    """
    import sys
    if getattr(sys, "frozen", False):
        # PyInstaller onefile: files are extracted to sys._MEIPASS
        candidate = Path(sys._MEIPASS) / "transistordatabase" / "gui_web" / "dist"
    else:
        candidate = Path(__file__).parent.parent / "dist"
    return candidate if candidate.exists() else None


_dist = _frontend_dist()
if _dist is not None:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
