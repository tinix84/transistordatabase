"""Validated in-memory downloads from complete core device records."""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
from typing import Literal
from urllib.parse import quote
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import Field, model_validator

from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict
from transistordatabase.gui_web.backend.plecs_routes import CoreSelectionRequest
from transistordatabase.native_export import render_gecko, render_matlab
from transistordatabase.plecs_document import render_core_plecs
from transistordatabase.plecs_core_export import render_core_selection

MAX_BYTES = 64 * 1024 * 1024


class GeckoSelection(Strict):
    """Declare all seven conditions, including each unknown gate condition."""

    component: Literal["switch"]
    t_j: float
    v_supply: float = Field(gt=0)
    v_g_on: float | None
    v_g_off: float | None
    r_g_on: float | None = Field(ge=0)
    r_g_off: float | None = Field(ge=0)


class EmptyOptions(Strict):
    """Formats without options reject ignored or filesystem parameters."""


class BatchOptions(Strict):
    """Choose one common slice or an exact device-to-slice mapping."""

    selection: GeckoSelection | CoreSelectionRequest | None = None
    selections: dict[str, GeckoSelection | CoreSelectionRequest] | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def exclusive_selection(self):
        """Reject ambiguous common and per-device selections."""
        if self.selection is not None and self.selections is not None:
            raise ValueError("Use selection or selections, not both")
        return self


class BatchDownload(Strict):
    """Validate every device before returning an archive."""

    transistor_ids: list[str] = Field(min_length=1, max_length=100)
    format: str
    options: BatchOptions = Field(default_factory=BatchOptions)


def _safe_name(name):
    """Refuse path syntax and ambiguous archive names without changing identity."""
    if (not isinstance(name, str) or not name.strip() or len(name) > 255
            or name in {".", ".."} or name.endswith((".", " "))
            or any(char in '<>:"/\\|?*' or ord(char) < 32
                   or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
                   for char in name)
            or name.split(".")[0].rstrip().lower() in {
                "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                *(f"lpt{i}" for i in range(1, 10))}):
        raise ValueError("Device IDs and export filenames must be safe plain names")
    return name


def _files(device, format, selection=None):
    """Render actual bytes without temporary files or browser number conversion."""
    name = _safe_name(device.metadata.name)
    if format not in ("gecko", "plecs") and selection is not None:
        raise ValueError("Conditions apply only to Gecko or PLECS export")
    if format == "json":
        return {f"{name}.json": json.dumps(core_to_json_document(device),
                ensure_ascii=True, allow_nan=False, indent=2).encode("utf-8")}
    if format == "matlab":
        return render_matlab(device)
    if format == "gecko":
        if not isinstance(selection, GeckoSelection):
            raise ValueError("Declare all seven Gecko conditions. Use null for recorded unknown gate conditions")
        return render_gecko(device, selection.model_dump())
    if format == "csv":
        stream = StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["name", "type", "manufacturer", "v_abs_max", "i_abs_max", "i_cont", "housing_type"])
        def text(value):
            return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
        writer.writerow([text(name), text(device.metadata.type), text(device.metadata.manufacturer),
                         device.electrical_ratings.v_abs_max, device.electrical_ratings.i_abs_max,
                         device.electrical_ratings.i_cont, text(device.metadata.housing_type)])
        return {f"{name}_ratings_summary.csv": stream.getvalue().encode("utf-8")}
    if format == "plecs":
        if selection is not None and not isinstance(selection, CoreSelectionRequest):
            raise ValueError("Use the PLECS source table selection contract")
        data = (render_core_selection(device, selection.model_dump()) if selection is not None
                else render_core_plecs(device))
        return {f"{name}.xml": data}
    if format in ("spice", "ltspice", "simulink"):
        raise ValueError("A calibrated simulator model is unavailable. Use the DPT workbench for explicit generic templates")
    raise ValueError("Supported downloads are json, csv, matlab, gecko, and source-backed or explicitly selected plecs")


def _archive(files):
    """Validate unique member names and output budgets before creating a ZIP."""
    seen = set()
    total = 0
    for name, data in files.items():
        for part in name.split("/"):
            _safe_name(part)
        if name.casefold() in seen:
            raise ValueError("Export archive member names must be unique")
        seen.add(name.casefold())
        total += len(data)
        if total > MAX_BYTES:
            raise ValueError("Export exceeds the 64 MiB uncompressed output limit")
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return stream.getvalue()


def _download(data, filename, media_type):
    if len(data) > MAX_BYTES:
        raise ValueError("Export exceeds the 64 MiB output limit")
    _safe_name(filename)
    fallback = "".join(char if ord(char) < 127 else "_" for char in filename)
    return Response(data, media_type=media_type, headers={
        "Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}",
        "X-Content-Type-Options": "nosniff",
    })


def create_export_router(get_transistor):
    """Use the application's repository lookup without a second repository."""
    router = APIRouter(tags=["Device downloads"], route_class=ParameterRoute)

    @router.post("/api/transistors/{transistor_id}/export/{format}")
    async def export_transistor(transistor_id: str, format: str,
                                request: GeckoSelection | CoreSelectionRequest | EmptyOptions = EmptyOptions()):
        try:
            _safe_name(transistor_id)
            device = get_transistor(transistor_id)
            files = _files(device, format, request if not isinstance(request, EmptyOptions) else None)
            if format == "gecko":
                return _download(_archive(files), f"{transistor_id}_gecko.zip", "application/zip")
            filename, data = next(iter(files.items()))
            media = {"json": "application/json", "csv": "text/csv", "matlab": "application/octet-stream", "plecs": "application/xml"}[format]
            return _download(data, filename, media)
        except (ValueError, OverflowError, NotImplementedError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/api/transistors/batch_export")
    async def batch_export(request: BatchDownload):
        try:
            ids = request.transistor_ids
            if len({name.casefold() for name in ids}) != len(ids):
                raise ValueError("Device IDs must be unique")
            for name in ids:
                _safe_name(name)
            options = request.options
            if request.format not in ("gecko", "plecs") and options.model_fields_set:
                raise ValueError("This format does not accept export options")
            if options.selections is not None and set(options.selections) != set(ids):
                raise ValueError("Per-device selections must match every requested device ID exactly")
            files = {}
            total = 0
            for name in ids:
                selection = (options.selections[name] if options.selections is not None else options.selection)
                for filename, data in _files(get_transistor(name), request.format, selection).items():
                    member = f"{name}/{filename}"
                    if member in files:
                        raise ValueError("Export archive member names must be unique")
                    total += len(data)
                    if total > MAX_BYTES:
                        raise ValueError("Export exceeds the 64 MiB uncompressed output limit")
                    files[member] = data
            return _download(_archive(files), f"transistor_exports_{request.format}.zip", "application/zip")
        except (ValueError, OverflowError, NotImplementedError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/transistors/{transistor_id}/export/{format}/preview")
    async def export_preview(transistor_id: str, format: str):
        try:
            _safe_name(transistor_id)
            if format in ("spice", "ltspice", "simulink"):
                raise ValueError("A calibrated simulator model is unavailable. Use the DPT workbench for explicit generic templates")
            if format not in ("json", "csv", "plecs"):
                raise ValueError("Preview supports native JSON, ratings summary CSV, and source-backed PLECS only")
            filename, data = next(iter(_files(get_transistor(transistor_id), format).items()))
            if len(data) > MAX_BYTES:
                raise ValueError("Export exceeds the 64 MiB output limit")
            lines = data.decode("utf-8").splitlines()
            return {"preview": "\n".join(lines[:100]), "total_lines": len(lines),
                    "preview_lines": min(100, len(lines)), "file_size": len(data),
                    "file_name": filename, "truncated": len(lines) > 100}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
