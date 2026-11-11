"""Typed uploaded-file transport for nonexecuting catalog source operations."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

from transistordatabase.catalog_sources import (
    MAX_BYTES,
    build_snapshot,
    compare_snapshots,
    preview_source,
    static_links,
    workbook_coefficients,
)
from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict

router = APIRouter(prefix="/api/catalog-sources", tags=["Catalog source files"],
                   route_class=ParameterRoute)


class SourceFile(Strict):
    """Exact uploaded bytes and declared text parsing conventions."""

    filename: str = Field(min_length=1, max_length=240)
    content_base64: str = Field(min_length=1, max_length=4 * ((MAX_BYTES + 2) // 3))
    encoding: Literal["utf-8-sig", "utf-8", "cp1252", "utf-16"] = "utf-8-sig"
    delimiter: Literal[",", ";", "\t", "|"] = ","
    decimal: Literal[".", ","] = "."


class Preview(SourceFile):
    """A small row sample plus complete sheet/error/link census."""

    preview_rows: int = Field(default=20, ge=1, le=200)


class ColumnMapping(Strict):
    """Unambiguous letter (OOXML) or one-based index (CSV), and explicit unit."""

    column: str = Field(min_length=1, max_length=8)
    unit: str = Field(min_length=1, max_length=30)


class SourceMetadata(Strict):
    """Declared quote and measurement context, never inferred from file age."""

    currency: str | None = Field(default=None, max_length=30)
    snapshot: str | None = Field(default=None, max_length=1000)
    quote_quantity: float | None = Field(default=None, ge=0)
    technology: str = Field(default="Unknown", max_length=100)
    label: str | None = Field(default=None, max_length=240)
    conditions: dict[str, float | str | None] = Field(default_factory=dict, max_length=100)
    workbook_profile: Literal["mosfet", "igbt"] | None = None


class Snapshot(SourceFile):
    """Explicit row mapping creates a new portable snapshot, with original bytes."""

    sheet: str = Field(min_length=1, max_length=240)
    header_row: int = Field(default=1, ge=1, le=10000)
    mapping: dict[str, ColumnMapping] = Field(min_length=1, max_length=48)
    metadata: SourceMetadata


class CellReference(Strict):
    """Exact source cell identity for a link lookup."""

    sheet: str = Field(min_length=1, max_length=240)
    cell: str = Field(min_length=1, max_length=40)


class Links(SourceFile):
    """List every static link, or selected cells with explicit absent results."""

    cells: list[CellReference] | None = Field(default=None, max_length=2000)


class SnapshotRecords(Strict):
    """Snapshot records needed by an explicit-key source comparison."""

    records: list[dict] = Field(max_length=10000)


class Difference(Strict):
    """Comparison is a report only; there is no automatic application endpoint."""

    before: SnapshotRecords
    after: SnapshotRecords
    key_fields: Annotated[list[str], Field(min_length=1, max_length=5)]


def _run(operation, request):
    try:
        return operation(**request.model_dump())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/preview")
def preview(request: Preview) -> dict:
    """Inspect uploaded source cells and supported mapping fields."""
    return _run(preview_source, request)


@router.post("/snapshot")
def snapshot(request: Snapshot) -> dict:
    """Return a new immutable catalog JSON; no server files are written."""
    return _run(build_snapshot, request)


@router.post("/links")
def links(request: Links) -> dict:
    """Return stored/literal/plain links, retaining original target semantics."""
    return _run(static_links, request)


@router.post("/coefficients")
def coefficients(request: SourceFile) -> dict:
    """Return documented workbook J/A fields and independent formula/cache evidence."""
    return _run(workbook_coefficients, request)


@router.post("/diff")
def difference(request: Difference) -> dict:
    """Report added, changed and missing rows using explicitly selected keys."""
    return _run(compare_snapshots, request)
