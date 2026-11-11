"""Typed, bounded transport for original-byte XYZ energy imports."""
from __future__ import annotations

import base64
import binascii
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import Field, StrictInt

from transistordatabase.gui_web.backend.analysis_routes import FixedConditions
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.gui_web.backend.parameter_routes import Strict
from transistordatabase.xyz_energy_import import (
    MAX_RAW_BYTES, MAX_RESPONSE_BYTES, XyzEnergyLimitError, import_xyz_energy_map,
)


class XyzRoute(FiniteJsonRoute):
    """Permit bounded base64 overhead independently of the raw source bound."""

    max_body_bytes = 6_000_000


class Columns(Strict):
    """Three different zero-based column positions, checked by the importer."""

    current: StrictInt = Field(ge=0, le=2)
    voltage: StrictInt = Field(ge=0, le=2)
    energy: StrictInt = Field(ge=0, le=2)


class Units(Strict):
    """Explicit source units without inferred prefixes."""

    current: Literal["A", "mA"]
    voltage: Literal["V", "kV"]
    energy: Literal["J", "mJ", "µJ", "uJ"]


class Options(Strict):
    """One declared delimiter, optional header, and unknown-energy vocabulary."""

    delimiter: Literal["csv", "tsv", "whitespace"]
    decimal: Literal["."]
    has_header: bool
    columns: Columns
    units: Units
    missing_energy_tokens: Annotated[list[Annotated[str, Field(max_length=128)]],
                                     Field(max_length=20)]


class XyzRequest(Strict):
    """Supplied source bytes and full independent metadata."""

    content_base64: str = Field(min_length=1, max_length=5_333_336)
    source_name: str = Field(min_length=1, max_length=500)
    input_kind: Literal["file", "paste"]
    source: str = Field(min_length=1, max_length=2000)
    source_kind: Literal["measured", "datasheet", "simulated", "manual", "unknown"]
    energy_kind: Literal["e_on", "e_off", "e_rr"]
    conditions: FixedConditions
    options: Options


Status = Literal["available", "missing", "unknown", "ambiguous"]


class Cell(Strict):
    """A coordinate refers to all source rows and one unambiguous known value."""

    current_a: float
    voltage_v: float
    status: Status
    energy_j: float | None
    row_ids: list[str]
    reason: str | None


class SourceRow(Strict):
    """Original row text, parsed tokens, and exact SI coordinate identities."""

    id: str
    line_number: int
    raw_record: str
    lexemes: list[str]
    current_decimal_key: str
    voltage_decimal_key: str
    current_a: float
    voltage_v: float
    energy_j: float | None
    status: Literal["available", "unknown"]
    cell_status: Literal["available", "unknown", "ambiguous"]
    reason: str | None


class Source(Strict):
    """A hash applies to the supplied bytes and the declared origin scope."""

    name: str
    source: str
    source_kind: Literal["measured", "datasheet", "simulated", "manual", "unknown"]
    input_kind: Literal["file", "paste"]
    sha256: str
    byte_count: int
    hash_scope: Literal["original-file-bytes", "utf8-paste-bytes"]
    encoding: Literal["utf-8"]


class Header(Strict):
    """Retain the declared header without deriving units or column positions."""

    line_number: int
    raw_record: str
    lexemes: list[str]


class Census(Strict):
    """Count source observations independently from grid cell availability."""

    source_rows: int
    available_rows: int
    unknown_rows: int
    available_cells: int
    missing_cells: int
    unknown_cells: int
    ambiguous_cells: int
    duplicate_coordinates: int
    total_cells: int


class XyzResponse(Strict):
    """The import result retains holes, source evidence, and request identity."""

    method_version: str
    energy_kind: Literal["e_on", "e_off", "e_rr"]
    conditions: FixedConditions
    source: Source
    options: Options
    request_sha256: str
    currents_a: list[float]
    voltages_v: list[float]
    current_decimal_keys: list[str]
    voltage_decimal_keys: list[str]
    units: Units
    cells: list[list[Cell]]
    rows: list[SourceRow]
    header: Header | None
    census: Census
    limits: dict[str, int]
    assumptions: list[str]


router = APIRouter(prefix="/api/analysis/energy-map", tags=["XYZ energy import"],
                   route_class=XyzRoute)


@router.post("/xyz", response_model=XyzResponse)
def import_xyz(request: XyzRequest):
    """Decode one original source. Do not access files or write database state."""
    try:
        raw = base64.b64decode(request.content_base64, validate=True)
        if len(raw) > MAX_RAW_BYTES:
            raise XyzEnergyLimitError("XYZ source exceeds the 4 MB limit")
        result = import_xyz_energy_map(raw, request.model_dump(exclude={"content_base64"}))
        # Validate the public response before its exact encoded-byte bound check.
        response = XyzResponse.model_validate(result)
        encoded = response.model_dump_json().encode("utf-8")
        if len(encoded) > MAX_RESPONSE_BYTES:
            raise XyzEnergyLimitError("XYZ result exceeds the 24 MB limit")
        return JSONResponse(content=response.model_dump())
    except XyzEnergyLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, TypeError, OverflowError, binascii.Error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
