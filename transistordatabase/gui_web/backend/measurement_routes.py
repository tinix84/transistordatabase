"""Strict transport for source-preserving DPT waveform and log imports."""
from importlib.util import find_spec
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field, field_validator

from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict
from transistordatabase.measurement_import import (
    MeasurementLimitError, extract_measurement, preview_measurement,
)


class TraceMapping(Strict):
    """Explicit column or trace identity, units, and probe direction."""

    name: str = Field(min_length=1, max_length=300)
    unit: str = Field(min_length=1, max_length=20)
    polarity: Literal[1, -1]

    @field_validator("polarity", mode="before")
    @classmethod
    def numeric_polarity(cls, value):
        """Reject boolean aliases for the integer probe direction."""
        if type(value) is not int:
            raise ValueError("Probe polarity must be the integer 1 or -1")
        return value


class Mappings(Strict):
    """Null mappings permit discovery without extracting arrays."""

    time: TraceMapping | None = None
    voltage: TraceMapping | None = None
    current: TraceMapping | None = None
    gate: TraceMapping | None = None


class CsvOptions(Strict):
    """Declared CSV syntax and an explicit step selector."""

    delimiter: str = Field(default=",", min_length=1, max_length=1)
    decimal: Literal[".", ","] = "."
    encoding: Literal["utf-8", "utf-8-sig", "utf-16", "cp1252"] = "utf-8-sig"
    step_column: str | None = None
    step_value: str | None = None


class LogMapping(Strict):
    """Supplied integrals and independent declared current and bus variables."""

    energy_on: str | None = None
    energy_off: str | None = None
    current: str | None = None
    bus: str | None = None
    energy_unit: str = "J"
    current_unit: str = "A"
    bus_unit: str = "V"


class MeasurementRequest(Strict):
    """Original source bytes and user-selected mappings."""

    format: Literal["csv", "raw", "log"]
    filename: str = Field(min_length=1, max_length=500)
    content_base64: str = Field(min_length=1, max_length=10_666_668)
    source_kind: Literal["measurement", "simulation", "manual", "unknown"]
    disclosure: str | None = Field(default=None, max_length=4000)
    companion_log_base64: str | None = Field(default=None, max_length=10_666_668)
    companion_log_filename: str | None = Field(default=None, max_length=500)
    step_index: int | None = Field(default=None, ge=0, le=9999)
    mappings: Mappings = Field(default_factory=Mappings)
    csv: CsvOptions = Field(default_factory=CsvOptions)
    log: LogMapping = Field(default_factory=LogMapping)


class Conditions(Strict):
    """Independent conditions without inferred device ratings."""

    t_j: float | None = Field(ge=-273.15)
    bus_voltage_v: float | None = Field(gt=0)
    v_g_on: float | None
    v_g_off: float | None
    r_g_on: float | None = Field(ge=0)
    r_g_off: float | None = Field(ge=0)
    commutation_device: str | None
    manual_override: str | None = None


class ExtractionRequest(MeasurementRequest):
    """Preview inputs plus explicit conditions and an optional native core target."""

    conditions: Conditions
    device_document: dict | None = None
    device_document_text: str | None = Field(default=None, max_length=40_000_000)


router = APIRouter(prefix="/api/measurement", tags=["Measurement imports"],
                   route_class=ParameterRoute)


@router.get("/capabilities")
def capabilities():
    """Report installed parsers, supported scope, and import budgets."""
    return {"csv": True, "raw": find_spec("spicelib") is not None,
            "log": find_spec("spicelib") is not None, "max_bytes": 8_000_000,
            "max_points": 100_000, "max_expanded_cells": 500_000,
            "scope": "Explicit source import only. No simulator execution or automatic database save."}


@router.post("/preview")
def preview(request: MeasurementRequest):
    """Preview declared mappings and source identity without persistence."""
    try:
        return preview_measurement(request.model_dump())
    except MeasurementLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/extract")
def extract(request: ExtractionRequest):
    """Return source-associated core points and a reviewable optional native archive."""
    try:
        source = request.model_dump(exclude={"conditions", "device_document", "device_document_text"})
        if request.device_document is not None and request.device_document_text is not None:
            raise ValueError("Supply one native document representation")
        native = json.loads(request.device_document_text) if request.device_document_text is not None else request.device_document
        return extract_measurement(source, request.conditions.model_dump(), native)
    except MeasurementLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
