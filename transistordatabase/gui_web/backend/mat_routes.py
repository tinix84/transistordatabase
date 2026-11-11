"""Typed byte-only MAT preview, archive and explicit branch finalization."""
from __future__ import annotations

import base64
from hashlib import sha256
import math
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import Field, JsonValue

from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict
from transistordatabase.gui_web.backend.plecs_routes import Areas, Ratings
from transistordatabase.legacy_mat import (
    MAX_SOURCE_BYTES, export_mat, finalize_mat, mat_array_blob, preview_mat,
)


class MatSourceRequest(Strict):
    """Original uploaded bytes. A filename is provenance only."""

    mat_base64: str = Field(min_length=1, max_length=(MAX_SOURCE_BYTES + 2) // 3 * 4)
    source_name: str | None = Field(default=None, max_length=2000)

    def source_bytes(self):
        """Decode a bounded source without filesystem access."""
        try:
            return base64.b64decode(self.mat_base64, validate=True)
        except ValueError as exc:
            raise ValueError("mat_base64 is not valid base64") from exc


class MatExportRequest(MatSourceRequest):
    """Choose exact source preservation or an explicit legacy format conversion."""

    mode: Literal["source-preserved", "legacy-mat5"]


class MatArrayRequest(MatSourceRequest):
    """Resolve a preview array reference. This is a field path, not a filesystem path."""

    array_path: list[Annotated[str, Field(min_length=1, max_length=1024)]] = Field(min_length=1, max_length=17)


class MatIdentity(Strict):
    """Separately declared core identity. No filename inference is permitted."""

    name: str = Field(min_length=1, max_length=1000)
    type: str = Field(min_length=1, max_length=1000)
    author: str = Field(min_length=1, max_length=1000)
    manufacturer: str = Field(min_length=1, max_length=1000)
    housing_type: str = Field(min_length=1, max_length=1000)


class MatConditions(Strict):
    """Independent conditions for this exact original branch curve."""

    t_j_c: float
    gate_voltage_v: float | None
    gate_off_voltage_v: float | None
    gate_resistance_ohm: float | None = Field(ge=0)
    supply_voltage_v: float | None = Field(gt=0)
    commutation_device: str | None = Field(max_length=2000)


class MatCurve(Strict):
    """Explicit branch domain, original sample selection and core destination."""

    kind: Literal["conduction", "turn-on", "turn-off"]
    branch: Literal["forward", "reverse"]
    target: Literal["switch", "diode"]
    source: Literal["original-trace", "combined"]
    coordinate_convention: Literal["trace-positive-magnitudes", "signed-passive"]
    sample_indices: list[int] | None = Field(max_length=20000)
    sample_order: Literal["recorded-order", "sort-distinct-current"]
    zero_policy: Literal["reject-zero", "recorded-branch-value"]
    conditions: MatConditions
    evidence_reference: str = Field(min_length=1, max_length=2000)
    extrapolation_policy: Literal["reject", "explicit-estimate"] = "reject"
    data_status: Literal["recorded-source-values", "source-zero-unverified", "declared-ideal-model"] = "recorded-source-values"
    ideal_model_scope: str | None = Field(default=None, min_length=1, max_length=2000)
    source_zero_acknowledgement: str | None = Field(default=None, min_length=1, max_length=2000)


class MatFinalizeRequest(MatSourceRequest):
    """Supply the missing scientific choices without changing source data."""

    identity: MatIdentity
    ratings: Ratings
    thermal_properties: Areas
    curves: list[MatCurve] = Field(min_length=1, max_length=6)
    omission_reasons: dict[str, Annotated[str, Field(min_length=1, max_length=2000)]] = Field(max_length=6)
    units: Literal["SC-editor-SI-confirmed"]
    declaration_source: str = Field(min_length=1, max_length=2000)


class MatCalculateRequest(MatFinalizeRequest):
    """Use the uniquely mapped channel at a positive bounded current."""

    target: Literal["switch", "diode"]
    current_a: float = Field(gt=0)


class MatArrayDescriptor(Strict):
    """Typed small values or an original source-array reference."""

    kind: Literal["array"]
    dtype: str
    itemsize: int
    shape: list[int]
    path: str
    values: list[JsonValue] | None = None
    encoding: Literal["source-array-reference"] | None = None
    byte_length: int | None = None
    sha256: str | None = None
    source_array_path: list[str] | None = None
    nonfinite: bool | None = None
    role: Literal["original-image"] | None = None


class MatStructDescriptor(Strict):
    """Original scalar structure fields retain their names and array shapes."""

    kind: Literal["struct"]
    shape: list[int]
    fields: dict[str, MatArrayDescriptor | MatStructDescriptor]


class MatSource(Strict):
    """Original source identity, independent of any declared core identity."""

    name: str | None
    sha256: str
    byte_length: int
    format: Literal["mat5", "octave-text-subset"]


class MatDiagnostic(Strict):
    """A source location and a container or scientific limitation."""

    path: str
    code: str
    message: str


class MatPreview(Strict):
    """Versioned original-byte archive with a bounded typed field tree."""

    schema_name: Literal["transistordatabase-legacy-mat"] = Field(alias="schema")
    version: Literal[1]
    parser_version: Literal[1]
    workspace_schema: Literal["scData", "hsData", "digitizer-traces", "numeric-workspace"]
    source: MatSource
    source_base64: str
    source_version: str | None
    conversion_history: list[dict[str, str]]
    variables: dict[str, MatArrayDescriptor | MatStructDescriptor]
    known_fields: list[str]
    unknown_fields: list[str]
    proposed_units: dict[str, JsonValue]
    legacy_coordinates: Literal["x/image-height,y/image-width"]
    calibrations: dict[str, JsonValue]
    diagnostics: list[MatDiagnostic]
    status: Literal["preview-only-requires-explicit-finalization"]


class MatArrayBlob(Strict):
    """One binary array with exact dtype, dimensions and provenance."""

    source_sha256: str
    array_path: list[str]
    dtype: str
    itemsize: int
    shape: list[int]
    encoding: Literal["base64-c-order"]
    byte_length: int
    sha256: str
    data_base64: str


class MatCalculation(Strict):
    """An existing bounded core result with source and declaration evidence."""

    status: Literal["supported", "supported-estimate"]
    target: Literal["switch", "diode"]
    current_a: float
    voltage_v: float
    resistance_ohm: float
    current_coverage_a: list[float]
    conditions: MatConditions
    evidence_reference: str
    extrapolation_policy: Literal["reject", "explicit-estimate"]
    data_status: Literal["recorded-source-values", "source-zero-unverified", "declared-ideal-model"]
    declaration_source: str
    source_sha256: str
    method: str
    method_version: Literal[1]
    units: dict[str, str]


router = APIRouter(prefix="/api/mat", tags=["Legacy MAT migration"], route_class=ParameterRoute)


@router.post("/preview", response_model=MatPreview, response_model_exclude_unset=True)
def preview(request: MatSourceRequest):
    """Preview actual arrays and unknown fields separately from physical validity."""
    try:
        return preview_mat(request.source_bytes(), request.source_name)
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/export")
def export(request: MatExportRequest):
    """Return exact source bytes or a semantic MAT5 conversion without persistence."""
    try:
        content = export_mat(request.source_bytes(), mode=request.mode)
        return Response(content, media_type="application/octet-stream", headers={
            "Content-Disposition": 'attachment; filename="legacy-sc-data.mat"',
            "X-MAT-Export-Scope": request.mode,
            "X-MAT-Loss-Report": "Source fields preserved; new core declarations require the native JSON document",
        })
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/array", response_model=MatArrayBlob)
def array_blob(request: MatArrayRequest):
    """Return one original image or array as an explicitly requested bounded binary blob."""
    try:
        return mat_array_blob(request.source_bytes(), request.array_path)
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/finalize")
def finalize(request: MatFinalizeRequest):
    """Return reviewable native JSON. No repository write occurs."""
    try:
        arguments = request.model_dump(exclude={"mat_base64", "source_name"})
        transistor = finalize_mat(request.source_bytes(), source_name=request.source_name, **arguments)
        return {"status": "finalized-not-persisted", "declarations": arguments,
                "source_name": request.source_name,
                "core_json_document": core_to_json_document(transistor),
                "limitations": [
                    "Only the selected branches enter the existing two-component core",
                    "All original branches, images and unknown fields remain in native source provenance",
                    "No interpolation across branch zero gaps or voltage normalization is applied",
                ]}
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/calculate", response_model=MatCalculation)
def calculate(request: MatCalculateRequest):
    """Delegate the explicit mapped channel to the existing bounded core lookup."""
    try:
        raw = request.source_bytes()
        arguments = request.model_dump(exclude={"mat_base64", "source_name", "target", "current_a"})
        transistor = finalize_mat(raw, source_name=request.source_name, **arguments)
        channels = getattr(transistor, request.target).channel_data
        if len(channels) != 1:
            raise ValueError("Select a target with exactly one explicitly mapped conduction curve")
        curve = channels[0]
        resistance = curve.get_resistance_at_current(request.current_a)
        voltage = resistance * request.current_a
        if not math.isfinite(resistance) or not math.isfinite(voltage):
            raise ValueError("Bounded channel result exceeds finite numeric range")
        selection = next(c for c in arguments["curves"] if c["kind"] == "conduction" and c["target"] == request.target)
        return {"status": "supported-estimate" if selection["extrapolation_policy"] == "explicit-estimate" else "supported", "target": request.target,
                "current_a": request.current_a, "voltage_v": voltage, "resistance_ohm": resistance,
                "current_coverage_a": [float(curve.graph_v_i[1, 0]), float(curve.graph_v_i[1, -1])],
                "conditions": selection["conditions"], "evidence_reference": selection["evidence_reference"],
                "extrapolation_policy": selection["extrapolation_policy"], "data_status": selection["data_status"],
                "declaration_source": request.declaration_source, "source_sha256": sha256(raw).hexdigest(),
                "method": "ChannelCharacteristics.get_resistance_at_current; bounded linear interpolation",
                "method_version": 1, "units": {"current": "A", "voltage": "V", "resistance": "ohm"}}
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
