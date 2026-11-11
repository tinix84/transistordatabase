"""Typed PLECS document preview, archive and explicit core mapping transport."""
from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, is_dataclass
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import Field, model_validator

from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict
from transistordatabase.plecs_document import (
    MAX_XML_BYTES, PlecsDocument, finalize_plecs_package,
    parse_plecs_document, render_core_plecs, render_plecs_document,
)
from transistordatabase.plecs_core_export import core_selection_document, render_core_selection
from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.core.repository import RepositoryPathError


class PlecsSourceRequest(Strict):
    """Exactly one Unicode text or original byte-preserving base64 source."""

    xml_text: str | None = Field(default=None, max_length=MAX_XML_BYTES)
    xml_base64: str | None = Field(default=None, max_length=(MAX_XML_BYTES + 2) // 3 * 4)
    source_name: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def one_source(self):
        """Require a single source identity instead of conflicting representations."""
        if (self.xml_text is None) == (self.xml_base64 is None):
            raise ValueError("Provide exactly one of xml_text or xml_base64")
        return self

    def document(self):
        """Decode bytes explicitly; never infer source conditions from names."""
        source = self.xml_text
        if self.xml_base64 is not None:
            try:
                source = base64.b64decode(self.xml_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("xml_base64 is not valid base64") from exc
        return parse_plecs_document(source, self.source_name)


class PlecsExportRequest(PlecsSourceRequest):
    """Supported model rendering differs explicitly from unvalidated source archive."""

    mode: Literal["supported-tables", "source-preserved"]


class Ratings(Strict):
    """Separately supplied datasheet limits, never curve coverage."""

    v_abs_max: float = Field(gt=0)
    i_abs_max: float = Field(gt=0)
    i_cont: float = Field(gt=0)
    t_j_max: float


class Areas(Strict):
    """Separately supplied physical areas and optional case interface values."""

    housing_area: float = Field(gt=0)
    cooling_area: float = Field(gt=0)
    r_th_cs: float | None = Field(default=None, ge=0)
    r_th_switch_cs: float | None = Field(default=None, ge=0)
    r_th_diode_cs: float | None = Field(default=None, ge=0)
    t_c_max: float | None = None


class PlecsFinalizeRequest(PlecsSourceRequest):
    """Declarations are distinct from the original unknown source metadata."""

    package_id: str
    ratings: Ratings
    thermal_properties: Areas
    block_targets: dict[str, Literal["switch", "diode"]]
    gate_voltage_by_table: dict[str, float] = Field(default_factory=dict)
    recovery_voltage_convention: Literal["preserve-signed", "negative-to-positive-bus"]
    package_thermal_target: Literal["switch", "diode"] | None = None


class ConductionSelection(Strict):
    """A required exact gate selector, including an explicit unknown null."""

    gate_voltage_v: float | None


class EnergySelection(ConductionSelection):
    """Independent exact energy gate/Rg context, without guessed defaults."""

    gate_resistance_ohm: float | None


class CoreSelectionRequest(Strict):
    """Exact fixed contexts and explicit omissions for an existing core record."""

    component: Literal["switch", "diode"]
    conduction: ConductionSelection | None
    turn_on: EnergySelection | None
    turn_off: EnergySelection | None
    include_thermal: bool
    omission_reasons: dict[str, str]
    recovery_voltage_convention: Literal["preserve-signed", "positive-bus-to-negative-plecs"]


def plecs_repository():
    """Require the application's existing repository without creating an alternate path."""
    raise HTTPException(status_code=503, detail="PLECS core repository is not configured")


def _repository_transistor(repository, transistor_id):
    try:
        return repository.get_by_name(transistor_id)
    except RepositoryPathError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


router = APIRouter(prefix="/api/plecs", tags=["PLECS documents"], route_class=ParameterRoute)


@router.post("/preview", response_model=PlecsDocument)
def preview(request: PlecsSourceRequest):
    """Return actual table evidence and structured unsupported content, without persistence."""
    try:
        return request.document()
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/export")
def export(request: PlecsExportRequest):
    """Download exact original bytes or validated supported numeric XML."""
    try:
        content = render_plecs_document(request.document(), mode=request.mode)
        return Response(content, media_type="application/xml", headers={
            "Content-Disposition": 'attachment; filename="plecs-document.xml"',
            "X-PLECS-Export-Scope": request.mode,
        })
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _json_core(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return _json_core(asdict(value))
    if isinstance(value, dict):
        return {key: _json_core(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_core(item) for item in value]
    return value


@router.post("/finalize")
def finalize(request: PlecsFinalizeRequest):
    """Produce reviewable core data and a verified XML roundtrip without saving it."""
    try:
        arguments = request.model_dump(exclude={"xml_text", "xml_base64", "source_name"})
        document = request.document()
        transistor = finalize_plecs_package(document, **arguments)
        content = render_core_plecs(transistor)
        components = {}
        for name in ("switch", "diode"):
            component = getattr(transistor, name)
            components[name] = {key: _json_core(getattr(component, key)) for key in (
                "channel_data", "thermal_foster", "e_on_data", "e_off_data", "e_rr_data"
            ) if hasattr(component, key)}
        return {"status": "finalized-not-persisted", "declarations": arguments,
                "source": asdict(document.source),
                "core": {"metadata": asdict(transistor.metadata),
                         "electrical_ratings": asdict(transistor.electrical_ratings),
                         "thermal_properties": asdict(transistor.thermal_properties), **components},
                "core_json_document": core_to_json_document(transistor),
                "roundtrip_xml_base64": base64.b64encode(content).decode("ascii")}
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/core/{transistor_id}/preview")
def core_preview(transistor_id: str, request: CoreSelectionRequest,
                 repository=Depends(plecs_repository)):
    """Preview exact selected recorded curves and every omission before exporting."""
    transistor = _repository_transistor(repository, transistor_id)
    if transistor is None:
        raise HTTPException(status_code=404, detail="Core transistor not found")
    try:
        document, report = core_selection_document(transistor, request.model_dump())
        return {"document": document.to_dict(), "report": report}
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/core/{transistor_id}/export")
def core_export(transistor_id: str, request: CoreSelectionRequest,
                repository=Depends(plecs_repository)):
    """Render the exact explicit subset directly to bytes, without cwd or file staging."""
    transistor = _repository_transistor(repository, transistor_id)
    if transistor is None:
        raise HTTPException(status_code=404, detail="Core transistor not found")
    try:
        return Response(render_core_selection(transistor, request.model_dump()),
                        media_type="application/xml", headers={
                            "Content-Disposition": 'attachment; filename="plecs-core-selection.xml"',
                            "X-PLECS-Export-Scope": "exact-core-subset",
                        })
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/core/{transistor_id}/contexts")
def core_contexts(transistor_id: str, repository=Depends(plecs_repository)):
    """List valid recorded contexts and per-record diagnostics without default selection."""
    import json
    from transistordatabase.plecs_core_export import _finite

    transistor = _repository_transistor(repository, transistor_id)
    if transistor is None:
        raise HTTPException(status_code=404, detail="Core transistor not found")
    result, diagnostics = {}, []
    record_budget = 0
    for component_name in ("switch", "diode"):
        component = getattr(transistor, component_name)
        result[component_name] = {"thermal_available": component.thermal_foster is not None}
        for key in ("conduction", "turn_on", "turn_off"):
            records = (component.channel_data if key == "conduction" else
                       component.e_rr_data if component_name == "diode" and key == "turn_off" else
                       [] if component_name == "diode" else
                       getattr(component, "e_on_data" if key == "turn_on" else "e_off_data"))
            groups = {}
            if not isinstance(records, (list, tuple)):
                diagnostics.append({"component": component_name, "kind": key,
                    "record_index": None, "code": "invalid-record-collection",
                    "message": "Core mechanism requires a record sequence"})
                records = []
            record_budget += len(records)
            if record_budget > 100_000:
                raise HTTPException(status_code=413, detail="Core context inventory exceeds 100000 records")
            for index, record in enumerate(records):
                try:
                    context = {"gate_voltage_v": _finite(record.v_g, "Recorded gate voltage", nullable=True)}
                    temperature = _finite(record.t_j, "Recorded junction temperature")
                    additional, supply, dataset_type = None, None, None
                    if key != "conduction":
                        resistance = _finite(record.r_g, "Recorded gate resistance", nullable=True)
                        if resistance is not None and resistance < 0:
                            raise ValueError("Recorded gate resistance must be nonnegative")
                        context["gate_resistance_ohm"] = resistance
                        supply = _finite(record.v_supply, "Recorded supply voltage")
                        if record.commutation_device is not None and not isinstance(record.commutation_device, str):
                            raise ValueError("Recorded commutation device must be text or unknown null")
                        if not isinstance(record.dataset_type, str):
                            raise ValueError("Recorded dataset type must be text")
                        dataset_type = record.dataset_type
                        additional = {"gate_off_voltage_v": _finite(record.v_g_off, "Recorded gate-off voltage", nullable=True),
                            "commutation_device": record.commutation_device,
                            "load_inductance_h": _finite(record.load_inductance, "Recorded load inductance", nullable=True),
                            "commutation_inductance_h": _finite(record.commutation_inductance, "Recorded commutation inductance", nullable=True)}
                    identity = json.dumps([context, additional], sort_keys=True, allow_nan=False)
                    group = groups.setdefault(identity, {**context, "additional_conditions": additional,
                        "curve_count": 0, "temperatures_c": [], "voltages_v": [], "dataset_types": []})
                    group["curve_count"] += 1
                    group["temperatures_c"].append(temperature)
                    if key != "conduction":
                        group["voltages_v"].append(supply)
                        group["dataset_types"].append(dataset_type)
                except (ValueError, TypeError, OverflowError, AttributeError) as exc:
                    diagnostics.append({"component": component_name, "kind": key,
                        "record_index": index, "code": "invalid-record-conditions", "message": str(exc)})
            for group in groups.values():
                group["temperatures_c"] = sorted(set(group["temperatures_c"]))
                group["voltages_v"] = sorted(set(group["voltages_v"])) if key != "conduction" else None
                group["dataset_types"] = sorted(set(group["dataset_types"]))
            result[component_name][key] = list(groups.values())
    return {"transistor_id": transistor_id, "components": result, "diagnostics": diagnostics,
            "selection_policy": "Explicit exact context; no default selection"}
