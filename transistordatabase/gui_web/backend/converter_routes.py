"""Typed transport for declarative supplied-waveform converter aggregation."""
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field, StrictInt

from transistordatabase.converter_losses import analyze_converter, converter_presets
from transistordatabase.circuit_summary import MAX_INPUT_BYTES, analyze_circuit_summary
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict
from transistordatabase.gui_web.backend.waveform_routes import (
    Conduction, EventPolicy, Provenance, Switching,
)
from transistordatabase.waveform_losses import WaveformResourceLimitError


class SignalBinding(Strict):
    """Named signal times a scalar, optional complement, or explicit constant."""

    signal: str | None = None
    value: float | None = None
    scale: float = 1
    complement: bool = False


class RoleBinding(Strict):
    """Physical positions represented by one independently modeled role."""

    role: str
    members: list[str]
    multiplicity: StrictInt = Field(ge=0, le=10000)
    conduction_current: SignalBinding
    switching_current: SignalBinding | None = None
    gate_signal: str | None = None
    bus_voltage: SignalBinding | None = None
    duty: SignalBinding | None = None
    frequency_hz: SignalBinding | None = None
    source_switching_omission: bool = False
    inactive_reason: str | None = None
    activation: Literal["negative-power"] | None = None


class RoleModel(Strict):
    """Source identity and existing strict directional scientific model shapes."""

    source: str | None = None
    conduction: Conduction | None = None
    switching: Switching | None = None
    events: EventPolicy | None = None


class Assumptions(Strict):
    """User-selected symmetry/envelope/omission scope, never inferred from names."""

    symmetry_selected: bool = False
    envelope_selected: bool = False
    source_omission_reason: str | None = None
    operating_power_w: float | None = None


class ConverterRequest(Strict):
    """Same time interval and named supplied columns for all physical roles."""

    preset: str
    operation: Literal["waveform", "held-envelope-estimate"]
    time_s: Annotated[list[float], Field(min_length=2, max_length=10001)]
    signals: dict[str, float | list[float]]
    role_models: dict[str, RoleModel]
    assumptions: Assumptions
    bindings: Annotated[list[RoleBinding], Field(min_length=1, max_length=32)] | None = None
    provenance: Provenance | None = None


class HeatOwner(Strict):
    """Explicit die and thermal owner. Null records an unknown owner."""

    die: Literal["channel", "diode"] | None
    junction_id: str | None


class DirectionalOwners(Strict):
    """Ownership follows current sign, including finite zero-current events."""

    positive: HeatOwner
    negative: HeatOwner
    zero: HeatOwner


class LossOwners(Strict):
    """Independent conduction and switching heat ownership."""

    conduction: DirectionalOwners
    turn_on: DirectionalOwners
    turn_off: DirectionalOwners


class CircuitJunction(Strict):
    """One explicit junction path to the ideal fixed sink."""

    id: str = Field(min_length=1, max_length=2000)
    rtheta_k_w: float | None = Field(ge=0)
    source: str | None = Field(max_length=2000)
    conditions: dict[str, str | float | None]


class CircuitMember(Strict):
    """One converter physical member and its complete or partial inventory."""

    id: str = Field(min_length=1, max_length=2000)
    role: str = Field(min_length=1, max_length=2000)
    layout_complete: bool
    loss_owners: LossOwners
    junctions: Annotated[list[CircuitJunction], Field(max_length=128)]


class CircuitBoundary(Strict):
    """A sourced ideal boundary without inferred sink heating."""

    kind: Literal["ideal-fixed-sink"]
    temperature_c: float
    source: str = Field(min_length=1, max_length=2000)
    conditions: dict[str, str | float | None]


class CircuitProvenance(Provenance):
    """Retain a caller-supplied source digest without claiming a manual source hash."""

    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")


class CircuitSummaryRequest(Strict):
    """Reuse the accepted converter request and declare physical heat owners."""

    configuration_id: str = Field(min_length=1, max_length=2000)
    scenario_id: str = Field(min_length=1, max_length=2000)
    converter: ConverterRequest
    boundary: CircuitBoundary
    members: Annotated[list[CircuitMember], Field(max_length=128)]
    acknowledge_known_only: bool = False
    provenance: CircuitProvenance | None = None


class CircuitMaximum(Strict):
    """Separate complete temperatures from acknowledged known-heat estimates."""

    tj_c: float | None
    junction_ids: list[str]
    known_heat_tj_c: float | None
    known_heat_junction_ids: list[str]
    status: Literal["complete", "incomplete"]


class CircuitElectrical(Strict):
    """Authoritative electrical totals and explicit partial-known subtotals."""

    known_component_energy_j: float | None
    known_component_power_w: float | None
    total_energy_j: float | None
    total_power_w: float | None
    total_status: Literal["computed", "partial-explicit-omissions"]
    components: dict[str, dict[str, object]] | None = None


class CircuitJunctionResult(CircuitJunction):
    """Independent heat coverage and thermal-path coverage for one junction."""

    heat_w: float | None
    known_heat_w: float | None
    tj_c: float | None
    known_heat_tj_c: float | None
    status: Literal["complete", "incomplete"]
    omissions: list[str]


class CircuitMemberResult(Strict):
    """Per-position electrical and physical heat results without role evidence copies."""

    id: str
    role: str
    layout_complete: bool
    loss_owners: LossOwners
    components: dict[str, dict[str, object]]
    electrical: CircuitElectrical
    die_losses: dict[str, dict[str, object]]
    junctions: list[CircuitJunctionResult]
    omissions: list[str]
    unallocated_junction_heat_w: float | None
    maximum: CircuitMaximum


class CircuitInputIdentity(Strict):
    """Canonical request identity and a nullable caller-declared source digest."""

    canonical_request_sha256: str
    converter_request_sha256: str
    upstream_source_sha256: str | None


class CircuitCoverage(Strict):
    """Explicit inventory counts and heat without a declared junction owner."""

    member_count: int
    layout_complete_members: int
    unallocated_known_heat_w: float | None
    omissions: list[str]


class CircuitSummaryResponse(Strict):
    """Typed circuit audit with exact server exports of the core payload."""

    method: Literal["explicit-physical-ownership-fixed-sink-v1"]
    input_sha256: str
    input_identity: CircuitInputIdentity
    configuration_id: str
    scenario_id: str
    duration_s: float
    boundary: CircuitBoundary
    members: list[CircuitMemberResult]
    electrical: CircuitElectrical
    maximum: CircuitMaximum
    evidence_by_role: dict[str, dict[str, object]]
    coverage: CircuitCoverage
    acknowledge_known_only: bool
    provenance: dict[str, object] | None
    request: dict[str, object]
    iteration_status: Literal["not-required-constant-heat"]
    null_csv: Literal["NULL"]
    csv_metadata_scope: str
    interpretation: str
    export_json: str
    export_csv: str


router = APIRouter(prefix="/api/converters", tags=["Converter losses"], route_class=ParameterRoute)


@router.get("/presets")
def presets():
    """Inspectable signs, sources and physical multiplicities."""
    return converter_presets()


def analyze(request: ConverterRequest):
    """Return SI evidence; reject missing data, unsupported scope and resource excess."""
    try:
        return analyze_converter(request.model_dump(exclude_none=True))
    except WaveformResourceLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def circuit_summary(request: CircuitSummaryRequest):
    """Return physical heat ownership and independent fixed-sink temperatures."""
    try:
        # Ownership and source-condition nulls are scientific evidence.
        payload = request.model_dump()
        payload["converter"] = request.converter.model_dump(exclude_none=True)
        return analyze_circuit_summary(payload)
    except WaveformResourceLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class CircuitSummaryRoute(FiniteJsonRoute):
    """Keep the circuit request within its decimal four-megabyte budget."""

    max_body_bytes = MAX_INPUT_BYTES


router.add_api_route("/analyze", analyze, methods=["POST"],
                     route_class_override=CircuitSummaryRoute)
router.add_api_route("/circuit-summary", circuit_summary, methods=["POST"],
                     route_class_override=CircuitSummaryRoute,
                     response_model=CircuitSummaryResponse,
                     response_model_exclude_unset=True)
