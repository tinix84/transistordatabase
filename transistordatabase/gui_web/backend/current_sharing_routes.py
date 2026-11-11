"""Typed HTTP transport for bounded reverse current sharing."""
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

from transistordatabase.reverse_current_sharing import solve_reverse_current_sharing
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.gui_web.backend.parameter_routes import Strict

Vector = Annotated[list[float], Field(min_length=2, max_length=10000)]


class Branch(Strict):
    """Independent branch magnitudes and their source conditions."""

    kind: Literal["channel-only", "diode-only"]
    current_a: Vector
    voltage_v: Vector
    source: str = Field(min_length=1, max_length=2000)
    isolation_evidence: str = Field(min_length=1, max_length=2000)
    recorded_conditions: dict[str, float | None]
    requested_conditions: dict[str, float | None]


class SharingRequest(Strict):
    """Explicit reverse operating point and compatible branch selection."""

    selected: bool
    compatible_operating_state: bool
    mode: Literal["strict-recorded", "ideal-diode-blocking"] = "strict-recorded"
    current_a: float
    junction_temperature_c: float
    channel: Branch
    diode: Branch


class Bounds(Strict):
    """Feasible channel current interval after conservation constraints."""

    lower_a: float
    upper_a: float


class RootEvidence(Strict):
    """Exact piecewise root classification and returned-point voltage residual."""

    isolated_root_count: int
    flat_intervals_a: list[list[float]]
    candidate_channel_current_magnitude_a: list[float]
    evaluated_knot_count: int
    voltage_residual_v: float | None


class SharingResult(Strict):
    """Supported static point or explicit null unsupported quantities."""

    status: Literal["supported", "unsupported", "ambiguous"]
    support: Literal["recorded", "estimate", "none"]
    diagnostic: str
    channel_current_a: float | None
    diode_current_a: float | None
    current_sum_a: float | None
    current_residual_a: float | None
    voltage_v: float | None
    channel_static_power_w: float | None
    diode_static_power_w: float | None
    total_static_power_w: float | None
    blocking_margin_v: float | None
    feasible_channel_current_magnitude_a: Bounds | None
    root_evidence: RootEvidence
    equations: list[str]
    scope: str
    units: dict[str, str]


class SharingEvidence(Strict):
    """Canonical input and result identities."""

    method: Literal["bounded-reverse-sharing-v1"]
    input_sha256: str
    result_sha256: str


class SharingResponse(Strict):
    """Reproducible sharing result envelope."""

    inputs: SharingRequest
    result: SharingResult
    evidence: SharingEvidence


router = APIRouter(prefix="/api/assemblies", tags=["Semiconductor assemblies"],
                   route_class=FiniteJsonRoute)


@router.post("/reverse-sharing", response_model=SharingResponse)
def reverse_sharing(request: SharingRequest):
    """Solve within recorded domains without clamp or extrapolation."""
    try:
        return solve_reverse_current_sharing(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
