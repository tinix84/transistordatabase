"""Typed transport for sample-only gate-state current allocation."""
from typing import Annotated, Literal
from decimal import Decimal

from fastapi import APIRouter, HTTPException
from pydantic import Field, StrictBool, StrictFloat, StrictInt, field_validator, model_serializer

from transistordatabase.current_allocation import allocate_gate_current
from transistordatabase.gui_web.backend.gate_policy import HysteresisPolicy, HysteresisEvidence
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.gui_web.backend.parameter_routes import Strict
from transistordatabase.gui_web.backend.waveform_routes import Provenance

Vector = Annotated[list[float], Field(min_length=2, max_length=10000)]


class BinaryGate(Strict):
    """One actual boolean or numeric 0/1 per current sample."""

    mode: Literal["binary"]
    state: Annotated[list[StrictBool | StrictInt | StrictFloat], Field(min_length=2, max_length=10000)]

    @field_validator("state", mode="before")
    @classmethod
    def exact_binary_literals(cls, values):
        """Check exact decimal membership before float conversion can round near-one values."""
        if isinstance(values, list) and any(
                not isinstance(value, (bool, int, float, Decimal)) or value not in (0, 1)
                for value in values):
            raise ValueError("Binary gate state accepts only actual booleans or numeric 0/1")
        return values


class ThresholdGate(Strict):
    """Declare voltage threshold with equality assigned to the low state."""

    mode: Literal["threshold"]
    gate_v: Vector
    threshold_v: float


class HysteresisGate(Strict):
    """Original gate samples with explicitly selected hysteresis."""

    mode: Literal["hysteresis"]
    gate_v: Vector
    digitizer: HysteresisPolicy


class AllocationRequest(Strict):
    """Reproducible samples, source conditions and explicit assumption selection."""

    time_s: Vector
    current_a: Vector
    gate: Annotated[BinaryGate | ThresholdGate | HysteresisGate, Field(discriminator="mode")]
    selected: bool
    source: str = Field(min_length=1, max_length=2000)
    conditions: dict[str, float | None]
    provenance: Provenance | None = None


class Sample(Strict):
    """Complementary signed allocation at one original sample time."""

    time_s: float
    current_a: float
    gate_on: bool
    gate_on_current_a: float
    gate_off_current_a: float


class AllocationResult(Strict):
    """Sample evidence without an interval integration claim."""

    samples: list[Sample]
    sample_count: int
    semantics: str
    scope: str
    units: dict[str, str]
    gate_digitization: HysteresisEvidence | None = None

    @model_serializer(mode="wrap")
    def preserve_existing_result(self, handler):
        """Keep legacy response bodies and result hashes unchanged."""
        data = handler(self)
        if self.gate_digitization is None:
            data.pop("gate_digitization", None)
        return data


class Evidence(Strict):
    """Canonical input and result identities."""

    method: Literal["gate-state-allocation-v1"]
    input_sha256: str
    result_sha256: str


class AllocationResponse(Strict):
    """Typed allocation envelope."""

    inputs: AllocationRequest
    result: AllocationResult
    evidence: Evidence


router = APIRouter(prefix="/api/waveform", tags=["Signed waveforms"], route_class=FiniteJsonRoute)


@router.post("/gate-allocation", response_model=AllocationResponse)
def allocate(request: AllocationRequest):
    """Allocate original samples or reject unsupported inputs."""
    try:
        return allocate_gate_current(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
