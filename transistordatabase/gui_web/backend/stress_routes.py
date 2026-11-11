"""Strict transport for reusable semiconductor stress proxies."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.semiconductor_stress import calculate_stress_indices


class Strict(BaseModel):
    """Reject extra fields, non-finite numbers, and implicit text conversion."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Transition(Strict):
    """One explicit frequency-weighted waveform sampling record."""

    kind: Literal["turn_on", "turn_off"]
    time_s: float
    frequency_hz: float = Field(ge=0)


Vector = Annotated[list[float], Field(min_length=2, max_length=20000)]


class StressRequest(Strict):
    """Complete waveform and explicit known or unknown conditions."""

    source: str = Field(min_length=1, max_length=2000)
    conditions: dict[str, float | None]
    time_s: Vector
    current_a: Vector
    voltage_v: Vector
    rds_on_ohm: float | None = Field(default=None, gt=0)
    transitions: Annotated[list[Transition], Field(max_length=20000)] | None = None


class Metric(Strict):
    """A calculated proxy or an explicitly unknown proxy."""

    value: float | None
    unit: str
    status: Literal["calculated", "unknown"]
    reason: str | None


class Metrics(Strict):
    """The four separate stress proxies."""

    conduction: Metric
    volume: Metric
    switching: Metric
    cost: Metric


class TransitionSample(Transition):
    """Recorded interpolation values and their signed contribution."""

    current_a: float
    voltage_v: float
    contribution: float


class StressResult(Strict):
    """Calculated proxies with explicit definitions and bounds."""

    current_rms_a: float
    current_mean_square_a2: float
    peak_voltage_v: float
    duration_s: float
    sample_count: int
    metrics: Metrics
    transition_samples: list[TransitionSample]
    definitions: dict[str, str]
    scope: str


class Evidence(Strict):
    """Canonical input and result hashes for reproducible evidence."""

    method: Literal["semiconductor-stress-v1"]
    input_sha256: str
    result_sha256: str


class StressResponse(Strict):
    """Typed calculation envelope."""

    inputs: StressRequest
    result: StressResult
    evidence: Evidence




router = APIRouter(prefix="/api/stress", tags=["Semiconductor stress proxies"],
                   route_class=FiniteJsonRoute)


@router.post("/calculate", response_model=StressResponse)
def calculate(request: StressRequest) -> dict:
    """Return auditable inputs, explicit proxy units, and result hashes."""
    try:
        return calculate_stress_indices(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
