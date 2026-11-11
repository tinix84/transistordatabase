"""Strict transport for the pure semiconductor assembly transformations."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict
from transistordatabase.semiconductor_assemblies import analyze_assembly

Vector = Annotated[list[float], Field(min_length=2, max_length=10000)]


class Table(Strict):
    """Bounded magnitude-current table with exact source conditions."""

    kind: Literal["channel", "diode", "energy"]
    x: Vector
    y: Vector
    conditions: dict[str, float | None]
    source: str = Field(min_length=1, max_length=2000)


class Foster(Strict):
    """Independent internal poles; no shared external sink transformation."""

    resistance_k_per_w: list[float] = Field(min_length=1, max_length=100)
    tau_s: list[float] = Field(min_length=1, max_length=100)
    capacity_j_per_k: list[float] | None = Field(default=None, min_length=1, max_length=100)


class Device(Strict):
    """Small transformation input, deliberately separate from a device hierarchy."""

    source: str = Field(min_length=1, max_length=2000)
    conduction: Table | None = None
    turn_on: Table | None = None
    turn_off: Table | None = None
    blocking_voltage_v: float | None = None
    current_rating_a: float | None = None
    area_m2: float | None = None
    cost: float | None = None
    currency: str | None = None
    weight_kg: float | None = None
    internal_resistance_k_per_w: float | None = None
    shared_sink_resistance_k_per_w: float | None = None
    foster: Foster | None = None


class Parallel(Strict):
    """Equal sharing must be explicitly selected in the submitted operation."""

    operation: Literal["parallel-bank"]
    device: Device
    count: int = Field(ge=1)


class VoltageEstimate(Strict):
    """Named linear-voltage estimate; other conditions remain exact."""

    selected: bool
    target_voltage_v: float = Field(gt=0)


class Composite(Strict):
    """Separate positive IGBT and negative diode branches, including E(0)."""

    operation: Literal["signed-composite"]
    positive: Device
    negative: Device
    common_resistance_k_per_w: float = Field(ge=0)
    voltage_estimate: VoltageEstimate | None = None


class Envelope(Strict):
    """Explicitly selected ideal reverse-on lower-drop envelope."""

    operation: Literal["reverse-envelope"]
    selected: bool
    channel: Table
    diode: Table


Request = Annotated[Parallel | Composite | Envelope, Field(discriminator="operation")]
router = APIRouter(prefix="/api/assemblies", tags=["Semiconductor assemblies"],
                   route_class=ParameterRoute)


@router.post("/analyze")
def analyze(request: Request):
    """Return canonical SI results and sources without mutating original inputs."""
    try:
        return analyze_assembly(request.model_dump())
    except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
