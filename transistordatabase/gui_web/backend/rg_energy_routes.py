"""Typed read-only transport for repository gate-resistance energy curves."""
from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.gui_web.backend.parameter_routes import Strict
from transistordatabase.rg_energy_access import (
    calculate_rg_energy, rg_energy_inventory,
)

Text = Annotated[str, Field(min_length=1, max_length=2000)]
Branch = Literal["e_on", "e_off", "e_rr"]
Representation = Literal["graph_r_e", "graph_i_e"]


class RgConditions(Strict):
    """Require all physical conditions, including each explicit unknown."""

    v_supply: float | None
    v_g: float | None
    v_g_off: float | None
    load_inductance: float | None = Field(ge=0)
    commutation_inductance: float | None = Field(ge=0)
    commutation_device: Text | None
    convention: Text | None
    event: Text | None
    method: Text | None


class RgQuery(Strict):
    """Select one branch, representation, current, temperature and context."""

    branch: Branch
    representation: Representation
    i_channel: float = Field(ge=0)
    t_j: float
    conditions: RgConditions


class EnergyAtRg(RgQuery):
    """Estimate energy at a bounded resistance."""

    operation: Literal["energy-at-rg"]
    r_g: float = Field(ge=0)


class RgAtEnergy(RgQuery):
    """Find one bounded resistance for the target energy."""

    operation: Literal["rg-at-energy"]
    e_target: float = Field(ge=0)


RgRequest = Annotated[EnergyAtRg | RgAtEnergy, Field(discriminator="operation")]


class Coverage(Strict):
    """Report measured domains without a promise of unique inverse solutions."""

    axis: Literal["r_g", "i_channel"]
    axis_unit: Literal["ohm", "A"]
    lower: float
    upper: float
    energy_min: float
    energy_max: float
    sample_count: int


class CurvePreview(Strict):
    """Bound source previews independently of the original file size."""

    axis: list[float]
    energy_j: list[float]
    truncated: bool


class DependencySnapshot(Strict):
    """Record normalized math dependencies without embedding original files."""

    representation: Representation
    t_j: float
    i_x: float | None
    r_g: float | None
    conditions: RgConditions
    axis: list[float]
    energy_j: list[float]


class CurveSource(Strict):
    """Identify a repository row and preserve available measurement evidence."""

    source_id: str
    source_kind: Literal["repository-locator", "measurement-evidence"]
    branch: Branch
    index: int
    representation: Representation
    t_j: float | None
    i_x: float | None
    r_g: float | None
    conditions: RgConditions | None
    source_filename: str | None
    source_sha256: str | None
    snapshot_sha256: str | None
    coverage: Coverage | None
    preview: CurvePreview | None
    status: Literal["available", "unavailable"]
    diagnostic: str | None
    unavailable_fields: list[str]


class ConsideredSource(CurveSource):
    """Provide replay dependencies for each considered branch curve."""

    snapshot: DependencySnapshot | None


class RgInventory(Strict):
    """List curves without an automatic first-context selection."""

    device_id: str
    curves: list[CurveSource]


class RgResult(Strict):
    """Return a bounded estimate and its declared source scope."""

    device_id: str
    operation: Literal["energy-at-rg", "rg-at-energy"]
    value: float
    unit: Literal["J", "ohm"]
    evidence_tag: Literal["bounded-interpolation-estimate"]
    query: EnergyAtRg | RgAtEnergy
    conditions: RgConditions
    source_scope: Literal["considered-branch"]
    sources: list[ConsideredSource]
    method: str


def create_rg_energy_router(get_transistor: Callable):
    """Reuse the application's repository lookup and preserve its 404 response."""
    router = APIRouter(tags=["Rg-energy"], route_class=FiniteJsonRoute)

    @router.get("/api/transistors/{transistor_id}/rg-energy",
                response_model=RgInventory)
    async def inventory(transistor_id: str):
        try:
            return rg_energy_inventory(get_transistor(transistor_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/api/transistors/{transistor_id}/rg-energy", response_model=RgResult)
    async def calculate(transistor_id: str, request: RgRequest):
        try:
            return calculate_rg_energy(get_transistor(transistor_id), request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
