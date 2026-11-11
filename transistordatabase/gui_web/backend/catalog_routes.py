"""FastAPI adapter for the reusable catalog FOM operations."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException
from typing import Any, Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator

from transistordatabase.catalog_fom import METRICS, load_catalog, screen_catalog
from transistordatabase.workbook_fom import DESCRIPTIONS, PROFILES
from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict

router = APIRouter(prefix="/api/catalog", tags=["Catalog FOM"], route_class=ParameterRoute)


@lru_cache(maxsize=1)
def catalog() -> dict:
    """Read the immutable bundled snapshot once per process."""
    return load_catalog()


class QuoteCohort(Strict):
    """Declare a comparable recorded currency and quantity tier."""

    currency: str = Field(pattern=r"^[A-Z]{3}$")
    quote_quantity: StrictFloat | StrictInt = Field(gt=0)


class Normalization(Strict):
    """Declare the metric denominator population."""

    mode: Literal["fixed", "visible"]
    metrics: list[str] = Field(min_length=1)
    population_ids: list[str] | None = None


class CatalogSnapshot(Strict):
    """Accept versioned source rows without changing the bundled catalog."""

    schema_version: StrictInt = Field(ge=1, le=1)
    sources: list[dict[str, Any]] = Field(min_length=1, max_length=64)
    records: list[dict[str, Any]] = Field(max_length=10000)


class WorkbookSourceDeclaration(Strict):
    """Declare source units without changing historical cells or source metadata."""

    source_id: str = Field(min_length=1, max_length=500)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    area_unit: Literal["m2", "mm2", "cm2"] | None = None


class WorkbookDerivation(Strict):
    """Acknowledge the fixed source estimate model and declare unknown units."""

    thermal_convention: Literal["source-constant-model-100C"] | None = None
    source_declarations: list[WorkbookSourceDeclaration] = Field(default_factory=list, max_length=64)


class ScreeningRequest(Strict):
    """Selection and objective settings for a reproducible screening run."""

    source_id: str | None = None
    source_ids: list[str] | None = None
    snapshot: CatalogSnapshot | None = None
    quote_cohort: QuoteCohort | None = None
    normalization: Normalization | None = None
    workbook_derivation: WorkbookDerivation | None = None
    panel_preset: Literal["default", "survey"] = "default"
    x_metric: str = "rds_ohm"
    y_metric: str = "cost_fom"
    x_direction: str = "min"
    y_direction: str = "min"
    technology: str = ""
    manufacturer: str = ""
    search: str = ""
    voltage_min: StrictFloat | StrictInt | None = None
    voltage_max: StrictFloat | StrictInt | None = None
    current_min: StrictFloat | StrictInt | None = None

    @model_validator(mode="after")
    def one_source_selection(self):
        """Require one source selection form."""
        if (self.source_id is None) == (self.source_ids is None):
            raise ValueError("Choose source_id or source_ids, exclusively")
        return self


@router.get("/sources")
def sources() -> dict:
    """Describe available worksheets and metrics without returning raw records."""
    data = catalog()
    return {
        "sources": data["sources"],
        "metrics": {
            key: {"label": v[0], "unit": v[1], "direction": v[2],
                  **({"description": DESCRIPTIONS[key]} if key in DESCRIPTIONS else {})}
            for key, v in METRICS.items()
        },
        "workbook_profiles": PROFILES,
        "facets": {
            s["id"]: {
                key: sorted(
                    {r[key] for r in data["records"] if r["source_id"] == s["id"]}
                )
                for key in ("technology", "manufacturer")
            }
            for s in data["sources"]
        },
    }


@router.post("/screen")
def screen(request: ScreeningRequest) -> dict:
    """Return filtered points, metric coverage, provenance and a Pareto frontier."""
    try:
        settings = request.model_dump()
        snapshot = settings.pop("snapshot")
        if settings["normalization"] is not None:
            settings["normalization"] = request.normalization.model_dump(exclude_none=True)
        return screen_catalog(snapshot if snapshot is not None else catalog(), **settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
