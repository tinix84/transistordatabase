"""Typed transport for stateless semiconductor training and bounded prediction."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, StrictInt

from transistordatabase.device_model_fitting import (
    evaluate_device_model, train_device_model,
)
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.gui_web.backend.parameter_routes import Strict

Conditions = dict[str, float | None]
Token = Annotated[str, Field(min_length=1, max_length=256)]


class Observation(Strict):
    """One source observation with a deliberate train or holdout assignment."""

    id: Token
    source: Token
    series: Token
    split: Literal["train", "holdout"]
    coordinates: Conditions
    value: float


class TrainingRequest(Strict):
    """One family, fixed polynomial order and explicit energy anchor."""

    family: Literal["ron", "diode", "energy"]
    degree: StrictInt = Field(default=1, ge=1, le=3)
    event: Literal["e_on", "e_off"] | None = None
    fixed_conditions: Conditions = Field(default_factory=dict)
    anchor: dict[str, float] | None = None
    observations: Annotated[list[Observation], Field(min_length=2, max_length=2000)]


class Polynomial(Strict):
    """Scaled ascending polynomial and numerical support evidence."""

    basis: Literal["ascending-powers-of-z"]
    degree: StrictInt
    coefficients: list[float]
    center: float
    scale: float
    support: list[float]
    training_ids: list[str]
    rank: StrictInt
    condition_number: float
    residual_degrees_of_freedom: StrictInt
    normalization: float | None


class ModelSlice(Strict):
    """One exact physical slice and its independent output fits."""

    id: str
    conditions: Conditions
    polynomials: dict[str, Polynomial]


class DeviceModel(Strict):
    """Versioned JSON data; evaluate also reproduces and validates the fit."""

    schema_version: StrictInt = Field(ge=1, le=1)
    method: Literal["scaled-polynomial-device-fit-v1"]
    family: Literal["ron", "diode", "energy"]
    degree: StrictInt
    event: Literal["e_on", "e_off"] | None
    fixed_conditions: Conditions
    anchor: dict[str, float] | None
    training_observations: list[Observation]
    slices: list[ModelSlice]


class ResidualRow(Strict):
    """A scored observation or an explicit unsupported prediction."""

    id: str
    source: str
    series: str
    observed: float
    predicted: float | None
    residual: float | None
    status: Literal["predicted", "rejected"]
    reason: str | None


class Metrics(Strict):
    """Errors calculated from successfully scored observations only."""

    rows: list[ResidualRow]
    count: int
    rmse: float | None
    mae: float | None
    r2: float | None


class Diagnostic(Strict):
    """Separate fitting and held-out diagnostics for each output and slice."""

    output: str
    slice: str
    train: Metrics
    holdout: Metrics | None


class PlotPoint(Strict):
    """One fitted point in the returned plot's explicit axis units."""

    x: float
    y: float


class PlotObservation(PlotPoint):
    """An observed point and its source and split assignment."""

    id: str
    source: str
    series: str
    split: Literal["train", "holdout"]


class FitPlot(Strict):
    """Server-generated fitted points and source observations in the same units."""

    id: str
    output: str
    slice: str
    axis: str
    x_label: str
    y_label: str
    samples: list[PlotPoint]
    observations: list[PlotObservation]
    conditions: Conditions
    normalization: float | None
    support: list[float]


class Evidence(Strict):
    """Canonical identities plus the exact HTTP request body identity."""

    method: str
    input_sha256: str
    model_sha256: str
    result_sha256: str
    dataset_sha256: str | None = None
    request_sha256: str | None = None


class TrainingResponse(Strict):
    """Fitted model, exact export text and separate validation evidence."""

    inputs: TrainingRequest
    model: DeviceModel
    model_json: str
    diagnostics: list[Diagnostic]
    plots: list[FitPlot]
    evidence: Evidence
    scope: str
    units: dict[str, str]


class Query(Strict):
    """One identified query for a declared physical output."""

    id: Token
    output: Literal["resistance", "voltage", "current", "energy"]
    coordinates: Conditions


class EvaluationRequest(Strict):
    """An imported own-format JSON model and bounded query list."""

    model: DeviceModel
    queries: Annotated[list[Query], Field(min_length=1, max_length=2000)]


class Prediction(Strict):
    """An accepted prediction or an explicit rejection reason."""

    id: str
    output: str
    status: Literal["predicted", "rejected"]
    value: float | None
    unit: str
    reason: str | None


class EvaluationResult(Strict):
    """Predictions with the applicable assumptions and scope."""

    predictions: list[Prediction]
    scope: str


class EvaluationResponse(Strict):
    """Validated inputs, prediction results and reproducible identities."""

    inputs: EvaluationRequest
    result: EvaluationResult
    evidence: Evidence


router = APIRouter(prefix="/api/models", tags=["Device model fitting"],
                   route_class=FiniteJsonRoute)


@router.post("/train", response_model=TrainingResponse)
async def train(body: TrainingRequest, request: Request):
    """Fit only the explicitly assigned training observations."""
    try:
        result = train_device_model(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["evidence"]["request_sha256"] = hashlib.sha256(await request.body()).hexdigest()
    return result


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(body: EvaluationRequest, request: Request):
    """Check all imported model state and return explicit bounded predictions."""
    try:
        # FiniteJsonRoute retains integer precision and decimal lexemes. Check
        # model state before typed floats can hide a changed source coordinate,
        # coefficient or support bound. Decimal equality accepts equivalent
        # numeric spellings (25, 25.0, 2.5e1). Strict typed fields have already
        # rejected booleans in every model numeric field.
        original_model = (await request.json())["model"]
        canonical_model = json.loads(body.model.model_dump_json(), parse_float=Decimal)
        if original_model != canonical_model:
            raise ValueError("Submitted model numbers change during numeric conversion")
        result = evaluate_device_model(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["evidence"]["request_sha256"] = hashlib.sha256(await request.body()).hexdigest()
    return result
