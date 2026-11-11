"""Typed HTTP boundary for finite waveform grids and conditioned energy maps."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field, field_validator, model_serializer

from transistordatabase.energy_maps import build_energy_map
from transistordatabase.grid_analysis import (
    AnalysisResourceLimitError, evaluate_grid, thermal_handoff,
)
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.gui_web.backend.parameter_routes import Strict
from transistordatabase.gui_web.backend.waveform_routes import (
    Conduction, EventPolicy, Provenance, Spectrum, Switching, Waveform, WaveformRequest,
)
from transistordatabase.gui_web.backend.thermal_routes import Event, Segment
from transistordatabase.waveform_losses import WaveformResourceLimitError

MAX_INPUT_BYTES = 4_000_000


class AnalysisRoute(FiniteJsonRoute):
    """Bound actual request bytes before JSON parsing and typed validation."""

    max_body_bytes = MAX_INPUT_BYTES


class Identity(Strict):
    """Stable user-selected identity with an inspectable source."""

    id: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=2000)

    @field_validator("id", "source")
    @classmethod
    def nonblank(cls, value):
        """Reject blank identities without changing supplied values."""
        if not value.strip():
            raise ValueError("Use a nonblank identity and source")
        return value


class Configuration(Identity):
    """Immutable model selection, using the accepted waveform schema."""

    conduction: Conduction
    switching: Switching


class OperatingPoint(Identity):
    """Explicit waveform, event policy, origin, and nonnegative weight."""

    weight: float = Field(ge=0)
    waveform: Waveform
    events: EventPolicy
    spectrum: Spectrum | None = None
    provenance: Provenance | None = None


class Metric(Strict):
    """A physical objective, with one unit and a declared direction."""

    name: Literal["total_power_w", "conduction_power_w", "turn_on_power_w", "turn_off_power_w"]
    unit: Literal["W"]
    direction: Literal["minimize", "maximize"]


class MeritObjective(Metric):
    """An explicit weight for one named physical power objective."""

    weight: float = Field(ge=0)


class GridRequest(Strict):
    """A finite configuration-by-point collection without executable expressions."""

    configurations: list[Configuration]
    operating_points: list[OperatingPoint]
    metric: Metric
    summary_mode: Literal["full", "successful-subset"] = "full"
    include_zero_weight_in_worst: bool = False
    grouped_series_exports: bool = False
    objectives: list[MeritObjective] | None = Field(default=None, min_length=1, max_length=4)

    @model_serializer(mode="wrap")
    def canonical_request(self, handler):
        """Preserve legacy snapshots when grouped exports are not selected."""
        data = handler(self)
        if not self.grouped_series_exports:
            data.pop("grouped_series_exports", None)
        if self.objectives is None:
            data.pop("objectives", None)
        return data


class MetricValue(Strict):
    """A missing metric remains unavailable, distinct from zero."""

    value: float | None
    unit: Literal["W"]
    status: Literal["available", "unavailable"]


class PairResult(Strict):
    """One pair retains complete inputs and the accepted waveform result."""

    configuration_id: str
    operating_point_id: str
    weight: float
    inputs: dict[str, Any]
    input_sha256: str
    method_version: str
    status: Literal["computed", "unsupported"]
    result: dict[str, Any] | None
    metrics: dict[str, MetricValue]
    error: str | None


class MetricSummary(Strict):
    """Full and subset values retain their weight coverage and worst-point policy."""

    unit: Literal["W"]
    full_mean: float | None
    subset_mean: float | None
    selected_mean: float | None
    selected_summary: Literal["full", "successful-subset"]
    retained_weight: float
    total_weight: float
    coverage: float
    complete: bool
    worst_value: float | None
    worst_point_ids: list[str]
    worst_complete: bool
    zero_weight_in_worst: bool


class ConfigurationSummary(Strict):
    """One configuration with four independent physical metric summaries."""

    configuration_id: str
    metrics: dict[str, MetricSummary]


class RankedConfiguration(Strict):
    """A selected-summary rank entry, ordered by its declared metric."""

    configuration_id: str
    value: float
    unit: Literal["W"]


class SeriesMetadata(Strict):
    """Preserve physical units and provenance on each exported row."""

    configuration_id: str
    configuration_source: str
    operating_point_id: str
    operating_point_source: str
    weight: float
    series: Literal["current_a", "gate_v", "device_voltage_v", "bus_voltage_v"]
    unit: Literal["A", "V"]
    status: Literal["computed", "unsupported", "not_requested"]
    pair_status: Literal["computed", "unsupported"]
    error: str | None
    input_sha256: str
    method_version: str
    grid_input_sha256: str
    window: Literal["rectangular", "hann"] | None
    detrend: Literal["none", "mean"] | None
    relative_threshold: float | None
    phase_reference: Literal["first supplied sample"] | None


class TimeSeriesRow(SeriesMetadata):
    """Discrete sample statistics remain distinct from interpolated time integrals."""

    sample_count: int | None
    minimum: float | None
    maximum: float | None
    arithmetic_mean: float | None
    sample_standard_deviation: float | None
    discrete_rms: float | None
    time_weighted_mean: float | None
    time_weighted_rms: float | None


class FrequencySeriesRow(SeriesMetadata):
    """A selected complex bin or a null-bin status manifest."""

    bin_index: int | None
    frequency_hz: float | None
    magnitude: float | None
    real: float | None
    imaginary: float | None
    phase_rad: float | None
    retained_bin_count: int | None


class GroupedSeriesExports(Strict):
    """One server projection supplies matching raw JSON rows and ready CSV text."""

    method_version: str
    scope: str
    time_rows: list[TimeSeriesRow]
    frequency_rows: list[FrequencySeriesRow]
    time_json: str
    frequency_json: str
    time_csv: str
    frequency_csv: str
    null_csv: Literal["NULL"]


class MeritRange(MeritObjective):
    """Use the same eligible population for every active objective range."""

    normalized_weight: float
    active: bool
    minimum: float | None
    maximum: float | None
    minimum_configuration_ids: list[str]
    maximum_configuration_ids: list[str]


class MeritEvidence(MeritRange):
    """Keep complete raw means, missing coverage, and score contributions."""

    full_mean: float | None
    complete: bool
    retained_weight: float
    total_weight: float
    coverage: float
    missing_operating_point_ids: list[str]
    penalty: float | None
    contribution: float | None


class MeritExclusion(Strict):
    """Identify every missing positive-weight operating point for an objective."""

    name: str
    missing_operating_point_ids: list[str]


class MeritRow(Strict):
    """An eligible score or an excluded configuration with its raw evidence."""

    configuration_id: str
    status: Literal["computed", "excluded"]
    score: float | None
    rank: int | None
    objectives: list[MeritEvidence]
    exclusion_reasons: list[MeritExclusion]


class MeritPopulation(Strict):
    """Freeze the eligible configuration and positive-weight point identities."""

    configuration_ids: list[str]
    operating_point_ids: list[str]
    zero_weight_operating_point_ids: list[str]
    configuration_count: int
    eligible_configuration_count: int
    configuration_coverage: float


class MeritRank(Strict):
    """A lower score ranks first, with exact ties sharing a competition rank."""

    configuration_id: str
    score: float
    rank: int


class MeritAuditRow(MeritEvidence):
    """One flat export row preserves the complete score and objective provenance."""

    method_version: str
    grid_input_sha256: str
    population_sha256: str
    configuration_id: str
    status: Literal["computed", "excluded"]
    score: float | None
    rank: int | None
    population_configuration_ids: list[str]
    population_operating_point_ids: list[str]
    zero_weight_operating_point_ids: list[str]


class MeritResponse(Strict):
    """Server scores and raw exports share one accepted grid population."""

    method_version: str
    grid_input_sha256: str
    population_sha256: str
    status: Literal["computed", "unavailable"]
    population: MeritPopulation
    objectives: list[MeritRange]
    rows: list[MeritRow]
    ranking: list[MeritRank]
    export_rows: list[MeritAuditRow]
    rows_json: str
    rows_csv: str
    null_csv: Literal["NULL"]


class GridResponse(Strict):
    """Auditable results shared by linked tables, plots, and exports."""

    method_version: str
    metric: Metric
    summary_mode: Literal["full", "successful-subset"]
    rows: list[PairResult]
    summaries: list[ConfigurationSummary]
    ranking: list[RankedConfiguration]
    inputs: GridRequest
    input_sha256: str
    limits: dict[str, int]
    grouped_series_exports: GroupedSeriesExports | None = None
    merit: MeritResponse | None = None


class FixedConditions(Strict):
    """All comparison conditions require known values or explicit nulls."""

    t_j_c: float | None
    v_g_on_v: float | None
    v_g_off_v: float | None
    r_g_on_ohm: float | None = Field(ge=0)
    r_g_off_ohm: float | None = Field(ge=0)
    commutation_device: str | None = Field(max_length=1000)
    commutation_source: str | None = Field(max_length=2000)
    energy_convention: str = Field(min_length=1, max_length=2000)


class EnergySlice(Identity):
    """One unchanged source curve, with independent kind and fixed conditions."""

    source_kind: Literal["measured", "datasheet", "simulated", "manual", "unknown"]
    energy_kind: Literal["on", "off", "rr"]
    voltage_v: float = Field(gt=0)
    current_a: list[float]
    energy_j: list[float]
    conditions: FixedConditions


class EnergyEstimate(Strict):
    """Explicitly selected interpolation within a complete displayed rectangle."""

    selected: bool
    current_a: float = Field(ge=0)
    voltage_v: float = Field(gt=0)


class EnergyMapRequest(Strict):
    """A recorded-slice view and an optional named I/V estimate."""

    energy_kind: Literal["on", "off", "rr"]
    conditions: FixedConditions
    slices: list[EnergySlice]
    currents_a: list[float]
    voltages_v: list[float]
    estimate: EnergyEstimate | None = None


class Candidate(Strict):
    """One source contribution, without claiming that all sources are measured."""

    slice_id: str
    source: str
    source_kind: str
    energy_j: float
    mode: Literal["recorded-point", "bounded-current-linear"]


class UnsupportedSlice(Strict):
    """A selected source that does not cover a displayed cell."""

    slice_id: str
    reason: str


class EnergyCell(Strict):
    """A cell is available only when exactly one source covers its current."""

    current_a: float
    voltage_v: float
    status: Literal["available", "ambiguous", "missing"]
    energy_j: float | None
    mode: Literal["recorded-point", "bounded-current-linear"] | None
    candidates: list[Candidate]
    unsupported_slices: list[UnsupportedSlice]


class EstimateResult(Strict):
    """A bounded estimate or an explicit unsupported query."""

    status: Literal["estimated", "unsupported"]
    energy_j: float | None
    method: Literal["bounded-bilinear-estimate"]
    query: EnergyEstimate
    corners: list[EnergyCell]
    error: str | None


class SliceDiagnostic(Strict):
    """A mismatched source remains visible with its exclusion reason."""

    slice_id: str
    status: Literal["selected", "excluded"]
    reason: str | None


class EnergyMapResponse(Strict):
    """Original slices, conditioned grid, and disclosed interpolation evidence."""

    method_version: str
    energy_kind: Literal["on", "off", "rr"]
    conditions: FixedConditions
    currents_a: list[float]
    voltages_v: list[float]
    units: dict[str, str]
    cells: list[list[EnergyCell]]
    estimate: EstimateResult | None
    source_slices: list[EnergySlice]
    slice_diagnostics: list[SliceDiagnostic]
    input_sha256: str
    assumptions: list[str]


class DirectionOwners(Strict):
    """Current sign does not imply physical junction identity."""

    positive: Literal["forward", "reverse"]
    negative: Literal["forward", "reverse"]
    zero: Literal["forward", "reverse"]


class ThermalHandoffRequest(Strict):
    """Recompute an accepted waveform request before staging exact known heat."""

    waveform_request: WaveformRequest
    direction_owners: DirectionOwners
    network_heat_owner: Literal["forward", "reverse", "shared-junction"]
    event_mode: Literal["impulse"]
    allow_partial: bool = False


class ThermalHeatInput(Strict):
    """Only heat is staged. The physical network and boundary remain required."""

    segments: list[Segment]
    events: list[Event]
    network_heat_owner: Literal["forward", "reverse", "shared-junction"]
    event_mode: Literal["impulse"]
    gap_policy: Literal["reject"]


class SegmentOwner(Strict):
    """Preserve each exact segment's declared physical owner."""

    segment_index: int
    heat_owner: Literal["forward", "reverse"]


class ThermalHandoffResponse(Strict):
    """Frozen electrical evidence and staged heat for the thermal workbench."""

    method_version: str
    source_request: WaveformRequest
    input_sha256: str
    source_input_sha256: str
    direction_owners: DirectionOwners
    segment_owners: list[SegmentOwner]
    omitted_components: list[str]
    omission_reasons: dict[str, str]
    scope: Literal["known-components-only", "complete-waveform-heat"]
    waveform_result: dict[str, Any]
    thermal_input: ThermalHeatInput
    required_inputs: list[str]
    assumptions: list[str]


router = APIRouter(prefix="/api/analysis", tags=["Finite analysis"], route_class=AnalysisRoute)


@router.post("/grid", response_model=GridResponse, response_model_exclude_unset=True)
def grid(request: GridRequest):
    """Evaluate the registered waveform operation, preserving individual failures."""
    try:
        return evaluate_grid(request.model_dump())
    except AnalysisResourceLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/energy-map", response_model=EnergyMapResponse)
def energy_map(request: EnergyMapRequest):
    """Build a conditioned view without inventing missing records."""
    try:
        return build_energy_map(request.model_dump())
    except AnalysisResourceLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/thermal-handoff", response_model=ThermalHandoffResponse, response_model_exclude_unset=True)
def handoff(request: ThermalHandoffRequest):
    """Stage exact electrical heat without fitting or inventing a thermal network."""
    try:
        return thermal_handoff(request.model_dump(exclude_unset=True))
    except (AnalysisResourceLimitError, WaveformResourceLimitError) as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
