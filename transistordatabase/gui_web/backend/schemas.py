"""
Pydantic schemas for API request/response validation.

Defines data models for curve management and other API operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator


class ChannelCurveCreate(BaseModel):
    """Schema for creating a channel characteristic curve."""

    t_j: float = Field(..., description="Junction temperature in deg C")
    v_g: Optional[float] = Field(None, description="Gate voltage in V (mandatory for switch)")
    v_data: List[float] = Field(..., description="Voltage data points in V", min_length=2)
    i_data: List[float] = Field(..., description="Current data points in A", min_length=2)

    @field_validator('v_data', 'i_data')
    @classmethod
    def check_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure voltage and current arrays have same length."""
        if info.field_name == 'i_data':
            # This runs after v_data, so we can check lengths
            values_dict = info.data
            if 'v_data' in values_dict and len(v) != len(values_dict['v_data']):
                raise ValueError(
                    f"v_data and i_data must have same length "
                    f"(got {len(values_dict['v_data'])} and {len(v)})"
                )
        return v

    @field_validator('t_j')
    @classmethod
    def check_temperature_range(cls, v: float) -> float:
        """Validate temperature is reasonable."""
        if not -273 <= v <= 300:
            raise ValueError(f"Temperature {v}°C is outside reasonable range [-273, 300]")
        return v


class SwitchingLossCreate(BaseModel):
    """Schema for creating switching loss data (e_on, e_off, e_rr)."""

    dataset_type: str = Field(
        ...,
        description="Type of dataset: 'single', 'graph_i_e', 'graph_r_e', 'graph_t_e'"
    )
    t_j: float = Field(..., description="Junction temperature in deg C")
    v_supply: float = Field(..., description="Supply voltage in V")
    v_g: float = Field(..., description="Gate voltage in V")
    v_g_off: Optional[float] = Field(None, description="Gate voltage for turn-off in V")
    r_g: Optional[float] = Field(None, description="Gate resistance in Ohm")
    i_x: Optional[float] = Field(None, description="Current point for single measurement in A")
    e_x: Optional[float] = Field(None, description="Energy for single measurement in J")

    # Graph data (as lists for JSON serialization)
    graph_i_e: Optional[List[List[float]]] = Field(
        None,
        description="Current vs energy graph [[i1,i2,...], [e1,e2,...]]"
    )
    graph_r_e: Optional[List[List[float]]] = Field(
        None,
        description="Resistance vs energy graph [[r1,r2,...], [e1,e2,...]]"
    )
    graph_t_e: Optional[List[List[float]]] = Field(
        None,
        description="Temperature vs energy graph [[t1,t2,...], [e1,e2,...]]"
    )

    # Measurement metadata
    comment: Optional[str] = None
    measurement_date: Optional[datetime] = None
    measurement_testbench: Optional[str] = None
    commutation_device: Optional[str] = None
    load_inductance: Optional[float] = Field(None, description="Load inductance in H")
    commutation_inductance: Optional[float] = Field(None, description="Commutation inductance in H")

    @field_validator('dataset_type')
    @classmethod
    def check_dataset_type(cls, v: str) -> str:
        """Validate dataset type."""
        valid_types = {'single', 'graph_i_e', 'graph_r_e', 'graph_t_e'}
        if v not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}, got '{v}'")
        return v


class GateChargeCurveCreate(BaseModel):
    """Schema for creating a gate charge curve."""

    v_supply: float = Field(..., description="Drain-source or collector-emitter voltage in V")
    t_j: float = Field(..., description="Junction temperature in deg C")
    i_channel: float = Field(..., description="Channel current during measurement in A")
    i_g: Optional[float] = Field(None, description="Gate current in A")
    q_data: List[float] = Field(..., description="Charge data points in C", min_length=2)
    v_data: List[float] = Field(..., description="Voltage data points in V", min_length=2)

    @field_validator('q_data')
    @classmethod
    def check_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure charge and voltage arrays have same length."""
        values_dict = info.data
        if 'v_data' in values_dict and len(v) != len(values_dict['v_data']):
            raise ValueError(
                f"q_data and v_data must have same length "
                f"(got {len(v)} and {len(values_dict['v_data'])})"
            )
        return v


class SOACreate(BaseModel):
    """Schema for creating Safe Operating Area data."""

    t_c: Optional[float] = Field(None, description="Case temperature in deg C")
    time_pulse: Optional[float] = Field(None, description="Pulse duration in seconds")
    v_data: List[float] = Field(..., description="Voltage data points in V", min_length=2)
    i_data: List[float] = Field(..., description="Current data points in A", min_length=2)

    @field_validator('i_data')
    @classmethod
    def check_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure voltage and current arrays have same length."""
        values_dict = info.data
        if 'v_data' in values_dict and len(v) != len(values_dict['v_data']):
            raise ValueError(
                f"v_data and i_data must have same length "
                f"(got {len(values_dict['v_data'])} and {len(v)})"
            )
        return v


class CapacitanceCurveCreate(BaseModel):
    """Schema for creating voltage-dependent capacitance curves (C_oss, C_iss, C_rss)."""

    t_j: float = Field(..., description="Junction temperature in deg C")
    v_data: List[float] = Field(..., description="Voltage data points in V", min_length=2)
    c_data: List[float] = Field(..., description="Capacitance data points in F", min_length=2)

    @field_validator('c_data')
    @classmethod
    def check_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure voltage and capacitance arrays have same length."""
        values_dict = info.data
        if 'v_data' in values_dict and len(v) != len(values_dict['v_data']):
            raise ValueError(
                f"v_data and c_data must have same length "
                f"(got {len(values_dict['v_data'])} and {len(v)})"
            )
        return v


class CurveUpdateRequest(BaseModel):
    """Optimistic concurrency preconditions and one raw curve patch."""

    expected_count: int = Field(..., ge=1)
    expected_token: str = Field(..., min_length=64, max_length=64, pattern="^[a-f0-9]{64}$")
    curve: dict[str, Any]


class CurveValidationResult(BaseModel):
    """Response schema for curve validation."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None


class ExportOptions(BaseModel):
    """Base schema for export options."""

    include_plots: bool = Field(False, description="Include plot images in export")
    comments: Optional[str] = Field(None, description="Additional comments for export")


class PLECSExportOptions(ExportOptions):
    """Export options for PLECS format."""

    recheck: bool = Field(True, description="Recheck data before export")
    gate_voltages: Optional[List[float]] = Field(
        None,
        description="Gate voltages to include in export"
    )
    include_thermal: bool = Field(True, description="Include thermal model")
    include_losses: bool = Field(True, description="Include loss data")


class MATLABExportOptions(ExportOptions):
    """Export options for MATLAB format."""

    variable_name: str = Field("transistor_data", description="MATLAB variable name")
    struct_array: bool = Field(True, description="Export as struct array")


class BatchExportRequest(BaseModel):
    """Request schema for batch export operations."""

    transistor_ids: List[str] = Field(..., min_length=1, description="List of transistor IDs to export")
    format: str = Field(..., description="Export format (json, csv, plecs, etc.)")
    options: Optional[Dict[str, Any]] = Field(None, description="Format-specific options")


class TopologySweepRequest(BaseModel):
    """Line or contour sweep generated by the Python calculation layer."""

    x_axis: Literal["v_in", "v_out", "p_out", "f_sw", "l_value", "c_out"]
    x_min: float = Field(..., gt=0, allow_inf_nan=False)
    x_max: float = Field(..., gt=0, allow_inf_nan=False)
    x_points: int = Field(50, ge=2, le=200)
    y_axis: Literal["v_in", "v_out", "p_out", "f_sw", "l_value", "c_out"] | None = None
    y_min: float | None = Field(None, gt=0, allow_inf_nan=False)
    y_max: float | None = Field(None, gt=0, allow_inf_nan=False)
    y_points: int | None = Field(None, ge=2, le=200)
    metric: Literal[
        "duty_cycle", "i_in_avg_a", "i_out_avg_a", "i_l_peak_a",
        "i_l_ripple_pp_a", "v_out_ripple_pp_v", "p_loss_total_w",
        "efficiency_percent", "t_switch_c", "t_diode_c",
    ] = "i_l_ripple_pp_a"

    @model_validator(mode="after")
    def validate_axis_ranges(
        self: TopologySweepRequest,
    ) -> TopologySweepRequest:
        """Require complete, non-duplicated axes for contour sweeps."""
        if self.x_axis == self.y_axis:
            raise ValueError("x_axis and y_axis must differ")
        if self.y_axis is None:
            if self.y_min is not None or self.y_max is not None or self.y_points is not None:
                raise ValueError("y sweep range requires y_axis")
        elif self.y_min is None or self.y_max is None or self.y_points is None:
            raise ValueError("y_axis requires y_min, y_max, and y_points")
        return self


class TopologyCalculationRequest(BaseModel):
    """SI-unit request for one headless topology operating point and optional sweep."""

    topology: str = Field(..., description="buck, boost, buck-boost, or flyback; other topologies are unsupported")
    mode: Literal["estimate", "conduction-only", "detailed", "loss-balanced"] = Field(
        ...,
        description=(
            "Select an ideal estimate, channel conduction only, channel conduction plus "
            "switch turn-on/turn-off losses, or channel-loss-balanced CCM/DCM "
            "operating points and bounded line/contour sweeps for Buck, Boost, "
            "Buck-Boost, and Flyback"
        ),
    )
    conduction_regime: Literal["CCM", "DCM"] = Field(
        "CCM", description="Explicitly select continuous or discontinuous conduction"
    )
    turns_ratio_nps: float | None = Field(
        None, description="Flyback primary-to-secondary turns ratio Np/Ns", gt=0,
        allow_inf_nan=False,
    )
    v_in: float = Field(..., description="Input voltage in V", gt=0, allow_inf_nan=False)
    v_out: float = Field(..., description="Output voltage magnitude in V", gt=0, allow_inf_nan=False)
    p_out: float = Field(..., description="Output power in W", gt=0, allow_inf_nan=False)
    f_sw: float = Field(..., description="Switching frequency in Hz", gt=0, allow_inf_nan=False)
    l_value: float = Field(..., description="Inductance in H", gt=0, allow_inf_nan=False)
    c_out: float = Field(..., description="Output capacitance in F", gt=0, allow_inf_nan=False)
    switch_transistor_id: str | None = Field(None, description="TDB device used for the switching leg")
    diode_transistor_id: str | None = Field(None, description="TDB device used for the diode leg")
    transistor_id: str | None = Field(None, description="Compatibility alias used for both device legs")
    parallel_switches: int = Field(1, ge=1, le=100)
    parallel_diodes: int = Field(1, ge=1, le=100)
    t_j: float | None = Field(None, description="Exact junction temperature in deg C")
    t_case: float | None = Field(None, description="Fixed case/sink reference temperature in deg C")
    v_g: float | None = Field(None, description="Exact switch gate voltage in V")
    diode_v_g: float | None = Field(
        None,
        description="Exact diode channel curve gate voltage in V when the TDB curve is gate-conditioned",
        allow_inf_nan=False,
    )
    v_g_off: float | None = Field(None, description="Exact switch turn-off gate voltage in V")
    r_g_on: float | None = Field(None, description="Switch turn-on gate resistance in Ohm")
    r_g_off: float | None = Field(None, description="Switch turn-off gate resistance in Ohm")
    sweep: TopologySweepRequest | None = None


class BoostParetoBoundaryRequest(BaseModel):
    """One shared SI-unit Boost CCM boundary for a semiconductor screen."""

    v_in: float = Field(..., gt=0, allow_inf_nan=False)
    v_out: float = Field(..., gt=0, allow_inf_nan=False)
    p_out: float = Field(..., gt=0, allow_inf_nan=False)
    f_sw: float = Field(..., gt=0, allow_inf_nan=False)
    l_value: float = Field(..., gt=0, allow_inf_nan=False)
    c_out: float = Field(..., gt=0, allow_inf_nan=False)
    t_j: float = Field(..., allow_inf_nan=False)
    t_case: float | None = Field(None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_boost_boundary(self) -> BoostParetoBoundaryRequest:
        """Require the voltage range supported by a Boost operating point.

        :returns: This boundary after successful validation.
        :rtype: BoostParetoBoundaryRequest
        :param self: Boundary request instance to validate.
        :type self: BoostParetoBoundaryRequest
        :raises ValueError: If output voltage does not exceed input voltage.
        """
        if self.v_out <= self.v_in:
            raise ValueError("Boost Pareto screening requires v_out > v_in.")
        return self


class ParetoCandidateRequest(BaseModel):
    """One physical switch/rectifier configuration for Pareto evaluation."""

    candidate_id: str = Field(..., min_length=1, max_length=64)
    label: str = Field(..., min_length=1, max_length=120)
    switch_transistor_id: str | None = None
    diode_transistor_id: str | None = None
    parallel_switches: int = Field(1, ge=1, le=100)
    parallel_diodes: int = Field(1, ge=1, le=100)
    v_g: float | None = Field(None, allow_inf_nan=False)
    diode_v_g: float | None = Field(None, allow_inf_nan=False)

    @field_validator("candidate_id", "label")
    @classmethod
    def strip_candidate_text(cls, value: str) -> str:
        """Reject blank candidate text and normalize surrounding whitespace.

        :param value: Candidate ID or display label supplied by the caller.
        :type value: str
        :returns: Trimmed nonblank value.
        :rtype: str
        :raises ValueError: If the value contains only whitespace.
        """
        value = value.strip()
        if not value:
            raise ValueError("Candidate identifiers and labels cannot be blank.")
        return value


class BoostSemiconductorParetoRequest(BaseModel):
    """Fixed-boundary Boost CCM semiconductor Pareto request."""

    boundary: BoostParetoBoundaryRequest
    candidates: list[ParetoCandidateRequest] = Field(..., min_length=2, max_length=20)
    catalog: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw sourced package and price metadata keyed by TDB identity",
    )

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> BoostSemiconductorParetoRequest:
        """Keep candidate identifiers unique after whitespace normalization.

        :returns: This request after successful validation.
        :rtype: BoostSemiconductorParetoRequest
        :param self: Pareto request instance to validate.
        :type self: BoostSemiconductorParetoRequest
        :raises ValueError: If any normalized candidate IDs are duplicated.
        """
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique.")
        return self


class AdvancedSearchRequest(BaseModel):
    """Request schema for advanced transistor search."""

    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filter criteria (type, v_abs_max range, etc.)"
    )
    sort_field: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field("asc", description="Sort order: 'asc' or 'desc'")
    page: int = Field(1, description="Page number", ge=1)
    per_page: int = Field(20, description="Items per page", ge=1, le=100)


class UserSettings(BaseModel):
    """User preferences and settings."""

    theme: str = Field("light", description="UI theme: 'light', 'dark', 'auto'")
    default_filters: Dict[str, Any] = Field(default_factory=dict)
    favorite_transistors: List[str] = Field(default_factory=list)
    recent_transistors: List[str] = Field(default_factory=list, max_length=20)
    plot_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Plot styling preferences"
    )
    auto_save: bool = Field(True, description="Enable auto-save for forms")


# ==================== Phase 6: Archive Integration Schemas ====================


class BielaModelRequest(BaseModel):
    """Request schema for Biela analytical model calculations."""

    c_oss: float = Field(..., description="Output capacitance in F", gt=0)
    q_g: float = Field(..., description="Total gate charge in C", gt=0)
    v_plateau: float = Field(..., description="Gate plateau voltage in V", gt=0)
    r_g_on: float = Field(..., description="Gate resistance turn-on in Ohm", gt=0)
    r_g_off: float = Field(..., description="Gate resistance turn-off in Ohm", gt=0)
    v_driver_on: float = Field(15.0, description="Gate driver voltage turn-on in V")
    v_driver_off: float = Field(-5.0, description="Gate driver voltage turn-off in V")

    # Operating conditions
    v_dc: float = Field(..., description="DC bus voltage in V", gt=0)
    i_load: float = Field(..., description="Load current in A", gt=0)
    t_j: float = Field(25.0, description="Junction temperature in deg C")

    # Optional: Calculate curve over current range
    i_min: Optional[float] = Field(None, description="Minimum current for curve in A", ge=0)
    i_max: Optional[float] = Field(None, description="Maximum current for curve in A", gt=0)
    i_points: int = Field(50, description="Number of points in curve", ge=2, le=1000)


class GateChargeModelRequest(BaseModel):
    """Request schema for gate charge switching time calculations."""

    q_gs: float = Field(..., description="Gate-source charge in C", gt=0)
    q_gd: float = Field(..., description="Gate-drain (Miller) charge in C", gt=0)
    q_g: float = Field(..., description="Total gate charge in C", gt=0)
    v_th: float = Field(..., description="Threshold voltage in V", gt=0)
    v_plateau: float = Field(..., description="Plateau voltage in V", gt=0)
    r_g: float = Field(..., description="External gate resistance in Ohm", ge=0)
    r_g_int: float = Field(0.0, description="Internal gate resistance in Ohm", ge=0)
    v_driver: float = Field(15.0, description="Gate driver voltage in V", gt=0)
    v_off: float = Field(0.0, description="Gate driver off-state voltage in V")


class DPTConfigRequest(BaseModel):
    """Request schema for LTspice DPT configuration."""

    v_dc: float = Field(400.0, description="DC bus voltage in V", gt=0)
    i_target: float = Field(20.0, description="Target current at turn-off in A", gt=0)
    l_load: float = Field(100e-6, description="Load inductance in H", gt=0)
    r_g_on: float = Field(10.0, description="Gate resistance turn-on in Ohm", ge=0)
    r_g_off: float = Field(10.0, description="Gate resistance turn-off in Ohm", ge=0)
    v_g_on: float = Field(15.0, description="Gate drive voltage turn-on in V")
    v_g_off: float = Field(-5.0, description="Gate drive voltage turn-off in V")
    t_dead: float = Field(5e-6, description="Dead time between pulses in s", gt=0)
    t_pulse2: float = Field(2e-6, description="Duration of second pulse in s", gt=0)


class DptRawEventWindow(BaseModel):
    """Explicit inclusive time window in seconds for one DPT switching event."""

    start_s: float = Field(..., allow_inf_nan=False)
    end_s: float = Field(..., allow_inf_nan=False)

    @model_validator(mode="after")
    def check_order(self) -> "DptRawEventWindow":
        """Reject reversed event boundaries before reading or writing records."""
        if self.end_s < self.start_s:
            raise ValueError("event window end_s must be at or after start_s")
        return self


class DptRawImportMetadata(BaseModel):
    """Optional test conditions stored in an existing raw DPT record."""

    t_j: float | None = Field(None, allow_inf_nan=False)
    v_supply: float | None = Field(None, allow_inf_nan=False)
    v_g: float | None = Field(None, allow_inf_nan=False)
    v_g_off: float | None = Field(None, allow_inf_nan=False)
    r_g: float | None = Field(None, allow_inf_nan=False)
    r_g_off: float | None = Field(None, allow_inf_nan=False)
    load_inductance: float | None = Field(None, allow_inf_nan=False)
    commutation_inductance: float | None = Field(None, allow_inf_nan=False)
    measurement_date: datetime | None = None
    measurement_testbench: str | None = Field(None, max_length=256)
    commutation_device: str | None = Field(None, max_length=256)
    comment: str | None = Field(None, max_length=2000)


class DptRawTraceSelectionRequest(BaseModel):
    """Explicit time, Vds, and Id trace choices from one RAW result."""

    time_trace: str = Field(..., min_length=1, max_length=256)
    vds_trace: str = Field(..., min_length=1, max_length=256)
    id_trace: str = Field(..., min_length=1, max_length=256)
    plot_index: StrictInt | None = Field(None, ge=0)
    step_indices: List[StrictInt] = Field(default_factory=list, max_length=50)
    turn_on: DptRawEventWindow | None = None
    turn_off: DptRawEventWindow | None = None
    step_selections: List[DptRawStepSelection] | None = Field(None, max_length=50)
    same_context_confirmed: bool = False
    metadata: DptRawImportMetadata = Field(default_factory=DptRawImportMetadata)
    review_token: str | None = Field(None, min_length=16, max_length=128)

    @model_validator(mode="after")
    def check_trace_mapping(self) -> "DptRawTraceSelectionRequest":
        """Require distinct, explicit time, Vds, and Id trace choices."""
        if len({self.time_trace, self.vds_trace, self.id_trace}) != 3:
            raise ValueError("time_trace, vds_trace, and id_trace must be distinct")
        if len(set(self.step_indices)) != len(self.step_indices):
            raise ValueError("step_indices must be unique")
        if self.step_selections is not None:
            step_indexes = [item.step_index for item in self.step_selections]
            if len(set(step_indexes)) != len(step_indexes):
                raise ValueError("step_selections must use each source step only once")
            if self.step_indices != step_indexes:
                raise ValueError(
                    "step_indices must exactly match step_selections in the reviewed order"
                )
        return self


class DptRawStepSelection(BaseModel):
    """Explicit current and two event windows for one selected RAW step."""

    step_index: StrictInt = Field(..., ge=0)
    i_x: float = Field(..., gt=0, allow_inf_nan=False, description="Acquisition current [A]")
    turn_on: DptRawEventWindow
    turn_off: DptRawEventWindow


class DptRawImportRequest(DptRawTraceSelectionRequest):
    """User-selected plot, traces, events, and conditions for one raw import."""

    @model_validator(mode="after")
    def check_event_selection(self) -> "DptRawImportRequest":
        """Require one legacy pair or an explicit unique step selection."""
        if self.step_selections is not None:
            if not self.step_selections:
                raise ValueError("step_selections must contain at least one selected step")
            if self.turn_on is not None or self.turn_off is not None:
                raise ValueError("Use either step_selections or turn_on/turn_off")
            step_indexes = [item.step_index for item in self.step_selections]
            if len(set(step_indexes)) != len(step_indexes):
                raise ValueError("step_selections must use each source step only once")
            if self.step_indices != step_indexes:
                raise ValueError(
                    "step_indices must exactly match step_selections in the reviewed order"
                )
            if not self.same_context_confirmed:
                raise ValueError(
                    "Confirm that all selected steps share the entered test conditions."
                )
        elif self.turn_on is None or self.turn_off is None:
            raise ValueError("Provide turn_on and turn_off event windows.")
        return self


class DptEnergyPointRequest(BaseModel):
    """One explicit current-axis point tied to a stored event acquisition."""

    measurement_index: StrictInt = Field(..., ge=0)
    acquisition_index: StrictInt = Field(..., ge=0)
    i_x: float = Field(..., gt=0, allow_inf_nan=False, description="Current axis in A")
    vds_sign: Literal[-1, 1] = Field(..., description="Explicit Vds polarity multiplier")
    id_sign: Literal[-1, 1] = Field(..., description="Explicit Id polarity multiplier")


class DptEnergyDerivationRequest(BaseModel):
    """Inputs for one E_on or E_off graph derived from exact stored samples."""

    loss_type: Literal["e_on", "e_off"]
    points: List[DptEnergyPointRequest] = Field(..., min_length=2, max_length=50)


class DptEnergySaveRequest(DptEnergyDerivationRequest):
    """Reviewed inputs plus server-issued source and request preconditions."""

    source_token: str = Field(..., min_length=64, max_length=64, pattern="^[a-f0-9]{64}$")
    request_token: str = Field(..., min_length=64, max_length=64, pattern="^[a-f0-9]{64}$")
    save_action: Literal["add", "extend"]
    existing_curve_index: Optional[StrictInt] = Field(None, ge=0)
    target_curve_count: StrictInt = Field(..., ge=0)
    target_edit_token: Optional[str] = Field(
        None, min_length=64, max_length=64, pattern="^[a-f0-9]{64}$"
    )

    @model_validator(mode="after")
    def validate_target_snapshot(self) -> "DptEnergySaveRequest":
        """Require a complete, internally consistent target state snapshot."""
        if self.save_action == "add":
            if self.existing_curve_index is not None or self.target_edit_token is not None:
                raise ValueError("An add target cannot include a curve index or edit token")
        elif (
            self.existing_curve_index is None
            or self.existing_curve_index >= self.target_curve_count
            or self.target_edit_token is None
        ):
            raise ValueError("An extend target needs its curve index and edit token")
        return self


class PLECSImportResponse(BaseModel):
    """Response schema for PLECS import operations."""

    success: bool
    transistors_imported: int
    transistor_ids: List[str]
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
