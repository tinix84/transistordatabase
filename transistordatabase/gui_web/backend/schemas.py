"""
Pydantic schemas for API request/response validation.

Defines data models for curve management and other API operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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


class TopologyCalculationRequest(BaseModel):
    """Request schema for topology calculations."""

    topology: str = Field(..., description="Topology type: 'buck', 'boost', 'buck-boost'")
    v_in: float = Field(..., description="Input voltage in V", gt=0)
    v_out: float = Field(..., description="Output voltage in V", gt=0)
    p_out: float = Field(..., description="Output power in W", gt=0)
    f_sw: float = Field(..., description="Switching frequency in Hz", gt=0)
    transistor_id: Optional[str] = Field(None, description="Transistor to use for calculations")

    # Optional parameters
    t_j: Optional[float] = Field(25.0, description="Junction temperature in deg C")
    v_g: Optional[float] = Field(None, description="Gate voltage in V")
    r_g_on: Optional[float] = Field(None, description="Gate resistance turn-on in Ohm")
    r_g_off: Optional[float] = Field(None, description="Gate resistance turn-off in Ohm")


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


class PLECSImportResponse(BaseModel):
    """Response schema for PLECS import operations."""

    success: bool
    transistors_imported: int
    transistor_ids: List[str]
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
