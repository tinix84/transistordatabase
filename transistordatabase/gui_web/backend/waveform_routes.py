"""Typed, side-effect-free transport for the signed waveform workbench."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field, model_serializer

from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict
from transistordatabase.waveform_losses import WaveformResourceLimitError, analyze_waveform

Vector = Annotated[list[float], Field(min_length=2, max_length=10000)]
Conditions = dict[str, float | None]


class Context(Strict):
    """Inspectable source identifier; science additionally rejects blank strings."""

    source: str = Field(min_length=1, max_length=2000)


class LinearDrop(Context):
    """Explicit threshold-plus-resistance estimate, in V and ohm."""

    mode: Literal["linear-drop-estimate"]
    selected: bool = False
    offset_v: float = Field(ge=0)
    resistance_ohm: float = Field(ge=0)
    conditions: Conditions


class RecordedDrop(Context):
    """Bounded magnitude-current V-I curve at exact declared conditions."""

    mode: Literal["recorded"]
    current_a: Vector
    drop_v: Vector
    recorded_conditions: Conditions
    requested_conditions: Conditions


class RecordedEnergy(Context):
    """Bounded measured current-energy slice, with independent test conditions."""

    mode: Literal["recorded"]
    current_a: Vector
    energy_j: Vector
    recorded_conditions: Conditions
    requested_conditions: Conditions


class ScaledEnergy(Context):
    """Explicit energy proportional to bus-voltage estimate on a bounded I slice."""

    mode: Literal["voltage-scaling-estimate"]
    selected: bool = False
    current_a: Vector
    energy_j: Vector
    recorded_conditions: Conditions
    requested_conditions: Conditions


Drop = Annotated[LinearDrop | RecordedDrop, Field(discriminator="mode")]
Energy = Annotated[RecordedEnergy | ScaledEnergy, Field(discriminator="mode")]


class Waveform(Strict):
    """Aligned SI samples; only the switching bus may be supplied as a scalar."""

    time_s: Vector
    current_a: Vector
    gate_v: Vector | None = None
    device_voltage_v: Vector | None = None
    bus_voltage_v: float | Vector | None = None


class Conduction(Strict):
    """Explicit signed branch models and independent conduction-duty weighting."""

    omit: bool = False
    omission_reason: str = ""
    positive: Drop | None = None
    negative: Drop | None = None
    duty: float | Vector = 1.0


class EventPolicy(Strict):
    """Gate > threshold, explicit N-1 masks, or a reasoned omission."""

    mode: Literal["gate", "masks", "omitted"]
    threshold_v: float | None = None
    turn_on: list[bool] | None = None
    turn_off: list[bool] | None = None
    omission_reason: str = ""


class SwitchingComponent(Strict):
    """One requested/omitted event-energy component with signed model selection."""

    omit: bool = False
    omission_reason: str = ""
    positive: Energy | None = None
    negative: Energy | None = None


class Switching(Strict):
    """Independent turn-on and turn-off models, conditions and omissions."""

    turn_on: SwitchingComponent
    turn_off: SwitchingComponent


class Spectrum(Strict):
    """Uniform-only discrete FFT choices; no hidden resampling."""

    window: Literal["rectangular", "hann"]
    detrend: Literal["none", "mean"]
    relative_threshold: float = Field(default=0.1, ge=0, le=1)

    @model_serializer(mode="wrap")
    def canonical_settings(self, handler):
        """Preserve legacy input hashes when the new setting is absent."""
        data = handler(self)
        if "relative_threshold" not in self.model_fields_set:
            data.pop("relative_threshold", None)
        return data


class Provenance(Strict):
    """User-declared waveform origin/calibration, without an invented measurement ID."""

    source: str | None = None
    calibration_notes: str | None = None
    imported_filename: str | None = None
    edited_after_import: bool = False


class WaveformRequest(Strict):
    """Canonical reproducible request shared by browser, Python and future MCP."""

    waveform: Waveform
    conduction: Conduction
    events: EventPolicy
    switching: Switching
    spectrum: Spectrum | None = None
    provenance: Provenance | None = None


router = APIRouter(prefix="/api/waveform", tags=["Signed waveforms"], route_class=ParameterRoute)


@router.post("/analyze")
def analyze(request: WaveformRequest):
    """Return SI evidence, or reject unsupported/missing conditions without fallback."""
    try:
        return analyze_waveform(request.model_dump(exclude_none=True))
    except WaveformResourceLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
