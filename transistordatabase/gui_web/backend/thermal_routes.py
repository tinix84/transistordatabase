"""Typed transport for explicit steady and fixed-case thermal calculations."""

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field, model_serializer

from transistordatabase.gui_web.backend.parameter_routes import Strict
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute
from transistordatabase.thermal import ThermalResourceLimitError, analyze_thermal

Values = Annotated[list[float], Field(min_length=1, max_length=10000)]


class Context(Strict):
    """A declared source with preserved known and unknown conditions."""

    source: str = Field(min_length=1, max_length=2000)
    conditions: dict[str, float | None]


class Interface(Context):
    """Area-specific interface resistivity on one selected physical path."""

    path: Literal["common", "forward", "reverse"]
    resistivity_k_m2_w: float = Field(ge=0)
    area_m2: float = Field(gt=0)


class ContactInterface(Interface):
    """One identified physical contact in an explicit series stack."""

    interface_id: str = Field(min_length=1, max_length=200)


class Steady(Context):
    """Explicit common and independent branch thermal paths."""

    operation: Literal["steady"]
    sink_temperature_c: float
    forward_power_w: float = Field(ge=0)
    reverse_power_w: float = Field(ge=0)
    common_resistance_k_w: float = Field(ge=0)
    forward_resistance_k_w: float = Field(ge=0)
    reverse_resistance_k_w: float = Field(ge=0)
    interface: Interface | None = None
    interfaces: Annotated[list[ContactInterface], Field(min_length=1, max_length=32)] | None = None

    @model_serializer(mode="wrap")
    def canonical_settings(self, handler):
        """Keep the existing steady request unchanged when no list is supplied."""
        result = handler(self)
        if "interfaces" not in self.model_fields_set:
            result.pop("interfaces", None)
        return result


class Network(Context):
    """Complete recorded Foster poles; a total alone is insufficient."""

    r_th_vector: Values
    tau_vector: Values | None = None
    c_th_vector: Values | None = None
    r_th_total: float | None = None


class Segment(Strict):
    """Exact ascending coefficients in normalized local time."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    power_coefficients_w: Annotated[list[float], Field(min_length=1, max_length=4)]
    sample_interval_index: int | None = None
    direction: Literal["positive", "negative", "zero"] | None = None
    heat_owner: Literal["forward", "reverse"] | None = None
    energy_j: float | None = None
    cumulative_start_j: float | None = None
    cumulative_end_j: float | None = None
    source: str | dict | None = None


class Event(Context):
    """Energy deposition with an explicit heat owner and temporal convention."""

    time_s: float = Field(ge=0)
    energy_j: float = Field(ge=0)
    heat_owner: Literal["forward", "reverse"]
    end_s: float | None = None


class Transient(Context):
    """One physical junction held at an explicit fixed case boundary."""

    operation: Literal["transient"]
    network: Network
    network_heat_owner: Literal["forward", "reverse", "shared-junction"]
    case_temperature_c: float
    initial_state: Literal["cold"] | Values
    gap_policy: Literal["reject", "zero-power"]
    event_mode: Literal["none", "impulse", "interval-average"]
    segments: Annotated[list[Segment], Field(min_length=1, max_length=10000)]
    events: Annotated[list[Event], Field(max_length=10000)] = []
    samples_per_segment: int = Field(ge=1, le=100)


class LinearLoss(Context):
    """Explicitly selected bounded temperature scaling estimate."""

    mode: Literal["linear-temperature-estimate"]
    selected: bool = False
    temperature_min_c: float
    temperature_max_c: float
    reference_temperature_c: float
    reference_power_w: float = Field(ge=0)
    coefficient_per_k: float


class RecordedLoss(Context):
    """Recorded loss versus temperature at unchanged other conditions."""

    mode: Literal["recorded"]
    temperature_min_c: float
    temperature_max_c: float
    temperature_c: Values
    power_w: Values


Loss = Annotated[LinearLoss | RecordedLoss, Field(discriminator="mode")]


class Electrothermal(Context):
    """One or two branches solved using bounded source losses."""

    operation: Literal["electrothermal"]
    sink_temperature_c: float
    common_resistance_k_w: float = Field(ge=0)
    branch_resistances_k_w: Values
    initial_temperatures_c: Values
    loss_models: Annotated[list[Loss], Field(min_length=1, max_length=2)]
    tolerance_k: float = Field(gt=0)
    max_iterations: int = Field(ge=1, le=10000)
    relaxation: float = Field(gt=0, le=1)


class PackageLimit(Context):
    """Separately reported package restriction and its applicability."""

    current_a: float = Field(ge=0)
    applicability: str = Field(min_length=1)


class Screening(Context):
    """Reference-path screening with explicit conduction model conditions."""

    operation: Literal["screening"]
    junction_limit_c: float
    reference_kind: Literal["case", "heatsink", "coolant"]
    reference_temperature_c: float
    path_resistance_k_w: float = Field(gt=0)
    threshold_v: float = Field(ge=0)
    resistance_ohm: float = Field(ge=0)
    package_limit: PackageLimit | None = None


Request = Annotated[
    Steady | Transient | Electrothermal | Screening, Field(discriminator="operation")
]
class ThermalJsonRoute(FiniteJsonRoute):
    """Preserve the existing thermal body policy with finite-literal validation."""

    max_body_bytes = None


router = APIRouter(
    prefix="/api/thermal", tags=["Thermal workbench"], route_class=ThermalJsonRoute
)


@router.post("/analyze")
def analyze(request: Request):
    """Return canonical evidence or a clear physical/resource rejection."""
    try:
        return analyze_thermal(request.model_dump())
    except ThermalResourceLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
