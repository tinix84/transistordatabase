"""Shared typed policies for sampled gate hysteresis."""
from typing import Annotated, Literal
from decimal import Decimal, InvalidOperation

from pydantic import Field, StrictBool, field_validator

from transistordatabase.gui_web.backend.parameter_routes import Strict


class AbsoluteThresholds(Strict):
    """Two explicitly supplied voltage thresholds."""

    scale: Literal["absolute"]
    low_v: float
    high_v: float


class RelativeThresholds(Strict):
    """Fractions of an explicitly declared calibrated voltage range."""

    scale: Literal["relative"]
    low_fraction: float
    high_fraction: float
    minimum_v: float
    maximum_v: float

    @field_validator("low_fraction", "high_fraction", mode="before")
    @classmethod
    def exact_fraction_bounds(cls, value):
        """Check original decimal limits before binary64 conversion can round them."""
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float, Decimal)) and 0 <= value <= 1
        except InvalidOperation:
            valid = False
        if not valid:
            raise ValueError("Hysteresis fractions must be finite numbers between 0 and 1")
        return value


class HysteresisPolicy(Strict):
    """Explicit hysteresis selection and optional state at sample zero."""

    thresholds: Annotated[AbsoluteThresholds | RelativeThresholds, Field(discriminator="scale")]
    initial_state: StrictBool | None = None


class HysteresisEvidence(Strict):
    """Resolved thresholds, original-sample states and pre-edge event brackets."""

    method: Literal["strict-hysteresis-v1"]
    threshold_scale: Literal["absolute", "relative"]
    low_v: float
    high_v: float
    calibration_min_v: float | None
    calibration_max_v: float | None
    initial_state: bool
    initial_state_source: Literal["declared", "first-sample-below-low", "first-sample-above-high"]
    states: list[bool]
    turn_on_brackets: list[int]
    turn_off_brackets: list[int]
    semantics: str
    units: dict[str, str]
