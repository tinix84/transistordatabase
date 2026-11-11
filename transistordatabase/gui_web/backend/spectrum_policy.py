"""Typed coordinate masks and declared discrete frequency bands."""
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, field_validator

from transistordatabase.gui_web.backend.parameter_routes import Strict


def _frequency_precision(values):
    for value in values:
        if isinstance(value, bool):
            raise ValueError("Frequency coordinates require actual numbers")
        if isinstance(value, (int, Decimal)):
            try:
                number = float(value)
                if Decimal(str(number)) != Decimal(value):
                    raise ValueError("Frequency precision exceeds accepted binary64 coordinates")
            except (ValueError, OverflowError) as error:
                raise ValueError("Frequency precision exceeds accepted binary64 coordinates") from error
    return values


class SpectrumMask(Strict):
    """A source-declared peak limit profile with explicit interpolation."""

    unit: Literal["A_peak", "V_peak"]
    frequency_hz: Annotated[list[float], Field(min_length=1, max_length=5001)]
    limit_peak: Annotated[list[float | None], Field(min_length=1, max_length=5001)]
    interpolation: Literal["exact", "linear-frequency", "log-frequency"] = "exact"
    source: str = Field(min_length=1, max_length=2000)

    @field_validator("frequency_hz", mode="before")
    @classmethod
    def exact_frequency_coordinates(cls, values):
        """Reject precision loss before declared coordinates collapse into bins."""
        return _frequency_precision(values) if isinstance(values, list) else values


class SpectrumBand(Strict):
    """A named range of discrete bins with a declared endpoint convention."""

    name: str = Field(min_length=1, max_length=200)
    lower_hz: float
    upper_hz: float
    endpoints: Literal["closed", "open", "left-closed", "right-closed"] = "closed"

    @field_validator("lower_hz", "upper_hz", mode="before")
    @classmethod
    def exact_band_coordinates(cls, value):
        """Retain the precision of declared frequency boundaries."""
        return _frequency_precision([value])[0]
