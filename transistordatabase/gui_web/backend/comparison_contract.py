"""Transport contract for observed curves and explicit static-duty estimates."""
from typing import Annotated

from pydantic import Field

from transistordatabase.gui_web.backend.parameter_routes import Strict

Values = Annotated[list[float | None], Field(min_length=2, max_length=3)]


class ComparisonConfig(Strict):
    """Omitted inputs leave scientific estimates unavailable, without defaults."""

    t_j: float | None = None
    i_channel: float | None = None
    f_sw: float | None = None
    duty: float | None = None
    output_power_w: float | None = None
    v_supply: Values | None = None
    v_g: Values | None = None
    v_g_off: Values | None = None
    r_g_on: Values | None = None
    r_g_off: Values | None = None
    parallel_count: Annotated[list[int], Field(min_length=2, max_length=3)] | None = None


class ComparisonRequest(Strict):
    """Two or three stored device IDs plus optional estimate inputs."""

    transistor_ids: Annotated[list[str], Field(min_length=2, max_length=3)]
    config: ComparisonConfig = Field(default_factory=ComparisonConfig)
