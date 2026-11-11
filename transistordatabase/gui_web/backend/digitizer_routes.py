"""Typed transport for exact-image calibration and ordered point export."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

from transistordatabase.digitization import (
    MAX_POINTS, DigitizerResourceError, inspect_image, transform_points,
)
from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute, Strict

router = APIRouter(prefix="/api/digitizer", tags=["Image digitization"],
                   route_class=ParameterRoute)


class ImageUpload(Strict):
    """Exact uploaded bytes, without server paths or remote addresses."""

    filename: str = Field(min_length=1, max_length=240)
    # The reusable operation returns 413 for encoded or decoded byte limits.
    content_base64: str = Field(min_length=1)


class ImageIdentity(Strict):
    """Calibration binding to exact bytes and natural image dimensions."""

    sha256: str = Field(pattern="^[0-9a-f]{64}$")
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class Axis(Strict):
    """Independent two-anchor axis in explicit physical units."""

    pixel_1: float
    value_1: float
    pixel_2: float
    value_2: float
    scale: Literal["linear", "log"]
    unit: str = Field(min_length=1, max_length=80)


class Calibration(Strict):
    """Independent axes and explicit permission for extrapolation."""

    image: ImageIdentity
    x: Axis
    y: Axis
    allow_extrapolation: bool = False


class PixelPoint(Strict):
    """A point in natural image pixels, with upper-left origin."""

    x: float
    y: float


class Transform(ImageUpload):
    """Calibration and ordered points, with declared source metadata."""

    calibration: Calibration
    points: list[PixelPoint] = Field(min_length=1, max_length=MAX_POINTS)
    source: str = Field(default="Unknown", min_length=1, max_length=2000)


class ImageMetadata(ImageIdentity):
    """Validated exact-byte identity and browser-compatible pixel convention."""

    filename: str
    byte_size: int
    format: Literal["PNG", "JPEG"]
    orientation: Literal[1]
    pixel_space: Literal["natural-image-pixels"]
    origin: Literal["top-left"]
    x_direction: Literal["right"]
    y_direction: Literal["down"]


class PhysicalPoint(Strict):
    """Ordered original pixels and their calculated physical values."""

    index: int
    pixel_x: float
    pixel_y: float
    x: float
    y: float


class AxisUnits(Strict):
    """Physical output units, without an inferred SI conversion."""

    x: str
    y: str


class TransformResult(Strict):
    """One calculated result supplies the table, JSON and CSV exports."""

    image: ImageMetadata
    calibration: Calibration
    points: list[PhysicalPoint]
    units: AxisUnits
    source: str
    csv: str


def _run(operation, request):
    try:
        return operation(**request.model_dump())
    except DigitizerResourceError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/image", response_model=ImageMetadata)
def image_metadata(request: ImageUpload):
    """Inspect exact uploaded raster bytes without saving or modifying them."""
    return _run(inspect_image, request)


@router.post("/transform", response_model=TransformResult)
def transform(request: Transform):
    """Calculate calibrated points and both exports through reusable Python."""
    return _run(transform_points, request)
