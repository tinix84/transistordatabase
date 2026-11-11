"""Independent natural-pixel calibration, image limits and HTTP evidence."""

import base64
import copy
import csv
import hashlib
import io
import json
import struct
import zlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from transistordatabase.digitization import (
    MAX_BYTES, DigitizerResourceError, inspect_image, transform_points,
)
from transistordatabase.gui_web.backend.digitizer_routes import router


def raster(image_format="PNG", size=(1000, 600), orientation=1, color="white"):
    """Create independent raster bytes entirely in memory."""
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", size, color).save(stream, format=image_format, exif=exif)
    data = stream.getvalue()
    return data, base64.b64encode(data).decode()


@pytest.fixture
def request_body():
    """Provide a logarithmic X and reversed-pixel linear Y fixture."""
    _, encoded = raster()
    metadata = inspect_image("fixture.png", encoded)
    return {"filename": "fixture.png", "content_base64": encoded,
            "calibration": {
                "image": {k: metadata[k] for k in ("sha256", "width", "height")},
                "x": {"pixel_1": 100., "value_1": 1., "pixel_2": 900.,
                      "value_2": 100., "scale": "log", "unit": "A"},
                "y": {"pixel_1": 500., "value_1": 0., "pixel_2": 100.,
                      "value_2": 20., "scale": "linear", "unit": "V"},
                "allow_extrapolation": False},
            "points": [{"x": 500., "y": 300.}, {"x": 900., "y": 100.}],
            "source": "Datasheet, page 3, curve 1. Tj unknown."}


@pytest.fixture
def client():
    """Exercise only the headless digitizer router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_independent_fixture_export_and_axis_independence(request_body):
    """The 1000 by 600 fixture maps the midpoint to 10 A and 10 V."""
    original = copy.deepcopy(request_body)
    result = transform_points(**request_body)
    assert request_body == original
    assert result["points"][0] == {
        "index": 0, "pixel_x": 500., "pixel_y": 300., "x": 10., "y": 10.}
    assert result["points"][1]["x"] == 100.
    assert result["points"][1]["y"] == 20.
    rows = list(csv.DictReader(io.StringIO(result["csv"])))
    for row, point in zip(rows, result["points"], strict=True):
        for name in ("pixel_x", "pixel_y", "x", "y"):
            assert float(row[name]) == point[name]
        assert row["source"] == request_body["source"]
        assert row["sha256"] == result["image"]["sha256"]
        assert row["x_unit"] == "A" and row["y_unit"] == "V"
    request_body["points"][0]["x"] = 700.
    changed = transform_points(**request_body)
    assert changed["points"][0]["x"] == pytest.approx(10 ** 1.5)
    assert changed["points"][0]["y"] == 10.


@pytest.mark.parametrize("x_scale,y_scale", [
    ("linear", "linear"), ("log", "linear"), ("linear", "log"), ("log", "log")])
def test_all_scale_pairs_and_reversed_anchors(request_body, x_scale, y_scale):
    """Independent scales and reversed physical/pixel directions remain valid."""
    for axis_name, scale in (("x", x_scale), ("y", y_scale)):
        axis = request_body["calibration"][axis_name]
        axis.update(scale=scale, value_1=100., value_2=1.)
    result = transform_points(**request_body)
    assert result["points"][0]["x"] == (10. if x_scale == "log" else 50.5)
    assert result["points"][0]["y"] == (10. if y_scale == "log" else 50.5)
    for axis_name in ("x", "y"):
        axis = request_body["calibration"][axis_name]
        axis["pixel_1"], axis["pixel_2"] = axis["pixel_2"], axis["pixel_1"]
        axis["value_1"], axis["value_2"] = axis["value_2"], axis["value_1"]
    assert transform_points(**request_body)["points"] == result["points"]


def test_stable_extreme_values_and_exact_endpoints(request_body):
    """Log ratios and linear differences need not fit in a float."""
    axis = request_body["calibration"]["x"]
    axis.update(value_1=1e-200, value_2=1e200)
    assert transform_points(**request_body)["points"][0]["x"] == 1.
    assert transform_points(**request_body)["points"][1]["x"] == 1e200
    axis.update(scale="linear", value_1=-1e308, value_2=1e308)
    assert transform_points(**request_body)["points"][0]["x"] == 0.
    assert transform_points(**request_body)["points"][1]["x"] == 1e308


@pytest.mark.parametrize("mutation", [
    lambda r: r["calibration"]["x"].update(pixel_2=100.),
    lambda r: r["calibration"]["x"].update(value_2=1.),
    lambda r: r["calibration"]["x"].update(value_1=0.),
    lambda r: r["calibration"]["x"].update(pixel_1=-1.),
    lambda r: r["calibration"]["x"].update(pixel_2=1000.),
    lambda r: r["points"][0].update(x=-1.),
    lambda r: r["points"][0].update(y=600.),
    lambda r: r["points"][0].update(x=50.),
    lambda r: r["points"][0].update(x=float("nan")),
    lambda r: r["points"][0].update(x=True),
    lambda r: r["calibration"].update(allow_extrapolation=1),
    lambda r: r["calibration"]["image"].update(width=999),
    lambda r: r["calibration"]["image"].update(sha256="0" * 64),
    lambda r: r["calibration"]["x"].update(unit=" "),
])
def test_invalid_calibrations_and_points(request_body, mutation):
    """Invalid inputs fail without silent extrapolation or geometry changes."""
    mutation(request_body)
    with pytest.raises(ValueError):
        transform_points(**request_body)


def test_explicit_extrapolation_overflow_and_underflow(request_body):
    """Extrapolation is selected explicitly and finite positive output is required."""
    request_body["calibration"]["allow_extrapolation"] = True
    request_body["points"] = [{"x": 0., "y": 0.}]
    result = transform_points(**request_body)
    assert result["points"][0]["x"] == pytest.approx(10 ** -.25)
    assert result["points"][0]["y"] == 25.
    axis = request_body["calibration"]["x"]
    axis.update(pixel_1=100., pixel_2=101., value_1=1., value_2=1e300)
    request_body["points"][0]["x"] = 102.
    with pytest.raises(ValueError, match="finite"):
        transform_points(**request_body)
    request_body["points"][0]["x"] = 0.
    with pytest.raises(ValueError, match="underflows"):
        transform_points(**request_body)


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_exact_raster_identity_and_orientation(image_format):
    """Exact bytes define identity and orientation never silently rotates pixels."""
    data, encoded = raster(image_format)
    result = inspect_image("test.image", encoded)
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["byte_size"] == len(data)
    assert (result["width"], result["height"]) == (1000, 600)
    assert result["origin"] == "top-left" and result["y_direction"] == "down"
    _, rotated = raster(image_format, orientation=6)
    with pytest.raises(ValueError, match="orientation"):
        inspect_image("rotated.image", rotated)


def test_image_replacement_rejects_calibration(request_body):
    """An image with identical dimensions but different bytes invalidates calibration."""
    _, request_body["content_base64"] = raster(color="black")
    with pytest.raises(ValueError, match="identity"):
        transform_points(**request_body)


def test_animation_rejected():
    """Multiple PNG frames cannot share one static calibration."""
    stream = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(
        stream, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (10, 10), "black")])
    with pytest.raises(ValueError, match="Animated"):
        inspect_image("animated.png", base64.b64encode(stream.getvalue()).decode())


def test_budgets_before_decode_and_invalid_rasters():
    """Oversized encoded bytes or header dimensions fail before raster expansion."""
    with pytest.raises(DigitizerResourceError):
        inspect_image("huge.png", "A" * (4 * ((MAX_BYTES + 2) // 3) + 1))
    data, _ = raster(size=(1, 1))
    header = struct.pack(">IIBBBBB", 8192, 8192, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + header
    huge = data[:8] + struct.pack(">I", len(header)) + chunk + struct.pack(">I", zlib.crc32(chunk)) + data[33:]
    with pytest.raises(DigitizerResourceError):
        inspect_image("huge.png", base64.b64encode(huge).decode())
    for invalid in ("%%invalid%%", base64.b64encode(b"not an image").decode(),
                    base64.b64encode(data[:40]).decode()):
        with pytest.raises(ValueError):
            inspect_image("bad.png", invalid)
    _, gif = raster("GIF")
    with pytest.raises(ValueError, match="PNG and JPEG"):
        inspect_image("bad.gif", gif)


def test_http_equality_typed_errors_and_resource_status(request_body, client):
    """Typed HTTP matches Python and returns safe 422 or resource 413 errors."""
    expected = transform_points(**request_body)
    response = client.post("/api/digitizer/transform", json=request_body)
    assert response.status_code == 200
    assert response.json() == expected
    metadata = client.post("/api/digitizer/image", json={
        key: request_body[key] for key in ("filename", "content_base64")})
    assert metadata.json() == expected["image"]
    for value in (True, "500", float("nan"), float("inf")):
        request_body["points"][0]["x"] = value
        response = client.post("/api/digitizer/transform",
                               content=json.dumps(request_body),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["detail"]
    response = client.post("/api/digitizer/image", json={
        "filename": "huge.png", "content_base64": "A" * (4 * ((MAX_BYTES + 2) // 3) + 1)})
    assert response.status_code == 413


@pytest.mark.parametrize("field", ["pixel_1", "value_1", "width", "allow_extrapolation"])
def test_http_strict_anchor_identity_and_selection_types(request_body, client, field):
    """Boolean numbers and numeric extrapolation selection never reach the science."""
    if field in ("pixel_1", "value_1"):
        request_body["calibration"]["x"][field] = True
    elif field == "width":
        request_body["calibration"]["image"][field] = True
    else:
        request_body["calibration"][field] = 1
    response = client.post("/api/digitizer/transform", json=request_body)
    assert response.status_code == 422
