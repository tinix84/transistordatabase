"""Independent byte-level, numerical and HTTP checks for image calibration."""

import base64
import csv
import hashlib
import io
import itertools
import json
import struct
import zlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from transistordatabase import digitization
from transistordatabase.gui_web.backend.digitizer_routes import router


def chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def png(width=1000, height=600, gray=70):
    """Write an 8-bit grayscale PNG with no EXIF metadata."""
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    pixels = (b"\0" + bytes([gray]) * width) * height
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def upload(data, name="curve.png"):
    return {"filename": name, "content_base64": base64.b64encode(data).decode()}


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def body():
    data = png()
    identity = {"sha256": hashlib.sha256(data).hexdigest(), "width": 1000, "height": 600}
    return {
        **upload(data), "source": 'Sheet, "Fig. 2"\nconditions unknown',
        "calibration": {
            "image": identity,
            "x": dict(pixel_1=100, pixel_2=900, value_1=1, value_2=100, scale="log", unit="nF"),
            "y": dict(pixel_1=500, pixel_2=100, value_1=0, value_2=20, scale="linear", unit="V"),
            "allow_extrapolation": False,
        },
        "points": [{"x": 500, "y": 300}, {"x": 900, "y": 100}, {"x": 100, "y": 500}],
    }


def test_no_exif_png_and_csv_text_policy(client, body):
    info = client.post("/api/digitizer/image", json=upload(png()))
    assert info.status_code == 200, info.text
    for key, expected in body["calibration"]["image"].items():
        assert info.json()[key] == expected
    body["filename"] = "=1+1.png"
    body["source"] = '\t@SUM(1,2)\nquoted "source"'
    body["calibration"]["y"].update(value_2=-20, unit="  +V")
    response = client.post("/api/digitizer/transform", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result == digitization.transform_points(**body)
    row = next(csv.DictReader(io.StringIO(result["csv"])))
    assert (row["filename"], row["source"], row["y_unit"]) == (
        "'" + body["filename"], "'" + body["source"], "'  +V")
    assert result["source"] == body["source"] and result["units"]["y"] == "  +V"
    assert float(row["y"]) == result["points"][0]["y"] == -10
    assert float(row["x"]) == result["points"][0]["x"] == 10


@pytest.mark.parametrize("xscale,yscale", itertools.product(("linear", "log"), repeat=2))
def test_ordered_axis_maps(client, body, xscale, yscale):
    calibration = body["calibration"]
    calibration["x"].update(scale=xscale, value_1=100, value_2=1)
    calibration["y"].update(scale=yscale, value_1=4, value_2=64)
    body["points"].append({"x": 300, "y": 200})
    response = client.post("/api/digitizer/transform", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result == digitization.transform_points(**body)
    expected_x = [50.5, 1, 100, 75.25] if xscale == "linear" else [10, 1, 100, 10**1.5]
    expected_y = [34, 64, 4, 49] if yscale == "linear" else [16, 64, 4, 32]
    for i, (point, row) in enumerate(zip(result["points"], csv.DictReader(io.StringIO(result["csv"])), strict=True)):
        assert point["index"] == i and int(row["index"]) == i
        assert [point["pixel_x"], point["pixel_y"]] == list(body["points"][i].values())
        assert point["x"] == pytest.approx(expected_x[i])
        assert point["y"] == pytest.approx(expected_y[i])
        assert [float(row["x"]), float(row["y"])] == [point["x"], point["y"]]
        assert (row["x_unit"], row["y_unit"], row["source"]) == ("nF", "V", body["source"])


def test_extremes_and_extrapolation(client, body):
    calibration = body["calibration"]
    calibration["x"].update(value_1=1e-200, value_2=1e200)
    calibration["y"].update(value_1=-1e308, value_2=1e308)
    answer = digitization.transform_points(**body)
    assert [(p["x"], p["y"]) for p in answer["points"]] == [(1, 0), (1e200, 1e308), (1e-200, -1e308)]
    calibration["x"].update(pixel_1=100, pixel_2=101, value_1=1, value_2=1e-200)
    body["points"] = [{"x": 102, "y": 300}]
    assert client.post("/api/digitizer/transform", json=body).status_code == 422
    calibration["allow_extrapolation"] = True
    assert client.post("/api/digitizer/transform", json=body).status_code == 422
    body["points"][0]["x"] = 101.25
    assert digitization.transform_points(**body)["points"][0]["x"] == pytest.approx(1e-250, abs=0)
    body["points"][0]["y"] = 600
    assert client.post("/api/digitizer/transform", json=body).status_code == 422


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1e3"])
def test_non_numeric_http(client, body, value):
    body["points"][0]["x"] = value
    response = client.post("/api/digitizer/transform", content=json.dumps(body),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text


def test_identity_and_disallowed_file_locators(client, body):
    calibration = body["calibration"]
    for key, value in (("width", 999), ("height", 601), ("sha256", "1" * 64)):
        wrong = {**calibration, "image": {**calibration["image"], key: value}}
        assert client.post("/api/digitizer/transform", json={**body, "calibration": wrong}).status_code == 422
    assert client.post("/api/digitizer/transform", json={**body, **upload(png(gray=71))}).status_code == 422
    for key, value in (("path", "/etc/hosts"), ("url", "https://example.invalid/figure.png")):
        assert client.post("/api/digitizer/image", json={key: value, "filename": "curve.png"}).status_code == 422


def test_image_budgets_before_raster_load(client, monkeypatch):
    data = png(1, 1)
    for w, h in ((8193, 1), (4001, 4000)):
        header = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        assert client.post("/api/digitizer/image", json=upload(data[:8] + header + data[33:])).status_code == 413
    monkeypatch.setattr(digitization, "MAX_EXPANDED_BYTES", 3)
    assert client.post("/api/digitizer/image", json=upload(data)).status_code == 413
    monkeypatch.setattr(digitization, "MAX_BYTES", 5)
    monkeypatch.setattr(digitization, "MAX_BASE64", 8)
    for raw in (b"123456", b"1234567"):
        assert client.post("/api/digitizer/image", json=upload(raw)).status_code == 413


@pytest.mark.parametrize("missing", [1, 2, 3, 4])
def test_complete_png_ends_with_valid_checksum(client, missing):
    data = png(1, 1)
    for damaged in (data[:-missing], data[:-1] + b"!"):
        response = client.post("/api/digitizer/image", json=upload(damaged))
        assert response.status_code == 422, response.text


@pytest.mark.parametrize("orientation,status", [(None, 200), (1, 200), (3, 422), (6, 422)])
def test_exact_jpeg_bytes(client, orientation, status):
    stream = io.BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    Image.new("RGB", (23, 31)).save(stream, format="JPEG", exif=exif)
    data = stream.getvalue()
    response = client.post("/api/digitizer/image", json=upload(data, "raster.jpg"))
    assert response.status_code == status
    if status == 200:
        assert response.json()["sha256"] == hashlib.sha256(data).hexdigest()
        assert (response.json()["width"], response.json()["height"]) == (23, 31)
