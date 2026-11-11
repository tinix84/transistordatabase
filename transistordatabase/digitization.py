"""Calibrate exact raster-image pixels without modifying the source image."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import math
import warnings

from PIL import Image, UnidentifiedImageError

MAX_BYTES = 8 * 1024 * 1024
MAX_BASE64 = 4 * ((MAX_BYTES + 2) // 3)
MAX_PIXELS = 16_000_000
MAX_SIDE = 8192
MAX_EXPANDED_BYTES = 64_000_000
MAX_POINTS = 20_000


class DigitizerResourceError(ValueError):
    """An uploaded image exceeds a declared resource budget."""


def inspect_image(filename: str, content_base64: str) -> dict:
    """Validate bounded PNG/JPEG bytes and return their exact image identity."""
    if not isinstance(filename, str) or not filename.strip() or len(filename) > 240:
        raise ValueError("Supply an image filename with at most 240 characters")
    if not isinstance(content_base64, str) or not content_base64:
        raise ValueError("Supply the exact image bytes as base64")
    if len(content_base64) > MAX_BASE64:
        raise DigitizerResourceError("Image bytes exceed the 8 MiB limit")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Image content must be valid base64") from exc
    if len(data) > MAX_BYTES:
        raise DigitizerResourceError("Image bytes exceed the 8 MiB limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            stream = io.BytesIO(data)
            with Image.open(stream) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise ValueError("Only PNG and JPEG images are supported")
                width, height = image.size
                # Image.open reads headers. Check budgets before verify/load.
                if (width > MAX_SIDE or height > MAX_SIDE
                        or width * height > MAX_PIXELS
                        or width * height * 4 > MAX_EXPANDED_BYTES):
                    raise DigitizerResourceError("Image dimensions exceed the pixel budget")
                if width < 1 or height < 1:
                    raise ValueError("Image dimensions must be positive")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported")
                image_format = image.format
                image.verify()
                if image_format == "PNG":
                    # Pillow verifies up to the IEND header. Check its empty payload and CRC.
                    stream.seek(-8, io.SEEK_CUR)
                    if stream.read(12) != b"\x00\x00\x00\x00IEND\xaeB\x60\x82":
                        raise ValueError("PNG end chunk is missing or corrupt")
            # PNG EXIF access can load pixels. Read it after the integrity check.
            with Image.open(io.BytesIO(data)) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("EXIF orientation must be 1 for natural-pixel calibration")
                image.load()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise DigitizerResourceError("Image dimensions exceed the pixel budget") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Image bytes do not contain a complete supported raster") from exc
    return {
        "filename": filename,
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width, "height": height,
        "format": image_format, "byte_size": len(data), "orientation": 1,
        "pixel_space": "natural-image-pixels", "origin": "top-left",
        "x_direction": "right", "y_direction": "down",
    }


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _pixel(value, dimension, name):
    result = _number(value, name)
    if not 0 <= result <= dimension - 1:
        raise ValueError(f"{name} must be within natural-image pixel bounds")
    return result


def _axis(axis, dimension, name):
    required = {"pixel_1", "pixel_2", "value_1", "value_2", "scale", "unit"}
    if not isinstance(axis, dict) or set(axis) != required:
        raise ValueError(f"{name} requires two pixel/value anchors, scale and unit")
    result = {key: _pixel(axis[key], dimension, f"{name} {key}")
              for key in ("pixel_1", "pixel_2")}
    result.update({key: _number(axis[key], f"{name} {key}")
                   for key in ("value_1", "value_2")})
    if result["pixel_1"] == result["pixel_2"]:
        raise ValueError(f"{name} pixel anchors must be distinct")
    if result["value_1"] == result["value_2"]:
        raise ValueError(f"{name} physical anchors must be distinct")
    if axis["scale"] not in ("linear", "log"):
        raise ValueError(f"{name} scale must be linear or log")
    if axis["scale"] == "log" and min(result["value_1"], result["value_2"]) <= 0:
        raise ValueError(f"{name} logarithmic anchors must be positive")
    if (not isinstance(axis["unit"], str) or not axis["unit"].strip()
            or len(axis["unit"]) > 80):
        raise ValueError(f"{name} requires an explicit unit with at most 80 characters")
    return {**result, "scale": axis["scale"], "unit": axis["unit"]}


def _map(pixel, axis, extrapolate):
    p1, p2 = axis["pixel_1"], axis["pixel_2"]
    v1, v2 = axis["value_1"], axis["value_2"]
    if not extrapolate and not min(p1, p2) <= pixel <= max(p1, p2):
        raise ValueError("Points outside calibration anchors require explicit extrapolation")
    if pixel == p1:
        return v1
    if pixel == p2:
        return v2
    fraction = (pixel - p1) / (p2 - p1)
    if not math.isfinite(fraction):
        raise ValueError("Pixel anchors cannot resolve this point with finite precision")
    try:
        if axis["scale"] == "log":
            exponent = (1 - fraction) * math.log10(v1) + fraction * math.log10(v2)
            value = 10.0 ** exponent
            if value <= 0:
                raise ValueError("Logarithmic output underflows positive floating-point values")
        elif 0 <= fraction <= 1:
            value = (1 - fraction) * v1 + fraction * v2
        else:
            # Endpoint basis avoids overflow from weighted same-sign terms.
            value = v1 + fraction * (v2 - v1)
    except OverflowError as exc:
        raise ValueError("Calibrated output exceeds finite floating-point values") from exc
    if not math.isfinite(value):
        raise ValueError("Calibrated output exceeds finite floating-point values")
    return value


def _csv_text(value: str) -> str:
    """Keep text that starts with a spreadsheet formula marker in a text cell."""
    if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def transform_points(filename: str, content_base64: str, calibration: dict,
                     points: list[dict], source: str = "Unknown") -> dict:
    """Return ordered physical points, calibration and CSV from one calculation.

    Physical outputs retain the declared units. This operation does not infer
    SI conversions, device fields, test conditions or measurement accuracy.
    """
    image = inspect_image(filename, content_base64)
    if (not isinstance(calibration, dict)
            or set(calibration) != {"image", "x", "y", "allow_extrapolation"}):
        raise ValueError("Supply an image-bound calibration and explicit extrapolation selection")
    identity = {key: image[key] for key in ("sha256", "width", "height")}
    supplied = calibration["image"]
    if (not isinstance(supplied, dict) or set(supplied) != set(identity)
            or any(type(supplied[key]) is not type(identity[key]) for key in identity)
            or supplied != identity):
        raise ValueError("Calibration identity does not match the exact uploaded image")
    extrapolate = calibration["allow_extrapolation"]
    if not isinstance(extrapolate, bool):
        raise ValueError("Extrapolation selection must be a boolean")
    axes = {name: _axis(calibration[name], image[dimension], name)
            for name, dimension in (("x", "width"), ("y", "height"))}
    if not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS:
        raise ValueError("Supply between 1 and 20000 selected points")
    if not isinstance(source, str) or not source.strip() or len(source) > 2000:
        raise ValueError("Supply source metadata with at most 2000 characters")
    mapped = []
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != {"x", "y"}:
            raise ValueError("Each selected point requires x and y natural-pixel coordinates")
        px = _pixel(point["x"], image["width"], "Point x")
        py = _pixel(point["y"], image["height"], "Point y")
        mapped.append({"index": index, "pixel_x": px, "pixel_y": py,
                       "x": _map(px, axes["x"], extrapolate),
                       "y": _map(py, axes["y"], extrapolate)})
    calibrated = {"image": identity, **axes, "allow_extrapolation": extrapolate}
    units = {name: axis["unit"] for name, axis in axes.items()}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    columns = ["index", "pixel_x", "pixel_y", "x", "y"]
    writer.writerow(columns + ["x_unit", "y_unit", "filename", "sha256",
                               "width", "height", "source"])
    metadata = [_csv_text(units["x"]), _csv_text(units["y"]), _csv_text(filename),
                image["sha256"], image["width"], image["height"], _csv_text(source)]
    for point in mapped:
        writer.writerow([point[key] for key in columns] + metadata)
    return {"image": image, "calibration": calibrated, "points": mapped,
            "units": units, "source": source, "csv": stream.getvalue()}
