# Calibrated image digitization usage

1. Send `filename` and exact `content_base64` bytes to `POST /api/digitizer/image`.
2. Record the returned SHA-256 hash, width and height in `calibration.image`.
3. Supply independent `calibration.x` and `calibration.y` axis objects.
4. Supply ordered natural-image pixel points to `POST /api/digitizer/transform`.
5. Include the same image bytes, filename and source description in the transform request.
6. Save the returned JSON and `csv` string as the exports.

Each axis requires `pixel_1`, `value_1`, `pixel_2`, `value_2`, `scale` and `unit`.
The scale is `linear` or `log`. Logarithmic physical anchors must be positive.
Pixel anchors and physical anchors must be distinct. Reversed directions are valid.
Set `calibration.allow_extrapolation` explicitly to permit points outside the anchor intervals.
The default is `false`.

The pixel origin is upper left. X increases rightward. Y increases downward.
Coordinates are natural-image pixel centers.
X ranges from zero through width minus one. Y ranges from zero through height minus one.
Display resizing does not change these coordinates.
The image identity must match the uploaded bytes and dimensions for every transform.

PNG and JPEG images must have EXIF orientation 1 or no orientation tag.
Animated images are unsupported. Upload limits are 8 MiB, 8192 pixels per side and 16 million pixels.
The expanded raster budget is 64 million bytes. Resource limits return HTTP 413.
Invalid raster data, calibration, numeric inputs or image identity return HTTP 422.
The operation does not save or modify the source image.

Physical outputs retain the declared units. No SI conversion is inferred.
The returned CSV contains original pixel coordinates, physical values, units, image identity and source metadata.
The JSON also contains the full calibration. Both exports use the same Python calculation.
Save the JSON to retain the exact supplied metadata text.
CSV text cells receive an apostrophe prefix if they start with a tab, carriage return, or line feed.
The same prefix applies when the first non-whitespace character is `=`, `+`, `-`, or `@`.
This prevents spreadsheet applications from interpreting those text cells as formulas.
Numeric CSV columns retain their serialized Python values. Ordinary text labels such as `A` and `V` remain unchanged.
Source metadata can name the document, page, curve and known or unknown test conditions.
Calibration does not establish device accuracy or infer test conditions.

For a 1000 by 600 image, calibrate X pixels 100 and 900 to 1 and 100 with logarithmic scale.
Calibrate Y pixels 500 and 100 to 0 and 20 V with linear scale.
The point at natural pixel 500/300 returns physical values 10/10.

This workflow provides selected-point export. It does not fit curves, recognize curves automatically or correct perspective.
It does not assign points to device fields or change database records.
