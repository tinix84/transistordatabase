# Legacy MAT data API

Upload original file bytes as base64. The API does not accept server filesystem paths.
A source filename identifies provenance. It does not supply identity, ratings, temperature, gate voltage, or gate resistance.

## Preview and archive

Send `POST /api/mat/preview` with this JSON structure:

```json
{"mat_base64": "<base64 of original bytes>", "source_name": "original.mat"}
```

Save the complete response as a canonical JSON archive.
The response contains the original bytes, SHA-256, schema version, source version, field tree, diagnostics, and calibration comparisons.
Array descriptors retain dtype and shape. Empty arrays retain their original dimensions and do not mean zero.
Unknown fields remain in the field tree and the original bytes.

Large arrays use `encoding: "source-array-reference"`.
Send `POST /api/mat/array` with the same source fields and the descriptor's `source_array_path` as `array_path`.
The response contains the original array as `base64-c-order`, with dtype, shape, byte count, and SHA-256.
Descriptors include `itemsize` in bytes. NumPy `U1` uses four bytes per Unicode code point.
Integers outside ±(2^53−1) use binary references to prevent browser JSON rounding.
For an RGB uint8 image, consecutive bytes represent consecutive channels, columns, and rows in that order.
No image transformation occurs.

Calibration comparisons preserve the historical `x/image-height,y/image-width` convention.
Residuals compare reconstructed values with stored values. They do not replace stored values.
Outside-calibration indices remain visible.

The Python equivalents are:

```python
from pathlib import Path
from transistordatabase.legacy_mat import (
    archive_from_json, mat_array_blob, preview_mat,
)

raw = Path("original.mat").read_bytes()
archive = preview_mat(raw, "original.mat")
verified_archive = archive_from_json(archive)
image = mat_array_blob(raw, ["scData", "forwardConductionTrace", "image"])
```

## Compatibility download

Send `POST /api/mat/export` with the source fields and an explicit `mode`:

| Mode | Result |
| --- | --- |
| `source-preserved` | Exact original bytes |
| `legacy-mat5` | Semantic SC MAT5 conversion through SciPy |

MAT5 conversion preserves source fields, arrays, trace images, and calibration values.
Version 0.0 conversion adds only three empty thermal-interface areas.
Version 0.1 conversion changes only the version to 0.2 and the binary format.
Neither conversion supplies missing measurement conditions or changes physical values.
Unsupported field names cause an explicit export error. Use the original-byte archive for those sources.
Empty character shapes that SciPy cannot preserve cause an explicit export error with the field path.
The writer compares exported fields, dimensions, dtypes, and values before returning bytes.
Keep new core declarations in the native JSON document. Legacy format conversion does not encode those declarations.

## Explicit core finalization

Send `POST /api/mat/finalize` with the source fields and these declarations:

| Field | Required content |
| --- | --- |
| `identity` | `name`, `type`, `author`, `manufacturer`, `housing_type` |
| `ratings` | `v_abs_max`, `i_abs_max`, `i_cont`, `t_j_max`, in V, A, A, °C |
| `thermal_properties` | Positive `housing_area`, `cooling_area`, in m². Optional explicit case-interface values use K/W |
| `units` | `SC-editor-SI-confirmed` |
| `declaration_source` | Nonblank citation for separately supplied identity, ratings, areas, and conditions |
| `curves` | One to six explicit selections, as described below |
| `omission_reasons` | A nonblank reason for every unselected original branch mechanism |

Each curve selection requires all these fields:

```json
{
  "kind": "conduction",
  "branch": "forward",
  "target": "switch",
  "source": "original-trace",
  "coordinate_convention": "trace-positive-magnitudes",
  "sample_indices": null,
  "sample_order": "sort-distinct-current",
  "zero_policy": "reject-zero",
  "evidence_reference": "Datasheet figure and independent test-condition reference",
  "extrapolation_policy": "reject",
  "conditions": {
    "t_j_c": 125.0,
    "gate_voltage_v": 15.0,
    "gate_off_voltage_v": null,
    "gate_resistance_ohm": null,
    "supply_voltage_v": null,
    "commutation_device": null
  }
}
```

Use `conduction`, `turn-on`, or `turn-off` for `kind`.
Use `forward` or `reverse` for `branch`.
Use `switch` or `diode` for `target`.
The existing diode core supports conduction and turn-off recovery. It does not represent a separate turn-on mechanism.
Each component mechanism accepts one selected branch. Original unselected branches remain in the archive.

Prefer `original-trace` when combined arrays contain synthetic points or conflicting reference voltages.
Recognized original labels use A, V, and J. Reversed axes and trailing label whitespace retain their original source values.
Unknown labels cause a finalization error.
Original points outside stored calibration must be excluded with `sample_indices` or selected with `extrapolation_policy: "explicit-estimate"`.
The default policy is `reject`. Explicit estimates retain original source values and the selected policy in provenance.

Combined curves require `source: "combined"` and `coordinate_convention: "signed-passive"`.
The selected current sign defines the branch. Reverse conduction maps both current and voltage to positive magnitudes.
Energy never changes sign automatically.
Branches remain separate. Finalization does not interpolate across zero or extend coverage to zero.

`sample_indices` selects original array indices. `null` selects all recorded samples before branch separation.
Exclude combined ±1e-6 A synthetic candidates explicitly.
Use `recorded-order` for increasing source current or explicitly select `sort-distinct-current`.
Duplicate current values cause an error. Sorting never discards conflicting values.
Use `recorded-branch-value` instead of `reject-zero` only when explicitly selecting a recorded branch zero value.

Declare conditions independently for each curve.
Energy selections require known gate voltage, nonnegative gate resistance, and positive supply voltage.
A source reference voltage must match the declaration. No voltage scaling occurs.
Unresolved negative energy and identified synthetic zero-energy placeholders cannot enter a passive loss model.
All-zero combined energy requires an original trace or an explicit interpretation.
Use `data_status: "source-zero-unverified"` with a nonblank `source_zero_acknowledgement` to retain an unverified recorded zero curve.
This declaration acknowledges that the original trace and measurement evidence are absent. It does not classify the curve as ideal.
Alternatively, use `data_status: "declared-ideal-model"` with a nonblank `ideal_model_scope`.
`recorded-source-values` describes preserved source numbers. It does not establish measurement accuracy or measured status.
The exact placeholder signature remains unsupported even when an ideal model status is declared.

Omission keys are the six original mechanisms:
`forwardConductionTrace`, `reverseConductionTrace`, `forwardTurnOnTrace`, `reverseTurnOnTrace`,
`forwardTurnOffTrace`, and `reverseTurnOffTrace`.
Omit keys for mechanisms selected in `curves`.

Finalization returns `core_json_document` without persistence.
The native document retains the original bytes and declarations in `switch.metadata.legacy_mat_archive`.
Use `core_from_json_document` to load that document. Source ratings and unknown fields remain independent of user declarations.

## Bounded channel calculation

Send `POST /api/mat/calculate` with the complete finalization payload, an explicit `target`, and positive `current_a`.
The target must have exactly one selected conduction curve.
The API uses the existing core `get_resistance_at_current` method. It does not supply another interpolation engine.
Requests outside the selected branch domain fail without extrapolation.

The response contains current, voltage, resistance, current coverage, conditions, source SHA-256, citations, method, and units.
An explicit calibration extrapolation selection returns `supported-estimate` and retains its policy.
For a selected curve with 2 V at 10 A and 6 V at 20 A, 15 A gives 4 V and 4/15 Ω.
A request at 5 A is outside that curve's coverage.

## Supported formats and limits

The reader supports numeric MAT5 arrays, character arrays, scalar structs, and the reviewed literal Octave text subset.
The Octave subset supports `scalar struct`, `scalar`, `matrix`, `uint8 matrix`, and single `sq_string` values.
String lengths count UTF-8 bytes. Matrix dimensions and payload counts must match exactly.
Each nonempty two-dimensional numeric row must contain exactly the declared column count.
Three-dimensional Octave images use the original column-major storage order before conversion to the array descriptor.

MATLAB objects, function handles, cells, sparse or complex arrays, non-scalar structs, and HDF5 v7.3 are unsupported.
The reader never executes source expressions or starts MATLAB, Octave, or Qt.
Heat-sink and raw/grid workspaces support preview and exact archival preservation. They do not enter the SC core finalizer.
Heat-sink interface coefficients use K·m²/W. An explicit area is necessary to derive a K/W resistance.

| Budget | Maximum |
| --- | --- |
| Uploaded original bytes | 24 MiB |
| Aggregate source and expanded MAT bytes | 96 MiB |
| Elements per array | 8,000,000 |
| Aggregate array elements | 24,000,000 |
| Variables | 128 |
| Nested fields | 4,096 |
| Nesting depth | 16 |
| Bytes per string / aggregate strings | 1,000,000 / 2,000,000 |
| Inline preview array values | 200,000 total, 4,096 per array |
| Calibration comparison locations | 20,000 total |
| Explicit array blob bytes | 24 MiB |

Expanded MAT tags, shapes, types, fields, and nesting are checked before SciPy materializes arrays.
Individual dimensions, prefix products, and aggregate nested containers have bounds even when the final array has zero elements.
Finalization accepts at most 20,000 selected points per curve.
Integer curve samples outside exact float64 core coverage cause an explicit finalization error.
Resource errors do not return partial scientific results.
Invalid requests return HTTP 422 with a diagnostic. Boolean and nonfinite declarations are rejected.
