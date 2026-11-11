# Import a three-column energy map

Open the [energy map panel](http://localhost:8002/index.html?view=energy-map). Select the XYZ input.

1. Select a UTF-8 file or paste the source text.
2. Declare the delimiter, header, column positions, and column units.
3. Declare the energy event, source, and common operating conditions.
4. Parse the source.
5. Inspect the matrix and each cell's original rows.
6. Download the response JSON, original source, source-row CSV, or cell CSV.

Use CSV, TSV, or whitespace delimiters with a decimal point. Each nonblank record must contain three columns.
Select current units A or mA, voltage units V or kV, and energy units J, mJ, µJ, or uJ.
The importer converts declared units to A, V, and J. It does not infer units from headers.

## Python and API example

The Python function accepts original bytes and explicit metadata. This example sends the same input to the running API.

```python
import base64
import hashlib
import json
from urllib.request import Request, urlopen

from transistordatabase.xyz_energy_import import import_xyz_energy_map

raw = b"9,600,90\n3,200,30\n0,600,12\n9,200,60\n0,200,0\n"
metadata = {
    "source_name": "fixture.csv",
    "input_kind": "file",
    "source": "Hand fixture",
    "source_kind": "manual",
    "energy_kind": "e_on",
    "conditions": {
        "t_j_c": None, "v_g_on_v": None, "v_g_off_v": None,
        "r_g_on_ohm": None, "r_g_off_ohm": None,
        "commutation_device": None, "commutation_source": None,
        "energy_convention": "Supplied event integral"
    },
    "options": {
        "delimiter": "csv", "decimal": ".", "has_header": False,
        "columns": {"current": 0, "voltage": 1, "energy": 2},
        "units": {"current": "A", "voltage": "V", "energy": "µJ"},
        "missing_energy_tokens": ["NA"]
    }
}
result = import_xyz_energy_map(raw, metadata)
assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()

payload = metadata | {"content_base64": base64.b64encode(raw).decode("ascii")}
request = Request(
    "http://localhost:8002/api/analysis/energy-map/xyz",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request) as response:
    result = json.load(response)

assert result["currents_a"] == [0, 3, 9]
assert result["voltages_v"] == [200, 600]
assert result["cells"][0][0]["energy_j"] == 0
assert result["cells"][1][1]["status"] == "missing"
assert result["cells"][1][1]["energy_j"] is None
```

The API returns ascending axes. Its matrix uses voltage rows and current columns.
The browser shows higher voltage rows above lower voltage rows.

## Cells preserve the source evidence

| Status | Meaning | Energy |
|---|---|---|
| available | Exactly one row supplies a known value | Recorded energy in J, including zero |
| missing | No source row supplies this coordinate | null |
| unknown | One row has blank energy or a declared missing token | null |
| ambiguous | More than one row supplies this coordinate | null |

Repeated identical values remain ambiguous. The importer does not average or sum duplicates.
The importer does not interpolate holes or create observations at zero.
Current, voltage, and known energy must be finite, nonnegative magnitudes.
Each fixed numeric condition requires an explicit finite value or null. Zero and null remain distinct.

The source SHA-256 covers the original file bytes, including a UTF-8 byte order mark and line endings.
For pasted text, it covers the UTF-8 paste bytes.
The request SHA-256 also binds the source digest to the declared metadata.
Row evidence retains line numbers, row text, tokens, and decimal coordinate identities.
Keep the original source download for exact byte replay.

JSON downloads preserve the successful HTTP response text.
CSV downloads distinguish null from zero and protect text cells against spreadsheet formulas.
A changed source or declaration invalidates the displayed result and its downloads.
The API reads supplied bytes and does not write device repository state.

## Resource limits

| Limit | Maximum |
|---|---|
| Raw source | 4,000,000 bytes |
| HTTP request body | 6,000,000 bytes |
| Source data rows | 100,000 |
| Distinct current coordinates | 256 |
| Distinct voltage coordinates | 128 |
| Current-axis count × voltage-axis count | 16,384 cells |
| Response evidence | 24,000,000 bytes |
| Numeric token length | 128 characters |
| Decimal digits | 80 |
| Decimal exponent magnitude | 400 |
| Declared missing-energy tokens | 20 |

The response evidence budget can reject a source before a row or cell limit is reached.
HTTP 413 reports a resource limit. HTTP 422 reports invalid metadata, syntax, encoding, or numeric values.
Distinct decimal coordinates that collapse to one binary floating-point value are rejected.
Nonzero values that underflow to zero are rejected.
