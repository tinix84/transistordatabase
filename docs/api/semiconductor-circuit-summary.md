# Physical circuit summary usage

`POST /api/converters/circuit-summary` returns physical loss ownership and steady
junction temperatures. The Python entry point is
`transistordatabase.circuit_summary.analyze_circuit_summary(request)`.
Use the [converter request contract](converter-losses.md) for `converter`.
The service evaluates that converter request once.

## Browser procedure

1. Open the Converter workbench.
2. Enable **Configure physical loss ownership and steady junction temperatures**.
3. Select **Stage members from current converter mapping**.
4. Enter the configuration ID and scenario ID.
5. Enter the ideal fixed sink temperature, source, and conditions.
6. Declare each member's complete junction layout.
7. Enter each junction's resistance, source, and conditions.
8. Select each component's die owner for each current direction.
9. Select each component's junction owner for each current direction.
10. Select **Calculate physical circuit summary**.
11. Download the circuit audit JSON or CSV.

Staging starts with unknown layout, ownership, and thermal resistance.
The independent two-device fixture provides a separate algebraic example.
It declares 1 A, 1 V, and 1 s without gate edges.
Each member dissipates 1 W. The circuit dissipates 2 W.
At 25 °C and 10 K/W per member, both junctions reach 35 °C [C].
This fixture does not establish device measurement accuracy.

## Python request example

This example wraps an existing accepted converter request.
The converter request must declare physical member `Q1` with role `position`.

```python
from transistordatabase.circuit_summary import analyze_circuit_summary

junction_id = "Q1.junction"
owners = {
    component: {
        direction: {"die": "channel", "junction_id": junction_id}
        for direction in ("positive", "negative", "zero")
    }
    for component in ("conduction", "turn_on", "turn_off")
}
request = {
    "configuration_id": "configuration-A",
    "scenario_id": "nominal-point",
    "converter": converter_request,
    "boundary": {
        "kind": "ideal-fixed-sink",
        "temperature_c": 25,
        "source": "Declared fixed boundary",
        "conditions": {"description": "Ideal fixed sink"},
    },
    "members": [{
        "id": "Q1",
        "role": "position",
        "layout_complete": True,
        "loss_owners": owners,
        "junctions": [{
            "id": junction_id,
            "rtheta_k_w": 10,
            "source": "Declared thermal estimate",
            "conditions": {"t_j_c": None},
        }],
    }],
    "acknowledge_known_only": False,
    "provenance": {"source": "Declared input", "source_sha256": None},
}
result = analyze_circuit_summary(request)
```

Choose ownership from physical evidence. The example assigns all heat to one
channel junction as an explicit assumption [A]. Negative current can belong to
the channel. A finite switching event at zero current uses the `zero` owner.
Declare `die` and `junction_id` independently. Either value can be `null`.
A shared channel and diode junction requires the same explicit junction ID.

Every physical converter member needs one matching `members` entry.
Member IDs and junction IDs must each be globally unique.
Multiplicity must equal the number of converter member IDs.
Do not multiply per-member results by multiplicity again.

## Result interpretation

`electrical.total_power_w` contains the full circuit total only when all required
components are available. `known_component_power_w` retains the known subtotal.
Each member retains component, channel, diode, junction, and omission evidence.

`heat_w` can remain known when thermal resistance is missing.
`tj_c` remains `null` without a sourced thermal path or complete heat coverage.
Missing resistance is not zero resistance.
An explicit sourced zero resistance represents an ideal path.
Unknown die ownership can coexist with known junction ownership.

`maximum.tj_c` requires complete member and junction coverage.
`maximum.junction_ids` retains all tied hottest junctions.
With `acknowledge_known_only=true`, `known_heat_tj_c` reports a separate partial
estimate. It excludes omitted heat and does not establish a full temperature.
`coverage` identifies incomplete layouts, unallocated heat, and omissions.

The method is `explicit-physical-ownership-fixed-sink-v1`.
The boundary is one ideal fixed sink with independent junction paths.
`iteration_status` is `not-required-constant-heat`.
The summary does not calculate thermal feedback, shared sink heating, or Foster
transients. Use the Thermal tools for their separately declared models.

## Exports, identity, and limits

`export_json` contains the complete core result as server-produced text.
`export_csv` contains junction rows with explicit `NULL` values.
CSV protects formula-like text. Each CSV row retains `input_sha256`.
Full model and boundary metadata remain in the linked JSON.

`input_identity.canonical_request_sha256` identifies the normalized request.
`converter_request_sha256` identifies the nested converter request.
`upstream_source_sha256` is nullable caller-declared evidence.
The service does not verify the upstream source bytes.
API defaults can make its canonical request differ from an unnormalized Python
request. Compare hashes only for identical normalized payloads.

The request limit is 4,000,000 bytes and 128 physical members.
Each member permits at most 128 junction declarations.
The complete response and exports must fit within 24,000,000 bytes.
Metadata expansion checks run before converter evaluation.
Resource excess returns HTTP 413. Invalid scientific input returns HTTP 422.
