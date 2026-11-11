# Native-device electrothermal feedback

`device_electrothermal.build_device_loss_table` and
`analyze_device_electrothermal` accept a core transistor and an explicit request.
They perform no filesystem access. The existing `thermal.electrothermal` function
solves the derived recorded loss table. There is no second iteration algorithm.

The thermal workbench offers **Native device electrothermal feedback**.
Existing manual steady, transient, electrothermal, and screening operations keep
their contracts. The new operation models one switch junction at a fixed sink.
It does not infer diode heat, package coupling, multiplicity, or shared junctions.

## Source selection

`GET /api/thermal/device/inventory/{transistor_id}` returns source metadata,
indices, fixed conditions, temperatures, and supported current ranges.
It does not repeat full graphs. Select explicit channel, turn-on, and turn-off
records independently. Each active family requires at least two unique temperatures.
A selected current must remain inside every selected current curve.
Single-energy measurements support only their exact recorded current.

Each temperature family retains one exact declared non-temperature context.
Known values, zero, and unknown null values remain distinct.
The gate voltage fields keep their original meanings. Turn-on and turn-off may
have different gate conditions. Commutation hardware, inductances, testbench,
event, convention, and method must remain consistent within each family.
Unknown context remains unverified even when its declared null pattern matches.
The model does not parse comments to infer physical conditions.

Primary energy selectors require `v_g`, `v_g_off`, `r_g`, and `v_supply`.
Additional native context selectors are supported. All recorded fixed context
is checked, hashed, and retained, including fields the caller does not restate.
The browser sends the complete context from the selected inventory rows.
Duplicate source temperatures require explicit disambiguation, not averaging.

## Fixed operating point

`POST /api/thermal/device/analyze` accepts the following structure.
Replace the device ID and source indices with the selected inventory values.

```json
{
  "transistor_id": "Your device ID",
  "selected": true,
  "switching_policy": "held-current-pwm",
  "source": "Declared junction-to-fixed-sink path source",
  "current_a": 2,
  "duty": 1,
  "frequency_hz": 0,
  "sink_temperature_c": 25,
  "resistance_k_w": 2,
  "initial_temperature_c": 25,
  "tolerance_k": 1e-8,
  "max_iterations": 200,
  "relaxation": 1,
  "channel": {"indices": [0, 1], "conditions": {"v_g": 15}},
  "turn_on": null,
  "turn_off": null
}
```

Held current and frequency are nonnegative. Duty is in [0,1].
Conduction uses `I*V(I,T)*duty` only for positive current and duty.
Zero current gives exact zero conduction by construction.
The selected PWM policy defines one on/off pair per declared period only when
frequency is positive and duty is strictly between zero and one.
Endpoint duty and zero frequency contain no switching events under this policy.
This is an explicit estimate, not an event count inferred from measured data.
Active switching at zero current still requires recorded `E(0)` coverage.
An all-inactive input has no inferred device temperature domain.
Use explicit zero-heat steady analysis for that case.

The calculation evaluates current within each selected temperature slice first.
It then interpolates each active contribution across temperature.
The combined table uses the union of knots inside the common supported interval.
It does not extrapolate, clamp, select a nearest record, or bridge a missing family.
The existing thermal solver reports convergence, iteration limit, residual,
and source-domain exit. Only `accepted_solution` is an accepted junction result.
A last iterate is not presented as a physical solution.

## Provenance and precision

`table.source_identity.selected_dependency_sha256` identifies immutable selected
representations, indices, sample arrays, and recorded physics conditions.
It excludes load-time metadata such as a synthesized creation timestamp.
It is not a persisted file hash.

The repository adapter reads bounded bytes once, hashes those bytes, and passes
the same decoded snapshot to the existing loader.
`repository_source.sha256` identifies those original bytes.
An in-memory repository has no fabricated raw-file digest.
The optional `expected_file_sha256` rejects changed source bytes with HTTP 409.
The browser sends this value after an inventory read whenever available.

Record proof contains original graph sample indices, explicit bracket coordinates,
validated sorted positions, evaluated values, and full fixed context.
Long comments use bounded previews and a separate comment digest.
Graphs are hashed once and are not repeated per temperature or iteration.
Changing irrelevant annotations can change the raw-file hash without changing
the selected physical dependency digest.

Original HTTP scalar tokens are validated before float conversion.
Lossy duty rounding cannot convert an interior duty into an inactive endpoint.
Nonzero current, frequency, products, or interpolation values cannot silently
become zero. Unsupported persisted lexical or declared-dtype numeric loss is
rejected by this new reader. The global legacy loader remains unchanged.

The response retains the scientific request, derived table, source identities,
solver request, history, status, and accepted component breakdown.
For API replay, combine `request` with `transistor_id=table.device_name` and the
recorded raw-file hash when available. JSON retains the full response.
The CSV contains derived temperature/loss rows and scalar source identities.
Input, source, and selection changes invalidate browser results and exports.

## Bounds and qualification

The source snapshot is limited to 16 MiB before JSON parsing.
Each family permits 32 selected records. Each graph permits 10,000 points.
Selected evidence permits 10,00000 points. Inventory permits 512 source records.
Responses permit 4 MiB. Solver iteration limits retain the existing 10,000 bound.
These limits reject unsupported requests without changing source files.

For a synthetic conduction-only model, use 2 A, unit duty, a 25 C sink, and 2 K/W.
With `R(T)=1+0.01*(T-25)` ohm, the junction root is
`25+8/0.92 = 33.69565217 C`.
For the read-only Fuji_2MBI100XAA120-50 qualification at 50 A, duty 0.5,
1,000 Hz, 45 C sink and 0.1 K/W, the computed root is 48.76305056 C.
These are calculated model checks, not device-rating or hardware measurements.
