# Web DPT raw-window switching-energy derivation sprint

## Goal

Turn already imported, user-selected DPT turn-on and turn-off sample windows into
evidence-backed `E_on` and `E_off` current-energy curves in the local web workflow.
The calculation uses the exact samples stored by the native LTspice RAW import. It
does not detect event boundaries or turn the result into a standards-normalized
switching-energy value.

## Scope

- Add a reusable headless Python operation that integrates one or more selected
  stored event acquisitions and returns an existing TDB `graph_i_e` dataset.
- Add FastAPI preview and confirmed-save routes. Preview is read-only and reports
  the calculation, interval, conditions, provenance, and whether save will add or
  extend a curve. Save checks that both the selected raw records and reviewed inputs
  are unchanged since preview.
- Add DPTManager controls to choose Eon or Eoff, stored record/acquisition pairs,
  explicit current-axis values, and explicit Vds and Id polarity multipliers. Show
  the exact preview before the user confirms save.
- A curve contains between 2 and 50 event windows with unique positive `i_x`
  values. Each selected stored event acquisition can appear only once in that curve.
- An existing matching `graph_i_e` curve is extended only when its embedded DPT
  provenance confirms the same complete condition context, every stored graph
  sample matches the union of its provenance points, and the selected raw
  acquisitions have not already been used in that curve. The save uses the
  existing optimistic-token curve update path and binds the reviewed add/extend
  action, curve index/count, and target edit token from preview. A matching curve
  without verifiable provenance, a changed target, or multiple TDB lookup matches
  is rejected as ambiguous.

## Equations and conventions

For each event, apply the two explicit polarity multipliers selected by the user:

```text
p[n] = (Vds_sign × Vds[n]) × (Id_sign × Id[n])
E    = Σ 0.5 × (p[n] + p[n+1]) × (t[n+1] - t[n])
```

`t`, `Vds`, and `Id` are the stored sample coordinates and values in seconds,
volts, and amperes. Integration starts at the first stored event sample and ends at
the last stored event sample. Repeated time coordinates retain source order and
contribute a zero-width trapezoid. There is no threshold extension, resampling,
interpolation, automatic boundary detection, or implicit polarity correction.
Zero or negative signed energy is rejected; the operation never applies `abs()`.
The UI labels the result a raw-window measurement and shows the interval, sample
count, signs, formula, and energy before save.

Every event requires a positive, explicit `i_x` from the user. Current is not
estimated from the transient. Vds/Id arrays must contain finite, aligned
`[time, value]` pairs with nondecreasing time. Energy data is sorted by `i_x` for
storage, but the measured waveform arrays remain unchanged.

## TDB condition and persistence contract

- Store the results in the existing `SwitchingLossData.graph_i_e` field with
  `dataset_type="graph_i_e"`. The current topology lookup uses exact TDB conditions
  for `t_j`, `v_supply`, `v_g`, and applicable `r_g`, with its existing absolute
  comparison tolerance of `1e-9`. It rejects a missing or multiple matching graph.
- `v_g` always carries the turn-on gate voltage. Both gate voltages must be present;
  `v_g_off` is stored explicitly for modern Eoff pairing. The `r_g` field carries the
  resistance for the derived side: raw `r_g` for Eon and raw `r_g_off` for Eoff.
- All points in a curve must have exactly the same temperature, supply voltage,
  both gate voltages, both recorded gate resistances, load and commutation
  inductances, commutation device, and test bench. Missing required TDB conditions,
  non-scalar/ambiguous condition values, and incompatible contexts are rejected.
- The RAW importer and `RawMeasurementData` contract store load and commutation
  inductance in µH, while `SwitchingLossData` stores these conditions in H. Convert
  only when the raw record has `import_provenance.parser="PyLTSpice.RawRead"`.
  If a legacy source has either inductance without that verified importer marker,
  reject it with an actionable unit-provenance error. If both are absent, no
  conversion is needed.
- Store source record/acquisition indices, exact intervals, per-point `i_x`, Vds/Id
  signs, sample counts, formula, and source SHA-256 in the existing curve comment.
  Do not add a stored field or change the core schema.
- Extension parses every DPT provenance marker and compares the stored temperature,
  supply/gate conditions, event-side gate resistance, inductances, commutation
  device, and test bench exactly against the marker. It also requires the complete
  persisted graph to equal the union of provenance `(i_x, energy_j)` pairs. Existing
  graph current and energy values must be positive, unique, and finite; merged
  curves cannot exceed 50 points. The other-side gate resistance is verified only
  through the provenance because `SwitchingLossData` has no field for it.
- Appending a curve uses `_persist_curve_mutation` and the established raw legacy
  curve writer. Extending a verified matching curve uses the existing token-checked
  curve update helper. Both preserve every top-level `raw_measurement_data` record.

## Protected boundaries

Do not change core schema/models, adapters, ABCs, device records, Qt code/runtime/
tests, ntbees2 interfaces/adapters/converters/evaluator, `trial.json`, or unrelated
user files. The known ntbees2 converter issue that can lose valid curve data remains
explicitly out of scope. Keep the accepted local-web migration direction; do not
run Qt or depend on the desktop DPT derivation code.

## Acceptance and validation

- Python/API coverage verifies the trapezoid result against hand-computed samples;
  sign selection; exact source sample usage; non-finite, misaligned, decreasing,
  insufficient, duplicate-current, and nonpositive-energy rejection; required and
  ambiguous conditions; incompatible-context rejection; verified µH-to-H conversion
  and unknown-provenance rejection; and stale source/request rejection.
- Save coverage proves Eon/Eoff remain separate, an eligible same-context graph is
  extended safely, ambiguous existing lookup conditions are rejected, source raw
  records remain byte-value-equivalent, and failed preview/save leaves the record
  unchanged. It also covers duplicate source-acquisition rejection, the 50-point
  merged limit, exact stored-field/provenance and graph/provenance agreement, and
  stale source, request, target-edit, and add-to-extend race rejection.
- Vue coverage verifies required current/polarity controls, exact interval/sign/
  energy preview, stale-preview invalidation including a visible 409 conflict,
  explicit confirmation, and DPT data refresh after save. Focused Chromium tests
  verify both the request/preview/save path and stale-target conflict handling.
- Run only focused non-Qt Python/API, Vue, and browser checks required by this
  sprint. Build the local web frontend. Do not run broad or Qt suites.

## Implementation evidence

Implemented the reusable exact-sample calculation, preview/save APIs, DPTManager
review and confirmation UI, and focused API/Vue/Chromium coverage. Saving an appended
curve uses `_persist_curve_mutation`; adding measured points to a matching curve
requires this workflow's full-context provenance and uses the optimistic-token curve
update helper. Both paths preserve the stored raw measurement records.

Focused validation on 2026-10-14:

- `conda run -n ntbees2 python -m pytest --confcutdir=tests/web_dpt_energy tests/web_dpt_energy/test_api.py -q` — 31 passed.
- `npm run test:run -- src/tests/unit/DPTManager.test.js` — 20 passed.
- `npm run test:e2e -- src/tests/e2e/dpt-energy.spec.js --project=chromium` — 2 Chromium tests passed, including a stale-target 409 flow. The worktree lacked frontend packages, so `npm install --ignore-scripts --no-audit --no-fund` installed 252 locked packages; the lockfile was restored unchanged afterward. The first Playwright launch could not resolve the Conda PowerShell command in its Windows child shell. Starting the configured Vite and FastAPI services explicitly in the active shell let the focused E2E run reuse both services and pass.
- `npm run build` — passed. Vite reports its existing warning that the JavaScript chunk exceeds 500 kB after minification.
- `conda run -n ntbees2 ruff check transistordatabase/gui_web/backend/dpt_energy.py transistordatabase/gui_web/backend/main.py transistordatabase/gui_web/backend/schemas.py tests/web_dpt_energy/test_api.py --ignore B905` — passed. `main.py` contains three pre-existing B905 findings outside this sprint's diff.
- `git -c core.whitespace=cr-at-eol diff --check` — passed for the repository's CRLF files.

No broad or Qt suite was run. No engineering decision remains open under the
user-specified raw-window signed-integration and explicit-current contract. These
points are labeled raw-window measurements and are not standards-normalized.
