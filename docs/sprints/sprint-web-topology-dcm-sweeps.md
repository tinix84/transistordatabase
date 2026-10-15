# Sprint: loss-balanced DCM topology sweeps

Status: implementation complete; Sol spec and standards reviews complete.

## Goal

Extend the existing bounded Python line and contour sweep operation to
loss-balanced DCM for Buck, Boost, inverting Buck-Boost, and Flyback. Each
requested coordinate must call the same scalar operating-point solver used by
the corresponding DCM result. FastAPI returns the generated arrays and Vue
displays and exports them with the request and curve provenance.

## Sweep contract

- Keep the existing SI axes and metrics, 2 to 200 points per axis, line and
  contour array shapes, and 2,500-point contour cap.
- Evaluate each line or contour coordinate independently through
  `calculate_topology_operating_point`; do not approximate or skip unsupported
  points. A failed point rejects the full sweep and reports its x/y coordinates
  and the scalar solver error.
- Reuse the DCM ramp and output-charge solve. The selected switch and diode
  curves must match the requested temperature and gate conditions exactly,
  include zero current, and cover the solved branch-current peaks. Do not
  extrapolate. Reject coordinates that reach or cross the topology's CCM
  boundary.
- Preserve the selected metric values and averaged inductor volt-second
  residual array. DCM sweeps also expose a same-shape power-balance residual
  array; the scalar DCM solver independently checks both residuals at every
  coordinate.
- Vue keeps line and contour controls available in loss-balanced DCM, renders
  the returned metric and residual information, keeps topology comparison
  disabled, and includes the complete request/result and curve-condition
  provenance in the JSON export.

## Model limits

- Losses include selected switch and diode channel conduction only. Switching,
  reverse-recovery, gate-drive, passive, magnetic, parasitic, and other losses
  are excluded. The power balance closes only for those included losses.
- DCM ramp integration uses interpolated exact-condition channel curves and
  numerical quadrature. The reported power and volt-second residuals are
  independent checks within the scalar solver tolerances; this is not a claim
  of exact nonlinear device-law closure outside those curve samples.
- Flyback retains the ideal transformer model and explicit primary-to-secondary
  turns ratio. Leakage, winding, core, and clamp effects remain excluded.
- Temperature metrics remain one-pass estimates at the selected curve
  temperature when the fixed case/sink inputs and positive thermal resistances
  support them. Missing thermal data remain unavailable; no temperature-driven
  curve iteration, shared heatsink model, or electrothermal closure is added.
- DCM is supported only strictly below the CCM boundary and inside exact
  channel-curve domains. Unsupported coordinates reject the entire sweep.
- Topology comparison remains unavailable for loss-balanced DCM. This sprint
  does not broaden the comparison model.
- Do not change persisted TDB records or schemas, core adapters, converters,
  Qt, or the separate `ntbees2` interfaces/adapters/converters. Do not claim
  that the known TDB/`ntbees2` converter integration bug is fixed.

## Acceptance and validation

- Python line and contour cases compare every topology's returned metric,
  power residual, and volt-second residual with direct scalar calls at the same
  coordinates.
- Unsupported DCM boundary points fail with coordinates. FastAPI cases cover
  DCM sweep serialization and coordinate errors.
- Vue unit coverage verifies a DCM contour request, visible residual tooltips,
  and disabled comparison. Chromium exercises a real FastAPI-backed DCM line
  sweep and JSON export; it uses the ideal estimate mode because the checkout's
  runtime device database is empty. Curve-backed loss-balanced DCM sweeps are
  covered by FastAPI tests with deterministic in-memory exact-condition curves
  and by the Vue workflow unit test.
- Focused Python/API validation passed: 126 tests using `ntbees2` and
  `--noconftest` (the repository conftest imports legacy PyQt5 code).
- Focused Vue validation passed: 20 `TopologyWorkflow` tests. The existing
  Chart.js canvas-context warning remains.
- Chromium validation passed: one real FastAPI-backed DCM line sweep and JSON
  export. Playwright output directories were kept intact.
- Vite production build passed (115 modules); Vite reported the existing
  bundle-size warning for the 625.74 kB JavaScript chunk.
- Ruff passed on the changed Python implementation and test files.
- `git diff --check` passed with `core.whitespace=cr-at-eol`, which treats the
  repository's CRLF line endings correctly.

## Completion rule

This sprint closes the integration gap for loss-balanced DCM line and contour
sweeps across the four supported topologies. It does not close total loss
coverage, nonlinear physical volt-second closure, electrothermal iteration,
DCM topology comparison, screenshot parity, or the broader desktop migration.
