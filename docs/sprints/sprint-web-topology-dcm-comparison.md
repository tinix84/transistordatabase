# Sprint: loss-balanced DCM topology comparison

Status: implementation complete; Sol spec and standards reviews passed.

## Goal

Enable the existing topology comparison workflow for loss-balanced DCM after
the scalar averaged solve is available for Buck, Boost, inverting Buck-Boost,
and Flyback. Evaluate every topology at the same SI boundary supplied by the
caller. Keep each topology's domain or curve failure in that topology's result
entry, and return the full scalar result for every supported point.

## Comparison contract

- Reuse `calculate_topology_operating_point` through the existing comparison
  service path. Do not change the boundary values to make a topology fit.
- Preserve the existing compare response: each topology includes its scalar
  model, calculations, losses, temperatures when available, performance,
  assumptions, curve conditions, and both DCM balance residuals. A failed
  topology keeps its own actionable error while other topologies are returned.
- Keep `efficiencies_percent`, `best_topology`, and `ranking_available` for
  clients that already consume the response. Add DCM-only `ranking_basis` so
  the new ranking is explicitly an included channel-conduction efficiency
  proxy; leave the existing CCM comparison response and behavior unchanged.
- In Vue, allow comparison in loss-balanced DCM, show the power-balance and
  averaged inductor volt-second residuals for successful topologies, and label
  the ranking as a channel-loss proxy rather than total converter efficiency.
- Make the Flyback turns ratio available from any selected topology for a
  loss-balanced DCM comparison and include it in that request without changing
  non-DCM comparison payloads.
- Keep estimate, conduction-only, detailed, and CCM comparison behavior
  unchanged. Keep the selected output voltage/current sign conventions,
  Flyback turns ratio, exact-condition curve requirements, no-extrapolation
  behavior, and per-topology validation messages.

## Model limits and compatibility

- Ranking includes selected switch and diode channel-conduction losses only.
  Switching, reverse-recovery, gate-drive, passive, magnetic, parasitic, and
  other losses are excluded; the ranking is not total converter efficiency.
- Each successful DCM result uses the existing ramp and output-charge solve
  and reports its power-balance and averaged volt-second residuals. DCM points
  must remain strictly below the CCM boundary and within exact channel-curve
  conditions and ranges.
- Comparison evaluates the requested boundary independently for each
  topology. A voltage or device-curve domain that does not support a topology
  remains a per-topology error; no equivalent boundary is inferred.
- This sprint does not add DCM sweep, Pareto, or electrothermal behavior.
  Temperature output remains the existing one-pass estimate where supported.
- Do not change Qt, TDB persisted schemas or device records, core adapters,
  converters, or any `ntbees2` interface, adapter, or converter. The existing
  TDB/`ntbees2` converter integration issue remains outside scope.

## Acceptance and validation

- FastAPI tests use deterministic in-memory exact-condition curves and a
  shared DCM request to verify successful scalar results, both residuals,
  channel-only ranking metadata, and an unsupported topology error beside
  valid results.
- Vue coverage verifies that a DCM request reaches the comparison API and
  renders residuals, the proxy label, and per-topology errors.
- Browser coverage verifies that a DCM comparison request from the default
  Buck selection includes the configured Flyback turns ratio and that
  per-topology missing-curve outcomes render. This checkout has no runtime DCM
  device records; the browser comparison response is a UI fixture and does
  not demonstrate curve-backed runtime comparison. The existing live FastAPI
  estimate-mode browser flow remains the live-server check.
- Focused Python/API validation passed: four comparison tests with
  `conda run -n ntbees2 python -m pytest --noconftest tests/test_topology_api.py -k comparison -q`.
- Focused Vue validation passed: all 21 `TopologyWorkflow` tests. The existing
  Chart.js canvas-context warnings remain in two sweep tests.
- Chromium validation passed: the DCM comparison UI fixture and the existing
  live FastAPI Boost estimate/invalidation flow. The DCM check verifies request
  submission and per-topology unavailable-curve rendering; it does not run
  curve-backed DCM through the browser because this checkout has no runtime DCM
  device records.
- Vite production build passed (115 modules); the existing large-chunk warning
  remains for the 626.07 kB JavaScript chunk.
- Ruff passed on the modified Python implementation and API test.
- `git diff --check` passed with `core.whitespace=cr-at-eol` for this checkout's
  CRLF files.
- Sol spec and standards review passed with no remaining actionable findings.

## Completion rule

This sprint closes the loss-balanced DCM topology-comparison integration gap
for the four scalar-supported converters. It does not close total device loss,
nonlinear volt-second closure, electrothermal iteration, screenshot parity, or
the broader desktop capability migration.
