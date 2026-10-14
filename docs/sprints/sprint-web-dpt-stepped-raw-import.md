# Stepped and multi-plot LTspice RAW import

Status: complete for bounded native LTspice `.raw` stepped and multi-plot DPT import. This sprint extends the existing one-plot, unstepped workflow; it does not close the broader local web GUI migration.

## Goal

Allow a user to import selected transient acquisitions from one stepped and/or multi-plot LTspice RAW file through the web DPT workflow. Preserve the current one-plot, unstepped import behavior.

## Scope and user decisions

- Accept native LTspice `.raw` uploads using the existing PyLTSpice dependency and temporary-file workflow.
- Preview the file's plots, plot names, trace names and step indexes. Require the user to select exactly one transient plot, then explicitly select one or more step indexes from that plot.
- Treat plot and step indexes as parser provenance only. Do not read, infer, display as conditions, or persist `.STEP` parameter values. The optional LTspice `.log` file is not required.
- Require explicit time, Vds and Id trace choices for the selected plot. Require separate turn-on and turn-off windows for every selected step; each selected step produces a distinct pair of raw DPT event acquisitions.
- Require the user to supply the current axis and DPT conditions. Conditions apply commonly to the selected acquisitions only after an explicit same-context confirmation. Never infer operating conditions from plot/step order, waveform values, filenames or `.STEP` values.
- Show the selected plot, step indexes, event windows, shared conditions and preview before the user confirms append. Invalidate previews when the file, plot, steps, traces, conditions or target device changes.
- Append only through the current locked legacy JSON raw-record writer, preserving prior raw records and unrelated device fields.

## Parsing and resource limits

- Keep the existing upload-byte, request-body, trace, point, preview, event-window, cumulative payload and legacy-record file limits.
- Add explicit finite caps for the number of plots and steps, and cap the sum of source samples selected for all acquisitions in one request before loading selected Vds/Id arrays or persisting JSON.
- Preflight every plot and step using metadata/axis data under a bounded read. Load value traces only for the selected transient plot and selected steps.
- A step boundary must be derived from the chosen RAW plot's time-axis data using a verified supported LTspice convention. Reject ambiguous, malformed, non-transient, missing-trace, or unbounded inputs with an actionable error. Do not guess step count or boundaries.
- Preserve source sample order and exact values; do not resample, interpolate, sort, or deduplicate waveform samples.
- Bind trace/plot/step selections, file identity, windows, conditions and current-axis values to preview/import validation so stale or altered requests cannot silently append different data.

## Protected compatibility boundary

This is a web import workflow only. Do not change the TDB schema, core models, core adapters, ABCs, ntbees2 interfaces/adapters/converters, persisted device records, converter outputs, Qt implementation, or unrelated import/export behavior. TDB remains complementary to ntbees2; this sprint makes no integration-completion or converter-parity claim. The existing legacy JSON writer remains necessary because the core adapter drops top-level raw records.

## Out of scope

- Other simulators, file formats, or simulator-generated metadata files.
- Automatic waveform/event detection, inferred `.STEP` values, condition splitting, waveform resampling, or energy derivation.
- Raw-record editing/deletion, general saved-energy-curve editing, Qt work, screenshot parity, or full desktop capability parity.

## Acceptance evidence

- API regressions cover unstepped compatibility, multiple plots, multiple selected steps, invalid plot/step/trace choices, exact per-step windows and values, missing logs, ambiguous or malformed boundaries, limits, preview tampering/staleness, additive persistence, and non-mutation on failure.
- Vue tests cover plot and step selection, per-step event windows/current axes, explicit common-condition confirmation, preview invalidation, confirmation, save and refresh/error behavior.
- A focused Chromium workflow verifies that the selected plot/steps and windows reach the API and that the resulting records are visible after save.
- The focused Python/API and Vue suites, the Chromium workflow, Vite production build, and Ruff checks pass. No Qt runtime or test is run.
- Inspect the final diff to confirm the protected ntbees2/TDB boundary above.

## Results

Implementation and initial independent Sol review completed on 2026-10-14. Sol's fresh code review found one actionable HTTP 500 error path; the regression fix was independently re-reviewed with no remaining actionable findings.

- Bounded preflight validates all plot declarations, binary payload bounds, ASCII rows, aggregate point/trace/plot limits, finite step caps, and event sample budgets before PyLTSpice loads values. Signal loading uses a temporary single-plot RAW containing only the selected steps. ASCII point indexes are rebased only in this temporary payload; source trace values and sample order are preserved.
- Step boundaries use repeated time-origin samples. The axis decoder follows PyLTSpice's `abs(time)` handling for native LTspice negative time markers. Ambiguous boundaries and complex transient plots are rejected; `.STEP` parameter values and `.log` files are not used.
- Stepped and multi-plot requests require a one-time server review token bound to the file SHA-256, target device and full plot/trace/step/window/current/condition request. Tokens expire after 15 minutes, the in-memory store is capped at 1,024 entries, and altered, expired, missing or replayed tokens are rejected before record mutation. A valid token is consumed once. The legacy single-plot, unstepped API payload remains compatible.
- Import appends through the existing mutation lock and legacy JSON writer. Regressions verify prior raw records and device fields are preserved and rejected imports leave the stored record unchanged. No TDB schema/core/adapter, ntbees2 surface, Qt implementation, device data or converter output changed.
- Validation passed: 38 focused Python/API tests, including regressions proving malformed grouped trace selections return HTTP 422 without mutating the record, 24 DPTManager Vue tests, one focused Chromium workflow, Vite production build, Ruff on the changed parser/schema/API tests, Ruff on `main.py` with the three pre-existing B905 findings ignored, and CRLF-aware `git diff --check`. The build retains the existing large-chunk warning. ESLint 10 could not run because the repository has no `eslint.config.*`. No Qt runtime or tests were run.
- Sol's final spec and standards reviews reported no remaining findings. The source-format scope remains native LTspice `.raw`; step splitting is limited to the verified repeated-origin convention. This completes only the scoped RAW import capability, not full desktop parity or the separate ntbees2 integration work.
