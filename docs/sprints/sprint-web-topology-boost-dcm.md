# Sprint: Boost DCM channel-loss feedback

Status: implementation complete; extends the merged CCM channel-feedback and CCM sweep workflows.

## Goal

Add one bounded DCM feedback case to the reusable topology workflow: scalar Boost DCM solved from exact-condition switch and diode channel curves. The solver must close the current ramps and output charge together, expose both power and volt-second residuals, and keep unsupported DCM cases explicit.

## Evidence and model

- The reusable operating-point module already calculates ideal Boost DCM duty, diode demagnetization fraction, zero-current idle fraction, current peak, output current, and ripple.
- The merged CCM loss-balanced operation provides the adjacent curve lookup, channel-loss reporting, FastAPI and Vue result contracts, and CCM sweep integration.
- Boost DCM has two nonzero intervals and a zero-current idle interval. For each trial peak, this solver integrates switch and diode current ramps using their measured current-dependent channel drops. It finds the peak that supplies the requested output charge per switching period, then independently reports input/output plus channel-loss power residual and averaged inductor volt-second residual.
- Curves must match the requested junction temperature and gate conditions and cover zero through the required peak current for both devices. No extrapolation, nearest-condition substitution, skipped points, or implicit fallback is allowed.

## Scope

1. Enable scalar Boost DCM `loss-balanced` channel feedback using exact switch and diode curves, including parallel-device sharing, branch/event current coverage, switch duty, diode demagnetization, idle fraction, RMS/peak/average currents, output ripple, and channel losses.
2. Use a bounded peak-current solve with a clear CCM-boundary failure. Reject unsupported Buck, Buck-Boost, and Flyback DCM feedback before attempting curve solving.
3. Expose power-balance and volt-second residuals, channel-only loss scope, solved-trajectory output ripple, and the DCM model limits in the Python result, FastAPI response, Vue UI, and JSON export.
4. Keep loss-balanced sweeps and topology comparison CCM-only. DCM comparison requests must fail at the service/API boundary because the scalar Boost DCM point cannot be ranked across unsupported topologies.
5. Update this plan and the master migration matrix with test evidence and the remaining full-migration gaps.

## Protected surfaces and limits

- This is an averaged Boost model with measured switch and diode channel drops. Switching, reverse recovery, gate drive, passive, magnetic, parasitic, and temperature-feedback losses are excluded.
- The current-dependent ramp integration closes both balances for the modeled channel curves. Residuals remain explicit so clients can inspect numerical closure; they do not establish a complete physical converter model.
- No topology beyond Boost DCM is added. CCM remains on the existing solve; all DCM sweeps remain unsupported. Explicit `estimate`, `conduction-only`, and `detailed` behavior remains available.
- Do not alter the TDB schema, core models, adapters, ABCs, device records, Qt code, `ntbees2` interfaces/adapters/converters/evaluator, or `trial.json`.

## Acceptance

- An independently derived flat-drop fixture checks duty, demagnetization, idle interval, requested output charge, input/output plus included channel-loss power balance, and volt-second balance.
- A separate dense nonlinear-ramp calculation checks capacitor charge ripple, and a rising switch-drop curve verifies the solver brackets below the positive-on-voltage cutoff.
- Boundary and domain tests cover a target at or beyond the CCM boundary, exact curve support at zero and peak/event currents, unsupported conditions, and unchanged source TDB curve arrays.
- Python tests confirm unsupported DCM topologies fail clearly and the scalar result retains SI units and channel-only scope.
- FastAPI returns the same Boost DCM fields and both residuals, and returns actionable 400 responses for unsupported cases and missing/out-of-range curve data.
- FastAPI rejects loss-balanced DCM comparison while preserving CCM comparison.
- Vue and browser checks cover a Boost DCM request, model label, both residuals, the CCM-only sweep hint, disabled topology comparison for DCM, and JSON export of request, result, and loss scope.
- Focused Python/API, Vue, browser, production-build, and Ruff results are recorded below and in the migration matrix.

## Validation record

- `conda run -n ntbees2 python -m pytest --noconftest tests/test_topology_operating_point.py tests/test_topology_api.py -q`: 93 passed. `--noconftest` avoids the repository conftest's PyQt5 import; Qt was not used. The `ntbees2` environment initially lacked `httpx`, so `httpx` and its `httpcore` dependency were installed there to run the API contract tests.
- `npm run test:run -- src/tests/unit/TopologyWorkflow.test.js`: 13 passed.
- Chromium browser workflow for Boost DCM request, result residuals, and JSON export: 1 passed. Playwright artifacts were directed to a separate temporary output path.
- `npm run build`: passed; Vite reported the existing 578.24 kB JavaScript chunk warning.
- `ruff check transistordatabase/topologies/operating_point.py transistordatabase/gui_web/backend/schemas.py transistordatabase/gui_web/backend/topology_service.py tests/test_topology_operating_point.py tests/test_topology_api.py`: passed.
- `git -c core.whitespace=cr-at-eol diff --check`: passed. The existing `core.autocrlf=true` checkout uses CRLF for `schemas.py`; the scoped Git whitespace setting recognizes those carriage returns without changing the user's Git configuration.

The independent flat-drop fixture verifies `duty + diode demagnetization + idle = 1`, analytically derived peak current and both balances to numerical tolerance. A dense independent nonlinear-ramp integration checks the output ripple; rising switch-drop data exercise the positive-on-voltage cutoff; analytic boundary tests reject equality while accepting a nearby DCM point. Additional tests cover current support gaps at zero and at the required peak, non-Boost DCM requests, and source curve immutability. API and browser results expose both residuals and the `channel_conduction_only` scope. DCM sweep and comparison requests remain deliberately unsupported.
