# Sprint: Buck DCM channel-loss feedback

Status: implementation complete; Sol review pending.

## Goal

Add scalar Buck DCM `loss-balanced` feedback using the selected switch and diode
channel curves. Preserve the existing Boost DCM solve, keep comparison and
sweeps CCM-only, and reject DCM feedback for Buck-Boost and Flyback.

## Model and evidence

In Buck DCM, inductor current begins at zero, rises during the switch interval,
falls to zero during the diode interval, then remains zero for the idle
interval. With output voltage reported as a positive magnitude, the inductor
voltages are `Vin - Vout - Vswitch(I)` during the switch interval and
`-Vout - Vdiode(I)` during the diode interval. The operation integrates each
current-dependent ramp using the measured channel curve and the corresponding
inductor voltage.

The output-charge condition includes both conducting intervals:

`L ∫ I/(Vin−Vout−Vswitch(I)) dI + L ∫ I/(Vout+Vdiode(I)) dI = (Pout/Vout)/fsw`.

The solver finds the peak current satisfying that condition. Input charge comes
from the switch interval alone. It independently integrates switch and diode
channel energy, then reports `Vin·Iin - Pout - Pswitch - Pdiode` and the
time-averaged signed inductor volt-second residual. Output ripple integrates
capacitor charge over the solved switch, diode, and idle trajectory.

The operation requires exact selected temperature/gate conditions, both curves
to include zero current and cover the per-device peak, and no extrapolation.
Parallel devices are assumed identical and share current evenly. It brackets
only peaks with positive switch-ramp voltage and rejects targets at or beyond
the DCM/CCM boundary.

## Scope

1. Add scalar Buck DCM feedback through the reusable Python operating-point
   calculation and FastAPI response. Return the solved input, output, inductor,
   and diode currents; separate switch and diode conduction losses; output
   ripple; and both balance residuals.
2. Preserve scalar Boost DCM behavior through the shared Buck/Boost ramp
   integrator. Keep Buck-Boost and Flyback DCM feedback unsupported.
3. Show the Buck DCM model and its results through the Vue workflow and include
   the request/result in the existing JSON export. Disable topology comparison
   for DCM and reject DCM sweep requests in the API.
4. Update the migration matrix with the supported DCM boundary and validation.

## Limits and protected surfaces

- This is an averaged channel-conduction model. Switching, reverse-recovery,
  gate-drive, passive, magnetic, parasitic, and temperature-feedback losses are
  excluded.
- Temperature is a one-pass estimate using the selected curve temperature;
  electrothermal closure is not implemented.
- Buck and Boost DCM are scalar-only. DCM topology comparison and all DCM
  sweeps remain unsupported. CCM support remains unchanged.
- Do not change the TDB schema, core models, adapters, ABCs, device records,
  Qt code, `ntbees2` interfaces/adapters/converters/evaluator, or `trial.json`.

## Acceptance

- An independent flat-drop derivation checks Buck duty, diode and idle
  fractions, peak/average/input/output/diode currents, RMS currents, separate
  switch/diode losses, power residual, and volt-second residual.
- A separate dense nonlinear integration checks current-ramp/charge solving,
  both residuals, separate losses, and output ripple. A derived flat-drop CCM
  boundary rejects equality while accepting a nearby DCM point.
- Python/API tests verify Buck success, unsupported Buck-Boost/Flyback DCM,
  DCM sweep rejection, and unchanged DCM comparison rejection.
- Vue and Chromium checks verify the Buck DCM model label, residual display,
  disabled comparison, and JSON request/result export.
- Run focused Python/API tests, Vue unit tests, Ruff, the Vite build, and the
  relevant Chromium workflow. Follow the repository's non-Qt test policy when
  choosing broader validation.

## Validation record

- `conda run -n ntbees2 python -m pytest --noconftest tests/test_topology_operating_point.py tests/test_topology_api.py -q`: 98 passed. `--noconftest` avoids the repository conftest's PyQt5 import; no Qt code or GUI was used.
- `conda run -n ntbees2 ruff check transistordatabase/topologies/operating_point.py transistordatabase/gui_web/backend/schemas.py transistordatabase/gui_web/backend/topology_service.py tests/test_topology_operating_point.py tests/test_topology_api.py`: passed.
- `node node_modules/vitest/vitest.mjs run src/tests/unit/TopologyWorkflow.test.js`: 17 passed. One expected chart-canvas warning occurs in the DOM test environment.
- `npm run build`: passed; Vite reports the existing JavaScript chunk above 500 kB.
- `npm run test:e2e -- src/tests/e2e/topology-migration.spec.js --grep "Buck DCM" --project=chromium`: 1 passed. FastAPI was started from the WSL `ntbees2` environment because the Windows Playwright web-server command could not resolve `conda`; the server was stopped after the check.
- A broad-suite attempt used `conda run -n ntbees2 python -m pytest --import-mode=importlib --noconftest tests/ -q --ignore-glob='test_gui_*.py'` intending to omit the Qt modules. On this Windows pytest invocation, that glob did not match the test paths, so the run unexpectedly executed legacy Qt tests. It reported 560 passed, 12 failed, 53 errors, 6 skipped, and 2 warnings in 56.26 seconds. Failures/errors included legacy `MainWindow` tests raising `NameError: app is not defined`, legacy web Playwright tests unable to connect to `localhost:5173`, and the DPT RAW async event-loop test. This broad result is not green and is not a clean non-Qt validation. This was an accidental validation-scope breach; no Qt code was changed and no interactive Qt GUI was launched.
- The plain full repository command was not run separately because the migration instruction excludes Qt tests. Sol review is pending.
