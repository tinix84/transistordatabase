# Sprint: Inverting Buck-Boost DCM channel-loss feedback

Status: implementation complete; parent review pending.

## Goal

Add scalar inverting Buck-Boost DCM `loss-balanced` feedback to the reusable
Python/FastAPI topology workflow and Vue client. Reuse the bounded exact-curve
current-ramp solver already used by Buck and Boost. Preserve positive SI
magnitudes for voltage and current outputs, and identify the physical output
polarity as negative.

## Model and equations

The inductor starts each switching period at zero current, ramps up during the
switch interval, ramps down through the diode interval, then remains at zero
for the idle interval. The requested output voltage is its positive magnitude;
the physical Buck-Boost output is inverting, so its signed value is
`−Vout`. With per-branch measured channel drops and equal sharing across
parallel devices:

- Switch interval: `vL(I) = Vin − Vswitch(I)`.
- Diode interval: `vL(I) = −(Vout + Vdiode(I))`.
- `dt_on = L·dI/(Vin−Vswitch(I))` and
  `dt_diode = L·dI/(Vout+Vdiode(I))`.
- The peak current is solved from diode-interval output charge:
  `L∫ I/(Vout+Vdiode(I)) dI = (Pout/Vout)/fsw`.
- Average input current is switch-interval charge per period;
  `Iin = (L/T)∫ I/(Vin−Vswitch(I)) dI`.
- Average inductor current includes both ramps; average output current equals
  the diode-interval charge per period and the requested `Pout/Vout`.
- `D = ton/T`, diode fraction is `tdiode/T`, and idle fraction is
  `1−D−tdiode`.
- Report the independently integrated residuals
  `Vin·Iin−Pout−Pswitch−Pdiode` and
  `D·(Vin−Vswitch_mean) + Ddiode·(−Vout−Vdiode_mean)`.

All curve lookups use the existing exact temperature/gate selection and
in-range current interpolation policy. The curves must include zero and cover
the per-device peak; extrapolation is rejected. The requested operating point
must be strictly inside the supported DCM range. A nonpositive switch-ramp
voltage, missing curve coverage, failed output-charge solve, or CCM-boundary
point produces an actionable error.

## Scope

1. Extend the headless scalar DCM ramp calculation to inverting Buck-Boost,
   with separate input/output/inductor/diode currents, switch and diode losses,
   RMS currents, DCM intervals, output ripple, and both residuals.
2. Expose the result through the existing FastAPI calculation contract with SI
   values and `output_polarity=-1`; retain the existing `buck-boost` topology
   identifier and alias behavior.
3. Update Vue labels and explanatory text so the result clearly states the
   inverting polarity and reports scalar DCM channel-feedback metrics. Keep
   loss-balanced DCM sweeps and cross-topology comparison explicitly
   unsupported, with actionable API errors and clear UI behavior.
4. Add independent flat-drop and nonlinear numerical fixtures, API coverage,
   Vue workflow assertions, and a focused Chromium check where the repository
   browser environment permits.
5. Update the migration matrix with the verified support boundary and focused
   evidence. Do not claim total-loss, electrothermal, sweep, or comparison
   parity.

## Limits and protected surfaces

- This is an averaged channel-conduction model. Switching, reverse recovery,
  gate-drive, dead-time, passive, magnetic, parasitic, and temperature-feedback
  losses are excluded. `loss_scope` remains `channel_conduction_only`.
- Exact-condition switch and diode channel curves are required. No nearest
  condition, missing-data fallback, or extrapolation is allowed.
- Parallel devices are identical and share current evenly.
- Buck-Boost DCM is scalar-only. DCM loss-balanced sweeps and topology
  comparison remain unsupported. Explicit estimate, conduction-only, detailed,
  and CCM behaviors remain unchanged.
- No TDB schema, core model, adapter, ABC, device data, Qt, `ntbees2` interface,
  adapter, converter, evaluator, or unrelated DPT changes.

## Acceptance and validation

- An independently derived flat-drop case checks inverting polarity, duty,
  diode/idle fractions, peak, average inductor/input/output/diode currents,
  RMS currents, channel losses, power residual, and volt-second residual.
- A separate nonlinear curve fixture independently integrates the ramps and
  verifies output charge, channel losses, residuals, and solved-trajectory
  output ripple. Boundary, exact curve support, no-extrapolation, and
  source-curve immutability cases are covered.
- FastAPI tests verify the SI result, negative output polarity, channel-only
  scope, actionable missing/range/boundary errors, and DCM sweep/comparison
  rejection without changing CCM behavior.
- Vue and browser checks verify a Buck-Boost DCM request, inverting output
  label, current/loss/residual display, unsupported sweep/comparison behavior,
  and result export where supported by the existing workflow.
- Run only focused non-Qt Python/API tests, Vue tests, Ruff, production build,
  and the targeted Chromium workflow. Record exact commands and outcomes.

## Validation record

- `conda run -n ntbees2 python -m pytest --noconftest tests/test_topology_operating_point.py tests/test_topology_api.py -q`: 110 passed. `--noconftest` avoids the repository conftest's PyQt5 import; the Qt runtime was not used.
- `node node_modules/vitest/vitest.mjs run src/tests/unit/TopologyWorkflow.test.js`: 18 passed. The test environment emitted one existing Chart.js canvas-context warning.
- `npm run test:e2e -- src/tests/e2e/topology-migration.spec.js --grep "inverting Buck-Boost DCM" --project=chromium --reporter=line --output=C:\Users\tinix\AppData\Local\Temp\codex-topology-bb-dcm-artifacts`: 1 passed. Playwright's child shell did not inherit the PowerShell `conda` command on the first attempt; the successful rerun temporarily prepended `C:\Users\tinix\miniconda3\condabin` and `C:\Users\tinix\miniconda3\Scripts` to `PATH`. Existing untracked Playwright report/result directories were not used or changed.
- `npm run build`: passed. Vite emitted the existing large-chunk warning; the generated JavaScript bundle is 604.91 kB.
- `conda run -n ntbees2 ruff check transistordatabase/topologies/operating_point.py transistordatabase/gui_web/backend/topology_service.py tests/test_topology_operating_point.py tests/test_topology_api.py`: passed.
- `git -c safe.directory=C:/Users/tinix/.codex/worktrees/dpt-stepped-raw/transistordatabase -c core.whitespace=cr-at-eol diff --check`: passed.

The flat-drop fixture checks independently derived current intervals and the
inverting output sign. A separate dense integration over nonlinear channel
curves checks output-charge solving, current averages, losses, ripple, and both
residuals. A derived flat-drop CCM boundary is rejected while a nearby DCM
point is accepted. API/UI tests preserve scalar-only sweeps/comparison and the
channel-conduction loss scope. No TDB schema/core/adapter/ABC, device data,
Qt, ntbees2, or unrelated DPT surface was changed; broad or Qt tests were not
run.

