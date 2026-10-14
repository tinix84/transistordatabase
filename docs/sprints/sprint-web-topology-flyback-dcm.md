# Sprint: Flyback DCM channel-loss feedback

Status: implementation in review.

## Goal

Add scalar Flyback DCM `loss-balanced` channel feedback to the reusable Python,
FastAPI, and Vue topology workflow. Use the accepted ideal transformer model
with explicit primary-to-secondary ratio `turns_ratio_nps = Np/Ns`. Keep DCM
sweeps and cross-topology comparison unsupported.

## Model and equations

Magnetizing inductance and ramp current are primary-referred. Let `n = Np/Ns`,
and let `I` denote primary current from zero to the trial peak `Ipk`. With
`Ns` parallel switch devices and `Nd` parallel secondary diodes, the channel
curves are queried at `I/Ns` and `n·I/Nd`, respectively:

- Switch ramp voltage: `Vin − Vswitch(I/Ns)`.
- Primary-referred diode ramp voltage: `−n·(Vout + Vdiode(n·I/Nd))`.
- `ton = Lm∫dI/(Vin−Vswitch(I/Ns))`; `tdiode = Lm∫dI/[n·(Vout+Vdiode(n·I/Nd))]`.
- `D=ton/T`, `Ddiode=tdiode/T`, and `Didle=1−D−Ddiode`, where `T=1/fsw`.
- Solve `Ipk` from secondary output charge:
  `n·Lm∫I/[n·(Vout+Vdiode(n·I/Nd))]dI / T = (Pout/Vout)`.
- Input current is the switch-ramp charge per period:
  `Iin = Lm∫I/(Vin−Vswitch(I/Ns))dI / T`.
- Integrate channel energy over each interval. In particular, the diode loss
  is `n·Lm∫I·Vdiode/[n·(Vout+Vdiode)]dI / T`; divide total switch/diode
  losses by the corresponding parallel count for per-device reports.
- Independently report `Vin·Iin−Pout−Psw−Pdiode` and the signed primary
  volt-second residual `D·(Vin−Vsw_mean)−Ddiode·n·(Vout+Vd_mean)`.
- Output ripple integrates capacitor charge over the solved trajectory:
  `−Iout` during switch and idle intervals, and `n·Iprimary−Iout` during
  diode conduction.

## Scope

1. Extend the existing scalar DCM ramp integration to Flyback, including
   secondary-referred diode curve lookup/current/rms, equal sharing for
   parallel devices, both balance residuals, and solved-trajectory ripple.
2. Expose the result through the existing FastAPI contract. Keep SI units and
   the existing explicit turns-ratio request field.
3. Update Vue to label primary magnetizing current separately from secondary
   diode current and identify the Flyback DCM loss-balanced model. Disable
   sweeps and comparison for every loss-balanced DCM topology.
4. Update this sprint and the master topology migration matrix with focused
   Python/API/Vue/Chromium evidence.

## Limits and protected surfaces

- This is an ideal-transformer averaged channel-conduction model. Leakage,
  winding copper, core, clamp, switching, reverse-recovery, gate-drive,
  dead-time, passive, and parasitic losses are excluded.
- Exact selected junction-temperature and gate-conditioned curves are
  required. Curves must include zero and cover each device's branch-current
  peak. No nearest-condition selection or current extrapolation is allowed.
- Parallel devices are assumed identical with equal current sharing.
- Output ripple is an ideal capacitor-charge estimate; ESR and switching
  ripple are excluded. Temperature is a one-pass estimate at the selected
  curve temperature; electrothermal closure is not implemented.
- DCM feedback remains scalar-only. DCM sweeps and topology comparison stay
  unsupported; CCM and explicit estimate/conduction-only/detailed modes keep
  their current behavior.
- Do not change TDB schema, core models, adapters, ABCs, device records, Qt,
  `ntbees2` interfaces/adapters/converters/evaluator, or unrelated DPT files.

## Acceptance

- Independent flat-drop fixture checks primary peak/average/input currents,
  secondary average/peak/RMS diode current, duty/demagnetization/idle
  fractions, parallel losses, ripple, and both balance residuals.
- Independent dense nonlinear integration checks branch-current scaling,
  output charge, channel losses, both residuals, and capacitor ripple.
- Boundary, exact curve condition, zero/peak curve range, missing/invalid
  ratio, and source-curve immutability behavior are covered.
- FastAPI exposes the scalar fields and rejects missing/invalid ratio,
  unsupported conditions/ranges, DCM sweeps, and DCM comparison.
- Vue unit and Chromium workflows show distinct primary/secondary currents,
  disabled sweep/comparison, and exported request/results.
- Run focused non-Qt Python/API tests, Vue unit tests, Chromium workflow,
  Ruff, and the Vite production build. Do not run Qt.

## Validation record

- `conda run --no-capture-output -n ntbees2 python -m pytest --noconftest tests/test_topology_operating_point.py tests/test_topology_api.py -q` — 117 passed. The repository conftest was skipped to avoid its legacy Qt import; no Qt runtime or test was used.
- `node node_modules/vitest/vitest.mjs run src/tests/unit/TopologyWorkflow.test.js` — 18 passed. The existing Chart.js DOM-canvas context warning remains in the suite output.
- `npm run build` — passed; Vite reports the existing large-chunk advisory (605.87 kB JS bundle).
- `conda run --no-capture-output -n ntbees2 ruff check transistordatabase/topologies/operating_point.py transistordatabase/gui_web/backend/schemas.py transistordatabase/gui_web/backend/topology_service.py tests/test_topology_operating_point.py tests/test_topology_api.py` — passed.
- `npm run test:e2e -- src/tests/e2e/topology-migration.spec.js --grep "Flyback DCM" --project=chromium --reporter=line --output=C:\Users\tinix\AppData\Local\Temp\codex-flyback-dcm-artifacts` — 1 passed. Playwright needed `C:\Users\tinix\miniconda3\condabin` and `C:\Users\tinix\miniconda3\Scripts` prepended to the child `PATH` so its FastAPI web server could resolve `conda`.
- `npm ci` used the existing lockfile to populate this new worktree's `node_modules`; the lockfile was not edited. It reported 27 dependency audit findings, which were not changed as part of this topology slice.
