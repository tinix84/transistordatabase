# Sprint: CCM loss-balanced topology sweeps

Status: implementation in review; based on merged CCM channel-loss feedback PR #11 at `6a7541f`.

## Goal

Extend the existing bounded Python line and contour sweep workflow to the `loss-balanced` CCM mode. Each coordinate is solved independently through the same scalar operating-point operation, then the existing FastAPI and Vue workflows return and display the selected metric.

## Evidence and boundary

- PR #11 added the scalar `loss-balanced` mode for Buck, Boost, inverting Buck-Boost, and ideal-transformer Flyback, with exact-condition channel curves, power-balance duty solving, and a separately reported averaged inductor volt-second residual.
- The reusable Python sweep already generates SI-unit line/contour values, enforces 2,500 contour points, and fails at the first unsupported coordinate. The service and Vue currently reject or disable this mode.
- Channel-current curves must exactly match the requested gate and temperature conditions and span each point's required current range. No nearest-condition selection, extrapolation, or skipped cells are allowed.

## Scope

1. Enable loss-balanced sweeps over the existing axes and metrics, preserving SI units, line/contour shapes, 200 points per axis, and the 2,500-point contour limit.
2. Ensure each sweep value is the corresponding scalar result at the same coordinates. Fail fast with coordinates and the underlying actionable error if any point is unsupported or curve data do not cover it. Reject DCM clearly.
3. Expose the mode through the existing FastAPI endpoint and Vue line/contour controls, charts, and JSON export.
4. Include machine-readable loss scope, a same-shape per-point voltage-law residual array, and its caveat in sweep responses; show these in the browser and retain them in exports.
5. Update the master migration matrix with verified coverage and remaining limits.

## Model limits and protected surfaces

- This remains a channel-conduction-only loss screen. Switching, reverse-recovery, gate-drive, passive, and magnetic losses are excluded.
- Each point power-balances included channel losses and reports its own averaged inductor volt-second residual. Nonlinear curve cases may not close both equations exactly; no exact nonlinear closure or electrothermal iteration is claimed.
- Exact-condition curves only; no extrapolation. One unsupported sweep point rejects the request rather than producing a partial grid.
- No changes to Qt, the persisted TDB schema, core models, adapters, ABCs, device records, or the separate `ntbees2` interfaces/adapters/converters. Do not alter `trial.json`; do not stage, commit, or push as part of this sprint.

## Acceptance

- Independent Python line and contour regressions compare both each selected metric and its voltage-law residual to a direct scalar call at that coordinate.
- Tests preserve axis units, shape, point limits, fail-fast behavior, actionable coordinate errors for out-of-range curves, and clear DCM rejection.
- FastAPI returns supported sweeps with scope/caveat metadata and a client error with coordinate context for unsupported curve ranges.
- Vue submits line and contour sweeps in loss-balanced CCM, renders the returned result and per-point residuals, and JSON export retains the request and full sweep metadata.
- Focused Python/API, Vue/browser, production build, and Ruff checks pass; results are recorded in the master migration plan.

## Completion rule

This sprint closes only the integration gap for CCM channel-loss line and contour sweeps. DCM feedback, exact nonlinear volt-second closure, total loss coverage, electrothermal iteration, screenshot parity, and broader desktop migration remain open.
