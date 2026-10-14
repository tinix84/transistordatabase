# Sprint: CCM channel-loss feedback vertical slice

Status: implemented, review pending; based on the checked-in records at `origin/main` commit `0141c29` (PR #10 merged).

## Goal

Add an explicitly bounded, reusable Python/FastAPI/UI operating mode that feeds exact-condition switch and diode channel-curve losses back into a regulated CCM operating point. The first vertical slice covers Buck, Boost, inverting Buck-Boost, and ideal-transformer Flyback when both exact-condition channel curves span the required current range. It is an averaged triangular-ripple screening model. It is not a complete topology loss model, exact nonlinear switching-period solution, or electrothermal closure.

The mode closes its included channel-loss input/output power balance by solving duty. It separately reports the averaged inductor volt-second residual. Constant-drop fixtures close both equations independently; the checked-in nonlinear gate-conditioned API fixture leaves a measured 0.0038754 V volt-second residual, so nonlinear exact closure is not claimed. No device data are altered or synthesized.

## Evidence and supported boundary

- TDB channel curves exist for checked-in records, and existing topology equations already define the CCM currents, ripple, polarity, and ideal Flyback turns-ratio convention.
- The Pareto example uses a fixed 100 V to 200 V, 1 kW, 20 kHz Boost CCM boundary with checked-in CREE and Infineon devices. Its losses remain channel conduction only.
- The inspected records do not provide positive paired switch/diode case-to-sink plus junction-to-case thermal resistances, so temperature outputs are unavailable for those candidates. Even when a temperature estimate is calculable, the selected curve temperature is fixed and the solver does not iterate the curve data against temperature.
- At the Pareto boundary, matched switching-energy voltage conditions are absent for the compared candidates. Switching, reverse-recovery, gate-drive, passive, and magnetic losses therefore remain excluded.
- DCM feedback is unsupported. Missing curves, mismatched conditions, and current-range extrapolation fail explicitly.

## Scope

1. Add `loss-balanced` to the reusable topology calculation for CCM on the four supported topologies. Solve the included channel-loss power balance, refine triangular ripple from the selected switch-channel voltage, and report both power and inductor volt-second residuals.
2. Keep the model limits machine-readable and visible: channel-conduction-only, no switching/recovery/passive/magnetic losses, exact-condition curves only, no extrapolation, and no iterative electrothermal closure.
3. Expose the mode through FastAPI and the existing Vue workflow, including the two balance residuals, thermal-data unavailability, unsupported DCM behavior, and topology comparison where supported.
4. Re-run the fixed-boundary Pareto notebook with the new mode only where exact curve data support the point. Retain the explicit one-boundary, Boost-only and channel-loss-only scope in the notebook and output.
5. Update the master migration matrix with verified evidence and remaining gaps.

## Out of scope and protected boundary

- No Qt execution, implementation, or dependency.
- No persisted TDB schema, core-model, adapter, ABC, device-record, or `ntbees2` interface/adapter/converter changes. The FastAPI request schema may add the selected calculation mode; the separate ntbees2 goal remains protected.
- No claimed switching-loss or total-loss solution, reverse recovery, gate-drive loss, passive/magnetic loss, or broad thermal-network solution.
- No DCM loss-feedback solve, topology sweep feedback integration, temperature-dependent curve interpolation, or electrothermal iteration.
- No screenshot parity claims or broad desktop capability closure.
- No edits to `trial.json`, staging, commit, or push.

## Acceptance and validation

- Independently derived flat-channel CCM fixtures verify duty, current, included losses, power balance, and inductor volt-second balance for Boost, Buck, Buck-Boost, and ideal-transformer Flyback.
- Nonlinear checked-in curves return the power-balance result and expose the separately measured volt-second residual. No assertion labels both balances as closed unless the returned residual demonstrates it.
- Tests reject DCM, unavailable exact conditions, and current extrapolation; they verify missing/non-positive thermal data are reported without invented temperatures and that computed temperature remains one-pass at the selected curve temperature.
- FastAPI tests return the same bounded result, both residuals, loss scope, and clear unavailable-data errors. Vue tests show the residuals/model scope and prevent unsupported DCM comparison.
- The fixed-boundary Pareto notebook executes from checked-in records and presents the Pareto trade-off across loss, package volume, and dated component cost. It labels all exclusions and unavailable thermal values.
- Focused independent Python, API, UI, and notebook validation results are recorded in the master migration plan. No full topology completion claim is made.

## Completion rule

This sprint is complete only when all accepted checks pass and the migration matrix still identifies nonlinear volt-second mismatch, switching and other missing losses, unavailable thermal closure, DCM feedback, screenshot parity, and broader desktop capabilities as open work.
