# Sprint: scalar Boost CCM switching-loss feedback

Status: implementation complete; independent Sol review passed.

## Goal and boundary

Add a distinct `switching-loss-balanced` mode that solves one scalar Boost CCM operating point with switch and diode channel losses plus exact-condition switch Eon/Eoff graph_i_e losses in the duty/current power balance. Existing `detailed` retains its ideal-point switching-loss semantics, and `loss-balanced` remains channel-only.

The solve requires exact junction temperature, Boost blocking voltage, turn-on and turn-off gate voltages, and the respective gate resistances. Eon is evaluated at per-switch valley current and Eoff at per-switch peak current. No current extrapolation or ambiguous curve selection is allowed. Eon records without an explicit turn-off gate and legacy Eoff records without an explicit gate pair are rejected.

Reverse recovery is excluded because current Eon provenance does not establish whether recovery is already included; adding a separately recorded recovery energy could double count. When case temperature and positive junction-to-case/case-to-sink thermal data are present, switch and diode junction temperatures are calculated once from resulting per-device losses. Curves remain at their selected temperature; this is not electrothermal iteration.

Only scalar Boost CCM is supported. DCM, sweeps, topology comparison, Pareto ranking, and new electrothermal iteration remain out of scope. The checked-in Pareto candidates lack matched switching curves and usable thermal data, so their notebook, records, and channel-only Pareto method remain unchanged.

## Protected integration boundary

No TDB schema, device records, core model, adapter, repository, converter, Qt, `ntbees2` source/interface, GitHub Actions, Pareto implementation, notebook, or `trial.json` was changed. Playwright validation left local untracked report and test-result files in the worktree; they remain preserved and outside the intended source changes. Existing topology API, Vue result/export flow, and the migration evidence matrix are the only integration surfaces.

## Validation evidence

- Independent Python fixtures cover switching loss in the power residual, power closure, ideal estimates below the solved CCM boundary, narrow measured-current windows, one-pass thermal output, unavailable thermal output, exact voltage/gate/resistance conditions, legacy Eon/Eoff gate rejection, and roots bracketed above/below measured switching-current limits.
- FastAPI tests cover returned loss/provenance fields, power closure, required exact conditions, scalar-only sweep rejection, and cross-topology comparison rejection.
- Vue unit coverage checks the request conditions, switching loss display, and disabled sweep/comparison controls.
- Chromium coverage checks scalar request/result/export and the UI scope restrictions.
- Focused results: 141 tests across the Python operating-point/API files passed; 23 Vue topology workflow tests passed; one Chromium workflow passed; Vite production build passed with its existing >500 kB chunk advisory. Ruff passed on changed Python files. `git diff --check` passed with `core.whitespace=cr-at-eol` for this CRLF checkout. Independent Sol review passed after fixes for two reported solver edge cases.
- Full pytest was not run; Qt tests are outside this sprint's validation scope.

## Follow-up limits

The scalar model retains triangular CCM ripple and averaged channel-drop volt-second approximation. It does not prove the measurement commutation device matches the selected diode. Losses still exclude reverse recovery, gate drive, dead time, passive, magnetic, and parasitic effects. These are partial switching-inclusive losses, not a total converter loss or a switching-loss Pareto result.
