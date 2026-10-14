# Local web transient thermal comparison sprint

## Goal

Show the transient thermal data already stored for both switch and diode components in the active web advanced comparison. Plot recorded `graph_t_rthjc` samples as measured curves, and show a separately labeled Foster-model curve only when that component has valid, paired resistance and time-constant vectors.

## Boundaries

- Keep the change in the existing TDB backend comparison service and its tests. Preserve the current advanced-comparison response fields: `curves`, per-curve `time_data` and `z_th_data`, title, axes, and logarithmic time axis.
- Read transistor objects without mutation. This is display/comparison behavior only; it does not add persistence, edit curves, or change scientific models or shared interfaces.
- Do not create estimates from `r_th_total`, `r_th_cs`, browser calculations, or any other fallback. Do not change Qt, ntbees2, adapters/converters, or device records.
- Keep DPT measurement handling and the remaining desktop migration gaps open.

## Implementation and acceptance

- For each compared device, include switch and diode `graph_t_rthjc` arrays as separate measured curves, retaining the stored time and impedance samples.
- Add a separately named Foster-model curve only when both vectors are non-empty one-dimensional numeric arrays of equal length, and all values are finite and positive. A missing or invalid model does not suppress valid measured points.
- Name each curve with the device, component, and `measured` or `Foster model` role. Keep the time axis in seconds and impedance axis in K/W.
- Sample Foster curves on the existing 100-point logarithmic display grid from 1 µs to 10 s. The grid is for displaying the model and does not replace or resample recorded measurement points.
- Test both components with measured and modeled data, measured-only data with absent/unpaired model vectors, a diode model when the switch model is absent, and that comparison leaves all input arrays and vectors unchanged.

## Validation evidence

- Focused Python tests: PowerShell `$env:PYTHONDONTWRITEBYTECODE='1'; & 'D:\codex-envs\tdb-web-lifecycle\Scripts\python.exe' -m pytest -p no:cacheprovider tests/test_core_services.py::TestComparisonService::test_thermal_comparison_includes_measured_and_paired_models tests/test_core_services.py::TestComparisonService::test_thermal_comparison_keeps_measured_data_without_valid_model tests/test_core_services.py::TestComparisonService::test_thermal_comparison_includes_diode_model_without_switch_model tests/test_core_services.py::TestComparisonService::test_thermal_comparison_does_not_mutate_transistors -q` — **4 passed**.
- Full concrete-service module: PowerShell `$env:PYTHONDONTWRITEBYTECODE='1'; & 'D:\codex-envs\tdb-web-lifecycle\Scripts\python.exe' -m pytest -p no:cacheprovider tests/test_core_services.py -q` — **31 passed**.
- Broader Python suite: PowerShell `$env:PYTHONDONTWRITEBYTECODE='1'; & 'D:\codex-envs\tdb-web-lifecycle\Scripts\python.exe' -m pytest -p no:cacheprovider --import-mode=importlib --ignore=tests/test_gui_playwright.py tests -q` — **480 passed, 6 skipped, 62 setup errors**. The setup errors are in legacy Qt GUI tests because `pytest-qt`'s `qtbot` fixture is unavailable in this environment. `tests/test_gui_playwright.py` was excluded to avoid running browser tests against the existing Playwright output directories. The legacy suite created an untracked root `trial.json`; it was left untouched.
- Ruff: `ruff check --extend-ignore D102 transistordatabase/backend/concrete_services.py tests/test_core_services.py` — passed. The ignore covers existing missing-docstring findings on `setup_method` in this test module.
- Whitespace: `git -c core.whitespace=cr-at-eol diff --check` — passed.
- No frontend files or chart contract changed, so no Vue/browser run was needed for this backend-only update. MkDocs could not be run in the available Windows environment (`mkdocs` is not installed); the attempted WSL command was rejected by the execution policy.

## Remaining gaps

- The advanced comparison renders the stored transient thermal arrays and available Foster models; broader thermal parity, DPT waveform viewing, and comparison behavior outside this thermal display remain open.
- This sprint is one slice of full local web replacement and does not close the [topology migration plan](sprint-web-topology-migration.md).
