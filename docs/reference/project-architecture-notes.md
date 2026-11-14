# Architecture — transistordatabase

Last modified: 2026-02-19

## Core Layer Architecture

All new code goes into the `core/` layer:

```
transistordatabase/core/
├── models.py        — Domain entities (Transistor, Switch, Diode, + 12 data classes)
├── services.py      — ABC interfaces (ICalculationService, IExportService, etc.)
├── repository.py    — JSON persistence (JsonTransistorRepository, JsonTransistorLoader)
└── adapters.py      — Bidirectional legacy ↔ core converters (legacy_to_core, core_to_legacy_dicts)

transistordatabase/backend/
└── concrete_services.py  — Concrete implementations: Plotting, Calculation, Export,
                            Validation, Comparison + ConcreteServiceFactory

transistordatabase/frontend/
├── interfaces.py    — UI widget ABCs and controllers
└── pyqt5_impl.py    — PyQt5 concrete implementations

transistordatabase/gui_web/backend/
└── main.py          — FastAPI REST API (CRUD, export, plot data endpoints)

transistordatabase/gui/
└── api_client.py    — REST client for PyQt5 GUI → FastAPI communication
```

## Adapter Bridge Pattern

Legacy and core are connected via `core/adapters.py`:
- `legacy_to_core(legacy_transistor)` → core `Transistor`
- `core_to_legacy_dicts(core_transistor)` → `(transistor_args, switch_args, diode_args)` for legacy constructor
- `JsonTransistorLoader.load_from_json()` uses: JSON → numpy conversion → legacy Transistor → `legacy_to_core()`
- Export services use: core Transistor → `core_to_legacy_dicts()` → legacy Transistor → legacy export method
- `DatabaseManager.load_transistor_core()` returns core models directly

## Core Object Hierarchy (core/models.py)

```
Transistor
├── metadata: TransistorMetadata (name, type, manufacturer, housing, etc.)
├── electrical_ratings: ElectricalRatings (v_abs_max, i_abs_max, i_cont, t_j_max)
├── thermal_properties: ThermalProperties (housing_area, cooling_area, r_th_*)
├── switch: Switch
│   ├── channel_data: list[ChannelCharacteristics]
│   ├── e_on_data / e_off_data: list[SwitchingLossData]
│   ├── thermal_foster: FosterThermalModel
│   ├── gate_charge_curves: list[GateChargeCurve]
│   ├── soa: list[SOA]
│   ├── r_channel_temp: list[TemperatureDependResistance]
│   └── linearized_model: list[LinearizedModel]
├── diode: Diode
│   ├── channel_data: list[ChannelCharacteristics]
│   ├── e_rr_data: list[SwitchingLossData]
│   ├── thermal_foster: FosterThermalModel
│   └── linearized_model: list[LinearizedModel]
├── c_oss / c_iss / c_rss: list[VoltageDependentCapacitance]
└── c_oss_er / c_oss_tr: EffectiveOutputCapacitance
```

## PyQt5 GUI Architecture (refactored in v0.6.0)

The `gui/` directory was refactored from a 5,958-line god class into a modular mixin-based architecture:

```
transistordatabase/gui/
├── gui.py                  — 950 LOC (MainWindow shell + helper classes)
├── _widgets.py             — Reusable widgets (MatplotlibWidget, PopOutPlotWindow, ViewCurveWindow)
├── _utils.py               — Shared utilities (resource_path)
├── api_client.py           — REST client for FastAPI backend communication
└── mixins/                 — 8 mixin modules (5,957 LOC total)
    ├── utils_mixin.py      — Utilities (show_popup_message, browse_file, webbrowser_*)
    ├── settings_mixin.py   — Settings save/load/export
    ├── search_mixin.py     — Database search and filtering
    ├── creation_mixin.py   — Transistor creation/editing
    ├── curve_mixin.py      — Curve add/view/delete (switch/diode/capacitance)
    ├── export_mixin.py     — Export to simulation tools
    ├── comparison_mixin.py — Transistor comparison plots
    └── topology_mixin.py   — Topology calculator (buck/boost/buck-boost)
```

**MainWindow** inherits from all 8 mixins via multiple inheritance. Helper classes (`CurveCheckerWindow`, `InformationWindow`) remain in `gui.py` as they reference the MainWindow singleton.

## Legacy Root Modules (still functional, bridged to core via adapters)

- **`transistor.py`** — Monolithic Transistor class (used by export bridge)
- **`data_classes.py`** — Legacy dataclasses (ChannelData, SwitchEnergyData, etc.)
- **`database_manager.py`** — Legacy DatabaseManager with `load_transistor_core()` shim
- **`helper_functions.py`** — Validation, CSV parsing (headless-safe)
- **`gui_web/`** — Vue 3 + FastAPI web interface (wired to real services)

## Directory Structure

```
transistordatabase/
├── examples/              — Jupyter notebooks and example scripts
│   └── transistordatabase_performance_dashboard.ipynb
├── scripts/               — Utility scripts for migration, validation, testing
├── docs/                  — Documentation source files (MkDocs + Sphinx)
│   ├── getting-started/   — Installation and quickstart guides
│   ├── guide/             — User guide documentation
│   ├── api/               — Auto-generated API reference
│   └── development/       — Contributing, testing, release guides
├── tests/                 — Test suite (pytest)
├── transistordatabase/    — Main package
└── .github/workflows/     — GitHub Actions CI/CD
```

## Git Hooks

The repository includes hooks in `.git/hooks/`:

**`pre-commit`** — runs automatically before each commit:
1. **Updates timestamps** in CLAUDE.md, docs/PRD.md, README.md, docs/ARCHITECTURE.md when Python files change
2. **Validates documentation** files for broken links and formatting issues
3. **Regenerates docs/ARCHITECTURE.md** from codebase structure (class/function names + first-line docstrings)
4. **Prompts for changelog** entry when `transistordatabase/*.py` files are modified

**`pre-push`** — runs automatically before each push:
1. **Checks the version** in `setup.py` against the README version comment without changing either file
2. **Builds MkDocs** in a fresh directory under `/tmp`, outside the checkout
3. **Retains build output and logs** for inspection. Build failures remain non-blocking.

The maintained hook is `scripts/pre-push`. See [the installation instructions](../development/docs.md#pre-push-checks).

To bypass hooks (not recommended):
```bash
git commit --no-verify
```

## Testing Strategy

Test framework: pytest. MongoDB mocking via `mongomock`. Browser testing via `playwright`.

```bash
pytest tests/ -q                   # Run all tests (~283 tests)
pytest tests/test_core_services.py # Core backend services
pytest tests/test_repository.py    # Repository + adapter bridge
pytest tests/test_rest_api.py      # FastAPI endpoints (needs fastapi, httpx)
pytest tests/test_adapters.py      # Legacy ↔ core roundtrip
pytest tests/test_tdb_classes.py   # Legacy transistor classes
pytest tests/test_database_manager.py  # Legacy DB manager
pytest tests/test_gui_playwright.py -v --headed  # E2E browser tests (requires playwright)
```

### E2E Testing with Playwright

```bash
# Install Playwright
pip install playwright pytest-playwright
playwright install chromium

# Run E2E tests (headless)
pytest tests/test_gui_playwright.py -v

# Run with visible browser (debugging)
pytest tests/test_gui_playwright.py -v --headed --slowmo 500

# See tests/PLAYWRIGHT_TESTING.md for detailed guide
```
