# CLAUDE.md

Last modified: 2026-02-17

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transistor Database (TDB) is a Python package for managing power semiconductor transistor data (MOSFETs, SiC-MOSFETs, IGBTs, GaN) and exporting to simulation tools (GeckoCIRCUITS, PLECS, Simulink, MATLAB, LTSpice). Developed by LEA at University of Paderborn.

**Status**: Clean architecture is wired and operational. The `core/` package is the source of truth, connected to legacy via bidirectional adapters.

## Architecture

### Core layer (all new code goes here)
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

### Adapter Bridge Pattern
Legacy and core are connected via `core/adapters.py`:
- `legacy_to_core(legacy_transistor)` → core `Transistor`
- `core_to_legacy_dicts(core_transistor)` → `(transistor_args, switch_args, diode_args)` for legacy constructor
- `JsonTransistorLoader.load_from_json()` uses: JSON → numpy conversion → legacy Transistor → `legacy_to_core()`
- Export services use: core Transistor → `core_to_legacy_dicts()` → legacy Transistor → legacy export method
- `DatabaseManager.load_transistor_core()` returns core models directly

### Core Object Hierarchy (core/models.py)
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

### New Modules (v0.6.0)
- **`plecs_importer.py`** — Import transistors from PLECS XML semiconductor libraries
- **`analytical_models.py`** — Biela, gate charge, IGBT analytical switching loss models
- **`waveform_losses.py`** — Time-domain conduction/switching loss from current waveforms
- **`catalog_importer.py`** — CSV catalog import with FOM ranking (Rds*Qg)
- **`rg_formula.py`** — Gate resistance dependent switching energy interpolation
- **`topologies/`** — Power converter topology analyzers (Buck, Boost, Buck-Boost)
- **`utils/ltspice_dpt.py`** — LTspice Double Pulse Test netlist generation and analysis

### PyQt5 GUI Architecture (refactored in v0.6.0)
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

### Legacy root modules (still functional, bridged to core via adapters)
- **`transistor.py`** — Monolithic Transistor class (used by export bridge)
- **`data_classes.py`** — Legacy dataclasses (ChannelData, SwitchEnergyData, etc.)
- **`database_manager.py`** — Legacy DatabaseManager with `load_transistor_core()` shim
- **`helper_functions.py`** — Validation, CSV parsing (headless-safe)
- **`gui_web/`** — Vue 3 + FastAPI web interface (wired to real services)

## Project Organization

### Directory Structure
```
transistordatabase/
├── examples/              — Jupyter notebooks and example scripts
│   └── transistordatabase_performance_dashboard.ipynb
├── scripts/               — Utility scripts for migration, validation, testing
├── docs/                  — Documentation source files (MkDocs + Sphinx)
│   ├── archive/           — Historical reports and temporary analysis files
│   ├── getting-started/   — Installation and quickstart guides
│   ├── guide/             — User guide documentation
│   ├── api/               — Auto-generated API reference
│   └── architecture/      — Architecture documentation
├── tests/                 — Test suite (pytest)
├── transistordatabase/    — Main package
└── .github/workflows/     — GitHub Actions CI/CD
```

### File Organization Guidelines

**Examples Directory (`examples/`)**
- Jupyter notebooks demonstrating package features
- Interactive dashboards and tutorials
- Example workflows and use cases

**Scripts Directory (`scripts/`)**
- One-off migration scripts
- Validation and testing utilities
- Database maintenance tools
- NOT part of the installable package

**Documentation Archive (`docs/archive/`)**
- Historical validation reports
- Migration documentation
- Temporary analysis files
- Excluded from MkDocs build
- Patterns: `*_REPORT.*`, `*_SUMMARY.*`, `*_EXECUTION_*.*`

### Git Hooks

The repository includes a pre-commit hook (`.git/hooks/pre-commit`) that automatically:

1. **Updates timestamps** in CLAUDE.md, PRD.md, ARCHITECTURE.md when Python files change
2. **Validates documentation** files for broken links and formatting issues
3. **Regenerates docs/ARCHITECTURE.md** from codebase structure (class/function names + first-line docstrings)
4. **Prompts for changelog** entry when `transistordatabase/*.py` files are modified

The hook runs automatically before each commit. To bypass (not recommended):
```bash
git commit --no-verify
```

## Common Commands

### Install & Setup
```bash
pip install -e .
```
Requires Python >= 3.10.

### Testing
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
Test framework: pytest. MongoDB mocking via `mongomock`. Browser testing via `playwright`.

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

### Core import check
```bash
python3 -c "from transistordatabase.core import Transistor"
```

### Linting
```bash
ruff check transistordatabase/
```
- **Ruff**: line-length 88, target Python 3.10, PEP257 docstrings. Config in `ruff.toml`.
- **pycodestyle**: line-length 160, legacy linter. Config in `tox.ini`.

### Documentation

**Build with MkDocs (recommended):**
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve  # Live preview at http://localhost:8000
mkdocs build  # Build static site to site/
```

**Build with Sphinx (legacy):**
```bash
pip install sphinx sphinx-multiversion sphinx_rtd_theme sphinxcontrib-email
cd docs/ && make html
```

**Auto-deployment:**
- Documentation is automatically built and deployed to GitHub Pages on every push to `main`
- Workflow: `.github/workflows/docs.yml`
- Deployed site: `https://upb-lea.github.io/transistordatabase/`

**API Reference:**
- Auto-generated from docstrings using mkdocstrings
- Source files: `docs/api/**/*.md`
- Shows class/function names and first-line docstrings only

## Code Conventions

- **Naming**: functions/methods in `lower_snake_case`, classes in `CamelCase`
- **Type hints**: required on all parameters and return values. Use `from __future__ import annotations`.
- **Docstrings**: Sphinx/reST format with `:param:`, `:type:`, `:return:`, `:rtype:`. First line ends with period. Imperative mood.
- **Python version**: >= 3.10. Use `X | None` over `Optional[X]`, `list[X]` over `List[X]`.
- **Imports**: ABC interfaces from `core/services.py`. No duplicate interfaces in backend.
- **Prefer `pathlib`** over `os.path` for path operations.
- **Testing**: pytest with fixtures, no unittest.

## Key Design Decisions

- **core/services.py** is the single source of truth for all ABC interfaces
- **backend/concrete_services.py** implements those ABCs — never define new ABCs in backend
- `transistor.metadata.name` (not `transistor.name`) — all access goes through proper attributes
- `transistor.switch.channel_data` (not `transistor.switch.channel`)
- `transistor.electrical_ratings` (not `transistor.electrical`)
- `transistor.thermal_properties` (not `transistor.thermal`)
- PyQt5 is isolated in `helper_pdf.py` and `gui/` — helper_functions.py is headless-safe
- **GUI mixin pattern**: MainWindow uses multiple inheritance from 8 mixins instead of one monolithic class
- **Circular import avoidance**: Independent widgets in `_widgets.py`, deferred imports for dependent classes
- **Helper classes stay in gui.py**: `CurveCheckerWindow` and `InformationWindow` reference the MainWindow singleton

## Version Locations

When releasing, update version in all of:
- `setup.py` (line `version=`)
- `docs/conf.py`
- `transistordatabase/__init__.py` (`__version__`)
- `transistordatabase/core/__init__.py` (`__version__`)
- `CHANGELOG.md`

## Branch Strategy

- `main` — stable releases
- `dev/merge-refactor` — umbrella branch for core refactoring + archive merge
