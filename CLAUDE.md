# CLAUDE.md

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
- **`topologies/`** — Power converter topology analyzers (Bridgeless PFC, DAB, LLC, SRC-ZVS)
- **`utils/ltspice_dpt.py`** — LTspice Double Pulse Test netlist generation and analysis

### Legacy root modules (still functional, bridged to core via adapters)
- **`transistor.py`** — Monolithic Transistor class (used by export bridge)
- **`data_classes.py`** — Legacy dataclasses (ChannelData, SwitchEnergyData, etc.)
- **`database_manager.py`** — Legacy DatabaseManager with `load_transistor_core()` shim
- **`helper_functions.py`** — Validation, CSV parsing (headless-safe)
- **`gui/`** — PyQt5 desktop GUI with `api_client.py` for REST communication
- **`gui_web/`** — Vue 3 + FastAPI web interface (wired to real services)

## Common Commands

### Install & Setup
```bash
pip install -e .
```
Requires Python >= 3.10.

### Testing
```bash
pytest tests/ -q                   # Run all tests (~292 tests)
pytest tests/test_core_services.py # Core backend services
pytest tests/test_repository.py    # Repository + adapter bridge
pytest tests/test_rest_api.py      # FastAPI endpoints (needs fastapi, httpx)
pytest tests/test_adapters.py      # Legacy ↔ core roundtrip
pytest tests/test_tdb_classes.py   # Legacy transistor classes
pytest tests/test_database_manager.py  # Legacy DB manager
```
Test framework: pytest. MongoDB mocking via `mongomock`.

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
```bash
pip install sphinx sphinx-multiversion sphinx_rtd_theme sphinxcontrib-email
cd docs/ && make html
```

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
