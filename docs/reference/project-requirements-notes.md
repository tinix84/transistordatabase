# Product Requirements — transistordatabase

Last modified: 2026-02-19

## Project Overview

Transistor Database (TDB) is a Python package for managing power semiconductor transistor data (MOSFETs, SiC-MOSFETs, IGBTs, GaN) and exporting to simulation tools (GeckoCIRCUITS, PLECS, Simulink, MATLAB, LTSpice). Developed by LEA at University of Paderborn.

## Status

**v1.0.1** released. Repository cleaned (stale docs, empty dirs, venv removed). Clean architecture is wired and operational. The `core/` package is the source of truth, connected to legacy via bidirectional adapters.

## New Modules (v0.6.0)

- **`plecs_importer.py`** — Import transistors from PLECS XML semiconductor libraries
- **`analytical_models.py`** — Biela, gate charge, IGBT analytical switching loss models
- **`waveform_losses.py`** — Time-domain conduction/switching loss from current waveforms
- **`catalog_importer.py`** — CSV catalog import with FOM ranking (Rds*Qg)
- **`rg_formula.py`** — Gate resistance dependent switching energy interpolation
- **`topologies/`** — Power converter topology analyzers (Buck, Boost, Buck-Boost)
- **`utils/ltspice_dpt.py`** — LTspice Double Pulse Test netlist generation and analysis

## File Organization Guidelines

**Examples Directory (`examples/`)**
- Jupyter notebooks demonstrating package features
- Interactive dashboards and tutorials
- Example workflows and use cases

**Scripts Directory (`scripts/`)**
- One-off migration scripts
- Validation and testing utilities
- Database maintenance tools
- NOT part of the installable package

**Local Safe Zone (`.trash/`)**
- Removed but locally preserved files (not committed)
- Gitignored — invisible to git
- Use `mv file .trash/` instead of `rm` for any file deletions during cleanup

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

## Common Pitfalls

- Do not use `transistor.name` — always use `transistor.metadata.name`
- Do not use `transistor.switch.channel` — always use `transistor.switch.channel_data`
- Do not define new ABCs in `backend/` — all interfaces belong in `core/services.py`
- Do not import PyQt5 outside `gui/` or `helper_pdf.py` — keep `helper_functions.py` headless-safe
- Do not use `os.path` — prefer `pathlib`

## Available Skills

Central pool at `/home/tinix/claude_wsl/agents_pool/`:
- Domain: **pe-expert** — Power Electronics expert for transistor/magnetics analysis

## Sprint History

Sprint plans follow the convention `docs/sprints/sprint-*.md`. Default model: **sonnet**.
