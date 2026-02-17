# Release Plan

Last updated: 2026-02-17

This document maps upcoming releases to sprint plans, PRD milestones, and architectural goals.
It supersedes the ad-hoc notes previously scattered across sprint files.

---

## Tag History

| Tag | Commit | Date | Description |
|-----|--------|------|-------------|
| `0.1.0`–`0.5.1` | (legacy) | 2021–2024 | Pre-refactor era (no `v` prefix) |
| `v0.6.0` | `ca75b02` | 2026-02-13 | Clean architecture refactor + archive feature merge |
| `v1.0.0` | `881a293` | 2026-02-14 | Stable release, GUI mixin refactor, quality gates |
| `v1.0.1` | (pending commit) | 2026-02-17 | Package distribution: pip extras, entry points, PyInstaller, GitHub Actions release |

> Tags `v*` are required to trigger the `release.yml` GitHub Actions workflow.
> Legacy tags `0.x.y` (without `v`) do not trigger it.

---

## v1.0.1 — Distribution & Packaging (current, ready to tag)

**Theme**: Make TDB installable in three self-contained flavours for end-users.

### Scope (implemented 2026-02-17)

| Area | Change |
|------|--------|
| `setup.py` | `extras_require`: `[webgui]`, `[pyqt]`, `[dev]`; removed PyQt5/mongomock/pytest from base install |
| Entry points | `tdb-backend`, `tdb-webgui`, `tdb-pyqt` CLI commands |
| `scripts/` | `tdb_backend.py`, `tdb_webgui.py`, `tdb_pyqt.py` |
| `main.py` | Static-file mount for Vue 3 `dist/` (in-source + PyInstaller `_MEIPASS`) |
| `packaging/` | `tdb_backend.spec`, `tdb_webgui.spec`, `tdb_pyqt.spec` (PyInstaller) |
| `.github/workflows/release.yml` | Builds wheel + 9 native executables (3 OS × 3 variants), attaches to GitHub Release |

### To release

```bash
# Commit the distribution changes (prompted by pre-commit hook for CHANGELOG entry)
git add setup.py transistordatabase/scripts/ transistordatabase/gui_web/backend/main.py \
        packaging/ .github/workflows/release.yml CHANGELOG.md docs/RELEASE_PLAN.md
git commit -m "v1.0.1: Package distribution — pip extras, CLI entry points, PyInstaller, GitHub Actions release"

# Tag → triggers release workflow
git tag -a v1.0.1 HEAD -m "Release v1.0.1: Package distribution"
git push origin main v1.0.1
```

---

## v1.1.0 — Christen-Biela Analytical Switching Loss Model

**Theme**: Replace the simplified `BielaModel` with a full physics-based half-bridge MOSFET
loss model from the IEEE TPEL 2019 paper (Christen / Biela).

**Sprint file**: `docs/SPRINT_CHRISTEN_BIELA.md`
**Target**: Q2 2026

### Why this release

The current `BielaModel` in `analytical_models.py` is a simplified approximation. The
Christen-Biela model covers:
- Iterative I_oss convergence with gm-dependent feedback
- Body-diode reverse recovery with three time constants (supplement-corrected equations)
- ZVS boundary calculation
- Automatic parameter extraction from `Transistor` objects (C(V) curves, channel data, diode fields)

This matters for SiC MOSFET design — E_on is ~10-20x larger than E_off and reverse recovery
dominates turn-on losses for SiC body diodes.

### Tasks (from SPRINT_CHRISTEN_BIELA.md)

Execution in 7 phases (CB.1-CB.11):

```
Phase 1 — parallel, no deps (~1 day):
  CB.1  Add q_rr/i_rr/t_rr/di_dt_rr fields to Diode model          [haiku]
  CB.2  HalfBridgeParams, TransconductanceParams, ReverseRecoveryParams, SwitchingEnergyResult dataclasses  [haiku]
  CB.3  calc_charge_equivalent_capacitance / calc_charge_stored / calc_device_capacitances  [haiku]
  CB.9  PACKAGE_INDUCTANCE dict + get_package_inductance()           [haiku]

Phase 2 — needs CB.2+CB.3 (~1 day):
  CB.4  fit_transconductance() from ChannelCharacteristics list      [sonnet]
  CB.6  extract_reverse_recovery_params() (eq 38-41 + errata)        [sonnet]

Phase 3 — needs CB.2+CB.3+CB.4 (~1 day):
  CB.5  ChristenBielaModel._solve_i_oss() + calc_turn_off_energy()  [sonnet]

Phase 4 — needs CB.5+CB.6 (~1 day):
  CB.7  ChristenBielaModel.calc_turn_on_energy() + _calc_reverse_recovery()  [sonnet]

Phase 5 — needs CB.1+CB.7+CB.9 (~1 day):
  CB.8  from_transistor() factory + calc_switching_loss_curve()      [sonnet]

Phase 6 — needs CB.8 (~0.5 day):
  CB.9b Remove old BielaModel + BielaModelParams                     [haiku]
  CB.10 Update __init__.py exports                                   [haiku]

Phase 7 — needs all (~1 day):
  CB.11 Rewrite tests/test_analytical_models.py (22+ test cases)    [sonnet]
```

### Acceptance criteria

- `ChristenBielaModel` importable from top-level: `from transistordatabase import ChristenBielaModel`
- `ChristenBielaModel.from_transistor(t, v_0=600)` works on any Transistor with C(V) + channel data
- E_T,off for C2M0080120D at 600 V / 20 A: within [5, 30] µJ (paper: 14.1 µJ)
- E_T,on for C2M0080120D at 600 V / 20 A: within [150, 500] µJ (paper: 274 µJ)
- E_on / E_off > 5 (matches SiC physics)
- `BielaModel` and `BielaModelParams` removed
- All 22+ CB tests pass; `GateChargeModel` and `IgbtModel` tests unchanged
- `ruff check` passes, no regressions in existing test suite

### CHANGELOG entry (draft)

```markdown
## [1.1.0] - 2026-Q2

### Added
- **ChristenBielaModel**: Full IEEE TPEL 2019 half-bridge MOSFET analytical switching loss model
  - Iterative I_oss solver with gm convergence (eq. 10 with supplement corrections)
  - Turn-off energy: voltage-rise + current-fall intervals (eqs. 13, 15-17, corrected)
  - Turn-on energy: current-rise + reverse recovery + voltage-fall intervals (eqs. 18-31, corrected)
  - Body-diode reverse recovery with tau_rr/tau_c/T_m model (eqs. 38-41, supplement corrected)
  - ZVS boundary: I_0,zvs calculation (eq. 14)
  - `from_transistor(t, v_0, t_j)` — one-call factory that extracts all parameters automatically
  - `calc_switching_loss_curve(v_0, currents)` — sweep over current range
- **New dataclasses**: `HalfBridgeParams`, `TransconductanceParams`,
  `ReverseRecoveryParams`, `SwitchingEnergyResult`
- **New functions**: `calc_charge_equivalent_capacitance`, `calc_charge_stored`,
  `calc_device_capacitances`, `fit_transconductance`, `extract_reverse_recovery_params`,
  `get_package_inductance`
- **Diode model**: `q_rr`, `i_rr`, `t_rr`, `di_dt_rr` optional fields for reverse recovery data

### Removed
- **BielaModel** and **BielaModelParams** (replaced by ChristenBielaModel)

### Migration
Replace:
    from transistordatabase import BielaModel, BielaModelParams
    params = BielaModelParams(c_oss=..., q_g=...)
    model = BielaModel(params)
    e_on = model.calc_e_on(v_dc, i_load)
With:
    from transistordatabase import ChristenBielaModel, HalfBridgeParams
    model = ChristenBielaModel.from_transistor(transistor, v_0=600)
    result = model.calc_switching_energy(HalfBridgeParams(v_0=600, i_0=20, ...))
    e_on = result["e_on"]
```

---

## v1.2.0 — Web GUI Parity with PyQt5

**Theme**: Bring the Vue 3 / FastAPI web interface to feature parity with the PyQt5 desktop GUI.

**Target**: Q3 2026

### Scope

| Feature | Current state | Target |
|---------|--------------|--------|
| Transistor list + search | ✅ REST API, partial Vue UI | Full search/filter UI matching PyQt5 |
| Curve add/view/delete | ✅ REST endpoints | Full form UI (switch, diode, capacitance, gate charge, SOA) |
| Export tools | ✅ REST endpoints | UI dropdowns for format selection + download |
| Comparison plots | ✅ REST `/api/comparison/advanced` | Interactive Chart.js / Plotly overlay plots |
| Topology calculator | ✅ REST endpoints | Web form with real-time Buck/Boost/BB results |
| File import (JSON/PLECS) | ✅ `/api/transistors/upload`, `/api/import/plecs` | Drag-and-drop UI |
| Settings | ✅ REST endpoints | Persistent per-browser settings panel |
| Test coverage | 84% | ≥ 90% (Vitest unit + Playwright E2E) |

### Non-functional targets

- Lighthouse performance score ≥ 90 on the main page
- Bundle size ≤ 500 KB gzipped
- All 31 REST API tests still passing
- Playwright E2E tests cover the 5 main user flows

---

## v1.3.0 — Data Ecosystem & Integrations

**Theme**: Expand data sources, persistence options, and sharing capabilities.

**Target**: Q4 2026

### Scope

| Feature | Details |
|---------|---------|
| MongoDB Atlas cloud | `JsonTransistorRepository` → `MongoTransistorRepository` behind the same ABC |
| Digikey live catalog | OAuth2 + REST call to Digikey API for live FOM ranking |
| Virtual datasheet v2 | PDF generation (WeasyPrint), improved HTML template with all curves |
| Switching-pair UI | Full Vue 3 UI for pair list, validation scores, CSV export |
| Batch operations | Multi-select in web UI → batch export / batch delete |
| PRD Phase 6 complete | All PRD milestones closed |

---

## v2.0.0 — Platform Evolution (long-term)

**Theme**: Web-first, community-shareable transistor database platform.

**Target**: 2027

### Vision

| Capability | Description |
|-----------|-------------|
| Multi-user | Authentication (OAuth2 / institutional SSO), per-user transistor libraries |
| Community DB | Shared public transistor entries with moderation workflow |
| REST API v2 | Versioned API with deprecation policy, OpenAPI spec auto-published |
| Advanced analytics | Loss breakdown dashboard, worst-case corner analysis |
| CI-driven data validation | GitHub Actions checks transistor JSON against schema on every PR |
| SDK | Official Python SDK (`import tdb; t = tdb.get("CoolMOS...")`) |

---

## PRD Phase Status

| Phase | Scope | Status | Release |
|-------|-------|--------|---------|
| Phase 0 | Core refactoring (models, services, interfaces) | ✅ Complete | v0.6.0 |
| Phase 1 | PLECS import, analytical models, Rg formula | ✅ Complete | v0.6.0 |
| Phase 2 | Waveform losses, topology analysis | ✅ Complete | v0.6.0 |
| Phase 3 | Legacy migration, GUI mixin refactor | ✅ Complete | v1.0.0 |
| Phase 4 | LTSpice DPT, Digikey catalog importer, docs | ✅ Complete | v1.0.0 |
| Phase 5 | Distribution packaging, release automation | ✅ Complete | **v1.0.1** |
| Phase 6 | Christen-Biela physics model | Planned | **v1.1.0** |
| Phase 7 | Web GUI parity | Planned | **v1.2.0** |
| Phase 8 | Data ecosystem & integrations | Planned | **v1.3.0** |
| Phase 9 | Platform evolution (community, multi-user) | Future | **v2.0.0** |

---

## Non-Functional Targets by Release

| Metric | v1.0.1 (now) | v1.1.0 | v1.2.0 | v1.3.0 |
|--------|-------------|--------|--------|--------|
| Test coverage (pytest) | ~85% | ≥87% | ≥90% | ≥90% |
| Ruff violations | 0 | 0 | 0 | 0 |
| Import time | <2s | <2s | <2s | <2s |
| JSON load | <100ms | <100ms | <100ms | <100ms |
| REST API tests | 31 pass | 31+ pass | 50+ pass | 60+ pass |
| Playwright E2E | 5 scenarios | 5 | 15 | 20 |
