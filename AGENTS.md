# transistordatabase — Power semiconductor database and simulation export tool

Last modified: 2026-02-19

**Stack**: Python (PyQt5, FastAPI, numpy)

## Build & Test

```bash
pip install -e .                   # Install (requires Python >= 3.10)
pytest tests/ -q                   # Run all tests (~283 tests)
ruff check transistordatabase/     # Lint (line-length 88, ruff.toml)
python3 -c "from transistordatabase.core import Transistor"  # Core import check
```

## Key Documents

- [PRD](docs/PRD.md) — requirements, status, new modules, conventions, design decisions
- [Architecture](docs/ARCHITECTURE.md) — core/adapter pattern, object hierarchy, GUI architecture
- [Auto-Context](docs/auto-context.md) — generated summary

## Sprint Plans

Convention: `docs/sprints/sprint-*.md`

## Skills

Central pool: `/home/tinix/claude_wsl/agents_pool/` | Domain: **pe-expert**

## Task Protocol

1. **Run tests**: `pytest tests/ -q` before committing

## Important Notes

- `core/services.py` is THE single source of truth for ABCs
- Always use `transistor.metadata.name` (not `transistor.name`)
- Always use `transistor.switch.channel_data` (not `transistor.switch.channel`)
- PyQt5 isolated in `gui/` — `helper_functions.py` is headless-safe
- Branch: `main` (stable), `dev/merge-refactor` (refactoring)

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|

## Python Environment

**Do not create `.venv` directories.** Use conda instead:
- `conda activate ntbees2` (default)
- `conda activate social_env` (social media work)

Never run `python -m venv`, `uv venv`, or `virtualenv`.

## Agent skills

### Issue tracker

Issues and PRDs live in GitHub Issues for tinix84/transistordatabase.
See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels.
See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout: root CONTEXT.md and docs/adr/.
See `docs/agents/domain.md`.

## Web GUI migration

The accepted direction is a local web GUI replacing all desktop capabilities. Qt is a screenshot reference only; do not run, maintain, refactor, or depend on it for migration work. Scientific calculations belong in reusable Python modules exposed through FastAPI, with future MCP reuse in mind.

See [the accepted decision](docs/adr/0001-local-web-gui-replaces-qt.md) and [the topology migration plan](docs/sprints/sprint-web-topology-migration.md). Existing Qt-specific instructions describe legacy code and do not override this migration direction.
