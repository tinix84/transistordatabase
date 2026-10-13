# Local web GUI replaces Qt

Accepted on 2026-10-02. Replace the Qt desktop GUI with a locally running web GUI covering all desktop capabilities; browser workflows may be reorganized. Keep FastAPI as the operation interface, put all scientific calculations in reusable Python modules, and allow a future MCP adapter to reuse those operations without browser automation or duplicated numerical logic.

Qt is a screenshot reference only: do not run, maintain, refactor, or depend on Qt to implement or validate the migration. Numerical validation uses Python modules, independent expected results and tests. Migrate complete workflows incrementally, starting with topology calculations; full capability coverage remains the overall acceptance target.

## Consequences

The browser owns interaction and visualization. Python owns scientific results, including sweeps, losses, thermal calculations and derived device requirements. Existing Qt implementation and Qt tests are not migration dependencies. Existing headless calculation modules may be reused after their assumptions and correctness are verified.

This replaces the earlier architecture-review proposal to deepen desktop topology mixins. Qt preservation or maintenance is not an alternative migration track. Preserve existing files; retirement or deletion requires separate explicit approval under the repository's deletion rules.

FastAPI transport details must stay outside the reusable calculation implementation. MCP implementation and shared-server deployment are deferred; neither is required for local web feature parity.

## Missing device data

Detailed calculations fail clearly when required device curves are missing. Estimates require an explicitly selected estimate mode with labeled results and assumptions; no automatic fallback is allowed. Enforce this rule in the reusable Python calculation module for consistent web and future MCP behavior.

## Unsupported operating regimes

Reject unsupported operating regimes with a clear error explaining the supported range or regime; do not silently extrapolate. Enforce validity in the reusable Python calculation module, including the validity limits of an explicitly selected estimate mode.
