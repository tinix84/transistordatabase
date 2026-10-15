# Transient thermal curve workflow sprint

## Goal

Expose the already-stored switch and diode transient junction-to-case thermal impedance curves in the local web editor. Each component has one optional raw `thermal_foster.graph_t_rthjc` array, with row 1 containing time in seconds and row 2 containing Zth(j-c) in K/W.

## Scope and storage boundary

- The browser loads each component's raw graph unchanged, plots it with explicit SI axis labels, and supports add, edit, and delete.
- FastAPI reads and patches only the selected component's `thermal_foster.graph_t_rthjc` field. If the Foster object is absent or null, reads treat it as an empty curve without writing; an add creates the legacy reader's valid empty Foster defaults plus the selected graph. Existing Foster values and every other raw record key remain in place.
- Every mutation includes a hash of the graph the user loaded. A changed graph returns HTTP 409 and is never silently overwritten.
- Input must contain two equally sized rows with at least two finite samples, non-negative time and impedance, and strictly increasing time.
- No TDB schema, device records, core models/adapters/converters, Qt, `ntbees2`, or topology Pareto code is part of this sprint.

## Verification

Focused API validation: 13 tests passed under the `ntbees2` environment with `--noconftest`. It covers both switch and diode API CRUD, unchanged sample reads, stale-write rejection, invalid-input rejection without file changes, non-finite schema rejection, preservation of unrelated Foster/raw values, and absent/null Foster reads and initialization. The focused Vue unit suite passed 48/48 and Chromium workflow coverage passed 5/5 in the dependency-equipped validation copy. JavaScript `node --check`, Vue SFC parse/script/template compilation, and `git diff --check` passed. Coverage includes unchanged and zero-valued samples, SI axes, empty/error states, both components, mutation tokens, component target locking during edit/delete, scientific-notation tooltips for microsecond time, and small positive Zth values. The full repository pytest suite was not run because it includes prohibited Qt tests.

The complete migration remains open: total device losses, iterative electrothermal closure, screenshot parity, and broader desktop capability coverage are not established by curve editing. The curve editor stores measured graph samples; it does not derive or synthesize them.
