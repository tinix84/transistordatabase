# Web curve workflow sprint

## Goal

Make the existing browser curve lists load the fields returned by FastAPI and provide an in-app plot for each graph format the endpoint currently returns. Keep curve mutations on their existing routes and carry the exact selected device name through each request.

## Evidence and scope

- `GET /api/transistors/{transistor_id}/curves` returns `channel_data`, `e_on_data`, `e_off_data`, `e_rr_data`, `gate_charge_curves`, `soa`, and `capacitances`. The form previously read different keys, so edit-mode lists appeared empty.
- The browser now maps those response keys directly and plots the endpoint's two-row arrays: channel and SOA use V/A; switching energy uses A/J or Ω/J for `graph_r_e`; gate charge converts stored coulombs to nC for an nC/V plot; capacitance uses V/F.
- The checked-in `CREE_C3M0060065J.json` gate-charge graph stores x values around `1.4e-9` to `4.6e-8` C, while the legacy curve plot labels the axis nC. The browser converts those stored values by `1e9`; the component regression supplies C and asserts nC plot points.
- Selecting View opens a shared `CurvePlotter` panel; Close removes it. Records without a returned graph show a clear message. The unsupported CurveList Edit action was removed; adding and deleting still use their existing API methods and exact `metadata.name` identity.
- This sprint changes only the web form, curve list, focused frontend tests, and migration documentation. It does not change the core TDB model, ntbees2 interfaces, API schemas, persistence code, converters, or database records.

## Acceptance

- All curve categories returned by the current endpoint load into their corresponding lists.
- Plot points retain the two-row API array order, except gate-charge x values are converted from the stored C unit to nC for the displayed axis. Switching-energy `graph_i_e` and `graph_r_e` records are both supported.
- View and Close work in the form, and no unsupported Edit button is shown.
- Deterministic tests verify exact device identity and unchanged add/delete route arguments without reaching the JSON database.
- The broader migration stays open until remaining desktop capabilities receive their own evidence.

## Validation evidence

- Vitest: `src/tests/unit/TransistorForm.test.js` — 40 passed. Mocked responses cover switch/diode channel, Eon/Eoff/Err, gate-charge C→nC conversion, SOA, capacitance, plot closure, exact identity, and add/delete routes for all five curve kinds.
- Playwright Chromium: `src/tests/e2e/curve-workflow.spec.js` — 1 passed. Every API request is intercepted; the test views and closes a plotted curve, confirms the encoded exact device path, deletes the curve, and fails unmatched requests.
- Vite production build: passed with the existing warning that the minified JavaScript chunk is larger than 500 kB. Build output was directed to a new directory under `D:\tmp` because the default C: temporary volume is full.

## Remaining curve parity gaps

- Editing saved curves is unsupported and has no UI action.
- The existing curve endpoint does not return a plottable value for single-point switching-energy records or `graph_t_e` energy curves; those show the no-graph message.
- Transient thermal impedance curves and DPT measurement datasets are outside this endpoint and are not shown here.
- The browser view does not infer or alter legacy-only/unknown curve fields. Those fields remain subject to the existing raw-record preservation behavior of the add/delete routes.
