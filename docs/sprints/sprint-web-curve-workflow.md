# Web curve workflow sprint

## Goal

Make the existing browser curve lists load the fields returned by FastAPI and provide an in-app plot for each graph format the endpoint currently returns. Keep curve mutations on their existing routes and carry the exact selected device name through each request.

## Evidence and scope

- `GET /api/transistors/{transistor_id}/curves` returns `channel_data`, `e_on_data`, `e_off_data`, `e_rr_data`, `gate_charge_curves`, `soa`, and `capacitances`. The form previously read different keys, so edit-mode lists appeared empty.
- The browser now maps those response keys directly and plots the endpoint's two-row arrays: channel and SOA use V/A; switching energy uses A/J or Ω/J for `graph_r_e`; gate charge converts stored coulombs to nC for an nC/V plot; capacitance uses V/F.
- The checked-in `CREE_C3M0060065J.json` gate-charge graph stores x values around `1.4e-9` to `4.6e-8` C, while the legacy curve plot labels the axis nC. The browser converts those stored values by `1e9`; the component regression supplies C and asserts nC plot points.
- Selecting View opens a shared `CurvePlotter` panel; Close removes it. Edit opens the selected saved curve in the existing editor and sends an optimistic update for only that raw legacy entry. The implementation and its boundaries are documented in the [saved-curve editing sprint](sprint-web-saved-curve-editing.md).
- Adding, editing, deleting, loading, and plotting carry the exact `metadata.name` identity. This workflow changes only the web application layer and does not change the core TDB model, ntbees2 interfaces, converters, or device records.

## Acceptance

- All curve categories returned by the current endpoint load into their corresponding lists.
- Plot points retain the two-row API array order, except gate-charge x values are converted from the stored C unit to nC for the displayed axis. Switching-energy `graph_i_e` and `graph_r_e` records are both supported.
- View, Close, Edit, Cancel, and Update work in the form. Editing sends the expected curve count and raw-entry token; stale edits are rejected without overwriting a changed curve.
- Deterministic backend and frontend tests verify exact device identity, targeted raw updates, and unchanged add/delete routes without modifying repository records.
- The broader migration stays open until remaining desktop capabilities receive their own evidence.

## Validation evidence

- Vitest and backend update evidence are maintained in the [saved-curve editing sprint](sprint-web-saved-curve-editing.md).
- Playwright Chromium: see the saved-curve sprint for edit-prefill, update request, successful refresh, view/close, delete, exact encoded identity, and unmatched-request evidence.
- Vite production build: passed with the existing warning that the minified JavaScript chunk is larger than 500 kB. Build output was directed to a new directory under `D:\tmp` because the default C: temporary volume is full.

## Remaining curve parity gaps

- The existing curve endpoint does not return a plottable value for single-point switching-energy records or `graph_t_e` energy curves; those show the no-graph message.
- Transient thermal impedance curves and DPT measurement datasets are outside this endpoint and are not shown here.
- Editing exposes the curve fields represented by this endpoint and editor. Unrepresented raw and unknown fields are retained by the backend patch path, but cannot be edited in the browser.
