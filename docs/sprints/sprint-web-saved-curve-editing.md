# Local web saved-curve editing sprint

## Goal

Allow users to edit saved curve entries from the active Vue transistor form for the curve categories already represented by `CurveEditor`: switch and diode channel curves; switch Eon/Eoff and diode Err; gate charge; SOA; and Coss/Ciss/Crss. Editing must patch the selected raw legacy entry while preserving its unknown fields and all other record content.

## Boundaries

- Changes stay in the existing TDB web application layer: FastAPI web routes and schemas, Vue components/services, focused tests, and migration documentation.
- Do not change Qt, the TDB core model or interfaces, ntbees2 interfaces/adapters/converters, scientific calculations, or checked-in/persisted device records.
- Use temporary copies of an existing fixture for backend tests and deterministic API mocks in browser tests. Do not edit repository database files.
- Do not add new curve categories or claim full parity for fields not exposed by the active editor.

## Implementation

The curve read endpoint now pairs each supported API projection with an opaque token derived from its corresponding raw JSON entry. The form keeps the list length and selected token when opening Edit. A `PUT /api/transistors/{transistor_id}/curves/{component}/{curve_type}/{index}` request carries that optimistic precondition and a field-limited curve patch.

The update route holds the existing cross-process mutation lock, checks the exact stored device name, category, index, list count, and raw-entry token, then updates only the selected raw dictionary. The existing legacy reader validates the candidate, and the existing atomic record writer publishes it. Stale list or entry preconditions fail before publication. Fields unknown to the active editor remain copied from the original raw entry.

The Vue form prefills the existing curve editor, supports cancel, displays API errors, and reloads the curve list only after a successful update. The UI routes the same component/type mapping used by deletion, including diode reverse-recovery curves and each capacitance list.

## Acceptance

- Edit is available for all ten supported category/component combinations listed in the goal.
- Graph and scalar switching-loss records can be edited in their existing format, including graph_i_e, graph_r_e, graph_t_e, and single-point values.
- The request identifies the exact selected device and curve and includes both expected list count and selected raw-entry token.
- Backend validation rejects unsupported patch fields, stale indexes/counts/tokens, exact-name mismatches, and invalid legacy candidates before file publication.
- A successful update changes only the targeted raw curve fields; unrelated curve entries, raw top-level/section fields, and unknown fields on the target remain unchanged.
- The browser prefills the editor, allows cancel, shows save failures without refreshing away the user's edit, and reloads curve data only after success.
- Tests use temporary fixture copies or deterministic mocks and never write repository records.

## Validation evidence

- Backend: with `PYTHONDONTWRITEBYTECODE=1`, `D:\codex-envs\tdb-web-lifecycle\Scripts\python.exe -m pytest -p no:cacheprovider --confcutdir=tests/web_saved_curve_edit tests/web_saved_curve_edit/test_api.py -q` — **23 passed**, with one existing Starlette/httpx deprecation warning. Coverage includes all ten curve categories, graph_i_e/graph_r_e/graph_t_e/single switching formats, raw-field preservation, stale preconditions, invalid candidates, exact record-name checks, rejection of non-finite patch values, and a byte-copied checked-in Infineon fixture containing four DPT `-Infinity` samples. GET and a valid curve edit succeed on the temporary copy, and the `raw_measurement_data` JSON value (including all four DPT sentinels) remains byte-identical apart from line-ending normalization.
- Vue unit tests: `npm run test:run -- src/tests/unit/CurveEditor.test.js src/tests/unit/TransistorForm.test.js` from `transistordatabase/gui_web` — **57 passed**. The editor tests verify prefill and raw patches across supported categories, graph_t_e edits with `t_j: null`, single-point switching values, changed-only parameter patches, and cancel; form tests verify exact device/category routes and update preconditions.
- Playwright Chromium: `npx playwright test src/tests/e2e/curve-workflow.spec.js --project=chromium --workers=1 --reporter=line --output=D:\tmp\saved-curve-graphte-e2e-final2-20261014` — **4 passed**. Vite and FastAPI were started on ports 5174 and 8002 from the local Node and isolated Python environments, then Playwright reused those servers. The browser tests verify prefill, edit A-to-B reset with the second curve's index and token, captured route after changing the Switch/Diode selector, Edit-to-Add reset, successful update and refresh, cancel, stale-update error with no refresh, exact encoded identity, view/close, delete, and no unmatched API requests. The configured Playwright webServer command could not resolve its `conda` command in the child shell on this Windows host.
- Production build: `npm run build -- --outDir D:\tmp\saved-curve-graphte-build-20261014` — **passed** (113 modules). Vite reports the existing chunk-size warning for the 545 kB minified JavaScript bundle.
- Documentation: `wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/d/repos/transistordatabase-sprint2 && python3 -m mkdocs build --strict --site-dir /tmp/saved-curve-edit-docs-20261014'` — MkDocs includes the new sprint page and navigation entry but exits in strict mode on **15 existing warnings**: 11 missing navigation targets, two old broken links in `FAQ.md`, and two griffe parameter warnings. No warning references the new sprint page or its links.
- `git -c core.whitespace=cr-at-eol diff --check` — passed with no whitespace errors while preserving the repository's existing CRLF working-tree convention.

## Remaining migration gaps

- Editing is limited to curve values and conditions represented in the active editor. Legacy-only and unknown fields are preserved by the raw patch but are not editable from the browser.
- Transient thermal-impedance curves and DPT measurement datasets remain outside the current curve endpoint and editor.
- A full migration parity review for the desktop curve workflow, including accepted reference screenshots, remains open.
- This sprint does not close the broader web replacement plan; the [topology migration plan](sprint-web-topology-migration.md) remains the tracking index.
