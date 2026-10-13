# Web database lifecycle sprint

## Goal

Complete the first reviewable web slice for creating, editing, and deleting database records from search results. The slice uses the existing FastAPI CRUD routes and keeps every browser test isolated from the repository's JSON database.

## Evidence and gap

- `SearchDatabase.vue` previously offered export, comparison, and topology actions in table and card results, but no edit or delete action.
- `App.vue` already had an edit handler, but search results did not emit an event to reach it. A delete handler existed without a visible search action or user feedback.
- The CRUD Playwright file used conditional checks and placeholder assertions such as `expect(hasDeleteButton || true)`, so a missing action could pass.
- The FastAPI update route replaced the loaded transistor with a bare object made from the general form. Since the form does not submit curve data, this could discard existing switch, diode, and capacitance data during an ordinary metadata edit.

## Vertical slice

1. Add edit and delete actions to each search table row and card.
2. Open the existing form from either result presentation. Show update and delete success or error feedback in the page.
3. Require browser confirmation before calling the existing delete endpoint.
4. Patch only metadata, ratings, and basic thermal fields in the existing legacy JSON record. Preserve all legacy-only values, curves, measured data, and unknown keys; do not serialize the partial form through the core-to-legacy adapter. Keep device names read-only during edits because pair references are keyed independently.
5. Use the exact device name in the form and query-name CRUD endpoints. Encode names as query values so spaces and URL-reserved characters cannot be mistaken for path syntax. Reject `/` and `\` identities explicitly because the JSON repository stores each identity as one filename.
6. Replace permissive CRUD E2E cases with deterministic Playwright route mocks that assert the UI actions and exact query name, method, payload, confirmation, and refresh behavior. Fail unmatched API requests; mocked records remain in memory.
7. Add backend API regressions that patch temporary raw JSON records containing legacy-only fields, switch/diode curves, measured data, and unknown keys. Validate the patched copy through the legacy constructor/catalog rules before writing, and prove invalid updates preserve the file bytes. Replace updates atomically and reject duplicate create names with HTTP 409 without overwriting a record, including when creates race. No real database records are written by these tests.

## Acceptance criteria

- Table and card results expose working Edit and Delete actions.
- Edit loads the selected device in the existing form, performs `PUT /api/transistor-record?name=...` with the exact identity, returns to search after success, and displays an error while keeping the form open after failure.
- Delete asks for confirmation, performs `DELETE /api/transistor-record?name=...` only when confirmed, refreshes the search results, and displays success or failure feedback.
- General edits change every form-owned field and preserve all other JSON data. Renaming is rejected by the API and disabled in the edit form until pair-reference migration is designed.
- Updates that violate legacy manufacturer or housing catalogs fail before writing, and successful updates reload through `JsonTransistorLoader`. Updates replace the record atomically. Duplicate creates return 409 without overwriting an existing populated record, including under concurrent requests.
- Names with spaces and URL-reserved characters remain exact through browser update/delete. Names containing `/` or `\` receive a clear rejection before create, edit, or delete.
- Browser tests mock every expected API request, fail on unmatched API calls, and mutate only fixture objects. Backend regressions use temporary raw JSON files. Repository construction is lazy at web-backend import, so test collection does not create the real database directories. The CI workflow explicitly runs the isolated backend regression module.
- No Qt code or ntbees2 interfaces, adapters, converters, or persisted device data are changed.

## Validation record

- `npx playwright test --config playwright.lifecycle.config.js src/tests/e2e/crud-workflows.spec.js --workers=1` — final rerun passed 9/9 Chromium tests in 10.3 seconds, including duplicate create rejection with the populated fixture unchanged. An earlier run had two transient 30-second locator timeouts; both cases passed individually and passed again in the full rerun. The test config starts only Vite; the spec intercepts every expected `/api/` request, fails unmatched API calls, and mutates fixture objects in memory. Coverage includes exact space/`#`/`?` identity, a similarly named device, and preserving `TO263` in the form.
- `npm run build` — passed. Vite reported its existing warning that one minified JavaScript chunk is larger than 500 kB.
- `npm run test:run -- src/tests/unit/App.test.js -t "Transistor Operations"` — 5 passed, 24 skipped.
- `npm run test:run -- src/tests/unit/api.test.js src/tests/unit/App.test.js -t "preserves spaced|path-separator"` — 2 focused tests passed; 53 were skipped by the filter.
- `D:\codex-envs\tdb-web-lifecycle\Scripts\python.exe -m pytest tests/test_web_database_lifecycle_api.py -q` — 11 passed. The external Windows venv avoids adding Python environment files to this repository. Regressions copy a checked-in valid CREE fixture into pytest's temporary directory, verify in-memory JSON upload conversion, reload successful edits and creates through `JsonTransistorLoader`, check invalid catalog updates preserve exact bytes, and verify concurrent duplicate creates publish one complete record without overwrite.
- Update writes flush a validated sibling temporary file before atomic replacement. Creates serialize a sibling temporary file and atomically publish it without replacing an existing path (`os.rename` on Windows; Linux `renameat2(RENAME_NOREPLACE)`). A persistent `.web-record-mutations.lock` coordinates create, edit, delete, curve changes, and imports across threads and server workers sharing the database. Blocking lock acquisition runs in worker threads so it does not freeze the FastAPI event loop; uploaded files are read and parsed before locking, and the lock covers repository writes. Repository construction is lazy at web-backend import. If a staging write or publish fails, its `.tmp` sibling is preserved for inspection; the backend does not automatically delete staging files. Atomic create publication requires Windows or Linux with `renameat2`; unsupported systems fail without replacing an existing record.
- A combined related Vitest run found existing stale UI expectations in `SearchDatabase.test.js`, `TransistorForm.test.js`, and App navigation assertions. The new App confirmation handling is covered by the focused passing App operation group and the browser tests.

## Migration status

### Review follow-up: raw record preservation and exact names

- The previous aggregate save was removed from general web updates because the adapter cannot represent all legacy JSON fields. The endpoint now validates the form payload and patches only its 12 general fields directly into the existing JSON object.
- Query-name CRUD routes carry the complete device name for the web UI. The UI no longer replaces spaces or slashes with underscores. Create, edit, get, and delete reject slash and backslash identities because repository records use one filename per identity; `?` and `#` are query data and cannot redirect an operation to a similarly named record.
- Backend regressions seed temporary files with distinct old metadata, `r_g_int`, recommended gate resistances, `graph_v_ecoss`, nested switch/diode curves, raw measurements, and unknown keys. They assert all form-owned fields update while every non-form field remains equal, and exact-name update/delete do not affect a similar record.
- Before writing an edit, the web layer runs a deep copy of the merged raw JSON through `_json_dict_to_legacy_transistor`, preserving the stored arrays/unknown values while applying the legacy reader's required catalog validation. Updates use atomic replacement after validation. Creates return 409 for an existing exact-name JSON file before model construction and atomically publish without replacing an existing file. The persistent cross-process mutation lock also covers existing web curve-save and import routes, preventing a stale curve write from racing a general edit.
- The housing selector sends canonical catalog values (`TO247`, `TO220`, `TO263`, `TO92`) while showing familiar labels. Existing unlisted housing values remain selectable during edits.
- Browser regressions use a `TO263` device with a spaced, `#`- and `?`-containing identity; they assert the exact captured query name for update and delete, preserve a similar fixture, and fail if any API call lacks an explicit mock.

This sprint closes the search-driven general CRUD entry points. The wider web replacement remains open under [the web topology migration plan](sprint-web-topology-migration.md): curve workflows, database import/export and backup, settings, and remaining documented desktop capabilities still need their own evidence and slices.
