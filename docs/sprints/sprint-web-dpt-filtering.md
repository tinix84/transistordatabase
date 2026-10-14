# Web DPT stored-measurement filtering sprint

## Goal

Let users find already-stored double-pulse-test (DPT) raw measurement records by their available metadata and test conditions in the local web GUI, both in the single-record viewer and in the comparison candidate list.

## Scope and compatibility

- Filter the existing read-only `raw_measurement_data` response in the browser. Search is case-insensitive and treats whitespace-separated terms as an AND query over the record's metadata values and the labels/values of displayed test conditions.
- Cover the metadata already returned for each stored record, including dataset type, junction temperature, supply/gate conditions, gate resistances, inductances, measurement date, test bench, commutation device, and comment when present. Do not search waveform sample values.
- In single-record mode, show only matching choices and move to the first matching record if the current selection is filtered out. Reset that record's acquisition selector when the record changes. Show a clear no-match state and a clear-search action.
- Keep comparison entry available whenever multiple raw records are stored, even when the search has only one match. In comparison mode, filter the records offered in the checklist. Keep already selected records in the overlays if the search hides them, preserve their independent acquisition selections, and explain that behavior in the UI. Clearing the search restores all candidates.
- Keep the filter entirely in the browser. Do not change API responses, stored records, raw sample order/values, persistence, TDB models/adapters, converter output, ntbees2 code or interfaces, device data, or Qt.
- Preserve existing single-record waveform viewing, comparison overlays, netlist generation, and switching-loss validation.

## Acceptance and evidence

- Vue coverage filters single-record choices by numeric conditions and text metadata, reports no matches, restores choices after clearing, and verifies the selected record's waveform remains available.
- Vue coverage filters comparison candidates while retaining hidden selected overlays and their independently selected acquisitions, then restores candidates on clear.
- Vue coverage confirms comparison can be entered with multiple stored records when only one matches, and that only matching records are offered as candidates.
- Filtering does not rewrite or reorder the API's raw measurement data.
- The focused DPT Vue suite passes and the Vite production build succeeds.
- This sprint adds read-only filtering only; the broader web migration remains open.

## Implementation evidence

- `DPTManager.vue` adds a metadata/condition search field with a visible match count, clear action, and no-match message. Matching uses case-insensitive terms across available metadata and formatted condition labels/values while retaining each record's original list index.
- The single-record selector offers matching records and resets the acquisition when the current record is excluded. Comparison candidates are filtered independently from selected overlays; hidden selections keep their overlay and acquisition and are called out until the search is cleared.
- The filter is computed from the existing response in Vue. No API, TDB model, persistence, converter, or device-record code changed; waveform samples remain untouched.
- `npm run test:run -- src/tests/unit/DPTManager.test.js`: 10 passed. Coverage includes condition/text search, empty-result recovery, comparison entry with one matching candidate, retained hidden overlays and per-record acquisition state, and raw response non-mutation.
- `npm run build`: passed. Vite reports the existing JavaScript chunk-size warning (558.61 kB minified).

## Remaining gaps

Raw DPT file import remains open. The current `transistordatabase/utils/ltspice_dpt.py` supports netlist generation, simulation launch, waveform analysis from caller-supplied arrays, and switching-loss data conversion; it does not read `.raw`, log, or CSV files. The existing core adapter also drops top-level raw measurements during legacy-to-core conversion and writes `raw_measurement_data: []` during core-to-legacy conversion. The web repository save path therefore cannot safely persist an appended raw measurement without changing the current adapter/persistence boundary. This sprint intentionally leaves that boundary unchanged.

Search is substring-based over metadata and displayed condition text. It does not offer numeric ranges, sample-value filters, interpolation/alignment, energy derivation, or broader Qt parity. DPT raw import, energy-curve derivation/editing, screenshot parity, and full desktop capability coverage remain open.
