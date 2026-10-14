# Web DPT raw waveform viewing sprint

## Goal

Show existing double-pulse-test (DPT) raw voltage and current measurements in the local web GUI. This is a read-only viewing slice; it makes stored waveforms visible without changing measurement import, energy calculation, persistence, or transistor records.

## Current gap

`DPTManager.vue` currently shows switching-energy availability and limited metadata. Its `/api/transistors/{transistor_id}/dpt_data` endpoint does not return raw waveforms. The core adapter intentionally drops legacy top-level `raw_measurement_data`, while the web backend already has a read-only raw-record reader that must be called under `_record_mutation()`.

## Scope and compatibility

- Extend the existing DPT-data response additively with the stored raw measurement records, preserving current `has_e_on`, `has_e_off`, `has_e_rr`, and `metadata` fields. Accept the legacy field as either one record object or a list. Read raw records through `_read_legacy_record_unlocked()` while holding `_record_mutation()`; do not change or bypass the core adapter.
- Include each record's available metadata and raw `dpt_on_vds`, `dpt_on_id`, `dpt_off_vds`, and `dpt_off_id` series. Each sample remains `[time_seconds, value]`, with volts for Vds and amperes for Id. Convert NumPy arrays for JSON serialization without modifying source records. Encode non-finite sample values as their exact strings (`"Infinity"`, `"-Infinity"`, `"NaN"`) so strict JSON preserves sentinel positions; render these values as gaps in charts.
- In the DPT data tab, let users select a stored measurement and acquisition, see its available test conditions, and inspect turn-on and turn-off Vds/Id against time in separate labeled plots. Display the time axis in microseconds with six-decimal tooltip precision for switching detail while keeping raw API samples in seconds. Support multiple stored records and acquisitions. Show a clear empty state for missing event data; do not interpolate, resample, derive energy, or replace missing samples.
- Preserve existing netlist generation and switching-loss validation behavior. No upload/edit/delete workflow is part of this sprint.
- Do not modify Qt, TDB persistence or data models, ntbees2 interfaces/adapters/converters, device records, or the converter bug. Do not claim raw waveform plotting is a validated loss calculation.

## Acceptance and evidence

- API tests cover additive response compatibility, JSON-serializable raw samples and metadata, singleton/list record shapes, non-finite sentinel encoding, absent waveform sides, multiple records/acquisitions, and source non-mutation.
- Web tests show the selected record's conditions and separate turn-on/turn-off voltage/current traces, allow selecting another acquisition, and display missing-data states without failing the whole page.
- Existing DPT netlist and validation routes remain unchanged.
- Update the migration status with the implementation and validation evidence. Keep screenshot parity and other desktop migration gaps open unless separately evidenced.

## Remaining gaps

This sprint only displays already-stored raw DPT waveforms. Importing raw files, deriving or editing switching-energy curves, comparing DPT datasets, broader DPT parity, and full desktop replacement remain open.

## Implementation evidence

- `GET /api/transistors/{transistor_id}/dpt_data` now appends `raw_measurement_data` while retaining its existing summary fields. The raw record is read under `_record_mutation()` and serialized without writing back to the database. Singleton legacy records normalize to a one-record list; NumPy arrays and non-finite values become strict-JSON-safe values while preserving sample order.
- `DPTManager.vue` provides measurement and acquisition selectors, condition metadata, four labeled time-series panels with a microsecond display axis and six-decimal tooltip precision, and explicit empty states. Non-finite ordinate sentinels render as gaps. Existing netlist generation and loss validation controls remain present.
- Focused backend API tests: 5 passed, covering additive compatibility, singleton and list shapes, multiple acquisitions, empty fields, NumPy conversion, strict JSON, checked-in `-Infinity` samples, and file-byte non-mutation.
- Focused DPT Vue tests: 4 passed, covering conditions, all four event/component panels, acquisition and record selection, gap handling, empty state, microsecond conversion, and preserved netlist/validation flows. A separate CurvePlotter regression passed, checking that adjacent raw timestamps remain distinct in the real tooltip formatter.
- `npm run build` passed. Vite reported the existing >500 kB JavaScript chunk warning.
- The non-Qt, non-E2E Python suite passed: 475 passed, 7 skipped, and 11 E2E tests deselected. It ran from a fresh temporary working directory; `--import-mode=importlib` avoids collisions between existing nested `test_api.py` modules. The eight legacy Qt GUI test modules were excluded under the accepted web migration direction. Focused API tests and the DPT Vue tests are also green. Screenshot parity and other migration gaps remain open.
