# Web DPT raw measurement comparison sprint

## Goal

Let users compare two or more already-stored double-pulse-test (DPT) raw measurement records in the local web GUI. Overlay corresponding turn-on and turn-off Vds/Id traces while retaining each record's identity, test conditions, and independently selected acquisition.

## Scope and compatibility

- Reuse the additive `raw_measurement_data` response from the [DPT waveform viewing sprint](sprint-web-dpt-waveforms.md). This sprint is read-only and does not change API responses, database records, adapters, or persistence.
- Allow selecting at least two measurement records. Provide an acquisition selector for every selected record with multiple acquisitions; each selection is independent, including when records have different acquisition counts.
- Overlay stored samples from each selected acquisition in four separate plots: turn-on Vds, turn-on Id, turn-off Vds, and turn-off Id. Identify each trace by its measurement record and acquisition, and show the selected record's available conditions beside the matching trace color.
- Use the raw sample order and values as stored. Convert finite time coordinates from seconds to microseconds only in the chart projection. Preserve non-finite ordinate samples as chart gaps. Use six-decimal DPT time tooltip precision.
- Show which selected records lack samples for a plotted event/component. Do not align records by interpolation, resample, filter, derive switching energy, or synthesize missing values.
- Preserve the existing single-record viewer, netlist generation, and switching-loss validation workflows. No raw import or measurement write path is included.
- Do not modify Qt, TDB models/persistence/device data, ntbees2 interfaces/adapters/converters, or the known converter bug.

## Acceptance and evidence

- Vue tests cover selecting multiple records, separately choosing acquisitions for records with different acquisition counts, correct overlay values and labels in the four plots, condition display, absent event data, and non-finite gaps.
- The production web build succeeds.
- The broader topology migration remains open; this sprint only closes read-only raw DPT dataset comparison.

## Implementation evidence

- `DPTManager.vue` adds comparison mode with a record checklist, per-record acquisition selection, color-matched record cards with test conditions, and four overlay plots. The existing single-record mode remains available.
- Comparison chart datasets are projections of the selected raw samples. Stored time values remain in seconds in the API data; chart x values are converted to microseconds, sample order is preserved, and non-finite y values remain `null` gaps. Missing event traces are named in each corresponding panel.
- Focused Vue DPT tests: 6 passed, covering single-record behavior, record selection, different acquisition counts and independent selectors, all four overlays, labels and conditions, missing event data, non-finite gaps, microsecond x coordinates, six-decimal tooltip configuration, and unchanged netlist/validation workflows.
- `npm run build` passed; Vite reported the existing bundle-size warning.

## Remaining gaps

Raw DPT file import, editing or deriving switching-energy curves, DPT filtering/measurement alignment, and broader Qt-to-web parity remain open. Comparison plots are observational views of stored samples and are not a validated switching-loss calculation.
