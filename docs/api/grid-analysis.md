# Finite waveform analysis and energy maps

Use these operations for finite declared inputs. All physical values use SI units.
The API rejects unknown fields and nonfinite values. It does not execute custom expressions.

## Configuration and operating-point grids

Send `POST /api/analysis/grid` with these fields:

- `configurations`: records with `id`, `source`, `conduction`, and `switching`.
- `operating_points`: records with `id`, `source`, `weight`, `waveform`, and `events`. Optional fields are `spectrum` and `provenance`.
- `metric`: a `name`, `unit: "W"`, and `direction: "minimize"` or `"maximize"`.
- `summary_mode`: `"full"` or explicitly selected `"successful-subset"`. The default is `"full"`.
- `include_zero_weight_in_worst`: include zero-weight points in worst-point reports. The default is `false`.

The waveform, model, event, and spectrum fields use the existing waveform API schema.
Supported metric names are `total_power_w`, `conduction_power_w`, `turn_on_power_w`, and `turn_off_power_w`.
Weights must be nonnegative. Their finite total must be positive. IDs must be unique within each collection.

Each `rows` entry contains its configuration ID, operating-point ID, full inputs, input hash, method version, and metrics.
Its `result` contains the exact Python waveform result for linked time, spectrum, and loss views.
An unsupported pair retains its inputs and error. Other pairs continue.
Omitted loss components remain unavailable. They do not become computed zero losses.

Each configuration summary contains four independent metric summaries:

- `full_mean` is unavailable if any positive-weight point lacks that metric.
- `subset_mean` reports the successful subset separately. `retained_weight`, `total_weight`, and `coverage` describe its population.
- `selected_mean` follows `summary_mode`. Only available selected means enter `ranking`.
- `worst_value`, `worst_point_ids`, and `worst_complete` describe available worst-point evidence and missing coverage.

For minimization, the worst value is the maximum. For maximization, the worst value is the minimum.
Equal ranking values use configuration IDs as stable ordering keys.

The grid supports at most 32 configurations, 64 operating points, and 256 pairs.
The complete expansion supports at most 200000 waveform samples.
The input limit is 4 MB. The response limit is 24 MB.
The HTTP byte limit includes whitespace and applies before JSON parsing to all three analysis endpoints.
The adapter checks pair, sample, and input limits before evaluation. It checks response size as results accumulate.
Before each calculation, it bounds repeated source and condition metadata using the accepted event detector's edge counts.
The thermal handoff uses the same metadata check and complete 24 MB response limit.
Resource failures return HTTP 413. Invalid declared populations return HTTP 422.
Expected scientific failures remain individual unsupported rows.

### Download grouped physical series

In **Analysis grid**, select **Include grouped series statistics and harmonic exports** before calculation. For the API, add `"grouped_series_exports": true` to the grid request. Omitting the field or setting it to false preserves the previous grid input hash.

The server projects the accepted pairs into time and frequency tables. Each table includes current in A and every supplied gate, device and bus voltage in V. A scalar bus voltage supplies the same value at each sample. Each row retains the configuration, operating point, weight, sources, series, unit, pair status, method and input hashes.

The time table reports sample count, minimum, maximum, arithmetic mean, sample standard deviation and discrete root mean square (RMS). Sample standard deviation uses an `N − 1` denominator. Separate time-weighted mean and RMS columns integrate the piecewise-linear waveform. For samples 0 and 2, discrete RMS is √2 and time-weighted RMS is 2/√3.

For a frequency table, set `spectrum` on the operating point. For example, use `{"window":"rectangular","detrend":"none","relative_threshold":0.5}`. The threshold defaults to 0.1 and accepts finite fractions from 0 to 1. The guided form omits the default fraction from its request.

The server keeps the first ten bins in ascending frequency whose magnitudes strictly exceed the largest magnitude times the fraction. The ENCO source calls these selected bins harmonics. They can have any frequency on the discrete Fourier transform grid. Each bin includes frequency in Hz, magnitude, real and imaginary components, and phase in radians relative to the first sample. The amplitude scale corrects for window coherent gain. It doubles each positive-frequency bin except the Nyquist bin for an even sample count. DC keeps its original scale. An exactly zero phasor has null phase. Each operating point retains its own phase.

A computed spectrum with zero retained bins has one manifest row with `retained_bin_count: 0` and null bin values. An unrequested spectrum has status `not_requested`. A failed series operation has status `unsupported` and an error. The pair status remains separate. A failed pair retains its error and null numerical values.

The four downloads contain the complete server-generated time or frequency rows. JSON preserves the server's number representations. CSV writes null as `NULL` and prefixes formula-like text with an apostrophe. Table pages and pair filters affect only the display. Row limits are 1,024 for time and 10,240 for frequency. The 24 MB response budget includes the original grid, both tables and their download text.

### Compare configurations with weighted merit

In **Analysis grid**, select **Compare weighted merit penalties**. Declare one to four distinct power metrics. For the API, include an `objectives` array in the grid request, for example:

```json
"objectives": [
  {"name": "conduction_power_w", "unit": "W", "direction": "minimize", "weight": 1},
  {"name": "turn_on_power_w", "unit": "W", "direction": "minimize", "weight": 3}
]
```

Each objective needs a finite, nonnegative weight. A positive weight activates the objective. The sum of weights must be positive and finite. Omit `objectives` or use `null` to retain the original grid response and input hash.

Operating-point weights determine each configuration's full mean in W. Objective weights determine the relative influence of those means on the dimensionless score. A configuration qualifies only when every active objective has a value at every positive-weight operating point. A missing value at a zero-weight point has no effect. Each excluded configuration retains its missing point IDs, raw means, and weight coverage. The successful-subset option applies to the original grid ranking only.

For each active objective, read the minimum and maximum full means across the same eligible configurations. For a mean `x`, use `(x − minimum) / (maximum − minimum)` for minimization or `(maximum − x) / (maximum − minimum)` for maximization. A constant range gives a penalty of zero. Divide each objective weight by the sum of weights and multiply by its penalty to get its contribution. Inactive objectives contribute zero. Excluded configurations have null scores, ranks, and active contributions.

The score is the weighted mean of active penalties. A lower score ranks first. Equal computed scores share a competition rank, with configuration IDs in ascending order within each tie. For example, two first places give the next configuration rank 3. If no configuration qualifies, the response includes the grid rows and reports merit as `unavailable`.

For the example weights above, suppose A has full means of 25 W conduction and 4 W turn-on. B has 20 W and 8 W. A scores 0.25 and B scores 0.75. Changing the turn-on direction to maximize gives B a score of zero and A a score of one.

Read `merit.population` for the eligible configuration IDs and the positive- and zero-weight point IDs. The response includes all tied extrema, the grid input hash, and the population hash. The server keeps the score between its smallest and largest active penalties. Rounding can cause a last-digit difference between the score and the sum of exported contributions. The API rejects a positive weight, penalty, or contribution that would round to zero.

Download the server's JSON or CSV to retain all configuration–objective rows. Pair filters and page changes affect only the display. CSV represents null as `NULL`. The maximum is 128 export rows. The 24 MB response budget covers the grid, merit data, and download text. The server reserves space for repeated identities before creating the exports. Long identities can trigger HTTP 413 even when the final data would fit.

## Conditioned energy maps

Send `POST /api/analysis/energy-map` with `energy_kind`, `conditions`, `slices`, `currents_a`, and `voltages_v`.
The energy kind is `"on"`, `"off"`, or `"rr"`.

Every fixed-condition object must declare these fields:

```json
{
  "t_j_c": 25,
  "v_g_on_v": 15,
  "v_g_off_v": -5,
  "r_g_on_ohm": 2,
  "r_g_off_ohm": 3,
  "commutation_device": null,
  "commutation_source": null,
  "energy_convention": "Declared event integral"
}
```

Use `null` for an unknown condition. The energy convention must be nonblank.
Unknown conditions match only unknown conditions. Numeric conditions use the common table helper's absolute tolerance of 1e-9.

Each source slice contains `id`, `source`, `source_kind`, `energy_kind`, `voltage_v`, `current_a`, `energy_j`, and `conditions`.
Supported source kinds are `measured`, `datasheet`, `simulated`, `manual`, and `unknown`.
Source IDs remain separate from fixed comparison conditions. All original points remain in `source_slices`.
Mismatched conditions or energy kinds produce visible exclusion diagnostics.

The response sorts displayed axes. `cells[voltage_index][current_index]` contains the energy in joules.
A cell with one covering source is `available`. Multiple covering sources make it `ambiguous`.
Absent coverage makes it `missing`. Each cell retains candidate sources and unsupported-source diagnostics.
`recorded-point` identifies original current samples. `bounded-current-linear` identifies bounded current interpolation.
No voltage scaling or nearest-condition selection occurs.

Select a bounded estimate with this optional request field:

```json
"estimate": {"selected": true, "current_a": 5, "voltage_v": 600}
```

The estimate requires four available displayed corners. It rejects extrapolation and ambiguous or missing corners.
It uses bilinear interpolation between displayed corners. Interior source knots are not inserted.
This is an estimate on the displayed grid. It is not a measured multidimensional surface.

Limits are 128 slices, 100000 total source-current samples, 256 displayed currents, and 128 displayed voltages.
The displayed grid supports at most 16384 cells. Input and response limits are 4 MB and 24 MB.
The complete response estimate includes repeated candidate metadata. Resource failures return HTTP 413.

### Inspect original energy observations in three dimensions

1. Open **Energy map** and calculate a prepared map or import an XYZ source.
2. Select **Source scatter** to inspect current in A, voltage in V and switching energy in J.
3. Select a point or its source row to inspect its original identity and values.
4. Choose **Source scatter with visual surface connections** to inspect adjacent source coordinates.

Each finite original observation contributes a point. Repeated coordinates retain separate source rows. A recorded zero appears at 0 J. An unknown energy remains in the table with a null value.

Surface connections require four unique available corners. The prepared map also requires source-slice edges between adjacent current samples. Gaps remain open. Use the bounded estimate in the previous section for a numerical energy query.

The scene displays up to 12,000 finite points. Above that limit, use the source table. Face construction supports up to 65,536 current-by-voltage coordinates. The source table displays 50 rows per page.

## Exact waveform heat transfer

Send `POST /api/analysis/thermal-handoff` with this structure:

```json
{
  "waveform_request": {},
  "direction_owners": {"positive": "forward", "negative": "reverse", "zero": "forward"},
  "network_heat_owner": "shared-junction",
  "event_mode": "impulse",
  "allow_partial": false
}
```

Replace `waveform_request` with the complete waveform API request.
Declare physical owners explicitly. Current sign does not identify a physical die.
A single-owner network rejects incompatible conduction or switching heat.
The `shared-junction` choice explicitly assigns both owners to one physical junction.

The adapter recomputes the waveform in Python. It preserves exact polynomial coefficients and absolute event times.
Each staged segment adds its declared `heat_owner`. The original electrical `direction` remains unchanged.
The thermal API honors an explicit segment owner. Without one, its standalone direction rule remains in effect.
An explicitly owned segment with all zero coefficients can remain in another network's observation interval.
Any nonzero coefficient requires compatible physical ownership. Stored energy metadata cannot override this check.
Negative time origins are unsupported. Switching events use ideal instantaneous energy deposition.
No thermal network, case temperature, or initial state is inferred.

The response contains `thermal_input`, `waveform_result`, `source_request`, and both source and complete-handoff input hashes.
The handoff preserves optional-field presence from the supplied accepted request. It does not insert absent fields into source evidence.
`thermal_input` supplies exact `segments`, `events`, ownership, and temporal conventions for the existing thermal API.
Supply a physical network, case temperature, initial state, and sampling choice before thermal analysis.

Omitted components require `allow_partial: true`. The response retains omission reasons and reports `scope: "known-components-only"`.
If conduction is omitted, the adapter stages a disclosed zero background for known switching heat only.
That background does not establish zero physical conduction loss or complete heat coverage.
