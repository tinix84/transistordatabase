# Signed waveform workbench

The headless entry point is `waveform_losses.analyze_waveform(request)`; the
typed transport is `POST /api/waveform/analyze` and `WaveformRequest` in
`gui_web/backend/waveform_routes.py`. The browser is available at
`/index.html?view=waveform`, also under More Tools → Waveform workbench.

All arrays use SI units. `waveform` requires `time_s` and signed `current_a`;
optional `gate_v` and measured `device_voltage_v` arrays align exactly. A separate
`bus_voltage_v` scalar or aligned array conditions switching energy. Time must be
finite and strictly increasing; no sorting, filtering or resampling is implicit.
`provenance` preserves user-declared source, SI calibration notes, imported
filename and whether the imported samples were edited. Missing metadata remains
unknown. An imported filename alone does not establish measured-device identity.

`conduction.positive` and `.negative` explicitly select magnitude-current models.
A `recorded` model supplies `current_a`, `drop_v`, `recorded_conditions`,
`requested_conditions` and `source`. `linear-drop-estimate` supplies nonnegative
`offset_v`, `resistance_ohm`, assumed `conditions`, `source` and `selected: true`.
There is no implicit borrowing of a forward curve for reverse current. Optional
`duty` is a scalar or aligned piecewise-linear weighting in [0,1]. Gate signals
provide switching evidence and never silently mask conduction.

Conduction intervals split at current zero and crossed table knots. For every
result `power_segments` row, `u=(t-start_s)/(end_s-start_s)` and
`P(u)=sum(power_coefficients_w[k]*u**k)` for k=0…3. Coefficients have units W.
The energy is the exact cubic integral, not a trapezoid of endpoint power. Rows
retain direction, original sample interval, source and cumulative energy bounds;
thermal adapters should consume these polynomials unchanged. Plot samples are
illustrations generated from those same polynomials.

## Hysteretic gate states

Select `events.mode: "hysteresis"` to derive sampled states with a deadband.
The allocation panel accepts the same `digitizer` settings under `gate.mode: "hysteresis"`.
Existing binary and single-threshold policies retain their behavior.

```json
{
  "mode": "hysteresis",
  "digitizer": {
    "thresholds": {"scale": "absolute", "low_v": 2.5, "high_v": 7.5},
    "initial_state": false
  }
}
```

A gate voltage strictly above `high_v` sets the state on.
A voltage strictly below `low_v` sets the state off.
Equality and the deadband retain the previous sampled state.
Without `initial_state`, sample zero must be strictly outside the band.
Otherwise, declare its state explicitly.
A declared state that contradicts an outside-band first sample is rejected.
There is no event before sample zero.

Relative thresholds require `scale: "relative"`, `low_fraction`, `high_fraction`,
`minimum_v`, and `maximum_v` instead of absolute threshold fields.
The fractions satisfy `0 <= low_fraction < high_fraction <= 1`.
The finite calibrated voltage bounds satisfy `minimum_v < maximum_v`.
The selected reference is explicit. The tool does not infer calibration from observed extrema.
Constant traces remain valid with a valid declared calibration or absolute thresholds.
A first sample on or inside the band still requires an initial state.
Resolved thresholds must remain distinct in the supported numeric range.

For `[0,8,4,6,4,2,10]` V with bounds 2.5 V and 7.5 V,
states are `[0,1,1,1,1,0,1]`.
Turn-on brackets start at indices 0 and 5. The turn-off bracket starts at index 4.
Switching uses the existing pre-edge sample and time bracket.
It does not interpolate a switching instant.
Gate states do not mask the conduction calculation or establish physical branch sharing.

`gate_digitization` records the resolved thresholds, initial-state source,
original-sample states, and transition brackets with the method and units.
The browser displays this evidence and retains it in JSON and gate-state CSV downloads.
Threshold, calibration, initial-state, and sample edits invalidate prior results and exports.
CSV rows include scalar method metadata without repeating complete state vectors.

`events.mode` is `gate` with a finite `threshold_v` (strict `gate > threshold`,
equality is low), `masks` with mutually exclusive boolean N−1 `turn_on`/`turn_off`
arrays, or `omitted` with a nonblank string reason. Gate and mask policies cannot
be combined. Every event samples current and bus at index i before bracket
[i,i+1]. No initial or wraparound event is invented.

`switching.turn_on` and `.turn_off` each independently specify positive/negative
energy models. `recorded` requires exact named conditions, including explicit
unknowns, and bounded `current_a`/`energy_j` pairs. A selected
`voltage-scaling-estimate` scales energy by requested/reference `v_supply_v`;
all other conditions must still match. Each event retains its index, bracket,
signed current, independent conditions, source, mode and energy. Finite E(0) is
preserved. Events are energy impulses; no arbitrary finite power pulse is drawn.

Every loss component is requested or explicitly `omit: true` with a nonblank
`omission_reason`. Missing requested data rejects, including when no events are
observed. A supported empty event list produces computed zero with event count0.
Omissions produce null values. `known_component_power_w` sums supported terms;
the full `total_power_w` is null while any component is omitted. Powers use the
complete observation interval; separate on/off event rates are reported.
Legacy `LossResult.n_cycles` is deprecated/unknown (`None`), because event counts
do not establish complete switching periods.

Statistics integrate piecewise-linear current exactly on irregular time grids.
Optional `spectrum` requires uniform samples within relative spacing tolerance
1e−9 and exposes explicit rectangular/Hann window and none/mean detrending.
One-sided peak amplitudes double interior bins, never DC or even-N Nyquist.
The last positive-frequency bin keeps its doubled amplitude for odd N.
The current spectrum includes `real_a`, `imaginary_a` and `phase_rad` for every bin.
Phase uses the first sample as its reference. An exactly zero phasor has null phase.
Set the optional `spectrum.relative_threshold` fraction to select up to ten bins.
The default is 0.1. The full spectrum remains available at every threshold.
For the selection rule and voltage exports, see [grouped physical series](grid-analysis.md#download-grouped-physical-series).
The discrete sequence period is N·dt, distinct from the N−1 interval observation.
Discrete/Parseval RMS is separate from continuous interpolated RMS. Normalized
bin mean-squares and the SI amplitude scale avoid claiming zero representable
RMS after squared-value underflow. The normalization is returned explicitly.

## Declared spectrum masks and band RMS

The optional spectrum fields `mask` and `bands` use all observed FFT bins.
They do not use the relative display threshold or the ten-bin selection.
Omit these fields to retain the existing request, result, and evidence hash.

```json
{
  "window": "rectangular",
  "detrend": "none",
  "mask": {
    "unit": "A_peak",
    "frequency_hz": [0, 1, 2, 4],
    "limit_peak": [1, 1.5, null, 1],
    "interpolation": "exact",
    "source": "Explicit illustrative current limit; no compliance claim"
  },
  "bands": [
    {"name": "Fundamental and second harmonic", "lower_hz": 1,
     "upper_hz": 2, "endpoints": "closed"}
  ]
}
```

Mask limits use coherent-gain-corrected one-sided peak amplitudes.
A strict amplitude greater than the limit is a violation. Equality passes.
`margin_peak` is limit minus amplitude; a negative margin is a violation.
`excess_peak` is the positive excess, or zero for a covered passing bin.
Uncovered bins have null limits and margins, and remain unknown.
A zero amplitude with a zero limit passes.
There is no implicit dB, RMS, detector, or physical compliance conversion.
The waveform transport fixes the quantity to current in `A_peak`.
The reusable engine also accepts an explicitly matching voltage peak mask.

Frequency coordinates must increase strictly and match the accepted binary64
FFT coordinates exactly. Export the spectrum coordinates before defining exact
limits. The tool does not join by row, choose a nearby frequency, or use a
comparison tolerance. Mask values use the same canonical peaks as the spectrum.
Optional `linear-frequency` and `log-frequency` interpolation operate only
between adjacent finite, nonnull limits. Null limits break a segment.
Log-frequency segments require positive endpoints. No mode extrapolates.
A broader mask may extend past Nyquist, but screening covers only observed bins.
A passing screen does not establish coverage of the complete physical profile.
Mask source is required; a filename alone does not establish a standard.

Band endpoints are `closed`, `open`, `left-closed`, or `right-closed`.
Bounds must remain within the physical Nyquist frequency, including for odd
sample counts where the final FFT bin is below Nyquist.
A band containing no bins returns null RMS with `no-bins` status.
Bands may overlap; their RMS values are independent and must not be added.

Band RMS uses the raw windowed, detrended discrete Parseval basis.
It does not apply coherent-gain or window-power-gain correction.
`analysis_basis` records both gains, the window, detrending, units, and the two
quantity bases. For five constant 1 A samples with a Hann window and no
detrending, the corrected DC peak is 1 A, the DC-only band RMS is 0.4 A,
and the all-bin RMS is sqrt(0.3) A. Mean detrending makes these values zero.
Small representable RMS values remain nonzero even when their squares cannot
be represented. Unsupported numeric precision is rejected explicitly.

The browser accepts mask and band CSV in the existing FFT section.
It shows all-bin counts, violations, unknown coverage, and a bounded table
preview. CSV and JSON downloads retain every bin and the declared basis.
Band-only results also provide CSV and exact spectrum JSON.
Any sample, mask, band, or normalization edit invalidates previous results.
The typed input allows up to 5,001 mask points and 32 bands.

Input arrays are limited to10000 samples per typed field. Exact output expansion
is limited to10000 polynomial segments /90000 plot records. Exceeding this budget
raises `WaveformResourceLimitError` (HTTP413); no interval is silently truncated.
Invalid or unsupported scientific inputs return HTTP422. Results and exports
copy caller input into plain independent snapshots.

The browser accepts numeric unquoted CSV/TSV and paired curve entry, preserves explicit
estimates/omissions, and invalidates results on input edits or stale replies.
All time plots use one visual range with independent zoom disabled; changing it
does not change the calculation interval. FFT has a separate frequency axis.
Exports include canonical request+result JSON, per-event CSV and exact interval
CSV. General spreadsheet parsing, image digitization and sensor calibration
fitting remain separate import operations.

## Gate-state current allocation

In the waveform workbench, load aligned time and signed current samples. Open the
gate-state allocation panel and select the command-state assumption. Supply the
source and conditions. Select binary gate states or a gate-voltage threshold.
The sample CSV and evidence JSON downloads contain every accepted sample.

`POST /api/waveform/gate-allocation` accepts 2–10,000 samples at strictly
increasing times. Each current and gate vector must match the time vector.
For example, this input assigns −3 A to the gate-on trace at 0 s and 2 A to
the gate-off trace at 1 ms.

```json
{
  "time_s": [0, 0.001],
  "current_a": [-3, 2],
  "gate": {"mode": "binary", "state": [true, false]},
  "selected": true,
  "source": "Illustrative signed-current samples",
  "conditions": {"junction_temperature_c": null}
}
```

Binary mode accepts booleans and numeric 0 or 1. For threshold mode, supply
`{"mode":"threshold","gate_v":[6,5],"threshold_v":5}`. A gate voltage above
the threshold gives the on state. Equality gives the off state. Use `null`
for an unknown condition.

At each supplied time, the two returned trace values sum to the signed device
current. The allocation expresses the selected gate-command assumption only at
those samples. Physical channel/diode sharing and interval losses require
separate device models.
