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
