# Recorded comparison and explicit static-duty estimates

`POST /api/comparison/advanced` now uses a strict typed request. Two or three
stored transistor IDs are required. `config: {}` is a useful request: it returns
every supported recorded channel, switching, capacitance, gate-charge and SOA
slice at its actual conditions, plus validated Foster impedance when available.
No requested temperature, voltage or gate resistance selects a nearby curve.
Labels, condition tables and evidence JSON retain the original conditions and
record index. Invalid source curves produce individual diagnostics.

Loss and efficiency rows remain `unsupported` with null values until all inputs
and active source data are supplied. Each device retains its own status/reason;
one unavailable device does not erase the other device's calculation or recorded
panels. Unknown conditions are explicit `null`, which never matches a known
condition. Missing data is never zero. Nonfinite request inputs produce HTTP422.

An explicit estimate request has this shape (all values in SI except °C):

```json
{
  "transistor_ids": ["device-a", "device-b"],
  "config": {
    "t_j": 25,
    "i_channel": 10,
    "f_sw": 100000,
    "duty": 0.25,
    "v_supply": [400, 400],
    "v_g": [15, 15],
    "v_g_off": [-3, -3],
    "r_g_on": [4, 4],
    "r_g_off": [7, 7],
    "parallel_count": [1, 1],
    "output_power_w": 1000
  }
}
```

Each per-device array follows the selected ID order. `i_channel` is the total
shared current for each alternative. Identical parallel devices carry `I/N` and
all component powers are multiplied by N. The browser requires the user to select
the held-current PWM estimate and explains ideal sharing and the selected duty.
Blank inputs never become zero. Per-device gate-off voltage can be explicitly
unknown. Source unknowns must match explicitly unknown requests.

`static_losses.static_duty_losses` is the reusable headless adapter, also used by
`CalculationService.calculate_losses`. It requires all scientific conditions;
the old hidden 25°C/15V/10kHz values and suppressed conduction exceptions are
gone. It delegates calculation to `analyze_converter` with a single physical
role and `held-envelope-estimate`, which uses the accepted exact waveform
conduction and switching-energy authority. Positive current only is supported
by this legacy adapter; signed models belong in the waveform/converter workbench.
Exactly one unambiguous channel record and independent on/off energy-current
records must match Tj, gate, gate-off, Rg and supply. Queries must stay inside
recorded current coverage. There is no extrapolation, clamping, nearest-point
selection, voltage scaling or arbitrary ZVS reduction.

For `0 < duty < 1`, one on/off event per PWM period is selected. D=0 or D=1, or
fs=0, explicitly disables switching and needs no unused energy curves. D=0 needs
no channel curve. At zero current conduction follows P=V×0 while finite E(0)
switching energy remains evaluable. The public
`CalculationService.interpolate_switching_losses` additionally supports a
single measured energy only at exactly its recorded current and conditions;
the static-duty adapter requires a bounded current-energy curve.

Efficiency is `Pout / (Pout + semiconductor loss) × 100`, only when the caller
supplies positive `output_power_w` and the full device loss is available. It is
explicitly a semiconductor-only estimate, excluding magnetics, drive,
auxiliaries and all other converter losses. No invented efficiency-vs-current
curve remains.

The old `find_optimal_working_point` and equal-score `rank_transistors` return a
clear `NotImplementedError`: these signatures never established an evaluated
objective. Generic `export_to_spice` no longer emits an NPN model for arbitrary
devices. `export_to_ltspice` no longer derives conditions from ratings. Those
legacy signatures now point to explicit supported DPT/manufacturer-model
workflows. Single export and preview HTTP routes return useful422 errors.

The shared `WaveformModelEditor` preserves Known state when a numeric field is
cleared, visibly marks it invalid, and both waveform/converter serializers reject
it before transport. An invalid Known value cannot silently serialize as JSON
null and become an unknown condition.

Validation uses independent 2V drop and 1/2mJ energies at10A, duty.25 and100kHz:
5W conduction +100W on +200W off =305W; doubling frequency gives605W. Tests cover
source ambiguity, T/V/gate/Rg mismatch, current bounds, independent on/off Rg,
explicit unknowns, inactive components, ideal sharing, representational current
underflow, immutable inputs, partial panels, typed HTTP and unsupported exports.
`tests/legacy_comparison_browser_server.py` serves the actual production app and
compiled UI with a read-only in-memory fixture repository on port8031; browser
tests perform actual HTTP/Python calculations without altering stored devices.

Owner verification (2026-10-16): 48 headless Python/API/core-service tests, 22
Vue tests (including adjacent waveform/converter Known-state regressions), and
four compiled Chromium workflows passed. The held-response test initially used
an ambiguous status selector; its corrected affected rerun passed. A final
CSS-only minimum cell width was added after visual review to keep device names
readable inside mobile horizontal tables; the affected mobile probe must run
after the coordinated preserved rebuild before final acceptance.
