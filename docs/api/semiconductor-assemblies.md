# Semiconductor assembly transformations

`POST /api/assemblies/analyze` and the **Semiconductor assemblies** browser view
(`index.html?view=assemblies`) call the same pure Python entry point:
`transistordatabase.semiconductor_assemblies.analyze_assembly(request)`.
There is no Qt dependency, device-balancing solver or second device class tree.
The supplied examples are illustrative data, not manufacturer measurements.

All quantities use SI, with junction temperature in °C and costs in their stated
currency. A missing optional quantity is `null`, not zero. Results include a
deep-copied canonical request, sources, conditions and assumptions. Editing an
input invalidates the displayed result and any pending response. JSON export
contains the same Python result and request used by the plots.

## Recorded curve and device input

Each curve uses the existing `MeasuredTable` contract:

```json
{
  "kind": "channel",
  "x": [0, 10],
  "y": [0, 2],
  "conditions": {"t_j_c": 25, "v_g_v": 15},
  "source": "Measurement or datasheet figure identifier"
}
```

`x` is nonnegative current magnitude in A. `y` is nonnegative voltage-drop
magnitude in V for `channel`/`diode`, or event energy in J for `energy`.
At least two distinct finite current samples are required; paired samples are
sorted together. Each curve is limited to 10,000 points in Python and HTTP.
Unknown condition values remain `null`. No nearest-temperature selection,
extrapolation, implicit resistance or zero-energy fallback is performed.

A transformation device is a small data record, with a nonblank `source` and
optional `conduction`, `turn_on` and `turn_off` curves. Each event has its own
conditions and source. Optional scalar fields are `blocking_voltage_v`,
`current_rating_a`, `area_m2`, `cost`, `currency`, `weight_kg`,
`internal_resistance_k_per_w` and `shared_sink_resistance_k_per_w`.
Finite nonnegative scalar values, including explicitly declared ideal zeros,
are supported. Missing fields become `null` in the transformed device.

Optional `foster` input contains complete positive `resistance_k_per_w` and
`tau_s` arrays and optional `capacity_j_per_k`. The shared
`thermal.foster_network` validator checks alignment, positivity, finite totals,
and `tau=R*C` to relative tolerance 1e-9 when C is recorded. Assembly inputs
require recorded R/tau; this interface does not infer an entire pole network
from scalar totals. Missing C remains unknown rather than being invented.

## Equal-sharing bank

```json
{"operation": "parallel-bank", "count": 3, "device": {"source": "My device"}}
```

`count` must be a non-Boolean integer ≥1. For N equal devices:

- Current axes and current rating multiply by N; blocking voltage stays fixed.
- Conduction voltage stays fixed; event-energy values multiply by N.
- Independent contact area, cost and weight multiply by N.
- Internal thermal resistance and each Foster resistance divide by N.
- Foster time constants stay fixed; recorded capacities multiply by N.
- An explicitly shared external sink resistance stays fixed.

This is an **ideal equal-sharing estimate**. It does not establish practical
parallel stability. Area scaling describes N independent contact areas; it does
not infer shared interface geometry. Count one produces an independent copy of
the same numerical data. The original data and arrays remain unchanged.

For a 10 A, 1 mJ, 1 K/W, .01 m², cost-2 device, N=3 gives 30 A, 3 mJ,
1/3 K/W, .03 m² and cost 6. Shared .2 K/W remains .2 K/W.

## Signed IGBT / diode composite

```json
{
  "operation": "signed-composite",
  "positive": {"source": "IGBT source"},
  "negative": {"source": "Diode source"},
  "common_resistance_k_per_w": 0.2,
  "voltage_estimate": null
}
```

The positive and negative devices remain separate magnitude-coordinate records
under `branches`. Browser conduction plots negate both I and V for the negative
branch and draw separate datasets, preserving any discontinuity at zero. Event
energy remains positive. No interpolation joins the two branches at zero.

The assembly blocking rating is the smaller of two known ratings. If either is
missing, the assembly rating remains unknown. Complete costs and weights add;
an incomplete total remains `null`, with `known_cost_sum` and
`known_weight_kg_sum` reported separately. Costs require the same known currency;
incompatible or unknown currency prevents even a known subtotal from being
represented as a single monetary quantity. Each branch's data remains visible.
Common thermal resistance must be supplied explicitly and is never inherited
from one device. This operation does not compute a junction temperature.

By default, corresponding supplied turn-on curves must have matching condition
names and values, and corresponding turn-off curves must independently match.
Turn-on and turn-off may use different reference voltages from each other.
If only one branch supplies an event curve, the other remains unavailable.
No new missing curve is synthesized.

Different supply test voltages require an explicitly selected named estimate:

```json
"voltage_estimate": {"selected": true, "target_voltage_v": 200}
```

Each supplied event table then uses its own recorded positive `v_supply_v`:
`Etarget = Erecorded * target_voltage_v / recorded_voltage_v`. Every other
corresponding condition remains exact, including unknown versus known values.
The result records separate event/branch scaling factors, recorded conditions,
requested conditions and original sources under `estimates`. This is a linear
voltage estimate, not a measured multi-condition surface. A missing recorded
supply voltage, unchecked selection or nonrepresentable factor is rejected.

## Ideal reverse-on envelope

```json
{
  "operation": "reverse-envelope",
  "selected": true,
  "channel": {"kind": "channel", "x": [0, 10], "y": [0, 2], "conditions": {"t_j_c": 25, "v_g_v": 15}, "source": "Channel source"},
  "diode": {"kind": "diode", "x": [0, 10], "y": [0.8, 1], "conditions": {"t_j_c": 25, "v_g_v": 0}, "source": "Diode source"}
}
```

On the overlapping recorded magnitude-current domain only, select the lower
piecewise-linear voltage-drop magnitude. Preserve every source knot in that
domain and add every exact crossing between adjacent linear pieces. Return the
result on an ascending negative-current axis with negative voltage. Independently
declared gate biases may differ by construction; all other named conditions
must match. Both original tables, sources and conditions remain in the result.

For channel `.2*j` and diode `.8+.02*j`, the exact crossing is `40/9 A`.
The mode is explicitly `ideal-lower-drop-envelope-estimate`; it does not predict
simultaneous nonlinear channel/diode current sharing. There is no fixed
101-point resampling or extrapolation. At a source endpoint of zero current,
the signed zero endpoint may be present; it does not join another branch.

## Errors and verification scope

HTTP returns 422 for invalid/missing input, nonfinite raw numeric tokens,
unsupported conditions, disjoint envelope domains, unchecked estimates and
nonrepresentable derived scaling/interpolation. Python raises `ValueError`
for these scientific validation failures. Derived positive values that underflow
to zero during bank or energy scaling are rejected rather than reported as
physical zeros. These checks validate the transformation and transport, not
the accuracy of the supplied device measurements.

Focused checks are in `tests/test_semiconductor_assemblies.py`,
`src/tests/unit/SemiconductorAssemblyWorkbench.test.js`, and
`src/tests/e2e/semiconductor-assemblies.spec.js` under the web frontend.
The science tests include source independence, signed zero boundaries,
independent event conditions, exact 40/9 A and multiple crossings, 1,001 envelope
probes, overflow, missing totals/currencies, and JSON-safe raw NaN rejection.

## Static reverse channel and diode sharing

Open the reverse current sharing panel in the assembly workbench. Supply
isolated channel and diode curves at one junction temperature. Enter a signed
reverse device current. Select the model and confirm that the branch curves
describe compatible operating states. Record each curve's source, isolation
evidence and gate bias. An unknown gate bias is `null`.

`POST /api/assemblies/reverse-sharing` uses branch current and voltage
magnitudes. This illustrative request uses two linear branch characteristics.

```json
{
  "selected": true, "compatible_operating_state": true,
  "mode": "strict-recorded", "current_a": -10, "junction_temperature_c": 25,
  "channel": {
    "kind": "channel-only", "current_a": [0,20], "voltage_v": [0,2],
    "source": "Analytic channel example", "isolation_evidence": "Separate channel equation",
    "recorded_conditions": {"junction_temperature_c":25,"gate_bias_v":15},
    "requested_conditions": {"junction_temperature_c":25,"gate_bias_v":15}
  },
  "diode": {
    "kind": "diode-only", "current_a": [0,20], "voltage_v": [0.6,1],
    "source": "Analytic diode example", "isolation_evidence": "Separate diode equation",
    "recorded_conditions": {"junction_temperature_c":25,"gate_bias_v":null},
    "requested_conditions": {"junction_temperature_c":25,"gate_bias_v":null}
  }
}
```

At −10 A, the channel carries approximately −6.667 A and the diode carries
−3.333 A at a common voltage of −0.667 V. The returned powers describe that
static operating point. Current and voltage residuals accompany the solution.

An unsupported or ambiguous strict query returns a diagnostic and null
solution values. The optional `ideal-diode-blocking` estimate permits zero
diode current under its declared blocking condition. The browser exports the
accepted evidence JSON and result CSV.
