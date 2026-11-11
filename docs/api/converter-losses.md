# Converter loss aggregation

`transistordatabase.converter_losses.analyze_converter(request)` is the reusable
Python authority. `GET /api/converters/presets` returns revisioned declarative
mappings and `POST /api/converters/analyze` uses the same function. All values are
SI. This is supplied-data postprocessing, not a new converter operating-point
solver or a physical proof of ZVS.

The presets are `sc-dab`, `sc-src-zcs`, `sc-src-zvs`, `sc-llc`, `sc-bridgeless`,
`sc-six-switch`, `sc-vr`, `sc-vrn`, `sc-ivr`, `sc-ivrn`, `sc-tvr` and
`inverter-six`. Sources and exact signs/counts are returned with each preset.
`custom` accepts additional explicitly named physical positions, including a
twelve-device ENCO-style configuration without claiming it is SC's eight-device
DAB. No reference implementation is copied, imported or executed.

## Request

```json
{
  "preset": "custom",
  "operation": "held-envelope-estimate",
  "time_s": [0, 1, 4],
  "signals": {"i": [10, 10], "d": [0.25, 0.25], "fs": 100000, "v": 200},
  "bindings": [{
    "role": "position", "members": ["s1"], "multiplicity": 1,
    "conduction_current": {"signal": "i", "scale": 1},
    "duty": {"signal": "d"}, "frequency_hz": {"signal": "fs"},
    "bus_voltage": {"signal": "v"}
  }],
  "role_models": {
    "position": {
      "source": "Declared position model",
      "conduction": {
        "positive": {
          "mode": "linear-drop-estimate", "selected": true,
          "offset_v": 2, "resistance_ohm": 0,
          "source": "Explicit constant-drop estimate", "conditions": {}
        }
      },
      "switching": {
        "turn_on": {"omit": true, "omission_reason": "No energy data supplied"},
        "turn_off": {"omit": true, "omission_reason": "No energy data supplied"}
      }
    }
  },
  "assumptions": {"envelope_selected": true, "symmetry_selected": false}
}
```

Each model's conduction, independent turn-on/off directional data and waveform
event policy use the [waveform contract](waveform-workbench.md). A model-level
`source` is optional extra position provenance; each actual scientific curve still
requires its own nonblank source and exact conditions. Known `null` conditions are
retained. Missing requested directional data rejects, including extrapolation and
bus-condition mismatch. Voltage-proportional switching energy is available only
as the named, explicitly selected estimate. Equal tiny/large reference and query
voltages retain representable energies without avoidable multiply/divide overflow
or underflow.

Preset bindings are immutable. Readable mapping edits can be submitted as a custom
mapping. A binding is a named signal times a finite scalar, optionally `1-value`,
or an explicit constant `{"value": ...}`. There is no expression evaluator.
Physical members must be globally unique, multiplicity equals their count, and
counts above one require `symmetry_selected=true`: identical loss behavior is an
explicit source symmetry assumption, not parallel-die current sharing. Equal die
sharing belongs to the separate assembly workbench.

## Recorded waveform operation

`operation="waveform"` takes N increasing `time_s` and N aligned values per signal;
explicit scalar bindings are broadcast with metadata. Conduction current and
commutation current have separate bindings. The shared waveform engine integrates
conduction and switching independently, then only physical requested components
are combined. Internal unused components do not become physical omissions.
For gate events a role supplies `events={"mode":"gate","threshold_v":...}`;
boolean N-1 explicit masks are also supported. An unavailable gate column is not
required for explicit masks. A pair of explicitly omitted switching components
needs neither gate nor switching bus observations.

Pre-edge sample i represents the bracket [i,i+1], with no invented first/wrapped
edge and no discarded finite E(0). Full supplied duration sets average power.
The DAB maps is2/is4/is6/is8 conduction independently from -itp/+itp/+its/-its
commutation and counts each role twice. These corrected pre-edge and independent
Eoff-reference rules intentionally differ from defects in the older SC helper.

SRC-ZCS omits six positions' switching in its source scope; SRC-ZVS/LLC omit four.
Selecting those presets requires a nonblank `assumptions.source_omission_reason`.
Secondary conduction models remain required. Omitted switching is null and makes
the complete converter total null. It is not physically proven zero by a topology
name. Use a custom mapping with actual additional data to extend that scope.

## Held frequency-envelope operation

`operation="held-envelope-estimate"` requires `envelope_selected=true`. N time
edges define N-1 intervals; current, duty, bus and frequency are held independently
on each interval. Signal arrays have exactly N-1 values; scalar broadcasts are
explicitly recorded. Conduction uses the shared signed engine with constant
current/duty on that interval. Switching uses the exact same directional,
conditioned energy evaluator as recorded events and multiplies by supplied Hz.
No synthetic PWM, measured switching events, cycle counts or event rates are
generated. Integrating local W times interval seconds gives J; the complete time
interval supplies the W denominator.

Duty must lie in [0,1] and frequency be nonnegative. D=0 gives known inactive zero
conduction and switching; D=1 gives conduction only; fs=0 suppresses switching
while retaining conduction. Those inactive switching mechanisms need no energy
curves. An explicitly selected omission still remains an omission. No recorded
energy curve is automatically scaled to another bus voltage.

Bridgeless reverse IGBTs are explicitly active only when selected
`assumptions.operating_power_w<0`; this activation does not change current signs.
Six-switch neutral devices use `3*ia` and line frequency `fl`, separately from
their multiplicity two. Rectifier counts are VR 6/0, VRN 6/6, IVR 12/0, IVRN
12/6 and TVR 0/0; D=1 rectifiers need no fabricated 5 V switching bus.
`inverter-six` uses all six supplied independent phase/position branches and works
for partial or unbalanced records without assuming representative-phase symmetry.
No ideal sinusoidal generator or representative-phase-times-three shortcut is
provided.

## Results and limits

Results contain canonical copied `request`, preset revision/source, expanded
`bindings`, `branches` and `totals`. Each branch preserves its `model_bindings`,
per-device `components`, multiplicity-scaled `scaled_components`, broadcasts and
scientific evidence. Scaling changes energies/powers and represented event counts,
not current amplitudes. Waveform evidence retains separate conduction/switching
signals, exact power polynomial segments/plots, local event records and cumulative
switching energy. Envelope evidence retains interval-held inputs and local power
plus independent energy source/condition evidence.

When both switching contributions are omitted, switching signal/cumulative
scientific evidence is empty and `switching_signals_status` is
`not-evaluated-explicit-omissions`. The raw supplied signals remain in the
canonical request; no internal unused current becomes a claimed commutation
observation.

`totals.components` distinguishes complete numeric energy/power from a partial
known subtotal. `total_power_w` and `total_energy_j` are null if any required
physical contribution is omitted; `known_component_*` remains explicit. Computed
zero, selected inactivity, explicit omission and missing requested data are
different states. Every physical contribution covers the same supplied interval.
Source arrays and results do not alias later caller edits.

Resource limits are 32 roles and 10,000 total role intervals, with a further
10,000-total-exact-segment expansion bound. Excess returns HTTP413; invalid,
missing, nonfinite, unsupported or overflowing inputs return HTTP422. There is no
automatic resampling/truncation or manufactured fallback. This interface does not
combine temperatures across a common sink, identify sensor calibration, or prove
measured-device accuracy.

## Browser workbench

Open `/index.html?view=converter` or choose **Converter loss workbench** in the
More menu. The preset selector loads the Python-owned source mapping, including
physical members, multiplicity, signed conduction and commutation references,
gate/bus identity, duty, frequency and source omissions. Editable named SI signal
rows and increasing times supply the data; waveform samples and held interval
values are explained separately. There is no converter operating-point generator.

Independent algebraic DAB, LLC-omission and six-switch-envelope fixtures are
available. They are explicitly labeled hand inputs, not manufacturer measurements.
Symmetry, the envelope estimate and source omission scope require separate user
selections. Each role expands into signed bounded curves, source identities,
independent recorded/requested conditions and named estimate controls. Waveform
events accept explicit gate thresholds or N−1 boolean bracket masks. Custom
mappings expose member counts, named signal references/scalars and duty complements;
the canonical JSON editor handles advanced import without replacing readable forms.

Results distinguish complete totals from known partial subtotals. Component powers
are shown per device and after physical multiplicity, with stacked W bars and
explicit omitted statuses. A role selector links current, commutation, gate,
polynomial-power, cumulative-energy and event views, or held duty/frequency/power
views in envelope mode. Every trace consumes returned Python evidence. Fully
omitted switching does not produce a commutation-current observation. One shared
visual range changes plot bounds without changing the integration interval.
Source and independent conditions remain readable in the result tables.

The request/result JSON download retains the canonical calculation and evidence;
branch CSV retains exact returned powers, member counts, sources and empty unknown
components. Editing any scientific input/source clears results, and late replies
cannot restore obsolete results. The browser consumes supplied SI calibration
notes but does not fit sensor calibration or claim physical ZVS verification.
