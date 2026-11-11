# Converter aggregation contract for semiconductor parity

Prepared 2026-10-16 by read-only source inspection for execution step 7. This is
implementation preparation, not completed or numerically validated coverage.
No reference application, MATLAB script, Python reference module or Qt code was
executed. The implementation must be independent: SC converter source also carries
BRUSA notices, and ENCO reserves rights to BRUSA HyPower. Behavioral mappings below
do not authorize copying package implementations.

Reference roots, preserved literally:

```text
\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc
D:\OneDrive\claude\bhy\enco
```

All SC locations below are relative to the first root; ENCO locations are relative
to the second. All input and output quantities use SI units. Follow
`docs/adr/0001-local-web-gui-replaces-qt.md`: reusable Python science, FastAPI
transport, browser interaction. No new converter operating-point solver is implied.

## Two distinct kinds of operation

1. **Recorded waveform aggregation** consumes complete supplied time/current/gate
   records. Conduction is integrated across the full record. Switching energy is
   summed over explicit event brackets and divided by the full record duration.
   DAB and SRC/LLC source functions are this kind of postprocessor.
2. **Hard-switching envelope estimates** consume supplied slowly varying current,
   duty, blocking voltage and switching frequency. They estimate local averaged
   conduction and switching power without resolving PWM events. PFC functions and
   the inverter example use this assumption. Do not manufacture event counts or
   present their power samples as measured switching transients.

The inverter example additionally synthesizes balanced sinusoidal fundamental
voltages/currents and negative-rail-clamped duty. That small optional, explicit
ideal waveform generator is separable from converter aggregation. The PFC and
resonant loss functions inspected here do not generate their waveforms.

## Shared scientific seam and result semantics

Step 5 currently exposes `waveform_losses.analyze_waveform(request: dict) -> dict`.
Its request has `waveform`, `events`, `conduction`, `switching`; waveform keys are
`time_s`, `current_a`, optional `gate_v`, `device_voltage_v`, `bus_voltage_v`.
The event policy is thresholded gate evidence, explicit boolean masks of length
N-1, or explicit omission. Event current and bus voltage use pre-edge sample i
for the bracket [i,i+1]. Directional models use current magnitude after retaining
its sign to select positive versus negative data. Finite E(0) is retained.

Conduction accepts separate positive/negative recorded models or explicitly
selected linear-drop estimates, and an optional scalar/aligned duty. Switching
has independent `turn_on` and `turn_off` directional energy models, including
explicitly selected voltage-scaling estimates. Recorded/requested conditions and
nonblank source identities are retained. `MeasuredTable` and `require_conditions`
in `condition_tables.py` own bounded lookup and exact condition matching.

Important adapter requirement: **one physical branch can have different conduction
and commutation current signals.** For example DAB s12 conduction uses is2 but
switching uses -itp. Do not replace the switching current with is2. A lean adapter
can call the same engine once for conduction and once for switching, then combine
only those requested component results. The intentionally unused contributions
of those internal calls must not become artificial physical omissions in the
combined branch result. Alternatively add a small explicit event-current input to
the shared engine after independently verifying compatibility; do not duplicate
its sign, energy lookup, condition or event logic in converter-specific modules.

Proposed external module interface:

```text
analyze_converter(request) -> branch results + converter totals + evidence
request = topology/preset, shared time/signals, role-to-model bindings,
          waveform or envelope operation, selected assumptions
```

Each declarative preset binds role name, physical member names, positive integer
multiplicity, conduction signal/expression, switching current signal/expression,
gate or explicit event masks, bus voltage, optional duty/frequency, and explicit
omission/inactive reasons. Expressions needed by these presets are only a named
signal times a finite scalar and the complement 1-D; do not build a string eval
language. Keep preset data separate from the common calculation implementation.
Expose expanded mappings in results so signs/counts are inspectable.

Return each contribution before and after multiplicity, complete branch source
bindings, event evidence or envelope-estimate evidence, duration and units. Sum
energies only across equal observation intervals; sum average powers only across
the same operating point/interval. Reject misaligned durations rather than mixing
them. Converter total is the sum of computed branch contributions only when every
required physical contribution is computed. Otherwise return a known subtotal,
null total, and explicit omissions. Missing requested data rejects; omitted is
null with a reason; supported computed zero is numeric zero. An absent topology
role with multiplicity zero is explicitly inactive, not a missing device model.

Symmetry multiplicity means *identical loss behavior by an explicitly selected
symmetry assumption*. It is not equal current sharing among parallel dies. Step 6
assembly scaling and device-pair composition remain separate operations.

## Exact SC waveform mappings

### DAB

Source `matlab_func\DCDC_DAB\internalFunctions\calc_DAB_Psemi.m:16–48`.
Every row represents two symmetric physical devices:

| Role/model argument | Physical members | Count | Conduction current | Gate | Commutation current | Switching bus |
|---|---|---:|---|---|---|---|
| s12 / scData_s12 | s1,s2 | 2 | is2 | vgs2 | -itp | vp |
| s34 / scData_s34 | s3,s4 | 2 | is4 | vgs4 | +itp | vp |
| s56 / scData_s56 | s5,s6 | 2 | is6 | vgs6 | +its | vs |
| s78 / scData_s78 | s7,s8 | 2 | is8 | vgs8 | -its | vs |

Source reads vs2/vs4/vs6/vs8 for edge extraction but discards extracted voltages;
the actual energy scaling uses vp/vs. Distinguish these bus voltages from optional
measured device voltage plots. The commented factor-of-two correction at line 49
is disabled; do not enable it or double the multiplicity again.

This source still calls `calc_switching_voltage_current.m:22–31` (post-edge sample)
and `calc_Psw_Semi.m:25–33` (filters zero-current events, and wrongly uses the
turn-on reference voltage for Eoff). Preserve role/sign/count mapping, not those
defects. Use the explicit shared-engine pre-edge convention and independent Eon /
Eoff conditions; document that historical numerical results may therefore differ.
Do not infer commutation current sign from direction of transferred power.

### SRC-ZCS

Source `matlab_func\DCDC_SRC_ZCS\internalFunctions\calc_SRCZCS_Psemi.m:17–49`.

| Role/model argument | Physical members | Count | Conduction current | Switching current/gate/bus |
|---|---|---:|---|---|
| inverter_hb1 / invhb1_scData | s1,s2 | 2 | is1 | +itp / vgs1 / vp |
| inverter_hb2 / invhb2_scData | d3,d4 | 2 | idp1 | omitted in source |
| rectifier_hb1 / rechb1_scData | d5,d6 | 2 | ids1 | omitted in source |
| rectifier_hb2 / rechb2_scData | d7,d8 | 2 | ids3 | omitted in source |

Eight devices contribute conduction; only two contribute source switching.
Source returns a switching vector of length two, not an eight-element zero-padded
vector. Preserve the six omitted devices as explicit unsupported/omitted switching
contributions with the selected source assumption. Do not claim physically zero
recovery or zero energy solely because the topology has ZCS in its name. Np/Ns/vs
are read at lines 38–40 but do not participate in this function's losses.

### SRC-ZVS and LLC

Sources `matlab_func\DCDC_SRC_ZVS\internalFunctions\calc_SRCZVS_Psemi.m:17–50`
and `matlab_func\LLC\internalFunctions\calc_llc_Psemi.m:17–48` have the same
mapping and may share one preset implementation:

| Role | Members | Count | Conduction | Switching current / gate / bus |
|---|---|---:|---|---|
| inverter_hb1 | s1,s2 | 2 | is2 | -itp / vgs2 / vp |
| inverter_hb2 | s3,s4 | 2 | is4 | +itp / vgs4 / vp |
| rectifier_hb1 | s5,s6 | 2 | is6 | omitted in source |
| rectifier_hb2 | s7,s8 | 2 | is8 | omitted in source |

Source conduction vector has eight entries; switching vector has four. Secondary
switching/recovery is omitted, not proven zero. These use the v2 pre-edge helper:
`mfiles\calc_Psw_Semi_v2.m:27–42`, preserving E(0) and separate voltage references.
Archive v2 compares raw neighboring gate amplitudes, whereas the shared engine
thresholds the gate into states or accepts explicit masks. Noisy/analog gate input
must have an explicit threshold/mask and no undocumented debounce inference.

## Exact SC hard-switching-envelope mappings

Primitive source `mfiles\HardSwitchedSemiconductorLoss.m:1–7` computes
`D * I * V(I)`, and `(0<D<1) * fs * Eon/off(I) * Vbus/Vreference_on/off`.
The source actually tests D!=0 and D!=1 without validating [0,1]; the new module
must reject invalid duties and frequencies. Signed model selection remains
explicit. Archive extrapolation is not a supported recorded-data behavior.

The envelope path needs a named estimate mode sharing the common conduction and
energy evaluators. Do not feed line-frequency gate states into edge detection to
pretend they represent fs PWM. For an initially lean, unambiguous envelope input,
accept interval-held I,D,Vbus,fs over N-1 intervals with N increasing time edges;
energy is the sum of local power times dt. State this representation explicitly.
Interpolated envelopes can be added with independently specified integration and
duty-endpoint rules, not by inheriting a sample-count mean. Step 5's existing
analyze_waveform does not itself supply this frequency-envelope operation.

### Bridgeless PFC

Source `matlab_func\Bridgeless_PFC\calc_bridgeless_semiconductor_loss.m:25–94`.

| Role | Members/count | Current | Duty | Frequency | Bus | Notes |
|---|---|---|---|---|---|---|
| MOSFET | s1,s2 / 2 | +iac | dn1 | fs | vdc | reverse-on ideal envelope selected in source |
| PFC diode | idpfc1,idpfc2 / 2 | -iac | dp1 | fs | vdc | signed diode model required |
| reverse IGBT | isrev1,isrev2 / 2 | -iac | dp1 | 0 | vdc | active only if P<0 |

For P>=0 the source explicitly emits zero reverse-IGBT contributions. For P<0 it
needs IGBT data, despite later name assignment tolerating an empty array. Do not
carry forward that inconsistent optional-data behavior. Frequency zero makes
switching computed zero by the selected operating assumption, not an omission.
P<0 is an activation condition; it must not reverse any current sign again.

The accompanying .py starts with invalid MATLAB-style syntax at line 1 and is not
an executable alternative; use the MATLAB source behavior above as evidence.

### Six-switch / three-phase totem-pole PFC

Source `matlab_func\PFC-SixSwitch\three_phase_totempole_pfc_semiconductor_loss.m:29–92`.

| Role | Members/count | Current | Duty | Frequency | Bus |
|---|---|---|---|---|---|
| low MOSFETs | sna,snb,snc / 3 | +ia | dna | fs | vdc |
| high MOSFETs | spa,spb,spc / 3 | -ia | dpa | fs | vdc |
| neutral IGBTs | spn,snn / 2 | 3*ia | dpn | fl | vdc |

Both MOSFET roles use the reverse-on ideal envelope in the source. Neutral
commutation uses line frequency fl, not fs. The factor 3 on current is distinct
from the factor 2 device multiplicity; collapsing these changes I²R losses.
This mapping is the source's representative-phase symmetry assumption, not a
universal circuit identity for arbitrary unbalanced three-phase waveforms.

### VR / VRN / IVR / IVRN / TVR PFC

Source `matlab_func\PFC_IVR\calc_vr_semiconductor_loss.m:27–53,95–133`.

| Role | Current | Duty | Frequency | Bus | Count |
|---|---|---|---|---|---:|
| PFC MOSFET | +ipfc | dpfc | fs | vdcp | 6 |
| PFC diode | -ipfc | 1-dpfc | fs | vdcp | 6 |
| phase rectifier diode | +idrl | 1 | no switching at D=1 | source passes 5 V | see below |
| neutral rectifier diode | +idrn | 1 | no switching at D=1 | source passes 5 V | see below |

| Topology | Phase rectifier count | Neutral rectifier count |
|---|---:|---:|
| VR | 6 | 0 |
| VRN | 6 | 6 |
| IVR | 12 | 0 |
| IVRN | 12 | 6 |
| TVR | 0 | 0 |

Do not infer two neutral devices from the comment naming drn1/drn2: executable
count is six for VRN/IVRN. Source's arbitrary 5 V rectifier argument has no loss
effect because D=1 suppresses switching. Do not expose it as measured blocking
voltage or require switching curves for these inactive switching contributions.
Reject unknown topology identifiers. Source indexes device arrays by operating
point i; the new interface should bind named role models explicitly, avoiding
accidental pairing of a device configuration with a different operating point.

### Balanced two-level inverter

Source `mfiles\EXAMPLE_TwoLevelInverter.m:31–79`:
three phase-to-neutral voltages have amplitude sqrt(2/3)*Uac, with Uac the
line-to-line RMS. For phase theta, current is
`sqrt(2)*(P*cos(theta)+Q*sin(theta))/(sqrt(3)*Uac)`; phase offsets are 0,
-2pi/3 and -4pi/3. Negative-rail-clamped zero-sequence voltage is
`u0n=-min(ua,ub,uc)`, upper duty dpk=(uk+u0n)/Udc, lower dnk=1-dpk.

Source calculates representative phase-a upper current +ia/dpa and lower current
-ia/dna, both bus Udc and frequency fs, then multiplies each mean by three. These
are three upper plus three lower composed switch/diode positions. The example's
15 parallel IGBTs and five parallel diodes per position are an explicit assembly,
not an implicit additional multiplier in the topology preset.

For arbitrary supplied three-phase waveforms, prefer all six explicit branches
with each phase's own current and duty. Representative phase times three requires
explicit balanced periodic symmetry, a complete fundamental interval and correct
integration weights. Do not apply symmetry to unbalanced or partial records.

Optional ideal generator must reject nonpositive Udc/Uac/fs/fundamental frequency,
nonfinite P/Q and modulation outside duty [0,1]. It does not solve dead time,
switching ripple, harmonics caused by PWM, neutral imbalance or resonant dynamics.

## Independent fixtures and acceptance conditions

These are proposed algebraic fixtures, not tests already run against a new adapter.

1. **Primitive duty/frequency:** I=10 A, Vdrop=2 V, D=.25, fs=100 kHz;
   Eon=1 mJ at 400 V, Eoff=2 mJ at 600 V; bus=200 V. With explicitly selected
   linear voltage scaling, per-device conduction/on/off is 5, 50, 66.6666666667 W;
   total 121.6666666667 W. Default recorded mode must reject these voltage
   mismatches. Matching 200 V recorded tables of .5 mJ and 2/3 mJ give the same
   result without scaling. D=0 gives computed zeros; D=1 gives 20 W conduction
   and no switching; fs=0 preserves 5 W conduction and zero switching. Negative
   fs, D outside [0,1], missing active directional data and extrapolation reject.
2. **Time weighting:** two interval-held powers 10 W for 1 s and 30 W for 3 s
   give 25 W, not the unweighted sample mean 20 W. Splitting an interval without
   changing held values leaves energy and power unchanged. Extending a no-event
   zero-current tail changes event power through full duration, not event spacing.
3. **DAB mapping/count:** give each role distinct per-device computed triples
   (Pc,Pon,Poff)=(1,10,100),(2,20,200),(3,30,300),(4,40,400) W. Converter sums
   are 20,200,2000 W and 2220 W total. Expanded physical count is eight. Each
   pair's event evidence has exactly opposite intended primary/secondary signs
   according to the mapping, not four uses of |itp|. Feed is2 differing from -itp
   to ensure the conduction and commutation signals are not conflated.
4. **Pre-edge/finite-zero:** t=[0,1,2,3,4] us, gate=[0,0,1,1,0],
   I=[0,2,9,3,0] A selects on 2 A and off 3 A. Independently conditioned
   Eon=I*1 mJ at 400 V, Eoff=I*2 mJ at 600 V, bus200 with selected scaling
   gives 1/2 mJ, 250/500 W over 4 us. Set a genuine gate edge at I=0 and a
   finite E(0); it must remain an event and contribute finite energy.
5. **SRC omissions:** four distinct per-device conduction powers [1,2,3,4] W
   sum to20 W for all three presets. SRC-ZCS has two represented switching
   devices; SRC-ZVS/LLC have four. Omitted secondary/recovery contributions keep
   the complete total null with a known subtotal, unless the caller supplies
   additional real branches/data. A topology label never makes them zero.
6. **Bridgeless direction:** P>=0 makes the reverse IGBT inactive computed zero;
   P<0 with Vdrop=2 V, iac=-10 A, dp1=.25 gives per-IGBT5 W and pair10 W,
   switching0 at fs=0. Switching energies need not be present for this inactive
   switching mechanism. Keep the signed MOSFET/diode paths independently bound.
7. **Six-switch scale:** constant ia=10 A, neutral dpn=1 and R=.1 ohm gives
   per-neutral90 W and pair180 W from (3*ia)²R; a mistaken count-times-current
   substitution fails. With D=.5, Eon=Eoff=1 mJ and fl=50 Hz, neutral pair
   switching=.2 W, not fs-based power. Keep all model voltage conditions matched.
8. **VR counts:** set each phase/neutral diode's computed conduction to2/3 W;
   incremental rectifier powers are VR12, VRN30, IVR24, IVRN42, TVR0 W.
   Remove both rectifier model bindings for TVR and verify explicit inactivity
   succeeds; remove required phase diode data for VR and verify it rejects.
9. **Inverter balance:** Udc400 V, Uac200 V, P3000 W, Q0. Over a complete
   ideal fundamental period, sum phase currents and voltages is0; integrated
   sum uk*ik is3000 W. Current peak=sqrt(150) A, phase RMS=sqrt(75) A.
   Voltage span max(uk)-min(uk)<=sqrt(2)*Uac=282.8427125 V, so clamped duties
   remain [0,1], at least one upper duty is0 and its lower duty1 at each phase
   angle. With identical pure R=.1 ohm bidirectional positions and complementary
   duty, all six conduction contributions sum to R*sum(phase RMS²)=22.5 W.
   Explicit six-branch integration and balanced representative-phase times3
   must agree; shifting all phases or permuting phase labels preserves totals.
   P0,Q3000 gives the same conduction22.5 W but zero average active output power.
10. **Evidence/state:** source objects/input arrays stay unchanged; input results
    do not alias future edits. Role-source identifiers and independent Eon/Eoff
    conditions survive JSON export. Multiplicity scales energy/power/counts but
    not current amplitudes or per-device temperatures. Measured-zero, inactive,
    omitted and missing are distinguishable. Overflow rejects before transport.

## ENCO behavioral evidence and limits

Read `enco\packages\enco-analysis\src\enco_analysis\circuit\circuit\mosfet.py`:
`attach_physical_indicators:271–443` loops explicitly instantiated MOSFETs, reads
named ::vgs/::vds/::ids records, computes per-device conduction/switching/diode
losses, and sums the respective totals. This supports generic named-device
aggregation and linked evidence, rather than many converter-specific kernels.
Its disabled loss booleans become numeric zero and IDEAL models are skipped;
the new module must instead preserve the selected inactive/omitted scope.
The RDS_ON ratio reused as a switching worst-case factor is a heuristic, not a
measured switching-temperature law or a default scaling rule.

Read `enco\packages\enco-components\src\enco_components\components\mosfet\__init__.py`:
`get_conduction_loss:182–215` multiplies current by supplied vgs then takes RMS,
so it assumes a normalized gating signal; feeding physical gate volts would
scale loss by its square. New inputs must distinguish physical Vgs from duty or
binary channel mask. `get_switching_energy_dataframe:217–306` defaults factor
to0 and drops null energy rows; neither behavior is a valid missing-data policy.
`get_switching_loss:308–349` divides by the first-to-last transition span and
returns0 for fewer than two transitions; use complete supplied duration and
retain single events instead. Do not transplant these bodies or dependencies.

Read `library-design\dcdc\dab_cf\src\project\models\dab.py:47–128,151–170,266–272`:
twelve individually instantiated MOSFETs (four primary and eight secondary),
supplied time series, and postprocessing. This is a different configured circuit
from SC's eight-device DAB; do not silently substitute its count. Its soft-switch
loop overwrites the aggregate boolean per device rather than accumulating it.

Read `library-design\dcdc\cllc\src\project\models\cllc.py:41–119,142–161,260–266`:
twelve MOSFET instances and supplied waveform postprocessing; the same overwritten
soft-switch flag. Negative turn-on current or an arbitrary 2 A threshold is not
evidence of adequate Coss commutation charge or physical ZVS across all devices.

Read `library-design\dcdc\staedler_llc\src\project\models\llc.py:25–107,153–197`:
named cloned devices, inherited indicator generation, and transformer winding
postprocessing. It defines unused primary_mosfet but clones secondary_mosfet for
the twelve devices shown; later plot lists include names not instantiated there.
No verified analytical LLC waveform solver follows from this source.

## Existing current code to reuse or avoid

- `topologies/operating_point.py:20` supports buck, boost, buck-boost and flyback.
  It is not an implementation of the listed new topologies. Exact condition
  selection/lookup primitives are useful, but its nonnegative channel convention
  cannot replace step5 signed directional dispatch. Do not invoke topology solvers
  to generate substitute DAB/PFC waveforms.
- `topologies/converter_common.py` is legacy mesh/vector code, with old
  `switch.channel` attributes, nearest/highest conditions, clamps and transistor.wp
  mutation. Its class names do not establish an advanced validated common engine.
  Reusing the new shared waveform/table implementation is leaner and safer than
  adapting each old method. This read-only preparation does not edit legacy code.
- `waveform_losses.py` owns exact piecewise-linear conduction, pre-edge event
  evidence, full-duration normalization, explicit estimates/omissions and exported
  request snapshots. Converter adapters must consume those results for waveform
  mode; no independent browser science is required.
- Phase6 assembly and thermal calculations should consume per-position losses.
  Common-sink thermal paths cannot be modeled by simply adding the hottest
  single-device temperature or multiplying temperature rise by device count.

## Browser and transport acceptance

Use the shared waveform workbench with a converter preset selector, visible role
mapping table and operation selector (recorded edge waveforms versus duty/frequency
estimate). Let callers bind named supplied columns and device/assembly models.
Show current sign/scalar, symmetry multiplicity, switching bus, duty/frequency,
conditions and omitted mechanisms before calculating. Keep advanced data-entry
JSON as import/export support, not the only explanation of the physical mapping.

Results require per-role and expanded/member-count views, stacked conduction/on/
off known power, a clearly partial total when scope is omitted, and linked branch
signals/event evidence or envelope power. Editing any mapping, waveform, model or
assumption invalidates results; late replies cannot restore obsolete output.
Exports include supplied SI inputs, preset revision, mappings, source identities,
selected assumptions, calculation evidence, missing/omitted scope and totals.
FastAPI must call the reusable Python interface; browser sums are presentation
checks, not scientific authority. Test API equality, six-branch/symmetry mappings,
omitted-state display, stale requests, downloads and narrow-screen plots.

## Historical fixture limitations

SC test scripts load external waveform MAT files and mostly print/plot results.
The scoped file inventory did not find the named converter inputData waveform
fixtures under SC; the inverter semiconductor and heatsink MAT files do exist.
`DCDC_DAB\test_calc_DAB_Psemi.m:34` also calls an older argument layout than the
four-model function signature. These source examples are not runnable parity
proof. Use new deterministic algebraic fixtures above and later user-supplied
recorded waveforms; preserve reference files and never execute source applications
as part of implementation validation.
