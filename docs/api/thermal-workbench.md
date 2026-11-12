# Thermal workbench

`POST /api/thermal/analyze` and `thermal.analyze_thermal(request)` share the same
scientific operations. Open `/index.html?view=thermal` or More Tools → Thermal
workbench. All displayed initial values are editable hand fixtures, identified
as such; they do not establish a device measurement or rating.

The typed operation discriminator accepts `steady`, `transient`,
`electrothermal` or `screening`. Each request carries a nonblank `source` and a
`conditions` mapping. Known numeric conditions stay explicit and unknown values
remain JSON null. Responses retain an independent canonical request and result.
The browser exports both together; transient/iteration evidence also exports CSV.
Editing any scientific/source field clears results and exports and discards an
in-flight stale response. The advanced JSON editor is optional; ordinary forms
expose all four calculations.

The browser currently accepts explicitly entered exact segments and event rows.
Python/API callers can pass waveform power-segment dictionaries, but a direct
waveform JSON download → thermal browser import has not been implemented or
verified in this phase. Source/event evidence preservation and explicit heat
allocation in that handoff remain part of the later data-workflow phase.

## Steady physical paths

Required fields are `sink_temperature_c`, `forward_power_w`, `reverse_power_w`,
`common_resistance_k_w`, `forward_resistance_k_w` and
`reverse_resistance_k_w`. All powers and paths are explicitly nonnegative;
declared ideal zero is different from missing data. The two rises are:

```
forward: (Pf + Pr)*Rc + Pf*Rf
reverse: (Pf + Pr)*Rc + Pr*Rr
```

An optional `interface` specifies `path` (common/forward/reverse),
`resistivity_k_m2_w`, positive `area_m2`, source and conditions. Its resistance
is rho/area, added once to that chosen path. The old service signature accepts
neither a physical reference temperature nor a time-weighted heat profile;
`CalculationService.calculate_thermal_resistance` now raises a migration error
instead of mislabeling case-to-sink resistance as junction-to-case, averaging
unweighted samples and assuming 25°C.

### Simultaneous physical contacts

For multiple contacts, supply `interfaces` instead of a nonnull `interface`.
The list contains 1 to 32 records with the existing interface fields and a
required unique `interface_id`. Each record retains its own source and conditions.
Distinct contacts on the same path form a series stack. Repeated physical IDs
are rejected. The common path carries total heat. Each branch path carries only
that branch's heat.

```json
{
  "interface_id": "forward-contact-1",
  "path": "forward",
  "resistivity_k_m2_w": 0.001,
  "area_m2": 0.01,
  "source": "Explicit illustrative contact; replace with recorded data",
  "conditions": {"pressure_pa": null}
}
```

The calculation resolves each SI ratio before rounding its resistance.
It sums the exact ratios on each path before rounding the effective resistance.
Contact order does not change the result. Nonzero resistance outside the
supported finite range is rejected. A sourced zero resistivity remains an
explicit ideal contact. Missing base resistances remain invalid.

Plural results contain `interface_contributions`, `added_path_resistances_k_w`,
`path_resistances_k_w`, and `shared_temperature_c`, with both junction temperatures.
Each contact row includes its resolved `resistance_k_w` and original evidence.
The result does not combine unrelated branch resistances into one scalar.
Existing absent and single-interface requests retain their response format.

With a 25 °C sink, 2 W forward heat, 1 W reverse heat, and ideal base paths,
contact contributions of 1, 2, and 3 K/W on common, forward, and reverse paths
give 28 °C shared, 32 °C forward, and 31 °C reverse temperatures.
These values are an independent synthetic fixture, not a device rating.

The steady form provides contact rows with separate source and condition entry.
Blank condition values mean unknown. The contact CSV includes each contact and
its added and effective path resistance. JSON retains the complete request and
result. Contact edits invalidate results and downloads.
The advanced editor sends its original JSON text for finite-literal validation.
Thermal input retains its existing body-size policy and numerical work budgets.

## Fixed-case Foster transient

The network has complete positive aligned `r_th_vector` and `tau_vector`, or
`r_th_vector` and `c_th_vector` with tau=R*C. If both tau and capacity are given,
they must agree to 1e-9 relative. Optional recorded `r_th_total` must agree with
the pole sum to 1e-9 relative; there is no fixed absolute tolerance that hides
inconsistent tiny resistances. A total alone cannot define transient poles.
Both existing impedance adapters delegate to this shared validation and the
stable step response sum(R*(-expm1(-t/tau))). Negative/nonfinite time fails.

Required `network_heat_owner` identifies `forward`, `reverse` or an explicitly
co-located `shared-junction`. An event for another physical junction and a signed
segment for another branch are rejected unless the caller explicitly declares
that both heat sources share this one junction. This is one network with a
fixed case boundary, not a coupled dynamic heatsink model.

`initial_state` is `cold` at the first segment start, or one nonnegative
temperature rise in K per pole. A scalar initial junction temperature cannot
determine multiple pole states. `case_temperature_c`, `gap_policy` (`reject` or
explicit `zero-power`), `event_mode` and `samples_per_segment` are required.

Each ordered nonoverlapping segment provides `start_s`, `end_s` and one through
four ascending `power_coefficients_w`. For u=(t-start)/(end-start),
P(u)=c0+c1*u+c2*u²+c3*u³. Extra evidence fields emitted by the waveform engine
are accepted and retained. The full polynomial is integrated, including
current-zero and table-knot splits; plotting points never replace it with a
linear approximation. Nonnegative power is checked at endpoints and all real
stationary points. The disclosed roundoff allowance is
32*float64-epsilon*sum(abs(coefficients)) W, permitting physical negative
coefficients without permitting material negative heat.

For each pole, exact convolution uses Bk(h)=h*integral(exp(-h*(1-u))*u^k du),
h=dt/tau. For h<=1 an independently convergent series avoids cancellation;
for h>1 expand about the evaluation endpoint, v=1-u. With P(1-v)=sum(dk*v^k),
use J0=-expm1(-h), Jk=(k/h)*J(k-1)-exp(-h). Compensated coefficient sums retain
exact endpoint zeros such as P=(1-u)^3. Endpoint derivatives are evaluated from
the original segment coefficients before translating any event/averaging cut.
Near the source endpoint the implementation first forms the original global
q(v)=P(1-v) basis and evaluates at v=(original_end-target_time)/duration directly.
Both local and endpoint expansions use that basis near the end and the original
u basis near the start. This preserves a tiny cubic tail without subtracting
rounded powers near u=1, including cuts within one tau of the end. Adding a
zero-energy event cannot perturb that physical heat. This avoids subtraction of almost equal
original-basis moments for large h. Calculations use extended internal
arithmetic to keep intermediate time ratios and tiny increments representable,
and reject outputs outside finite SI representation. Returned floats are finite.
Positive impedance time/tau ratios below the platform's normal internal range
are explicitly rejected. For forcing moments the lower ratio bound is four
times the internal smallest normal value, keeping the cubic series terms normal.
Multiplication by R or P could make a lost subnormal increment representable;
the calculation never silently reports that increment as zero.
If a nonzero endpoint coefficient needs a moment below the internal smallest
normal value, the request explicitly fails rather than silently losing recoverable
heat. Endpoint coefficient overflow also fails. These are unsupported precision
regimes, not physical device limitations. No automatic time/unit rescaling or
endpoint approximation is used. There is no arbitrary upper time-ratio cap.
Only physical forcing changes and switching events advance the network state;
plot samples are evaluated independently from each physical interval's initial
state. The final response does not depend on plot sample density.
Any negative convolution or pole state produced from accepted nonnegative heat
is rejected as unresolved numerical precision; it is never reported as cooling
or silently clamped to zero.
Each displayed maximum is explicitly **sampled**, with no global maximum claim.

Switching events carry `time_s`, nonnegative `energy_j`, source, conditions and
`heat_owner` (forward/reverse), including E(0). `impulse` deposits R*E/tau per
pole and preserves pre/post samples. `interval-average` also requires `end_s`
and deposits all E uniformly over that explicit width. These are selected
temporal approximations, not inferred measured heat-pulse shapes. No-event mode
rejects supplied events instead of discarding switching heat. Conduction and
switching input energies are reported separately and each event is deposited
once. Gaps require an explicit zero-power convention.

Resource limits reject work rather than truncating: impedance <=1,000,000
pole/time products; transient <=100 poles, 10,000 input segments, 10,000 events,
100,000 expanded plotting samples; input segment/event and expanded
sample/pole or sample/averaging-event products <=2,000,000. HTTP rejects
resource limits with 413 and invalid/unsupported science with 422.

## Bounded steady electrothermal iteration

Supply one or two `loss_models`, explicit `initial_temperatures_c`,
`branch_resistances_k_w`, `common_resistance_k_w`, `sink_temperature_c`,
`tolerance_k`, `max_iterations` and `relaxation` in (0,1]. Each recorded model
contains aligned increasing `temperature_c`/`power_w` and coverage endpoints
matching that table. Its other source conditions stay fixed. Alternatively,
`linear-temperature-estimate` requires `selected:true`, a declared bounded
temperature domain, reference temperature/power and coefficient per K.

Each branch source is evaluated at its own temperature. Iteration solves
F(T)=Tsink+M*P(T)-T, M=diag(branch paths)+common path. Acceptance requires the
actual unrelaxed residual within tolerance, even when the relaxed step is tiny.
Statuses are `converged`, `iteration-limit` and `source-domain-exit`. Only the
first returns an `accepted_solution`; history and residuals remain inspectable
for the others. Domain exit is detected before evaluating data outside its
coverage. Failed fixed-point iteration is not proof that no physical solution
exists. No transient electrothermal coupling is claimed.

## Current screening

Supply `junction_limit_c`, `reference_temperature_c`, `reference_kind`
(case/heatsink/coolant), positive complete `path_resistance_k_w`,
`threshold_v`, `resistance_ohm`, and model conditions/source.
Pbudget=(Tjmax-Tref)/Rpath. For positive resistance, the positive current root
uses 2*Pbudget/(V0+sqrt(V0²+4*r*Pbudget)); zero slope and zero budget have
explicit branches. A zero-loss conduction model has no finite thermal current
limit (null), rather than an invented rating. Package current/applicability
can be reported separately and is never conflated with this thermal screen.
The browser explicitly records model temperature and unknown/known gate bias.
Screening does not solve self-heating.

## Numerical evidence

Owner fixtures in `tests/test_thermal_workbench.py` cover both branch paths,
interface units, exact Foster steps, capacity conversion, irregular forcing,
the cubic P=100u²-100u³, impulses/averaging, stable small/large dt/tau,
bounded coupled/single-branch fixed points, domain exit/nonconvergence,
stable screening roots, unknown metadata, strict finite validation and 413.
The convolution moments are also compared to independent adaptive quadrature.
Browser and independent review results are recorded separately in the parity
sprint; fixtures establish equations and interfaces, not measured-device accuracy.

## Native-device curve feedback

The optional native-device source uses explicit indexed channel and switching
families to derive a bounded temperature/loss table.
It passes that table to the existing one-junction fixed-sink iteration.
See [native-device electrothermal feedback](device-electrothermal.md) for source
identity, fixed conditions, inactive terms, bounds, and replay details.
