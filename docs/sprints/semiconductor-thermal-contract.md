# Semiconductor assembly and thermal calculation contract

Prepared 2026-10-16 as a read-only coordination handoff for step 6 of
`sprint-semiconductor-function-parity.md`. This note does not advance or complete
that step. Step 5 remains subject to its own verification and approval state.
No production implementation or test suite was changed while preparing this note.

## Evidence and scope

Inspected reference files, without execution or modification:

- `\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc\matlab_func\ParallelSemiconductors.m`
- `\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc\matlab_func\ParallelIGBTandDiode.m`
- `\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc\matlab_func\reverseTurnedOnMOSFET.m`
- `\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc\matlab_func\SemiconductorTemperature.m`

Current evidence includes `core/models.py`, `core/pair_models.py`,
`core/services.py`, `backend/concrete_services.py`, the thermal result adapter in
`topologies/operating_point.py`, and the workbook audit
`semiconductor-workbook-function-audit.md`.

Scientific contracts below are proposed independent implementations of inspected
behavior with explicit corrections. Existing model declarations and matching
function names do not establish verified implementation coverage. The archive's
steady resistance calculation is distinct from a transient Foster network.

## Proposed edit boundaries

- `transistordatabase/semiconductor_assemblies.py`: small immutable curve/data
  transformations for equal sharing, signed composites and the ideal envelope.
  Reuse existing curve and condition primitives; avoid a second device hierarchy.
- `transistordatabase/thermal.py`: validated impedance, forcing, steady paths,
  screening and explicit electrothermal iteration.
- `core/models.py:FosterThermalModel.get_thermal_impedance` and the corresponding
  concrete-service method: thin delegation to the shared validated primitive.
- Web schemas, routes and components: adapters consuming Python results, with
  explicit estimate selection, stale-result invalidation and reproducible exports.

Keep `core/services.py` as the single ABC authority. Do not create new ABCs for
these calculations. Do not run, maintain or depend on Qt. All reference trees
remain read-only.

## Equal-sharing parallel bank

Proposed operation: `parallel_bank(device, count)`. Require a non-Boolean integer
count at least one and preserve the input, including its arrays and provenance.

For N identical devices with ideal equal current sharing:

```text
Vbank(I) = Vsingle(I/N)
Ebank(I) = N * Esingle(I/N)
```

Transform current axes and current ratings by N. Keep conduction voltage values
and blocking voltage unchanged. Scale event-energy values, area, cost and weight
by N. Missing values remain unavailable rather than becoming zero.

Independent internal thermal resistances scale by 1/N. A Foster bank has R/N and
unchanged time constants, corresponding to equivalent capacities N*C. Interface
resistance computed from an area-specific resistivity and N-fold independent
contact area also scales by 1/N. A shared external heatsink resistance remains
explicit and unchanged; do not divide it merely because the bank count changed.

Fixture: three devices described by 10 A, 1 mJ, 1 K/W, .01 m² and cost 2 produce
30 A, 3 mJ, 1/3 K/W, .03 m² and cost 6. Blocking voltage and curve conditions
remain unchanged. Count one must be numerically identical and still not alias
mutable source data. This is an equal-sharing estimate, not a balancing solver.

## Signed IGBT/diode composite

Preserve separate positive IGBT and negative diode branches, conditions and
sources. The blocking rating is the smaller rating; known cost and weight add.
Do not interpolate across a discontinuous zero-current branch boundary.

The archive rescales both energy curves to the smaller reference voltage. The
new default should require matching recorded conditions. Different reference
voltages need an explicitly selected linear-voltage estimate with recorded
scaling factors and requested/recorded conditions. Preserve turn-on and turn-off
conditions separately.

Common thermal resistance must be supplied explicitly. Inheriting the IGBT's
common path does not establish the geometry of an assembled IGBT/diode pair.
This operation selects signed branches, not a nonlinear current-sharing solution.

## Ideal reverse-on MOSFET envelope

For negative current, reproduce the inspected intent:

```text
Vreverse(I) = max(Vdiode(I), -Vforward(-I)), I < 0
```

In positive current-magnitude coordinates this chooses the smaller voltage-drop
magnitude. Restrict the operation to the overlapping recorded current domains.
Preserve source knots and add exact crossings of competing linear pieces. A fixed
101-point resampling, as used in the archive, can lose the crossover and should
not define the new result.

Fixture: forward drop `.2*j` and diode drop `.8+.02*j`, with j in [0,10] A.
The ideal envelope changes branch at j=40/9 A. Its sign is negative when returned
on a negative-current axis. Label this as an ideal lower-drop envelope; it does
not calculate simultaneous channel and diode sharing.

## Steady common and branch thermal paths

Use explicit nonnegative branch heat Pf and Pr:

```text
Tforward = Tsink + (Pf+Pr)*Rcommon + Pf*Rforward
Treverse = Tsink + (Pf+Pr)*Rcommon + Pr*Rreverse
```

An interface adds `rho_interface / area`, with rho in K*m²/W and area in m².
Require finite inputs and positive area when the interface term is used.
Missing path values must not become zero. Explicitly declared ideal zero
resistance can be supported separately from unavailable imported defaults.

Fixtures:

- Pf=10 W, Pr=0, Rc=.2 K/W, Rf=1 K/W, Rr=2 K/W gives rises 12 K and 2 K.
- Pf=0, Pr=10 W gives rises 2 K and 22 K.
- rho=.001 K*m²/W and area=.01 m² add .1 K/W to the selected path.

The archive allocates a zero-current nonzero loss fully to both branches. Correct
this by requiring a heat owner or allocation for zero-current heat, including
finite E(0). Direction alone is insufficient at zero. Do not double-count it.

## Fixed-case Foster response

For every pole, use the linear state equation:

```text
dx/dt = (R*P(t) - x)/tau
Tj = Tcase + sum(x)
Zth(t) = sum(R * (1-exp(-t/tau)))
```

Require complete aligned nonempty arrays of finite positive R and tau. Use
`-expm1(-t/tau)` for the step response. Reject negative/non-finite time. Total
resistance alone supports steady-state calculations, not a transient curve.
If capacities are an accepted alternative, require complete positive R/C arrays
and explicitly derive each tau=R*C; do not infer a multi-pole network from totals.
Validate any recorded total against the branch sum without silently rescaling.

Initial state must be explicitly cold at the fixed case or supplied per pole.
One initial junction temperature does not determine all pole states. Reject
overlapping or out-of-order forcing intervals. Any gaps require a declared
zero-power convention. Every returned temperature and intermediate state must
remain finite.

Vendor junction-to-case Foster data retains a fixed-case boundary. Do not silently
cascade fitted Foster blocks into an arbitrary moving-case/heatsink network.
The steady common/branch network above does not imply such a transient network.

### Exact polynomial forcing contract

Coordination with the step-5 waveform implementer established this segment shape:

```text
{
  start_s, end_s, sample_interval_index,
  direction: 'positive' | 'negative' | 'zero',
  power_coefficients_w: [c0,c1,c2,c3],
  energy_j, cumulative_start_j, cumulative_end_j, source
}
```

The coefficients are ascending powers of normalized local time:

```text
u = (t-start_s)/(end_s-start_s)
P(u) = c0 + c1*u + c2*u² + c3*u³
```

The waveform engine splits at zero crossings and recorded-table knots. Its
product of current, piecewise-linear voltage drop and interpolated duty can be
cubic. Thermal consumption must preserve those exact segments rather than
silently replacing them with linear interpolation between plotted samples.

For delta_t=end_s-start_s and h=delta_t/tau, derive:

```text
xend = exp(-h)*xstart + R*sum(ck*Bk(h))
Bk(h) = h * integral_0^1 exp(-h*(1-u))*u^k du
B0(h) = -expm1(-h)
Bk(h) = 1 - k*B(k-1)(h)/h
```

Use the recurrence only for h>1. For h<=1, use the cancellation-safe convergent
series independently for each k:

```text
term0 = h/(k+1)
term(m+1) = -h*term(m)/(k+m+2)
Bk = sum(termm)
```

At an interior fraction u, the exact state is:

```text
x(u) = exp(-h*u)*xstart + R*sum(ck*u^k*Bk(h*u))
```

The h=0 limit is zero forcing increment. Define behavior for underflow/overflow
of the dimensionless ratio explicitly; do not return non-finite states from
otherwise accepted finite inputs. Endpoint temperatures do not prove a global
transient maximum. If a plot samples the interior, label its reported maximum
as sampled unless a separate extrema calculation establishes more.

Physical heat must be nonnegative throughout a segment. Validate polynomial
values at both endpoints and every real stationary point in [0,1]. Negative
coefficients are permitted; checking coefficient signs is incorrect. For example,
`[0,0,100,-100]` gives nonnegative heat. Declare any numerical roundoff tolerance
used by this validation rather than silently accepting material negative heat.

The derived moment formula was independently checked against 96-point
Gauss-Legendre integration for h in {1e-12,1e-6,.01,1,10,100}, powers zero through
three. The largest relative difference was 1.3e-13. This was a mathematical
preparation check, not a production-code test or validation of a device model.

### Independent forcing fixtures

1. R=[.2,.3] K/W, tau=[.01,.1] s, P=100 W, Tcase=25 °C, cold initial state:
   - Tj(.01 s)=40.49728863549237 °C.
   - Tj(.1 s)=63.962708766261485 °C.
   - Infinite-time limit=75 °C.
2. Same network: 100 W on [0,.01] s, zero on [.01,.037] s, 50 W on [.037,.1] s.
   At .1 s the pole rises are 9.983197149716196 K and 8.171829541971245 K;
   Tj=43.15502669168744 °C. For each pole the closed form is
   `R*(100*(1-exp(-.01/tau))*exp(-.09/tau)+50*(1-exp(-.063/tau)))`.
3. One pole R=1 K/W, tau=1 s, cold start, interval [0,1] s, polynomial
   coefficients [0,0,100,-100] W. Input energy=25/3 J. Final temperature rise is
   `100*(3-8/e)=5.696447062846133 K`.

## Switching-event heat

The waveform contract keeps events separate:

```text
{
  event: 'turn_on' | 'turn_off', sample_index, bracket_s: [ti,ti+1],
  time_s: ti, current_a, direction, energy_j, source, mode,
  recorded_conditions, requested_conditions
}
```

Its current sampling convention is pre-edge. Conduction energy and event jumps
are separate evidence, not an undisclosed finite-width pulse.

When thermal input includes these events, require an explicit convention:

- Ideal impulse deposition: each pole state jumps by `R*E/tau`, then decays
  exponentially. Preserve pre/post-event samples if plotting the discontinuity.
- Explicit averaging over stated intervals: conserve the complete event energy
  and report interval widths and branch allocations.

Do not silently omit switching heat. Require a branch owner/allocation for E(0).
Using measured event energy with an impulse is a declared temporal approximation,
not a measured thermal pulse shape.

Fixture: in the irregular forcing example, an additional .1 J impulse at .037 s
adds `sum(R*.1/tau*exp(-.063/tau))=0.16345014985612696 K` at .1 s.

## Fixed-sink steady electrothermal iteration

Evaluate losses at each branch's own temperature using bounded recorded data or
an explicitly selected temperature estimate. Keep every other source condition
exact. No nearest-temperature substitution, silent extrapolation, invented
resistance or zero-loss fallback is permitted.

For the two-branch network, solve:

```text
M = [[Rc+Rf, Rc], [Rc, Rc+Rr]]
F(T) = Tsink + M*P(T) - T
```

Return finite iteration history, actual residual, tolerances, sources, estimate
scope and an explicit status. Suggested statuses are converged, iteration limit
and source-domain exit. Require a positive finite tolerance, integer iteration
limit and, if used, relaxation in (0,1]. A small relaxed temperature update alone
does not establish convergence: also test the actual fixed-point residual.
Do not report an unconverged last iterate as an accepted junction temperature.
Failed iteration is not proof that no physical solution exists.

Fixtures:

- Single branch: Tsink=25 °C, I=10 A, R25=.1 ohm, alpha=.004/K and thermal
  resistance=.5 K/W. For R(T)=R25*(1+alpha*(T-25)), the solution is
  Tj=30.10204081632653 °C and loss=10.204081632653061 W.
- Coupled: Rc=.2, Rf=1, Rr=2 K/W, Tsink=25 °C;
  P1=10*(1+.01*(T1-25)), P2=5*(1+.02*(T2-25)). Exact linear-system solution:
  T1=40.160349854227405 °C, T2=42.05539358600583 °C;
  P1=11.51603498542274 W, P2=6.705539358600583 W.
- Domain/runaway example: I=10 A, R25=.1 ohm, alpha=.04/K, Rth=3 K/W,
  source domain 25 through 175 °C, initial temperature 25 °C. Unrelaxed iterates
  are 25,55,91,134.2,186.04 °C. Report domain exit before evaluating a source
  outside its coverage. A deliberately small iteration limit must also return
  explicit nonconvergence in an otherwise convergent fixture.

## Workbook thermal screening

Expose explicit reference-path power budgets:

```text
Pbudget = (Tj_max - T_reference)/R_path
```

Require finite inputs, positive selected path resistance and Tj_max>=T_reference.
Record whether the reference is case, heatsink or coolant and select the matching
physical resistance path. Preserve the source and every explicitly added path
estimate; row-specific workbook additions are not universal thermal laws.

For threshold-plus-resistance conduction, solve `r*I²+V0*I=Pbudget` using the
cancellation-resistant positive root when r>0:

```text
I = 2*Pbudget / (V0 + sqrt(V0² + 4*r*Pbudget))
```

Handle zero budget and r=0 separately. For r=0,V0>0 use I=Pbudget/V0. A zero-loss
model does not produce a finite thermal current limit. Keep the model's
temperature/gate conditions explicit; this screening operation does not itself
solve self-heating. Return package limits and their applicability separately.
Do not present the thermal screening current as a datasheet current rating.
Area-times-resistance products remain dimensional screening indices, not cooling
geometry or transient results.

## Existing defects and compatibility limits

- `core/models.py:FosterThermalModel.get_thermal_impedance` returns total R as a
  constant when time constants are absent, including at t=0. It lacks complete
  physical/time validation.
- `backend/concrete_services.py:calculate_thermal_impedance` also substitutes a
  constant total resistance, defaulting missing data to zero.
- Its `calculate_thermal_resistance` mislabels r_th_cs as r_th_jc, takes an
  unweighted sample mean without time, and invents a 25 °C reference. It cannot
  represent the new contract without explicit inputs or a separate adapter.
- `core/pair_models.py` declares thermal and parallel fields, but declarations and
  count properties are not assembly or thermal algorithms. Default zero values
  must not be mistaken for recorded ideal paths.
- `topologies/operating_point.py` has a strict existing partial steady estimate;
  preserve its availability distinctions when sharing primitives. Avoid changing
  unrelated topology calculation behavior during this step.
- Source SemiconductorTemperature is steady algebra. Reverse-on and composite
  operations do not solve nonlinear electrical current sharing. Equal-sharing
  current and thermal scaling do not establish practical parallel stability.

## Verification required before accepting step 6

Verify source immutability; count-one identity; parallel current/energy axes and
external-common-path behavior; signed composite ownership and source conditions;
exact reverse-envelope crossings and bounded domains; common/branch/interface
units; missing-versus-declared-zero distinctions; Foster step and irregular/cubic
forcing fixtures; interior negative-polynomial rejection; very small and large
delta_t/tau; explicit initial states and event heat; bounded electrothermal
convergence, residuals and nonconvergence; screening zero-slope/zero-budget
boundaries; and browser plots/exports that use the same Python results.

Only subsequent implementation, numerical/API/browser checks and independent
review can establish completion. This handoff records the proposed contract and
derived fixtures, not measured-device accuracy or completed migration coverage.
