# Christen–Biela half-bridge estimate

The reusable implementation is `transistordatabase/biela.py`; the existing public
imports from `transistordatabase.analytical_models` remain available. All Python
and API quantities use SI units. The browser converts selected quantities for
display and preserves the input conditions and source in its JSON export.

This is the phase approximation from Christen and Biela, “Analytical Switching
Loss Modeling Based on Datasheet Parameters for MOSFETs in a Half-Bridge,”
DOI [10.1109/TPEL.2018.2851068](https://doi.org/10.1109/TPEL.2018.2851068).
The actual author manuscript was inspected through the
[author-text mirror](https://www.scribd.com/document/731572495/Analytical-Switching-Loss-Modeling-Based-on-Datasheet-Parameters-for-Mosfets-in-a-Half-Bridge).
Corrections follow the [ETH supplement](https://www.research-collection.ethz.ch/bitstreams/c8e37d49-333d-45f7-b92e-2d9e15822b9e/download).
The supplement's six corrections concern off-gate voltage signs, the channel
current used to update gm, the diode-current time intercept, the positive sign
of the turn-on quadratic, the reciprocal lifetime relation, and Qoss for one
device. The source reference does not validate the accuracy of arbitrary manual
parameters or the numerical fixtures in this repository.

The implementation rejects unsupported phases, insufficient data coverage and
numerical failure. An unsupported estimate does not show that a real device or
circuit is physically impossible. A failed fixed-point iteration can occur even
when the implicit nonlinear equation has a root; failure is not proof that no
model solution exists.

## Parameter definition and domains

The fitted transfer characteristic is equation (36):

`Ich = k1 * (Vgs - Vth)^x + k2`.

Its equation (37) gm is the **secant** `Ich/(Vgs-Vth)`, hence
`gm(Ich) = Ich * (k1/(Ich-k2))^(1/x)`. It is not the differential derivative of
the transfer characteristic. The implementation requires positive finite Ich
greater than k2. A transfer fit requires a known Vth, at least four distinct gate
voltages, exact common Tj and an explicitly chosen common Vds covered by every
selected output curve. It checks nonlinear solver success and Jacobian rank and
retains current bounds, residuals and source conditions. There is no inferred
threshold, varying median Vds, nearest-temperature choice or fallback linear fit.

The capacitance helpers delegate to the shared bounded piecewise-linear C(V)
integration. The entire interval `[0,V0]` must be recorded. `Qoss` is the charge
of **one** switch; `Cgs=Ciss-Crss`, `Cds=Coss-Crss`, `Cgd=Crss` must be physically
nonnegative. The Biela model requires strictly positive capacitances and Qoss,
and `Qoss=(Cgd+Cds)*V0` to within a disclosed 1% relative source-rounding
tolerance. Larger disagreement is rejected. The exact relative difference is
returned in diagnostics. The ZVS boundary uses the same Qoss as equation (10),
so accepted source rounding does not make the boundary inconsistent with that
quadratic; equation (14) is recovered when the charge equality is exact.
No extrapolated constant endpoint, negative-value clamp or invented parameter
is substituted. The web request requires the Qoss reference voltage to equal
Vdc and the parameter temperature to equal the requested Tj. It performs no
temperature or voltage scaling. A current sweep holds those conditions fixed.
`from_transistor()` retains the extraction voltage, temperature, device source
and fit evidence in the model. It rejects another bus voltage. The optional
`HalfBridgeParams.t_j` checks a supplied requested temperature against the stored
condition; omitted Tj uses the extracted model's stated condition, with no
temperature interpolation. A manual model without reference metadata cannot
establish a device temperature; callers remain responsible for its provenance.

## Corrected switching phases

For both Miller events equation (10) is

`a*Ioss² + b*Ioss + c = 0`,
`a=2*Ls/(Qoss*Rg)`, `b=2/(gm*Rg)+Cgd/(Cgd+Cds)`,
`c=(Vgate-Vth-I0/gm)/Rg`.

The root `-2*c/(b+sqrt(b²-4*a*c))` is continuous from the Ls=0 limit.
Turn-off Ioss is positive; turn-on Ioss is negative. Each iteration evaluates
`gm(I0-2*Ioss)` and checks both relative gm convergence and the final equation
residual. A negative discriminant at an intermediate gm iterate is reported as
a numerical failure; it neither establishes ZVS nor proves the implicit
nonlinear equation has no root. Failure to converge or leaving the transfer
domain during iteration also returns a numerical failure. No channel-current
floor is used.

Equations (13), corrected (15), (16) and (17) give turn-off voltage-rise time,
current-fall time, drain overvoltage and energy. The turn-off gate rail is used
in the logarithm. Equation (14)'s ZVS boundary is evaluated with a stable Ls=0
limit. This implementation rejects complete hard-switching calculations at or
below that boundary; zero-current Eon is not invented as zero.

Turn-on phase 1b uses `gm(I0)` and manuscript equation (18):
`tri=-ln(1-I0/[gm(I0)*(Von-Vth)])*(Cgs*Rg+Ls*gm(I0))`.
The gate drive must support that plateau. The phase voltage is
`Vds0=V0-Ld*I0/tri > 0`. Phase 3b separately solves the corrected positive
quadratic with gm evaluated at `I0-2*Ioss_on`. Manuscript equation (30) is
`tfv=-Qoss/Ioss_on`, with the single-device charge convention. The turn-on
energy partitions in manuscript equation (31) are

`0.5*tri*Vds0*I0 + trs*Vds0*I0 + Ers + 0.5*tfv*(I0-2*Ioss_on)*Vds0 + Erf`.

## Recovery extraction and energy

### Browser extraction and explicit application

1. Open **Analytical Models → Device Parameter Operations**.
2. Select **Nonlinear transfer fit for Biela** or **Reverse-recovery charge-control extraction**.
3. Replace the illustrative source, conditions and samples with recorded device data.
4. Calculate the parameters.
5. Inspect the diagnostics, coverage and source conditions.
6. Select **Review extracted parameters in Biela**.
7. Review the current and extracted values.
8. Select **Apply these extracted parameters**.

Transfer curves contain paired Vds/current samples for each gate voltage.
Multiple samples use bounded Vds interpolation. A single sample must record the exact selected Vds.
The fit retains the actual transfer Vds. Application does not change the model bus voltage.

Recovery fields display charge in nC, slope in GA/s and optional Trr in ns.
The request and result export use SI units.
Recovery Qoss records its own reference voltage and temperature.
These conditions must match the recovery test conditions.
Application selects charge-control recovery and copies only the extracted constants.
It does not change the model charge, capacitances, bus voltage or temperature.
Provide compatible model inputs before calculation. The model still checks charge-equivalent capacitance consistency.

Input edits invalidate extraction results and withdraw a pending Biela application.
An applied source record remains an immutable snapshot of its original inputs.
Model parameter edits mark the corresponding extraction evidence as historical.
The Biela request includes original extraction inputs and an `applied` or `edited` status.
The service recalculates those inputs. It checks applied parameters and required source/model conditions.
Both result exports retain the original source, samples, test current, conditions and diagnostics.
An edited record does not establish calibration of the current manual parameters.
The CSV download uses exact SI values from the same Python result as the tables and plots.
Transfer rows contain gate voltage, recorded/fitted current, residual and secant gm, with their source conditions.
The recovery row contains original test quantities, calculated constants, residual diagnostics and separate charge reference conditions.
CSV text cells escape spreadsheet formula markers. Numeric cells retain their values. JSON retains source text without CSV formula escaping.

### Typed extraction API

Send either operation to `POST /api/parameters/calculate`.
Transfer inputs use `operation="transfer-fit"`, `source`, `conditions={"v_ds_v":10,"t_j_c":25}`,
`v_th_v` and `curves`.
Each curve contains `v_g_v`, `t_j_c`, `voltage_v` and `current_a`.
Supply at least four distinct above-threshold gate voltages.
The result includes exact recorded/fitted/residual plot rows, secant gm, coverage and fit diagnostics.

Recovery inputs use this SI payload:

```json
{
  "operation": "reverse-recovery-extraction",
  "source": "Independent algebra verification fixture; not measured device data",
  "conditions": {
    "v_supply_v": 100,
    "t_j_c": 25,
    "q_oss_v_ref_v": 100,
    "q_oss_t_j_c": 25
  },
  "q_rr_c": 1e-7,
  "i_rr_a": 10,
  "di_dt_a_s": 2.4e9,
  "q_oss_c": 1.2e-8,
  "i_forward_test_a": 20,
  "t_rr_s": null
}
```

The result includes corrected charge, recovery constants, charge and lifetime residual diagnostics, and equation scope.
Unknown required extraction conditions, nonfinite values and boolean numeric inputs produce HTTP 422.
The Biela request also rejects boolean numeric values and non-text source identifiers.
The transfer adapter permits at most 100 curves and 10000 total recorded points.

For Biela application, send `extraction_evidence` as a list of records:
`{"inputs": <original extraction request>, "status": "applied"}`.
Supply the corresponding `result.biela_parameters` as explicit Biela input fields.
Use `status="edited"` when manual parameters replace those extracted values.
The Biela response includes recalculated complete extraction envelopes and their evidence status.

Recovery is explicitly omitted or explicitly selected; absent data never becomes
an unnoticed zero-recovery assumption. The stored parameter object requires
positive finite constants satisfying corrected equation (41):
`1/Tm=1/tau_rr-1/tau_c`, with `tau_c>tau_rr`.

Extraction (38)–(41) requires the same-condition forward test current in addition
to Qrr, Irr, slope and Qoss. Set `Q*=Qrr-Qoss`,
`Qrf=Q*-Irr²/(2*slope)`, `tau_rr=Qrf/Irr`,
`T1=(Iforward_test+Irr)/slope`. Solve
`Irr=slope*(tau_c-tau_rr)*(1-exp(-T1/tau_c))` in a dimensionless variable.
Nonpositive corrected charges and missing test current are rejected. The
datasheet threshold-defined trr is retained as a compatibility argument but is
not used in the infinite-exponential-tail model.

At the requested operating point, `s=I0/tri` and `T0=I0/s`.
The charge-control relation is
`qm(t)=s*tau_c*(T0+tau_c-t-tau_c*exp(-t/tau_c))`.
Solve `qm(T1)+Tm*(I0-s*T1)=0`, then
`trs=T1-T0`, `Irr=s*trs`, `Qrs=0.5*trs*Irr`, `Ers=Qrs*Vds0`.

Manuscript equation (27), independently checked by numerical integration of the
linear drain-voltage fall times the exponential recovery current, is

`Erf=Irr*Vds0*(tau_rr/tfv)*(tfv-tau_rr+tau_rr*exp(-tfv/tau_rr))`.

The complementary body diode's energy is reported separately, not added to the
switch's Eon. Equation (28) is

`Ediode=Irr*Vds0*(tau_rr/tfv)*(tau_rr-(tau_rr+tfv)*exp(-tfv/tau_rr))`
`       + tau_rr*Irr*V0*exp(-tfv/tau_rr)`.

The remaining tail sees V0 after the fall; it does not see Vds0 when Ld is
nonzero. A dedicated numerical-integral regression covers that distinction.

## Verification and limitations

The constant-gm algebra fixture uses Cgs=Cgd=Cds=1 nF, Qoss=200 nC,
k1=10, k2=0, x=1, Vth=3 V, V0=100 V, I0=10 A, Rg=10 Ω,
Von/Voff=15/0 V, Ls=Ld=0. It gives Eon=67.7077841576754 µJ,
Eoff=111.05118267395518 µJ and the ZVS boundary 1.2 A. With tau_rr=2 ns,
tau_c=5 ns and Tm=10/3 ns, Eon=71.056924268450 µJ. These verify equations,
units and energy partitions; they are not measured-device accuracy results.

The previous “C2M-like” approximate test constants produce a negative
discriminant in the turn-on fixed-point iteration under the corrected positive
quadratic. The new regression asserts an explicit numerical failure and no
longer labels a wide positivity band as paper parity. A separate fixture with
x=0.1 has a verified implicit nonlinear root despite the failed initial
quadratic; this guards the distinction between solver failure and nonexistence.

Supported browser output includes all phase quantities, energy partitions,
iteration counts and residuals, fixed conditions, source, assumptions and JSON
export. Sweep points retain `supported`, `unsupported` or `numerical_failure`
status with exact diagnostics. An input edit invalidates the result and prevents
an older response from restoring stale output. This work does not validate
ringing, waveform shape, nonlinear parameter temperature dependence, avalanche,
soft switching or arbitrary devices against experiment.
The browser rejects energy magnitudes whose separate µJ display conversion
would overflow; the SI API/Python model remains available at other scales.
