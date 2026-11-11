# Post-Miller gate settling estimate

`GateChargeModel.calc_switching_estimate` and `POST /api/analytical/gate_charge`
accept an optional `settling_target_v`. The existing gate charge calculator
provides the same opt-in target field. Omit the field to retain the previous
complete response, including both switching direction traces and timings.

The target satisfies `Vplateau < target < Von`. The total gate resistance is
positive. The model retains the existing constant-capacitance assumptions.
`Qg` remains measured at `Von`; the target does not redefine its charge reference.
This phase estimates gate charging. It does not establish physical drain
waveforms, drain switching energy, or nonlinear capacitance behavior.

For residual charge `Qpost = Qg - Qgs - Qgd` and total resistance `R`:

```text
Cpost = Qpost / (Von - Vplateau)
additional charge = Cpost * (target - Vplateau)
additional duration = R*Cpost*log1p((target - Vplateau)/(Von - target))
gate current = (Von - Vgate)/R
```

The optional phase accepts binary64 input values. Direct Python values with
additional precision are rejected if conversion would change their value.
Optional-mode HTTP validation also inspects the original numeric tokens.
Canonical-equivalent decimal notation is accepted. Numeric strings and lossy
literals are rejected. A residual-charge sign or zero/nonzero classification
change at binary64 precision is rejected. The no-target transport stays unchanged.
The calculation uses exact input charge and voltage differences, then extended
intermediate arithmetic. The positive-gap logarithm remains defined near either
boundary. Unsupported nonzero SI results are rejected instead of becoming zero.
The gate voltage trace uses `Vplateau + span*(-expm1(-relative_time/(R*Cpost)))`.

The response adds `estimate.turn_on_settling` as a separate object.
Existing `turn_on` and `turn_off` records remain unchanged.
`anchor_s` identifies the existing Miller endpoint.
`duration_s` and `relative_time_s` describe time after that anchor.
If a positive duration cannot advance the absolute binary64 timestamp,
`absolute_end_s` is null and `absolute_time_status` is `unresolved-use-relative-time`.
The local duration remains positive. This is not an instantaneous physical phase.

When the exact residual charge is zero, the response reports an
`instantaneous-ideal-crossing`. Duration and incremental charge are zero.
The two voltage endpoints share relative time zero.
Gate current endpoints and the current trace are null because no finite-time
current pulse is defined. Zero residual charge is explicit, not missing data.
A zero total gate resistance remains unsupported by this model.

For a 10 V rail, 4 V plateau, 7 V target, 2 ohm total resistance, and 1 nF
post-plateau capacitance, the additional duration is `2*ln(2)` ns.
The incremental charge is 3 nC. Gate current changes from 3 A to 1.5 A.
These values form an independent calculated fixture, not a device measurement.

The browser shows the separate phase, its relative time basis, and the absolute
timestamp availability. It plots gate voltage and finite gate current only.
JSON retains the model parameters and complete result.
The CSV retains relative timestamps, the anchor, target, charge, scope, and status.
Target, model, and selection edits invalidate results and exports, including
responses from an earlier request that arrives after an edit.
