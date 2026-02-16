# PRD: Christen-Biela Analytical Switching Loss Model

## Reference

- **Paper**: D. Christen, J. Biela, "Analytical Switching Loss Modelling based on Datasheet Parameters for MOSFETs in a Half-Bridge", IEEE Transactions on Power Electronics, vol. 34, no. 4, Apr 2019.
- **DOI**: 10.1109/TPEL.2018.2851068
- **Supplement**: Errata corrige with 6 equation corrections and implementation hints.
- **Local copies**:
  - `/home/tinix/claude_wsl/ntbees2/docs/reference/Transaction_final_.pdf`
  - `/home/tinix/claude_wsl/ntbees2/docs/reference/Supplement_file_final.pdf`

---

## 1. Problem Statement

The existing `BielaModel` in `analytical_models.py` is a simplified overlap-based switching loss estimator. It uses constant Q_g and C_oss values and simple triangular overlap integrals. It does **not** model:

- The half-bridge topology (two identical devices with coupled C_oss recharging)
- Nonlinear, current-dependent transconductance g_m(i_ch)
- Iterative I_oss determination from the quadratic circuit equation
- Body diode reverse recovery (Q_rr, I_rr, tau_rr)
- Parasitic source/drain inductance effects (L_s, L_d)
- Charge-equivalent capacitance integration from C(V) curves
- Per-interval energy breakdown (intervals 0a/1a/2a for turn-off, 0b/1b/2b/3b for turn-on)

The Christen-Biela model addresses all of these with <10% average error vs measurement (validated on C2M0080120D and SCH2080KEC).

---

## 2. Goals

1. Replace the simplified `BielaModel` with a physics-faithful `ChristenBielaModel` implementing the full paper equations (with all supplement corrections applied).
2. Add body diode reverse recovery fields to the core `Diode` model.
3. Provide automatic parameter extraction from TDB `Transistor` objects (C(V) integration, g_m fitting).
4. Validate against the paper's reference data (C2M0080120D, 600V, 20A).

---

## 3. Non-Goals

- Temperature-dependent reverse recovery (the paper does not deeply address this; future work).
- Multi-device paralleling (n_semi > 1 in the flowchart). Can be added later as a multiplier.
- Extending the model to IGBTs (IGBT tail current uses the separate `IgbtModel`).
- GUI integration (separate task).

---

## 4. Physics Model Summary

### 4.1 Topology

Homogeneous half-bridge: two identical MOSFETs S1 (high-side) and S2 (low-side). Load current I_0 = const during switching. Gate voltage V_g steps instantaneously.

### 4.2 MOSFET Equivalent Circuit (Fig. 1 of paper)

```
           S1
    V_0 ─┤├─── L_d ──┬── v_ds
           S2         │
                 ┌────┴────┐
          i_gd → C_gd    C_ds ← i_ds
                 └────┬────┘
    R_g ──── v_gs ─── C_gs
                       │
                       L_s ── I_0 = const
```

- Channel modeled as controlled current source: `i_ch = g_m(i_ch) * (v_gs - V_th)`
- Capacitor recharging current: `I_oss = i_gd + i_ds` (lossless energy buffer)
- `C_oss = C_gd + C_ds` per device

### 4.3 Charge-Equivalent Capacitances (eq 32-35)

From voltage-dependent C(V) curves in the datasheet:

```
C_v,q,eq = (1/V_0) * integral_0^V_0 C_v(v_ds) dv_ds     for v = oss, iss, rss
C_gs = C_iss,eq - C_rss,eq
C_ds = C_oss,eq - C_rss,eq
C_gd = C_rss,eq
Q_oss = integral_0^V_0 C_oss(v) dv     (charge in C_oss of ONE switch, per supplement Mistake 6)
```

### 4.4 Transconductance Model (eq 36-37)

Transfer characteristic fitted from channel data:

```
i_ch = k_1 * (v_gs - V_th)^x + k_2                    (eq 36)
g_m(i_ch) = (k_1 * i_ch^x / (i_ch - k_2))^(1/x)      (eq 37)
```

Parameters (k_1, k_2, x) found by `scipy.optimize.curve_fit` on the switch channel V-I data at a single temperature. V_th extracted as the x-intercept of the transfer curve.

### 4.5 Turn-Off Model (Section II-A)

**Interval 0a** — Gate voltage drops to V_g,off. Lossless (no current/voltage change).

**Interval 1a** — Voltage rises from 0 to V_0:

- Solve quadratic for I_oss (eq 10, CORRECTED eq 29 sign):

```
0 = (2*L_s / (Q_oss*R_g)) * I_oss^2
  + (2/(g_m*R_g) + C_gd/(C_gd+C_ds)) * I_oss
  + (1/R_g) * (V_g - V_th - I_0/g_m)
```

  Where V_g = V_g,off (turn-off gate supply, typically 0 or negative).
  **Iterative**: g_m = f(I_0) initially, then g_m = f(I_0 - 2*I_oss), repeat until delta_gm/gm < 1%.

- Drain/channel currents: `i_d = I_0 - I_oss`, `i_ch = I_0 - 2*I_oss`
- Miller voltage: `V_mil = V_th + i_ch / g_m`
- Voltage rise time: `t_rv = Q_oss / I_oss`

**Interval 2a** — Channel current falls to zero:

- Current fall time (CORRECTED eq 15):
  `t_fi = -ln((V_th - V_g) / (V_mil - V_g)) * (C_gs*R_g + L_s*g_m)`
  Where V_g = V_g,off.
- Overvoltage: `V_Ld = L_d * (I_0 - 2*I_oss) / t_fi`

**Turn-off energy (eq 17)**:

```
E_T,off = 0.5 * t_rv * V_0 * (I_0 - 2*I_oss)
        + 0.5 * t_fi * (V_0 + V_Ld) * (I_0 - 2*I_oss)
```

### 4.6 Turn-On Model (Section II-B)

**Interval 0b** — Gate charges to V_th. Lossless.

**Interval 1b** — Current rises to I_0:

- Current rise time (CORRECTED eq 18, with V_g = V_g,on):
  `t_ri = -ln(1 - I_0 / (g_m * (V_g - V_th))) * (C_gs*R_g + L_s*g_m)`
  g_m = f(I_0) from the transconductance model.
- Voltage reduction from stray inductance: `V_Ld = L_d * I_0 / t_ri`
- Initial drain-source voltage: `V_ds,0 = V_0 - V_Ld`

**Interval 2b** — Reverse recovery of body diode of S2:

- Diode current model (CORRECTED eq 19): `i_bd(t) = I_0 - (I_0/t_ri) * t` for t < T_1
- Reverse recovery parameters from datasheet:
  - `Q_rr* = Q_rr - Q_oss` (subtract capacitance charge, per supplement)
  - `Q_rf = Q_rr* - I_rr^2 / (2 * di_bd/dt)` (eq 38)
  - `tau_rr = Q_rf / I_rr` (eq 39; or `Q_rf / (0.9*I_rr)` if T_2 is finite, per supplement)
  - `I_rr,dat = |di_bd/dt| * (tau_c - tau_rr) * (1 - exp(-T_1/tau_c))` (eq 40)
  - `1/T_m = 1/tau_rr - 1/tau_c` (CORRECTED eq 41)
  - T_1 found by numerically solving `q_e(T_1) = 0` (eq 22)
- Reverse conducting time: `t_rs = T_1 - T_0` where `T_0 = t_ri`
- Peak reverse current: `I_rr = t_rs * I_0 / t_ri`
- Reverse conducting charge: `Q_rs = 0.5 * t_rs * I_rr`

For arbitrary operating points:
- `T_0 = t_ri` (time when current through S2 crosses zero)
- Generalized reverse recovery uses the extracted T_m, tau_c, tau_rr

**Interval 3b** — Voltage falls:

- Same quadratic as turn-off (eq 29) but I_oss becomes negative.
  Iterative: g_m = f(I_0) initially, then g_m = f(I_0 - 2*I_oss), converge.
- Voltage fall time: `t_fv = -Q_oss / I_oss` (eq 30, I_oss < 0 so t_fv > 0)
- Channel current: `i_ch = I_0 - 2*I_oss` (increased by 2|I_oss| due to C_oss discharge)

**Reverse recovery energy contributions (eqs 26-28)**:

```
E_rs = Q_rs * V_ds,0                                     (eq 26)
E_rf = I_rr * V_ds,0 * (tau_rr / t_fv) *
       (t_fv - tau_rr + tau_rr * exp(-t_fv/tau_rr))      (eq 27, integration T_1 to T_1+t_fv)
E_rr,S2 = I_rr * V_ds,0 * (tau_rr/t_fv) *               (eq 28, integration T_1 to +inf)
           (tau_rr - (tau_rr + t_fv) * exp(-t_fv/tau_rr))
         + tau_rr * I_rr * V_0 * exp(-t_fv/tau_rr)
```

**Turn-on energy (eq 31)**:

```
E_T,on = 0.5 * t_ri * V_ds,0 * I_0
       + 0.5 * t_fv * (I_0 - 2*I_oss) * V_ds,0
       + t_rs * V_ds,0 * I_0
       + E_rf + E_rs
```

### 4.7 ZVS Boundary (eq 14)

Maximum current for lossless turn-off (i_ch → 0, I_oss = I_0/2):

```
I_0,zvs = V_0 / (2*L_s) * (-R_g*C_gd + sqrt((R_g*C_gd)^2
          - 8*(V_g - V_th) * L_s*(C_gd+C_ds) / V_0))
```

### 4.8 Total Half-Bridge Switching Losses

Per switching event per device:
```
P_sw,Si = f_s * (E_T,on + E_D,off + E_T,off)
P_sw = n_semi * (P_sw,S1 + P_sw,S2)
```

Where `E_D,off,S2 = E_rr,S2` (body diode turn-off loss in the complementary device).

---

## 5. Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Location | New `ChristenBielaModel` class in `analytical_models.py` | Keeps calculations separate from data model |
| Old BielaModel | Replace entirely | User preference; cleaner codebase |
| Diode params | Add `q_rr`, `i_rr`, `t_rr`, `di_dt_rr` fields to core `Diode` in `models.py` | Long-term correctness |
| Parasitic L | Package lookup table + user override | Convenience + flexibility |
| g_m fitting | Auto-fit from `switch.channel_data` via `curve_fit` | Fully automated |
| Q_oss calc | Auto-integrate C_oss(V) via `np.trapz` | Per paper eq(32) |
| Convergence | 1% relative delta_gm | Fast, sufficient accuracy |
| Return value | Tiered: summary dict default, detailed dataclass with `detailed=True` | Flexibility |

---

## 6. Data Model Changes

### 6.1 Diode class (models.py) — New optional fields

```python
class Diode(ITransistorComponent):
    def __init__(self) -> None:
        # ... existing fields ...
        # NEW: Body diode reverse recovery parameters
        self.q_rr: float | None = None       # Reverse recovery charge in C
        self.i_rr: float | None = None       # Peak reverse recovery current in A
        self.t_rr: float | None = None       # Reverse recovery time in s
        self.di_dt_rr: float | None = None   # di/dt during reverse recovery in A/s
```

### 6.2 New dataclasses in analytical_models.py

```python
@dataclass
class HalfBridgeParams:
    """Operating conditions and circuit parameters for the half-bridge model."""
    v_0: float           # DC bus voltage in V
    i_0: float           # Load current in A
    v_g_on: float        # Turn-on gate voltage in V (e.g., +20)
    v_g_off: float       # Turn-off gate voltage in V (e.g., -5 or 0)
    r_g: float           # Total gate resistance (R_g,ext + R_g,int) in Ohm
    l_s: float = 4e-9    # Source parasitic inductance in H
    l_d: float = 10e-9   # Drain parasitic inductance in H

@dataclass
class TransconductanceParams:
    """Fitted transconductance model parameters."""
    k_1: float    # Coefficient
    k_2: float    # Offset
    x: float      # Exponent
    v_th: float   # Threshold voltage in V

@dataclass
class ReverseRecoveryParams:
    """Extracted body diode reverse recovery time constants."""
    tau_rr: float   # Recovery time constant in s
    tau_c: float    # Carrier lifetime in s
    t_m: float      # Drift region transit time in s
    q_rr_star: float  # Corrected Q_rr (minus Q_oss) in C

@dataclass
class SwitchingEnergyResult:
    """Detailed switching energy breakdown."""
    # Summary
    e_on: float         # Total turn-on energy in J
    e_off: float        # Total turn-off energy in J
    e_total: float      # e_on + e_off in J
    i_0_zvs: float      # Maximum ZVS current in A

    # Turn-off breakdown
    i_oss_off: float    # Capacitor charging current during turn-off in A
    v_mil_off: float    # Miller voltage during turn-off in V
    t_rv: float         # Voltage rise time in s
    t_fi: float         # Current fall time in s
    v_ld_off: float     # Overvoltage from L_d during turn-off in V
    gm_off: float       # Transconductance during turn-off in S

    # Turn-on breakdown
    i_oss_on: float     # Capacitor charging current during turn-on in A (negative)
    v_mil_on: float     # Miller voltage during turn-on in V
    t_ri: float         # Current rise time in s
    t_fv: float         # Voltage fall time in s
    v_ld_on: float      # Voltage reduction from L_d during turn-on in V
    gm_on: float        # Transconductance during turn-on (interval 1b) in S
    gm_on_3b: float     # Transconductance during turn-on (interval 3b) in S

    # Reverse recovery
    i_rr: float         # Peak reverse recovery current in A
    t_rs: float         # Reverse conducting time in s
    q_rs: float         # Reverse conducting charge in C
    e_rs: float         # Reverse conducting energy in J
    e_rf: float         # Recovery fall energy in J
    e_rr_s2: float      # Body diode loss in S2 in J
```

---

## 7. Package Inductance Lookup Table

Default parasitic inductance estimates by housing type:

| Housing Type | L_s (nH) | L_d (nH) | Source |
|-------------|----------|----------|--------|
| TO-247      | 4.0      | 10.0     | Paper [6], [26], [27] |
| TO-220      | 5.0      | 12.0     | Literature estimate |
| TO-263 (D2PAK) | 3.0  | 8.0      | Literature estimate |
| SOT-227     | 2.0      | 5.0      | Low-inductance package |
| QFN / DFN   | 0.5      | 1.0      | Minimal parasitics |
| Module      | 5.0      | 15.0     | Higher due to bonding |
| Default     | 4.0      | 10.0     | Paper assumption |

---

## 8. API Surface

### 8.1 Primary API

```python
from transistordatabase.analytical_models import ChristenBielaModel, HalfBridgeParams

model = ChristenBielaModel.from_transistor(transistor)
result = model.calc_switching_energy(
    HalfBridgeParams(v_0=600, i_0=20, v_g_on=20, v_g_off=-5, r_g=7.1)
)
# result = {"e_on": ..., "e_off": ..., "e_total": ..., "i_0_zvs": ...}

# Detailed breakdown
result = model.calc_switching_energy(params, detailed=True)
# result = SwitchingEnergyResult(e_on=..., e_off=..., t_rv=..., ...)
```

### 8.2 Standalone API (without Transistor object)

```python
model = ChristenBielaModel(
    c_gs=1080e-12, c_ds=130e-12, c_gd=14.5e-12,
    q_oss=86.56e-9,
    gm_params=TransconductanceParams(k_1=0.1319, k_2=-0.076, x=3.80, v_th=4.5),
    rr_params=ReverseRecoveryParams(tau_rr=8.6e-9, tau_c=16e-9, t_m=18.6e-9, q_rr_star=88e-9),
)
result = model.calc_switching_energy(params)
```

### 8.3 Curve sweep API

```python
currents = np.linspace(1, 40, 20)
e_on, e_off, e_total = model.calc_switching_loss_curve(
    v_0=600, currents=currents, v_g_on=20, v_g_off=-5, r_g=7.1
)
```

---

## 9. Validation Data (Paper Table I + Fig. 11)

Device: C2M0080120D (Wolfspeed SiC MOSFET, 1200V, 80mOhm)
Operating point: V_0 = 600V, I_0 = 20A, T_j = 25C
Gate drive: V_g,on = +20V, V_g,off = -5V, R_g = 2.5 + 4.6 = 7.1 Ohm
Parasitic: L_s = 4nH, L_d = 10nH

### Expected intermediates (Table I):

| Parameter | Turn-off | Turn-on |
|-----------|----------|---------|
| g_m (interval 1) | 1.02 S | 3.02 S |
| I_oss | 8.33 A | -6.06 A |
| I_ch | 3.32 A | 32.16 A |
| V_mil | 7.46 V | 12.16 V |
| t_rv / t_ri | 10.5 ns | 10.7 ns |
| t_fi / t_fv | 3.5 ns | 14.44 ns |
| t_rs | — | 4.6 ns |
| I_rr | — | 8.6 A |
| g_m (interval 3b) | — | 4.1 S |

### Expected energies:

| Quantity | Value |
|----------|-------|
| E_T,off (device internal) | 14.1 uJ |
| E_T,on (device internal) | 274 uJ |
| E_oss,S1 (stored energy) | 18.9 uJ |

### Expected trends (Fig. 11):

- E_on increases super-linearly with I_0 (due to reverse recovery growth)
- E_off increases roughly linearly with I_0
- E_on >> E_off for SiC MOSFETs (factor ~20x at 20A)
- E_on and E_off both scale roughly linearly with V_0
- Average error vs measurement: <10%

---

## 10. Acceptance Criteria

1. `ChristenBielaModel.from_transistor(t)` successfully extracts all parameters from a core `Transistor` object with C(V) and channel data.
2. `calc_switching_energy()` returns correct summary dict for the C2M0080120D reference point (within 15% of Table I values, accounting for supplement corrections).
3. `calc_switching_energy(params, detailed=True)` returns `SwitchingEnergyResult` with all intermediate values populated.
4. Iterative I_oss convergence completes within 20 iterations for all reasonable operating points.
5. `calc_switching_loss_curve()` generates monotonically increasing E_on and E_off vs current.
6. All existing `TestGateChargeModel` and `TestIgbtModel` tests still pass unchanged.
7. New `TestChristenBielaModel` tests pass (replacing old `TestBielaModel`).
8. `ruff check` passes on all modified files.
9. Graceful degradation: if reverse recovery data is missing, model skips E_rs/E_rf/E_rr contributions and logs a warning.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| g_m curve_fit fails to converge | Provide fallback: if fit fails, use linear g_m estimate from channel data |
| Missing C(V) data in Transistor | Raise clear `ValueError` with message listing which capacitance curves are needed |
| I_oss quadratic has no real root | Use discriminant check; if negative, I_oss = 0 (ZVS condition) |
| Division by zero (I_0 = 0, g_m = 0) | Guard all divisions; return E = 0 for I_0 = 0 |
| Errata values differ from original paper | All implementations use supplement-corrected equations exclusively |
