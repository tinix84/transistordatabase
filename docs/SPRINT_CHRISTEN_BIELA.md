# Sprint Plan: Christen-Biela Analytical Switching Loss Model

Implementation of the full Christen/Biela half-bridge MOSFET switching loss model
(IEEE TPEL 2019) with all supplement errata corrections applied.

See `PRD_CHRISTEN_BIELA.md` for the full physics reference and equation details.

---

## Conventions

- **Agent model**: `haiku` unless marked `sonnet` (multi-file reasoning or complex math)
- **Task ID format**: `CB.{seq}` (Christen-Biela task sequence)
- **Acceptance**: Every task must pass `ruff check` and not break existing tests
- **Context rule**: Each prompt includes exact file paths, equation references, and expected values
- **Errata**: ALL equations use the supplement-corrected versions. The prompt for each task explicitly states which corrections apply.

---

## Task Dependency Graph

```
CB.1 (Diode fields) ──────────────────────────────┐
CB.2 (Dataclasses) ──┬── CB.5 (Turn-off) ─────────┤
CB.3 (Capacitance) ──┤                             ├── CB.8 (from_transistor) ── CB.10 (Curve sweep)
CB.4 (gm fit) ───────┤                             │                            CB.11 (Tests)
                      ├── CB.6 (Reverse recovery) ──┤
                      └── CB.7 (Turn-on) ───────────┘
CB.9 (Package lookup) ─────────────────────────────┘
```

**Parallelism**: CB.1, CB.2, CB.3, CB.4, CB.9 can all run in parallel (no dependencies).
CB.5 and CB.6 can run in parallel after CB.2+CB.3+CB.4 complete.
CB.7 depends on CB.5+CB.6.
CB.8 depends on all of CB.1-CB.7.
CB.10 and CB.11 depend on CB.8.

---

## CB.1 — Add reverse recovery fields to Diode model

**Agent**: haiku
**Depends on**: none

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Add optional body diode reverse recovery parameter fields to the core Diode class.

TARGET FILE TO EDIT: transistordatabase/core/models.py

Read the file first. Find the `Diode` class `__init__` method (around line 322). After the existing
`self.metadata: Dict[str, Any] = {}` line, add these new fields:

    self.q_rr: float | None = None       # Reverse recovery charge in C
    self.i_rr: float | None = None       # Peak reverse recovery current in A
    self.t_rr: float | None = None       # Reverse recovery time in s
    self.di_dt_rr: float | None = None   # Current slope during reverse recovery in A/s

These are optional (None by default) because not all datasheets provide them.

CONSTRAINTS:
- Do NOT modify any other class or method
- Do NOT add imports
- Preserve all existing code exactly as-is
- Must pass `ruff check transistordatabase/core/models.py`

ACCEPTANCE:
- Diode class has 4 new optional fields
- All existing tests still pass: `pytest tests/test_analytical_models.py tests/test_core_services.py -q`
- `ruff check` passes
```

**Input files**: `transistordatabase/core/models.py`
**Output files**: `transistordatabase/core/models.py` (edited)
**Acceptance**: Diode has q_rr, i_rr, t_rr, di_dt_rr fields; existing tests pass

---

## CB.2 — Create dataclasses for the model

**Agent**: haiku
**Depends on**: none

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Add new dataclasses to analytical_models.py that will be used by the ChristenBielaModel.
Do NOT modify or delete any existing code yet — just ADD new classes at the END of the file.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Read the file first. After the last class (IgbtModel), add these dataclasses:

1. `HalfBridgeParams` — operating conditions:
   - v_0: float           # DC bus voltage in V
   - i_0: float           # Load current in A
   - v_g_on: float        # Turn-on gate voltage in V (e.g., +20)
   - v_g_off: float       # Turn-off gate voltage in V (e.g., -5 or 0)
   - r_g: float           # Total gate resistance (external + internal) in Ohm
   - l_s: float = 4e-9    # Source parasitic inductance in H
   - l_d: float = 10e-9   # Drain parasitic inductance in H

2. `TransconductanceParams` — fitted gm model:
   - k_1: float    # Coefficient in transfer char
   - k_2: float    # Offset in transfer char
   - x: float      # Exponent in transfer char
   - v_th: float   # Threshold voltage in V

   Add a method `def calc_gm(self, i_ch: float) -> float` that implements:
       g_m = (k_1 * i_ch**x / (i_ch - k_2))**(1/x)
   Guard against i_ch <= k_2 (return a large default gm = 100.0 if so).
   Guard against i_ch <= 0 (return 0.01).

3. `ReverseRecoveryParams` — extracted diode time constants:
   - tau_rr: float   # Recovery time constant in s
   - tau_c: float    # Effective carrier lifetime in s
   - t_m: float      # Drift region transit time in s
   - q_rr_star: float  # Corrected Q_rr (= Q_rr_datasheet - Q_oss) in C

4. `SwitchingEnergyResult` — detailed result:
   Fields (all floats):
   # Summary
   e_on, e_off, e_total, i_0_zvs
   # Turn-off breakdown
   i_oss_off, v_mil_off, t_rv, t_fi, v_ld_off, gm_off
   # Turn-on breakdown
   i_oss_on, v_mil_on, t_ri, t_fv, v_ld_on, gm_on_1b, gm_on_3b
   # Reverse recovery
   i_rr_calc, t_rs, q_rs, e_rs, e_rf, e_rr_s2

   Add a method `def to_dict(self) -> dict[str, float]` returning the summary:
       {"e_on": self.e_on, "e_off": self.e_off, "e_total": self.e_total,
        "i_0_zvs": self.i_0_zvs}

All classes must be @dataclass, use `from __future__ import annotations`, have Sphinx docstrings
with :param:/:type: for every field.

CONSTRAINTS:
- Do NOT modify or delete any existing classes (BielaModel, GateChargeModel, IgbtModel)
- Add the new classes AFTER the existing code
- Must pass `ruff check transistordatabase/analytical_models.py`
- Use `from __future__ import annotations` (already at top of file)

ACCEPTANCE:
- Four new dataclasses importable: HalfBridgeParams, TransconductanceParams,
  ReverseRecoveryParams, SwitchingEnergyResult
- TransconductanceParams.calc_gm(20.0) returns a positive float for reasonable k_1, k_2, x values
- SwitchingEnergyResult.to_dict() returns 4-key dict
- `ruff check` passes
- All existing tests pass unchanged
```

**Input files**: `transistordatabase/analytical_models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited, appended)
**Acceptance**: Four new dataclasses importable and functional

---

## CB.3 — Implement charge-equivalent capacitance calculation

**Agent**: haiku
**Depends on**: none

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Add standalone functions for charge-equivalent capacitance calculation to analytical_models.py.
These implement equations (32)-(35) from Christen/Biela 2019.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Add these functions AFTER all existing code:

1. `calc_charge_equivalent_capacitance(v_axis: np.ndarray, c_axis: np.ndarray, v_0: float) -> float`:
   - Implements eq(32): C_eq = (1/V_0) * integral_0^V_0 C(v) dv
   - Use np.trapz for numerical integration
   - Clip v_axis to [0, v_0] range before integrating
   - If v_0 is beyond the data range, extrapolate with the last C value (constant extrapolation)

2. `calc_charge_stored(v_axis: np.ndarray, c_axis: np.ndarray, v_0: float) -> float`:
   - Implements Q_oss = integral_0^V_0 C_oss(v) dv (charge stored in C_oss of ONE switch)
   - Use np.trapz
   - Same clipping/extrapolation as above

3. `calc_device_capacitances(c_iss_eq: float, c_oss_eq: float, c_rss_eq: float) -> tuple[float, float, float]`:
   - Implements eqs (33)-(35):
     C_gs = C_iss_eq - C_rss_eq
     C_ds = C_oss_eq - C_rss_eq
     C_gd = C_rss_eq
   - Return (c_gs, c_ds, c_gd)
   - Guard: if any result is negative, clamp to 0.0 and emit a warning via `warnings.warn()`

Add `import warnings` at the top of the file (after existing imports).

CONSTRAINTS:
- Do NOT modify existing code
- Append new functions after existing classes
- Must pass `ruff check`
- Sphinx docstrings with :param:, :return:

ACCEPTANCE:
- calc_charge_equivalent_capacitance([0, 100, 200], [1000e-12, 500e-12, 200e-12], 200.0)
  returns approximately 566.7e-12 (= (1/200)*trapz([1000,500,200]*1e-12, [0,100,200]))
- calc_charge_stored returns approximately 113.3e-9 for the same input
  (= trapz([1000,500,200]*1e-12 * [0,100,200], [0,100,200])... actually just
  integral of C(v)dv = trapz(c_axis, v_axis) over [0, v_0])
- calc_device_capacitances(1100e-12, 150e-12, 15e-12) returns (1085e-12, 135e-12, 15e-12)
```

**Input files**: `transistordatabase/analytical_models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited, appended)
**Acceptance**: Three functions return correct values for known inputs

---

## CB.4 — Implement transconductance fitting from channel data

**Agent**: sonnet
**Depends on**: CB.2

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Add a function to fit transconductance model parameters from switch channel V-I data.
This implements equations (36)-(37) from Christen/Biela 2019.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Read the file first. Read also transistordatabase/core/models.py to understand ChannelCharacteristics
(has graph_v_i as 2D array [voltage, current] and v_g gate voltage).

Add a function after existing code:

`fit_transconductance(channel_data: ChannelCharacteristics) -> TransconductanceParams`:

The transfer characteristic model is (eq 36):
    i_ch = k_1 * (v_gs - V_th)^x + k_2

To fit this from channel V-I data:
1. Extract v_ds (voltages) and i_d (currents) from channel_data.graph_v_i.
   Note: graph_v_i[0] = voltages, graph_v_i[1] = currents.
   However, channel data is V_ds vs I_d (drain characteristics), NOT V_gs vs I_d (transfer).
   We need the transfer characteristic. If channel_data has a single v_g value, we can use
   the relationship: in the linear region, V_ds ≈ R_ds_on * I_d, and in saturation,
   I_d = k_1 * (V_gs - V_th)^x + k_2.

   APPROACH: Since TDB channel data stores V_ds-I_d curves at a specific V_gs,
   and we need V_gs-I_d (transfer curve), we need MULTIPLE channel_data entries
   at different V_gs values to reconstruct the transfer characteristic.

   Accept a LIST: `fit_transconductance(channel_data_list: list[ChannelCharacteristics]) -> TransconductanceParams`

   For each ChannelCharacteristics entry:
   - Extract v_g (gate voltage)
   - Extract a reference current at a specific V_ds (use median V_ds point)
   - Build arrays: vgs_points[], id_points[]

   Then fit: i_d = k_1 * (v_gs - V_th)^x + k_2 using scipy.optimize.curve_fit.

2. Initial guess: V_th = minimum v_g where current > 0.1A, k_1 = 0.1, k_2 = 0, x = 2.0
3. Bounds: V_th in [0, 10], k_1 in [1e-6, 100], k_2 in [-10, 10], x in [1.0, 6.0]
4. If curve_fit fails, fall back to linear approximation:
   - V_th = min v_g with current > 0
   - g_m_linear = delta_I / delta_V_gs
   - Set x=1.0, k_1=g_m_linear, k_2=0

Import scipy.optimize at the top (soft import with try/except ImportError).

Add `from transistordatabase.core.models import ChannelCharacteristics` import.

CONSTRAINTS:
- Do NOT modify existing code
- Append after existing functions
- scipy.optimize.curve_fit for fitting
- Must pass `ruff check`
- Sphinx docstrings

ACCEPTANCE:
- For synthetic data: vgs=[5,6,7,8,9,10], id=[0.5,2,5,10,18,28], V_th=4.5:
  fit returns TransconductanceParams with v_th close to 4.5, x between 1.5-4.0
- TransconductanceParams.calc_gm(20.0) returns positive float
- Graceful fallback if curve_fit doesn't converge
```

**Input files**: `transistordatabase/analytical_models.py`, `transistordatabase/core/models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited, appended)
**Acceptance**: Fitting works on synthetic transfer data; fallback works when fit fails

---

## CB.5 — Implement turn-off energy calculation

**Agent**: sonnet
**Depends on**: CB.2, CB.3, CB.4

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Implement the turn-off switching energy calculation for the Christen/Biela model.
This covers Section II-A of the paper (intervals 0a, 1a, 2a) with supplement corrections.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Read the file first to see the dataclasses added in previous tasks (HalfBridgeParams,
TransconductanceParams, SwitchingEnergyResult).

Add a class `ChristenBielaModel` after all existing code. For now implement only __init__
and the turn-off part. The turn-on part will be added in CB.7.

class ChristenBielaModel:
    """Full Christen-Biela analytical switching loss model for half-bridge MOSFETs.

    Reference: Christen, Biela, IEEE TPEL 2019, DOI: 10.1109/TPEL.2018.2851068
    All equations use supplement errata corrections.
    """

    def __init__(
        self,
        c_gs: float,           # Gate-source capacitance (charge-equiv) in F
        c_ds: float,           # Drain-source capacitance (charge-equiv) in F
        c_gd: float,           # Gate-drain capacitance (charge-equiv) in F
        q_oss: float,          # Charge stored in C_oss of ONE switch at V_0, in C
        gm_params: TransconductanceParams,
        rr_params: ReverseRecoveryParams | None = None,  # None = skip reverse recovery
    ) -> None:
        Store all params as instance attributes.
        Compute c_oss = c_gd + c_ds.

    def _solve_i_oss(
        self, i_0: float, v_g: float, r_g: float, l_s: float,
        gm_initial: float | None = None, tol: float = 0.01, max_iter: int = 20,
    ) -> tuple[float, float, float]:
        """Solve iterative quadratic for I_oss with gm convergence.

        Implements eq(10) with CORRECTED sign (supplement Mistake 4 confirms eq(10) form):
            0 = (2*L_s / (Q_oss*R_g)) * I_oss^2
              + (2/(g_m*R_g) + C_gd/(C_gd+C_ds)) * I_oss
              + (1/R_g) * (V_g - V_th - I_0/g_m)

        ITERATION (from supplement corrected flowchart):
        - For turn-off: g_m = f(I_0) initially, then g_m = f(I_0 - 2*I_oss)
        - Repeat until |g_m_new - g_m_old| / g_m_new < tol (1%)

        Use np.roots() or quadratic formula to solve.
        Select the physically meaningful root (the one closer to 0 for turn-off,
        I_oss should be positive and < I_0/2 for hard-switching turn-off).

        Returns: (i_oss, gm_converged, v_mil)
        Where v_mil = V_th + (I_0 - 2*I_oss) / gm_converged
        """

    def calc_turn_off_energy(self, params: HalfBridgeParams) -> dict[str, float]:
        """Calculate turn-off switching energy E_T,off.

        Intervals:
        - 0a: gate drops to V_g,off (lossless)
        - 1a: voltage rises, eq(10) for I_oss, eq(13) for t_rv
        - 2a: current falls, CORRECTED eq(15) for t_fi, eq(16) for V_Ld

        CORRECTED eq(15):
            t_fi = -ln((V_th - V_g) / (V_mil - V_g)) * (C_gs*R_g + L_s*g_m)
            Where V_g = V_g,off (the TURN-OFF gate supply voltage)

        eq(17):
            E_T,off = 0.5*t_rv*V_0*(I_0 - 2*I_oss) + 0.5*t_fi*(V_0 + V_Ld)*(I_0 - 2*I_oss)

        Returns dict with: e_off, i_oss, v_mil, t_rv, t_fi, v_ld, gm
        """

    def calc_i_0_zvs(self, params: HalfBridgeParams) -> float:
        """Calculate maximum current for lossless turn-off (ZVS boundary).

        Implements eq(14):
            I_0,zvs = V_0/(2*L_s) * (-R_g*C_gd
                      + sqrt((R_g*C_gd)^2 - 8*(V_g-V_th)*L_s*(C_gd+C_ds)/V_0))

        Where V_g = V_g,off.
        If discriminant < 0, return 0 (no ZVS possible).
        """

IMPORTANT ERRATA NOTES:
- eq(10): The paper's eq(10) has a POSITIVE leading coefficient (2*L_s/(Q_oss*R_g)).
  Supplement Mistake 4 confirms eq(29) has the same form as eq(10). Use positive sign.
- eq(15): CORRECTED sign. Original had (V_th+V_g)/(V_mil+V_g), correct is (V_th-V_g)/(V_mil-V_g).
  V_g here is the turn-off gate voltage (0 or negative).
- For turn-off: V_g in the I_oss equation is V_g,off (negative or zero).

EDGE CASES:
- If I_0 < 2*I_oss: ZVS condition, E_T,off ≈ 0. Return e_off = 0.
- If I_0 = 0: return all zeros.
- If V_mil <= V_th: clamp t_fi to 0 (channel already off).

CONSTRAINTS:
- Use np.roots for quadratic, or explicit quadratic formula
- Must pass `ruff check`
- Sphinx docstrings with equation references

ACCEPTANCE:
- For C2M0080120D at 600V, 20A (paper Table I):
  I_oss ≈ 8.33A, V_mil ≈ 7.46V, t_rv ≈ 10.5ns, t_fi ≈ 3.5ns, E_T,off ≈ 14.1uJ
  (within ~20% tolerance due to supplement corrections and parameter uncertainty)
- E_T,off increases with I_0 and V_0
- E_T,off = 0 when I_0 = 0
```

**Input files**: `transistordatabase/analytical_models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited, class added)
**Acceptance**: Turn-off energy within 20% of paper reference; correct trends

---

## CB.6 — Implement reverse recovery parameter extraction

**Agent**: sonnet
**Depends on**: CB.2

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Add a function to extract body diode reverse recovery time constants from datasheet parameters.
This implements Section III-A.2 of Christen/Biela 2019 (eqs 38-41) with supplement corrections.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Read the file first. Add a function after existing code:

`extract_reverse_recovery_params(
    q_rr: float, i_rr: float, di_dt: float, q_oss: float,
    t_rr: float | None = None,
) -> ReverseRecoveryParams`:

Implementation steps (from paper Section III-A.2 + supplement):

Step I: Correct Q_rr by subtracting C_oss charge (supplement Mistake 6):
    Q_rr_star = Q_rr - Q_oss

Step II: Calculate Q_rf and tau_rr:
    Q_rf = Q_rr_star - I_rr^2 / (2 * |di_dt|)                  (eq 38)
    tau_rr = Q_rf / I_rr                                         (eq 39, assuming T_2 → +inf)

    SUPPLEMENT ATTENTION for eq(39): If the datasheet defines Q_rr with a finite T_2
    (e.g., 90% of Irr), use: tau_rr = Q_rf / (0.9 * I_rr) and T_2 = ln10 * tau_rr + T_1.
    For simplicity, use the T_2 → +inf assumption by default.

Step III: Find tau_c by numerically solving eq(40):
    I_rr = |di_dt| * (tau_c - tau_rr) * (1 - exp(-T_1 / tau_c))   (eq 40)

    Where T_1 = I_rr / |di_dt| (time from zero-crossing to peak reverse current).

    Use scipy.optimize.brentq to solve for tau_c. Search in range [tau_rr*1.01, tau_rr*100].
    The equation to solve is:
        f(tau_c) = |di_dt| * (tau_c - tau_rr) * (1 - exp(-T_1/tau_c)) - I_rr = 0

Step IV: Calculate T_m (CORRECTED eq 41):
    1/T_m = 1/tau_rr - 1/tau_c                                    (CORRECTED eq 41)
    T_m = 1 / (1/tau_rr - 1/tau_c)

    Supplement Mistake 5: Original paper had 1/tau_rr = 1/tau_c - 1/T_m (wrong).
    Correct: 1/T_m = 1/tau_rr - 1/tau_c.

Return ReverseRecoveryParams(tau_rr, tau_c, t_m, q_rr_star).

EDGE CASES:
- If Q_rf <= 0: tau_rr very small, set tau_rr = 1e-12 (negligible recovery)
- If brentq fails: fall back to tau_c = 2 * tau_rr (rough estimate)
- If 1/tau_rr - 1/tau_c <= 0: T_m = tau_rr * 10 (very long transit time)

Import scipy.optimize (soft import with try/except).

CONSTRAINTS:
- Must pass `ruff check`
- Sphinx docstrings with equation references
- Handle all edge cases gracefully

ACCEPTANCE:
- For C2M0080120D (paper Section III-A.2):
  Q_rr=192nC, I_rr=10A, di_dt=-2400A/us, Q_oss=86.56nC (at 800V from datasheet)
  Expected: tau_rr ≈ 8.6ns (6.7ns in paper, slight diff from Q_oss correction),
            tau_c ≈ 16ns, T_m ≈ 18.6ns
  (within 30% tolerance — exact values depend on supplement corrections vs original)
- Returns valid ReverseRecoveryParams with all positive time constants
```

**Input files**: `transistordatabase/analytical_models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited, function added)
**Acceptance**: Returns physically reasonable time constants for C2M0080120D parameters

---

## CB.7 — Implement turn-on energy calculation

**Agent**: sonnet
**Depends on**: CB.5, CB.6

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Add the turn-on energy calculation to ChristenBielaModel.
This covers Section II-B of the paper (intervals 0b, 1b, 2b, 3b) with supplement corrections.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Read the file first. Find the ChristenBielaModel class (added in CB.5). Add these methods:

1. `_calc_reverse_recovery(self, i_0: float, t_ri: float, v_ds_0: float, t_fv: float) -> dict`:
    """Calculate reverse recovery energy contributions (intervals 2b).

    Uses the diode model from eqs(19)-(28) with corrections:
    - CORRECTED eq(19): i_bd(t) = I_0 - (I_0/t_ri) * t  (sign fix from supplement Mistake 3)
    - T_0 = t_ri (time when current through S2 crosses zero)
    - t_rs = T_1 - T_0 (eq 23), where T_1 is found from the general model
    - I_rr = t_rs * I_0 / t_ri (eq 24)
    - Q_rs = 0.5 * t_rs * I_rr (eq 25)

    For the generalized model at arbitrary operating points, use the stored
    ReverseRecoveryParams (tau_rr, tau_c, T_m) to compute T_1 from:
        T_0 = t_ri
        di_bd/dt = I_0 / t_ri  (current slope)
        T_1 found by solving q_e(T_1) = 0 via eq(22) + eq(21) numerically

    Simplified approach (valid when t_ri is small relative to tau_c):
        T_1 ≈ T_0 + tau_c * ln(1 + I_0 * tau_c / (T_m * di_bd/dt * tau_c))
        Or simply: t_rs ≈ I_rr_scaled * t_ri / I_0 where I_rr_scaled uses operating point scaling

    For a practical implementation:
        t_rs_ref from datasheet params
        Scale: t_rs = t_rs_ref * (I_0 / I_0_ref) * (t_ri_ref / t_ri) approximately
        Or use the full numerical solve with brentq.

    Energy contributions (eqs 26-28):
    E_rs = Q_rs * V_ds,0                                          (eq 26)
    E_rf = I_rr * V_ds,0 * (tau_rr/t_fv) *                       (eq 27)
           (t_fv - tau_rr + tau_rr * exp(-t_fv/tau_rr))
           Integration limits: T_1 to T_1+t_fv (supplement attention for eq 27-28)
    E_rr_S2 = I_rr * V_ds,0 * (tau_rr/t_fv) *                    (eq 28)
              (tau_rr - (tau_rr + t_fv) * exp(-t_fv/tau_rr))
            + tau_rr * I_rr * V_0 * exp(-t_fv/tau_rr)
            Integration limits: T_1 to +inf (supplement attention for eq 27-28)

    If self.rr_params is None, return all zeros (skip reverse recovery).
    """
    Returns dict: {i_rr, t_rs, q_rs, e_rs, e_rf, e_rr_s2}

2. `calc_turn_on_energy(self, params: HalfBridgeParams) -> dict[str, float]`:
    """Calculate turn-on switching energy E_T,on.

    Intervals:
    - 0b: gate charges to V_th (lossless)
    - 1b: current rises, CORRECTED eq(18) for t_ri
    - 2b: reverse recovery
    - 3b: voltage falls, eq(29) CORRECTED for I_oss, eq(30) for t_fv

    CORRECTED eq(18):
        t_ri = -ln(1 - I_0 / (g_m * (V_g - V_th))) * (C_gs*R_g + L_s*g_m)
        V_g here = V_g,on (the TURN-ON gate supply voltage, positive)
        g_m = f(I_0) from transconductance model (interval 1b, V_ds ≈ const)

    eq(29) for interval 3b (CORRECTED, supplement Mistake 4):
        0 = (2*L_s / (Q_oss*R_g)) * I_oss^2
          + (2/(g_m*R_g) + C_gd/(C_gd+C_ds)) * I_oss
          + (1/R_g) * (V_g - V_th - I_0/g_m)
        V_g here = V_g,on. I_oss will be NEGATIVE (capacitances discharging).
        g_m = f(I_0) initially, iterate with g_m = f(I_0 - 2*I_oss) per supplement flowchart.

    eq(30): t_fv = -Q_oss / I_oss (I_oss < 0 → t_fv > 0)

    V_Ld = L_d * I_0 / t_ri (voltage reduction during current rise)
    V_ds,0 = V_0 - V_Ld (initial drain-source voltage, reduced by L_d)

    CORRECTED eq(31):
        E_T,on = 0.5 * t_ri * V_ds,0 * I_0
               + 0.5 * t_fv * (I_0 - 2*I_oss) * V_ds,0
               + t_rs * V_ds,0 * I_0
               + E_rf + E_rs

    Returns dict with: e_on, i_oss, v_mil, t_ri, t_fv, v_ld, gm_1b, gm_3b,
                        i_rr, t_rs, q_rs, e_rs, e_rf, e_rr_s2
    """

3. `calc_switching_energy(self, params: HalfBridgeParams, detailed: bool = False) -> dict | SwitchingEnergyResult`:
    """Calculate total switching energy for one switching event.

    Calls calc_turn_off_energy() and calc_turn_on_energy(), combines results.

    If detailed=False: return {"e_on": ..., "e_off": ..., "e_total": ..., "i_0_zvs": ...}
    If detailed=True: return SwitchingEnergyResult with all fields populated.
    """

EDGE CASES:
- I_0 = 0: return e_on = 0, skip reverse recovery
- If ln argument in eq(18) <= 0: I_0 >= g_m*(V_g-V_th), current can't reach I_0.
  In this case, use t_ri = C_gs*R_g + L_s*g_m (limiting case, ln → large)
  and cap t_ri at 1us max.
- I_oss from eq(29) for turn-on: select the negative root (capacitances discharging).
  Use _solve_i_oss() but for turn-on the V_g = V_g,on, and the physically meaningful
  root is the more negative one.
- t_fv: if I_oss = 0, set t_fv to a large value (1us) as limiting case.

CONSTRAINTS:
- Reuse _solve_i_oss() from CB.5 for both turn-off (eq 10) and turn-on (eq 29)
- Must pass `ruff check`
- Sphinx docstrings with equation references

ACCEPTANCE:
- For C2M0080120D at 600V, 20A (paper Table I):
  t_ri ≈ 10.7ns, I_oss_on ≈ -6.06A, t_fv ≈ 14.44ns,
  t_rs ≈ 4.6ns, I_rr ≈ 8.6A, E_T,on ≈ 274uJ
  (within 20% tolerance)
- E_T,on >> E_T,off for SiC MOSFETs (factor ~10-20x)
- E_T,on increases with I_0 and V_0
```

**Input files**: `transistordatabase/analytical_models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited, methods added)
**Acceptance**: Turn-on energy within 20% of paper reference; correct E_on >> E_off relationship

---

## CB.8 — Implement from_transistor() factory and finalize API

**Agent**: sonnet
**Depends on**: CB.1, CB.2, CB.3, CB.4, CB.5, CB.6, CB.7

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Add the from_transistor() factory method and calc_switching_loss_curve() to ChristenBielaModel.
This wires the model to TDB core Transistor objects for fully automated parameter extraction.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Read the file first. Also read transistordatabase/core/models.py to understand:
- Transistor.c_oss, c_iss, c_rss: list[VoltageDependentCapacitance] with graph_v_c [voltage, capacitance]
- Transistor.switch.channel_data: list[ChannelCharacteristics] with graph_v_i [voltage, current], v_g
- Transistor.diode.q_rr, i_rr, t_rr, di_dt_rr: optional floats (added in CB.1)
- Transistor.metadata.housing_type: str

Add to ChristenBielaModel:

1. `@classmethod from_transistor(cls, transistor: Transistor, v_0: float, t_j: float = 25.0) -> ChristenBielaModel`:
    """Create model from a TDB Transistor object.

    Extracts all parameters automatically:
    1. Charge-equivalent capacitances from C(V) curves at operating voltage v_0
       - Find c_oss, c_iss, c_rss entries closest to t_j
       - Use calc_charge_equivalent_capacitance() and calc_charge_stored()
       - Use calc_device_capacitances() to get C_gs, C_ds, C_gd
    2. Transconductance fit from channel data
       - Collect all ChannelCharacteristics at t_j (or nearest temperature)
       - Call fit_transconductance()
    3. Reverse recovery params from diode fields
       - If transistor.diode.q_rr is not None, call extract_reverse_recovery_params()
       - Else rr_params = None (will skip reverse recovery)
    4. Return ChristenBielaModel(c_gs, c_ds, c_gd, q_oss, gm_params, rr_params)

    Raise ValueError with descriptive message if:
    - No C_oss data available
    - No channel data available
    """

    Import: from transistordatabase.core.models import Transistor, VoltageDependentCapacitance

2. `calc_switching_loss_curve(
        self, v_0: float, currents: npt.NDArray[np.float64],
        v_g_on: float = 15.0, v_g_off: float = -5.0,
        r_g: float = 10.0, l_s: float = 4e-9, l_d: float = 10e-9,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]`:
    """Calculate switching loss curves over a range of currents.

    Returns: (e_on_array, e_off_array, e_total_array) in J.
    Creates HalfBridgeParams for each current and calls calc_switching_energy().
    """

3. Also add the package inductance lookup as a module-level dict:

    PACKAGE_INDUCTANCE: dict[str, tuple[float, float]] = {
        "TO-247": (4e-9, 10e-9),
        "TO-220": (5e-9, 12e-9),
        "TO-263": (3e-9, 8e-9),
        "D2PAK": (3e-9, 8e-9),
        "SOT-227": (2e-9, 5e-9),
        "QFN": (0.5e-9, 1e-9),
        "DFN": (0.5e-9, 1e-9),
        "Module": (5e-9, 15e-9),
    }

    def get_package_inductance(housing_type: str) -> tuple[float, float]:
        """Look up parasitic inductances for a housing type.
        Returns (l_s, l_d) in H. Defaults to (4e-9, 10e-9) if not found.
        """

CONSTRAINTS:
- Must pass `ruff check`
- Handle missing data gracefully (clear error messages)
- Sphinx docstrings

ACCEPTANCE:
- from_transistor() works on a Transistor with C_oss/C_iss/C_rss + channel data
- calc_switching_loss_curve() returns arrays of correct length
- get_package_inductance("TO-247") returns (4e-9, 10e-9)
- get_package_inductance("unknown") returns (4e-9, 10e-9) default
```

**Input files**: `transistordatabase/analytical_models.py`, `transistordatabase/core/models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited)
**Acceptance**: Factory method extracts params from Transistor; curve sweep works

---

## CB.9 — Remove old BielaModel

**Agent**: haiku
**Depends on**: CB.5 (ChristenBielaModel exists)

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Remove the old simplified BielaModel and BielaModelParams classes from analytical_models.py.
The ChristenBielaModel replaces them.

TARGET FILE TO EDIT: transistordatabase/analytical_models.py

Read the file first. Delete:
1. The `BielaModelParams` dataclass (starts with `@dataclass` before `class BielaModelParams`)
2. The `BielaModel` class (starts with `class BielaModel`)

Do NOT delete GateChargeModel, GateChargeModelParams, IgbtModel, IgbtModelParams,
or any of the new classes/functions added in CB.2-CB.8.

CONSTRAINTS:
- Only remove BielaModelParams and BielaModel
- Do NOT touch any other code
- Must pass `ruff check`

ACCEPTANCE:
- BielaModel and BielaModelParams no longer exist in the file
- GateChargeModel, IgbtModel, ChristenBielaModel all still present
- `ruff check` passes
```

**Input files**: `transistordatabase/analytical_models.py`
**Output files**: `transistordatabase/analytical_models.py` (edited)
**Acceptance**: Old classes removed; new and other classes intact

---

## CB.10 — Update __init__.py exports

**Agent**: haiku
**Depends on**: CB.8, CB.9

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Update transistordatabase/__init__.py to export the new analytical model classes.

TARGET FILE TO EDIT: transistordatabase/__init__.py

Read the file first. Add an import block for analytical_models:

    from transistordatabase.analytical_models import (
        ChristenBielaModel,
        HalfBridgeParams,
        TransconductanceParams,
        ReverseRecoveryParams,
        SwitchingEnergyResult,
        GateChargeModel,
        GateChargeModelParams,
        IgbtModel,
        IgbtModelParams,
    )

Place this after existing imports, following the file's existing import style.

CONSTRAINTS:
- Do NOT remove any existing imports or code
- Must pass `ruff check`
- Must not import BielaModel or BielaModelParams (they no longer exist)

ACCEPTANCE:
- `from transistordatabase import ChristenBielaModel` works
- `from transistordatabase import GateChargeModel` works
- `ruff check` passes
```

**Input files**: `transistordatabase/__init__.py`
**Output files**: `transistordatabase/__init__.py` (edited)
**Acceptance**: All new classes importable from top-level package

---

## CB.11 — Write comprehensive tests

**Agent**: sonnet
**Depends on**: CB.8, CB.9, CB.10

**Prompt**:
```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: Rewrite tests/test_analytical_models.py to test the new ChristenBielaModel
(replacing old BielaModel tests) while keeping GateChargeModel and IgbtModel tests.

TARGET FILE TO REWRITE: tests/test_analytical_models.py

Read the existing file first. Keep TestGateChargeModel and TestIgbtModel exactly as-is.
Replace TestBielaModel with TestChristenBielaModel.

New test class structure:

class TestChristenBielaModel:

    @pytest.fixture
    def c2m_model(self) -> ChristenBielaModel:
        """C2M0080120D model from paper Section III."""
        # Paper parameters:
        # C_gs = 1080pF, C_ds = 130pF, C_gd = 14.5pF (eqs 33-35)
        # Q_oss = 86.56nC at 600V
        # gm params: k_1=0.1319, k_2=-0.076, x=3.80, V_th=4.5V
        # RR params: tau_rr=8.6ns, tau_c=16ns, T_m=18.6ns, Q_rr*=88nC
        return ChristenBielaModel(
            c_gs=1080e-12, c_ds=130e-12, c_gd=14.5e-12,
            q_oss=86.56e-9,
            gm_params=TransconductanceParams(k_1=0.1319, k_2=-0.076, x=3.80, v_th=4.5),
            rr_params=ReverseRecoveryParams(tau_rr=8.6e-9, tau_c=16e-9, t_m=18.6e-9, q_rr_star=88e-9),
        )

    @pytest.fixture
    def default_params(self) -> HalfBridgeParams:
        """C2M0080120D operating point from paper Table I."""
        return HalfBridgeParams(v_0=600, i_0=20, v_g_on=20, v_g_off=-5, r_g=7.1, l_s=4e-9, l_d=10e-9)

    Tests to include:

    1. test_e_off_reference_point(c2m_model, default_params):
       - E_T,off should be within [5, 30] uJ (paper: 14.1uJ, allow wide margin for errata diffs)

    2. test_e_on_reference_point(c2m_model, default_params):
       - E_T,on should be within [150, 500] uJ (paper: 274uJ)

    3. test_e_on_much_greater_than_e_off(c2m_model, default_params):
       - E_on / E_off > 5 (paper shows ~20x, but ratio varies with corrections)

    4. test_energy_increases_with_voltage(c2m_model):
       - E_total at 800V > E_total at 400V

    5. test_energy_increases_with_current(c2m_model):
       - E_total at 30A > E_total at 10A

    6. test_zero_current(c2m_model):
       - E_on = 0, E_off = 0 for I_0 = 0

    7. test_zvs_boundary(c2m_model, default_params):
       - I_0_zvs > 0 and < 20A (paper shows ZVS up to ~5-8A for this device)

    8. test_detailed_result(c2m_model, default_params):
       - detailed=True returns SwitchingEnergyResult
       - All timing values > 0
       - i_oss_off > 0, i_oss_on < 0

    9. test_summary_dict(c2m_model, default_params):
       - detailed=False returns dict with exactly 4 keys

    10. test_switching_loss_curve(c2m_model):
        - Returns 3 arrays of correct length
        - All monotonically increasing with current

    11. test_no_reverse_recovery():
        - Model without rr_params: e_on should be smaller than with rr_params
        - No errors raised

    12. test_convergence_iteration(c2m_model, default_params):
        - Model converges (doesn't raise exception)
        - gm_off > 0, gm_on_1b > 0, gm_on_3b > 0

    13. test_turn_off_intermediates(c2m_model, default_params):
        - t_rv in [5, 20] ns (paper: 10.5ns)
        - t_fi in [1, 10] ns (paper: 3.5ns)
        - i_oss in [3, 15] A (paper: 8.33A)

    14. test_turn_on_intermediates(c2m_model, default_params):
        - t_ri in [5, 20] ns (paper: 10.7ns)
        - t_fv in [5, 30] ns (paper: 14.44ns)
        - i_rr in [3, 15] A (paper: 8.6A)

class TestCapacitanceFunctions:
    15. test_charge_equivalent_capacitance():
        - Known C(V) curve returns expected value

    16. test_charge_stored():
        - Integral of C(V) returns expected charge

    17. test_device_capacitances():
        - C_gs, C_ds, C_gd computed correctly

class TestTransconductanceFit:
    18. test_fit_synthetic_data():
        - Fit synthetic transfer curve, recover approximate k_1, x, V_th

class TestReverseRecoveryExtraction:
    19. test_c2m_reference():
        - Extract from C2M0080120D datasheet params
        - tau_rr, tau_c, T_m all positive and in [1ns, 100ns] range

    20. test_edge_case_zero_qrr():
        - Q_rr = Q_oss → Q_rr_star = 0 → returns near-zero time constants

class TestPackageLookup:
    21. test_known_packages():
        - TO-247, TO-220, QFN return expected values

    22. test_unknown_package():
        - Returns default (4e-9, 10e-9)

Update imports at top:
    from transistordatabase.analytical_models import (
        ChristenBielaModel,
        HalfBridgeParams,
        TransconductanceParams,
        ReverseRecoveryParams,
        SwitchingEnergyResult,
        GateChargeModel,
        GateChargeModelParams,
        IgbtModel,
        IgbtModelParams,
        calc_charge_equivalent_capacitance,
        calc_charge_stored,
        calc_device_capacitances,
        get_package_inductance,
    )

CONSTRAINTS:
- Keep TestGateChargeModel and TestIgbtModel EXACTLY as they are (copy from existing file)
- Use pytest.approx for floating point comparisons
- Use wide tolerances (paper values may shift due to errata corrections)
- Must pass `ruff check`
- All tests must pass: `pytest tests/test_analytical_models.py -v`

ACCEPTANCE:
- All 22 tests pass
- GateChargeModel tests unchanged and passing
- IgbtModel tests unchanged and passing
- `ruff check` passes
```

**Input files**: `tests/test_analytical_models.py`, `transistordatabase/analytical_models.py`
**Output files**: `tests/test_analytical_models.py` (rewritten)
**Acceptance**: All 22+ tests pass; no regressions in GateChargeModel/IgbtModel tests

---

## Execution Order & Parallelism

```
Phase 1 (parallel, no dependencies):
    CB.1 (Diode fields)           ← haiku, ~2 min
    CB.2 (Dataclasses)            ← haiku, ~5 min
    CB.3 (Capacitance funcs)      ← haiku, ~3 min
    CB.9 (Package lookup)         ← haiku, ~2 min

Phase 2 (parallel, needs CB.2+CB.3):
    CB.4 (gm fitting)            ← sonnet, ~5 min
    CB.6 (Reverse recovery)      ← sonnet, ~5 min

Phase 3 (needs CB.2+CB.3+CB.4):
    CB.5 (Turn-off energy)       ← sonnet, ~10 min

Phase 4 (needs CB.5+CB.6):
    CB.7 (Turn-on energy)        ← sonnet, ~10 min

Phase 5 (needs CB.1+CB.7+CB.9):
    CB.8 (from_transistor)       ← sonnet, ~8 min

Phase 6 (needs CB.8):
    CB.9b (Remove old BielaModel) ← haiku, ~2 min
    CB.10 (Exports)               ← haiku, ~2 min

Phase 7 (needs all):
    CB.11 (Tests)                ← sonnet, ~10 min
```

Total: 11 tasks, ~6 phases, ~4 can be parallelized in phase 1.

---

## Haiku Agent Invocation Template

```
You are working on the transistordatabase Python project at /home/tinix/claude_wsl/transistordatabase/.

TASK: {task description}

SOURCE FILE TO READ FIRST: {input file path}
TARGET FILE TO EDIT/CREATE: {output file path}

CONSTRAINTS:
- Must pass `ruff check` (line-length 88, PEP257 docstrings, Python 3.10 target)
- Type hints required on all function parameters and return values
- Use `from __future__ import annotations` at top of new files
- Sphinx docstrings with :param:, :type:, :return:, :rtype: sections
- Preserve all existing code in target file; only add new code (unless explicitly told to delete)

ACCEPTANCE: {acceptance criteria}
```

Tasks marked `sonnet` require multi-file reasoning or complex mathematical logic.
