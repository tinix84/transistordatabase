#!/usr/bin/env python3
"""Diagnostic script to understand Christen-Biela turn-off calculation."""
from __future__ import annotations

import numpy as np
from transistordatabase.analytical_models import (
    ChristenBielaModel,
    HalfBridgeParams,
    TransconductanceParams,
)


def diagnostic_test():
    """Run diagnostic test with detailed output."""
    # Try to reverse-engineer parameters from expected results
    # Expected at 600V, 20A:
    # - I_oss ≈ 8.33A
    # - t_rv ≈ 10.5ns => Q_oss = I_oss * t_rv = 8.33A * 10.5ns = 87.5nC
    # - V_mil ≈ 7.46V
    # - t_fi ≈ 3.5ns
    # - E_T,off ≈ 14.1uJ

    # Let's use Q_oss = 87.5nC from the expected t_rv calculation
    q_oss = 87.5e-9  # 87.5 nC

    # Device parameters
    c_gs = 2.5e-9  # 2.5 nF
    c_ds = 0.5e-9  # 0.5 nF
    c_gd = 0.3e-9  # 0.3 nF

    # Transconductance: at I_ch = I_0 - 2*I_oss = 20 - 2*8.33 = 3.34A
    # V_mil = V_th + I_ch/g_m => g_m = I_ch/(V_mil - V_th)
    # If V_mil = 7.46V, V_th = 3.5V, I_ch = 3.34A:
    # g_m = 3.34 / (7.46 - 3.5) = 0.84 S
    # But gm also depends on the transfer characteristic fit

    # Let's try different gm_params to match expected g_m ~ 0.84 S at I_ch ~ 3.34A
    gm_params = TransconductanceParams(
        k_1=0.5,      # Increase to get higher gm
        k_2=0.0,
        x=1.5,        # Try closer to linear
        v_th=3.5,
    )

    # Create model
    model = ChristenBielaModel(
        c_gs=c_gs,
        c_ds=c_ds,
        c_gd=c_gd,
        q_oss=q_oss,
        gm_params=gm_params,
        rr_params=None,
    )

    # Operating conditions
    params = HalfBridgeParams(
        v_0=600.0,
        i_0=20.0,
        v_g_on=20.0,
        v_g_off=-5.0,
        r_g=10.0,
        l_s=5e-9,
        l_d=10e-9,
    )

    # Test gm calculation
    print("=== Transconductance Test ===")
    for i_ch in [3.34, 5.0, 10.0, 15.0, 20.0]:
        gm_test = gm_params.calc_gm(i_ch)
        v_gs = gm_params.v_th + i_ch / gm_test
        print(f"I_ch = {i_ch:5.2f}A: g_m = {gm_test:.3f} S, V_gs = {v_gs:.2f}V")

    # Calculate turn-off energy
    result = model.calc_turn_off_energy(params)

    print("\n=== Turn-off Energy Results (Tuned Parameters) ===")
    print(f"E_T,off:  {result['e_off']*1e6:.2f} uJ  (target: ~14.1 uJ)")
    print(f"I_oss:    {result['i_oss']:.2f} A     (target: ~8.33 A)")
    print(f"V_mil:    {result['v_mil']:.2f} V     (target: ~7.46 V)")
    print(f"t_rv:     {result['t_rv']*1e9:.2f} ns    (target: ~10.5 ns)")
    print(f"t_fi:     {result['t_fi']*1e9:.2f} ns    (target: ~3.5 ns)")
    print(f"V_Ld:     {result['v_ld']:.2f} V")
    print(f"g_m:      {result['gm']:.3f} S       (target: ~0.84 S)")

    # Calculate channel current
    i_ch = params.i_0 - 2 * result['i_oss']
    print(f"I_ch:     {i_ch:.2f} A     (I_0 - 2*I_oss)")

    # Check energy breakdown
    e_1a = 0.5 * result['t_rv'] * params.v_0 * i_ch
    e_2a = 0.5 * result['t_fi'] * (params.v_0 + result['v_ld']) * i_ch
    print(f"\nEnergy breakdown:")
    print(f"  E_1a (voltage rise):  {e_1a*1e6:.2f} uJ")
    print(f"  E_2a (current fall):  {e_2a*1e6:.2f} uJ")
    print(f"  Total:                {result['e_off']*1e6:.2f} uJ")

    # Test the _solve_i_oss iteration
    print("\n=== I_oss Solver Iteration Debug ===")
    model_debug = ChristenBielaModel(
        c_gs=c_gs, c_ds=c_ds, c_gd=c_gd, q_oss=q_oss,
        gm_params=gm_params, rr_params=None
    )

    # Manual iteration to see convergence
    i_0 = 20.0
    v_g = -5.0
    r_g = 10.0
    l_s = 5e-9

    gm = gm_params.calc_gm(i_0)
    print(f"Initial: g_m(I_0={i_0:.1f}A) = {gm:.3f} S")

    for iteration in range(5):
        # Quadratic coefficients
        a = (2 * l_s) / (q_oss * r_g)
        b = (2 / (gm * r_g)) + (c_gd / (c_gd + c_ds))
        c = (1 / r_g) * (v_g - gm_params.v_th - i_0 / gm)

        discriminant = b**2 - 4*a*c

        if discriminant >= 0:
            root1 = (-b + np.sqrt(discriminant)) / (2*a)
            root2 = (-b - np.sqrt(discriminant)) / (2*a)
            i_oss = min(abs(root1), abs(root2))
        else:
            i_oss = 0.0

        i_ch = i_0 - 2 * i_oss
        gm_new = gm_params.calc_gm(max(i_ch, 0.1))
        v_mil = gm_params.v_th + i_ch / max(gm_new, 1e-6)

        print(f"Iter {iteration}: I_oss={i_oss:.2f}A, I_ch={i_ch:.2f}A, g_m={gm_new:.3f}S, V_mil={v_mil:.2f}V")

        if abs(gm_new - gm) / max(gm_new, 1e-6) < 0.01:
            print(f"  Converged (|Δg_m|/g_m < 1%)")
            break

        gm = gm_new


if __name__ == "__main__":
    print("Christen-Biela Turn-Off Diagnostic\n")
    print("=" * 70)
    diagnostic_test()
