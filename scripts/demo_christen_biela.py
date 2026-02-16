#!/usr/bin/env python3
"""Demonstration of ChristenBielaModel turn-off energy calculation.

Shows typical usage and key features of the implementation.
"""
from __future__ import annotations

import numpy as np
from transistordatabase.analytical_models import (
    ChristenBielaModel,
    HalfBridgeParams,
    TransconductanceParams,
)


def main():
    """Demonstrate ChristenBielaModel usage."""
    print("=" * 70)
    print("Christen-Biela Turn-Off Energy Model - Demonstration")
    print("=" * 70)

    # Define device parameters (example SiC MOSFET)
    print("\n[1] Device Parameters")
    print("-" * 70)

    gm_params = TransconductanceParams(
        k_1=0.4,      # Transfer characteristic coefficient
        k_2=0.0,      # Offset
        x=1.8,        # Exponent (1.5-2.5 typical for MOSFETs)
        v_th=3.5,     # Threshold voltage [V]
    )

    model = ChristenBielaModel(
        c_gs=2.5e-9,   # 2.5 nF gate-source capacitance
        c_ds=0.5e-9,   # 0.5 nF drain-source capacitance
        c_gd=0.3e-9,   # 0.3 nF gate-drain (Miller) capacitance
        q_oss=100e-9,  # 100 nC charge stored at 600V
        gm_params=gm_params,
        rr_params=None,  # No reverse recovery for this demo
    )

    print(f"  C_gs   = {model.c_gs*1e9:.1f} nF")
    print(f"  C_ds   = {model.c_ds*1e9:.1f} nF")
    print(f"  C_gd   = {model.c_gd*1e9:.1f} nF")
    print(f"  C_oss  = {model.c_oss*1e9:.1f} nF")
    print(f"  Q_oss  = {model.q_oss*1e9:.1f} nC")
    print(f"  V_th   = {gm_params.v_th:.1f} V")

    # Define operating conditions
    print("\n[2] Operating Conditions")
    print("-" * 70)

    params = HalfBridgeParams(
        v_0=600.0,      # 600V DC bus
        i_0=20.0,       # 20A load current
        v_g_on=20.0,    # +20V gate drive (turn-on)
        v_g_off=-5.0,   # -5V gate drive (turn-off)
        r_g=10.0,       # 10Ω total gate resistance
        l_s=5e-9,       # 5nH source inductance
        l_d=10e-9,      # 10nH drain inductance
    )

    print(f"  V_0      = {params.v_0:.0f} V")
    print(f"  I_0      = {params.i_0:.1f} A")
    print(f"  V_g,on   = {params.v_g_on:.0f} V")
    print(f"  V_g,off  = {params.v_g_off:.0f} V")
    print(f"  R_g      = {params.r_g:.1f} Ω")
    print(f"  L_s      = {params.l_s*1e9:.1f} nH")
    print(f"  L_d      = {params.l_d*1e9:.1f} nH")

    # Calculate turn-off energy
    print("\n[3] Turn-Off Energy Calculation")
    print("-" * 70)

    result = model.calc_turn_off_energy(params)

    print(f"  E_T,off  = {result['e_off']*1e6:.2f} µJ")
    print(f"  I_oss    = {result['i_oss']:.2f} A")
    print(f"  V_mil    = {result['v_mil']:.2f} V")
    print(f"  t_rv     = {result['t_rv']*1e9:.2f} ns")
    print(f"  t_fi     = {result['t_fi']*1e9:.2f} ns")
    print(f"  V_Ld     = {result['v_ld']:.2f} V")
    print(f"  g_m      = {result['gm']:.3f} S")

    i_ch = params.i_0 - 2 * result['i_oss']
    print(f"\n  I_ch     = {i_ch:.2f} A  (channel current)")
    print(f"  t_total  = {(result['t_rv'] + result['t_fi'])*1e9:.2f} ns")

    # ZVS boundary
    print("\n[4] ZVS Boundary Analysis")
    print("-" * 70)

    i_0_zvs = model.calc_i_0_zvs(params)
    print(f"  I_0,zvs  = {i_0_zvs:.2f} A")
    print(f"  → For I_0 < {i_0_zvs:.2f} A: lossless turn-off (ZVS)")
    print(f"  → For I_0 > {i_0_zvs:.2f} A: hard-switched turn-off")

    # Energy sweep over current
    print("\n[5] Turn-Off Energy vs. Load Current")
    print("-" * 70)

    currents = np.linspace(0, 40, 9)
    print(f"  {'I_0 [A]':>8s}  {'E_off [µJ]':>10s}  {'t_total [ns]':>12s}")
    print("  " + "-" * 34)

    for i_0 in currents:
        params_sweep = HalfBridgeParams(
            v_0=600.0, i_0=i_0, v_g_on=20.0, v_g_off=-5.0,
            r_g=10.0, l_s=5e-9, l_d=10e-9
        )
        res = model.calc_turn_off_energy(params_sweep)
        t_total = res['t_rv'] + res['t_fi']
        print(f"  {i_0:8.1f}  {res['e_off']*1e6:10.2f}  {t_total*1e9:12.2f}")

    # Energy breakdown
    print("\n[6] Energy Breakdown (I_0 = 20A)")
    print("-" * 70)

    e_1a = 0.5 * result['t_rv'] * params.v_0 * i_ch
    e_2a = 0.5 * result['t_fi'] * (params.v_0 + result['v_ld']) * i_ch

    print(f"  E_1a (voltage rise):   {e_1a*1e6:6.2f} µJ  ({e_1a/result['e_off']*100:.1f}%)")
    print(f"  E_2a (current fall):   {e_2a*1e6:6.2f} µJ  ({e_2a/result['e_off']*100:.1f}%)")
    print(f"  Total turn-off:        {result['e_off']*1e6:6.2f} µJ  (100.0%)")

    # Switching frequency impact
    print("\n[7] Average Power Loss vs. Switching Frequency")
    print("-" * 70)

    frequencies = [10e3, 50e3, 100e3, 200e3, 500e3]  # kHz
    print(f"  {'f_sw [kHz]':>12s}  {'P_sw [W]':>10s}")
    print("  " + "-" * 24)

    for f_sw in frequencies:
        p_sw = result['e_off'] * f_sw
        print(f"  {f_sw/1e3:12.0f}  {p_sw:10.2f}")

    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
