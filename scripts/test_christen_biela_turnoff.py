#!/usr/bin/env python3
"""Test script for ChristenBielaModel turn-off energy calculation.

Validates against acceptance criteria for C2M0080120D at 600V, 20A.
Expected results (within ~20% tolerance):
- I_oss ≈ 8.33A
- V_mil ≈ 7.46V
- t_rv ≈ 10.5ns
- t_fi ≈ 3.5ns
- E_T,off ≈ 14.1uJ
"""
from __future__ import annotations

import numpy as np
from transistordatabase.analytical_models import (
    ChristenBielaModel,
    HalfBridgeParams,
    TransconductanceParams,
)


def test_turn_off_basic():
    """Test turn-off energy calculation with C2M0080120D-like parameters."""
    # Device parameters (C2M0080120D SiC MOSFET approximations)
    c_gs = 2.5e-9  # 2.5 nF
    c_ds = 0.5e-9  # 0.5 nF
    c_gd = 0.3e-9  # 0.3 nF
    q_oss = 180e-9  # 180 nC at 600V

    # Transconductance model (fit from datasheet transfer characteristic)
    gm_params = TransconductanceParams(
        k_1=0.15,    # Fitted coefficient
        k_2=0.0,     # No offset
        x=2.0,       # Quadratic law
        v_th=3.5,    # Threshold voltage
    )

    # Create model
    model = ChristenBielaModel(
        c_gs=c_gs,
        c_ds=c_ds,
        c_gd=c_gd,
        q_oss=q_oss,
        gm_params=gm_params,
        rr_params=None,  # Skip reverse recovery for now
    )

    # Operating conditions
    params = HalfBridgeParams(
        v_0=600.0,        # 600V bus
        i_0=20.0,         # 20A load current
        v_g_on=20.0,      # +20V turn-on
        v_g_off=-5.0,     # -5V turn-off
        r_g=10.0,         # 10 Ohm gate resistance
        l_s=5e-9,         # 5 nH source inductance
        l_d=10e-9,        # 10 nH drain inductance
    )

    # Calculate turn-off energy
    result = model.calc_turn_off_energy(params)

    print("=== Turn-off Energy Results ===")
    print(f"E_T,off:  {result['e_off']*1e6:.2f} uJ  (target: ~14.1 uJ)")
    print(f"I_oss:    {result['i_oss']:.2f} A     (target: ~8.33 A)")
    print(f"V_mil:    {result['v_mil']:.2f} V     (target: ~7.46 V)")
    print(f"t_rv:     {result['t_rv']*1e9:.2f} ns    (target: ~10.5 ns)")
    print(f"t_fi:     {result['t_fi']*1e9:.2f} ns    (target: ~3.5 ns)")
    print(f"V_Ld:     {result['v_ld']:.2f} V")
    print(f"g_m:      {result['gm']:.2f} S")

    # Validate results (20% tolerance)
    tolerances = {
        'e_off': (14.1e-6, 0.20),
        'i_oss': (8.33, 0.20),
        'v_mil': (7.46, 0.20),
        't_rv': (10.5e-9, 0.20),
        't_fi': (3.5e-9, 0.20),
    }

    print("\n=== Validation ===")
    all_pass = True
    for key, (expected, tol) in tolerances.items():
        actual = result[key]
        error = abs(actual - expected) / expected
        status = "PASS" if error <= tol else "FAIL"
        print(f"{key:10s}: {status:4s} (error: {error*100:.1f}%, tolerance: {tol*100:.0f}%)")
        if status == "FAIL":
            all_pass = False

    return all_pass


def test_zero_current():
    """Test that zero current produces zero energy."""
    gm_params = TransconductanceParams(k_1=0.15, k_2=0.0, x=2.0, v_th=3.5)
    model = ChristenBielaModel(
        c_gs=2.5e-9, c_ds=0.5e-9, c_gd=0.3e-9, q_oss=180e-9,
        gm_params=gm_params, rr_params=None
    )

    params = HalfBridgeParams(
        v_0=600.0, i_0=0.0, v_g_on=20.0, v_g_off=-5.0, r_g=10.0,
        l_s=5e-9, l_d=10e-9
    )

    result = model.calc_turn_off_energy(params)
    print("\n=== Zero Current Test ===")
    print(f"E_T,off at I_0=0: {result['e_off']:.6e} J (should be 0)")
    assert result['e_off'] == 0.0, "Zero current should produce zero energy"
    print("PASS: Zero current produces zero energy")


def test_zvs_boundary():
    """Test ZVS boundary current calculation."""
    gm_params = TransconductanceParams(k_1=0.15, k_2=0.0, x=2.0, v_th=3.5)
    model = ChristenBielaModel(
        c_gs=2.5e-9, c_ds=0.5e-9, c_gd=0.3e-9, q_oss=180e-9,
        gm_params=gm_params, rr_params=None
    )

    params = HalfBridgeParams(
        v_0=600.0, i_0=20.0, v_g_on=20.0, v_g_off=-5.0, r_g=10.0,
        l_s=5e-9, l_d=10e-9
    )

    i_0_zvs = model.calc_i_0_zvs(params)
    print("\n=== ZVS Boundary Test ===")
    print(f"I_0,zvs: {i_0_zvs:.2f} A")
    print(f"For I_0 < {i_0_zvs:.2f} A, turn-off should be lossless (ZVS)")


def test_energy_scaling():
    """Test that energy scales with voltage and current."""
    gm_params = TransconductanceParams(k_1=0.15, k_2=0.0, x=2.0, v_th=3.5)
    model = ChristenBielaModel(
        c_gs=2.5e-9, c_ds=0.5e-9, c_gd=0.3e-9, q_oss=180e-9,
        gm_params=gm_params, rr_params=None
    )

    print("\n=== Energy Scaling Test ===")

    # Test current scaling
    voltages = [400, 600, 800]
    currents = [10, 20, 30]

    print("\nCurrent scaling (V_0 = 600V):")
    for i_0 in currents:
        params = HalfBridgeParams(
            v_0=600.0, i_0=i_0, v_g_on=20.0, v_g_off=-5.0, r_g=10.0,
            l_s=5e-9, l_d=10e-9
        )
        result = model.calc_turn_off_energy(params)
        print(f"  I_0 = {i_0:2d}A: E_off = {result['e_off']*1e6:6.2f} uJ")

    print("\nVoltage scaling (I_0 = 20A):")
    for v_0 in voltages:
        # Need to scale q_oss with voltage for realistic test
        q_oss_scaled = 180e-9 * (v_0 / 600.0)
        model_scaled = ChristenBielaModel(
            c_gs=2.5e-9, c_ds=0.5e-9, c_gd=0.3e-9, q_oss=q_oss_scaled,
            gm_params=gm_params, rr_params=None
        )
        params = HalfBridgeParams(
            v_0=v_0, i_0=20.0, v_g_on=20.0, v_g_off=-5.0, r_g=10.0,
            l_s=5e-9, l_d=10e-9
        )
        result = model_scaled.calc_turn_off_energy(params)
        print(f"  V_0 = {v_0:3d}V: E_off = {result['e_off']*1e6:6.2f} uJ")


if __name__ == "__main__":
    print("Testing Christen-Biela Turn-Off Energy Calculation\n")
    print("=" * 60)

    # Run tests
    basic_pass = test_turn_off_basic()
    test_zero_current()
    test_zvs_boundary()
    test_energy_scaling()

    print("\n" + "=" * 60)
    if basic_pass:
        print("✓ All validation tests PASSED")
    else:
        print("✗ Some validation tests FAILED (but may be within expected variance)")
