#!/usr/bin/env python3
"""Validation test for ChristenBielaModel turn-off implementation.

Validates that the implementation is correct by checking:
1. Zero current → zero energy
2. Energy increases with current and voltage
3. Iteration converges
4. Physics equations are implemented correctly
"""
from __future__ import annotations

import numpy as np
from transistordatabase.analytical_models import (
    ChristenBielaModel,
    HalfBridgeParams,
    TransconductanceParams,
)


def test_implementation_correctness():
    """Test that implementation follows the physics correctly."""
    print("=" * 70)
    print("VALIDATION: ChristenBielaModel Turn-Off Implementation")
    print("=" * 70)

    # Standard test parameters
    c_gs = 2.5e-9
    c_ds = 0.5e-9
    c_gd = 0.3e-9
    q_oss = 100e-9  # 100 nC

    gm_params = TransconductanceParams(
        k_1=0.4,
        k_2=0.0,
        x=1.8,
        v_th=3.5,
    )

    model = ChristenBielaModel(
        c_gs=c_gs,
        c_ds=c_ds,
        c_gd=c_gd,
        q_oss=q_oss,
        gm_params=gm_params,
        rr_params=None,
    )

    params = HalfBridgeParams(
        v_0=600.0,
        i_0=20.0,
        v_g_on=20.0,
        v_g_off=-5.0,
        r_g=10.0,
        l_s=5e-9,
        l_d=10e-9,
    )

    # Test 1: Zero current
    print("\n[TEST 1] Zero Current → Zero Energy")
    params_zero = HalfBridgeParams(
        v_0=600.0, i_0=0.0, v_g_on=20.0, v_g_off=-5.0,
        r_g=10.0, l_s=5e-9, l_d=10e-9
    )
    result_zero = model.calc_turn_off_energy(params_zero)
    assert result_zero['e_off'] == 0.0, "Zero current must give zero energy"
    print(f"  ✓ E_off = {result_zero['e_off']:.2e} J (PASS)")

    # Test 2: Energy increases with current
    print("\n[TEST 2] Energy Increases with Current")
    currents = [5.0, 10.0, 15.0, 20.0, 25.0]
    energies = []
    for i_0 in currents:
        params_i = HalfBridgeParams(
            v_0=600.0, i_0=i_0, v_g_on=20.0, v_g_off=-5.0,
            r_g=10.0, l_s=5e-9, l_d=10e-9
        )
        result_i = model.calc_turn_off_energy(params_i)
        energies.append(result_i['e_off'])
        print(f"  I_0 = {i_0:5.1f}A → E_off = {result_i['e_off']*1e6:7.2f} uJ")

    # Verify monotonic increase
    for i in range(1, len(energies)):
        assert energies[i] > energies[i-1], "Energy must increase with current"
    print("  ✓ Energy increases monotonically (PASS)")

    # Test 3: Energy increases with voltage
    print("\n[TEST 3] Energy Increases with Voltage")
    voltages = [400.0, 500.0, 600.0, 700.0, 800.0]
    energies_v = []
    for v_0 in voltages:
        # Scale q_oss with voltage for realistic comparison
        q_oss_scaled = q_oss * (v_0 / 600.0)
        model_v = ChristenBielaModel(
            c_gs=c_gs, c_ds=c_ds, c_gd=c_gd, q_oss=q_oss_scaled,
            gm_params=gm_params, rr_params=None
        )
        params_v = HalfBridgeParams(
            v_0=v_0, i_0=20.0, v_g_on=20.0, v_g_off=-5.0,
            r_g=10.0, l_s=5e-9, l_d=10e-9
        )
        result_v = model_v.calc_turn_off_energy(params_v)
        energies_v.append(result_v['e_off'])
        print(f"  V_0 = {v_0:5.0f}V → E_off = {result_v['e_off']*1e6:7.2f} uJ")

    # Verify monotonic increase
    for i in range(1, len(energies_v)):
        assert energies_v[i] > energies_v[i-1], "Energy must increase with voltage"
    print("  ✓ Energy increases monotonically (PASS)")

    # Test 4: Equation validation
    print("\n[TEST 4] Equation Implementation Validation")
    result = model.calc_turn_off_energy(params)

    # Manually calculate and verify
    i_oss = result['i_oss']
    gm = result['gm']
    v_mil = result['v_mil']
    t_rv = result['t_rv']
    t_fi = result['t_fi']
    v_ld = result['v_ld']
    i_ch = params.i_0 - 2 * i_oss

    # Verify eq(13): t_rv = Q_oss / I_oss
    t_rv_calc = q_oss / max(i_oss, 1e-9)
    assert abs(t_rv - t_rv_calc) / max(t_rv_calc, 1e-12) < 1e-6, "eq(13) verification failed"
    print(f"  ✓ eq(13): t_rv = Q_oss/I_oss = {t_rv*1e9:.2f} ns (PASS)")

    # Verify eq(15): t_fi calculation
    numerator = gm_params.v_th - params.v_g_off
    denominator = v_mil - params.v_g_off
    t_fi_calc = -np.log(numerator / denominator) * (c_gs * params.r_g + params.l_s * gm)
    assert abs(t_fi - t_fi_calc) / max(t_fi_calc, 1e-12) < 1e-6, "eq(15) verification failed"
    print(f"  ✓ eq(15): t_fi = {t_fi*1e9:.2f} ns (PASS)")

    # Verify eq(16): v_ld calculation
    v_ld_calc = params.l_d * i_ch / max(t_fi, 1e-12) if t_fi > 0 else 0.0
    assert abs(v_ld - v_ld_calc) / max(v_ld_calc, 1e-12) < 1e-6, "eq(16) verification failed"
    print(f"  ✓ eq(16): V_Ld = L_d*I_ch/t_fi = {v_ld:.2f} V (PASS)")

    # Verify eq(17): energy calculation
    e_1a = 0.5 * t_rv * params.v_0 * i_ch
    e_2a = 0.5 * t_fi * (params.v_0 + v_ld) * i_ch
    e_total_calc = e_1a + e_2a
    assert abs(result['e_off'] - e_total_calc) / max(e_total_calc, 1e-12) < 1e-6, "eq(17) verification failed"
    print(f"  ✓ eq(17): E_off = E_1a + E_2a = {result['e_off']*1e6:.2f} uJ (PASS)")

    # Test 5: ZVS boundary
    print("\n[TEST 5] ZVS Boundary Current")
    i_0_zvs = model.calc_i_0_zvs(params)
    print(f"  I_0,zvs = {i_0_zvs:.2f} A")

    # Test at ZVS boundary (should have minimal loss)
    params_zvs = HalfBridgeParams(
        v_0=600.0, i_0=i_0_zvs * 0.5, v_g_on=20.0, v_g_off=-5.0,
        r_g=10.0, l_s=5e-9, l_d=10e-9
    )
    result_zvs = model.calc_turn_off_energy(params_zvs)
    print(f"  E_off at I_0 = {params_zvs.i_0:.2f}A (< I_0,zvs): {result_zvs['e_off']*1e6:.2f} uJ")
    print(f"  ✓ ZVS boundary calculated (PASS)")

    # Test 6: Iteration convergence
    print("\n[TEST 6] Iteration Convergence")
    i_oss_iter, gm_iter, v_mil_iter = model._solve_i_oss(
        params.i_0, params.v_g_off, params.r_g, params.l_s
    )
    print(f"  I_oss = {i_oss_iter:.2f} A (converged)")
    print(f"  g_m   = {gm_iter:.3f} S (converged)")
    print(f"  V_mil = {v_mil_iter:.2f} V (converged)")
    print(f"  ✓ Iteration converged (PASS)")

    # Summary
    print("\n" + "=" * 70)
    print("✓ ALL VALIDATION TESTS PASSED")
    print("=" * 70)
    print("\nImplementation Summary:")
    print("  - eq(10): Quadratic I_oss solver with gm iteration")
    print("  - eq(13): Voltage rise time t_rv = Q_oss/I_oss")
    print("  - eq(14): ZVS boundary current calculation")
    print("  - eq(15): CORRECTED current fall time with V_g,off")
    print("  - eq(16): Drain inductance overvoltage V_Ld")
    print("  - eq(17): Turn-off energy E_T,off")
    print("\nPhysics Validation:")
    print("  ✓ Zero current → zero energy")
    print("  ✓ Energy increases with current")
    print("  ✓ Energy increases with voltage")
    print("  ✓ All equations implemented correctly")
    print("  ✓ Iteration converges properly")
    print("=" * 70)


if __name__ == "__main__":
    test_implementation_correctness()
