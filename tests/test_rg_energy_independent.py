"""Check measured gate-resistance lookups with independent energy curves."""
from dataclasses import replace

import numpy as np
import pytest

from transistordatabase.core.models import SwitchingLossData
from transistordatabase.rg_formula import (
    find_rg_for_target_energy,
    interpolate_energy_at_rg,
)


def resistance_curve():
    """Record a descending energy curve in unsorted source order."""
    return SwitchingLossData(
        dataset_type="graph_r_e", t_j=75.0, v_supply=400.0, v_g=12.0,
        i_x=10.0, graph_r_e=np.array([[9.0, 1.0, 5.0], [0.0, 0.008, 0.004]]),
    )


def current_curve(resistance, energy_at_ten):
    """Record a rising current sweep about its 10 A midpoint."""
    return SwitchingLossData(
        dataset_type="graph_i_e", t_j=75.0, v_supply=400.0, v_g=12.0,
        r_g=resistance,
        graph_i_e=np.array([[14.0, 6.0], [energy_at_ten + 0.001, energy_at_ten - 0.001]]),
    )


def test_descending_resistance_curve_exact_slice_and_zero_endpoint():
    """Recover 7 ohms at 2 mJ and preserve the measured zero at 9 ohms."""
    direct = resistance_curve()
    original = direct.graph_r_e.copy()
    records = [replace(direct, v_supply=800.0, i_x=20.0), replace(direct, t_j=25.0), direct]
    assert interpolate_energy_at_rg(records, 7, 10, 75) == pytest.approx(0.002)
    assert find_rg_for_target_energy(records, 0.002, 10, 75) == pytest.approx(7)
    assert interpolate_energy_at_rg(records, 9, 10, 75) == 0
    with pytest.raises(ValueError):
        interpolate_energy_at_rg(records, 9, 9, 75)
    np.testing.assert_array_equal(direct.graph_r_e, original)


def test_select_representation_and_preserve_measured_current_curves():
    """Allow each measured representation and reject incomplete inverse coverage."""
    direct = resistance_curve()
    curves = [current_curve(9, 0.004), current_curve(1, 0.012)]
    originals = [curve.graph_i_e.copy() for curve in curves]
    records = [direct, *curves]
    with pytest.raises(ValueError):
        find_rg_for_target_energy(records, 0.008, 10, 75)
    assert find_rg_for_target_energy(
        records, 0.008, 10, 75, representation="graph_i_e",
    ) == pytest.approx(5)
    assert find_rg_for_target_energy(
        records, 0.008, 10, 75, representation="graph_r_e",
    ) == 1
    for curve, original in zip(curves, originals, strict=True):
        np.testing.assert_array_equal(curve.graph_i_e, original)
    hole = replace(curves[0], r_g=5, graph_i_e=np.array([[15, 20], [0.006, 0.010]]))
    with pytest.raises(ValueError):
        find_rg_for_target_energy([*curves, hole], 0.008, 10, 75)
    with pytest.raises(ValueError):
        find_rg_for_target_energy(
            [*records, replace(direct, v_supply=800)], 0.008, 10, 75,
            representation="graph_r_e",
        )


@pytest.mark.parametrize("middle_energy,target", [(0.002, 0.004), (0.008, 0.008)])
def test_nonunique_inverse_preserves_source(middle_energy, target):
    """Reject multiple crossings and plateaus without changing the samples."""
    direct = replace(
        resistance_curve(), graph_r_e=np.array([[1, 4, 9], [0.008, middle_energy, 0.006]]),
    )
    original = direct.graph_r_e.copy()
    with pytest.raises(ValueError):
        find_rg_for_target_energy([direct], target, 10, 75)
    np.testing.assert_array_equal(direct.graph_r_e, original)


def test_zero_gate_bias_and_explicit_unknown_select_distinct_curves():
    """Distinguish a grounded gate from an unknown bias and invalid evidence."""
    direct = resistance_curve()
    records = [direct, replace(direct, v_g_off=0, graph_r_e=direct.graph_r_e * [[1], [2]])]
    assert interpolate_energy_at_rg(
        records, 5, 10, 75, conditions={"v_g_off": 0},
    ) == pytest.approx(0.008)
    assert interpolate_energy_at_rg(
        records, 5, 10, 75, conditions={"v_g_off": None},
    ) == pytest.approx(0.004)
    with pytest.raises(ValueError):
        interpolate_energy_at_rg([replace(direct, v_g_off=False)], 5, 10, 75)


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 1, 0)])
def test_all_currents_share_one_commutation_inductance(order):
    """Reject physically distinct endpoints in every source order."""
    # Adjacent conditions match the 1 nH tolerance. The endpoints differ by 1.5 nH.
    curves = [
        replace(current_curve(rg, energy), commutation_inductance=inductance)
        for rg, energy, inductance in (
            (1, 0.012, 10e-9), (5, 0.008, 10.75e-9), (9, 0.004, 11.5e-9),
        )
    ]
    with pytest.raises(ValueError):
        find_rg_for_target_energy([curves[index] for index in order], 0.006, 10, 75)
