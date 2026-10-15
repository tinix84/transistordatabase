"""Independent fixtures for the web topology calculation core."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from transistordatabase.core.models import (
    ChannelCharacteristics,
    ElectricalRatings,
    FosterThermalModel,
    SwitchingLossData,
    ThermalProperties,
    Transistor,
    TransistorMetadata,
)
from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.topologies.operating_point import (
    _dcm_output_ripple,
    calculate_topology_operating_point,
    calculate_topology_sweep,
)


def point(**overrides: Any) -> dict[str, Any]:
    """Return one independent baseline CCM case with overrides."""
    values = {
        "mode": "estimate",
        "v_in": 100.0,
        "v_out": 50.0,
        "p_out": 500.0,
        "f_sw": 10_000.0,
        "l_value": 200e-6,
        "c_out": 100e-6,
    }
    values.update(overrides)
    return values


def device(
    name: str = "curve-device", *, max_current: float = 100.0
) -> Transistor:
    """Build exact-condition switch, diode, and switching curves for tests."""
    transistor = Transistor(
        TransistorMetadata(name, "MOSFET", "test", "vendor", "package"),
        ElectricalRatings(1200, 100, 50, 175),
        ThermalProperties(0.001, 0.001, r_th_switch_cs=0.2, r_th_diode_cs=0.2),
    )
    currents = np.array([0.1, max_current])
    transistor.switch.channel_data.append(
        ChannelCharacteristics(25, np.array([[0.01, 10.0], currents]), 15)
    )
    transistor.diode.channel_data.append(
        ChannelCharacteristics(25, np.array([[0.4, 5.4], currents]), None)
    )
    transistor.switch.e_on_data.append(SwitchingLossData(
        "graph_i_e", 25, 100, 15, r_g=10, v_g_off=-5,
        graph_i_e=np.array([[0.1, max_current], [0.0001, 0.1]]),
        commutation_device="fixture-diode", measurement_testbench="independent-fixture",
    ))
    transistor.switch.e_off_data.append(SwitchingLossData(
        "graph_i_e", 25, 100, 15, r_g=10, v_g_off=-5,
        graph_i_e=np.array([[0.1, max_current], [0.0002, 0.2]]),
        commutation_device="fixture-diode", measurement_testbench="independent-fixture",
    ))
    return transistor


@pytest.mark.parametrize(
    ("topology", "values", "duty", "i_in", "i_out"),
    [
        ("buck", point(), 0.5, 5.0, 10.0),
        (
            "boost",
            point(v_in=100, v_out=200, p_out=1000, l_value=1e-3, f_sw=20_000),
            0.5,
            10.0,
            5.0,
        ),
        (
            "buck-boost",
            point(v_in=100, v_out=200, p_out=1000, l_value=1e-3, f_sw=20_000),
            2 / 3,
            10.0,
            5.0,
        ),
    ],
)
def test_ideal_operating_points_match_independent_power_balance(
    topology: str,
    values: dict[str, Any],
    duty: float,
    i_in: float,
    i_out: float,
) -> None:
    """Match independently calculated duty and ideal input/output currents."""
    result = calculate_topology_operating_point(topology, values)
    calculated = result["calculated"]
    assert calculated["duty_cycle"] == pytest.approx(duty)
    assert calculated["i_in_avg_a"] == pytest.approx(i_in)
    assert calculated["i_out_avg_a"] == pytest.approx(i_out)
    assert values["v_in"] * calculated["i_in_avg_a"] == pytest.approx(values["p_out"])
    assert values["v_out"] * calculated["i_out_avg_a"] == pytest.approx(values["p_out"])
    assert result["losses"] is None
    assert result["performance"]["efficiency_percent"] is None


def test_buck_ripple_matches_independently_derived_fixture() -> None:
    """Match the buck inductor and capacitor ripple equations."""
    result = calculate_topology_operating_point("buck", point())
    assert result["calculated"]["i_l_ripple_pp_a"] == pytest.approx(12.5)
    assert result["calculated"]["v_out_ripple_pp_v"] == pytest.approx(1.5625)
    assert result["calculated"]["i_l_avg_a"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("inductance", "expected_ripple"),
    [(27.7777777778e-6, 0.125), (300e-6, 0.045)],
)
def test_boost_ccm_output_ripple_integrates_capacitor_current_segments(
    inductance: float, expected_ripple: float
) -> None:
    """Cover diode-current crossing and non-crossing CCM ripple cases."""
    result = calculate_topology_operating_point(
        "boost",
        point(
            v_in=100,
            v_out=1000 / 9,
            p_out=1000,
            f_sw=20_000,
            l_value=inductance,
            c_out=1e-3,
        ),
    )
    assert result["calculated"]["v_out_ripple_pp_v"] == pytest.approx(
        expected_ripple, rel=1e-4
    )


def test_inverting_buck_boost_reports_output_polarity() -> None:
    """Report the conventional inverting output sign separately from magnitude."""
    result = calculate_topology_operating_point(
        "buck-boost", point(v_in=100, v_out=200, p_out=1000, l_value=1e-3, f_sw=20_000)
    )
    assert result["model"]["output_polarity"] == -1
    assert result["calculated"]["v_out_signed_v"] == -200
    assert result["calculated"]["i_l_avg_a"] == pytest.approx(15)


def test_mode_must_be_explicit_and_estimate_is_labeled() -> None:
    """Require and label an ideal estimate request."""
    with pytest.raises(ValueError, match="mode must be explicitly"):
        calculate_topology_operating_point("buck", {k: v for k, v in point().items() if k != "mode"})
    result = calculate_topology_operating_point("buck", point())
    assert result["model"]["mode"] == "estimate"
    assert any("does not calculate device" in item for item in result["model"]["assumptions"])


@pytest.mark.parametrize(
    ("topology", "values", "message"),
    [
        ("buck", point(v_in=100, v_out=100), "Buck requires 0 < v_out < v_in"),
        ("boost", point(v_in=100, v_out=90), "Boost requires v_out > v_in"),
        ("buck", point(l_value=1e-9), "Requested CCM point reaches zero inductor current; use DCM"),
    ],
)
def test_invalid_topologies_and_ccm_boundary_fail_with_supported_conditions(
    topology: str, values: dict[str, Any], message: str
) -> None:
    """Explain topology-ratio and requested-CCM boundary errors."""
    with pytest.raises(ValueError, match=message):
        calculate_topology_operating_point(topology, values)


def test_flyback_requires_a_positive_primary_to_secondary_turns_ratio() -> None:
    """Require transformer ratio rather than assuming a winding convention."""
    with pytest.raises(ValueError, match="Flyback requires a positive turns_ratio_nps"):
        calculate_topology_operating_point("flyback", point())


@pytest.mark.parametrize(
    ("topology", "values", "duty", "diode_fraction", "peak", "i_in", "i_out"),
    [
        (
            "buck", point(v_out=50, p_out=100, l_value=100e-6, f_sw=20_000, c_out=1e-3, conduction_regime="DCM"),
            0.2828427125, 0.2828427125, 7.071067812, 1.0, 2.0,
        ),
        (
            "boost", point(v_in=100, v_out=200, p_out=100, l_value=100e-6, f_sw=20_000, c_out=1e-3, conduction_regime="DCM"),
            0.1414213562, 0.1414213562, 7.071067812, 1.0, 0.5,
        ),
        (
            "buck-boost", point(v_in=100, v_out=50, p_out=100, l_value=100e-6, f_sw=20_000, c_out=1e-3, conduction_regime="DCM"),
            0.2, 0.4, 10.0, 1.0, 2.0,
        ),
    ],
)
def test_dcm_operating_points_match_independent_energy_and_current_fixtures(
    topology: str,
    values: dict[str, Any],
    duty: float,
    diode_fraction: float,
    peak: float,
    i_in: float,
    i_out: float,
) -> None:
    """Check analytical DCM duties, current triangles, idle interval, and power balance."""
    result = calculate_topology_operating_point(topology, values)
    calculated = result["calculated"]
    assert result["model"]["conduction_regime"] == "DCM"
    assert calculated["duty_cycle"] == pytest.approx(duty)
    assert calculated["diode_conduction_fraction"] == pytest.approx(diode_fraction)
    assert calculated["i_l_valley_a"] == 0
    assert calculated["i_l_peak_a"] == pytest.approx(peak)
    assert calculated["i_in_avg_a"] == pytest.approx(i_in)
    assert calculated["i_out_avg_a"] == pytest.approx(i_out)
    assert values["v_in"] * calculated["i_in_avg_a"] == pytest.approx(values["p_out"])
    assert values["v_out"] * calculated["i_out_avg_a"] == pytest.approx(values["p_out"])
    assert calculated["idle_fraction"] == pytest.approx(1 - duty - diode_fraction)
    assert min(result["waveforms"]["inductor_current_a"]) >= 0
    assert max(result["waveforms"]["inductor_current_a"]) == pytest.approx(peak)


def test_dcm_boost_output_ripple_integrates_the_capacitor_current() -> None:
    """Match the analytically integrated output capacitor charge extrema."""
    values = point(
        v_in=100, v_out=200, p_out=100, l_value=100e-6,
        f_sw=20_000, c_out=1e-3, conduction_regime="DCM",
    )
    result = calculate_topology_operating_point("boost", values)
    duty = np.sqrt(0.02)
    peak = np.sqrt(50.0)
    output_current = 0.5
    positive_charge_area = duty * (peak - output_current) ** 2 / (2 * peak)
    phase_charge = positive_charge_area - output_current * duty
    expected_ripple = (phase_charge + output_current * duty) / (20_000 * 1e-3)
    assert result["calculated"]["v_out_ripple_pp_v"] == pytest.approx(expected_ripple, rel=1e-4)


def test_dcm_request_at_or_past_boundary_explains_ccm_selection() -> None:
    """Do not return a DCM solution after the two conduction intervals overlap."""
    with pytest.raises(ValueError, match="at or above the CCM boundary.*select CCM"):
        calculate_topology_operating_point(
            "boost",
            point(v_in=100, v_out=200, p_out=1000, l_value=1e-3,
                  f_sw=20_000, conduction_regime="DCM"),
        )


def test_flyback_ccm_uses_explicit_turns_ratio_and_reports_voltage_stress() -> None:
    """Verify primary current, secondary current, and winding-referred stresses."""
    result = calculate_topology_operating_point(
        "flyback",
        point(v_in=100, v_out=50, p_out=100, l_value=1e-3,
              f_sw=20_000, c_out=1e-3, turns_ratio_nps=2),
    )
    calculated = result["calculated"]
    assert calculated["duty_cycle"] == pytest.approx(0.5)
    assert any(
        "ideal transformer with no leakage" in item
        for item in result["model"]["assumptions"]
    )
    assert calculated["i_in_avg_a"] == pytest.approx(1)
    assert calculated["i_out_avg_a"] == pytest.approx(2)
    assert calculated["i_l_avg_a"] == pytest.approx(2)
    assert calculated["i_diode_avg_a"] == pytest.approx(2)
    assert calculated["v_switch_block_v"] == pytest.approx(200)
    assert calculated["v_diode_block_v"] == pytest.approx(100)
    assert result["parameters"]["turns_ratio_nps"] == 2


def test_flyback_dcm_uses_secondary_current_and_discontinuous_intervals() -> None:
    """Check Flyback DCM current reflection and transformer duty boundary."""
    result = calculate_topology_operating_point(
        "flyback",
        point(v_in=100, v_out=50, p_out=100, l_value=100e-6,
              f_sw=20_000, c_out=1e-3, turns_ratio_nps=2,
              conduction_regime="DCM"),
    )
    calculated = result["calculated"]
    assert calculated["duty_cycle"] == pytest.approx(0.2)
    assert calculated["diode_conduction_fraction"] == pytest.approx(0.2)
    assert calculated["i_l_peak_a"] == pytest.approx(10)
    assert calculated["i_diode_peak_a"] == pytest.approx(20)
    assert calculated["i_in_avg_a"] == pytest.approx(1)
    assert calculated["i_out_avg_a"] == pytest.approx(2)
    assert calculated["idle_fraction"] == pytest.approx(0.6)


def test_detailed_mode_rejects_missing_required_curves() -> None:
    """Fail detailed mode when its switch channel curve is missing."""
    missing_curves = device()
    missing_curves.switch.channel_data.clear()
    with pytest.raises(ValueError, match="Missing required switch channel curve"):
        calculate_topology_operating_point(
            "buck", point(mode="detailed", t_j=25, v_g=15), missing_curves, device()
        )


def test_detailed_mode_uses_curves_read_only_and_records_provenance() -> None:
    """Calculate curve-backed partial losses without mutating source curves."""
    transistor = device()
    before = transistor.switch.e_on_data[0].graph_i_e.copy()
    result = calculate_topology_operating_point(
        "buck",
        point(mode="detailed", t_j=25, t_case=30, v_g=15, v_g_off=-5, r_g_on=10, r_g_off=10),
        transistor,
        transistor,
    )
    assert result["losses"]["conduction_switch_w"] > 0
    assert result["losses"]["conduction_diode_w"] > 0
    assert result["losses"]["switching_on_w"] > 0
    assert result["losses"]["switching_off_w"] > 0
    assert result["performance"]["p_in_w"] == pytest.approx(
        result["performance"]["p_out_w"] + result["losses"]["total_w"]
    )
    assert "ideal lossless operating point" in result["performance"]["efficiency_scope"]
    assert "loss feedback is not iterated" in result["performance"]["input_power_scope"]
    assert any(
        "calculated at the ideal lossless operating point" in item
        for item in result["model"]["assumptions"]
    )
    assert any(
        "Commutation-device metadata is recorded provenance" in item
        for item in result["model"]["assumptions"]
    )
    assert result["model"]["curve_conditions"]["e_on"]["commutation_device"] == "fixture-diode"
    assert result["model"]["curve_conditions"]["e_on"]["commutation_device_status"] == "unverified"
    assert result["model"]["curve_conditions"]["e_on"]["measurement_testbench"] == "independent-fixture"
    np.testing.assert_array_equal(transistor.switch.e_on_data[0].graph_i_e, before)


def test_conduction_only_mode_needs_channel_curves_but_not_switching_energy_curves() -> None:
    """Allow exact measured channel losses while keeping switching loss excluded."""
    transistor = device()
    transistor.switch.e_on_data.clear()
    transistor.switch.e_off_data.clear()
    result = calculate_topology_operating_point(
        "buck",
        point(mode="conduction-only", t_j=25, t_case=30, v_g=15),
        transistor,
        transistor,
    )
    assert result["losses"]["conduction_switch_w"] > 0
    assert result["losses"]["conduction_diode_w"] > 0
    assert result["losses"]["switching_on_w"] == 0
    assert result["losses"]["switching_off_w"] == 0
    assert result["losses"]["loss_scope"] == "channel_conduction_only"
    assert result["losses"]["switching_losses_included"] is False
    assert result["losses"]["reverse_recovery_losses_included"] is False
    assert result["losses"]["gate_drive_losses_included"] is False
    assert result["model"]["curve_conditions"]["diode_channel"]["v_g"] is None
    assert any(
        "not recalculated from device losses" in item
        for item in result["model"]["assumptions"]
    )
    assert "e_on" not in result["model"]["curve_conditions"]


def test_duplicate_zero_current_channel_points_use_the_upper_voltage_envelope() -> None:
    """Conservatively retain the higher diode voltage at duplicated zero current."""
    transistor = device()
    transistor.diode.channel_data[0].graph_v_i = np.array([
        [0.0, 1.44, 2.0],
        [0.0, 0.0, 100.0],
    ])

    result = calculate_topology_operating_point(
        "buck",
        point(mode="conduction-only", t_j=25, v_g=15),
        transistor,
        transistor,
    )

    average_current = 10.0
    ripple = 12.5
    expected_average_diode_power = 1.44 * average_current + 0.0056 * (
        average_current**2 + ripple**2 / 12
    )
    assert result["losses"]["conduction_diode_w"] == pytest.approx(
        0.5 * expected_average_diode_power, rel=1e-5
    )
    assert any(
        "maximum recorded voltage at each current" in assumption
        for assumption in result["model"]["assumptions"]
    )


def test_conduction_only_mode_calculates_dcm_channel_losses_when_curves_cover_zero_current() -> None:
    """Use exact channel curves for the switch/diode ramps in explicit DCM."""
    transistor = device()
    for curve in transistor.switch.channel_data + transistor.diode.channel_data:
        curve.graph_v_i[1, 0] = 0.0

    result = calculate_topology_operating_point(
        "buck",
        point(
            mode="conduction-only", t_j=25, v_g=15, v_in=100, v_out=50,
            p_out=100, l_value=100e-6, f_sw=20_000, c_out=1e-3,
            conduction_regime="DCM",
        ),
        transistor,
        transistor,
    )

    assert result["model"]["conduction_regime"] == "DCM"
    assert result["calculated"]["i_l_valley_a"] == 0
    assert result["losses"]["conduction_switch_w"] > 0
    assert result["losses"]["conduction_diode_w"] > 0
    assert result["losses"]["switching_losses_included"] is False


def test_flyback_dcm_channel_losses_use_secondary_referred_diode_current() -> None:
    """Use Np/Ns to select the Flyback rectifier's exact current range."""
    transistor = device()
    for curve in transistor.switch.channel_data + transistor.diode.channel_data:
        curve.graph_v_i[1, 0] = 0.0

    result = calculate_topology_operating_point(
        "flyback",
        point(
            mode="conduction-only", t_j=25, v_g=15,
            v_in=100, v_out=50, p_out=100, l_value=100e-6,
            f_sw=20_000, c_out=1e-3, turns_ratio_nps=2,
            conduction_regime="DCM",
        ),
        transistor,
        transistor,
    )

    assert result["calculated"]["i_l_peak_a"] == pytest.approx(10)
    assert result["calculated"]["i_diode_peak_a"] == pytest.approx(20)
    assert result["losses"]["conduction_switch_w"] > 0
    assert result["losses"]["conduction_diode_w"] > 0
    assert result["losses"]["loss_scope"] == "channel_conduction_only"


def test_conduction_only_mode_rejects_missing_required_channel_curves() -> None:
    """Do not silently substitute a channel estimate in the Pareto mode."""
    transistor = device()
    transistor.diode.channel_data.clear()
    with pytest.raises(ValueError, match="Missing required diode channel curve"):
        calculate_topology_operating_point(
            "boost", point(
                mode="conduction-only", t_j=25, v_g=15,
                v_in=100, v_out=200, p_out=100, l_value=1e-3, f_sw=20_000,
                conduction_regime="DCM",
            ),
            transistor, transistor,
        )


def test_loss_balanced_boost_matches_independent_constant_drop_power_balance() -> None:
    """Solve duty/current for a regulated Boost with exact flat channel curves."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = 1.0
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0
    values = point(
        mode="loss-balanced", topology="boost", v_in=100, v_out=200,
        p_out=1000, l_value=1e-3, f_sw=20_000, c_out=1e-3,
        t_j=25, v_g=15, t_case=60,
    )

    result = calculate_topology_operating_point(
        "boost", values, transistor, transistor
    )

    expected_duty = (200 + 2 - 100) / (200 + 1)
    calculated = result["calculated"]
    losses = result["losses"]
    assert calculated["duty_cycle"] == pytest.approx(expected_duty, rel=1e-9)
    assert calculated["i_l_avg_a"] == pytest.approx(
        1000 / (200 * (1 - expected_duty)), rel=1e-9
    )
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(0, abs=1e-6)
    assert result["performance"]["p_in_w"] == pytest.approx(
        100 * calculated["i_in_avg_a"]
    )
    assert result["performance"]["p_in_w"] == pytest.approx(
        1000 + losses["total_w"], rel=1e-8
    )
    assert result["performance"]["power_balance_residual_w"] == pytest.approx(0, abs=1e-6)
    assert losses["loss_scope"] == "channel_conduction_only"
    assert losses["switching_losses_included"] is False
    assert result["model"]["mode"] == "loss-balanced"
    assert result["model"]["thermal_closure"] == "unavailable"
    assert result["temperatures"]["thermal_closure"] == "unavailable"


def test_loss_balanced_ripple_refines_against_final_nonlinear_switch_drop() -> None:
    """Use the selected-curve mean drop, not the initial average-current drop."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i = np.array(
        [
            [0.5, 0.55, 0.6, 1.0, 2.0, 4.0, 10.0],
            [0.1, 8.0, 9.0, 10.0, 11.0, 12.0, 100.0],
        ]
    )
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0
    values = point(
        mode="loss-balanced", topology="boost", v_in=100, v_out=200,
        p_out=1000, l_value=1e-3, f_sw=20_000, c_out=1e-3,
        t_j=25, v_g=15,
    )

    result = calculate_topology_operating_point(
        "boost", values, transistor, transistor
    )
    calculated = result["calculated"]
    curve = transistor.switch.channel_data[0]
    valley = calculated["i_l_valley_a"]
    peak = calculated["i_l_peak_a"]
    currents = np.linspace(valley, peak, 65)
    voltages = np.interp(currents, curve.graph_v_i[1], curve.graph_v_i[0])
    mean_curve_drop = np.trapezoid(voltages, currents) / (peak - valley)
    ripple_from_final_curve_drop = (
        (values["v_in"] - mean_curve_drop)
        * calculated["duty_cycle"]
        / (values["l_value"] * values["f_sw"])
    )
    initial_average_drop = np.interp(
        calculated["i_l_avg_a"], curve.graph_v_i[1], curve.graph_v_i[0]
    )
    initial_average_drop_ripple = (
        (values["v_in"] - initial_average_drop)
        * calculated["duty_cycle"]
        / (values["l_value"] * values["f_sw"])
    )

    assert calculated["i_l_ripple_pp_a"] == pytest.approx(
        ripple_from_final_curve_drop, rel=1e-9
    )
    assert abs(calculated["i_l_ripple_pp_a"] - initial_average_drop_ripple) > 1e-4


def test_loss_balanced_boost_can_move_inside_ccm_when_ideal_point_is_on_boundary() -> None:
    """A loss-adjusted CCM solution may exist when ideal ripple reaches zero valley."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = 1.0
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0
    values = point(
        mode="loss-balanced", topology="boost", v_in=100, v_out=200,
        p_out=1000, l_value=125e-6, f_sw=20_000, c_out=1e-3,
        t_j=25, v_g=15,
    )

    result = calculate_topology_operating_point(
        "boost", values, transistor, transistor
    )

    calculated = result["calculated"]
    assert calculated["duty_cycle"] > 0.5
    assert calculated["i_l_valley_a"] == pytest.approx(0.10, abs=0.01)
    assert calculated["i_l_valley_a"] > 0
    assert result["performance"]["power_balance_residual_w"] == pytest.approx(
        0, abs=1e-6
    )


def test_loss_balanced_boost_advances_past_lower_curve_limit_to_supported_root() -> None:
    """Skip an unsupported ideal trial when the root remains in curve range."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = 1.0
    transistor.switch.channel_data[0].graph_v_i[1, 0] = 8.8
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0
    transistor.diode.channel_data[0].graph_v_i[1, 0] = 8.8

    result = calculate_topology_operating_point(
        "boost",
        point(
            mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
            f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
        ),
        transistor,
        transistor,
    )

    assert result["calculated"]["i_l_valley_a"] == pytest.approx(8.895, abs=0.01)
    assert result["calculated"]["duty_cycle"] > 0.5
    assert result["performance"]["power_balance_residual_w"] == pytest.approx(
        0, abs=1e-6
    )


def test_loss_balanced_boost_refines_first_supported_trial_above_root() -> None:
    """Bisect a lower curve boundary when its first valid trial overshoots root."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = 1.0
    transistor.switch.channel_data[0].graph_v_i[1, 0] = 8.89
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0
    transistor.diode.channel_data[0].graph_v_i[1, 0] = 8.89

    result = calculate_topology_operating_point(
        "boost",
        point(
            mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
            f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
        ),
        transistor,
        transistor,
    )

    assert result["calculated"]["duty_cycle"] == pytest.approx(
        (200 + 2 - 100) / (200 + 1), abs=1e-5
    )
    assert result["calculated"]["i_l_valley_a"] == pytest.approx(8.895, abs=0.01)
    assert result["performance"]["power_balance_residual_w"] == pytest.approx(
        0, abs=1e-6
    )


def test_loss_balanced_boost_brackets_root_before_upper_curve_limit() -> None:
    """Refine a coarse high-duty trial that exceeds the measured curve maximum."""
    transistor = device(max_current=11.5)
    transistor.switch.channel_data[0].graph_v_i[0] = 1.0
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0

    result = calculate_topology_operating_point(
        "boost",
        point(
            mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
            f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
        ),
        transistor,
        transistor,
    )

    assert result["calculated"]["duty_cycle"] == pytest.approx(
        (200 + 2 - 100) / (200 + 1), rel=1e-8
    )
    assert result["calculated"]["i_l_peak_a"] <= 11.5
    assert result["performance"]["power_balance_residual_w"] == pytest.approx(
        0, abs=1e-6
    )


def test_loss_balanced_boost_rejects_when_supported_interval_has_no_root() -> None:
    """Fail clearly when curves start above every CCM power-balanced point."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = 1.0
    transistor.switch.channel_data[0].graph_v_i[1, 0] = 10.5
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0
    transistor.diode.channel_data[0].graph_v_i[1, 0] = 10.5

    with pytest.raises(ValueError, match="No loss-balanced solution"):
        calculate_topology_operating_point(
            "boost",
            point(
                mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
                f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
            ),
            transistor,
            transistor,
        )


def test_loss_balanced_boost_rejects_no_positive_on_interval_voltage() -> None:
    """Reject a case whose measured switch drop leaves no CCM ramp solution."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = 100.0
    with pytest.raises(ValueError, match="no positive inductor-current ramp"):
        calculate_topology_operating_point(
            "boost",
            point(
                mode="loss-balanced", topology="boost", v_in=100, v_out=200,
                p_out=1000, l_value=1e-3, f_sw=20_000, t_j=25, v_g=15,
            ),
            transistor,
            transistor,
        )


def test_loss_balanced_flyback_dcm_closes_independent_flat_drop_fixture() -> None:
    """Check primary and secondary currents with flat Flyback channel drops."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="flyback", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=38.46153846153846, l_value=1e-3,
        f_sw=20_000, c_out=1e-3, t_j=25, v_g=15, turns_ratio_nps=2,
        parallel_switches=2, parallel_diodes=3,
    )
    switch_drop = 1.0
    diode_drop = 2.0
    n = values["turns_ratio_nps"]
    on_voltage = values["v_in"] - switch_drop
    off_voltage_primary = n * (values["v_out"] + diode_drop)
    period = 1 / values["f_sw"]
    peak = 2.0
    on_time = values["l_value"] * peak / on_voltage
    diode_time = values["l_value"] * peak / off_voltage_primary
    duty = on_time / period
    diode_fraction = diode_time / period
    idle_fraction = 1 - duty - diode_fraction
    input_current = 0.5 * peak * duty
    diode_current = n * 0.5 * peak * diode_fraction
    primary_average = 0.5 * peak * (duty + diode_fraction)
    switch_rms = np.sqrt(duty * peak**2 / 3)
    diode_rms = n * np.sqrt(diode_fraction * peak**2 / 3)
    expected_switch_loss = switch_drop * input_current
    expected_diode_loss = diode_drop * diode_current

    result = calculate_topology_operating_point(
        "flyback", values, transistor, transistor
    )

    calculated = result["calculated"]
    losses = result["losses"]
    assert calculated["duty_cycle"] == pytest.approx(duty, rel=1e-8)
    assert calculated["diode_conduction_fraction"] == pytest.approx(diode_fraction, rel=1e-8)
    assert calculated["idle_fraction"] == pytest.approx(idle_fraction, rel=1e-8)
    assert calculated["i_l_peak_a"] == pytest.approx(peak, rel=1e-8)
    assert calculated["i_l_avg_a"] == pytest.approx(primary_average, rel=1e-8)
    assert calculated["i_in_avg_a"] == pytest.approx(input_current, rel=1e-8)
    assert calculated["i_out_avg_a"] == pytest.approx(values["p_out"] / values["v_out"])
    assert calculated["i_diode_avg_a"] == pytest.approx(diode_current, rel=1e-8)
    assert calculated["i_diode_peak_a"] == pytest.approx(n * peak)
    assert calculated["i_switch_rms_a"] == pytest.approx(switch_rms, rel=1e-5)
    assert calculated["i_diode_rms_a"] == pytest.approx(diode_rms, rel=1e-5)
    assert losses["conduction_switch_w"] == pytest.approx(expected_switch_loss, rel=1e-8)
    assert losses["conduction_diode_w"] == pytest.approx(expected_diode_loss, rel=1e-8)
    assert losses["per_switch_device_w"] == pytest.approx(expected_switch_loss / 2, rel=1e-8)
    assert losses["per_diode_device_w"] == pytest.approx(expected_diode_loss / 3, rel=1e-8)
    assert result["performance"]["power_balance_residual_w"] == pytest.approx(0, abs=1e-8)
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(0, abs=1e-8)
    assert calculated["v_out_ripple_pp_v"] > 0


def test_loss_balanced_flyback_dcm_matches_independent_nonlinear_ramps() -> None:
    """Check nonlinear reflected ramps, secondary losses, and capacitor ripple."""
    transistor = _flat_dcm_device()
    switch_current = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    switch_voltage = np.array([0.8, 0.9, 1.1, 1.6, 2.5])
    diode_current = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    diode_voltage = np.array([1.5, 1.7, 2.0, 2.6, 3.5])
    transistor.switch.channel_data[0].graph_v_i = np.array(
        [switch_voltage, switch_current]
    )
    transistor.diode.channel_data[0].graph_v_i = np.array(
        [diode_voltage, diode_current]
    )
    values = point(
        mode="loss-balanced", topology="flyback", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=20, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15, turns_ratio_nps=2,
        parallel_switches=2, parallel_diodes=2,
    )
    n = values["turns_ratio_nps"]
    period = 1 / values["f_sw"]
    l_value = values["l_value"]
    target_charge = values["p_out"] / values["v_out"] * period
    result = calculate_topology_operating_point(
        "flyback", values, transistor, transistor
    )
    peak = result["calculated"]["i_l_peak_a"]

    current = np.linspace(0.0, peak, 100_001)
    switch_drop = np.interp(
        current / values["parallel_switches"], switch_current, switch_voltage
    )
    diode_drop = np.interp(
        n * current / values["parallel_diodes"], diode_current, diode_voltage
    )
    on_voltage = values["v_in"] - switch_drop
    off_voltage = n * (values["v_out"] + diode_drop)

    def integrate(samples: np.ndarray) -> float:
        return float(np.trapezoid(samples, current))

    def cumulative_time(voltage: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(
            0.5 * (1 / voltage[:-1] + 1 / voltage[1:])
            * np.diff(current) * l_value
        )))

    on_time_curve = cumulative_time(on_voltage)
    off_time_curve = cumulative_time(off_voltage)
    on_time = float(on_time_curve[-1])
    off_time = float(off_time_curve[-1])
    input_charge = l_value * integrate(current / on_voltage)
    output_charge = n * l_value * integrate(current / off_voltage)
    switch_energy = l_value * integrate(current * switch_drop / on_voltage)
    diode_energy = n * l_value * integrate(current * diode_drop / off_voltage)
    switch_voltage_time = l_value * integrate(switch_drop / on_voltage)
    diode_voltage_time = l_value * integrate(diode_drop / off_voltage)
    duty = on_time / period
    diode_fraction = off_time / period
    input_current = input_charge / period
    output_current = output_charge / period
    switch_loss = switch_energy / period
    diode_loss = diode_energy / period
    switch_mean = switch_voltage_time / on_time
    diode_mean = diode_voltage_time / off_time
    expected_power_residual = (
        values["v_in"] * input_current - values["p_out"] - switch_loss - diode_loss
    )
    expected_volt_second_residual = (
        duty * (values["v_in"] - switch_mean)
        - diode_fraction * n * (values["v_out"] + diode_mean)
    )

    # Independently integrate capacitor current over dense on, off, and idle
    # time samples. Primary current is reflected by n only during diode transfer.
    on_time_dense = on_time_curve
    off_time_dense = on_time + off_time - off_time_curve[::-1]
    time = np.concatenate((on_time_dense, off_time_dense[1:], [period]))
    primary_current = np.concatenate((current, current[-2::-1], [0.0]))
    capacitor_current = np.full(time.shape, -values["p_out"] / values["v_out"])
    diode_samples = (time >= on_time) & (time <= on_time + off_time)
    capacitor_current[diode_samples] += n * primary_current[diode_samples]
    charge = np.concatenate(([0.0], np.cumsum(
        0.5 * (capacitor_current[:-1] + capacitor_current[1:]) * np.diff(time)
    )))
    expected_ripple = float(np.ptp(charge) / values["c_out"])

    calculated = result["calculated"]
    assert calculated["duty_cycle"] == pytest.approx(duty, rel=2e-5)
    assert calculated["diode_conduction_fraction"] == pytest.approx(
        diode_fraction, rel=2e-5
    )
    assert calculated["i_in_avg_a"] == pytest.approx(input_current, rel=2e-5)
    assert calculated["i_diode_avg_a"] == pytest.approx(output_current, rel=2e-5)
    assert output_charge == pytest.approx(target_charge, rel=2e-8)
    assert result["losses"]["conduction_switch_w"] == pytest.approx(
        switch_loss, rel=2e-5
    )
    assert result["losses"]["conduction_diode_w"] == pytest.approx(
        diode_loss, rel=2e-5
    )
    assert result["performance"]["power_balance_residual_w"] == pytest.approx(
        expected_power_residual, abs=2e-5
    )
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(
        expected_volt_second_residual, abs=2e-5
    )
    assert calculated["v_out_ripple_pp_v"] == pytest.approx(
        expected_ripple, rel=2e-4
    )


@pytest.mark.parametrize(
    ("turns_ratio_nps", "expected_error"),
    [(None, "Flyback requires a positive turns_ratio_nps"),
     (0.0, "turns_ratio_nps must be a finite positive number")],
)
def test_loss_balanced_flyback_dcm_validates_turns_ratio(
    turns_ratio_nps: float | None, expected_error: str
) -> None:
    """Require a positive finite ratio before solving Flyback DCM feedback."""
    with pytest.raises(ValueError, match=expected_error):
        calculate_topology_operating_point(
            "flyback",
            point(
                mode="loss-balanced", conduction_regime="DCM",
                turns_ratio_nps=turns_ratio_nps,
            ),
            _flat_dcm_device(), _flat_dcm_device(),
        )


def test_loss_balanced_flyback_dcm_rejects_curve_range_and_condition_gaps() -> None:
    """Reject a missing exact condition or branch current beyond curve support."""
    values = point(
        mode="loss-balanced", topology="flyback", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=20, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15, turns_ratio_nps=2,
    )
    transistor = _flat_dcm_device(max_current=0.2)
    with pytest.raises(ValueError, match="exact channel-curve range"):
        calculate_topology_operating_point(
            "flyback", values, transistor, transistor
        )
    transistor = _flat_dcm_device()
    values["v_g"] = 14
    with pytest.raises(ValueError, match="Missing required switch channel curve"):
        calculate_topology_operating_point(
            "flyback", values, transistor, transistor
        )
    values["v_g"] = 15
    values["diode_v_g"] = 0
    with pytest.raises(ValueError, match="Missing required diode channel curve"):
        calculate_topology_operating_point(
            "flyback", values, transistor, transistor
        )


def test_loss_balanced_flyback_dcm_rejects_exact_ccm_boundary() -> None:
    """Reject the power derived independently from full-period ramp time."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="flyback", conduction_regime="DCM",
        v_in=100, v_out=50, l_value=1e-3, f_sw=20_000, c_out=1e-3,
        t_j=25, v_g=15, turns_ratio_nps=2,
    )
    on_voltage = values["v_in"] - 1.0
    off_voltage = values["turns_ratio_nps"] * (values["v_out"] + 2.0)
    period = 1 / values["f_sw"]
    boundary_peak = period / (
        values["l_value"] * (1 / on_voltage + 1 / off_voltage)
    )
    diode_fraction = values["l_value"] * boundary_peak / off_voltage / period
    boundary_power = (
        values["v_out"]
        * values["turns_ratio_nps"]
        * 0.5 * boundary_peak * diode_fraction
    )

    with pytest.raises(ValueError, match="at or above the CCM boundary"):
        calculate_topology_operating_point(
            "flyback", {**values, "p_out": boundary_power}, transistor, transistor
        )
    below = calculate_topology_operating_point(
        "flyback", {**values, "p_out": boundary_power * 0.99}, transistor, transistor
    )
    assert below["calculated"]["idle_fraction"] > 0


def test_loss_balanced_buck_boost_dcm_closes_flat_drop_fixture() -> None:
    """Check inverting Buck-Boost DCM from independent flat-drop equations."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="buck-boost", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    switch_drop = 1.0
    diode_drop = 2.0
    on_voltage = values["v_in"] - switch_drop
    off_voltage = values["v_out"] + diode_drop
    period = 1 / values["f_sw"]
    target_output_current = values["p_out"] / values["v_out"]
    peak = np.sqrt(
        2 * target_output_current * period * off_voltage / values["l_value"]
    )
    on_time = values["l_value"] * peak / on_voltage
    diode_time = values["l_value"] * peak / off_voltage
    duty = on_time / period
    diode_fraction = diode_time / period
    expected_input_current = 0.5 * peak * duty
    expected_output_current = 0.5 * peak * diode_fraction
    expected_inductor_average = 0.5 * peak * (duty + diode_fraction)
    expected_switch_loss = switch_drop * expected_input_current
    expected_diode_loss = diode_drop * expected_output_current
    expected_power_residual = (
        values["v_in"] * expected_input_current
        - values["p_out"]
        - expected_switch_loss
        - expected_diode_loss
    )
    expected_volt_second_residual = (
        duty * on_voltage - diode_fraction * off_voltage
    )

    result = calculate_topology_operating_point(
        "buck-boost", values, transistor, transistor
    )

    calculated = result["calculated"]
    assert result["model"]["output_polarity"] == -1
    assert calculated["v_out_signed_v"] == pytest.approx(-values["v_out"])
    assert calculated["duty_cycle"] == pytest.approx(duty, rel=1e-8)
    assert calculated["switch_conduction_fraction"] == pytest.approx(duty)
    assert calculated["diode_conduction_fraction"] == pytest.approx(
        diode_fraction, rel=1e-8
    )
    assert calculated["idle_fraction"] == pytest.approx(
        1 - duty - diode_fraction, rel=1e-8
    )
    assert calculated["idle_fraction"] > 0
    assert calculated["i_l_peak_a"] == pytest.approx(peak, rel=1e-8)
    assert calculated["i_l_avg_a"] == pytest.approx(expected_inductor_average, rel=1e-8)
    assert calculated["i_in_avg_a"] == pytest.approx(expected_input_current, rel=1e-8)
    assert calculated["i_out_avg_a"] == pytest.approx(target_output_current, rel=1e-8)
    assert calculated["i_diode_avg_a"] == pytest.approx(expected_output_current, rel=1e-8)
    assert calculated["i_switch_rms_a"] == pytest.approx(
        peak * np.sqrt(duty / 3), rel=1e-5
    )
    assert calculated["i_diode_rms_a"] == pytest.approx(
        peak * np.sqrt(diode_fraction / 3), rel=1e-5
    )
    assert result["losses"]["conduction_switch_w"] == pytest.approx(
        expected_switch_loss, rel=1e-8
    )
    assert result["losses"]["conduction_diode_w"] == pytest.approx(
        expected_diode_loss, rel=1e-8
    )
    assert calculated["power_balance_residual_w"] == pytest.approx(
        expected_power_residual, abs=1e-8
    )
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(
        expected_volt_second_residual, abs=1e-8
    )
    assert expected_power_residual == pytest.approx(0, abs=1e-10)
    assert expected_volt_second_residual == pytest.approx(0, abs=1e-10)
    assert result["performance"]["p_in_w"] == pytest.approx(
        values["v_in"] * expected_input_current, rel=1e-8
    )
    assert result["performance"]["efficiency_percent"] == pytest.approx(
        100 * values["p_out"] / (values["v_in"] * expected_input_current),
        rel=1e-8,
    )
    assert result["losses"]["loss_scope"] == "channel_conduction_only"


def test_loss_balanced_buck_boost_dcm_rejects_ccm_boundary() -> None:
    """Reject the flat-drop Buck-Boost point whose ramps fill the full period."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="buck-boost", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    on_voltage = values["v_in"] - 1.0
    off_voltage = values["v_out"] + 2.0
    period = 1 / values["f_sw"]
    boundary_peak = period / (
        values["l_value"] * (1 / on_voltage + 1 / off_voltage)
    )
    boundary_current = (
        values["l_value"] * boundary_peak**2
        / (2 * off_voltage * period)
    )
    boundary_values = {
        **values,
        "p_out": values["v_out"] * boundary_current,
    }

    with pytest.raises(ValueError, match="at or above the CCM boundary"):
        calculate_topology_operating_point(
            "buck-boost", boundary_values, transistor, transistor
        )

    below_boundary = {**boundary_values, "p_out": boundary_values["p_out"] * 0.95}
    result = calculate_topology_operating_point(
        "buck-boost", below_boundary, transistor, transistor
    )
    assert result["calculated"]["idle_fraction"] > 0


@pytest.mark.parametrize(
    ("curve_role", "condition_override"),
    [
        ("switch", {"v_g": 14}),
        ("diode", {"diode_v_g": 2}),
    ],
)
def test_loss_balanced_buck_boost_dcm_requires_exact_channel_condition(
    curve_role: str, condition_override: dict[str, float]
) -> None:
    """Reject a Buck-Boost switch or diode curve with the wrong gate condition."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="buck-boost", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    values.update(condition_override)

    with pytest.raises(ValueError, match=f"Missing required {curve_role} channel curve"):
        calculate_topology_operating_point(
            "buck-boost", values, transistor, transistor
        )


@pytest.mark.parametrize("curve_role", ["switch", "diode"])
def test_loss_balanced_buck_boost_dcm_requires_zero_current_coverage(
    curve_role: str,
) -> None:
    """Reject a selected Buck-Boost channel curve that starts above zero."""
    transistor = _flat_dcm_device()
    getattr(transistor, curve_role).channel_data[0].graph_v_i[1, 0] = 0.1
    values = point(
        mode="loss-balanced", topology="buck-boost", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )

    with pytest.raises(
        ValueError, match=rf"{curve_role} channel curve supports 0\.1 to .*requested 0 A"
    ):
        calculate_topology_operating_point(
            "buck-boost", values, transistor, transistor
        )


def test_loss_balanced_buck_boost_dcm_rejects_insufficient_peak_curve_range() -> None:
    """Do not extrapolate Buck-Boost curves beyond their measured peak range."""
    transistor = _flat_dcm_device(max_current=0.5)
    values = point(
        mode="loss-balanced", topology="buck-boost", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )

    with pytest.raises(
        ValueError,
        match="No supported Buck-Boost DCM solution inside the exact channel-curve range",
    ):
        calculate_topology_operating_point(
            "buck-boost", values, transistor, transistor
        )


def test_loss_balanced_buck_boost_dcm_matches_independent_nonlinear_ramps() -> None:
    """Compare nonlinear Buck-Boost DCM metrics against dense ramp integration."""
    transistor = _flat_dcm_device()
    curve_current = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    switch_voltage = np.array([1.0, 1.05, 1.2, 1.5, 2.0, 2.5, 3.0])
    diode_voltage = np.array([2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5])
    transistor.switch.channel_data[0].graph_v_i = np.array(
        [switch_voltage, curve_current]
    )
    transistor.diode.channel_data[0].graph_v_i = np.array(
        [diode_voltage, curve_current]
    )
    values = point(
        mode="loss-balanced", topology="buck-boost", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    original_switch_curve = transistor.switch.channel_data[0].graph_v_i.copy()
    original_diode_curve = transistor.diode.channel_data[0].graph_v_i.copy()
    period = 1 / values["f_sw"]
    inductance = values["l_value"]
    target_output_charge = values["p_out"] / values["v_out"] * period

    def independent_ramp(peak: float) -> dict[str, Any]:
        current = np.linspace(0.0, peak, 30_001)
        switch_drop = np.interp(current, curve_current, switch_voltage)
        diode_drop = np.interp(current, curve_current, diode_voltage)
        inverse_on = 1 / (values["v_in"] - switch_drop)
        inverse_off = 1 / (values["v_out"] + diode_drop)

        def integrate(samples: np.ndarray) -> float:
            trapezoids = 0.5 * (samples[:-1] + samples[1:]) * np.diff(current)
            return float(np.sum(trapezoids))

        on_time = inductance * integrate(inverse_on)
        diode_time = inductance * integrate(inverse_off)
        input_charge = inductance * integrate(current * inverse_on)
        output_charge = inductance * integrate(current * inverse_off)
        switch_energy = inductance * integrate(current * switch_drop * inverse_on)
        diode_energy = inductance * integrate(current * diode_drop * inverse_off)
        switch_voltage_time = inductance * integrate(switch_drop * inverse_on)
        diode_voltage_time = inductance * integrate(diode_drop * inverse_off)
        duty = on_time / period
        diode_fraction = diode_time / period
        input_current = input_charge / period
        output_current = output_charge / period
        switch_loss = switch_energy / period
        diode_loss = diode_energy / period
        on_time_samples = inductance * np.concatenate((
            [0.0], np.cumsum(
                0.5 * (inverse_on[:-1] + inverse_on[1:]) * np.diff(current)
            )
        ))
        off_time_samples = inductance * np.concatenate((
            [0.0], np.cumsum(
                0.5 * (inverse_off[:-1] + inverse_off[1:]) * np.diff(current)
            )
        ))
        falling_time = on_time + diode_time - off_time_samples[::-1]
        time = np.concatenate((on_time_samples, falling_time[1:], [period]))
        waveform = np.concatenate((current, current[-2::-1], [0.0]))
        target_output_current = values["p_out"] / values["v_out"]
        cumulative_charge = [0.0]
        charge = 0.0
        for index in range(len(time) - 1):
            midpoint = (time[index] + time[index + 1]) / 2
            if midpoint < on_time or midpoint >= on_time + diode_time:
                current_start = current_end = -target_output_current
            else:
                current_start = waveform[index] - target_output_current
                current_end = waveform[index + 1] - target_output_current
            charge += 0.5 * (current_start + current_end) * (
                time[index + 1] - time[index]
            )
            cumulative_charge.append(charge)
        switch_mean = switch_voltage_time / on_time
        diode_mean = diode_voltage_time / diode_time
        return {
            "peak": peak,
            "duty": duty,
            "diode_fraction": diode_fraction,
            "idle_fraction": 1 - duty - diode_fraction,
            "input_current": input_current,
            "output_current": output_current,
            "inductor_current": (input_charge + output_charge) / period,
            "switch_loss": switch_loss,
            "diode_loss": diode_loss,
            "switch_rms": np.sqrt(
                inductance * integrate(current**2 * inverse_on) / period
            ),
            "diode_rms": np.sqrt(
                inductance * integrate(current**2 * inverse_off) / period
            ),
            "power_residual": (
                values["v_in"] * input_current - values["p_out"]
                - switch_loss - diode_loss
            ),
            "volt_second_residual": (
                duty * (values["v_in"] - switch_mean)
                + diode_fraction * (-values["v_out"] - diode_mean)
            ),
            "ripple": np.ptp(cumulative_charge) / values["c_out"],
        }

    low, high = 0.0, float(curve_current[-1])
    for _ in range(70):
        midpoint = (low + high) / 2
        state = independent_ramp(midpoint)
        if state["output_current"] * period < target_output_charge:
            low = midpoint
        else:
            high = midpoint
    expected = independent_ramp((low + high) / 2)

    result = calculate_topology_operating_point(
        "buck-boost", values, transistor, transistor
    )
    calculated = result["calculated"]
    assert calculated["i_l_peak_a"] == pytest.approx(expected["peak"], rel=2e-5)
    assert calculated["duty_cycle"] == pytest.approx(expected["duty"], rel=2e-5)
    assert calculated["diode_conduction_fraction"] == pytest.approx(
        expected["diode_fraction"], rel=2e-5
    )
    assert calculated["idle_fraction"] == pytest.approx(
        expected["idle_fraction"], abs=2e-5
    )
    assert calculated["i_in_avg_a"] == pytest.approx(
        expected["input_current"], rel=2e-5
    )
    assert calculated["i_out_avg_a"] == pytest.approx(
        expected["output_current"], rel=2e-5
    )
    assert calculated["i_l_avg_a"] == pytest.approx(
        expected["inductor_current"], rel=2e-5
    )
    assert calculated["i_switch_rms_a"] == pytest.approx(
        expected["switch_rms"], rel=2e-5
    )
    assert calculated["i_diode_rms_a"] == pytest.approx(
        expected["diode_rms"], rel=2e-5
    )
    assert result["losses"]["conduction_switch_w"] == pytest.approx(
        expected["switch_loss"], rel=2e-5
    )
    assert result["losses"]["conduction_diode_w"] == pytest.approx(
        expected["diode_loss"], rel=2e-5
    )
    assert calculated["power_balance_residual_w"] == pytest.approx(
        expected["power_residual"], abs=2e-6
    )
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(
        expected["volt_second_residual"], abs=2e-6
    )
    assert calculated["v_out_ripple_pp_v"] == pytest.approx(
        expected["ripple"], rel=3e-3
    )
    assert np.array_equal(
        transistor.switch.channel_data[0].graph_v_i, original_switch_curve
    )
    assert np.array_equal(
        transistor.diode.channel_data[0].graph_v_i, original_diode_curve
    )


def _flat_dcm_device(
    name: str = "flat DCM device", *, max_current: float = 100.0
) -> Transistor:
    transistor = device(name, max_current=max_current)
    transistor.switch.channel_data[0].graph_v_i = np.array(
        [[1.0, 1.0], [0.0, max_current]]
    )
    transistor.diode.channel_data[0].graph_v_i = np.array(
        [[2.0, 2.0], [0.0, max_current]]
    )
    return transistor


def test_loss_balanced_buck_dcm_closes_independent_flat_drop_fixture() -> None:
    """Check Buck currents and losses from independently derived flat drops."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="buck", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    switch_drop = 1.0
    diode_drop = 2.0
    on_voltage = values["v_in"] - values["v_out"] - switch_drop
    off_voltage = values["v_out"] + diode_drop
    period = 1 / values["f_sw"]
    target_output_current = values["p_out"] / values["v_out"]
    peak = np.sqrt(
        2 * target_output_current * period
        / (values["l_value"] * (1 / on_voltage + 1 / off_voltage))
    )
    on_time = values["l_value"] * peak / on_voltage
    diode_time = values["l_value"] * peak / off_voltage
    expected_duty = on_time / period
    expected_diode_fraction = diode_time / period
    expected_input_current = 0.5 * peak * expected_duty
    expected_diode_current = 0.5 * peak * expected_diode_fraction
    expected_switch_rms = peak * np.sqrt(expected_duty / 3)
    expected_diode_rms = peak * np.sqrt(expected_diode_fraction / 3)
    expected_switch_loss = switch_drop * expected_input_current
    expected_diode_loss = diode_drop * expected_diode_current
    independent_power_residual = (
        values["v_in"] * expected_input_current
        - values["p_out"]
        - expected_switch_loss
        - expected_diode_loss
    )
    independent_volt_second_residual = (
        expected_duty * on_voltage - expected_diode_fraction * off_voltage
    )

    result = calculate_topology_operating_point(
        "buck", values, transistor, transistor
    )

    calculated = result["calculated"]
    assert calculated["duty_cycle"] == pytest.approx(expected_duty, rel=1e-8)
    assert calculated["switch_conduction_fraction"] == pytest.approx(
        expected_duty, rel=1e-8
    )
    assert calculated["diode_conduction_fraction"] == pytest.approx(
        expected_diode_fraction, rel=1e-8
    )
    assert calculated["idle_fraction"] == pytest.approx(
        1 - expected_duty - expected_diode_fraction, rel=1e-8
    )
    assert calculated["i_l_peak_a"] == pytest.approx(peak, rel=1e-8)
    assert calculated["i_l_avg_a"] == pytest.approx(target_output_current, rel=1e-8)
    assert calculated["i_in_avg_a"] == pytest.approx(expected_input_current, rel=1e-8)
    assert calculated["i_out_avg_a"] == pytest.approx(target_output_current, rel=1e-8)
    assert calculated["i_diode_avg_a"] == pytest.approx(expected_diode_current, rel=1e-8)
    assert calculated["i_switch_rms_a"] == pytest.approx(expected_switch_rms, rel=5e-6)
    assert calculated["i_diode_rms_a"] == pytest.approx(expected_diode_rms, rel=5e-6)
    assert result["losses"]["conduction_switch_w"] == pytest.approx(
        expected_switch_loss, rel=1e-8
    )
    assert result["losses"]["conduction_diode_w"] == pytest.approx(
        expected_diode_loss, rel=1e-8
    )
    assert calculated["power_balance_residual_w"] == pytest.approx(
        independent_power_residual, abs=1e-8
    )
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(
        independent_volt_second_residual, abs=1e-8
    )
    assert independent_power_residual == pytest.approx(0, abs=1e-10)
    assert independent_volt_second_residual == pytest.approx(0, abs=1e-10)
    assert result["losses"]["loss_scope"] == "channel_conduction_only"


def test_loss_balanced_buck_dcm_matches_independent_nonlinear_ramp_integrals() -> None:
    """Check Buck DCM currents, losses, balances, and ripple on rising curves."""
    transistor = _flat_dcm_device()
    curve_currents = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    switch_drops = np.array([1.0, 1.05, 1.2, 1.5, 2.0, 2.5, 3.0])
    diode_drops = np.array([2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5])
    transistor.switch.channel_data[0].graph_v_i = np.array(
        [switch_drops, curve_currents]
    )
    transistor.diode.channel_data[0].graph_v_i = np.array(
        [diode_drops, curve_currents]
    )
    values = point(
        mode="loss-balanced", topology="buck", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    period = 1 / values["f_sw"]
    target_output_current = values["p_out"] / values["v_out"]
    target_output_charge = target_output_current * period
    inductance = values["l_value"]

    def independent_ramp(peak: float) -> dict[str, Any]:
        current = np.linspace(0.0, peak, 20_001)
        switch_drop = np.interp(current, curve_currents, switch_drops)
        diode_drop = np.interp(current, curve_currents, diode_drops)
        inverse_on_voltage = 1 / (
            values["v_in"] - values["v_out"] - switch_drop
        )
        inverse_off_voltage = 1 / (values["v_out"] + diode_drop)

        def integral(integrand: np.ndarray) -> float:
            return float(
                np.sum(
                    0.5 * (integrand[:-1] + integrand[1:]) * np.diff(current)
                )
            )

        on_time = inductance * integral(inverse_on_voltage)
        diode_time = inductance * integral(inverse_off_voltage)
        charge_on = inductance * integral(current * inverse_on_voltage)
        charge_diode = inductance * integral(current * inverse_off_voltage)
        switch_energy = inductance * integral(
            current * switch_drop * inverse_on_voltage
        )
        diode_energy = inductance * integral(
            current * diode_drop * inverse_off_voltage
        )
        switch_voltage_time = inductance * integral(
            switch_drop * inverse_on_voltage
        )
        diode_voltage_time = inductance * integral(
            diode_drop * inverse_off_voltage
        )
        charge_current_out = (charge_on + charge_diode) / period
        duty = on_time / period
        diode_fraction = diode_time / period
        input_current = charge_on / period
        diode_current = charge_diode / period
        switch_loss = switch_energy / period
        diode_loss = diode_energy / period
        switch_mean = switch_voltage_time / on_time
        diode_mean = diode_voltage_time / diode_time
        volt_second_residual = (
            duty * (values["v_in"] - values["v_out"] - switch_mean)
            + diode_fraction * (-values["v_out"] - diode_mean)
        )
        return {
            "current": current,
            "on_time": on_time,
            "diode_time": diode_time,
            "charge_on": charge_on,
            "charge_diode": charge_diode,
            "output_current": charge_current_out,
            "input_current": input_current,
            "diode_current": diode_current,
            "switch_loss": switch_loss,
            "diode_loss": diode_loss,
            "duty": duty,
            "diode_fraction": diode_fraction,
            "idle_fraction": 1 - duty - diode_fraction,
            "peak": peak,
            "power_residual": (
                values["v_in"] * input_current
                - values["p_out"]
                - switch_loss
                - diode_loss
            ),
            "volt_second_residual": volt_second_residual,
            "on_time_by_current": inductance * np.concatenate(
                ([0.0], np.cumsum(
                    0.5
                    * (inverse_on_voltage[:-1] + inverse_on_voltage[1:])
                    * np.diff(current)
                ))
            ),
            "diode_time_by_current": inductance * np.concatenate(
                ([0.0], np.cumsum(
                    0.5
                    * (inverse_off_voltage[:-1] + inverse_off_voltage[1:])
                    * np.diff(current)
                ))
            ),
        }

    low, high = 0.0, 3.0
    for _ in range(60):
        midpoint = (low + high) / 2
        ramp = independent_ramp(midpoint)
        if ramp["charge_on"] + ramp["charge_diode"] < target_output_charge:
            low = midpoint
        else:
            high = midpoint
    expected = independent_ramp((low + high) / 2)
    independent_current = expected["current"]
    on_time_by_current = expected["on_time_by_current"]
    diode_time_by_current = expected["diode_time_by_current"]
    descending_current = independent_current[::-1]
    descending_diode_time = (
        expected["diode_time"] - diode_time_by_current[::-1]
    )
    phase_time = np.concatenate(
        (
            on_time_by_current,
            expected["on_time"] + descending_diode_time[1:],
            np.array([period]),
        )
    )
    ramp_current = np.concatenate(
        (independent_current, descending_current[1:], np.array([0.0]))
    )
    capacitor_current = np.concatenate(
        (ramp_current[:-1] - target_output_current, np.array([-target_output_current]))
    )
    charge_steps = (
        0.5 * (capacitor_current[:-1] + capacitor_current[1:]) * np.diff(phase_time)
    )
    charge = np.concatenate(([0.0], np.cumsum(charge_steps)))
    independent_ripple = float(np.ptp(charge) / values["c_out"])

    result = calculate_topology_operating_point(
        "buck", values, transistor, transistor
    )

    calculated = result["calculated"]
    assert calculated["i_l_peak_a"] == pytest.approx(expected["peak"], rel=3e-5)
    assert calculated["duty_cycle"] == pytest.approx(expected["duty"], rel=3e-5)
    assert calculated["diode_conduction_fraction"] == pytest.approx(
        expected["diode_fraction"], rel=3e-5
    )
    assert calculated["idle_fraction"] == pytest.approx(
        expected["idle_fraction"], abs=3e-5
    )
    assert calculated["i_in_avg_a"] == pytest.approx(
        expected["input_current"], rel=3e-5
    )
    assert calculated["i_out_avg_a"] == pytest.approx(
        expected["output_current"], rel=3e-5
    )
    assert calculated["i_l_avg_a"] == pytest.approx(
        expected["output_current"], rel=3e-5
    )
    assert calculated["i_diode_avg_a"] == pytest.approx(
        expected["diode_current"], rel=3e-5
    )
    assert result["losses"]["conduction_switch_w"] == pytest.approx(
        expected["switch_loss"], rel=3e-5
    )
    assert result["losses"]["conduction_diode_w"] == pytest.approx(
        expected["diode_loss"], rel=3e-5
    )
    assert calculated["power_balance_residual_w"] == pytest.approx(
        expected["power_residual"], abs=1e-7
    )
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(
        expected["volt_second_residual"], abs=1e-7
    )
    assert calculated["v_out_ripple_pp_v"] == pytest.approx(
        independent_ripple, rel=3e-4
    )


def test_loss_balanced_buck_dcm_rejects_exact_boundary_but_accepts_below() -> None:
    """Use flat-drop voltages to derive Buck's exact CCM boundary current."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="buck", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    l_f = values["l_value"] * values["f_sw"]
    on_voltage = values["v_in"] - values["v_out"] - 1.0
    off_voltage = values["v_out"] + 2.0
    boundary_peak = 1 / (l_f * (1 / on_voltage + 1 / off_voltage))
    boundary_power = values["v_out"] * boundary_peak / 2
    boundary_values = {**values, "p_out": boundary_power}

    with pytest.raises(ValueError, match="at or above the CCM boundary"):
        calculate_topology_operating_point(
            "buck", boundary_values, transistor, transistor
        )

    below_boundary = {**values, "p_out": boundary_power * 0.95}
    result = calculate_topology_operating_point(
        "buck", below_boundary, transistor, transistor
    )
    assert result["calculated"]["idle_fraction"] > 0
    assert result["calculated"]["power_balance_residual_w"] == pytest.approx(
        0, abs=1e-7
    )
    assert result["calculated"]["inductor_volt_second_residual_v"] == pytest.approx(
        0, abs=1e-7
    )


def test_loss_balanced_boost_dcm_closes_independent_power_and_volt_second_fixture() -> None:
    """Check both balances from independent flat-drop Boost DCM equations."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="boost", conduction_regime="DCM",
        v_in=100, v_out=200, p_out=100, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    original_switch = transistor.switch.channel_data[0].graph_v_i.copy()
    original_diode = transistor.diode.channel_data[0].graph_v_i.copy()

    result = calculate_topology_operating_point(
        "boost", values, transistor, transistor
    )

    calculated = result["calculated"]
    current_out = values["p_out"] / values["v_out"]
    off_voltage = values["v_out"] - values["v_in"] + 2.0
    peak = np.sqrt(
        2 * current_out * off_voltage / (values["l_value"] * values["f_sw"])
    )
    expected_duty = (
        values["l_value"] * values["f_sw"] * peak / (values["v_in"] - 1.0)
    )
    expected_diode_fraction = (
        values["l_value"] * values["f_sw"] * peak / off_voltage
    )
    assert calculated["i_l_peak_a"] == pytest.approx(peak, rel=1e-8)
    assert calculated["duty_cycle"] == pytest.approx(expected_duty, rel=1e-8)
    assert calculated["switch_conduction_fraction"] == pytest.approx(expected_duty)
    assert calculated["diode_conduction_fraction"] == pytest.approx(
        expected_diode_fraction, rel=1e-8
    )
    assert calculated["idle_fraction"] == pytest.approx(
        1 - expected_duty - expected_diode_fraction, rel=1e-8
    )
    assert calculated["idle_fraction"] > 0
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(0, abs=1e-9)
    assert calculated["power_balance_residual_w"] == pytest.approx(0, abs=1e-8)
    assert result["performance"]["p_in_w"] == pytest.approx(
        values["p_out"] + result["losses"]["total_w"], rel=1e-9
    )
    assert calculated["i_out_avg_a"] == pytest.approx(current_out)
    assert result["model"]["conduction_regime"] == "DCM"
    assert result["losses"]["loss_scope"] == "channel_conduction_only"
    assert np.array_equal(transistor.switch.channel_data[0].graph_v_i, original_switch)
    assert np.array_equal(transistor.diode.channel_data[0].graph_v_i, original_diode)
    assert min(result["waveforms"]["inductor_current_a"]) == pytest.approx(0)
    phase = np.asarray(result["waveforms"]["time_s"]) * values["f_sw"]
    waveform_current = np.asarray(result["waveforms"]["inductor_current_a"])
    switch_event = np.flatnonzero(np.isclose(phase, expected_duty))[-1]
    diode_end_event = np.flatnonzero(
        np.isclose(phase, expected_duty + expected_diode_fraction)
    )[-1]
    assert waveform_current[switch_event] == pytest.approx(peak)
    assert waveform_current[diode_end_event] == pytest.approx(0)


def test_loss_balanced_boost_dcm_ripple_uses_solved_nonlinear_diode_ramp() -> None:
    """Compare ripple with an independent dense integration of nonlinear ramps."""
    transistor = _flat_dcm_device()
    currents = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    transistor.switch.channel_data[0].graph_v_i = np.array(
        [[1.0, 1.05, 1.2, 1.5, 2.0, 2.5, 3.0], currents]
    )
    transistor.diode.channel_data[0].graph_v_i = np.array(
        [[2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5], currents]
    )
    values = point(
        mode="loss-balanced", topology="boost", conduction_regime="DCM",
        v_in=100, v_out=200, p_out=100, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )

    result = calculate_topology_operating_point(
        "boost", values, transistor, transistor
    )

    peak = result["calculated"]["i_l_peak_a"]
    dense_current = np.linspace(0.0, peak, 100_001)
    switch_drop = np.interp(
        dense_current, currents, transistor.switch.channel_data[0].graph_v_i[0]
    )
    diode_drop = np.interp(
        dense_current, currents, transistor.diode.channel_data[0].graph_v_i[0]
    )
    inverse_on_voltage = 1.0 / (values["v_in"] - switch_drop)
    inverse_off_voltage = 1.0 / (
        values["v_out"] - values["v_in"] + diode_drop
    )

    def cumulative_trapezoid(integrand: np.ndarray) -> np.ndarray:
        increments = (
            0.5 * (integrand[:-1] + integrand[1:])
            * np.diff(dense_current)
            * values["l_value"]
        )
        return np.concatenate(([0.0], np.cumsum(increments))
)

    on_time = cumulative_trapezoid(inverse_on_voltage)
    diode_time = cumulative_trapezoid(inverse_off_voltage)
    diode_duration = float(diode_time[-1])
    current_out = values["p_out"] / values["v_out"]
    charge = -current_out * float(on_time[-1])
    extrema = [0.0, charge]
    descending_current = dense_current[::-1]
    descending_time = diode_duration - diode_time[::-1]
    capacitor_current = descending_current - current_out
    diode_charge = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5
                * (capacitor_current[:-1] + capacitor_current[1:])
                * np.diff(descending_time)
            ),
        )
    )
    extrema.extend((charge + diode_charge).tolist())
    cycle_time = 1.0 / values["f_sw"]
    idle_charge = -current_out * (cycle_time - on_time[-1] - diode_duration)
    extrema.append(charge + float(diode_charge[-1]) + idle_charge)
    independent_ripple = (max(extrema) - min(extrema)) / values["c_out"]

    calculated = result["calculated"]
    assert calculated["v_out_ripple_pp_v"] == pytest.approx(
        independent_ripple, rel=2e-5
    )
    triangular_estimate = _dcm_output_ripple(
        "boost",
        calculated["duty_cycle"],
        calculated["diode_conduction_fraction"],
        peak,
        current_out,
        values["f_sw"],
        values["c_out"],
    )
    assert abs(calculated["v_out_ripple_pp_v"] - triangular_estimate) > 1e-5


def test_loss_balanced_boost_dcm_brackets_before_switch_drop_cutoff() -> None:
    """Find a valid root below the current where the switch drop reaches Vin."""
    transistor = _flat_dcm_device(max_current=10.0)
    currents = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    transistor.switch.channel_data[0].graph_v_i = np.array(
        [[1.0, 1.1, 1.2, 1.5, 3.0, 120.0], currents]
    )
    transistor.diode.channel_data[0].graph_v_i = np.array(
        [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], currents]
    )

    result = calculate_topology_operating_point(
        "boost",
        point(
            mode="loss-balanced", conduction_regime="DCM", v_in=100,
            v_out=200, p_out=100, l_value=1e-3, f_sw=20_000,
            t_j=25, v_g=15,
        ),
        transistor,
        transistor,
    )

    calculated = result["calculated"]
    assert 2.0 < calculated["i_l_peak_a"] < 3.0
    assert calculated["power_balance_residual_w"] == pytest.approx(0, abs=1e-7)
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(0, abs=1e-7)


def test_loss_balanced_boost_dcm_rejects_ccm_boundary_and_curve_gaps() -> None:
    """Fail clearly for a non-DCM target or channel curves without zero current."""
    transistor = _flat_dcm_device()
    with pytest.raises(ValueError, match="at or above the CCM boundary"):
        calculate_topology_operating_point(
            "boost",
            point(
                mode="loss-balanced", conduction_regime="DCM", v_in=100,
                v_out=200, p_out=150, l_value=1e-3, f_sw=20_000,
                t_j=25, v_g=15,
            ),
            transistor,
            transistor,
        )

    unsupported = device()
    unsupported.switch.channel_data[0].graph_v_i[1, 0] = 0.1
    unsupported.diode.channel_data[0].graph_v_i[1, 0] = 0.1
    with pytest.raises(ValueError, match="curve supports 0.1 to"):
        calculate_topology_operating_point(
            "boost",
            point(
                mode="loss-balanced", conduction_regime="DCM", v_in=100,
                v_out=200, p_out=100, l_value=1e-3, f_sw=20_000,
                t_j=25, v_g=15,
            ),
            unsupported,
            unsupported,
        )

    limited = _flat_dcm_device(max_current=2.0)
    with pytest.raises(ValueError, match="No supported Boost DCM solution inside"):
        calculate_topology_operating_point(
            "boost",
            point(
                mode="loss-balanced", conduction_regime="DCM", v_in=100,
                v_out=200, p_out=100, l_value=1e-3, f_sw=20_000,
                t_j=25, v_g=15,
            ),
            limited,
            limited,
        )


def test_loss_balanced_boost_dcm_rejects_exact_boundary_but_accepts_below() -> None:
    """Use an analytic flat-drop boundary and check a nearby feasible point."""
    transistor = _flat_dcm_device()
    values = point(
        mode="loss-balanced", topology="boost", conduction_regime="DCM",
        v_in=100, v_out=200, p_out=100, l_value=1e-3, f_sw=20_000,
        c_out=1e-3, t_j=25, v_g=15,
    )
    l_f = values["l_value"] * values["f_sw"]
    on_voltage = values["v_in"] - 1.0
    off_voltage = values["v_out"] - values["v_in"] + 2.0
    boundary_peak = 1.0 / (l_f * (1.0 / on_voltage + 1.0 / off_voltage))
    boundary_output_current = (
        l_f * boundary_peak**2 / (2.0 * off_voltage)
    )
    boundary_power = values["v_out"] * boundary_output_current

    with pytest.raises(ValueError, match="at or above the CCM boundary"):
        calculate_topology_operating_point(
            "boost", {**values, "p_out": boundary_power}, transistor, transistor
        )

    near_boundary = calculate_topology_operating_point(
        "boost",
        {**values, "p_out": boundary_power * (1.0 - 1e-6)},
        transistor,
        transistor,
    )
    assert near_boundary["calculated"]["idle_fraction"] > 0


@pytest.mark.parametrize(
    ("topology", "parameters", "expected_result"),
    [
        (
            "buck",
            point(
                mode="loss-balanced", v_in=100, v_out=50, p_out=500,
                l_value=1e-3, f_sw=20_000,
            ),
            {
                "duty_cycle": 52 / 101,
                "i_l_avg_a": 10,
                "i_in_avg_a": 520 / 101,
                "i_out_avg_a": 10,
                "i_diode_avg_a": 490 / 101,
                "conduction_switch_w": 520 / 101,
                "conduction_diode_w": 980 / 101,
                "v_out_signed_v": 50,
            },
        ),
        (
            "buck-boost",
            point(
                mode="loss-balanced", v_in=100, v_out=50, p_out=500,
                l_value=1e-3, f_sw=20_000,
            ),
            {
                "duty_cycle": 52 / 151,
                "i_l_avg_a": 1510 / 99,
                "i_in_avg_a": 520 / 99,
                "i_out_avg_a": 10,
                "i_diode_avg_a": 10,
                "conduction_switch_w": 520 / 99,
                "conduction_diode_w": 20,
                "v_out_signed_v": -50,
            },
        ),
        (
            "flyback",
            point(
                mode="loss-balanced", v_in=100, v_out=50, p_out=500,
                turns_ratio_nps=2, l_value=1e-3, f_sw=20_000,
            ),
            {
                "duty_cycle": 104 / 203,
                "i_l_avg_a": 1015 / 99,
                "i_in_avg_a": 520 / 99,
                "i_out_avg_a": 10,
                "i_diode_avg_a": 10,
                "conduction_switch_w": 520 / 99,
                "conduction_diode_w": 20,
                "v_out_signed_v": 50,
            },
        ),
    ],
)
def test_loss_balanced_ccm_topologies_match_independent_constant_drop_balance(
    topology: str,
    parameters: dict[str, Any],
    expected_result: dict[str, float],
) -> None:
    """Check independent flat-drop currents, losses, and balance equations."""
    # One device in each leg, exact 25 C / 15 V switch data, and flat 1 V / 2 V
    # switch/diode drops make the expected CCM currents and interval losses
    # explicit. The diode is ungated (v_g=None); Flyback uses the specified Np/Ns.
    switch_drop = 1.0
    diode_drop = 2.0
    v_in = parameters["v_in"]
    v_out = parameters["v_out"]
    p_out = parameters["p_out"]
    turns_ratio_nps = parameters.get("turns_ratio_nps", 1.0)

    # Before calling the solver, hand-derived currents are I_l=Pout/Vout for
    # Buck, Pout/(Vout*(1-D)) for Buck-Boost, and
    # Pout/(Vout*(Np/Ns)*(1-D)) for Flyback. Their table values also set
    # Iin=D*I_l and the diode's full-period average with each topology's scale.
    # Derive interval losses and signed volt-seconds from those expected values.
    expected_duty = expected_result["duty_cycle"]
    expected_i_l = expected_result["i_l_avg_a"]
    expected_i_in = expected_result["i_in_avg_a"]
    expected_switch_loss = expected_result["conduction_switch_w"]
    expected_diode_loss = expected_result["conduction_diode_w"]
    if topology == "buck":
        expected_switch_loss_from_equation = (
            expected_duty * switch_drop * expected_i_l
        )
        expected_diode_loss_from_equation = (
            (1 - expected_duty) * diode_drop * expected_i_l
        )
        expected_volt_second_residual = (
            expected_duty * (v_in - switch_drop - v_out)
            + (1 - expected_duty) * (-v_out - diode_drop)
        )
    elif topology == "buck-boost":
        expected_switch_loss_from_equation = (
            expected_duty * switch_drop * expected_i_l
        )
        expected_diode_loss_from_equation = (
            (1 - expected_duty) * diode_drop * expected_i_l
        )
        expected_volt_second_residual = (
            expected_duty * (v_in - switch_drop)
            + (1 - expected_duty) * (-v_out - diode_drop)
        )
    else:
        expected_switch_loss_from_equation = (
            expected_duty * switch_drop * expected_i_l
        )
        expected_diode_loss_from_equation = (
            (1 - expected_duty)
            * turns_ratio_nps
            * diode_drop
            * expected_i_l
        )
        expected_volt_second_residual = (
            expected_duty * (v_in - switch_drop)
            + (1 - expected_duty)
            * (-turns_ratio_nps * (v_out + diode_drop))
        )

    expected_power_balance_residual = (
        v_in * expected_i_in
        - p_out
        - expected_switch_loss
        - expected_diode_loss
    )
    assert expected_switch_loss_from_equation == pytest.approx(expected_switch_loss)
    assert expected_diode_loss_from_equation == pytest.approx(expected_diode_loss)
    assert expected_power_balance_residual == pytest.approx(0, abs=1e-10)
    assert expected_volt_second_residual == pytest.approx(0, abs=1e-12)
    assert expected_result["i_out_avg_a"] == pytest.approx(p_out / v_out, rel=1e-8)

    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = switch_drop
    transistor.diode.channel_data[0].graph_v_i[0] = diode_drop
    result = calculate_topology_operating_point(
        topology,
        {**parameters, "t_j": 25, "v_g": 15, "diode_v_g": None},
        transistor,
        transistor,
    )

    calculated = result["calculated"]
    losses = result["losses"]
    performance = result["performance"]
    assert calculated["duty_cycle"] == pytest.approx(expected_duty, rel=1e-8)
    assert calculated["i_l_avg_a"] == pytest.approx(expected_i_l, rel=1e-8)
    assert calculated["i_in_avg_a"] == pytest.approx(expected_i_in, rel=1e-8)
    assert calculated["i_out_avg_a"] == pytest.approx(p_out / v_out, rel=1e-8)
    assert calculated["i_out_avg_a"] == pytest.approx(
        expected_result["i_out_avg_a"], rel=1e-8
    )
    assert calculated["i_diode_avg_a"] == pytest.approx(
        expected_result["i_diode_avg_a"], rel=1e-8
    )
    assert calculated["v_out_signed_v"] == pytest.approx(
        expected_result["v_out_signed_v"]
    )
    assert losses["conduction_switch_w"] == pytest.approx(
        expected_switch_loss, rel=1e-8
    )
    assert losses["conduction_diode_w"] == pytest.approx(
        expected_diode_loss, rel=1e-8
    )
    assert performance["p_in_w"] == pytest.approx(v_in * expected_i_in, rel=1e-8)
    assert performance["p_in_w"] == pytest.approx(
        p_out + expected_switch_loss + expected_diode_loss, rel=1e-8
    )
    assert performance["power_balance_residual_w"] == pytest.approx(
        expected_power_balance_residual, abs=1e-6
    )
    assert calculated["inductor_volt_second_residual_v"] == pytest.approx(
        expected_volt_second_residual, abs=1e-6
    )


def test_loss_balanced_boost_rejects_channel_curve_extrapolation() -> None:
    """Reject a feedback point when its converged current exceeds TDB data."""
    limited = device(max_current=8.0)
    with pytest.raises(ValueError, match="switch channel curve supports"):
        calculate_topology_operating_point(
            "boost",
            point(
                mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
                l_value=1e-3, f_sw=20_000, t_j=25, v_g=15,
            ),
            limited,
            limited,
        )


@pytest.mark.parametrize(
    ("switch_name", "switch_gate", "parallel_switches"),
    [
        ("CREE_C3M0060065J", 15.0, 1),
        ("Infineon_IPBE65R050CFD7A", 10.0, 1),
        ("Infineon_IPBE65R050CFD7A", 10.0, 2),
    ],
)
def test_loss_balanced_mode_runs_checked_in_same_boundary_pareto_records(
    switch_name: str, switch_gate: float, parallel_switches: int
) -> None:
    """Exercise the notebook's actual candidate curves at the shared boundary."""
    records = Path(__file__).parents[1] / "transistordatabase" / "examples" / "tdb_example"
    loader = JsonTransistorLoader()
    switch = loader.load_from_json(records / f"{switch_name}.json")
    diode = loader.load_from_json(records / "CREE_C3M0060065J.json")
    result = calculate_topology_operating_point(
        "boost",
        point(
            mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
            f_sw=20_000, l_value=1e-3, c_out=1e-3,
            t_j=25, t_case=60, v_g=switch_gate, diode_v_g=0,
            parallel_switches=parallel_switches,
        ),
        switch,
        diode,
        include_waveforms=False,
    )
    assert result["performance"]["p_in_w"] == pytest.approx(
        1000 + result["losses"]["total_w"], rel=1e-7
    )
    assert result["calculated"]["duty_cycle"] > 0.5
    assert result["model"]["thermal_closure"] == "unavailable"


def test_loss_balanced_thermal_output_does_not_claim_temperature_closure() -> None:
    """Keep fixed-temperature curve results separate from thermal closure."""
    transistor = device()
    transistor.switch.thermal_foster = FosterThermalModel(r_th_total=1.0)
    transistor.diode.thermal_foster = FosterThermalModel(r_th_total=1.0)
    transistor.thermal_properties.r_th_switch_cs = 0.2
    transistor.thermal_properties.r_th_diode_cs = 0.2
    result = calculate_topology_operating_point(
        "boost",
        point(
            mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
            l_value=1e-3, f_sw=20_000, t_j=25, t_case=25, v_g=15,
        ),
        transistor,
        transistor,
    )
    assert result["temperatures"]["switch_c"] > 25
    assert result["temperatures"]["diode_c"] > 25
    assert result["model"]["thermal_closure"] == "not_closed"
    assert "differs from the selected exact curve temperature" in result["temperatures"]["thermal_closure_reason"]


def test_gate_conditioned_diode_curves_require_explicit_diode_gate_voltage() -> None:
    """Explain missing diode gate selection and preserve selected curve provenance."""
    transistor = device()
    transistor.diode.channel_data.clear()
    graph = np.array([[0.4, 5.4], [0.0, 100.0]])
    for gate_voltage in (0.0, -2.0, -4.0):
        transistor.diode.channel_data.append(
            ChannelCharacteristics(25, graph.copy(), gate_voltage)
        )

    parameters = point(mode="conduction-only", t_j=25, v_g=15)
    with pytest.raises(
        ValueError,
        match=r"gate-conditioned; provide diode_v_g \(available: -4 V, -2 V, 0 V\)",
    ):
        calculate_topology_operating_point("buck", parameters, transistor, transistor)

    selected = calculate_topology_operating_point(
        "buck", {**parameters, "diode_v_g": 0}, transistor, transistor
    )
    assert selected["model"]["curve_conditions"]["diode_channel"] == {
        "t_j": 25,
        "v_g": 0,
    }


def test_ungated_diode_curve_is_selected_when_mixed_with_gate_conditioned_curves() -> None:
    """Keep the exact ungated diode curve selectable when other gates exist."""
    transistor = device()
    transistor.diode.channel_data.append(
        ChannelCharacteristics(
            25,
            np.array([[0.5, 5.5], [0.1, 100.0]]),
            -4.0,
        )
    )

    result = calculate_topology_operating_point(
        "buck",
        point(mode="conduction-only", t_j=25, v_g=15),
        transistor,
        transistor,
    )

    assert result["model"]["curve_conditions"]["diode_channel"] == {
        "t_j": 25,
        "v_g": None,
    }


def test_detailed_mode_rejects_negative_voltage_anywhere_in_channel_curve() -> None:
    """Reject interior negative voltage samples rather than checking only endpoints."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0, 1] = -0.1
    with pytest.raises(ValueError, match="switch channel curve must use non-negative current and voltage"):
        calculate_topology_operating_point(
            "buck",
            point(mode="detailed", t_j=25, v_g=15, r_g_on=10, r_g_off=10),
            transistor,
            device(),
        )


def test_detailed_mode_matches_and_records_turn_off_voltage_for_both_switching_curves() -> None:
    """Use turn-on voltage for Eon and the requested off voltage for Eoff."""
    transistor = device()
    transistor.switch.e_on_data.append(SwitchingLossData(
        "graph_i_e", 25, 100, 15, r_g=10, v_g_off=-10,
        graph_i_e=np.array([[0.1, 100.0], [0.0003, 0.3]]),
        commutation_device="fixture-diode", measurement_testbench="independent-fixture",
    ))
    transistor.switch.e_off_data.append(SwitchingLossData(
        "graph_i_e", 25, 100, 15, r_g=10, v_g_off=-10,
        graph_i_e=np.array([[0.1, 100.0], [0.0004, 0.4]]),
        commutation_device="fixture-diode", measurement_testbench="independent-fixture",
    ))

    result = calculate_topology_operating_point(
        "buck",
        point(mode="detailed", t_j=25, v_g=15, v_g_off=-10, r_g_on=10, r_g_off=10),
        transistor,
        device(),
    )

    assert result["model"]["curve_conditions"]["e_on"]["v_g_off"] == -10
    assert result["model"]["curve_conditions"]["e_off"]["v_g_off"] == -10


@pytest.mark.parametrize("requested_v_g_off", [None, -4.0])
def test_detailed_mode_selects_legacy_cree_turn_off_gate_data(
    requested_v_g_off: float | None,
) -> None:
    """Match CREE legacy Eoff.v_g as the turn-off gate condition."""
    cree_path = Path(__file__).parent / "master_data" / "test_data_CREE_C3M0060065J.json"
    cree = JsonTransistorLoader().load_from_json(cree_path)
    e_on = next(
        curve for curve in cree.switch.e_on_data
        if all((
            curve.dataset_type == "graph_i_e",
            curve.t_j == 25,
            curve.v_supply == 400,
            curve.v_g == 15,
            curve.r_g == 2.5,
        ))
    )
    e_off = next(
        curve for curve in cree.switch.e_off_data
        if all((
            curve.dataset_type == "graph_i_e",
            curve.t_j == 25,
            curve.v_supply == 400,
            curve.v_g == -4,
            curve.v_g_off is None,
            curve.r_g == 2.5,
        ))
    )
    assert e_on.v_g_off is None

    transistor = device()
    transistor.switch.e_on_data = [e_on]
    transistor.switch.e_off_data = [e_off]
    transistor.diode.channel_data[0].v_g = 0
    result = calculate_topology_operating_point(
        "buck",
        point(
            mode="detailed",
            v_in=400,
            v_out=200,
            p_out=2000,
            f_sw=20_000,
            l_value=1e-3,
            c_out=1e-3,
            t_j=25,
            v_g=15,
            diode_v_g=0,
            v_g_off=requested_v_g_off,
            r_g_on=2.5,
            r_g_off=2.5,
        ),
        transistor,
        transistor,
    )

    assert result["model"]["curve_conditions"]["e_on"]["v_g"] == 15
    assert result["model"]["curve_conditions"]["e_on"]["v_g_off"] is None
    assert result["model"]["curve_conditions"]["e_off"]["v_g"] == -4
    assert result["model"]["curve_conditions"]["e_off"]["v_g_off"] is None
    assert result["losses"]["switching_off_w"] > 0


def test_detailed_mode_reports_ambiguous_legacy_turn_off_gate_conditions() -> None:
    """List legacy off-gate values when an omitted selection is ambiguous."""
    transistor = device()
    transistor.switch.e_off_data.extend([
        SwitchingLossData(
            "graph_i_e", 25, 100, -4, r_g=10,
            graph_i_e=np.array([[0.1, 100.0], [0.0003, 0.3]]),
        ),
        SwitchingLossData(
            "graph_i_e", 25, 100, -3, r_g=10,
            graph_i_e=np.array([[0.1, 100.0], [0.0003, 0.3]]),
        ),
    ])
    with pytest.raises(ValueError, match="available turn-off gate voltages: -5 V, -4 V, -3 V"):
        calculate_topology_operating_point(
            "buck",
            point(mode="detailed", t_j=25, v_g=15, r_g_on=10, r_g_off=10),
            transistor,
            device(),
        )


def test_detailed_mode_reports_available_turn_off_gates_when_requested_gate_is_missing() -> None:
    """Explain which Eoff gate condition is present when an exact gate is absent."""
    transistor = device()
    transistor.switch.e_on_data = [
        SwitchingLossData(
            "graph_i_e", 25, 100, 15, r_g=10,
            graph_i_e=np.array([[0.1, 100.0], [0.0001, 0.1]]),
        )
    ]
    transistor.switch.e_off_data = [
        SwitchingLossData(
            "graph_i_e", 25, 100, -5, r_g=10,
            graph_i_e=np.array([[0.1, 100.0], [0.0002, 0.2]]),
        )
    ]
    with pytest.raises(
        ValueError,
        match="Missing required e_off.*available turn-off gate voltages: -5 V",
    ):
        calculate_topology_operating_point(
            "buck",
            point(
                mode="detailed", t_j=25, v_g=15, v_g_off=-10,
                r_g_on=10, r_g_off=10,
            ),
            transistor,
            device(),
        )


def test_parallel_device_loss_uses_even_current_sharing_without_curve_scaling() -> None:
    """Scale per-device current while preserving the original curve samples."""
    transistor = device()
    parameters = point(mode="detailed", t_j=25, v_g=15, v_g_off=-5, r_g_on=10, r_g_off=10)
    single = calculate_topology_operating_point("buck", parameters, transistor, transistor)
    parallel = calculate_topology_operating_point(
        "buck", {**parameters, "parallel_switches": 2, "parallel_diodes": 2}, transistor, transistor
    )
    assert parallel["losses"]["conduction_switch_w"] < single["losses"]["conduction_switch_w"]
    assert parallel["losses"]["conduction_diode_w"] < single["losses"]["conduction_diode_w"]
    assert parallel["model"]["parallel_switches"] == 2


def test_detailed_mode_rejects_current_outside_device_curve_range() -> None:
    """Reject current values outside recorded channel data."""
    transistor = device(max_current=10)
    with pytest.raises(ValueError, match="switch channel curve supports"):
        calculate_topology_operating_point(
            "buck", point(mode="detailed", t_j=25, v_g=15, r_g_on=10, r_g_off=10),
            transistor, transistor,
        )


def test_detailed_mode_does_not_select_a_nearby_gate_condition() -> None:
    """Require an exact gate condition instead of selecting a nearby curve."""
    transistor = device()
    with pytest.raises(ValueError, match="Missing required switch channel curve at t_j=25 C, v_g=10 V"):
        calculate_topology_operating_point(
            "buck", point(mode="detailed", t_j=25, v_g=10, r_g_on=10, r_g_off=10),
            transistor, transistor,
        )


def test_steady_state_temperatures_use_per_device_loss_and_tdb_thermal_resistance() -> None:
    """Use per-device losses and recorded thermal resistances for temperatures."""
    transistor = device()
    transistor.switch.thermal_foster = FosterThermalModel(r_th_total=1.0)
    transistor.diode.thermal_foster = FosterThermalModel(r_th_total=2.0)
    result = calculate_topology_operating_point(
        "buck",
        point(mode="detailed", t_j=25, t_case=30, v_g=15, v_g_off=-5, r_g_on=10, r_g_off=10),
        transistor,
        transistor,
    )
    assert result["temperatures"]["switch_c"] == pytest.approx(
        30 + result["losses"]["per_switch_device_w"] * 1.2
    )
    assert result["temperatures"]["diode_c"] == pytest.approx(
        30 + result["losses"]["per_diode_device_w"] * 2.2
    )


def test_nonpositive_tdb_thermal_resistance_is_reported_as_unavailable() -> None:
    """Treat TDB zero sentinels as missing rather than a zero-rise path."""
    transistor = device()
    transistor.switch.thermal_foster = FosterThermalModel(r_th_total=1.0)
    transistor.diode.thermal_foster = FosterThermalModel(r_th_total=0.0)
    transistor.thermal_properties.r_th_diode_cs = 0.0

    result = calculate_topology_operating_point(
        "buck",
        point(
            mode="conduction-only", t_j=25, t_case=30, v_g=15,
        ),
        transistor,
        transistor,
    )

    assert result["temperatures"]["switch_c"] is not None
    assert result["temperatures"]["diode_c"] is None
    assert "must be finite and greater than zero" in result["temperatures"]["diode_unavailable_reason"]


def test_line_and_contour_sweeps_have_documented_axis_shapes_and_units() -> None:
    """Generate line and contour arrays with stable shapes and SI units."""
    line = calculate_topology_sweep(
        "buck", point(),
        {"x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000, "x_points": 3, "metric": "i_l_ripple_pp_a"},
    )
    assert line["kind"] == "line"
    assert line["x_unit"] == "Hz"
    assert len(line["values"]) == len(line["x_values"]) == 3
    assert line["values"][0] > line["values"][-1]

    contour = calculate_topology_sweep(
        "buck", point(),
        {
            "x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000, "x_points": 3,
            "y_axis": "l_value", "y_min": 200e-6, "y_max": 400e-6, "y_points": 2,
            "metric": "i_l_ripple_pp_a",
        },
    )
    assert contour["kind"] == "contour"
    assert len(contour["values"]) == 2
    assert all(len(row) == 3 for row in contour["values"])


def test_sweep_does_not_silently_skip_an_unsupported_operating_point() -> None:
    """Fail a sweep with the coordinate of its first unsupported point."""
    with pytest.raises(ValueError, match="Sweep point .* is unsupported: Boost requires"):
        calculate_topology_sweep(
            "boost", point(v_in=100, v_out=150),
            {"x_axis": "v_out", "x_min": 90, "x_max": 110, "x_points": 2, "metric": "duty_cycle"},
        )


def test_contour_sweep_enforces_work_limit() -> None:
    """Bound contour work before calculating any grid points."""
    with pytest.raises(ValueError, match="limited to 2500 operating points"):
        calculate_topology_sweep(
            "buck", point(),
            {
                "x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000, "x_points": 51,
                "y_axis": "l_value", "y_min": 200e-6, "y_max": 400e-6, "y_points": 51,
                "metric": "i_l_ripple_pp_a",
            },
        )


@pytest.mark.parametrize(
    "sweep",
    [
        {
            "x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000,
            "x_points": 201, "metric": "i_l_ripple_pp_a",
        },
        {
            "x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000,
            "x_points": 2, "y_axis": "l_value", "y_min": 200e-6,
            "y_max": 400e-6, "y_points": 201, "metric": "i_l_ripple_pp_a",
        },
    ],
    ids=["x-axis", "y-axis"],
)
def test_sweep_limits_each_axis_to_200_points(sweep: dict[str, Any]) -> None:
    """Reject line and contour axes above the documented 200-point cap."""
    with pytest.raises(ValueError, match="2 to 200 points"):
        calculate_topology_sweep("buck", point(), sweep)


@pytest.mark.parametrize(
    ("sweep", "expected_x_points", "expected_y_points"),
    [
        (
            {
                "x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000,
                "x_points": 200, "metric": "i_l_ripple_pp_a",
            },
            200,
            None,
        ),
        (
            {
                "x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000,
                "x_points": 2, "y_axis": "l_value", "y_min": 200e-6,
                "y_max": 400e-6, "y_points": 200, "metric": "i_l_ripple_pp_a",
            },
            2,
            200,
        ),
    ],
    ids=["line-x-axis", "contour-y-axis"],
)
def test_sweep_accepts_200_points_on_each_axis(
    sweep: dict[str, Any],
    expected_x_points: int,
    expected_y_points: int | None,
) -> None:
    """Allow the documented inclusive 200-point maximum on either axis."""
    result = calculate_topology_sweep("buck", point(), sweep)
    assert len(result["x_values"]) == expected_x_points
    if expected_y_points is not None:
        assert len(result["y_values"]) == expected_y_points
        assert len(result["values"]) == expected_y_points
        assert all(len(row) == expected_x_points for row in result["values"])


def test_loss_balanced_line_sweep_matches_independent_scalar_points() -> None:
    """Every sweep value is the scalar solver result at its exact coordinate."""
    switch = device("loss-balanced switch")
    diode = device("loss-balanced diode")
    parameters = point(
        mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
        f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
    )
    sweep = {
        "x_axis": "f_sw", "x_min": 20_000, "x_max": 30_000,
        "x_points": 3, "metric": "p_loss_total_w",
    }
    result = calculate_topology_sweep("boost", parameters, sweep, switch, diode)

    assert result["kind"] == "line"
    assert result["x_unit"] == "Hz"
    assert result["loss_scope"] == "channel_conduction_only"
    assert "volt-second residual" in result["voltage_law_note"]
    for frequency, value, residual in zip(
        result["x_values"],
        result["values"],
        result["voltage_law_residual_values_v"],
        strict=True,
    ):
        scalar_parameters = {**parameters, "f_sw": frequency}
        scalar = calculate_topology_operating_point(
            "boost", scalar_parameters, switch, diode
        )
        assert value == pytest.approx(scalar["losses"]["total_w"])
        assert residual == pytest.approx(
            scalar["calculated"]["inductor_volt_second_residual_v"]
        )


def test_loss_balanced_contour_sweep_matches_scalar_points() -> None:
    """Contour rows and columns preserve the scalar values and axis shape."""
    switch = device("loss-balanced switch")
    diode = device("loss-balanced diode")
    parameters = point(
        mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
        f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
    )
    result = calculate_topology_sweep(
        "boost", parameters,
        {
            "x_axis": "f_sw", "x_min": 20_000, "x_max": 30_000,
            "x_points": 3, "y_axis": "l_value", "y_min": 1e-3,
            "y_max": 2e-3, "y_points": 2, "metric": "efficiency_percent",
        },
        switch, diode,
    )

    assert result["kind"] == "contour"
    assert result["x_unit"] == "Hz"
    assert result["y_unit"] == "H"
    assert len(result["values"]) == 2
    assert all(len(row) == 3 for row in result["values"])
    residuals = result["voltage_law_residual_values_v"]
    assert len(residuals) == 2
    assert all(len(row) == 3 for row in residuals)
    for y_index, inductance in enumerate(result["y_values"]):
        for x_index, frequency in enumerate(result["x_values"]):
            scalar = calculate_topology_operating_point(
                "boost", {**parameters, "l_value": inductance, "f_sw": frequency},
                switch, diode,
            )
            assert result["values"][y_index][x_index] == pytest.approx(
                scalar["performance"]["efficiency_percent"]
            )
            assert residuals[y_index][x_index] == pytest.approx(
                scalar["calculated"]["inductor_volt_second_residual_v"]
            )


def test_loss_balanced_sweep_reports_out_of_range_coordinate() -> None:
    """A single unsupported curve point fails the sweep without extrapolation."""
    switch = device("limited switch", max_current=12)
    diode = device("limited diode", max_current=12)
    parameters = point(
        mode="loss-balanced", v_in=100, v_out=200, p_out=1000,
        f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
    )
    with pytest.raises(ValueError, match=r"Sweep point \(x=2000 W\).*curve supports"):
        calculate_topology_sweep(
            "boost", parameters,
            {"x_axis": "p_out", "x_min": 1000, "x_max": 3000,
             "x_points": 3, "metric": "efficiency_percent"},
            switch, diode,
        )


@pytest.mark.parametrize(
    ("topology", "point_overrides"),
    [
        ("buck", {"v_in": 100, "v_out": 50, "p_out": 10}),
        ("boost", {"v_in": 100, "v_out": 200, "p_out": 100}),
        ("buck-boost", {"v_in": 100, "v_out": 50, "p_out": 10}),
        (
            "flyback",
            {"v_in": 100, "v_out": 50, "p_out": 10, "turns_ratio_nps": 2},
        ),
    ],
)
def test_loss_balanced_dcm_line_and_contour_sweeps_match_scalar_points(
    topology: str,
    point_overrides: dict[str, float],
) -> None:
    """Each DCM sweep metric and both residuals match its scalar coordinate."""
    switch = _flat_dcm_device(f"{topology} DCM sweep switch")
    diode = _flat_dcm_device(f"{topology} DCM sweep diode")
    parameters = point(
        mode="loss-balanced", conduction_regime="DCM",
        f_sw=20_000, l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
        **point_overrides,
    )
    line = calculate_topology_sweep(
        topology,
        parameters,
        {
            "x_axis": "f_sw", "x_min": 19_000, "x_max": 20_000,
            "x_points": 3, "metric": "p_loss_total_w",
        },
        switch,
        diode,
    )

    assert line["kind"] == "line"
    assert line["x_unit"] == "Hz"
    assert line["conduction_regime"] == "DCM"
    assert line["loss_scope"] == "channel_conduction_only"
    assert len(line["values"]) == len(line["voltage_law_residual_values_v"]) == 3
    assert len(line["power_balance_residual_values_w"]) == 3
    for index, frequency in enumerate(line["x_values"]):
        scalar = calculate_topology_operating_point(
            topology,
            {**parameters, "f_sw": frequency},
            switch,
            diode,
        )
        assert line["values"][index] == pytest.approx(
            scalar["losses"]["total_w"]
        )
        assert line["voltage_law_residual_values_v"][index] == pytest.approx(
            scalar["calculated"]["inductor_volt_second_residual_v"]
        )
        assert line["power_balance_residual_values_w"][index] == pytest.approx(
            scalar["performance"]["power_balance_residual_w"]
        )

    contour = calculate_topology_sweep(
        topology,
        parameters,
        {
            "x_axis": "f_sw", "x_min": 19_000, "x_max": 20_000,
            "x_points": 3, "y_axis": "c_out", "y_min": 0.8e-3,
            "y_max": 1.2e-3, "y_points": 2,
            "metric": "v_out_ripple_pp_v",
        },
        switch,
        diode,
    )

    assert contour["kind"] == "contour"
    assert contour["x_unit"] == "Hz"
    assert contour["y_unit"] == "F"
    assert len(contour["values"]) == 2
    assert all(len(row) == 3 for row in contour["values"])
    for y_index, capacitance in enumerate(contour["y_values"]):
        for x_index, frequency in enumerate(contour["x_values"]):
            scalar = calculate_topology_operating_point(
                topology,
                {**parameters, "c_out": capacitance, "f_sw": frequency},
                switch,
                diode,
            )
            assert contour["values"][y_index][x_index] == pytest.approx(
                scalar["calculated"]["v_out_ripple_pp_v"]
            )
            assert contour["voltage_law_residual_values_v"][y_index][x_index] == pytest.approx(
                scalar["calculated"]["inductor_volt_second_residual_v"]
            )
            assert contour["power_balance_residual_values_w"][y_index][x_index] == pytest.approx(
                scalar["performance"]["power_balance_residual_w"]
            )


def test_loss_balanced_dcm_sweep_reports_unsupported_coordinate() -> None:
    """A DCM boundary crossing rejects the sweep with its power coordinate."""
    transistor = _flat_dcm_device()
    parameters = point(
        mode="loss-balanced", conduction_regime="DCM", topology="boost",
        v_in=100, v_out=200, p_out=100, f_sw=20_000,
        l_value=1e-3, c_out=1e-3, t_j=25, v_g=15,
    )
    with pytest.raises(
        ValueError,
        match=r"Sweep point \(x=150 W\) is unsupported: .*at or above the CCM boundary",
    ):
        calculate_topology_sweep(
            "boost",
            parameters,
            {
                "x_axis": "p_out", "x_min": 100, "x_max": 200,
                "x_points": 3, "metric": "p_loss_total_w",
            },
            transistor,
            transistor,
        )
