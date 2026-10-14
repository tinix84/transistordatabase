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


@pytest.mark.parametrize("topology", ["buck", "buck-boost", "flyback"])
def test_loss_balanced_mode_rejects_unsupported_dcm_topologies(topology: str) -> None:
    """Keep DCM channel feedback explicitly Boost-only."""
    with pytest.raises(ValueError, match="Boost only"):
        calculate_topology_operating_point(
            topology,
            point(mode="loss-balanced", conduction_regime="DCM"),
            device(),
            device(),
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
    ("topology", "parameters", "expected_duty"),
    [
        (
            "buck",
            point(mode="loss-balanced", v_in=100, v_out=50, p_out=500, l_value=1e-3, f_sw=20_000),
            52 / 101,
        ),
        (
            "buck-boost",
            point(mode="loss-balanced", v_in=100, v_out=50, p_out=500, l_value=1e-3, f_sw=20_000),
            52 / 151,
        ),
        (
            "flyback",
            point(mode="loss-balanced", v_in=100, v_out=50, p_out=500, turns_ratio_nps=2, l_value=1e-3, f_sw=20_000),
            104 / 203,
        ),
    ],
)
def test_loss_balanced_ccm_topologies_match_independent_constant_drop_balance(
    topology: str, parameters: dict[str, Any], expected_duty: float
) -> None:
    """Verify buck-derived duty with flat 1 V switch and 2 V diode drops."""
    transistor = device()
    transistor.switch.channel_data[0].graph_v_i[0] = 1.0
    transistor.diode.channel_data[0].graph_v_i[0] = 2.0
    result = calculate_topology_operating_point(
        topology,
        {**parameters, "t_j": 25, "v_g": 15, "diode_v_g": None},
        transistor,
        transistor,
    )
    assert result["calculated"]["duty_cycle"] == pytest.approx(expected_duty, rel=1e-8)
    assert result["calculated"]["inductor_volt_second_residual_v"] == pytest.approx(
        0, abs=1e-6
    )
    assert result["performance"]["p_in_w"] == pytest.approx(
        result["performance"]["p_out_w"] + result["losses"]["total_w"], rel=1e-8
    )
    assert result["performance"]["p_in_w"] == pytest.approx(
        parameters["v_in"] * result["calculated"]["i_in_avg_a"], rel=1e-8
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


def test_loss_balanced_sweep_rejects_dcm_with_actionable_error() -> None:
    """Reject DCM before beginning a loss-balanced sweep."""
    with pytest.raises(ValueError, match="support CCM only.*conduction_regime='CCM'"):
        calculate_topology_sweep(
            "boost", point(mode="loss-balanced", conduction_regime="DCM"),
            {"x_axis": "f_sw", "x_min": 10_000, "x_max": 20_000,
             "x_points": 3, "metric": "i_l_ripple_pp_a"},
        )
