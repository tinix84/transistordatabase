"""Independent inverse gate-resistance fixtures and numeric-domain checks."""

import math

import numpy as np
import pytest

from transistordatabase.device_parameters import signed_gate_slew
from transistordatabase.gate_resistance import gate_resistance_for_slew


BASE = dict(target_dv_dt_v_s=20e9, c_rss_min_f=100e-12,
            v_drive_high_v=15, v_drive_low_v=-5,
            v_plateau_low_v=5, v_plateau_high_v=7, selected=True)


def test_independent_legacy_fixture_and_plateau_interval():
    """Two amperes give 4..5 ohms on and 5..6 ohms off."""
    result = gate_resistance_for_slew(**BASE)
    for key, expected in {"gate_current_magnitude_a": 2, "r_on_min_ohm": 4,
                          "r_on_max_ohm": 5, "r_off_min_ohm": 5,
                          "r_off_max_ohm": 6, "r_on_ohm": 5,
                          "r_off_ohm": 6}.items():
        assert result[key] == pytest.approx(expected)
    assert "total series" in " ".join(result["assumptions"])
    assert result["unit"] == "ohm"
    assert result["model_source"].endswith("sc-92afe9737e47b855")
    assert "minimum total resistance" in result["sizing_policy"]


def test_original_source_fixture_retains_legacy_on_off_values():
    """The source's 4..7 V plateau fixture gives 5.5 and 6 ohms."""
    result = gate_resistance_for_slew(**{**BASE, "v_plateau_low_v": 4})
    assert result["r_on_ohm"] == pytest.approx(5.5)
    assert result["r_off_ohm"] == pytest.approx(6)


@pytest.mark.parametrize("plateau", [5, 6, 7])
@pytest.mark.parametrize("turn_on", [True, False])
def test_forward_authority_roundtrip_and_worst_case(plateau, turn_on):
    """Upper resistance bounds respect the target at both plateau endpoints."""
    result = gate_resistance_for_slew(**BASE)
    resistance = result["r_on_ohm" if turn_on else "r_off_ohm"]
    drive = 15 if turn_on else -5
    forward = signed_gate_slew(plateau, 0, 10, resistance, 2e-9, 100e-12, drive)
    assert abs(forward["dv_ce_dt_v_s"]) <= 20e9 * (1 + 1e-15)
    assert math.copysign(1, forward["dv_ce_dt_v_s"]) == (-1 if turn_on else 1)
    if plateau == (5 if turn_on else 7):
        assert abs(forward["dv_ce_dt_v_s"]) == pytest.approx(20e9)
    endpoint_resistance = (drive - plateau if turn_on else plateau - drive) / 2
    exact = signed_gate_slew(plateau, 0, 10, endpoint_resistance, 2e-9, 100e-12, drive)
    assert abs(exact["dv_ce_dt_v_s"]) == pytest.approx(BASE["target_dv_dt_v_s"])


@pytest.mark.parametrize("key", [key for key in BASE if key != "selected"])
@pytest.mark.parametrize("invalid", [True, np.bool_(False), 1j, np.complex128(1),
                                      "1", None, [1], np.array(1),
                                      math.nan, math.inf, -math.inf])
def test_reject_nonreal_nonfinite_or_nonscalar_inputs(key, invalid):
    """Direct Python calls enforce the same physical scalar contract as HTTP."""
    with pytest.raises(ValueError, match="finite real numeric scalar"):
        gate_resistance_for_slew(**{**BASE, key: invalid})


@pytest.mark.parametrize("changes", [
    {"target_dv_dt_v_s": 0}, {"target_dv_dt_v_s": -1}, {"c_rss_min_f": 0},
    {"c_rss_min_f": -1}, {"v_plateau_low_v": 8}, {"v_drive_high_v": 7},
    {"v_drive_low_v": 5}, {"v_drive_low_v": 20}, {"selected": False},
    {"selected": 1}, {"target_dv_dt_v_s": 1e308, "c_rss_min_f": 1e308},
    {"target_dv_dt_v_s": 1e-300, "c_rss_min_f": 1e-300},
])
def test_reject_invalid_domains_and_unrepresentable_outputs(changes):
    """Do not clamp reversed rails, invalid capacitance, or numeric overflow."""
    with pytest.raises(ValueError):
        gate_resistance_for_slew(**{**BASE, **changes})


def test_numpy_scalars_collapsed_plateau_and_si_scaling():
    """A single plateau collapses bounds and capacitance remains in farads."""
    result = gate_resistance_for_slew(**{**BASE, "v_plateau_high_v": np.float64(5),
                                       "v_drive_high_v": np.int64(15)})
    assert result["r_on_min_ohm"] == result["r_on_max_ohm"]
    assert result["r_off_min_ohm"] == result["r_off_max_ohm"]
    scaled = gate_resistance_for_slew(**{**BASE, "c_rss_min_f": 200e-12})
    assert scaled["r_on_ohm"] == pytest.approx(2.5)


def test_finite_extreme_difference_and_adjacent_float_precision():
    """A finite resistance can survive an overflowing float voltage difference."""
    result = gate_resistance_for_slew(1, 1e308, 1.6e308, -1.6e308,
                                     -8e307, 8e307, selected=True)
    assert result["r_on_ohm"] == pytest.approx(2.4)
    assert result["r_off_ohm"] == pytest.approx(2.4)
    near = math.nextafter(15.0, -math.inf)
    result = gate_resistance_for_slew(**{**BASE, "v_plateau_low_v": near,
                                       "v_plateau_high_v": near})
    assert result["r_on_ohm"] == pytest.approx((15 - near) / 2, rel=1e-15, abs=0)
