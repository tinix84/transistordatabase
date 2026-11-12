"""Independent post-Miller gate-only timing and evidence fixtures."""
from copy import deepcopy
import math

import numpy as np
import pytest

from transistordatabase.analytical_models import GateChargeModel, GateChargeModelParams


def model(qgs=4e-9, qgd=2e-9, qg=12e-9):
    """Create the constant-capacitance 10 V driver fixture."""
    return GateChargeModel(GateChargeModelParams(qgs, qgd, qg, 2, 4, 1.5, .5))


def test_target_phase_preserves_existing_directions_and_charge_reference():
    device = model()
    before = device.calc_switching_estimate(10, 0)
    untouched = deepcopy(before)
    result = device.calc_switching_estimate(10, 0, settling_target_v=7)
    assert before == untouched
    assert result['turn_on'] == before['turn_on']
    assert result['turn_off'] == before['turn_off']
    assert result['assumptions'] == before['assumptions']
    phase = result['turn_on_settling']
    assert phase['anchor_s'] == before['turn_on']['times_s']['t_total']
    assert phase['duration_s'] == pytest.approx(2e-9 * math.log(2), rel=1e-14)
    assert phase['post_plateau_capacitance_f'] == pytest.approx(1e-9, rel=1e-14)
    assert phase['incremental_charge_c'] == pytest.approx(3e-9, rel=1e-14)
    assert (phase['initial_gate_current_a'], phase['final_gate_current_a']) == (3, 1.5)
    assert 'Von' in phase['charge_reference']
    assert not any('energy' in key or 'drain' in key for key in phase)


def test_relative_trace_remains_a_resistive_rc_gate_trajectory():
    phase = model().calc_switching_estimate(10, -5, settling_target_v=7)['turn_on_settling']
    times = phase['relative_time_s']
    assert times[0] == 0 and times[-1] == phase['duration_s']
    assert all(b > a for a, b in zip(times, times[1:], strict=False))
    for time, voltage, current in zip(times, phase['v_gate_v'], phase['i_gate_a'], strict=True):
        expected_v = 4 + 6 * -math.expm1(-time / 2e-9)
        assert voltage == pytest.approx(expected_v, rel=1e-14)
        assert current == pytest.approx((10 - voltage) / 2, rel=1e-14)
    assert phase['v_gate_v'][0] == 4 and phase['v_gate_v'][-1] == 7


def test_zero_post_charge_is_an_explicit_ideal_crossing_without_a_current_pulse():
    qgs, qgd = 2.0**-30, 2.0**-31
    phase = model(qgs, qgd, qgs + qgd).calc_switching_estimate(10, 0, settling_target_v=7)['turn_on_settling']
    assert phase['status'] == 'instantaneous-ideal-crossing'
    assert phase['duration_s'] == phase['incremental_charge_c'] == 0
    assert phase['relative_time_s'] == [0, 0]
    assert phase['v_gate_v'] == [4, 7]
    assert phase['i_gate_a'] == [None, None]
    assert phase['initial_gate_current_a'] is None
    assert phase['final_gate_current_a'] is None


@pytest.mark.parametrize('target', [4, 10, -1, float('nan'), float('inf'), True])
def test_invalid_target_never_becomes_a_rail_asymptote_or_implicit_phase(target):
    with pytest.raises(ValueError):
        model().calc_switching_estimate(10, 0, settling_target_v=target)


def test_near_plateau_duration_remains_positive_when_absolute_time_is_unresolved():
    phase = model(4e-6, 2e-6, 6.006e-6).calc_switching_estimate(
        10, 0, settling_target_v=float(np.nextafter(4., 10.)))['turn_on_settling']
    assert 0 < phase['duration_s'] < math.ulp(phase['anchor_s'])
    assert phase['relative_time_s'][-1] > 0
    assert phase['incremental_charge_c'] > 0
    assert phase['absolute_end_s'] is None
    assert phase['absolute_time_status'] == 'unresolved-use-relative-time'
    assert not phase['instantaneous']


def test_target_just_below_the_rail_keeps_a_finite_positive_gap():
    target = float(np.nextafter(10., 4.))
    phase = model().calc_switching_estimate(10, 0, settling_target_v=target)['turn_on_settling']
    expected = 2e-9 * math.log1p((target - 4) / (10 - target))
    assert math.isfinite(phase['duration_s'])
    assert phase['duration_s'] == pytest.approx(expected, rel=1e-14)
    assert phase['final_gate_current_a'] > 0


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
                    reason='This platform has no extended input precision beyond binary64')
def test_extended_positive_charge_cannot_silently_become_an_ideal_zero_phase():
    extended = np.longdouble(1) + np.finfo(np.longdouble).eps
    device = model(np.longdouble(1), np.longdouble(0), extended)
    assert device.calc_switching_estimate(10, 0)['turn_off']['times_s']['t_delay'] > 0
    with pytest.raises(ValueError, match='binary64'):
        device.calc_switching_estimate(10, 0, settling_target_v=7)


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
                    reason='This platform has no extended input precision beyond binary64')
def test_extended_target_cannot_silently_round_to_another_gate_voltage():
    target = np.longdouble(7) + 8 * np.finfo(np.longdouble).eps
    assert target != float(target)
    with pytest.raises(ValueError, match='binary64'):
        model().calc_switching_estimate(10, 0, settling_target_v=target)


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
                    reason='This platform has no extended input precision beyond binary64')
def test_zero_charge_mode_validates_resistance_before_it_can_round_to_zero():
    device = GateChargeModel(GateChargeModelParams(1e300, 0, 1e300, 2, 4,
                                                  np.longdouble('1e-400'), 0))
    with pytest.raises(ValueError, match='binary64'):
        device.calc_switching_estimate(10, 0, settling_target_v=7)


def test_raw_optional_mode_does_not_change_charge_classification_or_coerce_strings():
    import json
    from fastapi.testclient import TestClient
    from transistordatabase.gui_web.backend.main import app

    client = TestClient(app)
    ordinary = {"q_gs": 4e-9, "q_gd": 2e-9, "q_g": 12e-9, "v_th": 2,
                "v_plateau": 4, "r_g": 2, "r_g_int": 0, "v_driver": 10,
                "v_off": 0, "settling_target_v": 7.0}
    canonical = json.dumps(ordinary).replace('4e-09', '4.000e-09').replace('7.0', '7.000')
    assert client.post('/api/analytical/gate_charge', content=canonical,
                       headers={'Content-Type': 'application/json'}).status_code == 200
    raw_positive = json.dumps({**ordinary, 'q_gs': 1, 'q_gd': 0, 'q_g': 1}).replace('"q_g": 1,', '"q_g": 1.0000000000000000001,')
    raw_zero = json.dumps({**ordinary, 'q_gs': .1, 'q_gd': .3, 'q_g': .4})
    string_target = json.dumps({**ordinary, 'settling_target_v': '7'})
    huge_integer = json.dumps({**ordinary, 'q_g': 10**400})
    for raw in (raw_positive, raw_zero, string_target, huge_integer):
        response = client.post('/api/analytical/gate_charge', content=raw,
                               headers={'Content-Type': 'application/json'})
        assert response.status_code == 422
    legacy = {key: value for key, value in ordinary.items() if key != 'settling_target_v'}
    legacy['q_gd'] = '2e-9'
    response = client.post('/api/analytical/gate_charge', json=legacy)
    assert response.status_code == 200
    assert 'turn_on_settling' not in response.json()['estimate']
