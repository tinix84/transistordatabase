"""Independent RC/charge fixtures and the gate estimate HTTP contract."""
from dataclasses import replace
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from transistordatabase.analytical_models import GateChargeModel, GateChargeModelParams
from transistordatabase.gui_web.backend.main import app


PARAMS = GateChargeModelParams(10e-9, 15e-9, 50e-9, 3, 5, 6)
PAYLOAD = vars(PARAMS) | {'v_driver': 15, 'v_off': 0}


def test_exact_independent_fixture():
    """Check the RC integration fixture in seconds, with tight absolute tolerance."""
    model = GateChargeModel(PARAMS)
    assert list(model.calc_turn_on_times().values()) == pytest.approx(
        np.array([2.67772261577, 2.18785868153, 9, 13.8655812973]) * 1e-9,
        rel=1e-10, abs=1e-20,
    )
    assert list(model.calc_turn_off_times().values()) == pytest.approx(
        np.array([16.4791843300, 18, 6.12990748519, 40.6090918152]) * 1e-9,
        rel=1e-10, abs=1e-20,
    )


def test_negative_bias_uses_off_rail_in_both_directions():
    """Account for the off rail in both exponential boundary ratios."""
    model = GateChargeModel(PARAMS)
    on = model.calc_turn_on_times(15, -5)
    off = model.calc_turn_off_times(15, -5)
    assert on['t_delay'] == pytest.approx(12e-9 * np.log(20 / 12))
    assert off['t_delay'] == pytest.approx(15e-9 * np.log(20 / 10))
    assert off['t_rise_v'] == pytest.approx(9e-9)
    assert off['t_fall_i'] == pytest.approx(12e-9 * np.log(10 / 8))


@pytest.mark.parametrize('factor', [0.25, 3])
@pytest.mark.parametrize('kind', ['resistance', 'charge'])
def test_dimensional_scaling(factor, kind):
    """Resistance and charge scale each interval linearly."""
    changes = ({'r_g': PARAMS.r_g * factor} if kind == 'resistance' else
               {key: getattr(PARAMS, key) * factor for key in ('q_g', 'q_gs', 'q_gd')})
    base = GateChargeModel(PARAMS)
    scaled = GateChargeModel(replace(PARAMS, **changes))
    for method in ('calc_turn_on_times', 'calc_turn_off_times'):
        assert list(getattr(scaled, method)().values()) == pytest.approx(
            np.array(list(getattr(base, method)().values())) * factor
        )


@pytest.mark.parametrize('changes', [
    {'q_gs': 0}, {'q_gd': -1e-9}, {'q_g': 20e-9}, {'v_th': 0},
    {'v_th': 5}, {'v_plateau': 2}, {'r_g': 0}, {'r_g_int': -1},
    {'r_g': float('inf')}, {'q_g': float('nan')},
])
def test_science_rejects_invalid_domains(changes):
    """All reusable operations reject invalid model inputs."""
    model = GateChargeModel(replace(PARAMS, **changes))
    for method in (model.calc_turn_on_times, model.calc_turn_off_times, model.calc_switching_estimate):
        with pytest.raises(ValueError):
            method()


@pytest.mark.parametrize('von,voff', [(5, 0), (4, 0), (15, 3), (15, 5), (np.inf, 0), (15, np.nan)])
def test_rejects_underdrive_and_nonfinite_rails(von, voff):
    """Reject unsupported driver conditions before generating data."""
    with pytest.raises(ValueError):
        GateChargeModel(PARAMS).calc_switching_estimate(von, voff)


@pytest.mark.parametrize('voff', [0, -5])
def test_waveforms_match_phases_continuity_and_endpoint_scope(voff):
    """Check joins, finite samples and physical scope of each endpoint."""
    result = GateChargeModel(PARAMS).calc_switching_estimate(15, voff)
    for key in ('turn_on', 'turn_off'):
        trace = result[key]
        assert trace['time_s'][0] == 0
        assert trace['time_s'][-1] == pytest.approx(trace['times_s']['t_total'])
        assert all(np.diff(trace['time_s']) >= 0)
        for field in ('v_gate_v', 'i_gate_a', 'id_normalized', 'vds_normalized'):
            assert len(trace[field]) == len(trace['time_s'])
            assert np.all(np.isfinite(trace[field]))
            # All ordinary interval joins are continuous, including gate current.
            for i in (33, 66):
                assert trace[field][i - 1] == pytest.approx(trace[field][i])
        assert min(trace['id_normalized']) >= 0
        assert max(trace['id_normalized']) <= 1
        assert min(trace['vds_normalized']) == 0
        assert max(trace['vds_normalized']) == 1
        assert trace['phases'][1]['start_s'] == trace['phases'][0]['end_s']
    assert result['turn_on']['v_gate_v'][0] == voff
    assert result['turn_on']['v_gate_v'][-1] == 5
    assert result['turn_off']['v_gate_v'][0] == 15
    assert result['turn_off']['v_gate_v'][-1] == 3
    assert all(i >= 0 for i in result['turn_on']['i_gate_a'])
    assert all(i <= 0 for i in result['turn_off']['i_gate_a'])


def test_zero_post_charge_is_explicit_instantaneous_jump():
    """Zero post-charge causes a labeled zero-duration gate jump."""
    result = GateChargeModel(replace(PARAMS, q_g=PARAMS.q_gs + PARAMS.q_gd)).calc_switching_estimate()
    off = result['turn_off']
    assert off['times_s']['t_delay'] == 0
    assert off['phases'][0] == {'name': 't_delay', 'start_s': 0, 'end_s': 0, 'instantaneous': True}
    assert off['time_s'][:2] == [0, 0]
    assert off['v_gate_v'][:2] == [15, 5]
    assert off['id_normalized'][:3] == [1, 1, 1]
    assert off['vds_normalized'][:3] == [0, 0, 0]


def test_zero_miller_charge_is_explicit_not_minimum_capacitance():
    """Zero Miller charge gives a zero-duration normalized voltage jump."""
    estimate = GateChargeModel(replace(PARAMS, q_gd=0)).calc_switching_estimate()
    assert estimate['turn_on']['times_s']['t_fall_v'] == 0
    assert estimate['turn_on']['phases'][-1]['instantaneous']
    assert estimate['turn_on']['vds_normalized'][-2:] == [1, 0]


def test_api_ns_and_si_traces_agree_with_reusable_science():
    """The transport only changes units, never the underlying calculation."""
    with TestClient(app) as client:
        response = client.post('/api/analytical/gate_charge', json=PAYLOAD | {'v_off': -5})
    assert response.status_code == 200, response.text
    data = response.json()
    expected = GateChargeModel(PARAMS).calc_switching_estimate(15, -5)
    assert data['estimate'] == expected
    for key in ('turn_on', 'turn_off'):
        assert data[key + '_times'] == {k: v * 1e9 for k, v in expected[key]['times_s'].items()}


@pytest.mark.parametrize('changes, message', [
    ({'v_driver': 5}, 'Vplateau < Von'), ({'v_off': 3}, 'Voff < Vth'),
    ({'q_g': 20e-9}, 'Qg >= Qgs + Qgd'), ({'r_g': 0}, 'total R > 0'),
    ({'v_th': 5}, '0 < Vth < Vplateau'),
])
def test_api_domain_errors_are_actionable_422(changes, message):
    """Return model domain diagnostics as 422 responses."""
    with TestClient(app) as client:
        response = client.post('/api/analytical/gate_charge', json=PAYLOAD | changes)
    assert response.status_code == 422
    assert message in response.json()['detail']


@pytest.mark.parametrize('key', list(PAYLOAD))
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_api_rejects_nonfinite_json_values(key, value):
    """Nonstandard nonfinite JSON values must not crash error serialization."""
    with TestClient(app) as client:
        response = client.post('/api/analytical/gate_charge',
                               content=json.dumps(PAYLOAD | {key: value}),
                               headers={'content-type': 'application/json'})
    assert response.status_code == 422


def test_api_rejects_nanosecond_conversion_overflow():
    """Finite SI times can still exceed the adapter's nanosecond range."""
    with TestClient(app) as client:
        response = client.post('/api/analytical/gate_charge', json=PAYLOAD | {'r_g': 1e308})
    assert response.status_code == 422
    assert 'nanosecond API range' in response.json()['detail']


def test_archive_notebook_turn_on_intent():
    """Reproduce RC/Miller intervals intended by Mosfet_switching_losses.ipynb."""
    ciss, cgd, threshold, current = 1850e-12, 7e-12, 4, 8.7
    gm = (30 - 8) / (7.8 - 6)
    plateau = threshold + current / gm
    resistance, drive, bus = 11, 13, 200
    qgs, qgd = ciss * plateau, cgd * bus
    model = GateChargeModel(GateChargeModelParams(
        qgs, qgd, qgs + qgd, threshold, plateau, resistance,
    ))
    times = model.calc_turn_on_times(drive)
    expected = {
        't_delay': -resistance * ciss * np.log(1 - threshold / drive),
        't_rise': resistance * ciss * np.log((drive - threshold) / (drive - plateau)),
        't_fall_v': bus * resistance * cgd / (drive - plateau),
    }
    for key, value in expected.items():
        assert times[key] == pytest.approx(value, rel=1e-12, abs=1e-22)
    assert times['t_total'] == pytest.approx(sum(expected.values()), rel=1e-12)
    assert model.calc_turn_off_times(drive)['t_delay'] == 0
