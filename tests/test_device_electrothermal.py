"""Source identity, resource and numeric fixtures for native thermal feedback."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from transistordatabase.core.models import ChannelCharacteristics, SwitchingLossData, MeasurementEvidence
from transistordatabase.core.repository import TransistorFactory
import transistordatabase.device_electrothermal as module


def device():
    """Create a synthetic one-switch model with a known affine resistance law."""
    item = TransistorFactory.create_empty_transistor('Owner_Synthetic_Device')
    for temperature in (25, 125):
        current = np.array([0., 2., 4.])
        resistance = 1 + .01 * (temperature - 25)
        item.switch.channel_data.append(ChannelCharacteristics(temperature, np.array([current * resistance, current]), 15))
        for family, gate, energy in [('e_on_data', 15, .002), ('e_off_data', -15, .003)]:
            getattr(item.switch, family).append(SwitchingLossData('graph_i_e', temperature, 600, gate,
                r_g=5.6, v_g_off=None, graph_i_e=np.array([current, energy + .001 * current]),
                commutation_device='Synthetic diode', load_inductance=.001, commutation_inductance=1e-9,
                measurement_testbench='Synthetic bench', evidence=MeasurementEvidence(method='direct', convention='passive', event=family)))
    return item


def request():
    """Declare the conduction-only closed-form fixture and fixed sink path."""
    return {'selected': True, 'switching_policy': 'held-current-pwm', 'source': 'Synthetic fixed-sink path',
            'current_a': 2., 'duty': 1., 'frequency_hz': 0., 'sink_temperature_c': 25.,
            'resistance_k_w': 2., 'initial_temperature_c': 25., 'tolerance_k': 1e-9,
            'max_iterations': 200, 'relaxation': 1.,
            'channel': {'indices': [0, 1], 'conditions': {'v_g': 15}}, 'turn_on': None, 'turn_off': None}


def switching_request():
    value = request()
    value.update(current_a=0., duty=.5, frequency_hz=1000., channel=None)
    for family, gate in [('turn_on', 15), ('turn_off', -15)]:
        value[family] = {'indices': [0, 1], 'conditions': {'v_g': gate, 'v_g_off': None, 'r_g': 5.6, 'v_supply': 600}}
    return value


def test_closed_form_root_and_input_graphs_remain_unchanged():
    source, values = device(), request()
    before = [row.graph_v_i.copy() for row in source.switch.channel_data]
    original = deepcopy(values)
    result = module.analyze_device_electrothermal(source, values)
    assert result['result']['status'] == 'converged'
    assert result['result']['accepted_solution']['temperature_c'][0] == pytest.approx(25 + 8 / .92, abs=1e-8)
    assert values == original
    for record, old in zip(source.switch.channel_data, before, strict=True):
        np.testing.assert_array_equal(record.graph_v_i, old)
    result['request']['channel']['conditions']['v_g'] = 99
    assert values == original


def test_comment_digest_is_bounded_and_excluded_from_calculation_dependency():
    source, values = device(), switching_request()
    source.switch.e_on_data[0].comment = 'x' * 3000
    first = module.build_device_loss_table(source, values)
    evidence = next(row for row in first['record_evidence'] if row['record_key'] == 'turn_on:0')
    assert evidence['source_comment'] is None and evidence['source_comment_truncated']
    assert len(evidence['source_comment_preview']) == 512
    assert evidence['source_comment_sha256'] == hashlib.sha256(('x' * 3000).encode()).hexdigest()
    source.switch.e_on_data[0].comment = 'Different irrelevant source annotation'
    second = module.build_device_loss_table(source, values)
    assert first['source_identity']['selected_dependency_sha256'] == second['source_identity']['selected_dependency_sha256']
    source.switch.e_on_data[0].comment = 'x' * (module.MAX_COMMENT_CHARS + 1)
    with pytest.raises(module.DeviceThermalResourceError):
        module.build_device_loss_table(source, values)


def test_full_graphs_are_not_repeated_in_evidence_or_temperature_rows():
    source, values = device(), request()
    small = module.build_device_loss_table(source, values)
    for row in source.switch.channel_data:
        current = np.linspace(0, 4, 10000)
        row.graph_v_i = np.array([current * (1 + .01 * (row.t_j - 25)), current])
    large = module.build_device_loss_table(source, values)
    assert large['resource_use']['selected_points'] == 20000
    assert len(large['record_evidence']) == 2
    assert len(json.dumps(large)) < 2 * len(json.dumps(small))


def test_source_point_and_record_limits_precede_dense_encoding(monkeypatch):
    source, values = device(), request()
    monkeypatch.setattr(module, '_encode', lambda value: pytest.fail('Resource validation must precede source encoding'))
    source.switch.channel_data[0].graph_v_i = np.zeros((2, 10001))
    with pytest.raises(module.DeviceThermalResourceError):
        module.build_device_loss_table(source, values)
    values['channel']['indices'] = list(range(33))
    with pytest.raises(module.DeviceThermalResourceError):
        module.build_device_loss_table(source, values)


def test_positive_switching_power_underflow_does_not_become_inactive_zero():
    source, values = device(), switching_request()
    values['frequency_hz'] = 1e-200
    for rows in (source.switch.e_on_data, source.switch.e_off_data):
        for row in rows:
            row.graph_i_e = np.array([[0., 1.], [1e-200, 1e-200]])
    with pytest.raises(ValueError, match='nonzero'):
        module.analyze_device_electrothermal(source, values)


def test_exact_recorded_zero_E0_is_valid_with_a_sourced_temperature_domain():
    source, values = device(), switching_request()
    for rows in (source.switch.e_on_data, source.switch.e_off_data):
        for row in rows:
            row.graph_i_e = np.array([[0., 1.], [0., .001]])
    result = module.analyze_device_electrothermal(source, values)
    assert result['result']['status'] == 'converged'
    assert result['result']['accepted_solution']['temperature_c'] == [25]
    assert result['accepted_component_power_w']['sum_component_power_w'] == 0


def test_positive_current_interpolation_underflow_is_explicitly_unsupported():
    source, values = device(), switching_request()
    values['current_a'] = .25
    values['channel'] = request()['channel']
    for rows in (source.switch.e_on_data, source.switch.e_off_data):
        for row in rows:
            row.graph_i_e = np.array([[0., 1.], [0., np.nextafter(0., 1.)]])
    with pytest.raises(ValueError, match='underflows'):
        module.build_device_loss_table(source, values)


def test_unavailable_raw_identity_is_not_a_fabricated_repository_file_digest():
    table = module.build_device_loss_table(device(), request())
    assert table['source_identity']['repository_source'] is None
    assert len(table['source_identity']['selected_dependency_sha256']) == 64
    with pytest.raises(ValueError):
        module.build_device_loss_table(device(), request(), repository_source={'graphs': np.zeros((2, 5))})


def test_declared_integer_source_cannot_truncate_a_fractional_sample(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from transistordatabase.core.repository import JsonTransistorRepository
    from transistordatabase.gui_web.backend.device_electrothermal_routes import router, device_repository

    source = device()
    for row in source.switch.channel_data:
        row.graph_v_i = row.graph_v_i.astype(np.int64)
    repository = JsonTransistorRepository(tmp_path)
    repository.save(source)
    path = repository._storage_path(source.metadata.name)
    original = path.read_bytes()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[device_repository] = lambda: repository
    client = TestClient(app)
    payload = {**request(), 'transistor_id': source.metadata.name}
    assert client.post('/api/thermal/device/analyze', json=payload).status_code == 200
    document = json.loads(original)
    pending, target = [document], None
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get('kind') == 'ndarray' and np.dtype(value['dtype']).kind == 'i':
                target = value
                break
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    assert target is not None
    target['values']['values'][0]['values'][0] = 1.5
    changed = json.dumps(document).encode()
    path.write_bytes(changed)
    response = client.post('/api/thermal/device/analyze', json=payload)
    assert response.status_code == 422
    assert path.read_bytes() == changed
