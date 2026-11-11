"""Independent equation fixtures, condition guards and transport for Biela."""
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.integrate import quad

from transistordatabase.biela import (
    BielaConvergenceError, ChristenBielaModel, HalfBridgeParams,
    ReverseRecoveryParams, TransconductanceParams,
    calc_charge_equivalent_capacitance, calc_charge_stored,
    extract_reverse_recovery_params, fit_transconductance,
)
from transistordatabase.core.models import ChannelCharacteristics
from transistordatabase.gui_web.backend.main import app


def model(rr=None):
    """Model."""
    return ChristenBielaModel(1e-9, 1e-9, 1e-9, 200e-9,
                              TransconductanceParams(10, 0, 1, 3), rr)


P = HalfBridgeParams(100, 10, 15, 0, 10, 0, 0)
PAYLOAD = dict(model_kind='christen_biela_estimate', source='Independent algebra fixture',
               c_gs=1e-9, c_ds=1e-9, c_gd=1e-9, q_oss=200e-9, q_oss_v_ref=100,
               k_1=10, k_2=0, x=1, v_th=3, gm_i_min=1.3, gm_i_max=50,
               v_dc=100, i_load=10, v_driver_on=15, v_driver_off=0,
               r_g=10, l_s=0, l_d=0, t_j=25, parameter_t_j=25, recovery_mode='omit')


def test_independent_constant_gm_fixture_and_partitions():
    """Independent constant gm fixture and partitions."""
    r = model().calc_switching_energy(P)
    for key, expected in dict(i_oss_off=.7692307692307693,
                              i_oss_on=-2.1153846153846154, t_rv=260e-9,
                              t_fi=2.4846135929849973e-9, t_ri=.8701137698962976e-9,
                              t_fv=94.54545454545454e-9,
                              e_off=111.05118267395518e-6, e_on=67.7077841576754e-6,
                              e_total=178.7589668316306e-6, i_0_zvs=1.2).items():
        assert getattr(r, key) == pytest.approx(expected, rel=1e-11)
    for event in ('turn_on', 'turn_off'):
        assert sum(r.diagnostics[event + '_energy_partitions_j'].values()) == pytest.approx(
            r.e_on if event == 'turn_on' else r.e_off, rel=1e-13)
        assert r.diagnostics[event]['converged']
        assert abs(r.diagnostics[event]['equation_residual_a']) < 1e-10
    assert r.e_rr_s2 == 0
    assert 'omitted' in r.diagnostics['recovery']


def test_recovery_partitions_match_independent_numerical_integration():
    """Recovery partitions match independent numerical integration."""
    rr = ReverseRecoveryParams(2e-9, 5e-9, 10e-9 / 3, 88e-9)
    result = model(rr).calc_switching_energy(P)
    slope = P.i_0 / result.t_ri
    assert result.t_rs == pytest.approx(.890359204892e-9, rel=1e-10)
    assert result.i_rr_calc == pytest.approx(10.232675722367, rel=1e-10)
    assert result.q_rs == pytest.approx(4.555378510040e-9, rel=1e-10)
    assert result.e_on == pytest.approx(71.056924268450e-6, rel=1e-11)
    qrs = quad(lambda u: slope * (u * result.t_rs) * result.t_rs, 0, 1,
               epsabs=1e-18)[0]
    erf = quad(lambda u: P.v_0 * (1-u) * result.i_rr_calc
               * np.exp(-u * result.t_fv / rr.tau_rr) * result.t_fv,
               0, 1, epsabs=1e-18)[0]
    assert result.q_rs == pytest.approx(qrs, rel=1e-11)
    assert result.e_rs == pytest.approx(qrs * P.v_0, rel=1e-11)
    assert result.e_rf == pytest.approx(erf, rel=1e-11)
    assert result.e_rf == pytest.approx(2.003243054879e-6, rel=1e-10)
    assert result.diagnostics['body_diode_energy_j'] == pytest.approx(.043292089595e-6, rel=1e-9)
    assert abs(result.diagnostics['recovery_storage_residual_a']) < 1e-9
    assert result.diagnostics['turn_on_energy_partitions_j']['storage_load'] == pytest.approx(
        result.t_rs * P.v_0 * P.i_0)


def test_positive_quadratic_and_zero_inductance_continuity():
    """Positive quadratic and zero inductance continuity."""
    baseline = model().calc_switching_energy(P)
    nearby = model().calc_switching_energy(HalfBridgeParams(100, 10, 15, 0, 10, 1e-18, 0))
    assert nearby.i_oss_on == pytest.approx(baseline.i_oss_on, rel=1e-9)
    assert nearby.i_oss_off == pytest.approx(baseline.i_oss_off, rel=1e-9)
    assert nearby.i_0_zvs == pytest.approx(baseline.i_0_zvs, rel=1e-9)
    a = 2e-18 / (200e-9 * 10)
    b = 2 / (10 * 10) + .5
    c = (15 - 3 - 10 / 10) / 10
    assert abs(a * nearby.i_oss_on ** 2 + b * nearby.i_oss_on + c) < 1e-10


def test_charge_equivalent_consistency_rejects_mismatch_and_discloses_rounding():
    """Use one charge convention and retain an internally consistent ZVS boundary."""
    inconsistent = ChristenBielaModel(1e-9, 1e-9, 1e-9, 100e-9, TransconductanceParams(10, 0, 1, 3))
    with pytest.raises(ValueError, match='Charge-equivalent Qoss'):
        inconsistent.calc_switching_energy(P)
    rounded = ChristenBielaModel(1e-9, 1e-9, 1e-9, 201e-9, TransconductanceParams(10, 0, 1, 3))
    point = HalfBridgeParams(100, 10, 15, 0, 10, .1e-9, 0)
    result = rounded.calc_switching_energy(point)
    assert result.diagnostics['charge_consistency']['relative_difference'] == pytest.approx(.005)
    assert result.diagnostics['charge_consistency']['source_rounding_relative_tolerance'] == .01
    boundary = result.i_0_zvs
    residual = point.l_s / (2 * rounded.q_oss) * boundary ** 2 + (
        point.r_g * rounded.c_gd / (2 * rounded.c_oss)) * boundary - 3
    assert abs(residual) < 1e-12
    response = TestClient(app).post('/api/analytical/biela', json=PAYLOAD | {'q_oss': 100e-9})
    assert response.status_code == 422
    assert 'Charge-equivalent Qoss' in response.json()['detail']


def test_body_diode_tail_uses_bus_voltage_after_fall_with_nonzero_drain_inductance():
    """Integrate manuscript eq28 independently where Vds0 differs from V0."""
    # Long recovery tail exposes the V0 versus Vds0 post-fall distinction.
    rr = ReverseRecoveryParams(40e-9, 60e-9, 120e-9, 88e-9)
    p = HalfBridgeParams(100, 10, 15, 0, 10, 0, 1e-9)
    result = model(rr).calc_switching_energy(p)
    voltage = p.v_0 - result.v_ld_on
    assert voltage < p.v_0
    normalized_end = result.t_fv / rr.tau_rr
    during = quad(lambda u: voltage * u / normalized_end * result.i_rr_calc
                  * np.exp(-u) * rr.tau_rr, 0, normalized_end, epsabs=1e-18)[0]
    tail = quad(lambda u: p.v_0 * result.i_rr_calc * np.exp(-u) * rr.tau_rr,
                normalized_end, np.inf, epsabs=1e-18)[0]
    assert result.diagnostics['body_diode_energy_j'] == pytest.approx(during + tail, rel=1e-11)
    incorrect_complement = voltage * result.i_rr_calc * rr.tau_rr - result.e_rf
    assert abs(result.diagnostics['body_diode_energy_j'] - incorrect_complement) > 1e-8


def test_old_paper_like_fixture_is_not_a_verified_paper_prediction():
    """Old paper like fixture is not a verified paper prediction."""
    old = ChristenBielaModel(1080e-12, 130e-12, 14.5e-12, 86.56e-9,
                            TransconductanceParams(.1319, -.076, 3.8, 4.5))
    with pytest.raises(BielaConvergenceError, match='negative discriminant'):
        old.calc_switching_energy(HalfBridgeParams(600, 20, 20, -5, 7.1, 4e-9, 10e-9))


def test_negative_iteration_discriminant_does_not_claim_absence_of_nonlinear_root():
    """An implicit root can coexist with a failed initial fixed-point quadratic."""
    from scipy.optimize import brentq
    gm = TransconductanceParams(10, 0, .1, 3)
    m = ChristenBielaModel(1e-9, 1e-9, 1e-9, 200e-9, gm)
    p = HalfBridgeParams(100, 10, 15, 0, 10, 200e-9, 0)
    def implicit(current):
        secant = gm.calc_gm(p.i_0 - 2 * current)
        return 2 * p.l_s / (m.q_oss * p.r_g) * current ** 2 + (
            2 / (secant * p.r_g) + .5) * current + (p.v_g_on - 3 - p.i_0 / secant) / p.r_g
    root = brentq(implicit, -2, -1, xtol=1e-13)
    assert root == pytest.approx(-1.220074751950, rel=1e-11)
    assert abs(implicit(root)) < 1e-12
    with pytest.raises(BielaConvergenceError, match='existence of a self-consistent root is unknown'):
        m.calc_turn_on_energy(p)


@pytest.mark.parametrize('current', [0, -1, .5, 1.2])
def test_no_invented_zero_current_or_zvs_energy(current):
    """No invented zero current or zvs energy."""
    with pytest.raises(ValueError):
        model().calc_switching_energy(HalfBridgeParams(100, current, 15, 0, 10, 0, 0))


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_gm_current_rejected(value):
    """Invalid gm current rejected."""
    with pytest.raises(ValueError):
        TransconductanceParams(10, 0, 1, 3).calc_gm(value)


def test_gm_is_secant_not_differential_and_coverage_is_bounded():
    """Gm is secant not differential and coverage is bounded."""
    gm = TransconductanceParams(2, 0, 2, 3, 1, 20)
    assert gm.calc_gm(8) == pytest.approx(4)  # I/(Vgs-Vth), differential is 8S.
    offset = TransconductanceParams(2, .1, 2, 3)
    # At Vgs=5, I=2*(5-3)^2+.1=8.1A and secant gm=8.1/(5-3)=4.05S.
    assert offset.calc_gm(8.1) == pytest.approx(4.05)
    with pytest.raises(ValueError, match='coverage'):
        gm.calc_gm(21)


def test_nonlinear_nonzero_offset_fixture_against_direct_root_and_phase_quadrature():
    """Check iterative nonlinear gm against a separate implicit root and integration."""
    from scipy.optimize import brentq
    gm = TransconductanceParams(2, .1, 2, 3)
    p = HalfBridgeParams(100, 10, 15, -1, 10, .1e-9, .1e-9)
    m = ChristenBielaModel(1e-9, 1e-9, 1e-9, 200e-9, gm)
    r = m.calc_switching_energy(p)
    def polynomial(current, gate):
        secant = (p.i_0 - 2 * current) / np.sqrt((p.i_0 - 2 * current - .1) / 2)
        return (2 * p.l_s / (m.q_oss * p.r_g) * current ** 2
                + (2 / (secant * p.r_g) + .5) * current
                + (gate - 3 - p.i_0 / secant) / p.r_g)
    off = brentq(lambda current: polynomial(current, p.v_g_off), 0, 4)
    on = brentq(lambda current: polynomial(current, p.v_g_on), -5, 0)
    assert r.i_oss_off == pytest.approx(off, rel=1e-9)
    assert r.i_oss_on == pytest.approx(on, rel=1e-9)
    current_off = p.i_0 - 2 * off
    gm_off = current_off / np.sqrt((current_off - .1) / 2)
    plateau_off = 3 + current_off / gm_off
    rise_voltage_time = m.q_oss / off
    fall_current_time = np.log((plateau_off-p.v_g_off)/(3-p.v_g_off)) * (
        m.c_gs*p.r_g+p.l_s*gm_off)
    overvoltage = p.l_d * current_off / fall_current_time
    gm_at_load = p.i_0 / np.sqrt((p.i_0-.1)/2)
    rise_current_time = -np.log(1-p.i_0/(gm_at_load*(p.v_g_on-3))) * (
        m.c_gs*p.r_g+p.l_s*gm_at_load)
    initial_voltage = p.v_0-p.l_d*p.i_0/rise_current_time
    fall_voltage_time = -m.q_oss/on
    off_integral = quad(lambda u: u*p.v_0*current_off*rise_voltage_time, 0, 1)[0]
    off_integral += quad(lambda u: (p.v_0+overvoltage)*current_off*(1-u)*fall_current_time, 0, 1)[0]
    on_integral = quad(lambda u: initial_voltage*p.i_0*u*rise_current_time, 0, 1)[0]
    on_integral += quad(lambda u: initial_voltage*(1-u)*(p.i_0-2*on)*fall_voltage_time, 0, 1)[0]
    assert r.e_on == pytest.approx(on_integral, rel=1e-9)
    assert r.e_off == pytest.approx(off_integral, rel=1e-9)


def test_solver_nonconvergence_is_not_silently_returned():
    """Solver nonconvergence is not silently returned."""
    with pytest.raises(BielaConvergenceError, match='did not converge'):
        model()._solve_i_oss(10, 15, 10, 0, max_iter=0)


def test_corrected_recovery_extraction_requires_forward_test_current():
    """Corrected recovery extraction requires forward test current."""
    with pytest.raises(ValueError, match='i_forward_test'):
        extract_reverse_recovery_params(100e-9, 10, 2.4e9, 12e-9)
    r = extract_reverse_recovery_params(100e-9, 10, 2.4e9, 12e-9, i_forward_test=20)
    assert r.tau_rr == pytest.approx(6.716666666667e-9, rel=1e-10)
    assert r.tau_c == pytest.approx(13.669756359629e-9, rel=1e-10)
    assert r.t_m == pytest.approx(13.204949301187e-9, rel=1e-10)
    assert r.q_rr_star == pytest.approx(88e-9)
    residual = 2.4e9 * (r.tau_c-r.tau_rr) * (1-np.exp(-30/2.4e9/r.tau_c)) - 10
    assert abs(residual) < 1e-10


@pytest.mark.parametrize('qrr,qoss', [(1e-9, 2e-9), (10e-9, 0)])
def test_invalid_recovery_charge_is_not_floored(qrr, qoss):
    """Invalid recovery charge is not floored."""
    with pytest.raises(ValueError, match='must be positive'):
        extract_reverse_recovery_params(qrr, 10, 2.4e9, qoss, i_forward_test=20)


def transfer_curves():
    """Transfer curves."""
    return [ChannelCharacteristics(25, np.array([[5, 10, 20], [value, value, value]]), gate)
            for gate, value in [(4, 2.1), (5, 8.1), (6, 18.1), (7, 32.1), (8, 50.1)]]


def test_known_threshold_transfer_fit_and_coverage():
    """Known threshold transfer fit and coverage."""
    fit = fit_transconductance(transfer_curves(), v_ds=10, t_j=25, v_th=3)
    assert (fit.k_1, fit.k_2, fit.x) == pytest.approx((2, .1, 2), rel=1e-8)
    assert fit.diagnostics['rmse_a'] < 1e-8
    assert fit.diagnostics['rank'] == 3
    assert (fit.current_min, fit.current_max) == (2.1, 50.1)
    with pytest.raises(ValueError, match='coverage'):
        fit.calc_gm(100)


def test_transfer_fit_requires_explicit_common_conditions():
    """Transfer fit requires explicit common conditions."""
    for changes in ({}, {'v_ds': 30, 't_j': 25, 'v_th': 3},
                    {'v_ds': 10, 't_j': 26, 'v_th': 3}):
        with pytest.raises(ValueError):
            fit_transconductance(transfer_curves(), **changes)
    with pytest.raises(ValueError, match='four distinct'):
        fit_transconductance(transfer_curves()[:3], v_ds=10, t_j=25, v_th=3)


def test_capacitance_integration_uses_interpolated_bounds_without_extrapolation():
    """Capacitance integration uses interpolated bounds without extrapolation."""
    assert calc_charge_stored([0, 100, 200], [3e-9, 2e-9, 1e-9], 50) == pytest.approx(137.5e-9)
    assert calc_charge_equivalent_capacitance([0, 100], [3e-9, 2e-9], 50) == pytest.approx(2.75e-9)
    with pytest.raises(ValueError):
        calc_charge_stored([10, 100], [3e-9, 2e-9], 50)


def test_from_transistor_rejects_nearest_temperature_and_unstated_recovery():
    """From transistor rejects nearest temperature and unstated recovery."""
    cap = SimpleNamespace(t_j=25, graph_v_c=np.array([[0, 100], [2e-9, 2e-9]]))
    transistor = SimpleNamespace(c_iss=[cap], c_oss=[cap], c_rss=[cap],
                                 switch=SimpleNamespace(channel_data=transfer_curves()))
    with pytest.raises(ValueError, match='recovery_mode'):
        ChristenBielaModel.from_transistor(transistor, 100)
    with pytest.raises(ValueError, match='exact Tj'):
        ChristenBielaModel.from_transistor(transistor, 100, 26, recovery_mode='omit')


def test_public_imports_are_headless_and_keep_compatibility():
    """Public imports are headless and keep compatibility."""
    code = ('import sys; from transistordatabase.analytical_models import '
            'ChristenBielaModel, HalfBridgeParams, GateChargeModel, IgbtModel; '
            'assert not any(k.startswith(("PyQt", "PySide")) for k in sys.modules)')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)


def test_api_uses_same_science_and_preserves_conditions_and_provenance():
    """Api uses same science and preserves conditions and provenance."""
    response = TestClient(app).post('/api/analytical/biela', json=PAYLOAD)
    assert response.status_code == 200, response.text
    output = response.json()
    expected = model().calc_switching_energy(P).to_dict()
    expected['diagnostics']['parameter_reference'] = {
        'v_ref_v': 100, 't_j_c': 25, 'source': PAYLOAD['source']}
    assert output['results'] == expected
    assert output['parameter_inputs_si']['source'] == PAYLOAD['source']
    assert output['parameter_inputs_si']['parameter_t_j'] == 25
    assert output['mode'] == 'estimate'
    assert any('omitted' in a for a in output['assumptions'])


def test_api_sweep_keeps_successes_and_explicit_zero_current_failure():
    """Api sweep keeps successes and explicit zero current failure."""
    response = TestClient(app).post('/api/analytical/biela', json=PAYLOAD | dict(i_min=0, i_max=20, i_points=11))
    assert response.status_code == 200, response.text
    sweep = response.json()['sweep']
    assert sweep['points'][0]['status'] == 'unsupported'
    assert 'I0' in sweep['points'][0]['error']
    assert 0 < sweep['supported_points'] < sweep['requested_points']
    assert any(p.get('results', {}).get('e_total', 0) > 0 for p in sweep['points'])


def test_api_retains_numerical_failure_separately_from_unsupported_domain(monkeypatch):
    """Transport a solver failure without declaring the configuration invalid."""
    original = ChristenBielaModel.calc_switching_energy
    def selectively_fail(self, params):
        if params.i_0 == 5:
            raise BielaConvergenceError('Iteration did not converge; existence of a solution is unknown')
        return original(self, params)
    monkeypatch.setattr(ChristenBielaModel, 'calc_switching_energy', selectively_fail)
    response = TestClient(app).post('/api/analytical/biela', json=PAYLOAD | dict(i_min=0, i_max=10, i_points=3))
    assert response.status_code == 200
    points = response.json()['sweep']['points']
    assert [p['status'] for p in points] == ['unsupported', 'numerical_failure', 'supported']
    assert 'unknown' in points[1]['error']


def test_flat_transfer_data_is_not_silently_a_valid_fit():
    """Reject a transfer record that contains no current dependence on gate voltage."""
    curves = [ChannelCharacteristics(25, np.array([[5, 10], [10, 10]]), gate)
              for gate in [4, 5, 6, 7]]
    with pytest.raises(ValueError, match='increase strictly'):
        fit_transconductance(curves, v_ds=10, t_j=25, v_th=3)


@pytest.mark.parametrize('changes,message', [
    ({'parameter_t_j': 100}, 'temperature'), ({'q_oss_v_ref': 200}, 'reference voltage'),
    ({'gm_i_max': 1}, 'coverage'), ({'i_min': 0}, 'both sweep'),
    ({'recovery_mode': 'charge_control'}, 'requires'), ({'tau_rr': 1e-9}, 'omitted'),
    ({'v_driver_on': 3}, 'Voff < Vth < Von'), ({'i_load': 100}, 'coverage'),
    ({'source': '   '}, 'source'), ({'i_min': 10, 'i_max': 5}, 'minimum'),
])
def test_api_actionable_domain_failures(changes, message):
    """Api actionable domain failures."""
    response = TestClient(app).post('/api/analytical/biela', json=PAYLOAD | changes)
    assert response.status_code == 422
    assert message in response.json()['detail']


@pytest.mark.parametrize('key', [key for key, v in PAYLOAD.items() if isinstance(v, (int, float))])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_api_nonfinite_raw_json_does_not_crash_error_serialization(key, value):
    """Api nonfinite raw json does not crash error serialization."""
    response = TestClient(app).post('/api/analytical/biela', content=json.dumps(PAYLOAD | {key: value}),
                                     headers={'content-type': 'application/json'})
    assert response.status_code == 422


def test_api_rejects_obsolete_simplified_contract():
    """Api rejects obsolete simplified contract."""
    response = TestClient(app).post('/api/analytical/biela', json={'c_oss': 1e-9, 'q_g': 1e-8, 'v_plateau': 5})
    assert response.status_code == 422


def test_extracted_model_retains_voltage_temperature_source_and_fit_evidence():
    """Prevent a model extracted at one bus from silently reusing its old charge."""
    def cap(value):
        return SimpleNamespace(t_j=25, graph_v_c=np.array([[0, 100], [value, value]]))
    transistor = SimpleNamespace(c_iss=[cap(3e-9)], c_oss=[cap(2e-9)], c_rss=[cap(1e-9)],
                                 metadata=SimpleNamespace(name='Condition fixture'),
                                 switch=SimpleNamespace(channel_data=transfer_curves()))
    extracted = ChristenBielaModel.from_transistor(transistor, 100, 25, v_ds_fit=10,
                                                   v_th=3, recovery_mode='omit')
    result = extracted.calc_switching_energy(P)
    assert result.diagnostics['parameter_reference']['v_ref_v'] == 100
    assert result.diagnostics['parameter_reference']['t_j_c'] == 25
    assert 'Condition fixture' in result.diagnostics['parameter_reference']['source']
    assert result.diagnostics['transfer_fit']['points'] == 5
    with pytest.raises(ValueError, match='fixed reference bus'):
        extracted.calc_switching_energy(HalfBridgeParams(110, 10, 15, 0, 10, 0, 0))
    with pytest.raises(ValueError, match='temperature'):
        extracted.calc_switching_energy(HalfBridgeParams(100, 10, 15, 0, 10, 0, 0, 26))
