"""Independent SI fixtures and actual transport for parameter operations."""
import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.condition_tables import (
    MeasuredTable, bounded_interpolate, separable_energy_estimate,
)
from transistordatabase.device_parameters import (
    decompose_capacitances, energy_current_coefficient, fit_linear_vi,
    integrate_capacitance, resistance_temperature_estimate, signed_gate_slew,
)
from transistordatabase.gui_web.backend.parameter_routes import router


def test_strict_linear_fit():
    """Strict linear fit."""
    result = fit_linear_vi([0, 5, 10, 20, 21], [999, 1.35, 1.5, 1.8, 999], 0, 21)
    assert result['v0_v'] == pytest.approx(1.2)
    assert result['resistance_ohm'] == pytest.approx(.03)
    assert result['sample_count'] == 3
    assert result['rmse_v'] < 1e-14
    assert result['r_squared'] == pytest.approx(1)
    with pytest.raises(ValueError):
        fit_linear_vi([0, 1], [1, 2], 0, 1)


@pytest.mark.parametrize('stop,charge', [(200, 110e-9), (150, 96.25e-9), (50, 43.75e-9)])
def test_piecewise_capacitance_charge(stop, charge):
    """Piecewise capacitance charge."""
    result = integrate_capacitance([0, 100, 200], [1e-9, .5e-9, .2e-9], 0, stop)
    assert result['charge_c'] == pytest.approx(charge)
    assert result['q_c'][-1] == result['charge_c']
    assert result['voltage_v'][-1] == stop


def test_exact_energy_integral_and_offset():
    """Exact energy integral and offset."""
    # C(V)=1e-9 - 5e-12 V; analytic integral over20..80.
    result = integrate_capacitance([0, 100], [1e-9, .5e-9], 20, 80)
    expected = 1e-9 * (80**2 - 20**2)/2 - 5e-12*(80**3 - 20**3)/3
    assert result['energy_j'] == pytest.approx(expected)
    assert result['q_c'][0] == result['e_j'][0] == 0


@pytest.mark.parametrize('vdrive,ig,di,dv', [(15,1.8,9e9,-18e9),(-5,-2.2,-11e9,22e9)])
def test_signed_collector_slew(vdrive, ig, di, dv):
    """Signed collector slew."""
    result = signed_gate_slew(4,20,10,5,2e-9,100e-12,vdrive)
    assert result['v_plateau_v'] == 6
    assert result['gate_current_a'] == pytest.approx(ig)
    assert result['di_dt_a_s'] == pytest.approx(di)
    assert result['dv_ce_dt_v_s'] == pytest.approx(dv)


def test_terminal_capacitances():
    """Terminal capacitances."""
    assert decompose_capacitances(2e-9, .5e-9, .1e-9) == pytest.approx(
        {'c_gs_f': 1.9e-9, 'c_ds_f': .4e-9, 'c_gd_f': .1e-9})
    with pytest.raises(ValueError):
        decompose_capacitances(1, 2, 1.1)


def test_slew_rejects_negative_implied_gate_emitter_capacitance():
    """Cres cannot exceed total input capacitance in the lumped model."""
    with pytest.raises(ValueError, match='Cres ≤ Cies'):
        signed_gate_slew(4,20,10,5,1e-9,2e-9,15)
    payload = {'operation':'gate-slew-estimate','source':'fixture','conditions':{},
               'v_th_v':4,'current_a':20,'gm_s':10,'r_g_ohm':5,
               'c_ies_f':1e-9,'c_res_f':2e-9,'v_drive_v':15,'selected':True}
    response = client.post('/api/parameters/calculate',json=payload)
    assert response.status_code == 422
    assert 'Cres ≤ Cies' in response.json()['detail']


def test_bounded_measured_tables_and_conditions():
    """Bounded measured tables and conditions."""
    table = MeasuredTable('energy', [0,10,20], [.001,.002,.004],
                          {'t_j_c':25,'r_g_ohm':None}, 'fixture')
    assert table.evaluate(5, {'t_j_c':25,'r_g_ohm':None}) == .0015
    assert table.evaluate(0, {'t_j_c':25,'r_g_ohm':None}) == .001
    for query, conditions in [(21,table.conditions), (1,{'t_j_c':26,'r_g_ohm':None}),
                              (1,{'t_j_c':25,'r_g_ohm':0}), (1,{'t_j_c':25})]:
        with pytest.raises(ValueError):
            table.evaluate(query, conditions)
    with pytest.raises(ValueError):
        table.x[0] = 3


def test_r_factor_and_explicit_estimates():
    """R factor and explicit estimates."""
    table = MeasuredTable('r_factor', [25,125], [1,1.5], {'v_g_v':15}, 'fixture')
    assert table.evaluate(75, {'v_g_v':15}, .04) == .05
    for r in [None,0,-1,math.inf]:
        with pytest.raises(ValueError):
            table.evaluate(75, {'v_g_v':15}, r)
    with pytest.raises(ValueError):
        resistance_temperature_estimate(25,.04,125,.06,150)
    estimated = resistance_temperature_estimate(25,.04,125,.06,150, selected=True,max_typ_ratio=1.2)
    assert estimated['resistance_ohm'] == pytest.approx(.065)
    assert estimated['estimated_max_ohm'] == pytest.approx(.078)
    assert estimated['extrapolated']
    with pytest.raises(ValueError):
        table.evaluate(150, {'v_g_v':15}, .04)


def test_typed_energy_coefficient():
    """Typed energy coefficient."""
    value = energy_current_coefficient(.000632,36,{'t_j_c':175,'r_g_ohm':None},'workbook Y2')
    assert value['coefficient_j_per_a'] == pytest.approx(.000632/36)
    assert value['unit'] == 'J/A'
    assert value['reference_current_a'] == 36
    assert value['conditions']['r_g_ohm'] is None
    with pytest.raises(ValueError):
        energy_current_coefficient(.01,0,{},'invalid')


def test_separable_estimate_normalization_and_residual():
    """Separable estimate normalization and residual."""
    baseline = MeasuredTable('energy',[0,10,20],[.001,.002,.004],
                             {'t_j_c':25,'v_supply_v':400},'baseline')
    sweep = MeasuredTable('energy-condition',[25,125],[.0022,.0033],
                          {'current_a':10,'v_supply_v':400},'temperature sweep')
    result = separable_energy_estimate(baseline,5,{'t_j_c':75,'v_supply_v':400},
                                      {'t_j_c':sweep}, selected=True)
    assert result['energy_j'] == pytest.approx(.0015*1.25)
    assert result['slice_diagnostics'][0]['baseline_residual_j'] == pytest.approx(.0002)
    for conditions in [{'t_j_c':150,'v_supply_v':400}, {'t_j_c':75,'v_supply_v':500}]:
        with pytest.raises(ValueError):
            separable_energy_estimate(baseline,5,conditions,{'t_j_c':sweep},selected=True)
    with pytest.raises(ValueError):
        separable_energy_estimate(baseline,5,baseline.conditions,{})


def test_separable_temperature_voltage_and_resistance_slices_are_independent_estimate():
    """Retain all reference conditions and reject a sweep at a mismatched baseline."""
    conditions = {'t_j_c':25,'v_supply_v':400,'r_g_ohm':5}
    baseline = MeasuredTable('energy',[0,10,20],[.001,.002,.004],conditions,'baseline')
    slices = {
        't_j_c': MeasuredTable('energy-condition',[25,125],[.002,.003],
                              {'current_a':10,'v_supply_v':400,'r_g_ohm':5},'T sweep'),
        'v_supply_v': MeasuredTable('energy-condition',[200,400,800],[.001,.002,.004],
                                   {'current_a':10,'t_j_c':25,'r_g_ohm':5},'V sweep'),
        'r_g_ohm': MeasuredTable('energy-condition',[2,5,10],[.001,.002,.004],
                                {'current_a':10,'t_j_c':25,'v_supply_v':400},'Rg sweep')}
    requested = {'t_j_c':75,'v_supply_v':600,'r_g_ohm':7.5}
    result = separable_energy_estimate(baseline,5,requested,slices,selected=True)
    assert result['energy_j'] == pytest.approx(.0015*1.25*1.5*1.5)
    assert len(result['slice_diagnostics']) == 3
    assert result['baseline_conditions'] == conditions
    slices['r_g_ohm'] = MeasuredTable('energy-condition',[2,5,10],[.001,.002,.004],
                                     {'current_a':10,'t_j_c':125,'v_supply_v':400},'wrong T')
    with pytest.raises(ValueError, match='conditions do not match'):
        separable_energy_estimate(baseline,5,requested,slices,selected=True)


@pytest.mark.parametrize('x,y,q', [([0,0],[1,2],0),([0,1],[1,math.inf],0),
                                 ([0,1],[1,2],-1),([0,1],[1,2],math.nan),
                                 ([0,1],[1],0)])
def test_invalid_bounded_data(x,y,q):
    """Invalid bounded data."""
    with pytest.raises(ValueError):
        bounded_interpolate(x,y,q)


@pytest.mark.parametrize('args', [([0,100],[1,2],0,101),([0,100],[1,-2],0,100),
                                 ([0,100],[1,2],20,10)])
def test_invalid_capacitance_integration(args):
    """Invalid capacitance integration."""
    with pytest.raises(ValueError):
        integrate_capacitance(*args)


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_api_roundtrip_and_si_plot():
    """Api roundtrip and si plot."""
    payload = {'operation':'capacitance-integration','source':'fixture',
               'conditions':{'t_j_c':25},'voltage_v':[0,100,200],
               'capacitance_f':[1e-9,.5e-9,.2e-9],'start_v':0,'end_v':150}
    response = client.post('/api/parameters/calculate',json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['inputs'] == payload
    assert data['result']['charge_c'] == pytest.approx(96.25e-9)
    assert data['plots'][0]['x'] == [0,100,150]


def test_api_unknown_conditions_and_estimate_selection():
    """Api unknown conditions and estimate selection."""
    table = {'kind':'energy','x':[0,10,20],'y':[.001,.002,.004],
             'conditions':{'t_j_c':25,'r_g_ohm':None},'source':'fixture'}
    payload = {'operation':'measured-table','table':table,'query':5,
               'conditions':table['conditions']}
    assert client.post('/api/parameters/calculate',json=payload).json()['result']['value'] == .0015
    payload['conditions'] = {'t_j_c':25,'r_g_ohm':0}
    assert client.post('/api/parameters/calculate',json=payload).status_code == 422
    payload = {'operation':'gate-slew-estimate','source':'fixture','conditions':{},
               'v_th_v':4,'current_a':20,'gm_s':10,'r_g_ohm':5,
               'c_ies_f':2e-9,'c_res_f':1e-10,'v_drive_v':15}
    assert client.post('/api/parameters/calculate',json=payload).status_code == 422
    payload['selected'] = True
    assert client.post('/api/parameters/calculate',json=payload).status_code == 200
    payload['current_a'] = None
    assert client.post('/api/parameters/calculate',json=payload).status_code == 422


def test_api_rejects_extra_and_nonfinite():
    """Api rejects extra and nonfinite."""
    payload = {'operation':'terminal-capacitances','source':'fixture','conditions':{},
               'c_iss_f':2,'c_oss_f':1,'c_rss_f':.1,'unexpected':True}
    assert client.post('/api/parameters/calculate',json=payload).status_code == 422
    payload.pop('unexpected')
    payload['c_iss_f'] = 'NaN'
    assert client.post('/api/parameters/calculate',json=payload).status_code == 422


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity'])
def test_api_raw_nonfinite_tokens_are_actionable_422(token):
    """Reject nonstandard raw JSON numbers without serializing them into errors."""
    raw = ('{"operation":"terminal-capacitances","source":"fixture",'
           '"conditions":{},"c_iss_f":TOKEN,"c_oss_f":1,"c_rss_f":0.1}')
    response = client.post('/api/parameters/calculate', content=raw.replace('TOKEN',token),
                           headers={'Content-Type':'application/json'})
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'Input should be a finite number'


def test_api_resistance_factor_result_has_physical_units_and_reference():
    """Do not export converted ohms with an unlabeled dimensionless factor."""
    payload = {'operation':'measured-table','table':{'kind':'r_factor',
               'x':[25,125],'y':[1,1.5],'conditions':{'v_g_v':15},'source':'fixture'},
               'query':75,'conditions':{'v_g_v':15},'r_nom_ohm':.04}
    response = client.post('/api/parameters/calculate',json=payload)
    assert response.status_code == 200
    result = response.json()['result']
    assert result['value'] == .05
    assert result['unit'] == 'ohm'
    assert result['query_unit'] == 'degC'
    assert result['factor'] == 1.25
    assert result['r_nom_ohm'] == .04


def test_api_whitespace_source_is_rejected():
    """Every operation requires nonblank source provenance."""
    payload = {'operation':'terminal-capacitances','source':'  ','conditions':{},
               'c_iss_f':2,'c_oss_f':1,'c_rss_f':.1}
    response = client.post('/api/parameters/calculate',json=payload)
    assert response.status_code == 422
    assert 'nonblank source' in response.text
