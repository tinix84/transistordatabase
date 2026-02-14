"""Tests for bidirectional legacy ↔ core model adapters."""
from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import transistordatabase as tdb
from transistordatabase.core import adapters
from transistordatabase.core import models as core


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def legacy_transistor():
    """Build a legacy Transistor with representative data for roundtrip tests."""
    t_j = 25
    t_j_max = 175
    v_g = 15
    v_supply = 600
    r_g = 1
    i_x = 400

    graph_v_i = np.array([
        [0, 0.59, 0.67, 0.75, 0.81, 0.9, 0.96, 1.07, 1.19, 1.31],
        [0, 1e-03, 2.38, 5.71, 10, 19, 26.6, 40.9, 60.9, 82.3],
    ])
    graph_i_e = np.array([
        [0, 5.79, 15, 26, 38, 47, 56, 69, 81, 94],
        [0, 4.9e-04, 9.3e-04, 1.27e-03, 1.51e-03, 1.71e-03, 1.85e-03, 2.07e-03, 2.22e-03, 2.41e-03],
    ])
    graph_r_e = np.array([
        [0.991, 1.404, 2.079, 3.134, 4.479, 6.998],
        [1.95e-02, 1.947e-02, 1.926e-02, 1.941e-02, 1.975e-02, 2.028e-02],
    ])
    graph_t_rthjc = np.array([
        [0.001, 0.01, 0.1, 1],
        [0.03, 0.16, 0.43, 0.55],
    ])
    graph_t_r = np.array([
        [-48.6, -29.9, -16.2, -2.5, 11.2, 24.9],
        [0.897, 1.141, 1.332, 1.530, 1.753, 1.986],
    ])
    graph_q_v = np.array([
        [0.0, 3.13e-10, 5.21e-10, 7.30e-10, 9.48e-10],
        [0.0, 0.735, 1.206, 1.680, 2.172],
    ])
    graph_i_v = np.array([
        [1.2, 1.7, 2.5, 3.5, 5.1],
        [5.1, 7.3, 10.4, 14.8, 21.3],
    ])

    foster_args = {
        'r_th_vector': [1, 2, 3],
        'r_th_total': 0.5,
        'c_th_vector': [1, 2, 3],
        'c_th_total': 2,
        'tau_vector': [1, 4, 9],
        'tau_total': 1,
        'graph_t_rthjc': graph_t_rthjc,
    }

    switch_channel = {'t_j': t_j, 'graph_v_i': graph_v_i, 'v_g': v_g}
    diode_channel = {'t_j': t_j, 'graph_v_i': graph_v_i}
    switch_energy_i_e = {
        'dataset_type': 'graph_i_e', 't_j': t_j, 'v_supply': v_supply,
        'v_g': v_g, 'r_g': r_g, 'graph_i_e': graph_i_e,
    }
    switch_energy_r_e = {
        'dataset_type': 'graph_r_e', 't_j': t_j, 'v_supply': v_supply,
        'v_g': v_g, 'r_g': None, 'graph_r_e': graph_r_e, 'i_x': i_x,
    }
    switch_ron_args = {
        'i_channel': 12, 'v_g': 15, 'dataset_type': 't_factor',
        'r_channel_nominal': 67, 'graph_t_r': graph_t_r,
    }
    switch_gate_charge = {
        'i_channel': 12.3, 't_j': 25, 'v_supply': 400,
        'i_g': None, 'graph_q_v': graph_q_v,
    }
    soa_object = {
        't_c': 25, 'time_pulse': 50e-6, 'graph_i_v': graph_i_v,
    }
    c_oss_v_c = {'t_j': t_j, 'graph_v_c': np.array([[1, 2, 4, 5], [3, 4, 5, 6]])}
    c_oss_er = {'c_o': 73e-12, 'v_gs': 0, 'v_ds': 400}

    transistor_args = {
        'name': 'Test-Transistor', 'type': 'IGBT', 'author': 'Test Author',
        'comment': 'test_comment', 'manufacturer': 'Fuji Electric',
        'datasheet_hyperlink': 'https://example.com', 'datasheet_date': '2023-01-01',
        'datasheet_version': '1.0.0', 'housing_area': 367e-6, 'cooling_area': 160e-6,
        'housing_type': 'TO247', 'v_abs_max': 200, 'i_abs_max': 200, 'i_cont': 200,
        'c_oss_fix': 1, 'c_iss_fix': 1, 'c_rss_fix': 1,
        'c_oss': c_oss_v_c, 'c_iss': c_oss_v_c, 'c_rss': c_oss_v_c,
        'c_oss_er': c_oss_er, 'c_oss_tr': None,
        'graph_v_ecoss': np.array([[1, 2, 4], [3, 4, 5]]),
        'r_g_int': 10, 'r_th_cs': 0.05, 'r_th_switch_cs': 0, 'r_th_diode_cs': 0,
    }
    switch_args = {
        't_j_max': t_j_max, 'comment': 'sw_comment', 'manufacturer': 'Fuji Electric',
        'technology': 'IGBT3', 'channel': [switch_channel],
        'e_on': [switch_energy_i_e, switch_energy_r_e],
        'e_off': [switch_energy_i_e],
        'thermal_foster': foster_args,
        'r_channel_th': switch_ron_args,
        'charge_curve': switch_gate_charge,
        'soa': soa_object,
    }
    diode_args = {
        't_j_max': t_j_max, 'comment': 'di_comment', 'manufacturer': 'Fuji Electric',
        'technology': 'IGBT3', 'channel': [diode_channel],
        'e_rr': [switch_energy_i_e], 'thermal_foster': foster_args,
        'soa': soa_object,
    }

    return tdb.Transistor(
        transistor_args, switch_args, diode_args,
        possible_housing_types=['TO247'],
        possible_module_manufacturers=['Fuji Electric'],
    )


# ---------------------------------------------------------------------------
# legacy_to_core tests
# ---------------------------------------------------------------------------

class TestLegacyToCore:
    """Tests for legacy → core conversion."""

    def test_metadata_fields(self, legacy_transistor):
        """Verify metadata fields are correctly mapped."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        m = core_t.metadata

        assert m.name == 'Test-Transistor'
        assert m.type == 'IGBT'
        assert m.author == 'Test Author'
        assert m.manufacturer == 'Fuji Electric'
        assert m.housing_type == 'TO247'
        assert m.comment == 'test_comment'
        assert m.datasheet_hyperlink == 'https://example.com'
        assert m.datasheet_version == '1.0.0'

    def test_electrical_ratings(self, legacy_transistor):
        """Verify electrical ratings are correctly mapped."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        e = core_t.electrical_ratings

        assert e.v_abs_max == 200
        assert e.i_abs_max == 200
        assert e.i_cont == 200
        assert e.t_j_max == 175

    def test_thermal_properties(self, legacy_transistor):
        """Verify thermal properties are correctly mapped."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        th = core_t.thermal_properties

        assert th.housing_area == approx(367e-6)
        assert th.cooling_area == approx(160e-6)
        assert th.r_th_cs == approx(0.05)
        assert th.r_th_switch_cs == 0
        assert th.r_th_diode_cs == 0

    def test_switch_channel_data(self, legacy_transistor):
        """Verify switch channel data conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.switch.channel_data) == 1
        ch = core_t.switch.channel_data[0]
        assert ch.t_j == 25
        assert ch.v_g == 15
        assert isinstance(ch.graph_v_i, np.ndarray)
        assert ch.graph_v_i.shape == (2, 10)

    def test_switch_switching_loss_data(self, legacy_transistor):
        """Verify switch e_on and e_off data conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.switch.e_on_data) == 2
        assert len(core_t.switch.e_off_data) == 1

        e_on_ie = core_t.switch.e_on_data[0]
        assert e_on_ie.dataset_type == 'graph_i_e'
        assert e_on_ie.t_j == 25
        assert e_on_ie.v_supply == 600
        assert e_on_ie.v_g == 15
        assert e_on_ie.r_g == 1
        assert isinstance(e_on_ie.graph_i_e, np.ndarray)

        e_on_re = core_t.switch.e_on_data[1]
        assert e_on_re.dataset_type == 'graph_r_e'
        assert e_on_re.i_x == 400
        assert isinstance(e_on_re.graph_r_e, np.ndarray)

    def test_switch_foster_thermal(self, legacy_transistor):
        """Verify switch foster thermal model conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        f = core_t.switch.thermal_foster
        assert f is not None
        assert f.r_th_vector == [1, 2, 3]
        assert f.r_th_total == 0.5
        assert f.c_th_vector == [1, 2, 3]
        assert f.tau_vector == [1, 4, 9]
        assert isinstance(f.graph_t_rthjc, np.ndarray)

    def test_switch_gate_charge(self, legacy_transistor):
        """Verify switch gate charge curve conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.switch.gate_charge_curves) == 1
        gc = core_t.switch.gate_charge_curves[0]
        assert gc.v_supply == 400
        assert gc.t_j == 25
        assert gc.i_channel == approx(12.3)
        assert isinstance(gc.graph_q_v, np.ndarray)

    def test_switch_soa(self, legacy_transistor):
        """Verify switch SOA conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.switch.soa) == 1
        soa = core_t.switch.soa[0]
        assert soa.t_c == 25
        assert soa.time_pulse == approx(50e-6)
        assert isinstance(soa.graph_i_v, np.ndarray)

    def test_switch_r_channel_temp(self, legacy_transistor):
        """Verify switch temperature-dependent resistance conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.switch.r_channel_temp) == 1
        tr = core_t.switch.r_channel_temp[0]
        assert tr.i_channel == 12
        assert tr.v_g == 15
        assert tr.dataset_type == 't_factor'
        assert tr.r_channel_nominal == 67
        assert isinstance(tr.graph_t_r, np.ndarray)

    def test_diode_channel_data(self, legacy_transistor):
        """Verify diode channel data conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.diode.channel_data) == 1
        ch = core_t.diode.channel_data[0]
        assert ch.t_j == 25
        assert isinstance(ch.graph_v_i, np.ndarray)

    def test_diode_e_rr_data(self, legacy_transistor):
        """Verify diode reverse recovery data conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.diode.e_rr_data) == 1
        e_rr = core_t.diode.e_rr_data[0]
        assert e_rr.dataset_type == 'graph_i_e'
        assert e_rr.t_j == 25

    def test_diode_foster_thermal(self, legacy_transistor):
        """Verify diode foster thermal model conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        f = core_t.diode.thermal_foster
        assert f is not None
        assert f.r_th_vector == [1, 2, 3]

    def test_diode_soa(self, legacy_transistor):
        """Verify diode SOA conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.diode.soa) == 1

    def test_capacitances(self, legacy_transistor):
        """Verify voltage-dependent capacitance conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert len(core_t.c_oss) == 1
        assert len(core_t.c_iss) == 1
        assert len(core_t.c_rss) == 1
        assert core_t.c_oss[0].t_j == 25
        assert isinstance(core_t.c_oss[0].graph_v_c, np.ndarray)

    def test_effective_output_capacitance(self, legacy_transistor):
        """Verify effective output capacitance conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        assert core_t.c_oss_er is not None
        assert core_t.c_oss_er.c_o == approx(73e-12)
        assert core_t.c_oss_er.v_gs == 0
        assert core_t.c_oss_er.v_ds == 400
        assert core_t.c_oss_tr is None


# ---------------------------------------------------------------------------
# core_to_legacy_dicts tests
# ---------------------------------------------------------------------------

class TestCoreToLegacyDicts:
    """Tests for core → legacy dict conversion."""

    def test_transistor_args_keys(self, legacy_transistor):
        """Verify the transistor_args dict has all required keys."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        t_args, sw_args, di_args = adapters.core_to_legacy_dicts(core_t)

        required_keys = [
            'name', 'type', 'author', 'manufacturer', 'housing_type',
            'v_abs_max', 'i_abs_max', 'i_cont',
            'housing_area', 'cooling_area', 'r_th_cs',
        ]
        for key in required_keys:
            assert key in t_args, f"Missing key: {key}"

    def test_switch_args_keys(self, legacy_transistor):
        """Verify the switch_args dict has all required keys."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, sw_args, _ = adapters.core_to_legacy_dicts(core_t)

        assert 't_j_max' in sw_args
        assert 'channel' in sw_args
        assert 'e_on' in sw_args
        assert 'e_off' in sw_args
        assert 'thermal_foster' in sw_args

    def test_diode_args_keys(self, legacy_transistor):
        """Verify the diode_args dict has all required keys."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, _, di_args = adapters.core_to_legacy_dicts(core_t)

        assert 't_j_max' in di_args
        assert 'channel' in di_args
        assert 'e_rr' in di_args
        assert 'thermal_foster' in di_args

    def test_field_values_preserved(self, legacy_transistor):
        """Verify field values survive core → legacy dict conversion."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        t_args, sw_args, di_args = adapters.core_to_legacy_dicts(core_t)

        assert t_args['name'] == 'Test-Transistor'
        assert t_args['type'] == 'IGBT'
        assert t_args['v_abs_max'] == 200
        assert t_args['i_abs_max'] == 200
        assert t_args['i_cont'] == 200
        assert t_args['housing_area'] == approx(367e-6)
        assert t_args['r_th_cs'] == approx(0.05)

        assert sw_args['t_j_max'] == 175
        assert len(sw_args['channel']) == 1
        assert len(sw_args['e_on']) == 2
        assert len(sw_args['e_off']) == 1

        assert di_args['t_j_max'] == 175
        assert len(di_args['channel']) == 1
        assert len(di_args['e_rr']) == 1


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Tests for legacy → core → legacy roundtrip fidelity."""

    def test_metadata_roundtrip(self, legacy_transistor):
        """Metadata survives legacy → core → legacy roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        t_args, _, _ = adapters.core_to_legacy_dicts(core_t)

        assert t_args['name'] == legacy_transistor.name
        assert t_args['type'] == legacy_transistor.type
        assert t_args['author'] == legacy_transistor.author
        assert t_args['manufacturer'] == legacy_transistor.manufacturer
        assert t_args['housing_type'] == legacy_transistor.housing_type
        assert t_args['comment'] == legacy_transistor.comment

    def test_electrical_roundtrip(self, legacy_transistor):
        """Electrical ratings survive roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        t_args, _, _ = adapters.core_to_legacy_dicts(core_t)

        assert t_args['v_abs_max'] == legacy_transistor.v_abs_max
        assert t_args['i_abs_max'] == legacy_transistor.i_abs_max
        assert t_args['i_cont'] == legacy_transistor.i_cont

    def test_thermal_roundtrip(self, legacy_transistor):
        """Thermal properties survive roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        t_args, _, _ = adapters.core_to_legacy_dicts(core_t)

        assert t_args['housing_area'] == approx(legacy_transistor.housing_area)
        assert t_args['cooling_area'] == approx(legacy_transistor.cooling_area)
        assert t_args['r_th_cs'] == approx(legacy_transistor.r_th_cs)
        assert t_args['r_th_switch_cs'] == legacy_transistor.r_th_switch_cs
        assert t_args['r_th_diode_cs'] == legacy_transistor.r_th_diode_cs

    def test_switch_channel_roundtrip(self, legacy_transistor):
        """Switch channel data survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, sw_args, _ = adapters.core_to_legacy_dicts(core_t)

        orig = legacy_transistor.switch.channel[0]
        rt = sw_args['channel'][0]
        assert rt['t_j'] == orig.t_j
        assert rt['v_g'] == orig.v_g
        np.testing.assert_array_almost_equal(rt['graph_v_i'], orig.graph_v_i)

    def test_switch_e_on_roundtrip(self, legacy_transistor):
        """Switch e_on data survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, sw_args, _ = adapters.core_to_legacy_dicts(core_t)

        assert len(sw_args['e_on']) == len(legacy_transistor.switch.e_on)
        orig = legacy_transistor.switch.e_on[0]
        rt = sw_args['e_on'][0]
        assert rt['dataset_type'] == orig.dataset_type
        assert rt['t_j'] == orig.t_j
        assert rt['v_supply'] == orig.v_supply
        assert rt['v_g'] == orig.v_g
        assert rt['r_g'] == orig.r_g
        np.testing.assert_array_almost_equal(rt['graph_i_e'], orig.graph_i_e)

    def test_switch_foster_roundtrip(self, legacy_transistor):
        """Switch foster thermal model survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, sw_args, _ = adapters.core_to_legacy_dicts(core_t)

        orig = legacy_transistor.switch.thermal_foster
        rt = sw_args['thermal_foster']
        assert rt['r_th_vector'] == orig.r_th_vector
        assert rt['r_th_total'] == orig.r_th_total
        assert rt['c_th_vector'] == orig.c_th_vector
        assert rt['tau_vector'] == orig.tau_vector

    def test_switch_gate_charge_roundtrip(self, legacy_transistor):
        """Switch gate charge data survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, sw_args, _ = adapters.core_to_legacy_dicts(core_t)

        orig = legacy_transistor.switch.charge_curve[0]
        rt = sw_args['charge_curve'][0]
        assert rt['v_supply'] == orig.v_supply
        assert rt['t_j'] == orig.t_j
        assert rt['i_channel'] == approx(orig.i_channel)
        np.testing.assert_array_almost_equal(rt['graph_q_v'], orig.graph_q_v)

    def test_switch_soa_roundtrip(self, legacy_transistor):
        """Switch SOA data survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, sw_args, _ = adapters.core_to_legacy_dicts(core_t)

        orig = legacy_transistor.switch.soa[0]
        rt = sw_args['soa'][0]
        assert rt['t_c'] == orig.t_c
        assert rt['time_pulse'] == approx(orig.time_pulse)
        np.testing.assert_array_almost_equal(rt['graph_i_v'], orig.graph_i_v)

    def test_switch_r_channel_th_roundtrip(self, legacy_transistor):
        """Switch temp-dependent resistance survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, sw_args, _ = adapters.core_to_legacy_dicts(core_t)

        orig = legacy_transistor.switch.r_channel_th[0]
        rt = sw_args['r_channel_th'][0]
        assert rt['i_channel'] == orig.i_channel
        assert rt['v_g'] == orig.v_g
        assert rt['dataset_type'] == orig.dataset_type
        assert rt['r_channel_nominal'] == orig.r_channel_nominal
        np.testing.assert_array_almost_equal(rt['graph_t_r'], orig.graph_t_r)

    def test_diode_channel_roundtrip(self, legacy_transistor):
        """Diode channel data survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, _, di_args = adapters.core_to_legacy_dicts(core_t)

        orig = legacy_transistor.diode.channel[0]
        rt = di_args['channel'][0]
        assert rt['t_j'] == orig.t_j
        np.testing.assert_array_almost_equal(rt['graph_v_i'], orig.graph_v_i)

    def test_diode_e_rr_roundtrip(self, legacy_transistor):
        """Diode e_rr data survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        _, _, di_args = adapters.core_to_legacy_dicts(core_t)

        assert len(di_args['e_rr']) == len(legacy_transistor.diode.e_rr)
        orig = legacy_transistor.diode.e_rr[0]
        rt = di_args['e_rr'][0]
        assert rt['dataset_type'] == orig.dataset_type
        assert rt['t_j'] == orig.t_j

    def test_capacitance_roundtrip(self, legacy_transistor):
        """Voltage-dependent capacitances survive roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        t_args, _, _ = adapters.core_to_legacy_dicts(core_t)

        assert len(t_args['c_oss']) == len(legacy_transistor.c_oss)
        orig = legacy_transistor.c_oss[0]
        rt = t_args['c_oss'][0]
        assert rt['t_j'] == orig.t_j
        np.testing.assert_array_almost_equal(rt['graph_v_c'], orig.graph_v_c)

    def test_effective_capacitance_roundtrip(self, legacy_transistor):
        """Effective output capacitance survives roundtrip."""
        core_t = adapters.legacy_to_core(legacy_transistor)
        t_args, _, _ = adapters.core_to_legacy_dicts(core_t)

        assert t_args['c_oss_er']['c_o'] == approx(legacy_transistor.c_oss_er.c_o)
        assert t_args['c_oss_er']['v_gs'] == legacy_transistor.c_oss_er.v_gs
        assert t_args['c_oss_er']['v_ds'] == legacy_transistor.c_oss_er.v_ds
        assert t_args['c_oss_tr'] is None
