"""Bidirectional adapters between legacy and core transistor models.

Provides ``legacy_to_core()`` and ``core_to_legacy()`` functions that map
every field between the legacy flat-attribute ``transistor.Transistor`` and
the clean-architecture ``core.models.Transistor``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from transistordatabase.core import models as core

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(val: Any) -> np.ndarray | None:
    """Convert a value to a numpy array, or return None."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val
    return np.array(val)


def _list_or_none(val: Any) -> list | None:
    """Convert numpy array to list, pass through lists, return None for None."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ---------------------------------------------------------------------------
# Legacy → Core
# ---------------------------------------------------------------------------

def _convert_foster_legacy_to_core(
    foster,
) -> core.FosterThermalModel | None:
    """Convert a legacy FosterThermalModel to a core FosterThermalModel."""
    if foster is None:
        return None
    return core.FosterThermalModel(
        r_th_vector=foster.r_th_vector,
        r_th_total=foster.r_th_total,
        c_th_vector=foster.c_th_vector,
        c_th_total=foster.c_th_total,
        tau_vector=foster.tau_vector,
        tau_total=foster.tau_total,
        graph_t_rthjc=_np(foster.graph_t_rthjc),
    )


def _convert_channel_legacy_to_core(ch) -> core.ChannelCharacteristics:
    """Convert a legacy ChannelData to a core ChannelCharacteristics."""
    return core.ChannelCharacteristics(
        t_j=ch.t_j,
        graph_v_i=_np(ch.graph_v_i),
        v_g=getattr(ch, 'v_g', None),
    )


def _convert_switching_legacy_to_core(sed) -> core.SwitchingLossData:
    """Convert a legacy SwitchEnergyData to a core SwitchingLossData."""
    return core.SwitchingLossData(
        dataset_type=sed.dataset_type,
        t_j=sed.t_j,
        v_supply=sed.v_supply,
        v_g=sed.v_g,
        e_x=getattr(sed, 'e_x', None),
        r_g=getattr(sed, 'r_g', None),
        i_x=getattr(sed, 'i_x', None),
        v_g_off=getattr(sed, 'v_g_off', None),
        graph_i_e=_np(getattr(sed, 'graph_i_e', None)),
        graph_r_e=_np(getattr(sed, 'graph_r_e', None)),
        graph_t_e=_np(getattr(sed, 'graph_t_e', None)),
        comment=getattr(sed, 'comment', None),
        measurement_date=getattr(sed, 'measurement_date', None),
        measurement_testbench=getattr(sed, 'measurement_testbench', None),
        commutation_device=getattr(sed, 'commutation_device', None),
        load_inductance=getattr(sed, 'load_inductance', None),
        commutation_inductance=getattr(sed, 'commutation_inductance', None),
    )


def _convert_linearized_legacy_to_core(lm) -> core.LinearizedModel:
    """Convert a legacy LinearizedModel to a core LinearizedModel."""
    return core.LinearizedModel(
        t_j=lm.t_j,
        i_channel=lm.i_channel,
        r_channel=lm.r_channel,
        v0_channel=lm.v0_channel,
        v_g=getattr(lm, 'v_g', None),
    )


def _convert_gate_charge_legacy_to_core(gc) -> core.GateChargeCurve:
    """Convert a legacy GateChargeCurve to a core GateChargeCurve."""
    return core.GateChargeCurve(
        v_supply=gc.v_supply,
        t_j=gc.t_j,
        i_channel=gc.i_channel,
        i_g=getattr(gc, 'i_g', None),
        graph_q_v=_np(getattr(gc, 'graph_q_v', None)),
    )


def _convert_soa_legacy_to_core(soa) -> core.SOA:
    """Convert a legacy SOA to a core SOA."""
    return core.SOA(
        t_c=getattr(soa, 't_c', None),
        time_pulse=getattr(soa, 'time_pulse', None),
        graph_i_v=_np(getattr(soa, 'graph_i_v', None)),
    )


def _convert_temp_resist_legacy_to_core(tr) -> core.TemperatureDependResistance:
    """Convert a legacy TemperatureDependResistance to core."""
    return core.TemperatureDependResistance(
        i_channel=tr.i_channel,
        v_g=tr.v_g,
        dataset_type=tr.dataset_type,
        graph_t_r=_np(getattr(tr, 'graph_t_r', None)),
        r_channel_nominal=getattr(tr, 'r_channel_nominal', None),
    )


def _convert_raw_meas_legacy_to_core(rm) -> core.RawMeasurementData:
    """Convert a legacy RawMeasurementData to core."""
    return core.RawMeasurementData(
        dataset_type=rm.dataset_type,
        comment=getattr(rm, 'comment', None),
        measurement_date=getattr(rm, 'measurement_date', None),
        measurement_testbench=getattr(rm, 'measurement_testbench', None),
        commutation_device=getattr(rm, 'commutation_device', None),
        t_j=getattr(rm, 't_j', None),
        v_supply=getattr(rm, 'v_supply', None),
        v_g=getattr(rm, 'v_g', None),
        v_g_off=getattr(rm, 'v_g_off', None),
        r_g=getattr(rm, 'r_g', None),
        r_g_off=getattr(rm, 'r_g_off', None),
        load_inductance=getattr(rm, 'load_inductance', None),
        commutation_inductance=getattr(rm, 'commutation_inductance', None),
        dpt_on_vds=getattr(rm, 'dpt_on_vds', None),
        dpt_on_id=getattr(rm, 'dpt_on_id', None),
        dpt_off_vds=getattr(rm, 'dpt_off_vds', None),
        dpt_off_id=getattr(rm, 'dpt_off_id', None),
    )


def _convert_vdc_legacy_to_core(vdc) -> core.VoltageDependentCapacitance:
    """Convert a legacy VoltageDependentCapacitance to core."""
    return core.VoltageDependentCapacitance(
        t_j=vdc.t_j,
        graph_v_c=_np(getattr(vdc, 'graph_v_c', None)),
    )


def _convert_eoc_legacy_to_core(eoc) -> core.EffectiveOutputCapacitance | None:
    """Convert a legacy EffectiveOutputCapacitance to core."""
    if eoc is None:
        return None
    return core.EffectiveOutputCapacitance(
        c_o=eoc.c_o,
        v_gs=eoc.v_gs,
        v_ds=eoc.v_ds,
    )


def legacy_to_core(legacy) -> core.Transistor:
    """Convert a legacy ``transistor.Transistor`` to a ``core.models.Transistor``.

    :param legacy: Legacy Transistor object.
    :return: Core Transistor object.
    """
    metadata = core.TransistorMetadata(
        name=legacy.name,
        type=legacy.type,
        author=legacy.author,
        manufacturer=legacy.manufacturer,
        housing_type=legacy.housing_type,
        comment=getattr(legacy, 'comment', None),
        datasheet_hyperlink=getattr(legacy, 'datasheet_hyperlink', None),
        datasheet_date=getattr(legacy, 'datasheet_date', None),
        datasheet_version=getattr(legacy, 'datasheet_version', None),
    )

    electrical = core.ElectricalRatings(
        v_abs_max=legacy.v_abs_max,
        i_abs_max=legacy.i_abs_max,
        i_cont=legacy.i_cont,
        t_j_max=legacy.switch.t_j_max,
    )

    thermal = core.ThermalProperties(
        housing_area=legacy.housing_area,
        cooling_area=legacy.cooling_area,
        r_th_cs=getattr(legacy, 'r_th_cs', None),
        r_th_switch_cs=getattr(legacy, 'r_th_switch_cs', None),
        r_th_diode_cs=getattr(legacy, 'r_th_diode_cs', None),
        t_c_max=getattr(legacy, 't_c_max', None),
    )

    t = core.Transistor(metadata=metadata, electrical=electrical, thermal=thermal)

    # --- Switch ---
    sw = legacy.switch
    t.switch.channel_data = [_convert_channel_legacy_to_core(c) for c in sw.channel]
    t.switch.e_on_data = [_convert_switching_legacy_to_core(e) for e in sw.e_on]
    t.switch.e_off_data = [_convert_switching_legacy_to_core(e) for e in sw.e_off]
    t.switch.thermal_foster = _convert_foster_legacy_to_core(sw.thermal_foster)
    t.switch.gate_charge_curves = [
        _convert_gate_charge_legacy_to_core(gc) for gc in getattr(sw, 'charge_curve', []) or []
    ]
    t.switch.soa = [_convert_soa_legacy_to_core(s) for s in getattr(sw, 'soa', []) or []]
    t.switch.r_channel_temp = [
        _convert_temp_resist_legacy_to_core(tr) for tr in getattr(sw, 'r_channel_th', []) or []
    ]
    t.switch.linearized_model = [
        _convert_linearized_legacy_to_core(lm) for lm in getattr(sw, 'linearized_switch', []) or []
    ]
    t.switch.raw_measurement_data = []

    # --- Diode ---
    di = legacy.diode
    t.diode.channel_data = [_convert_channel_legacy_to_core(c) for c in di.channel]
    t.diode.e_rr_data = [_convert_switching_legacy_to_core(e) for e in di.e_rr]
    t.diode.thermal_foster = _convert_foster_legacy_to_core(di.thermal_foster)
    t.diode.soa = [_convert_soa_legacy_to_core(s) for s in getattr(di, 'soa', []) or []]
    t.diode.linearized_model = [
        _convert_linearized_legacy_to_core(lm) for lm in getattr(di, 'linearized_diode', []) or []
    ]
    t.diode.raw_measurement_data = []

    # --- Capacitances ---
    t.c_oss = [_convert_vdc_legacy_to_core(c) for c in (legacy.c_oss or [])]
    t.c_iss = [_convert_vdc_legacy_to_core(c) for c in (legacy.c_iss or [])]
    t.c_rss = [_convert_vdc_legacy_to_core(c) for c in (legacy.c_rss or [])]
    t.c_oss_er = _convert_eoc_legacy_to_core(getattr(legacy, 'c_oss_er', None))
    t.c_oss_tr = _convert_eoc_legacy_to_core(getattr(legacy, 'c_oss_tr', None))

    return t


# ---------------------------------------------------------------------------
# Core → Legacy (dict form suitable for legacy Transistor constructor)
# ---------------------------------------------------------------------------

def _convert_foster_core_to_dict(foster: core.FosterThermalModel | None) -> dict:
    """Convert a core FosterThermalModel to a dict for legacy construction.

    Returns a valid default dict when *foster* is None, because the legacy
    Switch/Diode constructors require ``thermal_foster`` to be a dict with
    at least ``r_th_total``.
    """
    if foster is None:
        return {
            'r_th_vector': None, 'r_th_total': 0,
            'c_th_vector': None, 'c_th_total': None,
            'tau_vector': None, 'tau_total': None,
            'graph_t_rthjc': None,
        }
    return {
        'r_th_vector': foster.r_th_vector,
        'r_th_total': foster.r_th_total,
        'c_th_vector': foster.c_th_vector,
        'c_th_total': foster.c_th_total,
        'tau_vector': foster.tau_vector,
        'tau_total': foster.tau_total,
        'graph_t_rthjc': _list_or_none(foster.graph_t_rthjc),
    }


def _convert_channel_core_to_dict(ch: core.ChannelCharacteristics) -> dict:
    """Convert a core ChannelCharacteristics to dict for legacy ChannelData."""
    return {
        't_j': ch.t_j,
        'graph_v_i': _np(ch.graph_v_i),
        'v_g': ch.v_g,
    }


def _convert_switching_core_to_dict(sld: core.SwitchingLossData) -> dict:
    """Convert a core SwitchingLossData to dict for legacy SwitchEnergyData."""
    d: dict[str, Any] = {
        'dataset_type': sld.dataset_type,
        't_j': sld.t_j,
        'v_supply': sld.v_supply,
        'v_g': sld.v_g,
        'v_g_off': sld.v_g_off,
        'comment': sld.comment,
        'measurement_date': sld.measurement_date,
        'measurement_testbench': sld.measurement_testbench,
        'commutation_device': sld.commutation_device,
        'load_inductance': sld.load_inductance,
        'commutation_inductance': sld.commutation_inductance,
    }
    if sld.dataset_type == 'single':
        d['e_x'] = sld.e_x
        d['r_g'] = sld.r_g
        d['i_x'] = sld.i_x
    elif sld.dataset_type == 'graph_i_e':
        d['r_g'] = sld.r_g
        d['graph_i_e'] = _np(sld.graph_i_e)
    elif sld.dataset_type == 'graph_r_e':
        d['i_x'] = sld.i_x
        d['graph_r_e'] = _np(sld.graph_r_e)
    elif sld.dataset_type == 'graph_t_e':
        d['r_g'] = sld.r_g
        d['i_x'] = sld.i_x
        d['graph_t_e'] = _np(sld.graph_t_e)
    return d


def _convert_linearized_core_to_dict(lm: core.LinearizedModel) -> dict:
    """Convert a core LinearizedModel to dict for legacy construction."""
    return {
        't_j': lm.t_j,
        'v_g': lm.v_g,
        'i_channel': lm.i_channel,
        'r_channel': lm.r_channel,
        'v0_channel': lm.v0_channel,
    }


def _convert_gate_charge_core_to_dict(gc: core.GateChargeCurve) -> dict:
    """Convert a core GateChargeCurve to dict for legacy construction."""
    return {
        'v_supply': gc.v_supply,
        't_j': gc.t_j,
        'i_channel': gc.i_channel,
        'i_g': gc.i_g,
        'graph_q_v': _np(gc.graph_q_v),
    }


def _convert_soa_core_to_dict(soa: core.SOA) -> dict:
    """Convert a core SOA to dict for legacy construction."""
    return {
        't_c': soa.t_c,
        'time_pulse': soa.time_pulse,
        'graph_i_v': _np(soa.graph_i_v),
    }


def _convert_temp_resist_core_to_dict(tr: core.TemperatureDependResistance) -> dict:
    """Convert a core TemperatureDependResistance to dict for legacy."""
    return {
        'i_channel': tr.i_channel,
        'v_g': tr.v_g,
        'dataset_type': tr.dataset_type,
        'graph_t_r': _np(tr.graph_t_r),
        'r_channel_nominal': tr.r_channel_nominal,
    }


def _convert_vdc_core_to_dict(vdc: core.VoltageDependentCapacitance) -> dict:
    """Convert a core VoltageDependentCapacitance to dict for legacy."""
    return {
        't_j': vdc.t_j,
        'graph_v_c': _np(vdc.graph_v_c),
    }


def _convert_eoc_core_to_dict(eoc: core.EffectiveOutputCapacitance | None) -> dict | None:
    """Convert a core EffectiveOutputCapacitance to dict for legacy."""
    if eoc is None:
        return None
    return {
        'c_o': eoc.c_o,
        'v_gs': eoc.v_gs,
        'v_ds': eoc.v_ds,
    }


def core_to_legacy_dicts(
    transistor: core.Transistor,
) -> tuple[dict, dict, dict]:
    """Convert a core Transistor to the 3 dicts needed by the legacy constructor.

    :param transistor: Core Transistor object.
    :return: (transistor_args, switch_args, diode_args) ready for
        ``legacy.Transistor(transistor_args, switch_args, diode_args, ...)``.
    """
    m = transistor.metadata
    e = transistor.electrical_ratings
    th = transistor.thermal_properties
    sw = transistor.switch
    di = transistor.diode

    transistor_args: dict[str, Any] = {
        'name': m.name,
        'type': m.type,
        'author': m.author,
        'manufacturer': m.manufacturer,
        'housing_type': m.housing_type,
        'comment': m.comment,
        'datasheet_hyperlink': m.datasheet_hyperlink or '',
        'datasheet_date': m.datasheet_date,
        'datasheet_version': m.datasheet_version,
        'v_abs_max': e.v_abs_max,
        'i_abs_max': e.i_abs_max,
        'i_cont': e.i_cont,
        'housing_area': th.housing_area,
        'cooling_area': th.cooling_area,
        'r_th_cs': th.r_th_cs,
        'r_th_switch_cs': th.r_th_switch_cs,
        'r_th_diode_cs': th.r_th_diode_cs,
        't_c_max': th.t_c_max,
        # Fields that don't have a core equivalent - set to None
        'technology': None,
        'template_version': None,
        'template_date': None,
        'creation_date': None,
        'last_modified': None,
        'r_g_int': 0,
        'r_g_on_recommended': None,
        'r_g_off_recommended': None,
        'c_oss_fix': None,
        'c_iss_fix': None,
        'c_rss_fix': None,
        'graph_v_ecoss': None,
        'raw_measurement_data': [],
        'c_oss': [_convert_vdc_core_to_dict(c) for c in transistor.c_oss],
        'c_iss': [_convert_vdc_core_to_dict(c) for c in transistor.c_iss],
        'c_rss': [_convert_vdc_core_to_dict(c) for c in transistor.c_rss],
        'c_oss_er': _convert_eoc_core_to_dict(transistor.c_oss_er),
        'c_oss_tr': _convert_eoc_core_to_dict(transistor.c_oss_tr),
    }

    switch_args: dict[str, Any] = {
        't_j_max': e.t_j_max,
        'comment': None,
        'manufacturer': None,
        'technology': None,
        'thermal_foster': _convert_foster_core_to_dict(sw.thermal_foster),
        'channel': [_convert_channel_core_to_dict(c) for c in sw.channel_data],
        'e_on': [_convert_switching_core_to_dict(e_on) for e_on in sw.e_on_data],
        'e_off': [_convert_switching_core_to_dict(e_off) for e_off in sw.e_off_data],
        'e_on_meas': [],
        'e_off_meas': [],
        'linearized_switch': [_convert_linearized_core_to_dict(lm) for lm in sw.linearized_model],
        'r_channel_th': [_convert_temp_resist_core_to_dict(tr) for tr in sw.r_channel_temp],
        'charge_curve': [_convert_gate_charge_core_to_dict(gc) for gc in sw.gate_charge_curves],
        'soa': [_convert_soa_core_to_dict(s) for s in sw.soa],
    }

    diode_args: dict[str, Any] = {
        't_j_max': getattr(di, 'metadata', {}).get('t_j_max', e.t_j_max)
            if isinstance(getattr(di, 'metadata', None), dict) else e.t_j_max,
        'comment': None,
        'manufacturer': None,
        'technology': None,
        'thermal_foster': _convert_foster_core_to_dict(di.thermal_foster),
        'channel': [_convert_channel_core_to_dict(c) for c in di.channel_data],
        'e_rr': [_convert_switching_core_to_dict(e_rr) for e_rr in di.e_rr_data],
        'linearized_diode': [_convert_linearized_core_to_dict(lm) for lm in di.linearized_model],
        'soa': [_convert_soa_core_to_dict(s) for s in di.soa],
    }

    return transistor_args, switch_args, diode_args
