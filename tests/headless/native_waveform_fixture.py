"""Independent native-device and full-window waveform fixtures for issue 27."""
from copy import deepcopy

import numpy as np

from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, MeasurementEvidence,
    SwitchingLossData, ThermalProperties, Transistor, TransistorMetadata,
)


def inputs():
    """Return a core device, explicit selections, and the 3.25 W oracle."""
    device = Transistor(
        TransistorMetadata("oracle-switch", "MOSFET", "test", "test", "test"),
        ElectricalRatings(600.0, 100.0, 50.0, 175.0),
        ThermalProperties(.001, .001),
    )
    device.switch.channel_data = [ChannelCharacteristics(
        t_j=25.0, v_g=15.0, graph_v_i=np.array([[0., 2.], [0., 4.]]))]
    device.diode.channel_data = [ChannelCharacteristics(
        t_j=25.0, v_g=None, graph_v_i=np.array([[0., 4.], [0., 4.]]))]
    for term, gate, joules, family in (
        ("turn_on", 15.0, .002, "e_on_data"),
        ("turn_off", -5.0, .003, "e_off_data"),
    ):
        records = []
        for factor in (1, 2):
            records.append(SwitchingLossData(
                dataset_type="graph_i_e", t_j=25.0, v_supply=400.0, v_g=gate,
                v_g_off=None, r_g=2.0,
                graph_i_e=np.array([[0., 4.], [factor * joules] * 2]),
                evidence=MeasurementEvidence(
                    convention="device-energy", event=term,
                    method="double-pulse-integral"),
            ))
        setattr(device.switch, family, records)
    channel = {"component": "switch", "record_index": 0,
               "requested_conditions": {"t_j_c": 25.0, "v_g_v": 15.0}}
    selection = {
        "schema_version": 1, "input_temperature_K": 298.15,
        "conduction": {"mode": "recorded", "positive": channel, "negative": None},
    }
    for term, gate in (("turn_on", 15.0), ("turn_off", -5.0)):
        selector = {
            "component": "switch", "record_index": 0,
            "requested_conditions": {
                "t_j_c": 25.0, "v_supply_v": 400.0, "v_g_v": gate,
                "v_g_off_v": None, "r_g_ohm": 2.0,
                "load_inductance_h": None, "commutation_inductance_h": None,
            },
            "requested_context": {
                "commutation_device": None, "convention": "device-energy",
                "event": term, "method": "double-pulse-integral",
                "measurement_testbench": None,
            },
        }
        selection[term] = {"mode": "recorded", "positive": selector, "negative": None}
    case = {
        "schema_version": 1, "case_id": "fixture", "candidate_id": "candidate-A",
        "operating_point_id": "OP1", "position_id": "Q1",
        "waveform": {"time_s": [0., .001, .002, .003, .004],
                     "current_a": [2.] * 5, "bus_voltage_v": 400.0},
        "events": {"mode": "masks", "turn_on": [True, False, False, False],
                   "turn_off": [False, False, True, False]},
        "provenance": {"origin_kind": "declared", "source_id": "closed-form-fixture"},
    }
    return device, selection, case


def omit_switching(selection, case):
    """Declare conduction-only scope without fabricating zero switching losses."""
    for term in ("turn_on", "turn_off"):
        selection[term] = {"mode": "omitted", "reason": "No switching requested"}
    case["events"] = {"mode": "omitted", "omission_reason": "No events requested"}


def direct_request():
    """Build an engine request independently, with no adapter preparation."""
    conditions = {"t_j_c": 25.0, "v_g_v": 15.0}
    request = {
        "waveform": {"time_s": [0., .001, .002, .003, .004],
                     "current_a": [2.] * 5, "bus_voltage_v": [400.] * 5},
        "events": {"mode": "masks", "turn_on": [True, False, False, False],
                   "turn_off": [False, False, True, False]},
        "conduction": {"positive": {
            "mode": "recorded", "source": "independent-channel",
            "current_a": [0., 4.], "drop_v": [0., 2.],
            "recorded_conditions": conditions, "requested_conditions": conditions}},
        "switching": {},
    }
    for term, gate, energy in (("turn_on", 15., .002), ("turn_off", -5., .003)):
        cond = {
            "t_j_c": 25.0, "v_supply_v": 400.0, "v_g_v": gate,
            "v_g_off_v": None, "r_g_ohm": 2.0,
            "load_inductance_h": None, "commutation_inductance_h": None,
        }
        request["switching"][term] = {"positive": {
            "mode": "recorded", "source": "independent-" + term,
            "current_a": [0., 4.], "energy_j": [energy] * 2,
            "recorded_conditions": deepcopy(cond),
            "requested_conditions": deepcopy(cond),
        }}
    return request

