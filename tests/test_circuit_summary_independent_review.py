"""Independent energy, physical allocation, and input expansion checks."""
from copy import deepcopy

import pytest

from transistordatabase import circuit_summary
from transistordatabase.waveform_losses import WaveformResourceLimitError


def energy(joules):
    """Declare a constant, signed-direction switching-energy table."""
    conditions = {"v_supply_v": 200, "t_j_c": None}
    return {"mode": "recorded", "current_a": [0, 10], "energy_j": [joules, joules],
            "recorded_conditions": conditions, "requested_conditions": dict(conditions),
            "source": "Independent constant energy"}


def fixture(operation="held-envelope-estimate"):
    """Define two physical junctions with independent paths to a 10 °C sink."""
    drop = {"mode": "linear-drop-estimate", "selected": True, "offset_v": 1,
            "resistance_ohm": 0, "source": "Independent constant drop", "conditions": {}}
    model = {"conduction": {"positive": drop, "negative": deepcopy(drop)},
             "switching": {"turn_on": {"positive": energy(.25), "negative": energy(.5)},
                           "turn_off": {"positive": energy(0), "negative": energy(0)}},
             "events": {"mode": "gate", "threshold_v": .5}}
    binding = {"role": "leg", "members": ["Q1"], "multiplicity": 1,
               "conduction_current": {"signal": "i"}, "switching_current": {"signal": "ic"},
               "gate_signal": "g", "bus_voltage": {"value": 200},
               "duty": {"value": .5}, "frequency_hz": {"value": 2}}
    owners = {name: {direction: {"die": "channel", "junction_id": "Q1.ch"}
                     for direction in ("positive", "negative", "zero")}
              for name in ("conduction", "turn_on", "turn_off")}
    owners["conduction"]["positive"] = {"die": "diode", "junction_id": "Q1.d"}
    owners["turn_on"]["negative"] = {"die": "diode", "junction_id": "Q1.d"}
    junctions = [{"id": jid, "rtheta_k_w": resistance, "source": "Independent path", "conditions": {}}
                 for jid, resistance in (("Q1.ch", 8), ("Q1.d", 16))]
    if operation == "waveform":
        signals = {"i": 0, "ic": [0, -1, 1], "g": [0, 1, 1]}
    else:
        signals = {"i": [2, -4], "ic": [-1, 1]}
    return {"configuration_id": "independent", "scenario_id": "two-junctions",
            "boundary": {"kind": "ideal-fixed-sink", "temperature_c": 10,
                         "source": "Independent cold sink", "conditions": {}},
            "members": [{"id": "Q1", "role": "leg", "layout_complete": True,
                         "loss_owners": owners, "junctions": junctions}],
            "acknowledge_known_only": False,
            "converter": {"operation": operation, "preset": "custom", "time_s": [0, 1, 4],
                          "signals": signals, "bindings": [binding], "role_models": {"leg": model},
                          "assumptions": {"envelope_selected": True, "symmetry_selected": True}}}


def test_signed_held_two_networks_unequal_intervals():
    """Integrate 1 s then 3 s, using the other signal for turn-on energy."""
    result = circuit_summary.analyze_circuit_summary(fixture())
    device = result["members"][0]
    assert device["components"]["conduction"]["energy_j"] == pytest.approx(7)
    assert device["components"]["turn_on"]["energy_j"] == pytest.approx(2.5)
    assert result["electrical"]["total_energy_j"] == pytest.approx(9.5)
    assert device["die_losses"]["channel"]["power_w"] == pytest.approx(1.875)
    assert device["die_losses"]["diode"]["power_w"] == pytest.approx(.5)
    temps = {row["id"]: row["tj_c"] for row in device["junctions"]}
    assert temps == pytest.approx({"Q1.ch": 25, "Q1.d": 18})
    assert result["maximum"]["junction_ids"] == ["Q1.ch"]
    assert "events" not in result["evidence_by_role"]["leg"]["evidence"]


def test_nonzero_zero_current_event_uses_explicit_zero_owner():
    """Assign positive E(0) to the declared zero-current heat owner."""
    request = fixture("waveform")
    request["members"][0]["loss_owners"]["turn_on"]["zero"] = {
        "die": "diode", "junction_id": "Q1.d"}
    result = circuit_summary.analyze_circuit_summary(request)
    device = result["members"][0]
    assert result["electrical"]["total_energy_j"] == pytest.approx(.25)
    assert device["die_losses"]["diode"]["power_w"] == pytest.approx(.25 / 4)
    assert result["maximum"]["junction_ids"] == ["Q1.d"]
    assert result["maximum"]["tj_c"] == pytest.approx(11)


def test_missing_path_preserves_complete_allocated_heat_and_local_temperature():
    """Keep the electrical and heat results when a junction lacks a path."""
    request = fixture()
    request["members"][0]["junctions"][1]["rtheta_k_w"] = None
    result = circuit_summary.analyze_circuit_summary(request)
    device = result["members"][0]
    channel, diode = device["junctions"]
    assert channel["tj_c"] == pytest.approx(25)
    assert diode["heat_w"] == pytest.approx(.5)
    assert diode["tj_c"] is None
    assert result["maximum"]["tj_c"] is None
    assert result["electrical"]["total_power_w"] == pytest.approx(9.5 / 4)


def test_unknown_die_does_not_hide_known_thermal_owner():
    """Keep the junction result when the die label is unknown."""
    request = fixture()
    owner = request["members"][0]["loss_owners"]["conduction"]["negative"]
    owner["die"] = None
    result = circuit_summary.analyze_circuit_summary(request)
    device = result["members"][0]
    assert device["die_losses"]["channel"]["power_w"] is None
    assert device["die_losses"]["diode"]["power_w"] is None
    assert device["die_losses"]["unallocated"]["known_power_w"] == pytest.approx(1.5)
    assert result["maximum"]["tj_c"] == pytest.approx(25)


def test_directional_event_energies_accumulate_before_conversion_to_owner_power():
    """Retain a representable owner power from two subnormal directional energies."""
    request = fixture()
    tiny = float.fromhex("0x0.0000000000001p-1022")
    converter = request["converter"]
    converter["time_s"] = [0, 1, 2]
    converter["signals"] = {"i": 0, "ic": [1, -1]}
    converter["bindings"][0]["frequency_hz"] = {"value": 1}
    converter["role_models"]["leg"]["switching"]["turn_on"] = {
        direction: energy(tiny) for direction in ("positive", "negative")}
    member = request["members"][0]
    member["loss_owners"]["turn_on"]["negative"] = {"die": "channel", "junction_id": "Q1.ch"}
    member["junctions"][0]["rtheta_k_w"] = 1e308
    request["boundary"]["temperature_c"] = 0
    result = circuit_summary.analyze_circuit_summary(request)
    assert result["electrical"]["total_power_w"] == tiny
    assert result["members"][0]["die_losses"]["channel"]["power_w"] == tiny
    assert result["members"][0]["junctions"][0]["heat_w"] == tiny
    assert result["maximum"]["tj_c"] == tiny * 1e308


def test_repeated_event_sources_are_bounded_before_electrical_evaluation(monkeypatch):
    """Reject an event source whose expansion would exceed the output budget."""
    request = fixture("waveform")
    converter = request["converter"]
    count = 2000
    converter["time_s"] = list(range(count))
    converter["signals"] = {"g": [index % 2 for index in range(count)], "i": 1, "ic": 1}
    converter["role_models"]["leg"]["switching"]["turn_on"]["positive"]["source"] = "S" * 32000

    def unexpected_calculation(_):
        pytest.fail("Repeated event source metadata exceeded the response budget before calculation")

    monkeypatch.setattr(circuit_summary.converter_losses, "analyze_converter", unexpected_calculation)
    with pytest.raises(WaveformResourceLimitError):
        circuit_summary.analyze_circuit_summary(request)


def test_omission_repeated_over_members_with_no_junctions_is_bounded(monkeypatch):
    """Reject omission text repeated over 128 members before calculation."""
    request = fixture("waveform")
    template = request["members"][0]
    template["junctions"] = []
    template["layout_complete"] = False
    for component in template["loss_owners"].values():
        for owner in component.values():
            owner["junction_id"] = None
    names = [f"Q{number}" for number in range(128)]
    request["members"] = [{**template, "id": name} for name in names]
    converter = request["converter"]
    converter["bindings"][0].update(members=names, multiplicity=len(names))
    converter["role_models"]["leg"]["switching"]["turn_on"] = {
        "omit": True, "omission_reason": "S" * 96000}

    def unexpected_calculation(_):
        pytest.fail("Member omission expansion exceeds the response budget")

    monkeypatch.setattr(circuit_summary.converter_losses, "analyze_converter", unexpected_calculation)
    with pytest.raises(WaveformResourceLimitError):
        circuit_summary.analyze_circuit_summary(request)
