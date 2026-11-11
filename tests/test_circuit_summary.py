"""Independent physical heat ownership, null coverage and transport checks."""
from copy import deepcopy
import json
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.circuit_summary import analyze_circuit_summary
from transistordatabase.gui_web.backend.converter_routes import router
from transistordatabase.thermal import steady_junction
from transistordatabase.waveform_losses import WaveformResourceLimitError


def fixture():
    """Two independent 1 A, 1 V, 1 s positions. Each gives 1 W and 35 C."""
    drop = {"mode": "linear-drop-estimate", "selected": True, "offset_v": 1,
            "resistance_ohm": 0, "source": "Independent 1V estimate", "conditions": {}}
    energy = {"mode": "recorded", "current_a": [0, 10], "energy_j": [0.5, 0.5],
              "source": "Independent constant energy", "recorded_conditions": {}, "requested_conditions": {}}
    model = {"conduction": {"positive": drop, "negative": deepcopy(drop)},
             "switching": {name: {"positive": deepcopy(energy), "negative": deepcopy(energy)}
                           for name in ("turn_on", "turn_off")},
             "events": {"mode": "gate", "threshold_v": 0.5}}
    converter = {"preset": "custom", "operation": "waveform", "time_s": [0, 1],
                 "signals": {"i": [1, 1], "gate": [0, 0]},
                 "bindings": [{"role": "pair", "members": ["Q1", "Q2"], "multiplicity": 2,
                               "conduction_current": {"signal": "i"},
                               "switching_current": {"signal": "i"}, "gate_signal": "gate"}],
                 "role_models": {"pair": model}, "assumptions": {"symmetry_selected": True}}
    members = []
    for identifier in ("Q1", "Q2"):
        jid = f"{identifier}.junction"
        owners = {name: {d: {"die": "channel", "junction_id": jid}
                         for d in ("positive", "negative", "zero")}
                  for name in ("conduction", "turn_on", "turn_off")}
        members.append({"id": identifier, "role": "pair", "layout_complete": True,
                        "loss_owners": owners,
                        "junctions": [{"id": jid, "rtheta_k_w": 10,
                                       "source": "Independent fixed-sink estimate", "conditions": {"t_j_c": None}}]})
    return {"configuration_id": "independent-two-position", "scenario_id": "one-second",
            "converter": converter,
            "boundary": {"kind": "ideal-fixed-sink", "temperature_c": 25,
                         "source": "Declared ideal fixed sink", "conditions": {}},
            "members": members, "acknowledge_known_only": False,
            "provenance": {"source": "Hand reference"}}


def test_two_positions_count_once_and_independent_sink_paths():
    """Count each physical position once and retain both maximum ties."""
    request = fixture()
    original = deepcopy(request)
    from transistordatabase import converter_losses
    with patch.object(converter_losses, "analyze_converter", wraps=converter_losses.analyze_converter) as calculator:
        result = analyze_circuit_summary(request)
    assert calculator.call_count == 1
    assert request == original
    assert result["electrical"]["total_energy_j"] == 2
    assert result["electrical"]["total_power_w"] == 2
    assert [m["electrical"]["total_power_w"] for m in result["members"]] == [1, 1]
    assert result["maximum"]["tj_c"] == 35
    assert result["maximum"]["junction_ids"] == ["Q1.junction", "Q2.junction"]
    assert list(result["evidence_by_role"]) == ["pair"]
    assert result["iteration_status"] == "not-required-constant-heat"
    assert json.loads(result["export_json"]) == {k: v for k, v in result.items() if k not in ("export_json", "export_csv")}
    assert "Q1.junction" in result["export_csv"]


def test_omitted_diode_loss_retains_full_channel_and_known_heat_only():
    """Keep known channel heat separate from required unknown diode heat."""
    request = fixture()
    omitted = {"omit": True, "omission_reason": "Recovery energy unavailable"}
    request["converter"]["role_models"]["pair"]["switching"] = {"turn_on": omitted, "turn_off": deepcopy(omitted)}
    for member in request["members"]:
        for name in ("turn_on", "turn_off"):
            for owner in member["loss_owners"][name].values():
                owner["die"] = "diode"
    request["acknowledge_known_only"] = True
    result = analyze_circuit_summary(request)
    assert result["electrical"]["total_power_w"] is None
    member = result["members"][0]
    assert member["die_losses"]["channel"]["power_w"] == 1
    assert member["die_losses"]["diode"]["power_w"] is None
    assert member["junctions"][0]["tj_c"] is None
    assert member["junctions"][0]["known_heat_tj_c"] == 35
    assert result["maximum"]["tj_c"] is None
    assert result["maximum"]["known_heat_tj_c"] == 35


def test_negative_current_can_belong_to_channel():
    """Use declared die ownership instead of an inferred sign rule."""
    request = fixture()
    request["converter"]["signals"]["i"] = [-1, -1]
    result = analyze_circuit_summary(request)
    assert result["members"][0]["die_losses"]["channel"]["power_w"] == 1
    assert result["maximum"]["tj_c"] == 35


def test_finite_zero_current_event_uses_explicit_zero_owner():
    """Assign finite E(0) through the explicit zero-current owner."""
    request = fixture()
    request["converter"]["signals"].update(i=[0, 0], gate=[0, 1])
    for member in request["members"]:
        member["loss_owners"]["turn_on"]["positive"] = {"die": "channel", "junction_id": None}
        member["loss_owners"]["turn_on"]["zero"]["die"] = "diode"
    result = analyze_circuit_summary(request)
    assert result["members"][0]["die_losses"]["diode"]["power_w"] == 0.5
    assert result["maximum"]["tj_c"] == 30


def test_cold_complete_inventory_has_no_room_temperature_floor():
    """Use the supplied cold boundary only with complete physical inventory."""
    request = fixture()
    request["converter"]["signals"]["i"] = [0, 0]
    request["boundary"]["temperature_c"] = 10
    assert analyze_circuit_summary(request)["maximum"]["tj_c"] == 10
    request["members"][0]["layout_complete"] = False
    result = analyze_circuit_summary(request)
    assert result["maximum"]["tj_c"] is None
    assert result["members"][1]["maximum"]["tj_c"] == 10


@pytest.mark.parametrize("missing", ["rtheta_k_w", "source"])
def test_missing_thermal_path_keeps_known_heat_but_no_temperature(missing):
    """Keep electrical heat when an independent thermal path is unknown."""
    request = fixture()
    request["members"][0]["junctions"][0][missing] = None
    result = analyze_circuit_summary(request)
    junction = result["members"][0]["junctions"][0]
    assert junction["heat_w"] == 1
    assert junction["tj_c"] is None
    assert result["members"][1]["maximum"]["tj_c"] == 35
    assert result["maximum"]["tj_c"] is None


def test_sourced_ideal_zero_resistance_is_supported():
    """Accept an explicitly sourced ideal path without a positive floor."""
    request = fixture()
    for member in request["members"]:
        member["junctions"][0]["rtheta_k_w"] = 0
    assert analyze_circuit_summary(request)["maximum"]["tj_c"] == 25


@pytest.mark.parametrize("unknown", ["die", "junction_id"])
def test_die_and_junction_ownership_are_independent(unknown):
    """Track die allocation and thermal allocation as separate axes."""
    request = fixture()
    request["members"][0]["loss_owners"]["conduction"]["positive"][unknown] = None
    member = analyze_circuit_summary(request)["members"][0]
    if unknown == "die":
        assert member["die_losses"]["channel"]["power_w"] is None
        assert member["die_losses"]["unallocated"]["known_power_w"] == 1
        assert member["junctions"][0]["tj_c"] == 35
        assert member["unallocated_junction_heat_w"] == 0
    else:
        assert member["die_losses"]["channel"]["power_w"] == 1
        assert member["die_losses"]["unallocated"]["known_power_w"] == 0
        assert member["junctions"][0]["tj_c"] is None
        assert member["unallocated_junction_heat_w"] == 1


def test_all_omitted_known_subtotals_are_null():
    """Do not fabricate zero for a subtotal with no computed contribution."""
    request = fixture()
    omitted = {"omit": True, "omission_reason": "No physical data supplied"}
    request["converter"]["role_models"]["pair"] = {"conduction": deepcopy(omitted),
        "switching": {"turn_on": deepcopy(omitted), "turn_off": deepcopy(omitted)}}
    result = analyze_circuit_summary(request)
    assert result["electrical"]["known_component_power_w"] is None
    assert all(c["known_power_w"] is None for c in result["electrical"]["components"].values())
    assert result["members"][0]["electrical"]["known_component_power_w"] is None
    assert result["members"][0]["junctions"][0]["known_heat_w"] is None


@pytest.mark.parametrize("mutation", ["wrong-id", "wrong-role", "duplicate-junction", "wrong-count", "over-cap", "metadata-cap"])
def test_invalid_inventory_and_expansion_reject_before_calculator(mutation):
    """Reject invalid physical inventories and metadata before calculation."""
    request = fixture()
    if mutation == "wrong-id":
        request["members"][0]["id"] = "Q3"
    elif mutation == "wrong-role":
        request["members"][0]["role"] = "unknown"
    elif mutation == "duplicate-junction":
        request["members"][1]["junctions"][0]["id"] = "Q1.junction"
    elif mutation == "wrong-count":
        request["converter"]["bindings"][0]["multiplicity"] = 1
    elif mutation == "over-cap":
        request["converter"]["bindings"][0].update(members=[f"Q{i}" for i in range(129)], multiplicity=129)
    else:
        request["converter"]["time_s"] = list(range(10001))
        request["converter"]["role_models"]["pair"]["switching"]["turn_on"]["positive"]["source"] = "x" * 2000
    with patch("transistordatabase.converter_losses.analyze_converter") as calculator:
        with pytest.raises((ValueError, WaveformResourceLimitError)):
            analyze_circuit_summary(request)
    calculator.assert_not_called()


def test_route_preserves_nulls_and_rejects_unsafe_json_numbers():
    """Retain unknown conditions and reject unsafe raw numeric literals."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = fixture()
    response = client.post("/api/converters/circuit-summary", json=request)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["members"][0]["junctions"][0]["conditions"]["t_j_c"] is None
    assert json.loads(payload["export_json"]) == {k: v for k, v in payload.items() if k not in ("export_json", "export_csv")}
    body = json.dumps(request).replace('"temperature_c": 25', '"temperature_c": 1e-400')
    assert client.post("/api/converters/circuit-summary", content=body).status_code == 422
    assert client.post("/api/converters/circuit-summary", content=b" " * 4_000_001).status_code == 413
    request["members"][0]["loss_owners"]["conduction"]["positive"].pop("die")
    assert client.post("/api/converters/circuit-summary", json=request).status_code == 422


@pytest.mark.parametrize("power,resistance,temperature,source", [(-1, 1, 25, "s"), (1, -1, 25, "s"),
    (1, None, 25, "s"), (True, 1, 25, "s"), (1, 1, float("inf"), "s"), (1, 1, 25, " ")])
def test_steady_junction_rejects_invalid_physical_inputs(power, resistance, temperature, source):
    """Reject invalid power, paths, boundary temperatures and sources."""
    with pytest.raises(ValueError):
        steady_junction(power, resistance, temperature, source)


def test_omission_reason_expansion_without_junctions_rejects_before_calculator():
    """Bound repeated member omissions even when no junction exists."""
    request = fixture()
    identifiers = [f"Q{i}" for i in range(128)]
    request["converter"]["bindings"][0].update(members=identifiers, multiplicity=128)
    omitted = {"omit": True, "omission_reason": "U" * 100000}
    request["converter"]["role_models"]["pair"] = {"conduction": deepcopy(omitted),
        "switching": {"turn_on": deepcopy(omitted), "turn_off": deepcopy(omitted)}}
    member = request["members"][0]
    member.update(layout_complete=False, junctions=[])
    for directions in member["loss_owners"].values():
        for owner in directions.values():
            owner["junction_id"] = None
    request["members"] = [{**deepcopy(member), "id": identifier} for identifier in identifiers]
    with patch("transistordatabase.converter_losses.analyze_converter") as calculator:
        with pytest.raises(WaveformResourceLimitError):
            analyze_circuit_summary(request)
    calculator.assert_not_called()


def test_csv_uses_literal_null_and_formula_protection():
    """Keep unknown cells explicit and neutralize spreadsheet formulas."""
    request = fixture()
    request["configuration_id"] = "  =HYPERLINK(\"bad\")"
    request["members"][0]["junctions"][0]["source"] = None
    result = analyze_circuit_summary(request)
    assert result["null_csv"] == "NULL"
    assert "'  =HYPERLINK" in result["export_csv"]
    assert '"NULL"' in result["export_csv"]
    assert result["input_sha256"] in result["export_csv"]


def test_caller_source_digest_is_validated_without_invention():
    """Keep a supplied source digest and leave manual source identity unknown."""
    request = fixture()
    result = analyze_circuit_summary(request)
    assert result["input_identity"]["upstream_source_sha256"] is None
    request["provenance"]["source_sha256"] = "a" * 64
    assert analyze_circuit_summary(request)["input_identity"]["upstream_source_sha256"] == "a" * 64
    request["provenance"]["source_sha256"] = "invalid"
    with patch("transistordatabase.converter_losses.analyze_converter") as calculator:
        with pytest.raises(ValueError, match="64 hexadecimal"):
            analyze_circuit_summary(request)
    calculator.assert_not_called()
