"""Algebraic physical-role aggregation and strict converter transport fixtures."""
from copy import deepcopy
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.converter_losses import analyze_converter, converter_presets
from transistordatabase import converter_losses
from transistordatabase.gui_web.backend.converter_routes import router
from transistordatabase.waveform_losses import (
    evaluate_switching_energy, prepare_switching_models,
)


def drop(offset=2, resistance=0):
    """Build a declared threshold/resistance estimate."""
    return {"mode": "linear-drop-estimate", "selected": True, "source": "Hand V-I fixture",
            "offset_v": offset, "resistance_ohm": resistance, "conditions": {}}


def energy(voltage=200, value=.001, requested=200):
    """Build independent constant-energy source slices."""
    return {"mode": "voltage-scaling-estimate", "selected": True,
            "source": f"Hand {voltage}V independent energy slice", "current_a": [0, 100],
            "energy_j": [value, value], "recorded_conditions": {"v_supply_v": voltage, "t_j_c": 25},
            "requested_conditions": {"v_supply_v": requested, "t_j_c": 25}}


def model(offset=2, resistance=0):
    """Bind both current directions to explicit fixture data."""
    conduction = {"positive": drop(offset, resistance), "negative": drop(offset, resistance)}
    return {"source": "Independent explicit role model", "conduction": conduction,
            "events": {"mode": "gate", "threshold_v": .5},
            "switching": {"turn_on": {"positive": energy(400), "negative": energy(400)},
                          "turn_off": {"positive": energy(600, .002), "negative": energy(600, .002)}}}


def envelope():
    """Build a single held physical-position request."""
    binding = {"role": "single", "members": ["s1"], "multiplicity": 1,
               "conduction_current": {"signal": "i"}, "duty": {"signal": "d"},
               "frequency_hz": {"signal": "fs"}, "bus_voltage": {"signal": "v"}}
    return {"preset": "custom", "operation": "held-envelope-estimate", "time_s": [0, 1],
            "signals": {"i": [10], "d": [.25], "fs": 100000, "v": 200},
            "bindings": [binding], "role_models": {"single": model()},
            "assumptions": {"envelope_selected": True}}


def waveform(preset="sc-dab"):
    """Build distinct conduction, commutation and gate columns."""
    bindings = next(p["bindings"] for p in converter_presets()["presets"] if p["id"] == preset)
    signals = {name: [0, 2, 9, 3, 0] for name in
               ("is1", "is2", "is4", "is6", "is8", "idp1", "ids1", "ids3")}
    signals.update(itp=[0, 2, 9, 3, 0], its=[0, -2, -9, -3, 0], vp=200, vs=200)
    for name in ("vgs1", "vgs2", "vgs4", "vgs6", "vgs8"):
        signals[name] = [0, 0, 1, 1, 0]
    return {"preset": preset, "operation": "waveform", "time_s": [0, 1e-6, 2e-6, 3e-6, 4e-6],
            "signals": signals, "role_models": {r["role"]: model() for r in bindings
                                                  if not r.get("source_switching_omission")},
            "assumptions": {"symmetry_selected": True}}


@pytest.mark.parametrize("operation", ["waveform", "held-envelope-estimate"])
def test_repeated_source_conditions_reject_before_branch_evaluation(monkeypatch, operation):
    """Bound valid repeated condition keys before electrical evidence allocation."""
    request = envelope()
    request["operation"] = operation
    request["time_s"] = list(range(17))
    count = 17 if operation == "waveform" else 16
    request["signals"].update(i=[10] * count, d=[.25] * count)
    request["bindings"][0]["switching_current"] = {"signal": "i"}
    role = request["role_models"]["single"]
    role["conduction"] = {"omit": True, "omission_reason": "No conduction model supplied"}
    if operation == "waveform":
        role["events"] = {"mode": "masks", "turn_on": [True] * 16, "turn_off": [False] * 16}
    # Conditions permit long keys even when source identifiers have a short cap.
    for spec in role["switching"].values():
        for direction in ("positive", "negative"):
            for name in ("recorded_conditions", "requested_conditions"):
                spec[direction][name]["audit_" + "x" * 4000] = None
    accepted = analyze_converter(request)
    assert accepted["totals"]["total_power_w"] is None
    assert accepted["totals"]["known_component_power_w"] > 0
    monkeypatch.setattr(converter_losses, "MAX_RESPONSE_BYTES", 200_000)

    def unexpected_branch(*args):
        pytest.fail("Source-condition expansion must reject before branch evaluation")

    monkeypatch.setattr(converter_losses, "_waveform_branch", unexpected_branch)
    monkeypatch.setattr(converter_losses, "_envelope_branch", unexpected_branch)
    with pytest.raises(converter_losses.WaveformResourceLimitError, match="evidence expansion"):
        analyze_converter(request)


def test_converter_streamed_body_limit_precedes_dto_and_calculation(monkeypatch):
    """Reject an oversized valid JSON body before DTO validation and calculation."""
    from transistordatabase.gui_web.backend import converter_routes
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    body = json.dumps(envelope()).encode("utf-8")
    body += b" " * (4_000_001 - len(body))

    def unexpected_calculation(*args):
        pytest.fail("Oversized streamed input must reject before calculation")

    monkeypatch.setattr(converter_routes, "analyze_converter", unexpected_calculation)
    response = client.post("/api/converters/analyze", content=body,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_converter_actual_response_size_has_an_independent_guard(monkeypatch):
    """Reject an oversized encoded result even if the estimate is bypassed."""
    monkeypatch.setattr(converter_losses, "converter_evidence_budget", lambda *_: 0)
    monkeypatch.setattr(converter_losses, "MAX_RESPONSE_BYTES", 500)
    with pytest.raises(converter_losses.WaveformResourceLimitError, match="response exceeds"):
        analyze_converter(envelope())


def test_held_budget_does_not_invent_waveform_plots_or_missing_switching_metadata():
    """Keep a valid 6000-row cold-switching envelope within its evidence budget."""
    request = envelope()
    request["time_s"] = list(range(6001))
    request["signals"].update(i=1, d=1, fs=0)
    request["role_models"]["single"].pop("switching")
    projected = converter_losses.converter_evidence_budget(request, request["bindings"])
    assert projected < converter_losses.MAX_RESPONSE_BYTES


def test_budget_counts_global_omission_reason_for_each_source_role(monkeypatch):
    """Bound global omission text retained four times per affected role."""
    request = waveform("sc-src-zcs")
    bindings = next(p["bindings"] for p in converter_presets()["presets"] if p["id"] == "sc-src-zcs")
    request["assumptions"]["source_omission_reason"] = "S" * 4000
    request_bytes = len(json.dumps(request, separators=(",", ":")).encode("utf-8"))
    affected = sum(bool(binding.get("source_switching_omission")) for binding in bindings)
    projected = converter_losses.converter_evidence_budget(request, bindings)
    assert projected >= 4 * request_bytes + 4 * affected * 4000
    monkeypatch.setattr(converter_losses, "MAX_RESPONSE_BYTES", 4 * request_bytes + 4 * affected * 4000 - 1)
    with pytest.raises(converter_losses.WaveformResourceLimitError):
        converter_losses.converter_evidence_budget(request, bindings)


def test_held_frequency_envelope_independent_references():
    """Check independent on/off voltage references and absent synthetic counts."""
    result = analyze_converter(envelope())
    parts = result["branches"][0]["components"]
    assert parts["conduction"]["power_w"] == 5
    assert parts["turn_on"]["power_w"] == 50
    assert parts["turn_off"]["power_w"] == pytest.approx(200 / 3)
    assert result["totals"]["total_power_w"] == pytest.approx(365 / 3)
    assert "event_count" not in parts["turn_on"]
    assert result["branches"][0]["evidence"]["intervals"][0]["energy_evidence"]["turn_off"]["recorded_conditions"]["v_supply_v"] == 600


@pytest.mark.parametrize("duty,frequency,expected", [(0, 100000, 0), (1, 100000, 20), (.25, 0, 5)])
def test_duty_endpoints_and_frequency_zero_need_no_energy_models(duty, frequency, expected):
    """Known inactive mechanisms require no invented energy curves."""
    request = envelope()
    request["signals"].update(d=[duty], fs=frequency)
    request["role_models"]["single"].pop("switching")
    if duty == 0:
        request["role_models"]["single"].pop("conduction")
    assert analyze_converter(request)["totals"]["total_power_w"] == expected


def test_irregular_held_time_weighting_and_refinement():
    """Held integration uses duration weights and preserves interval refinements."""
    request = envelope()
    request["time_s"] = [0, 1, 4]
    request["signals"].update(i=[5, 15], d=[1, 1])
    assert analyze_converter(request)["totals"]["total_power_w"] == 25
    request["time_s"] = [0, .5, 1, 4]
    request["signals"].update(i=[5, 5, 15], d=[1, 1, 1])
    assert analyze_converter(request)["totals"]["total_power_w"] == 25


def test_negative_current_zero_duty_requires_no_conduction_model():
    """Inactive conduction preserves negative input evidence without a fabricated model."""
    request = envelope()
    request["signals"].update(i=[-10], d=[0])
    request["role_models"]["single"] = {}
    result = analyze_converter(request)
    assert result["totals"]["total_power_w"] == 0
    assert result["branches"][0]["evidence"]["intervals"][0]["current_a"] == -10


def test_omitted_switching_does_not_require_unobserved_gate_or_bus():
    """Unused gate/bus bindings remain mapping evidence rather than mandatory data."""
    request = waveform()
    omitted = {"omit": True, "omission_reason": "No switching observation supplied"}
    for item in request["role_models"].values():
        item["switching"] = {"turn_on": omitted, "turn_off": omitted}
        item.pop("events")
    request["signals"] = {key: value for key, value in request["signals"].items() if key.startswith("is")}
    result = analyze_converter(request)
    assert result["totals"]["total_power_w"] is None
    assert result["branches"][0]["evidence"]["switching_signals"] == {}
    assert result["branches"][0]["evidence"]["switching_signals_status"] == "not-evaluated-explicit-omissions"
    assert result["branches"][0]["evidence"]["switching_cumulative_plot"] == []


def test_masks_do_not_require_gate_but_gate_mode_rejects_absent_gate():
    """Explicit edge evidence does not depend on an unavailable analog gate column."""
    request = waveform()
    request["signals"] = {key: value for key, value in request["signals"].items() if not key.startswith("vgs")}
    with pytest.raises(KeyError):
        analyze_converter(request)
    for item in request["role_models"].values():
        item["events"] = {"mode": "masks", "turn_on": [False, True, False, False],
                          "turn_off": [False, False, False, True]}
    assert analyze_converter(request)["totals"]["total_power_w"] is not None


def test_dab_conduction_commutation_separation_and_signs():
    """Conduction differs from signed pre-edge commutation in every DAB pair."""
    request = waveform()
    for name in ("is2", "is4", "is6", "is8"):
        request["signals"][name] = [10] * 5
    result = analyze_converter(request)
    assert result["physical_member_count"] == 8
    assert result["totals"]["components"]["conduction"]["power_w"] == 160
    assert [r["evidence"]["events"][0]["current_a"] for r in result["branches"]] == [-2, 2, -2, 2]
    assert [r["evidence"]["events"][1]["current_a"] for r in result["branches"]] == [-3, 3, -3, 3]
    assert result["totals"]["components"]["turn_on"]["power_w"] == pytest.approx(1000)
    assert result["totals"]["components"]["turn_off"]["power_w"] == pytest.approx(4000 / 3)


@pytest.mark.parametrize("preset,represented", [("sc-src-zcs", 2), ("sc-src-zvs", 4), ("sc-llc", 4)])
def test_source_omissions_are_explicit_partial_not_zero(preset, represented):
    """Historical source scope leaves secondary switching unknown."""
    request = waveform(preset)
    bindings = next(p["bindings"] for p in converter_presets()["presets"] if p["id"] == preset)
    for role in bindings:
        if role.get("source_switching_omission"):
            request["role_models"][role["role"]] = {"conduction": model()["conduction"]}
    with pytest.raises(ValueError, match="omission reason"):
        analyze_converter(request)
    request["assumptions"]["source_omission_reason"] = "Selected historical source scope; secondary recovery unknown"
    result = analyze_converter(request)
    assert result["totals"]["total_power_w"] is None
    assert sum(b["multiplicity"] for b in result["branches"] if b["components"]["turn_on"]["status"] == "computed") == represented
    assert result["physical_member_count"] == 8


@pytest.mark.parametrize("preset,expected", [("sc-vr", 12), ("sc-vrn", 30), ("sc-ivr", 24), ("sc-ivrn", 42), ("sc-tvr", 0)])
def test_vr_rectifier_counts_without_fabricated_voltage(preset, expected):
    """Physical rectifier multiplicities match every named variant."""
    request = {"preset": preset, "operation": "held-envelope-estimate", "time_s": [0, 1],
               "signals": {"ipfc": 0, "dpfc": 0, "fs": 0, "vdcp": 200, "idrl": 1, "idrn": 1},
               "assumptions": {"envelope_selected": True, "symmetry_selected": True},
               "role_models": {"pfc_mosfet": {}, "pfc_diode": {"conduction": model(0)["conduction"]}}}
    bindings = next(p["bindings"] for p in converter_presets()["presets"] if p["id"] == preset)
    for role in bindings[2:]:
        if role["multiplicity"]:
            request["role_models"][role["role"]] = {"conduction": model(2 if role["role"] == "phase_rectifier" else 3)["conduction"]}
    assert analyze_converter(request)["totals"]["total_power_w"] == expected


def test_six_switch_neutral_current_scale_and_line_frequency():
    """The threefold neutral current and pair count act separately."""
    request = {"preset": "sc-six-switch", "operation": "held-envelope-estimate", "time_s": [0, 1],
               "signals": {"ia": 10, "dna": 0, "dpa": 0, "dpn": 1, "fs": 100000, "fl": 50, "vdc": 200},
               "assumptions": {"envelope_selected": True, "symmetry_selected": True},
               "role_models": {"low_mosfets": {}, "high_mosfets": {}, "neutral_igbts": model(0, .1)}}
    result = analyze_converter(request)
    assert result["totals"]["total_power_w"] == 180
    request["signals"]["dpn"] = .5
    for component in ("turn_on", "turn_off"):
        request["role_models"]["neutral_igbts"]["switching"][component]["positive"] = energy(200)
    result = analyze_converter(request)
    assert result["totals"]["components"]["turn_on"]["power_w"] == .1
    assert result["totals"]["components"]["turn_off"]["power_w"] == .1
    assert result["totals"]["components"]["conduction"]["power_w"] == 90


def test_bridgeless_activation_does_not_flip_current_twice():
    """Operating-power activation does not change explicit current polarity."""
    request = {"preset": "sc-bridgeless", "operation": "held-envelope-estimate", "time_s": [0, 1],
               "signals": {"iac": -10, "dn1": 0, "dp1": .25, "fs": 0, "vdc": 200},
               "assumptions": {"envelope_selected": True, "symmetry_selected": True, "operating_power_w": 1},
               "role_models": {"mosfet": {}, "pfc_diode": model(0)}}
    assert analyze_converter(request)["totals"]["total_power_w"] == 0
    request["assumptions"]["operating_power_w"] = -1
    with pytest.raises(ValueError, match="reverse_igbt"):
        analyze_converter(request)
    request["role_models"]["reverse_igbt"] = model()
    assert analyze_converter(request)["totals"]["total_power_w"] == 10


def test_snapshot_immutability_and_metadata():
    """Canonical source data remains independent of later caller edits."""
    request = envelope()
    original = deepcopy(request)
    result = analyze_converter(request)
    assert request == original
    request["signals"]["i"][0] = 99
    assert result["request"]["signals"]["i"] == [10]
    assert result["branches"][0]["model_bindings"]["source"] == "Independent explicit role model"


@pytest.mark.parametrize("field,value", [("d", [-.01]), ("d", [1.01]), ("fs", -1), ("v", -1), ("i", [101])])
def test_invalid_held_inputs_reject(field, value):
    """Invalid duty, frequency, bus and extrapolated current fail."""
    request = envelope()
    request["signals"][field] = value
    with pytest.raises(ValueError):
        analyze_converter(request)


def test_energy_scaling_preserves_tiny_and_avoids_intermediate_overflow():
    """Equal reference/query voltages retain representable extreme energies."""
    for value, voltage in ((1e-308, 1e-308), (1e308, 1e308)):
        models = prepare_switching_models({"positive": energy(voltage, value, voltage)})
        assert evaluate_switching_energy(models, 0, voltage)["energy_j"] == pytest.approx(value, rel=1e-14, abs=0)


@pytest.mark.parametrize("reason", ["", "   ", True])
def test_zero_member_role_requires_a_nonblank_string_reason(reason):
    """Inactive physical scope is reasoned, never a truthy placeholder."""
    request = envelope()
    request["bindings"][0].update(multiplicity=0, members=[], inactive_reason=reason)
    with pytest.raises(ValueError, match="inactive reason"):
        analyze_converter(request)


def test_typed_api_canonical_equality_safe_nonfinite_and_resource_limit():
    """Exercise normalized API equality and safe validation/resource responses."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = envelope()
    response = client.post("/api/converters/analyze", json=request)
    assert response.status_code == 200, response.text
    # Replaying the returned normalized request is the canonical API/core contract.
    result = response.json()
    assert result == analyze_converter(result["request"])
    assert len(client.get("/api/converters/presets").json()["presets"]) == 12
    response = client.post("/api/converters/analyze", content='{"preset":"custom","operation":"waveform","time_s":[0,NaN],"signals":{},"role_models":{},"assumptions":{}}', headers={"content-type": "application/json"})
    assert response.status_code == 422
    request = envelope()
    request["bindings"] *= 2
    request["time_s"] = list(range(6001))
    assert client.post("/api/converters/analyze", json=request).status_code == 413


def test_explicit_inverter_six_supplied_unbalanced_branches():
    """Arbitrary phase records use all six explicit positions without symmetry."""
    request = {"preset": "inverter-six", "operation": "held-envelope-estimate", "time_s": [0, 1, 4],
               "signals": {"ia": [10, 20], "ib": [3, 5], "ic": [-8, -2], "vdc": 400, "fs": 0},
               "assumptions": {"envelope_selected": True}, "role_models": {}}
    for phase in "abc":
        request["signals"][f"dp{phase}"] = [.25, .7]
        request["signals"][f"dn{phase}"] = [.75, .3]
        for position in ("upper", "lower"):
            request["role_models"][f"{phase}_{position}"] = {"conduction": model(0, .1)["conduction"]}
    result = analyze_converter(request)
    expected = .1 * ((100 + 9 + 64) + 3 * (400 + 25 + 4)) / 4
    assert result["totals"]["total_power_w"] == pytest.approx(expected)
    assert result["physical_member_count"] == 6
