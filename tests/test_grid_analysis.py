"""Independent weighted-grid fixtures and registered-operation transport checks."""
from copy import deepcopy
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.grid_analysis import (
    AnalysisResourceLimitError, evaluate_grid, thermal_handoff,
)
from transistordatabase.gui_web.backend.analysis_routes import GridRequest, router
from transistordatabase.thermal import analyze_thermal


def fixture():
    """Use constant current and exact resistance power, without switching heat."""
    configs = []
    for name, resistance in (("A", 1), ("B", 2)):
        configs.append({"id": name, "source": "Independent fixture",
                        "conduction": {"positive": {
                            "mode": "linear-drop-estimate", "selected": True,
                            "offset_v": 0, "resistance_ohm": resistance,
                            "conditions": {}, "source": "Declared resistance"}},
                        "switching": {name: {"omit": True, "omission_reason": "Conduction fixture"}
                                      for name in ("turn_on", "turn_off")}})
    points = [{"id": name, "source": "Independent waveform", "weight": weight,
               "waveform": {"time_s": [0, 1], "current_a": [current, current]},
               "events": {"mode": "omitted", "omission_reason": "Conduction fixture"}}
              for name, weight, current in (("p1", 1, 1), ("p2", 3, 2))]
    return {"configurations": configs, "operating_points": points,
            "metric": {"name": "conduction_power_w", "unit": "W", "direction": "minimize"}}


def evaluator(data):
    """Independent A=[10,30], B=[20,20] aggregation oracle."""
    resistance = data["conduction"]["positive"]["resistance_ohm"]
    value = (10 if data["waveform"]["current_a"][0] == 1 else 30) if resistance == 1 else 20
    return {"total_power_w": value, "components": {
        "conduction": {"power_w": value}, "turn_on": {"power_w": 0}, "turn_off": {"power_w": 0}}}


def summary(result, name="A"):
    """Select one configuration's conduction summary."""
    return next(s["metrics"]["conduction_power_w"] for s in result["summaries"]
                if s["configuration_id"] == name)


def test_independent_weighted_fixture_and_order_identity():
    """Verify independent means and stable pair identities."""
    request = fixture()
    before = deepcopy(request)
    result = evaluate_grid(request, _evaluator=evaluator)
    assert summary(result)["full_mean"] == 25
    assert summary(result)["worst_value"] == 30
    assert summary(result)["worst_point_ids"] == ["p2"]
    assert summary(result, "B")["full_mean"] == 20
    assert [r["configuration_id"] for r in result["ranking"]] == ["B", "A"]
    request["configurations"].reverse()
    request["operating_points"].reverse()
    reverse = evaluate_grid(request, _evaluator=evaluator)
    key = lambda r: (r["configuration_id"], r["operating_point_id"])
    assert {key(r): r["input_sha256"] for r in result["rows"]} == {
        key(r): r["input_sha256"] for r in reverse["rows"]}
    request["configurations"].reverse()
    request["operating_points"].reverse()
    assert request == before


def test_failure_preserves_other_pairs_and_requires_subset_selection():
    """Keep successful rows and disclose incomplete population coverage."""
    def fails(data):
        if data["conduction"]["positive"]["resistance_ohm"] == 1 and data["waveform"]["current_a"][0] == 2:
            raise ValueError("Missing bounded condition")
        return evaluator(data)
    request = fixture()
    result = evaluate_grid(request, _evaluator=fails)
    assert [r["status"] for r in result["rows"]] == ["computed", "unsupported", "computed", "computed"]
    a = summary(result)
    assert a["full_mean"] is None and a["selected_mean"] is None
    assert (a["subset_mean"], a["retained_weight"], a["total_weight"], a["coverage"]) == (10, 1, 4, .25)
    request["summary_mode"] = "successful-subset"
    assert summary(evaluate_grid(request, _evaluator=fails))["selected_mean"] == 10
    assert summary(evaluate_grid(fixture(), _evaluator=evaluator))["full_mean"] == 25


def test_zero_weight_failure_and_worst_policy():
    """Separate weighted coverage from the selected worst-point population."""
    request = fixture()
    request["operating_points"][1]["weight"] = 0
    result = evaluate_grid(request, _evaluator=evaluator)
    assert summary(result)["full_mean"] == 10
    assert summary(result)["worst_value"] == 10
    request["include_zero_weight_in_worst"] = True
    assert summary(evaluate_grid(request, _evaluator=evaluator))["worst_value"] == 30
    def fail_zero(data):
        if data["waveform"]["current_a"][0] == 2:
            raise ValueError("Zero-weight failure")
        return evaluator(data)
    result = evaluate_grid(request, _evaluator=fail_zero)
    assert summary(result)["full_mean"] == 10
    assert not summary(result)["worst_complete"]


@pytest.mark.parametrize("weights", [[0, 0], [-1, 3], [math.inf, 1], [math.nan, 1], [1e308, 1e308]])
def test_bad_weights(weights):
    """Reject invalid or unrepresentable total weights."""
    request = fixture()
    for p, weight in zip(request["operating_points"], weights, strict=True):
        p["weight"] = weight
    with pytest.raises(ValueError):
        evaluate_grid(request)


def test_duplicate_identity_product_and_preallocation_limits():
    """Reject duplicate IDs and excessive work before evaluation."""
    request = fixture()
    request["configurations"][1]["id"] = "A"
    with pytest.raises(ValueError, match="unique"):
        evaluate_grid(request)
    request = fixture()
    request["operating_points"] = [dict(request["operating_points"][0], id=str(n)) for n in range(65)]
    with pytest.raises(AnalysisResourceLimitError):
        evaluate_grid(request)
    request = fixture()
    request["operating_points"][0]["waveform"] = {"time_s": list(range(100001)), "current_a": [1] * 100001}
    with pytest.raises(AnalysisResourceLimitError, match="samples"):
        evaluate_grid(request)


def test_response_budget_is_global_and_not_a_scientific_pair_failure(monkeypatch):
    """Reject the complete response when the accumulated byte budget is exceeded."""
    monkeypatch.setattr("transistordatabase.grid_analysis.MAX_RESPONSE_BYTES", 1000)
    with pytest.raises(AnalysisResourceLimitError, match="response"):
        evaluate_grid(fixture())


def test_programmer_errors_propagate_and_evaluator_cannot_mutate_sources():
    """Isolate scientific inputs without suppressing programmer errors."""
    request = fixture()
    before = deepcopy(request)
    def mutates(data):
        result = evaluator(data)
        data["waveform"]["current_a"][0] = 999
        return result
    result = evaluate_grid(request, _evaluator=mutates)
    assert request == before
    assert result["rows"][0]["inputs"]["waveform_request"]["waveform"]["current_a"][0] == 1
    def broken(data):
        raise TypeError("Programmer bug")
    with pytest.raises(TypeError, match="Programmer"):
        evaluate_grid(request, _evaluator=broken)


def client():
    """Create the isolated registered analysis API."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_http_equals_real_engine_and_failures_remain_pairs():
    """Compare typed transport with the accepted waveform calculation."""
    request = fixture()
    expected = evaluate_grid(GridRequest.model_validate(request).model_dump())
    response = client().post("/api/analysis/grid", json=request)
    assert response.status_code == 200, response.text
    actual = response.json()
    assert summary(actual)["full_mean"] == pytest.approx(3.25)
    assert actual["rows"][0]["result"] == expected["rows"][0]["result"]
    request["configurations"][0]["conduction"]["positive"]["selected"] = False
    actual = client().post("/api/analysis/grid", json=request).json()
    assert [r["status"] for r in actual["rows"]] == ["unsupported", "unsupported", "computed", "computed"]


def test_http_rejects_custom_operations_extra_fields_and_nonfinite():
    """Reject executable extensions, unknown fields, and oversized grids."""
    request = fixture()
    request["operation"] = "python"
    assert client().post("/api/analysis/grid", json=request).status_code == 422
    request = fixture()
    request["operating_points"] = [dict(request["operating_points"][0], id=str(n)) for n in range(65)]
    assert client().post("/api/analysis/grid", json=request).status_code == 413
    response = client().post("/api/analysis/grid", content='{"configurations":[],"operating_points":[],"metric":{"name":"total_power_w","unit":"W","direction":"minimize"},"bad":NaN}', headers={"content-type": "application/json"})
    assert response.status_code == 422


def handoff_request():
    """Declare exact conduction heat and explicit physical ownership."""
    data = fixture()
    return {"waveform_request": {"waveform": data["operating_points"][0]["waveform"],
                                "events": data["operating_points"][0]["events"],
                                "conduction": data["configurations"][0]["conduction"],
                                "switching": data["configurations"][0]["switching"]},
            "direction_owners": {"positive": "forward", "negative": "reverse", "zero": "forward"},
            "network_heat_owner": "forward", "event_mode": "impulse", "allow_partial": True}


def test_handoff_exact_coefficients_partial_scope_and_thermal_identity():
    """Preserve coefficients and pass staged heat to the thermal engine."""
    request = handoff_request()
    result = thermal_handoff(request)
    assert result["thermal_input"]["segments"] == [
        dict(segment, heat_owner="forward") for segment in result["waveform_result"]["power_segments"]]
    assert result["scope"] == "known-components-only"
    assert result["omission_reasons"] == {"turn_on": "Conduction fixture", "turn_off": "Conduction fixture"}
    thermal_request = {**result["thermal_input"], "operation": "transient",
                       "source": result["source_input_sha256"], "conditions": {},
                       "network": {"source": "Independent one pole", "conditions": {},
                                   "r_th_vector": [2], "tau_vector": [1]},
                       "case_temperature_c": 25, "initial_state": "cold", "samples_per_segment": 1}
    thermal = analyze_thermal(thermal_request)
    assert thermal is not None
    response = client().post("/api/analysis/thermal-handoff", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["thermal_input"]["segments"] == result["thermal_input"]["segments"]
    request["allow_partial"] = False
    assert client().post("/api/analysis/thermal-handoff", json=request).status_code == 422


def test_handoff_rejects_owner_mismatch_and_negative_origin():
    """Reject incompatible physical ownership and unsupported time origins."""
    request = handoff_request()
    request["network_heat_owner"] = "reverse"
    with pytest.raises(ValueError, match="owner"):
        thermal_handoff(request)
    request = handoff_request()
    request["waveform_request"]["waveform"]["time_s"] = [-1, 0]
    with pytest.raises(ValueError, match="origin"):
        thermal_handoff(request)


def test_weighted_mean_preserves_representable_small_contribution():
    """Keep a representable contribution despite an underflowing weight ratio."""
    request = fixture()
    request["operating_points"][0]["weight"] = 1e-300
    request["operating_points"][1]["weight"] = 1e100
    def extreme(data):
        result = evaluator(data)
        value = 1e300 if data["waveform"]["current_a"][0] == 1 else 0
        result["total_power_w"] = value
        result["components"]["conduction"]["power_w"] = value
        return result
    assert summary(evaluate_grid(request, _evaluator=extreme))["full_mean"] == pytest.approx(1e-100, rel=1e-12, abs=0)


def test_derived_overflow_is_pair_failure():
    """Keep unaffected configurations when finite source values overflow."""
    request = fixture()
    request["configurations"][0]["conduction"]["positive"]["resistance_ohm"] = 1e308
    result = evaluate_grid(request)
    assert result["rows"][1]["status"] == "unsupported"
    assert all(r["status"] == "computed" for r in result["rows"][2:])


def test_handoff_keeps_finite_zero_current_event_energy_and_explicit_owner():
    """Keep finite E(0), event conditions, and explicit zero-current heat ownership."""
    request = handoff_request()
    waveform = request["waveform_request"]
    waveform["waveform"] = {"time_s": [0, 1], "current_a": [0, 0], "gate_v": [0, 1]}
    waveform["events"] = {"mode": "gate", "threshold_v": .5}
    waveform["switching"]["turn_on"] = {"positive": {
        "mode": "recorded", "source": "Finite E0 fixture", "current_a": [0, 10],
        "energy_j": [.001, .002], "recorded_conditions": {"t_j_c": 25},
        "requested_conditions": {"t_j_c": 25}}}
    request["direction_owners"]["zero"] = "reverse"
    request["network_heat_owner"] = "shared-junction"
    result = thermal_handoff(request)
    event = result["thermal_input"]["events"][0]
    source_event = result["waveform_result"]["events"][0]
    assert (event["energy_j"], event["time_s"]) == (source_event["energy_j"], source_event["time_s"])
    assert event["energy_j"] == .001
    assert event["heat_owner"] == "reverse"
    assert event["conditions"] == {"t_j_c": 25}
    request["network_heat_owner"] = "forward"
    with pytest.raises(ValueError, match="heat owner"):
        thermal_handoff(request)


def test_tiny_summands_only_scale_after_the_complete_sum():
    """Keep representable totals when individual normalized terms would underflow."""
    request = fixture()
    request["configurations"] = request["configurations"][:1]
    request["operating_points"] = [dict(request["operating_points"][0], id=str(n), weight=1)
                                   for n in range(64)]
    def tiny(data):
        result = evaluator(data)
        result["total_power_w"] = 5e-324
        result["components"]["conduction"]["power_w"] = 5e-324
        return result
    assert summary(evaluate_grid(request, _evaluator=tiny))["full_mean"] == 5e-324


@pytest.mark.parametrize("path", ["grid", "energy-map", "thermal-handoff"])
def test_actual_body_bytes_are_bounded_before_schema_validation(path):
    """Count whitespace bytes and apply the same limit to all analysis endpoints."""
    response = client().post(f"/api/analysis/{path}", content=b" " * 4_000_001,
                             headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_explicit_direction_owner_survives_signed_zero_power_segments():
    """Keep electrical signs while staging every segment for one declared junction."""
    request = handoff_request()
    request["direction_owners"] = {"positive": "reverse", "negative": "forward", "zero": "forward"}
    request["network_heat_owner"] = "reverse"
    waveform = request["waveform_request"]
    waveform["waveform"] = {"time_s": [0, 1, 2], "current_a": [1, 0, -1]}
    waveform["conduction"]["negative"] = deepcopy(waveform["conduction"]["positive"])
    waveform["conduction"]["duty"] = 0
    result = thermal_handoff(request)
    assert {segment["direction"] for segment in result["thermal_input"]["segments"]} == {"positive", "negative"}
    thermal_request = {**result["thermal_input"], "operation": "transient",
                       "source": result["source_input_sha256"], "conditions": {},
                       "network": {"source": "One pole", "conditions": {},
                                   "r_th_vector": [2], "tau_vector": [1]},
                       "case_temperature_c": 25, "initial_state": "cold", "samples_per_segment": 1}
    app = FastAPI()
    from transistordatabase.gui_web.backend.thermal_routes import router as thermal_router
    app.include_router(thermal_router)
    response = TestClient(app).post("/api/thermal/analyze", json=thermal_request)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["final_temperature_c"] == 25
    assert response.json()["result"]["conduction_energy_j"] == 0
    # Metadata cannot suppress nonzero heat or change the physical network check.
    thermal_request["segments"][1]["power_coefficients_w"] = [1]
    thermal_request["segments"][1]["energy_j"] = 0
    response = TestClient(app).post("/api/thermal/analyze", json=thermal_request)
    assert response.status_code == 422
    assert "heat owner" in response.json()["detail"]
    # Without explicit ownership, retain the standalone signed-segment rule.
    thermal_request["segments"][1]["power_coefficients_w"] = [0]
    del thermal_request["segments"][1]["heat_owner"]
    thermal_request["segments"][1]["direction"] = "positive"
    response = TestClient(app).post("/api/thermal/analyze", json=thermal_request)
    assert response.status_code == 422


@pytest.mark.parametrize("from_grid", [False, True])
def test_handoff_preserves_source_field_presence_and_exact_waveform_result(from_grid):
    """Keep both original omitted fields and grid-supplied explicit null fields."""
    app = FastAPI()
    app.include_router(router)
    from transistordatabase.gui_web.backend.waveform_routes import router as waveform_router
    app.include_router(waveform_router)
    transport = TestClient(app)
    handoff = handoff_request()
    if from_grid:
        response = transport.post("/api/analysis/grid", json=fixture())
        assert response.status_code == 200, response.text
        expected = response.json()["rows"][0]["result"]
    else:
        response = transport.post("/api/waveform/analyze", json=handoff["waveform_request"])
        assert response.status_code == 200, response.text
        expected = response.json()
    handoff["waveform_request"] = expected["request"]
    result = transport.post("/api/analysis/thermal-handoff", json=handoff)
    assert result.status_code == 200, result.text
    result = result.json()
    assert result["waveform_result"] == expected
    assert result["source_request"] == expected["request"]
    from transistordatabase.grid_analysis import input_hash
    assert result["source_input_sha256"] == input_hash(expected["request"])


def test_handoff_metadata_budget_precedes_full_waveform_construction(monkeypatch):
    """Reject repeated switching metadata before constructing the waveform result."""
    request = handoff_request()
    waveform = request["waveform_request"]
    waveform["waveform"] = {"time_s": list(range(100)), "current_a": [1] * 100,
                             "gate_v": [k % 2 for k in range(100)]}
    waveform["events"] = {"mode": "gate", "threshold_v": .5}
    conditions = {"Large declared source " + "x" * 1000: 1}
    for name in ("turn_on", "turn_off"):
        waveform["switching"][name] = {"positive": {
            "mode": "recorded", "source": "Independent switching fixture",
            "current_a": [0, 10], "energy_j": [.001, .002],
            "recorded_conditions": conditions, "requested_conditions": conditions}}
    monkeypatch.setattr("transistordatabase.grid_analysis.MAX_RESPONSE_BYTES", 80000)
    def unexpected(data):
        pytest.fail("The full waveform evaluator must not start")
    monkeypatch.setattr("transistordatabase.grid_analysis.analyze_waveform", unexpected)
    response = client().post("/api/analysis/thermal-handoff", json=request)
    assert response.status_code == 413


def test_handoff_checks_complete_response_budget(monkeypatch):
    """Bound the complete staged envelope after a safe waveform evaluation."""
    monkeypatch.setattr("transistordatabase.grid_analysis.MAX_RESPONSE_BYTES", 1000)
    monkeypatch.setattr("transistordatabase.grid_analysis._waveform_metadata_budget", lambda *args, **kwargs: None)
    with pytest.raises(AnalysisResourceLimitError, match="handoff response"):
        thermal_handoff(handoff_request())


def test_eight_pairs_with_thousand_samples_remain_usable():
    """Keep a practical grid available when repeated model metadata is small."""
    request = fixture()
    request["configurations"] = [dict(request["configurations"][0], id=str(n)) for n in range(8)]
    point = request["operating_points"][0]
    point["waveform"] = {"time_s": [n / 1000 for n in range(1000)], "current_a": [1] * 1000}
    request["operating_points"] = [point]
    result = evaluate_grid(request)
    assert len(result["rows"]) == 8
    assert all(row["status"] == "computed" for row in result["rows"])
