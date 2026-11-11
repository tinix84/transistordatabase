"""Independent weighted-merit populations, extrema, ties, and HTTP fixtures."""
from copy import deepcopy
import csv
import io
import json
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.grid_analysis import AnalysisResourceLimitError, evaluate_grid, input_hash
from transistordatabase.gui_web.backend.analysis_routes import GridRequest, router


def merit_fixture():
    """Use real constant-drop models and one constant-energy turn-on event."""
    configs = []
    for identity, drop, energy in (("A", 10, 4), ("B", 8, 8)):
        configs.append({"id": identity, "source": "Independent merit fixture",
            "conduction": {"positive": {"mode": "linear-drop-estimate", "selected": True,
                "offset_v": drop, "resistance_ohm": 0, "conditions": {}, "source": "Declared drop"}},
            "switching": {
                "turn_on": {"positive": {"mode": "recorded", "current_a": [1, 3],
                    "energy_j": [energy, energy], "recorded_conditions": {},
                    "requested_conditions": {}, "source": "Declared event energy"}},
                "turn_off": {"omit": True, "omission_reason": "No turn-off event"}}})
    points = [{"id": identity, "source": "Independent constant current", "weight": weight,
               "waveform": {"time_s": [0, 1], "current_a": [current, current], "gate_v": [0, 1]},
               "events": {"mode": "gate", "threshold_v": .5}}
              for identity, weight, current in (("p1", 1, 1), ("p2", 3, 3))]
    return {"configurations": configs, "operating_points": points,
            "metric": {"name": "conduction_power_w", "unit": "W", "direction": "minimize"},
            "objectives": [{"name": "conduction_power_w", "unit": "W", "direction": "minimize", "weight": 1},
                           {"name": "turn_on_power_w", "unit": "W", "direction": "minimize", "weight": 3}]}


def oracle(data):
    """Use the independent A=[10,30]/4 and B=[20,20]/8 objective fixture."""
    is_a = data["conduction"]["positive"]["offset_v"] == 10
    cond = (10 if data["waveform"]["current_a"][0] == 1 else 30) if is_a else 20
    on = 4 if is_a else 8
    return {"total_power_w": cond + on, "components": {
        "conduction": {"power_w": cond}, "turn_on": {"power_w": on}, "turn_off": {"power_w": 0}}}


def test_independent_weighted_fixture_and_reversed_direction():
    """Check raw weighted means and both objective directions independently."""
    request = merit_fixture()
    before = deepcopy(request)
    result = evaluate_grid(request, _evaluator=oracle)
    merit = result["merit"]
    assert request == before
    assert [(row["configuration_id"], row["score"], row["rank"]) for row in merit["ranking"]] == [
        ("A", .25, 1), ("B", .75, 2)]
    assert [row["objectives"][0]["full_mean"] for row in merit["rows"]] == [25, 20]
    assert [objective["normalized_weight"] for objective in merit["objectives"]] == [.25, .75]
    assert merit["objectives"][0]["minimum_configuration_ids"] == ["B"]
    assert merit["objectives"][1]["maximum_configuration_ids"] == ["B"]
    assert merit["grid_input_sha256"] == result["input_sha256"]
    request["objectives"][1]["direction"] = "maximize"
    reversed_result = evaluate_grid(request, _evaluator=oracle)["merit"]
    assert [(row["configuration_id"], row["score"]) for row in reversed_result["ranking"]] == [("B", 0), ("A", 1)]


def test_http_real_models_and_matching_raw_exports():
    """Check the registered route with real electrical models and raw export strings."""
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post("/api/analysis/grid", json=merit_fixture())
    assert response.status_code == 200, response.text
    merit = response.json()["merit"]
    assert [row["score"] for row in merit["ranking"]] == [.25, .75]
    assert json.loads(merit["rows_json"]) == merit["export_rows"]
    csv_rows = list(csv.DictReader(io.StringIO(merit["rows_csv"])))
    assert len(csv_rows) == 4
    assert csv_rows[0]["population_operating_point_ids"] == '["p1","p2"]'
    assert merit["null_csv"] == "NULL"


def test_one_fixed_population_excludes_incomplete_configuration():
    """Exclude incomplete configurations from every active objective range."""
    def failing(data):
        if data["conduction"]["positive"]["offset_v"] == 10 and data["waveform"]["current_a"][0] == 3:
            raise ValueError("Independent missing positive-weight point")
        return oracle(data)
    request = merit_fixture()
    request["summary_mode"] = "successful-subset"
    merit = evaluate_grid(request, _evaluator=failing)["merit"]
    assert merit["population"]["configuration_ids"] == ["B"]
    assert merit["population"]["configuration_coverage"] == .5
    a = merit["rows"][0]
    assert (a["status"], a["score"], a["rank"]) == ("excluded", None, None)
    assert a["objectives"][0]["coverage"] == .25
    assert a["objectives"][0]["full_mean"] is None
    assert all(objective["contribution"] is None for objective in a["objectives"])
    assert a["exclusion_reasons"][0]["missing_operating_point_ids"] == ["p2"]
    assert all(objective["minimum_configuration_ids"] == ["B"] for objective in merit["objectives"])
    assert merit["ranking"] == [{"configuration_id": "B", "score": 0, "rank": 1}]


def test_zero_objectives_and_zero_weight_points_do_not_restrict_population():
    """Keep inactive raw evidence without changing the common eligible population."""
    request = merit_fixture()
    request["objectives"][1] = {"name": "total_power_w", "unit": "W", "direction": "maximize", "weight": 0}
    zero = deepcopy(request["operating_points"][0])
    zero.update(id="zero", weight=0)
    zero["waveform"]["time_s"] = [0, 0]
    request["operating_points"].append(zero)
    merit = evaluate_grid(request)["merit"]
    assert merit["population"]["configuration_ids"] == ["A", "B"]
    assert merit["population"]["operating_point_ids"] == ["p1", "p2"]
    assert merit["population"]["zero_weight_operating_point_ids"] == ["zero"]
    inactive = merit["rows"][0]["objectives"][1]
    assert inactive["full_mean"] is None
    assert (inactive["active"], inactive["normalized_weight"], inactive["penalty"], inactive["contribution"]) == (False, 0, None, 0)


def test_constant_ranges_and_competition_ranks_keep_all_ties():
    """Use zero penalties for constant objectives and equal ranks for exact ties."""
    request = merit_fixture()
    request["objectives"] = [{"name": "turn_off_power_w", "unit": "W", "direction": "maximize", "weight": 1}]
    merit = evaluate_grid(request, _evaluator=oracle)["merit"]
    assert [row["rank"] for row in merit["ranking"]] == [1, 1]
    assert [row["score"] for row in merit["ranking"]] == [0, 0]
    assert merit["objectives"][0]["minimum_configuration_ids"] == ["A", "B"]
    assert merit["objectives"][0]["maximum_configuration_ids"] == ["A", "B"]
    single = deepcopy(request)
    single["configurations"] = single["configurations"][:1]
    assert evaluate_grid(single, _evaluator=oracle)["merit"]["ranking"] == [{"configuration_id": "A", "score": 0, "rank": 1}]
    competition = merit_fixture()
    competition["configurations"].append(dict(competition["configurations"][0], id="A2"))
    assert [row["rank"] for row in evaluate_grid(competition, _evaluator=oracle)["merit"]["ranking"]] == [1, 1, 3]


def test_no_eligible_population_is_unavailable_without_changing_grid_rows():
    """Keep incomplete electrical evidence when no merit population is available."""
    request = merit_fixture()
    request["objectives"] = [{"name": "total_power_w", "unit": "W", "direction": "minimize", "weight": 1}]
    result = evaluate_grid(request)
    merit = result["merit"]
    assert all(row["status"] == "computed" for row in result["rows"])
    assert (merit["status"], merit["ranking"], merit["population"]["configuration_ids"]) == ("unavailable", [], [])
    assert merit["objectives"][0]["minimum"] is None


@pytest.mark.parametrize("weights", [[True, 1], [-1, 1], [0, 0], [math.inf, 1], [math.nan, 1],
                                    [1e308, 1e308], [5e-324, 1e308]])
def test_invalid_or_unrepresentable_objective_weights_fail_early(weights):
    """Reject invalid totals and normalized positive-weight underflow before evaluation."""
    request = merit_fixture()
    for objective, weight in zip(request["objectives"], weights, strict=True):
        objective["weight"] = weight
    def must_not_evaluate(data):
        raise AssertionError("Invalid objective weights reached pair evaluation")
    with pytest.raises(ValueError):
        evaluate_grid(request, _evaluator=must_not_evaluate)


def test_duplicate_and_empty_objectives_are_invalid():
    """Reject duplicate objective names even when one duplicate has zero weight."""
    request = merit_fixture()
    request["objectives"][1].update(name="conduction_power_w", weight=0)
    with pytest.raises(ValueError, match="unique"):
        evaluate_grid(request)
    request["objectives"] = []
    with pytest.raises(ValueError):
        evaluate_grid(request)
    with pytest.raises(ValueError):
        GridRequest.model_validate(request)


@pytest.mark.parametrize("collection", ["objectives", "operating_points"])
def test_http_rejects_nonzero_json_weight_underflow_and_accepts_explicit_zero(collection):
    """Preserve the difference between explicit zero and unrepresentable positive literals."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = merit_fixture()
    request[collection][0]["weight"] = "NONZERO_WEIGHT_TOKEN"
    raw = json.dumps(request).replace('"NONZERO_WEIGHT_TOKEN"', "1e-999")
    assert client.post("/api/analysis/grid", content=raw,
                       headers={"content-type": "application/json"}).status_code == 422
    request[collection][0]["weight"] = 0
    assert client.post("/api/analysis/grid", json=request).status_code == 200


def test_absent_objectives_preserve_legacy_inputs_hashes_and_ranking():
    """Keep optional merit selection out of canonical legacy input snapshots."""
    request = merit_fixture()
    request.pop("objectives")
    typed = GridRequest.model_validate(request).model_dump()
    assert "objectives" not in typed
    legacy = evaluate_grid(typed, _evaluator=oracle)
    typed["objectives"] = None
    disabled = evaluate_grid(typed, _evaluator=oracle)
    assert "merit" not in disabled
    assert disabled["input_sha256"] == legacy["input_sha256"]
    assert disabled["ranking"] == legacy["ranking"]


def test_population_hash_binds_original_inputs_weights_and_raw_means():
    """Bind the population fingerprint to original inputs and raw means."""
    request = merit_fixture()
    result = evaluate_grid(request, _evaluator=oracle)
    merit = result["merit"]
    normalized = [{key: objective[key] for key in ("name", "unit", "direction", "weight", "normalized_weight", "active")}
                  for objective in merit["objectives"]]
    evidence = {"grid_input_sha256": result["input_sha256"], "population": merit["population"],
        "objectives": normalized,
        "raw_means": [{"configuration_id": row["configuration_id"],
                       "means": {objective["name"]: objective["full_mean"] for objective in row["objectives"]}}
                      for row in merit["rows"]]}
    assert input_hash(evidence) == merit["population_sha256"]
    request["objectives"][0]["weight"] = 2
    assert evaluate_grid(request, _evaluator=oracle)["merit"]["population_sha256"] != merit["population_sha256"]


def test_merit_metadata_budget_precedes_export_materialization(monkeypatch):
    """Reject a complete response that cannot hold the repeated merit exports."""
    monkeypatch.setattr("transistordatabase.grid_analysis.MAX_RESPONSE_BYTES", 30000)
    with pytest.raises(AnalysisResourceLimitError, match="Merit"):
        evaluate_grid(merit_fixture(), _evaluator=oracle)
