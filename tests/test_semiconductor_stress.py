"""Independent hand calculations and strict stress-proxy transport checks."""
import copy
import hashlib
import json
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.stress_routes import router
from transistordatabase.semiconductor_stress import calculate_stress_indices


def fixture():
    """Supply a ramp with an independently known exact second moment."""
    return {"source": "Independent ramp", "conditions": {"t_j_c": None},
            "time_s": [0, 1], "current_a": [0, 6], "voltage_v": [400, 400],
            "rds_on_ohm": .1,
            "transitions": [{"kind": "turn_on", "time_s": .5,
                             "frequency_hz": 1000}]}


def client():
    """Use an isolated application without repository data fixtures."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_exact_ramp_and_explicit_event():
    """The exact ramp square differs from trapezoidal squared samples."""
    result = calculate_stress_indices(**fixture())["result"]
    assert result["current_mean_square_a2"] == pytest.approx(12)
    assert result["current_rms_a"] == pytest.approx(math.sqrt(12))
    assert result["peak_voltage_v"] == 400
    expected = {"conduction": (12, "A^2"), "volume": (1920000, "A^2 V^2"),
                "switching": (3600000, "Hz A^2 V"),
                "cost": (.0000625, "1/(ohm V^2)")}
    for key, (value, unit) in expected.items():
        metric = result["metrics"][key]
        assert metric == {"value": pytest.approx(value), "unit": unit,
                          "status": "calculated", "reason": None}
    assert result["transition_samples"][0]["current_a"] == 3


def test_nonuniform_signed_ramp_and_peak():
    """Keep signed maximum voltage and integrate the full nonuniform interval."""
    inputs = fixture() | {"time_s": [5, 6, 8], "current_a": [-3, 3, -3],
                          "voltage_v": [-400, -100, -200], "transitions": [
                              {"kind": "turn_on", "time_s": 5, "frequency_hz": 2},
                              {"kind": "turn_off", "time_s": 8, "frequency_hz": 3}]}
    result = calculate_stress_indices(**inputs)["result"]
    assert result["duration_s"] == 3
    assert result["current_mean_square_a2"] == pytest.approx(3)
    assert result["peak_voltage_v"] == -100
    assert result["metrics"]["volume"]["value"] == pytest.approx(30000)
    assert result["metrics"]["switching"]["value"] == -12600
    assert result["metrics"]["cost"]["value"] == pytest.approx(.001)
    assert [s["current_a"] for s in result["transition_samples"]] == [-3, -3]


def test_unknown_and_explicit_empty_events_differ():
    """Missing events and resistance remain unknown instead of becoming zero."""
    inputs = fixture() | {"transitions": None, "rds_on_ohm": None}
    metrics = calculate_stress_indices(**inputs)["result"]["metrics"]
    for key in ("switching", "cost"):
        assert metrics[key]["status"] == "unknown"
        assert metrics[key]["value"] is None
        assert metrics[key]["reason"]
    inputs["transitions"] = []
    assert calculate_stress_indices(**inputs)["result"]["metrics"]["switching"]["value"] == 0


def test_zero_voltage_and_zero_current_are_explicit():
    """Retain valid zero stress and mark a zero-denominator cost as undefined."""
    result = calculate_stress_indices(**(fixture() | {
        "current_a": [0, 0], "voltage_v": [0, 0]}))["result"]
    assert result["current_rms_a"] == 0
    assert result["metrics"]["volume"]["value"] == 0
    assert result["metrics"]["switching"]["value"] == 0
    assert result["metrics"]["cost"]["value"] is None
    assert "zero" in result["metrics"]["cost"]["reason"]


def test_evidence_hashes_and_no_input_mutation():
    """Bind reproducible evidence to normalized inputs and returned definitions."""
    inputs = fixture()
    original = copy.deepcopy(inputs)
    envelope = calculate_stress_indices(**inputs)
    assert inputs == original
    for field in ("inputs", "result"):
        encoded = json.dumps(envelope[field], sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()
        key = "input_sha256" if field == "inputs" else "result_sha256"
        assert envelope["evidence"][key] == hashlib.sha256(encoded).hexdigest()
    changed = calculate_stress_indices(**(inputs | {"source": "Changed source"}))
    assert changed["evidence"]["input_sha256"] != envelope["evidence"]["input_sha256"]


@pytest.mark.parametrize("update", [
    {"time_s": [0, 0]}, {"time_s": [1, 0]}, {"current_a": [1]},
    {"voltage_v": [1, 2, 3]}, {"source": " "}, {"conditions": {"": 2}},
    {"conditions": {"t_j_c": True}}, {"conditions": {"t_j_c": "25"}},
    {"current_a": [True, 2]}, {"current_a": ["1", 2]},
    {"current_a": [math.inf, 1]}, {"rds_on_ohm": 0},
    {"rds_on_ohm": -1}, {"rds_on_ohm": True},
    {"transitions": [{"kind": "turn_on", "time_s": -1, "frequency_hz": 1}]},
    {"transitions": [{"kind": "turn_on", "time_s": 2, "frequency_hz": 1}]},
    {"transitions": [{"kind": "turn_on", "time_s": 0, "frequency_hz": -1}]},
    {"transitions": [{"kind": "unknown", "time_s": 0, "frequency_hz": 1}]},
    {"transitions": [{"kind": "turn_on", "time_s": 0, "frequency_hz": True}]},
    {"transitions": [{"kind": "turn_on", "time_s": 0, "frequency_hz": 1,
                      "current_a": 99}]},
])
def test_invalid_inputs_are_rejected_in_core_and_transport(update):
    """Reject invalid domains without conversions, extrapolation, or clamping."""
    inputs = fixture() | update
    with pytest.raises(ValueError):
        calculate_stress_indices(**inputs)
    response = client().post("/api/stress/calculate", content=json.dumps(inputs),
                             headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_transport_roundtrip_and_json_nan():
    """Keep strict validation errors JSON-safe for nonstandard JSON numbers."""
    api = client()
    response = api.post("/api/stress/calculate", json=fixture())
    assert response.status_code == 200
    assert response.json() == calculate_stress_indices(**fixture())
    invalid = json.dumps(fixture()).replace('"rds_on_ohm": 0.1', '"rds_on_ohm": NaN')
    assert api.post("/api/stress/calculate", content=invalid,
                    headers={"Content-Type": "application/json"}).status_code == 422


def test_actual_body_limit_does_not_trust_content_length():
    """Bound actual bytes before parsing or validating the input."""
    response = client().post("/api/stress/calculate", content=b" " * (4*1024*1024 + 1),
                             headers={"Content-Length": "1"})
    assert response.status_code == 413


def test_sample_transition_and_condition_count_limits():
    """Enforce bounded public core inputs without filesystem test data."""
    for update in ({"time_s": [0] * 20001}, {"transitions": [{}] * 20001},
                   {"conditions": {str(i): None for i in range(101)}}):
        with pytest.raises(ValueError):
            calculate_stress_indices(**(fixture() | update))


@pytest.mark.parametrize("update", [
    {"current_a": [0, 10**400]}, {"conditions": {"t_j_c": 10**400}},
    {"current_a": [1e308, 1e308]}, {"current_a": [1e-300, 1e-300]},
    {"time_s": [0, 1e-300, 1e100], "current_a": [1e100, 0, 0],
     "voltage_v": [400, 400, 400]},
    {"time_s": [-1e308, 1e308]},
    {"rds_on_ohm": 1e-300, "voltage_v": [1e-300, 1e-300]},
])
def test_numeric_overflow_and_underflow_are_errors(update):
    """Reject unrepresentable arithmetic instead of producing fabricated zero."""
    with pytest.raises(ValueError):
        calculate_stress_indices(**(fixture() | update))


def test_huge_json_integer_is_validation_error():
    """A finite-range conversion failure must not become a server error."""
    inputs = fixture() | {"conditions": {"t_j_c": 10**400}}
    assert client().post("/api/stress/calculate", json=inputs).status_code == 422
