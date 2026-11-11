"""Hand fixtures for sample allocation and strict public transport."""
import copy
import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.current_allocation import allocate_gate_current
from transistordatabase.gui_web.backend.current_allocation_routes import router


def fixture():
    return {"time_s": [5, 6, 8, 10], "current_a": [-3, 4, -5, 0],
            "gate": {"mode": "binary", "state": [False, 1, True, 0.0]},
            "selected": True, "source": "Independent signed sample fixture",
            "conditions": {"junction_temperature_c": None}, "provenance": None}


def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_complementary_signed_original_samples_and_hashes():
    inputs = fixture()
    saved = copy.deepcopy(inputs)
    result = allocate_gate_current(**inputs)
    rows = result["result"]["samples"]
    assert inputs == saved
    assert [row["time_s"] for row in rows] == [5, 6, 8, 10]
    assert [row["gate_on_current_a"] for row in rows] == [0, 4, -5, 0]
    assert [row["gate_off_current_a"] for row in rows] == [-3, 0, 0, 0]
    assert all(row["gate_on_current_a"] + row["gate_off_current_a"] == row["current_a"] for row in rows)
    assert result["inputs"]["conditions"]["junction_temperature_c"] is None
    assert "No state or interpolation" in result["result"]["semantics"]
    assert "physical current paths" in result["result"]["scope"]
    for name, key in [("inputs", "input_sha256"), ("result", "result_sha256")]:
        canonical = json.dumps(result[name], sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert result["evidence"][key] == hashlib.sha256(canonical.encode()).hexdigest()


def test_threshold_equality_is_low_and_negative_gate_voltage_is_valid():
    result = allocate_gate_current(**(fixture() | {"gate": {
        "mode": "threshold", "gate_v": [-5, .5, 2, 0], "threshold_v": .5}}))
    assert [row["gate_on"] for row in result["result"]["samples"]] == [False, False, True, False]


@pytest.mark.parametrize("patch", [
    {"selected": False}, {"selected": 1}, {"source": " "}, {"source": "x" * 2001},
    {"conditions": None}, {"conditions": {"": None}}, {"conditions": {"t": True}},
    {"time_s": [0, 0, 1, 2]}, {"time_s": [0, 2, 1, 3]},
    {"time_s": [0, float("inf"), 2, 3]}, {"current_a": [0, float("nan"), 2, 3]},
    {"current_a": [False, 1, 2, 3]}, {"current_a": ["0", 1, 2, 3]},
    {"current_a": [1, 2]}, {"time_s": [0], "current_a": [0]},
    {"gate": {"mode": "binary", "state": [0, 1, 2, 0]}},
    {"gate": {"mode": "binary", "state": ["false", 1, 0, 1]}},
    {"gate": {"mode": "binary", "state": [0, 1]}},
    {"gate": {"mode": "binary", "state": [0, 1, 0, 1], "threshold_v": 0}},
    {"gate": {"mode": "threshold", "gate_v": [0, 1], "threshold_v": 0}},
    {"gate": {"mode": "threshold", "gate_v": [0, 1, 0, 1], "threshold_v": True}},
    {"gate": {}}, {"provenance": {"edited_after_import": 1}},
])
def test_science_rejects_ambiguous_or_invalid_input(patch):
    with pytest.raises(ValueError):
        allocate_gate_current(**(fixture() | patch))


def test_typed_api_matches_python_and_preserves_provenance():
    inputs = fixture() | {"provenance": {"source": None, "calibration_notes": "SI assumed",
                                       "imported_filename": "signed.csv", "edited_after_import": True}}
    response = client().post("/api/waveform/gate-allocation", json=inputs)
    assert response.status_code == 200
    assert response.json() == allocate_gate_current(**inputs)


def test_partial_provenance_has_identical_science_and_transport_hashes():
    inputs = fixture() | {"provenance": {"imported_filename": "signed.csv"}}
    response = client().post("/api/waveform/gate-allocation", json=inputs)
    assert response.status_code == 200
    assert response.json() == allocate_gate_current(**inputs)


@pytest.mark.parametrize("state", [False, True, 0, 1.0])
def test_all_low_and_all_high_are_valid_without_invented_transitions(state):
    inputs = fixture() | {"gate": {"mode": "binary", "state": [state] * 4}}
    result = allocate_gate_current(**inputs)["result"]
    active = "gate_on_current_a" if state else "gate_off_current_a"
    assert [row[active] for row in result["samples"]] == inputs["current_a"]
    assert result["sample_count"] == 4


@pytest.mark.parametrize("patch", [
    {"selected": 1}, {"selected": "true"}, {"current_a": [False, 1, 2, 3]},
    {"extra": 0}, {"gate": {"mode": "binary", "state": [0, "1", 0, 1]}},
    {"conditions": {"t": "25"}}, {"time_s": list(range(10001))},
])
def test_api_rejects_coercion_extra_fields_and_sample_limit(patch):
    assert client().post("/api/waveform/gate-allocation", json=fixture() | patch).status_code == 422


@pytest.mark.parametrize("literal", ["1e-999", "1e999", "NaN", "Infinity"])
def test_api_rejects_lossy_and_nonfinite_json_numbers(literal):
    text = json.dumps(fixture()).replace('-3', literal, 1)
    assert client().post("/api/waveform/gate-allocation", content=text, headers={"Content-Type": "application/json"}).status_code == 422


def test_actual_request_bytes_are_bounded():
    response = client().post("/api/waveform/gate-allocation", content=b" " * (4 * 1024 * 1024 + 1))
    assert response.status_code == 413


@pytest.mark.parametrize("literal", ["1.00000000000000001", ".99999999999999999"])
def test_api_rejects_nonbinary_decimal_before_float_rounding(literal):
    # JSON requires the leading zero for a fractional numeric token.
    literal = "0" + literal if literal.startswith(".") else literal
    text = json.dumps(fixture()).replace('false', literal, 1)
    assert client().post("/api/waveform/gate-allocation", content=text,
                         headers={"Content-Type": "application/json"}).status_code == 422


def test_science_rejects_small_exact_fraction_in_binary_mask():
    from fractions import Fraction
    with pytest.raises(ValueError):
        allocate_gate_current(**(fixture() | {"gate": {"mode": "binary", "state": [0, 1, Fraction(1, 10**400), 0]}}))
