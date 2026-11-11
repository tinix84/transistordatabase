"""Independent gate-state, energy, and API compatibility checks."""
from copy import deepcopy
from hashlib import sha256
import json
import math
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gate_states import hysteretic_gate_states
from transistordatabase.gui_web.backend.current_allocation_routes import router as allocation_router
from transistordatabase.gui_web.backend.waveform_routes import router as waveform_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(allocation_router)
    app.include_router(waveform_router)
    return TestClient(app)


def absolute(low=2.5, high=7.5):
    return {"thresholds": {"scale": "absolute", "low_v": low, "high_v": high}}


def energy_curve(energy_j):
    return {"positive": {"mode": "recorded", "source": "constant-energy test fixture",
                         "current_a": [0, 2], "energy_j": [energy_j, energy_j],
                         "recorded_conditions": {"t_j_c": 25, "v_supply_v": 400},
                         "requested_conditions": {"t_j_c": 25, "v_supply_v": 400}}}


def test_noise_has_three_brackets_seven_millijoules_and_no_conduction_inference(client):
    gate_v = [0, 8, 4, 6, 4, 2, 10]
    time_s = [n / 1000 for n in range(7)]
    digitizer = absolute()
    payload = {"waveform": {"time_s": time_s, "current_a": [1] * 7, "gate_v": gate_v},
               "events": {"mode": "hysteresis", "digitizer": digitizer},
               "conduction": {"omit": True, "omission_reason": "switching only"},
               "switching": {"turn_on": energy_curve(0.002), "turn_off": energy_curve(0.003)}}
    response = client.post("/api/waveform/analyze", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["gate_digitization"]["states"] == [False, True, True, True, True, False, True]
    assert [(row["event"], row["sample_index"]) for row in body["events"]] == [
        ("turn_on", 0), ("turn_off", 4), ("turn_on", 5)]
    assert body["known_component_energy_j"] == pytest.approx(0.007)
    assert body["known_component_power_w"] == pytest.approx(7 / 6)
    assert body["total_energy_j"] is None
    assert body["request"]["waveform"]["gate_v"] == gate_v

    signed = [-3, 2, -1, 0, 7, 9, -5]
    allocation = {"time_s": time_s, "current_a": signed, "source": "synthetic signed trace",
                  "selected": True, "conditions": {},
                  "gate": {"mode": "hysteresis", "gate_v": gate_v, "digitizer": digitizer}}
    response = client.post("/api/waveform/gate-allocation", json=allocation)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["gate_digitization"] == body["gate_digitization"]
    assert [s["gate_on_current_a"] for s in result["samples"]] == [0, 2, -1, 0, 7, 0, -5]
    assert [s["gate_off_current_a"] for s in result["samples"]] == [-3, 0, 0, 0, 0, 9, 0]


def test_equalities_hold_the_preceding_state_without_mutating_evidence():
    trace = [0, 7, 8, 7, 2, 1, 2]
    policy = absolute(2, 7)
    original = deepcopy((trace, policy))
    evidence = hysteretic_gate_states(trace, policy)["evidence"]
    assert evidence["states"] == [False, False, True, True, True, False, False]
    assert evidence["turn_on_brackets"] == [1]
    assert evidence["turn_off_brackets"] == [4]
    assert (trace, policy) == original


def test_undecided_first_sample_requires_seed_and_never_implies_an_initial_event():
    trace = [4, 5, 8]
    with pytest.raises(ValueError, match="initial_state"):
        hysteretic_gate_states(trace, absolute())
    for seed, expected_on in [(True, []), (False, [1])]:
        result = hysteretic_gate_states(trace, absolute() | {"initial_state": seed})["evidence"]
        assert result["initial_state_source"] == "declared"
        assert result["turn_on_brackets"] == expected_on
        assert result["turn_off_brackets"] == []
    with pytest.raises(ValueError, match="conflicts"):
        hysteretic_gate_states([8, 6], absolute() | {"initial_state": False})


def test_extreme_calibration_is_finite_and_collapsed_thresholds_are_rejected():
    limit = sys.float_info.max
    thresholds = {"scale": "relative", "minimum_v": -limit, "maximum_v": limit,
                  "low_fraction": 0.25, "high_fraction": 0.75}
    observed = hysteretic_gate_states([-limit, 0, limit], {"thresholds": thresholds})["evidence"]
    assert observed["low_v"] == pytest.approx(-limit / 2)
    assert observed["high_v"] == pytest.approx(limit / 2)
    assert observed["states"] == [False, False, True]
    near = thresholds | {"minimum_v": 1, "maximum_v": math.nextafter(1, 2),
                         "low_fraction": 0, "high_fraction": 0.125}
    with pytest.raises(ValueError, match="threshold"):
        hysteretic_gate_states([0, 2], {"thresholds": near})


def test_underflowed_calibration_cannot_change_first_sample_evidence():
    thresholds = {"scale": "relative", "minimum_v": 0, "maximum_v": 1e-200,
                  "low_fraction": 1e-200, "high_fraction": 1}
    # Exact low is 1e-400 V, so a zero first sample cannot establish an on state.
    with pytest.raises(ValueError):
        hysteretic_gate_states([0, 2e-200], {"thresholds": thresholds, "initial_state": True})


def test_trace_size_and_constant_input_with_fixed_calibration():
    policy = absolute() | {"initial_state": True}
    result = hysteretic_gate_states([5] * 10_000, policy)["evidence"]
    assert result["states"] == [True] * 10_000
    assert result["turn_on_brackets"] == result["turn_off_brackets"] == []
    with pytest.raises(ValueError):
        hysteretic_gate_states([5] * 10_001, policy)
    relative = {"thresholds": {"scale": "relative", "minimum_v": 0, "maximum_v": 10,
                               "low_fraction": 0.25, "high_fraction": 0.75},
                "initial_state": True}
    seeded = hysteretic_gate_states([5, 5], relative)["evidence"]
    assert seeded["states"] == [True, True]
    assert seeded["turn_on_brackets"] == seeded["turn_off_brackets"] == []
    inferred = hysteretic_gate_states([0, 0], {"thresholds": relative["thresholds"]})["evidence"]
    assert inferred["states"] == [False, False]
    assert inferred["initial_state_source"] == "first-sample-below-low"


# Captured through HTTP on the merged e41cc8a application before adding hysteresis.
@pytest.mark.parametrize(("gate", "input_hash", "result_hash"), [
    ({"mode": "binary", "state": [False, True, False]},
     "9c7275627ddd2b1a446375dde2ad04598f246553a567b37c4e9b2c313da2b175",
     "3fc951f18c4b1b7793c89051a6a27e2ed51aa8078d172cce06101a6fe6dbc351"),
    ({"mode": "threshold", "gate_v": [0, 5, 6], "threshold_v": 5},
     "f13ef98c242c048e06ed56f109699d12b3a2d3acfa188cb6649888a23baa3a57",
     "f8a9d0fe2da36d6ec6f23e60f1c2e7b493c45779d7c2f8ecab3aa97f50c3c5ed"),
])
def test_legacy_allocation_keeps_baseline_http_evidence(client, gate, input_hash, result_hash):
    payload = {"time_s": [0, 1, 2], "current_a": [-2, 0, 3], "selected": True,
               "source": "compatibility fixture", "conditions": {}, "gate": gate}
    response = client.post("/api/waveform/gate-allocation", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["evidence"]["input_sha256"] == input_hash
    assert body["evidence"]["result_sha256"] == result_hash
    assert "gate_digitization" not in body["result"]


def test_legacy_waveform_keeps_complete_baseline_http_response(client):
    omission = {"omit": True, "omission_reason": "energy unavailable"}
    payload = {"waveform": {"time_s": [0, 1, 2], "current_a": [1, 1, 1], "gate_v": [0, 5, 6]},
               "conduction": {"omit": True, "omission_reason": "switching only"},
               "events": {"mode": "gate", "threshold_v": 5},
               "switching": {"turn_on": omission, "turn_off": omission}}
    response = client.post("/api/waveform/analyze", json=payload)
    assert response.status_code == 200, response.text
    canonical = json.dumps(response.json(), sort_keys=True, separators=(",", ":")).encode()
    assert sha256(canonical).hexdigest() == "b1edf8641807a0ef50e4ca0258f4d498b618bc01acc3aa70812eda755914d6ba"
