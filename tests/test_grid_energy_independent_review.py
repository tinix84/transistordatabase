"""Check scientific invariants across the public electrical and thermal routes."""
import copy
import hashlib
import json
import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.gui_web.backend import analysis_routes, thermal_routes, waveform_routes


app = FastAPI()
for routes in (analysis_routes, waveform_routes, thermal_routes):
    app.include_router(routes.router)
client = TestClient(app)


def accepted(path, payload):
    """Read the accepted public response."""
    response = client.post(f"/api/{path}", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def grid():
    """Two held currents give 10 and 30 W for A and 20 and 20 W for B."""
    energy = {"mode": "recorded", "current_a": [0, 3], "energy_j": [.001, .002],
              "source": "Unused event model", "recorded_conditions": {}, "requested_conditions": {}}
    return {
        "configurations": [{
            "id": name, "source": f"Grid fixture {name}",
            "conduction": {"positive": {
                "mode": "recorded", "current_a": [1, 2], "drop_v": drops,
                "source": f"Declared drop {name}", "recorded_conditions": {},
                "requested_conditions": {}}},
            "switching": {name: {"positive": energy} for name in ("turn_on", "turn_off")},
        } for name, drops in (("A", [10, 15]), ("B", [20, 10]))],
        "operating_points": [{
            "id": f"I{amps}", "source": "Constant-current fixture", "weight": weight,
            "waveform": {"time_s": [0, 1], "current_a": [amps, amps], "gate_v": [0, 0]},
            "events": {"mode": "gate", "threshold_v": .5},
        } for amps, weight in ((1, 1), (2, 3))],
        "metric": {"name": "total_power_w", "unit": "W", "direction": "minimize"},
    }


def digest(evidence):
    """Reconstruct the hash from response evidence without application helpers."""
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_grid_physical_powers_coverage_ranking_and_repeat():
    """Calculate real pair losses and isolate one out-of-range model."""
    request = grid()
    real = accepted("analysis/grid", request)
    assert [r["result"]["total_power_w"] for r in real["rows"]] == [10, 30, 20, 20]
    assert [(r["configuration_id"], r["value"]) for r in real["ranking"]] == [("B", 20), ("A", 25)]
    assert all(r["input_sha256"] == digest(r["inputs"]) for r in real["rows"])
    assert digest(real["inputs"]) == real["input_sha256"]
    request["configurations"][0]["conduction"]["positive"]["current_a"] = [.5, 1]
    request["configurations"][0]["conduction"]["positive"]["drop_v"] = [8, 10]
    partial = accepted("analysis/grid", request)
    a = partial["summaries"][0]["metrics"]["total_power_w"]
    assert [r["status"] for r in partial["rows"]] == ["computed", "unsupported", "computed", "computed"]
    assert (a["full_mean"], a["selected_mean"], a["subset_mean"], a["coverage"]) == (None, None, 10, .25)
    assert len(partial["ranking"]) == 1
    request["summary_mode"] = "successful-subset"
    assert accepted("analysis/grid", request)["ranking"][0]["value"] == 10
    request["operating_points"][1]["weight"] = 0
    request["include_zero_weight_in_worst"] = True
    a = accepted("analysis/grid", request)["summaries"][0]["metrics"]["total_power_w"]
    assert (a["full_mean"], a["complete"], a["worst_complete"]) == (10, True, False)
    assert accepted("analysis/grid", grid()) == real


def test_tiny_individual_weight_does_not_invalidate_finite_mean():
    """Keep the mean when a single normalized term rounds to zero."""
    request = grid()
    for op, weight in zip(request["operating_points"], [1e-300, 1e100], strict=True):
        op["weight"] = weight
    assert [r["value"] for r in accepted("analysis/grid", request)["ranking"]] == [20, 30]


def test_transport_enforces_real_input_byte_limit():
    """Reject a large raw body even when its parsed request is small."""
    body = b" " * 4_000_000 + json.dumps(grid()).encode()
    response = client.post("/api/analysis/grid", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_repeated_event_metadata_budget_precedes_full_waveform_construction(monkeypatch):
    """Reject repeated condition evidence before the waveform engine allocates it."""
    from transistordatabase import grid_analysis

    request = grid()
    waveform = {"time_s": list(range(100)), "current_a": [1] * 100, "gate_v": [k % 2 for k in range(100)]}
    request["configurations"] = request["configurations"][:1]
    request["operating_points"] = [dict(request["operating_points"][0], waveform=waveform)]
    conditions = {"Source condition " + "x" * 1000: 1}
    for event in request["configurations"][0]["switching"].values():
        event["positive"].update(recorded_conditions=conditions, requested_conditions=conditions)
    assert len(json.dumps(request)) < 16000
    monkeypatch.setattr(grid_analysis, "MAX_RESPONSE_BYTES", 80000)

    def reject_full_evaluation(_request):
        pytest.fail("Repeated event evidence alone exceeds the response budget")

    monkeypatch.setattr(grid_analysis, "analyze_waveform", reject_full_evaluation)
    response = client.post("/api/analysis/grid", json=request)
    assert response.status_code == 413


def test_coherent_spectrum_has_independent_frequency_and_amplitude():
    """Recover the declared DC and AC amplitudes from a finite sampled signal."""
    request = grid()
    point = request["operating_points"][0]
    point["spectrum"] = {"window": "rectangular", "detrend": "none"}
    point["waveform"] = {
        "time_s": [k / 128 for k in range(32)],
        "current_a": [1.5 + .25 * math.sin(2 * math.pi * 3 * k / 32) for k in range(32)],
        "gate_v": [0] * 32,
    }
    result = accepted("analysis/grid", request)["rows"][0]["result"]
    spectrum = result["spectrum"]
    assert spectrum["frequency_hz"][3] == 12
    assert spectrum["amplitude_a"][0] == pytest.approx(1.5)
    assert spectrum["amplitude_a"][3] == pytest.approx(.25)
    assert spectrum["sequence_period_s"] == .25
    assert result["duration_s"] == 31 / 128
    assert spectrum["parseval_rms_a"] == pytest.approx(math.sqrt(1.5**2 + .25**2 / 2))


def test_absolute_pre_edge_impulse_and_exact_cubic_foster_convolution():
    """Compare the thermal result to the analytical single-pole temperature."""
    data = grid()
    # I=1+u, V=2I and duty=u give P=2u+4u²+2u³ over one second.
    model = {"mode": "linear-drop-estimate", "selected": True, "source": "2 ohm",
             "conditions": {}, "offset_v": 0, "resistance_ohm": 2}
    heat = {"mode": "recorded", "source": "Finite switching test",
            "recorded_conditions": {}, "requested_conditions": {},
            "current_a": [0, 2], "energy_j": [.5, .5]}
    waveform = {
        "waveform": {"time_s": [3, 4], "current_a": [1, 2], "gate_v": [0, 1]},
        "events": data["operating_points"][0]["events"],
        "conduction": {"positive": model, "duty": [0, 1]},
        "switching": {"turn_on": {"positive": heat}, "turn_off": {"positive": heat}},
    }
    original = copy.deepcopy(waveform)
    direct = accepted("waveform/analyze", waveform)
    result = accepted("analysis/thermal-handoff", {
        "waveform_request": waveform, "direction_owners": {
            "positive": "reverse", "negative": "forward", "zero": "forward"},
        "network_heat_owner": "reverse", "event_mode": "impulse",
    })
    stage = result["thermal_input"]
    assert digest(result["source_request"]) == result["source_input_sha256"]
    assert result["scope"] == "complete-waveform-heat"
    assert result["waveform_result"]["power_segments"] == direct["power_segments"]
    assert [{k: v for k, v in seg.items() if k != "heat_owner"} for seg in stage["segments"]] == direct["power_segments"]
    assert stage["segments"][0]["heat_owner"] == "reverse"
    assert stage["segments"][0]["power_coefficients_w"] == [0, 2, 4, 2]
    assert stage["events"][0]["time_s"] == 3
    assert stage["events"][0]["heat_owner"] == "reverse"
    assert stage["events"][0]["energy_j"] == .5
    thermal = accepted("thermal/analyze", {
        "operation": "transient", "source": result["source_input_sha256"], "conditions": {},
        **stage, "case_temperature_c": 25, "initial_state": "cold", "samples_per_segment": 3,
        "network": {"source": "Declared one-pole R=2, tau=1", "conditions": {},
                    "r_th_vector": [2], "tau_vector": [1]},
    })["result"]
    # Integral exp(u)*(2u+4u²+2u³) du from 0 to 1 is 6.
    expected_temperature = 25 + 2 * 6 / math.e + 1 / math.e
    assert thermal["final_temperature_c"] == pytest.approx(expected_temperature)
    assert thermal["conduction_energy_j"] == pytest.approx(1 + 4 / 3 + .5)
    assert thermal["switching_energy_j"] == .5
    assert waveform == original


def test_energy_map_source_evidence_interior_knots_ambiguity_and_exclusions():
    """Keep original source knots and disclose ambiguous or missing cells."""
    conditions = {"t_j_c": None, "v_g_on_v": 15, "v_g_off_v": -3, "r_g_on_ohm": 2,
                  "r_g_off_ohm": 4, "commutation_device": "D", "commutation_source": None,
                  "energy_convention": "Terminal energy integral"}
    base = {"energy_kind": "on", "source_kind": "simulated", "conditions": conditions}
    request = {
        "energy_kind": "on", "conditions": conditions, "currents_a": [10, 0], "voltages_v": [800, 400],
        "slices": [
            {**base, "id": "lo", "source": "400 V original", "voltage_v": 400,
             "current_a": [10, 5, 0], "energy_j": [.001, .0008, 0]},
            {**base, "id": "hi", "source": "800 V original", "voltage_v": 800,
             "current_a": [0, 10], "energy_j": [0, .002]},
        ],
        "estimate": {"selected": True, "current_a": 5, "voltage_v": 600},
    }
    original = copy.deepcopy(request)
    result = accepted("analysis/energy-map", request)
    assert result["estimate"]["energy_j"] == pytest.approx(.00075)
    assert result["source_slices"] == request["slices"]
    assert result["cells"][0][0]["energy_j"] == 0
    assert result["cells"][1][1]["candidates"][0]["source_kind"] == "simulated"
    assert request == original
    request["slices"][0]["energy_j"][-1] = .0002
    assert accepted("analysis/energy-map", request)["cells"][0][0]["energy_j"] == .0002
    conflicting = dict(request["slices"][0], id="duplicate", source_kind="datasheet", energy_j=[.002, .001, 0])
    request["slices"].append(conflicting)
    request["slices"][1]["conditions"] = dict(conditions, commutation_source="Known D data")
    failed = accepted("analysis/energy-map", request)
    assert failed["cells"][0][0]["status"] == "ambiguous"
    assert {c["slice_id"] for c in failed["cells"][0][0]["candidates"]} == {"lo", "duplicate"}
    assert failed["cells"][1][0]["status"] == "missing"
    assert failed["estimate"]["status"] == "unsupported"
    assert failed["source_slices"] == request["slices"]
    assert failed["slice_diagnostics"][1]["reason"] == "fixed-condition-mismatch"
    assert failed["input_sha256"] != result["input_sha256"]
