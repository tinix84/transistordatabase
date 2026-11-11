"""Typed HTTP transport for inverse gate-loop resistance estimates."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.parameter_routes import router


app = FastAPI()
app.include_router(router)
client = TestClient(app)
PAYLOAD = {"operation": "gate-resistance-estimate", "selected": True,
           "source": "Independent algebra fixture; not a measured device",
           "conditions": {"t_j_c": 25, "v_supply_v": 400, "current_a": None},
           "target_dv_dt_v_s": 20e9, "c_rss_min_f": 100e-12,
           "v_drive_high_v": 15, "v_drive_low_v": -5,
           "v_plateau_low_v": 5, "v_plateau_high_v": 7}


def test_exact_si_request_context_and_total_resistance_results():
    """Preserve supplied source, unknown conditions, SI inputs and model limits."""
    response = client.post("/api/parameters/calculate", json=PAYLOAD)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["inputs"] == PAYLOAD
    assert data["context"] == {"source": PAYLOAD["source"], "conditions": PAYLOAD["conditions"]}
    assert data["result"]["r_on_ohm"] == pytest.approx(5)
    assert data["result"]["r_off_ohm"] == pytest.approx(6)
    assert data["result"]["unit"] == "ohm"
    assert "driver, internal and external" in data["result"]["scope"]
    assert len(data["result"]["assumptions"]) >= 3
    assert data["plots"] == []


@pytest.mark.parametrize("changes", [
    {"selected": False}, {"selected": 1}, {"target_dv_dt_v_s": True},
    {"c_rss_min_f": "100e-12"}, {"c_rss_min_f": 0}, {"target_dv_dt_v_s": -20e9},
    {"v_drive_low_v": 5}, {"v_drive_high_v": 7}, {"v_plateau_low_v": 8},
    {"v_plateau_high_v": None}, {"target_dv_dt_kv_us": 20}, {"source": " "},
    {"conditions": {"t_j_c": True}}, {"conditions": {"v_supply_v": "400"}},
    {"target_dv_dt_v_s": 1e308, "c_rss_min_f": 1e308},
])
def test_invalid_requests_return_actionable_422(changes):
    """Do not coerce units, booleans, strings, missing values or invalid domains."""
    response = client.post("/api/parameters/calculate", json={**PAYLOAD, **changes})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_missing_estimate_selection_is_rejected():
    """An omitted estimate acknowledgement cannot execute the model."""
    request = {key: value for key, value in PAYLOAD.items() if key != "selected"}
    response = client.post("/api/parameters/calculate", json=request)
    assert response.status_code == 422
    assert "Select" in response.json()["detail"]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_nonfinite_json_is_json_safe(token):
    """The shared route returns a valid JSON validation response for raw tokens."""
    raw = json.dumps({**PAYLOAD, "target_dv_dt_v_s": "TOKEN"}).replace('"TOKEN"', token)
    response = client.post("/api/parameters/calculate", content=raw,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "finite" in response.text
