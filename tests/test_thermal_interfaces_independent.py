"""Independent heat-flow, source preservation, and legacy HTTP checks."""
from copy import deepcopy
from hashlib import sha256
import json
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.thermal_routes import router
from transistordatabase.thermal import analyze_thermal


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def steady(**fields):
    return {
        "operation": "steady",
        "source": "synthetic thermal compatibility fixture",
        "conditions": {"clamp_pressure_pa": None},
        "sink_temperature_c": 25,
        "forward_power_w": 2,
        "reverse_power_w": 1,
        "common_resistance_k_w": 1,
        "forward_resistance_k_w": 2,
        "reverse_resistance_k_w": 3,
        **fields,
    }


def contact(identifier, path, rho, area):
    return {"interface_id": identifier, "path": path,
            "resistivity_k_m2_w": rho, "area_m2": area,
            "source": f"synthetic contact {identifier}", "conditions": {"clamp_pressure_pa": None}}


def public_result(client, payload):
    response = client.post("/api/thermal/analyze", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_two_series_contacts_and_independent_branch_contacts_preserve_evidence(client):
    contacts = [contact("shared left", "common", 0.5, 2),
                contact("junction A", "forward", 6, 3),
                contact("shared right", "common", 3, 4),
                contact("junction B", "reverse", 6, 2)]
    payload = steady(interfaces=contacts, common_resistance_k_w=0,
                     forward_resistance_k_w=0, reverse_resistance_k_w=0)
    saved = deepcopy(payload)
    core = analyze_thermal(payload)
    assert payload == saved
    assert core["result"] == public_result(client, payload)["result"]
    result = core["result"]
    assert result["added_path_resistances_k_w"] == {"common": 1, "forward": 2, "reverse": 3}
    assert result["path_resistances_k_w"] == {"common": 1, "forward": 2, "reverse": 3}
    assert result["shared_temperature_c"] == 28
    assert result["forward_temperature_c"] == 32
    assert result["reverse_temperature_c"] == 31
    assert "interface_resistance_k_w" not in result
    expected = dict(zip([c["interface_id"] for c in contacts], [0.25, 2, 0.75, 3], strict=True))
    rows = {row["interface_id"]: row for row in result["interface_contributions"]}
    assert rows.keys() == expected.keys()
    for submitted in contacts:
        row = rows[submitted["interface_id"]]
        assert row["resistance_k_w"] == expected[submitted["interface_id"]]
        for field, original in submitted.items():
            assert row[field] == original


def test_reverse_junction_with_no_local_heat_still_follows_the_common_boundary(client):
    payload = steady(forward_power_w=5, reverse_power_w=0, common_resistance_k_w=0,
                     forward_resistance_k_w=0, reverse_resistance_k_w=0,
                     interfaces=[contact("mount", "common", 0.5, 0.25),
                                 contact("junction A", "forward", 3, 2),
                                 contact("junction B", "reverse", 1000, 1)])
    result = public_result(client, payload)["result"]
    assert result["shared_temperature_c"] == result["reverse_temperature_c"] == 35
    assert result["forward_temperature_c"] == 42.5


def test_permuted_sub_ulp_contacts_have_the_same_exact_resistance_and_temperature(client):
    contacts = [contact("base washer", "common", 1, 1),
                contact("film A", "common", 2 ** -53, 1),
                contact("film B", "common", 2 ** -53, 1)]
    for order in (contacts, list(reversed(contacts))):
        result = public_result(client, steady(interfaces=order, forward_power_w=2**52, reverse_power_w=0,
                    sink_temperature_c=0, common_resistance_k_w=0, forward_resistance_k_w=0,
                    reverse_resistance_k_w=0))["result"]
        assert result["path_resistances_k_w"]["common"] == math.nextafter(1.0, 2.0)
        assert result["shared_temperature_c"] == 2**52 + 1
        assert result["forward_temperature_c"] == result["reverse_temperature_c"] == 2**52 + 1


def test_tiny_finite_interface_can_cause_a_measurable_temperature_rise(client):
    payload = steady(interfaces=[contact("synthetic resistance", "common", 1e-200, 1e100)],
                     sink_temperature_c=0, forward_power_w=1e300, reverse_power_w=0,
                     common_resistance_k_w=0, forward_resistance_k_w=0, reverse_resistance_k_w=0)
    result = public_result(client, payload)["result"]
    assert math.isclose(result["added_path_resistances_k_w"]["common"], 1e-300, rel_tol=1e-14)
    assert math.isclose(result["shared_temperature_c"], 1, rel_tol=1e-14)


def test_nonzero_unrepresentable_contribution_is_not_an_ideal_thermal_contact(client):
    payload = steady(interfaces=[contact("subnormal limit", "forward", 1e-200, 1e200)])
    response = client.post("/api/thermal/analyze", json=payload)
    assert response.status_code == 422, response.text
    payload["interfaces"][0]["resistivity_k_m2_w"] = 0
    result = public_result(client, payload)["result"]
    assert result["interface_contributions"][0]["resistance_k_w"] == 0


def test_duplicate_physical_contact_on_different_paths_is_rejected(client):
    contacts = [contact("shared pad", "common", 1, 1), contact("shared pad", "forward", 1, 1)]
    response = client.post("/api/thermal/analyze", json=steady(interfaces=contacts))
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("literal", ["1e-400", "-1e-400"])
def test_raw_contact_literal_cannot_silently_become_ideal_zero(client, literal):
    interface = contact("calibrated pad", "common", 73, 1)
    payload = steady(interfaces=[interface])
    raw = json.dumps(payload).replace('"resistivity_k_m2_w": 73', f'"resistivity_k_m2_w": {literal}')
    response = client.post("/api/thermal/analyze", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize(("single", "expected_hash"), [
    (False, "da74222662a6d19d5c58a046ad38b5c5dd64d9c7fe866d0a37a64399443a9b15"),
    (True, "468c542495c5ce9d239dfd2687976b011dba0861dd5115b324f8bee2de1f5877"),
])
def test_historical_interface_requests_keep_complete_http_results(client, single, expected_hash):
    # Captured from e41cc8a with a synthetic model and no database access.
    payload = steady()
    if single:
        payload["interface"] = {"path": "common", "resistivity_k_m2_w": 0.002,
                                "area_m2": 0.001, "source": "synthetic interface",
                                "conditions": {"clamp_pressure_pa": None}}
    response = public_result(client, payload)
    canonical = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    assert sha256(canonical).hexdigest() == expected_hash


def test_singular_and_plural_models_cannot_count_one_contact_twice(client):
    interface = contact("bridge", "common", 1, 1)
    singular = {k: v for k, v in interface.items() if k != "interface_id"}
    response = client.post("/api/thermal/analyze", json=steady(interface=singular, interfaces=[interface]))
    assert response.status_code == 422, response.text


def test_thermal_transport_exception_preserves_nonthermal_stream_limit_and_unicode_guard():
    from transistordatabase.gui_web.backend.waveform_routes import router as waveform_router

    app = FastAPI()
    app.include_router(waveform_router)
    client = TestClient(app)
    chunk = b"?" * 65536
    body = (chunk if i < 64 else b"?" for i in range(65))
    response = client.post("/api/waveform/analyze", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert "4194304" in response.json()["detail"]

    source = br'{"provenance":{"source":"\ud800"}}'
    response = client.post("/api/waveform/analyze", content=source, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "UTF-8" in response.json()["detail"]
