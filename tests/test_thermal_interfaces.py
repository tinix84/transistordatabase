"""Independent contact-stack fixtures for explicit steady thermal paths."""
from copy import deepcopy

import pytest

from transistordatabase.thermal import analyze_thermal, steady_paths


def contact(identifier, path, rho, area=1):
    """Declare one physical contact with SI values and explicit evidence."""
    return {"interface_id": identifier, "path": path, "resistivity_k_m2_w": rho,
            "area_m2": area, "source": "Independent synthetic contact",
            "conditions": {"pressure_pa": None}}


def request():
    """Return an ideal-base, two-branch fixture with simultaneous contacts."""
    return {"operation": "steady", "source": "Independent path fixture", "conditions": {},
            "sink_temperature_c": 25, "forward_power_w": 2, "reverse_power_w": 1,
            "common_resistance_k_w": 0, "forward_resistance_k_w": 0,
            "reverse_resistance_k_w": 0, "interfaces": [
                contact("sink-contact", "common", 1), contact("forward-contact", "forward", 2),
                contact("reverse-contact", "reverse", 3)]}


def test_three_physical_contacts_keep_shared_and_local_heat_separate():
    result = steady_paths(request())
    assert result["shared_temperature_c"] == 28
    assert result["forward_temperature_c"] == 32
    assert result["reverse_temperature_c"] == 31
    assert result["added_path_resistances_k_w"] == {"common": 1, "forward": 2, "reverse": 3}
    assert "interface_resistance_k_w" not in result


def test_distinct_series_contacts_on_one_path_are_additive_and_order_independent():
    value = request()
    value["interfaces"] += [contact("forward-pad-b", "forward", .5), contact("forward-grease", "forward", .5)]
    result = steady_paths(value)
    assert result["forward_temperature_c"] == 34
    assert result["path_resistances_k_w"]["forward"] == 3
    value["interfaces"].reverse()
    repeated = steady_paths(value)
    assert repeated["path_resistances_k_w"] == result["path_resistances_k_w"]
    assert repeated["forward_temperature_c"] == result["forward_temperature_c"]


def test_source_and_unknown_conditions_remain_independent_result_evidence():
    value = request()
    before = deepcopy(value)
    result = analyze_thermal(value)
    assert value == before
    assert result["request"] == before
    rows = result["result"]["interface_contributions"]
    assert rows[0]["interface_id"] == "sink-contact"
    assert rows[0]["conditions"] == {"pressure_pa": None}
    assert rows[0]["source"] == "Independent synthetic contact"
    rows[0]["conditions"]["pressure_pa"] = 123
    assert value == before
    assert result["request"] == before


def test_known_zero_contact_is_not_an_unknown_path():
    value = request()
    value["interfaces"] = [contact("ideal-common", "common", 0)]
    assert steady_paths(value)["shared_temperature_c"] == 25
    value["common_resistance_k_w"] = None
    with pytest.raises((ValueError, TypeError)):
        steady_paths(value)


@pytest.mark.parametrize("changes", [
    {"area_m2": 0}, {"area_m2": -1}, {"resistivity_k_m2_w": -1},
    {"resistivity_k_m2_w": float("inf")}, {"source": "  "}, {"path": "either"},
    {"interface_id": "  "},
])
def test_invalid_contact_never_becomes_a_zero_or_unassigned_path(changes):
    value = request()
    value["interfaces"][0].update(changes)
    with pytest.raises(ValueError):
        steady_paths(value)


def test_ambiguous_physical_identity_or_singular_plus_list_rejects():
    value = request()
    value["interfaces"][1]["interface_id"] = "sink-contact"
    with pytest.raises(ValueError):
        steady_paths(value)
    value = request()
    value["interface"] = {k: v for k, v in value["interfaces"][0].items() if k != "interface_id"}
    with pytest.raises(ValueError):
        steady_paths(value)


def test_tiny_representable_contact_resistance_produces_a_finite_kelvin_rise():
    value = request()
    value.update(sink_temperature_c=25, forward_power_w=1e300, reverse_power_w=0)
    value["interfaces"] = [contact("synthetic-extreme", "common", 1e-200, 1e100)]
    result = steady_paths(value)
    assert result["added_path_resistances_k_w"]["common"] == pytest.approx(1e-300, rel=1e-14, abs=0)
    assert result["shared_temperature_c"] == pytest.approx(26)


@pytest.mark.parametrize("rho,area", [(1e-200, 1e200), (1e308, 1e-308)])
def test_unrepresentable_nonzero_resistance_rejects_explicitly(rho, area):
    value = request()
    value["interfaces"] = [contact("unsupported-range", "common", rho, area)]
    with pytest.raises(ValueError):
        steady_paths(value)


def test_raw_contact_decimal_underflow_is_rejected_before_becoming_ideal_zero():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import json
    from transistordatabase.gui_web.backend.thermal_routes import router

    app = FastAPI()
    app.include_router(router)
    value = request()
    value["interfaces"][0]["resistivity_k_m2_w"] = 0.123456789
    raw = json.dumps(value).replace("0.123456789", "-1e-400")
    with TestClient(app) as client:
        response = client.post("/api/thermal/analyze", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
