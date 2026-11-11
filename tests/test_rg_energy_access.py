"""Check repository selection, helper equivalence and strict read-only HTTP access."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase.core.models import (
    ElectricalRatings, MeasurementEvidence, SwitchingLossData, ThermalProperties,
    Transistor, TransistorMetadata,
)
from transistordatabase.gui_web.backend.rg_energy_routes import create_rg_energy_router
from transistordatabase.rg_energy_access import (
    calculate_rg_energy, rg_energy_inventory,
)
from transistordatabase.rg_formula import (
    find_rg_for_target_energy, interpolate_energy_at_rg,
)


def part():
    """Build independent analytical curves without any stored repository writes."""
    device = Transistor(
        TransistorMetadata("Browser Rg energy", "SiC-MOSFET", "fixture", "A", "B"),
        ElectricalRatings(900, 65, 45, 160), ThermalProperties(.01, .01),
    )
    direct = SwitchingLossData(
        "graph_r_e", 75, 400, 12, i_x=10,
        graph_r_e=np.array([[1., 4., 9.], [.008, .002, .006]]),
    )
    currents = [SwitchingLossData(
        "graph_i_e", 75, 400, 12, r_g=rg,
        graph_i_e=np.array([[6., 14.], energies]),
    ) for rg, energies in [(1., [.011, .013]), (9., [.003, .005])]]
    device.switch.e_on_data = [direct, *currents]
    device.switch.e_off_data = [replace(
        direct, graph_r_e=np.array([[1., 4., 9.], [.008, .004, 0.]]),
        evidence=MeasurementEvidence(filename="recorded.csv", sha256="a" * 64,
                                     original_base64="must never leave repository"),
    )]
    device.diode.e_rr_data = [replace(
        direct, v_g=None, graph_r_e=np.array([[1., 4., 9.], [.008, .002, 0.]]),
    ), replace(direct, v_g=0)]
    return device


def query(**updates):
    """Declare every physical selector, including null evidence fields."""
    return {"operation": "energy-at-rg", "branch": "e_on",
            "representation": "graph_r_e", "i_channel": 10., "t_j": 75.,
            "r_g": 1.5, "conditions": {
                "v_supply": 400., "v_g": 12., "v_g_off": None,
                "load_inductance": None, "commutation_inductance": None,
                "commutation_device": None, "convention": None,
                "event": None, "method": None}} | updates


def inverse(**updates):
    """Request one unique bounded inverse solution."""
    result = query(operation="rg-at-energy")
    result.pop("r_g")
    return result | {"e_target": .007} | updates


@pytest.fixture
def client():
    """Inject a repository lookup with a real missing-device response."""
    device = part()
    def read(name):
        if name != device.metadata.name:
            raise HTTPException(404, "missing fixture device")
        return device
    app = FastAPI()
    app.include_router(create_rg_energy_router(read))
    with TestClient(app) as session:
        yield session


URL = "/api/transistors/Browser%20Rg%20energy/rg-energy"


@pytest.mark.parametrize("payload,expected,unit", [
    (query(), .007, "J"), (inverse(), 1.5, "ohm"),
    (inverse(representation="graph_i_e", e_target=.008), 5., "ohm"),
    (query(representation="graph_i_e", r_g=9), .004, "J"),
])
def test_analytical_results_headless_and_http(client, payload, expected, unit):
    """Verify direct and inverse values against independent segment equations."""
    result = calculate_rg_energy(part(), payload)
    assert result["value"] == pytest.approx(expected)
    assert result["unit"] == unit
    response = client.post(URL, json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == result
    assert result["source_scope"] == "considered-branch"
    assert result["evidence_tag"] == "bounded-interpolation-estimate"


def test_inventory_locator_evidence_snapshot_hash_and_no_embedded_original(client):
    """Preserve real source evidence and distinguish normalized dependency hashes."""
    response = client.get(URL)
    assert response.status_code == 200, response.text
    inventory = response.json()
    assert len(inventory["curves"]) == 6
    first = inventory["curves"][0]
    assert first["source_id"] == "Browser Rg energy:switch.e_on_data[0]"
    assert first["source_kind"] == "repository-locator"
    assert first["source_filename"] is first["source_sha256"] is None
    assert first["coverage"] == {
        "axis": "r_g", "axis_unit": "ohm", "lower": 1., "upper": 9.,
        "energy_min": .002, "energy_max": .008, "sample_count": 3,
    }
    source = inventory["curves"][3]
    assert source["source_kind"] == "measurement-evidence"
    assert source["source_filename"] == "recorded.csv"
    assert source["source_sha256"] == "a" * 64
    result = client.post(URL, json=query()).json()
    for source in result["sources"]:
        digest = hashlib.sha256(json.dumps(source["snapshot"], sort_keys=True,
            allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        assert digest == source["snapshot_sha256"]
    assert "original_base64" not in response.text
    assert "must never leave repository" not in response.text


@pytest.mark.parametrize("payload", [
    inverse(e_target=.004), query(r_g=0), query(r_g=10), query(i_channel=11),
    query(t_j=76), query(representation="graph_i_e", r_g=5),
])
def test_native_helper_diagnostic_is_unchanged(client, payload):
    """Do not suppress bounded-domain and multiple-root diagnostics."""
    helper = (interpolate_energy_at_rg if payload["operation"] == "energy-at-rg"
              else find_rg_for_target_energy)
    target = payload.get("r_g", payload.get("e_target"))
    with pytest.raises(ValueError) as caught:
        helper(part().switch.e_on_data, target, payload["i_channel"], payload["t_j"],
               conditions=payload["conditions"], representation=payload["representation"])
    response = client.post(URL, json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == str(caught.value)


def test_duplicates_and_inverse_current_hole_remain_visible():
    """Pass the complete selected branch to the public helper."""
    device = part()
    device.switch.e_on_data.append(deepcopy(device.switch.e_on_data[0]))
    with pytest.raises(ValueError, match="unambiguous"):
        calculate_rg_energy(device, query())
    device = part()
    device.switch.e_on_data.append(replace(
        device.switch.e_on_data[1], r_g=5,
        graph_i_e=np.array([[15., 20.], [.006, .010]])))
    with pytest.raises(ValueError, match="supports"):
        calculate_rg_energy(device, inverse(representation="graph_i_e", e_target=.008))


def test_unknown_and_zero_gate_bias_and_zero_energy_are_distinct(client):
    """Preserve null selectors and exact measured zero results."""
    unknown = query(branch="e_rr", r_g=9)
    unknown["conditions"]["v_g"] = None
    assert client.post(URL, json=unknown).json()["value"] == 0
    grounded = deepcopy(unknown)
    grounded["conditions"]["v_g"] = 0
    assert client.post(URL, json=grounded).json()["value"] == .006
    supplied = deepcopy(unknown)
    supplied["conditions"]["v_g"] = 12
    assert client.post(URL, json=supplied).status_code == 422


def test_literal_unknown_method_is_not_missing_evidence(client):
    """A recorded method unknown remains literal categorical evidence."""
    missing = query(branch="e_off", r_g=9)
    assert client.post(URL, json=missing).status_code == 422
    missing["conditions"]["method"] = "unknown"
    assert client.post(URL, json=missing).json()["value"] == 0


@pytest.mark.parametrize("field,value", [
    ("r_g", True), ("r_g", "1.5"), ("r_g", -1), ("t_j", False),
    ("i_channel", None), ("operation", []), ("branch", []),
    ("representation", []), ("e_target", .002), ("unused", 1),
])
def test_strict_headless_and_http_queries(client, field, value):
    """Reject coercion, ignored fields and malformed selectors independently."""
    request = query() | {field: value}
    with pytest.raises(ValueError):
        calculate_rg_energy(part(), request)
    assert client.post(URL, json=request).status_code == 422


@pytest.mark.parametrize("name,value", [
    ("v_supply", False), ("v_g", "12"), ("load_inductance", -1),
    ("method", " "), ("convention", 0), ("event", "x" * 2001),
])
def test_strict_conditions(client, name, value):
    """Require finite numeric or exact categorical conditions."""
    request = query()
    request["conditions"][name] = value
    with pytest.raises(ValueError):
        calculate_rg_energy(part(), request)
    assert client.post(URL, json=request).status_code == 422


@pytest.mark.parametrize("name", list(query()["conditions"]))
def test_every_condition_must_be_declared(client, name):
    """Never turn a missing selector into a wildcard or a zero."""
    request = query()
    request["conditions"].pop(name)
    with pytest.raises(ValueError, match="all nine"):
        calculate_rg_energy(part(), request)
    assert client.post(URL, json=request).status_code == 422


@pytest.mark.parametrize("body", [b"{", b"\xff", br'{"operation":"\ud800"}',
    b'{"r_g":1e-400}', b'{"r_g":1e400}', b'{"r_g":NaN}'])
def test_invalid_json_and_finite_literal_limits(client, body):
    """Reject malformed JSON, invalid text and numerical underflow or overflow."""
    assert client.post(URL, content=body,
                       headers={"Content-Type": "application/json"}).status_code == 422


def test_request_size_and_missing_repository_device(client):
    """Preserve the body budget and injected missing-device response."""
    assert client.post(URL, content=b" " * (4 * 1024 * 1024 + 1)).status_code == 413
    assert client.get("/api/transistors/missing/rg-energy").status_code == 404
    assert client.post("/api/transistors/missing/rg-energy", json=query()).status_code == 404


def test_no_source_array_or_request_mutation():
    """Keep cached repository arrays and input objects unchanged."""
    device = part()
    original = deepcopy(device.switch.e_on_data)
    request = inverse(representation="graph_i_e", e_target=.008)
    saved = deepcopy(request)
    rg_energy_inventory(device)
    calculate_rg_energy(device, request)
    assert request == saved
    for record, previous in zip(device.switch.e_on_data, original, strict=True):
        for name in ("graph_i_e", "graph_r_e"):
            np.testing.assert_array_equal(getattr(record, name), getattr(previous, name))


def test_empty_repository_inventory_and_source_bounds():
    """Report absent curves without fabricating a default context."""
    device = part()
    device.switch.e_on_data = []
    device.switch.e_off_data = []
    device.diode.e_rr_data = []
    assert rg_energy_inventory(device)["curves"] == []
    device.switch.e_on_data = [SwitchingLossData(
        "graph_r_e", 75, 400, 12, i_x=10,
        graph_r_e=np.zeros((2, 20001)))]
    with pytest.raises(ValueError, match="20000 sample"):
        rg_energy_inventory(device)
    device = part()
    device.switch.e_on_data = [device.switch.e_on_data[0]] * 1001
    with pytest.raises(ValueError, match="1000 curves"):
        rg_energy_inventory(device)


def test_zero_resistance_current_and_temperature_are_not_missing():
    """Accept recorded physical zeros and a finite negative temperature."""
    device = part()
    record = replace(device.switch.e_on_data[0], i_x=0, t_j=-25,
                     graph_r_e=np.array([[0., 1.], [0., .002]]))
    device.switch.e_on_data = [record]
    result = calculate_rg_energy(device, query(i_channel=0, t_j=-25, r_g=0))
    assert result["value"] == 0
    assert result["sources"][0]["i_x"] == 0


def test_inventory_preview_is_bounded_and_snapshot_preserves_full_curve():
    """Mark preview thinning and retain every normalized replay sample."""
    device = part()
    axis = np.linspace(1., 9., 201)
    device.switch.e_on_data = [replace(
        device.switch.e_on_data[0], graph_r_e=np.array([axis, axis * .001]))]
    curve = rg_energy_inventory(device)["curves"][0]
    assert curve["preview"]["truncated"] is True
    assert len(curve["preview"]["axis"]) == 200
    assert curve["preview"]["axis"][0] == 1
    assert curve["preview"]["axis"][-1] == 9
    result = calculate_rg_energy(device, query())
    assert len(result["sources"][0]["snapshot"]["axis"]) == 201


def test_output_budget_checks_encoded_bytes(monkeypatch):
    """Apply the response budget to escaped source text and snapshots."""
    import transistordatabase.rg_energy_access as access
    monkeypatch.setattr(access, "MAX_OUTPUT_BYTES", 100)
    with pytest.raises(ValueError, match="24 MiB"):
        rg_energy_inventory(part())
    with pytest.raises(ValueError, match="24 MiB"):
        calculate_rg_energy(part(), query())


def test_nontransitive_context_ambiguity_survives_full_branch_access():
    """Do not merge endpoints that only match an intermediate request context."""
    device = part()
    direct = device.switch.e_on_data[0]
    device.switch.e_on_data = [replace(
        direct, dataset_type="graph_i_e", r_g=resistance,
        graph_i_e=np.array([[6., 14.], [energy - .001, energy + .001]]),
        commutation_inductance=inductance,
    ) for resistance, energy, inductance in (
        (1, .012, 10e-9), (5, .008, 10.75e-9), (9, .004, 11.5e-9))]
    request = inverse(representation="graph_i_e", e_target=.008)
    request["conditions"]["commutation_inductance"] = 10.75e-9
    with pytest.raises(ValueError, match="unambiguous physical"):
        calculate_rg_energy(device, request)


@pytest.mark.parametrize("change", [
    {"sha256": "invalid"}, {"filename": " "}, {"method": " "},
])
def test_invalid_recorded_evidence_is_not_published(change):
    """Disclose invalid evidence without fabricating a filename or source hash."""
    device = part()
    device.switch.e_on_data[0].evidence = MeasurementEvidence(**change)
    curve = rg_energy_inventory(device)["curves"][0]
    assert curve["diagnostic"]
    if "method" in change:
        assert curve["status"] == "unavailable"
        assert curve["conditions"] is None
        assert "conditions" in curve["unavailable_fields"]
        assert curve["snapshot_sha256"] is None
    else:
        assert curve["status"] == "available"
        assert curve["snapshot_sha256"]
        assert curve["source_filename"] is None
        assert curve["source_sha256"] is None
        request = query()
        request["conditions"]["method"] = "unknown"
        result = calculate_rg_energy(device, request)
        assert result["value"] == pytest.approx(.007)
        assert result["sources"][0]["snapshot"] is not None


def test_unrelated_unusable_source_remains_attributable_and_does_not_disable_math():
    """Keep an invalid unused row visible while the helper selects a healthy slice."""
    device = part()
    bad = replace(device.switch.e_on_data[0], t_j=100, graph_r_e=None,
                  evidence=MeasurementEvidence(original_base64="private file bytes"))
    device.switch.e_on_data.append(bad)
    inventory = rg_energy_inventory(device)
    assert len(inventory["curves"]) == 7
    row = inventory["curves"][3]
    assert row["status"] == "unavailable"
    assert row["source_id"] == "Browser Rg energy:switch.e_on_data[3]"
    assert row["t_j"] == 100
    assert row["conditions"]["method"] == "unknown"
    assert row["preview"] is row["coverage"] is row["snapshot_sha256"] is None
    assert row["diagnostic"]
    result = calculate_rg_energy(device, query())
    assert result["value"] == pytest.approx(.007)
    assert result["sources"][3]["snapshot"] is None
    assert result["sources"][3]["status"] == "unavailable"
    assert "private file bytes" not in json.dumps(result)
    same_context = query(t_j=100)
    same_context["conditions"]["method"] = "unknown"
    with pytest.raises(ValueError, match="real numeric source samples"):
        calculate_rg_energy(device, same_context)


def test_ragged_unused_graph_is_visible_without_numpy_coercion_failure():
    """Bound malformed source sequences before the native helper selects a slice."""
    device = part()
    device.switch.e_on_data.append(replace(
        device.switch.e_on_data[0], t_j=100, graph_r_e=[[1, 2], [.001]]))
    assert calculate_rg_energy(device, query())["value"] == pytest.approx(.007)
    row = rg_energy_inventory(device)["curves"][3]
    assert row["status"] == "unavailable"
    assert row["coverage"] is None


@pytest.mark.parametrize("representation,name", [("graph_r_e", "r_g"), ("graph_i_e", "i_x")])
def test_invalid_unused_scalar_retains_complete_math_proof(representation, name):
    """Disclose invalid non-dependencies without losing valid normalized replay data."""
    device = part()
    index = 0 if representation == "graph_r_e" else 2
    setattr(device.switch.e_on_data[index], name, -1)
    result = calculate_rg_energy(device, query(
        representation=representation, r_g=1.5 if index == 0 else 9))
    row = result["sources"][index]
    assert result["value"] == pytest.approx(.007 if index == 0 else .004)
    assert row["status"] == "available"
    assert name in row["unavailable_fields"]
    assert row[name] is row["snapshot"][name] is None
    assert row["snapshot_sha256"]


def test_invalid_unrelated_http_source_preserves_helper_error_for_queried_slice():
    """Serve valid estimates and attributable invalid inventory through typed transport."""
    device = part()
    device.switch.e_on_data.append(replace(device.switch.e_on_data[0], t_j=100,
                                           graph_r_e=None))
    app = FastAPI()
    app.include_router(create_rg_energy_router(lambda name: device))
    with TestClient(app) as session:
        response = session.get(URL)
        assert response.status_code == 200, response.text
        assert response.json()["curves"][3]["status"] == "unavailable"
        response = session.post(URL, json=query())
        assert response.status_code == 200, response.text
        assert response.json()["value"] == pytest.approx(.007)
        assert response.json()["sources"][3]["snapshot"] is None
        failed = session.post(URL, json=query(t_j=100))
        assert failed.status_code == 422
        assert failed.json()["detail"] == "graph_r_e requires real numeric source samples"
