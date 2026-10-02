"""FastAPI contract checks for the new topology workflow."""
from __future__ import annotations

import pytest
import numpy as np

from transistordatabase.core.models import (
    ChannelCharacteristics,
    ElectricalRatings,
    ThermalProperties,
    Transistor,
    TransistorMetadata,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import transistordatabase.gui_web.backend.main as backend


class InMemoryRepo:
    """Minimal repository stub for the topology API tests."""

    def __init__(self, devices):
        self.devices = devices

    def get_by_name(self, name):
        """Return the named fixture device, or ``None`` when absent."""
        return self.devices.get(name)


@pytest.fixture
def client(monkeypatch):
    """Create an API client backed by an in-memory device repository."""
    device = Transistor(
        TransistorMetadata("bare-device", "MOSFET", "test", "vendor", "package"),
        ElectricalRatings(1200, 100, 50, 175),
        ThermalProperties(0.001, 0.001),
    )
    monkeypatch.setattr(backend, "_repo", InMemoryRepo({"bare-device": device}))
    return TestClient(backend.app)


@pytest.fixture
def gate_conditioned_client(monkeypatch):
    """Expose exact switch and gate-conditioned diode curves to the API."""
    device = Transistor(
        TransistorMetadata("gated-device", "MOSFET", "test", "vendor", "package"),
        ElectricalRatings(1200, 100, 50, 175),
        ThermalProperties(0.001, 0.001),
    )
    switch_graph = np.array([[0.01, 10.0], [0.0, 100.0]])
    diode_graph = np.array([[0.4, 5.4], [0.0, 100.0]])
    device.switch.channel_data.append(ChannelCharacteristics(25, switch_graph, 15))
    device.diode.channel_data.append(ChannelCharacteristics(25, diode_graph, 0))
    monkeypatch.setattr(backend, "_repo", InMemoryRepo({"gated-device": device}))
    return TestClient(backend.app)


def payload(**changes):
    """Return a valid baseline SI request with selected overrides."""
    body = {
        "topology": "buck",
        "mode": "estimate",
        "v_in": 100,
        "v_out": 50,
        "p_out": 500,
        "f_sw": 10000,
        "l_value": 0.0002,
        "c_out": 0.0001,
    }
    body.update(changes)
    return body


def test_openapi_describes_flyback_as_a_supported_topology(client):
    """Keep generated API docs aligned with the implemented topology set."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    request_schema = response.json()["components"]["schemas"][
        "TopologyCalculationRequest"
    ]["properties"]
    topology_description = request_schema["topology"]["description"]
    assert "buck, boost, buck-boost, or flyback" in topology_description
    assert "other topologies are unsupported" in topology_description
    assert "switch turn-on/turn-off losses" in request_schema["mode"]["description"]


def test_calculate_endpoint_uses_si_contract_and_matches_core(client):
    """Return unit-consistent operating values from the API."""
    response = client.post("/api/topology/calculate", json=payload())
    assert response.status_code == 200
    result = response.json()
    assert result["calculated"]["duty_cycle"] == pytest.approx(0.5)
    assert result["calculated"]["i_in_avg_a"] == pytest.approx(5)
    assert result["calculated"]["i_l_ripple_pp_a"] == pytest.approx(12.5)
    assert result["model"]["mode"] == "estimate"


def test_calculate_endpoint_requires_explicit_mode(client):
    """Reject requests that do not explicitly choose a model mode."""
    body = payload()
    body.pop("mode")
    response = client.post("/api/topology/calculate", json=body)
    assert response.status_code == 422


def test_calculate_endpoint_requires_flyback_ratio_and_calculates_when_supplied(client):
    """Require an explicit turns ratio and return the isolated operating point."""
    response = client.post("/api/topology/calculate", json=payload(topology="flyback"))
    assert response.status_code == 400
    assert "Flyback requires a positive turns_ratio_nps" in response.json()["detail"]

    response = client.post(
        "/api/topology/calculate",
        json=payload(topology="flyback", turns_ratio_nps=2, conduction_regime="DCM"),
    )
    assert response.status_code == 200
    assert response.json()["model"]["conduction_regime"] == "DCM"
    assert response.json()["calculated"]["duty_cycle"] > 0


def test_calculate_endpoint_accepts_explicit_dcm_regime(client):
    """Return supported DCM duty and an idle interval from the reusable core."""
    response = client.post(
        "/api/topology/calculate",
        json=payload(conduction_regime="DCM", p_out=100),
    )
    assert response.status_code == 200
    result = response.json()
    assert result["model"]["conduction_regime"] == "DCM"
    assert result["calculated"]["idle_fraction"] > 0


def test_detailed_mode_does_not_swallow_missing_curves_or_device_ids(client):
    """Reject missing detailed curves and preserve missing-device 404s."""
    response = client.post("/api/topology/calculate", json=payload(
        mode="detailed", t_j=25, v_g=15,
        switch_transistor_id="bare-device", diode_transistor_id="bare-device",
    ))
    assert response.status_code == 400
    assert "Missing required switch channel curve" in response.json()["detail"]

    missing_device = client.post("/api/topology/calculate", json=payload(
        switch_transistor_id="missing-device",
    ))
    assert missing_device.status_code == 404
    assert "missing-device" in missing_device.json()["detail"]


def test_conduction_only_mode_requires_device_channels_at_api_boundary(client):
    """Do not allow the curve-backed Pareto mode to fall back to an estimate."""
    response = client.post("/api/topology/calculate", json=payload(
        mode="conduction-only", t_j=25, v_g=15,
        switch_transistor_id="bare-device", diode_transistor_id="bare-device",
    ))
    assert response.status_code == 400
    assert "Missing required switch channel curve" in response.json()["detail"]


def test_api_requires_and_passes_gate_condition_for_diode_channel_curves(gate_conditioned_client):
    """Require explicit diode gate selection and return its curve provenance."""
    request = payload(
        mode="conduction-only", t_j=25, v_g=15,
        switch_transistor_id="gated-device", diode_transistor_id="gated-device",
    )
    missing = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert missing.status_code == 400
    assert "provide diode_v_g" in missing.json()["detail"]
    assert "0 V" in missing.json()["detail"]

    selected = gate_conditioned_client.post(
        "/api/topology/calculate", json={**request, "diode_v_g": 0}
    )
    assert selected.status_code == 200
    assert selected.json()["model"]["curve_conditions"]["diode_channel"] == {
        "t_j": 25,
        "v_g": 0,
    }


def test_line_sweep_is_generated_and_serialized_by_backend(client):
    """Return Python-generated sweep points in the API response."""
    response = client.post("/api/topology/calculate", json=payload(sweep={
        "x_axis": "f_sw", "x_min": 10000, "x_max": 20000,
        "x_points": 3, "metric": "i_l_ripple_pp_a",
    }))
    assert response.status_code == 200
    sweep = response.json()["sweep"]
    assert sweep["kind"] == "line"
    assert sweep["x_unit"] == "Hz"
    assert sweep["values"][0] == pytest.approx(12.5)
    assert sweep["values"][-1] == pytest.approx(6.25)


def test_estimate_comparison_does_not_rank_unmodeled_efficiencies(client):
    """Do not rank ideal estimates by fabricated efficiency values."""
    response = client.post("/api/topology/compare", json={"parameters": payload()})
    assert response.status_code == 200
    comparison = response.json()["comparison"]
    assert comparison["ranking_available"] is False
    assert comparison["best_topology"] is None


def test_conduction_only_comparison_explains_missing_curves_without_claiming_estimate(client):
    """Report the loss scope and accurate reason when no channel data is available."""
    response = client.post("/api/topology/compare", json={
        "parameters": payload(mode="conduction-only"),
        "switch_transistor_id": "bare-device",
        "diode_transistor_id": "bare-device",
    })
    assert response.status_code == 200
    comparison = response.json()["comparison"]
    assert comparison["ranking_available"] is False
    assert comparison["loss_scope"] == "channel_conduction_only"
    assert "required device curves" in comparison["reason"]
    assert "Estimate mode" not in comparison["reason"]
