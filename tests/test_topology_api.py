"""FastAPI contract checks for the new topology workflow."""
from __future__ import annotations

from typing import Any

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
from fastapi.testclient import TestClient  # noqa: E402

import transistordatabase.gui_web.backend.main as backend  # noqa: E402


class InMemoryRepo:
    """Minimal repository stub for the topology API tests."""

    def __init__(self: InMemoryRepo, devices: dict[str, Transistor]) -> None:
        self.devices = devices

    def get_by_name(self: InMemoryRepo, name: str) -> Transistor | None:
        """Return the named fixture device, or ``None`` when absent."""
        return self.devices.get(name)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create an API client backed by an in-memory device repository."""
    device = Transistor(
        TransistorMetadata("bare-device", "MOSFET", "test", "vendor", "package"),
        ElectricalRatings(1200, 100, 50, 175),
        ThermalProperties(0.001, 0.001),
    )
    monkeypatch.setattr(backend, "_repo", InMemoryRepo({"bare-device": device}))
    return TestClient(backend.app)


@pytest.fixture
def gate_conditioned_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
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


def payload(**changes: Any) -> dict[str, Any]:
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


def test_openapi_describes_flyback_as_a_supported_topology(client: TestClient) -> None:
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


def test_calculate_endpoint_uses_si_contract_and_matches_core(client: TestClient) -> None:
    """Return unit-consistent operating values from the API."""
    response = client.post("/api/topology/calculate", json=payload())
    assert response.status_code == 200
    result = response.json()
    assert result["calculated"]["duty_cycle"] == pytest.approx(0.5)
    assert result["calculated"]["i_in_avg_a"] == pytest.approx(5)
    assert result["calculated"]["i_l_ripple_pp_a"] == pytest.approx(12.5)
    assert result["model"]["mode"] == "estimate"


def test_calculate_endpoint_requires_explicit_mode(client: TestClient) -> None:
    """Reject requests that do not explicitly choose a model mode."""
    body = payload()
    body.pop("mode")
    response = client.post("/api/topology/calculate", json=body)
    assert response.status_code == 422


def test_calculate_endpoint_requires_flyback_ratio_and_calculates_when_supplied(client: TestClient) -> None:
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


def test_calculate_endpoint_accepts_explicit_dcm_regime(client: TestClient) -> None:
    """Return supported DCM duty and an idle interval from the reusable core."""
    response = client.post(
        "/api/topology/calculate",
        json=payload(conduction_regime="DCM", p_out=100),
    )
    assert response.status_code == 200
    result = response.json()
    assert result["model"]["conduction_regime"] == "DCM"
    assert result["calculated"]["idle_fraction"] > 0


def test_detailed_mode_does_not_swallow_missing_curves_or_device_ids(client: TestClient) -> None:
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


def test_conduction_only_mode_requires_device_channels_at_api_boundary(client: TestClient) -> None:
    """Do not allow the curve-backed Pareto mode to fall back to an estimate."""
    response = client.post("/api/topology/calculate", json=payload(
        mode="conduction-only", t_j=25, v_g=15,
        switch_transistor_id="bare-device", diode_transistor_id="bare-device",
    ))
    assert response.status_code == 400
    assert "Missing required switch channel curve" in response.json()["detail"]


def test_loss_balanced_boost_api_returns_power_closed_channel_result(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose the direct Boost CCM solver and its channel-only scope via HTTP."""
    request = payload(
        topology="boost",
        mode="loss-balanced",
        v_in=100,
        v_out=200,
        p_out=1000,
        f_sw=20_000,
        l_value=0.001,
        c_out=0.001,
        t_j=25,
        v_g=15,
        diode_v_g=0,
        switch_transistor_id="gated-device",
        diode_transistor_id="gated-device",
    )
    response = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["model"]["mode"] == "loss-balanced"
    assert result["losses"]["loss_scope"] == "channel_conduction_only"
    assert 0 < result["calculated"]["inductor_volt_second_residual_v"] < 0.01
    assert result["performance"]["p_in_w"] == pytest.approx(
        100 * result["calculated"]["i_in_avg_a"]
    )
    assert result["performance"]["p_in_w"] == pytest.approx(
        1000 + result["losses"]["total_w"], rel=1e-7
    )


def test_loss_balanced_api_rejects_unsupported_regime(client: TestClient) -> None:
    """Return a clear client error for unsupported loss-balanced DCM."""
    response = client.post(
        "/api/topology/calculate",
        json=payload(
            topology="buck-boost", mode="loss-balanced", conduction_regime="DCM"
        ),
    )
    assert response.status_code == 400
    assert "Buck and Boost only" in response.json()["detail"]


def test_loss_balanced_buck_dcm_api_exposes_both_balance_residuals(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose Buck DCM current ramps and balances through the SI API contract."""
    request = payload(
        topology="buck", mode="loss-balanced", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, f_sw=20_000,
        l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
        switch_transistor_id="gated-device", diode_transistor_id="gated-device",
    )
    response = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert response.status_code == 200
    result = response.json()
    calculated = result["calculated"]
    assert result["model"]["conduction_regime"] == "DCM"
    assert result["losses"]["loss_scope"] == "channel_conduction_only"
    assert calculated["idle_fraction"] > 0
    assert calculated["switch_conduction_fraction"] + calculated[
        "diode_conduction_fraction"
    ] + calculated["idle_fraction"] == pytest.approx(1)
    assert calculated["i_l_avg_a"] == pytest.approx(
        calculated["i_out_avg_a"]
    )
    assert calculated["i_diode_avg_a"] < calculated["i_out_avg_a"]
    assert abs(calculated["inductor_volt_second_residual_v"]) < 1e-7
    assert abs(calculated["power_balance_residual_w"]) < 1e-7
    assert result["performance"]["p_in_w"] == pytest.approx(
        request["v_in"] * calculated["i_in_avg_a"]
    )


def test_loss_balanced_boost_dcm_api_exposes_both_balance_residuals(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose channel-only Boost DCM feedback and both independent balances."""
    request = payload(
        topology="boost", mode="loss-balanced", conduction_regime="DCM",
        v_in=100, v_out=200, p_out=100, f_sw=20_000,
        l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
        switch_transistor_id="gated-device", diode_transistor_id="gated-device",
    )
    response = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert response.status_code == 200
    result = response.json()
    calculated = result["calculated"]
    assert result["model"]["conduction_regime"] == "DCM"
    assert result["losses"]["loss_scope"] == "channel_conduction_only"
    assert calculated["idle_fraction"] > 0
    assert calculated["switch_conduction_fraction"] + calculated[
        "diode_conduction_fraction"
    ] + calculated["idle_fraction"] == pytest.approx(1)
    assert abs(calculated["inductor_volt_second_residual_v"]) < 1e-8
    assert abs(calculated["power_balance_residual_w"]) < 1e-6
    assert result["performance"]["p_in_w"] == pytest.approx(
        request["v_in"] * calculated["i_in_avg_a"]
    )


def test_loss_balanced_boost_dcm_api_rejects_target_at_ccm_boundary(
    gate_conditioned_client: TestClient,
) -> None:
    """Return an actionable API error when the requested point is not DCM."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="boost", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=200, p_out=150, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        ),
    )
    assert response.status_code == 400
    assert "at or above the CCM boundary" in response.json()["detail"]


def test_loss_balanced_buck_dcm_api_rejects_sweeps(
    gate_conditioned_client: TestClient,
) -> None:
    """Keep Buck DCM scalar-only at the FastAPI boundary."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="buck", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
            sweep={
                "x_axis": "f_sw", "x_min": 20_000, "x_max": 30_000,
                "x_points": 2, "metric": "efficiency_percent",
            },
        ),
    )
    assert response.status_code == 400
    assert "Loss-balanced topology sweeps support CCM only" in response.json()["detail"]


def test_api_requires_and_passes_gate_condition_for_diode_channel_curves(gate_conditioned_client: TestClient) -> None:
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


def test_line_sweep_is_generated_and_serialized_by_backend(client: TestClient) -> None:
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


def test_loss_balanced_api_returns_line_sweep_scope_and_points(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose loss-balanced line values and model limits in the API response."""
    request = payload(
        topology="boost", mode="loss-balanced", v_in=100, v_out=200,
        p_out=1000, f_sw=20_000, l_value=1e-3, c_out=1e-3,
        t_j=25, v_g=15, diode_v_g=0,
        switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        sweep={"x_axis": "f_sw", "x_min": 20_000, "x_max": 30_000,
               "x_points": 3, "metric": "p_loss_total_w"},
    )
    response = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert response.status_code == 200
    result = response.json()
    sweep = result["sweep"]
    assert sweep["kind"] == "line"
    assert sweep["x_unit"] == "Hz"
    assert sweep["loss_scope"] == "channel_conduction_only"
    assert "volt-second residual" in sweep["voltage_law_note"]
    assert len(sweep["values"]) == 3
    assert len(sweep["voltage_law_residual_values_v"]) == 3
    assert all(value is not None for value in sweep["voltage_law_residual_values_v"])

    contour_response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json={
            **request,
            "sweep": {
                "x_axis": "f_sw", "x_min": 20_000, "x_max": 30_000,
                "x_points": 3, "y_axis": "l_value", "y_min": 0.001,
                "y_max": 0.002, "y_points": 2, "metric": "efficiency_percent",
            },
        },
    )
    assert contour_response.status_code == 200
    contour = contour_response.json()["sweep"]
    assert contour["kind"] == "contour"
    assert len(contour["values"]) == 2
    assert all(len(row) == 3 for row in contour["values"])
    assert len(contour["voltage_law_residual_values_v"]) == 2
    assert all(len(row) == 3 for row in contour["voltage_law_residual_values_v"])
    assert contour["loss_scope"] == "channel_conduction_only"


def test_loss_balanced_api_reports_sweep_curve_failure_coordinate(
    gate_conditioned_client: TestClient,
) -> None:
    """Return a client error naming the unsupported coordinate and curve range."""
    request = payload(
        topology="boost", mode="loss-balanced", v_in=100, v_out=200,
        p_out=1000, f_sw=20_000, l_value=1e-3, c_out=1e-3,
        t_j=25, v_g=15, diode_v_g=0,
        switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        sweep={"x_axis": "p_out", "x_min": 1000, "x_max": 100_000,
               "x_points": 2, "metric": "efficiency_percent"},
    )
    response = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert response.status_code == 400
    assert "Sweep point (x=100000 W)" in response.json()["detail"]
    assert "curve supports" in response.json()["detail"]


def test_loss_balanced_comparison_returns_scope_with_efficiency_ranking(
    gate_conditioned_client: TestClient,
) -> None:
    """Label the partial-loss scope when the comparison ranking is available."""
    response = gate_conditioned_client.post(
        "/api/topology/compare",
        json={
            "parameters": payload(
                topology="boost", mode="loss-balanced", v_in=100, v_out=200,
                p_out=1000, f_sw=20_000, l_value=1e-3, c_out=1e-3,
                t_j=25, v_g=15, diode_v_g=0,
            ),
            "switch_transistor_id": "gated-device",
            "diode_transistor_id": "gated-device",
        },
    )
    assert response.status_code == 200
    comparison = response.json()["comparison"]
    assert comparison["ranking_available"] is True
    assert comparison["loss_scope"] == "channel_conduction_only"


def test_loss_balanced_dcm_topology_comparison_is_rejected(
    gate_conditioned_client: TestClient,
) -> None:
    """Do not rank scalar Buck/Boost DCM results across topologies."""
    response = gate_conditioned_client.post(
        "/api/topology/compare",
        json={
            "parameters": payload(
                topology="boost", mode="loss-balanced", conduction_regime="DCM",
                v_in=100, v_out=200, p_out=100, f_sw=20_000,
                l_value=1e-3, c_out=1e-3, t_j=25, v_g=15, diode_v_g=0,
            ),
            "switch_transistor_id": "gated-device",
            "diode_transistor_id": "gated-device",
        },
    )
    assert response.status_code == 400
    assert "does not support loss-balanced DCM" in response.json()["detail"]


def test_estimate_comparison_does_not_rank_unmodeled_efficiencies(client: TestClient) -> None:
    """Do not rank ideal estimates by fabricated efficiency values."""
    response = client.post("/api/topology/compare", json={"parameters": payload()})
    assert response.status_code == 200
    comparison = response.json()["comparison"]
    assert comparison["ranking_available"] is False
    assert comparison["best_topology"] is None


def test_conduction_only_comparison_explains_missing_curves_without_claiming_estimate(client: TestClient) -> None:
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
