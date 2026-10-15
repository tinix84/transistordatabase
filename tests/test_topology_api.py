"""FastAPI contract checks for the new topology workflow."""
from __future__ import annotations

import json
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


def test_boost_pareto_api_keeps_candidate_errors_local(
    gate_conditioned_client: TestClient,
) -> None:
    """Return candidate-specific curve and lookup failures beside valid rows.

    :param gate_conditioned_client: API client with exact-condition TDB curves.
    :type gate_conditioned_client: TestClient
    :returns: None.
    :rtype: None
    """
    request = {
        "boundary": {
            "v_in": 100,
            "v_out": 200,
            "p_out": 1000,
            "f_sw": 20_000,
            "l_value": 0.001,
            "c_out": 0.001,
            "t_j": 25,
            "t_case": 60,
        },
        "candidates": [
            {
                "candidate_id": "valid",
                "label": "Valid device pair",
                "switch_transistor_id": "gated-device",
                "diode_transistor_id": "gated-device",
                "parallel_switches": 1,
                "parallel_diodes": 1,
                "v_g": 15,
                "diode_v_g": 0,
            },
            {
                "candidate_id": "wrong-gate",
                "label": "Wrong exact switch gate",
                "switch_transistor_id": "gated-device",
                "diode_transistor_id": "gated-device",
                "v_g": 14,
                "diode_v_g": 0,
            },
            {
                "candidate_id": "missing-device",
                "label": "Missing TDB record",
                "switch_transistor_id": "not-in-database",
                "diode_transistor_id": "gated-device",
                "v_g": 15,
                "diode_v_g": 0,
            },
        ],
        "catalog": {
            "gated-device": {
                "orderable_part": "GATED-DEVICE-1",
                "unit_cost_usd": 5.0,
                "length_mm": 10,
                "width_mm": 5,
                "height_mm": 2,
                "price_source_url": "https://example.com/price",
                "cost_accessed_on": "2026-10-14",
                "dimensions_source_url": "https://example.com/datasheet",
            }
        },
    }

    response = gate_conditioned_client.post("/api/topology/pareto", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["eligible_candidate_count"] == 1
    assert result["pareto_candidate_ids"] == ["valid"]
    candidates = {item["candidate_id"]: item for item in result["candidates"]}
    assert candidates["valid"]["eligible"] is True
    assert candidates["valid"]["catalog"]["switch"]["cost_accessed_on"] == "2026-10-14"
    assert candidates["wrong-gate"]["eligible"] is False
    assert any(
        "Missing required switch channel curve" in reason
        for reason in candidates["wrong-gate"]["ineligible_reasons"]
    )
    assert candidates["missing-device"]["eligible"] is False
    assert any(
        "not-in-database" in reason
        for reason in candidates["missing-device"]["ineligible_reasons"]
    )
    assert "Exact objective ties remain non-dominated." in result["dominance_rule"]


def test_boost_pareto_api_keeps_malformed_catalog_rows_local(
    gate_conditioned_client: TestClient,
) -> None:
    """Keep malformed market data from rejecting a mixed candidate request.

    :param gate_conditioned_client: API client with exact-condition TDB curves.
    :type gate_conditioned_client: TestClient
    :returns: None.
    :rtype: None
    """
    request = {
        "boundary": {
            "v_in": 100,
            "v_out": 200,
            "p_out": 1000,
            "f_sw": 20_000,
            "l_value": 0.001,
            "c_out": 0.001,
            "t_j": 25,
            "t_case": 60,
        },
        "candidates": [
            {
                "candidate_id": "valid",
                "label": "Valid sourced catalog",
                "switch_transistor_id": "gated-device",
                "diode_transistor_id": "gated-device",
                "v_g": 15,
                "diode_v_g": 0,
            },
            {
                "candidate_id": "malformed-catalog",
                "label": "Malformed sourced catalog",
                "switch_transistor_id": "missing-catalog-device",
                "diode_transistor_id": "missing-catalog-device",
                "v_g": 15,
                "diode_v_g": 0,
            },
        ],
        "catalog": {
            "gated-device": {
                "orderable_part": "GATED-DEVICE-1",
                "unit_cost_usd": 5,
                "length_mm": 10,
                "width_mm": 5,
                "height_mm": 2,
                "price_source_url": "https://example.com/price",
                "cost_accessed_on": "2026-10-14",
                "dimensions_source_url": "https://example.com/datasheet",
            },
            "missing-catalog-device": {
                "orderable_part": "BAD-DEVICE-1",
                "unit_cost_usd": 0,
                "length_mm": -1,
                "width_mm": 5,
                "height_mm": 2,
                "price_source_url": "not-a-url",
                "cost_accessed_on": "not-a-date",
                "dimensions_source_url": "ftp://example.com/datasheet",
            },
        },
    }

    response = gate_conditioned_client.post("/api/topology/pareto", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["eligible_candidate_count"] == 1
    assert result["pareto_candidate_ids"] == ["valid"]
    candidates = {item["candidate_id"]: item for item in result["candidates"]}
    assert candidates["valid"]["eligible"] is True
    malformed = candidates["malformed-catalog"]
    assert malformed["eligible"] is False
    reasons = " ".join(malformed["ineligible_reasons"])
    assert "unit_cost_usd" in reasons
    assert "length_mm" in reasons
    assert "price_source_url" in reasons
    assert "cost_accessed_on" in reasons
    assert "dimensions_source_url" in reasons


def test_boost_pareto_api_keeps_untyped_catalog_values_local(
    gate_conditioned_client: TestClient,
) -> None:
    """Screen valid alternatives when one catalog row has untyped values.

    :param gate_conditioned_client: API client with exact-condition TDB curves.
    :type gate_conditioned_client: TestClient
    :returns: None.
    :rtype: None
    """
    backend._repo.devices["malformed-types-device"] = backend._repo.devices[
        "gated-device"
    ]
    backend._repo.devices["non-object-catalog-device"] = backend._repo.devices[
        "gated-device"
    ]
    request = {
        "boundary": {
            "v_in": 100,
            "v_out": 200,
            "p_out": 1000,
            "f_sw": 20_000,
            "l_value": 0.001,
            "c_out": 0.001,
            "t_j": 25,
            "t_case": 60,
        },
        "candidates": [
            {
                "candidate_id": "valid",
                "label": "Valid catalog",
                "switch_transistor_id": "gated-device",
                "diode_transistor_id": "gated-device",
                "v_g": 15,
                "diode_v_g": 0,
            },
            {
                "candidate_id": "malformed-types",
                "label": "Malformed catalog values",
                "switch_transistor_id": "malformed-types-device",
                "diode_transistor_id": "malformed-types-device",
                "v_g": 15,
                "diode_v_g": 0,
            },
            {
                "candidate_id": "non-object-catalog",
                "label": "Non-object catalog row",
                "switch_transistor_id": "non-object-catalog-device",
                "diode_transistor_id": "non-object-catalog-device",
                "v_g": 15,
                "diode_v_g": 0,
            },
        ],
        "catalog": {
            "gated-device": {
                "orderable_part": "GATED-DEVICE-1",
                "unit_cost_usd": 5,
                "length_mm": 10,
                "width_mm": 5,
                "height_mm": 2,
                "price_source_url": "https://example.com/price",
                "cost_accessed_on": "2026-10-14",
                "dimensions_source_url": "https://example.com/datasheet",
            },
            "malformed-types-device": {
                "orderable_part": "BAD-DEVICE-1",
                "unit_cost_usd": "unknown",
                "length_mm": 1e309,
                "width_mm": 5,
                "height_mm": 2,
                "price_source_url": 123,
                "cost_accessed_on": "2026-10-14",
                "dimensions_source_url": "",
            },
            "non-object-catalog-device": 123,
        },
    }
    # Preserve a JSON numeric overflow token so the request exercises Pydantic's
    # handling of untyped 1e309 as a non-finite float.
    request_json = json.dumps(request).replace(
        '"length_mm": Infinity', '"length_mm": 1e309'
    )

    response = gate_conditioned_client.post(
        "/api/topology/pareto",
        content=request_json,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["eligible_candidate_count"] == 1
    assert result["pareto_candidate_ids"] == ["valid"]
    candidates = {item["candidate_id"]: item for item in result["candidates"]}
    assert candidates["valid"]["eligible"] is True
    malformed = candidates["malformed-types"]
    assert malformed["eligible"] is False
    reasons = " ".join(malformed["ineligible_reasons"])
    assert "unit_cost_usd" in reasons
    assert "length_mm" in reasons
    assert "price_source_url" in reasons
    assert "package-dimensions source URL" in reasons
    non_object = candidates["non-object-catalog"]
    assert non_object["eligible"] is False
    assert any(
        "No sourced package and cost data" in reason
        for reason in non_object["ineligible_reasons"]
    )


def test_boost_pareto_api_rejects_invalid_boundary_and_duplicate_ids(
    gate_conditioned_client: TestClient,
) -> None:
    """Validate shared operating bounds and stable candidate identifiers.

    :param gate_conditioned_client: API client with exact-condition TDB curves.
    :type gate_conditioned_client: TestClient
    :returns: None.
    :rtype: None
    """
    candidate = {
        "candidate_id": "same-id",
        "label": "Candidate",
        "switch_transistor_id": "gated-device",
        "diode_transistor_id": "gated-device",
        "v_g": 15,
        "diode_v_g": 0,
    }
    request = {
        "boundary": {
            "v_in": 100,
            "v_out": 90,
            "p_out": 1000,
            "f_sw": 20_000,
            "l_value": 0.001,
            "c_out": 0.001,
            "t_j": 25,
        },
        "candidates": [candidate, {**candidate, "candidate_id": "other"}],
        "catalog": {},
    }
    invalid_boundary = gate_conditioned_client.post(
        "/api/topology/pareto", json=request
    )
    assert invalid_boundary.status_code == 422
    assert "v_out > v_in" in invalid_boundary.text

    request["boundary"]["v_out"] = 200
    request["candidates"][1] = {
        **candidate,
        "candidate_id": " same-id ",
        "label": "Duplicate id",
    }
    duplicate_ids = gate_conditioned_client.post("/api/topology/pareto", json=request)
    assert duplicate_ids.status_code == 422
    assert "candidate_id values must be unique" in duplicate_ids.text


def test_loss_balanced_flyback_dcm_api_exposes_primary_and_secondary_currents(
    gate_conditioned_client: TestClient,
) -> None:
    """Return the ratio-scaled scalar Flyback DCM result through FastAPI."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="flyback", mode="loss-balanced", conduction_regime="DCM",
            turns_ratio_nps=2, v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        ),
    )
    assert response.status_code == 200
    result = response.json()
    calculated = result["calculated"]
    assert result["topology"] == "flyback"
    assert result["model"]["conduction_regime"] == "DCM"
    assert calculated["i_diode_peak_a"] == pytest.approx(
        2 * calculated["i_l_peak_a"]
    )
    assert calculated["i_diode_avg_a"] == pytest.approx(
        calculated["i_out_avg_a"], rel=1e-8
    )
    assert calculated["idle_fraction"] > 0
    assert abs(calculated["inductor_volt_second_residual_v"]) < 1e-7
    assert abs(calculated["power_balance_residual_w"]) < 1e-6
    assert result["losses"]["loss_scope"] == "channel_conduction_only"


def test_loss_balanced_flyback_dcm_api_rejects_missing_ratio_and_curve_condition(
    gate_conditioned_client: TestClient,
) -> None:
    """Return actionable errors for a missing ratio or exact TDB curve."""
    missing_ratio = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="flyback", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        ),
    )
    assert missing_ratio.status_code == 400
    assert "Flyback requires a positive turns_ratio_nps" in missing_ratio.json()["detail"]
    wrong_condition = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="flyback", mode="loss-balanced", conduction_regime="DCM",
            turns_ratio_nps=2, v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=14, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        ),
    )
    assert wrong_condition.status_code == 400
    assert "Missing required switch channel curve" in wrong_condition.json()["detail"]


def test_loss_balanced_flyback_dcm_api_sweeps_and_rejects_invalid_ratio(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose Flyback DCM sweep results and reject invalid ratios."""
    request = payload(
        topology="flyback", mode="loss-balanced", conduction_regime="DCM",
        turns_ratio_nps=2, v_in=100, v_out=50, p_out=10, f_sw=20_000,
        l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
        switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        sweep={
            "x_axis": "f_sw", "x_min": 20_000, "x_max": 21_000,
            "x_points": 2, "metric": "efficiency_percent",
        },
    )
    sweep_response = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert sweep_response.status_code == 200
    sweep = sweep_response.json()["sweep"]
    assert sweep["mode"] == "loss-balanced"
    assert sweep["conduction_regime"] == "DCM"
    assert sweep["loss_scope"] == "channel_conduction_only"
    assert len(sweep["values"]) == 2
    assert len(sweep["voltage_law_residual_values_v"]) == 2
    assert len(sweep["power_balance_residual_values_w"]) == 2
    assert "CCM boundary" in sweep["voltage_law_note"]
    request.pop("sweep")
    request["turns_ratio_nps"] = 0
    invalid_ratio = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert invalid_ratio.status_code == 422


def test_loss_balanced_buck_boost_dcm_api_exposes_inverting_scalar_result(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose inverting Buck-Boost DCM current ramps and balances through API."""
    request = payload(
        topology="buck-boost", mode="loss-balanced", conduction_regime="DCM",
        v_in=100, v_out=50, p_out=10, f_sw=20_000,
        l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
        switch_transistor_id="gated-device", diode_transistor_id="gated-device",
    )
    response = gate_conditioned_client.post("/api/topology/calculate", json=request)
    assert response.status_code == 200
    result = response.json()
    calculated = result["calculated"]
    assert result["topology"] == "buck-boost"
    assert result["model"]["conduction_regime"] == "DCM"
    assert result["model"]["output_polarity"] == -1
    assert calculated["v_out_signed_v"] == -request["v_out"]
    assert calculated["idle_fraction"] > 0
    assert calculated["i_l_avg_a"] > calculated["i_in_avg_a"]
    assert calculated["i_diode_avg_a"] == pytest.approx(
        calculated["i_out_avg_a"], rel=1e-8
    )
    assert abs(calculated["inductor_volt_second_residual_v"]) < 1e-7
    assert abs(calculated["power_balance_residual_w"]) < 1e-6
    assert result["losses"]["loss_scope"] == "channel_conduction_only"
    assert result["performance"]["p_in_w"] == pytest.approx(
        request["v_in"] * calculated["i_in_avg_a"], rel=1e-8
    )


def test_loss_balanced_buck_boost_dcm_api_rejects_wrong_exact_gate_condition(
    gate_conditioned_client: TestClient,
) -> None:
    """Report a missing exact switch gate curve as an actionable API error."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="buck-boost", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=14, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        ),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Missing required switch channel curve at t_j=25 C, v_g=14 V"
    )


def test_loss_balanced_buck_boost_dcm_api_rejects_insufficient_curve_range(
    gate_conditioned_client: TestClient,
) -> None:
    """Return the exact curve-range limitation when peak current is unsupported."""
    device = backend._repo.get_by_name("gated-device")
    assert device is not None
    device.switch.channel_data[0].graph_v_i[1, 1] = 0.1
    device.diode.channel_data[0].graph_v_i[1, 1] = 0.1

    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="buck-boost", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        ),
    )
    assert response.status_code == 400
    assert "No supported Buck-Boost DCM solution inside the exact channel-curve range" in response.json()["detail"]

def test_loss_balanced_buck_boost_dcm_api_returns_sweep_residuals(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose Buck-Boost DCM sweep values and both balance residuals."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="buck-boost", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
            sweep={
                "x_axis": "f_sw", "x_min": 20_000, "x_max": 21_000,
                "x_points": 2, "metric": "efficiency_percent",
            },
        ),
    )
    assert response.status_code == 200
    sweep = response.json()["sweep"]
    assert sweep["kind"] == "line"
    assert sweep["conduction_regime"] == "DCM"
    assert len(sweep["values"]) == len(sweep["power_balance_residual_values_w"]) == 2


def test_loss_balanced_buck_boost_dcm_api_rejects_ccm_boundary(
    gate_conditioned_client: TestClient,
) -> None:
    """Return an actionable error when Buck-Boost output power reaches CCM."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="buck-boost", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=50, p_out=1000, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
        ),
    )
    assert response.status_code == 400
    assert "at or above the CCM boundary" in response.json()["detail"]


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


def test_loss_balanced_boost_dcm_api_returns_line_sweep_residuals(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose Boost DCM line metrics and both scalar-checked residual arrays."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="boost", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=200, p_out=100, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
            sweep={
                "x_axis": "f_sw", "x_min": 20_000, "x_max": 21_000,
                "x_points": 3, "metric": "p_loss_total_w",
            },
        ),
    )

    assert response.status_code == 200
    sweep = response.json()["sweep"]
    assert sweep["kind"] == "line"
    assert sweep["conduction_regime"] == "DCM"
    assert len(sweep["values"]) == 3
    assert len(sweep["voltage_law_residual_values_v"]) == 3
    assert len(sweep["power_balance_residual_values_w"]) == 3


def test_loss_balanced_boost_dcm_api_reports_unsupported_sweep_coordinate(
    gate_conditioned_client: TestClient,
) -> None:
    """Reject a CCM-boundary DCM sweep point with its SI coordinate."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="boost", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=200, p_out=100, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
            sweep={
                "x_axis": "p_out", "x_min": 100, "x_max": 200,
                "x_points": 3, "metric": "p_loss_total_w",
            },
        ),
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Sweep point (x=150 W) is unsupported" in detail
    assert "at or above the CCM boundary" in detail


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


def test_loss_balanced_buck_dcm_api_returns_sweep_residuals(
    gate_conditioned_client: TestClient,
) -> None:
    """Expose Buck DCM sweep values and both balance residuals."""
    response = gate_conditioned_client.post(
        "/api/topology/calculate",
        json=payload(
            topology="buck", mode="loss-balanced", conduction_regime="DCM",
            v_in=100, v_out=50, p_out=10, f_sw=20_000,
            l_value=0.001, c_out=0.001, t_j=25, v_g=15, diode_v_g=0,
            switch_transistor_id="gated-device", diode_transistor_id="gated-device",
            sweep={
                "x_axis": "f_sw", "x_min": 20_000, "x_max": 21_000,
                "x_points": 2, "metric": "efficiency_percent",
            },
        ),
    )
    assert response.status_code == 200
    sweep = response.json()["sweep"]
    assert sweep["kind"] == "line"
    assert sweep["conduction_regime"] == "DCM"
    assert len(sweep["values"]) == len(sweep["power_balance_residual_values_w"]) == 2


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
    """Do not rank scalar DCM results across topologies."""
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
    assert "Flyback DCM operating points are supported individually" in response.json()["detail"]


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
