"""Independent operating-point oracles for static parallel conduction."""
import json
from fractions import Fraction
from math import nextafter

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.current_sharing_routes import router
from transistordatabase.reverse_current_sharing import solve_reverse_current_sharing


def operating_point(channel, diode, demand, mode="strict-recorded"):
    """Declare two independent fixture curves at one junction temperature."""
    inputs = {
        "selected": True, "compatible_operating_state": True, "current_a": -demand,
        "junction_temperature_c": 80, "mode": mode,
    }
    for kind, (currents, voltages), gate in (
        ("channel", channel, 12), ("diode", diode, None),
    ):
        conditions = {"junction_temperature_c": 80, "gate_bias_v": gate}
        inputs[kind] = {
            "kind": f"{kind}-only", "current_a": currents, "voltage_v": voltages,
            "source": f"Independent analytic {kind} fixture",
            "isolation_evidence": f"The {kind} function supplies one branch current.",
            "recorded_conditions": conditions, "requested_conditions": conditions.copy(),
        }
    return inputs


def test_sweep_from_independent_inverse_voltage_laws():
    """Construct the total current from voltage without a root search."""
    channel = ([40, 42, 58], [0, 1, 3])
    diode = ([1, 9, 17], [0, 2, 3])
    for step in range(25):
        voltage = Fraction(step, 8)
        ic = 40 + 2 * voltage + 6 * max(voltage - 1, 0)
        id_ = 1 + 4 * voltage + 4 * max(voltage - 2, 0)
        total = ic + id_
        inputs = operating_point(channel, diode, float(total))
        result = solve_reverse_current_sharing(**inputs)["result"]
        assert (result["status"], result["support"]) == ("supported", "recorded")
        assert result["channel_current_a"] == -ic
        assert result["diode_current_a"] == -id_
        assert result["voltage_v"] == -voltage
        assert result["channel_static_power_w"] == voltage * ic
        assert result["diode_static_power_w"] == voltage * id_
        assert result["total_static_power_w"] == voltage * total
        assert result["current_residual_a"] == 0
        assert result["root_evidence"]["voltage_residual_v"] == 0


@pytest.mark.parametrize("demand,interval", [
    (7, (5, 5)), (7.5, (5, 5.5)), (9, (5, 7)), (10.5, (6.5, 7)), (11, (7, 7)),
])
def test_flat_contact_from_branch_current_ranges(demand, interval):
    """At 1 V, the channel permits 5–7 A and the diode permits 2–4 A."""
    voltage = [0, 1, 1, 1, 2]
    inputs = operating_point(([4, 5, 6, 7, 8], voltage), ([1, 2, 3, 4, 5], voltage), demand)
    result = solve_reverse_current_sharing(**inputs)["result"]
    lo, hi = interval
    if lo == hi:
        assert result["status"] == "supported"
        assert result["channel_current_a"] == -lo
        assert result["diode_current_a"] == -(demand - lo)
        assert result["root_evidence"]["isolated_root_count"] == 1
    else:
        assert result["status"] == "ambiguous"
        assert result["root_evidence"]["flat_intervals_a"] == [[lo, hi]]
        assert result["root_evidence"]["isolated_root_count"] == 0
        assert all(result[key] is None for key in (
            "channel_current_a", "diode_current_a", "current_sum_a", "current_residual_a",
            "voltage_v", "channel_static_power_w", "diode_static_power_w",
            "total_static_power_w", "blocking_margin_v",
        ))


def test_threshold_at_adjacent_representable_currents():
    """Check each side of the diode threshold without a voltage tolerance."""
    channel = ([0, 2], [0, 2])
    diode = ([0, 2], [1, 3])
    for demand, expected in [
        (nextafter(1, 0), "estimate"), (1, "recorded"), (nextafter(1, 2), "recorded"),
    ]:
        inputs = operating_point(channel, diode, demand, "ideal-diode-blocking")
        result = solve_reverse_current_sharing(**inputs)["result"]
        assert result["support"] == expected
        assert (result["diode_current_a"] < 0) == (demand > 1)
        if demand < 1:
            assert result["blocking_margin_v"] == float(1 - Fraction(str(demand)))
        else:
            assert result["blocking_margin_v"] is None


def test_invalid_utf8_is_rejected_at_transport_boundary():
    """A source containing invalid UTF-8 bytes returns a JSON validation error."""
    app = FastAPI()
    app.include_router(router)
    sample = ([0, 1], [0, 1])
    request = operating_point(sample, sample, 1)
    body = json.dumps(request).encode().replace(b"analytic channel", b"analytic \xff channel")
    with TestClient(app) as client:
        response = client.post("/api/assemblies/reverse-sharing", content=body,
                               headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]
