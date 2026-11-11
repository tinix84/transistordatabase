"""Independent transport, numerical and provenance checks for stress indices."""

import hashlib
import json
from fractions import Fraction

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.stress_routes import router
from transistordatabase.semiconductor_stress import calculate_stress_indices


ENDPOINT = "/api/stress/calculate"


def case():
    """Return an asymmetric waveform with signed voltage and two events."""
    return {
        "source": "Independent three-segment fixture Ω",
        "conditions": {"temperature_c": None, "offset_v": -0.0},
        "time_s": [-3, -2, 0, 4],
        "current_a": [5, -1, 3, -1],
        "voltage_v": [-100, -40, -8, -32],
        "rds_on_ohm": 0.125,
        "transitions": [
            {"kind": "turn_off", "time_s": -2.5, "frequency_hz": 3},
            {"kind": "turn_on", "time_s": 1, "frequency_hz": 5},
        ],
    }


@pytest.fixture
def client():
    """Serve the stress route without opening a project database."""
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as api:
        yield api


def test_asymmetric_signed_waveform_and_exact_oracle(client):
    """Check all four indices against a rational integration result."""
    response = client.post(ENDPOINT, json=case())
    assert response.status_code == 200
    result = response.json()["result"]
    # Simpson integration of each squared linear segment is exact.
    # The segment midpoint currents are 2, 1 and 1 A, respectively.
    integral = Fraction(25 + 4 * 4 + 1, 6)
    integral += 2 * Fraction(1 + 4 + 9, 6)
    integral += 4 * Fraction(9 + 4 + 1, 6)
    mean_square = integral / 7
    expected = {
        "conduction": (mean_square, "A^2"),
        "volume": (64 * mean_square, "A^2 V^2"),
        "switching": (-840 - 280, "Hz A^2 V"),
        "cost": (Fraction(1, 8), "1/(ohm V^2)"),
    }
    assert result["peak_voltage_v"] == -8
    for key, (value, unit) in expected.items():
        metric = result["metrics"][key]
        assert (metric["status"], metric["unit"]) == ("calculated", unit)
        assert metric["value"] == pytest.approx(float(value))


@pytest.mark.parametrize("scale,offset", [(1, 0), (8, 50)])
def test_knots_and_time_coordinate_do_not_change_indices(scale, offset):
    """Preserve the indices under subdivision and an affine change of time."""
    inputs = case()
    expected = calculate_stress_indices(**inputs)["result"]["metrics"]
    # Insert samples exactly halfway along the first and final segments.
    inputs.update(
        time_s=[scale * t + offset for t in [-3, -2.5, -2, 0, 2, 4]],
        current_a=[5, 2, -1, 3, 1, -1],
        voltage_v=[-100, -70, -40, -8, -20, -32],
    )
    for record in inputs["transitions"]:
        record["time_s"] = scale * record["time_s"] + offset
    actual = calculate_stress_indices(**inputs)["result"]["metrics"]
    for key, metric in expected.items():
        assert actual[key]["value"] == pytest.approx(metric["value"])


@pytest.mark.parametrize(
    "times,current,stamp",
    [([-1, 0], [1, 0], -1e-17), ([0, 1], [0, 1], 1e-17)],
)
def test_resolvable_event_close_to_zero_endpoint(times, current, stamp):
    """Preserve the small current near either endpoint of a linear ramp."""
    inputs = case() | {
        "time_s": times, "current_a": current, "voltage_v": [1, 1],
        "transitions": [{"kind": "turn_on", "time_s": stamp, "frequency_hz": 1}],
    }
    result = calculate_stress_indices(**inputs)["result"]
    assert result["transition_samples"][0]["current_a"] == pytest.approx(
        1e-17, abs=0, rel=1e-15
    )
    assert result["metrics"]["switching"]["value"] == pytest.approx(
        1e-34, abs=0, rel=1e-15
    )


def test_known_signed_cancellation_preserves_both_events(client):
    """Keep both events when their signed contributions cancel."""
    inputs = case() | {
        "time_s": [0, 1], "current_a": [-2, -2], "voltage_v": [3, -3],
        "transitions": [{"kind": kind, "time_s": t, "frequency_hz": 7}
                        for kind, t in [("turn_on", 0), ("turn_off", 1)]],
    }
    response = client.post(ENDPOINT, json=inputs)
    assert response.status_code == 200
    result = response.json()["result"]
    assert [row["contribution"] for row in result["transition_samples"]] == [84, -84]
    switching = result["metrics"]["switching"]
    assert switching["status"] == "calculated"
    assert switching["value"] == 0


@pytest.mark.parametrize("token", ["1e-400", "-9.9e-999"])
def test_http_nonzero_current_cannot_become_zero_evidence(client, token):
    """Reject a nonzero JSON current that falls below the float range."""
    inputs = case() | {"current_a": ["REPLACE", -1, 3, -1]}
    body = json.dumps(inputs).replace('"REPLACE"', token)
    response = client.post(ENDPOINT, content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_raw_http_review_json_verifies_both_hashes(client):
    """Reproduce the evidence hashes from the original HTTP response."""
    response = client.post(ENDPOINT, json=case())
    assert response.status_code == 200
    review = json.loads(response.content)
    for key, digest in [("inputs", "input_sha256"), ("result", "result_sha256")]:
        canonical = json.dumps(
            review[key], sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == review["evidence"][digest]
