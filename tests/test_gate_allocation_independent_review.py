"""Independent checks of exact binary states, current conservation and review evidence."""

import hashlib
import itertools
import json
import math
from fractions import Fraction

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.current_allocation import allocate_gate_current
from transistordatabase.gui_web.backend.current_allocation_routes import router


URL = "/api/waveform/gate-allocation"


@pytest.fixture
def client():
    """Serve the allocation route without opening a project database."""
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as application:
        yield application


def case():
    """Return irregularly spaced signed samples and a known negative zero."""
    return {
        "time_s": [-2, -1.25, 0, 0.5, 3],
        "current_a": [7.5, -0.0, -12.5, 3, -8],
        "gate": {"mode": "binary", "state": [1, True, 0.0, 0, False]},
        "source": "signed fixture μ",
        "selected": True,
        "conditions": {"offset": -0.0, "temperature": None},
    }


def test_exchanging_masks_exchanges_signed_allocations():
    """Check all masks and their complements against the same signed samples."""
    inputs = case()
    currents = inputs["current_a"]

    def allocate(states):
        gate = {"mode": "binary", "state": states}
        return allocate_gate_current(**(inputs | {"gate": gate}))["result"]["samples"]

    for state in itertools.product([False, True], repeat=len(currents)):
        complementary = tuple(not sample for sample in state)
        for current, a, b in zip(currents, allocate(state), allocate(complementary), strict=True):
            assert a["gate_on_current_a"] == b["gate_off_current_a"]
            assert a["gate_off_current_a"] == b["gate_on_current_a"]
            assert a["gate_on_current_a"] + a["gate_off_current_a"] == current
            active = a["gate_on_current_a"] if a["gate_on"] else a["gate_off_current_a"]
            assert math.copysign(1, active) == math.copysign(1, current)


@pytest.mark.parametrize("state", [Fraction(10**25 - 1, 10**25), Fraction(1, 10**400)])
def test_science_checks_real_membership_before_rounding(state):
    """Reject nonbinary values that round to zero or one."""
    inputs = case()
    inputs["gate"]["state"][1] = state
    with pytest.raises(ValueError):
        allocate_gate_current(**inputs)


def test_science_preserves_distinction_between_tiny_current_and_known_zero():
    """Reject a current below the supported numerical range."""
    inputs = case()
    inputs["current_a"][1] = Fraction(-1, 10**401)
    with pytest.raises(ValueError):
        allocate_gate_current(**inputs)


@pytest.mark.parametrize("literal,accepted", [
    ("0.99999999999999999999999999999", False),
    ("1.00000000000000000000000000001", False),
    ("100000000000000000000e-20", True),
    ("-0e100000000000000000000", True),
])
def test_binary_json_checks_original_decimal(client, literal, accepted):
    """Validate the original decimal token before binary64 rounding."""
    inputs = case()
    inputs["gate"]["state"][1] = "substitute_literal"
    wire = json.dumps(inputs).replace('"substitute_literal"', literal)
    response = client.post(URL, content=wire, headers={"Content-Type": "application/json"})
    assert response.status_code == (200 if accepted else 422)
    if accepted:
        assert response.json()["inputs"]["gate"]["state"][1] is bool(float(literal))


def test_downloaded_response_hashes_and_replay_with_analog_gate(client):
    """Replay accepted analog inputs and reproduce both hashes from response bytes."""
    inputs = case()
    inputs["gate"] = {"mode": "threshold", "gate_v": [-2, -1, 0, 0.5, 3], "threshold_v": -1}
    inputs["provenance"] = {"imported_filename": "fixture.csv", "edited_after_import": True}
    response = client.post(URL, json=inputs)
    assert response.status_code == 200
    envelope = json.loads(response.content)
    assert envelope["inputs"]["conditions"] == {"temperature": None, "offset": 0}
    assert math.copysign(1, envelope["inputs"]["conditions"]["offset"]) == -1
    assert [row["gate_on"] for row in envelope["result"]["samples"]] == [False, False, True, True, True]
    for key, digest_key in (("result", "result_sha256"), ("inputs", "input_sha256")):
        section = json.dumps(envelope[key], sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert hashlib.sha256(section.encode()).hexdigest() == envelope["evidence"][digest_key]
    assert client.post(URL, json=envelope["inputs"]).json() == envelope


def test_request_byte_limit_applies_to_stream_with_understated_header(client):
    """Count actual streamed bytes independently of the declared size."""
    chunks = (b" " * (256 * 1024) for _ in range(17))
    response = client.post(URL, content=chunks, headers={"Content-Length": "1"})
    assert response.status_code == 413


@pytest.mark.parametrize("context", [
    {"source": chr(0xD800)},
    {"provenance": {"calibration_notes": chr(0xD800)}},
    {"conditions": {chr(0xD800): None}},
])
def test_reject_context_that_cannot_round_trip_as_unicode(client, context):
    """Reject isolated surrogates in source and provenance strings and condition names."""
    inputs = case() | context
    response = client.post(URL, content=json.dumps(inputs), headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    with pytest.raises(ValueError):
        allocate_gate_current(**inputs)
