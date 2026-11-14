"""Transport parity, lexical numeric safety, and bounds for issue 27."""
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from native_waveform_fixture import inputs
from transistordatabase.gui_web.backend import native_waveform_routes as routes
from transistordatabase.native_waveform import (
    MAX_REQUEST_BYTES, NativeWaveformError, evaluate_native_waveform,
)


class Repository:
    """Return one test-owned core object without touching persistent storage."""

    def __init__(self, device):
        self.device = device
        self.reads = []

    def get_by_name(self, name):
        self.reads.append(name)
        return self.device if name == self.device.metadata.name else None


@pytest.fixture
def transport():
    device, selection, case = inputs()
    repository = Repository(device)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.device_repository] = lambda: repository
    payload = {"device_id": device.metadata.name, "selection": selection, "case": case}
    return TestClient(app), repository, payload


def test_http_result_exactly_matches_qualified_pure_operation(transport, monkeypatch):
    client, repository, payload = transport
    expected = evaluate_native_waveform(repository.device, payload["selection"], payload["case"])
    calls = []
    original = routes.evaluate_native_waveform

    def counted(device, selection, case):
        calls.append((device, selection, case))
        return original(device, selection, case)

    monkeypatch.setattr(routes, "evaluate_native_waveform", counted)
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert len(calls) == 1
    assert repository.reads == [payload["device_id"]]
    assert response.json()["analysis"]["total_power_w"] == pytest.approx(3.25)


def test_missing_device_has_no_fallback(transport):
    client, repository, payload = transport
    payload["device_id"] = "does-not-exist"
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 404
    assert repository.reads == ["does-not-exist"]


def test_stored_identity_mismatch_is_preserved(transport, monkeypatch):
    client, repository, payload = transport
    monkeypatch.setattr(repository, "get_by_name", lambda name: inputs()[0])
    payload["device_id"] = "different-stored-name"
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 409


@pytest.mark.parametrize("identity", ["../outside", "a/b", "a\\b", "C:outside", "..", ".", " ", "bad\x00name"])
def test_storage_identity_rejects_paths_before_repository_access(transport, identity):
    client, repository, payload = transport
    payload["device_id"] = identity
    assert client.post("/api/waveform/native/analyze", json=payload).status_code == 422
    assert repository.reads == []


def test_extra_envelope_fields_reject(transport):
    client, repository, payload = transport
    payload["extra"] = True
    assert client.post("/api/waveform/native/analyze", json=payload).status_code == 422
    assert repository.reads == []


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p["selection"].update(input_temperature_K=323.15), "condition-mismatch"),
    (lambda p: p["selection"]["conduction"].update(positive=None), "missing-model"),
    (lambda p: p["case"]["waveform"].update(current_a=[5.] * 5), "unsupported-domain"),
    (lambda p: p["case"].update(extra=True), "invalid-input"),
    (lambda p: p["case"]["waveform"].update(current_a=[True] * 5), "invalid-input"),
    (lambda p: p["case"]["events"].update(turn_on=[1, 0, 0, 0]), "invalid-input"),
])
def test_stable_semantic_error_mapping(transport, mutation, code):
    client, _repository, payload = transport
    mutation(payload)
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code
    assert response.json()["detail"]["message"]


def test_declared_omissions_preserve_null_totals(transport):
    client, repository, payload = transport
    payload["selection"]["turn_on"] = {"mode": "omitted", "reason": "Not evaluated"}
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result == evaluate_native_waveform(repository.device, payload["selection"], payload["case"])
    assert result["analysis"]["total_power_w"] is None
    assert result["analysis"]["known_component_power_w"] == pytest.approx(2.75)


def test_sample_limit_rejects_before_source_read(transport):
    client, repository, payload = transport
    payload["case"]["waveform"]["time_s"] = list(range(10001))
    payload["case"]["waveform"]["current_a"] = [2.] * 10001
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "resource-limit"
    assert repository.reads == []


def test_source_read_budget_maps_to_413(transport, monkeypatch):
    client, _repository, payload = transport

    def oversized(*_args, **_kwargs):
        raise routes.DeviceThermalResourceError("Source exceeds its bounded read allowance")

    monkeypatch.setattr(routes, "_snapshot", oversized)
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "resource-limit"


def test_core_output_budget_maps_to_413(transport, monkeypatch):
    client, _repository, payload = transport

    def oversized(*_args, **_kwargs):
        raise NativeWaveformError("resource-limit", "Synthetic output bound")

    monkeypatch.setattr(routes, "evaluate_native_waveform", oversized)
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "resource-limit"


def test_raw_body_exact_limit_and_one_byte_beyond(transport):
    client, repository, payload = transport
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    exact = encoded + b" " * (MAX_REQUEST_BYTES - len(encoded))
    headers = {"Content-Type": "application/json"}
    response = client.post("/api/waveform/native/analyze", content=exact, headers=headers)
    assert response.status_code == 200, response.text
    repository.reads.clear()
    response = client.post("/api/waveform/native/analyze", content=exact + b" ", headers=headers)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "resource-limit"
    assert repository.reads == []


def test_body_cap_ignores_misleading_content_length_and_precedes_json(transport):
    client, repository, _payload = transport
    response = client.post(
        "/api/waveform/native/analyze",
        content=b"x" * (MAX_REQUEST_BYTES + 1),
        headers={"Content-Type": "application/json", "Content-Length": "1"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "resource-limit"
    assert repository.reads == []


def test_chunked_body_cap_before_repository_or_json_parse(transport):
    client, repository, _payload = transport

    def chunks():
        for _ in range(1025):
            yield b"x" * 4096

    response = client.post("/api/waveform/native/analyze", content=chunks(),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "resource-limit"
    assert repository.reads == []


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400", "1e-400", "9007199254740993"])
def test_unsafe_original_numeric_tokens_reject_before_source_lookup(transport, literal):
    client, repository, payload = transport
    payload["case"]["waveform"]["current_a"][0] = "NUMERIC_TOKEN"
    raw = json.dumps(payload).replace('"NUMERIC_TOKEN"', literal)
    response = client.post("/api/waveform/native/analyze", content=raw,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert repository.reads == []


def test_decimal_normalization_keeps_schema_indices_and_masks_typed(transport):
    client, repository, payload = transport
    encoded = json.dumps(payload, separators=(",", ":"))
    response = client.post("/api/waveform/native/analyze", content=encoded,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 200, response.text
    native = response.json()["native_request_snapshot"]
    assert type(native["selection"]["schema_version"]) is int
    assert type(native["selection"]["turn_on"]["positive"]["record_index"]) is int
    assert type(native["case"]["events"]["turn_on"][0]) is bool
    assert response.json() == evaluate_native_waveform(repository.device, payload["selection"], payload["case"])


def test_malformed_json_and_non_object_payload_reject(transport):
    client, repository, _payload = transport
    for raw in ('{"broken":', "null", "[]"):
        assert client.post("/api/waveform/native/analyze", content=raw,
                           headers={"Content-Type": "application/json"}).status_code == 422
    assert repository.reads == []


def test_unicode_provenance_is_identical_to_python(transport):
    client, repository, payload = transport
    payload["case"]["candidate_id"] = "München-半導体"
    payload["case"]["provenance"]["calibration_notes"] = "Ω / declared fixture"
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == evaluate_native_waveform(repository.device, payload["selection"], payload["case"])


def test_native_route_is_registered_in_actual_application():
    from transistordatabase.gui_web.backend.main import app
    schema = app.openapi()
    assert "post" in schema["paths"]["/api/waveform/native/analyze"]
    assert "post" in schema["paths"]["/api/waveform/analyze"]


def test_persisted_device_codec_snapshot_matches_pure_result(transport):
    from io import BytesIO
    from transistordatabase.core.serialization import core_to_json_document
    client, repository, payload = transport
    raw = json.dumps(core_to_json_document(repository.device), allow_nan=False).encode()
    calls = []

    class Source:
        def open(self, mode):
            assert mode == "rb"
            calls.append("bounded-source-read")
            return BytesIO(raw)

    repository._storage_path = lambda name: Source()
    expected = evaluate_native_waveform(repository.device, payload["selection"], payload["case"])
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert calls == ["bounded-source-read"]
    assert repository.reads == []


def test_real_source_byte_cap_rejects_before_decode(transport):
    from io import BytesIO
    from transistordatabase.gui_web.backend.device_electrothermal_routes import MAX_SOURCE_BYTES
    client, repository, payload = transport

    class Source:
        def open(self, mode):
            assert mode == "rb"
            return BytesIO(b"x" * (MAX_SOURCE_BYTES + 1))

    repository._storage_path = lambda name: Source()
    response = client.post("/api/waveform/native/analyze", json=payload)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "resource-limit"
    assert repository.reads == []


