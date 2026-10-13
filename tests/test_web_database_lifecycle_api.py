"""Isolated regressions for the web database edit lifecycle."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from transistordatabase.core import repository as repository_module
from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.gui_web.backend import main


DEVICE_NAME = "Lifecycle Device #Primary"
SIMILAR_DEVICE_NAME = "Lifecycle Device"


def _edit_payload(name: str = DEVICE_NAME) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "type": "MOSFET",
            "manufacturer": "Infineon",
            "housing_type": "TO247",
            "author": "Updated Author",
            "comment": "Updated comment",
        },
        "electrical": {
            "v_abs_max": 800,
            "i_abs_max": 40,
            "i_cont": 35,
            "t_j_max": 180,
        },
        "thermal": {
            "r_th_cs": 0.4,
            "housing_area": 0.001,
            "cooling_area": 0.0005,
        },
    }


def _create_payload(name: str = DEVICE_NAME) -> dict[str, Any]:
    payload = _edit_payload(name)
    payload["thermal"].update(
        {"r_th_switch_cs": 0.4, "r_th_diode_cs": 0.4}
    )
    return payload


def _legacy_record(name: str) -> dict[str, object]:
    """Copy a checked-in valid record and retain representative raw-only data."""
    fixture_path = Path(__file__).resolve().parent / "test_data" / "CREE_C3M0060065J.json"
    record = json.loads(fixture_path.read_text(encoding="utf-8"))
    record["name"] = name
    record["unknown_future_field"] = {"keep": True, "values": ["a", "b"]}
    return record


class TemporaryJsonRepository:
    """Repository facade over test-owned records in pytest's temporary folder."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory
        self.deleted_names: list[str] = []
        self.saved_names: list[str] = []

    def save(self, transistor: object) -> None:
        """Record the name saved by the endpoint without touching files."""
        self.saved_names.append(transistor.metadata.name)

    def delete(self, name: str) -> bool:
        """Record requested deletion while retaining the fixture file."""
        if not (self.data_directory / f"{name}.json").exists():
            return False
        self.deleted_names.append(name)
        return True


@pytest.fixture
def isolated_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]]:
    """Create isolated records and point the API at the temporary repository."""
    original_records = {
        DEVICE_NAME: _legacy_record(DEVICE_NAME),
        SIMILAR_DEVICE_NAME: _legacy_record(SIMILAR_DEVICE_NAME),
    }
    for name, record in original_records.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")

    repository = TemporaryJsonRepository(tmp_path)
    pair_repository = object()
    monkeypatch.setattr(main, "_repo", repository)
    monkeypatch.setattr(main, "_pair_repo", pair_repository)
    client = TestClient(main.app)
    return client, repository, original_records


def test_json_upload_converts_from_memory_before_locked_repository_save(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Load an uploaded raw record without temporary-file cleanup or deletion."""
    client, repository, _originals = isolated_database
    record = _legacy_record("Uploaded Lifecycle Device")

    response = client.post(
        "/api/transistors/upload",
        files={"file": ("uploaded.json", json.dumps(record), "application/json")},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Uploaded Lifecycle Device"
    assert repository.saved_names == ["Uploaded Lifecycle Device"]


def test_update_patches_only_general_fields_in_temporary_legacy_json(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Patch form-owned fields while retaining every raw legacy-only value."""
    client, repository, originals = isolated_database
    record_path = repository.data_directory / f"{DEVICE_NAME}.json"

    response = client.put(
        "/api/transistor-record",
        params={"name": DEVICE_NAME},
        json=_edit_payload(),
    )

    assert response.status_code == 200
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored["name"] == DEVICE_NAME
    assert stored["type"] == "MOSFET"
    assert stored["manufacturer"] == "Infineon"
    assert stored["housing_type"] == "TO247"
    assert stored["author"] == "Updated Author"
    assert stored["comment"] == "Updated comment"
    assert stored["v_abs_max"] == 800
    assert stored["i_abs_max"] == 40
    assert stored["i_cont"] == 35
    assert stored["switch"]["t_j_max"] == 180
    assert stored["r_th_cs"] == 0.4
    assert stored["housing_area"] == 0.001
    assert stored["cooling_area"] == 0.0005

    original = originals[DEVICE_NAME]
    for legacy_only_key in (
        "r_g_int",
        "r_g_on_recommended",
        "r_g_off_recommended",
        "graph_v_ecoss",
        "raw_measurement_data",
        "unknown_future_field",
    ):
        assert stored[legacy_only_key] == original[legacy_only_key]
    original_switch = dict(original["switch"])
    original_switch["t_j_max"] = 180
    assert stored["switch"] == original_switch
    assert stored["diode"] == original["diode"]
    assert repository.deleted_names == []

    loaded = repository_module.JsonTransistorLoader().load_from_json(record_path)
    assert loaded.metadata.manufacturer == "Infineon"
    assert loaded.metadata.housing_type == "TO247"
    assert loaded.electrical_ratings.t_j_max == 180
    assert len(loaded.switch.channel_data) == len(original["switch"]["channel"])
    assert len(loaded.diode.channel_data) == len(original["diode"]["channel"])


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("manufacturer", "Not in catalog"), ("housing_type", "TO-247")],
)
def test_invalid_catalog_update_leaves_exact_record_bytes_unchanged(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
    field: str,
    invalid_value: str,
) -> None:
    """Reject invalid catalog values without changing the stored bytes."""
    client, repository, _originals = isolated_database
    record_path = repository.data_directory / f"{DEVICE_NAME}.json"
    original_bytes = record_path.read_bytes()
    payload = _edit_payload()
    payload["metadata"][field] = invalid_value

    response = client.put(
        "/api/transistor-record",
        params={"name": DEVICE_NAME},
        json=payload,
    )

    assert response.status_code == 400
    assert record_path.read_bytes() == original_bytes
    assert repository.deleted_names == []
    assert repository.saved_names == []


def test_query_name_crud_keeps_spaces_and_url_reserved_characters_exact(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Keep query-encoded identities exact through update and delete routes."""
    client, repository, originals = isolated_database
    target_path = repository.data_directory / f"{DEVICE_NAME}.json"
    similar_path = repository.data_directory / f"{SIMILAR_DEVICE_NAME}.json"

    # A query parameter preserves the entire identity; it cannot become a
    # fragment or be truncated at '?' as a raw path interpolation would.
    missing_name = "Lifecycle Device?preview=Lifecycle Device #Primary"
    missing = client.put(
        "/api/transistor-record",
        params={"name": missing_name},
        json=_edit_payload(missing_name),
    )
    assert missing.status_code in {400, 404}
    assert json.loads(target_path.read_text(encoding="utf-8")) == originals[DEVICE_NAME]
    assert json.loads(similar_path.read_text(encoding="utf-8")) == originals[SIMILAR_DEVICE_NAME]

    updated = client.put(
        "/api/transistor-record",
        params={"name": DEVICE_NAME},
        json=_edit_payload(),
    )
    assert updated.status_code == 200

    deleted = client.delete("/api/transistor-record", params={"name": DEVICE_NAME})
    assert deleted.status_code == 200
    assert repository.deleted_names == [DEVICE_NAME]
    assert target_path.exists()
    assert json.loads(similar_path.read_text(encoding="utf-8")) == originals[SIMILAR_DEVICE_NAME]


def test_path_separator_identities_are_rejected_without_touching_records(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Reject slash identities before any record can be addressed."""
    client, repository, originals = isolated_database

    updated = client.put(
        "/api/transistor-record",
        params={"name": "Lifecycle/Device"},
        json=_edit_payload("Lifecycle/Device"),
    )
    deleted = client.delete(
        "/api/transistor-record",
        params={"name": "Lifecycle\\Device"},
    )

    assert updated.status_code == 400
    assert deleted.status_code == 400
    assert repository.deleted_names == []
    for name, original in originals.items():
        record_path = repository.data_directory / f"{name}.json"
        assert json.loads(record_path.read_text(encoding="utf-8")) == original


def test_create_rejects_path_separator_identities(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Reject path separators in new record identities."""
    client, repository, _ = isolated_database

    response = client.post("/api/transistors", json=_edit_payload("Nested/Device"))

    assert response.status_code == 400
    assert "path separators" in response.json()["detail"]
    assert not (repository.data_directory / "Nested/Device.json").exists()


def test_create_rejects_duplicate_without_changing_populated_record(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Reject duplicate names without replacing populated legacy data."""
    client, repository, originals = isolated_database
    record_path = repository.data_directory / f"{DEVICE_NAME}.json"
    original_bytes = record_path.read_bytes()
    original = originals[DEVICE_NAME]
    assert original["switch"]["channel"]
    assert original["diode"]["channel"]

    response = client.post("/api/transistors", json=_create_payload())

    assert response.status_code == 409
    assert record_path.read_bytes() == original_bytes
    assert repository.saved_names == []


def test_create_writes_new_record_without_overwriting_repository_contents(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Publish a new record that reloads through the legacy loader."""
    client, repository, _originals = isolated_database
    created_name = "New Lifecycle Device"

    response = client.post("/api/transistors", json=_create_payload(created_name))

    assert response.status_code == 200
    created_path = repository.data_directory / f"{created_name}.json"
    assert created_path.exists()
    assert JsonTransistorLoader().load_from_json(created_path).metadata.name == created_name


def test_concurrent_create_requests_publish_one_complete_record(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Serialize competing creates and publish exactly one complete record."""
    _client, repository, _originals = isolated_database
    created_name = "Concurrent Lifecycle Device"
    payload = _create_payload(created_name)

    def create_record() -> int:
        with TestClient(main.app) as client:
            return client.post("/api/transistors", json=payload).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(lambda _index: create_record(), range(2)))

    assert sorted(statuses) == [200, 409]
    created_path = repository.data_directory / f"{created_name}.json"
    loaded = JsonTransistorLoader().load_from_json(created_path)
    assert loaded.metadata.name == created_name


def test_update_rejects_name_change_without_creating_duplicate(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Reject edits that attempt to rename a device through the general form."""
    client, repository, originals = isolated_database

    response = client.put(
        "/api/transistor-record",
        params={"name": DEVICE_NAME},
        json=_edit_payload(name="Renamed Device"),
    )

    assert response.status_code == 400
    assert "names cannot be changed" in response.json()["detail"]
    assert repository.deleted_names == []
    for name, original in originals.items():
        record_path = repository.data_directory / f"{name}.json"
        assert json.loads(record_path.read_text(encoding="utf-8")) == original
