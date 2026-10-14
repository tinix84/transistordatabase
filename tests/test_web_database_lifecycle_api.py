"""Isolated regressions for the web database edit lifecycle."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
            "r_th_switch_cs": 0.4,
            "r_th_diode_cs": 0.4,
            "housing_area": 0.001,
            "cooling_area": 0.0005,
        },
    }


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

    def get_by_name(self, name: str) -> object | None:
        """Load an isolated core record as the production repository does."""
        record_path = self.data_directory / f"{name}.json"
        if not record_path.exists():
            return None
        return JsonTransistorLoader().load_from_json(record_path)

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


@pytest.mark.parametrize(
    ("add_path", "delete_component", "delete_type", "payload", "storage_path"),
    [
        (
            "channel?component=switch", "switch", "channel",
            {"t_j": 75, "v_g": 15, "v_data": [0, 1], "i_data": [0, 10]},
            ("switch", "channel"),
        ),
        (
            "channel?component=diode", "diode", "channel",
            {"t_j": 75, "v_data": [0, 1], "i_data": [0, 10]},
            ("diode", "channel"),
        ),
        (
            "switching/e_on", "switch", "e_on",
            {"dataset_type": "graph_i_e", "t_j": 75, "v_supply": 400, "v_g": 15,
             "r_g": 5, "graph_i_e": [[0, 10], [0, 0.001]]},
            ("switch", "e_on"),
        ),
        (
            "switching/e_off", "switch", "e_off",
            {"dataset_type": "graph_i_e", "t_j": 75, "v_supply": 400, "v_g": 15,
             "r_g": 5, "graph_i_e": [[0, 10], [0, 0.001]]},
            ("switch", "e_off"),
        ),
        (
            "switching/e_rr", "diode", "e_rr",
            {"dataset_type": "graph_i_e", "t_j": 75, "v_supply": 400, "v_g": 0,
             "r_g": 5, "graph_i_e": [[0, 10], [0, 0.001]]},
            ("diode", "e_rr"),
        ),
        (
            "gate_charge", "switch", "gate_charge",
            {"v_supply": 400, "t_j": 75, "i_channel": 10, "i_g": 0.1,
             "q_data": [0, 0.000001], "v_data": [0, 15]},
            ("switch", "charge_curve"),
        ),
        (
            "soa", "switch", "soa",
            {"t_c": 75, "time_pulse": 0.001, "v_data": [0, 400], "i_data": [10, 0]},
            ("switch", "soa"),
        ),
        (
            "capacitance/c_oss", "capacitance", "c_oss",
            {"t_j": 75, "v_data": [0, 400], "c_data": [0.000001, 0.0000001]},
            (None, "c_oss"),
        ),
        (
            "capacitance/c_iss", "capacitance", "c_iss",
            {"t_j": 75, "v_data": [0, 400], "c_data": [0.000001, 0.0000001]},
            (None, "c_iss"),
        ),
        (
            "capacitance/c_rss", "capacitance", "c_rss",
            {"t_j": 75, "v_data": [0, 400], "c_data": [0.000001, 0.0000001]},
            (None, "c_rss"),
        ),
    ],
)
def test_curve_add_and_delete_preserve_unrelated_legacy_record_data(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
    add_path: str,
    delete_component: str,
    delete_type: str,
    payload: dict[str, Any],
    storage_path: tuple[str | None, str],
) -> None:
    """Persist a curve without round-tripping unrelated legacy-only values."""
    client, repository, originals = isolated_database
    name = SIMILAR_DEVICE_NAME
    encoded_name = quote(name, safe="")
    record_path = repository.data_directory / f"{name}.json"
    original = originals[name]
    section, field = storage_path

    def curve_list(record: dict[str, Any]) -> list[Any]:
        target = record if section is None else record[section]
        return target.get(field) or []

    original_curves = curve_list(original)
    response = client.post(
        f"/api/transistors/{encoded_name}/curves/{add_path}",
        json=payload,
    )

    assert response.status_code == 200, response.text
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert len(curve_list(stored)) == len(original_curves) + 1
    assert curve_list(stored)[:-1] == original_curves

    for legacy_only_key in (
        "r_g_int",
        "r_g_on_recommended",
        "r_g_off_recommended",
        "graph_v_ecoss",
        "raw_measurement_data",
        "unknown_future_field",
    ):
        assert stored[legacy_only_key] == original[legacy_only_key]

    deleted = client.delete(
        f"/api/transistors/{encoded_name}/curves/{delete_component}/{delete_type}/{len(original_curves)}"
    )

    assert deleted.status_code == 200, deleted.text
    after_delete = json.loads(record_path.read_text(encoding="utf-8"))
    assert curve_list(after_delete) == original_curves
    for legacy_only_key in (
        "r_g_int",
        "r_g_on_recommended",
        "r_g_off_recommended",
        "graph_v_ecoss",
        "raw_measurement_data",
        "unknown_future_field",
    ):
        assert after_delete[legacy_only_key] == original[legacy_only_key]
    assert repository.saved_names == []


def test_curve_add_rejects_record_filename_name_mismatch_without_changing_either_file(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Do not let a mismatched record name redirect an add to another device."""
    client, repository, _originals = isolated_database
    requested_path = repository.data_directory / f"{DEVICE_NAME}.json"
    other_path = repository.data_directory / f"{SIMILAR_DEVICE_NAME}.json"
    requested_record = _legacy_record(SIMILAR_DEVICE_NAME)
    requested_path.write_text(json.dumps(requested_record), encoding="utf-8")
    before_requested = requested_path.read_bytes()
    before_other = other_path.read_bytes()

    response = client.post(
        f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/channel?component=switch",
        json={"t_j": 75, "v_g": 15, "v_data": [0, 1], "i_data": [0, 10]},
    )

    assert response.status_code == 409, response.text
    assert requested_path.read_bytes() == before_requested
    assert other_path.read_bytes() == before_other


def test_json_upload_preserves_raw_record_and_publishes_create_only(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Preserve an uploaded raw record instead of saving an adapted core object."""
    client, repository, _originals = isolated_database
    record = _legacy_record("Uploaded Lifecycle Device")

    response = client.post(
        "/api/transistors/upload",
        files={"file": ("uploaded.json", json.dumps(record), "application/json")},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Uploaded Lifecycle Device"
    assert repository.saved_names == []
    stored = json.loads(
        (repository.data_directory / "Uploaded Lifecycle Device.json").read_text(encoding="utf-8")
    )
    assert stored == record


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
    assert stored["r_th_switch_cs"] == 0.4
    assert stored["r_th_diode_cs"] == 0.4
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

    response = client.post("/api/transistors", json=_edit_payload())

    assert response.status_code == 409
    assert record_path.read_bytes() == original_bytes
    assert repository.saved_names == []


def test_create_writes_new_record_without_overwriting_repository_contents(
    isolated_database: tuple[TestClient, TemporaryJsonRepository, dict[str, dict[str, object]]],
) -> None:
    """Publish a new record that reloads through the legacy loader."""
    client, repository, _originals = isolated_database
    created_name = "New Lifecycle Device"

    response = client.post("/api/transistors", json=_edit_payload(created_name))

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
    payload = _edit_payload(created_name)

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
