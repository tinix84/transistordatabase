"""Isolated tests for raw JSON database backup and restore."""

from __future__ import annotations

import json
import ntpath
import os
from pathlib import Path, PureWindowsPath
from typing import Any

import pytest
from fastapi.testclient import TestClient

from transistordatabase.gui_web.backend import main


BACKUP_FORMAT = "transistordatabase.database-backup"


class TemporaryJsonRepository:
    """Repository facade over records owned by pytest's temporary folder."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory


@pytest.fixture
def isolated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Point the web API at an empty test-owned JSON directory."""
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(tmp_path))
    return TestClient(main.app)


def _legacy_record(name: str) -> dict[str, Any]:
    """Copy a valid legacy record and attach raw-only JSON values."""
    fixture = Path(__file__).resolve().parents[1] / "test_data" / "CREE_C3M0060065J.json"
    record = json.loads(fixture.read_text(encoding="utf-8"))
    record["name"] = name
    record["unknown_future_field"] = {"preserve": [None, True, 4.5, "raw"]}
    return record


def _backup(*records: dict[str, Any]) -> dict[str, Any]:
    return {"format": BACKUP_FORMAT, "version": 1, "records": list(records)}


def _upload_backup(client: TestClient, document: dict[str, Any], endpoint: str) -> Any:
    return client.post(
        endpoint,
        files={"file": ("database-backup.json", json.dumps(document), "application/json")},
    )


def test_backup_and_restore_preserve_raw_legacy_fields_and_unknown_values(
    isolated_database: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export raw records and restore their JSON values without core conversion."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_record = _legacy_record("Backup Device")
    source_record["r_g_int"] = 2.75
    source_path = source_dir / "Backup Device.json"
    source_path.write_text(json.dumps(source_record), encoding="utf-8")
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(source_dir))

    exported = isolated_database.get("/api/database/backup")

    assert exported.status_code == 200, exported.text
    document = exported.json()
    assert document["format"] == BACKUP_FORMAT
    assert document["version"] == 1
    assert document["records"] == [source_record]

    destination_dir = tmp_path / "destination"
    destination_dir.mkdir()
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(destination_dir))
    restored = isolated_database.post(
        "/api/database/backup/restore",
        files={"file": ("backup.json", exported.content, "application/json")},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json()["restored"] == ["Backup Device"]
    stored = json.loads((destination_dir / "Backup Device.json").read_text(encoding="utf-8"))
    assert stored == source_record


@pytest.mark.skipif(os.name == "nt", reason="Requires a case-sensitive destination filesystem")
def test_case_distinct_names_export_and_restore_on_case_sensitive_filesystem(
    isolated_database: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX export and restore preserve names that differ only by case."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    records = [_legacy_record("Device"), _legacy_record("device")]
    for record in records:
        (source_dir / f"{record['name']}.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(source_dir))

    exported = isolated_database.get("/api/database/backup")

    assert exported.status_code == 200, exported.text
    exported_records = exported.json()["records"]
    assert {record["name"] for record in exported_records} == {"Device", "device"}

    destination_dir = tmp_path / "destination"
    destination_dir.mkdir()
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(destination_dir))
    restored = isolated_database.post(
        "/api/database/backup/restore",
        files={"file": ("backup.json", exported.content, "application/json")},
    )

    assert restored.status_code == 200, restored.text
    assert set(restored.json()["restored"]) == {"Device", "device"}
    assert json.loads((destination_dir / "Device.json").read_text(encoding="utf-8")) == records[0]
    assert json.loads((destination_dir / "device.json").read_text(encoding="utf-8")) == records[1]


@pytest.mark.skipif(os.name == "nt", reason="The fixture needs two case-distinct directory entries")
def test_export_reports_case_collisions_using_destination_semantics(
    isolated_database: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export detects duplicate identities when the target semantics fold case."""
    for name in ("Device", "device"):
        (tmp_path / f"{name}.json").write_text(json.dumps(_legacy_record(name)), encoding="utf-8")
    monkeypatch.setattr(main, "_record_identity_key", ntpath.normcase)

    response = isolated_database.get("/api/database/backup")

    assert response.status_code == 409
    assert any(
        "Duplicate record identity" in record["error"]
        for record in response.json()["detail"]["records"]
    )


def test_backup_reports_corrupt_records_instead_of_omitting_them(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    """A corrupt database member prevents an incomplete backup download."""
    (tmp_path / "corrupt.json").write_text("{broken", encoding="utf-8")

    response = isolated_database.get("/api/database/backup")

    assert response.status_code == 409
    assert "corrupt.json" in response.json()["detail"]["records"][0]["file"]


def test_restore_skips_conflicts_without_changing_existing_bytes(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    """Restore only new names and keep a same-name record byte-for-byte intact."""
    existing_path = tmp_path / "Existing Device.json"
    existing_path.write_text('{ "name": "untouched" }\n', encoding="utf-8")
    original_bytes = existing_path.read_bytes()
    conflict = _legacy_record("Existing Device")
    addition = _legacy_record("New Device")

    response = _upload_backup(
        isolated_database,
        _backup(conflict, addition),
        "/api/database/backup/restore",
    )

    assert response.status_code == 200, response.text
    assert response.json()["restored"] == ["New Device"]
    assert response.json()["conflicts"] == ["Existing Device"]
    assert existing_path.read_bytes() == original_bytes
    assert json.loads((tmp_path / "New Device.json").read_text(encoding="utf-8")) == addition


def test_case_insensitive_existing_conflicts_match_preview_and_restore(
    isolated_database: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows case-insensitive identity semantics agree across preview and restore."""
    monkeypatch.setattr(main, "_record_identity_key", ntpath.normcase)
    existing_path = tmp_path / "Existing Device.json"
    existing_path.write_text('{ "name": "do not replace" }\n', encoding="utf-8")
    original_bytes = existing_path.read_bytes()
    record = _legacy_record("existing device")

    preview = _upload_backup(
        isolated_database,
        _backup(record),
        "/api/database/backup/validate",
    )
    restored = _upload_backup(
        isolated_database,
        _backup(record),
        "/api/database/backup/restore",
    )

    assert preview.status_code == 200, preview.text
    assert preview.json()["conflicts"] == ["existing device"]
    assert preview.json()["restore_count"] == 0
    assert restored.status_code == 200, restored.text
    assert restored.json()["restored"] == []
    assert restored.json()["conflicts"] == ["existing device"]
    assert existing_path.read_bytes() == original_bytes
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_validate_backup_previews_conflicts_without_writing_records(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    """The validation step reports the add set and conflict set without writes."""
    existing = _legacy_record("Existing Preview Device")
    existing_path = tmp_path / "Existing Preview Device.json"
    existing_path.write_text(json.dumps(existing), encoding="utf-8")
    new_record = _legacy_record("New Preview Device")

    response = _upload_backup(
        isolated_database,
        _backup(existing, new_record),
        "/api/database/backup/validate",
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "valid": True,
        "format": BACKUP_FORMAT,
        "version": 1,
        "record_count": 2,
        "restore_count": 1,
        "restore_names": ["New Preview Device"],
        "conflicts": ["Existing Preview Device"],
    }
    assert list(tmp_path.glob("*.json")) == [existing_path]


@pytest.mark.parametrize(
    "records",
    [
        [_legacy_record("Valid Before Corrupt"), {"name": "Missing Curves"}],
        [_legacy_record("Duplicate Device"), _legacy_record("Duplicate Device")],
    ],
)
def test_invalid_or_duplicate_backup_is_rejected_before_any_record_write(
    isolated_database: TestClient,
    tmp_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Validate the entire backup before publishing any otherwise-valid entry."""
    response = _upload_backup(
        isolated_database,
        _backup(*records),
        "/api/database/backup/restore",
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("endpoint", ["/api/database/backup/validate", "/api/database/backup/restore"])
def test_case_insensitive_duplicate_identities_are_rejected_consistently(
    isolated_database: TestClient,
    tmp_path: Path,
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows path identity semantics reject case-only duplicate input names."""
    monkeypatch.setattr(main, "_record_identity_key", ntpath.normcase)
    response = _upload_backup(
        isolated_database,
        _backup(_legacy_record("Duplicate Device"), _legacy_record("duplicate device")),
        endpoint,
    )

    assert response.status_code == 400
    assert "current filesystem path semantics" in response.json()["detail"]["errors"][0]["error"]
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("number_literal", ["1e400", "1e-400"])
def test_restore_rejects_unrepresentable_number_before_any_record_write(
    isolated_database: TestClient,
    tmp_path: Path,
    number_literal: str,
) -> None:
    """An unrepresentable float anywhere in the file blocks every write."""
    valid = json.dumps(_legacy_record("Valid Before Overflow"))
    overflow = json.dumps(_legacy_record("Overflow Number"))[:-1] + f',"unknown_number":{number_literal}}}'
    content = (
        '{"format":"transistordatabase.database-backup","version":1,"records":'
        f"[{valid},{overflow}]}}"
    )
    response = isolated_database.post(
        "/api/database/backup/restore",
        files={"file": ("backup.json", content, "application/json")},
    )

    assert response.status_code == 400
    assert "finite float range" in response.json()["detail"]["errors"][0]["error"]
    assert list(tmp_path.glob("*.json")) == []


def test_restore_rejects_unpaired_surrogate_before_any_record_write(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    """Every raw record must survive UTF-8 encoding before publication starts."""
    valid = json.dumps(_legacy_record("Valid Before Surrogate"))
    invalid_utf8 = json.dumps(_legacy_record("Unpaired Surrogate"))[:-1] + r',"unknown_field":"\ud800"}'
    content = (
        '{"format":"transistordatabase.database-backup","version":1,"records":'
        f"[{valid},{invalid_utf8}]}}"
    )
    response = isolated_database.post(
        "/api/database/backup/restore",
        files={"file": ("backup.json", content, "application/json")},
    )

    assert response.status_code == 400
    assert "UTF-8 JSON" in response.json()["detail"]["errors"][0]["error"]
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize(
    "unsafe_name",
    ["CON", "con.txt", "PRN", "AUX", "NUL", "COM1", "LPT9", "CONIN$", "COM¹", "bad<name", "trailing.", "trailing "],
)
def test_restore_rejects_nonportable_windows_filename_identities_before_writes(
    isolated_database: TestClient,
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    response = _upload_backup(
        isolated_database,
        _backup(_legacy_record(unsafe_name)),
        "/api/database/backup/restore",
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_restore_rejects_traversal_and_separator_identities_before_writes(
    isolated_database: TestClient,
    tmp_path: Path,
    separator: str,
) -> None:
    """A backup identity must map to one safe repository filename."""
    outside_name = f"{tmp_path.name}-outside"
    name = f"..{separator}{outside_name}"
    outside_path = tmp_path.parent / f"{outside_name}.json"
    assert not outside_path.exists()
    response = _upload_backup(
        isolated_database,
        _backup(_legacy_record(name)),
        "/api/database/backup/restore",
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []
    assert not outside_path.exists()


def test_restore_rejects_drive_relative_identity_before_writes(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    """A drive-relative identity cannot redirect a Windows record path."""
    # Reproduce the host-independent Windows joining behavior without touching
    # the path: a drive-relative leaf changes the drive from the database root.
    windows_root = PureWindowsPath(r"D:\database")
    windows_escape = windows_root / "C:outside.json"
    assert windows_escape.drive == "C:"
    with pytest.raises(ValueError):
        windows_escape.relative_to(windows_root)

    response = _upload_backup(
        isolated_database,
        _backup(_legacy_record("C:outside")),
        "/api/database/backup/restore",
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


def test_restore_rejects_windows_alternate_data_stream_identity_before_writes(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    response = _upload_backup(
        isolated_database,
        _backup(_legacy_record("Device:stream")),
        "/api/database/backup/restore",
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


def test_record_path_stays_inside_configured_repository_directory(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    """Generated record paths remain children under local and Windows semantics."""
    root = tmp_path.resolve()
    record_path = main._legacy_record_path("Contained Device")

    assert record_path.resolve().parent == root
    assert record_path.resolve().is_relative_to(root)
    windows_root = PureWindowsPath(str(tmp_path))
    windows_record_path = PureWindowsPath(str(record_path))
    assert windows_record_path.is_relative_to(windows_root)
    assert not record_path.exists()


@pytest.mark.parametrize(
    "document",
    [
        {"format": "other", "version": 1, "records": [_legacy_record("Unsupported Format")]},
        {"format": BACKUP_FORMAT, "version": 2, "records": [_legacy_record("Unsupported Version")]},
    ],
)
def test_restore_rejects_unsupported_format_or_version_before_writes(
    isolated_database: TestClient,
    tmp_path: Path,
    document: dict[str, Any],
) -> None:
    response = _upload_backup(isolated_database, document, "/api/database/backup/restore")

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


def test_restore_rejects_duplicate_json_object_keys_before_writes(
    isolated_database: TestClient,
    tmp_path: Path,
) -> None:
    content = (
        '{"format":"transistordatabase.database-backup","version":1,"records":'
        '[{"name":"First Value","name":"Second Value"}]}'
    )
    response = isolated_database.post(
        "/api/database/backup/restore",
        files={"file": ("backup.json", content, "application/json")},
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


def test_single_record_upload_is_raw_preserving_and_create_only(
    isolated_database: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing upload path shares additive raw-record publication rules."""
    original = _legacy_record("Uploaded Raw Device")
    first = isolated_database.post(
        "/api/transistors/upload",
        files={"file": ("device.json", json.dumps(original), "application/json")},
    )
    stored_path = tmp_path / "Uploaded Raw Device.json"

    assert first.status_code == 200, first.text
    assert json.loads(stored_path.read_text(encoding="utf-8")) == original
    original_bytes = stored_path.read_bytes()

    changed = dict(original, unknown_future_field={"overwrite": True})
    duplicate = isolated_database.post(
        "/api/transistors/upload",
        files={"file": ("device.json", json.dumps(changed), "application/json")},
    )

    assert duplicate.status_code == 409
    assert stored_path.read_bytes() == original_bytes

    monkeypatch.setattr(main, "_record_identity_key", ntpath.normcase)
    case_variant = dict(original, name="uploaded raw device")
    case_duplicate = isolated_database.post(
        "/api/transistors/upload",
        files={"file": ("device.json", json.dumps(case_variant), "application/json")},
    )

    assert case_duplicate.status_code == 409
    assert stored_path.read_bytes() == original_bytes


@pytest.mark.parametrize("name", ["../outside", r"..\outside"])
def test_single_upload_rejects_separator_identity(
    isolated_database: TestClient,
    tmp_path: Path,
    name: str,
) -> None:
    """Single-record uploads cannot create a path outside the record directory."""
    response = isolated_database.post(
        "/api/transistors/upload",
        files={"file": ("device.json", json.dumps(_legacy_record(name)), "application/json")},
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize(
    "name",
    ["C:outside", "Device:stream", "CON", "PRN.txt", "COM2", "LPT³", "bad?name", "trailing."],
)
def test_single_upload_rejects_windows_unsafe_identity(
    isolated_database: TestClient,
    tmp_path: Path,
    name: str,
) -> None:
    response = isolated_database.post(
        "/api/transistors/upload",
        files={"file": ("device.json", json.dumps(_legacy_record(name)), "application/json")},
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.json")) == []
