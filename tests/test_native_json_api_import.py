"""Actual file/API native JSON import through an isolated real repository."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from transistordatabase.core.repository import (
    JsonTransistorLoader, JsonTransistorRepository, RepositoryPathError,
)
from transistordatabase.core.serialization import VERSION, core_to_json_document
from transistordatabase.gui_web.backend import main
from transistordatabase.plecs_document import (
    finalize_plecs_package, parse_plecs_document, render_core_plecs,
)


def source_backed_fixture():
    """Supply required metadata independently, preserving actual unknown housing."""
    document = parse_plecs_document((Path(__file__).parent / "test_data/sample_plecs_switch.xml").read_bytes())
    transistor = finalize_plecs_package(document, package_id="0",
        ratings=dict(v_abs_max=1200., i_abs_max=75., i_cont=40., t_j_max=175.),
        thermal_properties=dict(housing_area=.0001, cooling_area=.0002),
        block_targets={"0":"switch"},
        gate_voltage_by_table={t.source_path:15. for t in document.packages[0].blocks[0].tables})
    return document, transistor


@pytest.fixture
def isolated_api(tmp_path, monkeypatch):
    """Use a fresh real repository; never persist into the user's database."""
    repository = JsonTransistorRepository(tmp_path / "isolated-repository")
    monkeypatch.setattr(main, "_repo", repository)
    return TestClient(main.app), repository


def test_native_archive_file_upload_save_reload_json_download(tmp_path, isolated_api):
    """Preserve the complete archive through the existing multipart API and exporter."""
    client, repository = isolated_api
    document, transistor = source_backed_fixture()
    archive = core_to_json_document(transistor)
    source = tmp_path / "source-finalized-native.json"
    source.write_text(json.dumps(archive), encoding="utf-8")
    original_source = source.read_bytes()
    response = client.post("/api/transistors/upload", files={
        "file":(source.name, original_source, "application/json")})
    assert response.status_code == 200, response.text
    assert response.json()["id"] == transistor.metadata.name
    saved = repository.get_by_name(transistor.metadata.name)
    assert saved.metadata.manufacturer == "TestVendor"
    assert saved.metadata.housing_type == ""
    assert core_to_json_document(saved) == archive
    assert parse_plecs_document(render_core_plecs(saved)).packages == document.packages
    exported = client.post(f"/api/transistors/{transistor.metadata.name}/export/json")
    assert exported.status_code == 200, exported.text
    assert exported.json() == archive
    download = tmp_path / "browser-downloaded-native.json"
    download.write_bytes(exported.content)
    restored = JsonTransistorLoader().load_from_json(download)
    assert parse_plecs_document(render_core_plecs(restored)).packages == document.packages
    assert source.read_bytes() == original_source


def test_native_dictionary_create_dispatch_preserves_all_fields(isolated_api):
    """The dictionary CRUD path recognizes a native archive before form parsing."""
    client, repository = isolated_api
    _, transistor = source_backed_fixture()
    archive = core_to_json_document(transistor)
    response = client.post("/api/transistors", json=archive)
    assert response.status_code == 200, response.text
    assert core_to_json_document(repository.get_by_name(transistor.metadata.name)) == archive


@pytest.mark.parametrize("malformation", ["schema", "version", "missing-schema"])
@pytest.mark.parametrize("operation", ["upload", "create"])
def test_unknown_native_schema_never_becomes_flat_unknown_record(isolated_api, malformation, operation):
    """Reject unsupported native documents without saving placeholder identities."""
    client, repository = isolated_api
    _, transistor = source_backed_fixture()
    archive = core_to_json_document(transistor)
    if malformation == "schema":
        archive["schema"] = "another-native-schema"
    elif malformation == "version":
        archive["version"] = VERSION + 1
    else:
        archive.pop("schema")
    response = (client.post("/api/transistors/upload", files={"file":("unsupported.json", json.dumps(archive).encode(), "application/json")})
                if operation == "upload" else client.post("/api/transistors", json=archive))
    assert response.status_code == 400, response.text
    assert repository.list_all() == []


@pytest.mark.parametrize("operation", ["upload", "create"])
def test_existing_flat_legacy_json_preserves_curves_without_source_changes(isolated_api, operation):
    """Retain complete existing flat JSON, rather than reconstructing summary fields."""
    client, repository = isolated_api
    source = Path(__file__).parent / "test_data/database/CREE_C3M0016120K.json"
    original = source.read_bytes()
    expected = JsonTransistorLoader().load_from_json(source)
    response = (client.post("/api/transistors/upload", files={"file":(source.name, original, "application/json")})
                if operation == "upload" else client.post("/api/transistors", json=json.loads(original)))
    assert response.status_code == 200, response.text
    restored = repository.get_by_name(expected.metadata.name)
    assert core_to_json_document(restored) == core_to_json_document(expected)
    assert source.read_bytes() == original


def test_nested_browser_form_remains_supported(isolated_api):
    """Keep the established explicitly supplied nested metadata form shape."""
    client, repository = isolated_api
    data = {"metadata":{"name":"Nested_Form_Device","type":"MOSFET","manufacturer":"Explicit Vendor",
                        "housing_type":"","author":"source"},
            "electrical":{"v_abs_max":600.,"i_abs_max":50.,"i_cont":30.,"t_j_max":150.},
            "thermal":{"housing_area":.0001,"cooling_area":.0002,"r_th_cs":None}}
    response = client.post("/api/transistors", json=data)
    assert response.status_code == 200, response.text
    assert repository.get_by_name("Nested_Form_Device").metadata.manufacturer == "Explicit Vendor"


@pytest.mark.parametrize("identity", ["../outside", "..\\outside"])
def test_native_import_storage_identity_rejected_without_outside_write(tmp_path, isolated_api, identity):
    """Preserve an unsafe identity as data while rejecting it at both import boundaries."""
    client, repository = isolated_api
    _, transistor = source_backed_fixture()
    transistor.metadata.name = identity
    archive = core_to_json_document(transistor)
    response = client.post("/api/transistors", json=archive)
    assert response.status_code == 422, response.text
    uploaded = client.post("/api/transistors/upload", files={
        "file":("native-unsafe-name.json", json.dumps(archive).encode(), "application/json")})
    assert uploaded.status_code == 422, uploaded.text
    assert repository.list_all() == []
    assert not (tmp_path / "outside.json").exists()
    assert transistor.metadata.name == identity


def test_repository_absolute_names_rejected_before_any_writer(tmp_path, monkeypatch):
    """Reject absolute/drive names at storage access without changing the source identity."""
    repository = JsonTransistorRepository(tmp_path / "repository")
    _, transistor = source_backed_fixture()
    writes = []
    monkeypatch.setattr(JsonTransistorLoader, "save_to_json", lambda *args: writes.append(args))
    for identity in (str(tmp_path / "outside"), "C:\\outside", "C:outside", ".", ".."):
        transistor.metadata.name = identity
        with pytest.raises(RepositoryPathError):
            repository.save(transistor)
        with pytest.raises(RepositoryPathError):
            repository.get_by_name(identity)
        assert transistor.metadata.name == identity
    assert writes == []


def test_repository_existing_link_outside_rejected_without_target_change(tmp_path, isolated_api):
    """Resolve an existing link before either a read or save and preserve its target."""
    client, repository = isolated_api
    outside = tmp_path / "retained-outside.json"
    original = b"retained independent source bytes"
    outside.write_bytes(original)
    link = repository.data_directory / "linked-device.json"
    link.symlink_to(outside)
    _, transistor = source_backed_fixture()
    transistor.metadata.name = "linked-device"
    with pytest.raises(RepositoryPathError, match="exact repository"):
        repository.get_by_name("linked-device")
    with pytest.raises(RepositoryPathError, match="exact repository"):
        repository.save(transistor)
    response = client.post("/api/transistors", json=core_to_json_document(transistor))
    assert response.status_code == 422, response.text
    assert client.get("/api/transistors/linked-device").status_code == 422
    assert outside.read_bytes() == original
    assert link.is_symlink()
