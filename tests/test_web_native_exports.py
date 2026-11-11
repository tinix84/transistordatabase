"""Consumer checks for validated complete-device HTTP downloads."""
from copy import deepcopy
from io import BytesIO
import json
import csv
import runpy
from pathlib import Path
from urllib.parse import quote
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.io import loadmat

from transistordatabase.core.serialization import core_from_json_document, core_to_json_document
from transistordatabase.gui_web.backend.export_routes import create_export_router

library = runpy.run_path(str(Path(__file__).with_name("test_public_export_independent_review.py")))


@pytest.fixture
def client():
    """Use a private device lookup without filesystem writes."""
    device = library["part"]()
    device.switch.metadata["source_id"] = 2**60 + 1
    other = deepcopy(device)
    other.metadata.name = "other"
    devices = {"example": device, "other": other}
    for name in ("My device", "Módulo 雪", "a..b"):
        named = deepcopy(device)
        named.metadata.name = name
        devices[name] = named
    device.metadata.manufacturer = "=Source()"
    device.metadata.type = " +Source()"
    device.metadata.housing_type = "@Source()"
    device.electrical_ratings.i_cont = -3
    plecs_library = runpy.run_path(str(Path(__file__).with_name("test_plecs_core_export.py")))
    plecs_device, plecs_selection = plecs_library["fixture"]()
    devices[plecs_device.metadata.name] = plecs_device
    from transistordatabase.plecs_document import finalize_plecs_package, parse_plecs_document
    source = parse_plecs_document((Path(__file__).parent / "test_data/sample_plecs_switch.xml").read_bytes())
    source_device = finalize_plecs_package(source, package_id="0",
        ratings=dict(v_abs_max=1200., i_abs_max=75., i_cont=40., t_j_max=175.),
        thermal_properties=dict(housing_area=.0001, cooling_area=.0002),
        block_targets={"0": "switch"},
        gate_voltage_by_table={t.source_path: 15. for t in source.packages[0].blocks[0].tables})
    devices["source-backed"] = source_device

    def lookup(name):
        if name not in devices:
            raise HTTPException(404, "Unknown device")
        return devices[name]

    app = FastAPI()
    app.include_router(create_export_router(lookup))
    return TestClient(app)


@pytest.mark.parametrize("name", ["My device", "Módulo 雪", "a..b"])
def test_safe_existing_names_preserve_native_identity_and_utf8_download_header(client, name):
    """Keep benign identity characters in both JSON and safe archive members."""
    response = client.post(f"/api/transistors/{quote(name, safe='')}/export/json", json={})
    assert response.status_code == 200
    assert core_from_json_document(response.json()).metadata.name == name
    assert "filename*=UTF-8''" + quote(name + ".json", safe="") in response.headers["content-disposition"]
    response = client.post("/api/transistors/batch_export", json={"transistor_ids": [name], "format": "json"})
    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == [f"{name}/{name}.json"]


def test_csv_neutralizes_text_formulas_and_preserves_numeric_sign(client):
    """Neutralize metadata formulas without changing numerical fields."""
    response = client.post("/api/transistors/example/export/csv", json={})
    assert response.status_code == 200
    [row] = csv.DictReader(response.text.splitlines())
    assert row["manufacturer"] == "'=Source()"
    assert row["type"] == "' +Source()"
    assert row["housing_type"] == "'@Source()"
    assert row["i_cont"] == "-3"


def test_existing_source_backed_and_exact_selected_plecs_exports_remain_available(client):
    """Keep both accepted PLECS export contracts available."""
    response = client.post("/api/transistors/source-backed/export/plecs", json={})
    assert response.status_code == 200
    from transistordatabase.plecs_document import parse_plecs_document
    assert parse_plecs_document(response.content).can_export
    plecs_library = runpy.run_path(str(Path(__file__).with_name("test_plecs_core_export.py")))
    device, selection = plecs_library["fixture"]()
    response = client.post(f"/api/transistors/{device.metadata.name}/export/plecs", json=selection)
    assert response.status_code == 200
    assert parse_plecs_document(response.content).packages[0].fixed_conditions["selection"] == selection
    response = client.post("/api/transistors/batch_export", json={"transistor_ids": [device.metadata.name],
        "format": "plecs", "options": {"selection": selection}})
    assert response.status_code == 200


def test_native_json_and_mat5_use_complete_source_bytes(client):
    """Restore the exact core record through two independent consumer formats."""
    response = client.post("/api/transistors/example/export/json", json={})
    assert response.status_code == 200
    assert b"1152921504606846977" in response.content
    restored = core_from_json_document(response.json())
    assert restored.switch.metadata["source_id"] == 2**60 + 1
    np.testing.assert_array_equal(restored.switch.channel_data[0].graph_v_i, library["part"]().switch.channel_data[0].graph_v_i)
    response = client.post("/api/transistors/example/export/matlab", json={})
    assert response.status_code == 200
    mat = loadmat(BytesIO(response.content), simplify_cells=True)
    restored = core_from_json_document(json.loads(str(mat["core_json"])))
    assert restored.switch.metadata["source_id"] == 2**60 + 1
    assert np.isnan(mat["example"]["switch"]["thermal_foster"]["r_th_total"])


def test_gecko_archive_retains_both_files_exact_curves_and_native_manifest(client):
    """Check SCL bytes, explicit conditions, coverage, and native provenance."""
    selection = library["conditions"]()
    response = client.post("/api/transistors/example/export/gecko", json=selection)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-content-type-options"] == "nosniff"
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"example_Switch.scl", "example_Switch.manifest.json"}
        manifest = json.loads(archive.read("example_Switch.manifest.json"))
        assert manifest["selection"] == selection
        assert manifest["switching_current_coverage_a"] == [3, 17]
        restored = core_from_json_document(manifest["core_document"])
        assert restored.switch.metadata["source_id"] == 2**60 + 1
        from transistordatabase.native_export import render_gecko
        assert archive.read("example_Switch.scl") == render_gecko(restored, selection)["example_Switch.scl"]


@pytest.mark.parametrize("changes", [{"r_g_on": -1}, {"v_supply": 0}, {"v_g_on": True},
    {"output_path": "/tmp/not-used"}, {"recheck": False}, {"unexpected": 1}, {"component": "diode"}, {"t_j": 38}])
def test_bad_conditions_return_error_without_download(client, changes):
    """Reject unsupported or invalid selectors before returning a file."""
    response = client.post("/api/transistors/example/export/gecko", json=library["conditions"](**changes))
    assert response.status_code == 422
    assert "content-disposition" not in response.headers


def test_missing_nullable_keys_and_nonfinite_tokens_are_rejected(client):
    """Require explicit unknown conditions and JSON-safe finite validation."""
    assert client.post("/api/transistors/example/export/gecko", json={}).status_code == 422
    conditions = library["conditions"]()
    conditions.pop("v_g_off")
    assert client.post("/api/transistors/example/export/gecko", json=conditions).status_code == 422
    for token in ("NaN", "Infinity", "-Infinity"):
        body = json.dumps(library["conditions"]()).replace('"t_j": 37', '"t_j": ' + token)
        response = client.post("/api/transistors/example/export/gecko", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)


def test_batch_validates_all_ids_and_conditions_before_returning_zip(client):
    """Require every requested device and its exact source selection."""
    request = {"transistor_ids": ["example", "other"], "format": "gecko",
               "options": {"selection": library["conditions"]()}}
    response = client.post("/api/transistors/batch_export", json=request)
    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert len(archive.namelist()) == 4
        assert len([name for name in archive.namelist() if name.endswith("manifest.json")]) == 2
    request["options"] = {"selections": {name: library["conditions"]() for name in request["transistor_ids"]}}
    assert client.post("/api/transistors/batch_export", json=request).status_code == 200
    request["options"]["selections"].pop("other")
    assert client.post("/api/transistors/batch_export", json=request).status_code == 422
    request["options"] = {"selection": library["conditions"]()}
    request["transistor_ids"] = ["example", "missing"]
    response = client.post("/api/transistors/batch_export", json=request)
    assert response.status_code == 404
    assert "content-disposition" not in response.headers


def test_output_budget_returns_an_error_before_download_headers(client, monkeypatch):
    """Reject oversized single and aggregate downloads before returning an archive."""
    from transistordatabase.gui_web.backend import export_routes
    monkeypatch.setattr(export_routes, "MAX_BYTES", 30)
    for path, body in [("/api/transistors/example/export/json", {}),
                       ("/api/transistors/batch_export", {"transistor_ids": ["example", "other"], "format": "json"})]:
        response = client.post(path, json=body)
        assert response.status_code == 422
        assert "content-disposition" not in response.headers


def test_archive_rejects_case_colliding_or_unsafe_renderer_names():
    """Require safe unique paths even when a renderer returns unexpected filenames."""
    from transistordatabase.gui_web.backend.export_routes import _archive
    for files in ({"A.json": b"a", "a.json": b"b"}, {"../part.json": b"a"}, {"a\\b.json": b"a"}):
        with pytest.raises(ValueError):
            _archive(files)


@pytest.mark.parametrize("ids", [["../bad"], ["a\\b"], ["example", "EXAMPLE"], ["example", "example"], ["con"], ["bad\r\nheader"]])
def test_unsafe_or_duplicate_ids_never_become_archive_names(client, ids):
    """Reject traversal, control characters, reserved names, and duplicates."""
    assert client.post("/api/transistors/batch_export", json={"transistor_ids": ids, "format": "json"}).status_code == 422


@pytest.mark.parametrize("format", ["json", "csv", "matlab"])
def test_batch_formats_have_real_bytes_and_reject_ignored_options(client, format):
    """Read each batch format with its consumer and reject unused options."""
    request = {"transistor_ids": ["example"], "format": format}
    response = client.post("/api/transistors/batch_export", json=request)
    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        [name] = archive.namelist()
        data = archive.read(name)
        if format == "json":
            assert core_from_json_document(json.loads(data)).switch.metadata["source_id"] == 2**60 + 1
        elif format == "matlab":
            assert "core_json" in loadmat(BytesIO(data))
        else:
            assert "ratings_summary.csv" in name
            assert data.startswith(b"name,type,manufacturer,v_abs_max")
    request["options"] = {"selection": library["conditions"]()}
    assert client.post("/api/transistors/batch_export", json=request).status_code == 422


@pytest.mark.parametrize("shape", [[-1], [True], [1.0], [0, 10_000_001], [2, 0, -1], [1, 0, 3], [2, 3]])
def test_empty_array_shape_cannot_reinterpret_values_or_allocate_unbounded_dimensions(shape):
    """Reject invalid or unbounded declared shapes before reshaping empty arrays."""
    device = library["part"]()
    device.switch.metadata["empty"] = np.empty((2, 0, 3))
    document = core_to_json_document(device)
    encoded = document["transistor"]["fields"]["switch"]["fields"]["metadata"]["values"]["empty"]
    encoded["shape"] = shape
    with pytest.raises(ValueError):
        core_from_json_document(document)


@pytest.mark.parametrize("format", ["spice", "ltspice", "simulink"])
def test_simulator_preview_requires_explicit_generic_dpt_workflow(client, format):
    """Explain the supported scientific path without inventing a calibrated model."""
    response = client.get(f"/api/transistors/example/export/{format}/preview")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "calibrated simulator model is unavailable" in detail
    assert "DPT workbench" in detail
    assert "explicit generic templates" in detail


def test_source_backed_plecs_preview_matches_validated_download_and_byte_budget(client, monkeypatch):
    """Preview actual source-backed XML within the existing line and byte limits."""
    from transistordatabase.gui_web.backend import export_routes

    uri = "/api/transistors/source-backed/export/plecs"
    download = client.post(uri, json={})
    assert download.status_code == 200
    response = client.get(uri + "/preview")
    assert response.status_code == 200
    result = response.json()
    lines = download.content.decode("utf-8").splitlines()
    assert result["preview"] == "\n".join(lines[:100])
    assert result["total_lines"] == len(lines)
    assert result["preview_lines"] == min(100, len(lines))
    assert result["file_size"] == len(download.content)
    assert result["truncated"] == (len(lines) > 100)
    assert result["file_name"].endswith(".xml")

    monkeypatch.setattr(export_routes, "MAX_BYTES", len(download.content) - 1)
    bounded = client.get(uri + "/preview")
    assert bounded.status_code == 422
    assert "64 MiB" in bounded.json()["detail"]
    assert "preview" not in bounded.json()


def test_plecs_preview_never_selects_generic_core_conditions(client):
    """Require source evidence instead of selecting the first recorded core slice."""
    plecs_library = runpy.run_path(str(Path(__file__).with_name("test_plecs_core_export.py")))
    device, _ = plecs_library["fixture"]()
    response = client.get(f"/api/transistors/{device.metadata.name}/export/plecs/preview")
    assert response.status_code == 422
    assert "source-backed" in response.json()["detail"]
