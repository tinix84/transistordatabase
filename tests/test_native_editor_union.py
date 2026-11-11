"""Exercise existing web editors with source-backed native records in a real repository."""
from __future__ import annotations

import base64
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
from spicelib import RawWrite, Trace

from transistordatabase.core.models import MeasurementEvidence
from transistordatabase.core.repository import JsonTransistorRepository
from transistordatabase.core.serialization import core_from_json_document, core_to_json_document
from transistordatabase.gui_web.backend import main
from transistordatabase.plecs_document import finalize_plecs_package, parse_plecs_document


@pytest.fixture
def uploaded(tmp_path, monkeypatch):
    """Upload a native device with four known thermal poles and source trace bytes."""
    repository = JsonTransistorRepository(tmp_path / "native")
    monkeypatch.setattr(main, "_repo", repository)
    client = TestClient(main.app)
    source = Path(__file__).parent / "test_data" / "sample_plecs_switch.xml"
    xml = source.read_bytes()
    document = parse_plecs_document(xml, source.name)
    device = finalize_plecs_package(
        document, package_id="0", block_targets={"0": "switch"},
        ratings={"v_abs_max": 1200., "i_abs_max": 75., "i_cont": 40., "t_j_max": 175.},
        thermal_properties={"housing_area": .0001, "cooling_area": .0002},
        gate_voltage_by_table={t.source_path: 15. for t in document.packages[0].blocks[0].tables},
    )
    trace = b"current_A,energy_J\r\n4,0.001\r\n"
    device.switch.e_on_data[0].evidence = MeasurementEvidence(
        filename="measured.csv", source_kind="test-trace",
        sha256=hashlib.sha256(trace).hexdigest(), original_base64=base64.b64encode(trace).decode(),
    )
    archive = core_to_json_document(device)
    response = client.post("/api/transistors/upload", files={
        "file": ("test-native.json", json.dumps(archive), "application/json")
    })
    assert response.status_code == 200, response.text
    path = repository.data_directory / f"{device.metadata.name}.json"
    assert json.loads(path.read_bytes()) == archive
    assert archive["schema"] == "transistordatabase-core"
    assert base64.b64decode(json.loads(device.metadata.comment)["source_base64"]) == xml
    return client, device, path


def test_general_form_updates_fields_and_keeps_complete_native_archive(uploaded):
    """Compare an independently updated model with the complete on-disk record."""
    client, expected, path = uploaded
    params = {"name": expected.metadata.name}
    read = client.get("/api/transistor-record", params=params)
    assert read.status_code == 200, read.text
    form = {key: read.json()[key] for key in ("metadata", "electrical", "thermal")}
    form["metadata"].update(author="Edited author", manufacturer="Infineon", housing_type="TO247")
    form["electrical"]["i_cont"] = 37.5
    response = client.put("/api/transistor-record", params=params, json=form)
    assert response.status_code == 200, response.text
    expected.metadata.author = "Edited author"
    expected.metadata.manufacturer = "Infineon"
    expected.metadata.housing_type = "TO247"
    expected.electrical_ratings.i_cont = 37.5
    assert json.loads(path.read_bytes()) == core_to_json_document(expected)


def test_channel_add_edit_stale_request_and_removal_leave_original_native_curves(uploaded):
    """Exercise the browser token and restore the complete initial curve population."""
    client, expected, path = uploaded
    endpoint = f"/api/transistors/{expected.metadata.name}/curves"
    index = len(expected.switch.channel_data)
    added = client.post(f"{endpoint}/channel", params={"component": "switch"}, json={
        "t_j": 100., "v_g": 12., "v_data": [0., 1.], "i_data": [0., 10.],
    })
    assert added.status_code == 200, added.text
    read = client.get(endpoint)
    assert read.status_code == 200, read.text
    curves = read.json()["switch"]["channel_data"]
    assert len(curves) == index + 1
    assert curves[index]["graph_v_i"] == [[0., 1.], [0., 10.]]
    url = f"{endpoint}/switch/channel/{index}"
    graph = [[0., 2.], [0., 10.]]
    edit = {"expected_token": curves[index]["_edit_token"], "expected_count": len(curves),
            "curve": {"graph_v_i": graph}}
    updated = client.put(url, json=edit)
    assert updated.status_code == 200, updated.text
    assert client.get(endpoint).json()["switch"]["channel_data"][index]["graph_v_i"] == graph
    saved = path.read_bytes()
    assert client.put(url, json=edit).status_code == 409
    assert path.read_bytes() == saved
    removed = client.delete(url)
    assert removed.status_code == 200, removed.text
    assert json.loads(path.read_bytes()) == core_to_json_document(expected)


def test_editing_source_backed_switching_curve_retains_measurement_evidence(uploaded):
    """Change one energy trace while retaining source bytes and matched conditions."""
    client, expected, path = uploaded
    endpoint = f"/api/transistors/{expected.metadata.name}/curves"
    read = client.get(endpoint)
    assert read.status_code == 200, read.text
    curves = read.json()["switch"]["e_on_data"]
    graph = [[0., 10., 25.], [0., .001, .003]]
    response = client.put(f"{endpoint}/switch/e_on/0", json={
        "expected_count": len(curves), "expected_token": curves[0]["_edit_token"],
        "curve": {"graph_i_e": graph},
    })
    assert response.status_code == 200, response.text
    expected.switch.e_on_data[0].graph_i_e = np.array(graph)
    assert json.loads(path.read_bytes()) == core_to_json_document(expected)


def test_measured_thermal_edits_preserve_foster_poles_and_source_xml(uploaded):
    """Store measured impedance alongside the four original PLECS Foster poles."""
    client, expected, path = uploaded
    curves = f"/api/transistors/{expected.metadata.name}/curves"
    endpoint = f"{curves}/thermal/switch"
    assert expected.switch.thermal_foster.r_th_vector == [.05, .1, .15, .2]
    assert expected.switch.thermal_foster.tau_vector == [.001, .01, .1, 1.]
    for method, ordinates in [(client.post, [.05, .2]), (client.put, [.07, .23])]:
        read = client.get(curves)
        assert read.status_code == 200, read.text
        token = read.json()["thermal_foster"]["switch"]["_edit_token"]
        graph = [[.001, 1.], ordinates]
        response = method(endpoint, json={"expected_token": token, "graph_t_rthjc": graph})
        assert response.status_code == 200, response.text
        expected.switch.thermal_foster.graph_t_rthjc = np.array(graph)
        assert json.loads(path.read_bytes()) == core_to_json_document(expected)


def test_native_backup_restores_missing_device_and_keeps_conflicting_bytes(uploaded, tmp_path, monkeypatch):
    """Restore a native archive alongside a distinct existing version in one operation."""
    client, device, source_path = uploaded
    exported = client.get("/api/database/backup")
    assert exported.status_code == 200, exported.text
    document = exported.json()
    assert document["records"] == [json.loads(source_path.read_bytes())]
    existing_name = device.metadata.name
    device.metadata.name = "Second native device"
    second = core_to_json_document(device)
    document["records"].append(second)
    destination = JsonTransistorRepository(tmp_path / "restored")
    monkeypatch.setattr(main, "_repo", destination)
    device.metadata.name = existing_name
    device.metadata.author = "Existing destination version"
    destination.save(device)
    existing_path = destination.data_directory / f"{existing_name}.json"
    original_bytes = existing_path.read_bytes()
    files = {"file": ("backup.json", json.dumps(document), "application/json")}
    endpoint = "/api/database/backup"
    preview = client.post(f"{endpoint}/validate", files=files)
    assert preview.status_code == 200, preview.text
    assert preview.json()["conflicts"] == [existing_name]
    assert preview.json()["restore_names"] == ["Second native device"]
    assert destination.list_all() == [existing_name]
    result = client.post(f"{endpoint}/restore", files=files)
    assert result.status_code == 200, result.text
    assert result.json()["restored"] == ["Second native device"]
    assert result.json()["conflicts"] == [existing_name]
    assert existing_path.read_bytes() == original_bytes
    assert json.loads((destination.data_directory / "Second native device.json").read_bytes()) == second


def test_native_raw_import_preserves_xml_and_reopens_measured_energy(uploaded, tmp_path):
    """Integrate two constant-power windows after saving and reloading native RAW data."""
    client, original, record_path = uploaded
    base = f"/api/transistors/{original.metadata.name}"
    selection = {
        "time_trace": "time", "vds_trace": "V(ds)", "id_trace": "I(M1)",
        "turn_on": {"start_s": 0., "end_s": 2e-6},
        "turn_off": {"start_s": 4e-6, "end_s": 6e-6},
        "metadata": {"t_j": 80., "v_supply": 300., "v_g": 12., "v_g_off": -3.,
                     "r_g": 4., "r_g_off": 5., "load_inductance": .75},
    }
    hashes = []
    for current in (5., 10.):
        path = tmp_path / f"pulse-{current:g}A.raw"
        writer = RawWrite(plot_name="Transient Analysis", fastacces=False)
        writer.add_trace(Trace("time", np.arange(7) * 1e-6, whattype="time"))
        writer.add_trace(Trace("V(ds)", np.full(7, 200.)))
        writer.add_trace(Trace("I(M1)", np.full(7, current), whattype="current"))
        writer.save(str(path))
        content = path.read_bytes()
        hashes.append(hashlib.sha256(content).hexdigest())
        response = client.post(
            f"{base}/dpt_raw_import",
            files={"file": (path.name, content, "application/octet-stream")},
            data={"selection_json": json.dumps(selection)},
        )
        assert response.status_code == 200, response.text
    saved_bytes = record_path.read_bytes()
    saved = core_from_json_document(json.loads(saved_bytes))
    assert len(saved.switch.raw_measurement_data) == 2
    for index, raw in enumerate(saved.switch.raw_measurement_data):
        assert raw.evidence.sha256 == hashes[index]
        assert raw.evidence.parser == "spicelib.RawRead"
        assert raw.evidence.parser_version == version("spicelib")
        np.testing.assert_allclose(raw.dpt_on_vds[0], [[0., 200.], [1e-6, 200.], [2e-6, 200.]])
    saved.switch.raw_measurement_data = []
    assert core_to_json_document(saved) == core_to_json_document(original)

    read = client.get(f"{base}/dpt_data")
    assert read.status_code == 200, read.text
    assert len(read.json()["raw_measurement_data"]) == 2
    points = [{"measurement_index": i, "acquisition_index": 0, "i_x": current,
               "vds_sign": 1, "id_sign": 1} for i, current in enumerate((5., 10.))]
    preview = client.post(f"{base}/dpt_energy/preview", json={"loss_type": "e_on", "points": points})
    assert preview.status_code == 200, preview.text
    np.testing.assert_allclose(preview.json()["graph_i_e"], [[5., 10.], [.002, .004]])
    assert record_path.read_bytes() == saved_bytes


@pytest.mark.parametrize("voltage", ["0.002", "1e-4000", "1e4000"])
def test_native_extended_curve_projection_preserves_stored_dtype(uploaded, voltage):
    """Return browser numbers only when the finite native samples fit that range."""
    if voltage != "0.002" and np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp:
        pytest.skip("This platform has no extended floating-point exponent range")
    client, device, first_path = uploaded
    device.metadata.name = "Extended channel samples"
    device.switch.channel_data[0].graph_v_i = np.array(
        [["0", voltage], ["0", "5"]], dtype=np.longdouble
    )
    archive = core_to_json_document(device)
    imported = client.post("/api/transistors/upload", files={
        "file": ("extended.json", json.dumps(archive), "application/json")
    })
    assert imported.status_code == 200, imported.text
    path = first_path.with_name(f"{device.metadata.name}.json")
    original_bytes = path.read_bytes()
    response = client.get(f"/api/transistors/{device.metadata.name}/curves")
    if voltage == "0.002":
        assert response.status_code == 200, response.text
        assert response.json()["switch"]["channel_data"][0]["graph_v_i"] == [[0., .002], [0., 5.]]
    else:
        assert response.status_code in (400, 422), response.text
    assert path.read_bytes() == original_bytes
    assert json.loads(original_bytes) == archive
