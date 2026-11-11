"""Read the public downloads as a consumer using an in-memory repository."""
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree
from zipfile import ZipFile

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.io import loadmat

from test_public_export_independent_review import conditions, part, scl_blocks
from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.gui_web.backend.export_routes import create_export_router
from transistordatabase.plecs_document import finalize_plecs_package, parse_plecs_document


def client_for(*devices):
    """Give the router a lookup that cannot touch a file or stored device."""
    records = {d.metadata.name: d for d in devices}

    def existing(key):
        if key not in records:
            raise HTTPException(status_code=404, detail="No such fixture")
        return records[key]

    app = FastAPI()
    app.include_router(create_export_router(existing))
    return TestClient(app, raise_server_exceptions=False)


def test_http_byte_consumers_and_ratings_summary():
    """Read the same native document in a Unicode attachment, MAT5, and ZIP."""
    device = part()
    device.metadata.name = "P..Q Ω"
    device.metadata.manufacturer = " =untrusted"
    device.switch.metadata = {"source_id": 2**61 + 13, "sensor": None,
                              "shape": np.empty((1, 0, 2), dtype=np.int8)}
    expected = core_to_json_document(device)
    client = client_for(device)
    prefix = f"/api/transistors/{quote(device.metadata.name, safe='')}/export/"
    source = client.post(prefix + "json")
    assert source.status_code == 200
    assert b"2305843009213693965" in source.content
    assert json.loads(source.content) == expected
    assert quote("P..Q Ω.json", safe="") in source.headers["content-disposition"]
    mat = client.post(prefix + "matlab")
    assert mat.status_code == 200
    data = loadmat(BytesIO(mat.content), simplify_cells=True)
    assert json.loads(data["core_json"]) == expected
    numeric, = (value for key, value in data.items() if not key.startswith("_") and key != "core_json")
    np.testing.assert_array_equal(numeric["switch"]["channel"]["graph_v_i"],
                                  [[1.5, -3, 2.5, 0, -2], [6, -12, 22, 0, -4]])
    summary = client.post(prefix + "csv")
    assert summary.status_code == 200
    [row] = csv.DictReader(StringIO(summary.text))
    assert row["manufacturer"] == "' =untrusted"
    assert row["v_abs_max"] == "900"
    assert "ratings_summary.csv" in summary.headers["content-disposition"]
    batch = client.post("/api/transistors/batch_export", json={
        "transistor_ids": [device.metadata.name], "format": "json"})
    assert batch.status_code == 200
    with ZipFile(BytesIO(batch.content)) as archive:
        assert archive.read("P..Q Ω/P..Q Ω.json") == source.content
    assert core_to_json_document(device) == expected


def test_per_device_gecko_slices_preserve_selection_and_scaled_data():
    """Check two different gate contexts and solve corner energies by hand."""
    a, b = part(), part(.5)
    a.metadata.name, b.metadata.name = "A", "B"
    for record in (b.switch.channel_data[0], b.switch.e_on_data[1], b.switch.e_off_data[1]):
        record.v_g = None
    for record in (b.switch.e_on_data[1], b.switch.e_off_data[1]):
        record.r_g = 0
    selected = {"A": conditions(), "B": conditions(v_g_on=None, r_g_on=0, r_g_off=0)}
    request = {"transistor_ids": ["A", "B"], "format": "gecko", "options": {"selections": selected}}
    client = client_for(a, b)
    response = client.post("/api/transistors/batch_export", json=request)
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        assert len(archive.namelist()) == 4
        for name, scale, device in (("A", 1, a), ("B", .5, b)):
            stem = f"{name}/{name}_Switch"
            (_, _), (energy, observed) = scl_blocks(archive.read(stem + ".scl"))
            assert observed == {"tj": 37, "uBlock": 430}
            np.testing.assert_array_equal(energy[0], [3, 5, 9, 13, 16, 17])
            np.testing.assert_allclose(energy[1:] / (scale * 1e-6),
                                      [[9, 15, 17, 19, 28, 31], [24, 36, 60, 72, 81, 82]])
            manifest = json.loads(archive.read(stem + ".manifest.json"))
            assert manifest["selection"] == selected[name]
            assert manifest["core_document"] == core_to_json_document(device)
    selected["B"]["v_g_on"] = 0
    failure = client.post("/api/transistors/batch_export", json=request)
    assert failure.status_code == 422
    assert failure.headers["content-type"].startswith("application/json")
    assert "content-disposition" not in failure.headers


@pytest.mark.parametrize("second,status", [("missing", 404), ("../B", 422), ("a", 422)])
def test_batch_rejects_invalid_second_id_before_publishing_archive(second, status):
    """A valid first item does not turn an invalid batch into a download."""
    device = part()
    device.metadata.name = "A"
    response = client_for(device).post("/api/transistors/batch_export", json={
        "transistor_ids": ["A", second], "format": "matlab"})
    assert response.status_code == status
    assert "content-disposition" not in response.headers
    assert response.json()["detail"]


def test_source_backed_plecs_scientific_corner_and_stale_source_guard():
    """Read four physical quantities from returned XML and reject a changed corner."""
    base = Path(__file__).parent / "test_data/sample_plecs_switch.xml"
    document = parse_plecs_document(base.read_bytes())
    device = finalize_plecs_package(document, package_id="0",
        ratings={"v_abs_max": 875., "i_abs_max": 52., "i_cont": 35., "t_j_max": 160.},
        thermal_properties={"housing_area": .001, "cooling_area": .002},
        block_targets={"0": "switch"}, gate_voltage_by_table={
            t.source_path: 13. for t in document.packages[0].blocks[0].tables})
    client = client_for(device)
    uri = f"/api/transistors/{device.metadata.name}/export/plecs"
    received = client.post(uri)
    assert received.status_code == 200
    tree = ElementTree.fromstring(received.content)
    corner = tree.find(".//{*}TurnOffLoss/{*}Energy")
    assert corner is not None
    voltages = corner.findall("{*}Temperature")[-1].findall("{*}Voltage")
    assert float(voltages[-1].text.split()[-1]) * float(corner.get("scale", 1)) == pytest.approx(.00135)
    assert tree.find(".//{*}TurnOffLoss/{*}TemperatureAxis").text.split()[-1] in ("150", "150.0")
    assert float(tree.find(".//{*}TurnOffLoss/{*}VoltageAxis").text.split()[-1]) == 800.
    assert float(tree.find(".//{*}TurnOffLoss/{*}CurrentAxis").text.split()[-1]) == 50.
    device.switch.e_off_data[-1].graph_i_e[1, -1] = .7
    stale = client.post(uri)
    assert stale.status_code == 422
    assert "content-disposition" not in stale.headers


@pytest.mark.parametrize("comment", ["[]", "null", '{"plecs_document_version": 1}'])
def test_plecs_rejects_non_mapping_or_incomplete_source_metadata(comment):
    """Ordinary comments cannot produce an internal server error on export."""
    device = part()
    device.metadata.comment = comment
    error = client_for(device).post("/api/transistors/example/export/plecs")
    assert error.status_code == 422
    assert error.json()["detail"]
