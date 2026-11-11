"""Independent numerical and format acceptance for the 8A/8D adapters."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase.backend.concrete_services import ComparisonService, ExportService
from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, SwitchingLossData,
    ThermalProperties, Transistor, TransistorMetadata,
    EffectiveOutputCapacitance, RawMeasurementData, VoltageDependentCapacitance,
    LinearizedModel,
)
from transistordatabase.core.repository import JsonTransistorLoader, JsonTransistorRepository
from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.gui_web.backend.plecs_routes import router
from transistordatabase.gui_web.backend.plecs_routes import plecs_repository
from transistordatabase.plecs_core_export import core_selection_document
from transistordatabase.plecs_document import (
    NAMESPACE, finalize_plecs_package, parse_plecs_document,
    render_core_plecs, render_plecs_document,
)
from transistordatabase.static_losses import static_duty_losses

SAMPLE = Path(__file__).parent / "test_data/sample_plecs_switch.xml"
SC = Path("/home/tinix/claude_wsl/archive_ntbees2/sc")
ENCO = Path("/mnt/d/OneDrive/claude/bhy/enco/enco")


def review_device(name="review"):
    """Use independent finite E(0) evidence and different on/off supplies."""
    transistor = Transistor(
        TransistorMetadata(name, "IGBT", "review", "fixture", "fixture"),
        ElectricalRatings(1200, 100, 50, 175), ThermalProperties(.001, .001),
    )
    transistor.switch.channel_data = [ChannelCharacteristics(
        40, np.array([[1., 3.], [0., 20.]]), 12)]
    for field, supply, rg, energy in (
        ("e_on_data", 400, 2, [.0002, .0012, .0022]),
        ("e_off_data", 600, 5, [.0004, .0024, .0044]),
    ):
        setattr(transistor.switch, field, [SwitchingLossData(
            "graph_i_e", 40, supply, 12, r_g=rg, v_g_off=-4,
            graph_i_e=np.array([[0., 10., 20.], energy]))])
    return transistor


def review_op():
    """Declare each operating condition without defaults."""
    return dict(t_j=40, v_g=12, v_g_off=-4, i_channel=10,
                f_sw=100_000, duty=.25, v_supply=400, r_g_on=2, r_g_off=5)


def mapping(document):
    """Keep explicit fixture declarations separate from the source."""
    return dict(package_id="0", ratings=dict(v_abs_max=1200., i_abs_max=80.,
        i_cont=40., t_j_max=175.), thermal_properties=dict(housing_area=.0001,
        cooling_area=.0002), block_targets={"0": "switch"},
        gate_voltage_by_table={t.source_path: 15. for t in document.packages[0].blocks[0].tables},
        recovery_voltage_convention="preserve-signed")


def client():
    """Expose only the production typed PLECS router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_legacy_different_measured_supplies_are_not_borrowed():
    """A single bus request cannot match 400 V Eon and 600 V Eoff."""
    with pytest.raises(ValueError, match="exact"):
        static_duty_losses(review_device(), review_op())


@pytest.mark.parametrize("current,count,expected", [(10, 1, 365.), (10, 2, 423.75), (0, 3, 180.)])
def test_parallel_total_current_and_nonzero_zero_current_energy(current, count, expected):
    """E(I)=E0+kI means device multiplicity scales E0 as well as current."""
    a, b = review_device(), review_device("second")
    for transistor in (a, b):
        transistor.switch.e_off_data[0].v_supply = 400
    point = review_op() | {"i_channel": current}
    arrays = {"v_g", "v_g_off", "v_supply", "r_g_on", "r_g_off"}
    config = {k: [v, v] if k in arrays else v for k, v in point.items()}
    config["parallel_count"] = [count, 1]
    before = deepcopy(config)
    result = ComparisonService().generate_advanced_comparison([a, b], config)
    row = result["loss_breakdown"]["rows"][0]
    assert row["total"] == pytest.approx(expected)
    assert row["calculation"]["conditions"]["i_channel"] == current / count
    assert result["efficiency"]["rows"][0]["efficiency_percent"] is None
    assert config == before


@pytest.mark.parametrize("key,value", [("t_j", 40.000001), ("v_g", None), ("r_g_on", None), ("v_g_off", -3.999999)])
def test_nearby_or_unknown_conditions_do_not_match_measured_values(key, value):
    """Require exact known or explicitly unknown conditions."""
    transistor = review_device()
    transistor.switch.e_off_data[0].v_supply = 400
    with pytest.raises(ValueError):
        static_duty_losses(transistor, review_op() | {key: value})


def test_invalid_record_condition_preserves_other_device_observation_and_panel():
    """Keep valid recorded evidence when another source condition is invalid."""
    a, b = review_device(), review_device("valid")
    a.switch.channel_data[0].t_j = float("nan")
    result = ComparisonService().generate_advanced_comparison([a, b], {})
    assert [x["device"] for x in result["channel_characteristics"]["curves"]] == ["valid"]
    assert result["channel_characteristics"]["diagnostics"][0]["status"] == "invalid-data"
    assert len(result["switching_losses_eon"]["curves"]) == 2
    assert all(x["total"] is None for x in result["loss_breakdown"]["rows"])


@pytest.mark.parametrize("defect", ["unknown-rating", "missing-rating", "bool-rating", "unknown-area", "negative-interface"])
def test_typed_finalize_invalid_declarations_are_422(defect):
    """Reject malformed declarations before constructing core records."""
    doc = parse_plecs_document(SAMPLE.read_bytes())
    args = mapping(doc)
    if defect == "unknown-rating":
        args["ratings"]["invented"] = 123
    elif defect == "missing-rating":
        args["ratings"].pop("i_cont")
    elif defect == "bool-rating":
        args["ratings"]["i_cont"] = True
    elif defect == "unknown-area":
        args["thermal_properties"]["invented"] = 123
    else:
        args["thermal_properties"]["r_th_cs"] = -1
    response = client().post("/api/plecs/finalize", json={"xml_base64": base64.b64encode(SAMPLE.read_bytes()).decode(), **args})
    assert response.status_code == 422, response.text


def test_core_json_save_reload_then_source_export_is_semantically_exact(tmp_path):
    """Persist declarations and all arrays through the actual existing loader."""
    doc = parse_plecs_document(SAMPLE.read_bytes())
    transistor = finalize_plecs_package(doc, **mapping(doc))
    path = tmp_path / "review-source-backed.json"
    loader = JsonTransistorLoader()
    loader.save_to_json(transistor, path)
    restored = loader.load_from_json(path)
    result = parse_plecs_document(render_core_plecs(restored))
    assert result.packages == doc.packages
    assert restored.switch.e_on_data[3].r_g is None
    assert restored.electrical_ratings.v_abs_max == 1200
    exported_path = tmp_path / "review-source-service-export.json"
    assert ExportService().export_to_json(restored, exported_path)
    restored_again = loader.load_from_json(exported_path)
    assert parse_plecs_document(render_core_plecs(restored_again)).packages == doc.packages


def test_http_finalization_native_download_reloads_with_complete_source(tmp_path):
    """The actual downloadable JSON reloads and retains its source roundtrip."""
    doc = parse_plecs_document(SAMPLE.read_bytes())
    response = client().post("/api/plecs/finalize", json={
        "xml_base64": base64.b64encode(SAMPLE.read_bytes()).decode(), **mapping(doc)})
    assert response.status_code == 200, response.text
    path = tmp_path / "http-declared-core-download.json"
    path.write_text(json.dumps(response.json()["core_json_document"]), encoding="utf-8")
    restored = JsonTransistorLoader().load_from_json(path)
    assert parse_plecs_document(render_core_plecs(restored)).packages == doc.packages


def test_native_full_json_export_restores_arrays_dates_unknowns_and_working_point(tmp_path):
    """Retain complete aggregate values through the public export service."""
    transistor = review_device("Unlisted vendor record")
    stamp = datetime(2026, 10, 16, 12, 34, 56, tzinfo=timezone.utc)
    transistor.metadata.datasheet_date = stamp
    transistor.metadata.comment = "Unicode provenance é; unknowns are explicit"
    transistor.switch.e_on_data[0].measurement_date = stamp
    transistor.c_oss = [VoltageDependentCapacitance(40, np.array([[0., 400.], [1e-9, 1e-10]]))]
    transistor.c_oss_er = EffectiveOutputCapacitance(2e-10, 0, 400)
    transistor._working_point = {"switch": LinearizedModel(40, 10, .1, 1., 12), "notes": [None, "recorded"]}
    transistor.switch.raw_measurement_data = [RawMeasurementData("dpt_u_i", measurement_date=stamp,
        dpt_on_id=[np.array([[0., 1e-6], [0., 10.]])])]
    path = tmp_path / "full-native-export.json"
    assert ExportService().export_to_json(transistor, path)
    restored = JsonTransistorLoader().load_from_json(path)
    assert restored.metadata == transistor.metadata
    assert restored.switch.e_on_data[0].measurement_date == stamp
    assert restored.switch.e_on_data[0].commutation_device is None
    np.testing.assert_array_equal(restored.switch.e_on_data[0].graph_i_e, transistor.switch.e_on_data[0].graph_i_e)
    np.testing.assert_array_equal(restored.c_oss[0].graph_v_c, transistor.c_oss[0].graph_v_c)
    assert restored.c_oss_er == transistor.c_oss_er
    assert restored.working_point == transistor.working_point
    np.testing.assert_array_equal(restored.switch.raw_measurement_data[0].dpt_on_id[0], transistor.switch.raw_measurement_data[0].dpt_on_id[0])


def test_existing_legacy_json_still_loads_without_source_mutation():
    """Native serialization must retain old flat JSON compatibility."""
    path = SAMPLE.parent / "CREE_C3M0060065J.json"
    before = path.read_bytes()
    restored = JsonTransistorLoader().load_from_json(path)
    assert restored.metadata.name == "CREE_C3M0060065J"
    assert restored.switch.channel_data
    assert sha256(path.read_bytes()).digest() == sha256(before).digest()


def test_context_inventory_preserves_valid_groups_beside_invalid_other_component():
    """One damaged diode observation must not hide usable switch selectors."""
    transistor = review_device()
    transistor.diode.channel_data = [ChannelCharacteristics(float("nan"), np.array([[0., 1.], [0., 10.]]), None)]
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[plecs_repository] = lambda: SimpleNamespace(get_by_name=lambda _: transistor)
    response = TestClient(app).get("/api/plecs/core/review/contexts")
    assert response.status_code == 200, response.text
    assert response.json()["components"]["switch"]["conduction"][0]["gate_voltage_v"] == 12
    assert response.json()["components"]["diode"]["conduction"] == []
    assert response.json()["diagnostics"]


@pytest.mark.parametrize("operation", ["upload", "create"])
def test_native_archive_existing_http_import_persist_reload_and_export(tmp_path, monkeypatch, operation):
    """A downloaded declaration archive completes the existing real import flow."""
    from transistordatabase.gui_web.backend import main

    doc = parse_plecs_document(SAMPLE.read_bytes())
    transistor = finalize_plecs_package(doc, **mapping(doc))
    archive = core_to_json_document(transistor)
    repository = JsonTransistorRepository(tmp_path / "isolated-native-review-repository")
    monkeypatch.setattr(main, "_repo", repository)
    transport = TestClient(main.app)
    payload = json.dumps(archive).encode("utf-8")
    response = (transport.post("/api/transistors/upload", files={"file": ("declared-core.json", payload, "application/json")})
                if operation == "upload" else transport.post("/api/transistors", json=archive))
    assert response.status_code == 200, response.text
    assert repository.list_all() == [transistor.metadata.name]
    restored = repository.get_by_name(transistor.metadata.name)
    assert core_to_json_document(restored) == archive
    assert restored.metadata.housing_type == ""
    exported = transport.post(f"/api/transistors/{transistor.metadata.name}/export/json")
    assert exported.status_code == 200, exported.text
    assert exported.json() == archive
    file = tmp_path / "final-downloaded-core.json"
    file.write_bytes(exported.content)
    reloaded = JsonTransistorLoader().load_from_json(file)
    assert parse_plecs_document(render_core_plecs(reloaded)).packages == doc.packages


@pytest.mark.parametrize("operation", ["upload", "create"])
def test_unsupported_native_version_cannot_create_unknown_placeholder(tmp_path, monkeypatch, operation):
    """A rejected archive leaves the isolated database empty."""
    from transistordatabase.gui_web.backend import main

    repository = JsonTransistorRepository(tmp_path / "isolated-rejected-review-repository")
    monkeypatch.setattr(main, "_repo", repository)
    transport = TestClient(main.app)
    archive = core_to_json_document(review_device()) | {"version": 999}
    response = (transport.post("/api/transistors/upload", files={"file": ("unsupported.json", json.dumps(archive).encode(), "application/json")})
                if operation == "upload" else transport.post("/api/transistors", json=archive))
    assert response.status_code == 400, response.text
    assert repository.list_all() == []


def test_existing_legacy_http_upload_preserves_full_curves_and_source_hash(tmp_path, monkeypatch):
    """An immutable legacy fixture also imports through the real upload adapter."""
    from transistordatabase.gui_web.backend import main

    source = SAMPLE.parent / "CREE_C3M0060065J.json"
    original = source.read_bytes()
    expected = JsonTransistorLoader().load_from_json(source)
    repository = JsonTransistorRepository(tmp_path / "isolated-legacy-review-repository")
    monkeypatch.setattr(main, "_repo", repository)
    response = TestClient(main.app).post("/api/transistors/upload", files={
        "file": (source.name, original, "application/json")})
    assert response.status_code == 200, response.text
    restored = repository.get_by_name(expected.metadata.name)
    assert core_to_json_document(restored) == core_to_json_document(expected)
    assert sha256(source.read_bytes()).digest() == sha256(original).digest()


@pytest.mark.parametrize("name", ["../outside", "..\\outside", "C:outside"])
def test_native_upload_storage_identity_rejected_before_any_file_write(tmp_path, monkeypatch, name):
    """Reject path syntax without rewriting identity or attempting an outside write."""
    from transistordatabase.gui_web.backend import main

    repository = JsonTransistorRepository(tmp_path / "isolated-boundary-review-repository")
    monkeypatch.setattr(main, "_repo", repository)
    reached = []

    def prohibited_write(self, transistor, path):
        reached.append(path)
        raise AssertionError("Unsafe storage identity reached the writer")

    monkeypatch.setattr(JsonTransistorLoader, "save_to_json", prohibited_write)
    transistor = review_device(name)
    archive = core_to_json_document(transistor)
    response = TestClient(main.app).post("/api/transistors/upload", files={
        "file": ("unsafe-identity.json", json.dumps(archive).encode(), "application/json")})
    assert response.status_code in (400, 422), response.text
    assert reached == []
    assert repository.list_all() == []
    assert transistor.metadata.name == name


@pytest.mark.parametrize("encoding,bom", [("utf-8", b"\xef\xbb\xbf"), ("iso-8859-1", b""), ("windows-1252", b"")])
def test_http_original_bytes_archived_exactly(encoding, bom):
    """Hash and download the actual encoded file, including its byte order mark."""
    text = SAMPLE.read_bytes().decode("iso-8859-1")
    text = text.replace('encoding="ISO-8859-1"', f'encoding="{encoding}"').replace("TestVendor", "Véndor")
    raw = bom + text.encode(encoding)
    transport = {"xml_base64": base64.b64encode(raw).decode(), "source_name": "independent.xml"}
    response = client().post("/api/plecs/preview", json=transport)
    assert response.status_code == 200, response.text
    assert response.json()["source"]["sha256"] == sha256(raw).hexdigest()
    assert response.json()["packages"][0]["vendor"] == "Véndor"
    exported = client().post("/api/plecs/export", json=transport | {"mode": "source-preserved"})
    assert exported.content == raw


@pytest.mark.parametrize("path", [
    SC / "db/plecs/Infineon_IGBT/IGW60T120_IGBT.xml",
    SC / "db/plecs/Infineon_IGBT/IKW20N60TA_Diode.xml",
    ENCO / "packages/enco-components/tests/mosfet/_mock/G3R40MT12K.xml",
])
def test_actual_sources_exact_archive_and_table_or_unsupported_contract(path):
    """Read supplied SC and ENCO evidence without running or modifying it."""
    assert path.is_file(), f"Required read-only source fixture unavailable: {path}"
    original = path.read_bytes()
    doc = parse_plecs_document(original, path.name)
    assert doc.packages[0].ratings is None
    assert render_plecs_document(doc, mode="source-preserved") == original
    if path.name == "G3R40MT12K.xml":
        assert len(doc.packages[0].blocks) == 1
        tables = doc.packages[0].blocks[0].tables
        assert sum(t.kind == "conduction" for t in tables) == 2
        assert all(not t.slices for t in tables if t.formula)
        assert any(t.network_type == "Cauer" for t in doc.packages[0].blocks[0].thermal)
        assert any(t.network_type == "StateSpaceModel" for t in doc.packages[0].thermal)
        assert not doc.can_export
        with pytest.raises(ValueError):
            render_plecs_document(doc)
    else:
        assert parse_plecs_document(render_plecs_document(doc)).packages == doc.packages
        if "Diode" in path.name:
            assert any(s.supply_voltage_v < 0 for t in doc.packages[0].blocks[0].tables for s in t.slices if s.supply_voltage_v is not None)
    assert sha256(path.read_bytes()).digest() == sha256(original).digest()


@pytest.mark.parametrize("defect", ["foreign", "mixed", "duplicate", "nan", "shape", "dtd"])
def test_invalid_xml_never_returns_plausible_preview(defect):
    """Reject ambiguity, malformed values and unsupported namespace/entity input."""
    text = SAMPLE.read_text()
    if defect == "foreign":
        text = text.replace(NAMESPACE, "urn:foreign")
    elif defect == "mixed":
        text = text.replace("<CurrentAxis>", '<CurrentAxis xmlns="urn:foreign">', 1)
    elif defect == "duplicate":
        text = text.replace("0 10 20 30 40 50", "0 10 10 30 40 50", 1)
    elif defect == "nan":
        text = text.replace("0.00 0.10 0.30", "0.00 NaN 0.30", 1)
    elif defect == "shape":
        text = text.replace("0.00 0.10 0.30", "0.00 0.30", 1)
    else:
        text = text.replace("<SemiconductorLibrary", '<!DOCTYPE SemiconductorLibrary [<!ENTITY e "x">]><SemiconductorLibrary', 1)
    response = client().post("/api/plecs/preview", json={"xml_text": text})
    assert response.status_code == 422


def test_derived_plot_budget_preflight_and_scale_overflow():
    """Bound expanded output before constructing plots or overflowing SI values."""
    root = ET.fromstring(SAMPLE.read_bytes())
    block = root.find(f"{{{NAMESPACE}}}Package/{{{NAMESPACE}}}SemiconductorData")
    # Each small source table advertises a >100000-cell Cartesian product.
    for node in block.iter(f"{{{NAMESPACE}}}CurrentAxis"):
        node.text = " ".join(str(i) for i in range(10001))
    with pytest.raises(ValueError, match="numeric/plot"):
        parse_plecs_document(ET.tostring(root))
    with pytest.raises(ValueError, match="overflow"):
        parse_plecs_document(SAMPLE.read_text().replace('scale="0.001"', 'scale="1e308"'))


def test_generic_export_preserves_exact_unknown_and_mixed_context_rejects():
    """An unknown selector never borrows a known record to fill a rectangle."""
    doc = parse_plecs_document(SAMPLE.read_bytes())
    transistor = finalize_plecs_package(doc, **mapping(doc))
    transistor.metadata.comment = "Independent ordinary core record"
    selected = dict(component="switch", conduction={"gate_voltage_v": 15.},
        turn_on={"gate_voltage_v": 15., "gate_resistance_ohm": None},
        turn_off={"gate_voltage_v": 15., "gate_resistance_ohm": None},
        include_thermal=True, omission_reasons={}, recovery_voltage_convention="preserve-signed")
    exported, report = core_selection_document(transistor, selected)
    assert report["evidence"][1]["curve_count"] == 4
    assert exported.packages[0].fixed_conditions["selection"] == selected
    transistor.switch.e_on_data[0].r_g = 0
    with pytest.raises(ValueError, match="incomplete"):
        core_selection_document(transistor, selected)
