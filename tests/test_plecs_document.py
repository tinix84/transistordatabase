"""Independent corners and real read-only source format fixtures for PLECS."""
import base64
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase.plecs_document import (
    NAMESPACE, finalize_plecs_package, parse_plecs_document,
    render_core_plecs, render_plecs_document,
)
from transistordatabase.gui_web.backend.plecs_routes import router

SAMPLE = Path(__file__).parent / "test_data/sample_plecs_switch.xml"
SC = Path("/home/tinix/claude_wsl/archive_ntbees2/sc")
ENCO = Path("/mnt/d/OneDrive/claude/bhy/enco/enco")


def declarations(document):
    """Provide explicit source-independent fixture ratings and gate conditions."""
    return dict(package_id="0", ratings=dict(v_abs_max=1200., i_abs_max=75.,
                i_cont=40., t_j_max=175.),
                thermal_properties=dict(housing_area=0.0001, cooling_area=0.0002),
                block_targets={"0": "switch"},
                gate_voltage_by_table={t.source_path: 15. for t in document.packages[0].blocks[0].tables},
                recovery_voltage_convention="preserve-signed")


def test_exact_numeric_corners_scale_and_unknowns():
    """Exact numeric corners scale and unknowns."""
    doc = parse_plecs_document(SAMPLE.read_bytes(), SAMPLE.name)
    assert doc.can_export and doc.status == "supported-tables"
    assert doc.packages[0].ratings is None
    table = doc.packages[0].blocks[0].tables[0]
    assert table.gate_voltage_v is None and table.gate_resistance_ohm is None
    assert table.scale == .001 and table.values[1][1][-1] == 2.70
    assert table.slices[-1].plot_data[-1] == [50., .0027]
    assert doc.source.sha256 == sha256(SAMPLE.read_bytes()).hexdigest()


@pytest.mark.parametrize("namespace", ["default", "prefix", "none"])
def test_namespace_semantic_roundtrip(namespace):
    """Namespace semantic roundtrip."""
    text = SAMPLE.read_text()
    if namespace == "none":
        text = text.replace(f' xmlns="{NAMESPACE}"', "")
    elif namespace == "prefix":
        import re
        text = text.replace(f'xmlns="{NAMESPACE}"', f'xmlns:p="{NAMESPACE}"')
        text = re.sub(r"<(\/?)([A-Za-z][\w]*)", r"<\1p:\2", text)
    before = parse_plecs_document(text)
    after = parse_plecs_document(render_plecs_document(before))
    assert after.packages == before.packages


def test_source_bytes_archive_exact_including_encoding_bom_and_nonascii():
    """Source bytes archive exact including encoding bom and nonascii."""
    raw = SAMPLE.read_bytes().replace(b"TestVendor", b"V\xe9ndor")
    doc = parse_plecs_document(raw)
    assert doc.packages[0].vendor == "Véndor"
    assert doc.source.hash_scope == "original-bytes"
    assert render_plecs_document(doc, mode="source-preserved") == raw
    text_doc = parse_plecs_document(raw.decode("iso-8859-1"))
    assert text_doc.packages[0].vendor == "Véndor"
    assert parse_plecs_document(render_plecs_document(text_doc)).packages[0].vendor == "Véndor"
    with pytest.raises(ValueError, match="declaration"):
        render_plecs_document(text_doc, mode="source-preserved")


def test_finalize_core_roundtrip_keeps_declarations_distinct():
    """Finalize core roundtrip keeps declarations distinct."""
    doc = parse_plecs_document(SAMPLE.read_bytes())
    t = finalize_plecs_package(doc, **declarations(doc))
    assert t.electrical_ratings.v_abs_max == 1200
    assert len(t.switch.e_on_data) == 4
    assert t.switch.e_on_data[0].v_g == 15
    assert t.switch.e_on_data[0].r_g is None
    np.testing.assert_array_equal(t.switch.e_on_data[0].graph_i_e[1], [0,.0001,.0003,.0006,.001,.0015])
    after = parse_plecs_document(render_core_plecs(t))
    assert after.packages == doc.packages
    assert after.packages[0].ratings is None
    assert after.packages[0].blocks[0].tables[0].gate_voltage_v is None
    assert t.switch.thermal_foster.r_th_total == .5
    assert "declarations" in t.metadata.comment
    t.switch.e_on_data[0].graph_i_e[1][1] *= 2
    with pytest.raises(ValueError, match="changed"):
        render_core_plecs(t)


@pytest.mark.parametrize("change,message", [
    (lambda s:s.replace("0 10 20 30 40 50", "0 10 10 30 40 50"), "unique"),
    (lambda s:s.replace("0.00 0.10 0.30", "0.00 0.10"), "row length"),
    (lambda s:s.replace("0.00 0.10 0.30", "0.00 NaN 0.30"), "nonfinite"),
    (lambda s:s.replace("<TemperatureAxis>25 150</TemperatureAxis>", ""), "missing"),
    (lambda s:s.replace('scale="0.001"', 'scale="-1"'), "positive"),
])
def test_malformed_data_fails_instead_of_skipping(change, message):
    """Malformed data fails instead of skipping."""
    with pytest.raises(ValueError, match=message):
        parse_plecs_document(change(SAMPLE.read_text()))


def test_duplicate_overlapping_conditions_rejected_but_distinct_rg_retained():
    """Duplicate overlapping conditions rejected but distinct rg retained."""
    from xml.etree import ElementTree as ET
    root = ET.fromstring(SAMPLE.read_bytes())
    block = root.find(f"{{{NAMESPACE}}}Package/{{{NAMESPACE}}}SemiconductorData")
    original = block[0]
    original.set("Rg", "2")
    duplicate = deepcopy(original)
    duplicate.set("Rg", "5")
    block.append(duplicate)
    assert len(parse_plecs_document(ET.tostring(root)).packages[0].blocks[0].tables) == 4
    duplicate.set("Rg", "2")
    with pytest.raises(ValueError, match="duplicate table conditions"):
        parse_plecs_document(ET.tostring(root))


def test_formula_preview_preserves_but_does_not_plot_or_evaluate():
    """Formula preview preserves but does not plot or evaluate."""
    source = SAMPLE.read_text().replace("Table only", "Table and formula", 1).replace(
        "</ComputationMethod>", "</ComputationMethod><Formula>arbitrary(i)</Formula>", 1)
    doc = parse_plecs_document(source)
    assert not doc.can_export
    table = doc.packages[0].blocks[0].tables[0]
    assert table.formula == "arbitrary(i)" and not table.slices
    assert doc.packages[0].blocks[0].tables[1].slices  # independent supported evidence
    with pytest.raises(ValueError, match="Formula"):
        render_plecs_document(doc)
    assert render_plecs_document(doc, mode="source-preserved") == source.encode()


@pytest.mark.parametrize("source", [b'<!DOCTYPE SemiconductorLibrary><SemiconductorLibrary/>',
                                    '<!DOCTYPE SemiconductorLibrary><SemiconductorLibrary/>'.encode("utf-16")])
def test_no_entity_expansion_for_any_accepted_encoding(source):
    """No entity expansion for any accepted encoding."""
    with pytest.raises(ValueError, match="DTD|encoding"):
        parse_plecs_document(source)


@pytest.mark.parametrize("fixture, expected", [
    (SC / "db/plecs/Infineon_IGBT/IGW60T120_IGBT.xml", "supported-tables"),
    (SC / "db/plecs/Infineon_IGBT/IKW20N60TA_Diode.xml", "supported-tables"),
    (ENCO / "packages/enco-components/tests/mosfet/_mock/G3R40MT12K.xml", "unsupported-content"),
])
def test_real_reference_fixtures_readonly(fixture, expected):
    """Real reference fixtures readonly."""
    if not fixture.exists():
        pytest.skip("Read-only reference fixture not available")
    before = fixture.read_bytes()
    doc = parse_plecs_document(before, str(fixture))
    assert doc.status == expected
    assert render_plecs_document(doc, mode="source-preserved") == before
    if expected == "supported-tables":
        assert parse_plecs_document(render_plecs_document(doc)).packages == doc.packages
    else:
        assert len(doc.packages[0].blocks) == 1
        assert len(doc.packages[0].blocks[0].tables) == 4
        assert any(t.network_type == "Cauer" for b in doc.packages[0].blocks for t in b.thermal)
        assert {d.code for d in doc.diagnostics} >= {"unsupported-formula", "unsupported-thermal"}
    assert sha256(fixture.read_bytes()).digest() == sha256(before).digest()


def test_multiple_packages_and_blocks_preserve_independent_identity():
    """Multiple packages and blocks preserve independent identity."""
    from xml.etree import ElementTree as ET
    root = ET.fromstring(SAMPLE.read_bytes())
    second_package = deepcopy(root[0])
    second_package.set("partnumber", "SEPARATE_DEVICE")
    root.append(second_package)
    second_block = deepcopy(root[0].find(f"{{{NAMESPACE}}}SemiconductorData"))
    second_block.set("type", "BodyDiode")
    root[0].append(second_block)
    before = parse_plecs_document(ET.tostring(root))
    assert len(before.packages) == 2 and len(before.packages[0].blocks) == 2
    assert before.packages[1].partnumber == "SEPARATE_DEVICE"
    assert parse_plecs_document(render_plecs_document(before)).packages == before.packages


def test_signed_diode_recovery_core_mapping_reversible():
    """Signed diode recovery core mapping reversible."""
    fixture = SC / "db/plecs/Infineon_IGBT/IKW20N60TA_Diode.xml"
    if not fixture.exists():
        pytest.skip("SC reference unavailable")
    doc = parse_plecs_document(fixture.read_bytes())
    args = declarations(doc)
    args["block_targets"] = {"0": "diode"}
    args["recovery_voltage_convention"] = "negative-to-positive-bus"
    t = finalize_plecs_package(doc, **args)
    assert len(t.diode.e_rr_data) == 4
    assert t.diode.e_rr_data[0].v_supply == 400
    assert not t.switch.e_on_data
    assert parse_plecs_document(render_core_plecs(t)).packages == doc.packages


def test_http_preview_archive_errors_and_finalize():
    """Http preview archive errors and finalize."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    source = dict(xml_base64=base64.b64encode(SAMPLE.read_bytes()).decode(), source_name=SAMPLE.name)
    response = client.post("/api/plecs/preview", json=source)
    assert response.status_code == 200, response.text
    assert response.json()["source"]["sha256"] == sha256(SAMPLE.read_bytes()).hexdigest()
    response = client.post("/api/plecs/export", json={**source, "mode":"source-preserved"})
    assert response.content == SAMPLE.read_bytes()
    assert client.post("/api/plecs/preview", json={**source,"xml_text":"bad"}).status_code == 422
    assert client.post("/api/plecs/preview", json={"xml_base64":"!!"}).status_code == 422
    doc = parse_plecs_document(SAMPLE.read_bytes())
    response = client.post("/api/plecs/finalize", json={**source, **declarations(doc)})
    assert response.status_code == 200, response.text
    assert response.json()["core"]["switch"]["e_on_data"][0]["v_g"] == 15


def test_export_service_no_cwd_mutation_and_no_overwrite(tmp_path, monkeypatch):
    """Export service no cwd mutation and no overwrite."""
    from transistordatabase.backend.concrete_services import ExportService
    document = parse_plecs_document(SAMPLE.read_bytes())
    transistor = finalize_plecs_package(document, **declarations(document))
    original = os.getcwd()
    monkeypatch.setattr(os, "chdir", lambda *a: pytest.fail("process cwd mutation"))
    ExportService().export_to_plecs(transistor, tmp_path)
    assert os.getcwd() == original
    assert list(tmp_path.glob("*.xml"))
    with pytest.raises(FileExistsError):
        ExportService().export_to_plecs(transistor, tmp_path)
