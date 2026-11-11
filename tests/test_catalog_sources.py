"""Independent uploaded-source semantics and exact historical-file evidence."""

import base64
import hashlib
import io
from pathlib import Path
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.catalog_sources import (
    build_snapshot, compare_snapshots, decode_source, preview_source,
    read_source, static_links, workbook_coefficients,
)
from transistordatabase.gui_web.backend.catalog_source_routes import router


def b64(data):
    """Return the exact upload representation."""
    return base64.b64encode(data).decode()


def xlsx(sheet_xml, rels=""):
    """Create minimal OOXML bytes, with explicit saved caches and no Excel engine."""
    contents = {
        "[Content_Types].xml": '''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>''',
        "_rels/.rels": '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>''',
        "xl/workbook.xml": '''<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="MOSFET" sheetId="1" r:id="rId1"/></sheets></workbook>''',
        "xl/_rels/workbook.xml.rels": '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>''',
        "xl/worksheets/sheet1.xml": sheet_xml,
    }
    if rels:
        contents["xl/worksheets/_rels/sheet1.xml.rels"] = rels
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    return stream.getvalue()


@pytest.fixture
def workbook():
    """Two duplicate headers, hidden row, cached formulas/errors and static links."""
    return xlsx('''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><dimension ref="A1:AA3"/><sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>Part-No.</t></is></c><c r="B1" t="inlineStr"><is><t>Rating</t></is></c><c r="C1" t="inlineStr"><is><t>Rating</t></is></c><c r="Y1" t="inlineStr"><is><t>EON (TJ=175°C)</t></is></c><c r="Z1" t="inlineStr"><is><t>EOFF (TJ=175°C)</t></is></c><c r="AA1" t="inlineStr"><is><t>ERR (TJ=175°C)</t></is></c></row>
    <row r="2" hidden="1"><c r="A2" t="inlineStr"><is><t>same</t></is></c><c r="B2"><f>1+1</f><v>2</v></c><c r="C2" t="e"><f>1/0</f><v>#DIV/0!</v></c><c r="D2" t="str"><f>HYPERLINK("../datasheet.pdf","quoted ""label""")</f><v>label</v></c><c r="E2" t="str"><f>GetURL(A2)</f><v>cached-not-a-link</v></c><c r="F2" t="inlineStr"><is><t>https://example.test/data</t></is></c><c r="X2"><v>600</v></c><c r="Y2"><f>0.000632/36</f><v>0.00001755555555555556</v></c><c r="Z2"><f>0.003/80</f><v>0.9</v></c><c r="AA2"><f>0.1/20</f></c></row>
    <row r="3"><c r="A3" t="inlineStr"><is><t>same</t></is></c><c r="B3"><v>0</v></c></row></sheetData><hyperlinks><hyperlink ref="A2" r:id="rId1" tooltip="source"/><hyperlink ref="B2" location="MOSFET!A1" display="top"/><hyperlink ref="C2" r:id="missing"/></hyperlinks></worksheet>''',
    '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="../original.pdf" TargetMode="External"/></Relationships>''')


def test_quoted_csv_bom_decimal_and_distinct_duplicate_parts():
    """Embedded separators/newlines/quotes and BOM survive explicit mapping."""
    data = '\ufeffPart;V;R\r\n"P,one";"600";"2,5"\r\n"multi\n""line""";0;bad\r\n"P,one";;3\r\n'.encode()
    encoded = b64(data)
    preview = preview_source("survey.csv", encoded, delimiter=";", decimal=",")
    assert preview["sheets"][0]["row_count"] == 4
    assert preview["sheets"][0]["headers"][0]["label"] == "Part"
    snapshot = build_snapshot("survey.csv", encoded, sheet="CSV", delimiter=";", decimal=",",
        mapping={"part_number": {"column": "1", "unit": "text"},
                 "voltage_v": {"column": "2", "unit": "V"},
                 "rds_ohm": {"column": "3", "unit": "mOhm"}}, metadata={})
    assert len(snapshot["records"]) == 3
    assert len({r["id"] for r in snapshot["records"]}) == 3
    assert snapshot["records"][0]["rds_ohm"] == .0025
    assert snapshot["records"][1]["part_number"] == 'multi\n"line"'
    assert snapshot["records"][1]["voltage_v"] == 0
    assert snapshot["records"][1]["rds_ohm"] is None
    assert snapshot["records"][2]["voltage_v"] is None
    assert base64.b64decode(snapshot["source_file"]["content_base64"]) == data
    assert snapshot["snapshot_id"] == build_snapshot("survey.csv", encoded, sheet="CSV", delimiter=";", decimal=",", mapping=snapshot["mapping"]["mapping"], metadata={})["snapshot_id"]


def test_duplicate_headers_hidden_rows_formula_cache_and_source_errors(workbook):
    """No ambiguous header collapse or cached-error substitution occurs."""
    source = preview_source("test.xlsm", b64(workbook), preview_rows=2)
    sheet = source["sheets"][0]
    assert [c["column"] for c in sheet["headers"] if c["label"] == "Rating"] == ["B", "C"]
    assert all(c["duplicate"] for c in sheet["headers"] if c["label"] == "Rating")
    assert sheet["hidden_rows"] == [2] and sheet["row_count"] == 3
    assert sheet["rows"][1]["hidden"]
    assert sheet["rows"][1]["cells"][1]["formula"] == "=1+1"
    assert sheet["rows"][1]["cells"][1]["cached_value"] == 2
    assert sheet["rows"][1]["cells"][2]["error"] == "#DIV/0!"
    snapshot = build_snapshot("test.xlsm", b64(workbook), sheet="MOSFET", mapping={
        "part_number": {"column": "A", "unit": "text"},
        "voltage_v": {"column": "C", "unit": "V"}}, metadata={})
    assert len(snapshot["records"]) == 2
    assert snapshot["records"][0]["voltage_v"] is None
    assert snapshot["row_errors"][0]["errors"][0]["reason"] == "cached-formula-error"


def test_static_hyperlinks_distinguish_relationships_formulas_text_and_absence(workbook):
    """Relative URLs remain relative to the original workbook, never the server."""
    result = static_links("test.xlsx", b64(workbook))
    assert result["native_link_count"] == 3
    assert result["geturl_formula_count"] == 1
    assert result["geturl_formula_locations"][0]["status"] == "not-evaluated"
    by_cell = {e["cell"]: e for e in result["entries"]}
    assert by_cell["A2"]["target"] == "../original.pdf"
    assert by_cell["A2"]["resolution"] == "relative-to-original-workbook-no-server-resolution"
    assert by_cell["B2"]["resolution"] == "internal-workbook-location"
    assert by_cell["C2"]["resolution"] == "missing-target"
    assert by_cell["D2"]["display"] == 'quoted "label"'
    assert by_cell["F2"]["kind"] == "plain-URL-text"
    assert "E2" not in by_cell
    batch = static_links("test.xlsx", b64(workbook), cells=[{"sheet": "MOSFET", "cell": "E2"}, {"sheet": "MOSFET", "cell": "A2"}])
    assert batch["entries"][0]["links"] == []
    assert len(batch["entries"][1]["links"]) == 1


def test_typed_coefficients_literal_division_is_not_an_event_energy(workbook):
    """Independent J/A arithmetic preserves missing caches and disagreements."""
    result = workbook_coefficients("test.xlsm", b64(workbook))
    row = {c["component"]: c for c in result["coefficients"] if c["row"] == 2}
    assert row["e_on"]["unit"] == "J/A"
    assert row["e_on"]["recognized_literal_coefficient_j_per_a"] == pytest.approx(.000632 / 36)
    assert row["e_on"]["reference_current_a"] == 36
    assert row["e_on"]["literal_energy_j"] == .000632
    assert row["e_on"]["reference_voltage_cell"]["value"] == 600
    assert row["e_on"]["reference_temperature_c"] == 175
    assert row["e_off"]["status"] == "cache-formula-disagreement"
    assert row["e_rr"]["status"] == "literal-division-cache-missing"
    assert row["e_rr"]["cached_coefficient_j_per_a"] is None


def test_version_comparison_keys_are_explicit_and_duplicate_parts_cannot_collapse():
    """Only a changed source value changes an aligned row, despite new source IDs."""
    before = {"records": [{"id": "old:2", "row": 2, "part_number": "A", "voltage_v": 600}, {"id": "old:3", "row": 3, "part_number": "B"}]}
    after = {"records": [{"id": "new:2", "row": 2, "part_number": "A", "voltage_v": 650}, {"id": "new:4", "row": 4, "part_number": "C"}]}
    result = compare_snapshots(before, after, key_fields=["part_number"])
    assert result["added"] == ["new:4"] and result["missing"] == ["old:3"]
    assert result["changed"] == [{"key": ["A"], "before_id": "old:2", "after_id": "new:2", "fields": ["voltage_v"]}]
    duplicates = {"records": [{"id": "x:2", "row": 2, "part_number": "A"}, {"id": "x:3", "row": 3, "part_number": "A"}]}
    with pytest.raises(ValueError, match="unique"):
        compare_snapshots(duplicates, duplicates, key_fields=["part_number"])
    assert compare_snapshots(duplicates, duplicates, key_fields=["part_number", "row"])["changed"] == []


@pytest.mark.parametrize("filename,payload", [("bad.xls", b"anything"), ("x.csv", b'"unterminated'), ("x.xlsx", b"not a zip")])
def test_invalid_source_formats_fail_explicitly(filename, payload):
    """Unsupported input cannot masquerade as a valid empty survey."""
    with pytest.raises(ValueError):
        read_source(filename, b64(payload))


def test_resource_and_xml_declaration_guards():
    """Reject expanded workbooks and XML entity declarations before parsing cells."""
    with pytest.raises(ValueError, match="base64"):
        decode_source("x.csv", "!!")
    with pytest.raises(ValueError, match="8 MB"):
        decode_source("x.csv", "A" * 10_666_673)
    data = xlsx('<!DOCTYPE worksheet [<!ENTITY x "bad">]><worksheet/>')
    with pytest.raises(ValueError):
        read_source("x.xlsx", b64(data))


def test_declared_unit_prefixes_preserve_condition_tails_and_reject_wrong_decimal():
    """Recognized physical units do not consume current/gate test conditions."""
    request = {"filename": "x.csv", "content_base64": b64(b'part;voltage;resistance\np;1.2kV;3.4Ohm @ 500mA, 10V\nq;600V;41mOhm bei 32A, 8V\nr;"1,2kV";unknown\n'),
               "sheet": "CSV", "delimiter": ";", "mapping": {
                   "part_number": {"column": "1", "unit": "text"},
                   "voltage_v": {"column": "2", "unit": "V"},
                   "rds_ohm": {"column": "3", "unit": "Ohm"}}, "metadata": {}}
    result = build_snapshot(**request)
    assert result["records"][0]["voltage_v"] == 1200
    assert result["records"][0]["rds_ohm"] == 3.4
    assert result["records"][1]["rds_ohm"] == .041
    assert result["records"][1]["raw"]["3"]["value"] == "41mOhm bei 32A, 8V"
    assert result["records"][2]["voltage_v"] is None


def test_misleading_ooxml_dimension_cannot_drop_stored_cells():
    """A false small dimension is a validation error rather than silent data loss."""
    data = xlsx('''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="2"><c r="A2"><v>10</v></c></row></sheetData></worksheet>''')
    with pytest.raises(ValueError, match="dimension excludes"):
        read_source("x.xlsx", b64(data))


def test_http_equality_and_rejected_units(workbook):
    """Typed transport preserves exact headless results and clear errors."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = {"filename": "test.xlsm", "content_base64": b64(workbook)}
    response = client.post("/api/catalog-sources/coefficients", json=request)
    assert response.status_code == 200
    assert response.json() == workbook_coefficients(**request)
    assert client.post("/api/catalog-sources/preview", json={**request, "preview_rows": 201}).status_code == 422
    snapshot = {**request, "sheet": "MOSFET", "metadata": {}, "mapping": {
        "part_number": {"column": "A", "unit": "text"}, "rds_ohm": {"column": "B", "unit": "V"}}}
    assert client.post("/api/catalog-sources/snapshot", json=snapshot).status_code == 422
    snapshot["mapping"]["rds_ohm"]["unit"] = "Ohm"
    snapshot["metadata"] = {"quote_quantity": 1000, "conditions": {"tj_c": 25}}
    response = client.post("/api/catalog-sources/snapshot", json=snapshot)
    assert response.status_code == 200
    assert response.json() == build_snapshot(**snapshot)
    before = {"records": response.json()["records"]}
    assert client.post("/api/catalog-sources/diff", json={"before": before, "after": before, "key_fields": ["row"]}).json() == compare_snapshots(before, before, key_fields=["row"])


def test_mapping_snapshot_does_not_alias_callers_mutable_input():
    """Later input edits cannot silently alter an already-created snapshot."""
    mapping = {"part_number": {"column": "1", "unit": "text"}}
    metadata = {"conditions": {"tj_c": 25}}
    result = build_snapshot("x.csv", b64(b"part\nP\n"), sheet="CSV", mapping=mapping, metadata=metadata)
    mapping["part_number"]["column"] = "2"
    metadata["conditions"]["tj_c"] = 150
    assert result["mapping"]["mapping"]["part_number"]["column"] == "1"
    assert result["records"][0]["conditions"]["tj_c"] == 25


ARCHIVE = Path("/home/tinix/claude_wsl/archive_ntbees2/sc/db")
FILES = {
    "201210_sic_digikey.xlsx": "aea21eb6f1f44267f8c454393be63d052f5f4d0b9b6cc6183151edd4df7c2581",
    "SiC_MOSFET_FOM.xlsx": "1470636e91595cf4f0a48a8a30fe50e3558710cde3d4b7bf5abac59bd99bc641",
    "201210_mos_digikey.csv": "a56d5f7319caa70f1b2b43954c60111a29359f800810e9db16f6e745c7f5ad4f",
    "201210_gan_digikey.csv": "7d97978c7015b45f4fdfe4f1dbfb1bc75ea44547b4fbb5321231f6221cece28f",
    "SemiconductorsData.xlsm": "f7b4c0424f1329c469dbffd96b4ab44fe6ac448ccb7ff0e6f331159c8a0b3ab0",
}


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Read-only historical source directory unavailable")
@pytest.mark.parametrize("filename,digest", FILES.items())
def test_exact_archive_source_census_and_hash_unchanged(filename, digest):
    """Every named supplied source remains byte-identical after inspection."""
    path = ARCHIVE / filename
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == digest
    result = preview_source(filename, b64(original), encoding="cp1252" if filename.endswith(".csv") else "utf-8-sig")
    assert result["source_file"]["sha256"] == digest
    if filename == "SemiconductorsData.xlsm":
        links = static_links(filename, b64(original))
        assert links["native_link_count"] == 59
        assert links["geturl_formula_count"] == 63
        assert sum(e["kind"] == "stored-link" and e["sheet"] == "MOSFET" for e in links["entries"]) == 23
        assert sum(e["kind"] == "stored-link" and e["sheet"] == "IGBT" for e in links["entries"]) == 36
        coeff = workbook_coefficients(filename, b64(original))
        y2 = next(c for c in coeff["coefficients"] if c["sheet"] == "MOSFET" and c["cell"]["cell"] == "Y2")
        assert y2["recognized_literal_coefficient_j_per_a"] == pytest.approx(.000632 / 36)
    if filename == "SiC_MOSFET_FOM.xlsx":
        assert len(next(s for s in result["sheets"] if s["name"] == "600V")["hidden_rows"]) == 1077
        assert sum(s["formula_count"] for s in result["sheets"]) == 3631
    assert path.read_bytes() == original


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Read-only historical source directory unavailable")
@pytest.mark.parametrize("filename,sheet,column,count", [
    ("201210_sic_digikey.xlsx", "in", "D", 286),
    ("SiC_MOSFET_FOM.xlsx", "1200V", "D", 174),
    ("SiC_MOSFET_FOM.xlsx", "600V", "D", 1137),
    ("SiC_MOSFET_FOM.xlsx", "Tabelle8", "D", 233),
    ("201210_mos_digikey.csv", "CSV", "4", 500),
    ("201210_gan_digikey.csv", "CSV", "4", 99),
    ("SemiconductorsData.xlsm", "MOSFET", "A", 23),
    ("SemiconductorsData.xlsm", "IGBT", "A", 38),
])
def test_explicit_mapping_retains_all_known_catalog_rows(filename, sheet, column, count):
    """The new general mapper reconciles all eight historical populations."""
    data = (ARCHIVE / filename).read_bytes()
    result = build_snapshot(filename, b64(data), sheet=sheet,
        mapping={"part_number": {"column": column, "unit": "text"}}, metadata={},
        encoding="cp1252" if filename.endswith(".csv") else "utf-8-sig")
    assert len(result["records"]) == count
    assert len({r["id"] for r in result["records"]}) == count
    assert (ARCHIVE / filename).read_bytes() == data
