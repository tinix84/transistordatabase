"""Cross-check catalog imports and screening against independent source evidence."""

import base64
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.catalog_fom import load_catalog, metric_evidence, screen_catalog
from transistordatabase.catalog_sources import (
    build_snapshot, compare_snapshots, preview_source, static_links, workbook_coefficients,
)
from transistordatabase.gui_web.backend.catalog_routes import router as screening_router
from transistordatabase.gui_web.backend.catalog_source_routes import router as source_router


ARCHIVE = Path("/home/tinix/claude_wsl/archive_ntbees2/sc/db")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def upload(filename, data):
    """Encode the exact bytes accepted by the transport."""
    return {"filename": filename, "content_base64": base64.b64encode(data).decode()}


@pytest.fixture(scope="module")
def client():
    """Run both API adapters without exposing a network port."""
    app = FastAPI()
    app.include_router(source_router)
    app.include_router(screening_router)
    return TestClient(app, raise_server_exceptions=False)


def package(members):
    """Build an in-memory ZIP from the supplied members."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as book:
        for name, data in members:
            book.writestr(name, data)
    return output.getvalue()


def test_unsafe_xml_encodings_and_broken_packages_reject_before_import(client):
    """Check parser rejection for DTDs in UTF-8 and UTF-16 packages."""
    # Inspect the actual parser result for an unused member of a valid workbook.
    from openpyxl import Workbook

    valid = io.BytesIO()
    Workbook().save(valid)
    with zipfile.ZipFile(valid) as book:
        members = [(name, book.read(name)) for name in book.namelist()]
    entity = '<!DOCTYPE r [<!ENTITY payload "unexpected">]><r>&payload;</r>'
    for encoding in ("utf-8", "utf-16", "utf-16-be"):
        source = upload("unsafe.xlsx", package([*members, ("customXml/part.xml", entity.encode(encoding))]))
        response = client.post("/api/catalog-sources/preview", json=source)
        assert response.status_code == 422, (encoding, response.text)
    for members in ([("document.xml", "<r/>")], [("xl/workbook.xml", "<r/>")]):
        response = client.post("/api/catalog-sources/preview", json=upload("bad.xlsx", package(members)))
        assert response.status_code == 422


def test_zip_budgets_are_checked_before_opening_a_worksheet(monkeypatch):
    """Exercise both ZIP limits with a small synthetic archive."""
    from transistordatabase import catalog_sources

    monkeypatch.setattr(catalog_sources, "MAX_EXPANDED", 128)
    for members in ([("blob.bin", b"a" * 129)], [(str(i), b"") for i in range(2001)]):
        with pytest.raises(ValueError, match="expanded ZIP resource"):
            preview_source(**upload("limits.xlsx", package(members)))


def test_sparse_ooxml_numeric_underflow_retains_token_and_real_zero():
    """Distinguish a stored zero from an unrepresentable lexical number."""
    from openpyxl import Workbook

    book, original = Workbook(), io.BytesIO()
    for row in (["Part", "Voltage"], ["tiny", 111], ["zero", 0]):
        book.active.append(row)
    book.save(original)
    with zipfile.ZipFile(original) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()]
    members = [(name, data.replace(b"<v>111</v>", b"<v>1e-999</v>").replace(b'<dimension ref="A1:B3"/>', b'')
                if name == "xl/worksheets/sheet1.xml" else data) for name, data in members]
    snapshot = build_snapshot(**upload("sparse.xlsx", package(members)), sheet="Sheet", mapping={
        "part_number": {"column": "A", "unit": "text"}, "voltage_v": {"column": "B", "unit": "V"},
    }, metadata={})
    tiny, zero = snapshot["records"]
    assert tiny["voltage_v"] is None and tiny["raw"]["B"]["raw_numeric_token"] == "1e-999"
    assert snapshot["row_errors"][0]["errors"][0]["reason"] == "numeric-underflow"
    assert zero["voltage_v"] == 0 and zero["raw"]["B"]["error"] is None


def test_shared_string_fanout_and_snapshot_comparison_have_output_budgets(monkeypatch):
    """Count every use of a shared string and every serialized record."""
    from openpyxl import Workbook
    from transistordatabase import catalog_sources

    book, original = Workbook(), io.BytesIO()
    for _ in range(10):
        book.active.append([333])
    book.save(original)
    path = "xl/sharedStrings.xml"
    text = "long shared value " * 10
    with zipfile.ZipFile(original) as archive:
        members = []
        for name in archive.namelist():
            data = archive.read(name)
            if name == "xl/worksheets/sheet1.xml":
                data = data.replace(b't="n"><v>333</v>', b't="s"><v>0</v>')
            if name == "[Content_Types].xml":
                override = f'<Override PartName="/{path}" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
                data = data.replace(b"</Types>", override.encode() + b"</Types>")
            members.append((name, data))
    members.append((path, f'<sst xmlns="{NS["s"]}"><si><t>{text}</t></si></sst>'))
    source = upload("shared.xlsx", package(members))
    assert preview_source(**source)["sheets"][0]["rows"][1]["cells"][0]["value"] == text
    # A content type can point to XML with an arbitrary extension.
    alias = "xl/strings.data"
    aliased = []
    for name, data in members:
        if name == path:
            name = alias
            data = ('<!DOCTYPE sst [<!ENTITY test "unexpected">]>' + data.replace(text, "&test;")).encode("utf-16")
        if name == "[Content_Types].xml":
            data = data.replace(path.encode(), alias.encode())
        aliased.append((name, data))
    with pytest.raises(ValueError, match="Invalid workbook XML"):
        preview_source(**upload("disguised.xlsx", package(aliased)))
    monkeypatch.setattr(catalog_sources, "MAX_CELL_TEXT_BYTES", 1000)
    with pytest.raises(ValueError, match="cell text"):
        preview_source(**source)
    monkeypatch.setattr(catalog_sources, "MAX_SNAPSHOT_BYTES", 100)
    with pytest.raises(ValueError, match="50 MB"):
        mapped("part;V;I;R;price;tier;Qg\nP;2;2;2;2;2;2\n")
    with pytest.raises(ValueError, match="50 MB"):
        compare_snapshots({"records": [{"id": "A", "row": 1, "x": text}]}, {"records": []}, key_fields=["row"])


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-Infinity"])
def test_comparison_rejects_nonfinite_nested_json(client, invalid):
    """Reject invalid numeric JSON even inside an unconstrained source cell."""
    body = ('{"before":{"records":[{"id":"first","row":2,"raw":{"value":' + invalid
            + '}}]},"after":{"records":[]},"key_fields":["row"]}')
    response = client.post("/api/catalog-sources/diff", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("metadata", [{"quote_quantity": True}, {"conditions": {"voltage": False}}])
def test_headless_and_http_reject_boolean_quote_metadata(client, metadata):
    """Keep boolean switches separate from recorded numbers."""
    request = {**upload("quoted.csv", b"part\nA\n"), "sheet": "CSV",
               "mapping": {"part_number": {"unit": "text", "column": "1"}}, "metadata": metadata}
    with pytest.raises(ValueError):
        build_snapshot(**request)
    assert client.post("/api/catalog-sources/snapshot", json=request).status_code == 422


def mapped(text, *, filename="quoted.csv", **meta):
    """Map an explicit seven-column survey with comma decimals."""
    return build_snapshot(
        **upload(filename, text.encode("utf-8-sig")), sheet="CSV", delimiter=";", decimal=",",
        mapping={field: {"column": str(index), "unit": unit} for index, (field, unit) in enumerate([
            ("part_number", "text"), ("voltage_v", "V"), ("current_a", "A"), ("rds_ohm", "mOhm"),
            ("price", "source currency"), ("quote_quantity", "pieces"), ("qg_c", "nC"),
        ], 1)},
        metadata={"technology": "SiC", "currency": "CHF", "snapshot": "survey A", **meta},
    )


def test_source_identity_exact_quotes_real_zero_invalid_units_and_tiers():
    """Trace duplicate parts, invalid scales and mapped tiers to their source rows."""
    text = 'Part;V;I;R;price;tier;Qg\r\n"A;""B""";650;20;2,5;3;100;1e-319\r\n"A;""B""";0;20;0;1e-999;;0\r\n'
    first = mapped(text, quote_quantity=1000)
    assert base64.b64decode(first["source_file"]["content_base64"]) == text.encode("utf-8-sig")
    assert first["source_file"]["sha256"] == hashlib.sha256(text.encode("utf-8-sig")).hexdigest()
    a, b = first["records"]
    assert a["id"] != b["id"] and a["part_number"] == b["part_number"] == 'A;"B"'
    assert a["rds_ohm"] == 0.0025 and b["voltage_v"] == b["rds_ohm"] == b["qg_c"] == 0
    assert a["qg_c"] is None and b["price"] is None
    assert a["quote_context"]["quote_quantity"] == a["quote_quantity"] == 100
    assert b["quote_context"]["quote_quantity"] is b["quote_quantity"] is None
    assert first["sources"][0]["snapshot_id"] == first["snapshot_id"]
    assert first["snapshot_id"] == mapped(text, quote_quantity=1000)["snapshot_id"]
    assert first["snapshot_id"] != mapped(text, quote_quantity=10)["snapshot_id"]
    with pytest.raises(ValueError, match="unique"):
        compare_snapshots(first, first, key_fields=["part_number"])
    changed = mapped(text.replace("2,5", "3,1"), quote_quantity=1000)
    delta = compare_snapshots(first, changed, key_fields=["part_number", "row"])
    assert not delta["added"] and not delta["missing"]
    assert len(delta["changed"]) == 1 and "rds_ohm" in delta["changed"][0]["fields"]
    assert json.loads(json.dumps(first)) == first


def test_two_source_frontier_and_independent_cost_coverage(client):
    """Check every physical frontier member and each panel population."""
    header = "part;V;I;R;price;tier;Qg\n"
    a = mapped(header + "A1;650;20;1;4;10;4\nA2;650;20;3;5;10;5\nA3;650;20;50;;;;\n")
    b = mapped(header + "B1;650;20;2;2;10;2\nB2;650;20;4;1;10;1\nT;650;20;2;2;10;2\n", snapshot="survey B")
    catalog = {"schema_version": 1, "sources": a["sources"] + b["sources"], "records": a["records"] + b["records"]}
    selection = {"source_ids": [s["id"] for s in catalog["sources"]], "x_metric": "rds_ohm", "y_metric": "qg_c"}
    output = screen_catalog(catalog, **selection)
    flags = {r["part_number"]: r["pareto"] for r in output["candidates"]}
    assert flags == {"A1": True, "A2": False, "A3": None, "B1": True, "B2": True, "T": True}
    assert [p["plotted_count"] for p in output["panels"]] == [5, 0, 6, 5]
    response = client.post("/api/catalog/screen", json={"snapshot": catalog, **selection})
    assert response.status_code == 200 and response.json() == output
    quote = {"currency": "CHF", "quote_quantity": 10}
    priced = screen_catalog(catalog, **selection, quote_cohort=quote)
    assert [p["plotted_count"] for p in priced["panels"]] == [5, 5, 6, 5]
    for column, invalid in (("quote_quantity", None), ("currency", "EUR")):
        incompatible = json.loads(json.dumps(catalog))
        record = incompatible["records"][-1]
        record[column] = record["quote_context"][column] = invalid
        mixed = screen_catalog(incompatible, **selection, quote_cohort=quote)
        assert mixed["cost_cohort"]["status"] == "unavailable"
        assert [p["plotted_count"] for p in mixed["panels"]] == [5, 0, 6, 5]


def test_declared_normalization_reports_all_minima_and_reproduces_fixed_population(client):
    """Check tied denominators and distinguish fixed and visible populations."""
    catalog = mapped("part;V;I;R;price;tier;Qg\nL1;1;2;1;1;1;1\nL2;1;2;1;1;1;1\nM;4;4;1;1;1;1\nH;8;8;1;1;1;1\nZero;1;0;1;1;1;1\n")
    catalog = {k: catalog[k] for k in ("schema_version", "sources", "records")}
    base = {"snapshot": catalog, "source_id": catalog["sources"][0]["id"], "x_metric": "voltage_v", "y_metric": "current_a"}
    def run(mode, **filters):
        response = client.post("/api/catalog/screen", json={**base, "normalization": {"mode": mode, "metrics": ["current_a"]}, **filters})
        assert response.status_code == 200
        return response.json()
    original, fixed, visible = run("fixed"), run("fixed", voltage_min=4), run("visible", voltage_min=4)
    expected_ids = sorted(r["id"] for r in catalog["records"][:2])
    evidence = fixed["normalization"]["metrics"]["current_a"]
    assert evidence["denominator"] == 2 and evidence["denominator_ids"] == expected_ids
    assert evidence["exclusions"] == [{"id": catalog["records"][-1]["id"], "reason": "nonpositive"}]
    assert fixed["normalization"]["population_hash"] == original["normalization"]["population_hash"]
    assert visible["normalization"]["population_hash"] != fixed["normalization"]["population_hash"]
    assert [r["normalization"]["current_a"]["relative"] for r in fixed["candidates"]] == [2, 4]
    assert [r["normalization"]["current_a"]["relative"] for r in visible["candidates"]] == [1, 2]
    assert visible["normalization"]["metrics"]["current_a"]["denominator"] == 4
    assert visible["pareto_count"] == fixed["pareto_count"]


def test_bad_operands_and_scaled_extreme_finite_arithmetic():
    """Retain valid physical operands when a derived ratio exceeds its range."""
    for value in (False, True, 10**400):
        values, reasons = metric_evidence({"current_a": value, "voltage_v": 650})
        assert values["voltage_v"] == 650 and values["current_a"] is None and reasons["vi"]
    for scalar in (1e250, 1e-250):
        values, reasons = metric_evidence({"price": scalar, "rds_ohm": scalar, "voltage_v": scalar, "qg_c": scalar})
        assert values["cost_fom"] == pytest.approx(1)
        assert values["rds_qg"] is None and reasons["rds_qg"].startswith("derived-")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Historical source archive unavailable")
def test_reference_sources_and_static_ooxml_match_the_bundled_artifact():
    """Compare raw XML relationships and shared formulas with the imported evidence."""
    bundled = load_catalog()
    by_file = {source["file"]: source["sha256"] for source in bundled["sources"]}
    assert len(by_file) == 5
    for filename, digest in by_file.items():
        assert hashlib.sha256((ARCHIVE / filename).read_bytes()).hexdigest() == digest
    census = Counter(r["source_id"] for r in bundled["records"])
    assert sorted(census.values()) == [23, 38, 99, 174, 233, 286, 500, 1137]
    for source in bundled["sources"]:
        if source["file"].endswith(".csv"):
            with (ARCHIVE / source["file"]).open(encoding="cp1252", newline="") as stream:
                assert sum(1 for _ in csv.reader(stream)) - 1 == census[source["id"]]
    name = "SemiconductorsData.xlsm"
    raw = (ARCHIVE / name).read_bytes()
    request = upload(name, raw)
    links, coeffs = static_links(**request), workbook_coefficients(**request)
    inspected = {(e["sheet"], e["cell"]) for e in links["entries"] if e["kind"] == "stored-link"}
    formulas = {(e["sheet"], e["cell"]) for e in links["geturl_formula_locations"]}
    with zipfile.ZipFile(io.BytesIO(raw)) as book:
        for i, (sheet, count) in enumerate((("MOSFET", 23), ("IGBT", 36)), 1):
            root = ET.fromstring(book.read(f"xl/worksheets/sheet{i}.xml"))
            native = {n.get("ref") for n in root.findall("s:hyperlinks/s:hyperlink", NS)}
            assert len(native) == count and {c for s, c in inspected if s == sheet} == native
            cells = [(c.get("r"), c.find("s:f", NS)) for c in root.findall(".//s:c", NS)]
            groups = {f.get("si") for _, f in cells if f is not None and f.get("t") == "shared" and (f.text or "").startswith("GetURL(")}
            xml_calls = {coord for coord, f in cells if f is not None and
                         ((f.text or "").startswith("GetURL(") or (f.get("t") == "shared" and f.get("si") in groups))}
            assert {c for s, c in formulas if s == sheet} == xml_calls
    assert len(inspected) == 59 and len(formulas) == 63
    for sheet, energy, current, volts in (("MOSFET", .000632, 36, 600), ("IGBT", .003, 80, 400)):
        entry = next(c for c in coeffs["coefficients"] if c["sheet"] == sheet and c["cell"]["cell"] == "Y2")
        assert entry["unit"] == "J/A" and entry["reference_temperature_c"] == 175
        assert entry["reference_current_a"] == current and entry["reference_voltage_v"] == volts
        assert entry["recognized_literal_coefficient_j_per_a"] == pytest.approx(energy / current)
        assert entry["gate_resistance_ohm"] is None
    assert (ARCHIVE / name).read_bytes() == raw
