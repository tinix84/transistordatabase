"""Independent checks of the archived workbook rows and portable screening API."""

import base64
from copy import deepcopy
from decimal import Decimal, localcontext
from hashlib import sha256
from io import BytesIO
from xml.etree import ElementTree
from zipfile import ZipFile

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook
import pytest

from transistordatabase.catalog_fom import CATALOG_PATH, load_catalog, screen_catalog
from transistordatabase.catalog_sources import build_snapshot
from transistordatabase.gui_web.backend.catalog_routes import router


MOS = "SemiconductorsData/MOSFET"
IGBT = "SemiconductorsData/IGBT"
ACK = {"thermal_convention": "source-constant-model-100C"}


def source_rows(*ids):
    """Select the first data row of each specified bundled sheet."""
    data = load_catalog()
    return {
        "schema_version": 1,
        "sources": [s for s in data["sources"] if s["id"] in ids],
        "records": [r for r in data["records"] if r["source_id"] in ids and r["row"] == 2],
    }


def apply_screen(data, x, y, **options):
    """Screen the selected sheets through the public Python entry point."""
    return screen_catalog(data, source_ids=[s["id"] for s in data["sources"]],
                          x_metric=x, y_metric=y, **options)


def conduction_root(power, threshold, resistance):
    """Bracket the positive root by its dissipated heat, then bisect."""
    low, high = Decimal(0), Decimal(1)
    while high * (threshold + resistance * high) < power:
        high *= 2
    for _ in range(180):
        mid = (low + high) / 2
        if mid * (threshold + resistance * mid) < power:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)


@pytest.mark.parametrize("sid", [IGBT, MOS])
def test_actual_source_current_cost_conditions_and_file_fingerprint(sid):
    """Recompute both original source rows from their recorded operands."""
    before = sha256(CATALOG_PATH.read_bytes()).hexdigest()
    source = source_rows(sid)
    original = deepcopy(source)
    result = apply_screen(source, "thermal_current_a", "thermal_vi_per_price",
                          workbook_derivation={**ACK, "source_declarations": [
                              {"source_id": sid, "currency": "CHF"}]})
    point, = result["points"]
    metrics, proof = point["metrics"], point["workbook_evidence"]
    with localcontext() as ctx:
        ctx.prec = 65
        if sid == IGBT:
            budget = Decimal(75) / Decimal("1.06")
            current = conduction_root(budget, Decimal(".85"), Decimal(1) / 65)
            voltage, price, gate, model_temp, ref = 650, 7.43, 15, 175, "heatsink"
            assert metrics["voltage_energy_benchmark_hz"] is None
            assert "err_j_a" in point["metric_exclusions"]["voltage_energy_benchmark_hz"]
            assert metrics["rce_coes_s"] == pytest.approx(175e-12 / 65, rel=1e-13, abs=0)
            assert metrics["rce_qg_vs"] == pytest.approx(175e-9 / 65, rel=1e-13, abs=0)
        else:
            budget = Decimal(75) / Decimal(".35")
            current = conduction_root(budget, Decimal(0), Decimal(".03575"))
            voltage, price, gate, model_temp, ref = 1200, 63.2, 18, 150, "case"
            expected_hz = float(Decimal(600) * 36 / Decimal(".000875"))
            assert metrics["voltage_energy_benchmark_hz"] == pytest.approx(expected_hz)
            assert metrics["voltage_energy_per_price"] == pytest.approx(expected_hz / price)
            assert proof["source_inputs"]["err_j_a"]["value"] == 0
    assert point["x"] == pytest.approx(current, rel=1e-13)
    assert point["y"] == pytest.approx(voltage * current / price, rel=1e-13)
    assert metrics["thermal_power_budget_w"] == pytest.approx(float(budget))
    assert proof["conditions"]["conduction_model_temperature_c"] == model_temp
    assert proof["conditions"]["conduction_model_gate_voltage_v"] == gate
    assert proof["conditions"]["thermal_reference_kind"] == ref
    assert proof["conditions"]["switching_coefficient_temperature_c"] == 175
    assert proof["quote_origin"] == "octoparts" and proof["quote_quantity"] == 1000
    assert proof["original_currency"] == "unspecified"
    assert proof["currency_classification"] == "A"
    assert source == original
    assert sha256(CATALOG_PATH.read_bytes()).hexdigest() == before


def test_archived_area_paths_and_declared_unit_scaling():
    """Keep the diode and IGBT paths distinct under three declared units."""
    data = source_rows(IGBT)
    row = data["records"][0]
    # Replace only the blank geometry and its two matching thermal paths.
    for col, value in {"30": 4, "31": 3, "36": .25, "37": .5}.items():
        row["raw"][col]["value"] = value
    x, y = "igbt_area_thermal_k_m2_w", "diode_area_thermal_k_m2_w"
    undecided = apply_screen(data, x, y)
    assert undecided["plotted_count"] == 0
    assert undecided["candidates"][0]["metrics"][x] is None
    for unit, factor in (("mm2", 1e-6), ("cm2", 1e-4), ("m2", 1)):
        settings = {"source_declarations": [{"source_id": IGBT, "area_unit": unit}]}
        point, = apply_screen(data, x, y, workbook_derivation=settings)["points"]
        assert (point["x"], point["y"]) == pytest.approx((factor, 1.5 * factor))
        assert point["workbook_evidence"]["area_basis"]["source_unit"] == "unknown"


def test_cost_overlay_requires_compatible_tiers_and_stays_population_scoped():
    """Keep an incompatible quote in the overlay comparison after a row filter."""
    data = source_rows(MOS, IGBT)
    options = {"workbook_derivation": {**ACK, "source_declarations": [
        {"source_id": sid, "currency": "CHF"} for sid in (MOS, IGBT)]}}
    axes = ("thermal_current_a", "thermal_vi_per_price")
    assert apply_screen(data, *axes, **options)["plotted_count"] == 0
    options["quote_cohort"] = {"currency": "CHF", "quote_quantity": 1000}
    assert apply_screen(data, *axes, **options)["plotted_count"] == 2
    data["records"][0]["quote_quantity"] = 500
    other = data["records"][1]
    filtered = apply_screen(data, *axes, search=other["part_number"], **options)
    assert filtered["matched_count"] == 1
    assert filtered["coverage"]["thermal_current_a"] == 1
    assert filtered["plotted_count"] == 0
    assert filtered["cost_cohort"]["incompatible_ids"] == [data["records"][0]["id"]]


@pytest.mark.parametrize("cache", [True, "bad numeric cache", .5])
def test_archived_literal_and_present_invalid_cache_stay_unavailable(cache):
    """Do not publish a benchmark from contradictory source evidence."""
    data = source_rows(MOS)
    data["records"][0]["raw"]["25"]["value"] = cache
    point, = apply_screen(data, "voltage_energy_benchmark_hz", "voltage_v")["candidates"]
    assert point["metrics"]["voltage_energy_benchmark_hz"] is None
    assert point["workbook_evidence"]["source_inputs"]["eon_j_a"]["source_value"] == cache


def test_uploaded_xlsx_coefficient_rejects_contradictory_cache():
    """Map an uploaded formula with a stale cached value through the public importer."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["part", "voltage", "eon", "eoff", "err"])
    sheet.append(["example", 600, "=0.000632/36", .00001, 0])
    original, changed = BytesIO(), BytesIO()
    workbook.save(original)
    with ZipFile(original) as source, ZipFile(changed, "w") as target:
        for info in source.infolist():
            content = source.read(info)
            if info.filename == "xl/worksheets/sheet1.xml":
                root = ElementTree.fromstring(content)
                root.find(".//{*}c[@r='C2']/{*}v").text = "0.5"
                content = ElementTree.tostring(root)
            target.writestr(info, content)
    columns = [("part_number", "text"), ("switch_voltage_v", "V"),
               ("eon_j_a", "J/A"), ("eoff_j_a", "J/A"), ("err_j_a", "J/A")]
    data = build_snapshot(
        "stale.xlsx", base64.b64encode(changed.getvalue()).decode(), sheet="Sheet",
        mapping={name: {"column": column, "unit": unit}
                 for column, (name, unit) in zip("ABCDE", columns, strict=True)},
        metadata={"workbook_profile": "mosfet"},
    )
    candidate, = apply_screen(data, "voltage_energy_benchmark_hz", "voltage_v")["candidates"]
    assert candidate["metrics"]["voltage_energy_benchmark_hz"] is None
    assert "cache" in candidate["metric_exclusions"]["voltage_energy_benchmark_hz"]
    assert candidate["raw"]["C"]["formula"] == "=0.000632/36"
    assert candidate["raw"]["C"]["cached_value"] == .5


def mapped_source():
    """Upload three CSV rows with positive, zero, and missing resistance."""
    payload = b"name,r,cap,junction,area,path\nknown,1,4,150,9,.5\nzero,0,4,150,9,.5\nunknown,,4,150,9,.5\n"
    columns = [("part_number", "text"), ("model_resistance_ohm", "Ohm"),
               ("coes_f", "pF"), ("junction_limit_c", "degC"), ("igbt_area_m2", "mm2"),
               ("igbt_rth_jf_k_w", "K/W")]
    return build_snapshot(
        "comparison.csv", base64.b64encode(payload).decode(), sheet="CSV",
        mapping={field: {"column": str(i), "unit": unit}
                 for i, (field, unit) in enumerate(columns, 1)},
        metadata={"workbook_profile": "igbt"},
    )


def test_csv_mapped_known_zero_absent_and_strictly_positive_reference():
    """Verify panel coverage, source units, and the normalization population."""
    snapshot = mapped_source()
    method = {"mode": "fixed", "metrics": ["rce_coes_s"]}
    key, area = "rce_coes_s", "igbt_area_thermal_k_m2_w"
    screen = apply_screen(snapshot, key, area, normalization=method)
    assert screen["matched_count"] == 3 and screen["plotted_count"] == 2
    known, zero, missing = screen["candidates"]
    assert known["metrics"][key] == pytest.approx(4e-12, rel=1e-13, abs=0)
    assert zero["metrics"][key] == 0 and missing["metrics"][key] is None
    metric = screen["normalization"]["metrics"][key]
    assert metric["denominator_ids"] == [known["id"]]
    assert [r["relative"] for r in metric["values"]] == [1, 0, None]
    assert known["metrics"][area] == pytest.approx(4.5e-6, rel=1e-13)
    proof = known["workbook_evidence"]
    assert "igbt_area_m2" in proof["metrics"][area]["inputs"]
    assert proof["source_inputs"]["igbt_area_m2"]["source_cell"]["value"] == "9"
    filtered = apply_screen(snapshot, key, area, normalization=method, search="zero")
    assert filtered["normalization"]["population_hash"] == screen["normalization"]["population_hash"]
    visible = apply_screen(snapshot, key, area, normalization={**method, "mode": "visible"},
                           search="zero")
    assert visible["normalization"]["metrics"][key]["denominator"] is None


@pytest.fixture(scope="module")
def client():
    """Observe public HTTP status codes, including server errors."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def post(client, data, **fields):
    """Submit a portable snapshot through the typed screening API."""
    return client.post("/api/catalog/screen", json={
        "snapshot": {key: data[key] for key in ("schema_version", "sources", "records")},
        "source_id": data["sources"][0]["id"], "x_metric": "rce_coes_s",
        "y_metric": "thermal_current_a", "workbook_derivation": ACK, **fields})


@pytest.mark.parametrize("key,replacement", [
    ("mapping_evidence", None), ("mapping_evidence", {"coes_f": []}),
])
def test_portable_malformed_scientific_evidence_returns_client_error(client, key, replacement):
    """Reject an invalid mapping object at the transport boundary."""
    data = mapped_source()
    data["records"][0][key] = replacement
    assert post(client, data).status_code == 422


def test_archived_malformed_raw_cell_returns_client_error(client):
    """Reject a scalar in a profile that requires source-cell objects."""
    data = source_rows(IGBT)
    data["records"][0]["raw"]["22"] = 5
    assert post(client, data).status_code == 422


def test_portable_metadata_cannot_override_canonical_scientific_value(client):
    """Keep a metadata field from replacing a scientific operand."""
    data = mapped_source()
    data["records"][0]["mapping_evidence"]["model_resistance_ohm"]["value"] = 123
    response = post(client, data)
    if response.status_code == 422:
        return
    assert response.status_code == 200
    known = response.json()["candidates"][0]
    assert known["metrics"]["rce_coes_s"] == pytest.approx(4e-12, rel=1e-13, abs=0)


def test_out_of_range_json_integer_cannot_abort_other_metrics(client):
    """Handle a finite JSON integer outside the scientific float range."""
    data = mapped_source()
    data["records"][0].update(model_temperature_c=10**400, model_gate_voltage_v=15,
                              model_threshold_v=1, thermal_path_k_w=1)
    response = post(client, data)
    if response.status_code == 422:
        return
    assert response.status_code == 200
    known = response.json()["candidates"][0]
    assert known["metrics"]["thermal_current_a"] is None
    assert known["metrics"]["rce_coes_s"] == pytest.approx(4e-12, rel=1e-13, abs=0)
