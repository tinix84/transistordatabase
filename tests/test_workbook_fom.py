"""Independent workbook equations and immutable catalog transport evidence."""

import base64
from copy import deepcopy
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.catalog_fom import load_catalog, screen_catalog
from transistordatabase.catalog_sources import (
    build_snapshot, coefficient_cache_evidence, literal_energy_coefficient,
)
from transistordatabase.gui_web.backend.catalog_routes import router
from transistordatabase.workbook_fom import CONVENTION


def fixture_catalog():
    """Use a 75 W budget and an independently soluble 5 A conduction model."""
    return {"schema_version": 1,
            "sources": [{"id": "fixture", "sha256": "a" * 64, "currency": "unspecified"}],
            "records": [{"id": "fixture:2", "source_id": "fixture", "row": 2,
                         "part_number": "fixture", "manufacturer": "fixture", "technology": "IGBT",
                         "workbook_profile": "igbt", "voltage_v": 1200, "current_a": 100,
                         "model_gate_voltage_v": 15, "model_temperature_c": 125,
                         "model_resistance_ohm": 2, "model_threshold_v": 5,
                         "junction_limit_c": 175, "thermal_path_k_w": 1,
                         "switch_voltage_v": 600, "eon_j_a": 1e-5, "eoff_j_a": 2e-5, "err_j_a": 0,
                         "coes_f": 200e-12, "source_qg_c": 100e-9,
                         "igbt_area_m2": 2e-6, "diode_area_m2": 1e-6,
                         "igbt_rth_jf_k_w": .5, "diode_rth_jf_k_w": .75,
                         "price": 10, "quote_quantity": 1000, "quote_origin": "fixture quote"}]}


def run(data=None, **overrides):
    """Screen actual derived axes, through the public catalog authority."""
    return screen_catalog(data or fixture_catalog(), source_id="fixture",
                          x_metric="thermal_power_budget_w", y_metric="thermal_current_a", **overrides)


def settings(currency="CHF"):
    """Declare the exact user assumptions needed by this fixture."""
    return {"thermal_convention": CONVENTION,
            "source_declarations": [{"source_id": "fixture", "currency": currency}]}


def test_named_equations_use_thermal_current_and_test_voltage_with_evidence():
    """Check all requested physical metrics against independently chosen values."""
    data = fixture_catalog()
    original = deepcopy(data)
    result = run(data, workbook_derivation=settings())
    point = result["points"][0]
    values = point["metrics"]
    assert values["thermal_power_budget_w"] == 75
    assert values["thermal_current_a"] == 5
    assert values["thermal_current_per_price"] == .5
    assert values["thermal_vi_per_price"] == 600
    assert values["voltage_energy_benchmark_hz"] == pytest.approx(20e6)
    assert values["voltage_energy_per_price"] == pytest.approx(2e6)
    assert values["rce_coes_s"] == pytest.approx(400e-12)
    assert values["rce_qg_vs"] == pytest.approx(200e-9)
    assert values["igbt_area_thermal_k_m2_w"] == pytest.approx(1e-6)
    assert values["diode_area_thermal_k_m2_w"] == pytest.approx(.75e-6)
    proof = point["workbook_evidence"]
    assert proof["conditions"]["conduction_model_temperature_c"] == 125
    assert proof["conditions"]["thermal_reference_kind"] == "heatsink"
    assert proof["currency"] == "CHF" and proof["original_currency"] == "unspecified"
    assert proof["currency_classification"] == "A"
    assert proof["quote_origin"] == "fixture quote" and proof["quote_quantity"] == 1000
    assert proof["metrics"]["thermal_current_a"]["classification"] == "E"
    assert proof["metrics"]["thermal_current_a"]["equation"].startswith("Solve V0")
    assert data == original


def test_default_estimates_unknown_currency_and_missing_coefficients_stay_unavailable():
    """A blank recovery coefficient never becomes a known zero."""
    data = fixture_catalog()
    values = run(data)["candidates"][0]["metrics"]
    assert values["thermal_current_a"] is None
    assert values["thermal_power_budget_w"] is None
    assert values["rce_coes_s"] == pytest.approx(400e-12)
    assert values["voltage_energy_per_price"] is None
    data["records"][0]["err_j_a"] = None
    point = run(data, workbook_derivation=settings())["candidates"][0]
    assert point["metrics"]["voltage_energy_benchmark_hz"] is None
    assert point["metric_exclusions"]["voltage_energy_benchmark_hz"] == "missing-input: err_j_a"
    assert point["metrics"]["thermal_current_a"] == 5


@pytest.mark.parametrize("field", ["model_temperature_c", "model_gate_voltage_v", "model_threshold_v", "model_resistance_ohm"])
def test_budget_is_independent_of_missing_conduction_model(field):
    """AV remains available when an AW-only prerequisite is missing."""
    data = fixture_catalog()
    data["records"][0][field] = None
    point = run(data, workbook_derivation=settings())["candidates"][0]
    assert point["metrics"]["thermal_power_budget_w"] == 75
    assert point["metrics"]["thermal_current_a"] is None


def test_zero_budget_ideal_model_and_zero_coefficients_have_explicit_domains():
    """Keep a known zero result and reject an unbounded ideal-loss current."""
    data = fixture_catalog()
    row = data["records"][0]
    row["junction_limit_c"] = 100
    point = run(data, workbook_derivation=settings())["points"][0]
    assert point["x"] == 0 and point["y"] == 0
    row["model_resistance_ohm"] = row["model_threshold_v"] = 0
    point = run(data, workbook_derivation=settings())["candidates"][0]
    assert point["metrics"]["thermal_power_budget_w"] == 0
    assert point["metrics"]["thermal_current_a"] is None
    assert point["metric_exclusions"]["thermal_current_a"] == "no-finite-thermal-current-limit"
    row["junction_limit_c"] = 99
    assert run(data, workbook_derivation=settings())["candidates"][0]["metrics"]["thermal_power_budget_w"] is None
    row["eon_j_a"] = row["eoff_j_a"] = row["err_j_a"] = 0
    point = run(data)["candidates"][0]
    assert point["metric_exclusions"]["voltage_energy_benchmark_hz"] == "zero-aggregate-energy-coefficient"


@pytest.mark.parametrize("field,value", [("err_j_a", True), ("model_resistance_ohm", -1), ("thermal_path_k_w", 0), ("igbt_area_m2", 0)])
def test_invalid_scientific_inputs_only_exclude_dependent_metrics(field, value):
    """Reject invalid operands without filtering unrelated physical panels."""
    data = fixture_catalog()
    data["records"][0][field] = value
    point = run(data, workbook_derivation=settings())["candidates"][0]
    metric = {"err_j_a": "voltage_energy_benchmark_hz", "model_resistance_ohm": "thermal_current_a",
              "thermal_path_k_w": "thermal_power_budget_w", "igbt_area_m2": "igbt_area_thermal_k_m2_w"}[field]
    assert point["metrics"][metric] is None
    assert point["metrics"]["voltage_v"] == 1200


def test_normalization_proofs_include_assumptions_and_zero_is_not_a_denominator():
    """Hash the source assumptions and use only a positive denominator."""
    data = fixture_catalog()
    zero = deepcopy(data["records"][0])
    zero.update({"id": "fixture:3", "junction_limit_c": 100})
    data["records"].append(zero)
    arguments = {"normalization": {"mode": "fixed", "metrics": ["thermal_current_a"]}}
    result = run(data, workbook_derivation=settings(), **arguments)
    metric = result["normalization"]["metrics"]["thermal_current_a"]
    assert metric["denominator"] == 5
    assert [v["relative"] for v in metric["values"]] == [1, 0]
    altered = run(data, workbook_derivation=settings("EUR"), **arguments)
    assert altered["normalization"]["population_hash"] != result["normalization"]["population_hash"]
    filtered = run(data, workbook_derivation=settings(), search="absent", **arguments)
    assert filtered["normalization"]["population_hash"] == result["normalization"]["population_hash"]


def test_quote_origin_tier_and_currency_declarations_remain_explicit():
    """Require recorded quote conditions and prevent a currency relabeling."""
    for field in ("quote_origin", "quote_quantity"):
        data = fixture_catalog()
        data["records"][0][field] = None
        point = run(data, workbook_derivation=settings())["candidates"][0]
        assert point["metrics"]["thermal_current_per_price"] is None
        assert point["metrics"]["thermal_current_a"] == 5
    data = fixture_catalog()
    data["sources"][0]["currency"] = "EUR"
    with pytest.raises(ValueError, match="cannot replace"):
        run(data, workbook_derivation=settings())


def test_archived_igbt_row_two_uses_source_path_and_keeps_missing_err():
    """Check the actual IGBT source row independently of cached output cells."""
    data = load_catalog()
    original = deepcopy(data)
    identity = "SemiconductorsData/IGBT"
    result = screen_catalog(data, source_id=identity, x_metric="thermal_power_budget_w", y_metric="thermal_current_a",
                            workbook_derivation={"thermal_convention": CONVENTION})
    point = next(r for r in result["candidates"] if r["id"] == identity + ":2")
    assert point["metrics"]["thermal_power_budget_w"] == pytest.approx(70.75471698113208)
    assert point["metrics"]["thermal_current_a"] == pytest.approx(45.6020252633)
    assert point["metrics"]["current_a"] == 100
    assert point["metrics"]["voltage_energy_benchmark_hz"] is None
    assert point["workbook_evidence"]["source_inputs"]["coes_f"]["source_value"] == 175
    assert point["metrics"]["rce_coes_s"] is not None
    assert point["metrics"]["thermal_vi_per_price"] is None
    assert data == original


def test_archived_mosfet_literal_coefficient_retains_reference_point():
    """Retain a reviewed literal coefficient and its actual source point."""
    data = load_catalog()
    identity = "SemiconductorsData/MOSFET"
    result = screen_catalog(data, source_id=identity, x_metric="voltage_v", y_metric="voltage_energy_benchmark_hz")
    point = next(r for r in result["candidates"] if r["id"] == identity + ":2")
    coefficient = point["workbook_evidence"]["source_inputs"]["eon_j_a"]
    assert coefficient["value"] == pytest.approx(.000632 / 36)
    assert coefficient["coefficient_reference"]["reference_current_a"] == 36
    assert coefficient["conditions"]["test_voltage_v"] == 600
    assert coefficient["conditions"]["temperature_c"] == 175


def test_source_unit_underflow_retains_the_source_value_and_specific_reason():
    """Keep a nonzero source operand distinct from missing data or known zero."""
    data = load_catalog()
    identity = "SemiconductorsData/MOSFET"
    row = deepcopy(next(r for r in data["records"] if r["id"] == identity + ":2"))
    row["raw"]["22"]["value"] = math.ulp(0.0)
    data["records"] = [row]
    result = screen_catalog(data, source_id=identity, x_metric="thermal_current_a", y_metric="thermal_power_budget_w",
                            workbook_derivation={"thermal_convention": CONVENTION})
    point = result["candidates"][0]
    assert point["metrics"]["thermal_current_a"] is None
    assert point["metric_exclusions"]["thermal_current_a"] == "source-unit-conversion-underflow: model_resistance_ohm"
    source = point["workbook_evidence"]["source_inputs"]["model_resistance_ohm"]
    assert source["source_value"] == math.ulp(0.0)
    assert source["value"] is None


def test_source_mapping_declares_profile_and_area_units_without_formula_execution():
    """Map known units into a new snapshot without changing the source bytes."""
    content = b"part,R,V0,gate,T,Tj,Rth,V,Eon,Eoff,Err,area,Rjf,origin,price\np,2,5,15,125,175,1,600,.00001,.00002,0,2,.5,quote,10\n"
    fields = {"part_number": (1, "text"), "model_resistance_ohm": (2, "Ohm"),
              "model_threshold_v": (3, "V"), "model_gate_voltage_v": (4, "V"),
              "model_temperature_c": (5, "degC"), "junction_limit_c": (6, "degC"),
              "thermal_path_k_w": (7, "K/W"), "switch_voltage_v": (8, "V"),
              "eon_j_a": (9, "J/A"), "eoff_j_a": (10, "J/A"), "err_j_a": (11, "J/A"),
              "igbt_area_m2": (12, "mm2"), "igbt_rth_jf_k_w": (13, "K/W"),
              "quote_origin": (14, "text"), "price": (15, "source currency")}
    snapshot = build_snapshot("fixture.csv", base64.b64encode(content).decode(), sheet="CSV",
                              mapping={k: {"column": str(v[0]), "unit": v[1]} for k, v in fields.items()},
                              metadata={"workbook_profile": "igbt", "currency": "CHF", "quote_quantity": 1000})
    identity = snapshot["sources"][0]["id"]
    result = screen_catalog(snapshot, source_id=identity, x_metric="thermal_current_a", y_metric="igbt_area_thermal_k_m2_w",
                            workbook_derivation={"thermal_convention": CONVENTION})
    point = result["points"][0]
    assert point["x"] == 5 and point["y"] == pytest.approx(1e-6)
    area_input = point["workbook_evidence"]["source_inputs"]["igbt_area_m2"]
    assert area_input["unit"] == "m2"
    assert area_input["mapping_evidence"]["unit"] == "mm2"
    area_proof = point["workbook_evidence"]["metrics"]["igbt_area_thermal_k_m2_w"]
    assert area_proof["inputs"] == ["igbt_area_m2", "igbt_rth_jf_k_w"]
    assert area_proof["area_conversion"]["factor_to_m2"] == 1
    assert snapshot["source_file"]["content_base64"] == base64.b64encode(content).decode()


@pytest.mark.parametrize("expression", ["=A2/B2", "=SUM(1,2)", "=1/0", "=1e-400/1", "=1/1e-400", "=1e308/1e-308"])
def test_literal_coefficient_recognizer_has_a_fixed_finite_grammar(expression):
    """Reject arbitrary expressions and unrepresentable literal divisions."""
    assert literal_energy_coefficient(expression) is None


@pytest.mark.parametrize("cache", [.5, True, False, "bad", "#VALUE!"])
def test_literal_coefficient_rejects_a_contradictory_or_malformed_saved_cache(cache):
    """Never promote a present invalid cache into a valid energy coefficient."""
    evidence = coefficient_cache_evidence("=0.000632/36", cache)
    assert evidence["value"] is None
    assert "cache" in evidence["validation"]
    assert evidence["literal"]["reference_current_a"] == 36


def test_coefficient_source_cache_policy_preserves_explicit_literal_and_general_cache():
    """Recognize a named literal while preserving other saved formulas without execution."""
    expected = .000632 / 36
    for cache in (None, expected, "0.000632/36"):
        evidence = coefficient_cache_evidence("=0.000632/36", cache)
        assert evidence["value"] == pytest.approx(expected)
        assert evidence["validation"] is None
    general = coefficient_cache_evidence("=A2/B2", .5)
    assert general["literal"] is None and general["value"] == .5


@pytest.mark.parametrize("expression", ["=1/0", "=1/1e-400", "=1e-400/1", "=1e308/1e-308"])
def test_invalid_literal_cannot_fall_back_to_a_present_general_formula_cache(expression):
    """Reject invalid literal arithmetic even when a finite saved cache exists."""
    evidence = coefficient_cache_evidence(expression, .5)
    assert evidence["literal_syntax"] is True
    assert evidence["value"] is None
    assert evidence["validation"] == "invalid-or-unrepresentable-coefficient-literal"


def test_typed_api_transports_evidence_and_rejects_implicit_or_malformed_declarations():
    """Preserve the declared assumptions through strict HTTP transport."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = {"snapshot": fixture_catalog(), "source_id": "fixture", "x_metric": "thermal_current_a",
               "y_metric": "thermal_power_budget_w", "workbook_derivation": settings()}
    response = client.post("/api/catalog/screen", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["points"][0]["x"] == 5
    assert response.json()["workbook_derivation"] == {**settings(), "source_declarations": [{"source_id": "fixture", "currency": "CHF", "area_unit": None}]}
    for declaration in ({"thermal_convention": True}, {"area_unit": "mm2"},
                        {"source_declarations": [{"source_id": "fixture", "currency": True}]},
                        {"source_declarations": [{"source_id": "absent", "currency": "CHF"}]},
                        {"source_declarations": [{"source_id": "fixture", "area_unit": "in2"}]}):
        changed = {**request, "workbook_derivation": declaration}
        assert client.post("/api/catalog/screen", json=changed).status_code == 422
    metadata = client.get("/api/catalog/sources").json()
    assert "description" in metadata["metrics"]["voltage_energy_benchmark_hz"]
    assert all(math.isfinite(v) for p in response.json()["points"] for v in (p["x"], p["y"]))
