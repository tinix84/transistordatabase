"""Independent fixtures for source overlays and declared normalization populations."""

from copy import deepcopy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.catalog_fom import metric_evidence, screen_catalog
from transistordatabase.gui_web.backend.catalog_routes import router


def snapshot():
    """Provide independently defined physical points with duplicate part names."""
    sources = [
        {"id": name, "sha256": digest * 64, "currency": "CHF",
         "sheet": "survey", "snapshot": f"historical-{name}"}
        for name, digest in (("A", "a"), ("B", "b"))
    ]
    records = [
        {"id": name, "source_id": name[0], "row": index + 2,
         "part_number": "duplicate-part", "technology": "SiC",
         "manufacturer": "fixture", "voltage_v": x, "current_a": y,
         "rds_ohm": .05, "price": 2, "quote_quantity": 100,
         "conditions": {"resistance": f"condition-{name}"},
         "raw": {"1": {"value": name}}}
        for index, (name, x, y) in enumerate([
            ("A1", 1, 4), ("B1", 2, 2), ("B2", 4, 1),
            ("A2", 3, 5), ("B3", 2, 2),
        ])
    ]
    return {"schema_version": 1, "sources": sources, "records": records}


def test_physical_overlay_frontier_identity_and_independent_panel_coverage():
    """Keep complete physical points when cost or other objectives are absent."""
    data = snapshot()
    data["records"].append({"id": "A3", "source_id": "A", "row": 9,
                            "voltage_v": 650, "current_a": 20, "rds_ohm": .05})
    original = deepcopy(data)
    result = screen_catalog(data, source_ids=["A", "B"],
                            x_metric="voltage_v", y_metric="current_a")
    assert {p["id"] for p in result["points"] if p["pareto"]} == {"A1", "B1", "B2", "B3"}
    assert result["matched_count"] == 6
    assert result["cost_cohort"]["status"] == "unavailable"
    assert result["panels"][2]["plotted_count"] == 6
    assert result["panels"][1]["plotted_count"] == 0
    assert all("raw" not in p for panel in result["panels"] for p in panel["points"])
    assert data == original
    costs = screen_catalog(data, source_ids=["A"], x_metric="rds_ohm", y_metric="price")
    assert next(p for p in costs["candidates"] if p["id"] == "A3")["pareto"] is None
    for p in result["candidates"][:5]:
        assert p["source"]["sha256"] == ("a" if p["source_id"] == "A" else "b") * 64
        assert p["conditions"]["resistance"] == f"condition-{p['id']}"


def test_cost_cohort_requires_known_matching_currency_and_recorded_quote_tier():
    """Keep invalid mixed quotes out of cost comparisons only."""
    data = snapshot()
    args = {"source_ids": ["A", "B"], "quote_cohort": {"currency": "CHF", "quote_quantity": 100}}
    assert screen_catalog(data, **args)["cost_cohort"]["status"] == "compatible-explicit"
    for field, value in (("currency", "EUR"), ("currency", None), ("quote_quantity", None)):
        changed = deepcopy(data)
        changed["records"][0][field] = value
        result = screen_catalog(changed, **args)
        assert result["cost_cohort"]["status"] == "unavailable"
        assert result["panels"][2]["plotted_count"] == 5
        assert result["coverage"]["price"] == 0


def test_fixed_and_visible_normalization_have_distinct_reproducible_populations():
    """Verify exact independent denominators, ties, hashes, and empty populations."""
    data = snapshot()
    data["records"] = data["records"][:3]
    for row, number in zip(data["records"], [2, 4, 8], strict=True):
        row["voltage_v"] = number
    args = {"source_ids": ["A", "B"], "x_metric": "voltage_v", "y_metric": "current_a"}
    fixed = screen_catalog(data, **args, normalization={"mode": "fixed", "metrics": ["voltage_v"]})
    hidden = screen_catalog(data, **args, voltage_min=4, normalization={"mode": "fixed", "metrics": ["voltage_v"]})
    visible = screen_catalog(data, **args, voltage_min=4, normalization={"mode": "visible", "metrics": ["voltage_v"]})
    assert [r["normalization"]["voltage_v"]["relative"] for r in fixed["candidates"]] == [1, 2, 4]
    assert [r["normalization"]["voltage_v"]["relative"] for r in hidden["candidates"]] == [2, 4]
    assert [r["normalization"]["voltage_v"]["relative"] for r in visible["candidates"]] == [1, 2]
    assert fixed["normalization"]["population_hash"] == hidden["normalization"]["population_hash"]
    assert visible["normalization"]["population_hash"] != hidden["normalization"]["population_hash"]
    assert visible["normalization"]["metrics"]["voltage_v"]["denominator_ids"] == ["B1"]
    tied = deepcopy(data)
    tied["records"][1]["voltage_v"] = 2
    evidence = screen_catalog(tied, **args, normalization={"mode": "fixed", "metrics": ["voltage_v"]})["normalization"]["metrics"]["voltage_v"]
    assert evidence["denominator_ids"] == ["A1", "B1"]
    empty = screen_catalog(data, **args, normalization={"mode": "fixed", "metrics": ["qg_c"]})
    assert empty["normalization"]["metrics"]["qg_c"]["status"] == "unavailable"
    assert len(empty["normalization"]["metrics"]["qg_c"]["exclusions"]) == 3


def test_stable_arithmetic_rejects_unrepresentable_results_and_bool_operands():
    """Preserve representable ratios and reject invalid derived values."""
    values, reasons = metric_evidence({"price": 1e308, "rds_ohm": 1e308, "voltage_v": 1e308,
                                      "qg_c": 1e308, "coss_f": 1e-320, "current_a": True})
    assert values["cost_fom"] == pytest.approx(1)
    assert values["rds_qg"] is None and reasons["rds_qg"] == "derived-overflow"
    assert values["current_a"] is None and reasons["current_a"] == "not-numeric"
    assert metric_evidence({"rds_ohm": 1e-300, "qg_c": 1e-300})[1]["rds_qg"] == "derived-underflow"
    assert metric_evidence({"voltage_v": 10**400})[1]["voltage_v"] == "nonfinite"


def test_typed_api_supplied_snapshot_matches_core_and_rejects_invalid_selection():
    """Exercise snapshot equality and transport validation without global mutation."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    data = snapshot()
    request = {"snapshot": data, "source_ids": ["A", "B"],
               "x_metric": "voltage_v", "y_metric": "current_a", "panel_preset": "survey",
               "normalization": {"mode": "fixed", "metrics": ["voltage_v"]}}
    response = client.post("/api/catalog/screen", json=request)
    assert response.status_code == 200, response.text
    assert response.json() == screen_catalog(data, **{k: v for k, v in request.items() if k != "snapshot"})
    assert len(response.json()["panels"]) == 6
    for changed in ({"source_id": "A"}, {"source_ids": []}, {"voltage_min": True},
                    {"quote_cohort": {"currency": "CHF", "quote_quantity": True}}):
        assert client.post("/api/catalog/screen", json={**request, **changed}).status_code == 422
    malformed = deepcopy(data)
    malformed["records"][0]["source_id"] = {}
    assert client.post("/api/catalog/screen", json={**request, "snapshot": malformed}).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_nonfinite_json_is_a_safe_validation_error(token):
    """Return actionable 422 errors for nonstandard numeric JSON tokens."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = '{"source_ids":["A"],"voltage_min":' + token + '}'
    assert client.post("/api/catalog/screen", content=request).status_code == 422
    data = snapshot()
    data["records"][0]["voltage_v"] = float(token)
    assert client.post("/api/catalog/screen", content=json.dumps({"snapshot": data, "source_ids": ["A"]})).status_code == 422
