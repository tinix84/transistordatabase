"""Numerical, provenance and transport tests for catalog screening."""

import math
import random

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.catalog_fom import (
    load_catalog,
    metric_values,
    pareto_ids,
    parse_quantity,
    screen_catalog,
)
from transistordatabase.gui_web.backend.catalog_routes import router


@pytest.mark.parametrize(
    "raw,unit,expected",
    [
        ("1.2kV", "V", 1200),
        ("2.6A (Tc)", "A", 2.6),
        ("10mA (Ta)", "A", 0.01),
        ("41mOhm @ 32A, 8V", "Ohm", 0.041),
        ("3.3Ohm @ 50mA, 5V", "Ohm", 3.3),
        ("4,3V bei 20mA", "V", 4.3),
        ("203nC @ 20V", "C", 203e-9),
        ("2890pF @ 800V", "F", 2890e-12),
        ("9,8A (Ta), 112A (Tc)", "A", 9.8),
        ("350mΩ", "Ohm", 0.35),
        ("2µC", "C", 2e-6),
        ("56.8W (Tc)", "W", 56.8),
    ],
)
def test_quantity_units_and_conditions(raw: str, unit: str, expected: float) -> None:
    """Preserve decimals and prefixes that legacy regex stripping corrupted."""
    assert parse_quantity(raw, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw", [None, "-", "#VALUE!", "=A2*B2", "nan", float("inf"), 0, -1]
)
def test_missing_and_invalid_quantities(raw: object) -> None:
    """Never turn missing values, formula text or invalid data into ratings."""
    assert parse_quantity(raw, "V") is None


def test_independent_fom_expectations() -> None:
    """Check workbook cost formula and SI electrical figures independently."""
    m = metric_values(
        {
            "rds_ohm": 0.028,
            "qg_c": 203e-9,
            "coss_f": 213e-12,
            "voltage_v": 1200,
            "current_a": 103,
            "price": 15.04912,
        }
    )
    assert m["cost_fom"] == pytest.approx(2.926217777777778e-7)
    assert m["rds_qg"] == pytest.approx(5.684e-9)
    assert m["rds_coss"] == pytest.approx(5.964e-12)
    assert m["vi"] == 123600
    assert metric_values({"rds_ohm": 0.1})["rds_qg"] is None
    assert (
        metric_values({"price": 0, "rds_ohm": 0.1, "voltage_v": 650})["cost_fom"]
        is None
    )


@pytest.mark.parametrize(
    "xd,yd", [("min", "min"), ("min", "max"), ("max", "min"), ("max", "max")]
)
def test_frontier_matches_brute_force_with_ties(xd: str, yd: str) -> None:
    """Compare the sorted algorithm to an independent dominance definition."""
    rng = random.Random(42)
    points = [
        {"id": str(i), "x": rng.randint(1, 20), "y": rng.randint(1, 20)}
        for i in range(180)
    ]
    sx, sy = (1 if d == "min" else -1 for d in (xd, yd))
    expected = {
        p["id"]
        for p in points
        if not any(
            sx * q["x"] <= sx * p["x"]
            and sy * q["y"] <= sy * p["y"]
            and (q["x"] != p["x"] or q["y"] != p["y"])
            for q in points
        )
    }
    assert pareto_ids(points, xd, yd) == expected


def test_bundled_sources_and_real_workbook_cell() -> None:
    """Ensure all five requested files are portable and source-row preserving."""
    data = load_catalog()
    assert len(data["sources"]) == 8
    assert len(data["records"]) == 2490
    assert len({s["file"] for s in data["sources"]}) == 5
    assert len({r["id"] for r in data["records"]}) == len(data["records"])
    assert all(len(s["sha256"]) == 64 for s in data["sources"])
    row = next(r for r in data["records"] if r["id"] == "SiC_MOSFET_FOM/1200V:2")
    assert row["part_number"] == "NTHL020N120SC1"
    assert row["rds_ohm"] == 0.028
    assert row["quote_quantity"] == 10000
    assert row["raw"]["32"]["value"] == pytest.approx(metric_values(row)["cost_fom"])
    assert row["formulas"]["32"] == "=I2/(1200^2/U2)"


def test_manual_mosfet_and_igbt_conditions_are_distinct() -> None:
    """Do not substitute IGBT slope resistance or typical MOSFET resistance."""
    data = load_catalog()
    mos = next(r for r in data["records"] if r["id"] == "SemiconductorsData/MOSFET:2")
    igbt = next(r for r in data["records"] if r["id"] == "SemiconductorsData/IGBT:2")
    assert mos["rds_ohm"] == pytest.approx(0.0286)
    assert mos["qg_c"] == pytest.approx(178e-9)
    assert "25 °C" in mos["conditions"]["resistance"]
    assert igbt["rds_ohm"] is None
    assert metric_values(igbt)["rds_qg"] is None
    assert igbt["technology"] == "IGBT"


def test_filtering_exclusions_and_source_isolation() -> None:
    """Return exact exclusion counts and restrict the dominance population."""
    result = screen_catalog(
        load_catalog(),
        source_id="SiC_MOSFET_FOM/1200V",
        technology="SiC",
        voltage_min=1200,
        voltage_max=1200,
    )
    assert result["plotted_count"] > 50
    assert result["matched_count"] == result["plotted_count"] + result["excluded_count"]
    assert all(
        p["voltage_v"] == 1200 and p["technology"] == "SiC" for p in result["points"]
    )
    assert result["source"]["currency"] == "CHF"
    empty = screen_catalog(
        load_catalog(), source_id="SiC_MOSFET_FOM/1200V", search="not-a-part-number"
    )
    assert empty["points"] == []
    assert empty["pareto_count"] == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"x_metric": "invalid"},
        {"x_metric": "cost_fom"},
        {"voltage_min": math.inf},
        {"voltage_min": 1200, "voltage_max": 650},
        {"x_direction": "unknown"},
    ],
)
def test_invalid_screening_requests(settings: dict) -> None:
    """Reject invalid objectives and filter domains explicitly."""
    with pytest.raises(ValueError):
        screen_catalog(load_catalog(), source_id="SiC_MOSFET_FOM/1200V", **settings)


def test_api_matches_direct_calculation_and_reports_invalid_input() -> None:
    """Verify the real adapter without starting Qt or touching a database."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert len(client.get("/api/catalog/sources").json()["sources"]) == 8
    request = {
        "source_id": "201210_gan_digikey/CSV",
        "x_metric": "voltage_v",
        "y_metric": "current_a",
        "x_direction": "max",
        "y_direction": "max",
    }
    response = client.post("/api/catalog/screen", json=request)
    assert response.status_code == 200
    assert response.json() == screen_catalog(load_catalog(), **request)
    assert (
        client.post("/api/catalog/screen", json={"source_id": "unknown"}).status_code
        == 422
    )
    assert (
        client.post("/api/catalog/screen", json={**request, "arbitrary": 1}).status_code
        == 422
    )
