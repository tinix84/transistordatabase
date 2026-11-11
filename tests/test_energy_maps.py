"""Independent rectangular energy-map and bounded interpolation fixtures."""
from copy import deepcopy

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.energy_maps import build_energy_map
from transistordatabase.grid_analysis import AnalysisResourceLimitError
from transistordatabase.gui_web.backend.analysis_routes import EnergyMapRequest, router


def fixture():
    """Declare two independent energy-current curves with equal fixed conditions."""
    conditions = {"t_j_c": 25, "v_g_on_v": 15, "v_g_off_v": -5,
                  "r_g_on_ohm": 2, "r_g_off_ohm": 3, "commutation_device": None,
                  "commutation_source": None, "energy_convention": "Declared event integral"}
    return {"energy_kind": "on", "conditions": conditions,
            "slices": [{"id": f"v{v}", "source": f"Independent {v} V fixture", "source_kind": "unknown",
                        "energy_kind": "on", "voltage_v": v, "current_a": [0, 10],
                        "energy_j": [0, energy], "conditions": deepcopy(conditions)}
                       for v, energy in ((400, .001), (800, .002))],
            "currents_a": [0, 10], "voltages_v": [400, 800],
            "estimate": {"selected": True, "current_a": 5, "voltage_v": 600}}


def test_corners_and_bilinear_independent_fixture_preserve_source():
    """Verify exact corners, bounded interpolation, and unchanged sources."""
    request = fixture()
    before = deepcopy(request)
    result = build_energy_map(request)
    assert [[cell["energy_j"] for cell in row] for row in result["cells"]] == [[0, .001], [0, .002]]
    assert result["estimate"]["energy_j"] == pytest.approx(.00075)
    assert result["estimate"]["status"] == "estimated"
    assert all(cell["mode"] == "recorded-point" for row in result["cells"] for cell in row)
    assert all(c["candidates"][0]["source_kind"] == "unknown" for row in result["cells"] for c in row)
    assert request == before
    result["source_slices"][0]["energy_j"][0] = 99
    assert request == before


def test_current_linear_grid_missing_voltage_and_no_implicit_voltage_scaling():
    """Distinguish current interpolation from absent voltage slices."""
    request = fixture()
    request["currents_a"] = [0, 5, 10]
    request["voltages_v"] = [400, 600, 800]
    result = build_energy_map(request)
    assert result["cells"][0][1]["energy_j"] == .0005
    assert result["cells"][0][1]["mode"] == "bounded-current-linear"
    assert all(c["status"] == "missing" for c in result["cells"][1])
    assert result["estimate"]["status"] == "unsupported"


@pytest.mark.parametrize("mutation", ["outside", "missing-corner", "duplicate", "conditions", "unknown"])
def test_unsupported_estimate_and_visible_failures(mutation):
    """Keep missing coverage and conflicting sources visible."""
    request = fixture()
    if mutation == "outside":
        request["estimate"]["voltage_v"] = 900
    elif mutation == "missing-corner":
        request["slices"][1]["current_a"] = [1, 10]
    elif mutation == "duplicate":
        request["slices"].append(dict(request["slices"][0], id="conflict", energy_j=[.001, .004]))
    elif mutation == "conditions":
        request["slices"][1]["conditions"]["r_g_off_ohm"] = 99
    else:
        request["slices"][1]["conditions"]["t_j_c"] = None
    result = build_energy_map(request)
    assert result["estimate"]["status"] == "unsupported"
    assert result["estimate"]["energy_j"] is None
    if mutation == "duplicate":
        assert result["cells"][0][0]["status"] == "ambiguous"
        assert len(result["cells"][0][0]["candidates"]) == 2
    if mutation in {"conditions", "unknown"}:
        assert result["slice_diagnostics"][1]["reason"] == "fixed-condition-mismatch"


def test_finite_energy_zero_retained_kind_exclusion_and_known_unknown_exactness():
    """Retain finite zero-current energy without mixing conditions or kinds."""
    request = fixture()
    request["slices"][0]["energy_j"][0] = .0001
    assert build_energy_map(request)["cells"][0][0]["energy_j"] == .0001
    request["slices"][1]["energy_kind"] = "rr"
    result = build_energy_map(request)
    assert result["slice_diagnostics"][1]["reason"] == "energy-kind-mismatch"
    request = fixture()
    request["conditions"]["commutation_device"] = "Known diode"
    assert all(c["status"] == "missing" for row in build_energy_map(request)["cells"] for c in row)


@pytest.mark.parametrize("field,value", [("currents_a", [0, 0]), ("currents_a", [-1, 10]),
                                         ("currents_a", [0, float("nan")]),
                                         ("voltages_v", [0, 800]), ("voltages_v", [400, float("inf")])])
def test_invalid_grid_axes(field, value):
    """Reject invalid displayed current and voltage axes."""
    request = fixture()
    request[field] = value
    with pytest.raises(ValueError):
        build_energy_map(request)


@pytest.mark.parametrize("field,value", [("current_a", [0, 0]), ("energy_j", [0, -1]),
                                         ("energy_j", [0, float("inf")]), ("energy_j", [0])])
def test_invalid_source_arrays(field, value):
    """Reject invalid source samples before interpolation."""
    request = fixture()
    request["slices"][0][field] = value
    with pytest.raises(ValueError):
        build_energy_map(request)


def test_unsorted_source_pairs_preserve_original_order_and_units():
    """Sort only internal lookup samples while preserving original source order."""
    request = fixture()
    for source in request["slices"]:
        source["current_a"].reverse()
        source["energy_j"].reverse()
    result = build_energy_map(request)
    assert result["estimate"]["energy_j"] == pytest.approx(.00075)
    assert result["source_slices"][0]["current_a"] == [10, 0]
    assert result["units"] == {"current": "A", "voltage": "V", "energy": "J"}


def test_limits_checked_before_grid_allocation():
    """Reject excessive displayed axes before grid allocation."""
    request = fixture()
    request["currents_a"] = list(range(257))
    with pytest.raises(AnalysisResourceLimitError):
        build_energy_map(request)


def test_candidate_metadata_budget_rejects_before_interpolation(monkeypatch):
    """Bound repeated source identities before allocating candidate cells."""
    request = fixture()
    request["currents_a"] = list(range(256))
    request["slices"] = [dict(request["slices"][0], id=str(n), source="x" * 2000) for n in range(128)]
    def unexpected(*args):
        raise AssertionError("Interpolation must not start")
    monkeypatch.setattr("transistordatabase.energy_maps.MeasuredTable.evaluate", unexpected)
    with pytest.raises(AnalysisResourceLimitError, match="response"):
        build_energy_map(request)


def test_bilinear_display_grid_coarsening_is_explicit():
    """Disclose displayed-corner interpolation when a source has an interior knot."""
    request = fixture()
    request["slices"][0]["current_a"] = [0, 5, 10]
    request["slices"][0]["energy_j"] = [0, .0008, .001]
    result = build_energy_map(request)
    assert result["estimate"]["energy_j"] == pytest.approx(.00075)
    assert "Interior source knots" in result["assumptions"][-1]
    assert result["source_slices"][0]["energy_j"][1] == .0008


def test_typed_http_equality_explicit_unknown_conditions_and_resource_status():
    """Verify typed HTTP equality and explicit missing-condition validation."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = fixture()
    response = client.post("/api/analysis/energy-map", json=request)
    assert response.status_code == 200, response.text
    assert response.json() == build_energy_map(EnergyMapRequest.model_validate(request).model_dump())
    del request["conditions"]["commutation_source"]
    assert client.post("/api/analysis/energy-map", json=request).status_code == 422
    request = fixture()
    request["currents_a"] = list(range(257))
    assert client.post("/api/analysis/energy-map", json=request).status_code == 413
