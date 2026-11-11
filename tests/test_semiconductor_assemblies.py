"""Assembly arithmetic and transport tests using independently derived fixtures."""
from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.gui_web.backend.assembly_routes import router
from transistordatabase.semiconductor_assemblies import (
    analyze_assembly,
    parallel_bank,
    reverse_envelope,
    signed_composite,
)


def table(kind="channel", x=None, y=None, voltage=None):
    """Construct a condition-specific independent test slice."""
    conditions = {"t_j_c": 25, "v_g_v": 15}
    if kind == "energy":
        conditions.update(v_supply_v=voltage or 400, r_g_ohm=None)
    return dict(kind=kind, x=x or [0, 10], y=y or [0, 2],
                conditions=conditions, source="Independent recorded fixture")


def device():
    """Return one fully specified transform fixture."""
    return dict(source="Fixture device", conduction=table(),
                turn_on=table("energy", y=[.0001, .001]),
                turn_off=table("energy", y=[.0002, .002], voltage=600),
                blocking_voltage_v=1200, current_rating_a=10, area_m2=.01,
                cost=2, currency="CHF", weight_kg=.02,
                internal_resistance_k_per_w=1, shared_sink_resistance_k_per_w=.2,
                foster=dict(resistance_k_per_w=[.2, .3], tau_s=[.01, .1],
                            capacity_j_per_k=[.05, 1 / 3]))


def test_parallel_scaling_and_immutable_sources():
    """Scale physical quantities while retaining external paths and originals."""
    source = device()
    snapshot = deepcopy(source)
    result = parallel_bank(source, 3)["device"]
    assert result["current_rating_a"] == 30
    assert result["turn_on"]["x"] == [0, 30]
    assert result["turn_on"]["y"] == pytest.approx([.0003, .003])
    assert result["turn_off"]["conditions"]["v_supply_v"] == 600
    assert result["conduction"]["y"] == [0, 2]
    assert result["blocking_voltage_v"] == 1200
    assert result["area_m2"] == .03 and result["cost"] == 6
    assert result["internal_resistance_k_per_w"] == pytest.approx(1 / 3)
    assert result["shared_sink_resistance_k_per_w"] == .2
    assert result["foster"]["resistance_k_per_w"] == pytest.approx([.2 / 3, .1])
    assert result["foster"]["tau_s"] == [.01, .1]
    assert result["foster"]["capacity_j_per_k"] == pytest.approx([.15, 1])
    result["conduction"]["x"][0] = 99
    assert source == snapshot


def test_count_one_is_independent_and_unknowns_preserved():
    """Identity transformations copy their source and preserve missing values."""
    source = device()
    source["cost"] = None
    result = parallel_bank(source, 1)["device"]
    assert result == source
    assert result is not source and result["turn_on"] is not source["turn_on"]
    assert result["turn_on"]["conditions"] is not source["turn_on"]["conditions"]
    assert result["cost"] is None


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.5, "3", 10**400])
def test_count_rejection(count):
    """Reject Boolean, fractional, nonpositive or nonrepresentable counts."""
    with pytest.raises(ValueError):
        parallel_bank(device(), count)


@pytest.mark.parametrize("field", ["cost", "area_m2", "current_rating_a", "weight_kg"])
def test_scaling_overflow(field):
    """Finite input must not yield nonfinite transformed values."""
    source = device()
    source[field] = 1e308
    with pytest.raises(ValueError, match="overflow"):
        parallel_bank(source, 3)


def test_signed_branches_preserve_zero_discontinuity_and_conditions():
    """Keep both zero-current branches and incomplete totals distinct."""
    positive, negative = device(), device()
    negative["conduction"] = table("diode", y=[.8, 1])
    negative["cost"] = None
    result = signed_composite(positive, negative, .2)
    assert result["branches"]["negative"]["conduction"]["y"] == [.8, 1]
    assert result["summary"]["cost"] is None
    assert result["summary"]["known_cost_sum"] == 2
    assert result["summary"]["weight_kg"] == .04
    assert result["summary"]["blocking_voltage_v"] == 1200
    assert result["common_resistance_k_per_w"] == .2
    assert result["branches"]["positive"]["turn_off"]["conditions"]["v_supply_v"] == 600


@pytest.mark.parametrize("currencies", [("CHF", "EUR"), (None, None), (None, "CHF")])
def test_incompatible_cost_units_never_summed(currencies):
    """Costs require the same known currency before summing."""
    a, b = device(), device()
    a["currency"], b["currency"] = currencies
    result = signed_composite(a, b, 0)
    assert result["summary"]["cost"] is None
    assert result["summary"]["known_cost_sum"] is None


def test_voltage_estimate_separate_event_references_and_source_integrity():
    """Scale each event using its own recorded voltage only when selected."""
    a, b = device(), device()
    b["turn_on"]["conditions"]["v_supply_v"] = 800
    b["turn_off"]["conditions"]["v_supply_v"] = 1200
    snapshot = deepcopy((a, b))
    with pytest.raises(ValueError, match="conditions"):
        signed_composite(a, b, .2)
    result = signed_composite(a, b, .2, dict(selected=True, target_voltage_v=200))
    assert [v["factor"] for v in result["estimates"]] == [.5, .25, 1 / 3, 1 / 6]
    assert result["branches"]["positive"]["turn_on"]["y"] == pytest.approx([.00005, .0005])
    assert result["estimates"][2]["recorded_conditions"]["v_supply_v"] == 600
    assert (a, b) == snapshot
    b["turn_off"]["conditions"]["t_j_c"] = 26
    with pytest.raises(ValueError, match="conditions"):
        signed_composite(a, b, .2, dict(selected=True, target_voltage_v=200))


def test_unknown_condition_is_not_known_zero():
    """Unknown gate resistance must not match a known ideal zero."""
    a, b = device(), device()
    b["turn_off"]["conditions"]["r_g_ohm"] = 0
    with pytest.raises(ValueError, match="conditions"):
        signed_composite(a, b, 0)


def test_exact_reverse_crossing_and_overlap():
    """Check the exact 40/9 crossing and bounded negative-current envelope."""
    channel = table(y=[0, 2])
    diode = table("diode", x=[2, 4, 8, 12], y=[.84, .88, .96, 1.04])
    diode["conditions"]["v_g_v"] = 0
    snapshot = deepcopy((channel, diode))
    result = reverse_envelope(channel, diode)
    assert result["crossings_magnitude_a"] == pytest.approx([40 / 9])
    assert result["overlap_magnitude_a"] == [2, 10]
    assert result["current_a"] == pytest.approx([-10, -8, -40 / 9, -4, -2])
    assert result["voltage_v"] == pytest.approx([-1, -.96, -8 / 9, -.8, -.4])
    assert (channel, diode) == snapshot
    for query in np.linspace(-10, -2, 1001):
        expected = -min(.2 * abs(query), .8 + .02 * abs(query))
        assert np.interp(query, result["current_a"], result["voltage_v"]) == pytest.approx(expected)


def test_multiple_piecewise_crossings_retains_knots():
    """Keep every measured knot and every crossing between linear pieces."""
    result = reverse_envelope(table(x=[0, 2, 4, 6], y=[0, 4, 0, 4]),
                              table("diode", x=[0, 6], y=[2, 2]))
    assert result["crossings_magnitude_a"] == [1, 3, 5]
    assert result["current_a"] == [-6, -5, -4, -3, -2, -1, 0]


@pytest.mark.parametrize("edit", ["nonfinite", "negative", "no-overlap", "temperature"])
def test_reverse_invalid_sources(edit):
    """Reject bad samples, disjoint domains and mismatched temperatures."""
    channel, diode = table(), table("diode")
    if edit == "nonfinite":
        diode["y"][1] = float("nan")
    elif edit == "negative":
        diode["y"][0] = -1
    elif edit == "no-overlap":
        diode["x"] = [20, 30]
    else:
        diode["conditions"]["t_j_c"] = None
    with pytest.raises(ValueError):
        reverse_envelope(channel, diode)


@pytest.fixture
def client():
    """Use a small headless router app with no desktop imports."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_http_canonical_and_headless_match(client):
    """HTTP uses the same pure transformation and preserves input provenance."""
    request = dict(operation="parallel-bank", count=3, device=device())
    result = client.post("/api/assemblies/analyze", json=request)
    assert result.status_code == 200
    assert result.json()["device"] == analyze_assembly(request)["device"]
    assert result.json()["request"]["device"]["cost"] == 2


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_raw_nonfinite_safe_422(client, bad):
    """Nonstandard JSON numeric tokens still return JSON-safe validation errors."""
    response = client.post("/api/assemblies/analyze", content='{"operation":"parallel-bank","count":3,"device":{"source":"x","cost":' + bad + '}}', headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("payload", [
    dict(operation="parallel-bank", count=True, device=device()),
    dict(operation="parallel-bank", count=2, device={**device(), "source": " "}),
    dict(operation="reverse-envelope", selected=False, channel=table(), diode=table("diode")),
    dict(operation="signed-composite", positive=device(), negative=device()),
])
def test_http_rejects_unsupported_request(client, payload):
    """Typed transport rejects missing choices, sources and common paths."""
    assert client.post("/api/assemblies/analyze", json=payload).status_code == 422


def test_nonrepresentable_interpolation_rejected_in_python_and_http(client):
    """Extreme finite slopes must not leak infinity into a JSON response."""
    payload = dict(operation="reverse-envelope", selected=True,
                   channel=table(x=[0, 1e-320], y=[0, 1e308]),
                   diode=table("diode", x=[0, 5e-321, 1e-320], y=[1, 1, 1]))
    with pytest.raises(ValueError, match="representable"):
        analyze_assembly(payload)
    assert client.post("/api/assemblies/analyze", json=payload).status_code == 422


def test_headless_curve_budget_matches_transport(client):
    """Headless callers cannot bypass the bounded recorded-curve input budget."""
    payload = dict(operation="parallel-bank", count=1, device=device())
    payload["device"]["conduction"]["x"] = list(range(10001))
    payload["device"]["conduction"]["y"] = [1] * 10001
    with pytest.raises(ValueError, match="10000"):
        analyze_assembly(payload)
    assert client.post("/api/assemblies/analyze", json=payload).status_code == 422


def test_positive_internal_resistance_underflow_rejected():
    """A tiny known positive resistance must not turn into a declared ideal zero."""
    source = device()
    source["internal_resistance_k_per_w"] = 5e-324
    with pytest.raises(ValueError, match="underflow"):
        parallel_bank(source, 3)


def test_nonrepresentable_voltage_factor_rejected():
    """A positive voltage ratio that rounds to zero cannot erase event heat."""
    with pytest.raises(ValueError, match="factor"):
        signed_composite(device(), device(), 0,
                         dict(selected=True, target_voltage_v=5e-324))


def test_foster_validation_uses_complete_relative_consistency():
    """The shared pole validator rejects mismatched tiny R/C/tau data."""
    source = device()
    source["foster"] = dict(resistance_k_per_w=[1e-100], tau_s=[1e-200],
                            capacity_j_per_k=[2e-100])
    with pytest.raises(ValueError, match="disagree"):
        parallel_bank(source, 1)
