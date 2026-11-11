"""Independent numerical oracles and untrusted-model transport boundaries."""
from copy import deepcopy
import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.device_model_fitting import (
    evaluate_device_model, train_device_model, validate_device_model,
)
from transistordatabase.gui_web.backend.model_fit_routes import router


def row(identifier, coordinates, value, split="train"):
    """Identify one measurement without hiding its split assignment."""
    return dict(id=identifier, source="Measured fixture", series="A", split=split,
                coordinates=coordinates, value=value)


def ron():
    """Use the independent resistance-temperature fixture."""
    return dict(family="ron", fixed_conditions={"v_g_v": 12, "current_a": None},
                observations=[row("r25", {"t_j_c": 25}, .010),
                              row("r75", {"t_j_c": 75}, .0125),
                              row("r125", {"t_j_c": 125}, .015),
                              row("held", {"t_j_c": 50}, .01150, "holdout")])


def diode():
    """Use a known linear positive-magnitude diode slice."""
    return dict(family="diode", observations=[row(
        f"d{i}", {"current_a": i, "t_j_c": 25, "v_g_v": 0}, v)
        for i, v in ((1, .55), (3, .65), (5, .75))])


def energy(event="e_on", multiplier=1):
    """Use isolated sweeps with a shared real nominal observation."""
    anchor = dict(current_a=20, voltage_v=400, t_j_c=25, r_g_ohm=5)
    data = [("i10", {"current_a": 10}, .001),
            ("nominal", {}, .002), ("i30", {"current_a": 30}, .003),
            ("v200", {"voltage_v": 200}, .001),
            ("v600", {"voltage_v": 600}, .003),
            ("t125", {"t_j_c": 125}, .003),
            ("r2", {"r_g_ohm": 2}, .0014),
            ("r8", {"r_g_ohm": 8}, .0026)]
    observations = [row(key, anchor | changes, value * multiplier)
                    for key, changes, value in data]
    observations.append(row("held", dict(current_a=15, voltage_v=500,
                                         t_j_c=75, r_g_ohm=6),
                            .002678125 * multiplier, "holdout"))
    return dict(family="energy", event=event, anchor=anchor,
                fixed_conditions={"v_g_v": 12}, observations=observations)


def query(model, output, coordinates):
    """Return one bounded headless prediction."""
    return evaluate_device_model(dict(model=model, queries=[dict(
        id="q", output=output, coordinates=coordinates)]))["result"]["predictions"][0]


def client():
    """Use the router without database or application fixtures."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_ron_exact_fit_and_true_holdout():
    """Calculate held-out residuals from observations excluded from fitting."""
    result = train_device_model(ron())
    diagnostic = result["diagnostics"][0]
    assert diagnostic["train"]["rmse"] == pytest.approx(0, abs=1e-17)
    assert diagnostic["holdout"]["rows"][0]["predicted"] == pytest.approx(.01125)
    assert diagnostic["holdout"]["rows"][0]["residual"] == pytest.approx(.00025)
    assert diagnostic["holdout"]["rmse"] == pytest.approx(.00025)
    assert diagnostic["holdout"]["mae"] == pytest.approx(.00025)
    assert diagnostic["holdout"]["r2"] is None
    assert result["model"]["slices"][0]["polynomials"]["resistance"][
        "residual_degrees_of_freedom"] == 1
    assert len(result["plots"][0]["samples"]) == 61
    prediction = query(result["model"], "resistance",
                       dict(t_j_c=50, v_g_v=12, current_a=None))
    assert prediction["value"] == pytest.approx(.01125)


def test_holdout_changes_do_not_change_model_or_fit_curve():
    """Keep fit state independent of held-out coordinates and values."""
    first = train_device_model(ron())
    changed = ron()
    changed["observations"][-1]["value"] = 200
    changed["observations"][-1]["coordinates"]["t_j_c"] = -100
    second = train_device_model(changed)
    assert first["model"] == second["model"]
    assert first["plots"][0]["samples"] == second["plots"][0]["samples"]
    assert first["evidence"]["model_sha256"] == second["evidence"]["model_sha256"]
    assert first["evidence"]["dataset_sha256"] != second["evidence"]["dataset_sha256"]
    held = second["diagnostics"][0]["holdout"]
    assert held["count"] == 0 and held["rmse"] is None
    assert held["rows"][0]["status"] == "rejected"
    assert held["rows"][0]["predicted"] is None


def test_diode_forward_and_independent_inverse_and_exact_conditions():
    """Fit each direction and reject unmatched physical slice conditions."""
    model = train_device_model(diode())["model"]
    assert query(model, "voltage", dict(current_a=2, t_j_c=25, v_g_v=0))[
        "value"] == pytest.approx(.60)
    assert query(model, "current", dict(voltage_v=.70, t_j_c=25, v_g_v=0))[
        "value"] == pytest.approx(4)
    for coordinates in (dict(current_a=2, t_j_c=25, v_g_v=None),
                        dict(current_a=2, t_j_c=25.0000000001, v_g_v=0),
                        dict(current_a=0, t_j_c=25, v_g_v=0)):
        result = query(model, "voltage", coordinates)
        assert result["status"] == "rejected" and result["value"] is None


def test_diode_separate_slices_and_unmatched_holdout_not_silently_dropped():
    """Retain unknown gates and explicitly reject untrained temperatures."""
    inputs = diode()
    inputs["observations"] += [row("warm" + str(i),
        dict(current_a=i, t_j_c=75, v_g_v=None), v)
        for i, v in ((1, .4), (3, .5), (5, .6))]
    inputs["observations"].append(row("unknownslice", dict(
        current_a=2, t_j_c=50, v_g_v=0), .6, "holdout"))
    result = train_device_model(inputs)
    assert len(result["model"]["slices"]) == 2
    assert query(result["model"], "voltage", dict(current_a=2, t_j_c=75,
                                               v_g_v=None))["value"] == pytest.approx(.45)
    unmatched = [item for item in result["diagnostics"] if item["slice"] == "unmatched"]
    assert len(unmatched) == 2
    assert all(item["holdout"]["rows"][0]["status"] == "rejected" for item in unmatched)


def test_energy_separable_oracle_and_shared_nominal_row():
    """Match the independent four-axis energy product and residual oracle."""
    result = train_device_model(energy())
    coords = dict(current_a=15, voltage_v=500, t_j_c=75, r_g_ohm=6, v_g_v=12)
    assert query(result["model"], "energy", coords)["value"] == pytest.approx(.002578125)
    held = result["diagnostics"][0]["holdout"]
    assert held["rows"][0]["residual"] == pytest.approx(.0001)
    for axis, poly in result["model"]["slices"][0]["polynomials"].items():
        assert "nominal" in poly["training_ids"]
        assert poly["normalization"] is None if axis == "current_a" else (
            poly["normalization"] == pytest.approx(.002))
    assert len(result["plots"]) == 4
    assert result["plots"][1]["observations"][1]["y"] == pytest.approx(.5)


def test_energy_events_are_independent_and_no_zero_voltage_anchor():
    """Keep turn-on and turn-off fits independent with positive anchors."""
    on = train_device_model(energy())["model"]
    off = train_device_model(energy("e_off", 2))["model"]
    coords = dict(current_a=15, voltage_v=500, t_j_c=75, r_g_ohm=6, v_g_v=12)
    assert query(off, "energy", coords)["value"] == pytest.approx(
        2 * query(on, "energy", coords)["value"])
    assert on["event"] == "e_on" and off["event"] == "e_off"
    invalid = energy()
    invalid["anchor"]["voltage_v"] = 0
    with pytest.raises(ValueError, match="nominal current and voltage"):
        train_device_model(invalid)


def test_energy_noisy_sweep_normalizes_its_fitted_anchor_not_baseline_anchor():
    """Normalize every fitted factor by its own fitted nominal energy."""
    inputs = energy()
    # Voltage sweep linear least squares no longer passes through the shared anchor.
    next(item for item in inputs["observations"] if item["id"] == "v600")["value"] = .0036
    model = train_device_model(inputs)["model"]
    anchor = inputs["anchor"] | {"v_g_v": 12}
    assert query(model, "energy", anchor)["value"] == pytest.approx(.002)
    poly = model["slices"][0]["polynomials"]["voltage_v"]
    assert poly["normalization"] == pytest.approx(.0022)


def test_energy_missing_axes_fixed_and_mixed_training_rejected():
    """Fix absent sweeps at nominal and reject mixed training corners."""
    inputs = energy()
    inputs["observations"] = inputs["observations"][:3]
    model = train_device_model(inputs)["model"]
    coords = inputs["anchor"] | {"current_a": 15, "v_g_v": 12}
    assert query(model, "energy", coords)["value"] == pytest.approx(.0015)
    assert query(model, "energy", coords | {"voltage_v": 500})["status"] == "rejected"
    inputs = energy()
    inputs["observations"][-1]["split"] = "train"
    with pytest.raises(ValueError, match="isolated"):
        train_device_model(inputs)


def test_physical_domain_checks_interior_extrema_without_forcing_ron_monotone():
    """Accept a positive resistance minimum and reject an interior negative value."""
    inputs = ron() | {"degree": 2}
    inputs["observations"] = [row(str(x), {"t_j_c": x}, .01 + .002 * x*x)
                              for x in (-1, 0, 1)]
    model = train_device_model(inputs)["model"]
    assert query(model, "resistance", dict(t_j_c=0, v_g_v=12,
                                          current_a=None))["value"] == pytest.approx(.01)
    for item in inputs["observations"]:
        x = item["coordinates"]["t_j_c"]
        item["value"] = (x - .5) ** 2 - .1
    assert all(item["value"] > 0 for item in inputs["observations"])
    with pytest.raises(ValueError, match="inside its support"):
        train_device_model(inputs)


def test_diode_interior_slope_check_and_signed_magnitudes():
    """Reject hidden negative slopes and signed directional currents."""
    inputs = diode() | {"degree": 3}
    inputs["observations"] = [row(str(x), dict(current_a=x + 1,
        t_j_c=25, v_g_v=0), 3 + x ** 3 - .1*x) for x in (-1, -.5, .5, 1)]
    with pytest.raises(ValueError, match="slope"):
        train_device_model(inputs)
    invalid = diode()
    invalid["observations"][0]["coordinates"]["current_a"] = -1
    with pytest.raises(ValueError, match="magnitudes"):
        train_device_model(invalid)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(degree=True),
    lambda r: r.update(degree=4),
    lambda r: r["observations"][0].update(value=True),
    lambda r: r["observations"][0].update(value=".01"),
    lambda r: r["observations"][0].update(value=float("nan")),
    lambda r: r["observations"][0].update(source=" "),
    lambda r: r["observations"][0].update(source="x" * 257),
    lambda r: r["observations"][0].update(split="validation"),
    lambda r: r["observations"][1].update(id="r25"),
    lambda r: r["observations"][0]["coordinates"].update(extra=1),
    lambda r: r["fixed_conditions"].pop("current_a"),
    lambda r: r.update(observations=r["observations"][-1:]),
])
def test_strict_training_boundaries(mutation):
    """Reject malformed observations, conditions and polynomial orders."""
    inputs = ron()
    mutation(inputs)
    with pytest.raises(ValueError):
        train_device_model(inputs)


def test_distinct_coordinate_and_condition_guard():
    """Require enough distinct points and a numerically usable polynomial basis."""
    inputs = ron() | {"degree": 3}
    with pytest.raises(ValueError, match="distinct"):
        train_device_model(inputs)
    # Distinct coordinates exist, but nearly coincident knots make cubic rank unsafe.
    inputs["observations"] = [row(str(i), {"t_j_c": x}, .01 + .001*x)
                              for i, x in enumerate((0, 1e-14, 2e-14, 1))]
    with pytest.raises(ValueError, match="rank deficient|condition"):
        train_device_model(inputs)


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(schema_version=2),
    lambda m: m.update(schema_version=True),
    lambda m: m.update(method="pickle"),
    lambda m: m.update(path="/tmp/model"),
    lambda m: m["slices"][0]["polynomials"]["resistance"]["coefficients"].append(0),
    lambda m: m["slices"][0]["polynomials"]["resistance"]["coefficients"].__setitem__(0, -1),
    lambda m: m["slices"][0]["polynomials"]["resistance"].update(support=[-100, 1000]),
    lambda m: m["slices"][0]["polynomials"]["resistance"].update(scale=0),
    lambda m: m["slices"][0]["polynomials"]["resistance"].update(rank=0),
    lambda m: m["slices"][0]["polynomials"]["resistance"].update(condition_number=1e20),
    lambda m: m["training_observations"][0].update(split="holdout"),
])
def test_untrusted_model_rejected(mutation):
    """Validate all imported model state against its declared training data."""
    model = train_device_model(ron())["model"]
    mutation(model)
    with pytest.raises(ValueError):
        validate_device_model(model)


def test_no_mutation_model_json_roundtrip_and_fingerprints():
    """Preserve caller data and bind the exact saved JSON text to its hash."""
    inputs = ron()
    original = deepcopy(inputs)
    trained = train_device_model(inputs)
    assert inputs == original
    model = json.loads(json.dumps(trained["model"]))
    assert validate_device_model(model) == trained["model"]
    expected = hashlib.sha256(json.dumps(model, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert trained["evidence"]["model_sha256"] == expected
    assert "held" not in [item["id"] for item in model["training_observations"]]
    assert hashlib.sha256(trained["model_json"].encode()).hexdigest() == expected


def test_headless_integral_json_numbers_and_response_isolation():
    """Accept equivalent JSON numbers and isolate all public response data."""
    trained = train_device_model(ron())
    def integral_numbers(value):
        if isinstance(value, dict):
            return {key: integral_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [integral_numbers(item) for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    converted = integral_numbers(trained["model"])
    assert validate_device_model(converted) == trained["model"]
    trained["units"]["energy"] = "wrong"
    trained["inputs"]["fixed_conditions"]["v_g_v"] = 99
    trained["plots"][0]["conditions"]["v_g_v"] = 88
    assert trained["model"]["fixed_conditions"]["v_g_v"] == 12
    assert train_device_model(ron())["units"]["energy"] == "J"


def test_single_off_nominal_sweep_observation_is_not_discarded():
    """Reject a supplied sweep with too few distinct training coordinates."""
    inputs = energy()
    # The voltage sweep has one measured off-anchor point and no real shared anchor.
    inputs["observations"] = [item for item in inputs["observations"]
                              if item["id"] not in ("nominal", "v600")]
    with pytest.raises(ValueError, match="distinct"):
        train_device_model(inputs)


def test_adjacent_float_temperature_support_uses_actual_transformed_domain():
    """Check physical limits on the coordinates actually used by the fit."""
    import math
    inputs = ron()
    lo, hi = 25.0, math.nextafter(25.0, math.inf)
    inputs["observations"] = [row("lo", {"t_j_c": lo}, .001),
                              row("hi", {"t_j_c": hi}, .004)]
    model = train_device_model(inputs)["model"]
    assert query(model, "resistance", dict(t_j_c=hi, v_g_v=12,
                                          current_a=None))["value"] == pytest.approx(.004)


def test_constant_outputs_r2_and_absent_holdout():
    """Represent undefined R2 and absent validation data as null."""
    inputs = ron()
    inputs["observations"] = inputs["observations"][:3]
    for item in inputs["observations"]:
        item["value"] = .01
    result = train_device_model(inputs)
    assert result["diagnostics"][0]["train"]["r2"] is None
    assert result["diagnostics"][0]["holdout"] is None


def test_api_train_evaluate_raw_bytes_and_invalid_model():
    """Round-trip the typed model and hash the exact HTTP request bytes."""
    app = client()
    raw = json.dumps(energy(), indent=2).encode()
    response = app.post("/api/models/train", content=raw,
                        headers={"content-type": "application/json"})
    assert response.status_code == 200, response.text
    trained = response.json()
    assert trained["evidence"]["request_sha256"] == hashlib.sha256(raw).hexdigest()
    request = dict(model=trained["model"], queries=[dict(id="q", output="energy",
        coordinates=dict(current_a=15, voltage_v=500, t_j_c=75,
                         r_g_ohm=6, v_g_v=12))])
    evaluated = app.post("/api/models/evaluate", json=request)
    assert evaluated.status_code == 200, evaluated.text
    assert evaluated.json()["result"]["predictions"][0]["value"] == pytest.approx(.002578125)
    request["model"]["slices"][0]["polynomials"]["current_a"]["coefficients"][0] = -1
    assert app.post("/api/models/evaluate", json=request).status_code == 422


@pytest.mark.parametrize("literal", ["true", '"0.01"', "NaN", "Infinity", "1e999", "1e-999"])
def test_api_numeric_literals_rejected(literal):
    """Reject booleans, strings, nonfinite and underflowing numeric literals."""
    raw = json.dumps(ron()).replace('"value": 0.01', '"value": ' + literal, 1)
    response = client().post("/api/models/train", content=raw,
                             headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_api_request_actual_byte_bound_and_extra_keys():
    """Apply the streamed body limit and reject undeclared side effects."""
    app = client()
    assert app.post("/api/models/train", content=b" " * (4 * 1024 * 1024 + 1),
                    headers={"content-type": "application/json"}).status_code == 413
    assert app.post("/api/models/train", json=ron() | {"save": True}).status_code == 422


def test_api_ron_unknown_current_and_gate_condition_zero():
    """Match an unknown fixed current only with an explicitly unknown query."""
    app = client()
    trained = app.post("/api/models/train", json=ron())
    assert trained.status_code == 200, trained.text
    response = app.post("/api/models/evaluate", json=dict(model=trained.json()["model"],
        queries=[dict(id="unknown", output="resistance", coordinates=dict(
            t_j_c=50, current_a=None, v_g_v=12)),
            dict(id="zero", output="resistance", coordinates=dict(
                t_j_c=50, current_a=0, v_g_v=12))]))
    assert response.status_code == 200, response.text
    assert [r["status"] for r in response.json()["result"]["predictions"]] == [
        "predicted", "rejected"]
