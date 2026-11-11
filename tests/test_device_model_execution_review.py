"""Independent checks of fitted functions, residuals, plots, and saved model replay."""

from copy import deepcopy
from hashlib import sha256
import json
from math import inf, nextafter, sqrt

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.device_model_fitting import (
    evaluate_device_model, train_device_model, validate_device_model,
)
from transistordatabase.gui_web.backend.model_fit_routes import router


def observation(key, coordinates, value, split="train"):
    """Identify each observation and its source series."""
    return dict(id=key, source="Independent fixture µ", series="S2", split=split,
                coordinates=coordinates, value=value)


def predict(model, output, coordinates):
    """Evaluate all supplied conditions in one call."""
    queries = [dict(id=str(i), output=output, coordinates=c)
               for i, c in enumerate(coordinates)]
    return evaluate_device_model({"model": model, "queries": queries})[
        "result"]["predictions"]


def test_quadratic_separable_energy_with_two_missing_factors():
    """Check the energy product, fixed axes, and held-out row accounting."""
    anchor = dict(current_a=8, voltage_v=300, t_j_c=60, r_g_ohm=4)
    fixed = {"v_g_v": None}

    def energy(i, v):
        return (1 + i * i / 64) * 1e-3 * (1 + ((v - 300) / 200) ** 2)

    rows = [
        observation(str((i, v)), anchor | dict(current_a=i, voltage_v=v), energy(i, v))
        for i, v in [(4, 300), (8, 300), (12, 300), (8, 100), (8, 500)]
    ]
    coordinates = anchor | dict(current_a=6, voltage_v=400)
    request = dict(family="energy", event="e_off", degree=2, anchor=anchor,
                   fixed_conditions=fixed, observations=rows)
    fit = train_device_model(request)
    assert fit["diagnostics"][0]["holdout"] is None
    query = coordinates | fixed
    points = predict(fit["model"], "energy",
                     [query, query | {"t_j_c": 61}, query | {"v_g_v": 0}])
    assert points[0]["value"] == pytest.approx(0.001953125)
    assert all(p["status"] == "rejected" and p["reason"] for p in points[1:])
    held = observation("mixed", coordinates, .002, "holdout")
    outside = observation("outside", coordinates | {"r_g_ohm": 6}, .003, "holdout")
    augmented = train_device_model(request | {"observations": [outside, *rows, held]})
    assert augmented["model_json"] == fit["model_json"]
    diagnostics = augmented["diagnostics"][0]
    assert diagnostics["train"]["count"] == 5
    assert diagnostics["holdout"]["count"] == 1
    assert [p["id"] for p in diagnostics["holdout"]["rows"]] == ["outside", "mixed"]
    assert diagnostics["holdout"]["mae"] == pytest.approx(.002 - .001953125)
    assert diagnostics["holdout"]["r2"] is None
    assert augmented["evidence"]["dataset_sha256"] != fit["evidence"]["dataset_sha256"]


def test_cubic_resistance_and_independent_holdout_statistics():
    """Compare each plotted resistance and held-out statistic to a known cubic."""
    def resistance(x):
        return .01 + .00001 * (x - 20) ** 2 + .00000005 * (x - 20) ** 3

    rows = [observation(f"train-{x}", {"t_j_c": x}, resistance(x))
            for x in (0, 10, 30, 40)]
    targets = [resistance(5) + .001, resistance(25) - .002]
    rows.extend(observation(f"held-{x}", {"t_j_c": x}, y, "holdout")
                for x, y in zip([5, 25], targets, strict=True))
    fit = train_device_model(dict(family="ron", degree=3,
                                  fixed_conditions={"current_a": None, "v_g_v": -4},
                                  observations=rows))
    error = fit["diagnostics"][0]["holdout"]
    assert error["mae"] == pytest.approx(.0015)
    assert error["rmse"] == pytest.approx(sqrt(.000005 / 2))
    assert error["r2"] == pytest.approx(1 - .000005 / ((targets[0] - targets[1])**2 / 2))
    assert [r["residual"] for r in error["rows"]] == pytest.approx([.001, -.002])
    plot = fit["plots"][0]["samples"]
    assert len(plot) == 61
    assert [p["y"] for p in plot] == pytest.approx([resistance(p["x"]) for p in plot])
    assert sha256(fit["model_json"].encode()).hexdigest() == fit["evidence"]["model_sha256"]


@pytest.mark.parametrize("lower", [-100., .25, 75.])
def test_adjacent_plot_samples_stay_on_representable_support(lower):
    """Use the two representable temperatures without extending the fitted domain."""
    upper = nextafter(lower, inf)
    fixed = dict(v_g_v=None, current_a=None)
    fit = train_device_model(dict(family="ron", fixed_conditions=fixed,
                                  observations=[observation("low", {"t_j_c": lower}, .2),
                                                observation("high", {"t_j_c": upper}, .1)]))
    samples = fit["plots"][0]["samples"]
    assert len(samples) == 61 and {p["x"] for p in samples} == {lower, upper}
    assert [p["y"] for p in samples] == pytest.approx(
        [.2 if p["x"] == lower else .1 for p in samples])
    rejected = predict(fit["model"], "resistance",
                       [fixed | {"t_j_c": nextafter(x, d)}
                        for x, d in [(lower, -inf), (upper, inf)]])
    assert all(p["status"] == "rejected" and p["value"] is None for p in rejected)


def test_api_model_replay_preserves_integer_distinctions_before_float_conversion():
    """Reject a changed support endpoint that rounds to the original float."""
    fixed = dict(v_g_v=None, current_a=None)
    large = 2**53
    fit = train_device_model(dict(family="ron", fixed_conditions=fixed,
                                  observations=[observation("low", {"t_j_c": large}, .1),
                                                observation("high", {"t_j_c": large + 4}, .2)]))
    query = dict(id="a", output="resistance", coordinates=fixed | {"t_j_c": large})
    model = deepcopy(fit["model"])
    model["slices"][0]["polynomials"]["resistance"]["support"][0] = large + 1
    with pytest.raises(ValueError):
        validate_device_model(model)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        for candidate, expected in [(json.loads(fit["model_json"]), 200), (model, 422)]:
            raw = json.dumps(dict(model=candidate, queries=[query]),
                             ensure_ascii=False, indent=1).encode()
            response = client.post("/api/models/evaluate", content=raw,
                                   headers={"content-type": "application/json"})
            assert response.status_code == expected
            if expected == 200:
                assert response.json()["evidence"]["request_sha256"] == sha256(raw).hexdigest()


@pytest.mark.parametrize(("literal", "expected"), [
    ("25.0000000000000000", 200), ("2.5e1", 200), ("25.0000000000000001", 422),
])
def test_api_support_uses_the_submitted_decimal_value(literal, expected):
    """Accept equivalent decimal spellings and reject a changed temperature endpoint."""
    fixed = dict(v_g_v=0, current_a=10)
    fit = train_device_model(dict(family="ron", fixed_conditions=fixed,
                                  observations=[observation("low", {"t_j_c": 25}, .01),
                                                observation("high", {"t_j_c": 75}, .02)]))
    model = deepcopy(fit["model"])
    model["slices"][0]["polynomials"]["resistance"]["support"][0] = "replace-this-token"
    query = dict(id="q", output="resistance", coordinates=fixed | {"t_j_c": 45})
    raw = json.dumps(dict(model=model, queries=[query])).replace('"replace-this-token"', literal)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/api/models/evaluate", content=raw,
                               headers={"content-type": "application/json"})
    assert response.status_code == expected
