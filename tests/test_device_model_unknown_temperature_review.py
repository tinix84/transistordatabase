"""Check the difference between an unknown slice condition and a swept coordinate."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.device_model_fitting import train_device_model
from transistordatabase.gui_web.backend.model_fit_routes import router


def observation(identifier, coordinates, value, split="train"):
    """Identify one observed point and its explicit validation split."""
    return {
        "id": identifier,
        "source": "Analytical fixture",
        "series": "Fixed temperature",
        "split": split,
        "coordinates": coordinates,
        "value": value,
    }


def test_diode_null_temperature_stays_distinct_in_saved_model_and_predictions():
    """Keep an unknown fixed temperature distinct from every known temperature."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    rows = []
    for label, temperature, intercept in [("unknown", None, .5), ("zero", 0, .8)]:
        for current in [1, 3]:
            coords = {"current_a": current, "t_j_c": temperature, "v_g_v": None}
            voltage = intercept + .05 * current
            rows.append(observation(f"{label}-{current}", coords, voltage))
    held = {"current_a": 2, "t_j_c": None, "v_g_v": None}
    rows.append(observation("unknown-held", held, .61, "holdout"))
    rows.append(observation("unseen-held", held | {"t_j_c": 25}, .62, "holdout"))
    response = client.post(
        "/api/models/train", json={"family": "diode", "observations": rows}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    by_temperature = {
        s["conditions"]["t_j_c"]: s["id"] for s in result["model"]["slices"]
    }
    assert set(by_temperature) == {None, 0}
    forward = {
        d["slice"]: d for d in result["diagnostics"] if d["output"] == "voltage"
    }
    unknown_row = forward[by_temperature[None]]["holdout"]["rows"][0]
    assert unknown_row["residual"] == pytest.approx(.01)
    assert forward[by_temperature[0]]["holdout"] is None
    assert forward["unmatched"]["holdout"]["count"] == 0
    assert forward["unmatched"]["holdout"]["rows"][0]["id"] == "unseen-held"

    cases = [
        ("unknown", "voltage", held),
        ("zero", "voltage", held | {"t_j_c": 0}),
        ("inverse", "current", {"voltage_v": .6, "t_j_c": None, "v_g_v": None}),
        ("unseen", "voltage", held | {"t_j_c": 25}),
        ("missing", "voltage", {"current_a": 2, "v_g_v": None}),
        ("gate-zero", "voltage", held | {"v_g_v": 0}),
    ]
    queries = [
        {"id": key, "output": output, "coordinates": coords}
        for key, output, coords in cases
    ]
    response = client.post(
        "/api/models/evaluate", json={"model": result["model"], "queries": queries}
    )
    assert response.status_code == 200, response.text
    predictions = response.json()["result"]["predictions"]
    assert [item["value"] for item in predictions[:3]] == pytest.approx([.6, .9, 2])
    for item in predictions[3:]:
        assert item["status"] == "rejected" and item["value"] is None and item["reason"]


@pytest.mark.parametrize("family", ["ron", "energy"])
def test_swept_temperature_requires_a_number(family):
    """Require known numerical temperature on each fitted temperature axis."""
    coords = {"t_j_c": None}
    request = {"family": family, "fixed_conditions": {"v_g_v": None}}
    if family == "ron":
        request["fixed_conditions"]["current_a"] = None
    else:
        anchor = {"current_a": 20, "voltage_v": 400, "t_j_c": 25, "r_g_ohm": 4}
        request.update(anchor=anchor, event="e_on")
        coords = anchor | coords
    request["observations"] = [
        observation("unknown", coords, .01),
        observation("known", coords | {"t_j_c": 75}, .015),
    ]
    with pytest.raises(ValueError, match="t_j_c"):
        train_device_model(request)
