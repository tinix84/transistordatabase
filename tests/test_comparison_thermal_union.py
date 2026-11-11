"""Keep measured evidence and validated Foster models in comparison results."""
from copy import deepcopy
import json

import numpy as np
import pytest

from transistordatabase.backend.concrete_services import ComparisonService
from transistordatabase.core.models import (
    ElectricalRatings,
    FosterThermalModel,
    ThermalProperties,
    Transistor,
    TransistorMetadata,
)


def device(name):
    """Create an independent device without inferred thermal parameters."""
    return Transistor(
        metadata=TransistorMetadata(name, "IGBT", "test", "fixture", "fixture"),
        electrical=ElectricalRatings(1200, 100, 50, 175),
        thermal=ThermalProperties(0.001, 0.001),
    )


def thermal_panel(*devices):
    """Exercise the public comparison result with explicit missing loss inputs."""
    return ComparisonService().generate_advanced_comparison(
        list(devices), {}
    )["thermal_impedance"]


def test_rc_only_foster_matches_independent_two_pole_response():
    """Calculate tau as R times C without a recorded tau vector."""
    source = device("RC source")
    measured = np.array([[0.002, 0.02, 0.2], [0.01, 0.12, 0.35]])
    source.switch.thermal_foster = FosterThermalModel(
        r_th_vector=[0.2, 0.3],
        c_th_vector=[0.5, 2.0],
        r_th_total=0.5,
        graph_t_rthjc=measured.copy(),
    )
    before = deepcopy(source.switch.thermal_foster)

    panel = thermal_panel(source, device("No thermal evidence"))
    curves = {curve["name"]: curve for curve in panel["curves"]}
    model = curves["RC source / switch - Foster model"]
    times = np.asarray(model["time_data"])
    expected = 0.2 * -np.expm1(-times / 0.1)
    expected += 0.3 * -np.expm1(-times / 0.6)

    np.testing.assert_allclose(model["z_th_data"], expected, rtol=1e-13)
    assert len(times) > 1 and np.all(np.diff(times) > 0)
    assert curves["RC source / switch - measured"]["time_data"] == measured[0].tolist()
    assert curves["RC source / switch - measured"]["z_th_data"] == measured[1].tolist()
    assert source.switch.thermal_foster.tau_vector is None
    assert source.switch.thermal_foster.r_th_vector == before.r_th_vector
    assert source.switch.thermal_foster.c_th_vector == before.c_th_vector
    np.testing.assert_array_equal(source.switch.thermal_foster.graph_t_rthjc, measured)
    json.dumps(panel, allow_nan=False)


@pytest.mark.parametrize(
    "model",
    [
        {"r_th_vector": [0.2], "tau_vector": [0.1], "r_th_total": 0.8},
        {"r_th_vector": [0.2], "tau_vector": [0.1], "c_th_vector": [2.0]},
    ],
    ids=["total-resistance-mismatch", "tau-and-RC-mismatch"],
)
def test_invalid_foster_keeps_measurement_other_component_and_other_device(model):
    """A rejected model reports its component without aborting the comparison."""
    source = device("Rejected model")
    measured = np.array([[0.002, 0.02], [0.01, 0.12]])
    source.switch.thermal_foster = FosterThermalModel(
        **deepcopy(model), graph_t_rthjc=measured.copy()
    )
    source.diode.thermal_foster = FosterThermalModel(
        r_th_vector=[0.4], tau_vector=[0.03]
    )
    other = device("Valid model")
    other.switch.thermal_foster = FosterThermalModel(
        r_th_vector=[0.3], c_th_vector=[0.2]
    )

    panel = thermal_panel(source, other)
    curves = {curve["name"]: curve for curve in panel["curves"]}

    assert "Rejected model / switch - Foster model" not in curves
    assert "Rejected model / diode - Foster model" in curves
    assert "Valid model / switch - Foster model" in curves
    recorded_curve = curves["Rejected model / switch - measured"]
    assert recorded_curve["time_data"] == measured[0].tolist()
    assert recorded_curve["z_th_data"] == measured[1].tolist()
    assert any(
        item["name"] == "Rejected model"
        and item["component"] == "switch"
        and item["status"] == "unsupported"
        and item["reason"]
        for item in panel["diagnostics"]
    )
    json.dumps(panel, allow_nan=False)


@pytest.mark.parametrize(
    "measured",
    [
        [[0.001, 0.01], [0.1, float("nan")]],
        [[0.001, float("inf")], [0.1, 0.2]],
        [[-0.001, 0.01], [0.1, 0.2]],
        [[0.01, 0.001], [0.1, 0.2]],
        [[0.001, 0.001], [0.1, 0.2]],
        [[0.001, 0.01], [-0.1, 0.2]],
        [["0.001", "0.01"], ["0.1", "0.2"]],
    ],
    ids=[
        "nonfinite-impedance", "nonfinite-time", "negative-time",
        "descending-time", "duplicate-time", "negative-impedance", "text-samples",
    ],
)
def test_invalid_measurement_keeps_valid_models_and_other_component(measured):
    """Invalid measured samples cannot contaminate JSON or hide valid evidence."""
    source = device("Invalid measurement")
    recorded = np.asarray(measured)
    before = recorded.copy()
    source.switch.thermal_foster = FosterThermalModel(
        r_th_vector=[0.2], tau_vector=[0.1], graph_t_rthjc=recorded
    )
    diode_measurement = np.array([[0.003, 0.03], [0.02, 0.21]])
    source.diode.thermal_foster = FosterThermalModel(
        graph_t_rthjc=diode_measurement.copy()
    )
    other = device("Other device")
    other.switch.thermal_foster = FosterThermalModel(
        r_th_vector=[0.4], tau_vector=[0.2]
    )

    panel = thermal_panel(source, other)
    curves = {curve["name"]: curve for curve in panel["curves"]}

    assert "Invalid measurement / switch - measured" not in curves
    assert "Invalid measurement / switch - Foster model" in curves
    assert "Other device / switch - Foster model" in curves
    recorded_curve = curves["Invalid measurement / diode - measured"]
    assert recorded_curve["time_data"] == diode_measurement[0].tolist()
    assert recorded_curve["z_th_data"] == diode_measurement[1].tolist()
    assert any(
        item["name"] == "Invalid measurement"
        and item["component"] == "switch"
        and item["status"] == "invalid-data"
        and item["reason"]
        for item in panel["diagnostics"]
    )
    np.testing.assert_array_equal(recorded, before)
    json.dumps(panel, allow_nan=False)