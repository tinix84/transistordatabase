"""Numerical fixtures independent of UI interpolation and result rendering."""

from copy import deepcopy
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.integrate import quad

from transistordatabase.core.models import FosterThermalModel
from transistordatabase.gui_web.backend.thermal_routes import router
from transistordatabase.thermal import (
    _moments,
    analyze_thermal,
    electrothermal,
    screen_current,
    simulate_foster,
    steady_paths,
    thermal_impedance,
)


def transient():
    """Return the independently derived two-pole fixture."""
    return {
        "operation": "transient",
        "source": "Independent fixture",
        "conditions": {"gate_v": None},
        "network": {
            "r_th_vector": [0.2, 0.3],
            "tau_vector": [0.01, 0.1],
            "source": "Fixture poles",
            "conditions": {},
        },
        "network_heat_owner": "forward",
        "case_temperature_c": 25,
        "initial_state": "cold",
        "gap_policy": "reject",
        "event_mode": "none",
        "segments": [{"start_s": 0, "end_s": 0.1, "power_coefficients_w": [100]}],
        "samples_per_segment": 10,
        "events": [],
    }


def steady():
    """Return an explicit common-path fixture."""
    return dict(
        operation="steady",
        source="Fixture",
        conditions={},
        sink_temperature_c=25,
        forward_power_w=10,
        reverse_power_w=0,
        common_resistance_k_w=0.2,
        forward_resistance_k_w=1,
        reverse_resistance_k_w=2,
    )


def estimate(power=10, coefficient=0.01):
    """Declare a selected bounded linear estimate for test data."""
    return dict(
        mode="linear-temperature-estimate",
        selected=True,
        source="Selected fixture estimate",
        conditions={"gate_v": None},
        temperature_min_c=25,
        temperature_max_c=175,
        reference_temperature_c=25,
        reference_power_w=power,
        coefficient_per_k=coefficient,
    )


def electro():
    """Return the coupled fixed-point fixture."""
    return dict(
        operation="electrothermal",
        source="Fixture",
        conditions={},
        sink_temperature_c=25,
        common_resistance_k_w=0.2,
        branch_resistances_k_w=[1, 2],
        initial_temperatures_c=[25, 25],
        loss_models=[estimate(), estimate(5, 0.02)],
        tolerance_k=1e-10,
        relaxation=1.0,
        max_iterations=1000,
    )


@pytest.mark.parametrize("reverse,expected", [(False, (37, 27)), (True, (27, 47))])
def test_common_branch_paths(reverse, expected):
    """Keep common heat coupling separate from each branch's own heat."""
    request = steady()
    if reverse:
        request.update(forward_power_w=0, reverse_power_w=10)
    result = steady_paths(request)
    assert (
        result["forward_temperature_c"],
        result["reverse_temperature_c"],
    ) == expected


def test_interface_units_and_declared_zero():
    """Check area division and preserve missing-versus-zero distinction."""
    request = steady()
    request["interface"] = dict(
        path="forward",
        resistivity_k_m2_w=0.001,
        area_m2=0.01,
        source="Interface fixture",
        conditions={},
    )
    assert steady_paths(request)["forward_temperature_c"] == 38
    request["common_resistance_k_w"] = None
    with pytest.raises(ValueError):
        steady_paths(request)


def test_foster_step_adapter_and_complete_data():
    """Compare analytic steps and the corrected core adapter."""
    request = transient()
    z = thermal_impedance(request["network"], [0, 0.01, 0.1])
    np.testing.assert_allclose(
        25 + 100 * z, [25, 40.49728863549237, 63.962708766261485], rtol=1e-14
    )
    model = FosterThermalModel(r_th_vector=[0.2, 0.3], c_th_vector=[0.05, 1 / 3])
    assert model.get_thermal_impedance(0.1) == pytest.approx(z[-1])
    with pytest.raises(ValueError, match="R poles"):
        FosterThermalModel(r_th_total=0.5).get_thermal_impedance(0)


@pytest.mark.parametrize(
    "change",
    [
        dict(r_th_vector=[0]),
        dict(tau_vector=[0, 0.1]),
        dict(tau_vector=[0.1]),
        dict(r_th_total=0.6),
        dict(c_th_vector=[1, 1]),
        dict(r_th_vector=[1e-20, 1e-20], r_th_total=1e-20),
    ],
)
def test_invalid_network(change):
    """Reject incomplete poles and relatively inconsistent tiny totals."""
    model = transient()["network"]
    model.update(change)
    with pytest.raises(ValueError):
        thermal_impedance(model, 1)


@pytest.mark.parametrize("time", [-1, float("nan"), float("inf"), True])
def test_invalid_time(time):
    """Reject invalid scalar time instead of inventing a curve."""
    with pytest.raises(ValueError):
        thermal_impedance(transient()["network"], time)


@pytest.mark.parametrize("h", [1e-12, 1e-6, 0.01, 1, 10, 100])
def test_moments_independent_quadrature(h):
    """Compare convolution moments to adaptive scalar quadrature."""
    for k, actual in enumerate(_moments(np.longdouble(h))):
        expected = quad(
            lambda u, power=k: h * math.exp(-h * (1 - u)) * u**power,
            0,
            1,
            epsabs=1e-14,
            epsrel=1e-13,
        )[0]
        assert float(actual) == pytest.approx(expected, rel=2e-13, abs=1e-25)


def test_irregular_segments_and_event_conserve_energy():
    """Check closed-form irregular heat and exact impulse energy allocation."""
    request = transient()
    request["segments"] = [
        dict(start_s=0, end_s=0.01, power_coefficients_w=[100]),
        dict(start_s=0.01, end_s=0.037, power_coefficients_w=[0]),
        dict(start_s=0.037, end_s=0.1, power_coefficients_w=[50]),
    ]
    result = simulate_foster(request)
    np.testing.assert_allclose(
        result["final_pole_rises_k"], [9.983197149716196, 8.171829541971245], rtol=1e-14
    )
    assert result["final_temperature_c"] == pytest.approx(43.15502669168744, rel=1e-14)
    request["event_mode"] = "impulse"
    request["events"] = [
        dict(
            time_s=0.037,
            energy_j=0.1,
            heat_owner="forward",
            source="Event fixture",
            conditions={},
        )
    ]
    impulse = simulate_foster(request)
    assert impulse["final_temperature_c"] - result[
        "final_temperature_c"
    ] == pytest.approx(0.16345014985612696, rel=1e-13)
    assert impulse["conduction_energy_j"] == pytest.approx(4.15)
    assert impulse["switching_energy_j"] == 0.1
    assert [v["kind"] for v in impulse["samples"]].count("post-event") == 1
    request["event_mode"] = "interval-average"
    request["events"][0]["end_s"] = 0.038
    averaged = simulate_foster(request)
    assert averaged["switching_energy_j"] == 0.1
    assert averaged["final_temperature_c"] > result["final_temperature_c"]


def test_cubic_exact_nonnegative_and_interior_negative():
    """Accept physical negative coefficients and reject negative interior heat."""
    request = transient()
    request["network"].update(r_th_vector=[1], tau_vector=[1])
    request["segments"] = [
        dict(start_s=0, end_s=1, power_coefficients_w=[0, 0, 100, -100])
    ]
    result = simulate_foster(request)
    assert result["conduction_energy_j"] == pytest.approx(25 / 3)
    assert result["final_temperature_c"] - 25 == pytest.approx(
        100 * (3 - 8 / math.e), rel=1e-14
    )
    request["segments"][0]["power_coefficients_w"] = [0.1, -1, 1]
    with pytest.raises(ValueError, match="negative inside"):
        simulate_foster(request)


@pytest.mark.parametrize("ratio", [1e-300, 1e300])
def test_extreme_time_constant_ratios(ratio):
    """Keep extreme finite time ratios representable."""
    request = transient()
    request["network"].update(r_th_vector=[1], tau_vector=[ratio])
    result = simulate_foster(request)
    assert math.isfinite(result["final_temperature_c"])


def test_gap_and_initial_state_are_explicit():
    """Require gap conventions and a fully specified per-pole initial state."""
    request = transient()
    request["segments"] += [dict(start_s=0.2, end_s=0.3, power_coefficients_w=[0])]
    with pytest.raises(ValueError, match="gaps"):
        simulate_foster(request)
    request["gap_policy"] = "zero-power"
    request["initial_state"] = [1, 2]
    assert simulate_foster(request)["samples"][0]["temperature_c"] == 28
    request["initial_state"] = [3]
    with pytest.raises(ValueError, match="one rise"):
        simulate_foster(request)


def test_electrothermal_coupled_and_single_fixture():
    """Compare accepted solutions with independent linear algebra fixtures."""
    request = electro()
    result = electrothermal(request)
    assert result["status"] == "converged"
    np.testing.assert_allclose(
        result["accepted_solution"]["temperature_c"],
        [40.160349854227405, 42.05539358600583],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["accepted_solution"]["power_w"],
        [11.51603498542274, 6.705539358600583],
        atol=1e-11,
    )
    request.update(
        common_resistance_k_w=0,
        branch_resistances_k_w=[0.5],
        initial_temperatures_c=[25],
        loss_models=[estimate(10, 0.004)],
    )
    result = electrothermal(request)
    assert result["accepted_solution"]["temperature_c"][0] == pytest.approx(
        30.10204081632653, abs=1e-10
    )


def test_nonconvergence_domain_and_actual_residual():
    """Distinguish limits/domain exit and test the unrelaxed residual."""
    request = electro()
    request["max_iterations"] = 1
    result = electrothermal(request)
    assert result["status"] == "iteration-limit" and result["accepted_solution"] is None
    request.update(
        common_resistance_k_w=0,
        branch_resistances_k_w=[3],
        initial_temperatures_c=[25],
        loss_models=[estimate(10, 0.04)],
        max_iterations=100,
    )
    result = electrothermal(request)
    assert result["status"] == "source-domain-exit"
    assert result["history"][-1]["temperature_c"][0] == pytest.approx(186.04)
    request = electro()
    request.update(relaxation=1e-12, max_iterations=2)
    assert electrothermal(request)["status"] == "iteration-limit"
    request["loss_models"][0]["selected"] = False
    with pytest.raises(ValueError, match="explicitly"):
        electrothermal(request)


@pytest.mark.parametrize(
    "v0,slope,budget,expected",
    [(1, 0.1, 20, 10), (2, 0, 20, 10), (1, 0.1, 0, 0), (0, 0.1, 10, 10)],
)
def test_screening_root(v0, slope, budget, expected):
    """Check quadratic/zero-slope/zero-budget screening boundaries."""
    request = dict(
        junction_limit_c=25 + budget,
        reference_temperature_c=25,
        reference_kind="case",
        path_resistance_k_w=1,
        threshold_v=v0,
        resistance_ohm=slope,
    )
    assert screen_current(request)["thermal_screening_current_a"] == pytest.approx(
        expected
    )
    request.update(threshold_v=0, resistance_ohm=0)
    assert screen_current(request)["thermal_screening_current_a"] is None


def test_api_canonical_unknowns_and_errors():
    """Verify typed transport, canonical unknowns and finite/resource errors."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = transient()
    original = deepcopy(request)
    response = client.post("/api/thermal/analyze", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["request"]["conditions"]["gate_v"] is None
    assert request == original
    for operation in (steady(), electro()):
        assert client.post("/api/thermal/analyze", json=operation).status_code == 200
    assert (
        client.post(
            "/api/thermal/analyze",
            content='{"operation":"steady","source":"x","sink_temperature_c":NaN}',
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )
    request["samples_per_segment"] = 100
    request["segments"] = [
        dict(start_s=i, end_s=i + 1, power_coefficients_w=[1]) for i in range(1000)
    ]
    response = client.post("/api/thermal/analyze", json=request)
    assert response.status_code == 413, response.text


def test_metadata_nonfinite_rejected_and_no_input_alias():
    """Keep request provenance finite and return an independent canonical copy."""
    request = steady()
    result = analyze_thermal(request)
    result["request"]["conditions"]["new"] = 1
    assert request["conditions"] == {}
    request["conditions"]["unknown"] = float("nan")
    with pytest.raises(ValueError, match="metadata"):
        analyze_thermal(request)
