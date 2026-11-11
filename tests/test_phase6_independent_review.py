"""Independent phase-6 review against quadrature and direct linear algebra."""
from copy import deepcopy
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.integrate import quad

import transistordatabase.thermal as thermal_module
from transistordatabase.gui_web.backend.assembly_routes import router as assembly_router
from transistordatabase.gui_web.backend.thermal_routes import router as thermal_router
from transistordatabase.semiconductor_assemblies import (
    parallel_bank, reverse_envelope, signed_composite,
)
from transistordatabase.thermal import (
    analyze_thermal, electrothermal, simulate_foster, thermal_impedance,
)


def context():
    """Declare fixture provenance and an explicitly unknown gate bias."""
    return {"source": "independent review fixture", "conditions": {"gate_v": None}}


def transient():
    """Build a cold two-pole fixed-case reference request."""
    return {**context(), "operation": "transient", "network": {
        **context(), "r_th_vector": [.2, .3], "tau_vector": [.01, .1]},
        "network_heat_owner": "forward", "case_temperature_c": 25.,
        "initial_state": "cold", "gap_policy": "reject", "event_mode": "none",
        "segments": [{"start_s": 0., "end_s": .1,
                      "power_coefficients_w": [100.]}],
        "events": [], "samples_per_segment": 7}


def table(x, y, kind="channel"):
    """Build a measured magnitude table without synthetic conditions."""
    return {"kind": kind, "x": list(x), "y": list(y), **context()}


def test_random_cubic_transients_against_physical_quadrature():
    """Compare physical convolution integrals with the production solver."""
    rng = np.random.default_rng(2026101606)
    for _ in range(40):
        r = np.exp(rng.uniform(-3, 1, 3))
        tau = np.exp(rng.uniform(-5, 2, 3))
        bounds = np.r_[0., np.cumsum(rng.uniform(.001, .2, 4))]
        # Square of affine function times nonnegative affine duty: positive cubic.
        coefficients = [np.polynomial.polynomial.polymul(
            np.polynomial.polynomial.polymul([a, b], [a, b]), [d, e])
            for a, b, d, e in rng.uniform(.01, 5, (4, 4))]
        initial = rng.uniform(0, 2, 3)
        request = transient()
        request.update(initial_state=initial.tolist())
        request["network"].update(r_th_vector=r.tolist(), tau_vector=tau.tolist())
        request["segments"] = [dict(start_s=float(a), end_s=float(b),
                                    power_coefficients_w=c.tolist())
                               for a, b, c in zip(bounds[:-1], bounds[1:], coefficients, strict=True)]
        actual = simulate_foster(request)
        sparse_request = deepcopy(request)
        sparse_request["samples_per_segment"] = 1
        np.testing.assert_allclose(
            simulate_foster(sparse_request)["final_pole_rises_k"],
            actual["final_pole_rises_k"], rtol=2e-13, atol=0,
        )
        final = bounds[-1]
        expected = []
        for resistance, time_constant, state0 in zip(r, tau, initial, strict=True):
            state = state0*math.exp(-final/time_constant)
            for a, b, c in zip(bounds[:-1], bounds[1:], coefficients, strict=True):
                state += quad(_convolution, a, b,
                              args=(resistance, time_constant, final, a, b, c),
                              epsabs=1e-11, epsrel=2e-12)[0]
            expected.append(state)
        np.testing.assert_allclose(actual["final_pole_rises_k"], expected,
                                   rtol=3e-12, atol=1e-12)
        expected_energy = sum((b-a)*sum(c[k]/(k+1) for k in range(4))
                              for a, b, c in zip(bounds[:-1], bounds[1:], coefficients, strict=True))
        assert actual["conduction_energy_j"] == pytest.approx(expected_energy, rel=2e-14)


def _convolution(t, resistance, time_constant, final, a, b, coefficients):
    return resistance/time_constant*math.exp(-(final-t)/time_constant)*np.polynomial.polynomial.polyval((t-a)/(b-a), coefficients)


@pytest.mark.parametrize("mode", ["impulse", "interval-average"])
def test_event_deposition_closed_form_and_duplicate_boundary(mode):
    """Deposit each duplicate-time event exactly once and conserve energy."""
    request = transient()
    request.update(event_mode=mode)
    request["segments"] = [dict(start_s=0., end_s=.037, power_coefficients_w=[0]),
                           dict(start_s=.037, end_s=.1, power_coefficients_w=[0])]
    events = [dict(time_s=t, energy_j=e, end_s=t+.007,
                   heat_owner="forward", **context())
              for t, e in [(0., .2), (.037, .1), (.037, .3), (.09, .04)]]
    request["events"] = events
    actual = simulate_foster(request)
    expected = []
    for resistance, tau in zip([.2, .3], [.01, .1], strict=True):
        if mode == "impulse":
            expected.append(sum(resistance*e["energy_j"]/tau * math.exp(
                -(.1-e["time_s"])/tau) for e in events))
        else:
            expected.append(sum(resistance*e["energy_j"]/.007 * (
                math.exp(-(.1-e["end_s"])/tau)-math.exp(-(.1-e["time_s"])/tau))
                for e in events))
    np.testing.assert_allclose(actual["final_pole_rises_k"], expected, rtol=1e-13)
    assert actual["switching_energy_j"] == pytest.approx(.64)
    if mode == "impulse":
        assert sum(s["kind"] == "post-event" for s in actual["samples"]) == 3


def test_event_at_final_time_and_explicit_branch_owner():
    """Retain end-time impulse heat and reject a different physical junction."""
    request = transient()
    request.update(event_mode="impulse")
    request["events"] = [dict(time_s=.1, energy_j=.1,
                              heat_owner="forward", **context())]
    baseline = simulate_foster({**request, "events": []})
    result = simulate_foster(request)
    assert result["final_temperature_c"]-baseline["final_temperature_c"] == pytest.approx(2.3)
    request["events"][0]["heat_owner"] = "reverse"
    with pytest.raises(ValueError, match="owner"):
        simulate_foster(request)
    request["network_heat_owner"] = "shared-junction"
    assert simulate_foster(request)["final_temperature_c"] == result["final_temperature_c"]


def test_random_electrothermal_against_direct_matrix_solution():
    """Check converged nonlinear iterations against independent linear solves."""
    rng = np.random.default_rng(16062026)
    for _ in range(25):
        common = rng.uniform(.01, .2)
        branches = rng.uniform(.1, 1., 2)
        powers = rng.uniform(1., 10., 2)
        alpha = rng.uniform(0., .01, 2)
        matrix = np.full((2, 2), common)+np.diag(branches)
        rise = np.linalg.solve(np.eye(2)-matrix@np.diag(powers*alpha), matrix@powers)
        request = {**context(), "operation": "electrothermal", "sink_temperature_c": 25.,
                   "common_resistance_k_w": common, "branch_resistances_k_w": branches.tolist(),
                   "initial_temperatures_c": [25., 25.], "max_iterations": 500,
                   "relaxation": .3, "tolerance_k": 1e-11,
                   "loss_models": [{**context(), "mode": "linear-temperature-estimate",
                       "selected": True, "temperature_min_c": 25., "temperature_max_c": 175.,
                       "reference_temperature_c": 25., "reference_power_w": p,
                       "coefficient_per_k": a} for p, a in zip(powers, alpha, strict=True)]}
        result = electrothermal(request)
        assert result["status"] == "converged"
        np.testing.assert_allclose(result["accepted_solution"]["temperature_c"],
                                   25+rise, atol=2e-11, rtol=0)
        assert max(abs(v) for v in result["accepted_solution"]["residual_k"]) <= 1e-11


def test_parallel_curve_evaluation_and_count_one_no_alias():
    """Verify the physical bank identities throughout the recorded domains."""
    source = {**context(), "conduction": table([0, 2, 10], [1., 1.4, 2.]),
              "turn_on": table([0, 3, 10], [.001, .003, .008], "energy"),
              "internal_resistance_k_per_w": 1., "shared_sink_resistance_k_per_w": .2}
    # Device context has no top-level conditions field.
    source.pop("conditions")
    snapshot = deepcopy(source)
    result = parallel_bank(source, 7)["device"]
    for current in np.linspace(0, 70, 101):
        assert np.interp(current, result["conduction"]["x"], result["conduction"]["y"]) == pytest.approx(
            np.interp(current/7, source["conduction"]["x"], source["conduction"]["y"]))
        assert np.interp(current, result["turn_on"]["x"], result["turn_on"]["y"]) == pytest.approx(
            7*np.interp(current/7, source["turn_on"]["x"], source["turn_on"]["y"]))
    result["turn_on"]["conditions"]["gate_v"] = 10
    assert source == snapshot


def test_random_reverse_envelopes_against_pointwise_minimum():
    """Compare arbitrary crossing-rich envelopes at 30030 signed currents."""
    rng = np.random.default_rng(60616)
    for _ in range(30):
        x1 = np.r_[0., np.sort(rng.uniform(0, 10, 8)), 10.]
        x2 = np.r_[1., np.sort(rng.uniform(1, 12, 6)), 12.]
        y1, y2 = rng.uniform(0, 3, 10), rng.uniform(0, 3, 8)
        result = reverse_envelope(table(x1, y1), table(x2, y2, "diode"))
        query = np.linspace(-10, -1, 1001)
        actual = np.interp(query, result["current_a"], result["voltage_v"])
        expected = -np.minimum(np.interp(-query, x1, y1), np.interp(-query, x2, y2))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_core_and_api_canonical_equality_and_safe_validation():
    """Check transport values, immutability, unknowns and raw nonfinite errors."""
    app = FastAPI()
    app.include_router(thermal_router)
    app.include_router(assembly_router)
    client = TestClient(app)
    request = transient()
    original = deepcopy(request)
    response = client.post("/api/thermal/analyze", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["result"] == analyze_thermal(request)["result"]
    assert request == original
    for invalid in ("NaN", "Infinity", "-Infinity"):
        raw = '{"operation":"parallel-bank","count":1,"device":{"source":"x","cost":'+invalid+'}}'
        response = client.post("/api/assemblies/analyze", content=raw,
                               headers={"content-type": "application/json"})
        assert response.status_code == 422


def test_foster_tiny_ratios_and_nonzero_initial_state():
    """Preserve representable small responses and exact cold-free decay."""
    assert thermal_impedance({"r_th_vector": [1.], "tau_vector": [1e300]}, 1.) == pytest.approx(1e-300, rel=1e-14, abs=0)
    request = transient()
    request["network"].update(r_th_vector=[1.], tau_vector=[1.])
    request.update(initial_state=[3.])
    request["segments"] = [dict(start_s=0., end_s=1., power_coefficients_w=[0.])]
    assert simulate_foster(request)["final_pole_rises_k"][0] == pytest.approx(3/math.e, rel=1e-14)


def test_composite_event_conditions_and_missing_rating_are_independent():
    """Require the voltage estimate without promoting missing blocking data."""
    a = {"source": "positive", "blocking_voltage_v": None,
         "turn_on": table([0, 10], [.001, .002], "energy")}
    b = {"source": "negative", "blocking_voltage_v": 1200,
         "turn_on": table([0, 10], [.002, .004], "energy")}
    a["turn_on"]["conditions"]["v_supply_v"] = 400.
    b["turn_on"]["conditions"]["v_supply_v"] = 800.
    with pytest.raises(ValueError, match="conditions"):
        signed_composite(a, b, 0.)
    result = signed_composite(a, b, 0., {"selected": True, "target_voltage_v": 200.})
    assert result["summary"]["blocking_voltage_v"] is None
    assert result["branches"]["positive"]["turn_on"]["y"] == [.0005, .001]
    assert result["branches"]["negative"]["turn_on"]["y"] == [.0005, .001]


def test_zero_and_negative_polynomial_with_float64_longdouble(monkeypatch):
    """Model Windows' float64 longdouble without accepting infinite tolerance."""
    class Float64Numpy:
        longdouble = np.float64

        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setattr(thermal_module, "np", Float64Numpy())
    np.testing.assert_array_equal(thermal_module._polynomial([0, 0, 0, 0]), [0, 0, 0, 0])
    with pytest.raises(ValueError, match="negative"):
        thermal_module._polynomial([1e308, -1e308, -1e308, 1e307])


def test_float64_ratio_underflow_does_not_fabricate_zero_heat(monkeypatch):
    """Reject unsupported ratios or preserve their representable final response."""
    class Float64Numpy:
        longdouble = np.float64

        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setattr(thermal_module, "np", Float64Numpy())
    model = {"r_th_vector": [1e308], "tau_vector": [1e308]}
    try:
        impedance = thermal_module.thermal_impedance(model, 1e-308)
    except ValueError as exc:
        assert "represent" in str(exc).lower() or "underflow" in str(exc).lower()
    else:
        assert impedance == pytest.approx(1e-308, rel=1e-14, abs=0)
    request = transient()
    request["network"].update(model)
    request["segments"] = [dict(start_s=0., end_s=1e-308, power_coefficients_w=[1e308])]
    request["samples_per_segment"] = 1
    try:
        result = thermal_module.simulate_foster(request)
    except ValueError as exc:
        assert "represent" in str(exc).lower() or "underflow" in str(exc).lower()
    else:
        assert result["final_temperature_c"] == pytest.approx(26., rel=1e-14)


def test_large_ratio_cancellation_does_not_fabricate_zero_heat():
    """Retain or explicitly reject a representable high-h linear-ramp response."""
    request = transient()
    request["network"].update(r_th_vector=[1e308], tau_vector=[1e-308])
    request.update(case_temperature_c=0.)
    request["segments"] = [dict(start_s=0., end_s=1.,
                                power_coefficients_w=[1., -1.])]
    request["samples_per_segment"] = 1
    try:
        result = thermal_module.simulate_foster(request)
    except ValueError as exc:
        assert any(term in str(exc).lower() for term in ("represent", "precision", "ratio", "overflow"))
    else:
        assert result["final_temperature_c"] == pytest.approx(1., rel=1e-14)


@pytest.mark.parametrize("ratio", [2., 10., 1e3, 1e8, 1e12, 1e20, 1e300])
def test_linear_end_response_conditioning_across_large_ratios(ratio):
    """Use a closed form that never subtracts two almost-equal B moments."""
    request = transient()
    request["network"].update(r_th_vector=[ratio], tau_vector=[1/ratio])
    request.update(case_temperature_c=0.)
    request["segments"] = [dict(start_s=0., end_s=1., power_coefficients_w=[1., -1.])]
    request["samples_per_segment"] = 1
    expected = 1-(ratio+1)*math.exp(-ratio)
    try:
        result = thermal_module.simulate_foster(request)
    except ValueError as exc:
        assert ratio >= 1e8
        assert any(term in str(exc).lower() for term in ("represent", "precision", "ratio", "overflow"))
    else:
        assert result["final_temperature_c"] == pytest.approx(expected, rel=2e-12)


@pytest.mark.parametrize("samples", [1, 7, 100])
def test_cubic_endpoint_cancellation_and_sampling_independence(samples):
    """P=(1-u)^3 has a nonzero 6 K endpoint despite three cancelling moments."""
    request = transient()
    request["network"].update(r_th_vector=[1e18], tau_vector=[1e-6])
    request.update(case_temperature_c=0., samples_per_segment=samples)
    request["segments"] = [dict(start_s=0., end_s=1.,
                                power_coefficients_w=[1., -3., 3., -1.])]
    result = thermal_module.simulate_foster(request)
    assert result["final_temperature_c"] == pytest.approx(6., rel=2e-12)
    assert result["conduction_energy_j"] == .25


@pytest.mark.parametrize("time,energy", [(.01, 0.), (.01, .1), (.9, 0.), (.9, .1),
                                      (.99999, 0.), (.999999, 0.), (.9999999, 0.)])
def test_cubic_endpoint_remains_correct_after_event_cut(time, energy):
    """Physical cuts must preserve the original cubic; impulse has decayed away."""
    request = transient()
    request["network"].update(r_th_vector=[1e18], tau_vector=[1e-6])
    request.update(case_temperature_c=0., samples_per_segment=1, event_mode="impulse")
    request["segments"] = [dict(start_s=0., end_s=1., power_coefficients_w=[1., -3., 3., -1.])]
    request["events"] = [dict(time_s=time, energy_j=energy,
                              heat_owner="forward", **context())]
    result = thermal_module.simulate_foster(request)
    assert result["final_temperature_c"] == pytest.approx(6., rel=2e-12)
