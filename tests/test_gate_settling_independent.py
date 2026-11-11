"""Gate settling checks from the series RC equation and pinned legacy HTTP output."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math

from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase.analytical_models import GateChargeModel, GateChargeModelParams
from transistordatabase.gui_web.backend.main import app


URL = "/api/analytical/gate_charge"
PARAMETERS = {"q_gs": 4e-9, "q_gd": 2e-9, "q_g": 12e-9, "v_th": 2, "v_plateau": 4,
              "r_g": 1.5, "r_g_int": 0.5, "v_driver": 10, "v_off": 0}


def public(**changes):
    response = TestClient(app).post(URL, json=PARAMETERS | changes)
    assert response.status_code == 200, response.text
    return response.json()


def test_default_response_is_identical_to_merged_application():
    # Captured from the merged e41cc8a application at its normal HTTP entry point.
    response = public()
    canonical = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    assert sha256(canonical).hexdigest() == "1981f6f3718173cdb751bc81ad3372318acbc5fe9968cd9917531f295f702745"
    assert "turn_on_settling" not in response["estimate"]
    assert "settling_target_v" not in response["parameters"]


def test_rc_charge_timing_and_gate_current_match_continuous_solution():
    legacy = public()
    target = public(settling_target_v=7)
    for key in ("turn_on_times", "turn_off_times"):
        assert target[key] == legacy[key]
    for key in ("turn_on", "turn_off"):
        assert target["estimate"][key] == legacy["estimate"][key]
    settling = target["estimate"]["turn_on_settling"]
    # Cpre = 1 nF. Miller current = 3 A. The voltage plateau starts after RC ln(10/6).
    anchor = 2e-9 * math.log(10 / 6) + 2e-9 / 3
    tau = 2e-9
    duration = tau * math.log(2)
    assert settling["status"] == "computed"
    assert math.isclose(settling["anchor_s"], anchor, rel_tol=1e-13)
    assert math.isclose(settling["duration_s"], duration, rel_tol=1e-13)
    assert math.isclose(settling["absolute_end_s"], anchor + duration, rel_tol=1e-13)
    assert math.isclose(settling["post_plateau_capacitance_f"], 1e-9, rel_tol=1e-13)
    assert math.isclose(settling["residual_charge_c"], 6e-9, rel_tol=1e-13)
    assert math.isclose(settling["incremental_charge_c"], 3e-9, rel_tol=1e-13)
    assert settling["initial_gate_current_a"] == 3
    assert settling["final_gate_current_a"] == 1.5
    times = settling["relative_time_s"]
    assert times[0] == 0 and math.isclose(times[-1], duration, rel_tol=1e-13)
    for t, v_gate, i_gate in zip(times, settling["v_gate_v"], settling["i_gate_a"], strict=True):
        assert math.isclose(v_gate, 10 - 6 * math.exp(-t / tau), rel_tol=1e-13)
        assert math.isclose(i_gate, 3 * math.exp(-t / tau), rel_tol=1e-13)
    assert not any("energy_j" in key or "drain_current" in key for key in settling)


@pytest.mark.parametrize("target", [4, 10, 3, 11, True, "7"])
def test_target_must_be_strictly_between_plateau_and_driver_rail(target):
    response = TestClient(app).post(URL, json=PARAMETERS | {"settling_target_v": target})
    assert response.status_code == 422, response.text


def test_next_float_below_rail_produces_finite_positive_current():
    target = math.nextafter(10.0, 4.0)
    settling = public(settling_target_v=target)["estimate"]["turn_on_settling"]
    # At the end the remaining driver-to-gate voltage is one representable step.
    gap = 10 - target
    expected = 2e-9 * (math.log(6) - math.log(gap))
    assert math.isclose(settling["duration_s"], expected, rel_tol=1e-13)
    assert settling["final_gate_current_a"] == gap / 2
    assert settling["v_gate_v"][-1] == target


def test_finite_si_settling_duration_survives_nanosecond_display_overflow():
    plateau = 1.0
    target = math.nextafter(plateau, 2.0)
    rail = math.nextafter(target, 2.0)
    response = public(q_gs=1e-9, q_gd=0, q_g=4.440892098500626e283, v_th=0.5,
                      v_plateau=plateau, v_driver=rail, r_g=100, r_g_int=0,
                      settling_target_v=target)
    settling = response["estimate"]["turn_on_settling"]
    expected = 1e301 * math.log(2)
    assert math.isfinite(settling["duration_s"])
    assert math.isclose(settling["duration_s"], expected, rel_tol=1e-13)
    assert settling["duration_s"] > 1e300
    assert settling["relative_time_s"][-1] == settling["duration_s"]


def test_positive_local_interval_survives_unresolvable_absolute_tick():
    target = math.nextafter(4.0, 10.0)
    settling = public(q_gs=4e-6, q_gd=2e-6, q_g=6.006e-6, settling_target_v=target)["estimate"]["turn_on_settling"]
    dt = settling["duration_s"]
    # When deltaV is one ULP at 4 V, Q=C*deltaV, and the current stays near 3 A.
    assert dt > 0 and math.isclose(dt, settling["incremental_charge_c"] / 3, rel_tol=1e-13)
    assert settling["anchor_s"] + dt == settling["anchor_s"]
    assert settling["status"] == "computed"
    assert settling["absolute_end_s"] is None
    assert settling["absolute_time_status"] == "unresolved-use-relative-time"
    assert settling["relative_time_s"][0] == 0 and settling["relative_time_s"][-1] == dt
    assert settling["v_gate_v"][-1] == target


def test_zero_residual_is_instantaneous_without_an_assumed_current_pulse():
    q_gs, q_gd = 2**-28, 2**-29
    settling = public(q_gs=q_gs, q_gd=q_gd, q_g=q_gs + q_gd, settling_target_v=7)["estimate"]["turn_on_settling"]
    assert settling["status"] == "instantaneous-ideal-crossing"
    assert settling["duration_s"] == settling["incremental_charge_c"] == settling["residual_charge_c"] == 0
    assert settling["relative_time_s"] == [0, 0]
    assert settling["v_gate_v"] == [4, 7]
    assert settling["i_gate_a"] == [None, None]
    assert settling["initial_gate_current_a"] is None and settling["final_gate_current_a"] is None


@pytest.mark.parametrize("unsupported", [{"q_g": None}, {"r_g": 0, "r_g_int": 0}])
def test_missing_total_charge_or_zero_series_resistance_cannot_imply_an_ideal_crossing(unsupported):
    response = TestClient(app).post(URL, json=PARAMETERS | {"settling_target_v": 7} | unsupported)
    assert response.status_code == 422, response.text


def test_two_target_queries_keep_the_original_device_charge_reference():
    fields = {k: v for k, v in PARAMETERS.items() if k not in ("v_driver", "v_off")}
    parameters = GateChargeModelParams(**fields)
    original = asdict(parameters)
    model = GateChargeModel(parameters)
    near = model.calc_switching_estimate(10, 0, settling_target_v=5)
    saved = deepcopy(near)
    far = model.calc_switching_estimate(10, 0, settling_target_v=8)
    assert asdict(parameters) == original
    assert near == saved
    a, b = near["turn_on_settling"], far["turn_on_settling"]
    assert a["post_plateau_capacitance_f"] == b["post_plateau_capacitance_f"]
    assert a["residual_charge_c"] == b["residual_charge_c"]
    assert math.isclose(b["incremental_charge_c"], 4 * a["incremental_charge_c"], rel_tol=1e-13)
    assert near["turn_on"] == far["turn_on"]


@pytest.mark.parametrize("field", ["q_g", "r_g"])
def test_direct_extended_input_cannot_be_rounded_into_an_ideal_phase(field):
    one = np.longdouble(1)
    adjacent = np.nextafter(one, np.longdouble(2))
    if float(adjacent) != 1:
        pytest.skip("This platform's longdouble has binary64 precision")
    fields = {key: value for key, value in PARAMETERS.items() if key not in ("v_driver", "v_off")}
    fields.update(q_gs=1, q_gd=0, q_g=1, r_g=1, r_g_int=0)
    fields[field] = adjacent
    assert adjacent > one
    with pytest.raises(ValueError):
        GateChargeModel(GateChargeModelParams(**fields)).calc_switching_estimate(10, 0, settling_target_v=7)


def raw_gate(q_g_token, target_token):
    # Embed the literal input tokens so the HTTP client cannot round them first.
    body = ('{"q_gs":1,"q_gd":0,"v_th":2,"v_plateau":4,"r_g":1.5,'
            '"r_g_int":0.5,"v_driver":10,"v_off":0,"q_g":' + q_g_token
            + ',"settling_target_v":' + target_token + '}')
    return TestClient(app).post(URL, content=body, headers={"Content-Type": "application/json"})


@pytest.mark.parametrize("charge", ["1.0000000000000000001", '"1.0000000000000000001"', "9" * 400])
def test_active_raw_charge_must_survive_the_transport_boundary(charge):
    response = raw_gate(charge, "7")
    assert response.status_code == 422, response.text


def test_exact_decimal_equivalents_preserve_the_public_rc_result():
    reference = raw_gate("2", "7")
    assert reference.status_code == 200, reference.text
    equivalent = raw_gate("2000.000e-3", "70.000e-1")
    assert equivalent.status_code == 200, equivalent.text
    assert equivalent.json() == reference.json()
    settling = equivalent.json()["estimate"]["turn_on_settling"]
    assert settling["residual_charge_c"] == 1
    assert math.isclose(settling["duration_s"], math.log(2) / 3)
