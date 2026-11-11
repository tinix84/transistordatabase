"""Verify extraction evidence through both production HTTP endpoints."""
from copy import deepcopy
import json
from math import expm1, isfinite

import pytest
from fastapi.testclient import TestClient

from transistordatabase.gui_web.backend.main import app


http = TestClient(app, raise_server_exceptions=False)
workbench = "/api/parameters/calculate"
model_route = "/api/analytical/biela"


def extract(request):
    """Calculate the evidence with the production router."""
    response = http.post(workbench, json=request)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def records():
    """Use independent drain-voltage grids that each cover 10 V."""
    currents = [2.1, 8.1, 18.1, 32.1, 50.1]
    curves = [{"v_g_v": 4 + i, "t_j_c": 25, "voltage_v": [9-i, 11+i],
               "current_a": [current-1, current+1]} for i, current in enumerate(currents)]
    curves[0].update(voltage_v=[10], current_a=[2.1])
    transfer = extract({"operation": "transfer-fit", "source": "fixture/transfer-25C.csv",
                        "conditions": {"v_ds_v": 10, "t_j_c": 25},
                        "curves": list(reversed(curves)), "v_th_v": 3})
    recovery = extract({"operation": "reverse-recovery-extraction",
                        "source": "fixture/recovery-400V.dat",
                        "conditions": {"v_supply_v": 400, "t_j_c": 25,
                                       "q_oss_v_ref_v": 400, "q_oss_t_j_c": 25},
                        "q_rr_c": 100e-9, "i_rr_a": 10, "i_forward_test_a": 20,
                        "di_dt_a_s": 2.4e9, "q_oss_c": 12e-9, "t_rr_s": 30e-9})
    return transfer, recovery


@pytest.fixture
def model(records):
    """Use the extracted parameters and the charge of one switch at 400 V."""
    transfer, recovery = records
    return {"model_kind": "christen_biela_estimate", "source": "400 V model fixture",
            "c_gs": 1e-9, "c_ds": 15e-12, "c_gd": 15e-12,
            "q_oss": 12e-9, "q_oss_v_ref": 400, "v_dc": 400, "i_load": 10,
            "v_driver_on": 15, "v_driver_off": 0, "r_g": 10, "l_s": 0, "l_d": 0,
            "t_j": 25, "parameter_t_j": 25, "recovery_mode": "charge_control",
            **transfer["result"]["biela_parameters"], **recovery["result"]["biela_parameters"],
            "extraction_evidence": [{"inputs": deepcopy(envelope["inputs"]), "status": "applied"}
                                    for envelope in records]}


def test_source_curves_reconstruct_recorded_fitted_residual_and_secant_gm(records):
    """Reconstruct every plotted series from the quadratic current fixture."""
    envelope, _ = records
    result = envelope["result"]
    assert envelope["inputs"]["curves"][0]["voltage_v"] == [5, 15]
    assert envelope["context"]["conditions"] == {"v_ds_v": 10, "t_j_c": 25}
    samples = [2 * (gate - 3)**2 + .1 for gate in range(4, 9)]
    assert result["current_a"] == pytest.approx(samples)
    assert [result[key] for key in ("k_1", "k_2", "x")] == pytest.approx([2, .1, 2])
    assert [result["current_min"], result["current_max"]] == pytest.approx([2.1, 50.1])
    diagnostics = result["diagnostics"]
    assert diagnostics["rank"] == 3 and isfinite(diagnostics["jacobian_condition"])
    assert diagnostics["rmse_a"] < 1e-8 and diagnostics["points"] == len(samples)
    fit_plot, residual_plot = envelope["plots"]
    assert fit_plot["x"] == residual_plot["x"] == list(range(4, 9))
    assert fit_plot["series"]["Recorded"] == pytest.approx(samples)
    for i, gate in enumerate(fit_plot["x"]):
        fitted = fit_plot["series"]["Fitted"][i]
        assert fitted == pytest.approx(samples[i], abs=1e-8)
        assert residual_plot["series"]["Residual"][i] == fitted - result["current_a"][i]
        assert result["secant_gm_s"][i] == pytest.approx(samples[i] / (gate - 3))


def test_recovery_time_and_charge_identities(records):
    """Check lifetimes and current at the peak of reverse recovery."""
    _, envelope = records
    result = envelope["result"]
    rr, carrier, transit = (result[key] for key in ("tau_rr", "tau_c", "t_m"))
    assert [v * 1e9 for v in (rr, carrier, transit)] == pytest.approx(
        [6.716666666667, 13.669756359629, 13.204949301187], rel=1e-10)
    assert result["q_rr_star"] * 1e9 == pytest.approx(88)
    assert result["t_rr_used"] is False
    rise_charge = .5 * 10 * (10 / 2.4e9)
    assert (rise_charge + 10 * rr + 12e-9) * 1e9 == pytest.approx(100)
    assert -expm1(-30 / (2.4e9 * carrier)) * 2.4e9 * (carrier - rr) == pytest.approx(10)
    assert transit * (1 / rr - 1 / carrier) == pytest.approx(1)


def test_combined_evidence_keeps_distinct_transfer_voltage_and_test_current(model, records):
    """Keep the original test conditions when the model uses another operating point."""
    response = http.post(model_route, json=model)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["operating_point"] == {"v_dc": 400, "t_j": 25, "i_load": 10}
    expected = [{"status": "applied", **envelope} for envelope in records]
    assert data["extraction_evidence"] == expected
    assert data["parameter_inputs_si"]["extraction_evidence"] == model["extraction_evidence"]
    assert data["extraction_evidence"][1]["inputs"]["i_forward_test_a"] == 20


@pytest.mark.parametrize(("field", "value"), [
    ("source", float("nan")), ("v_driver_off", False), ("l_d", False),
    ("t_j", float("inf")), ("parameter_t_j", 26), ("k_2", .5),
    ("gm_i_max", 55), ("tau_rr", 7e-9), ("q_oss", 12.01e-9),
])
def test_bad_model_values_return_structured_422(model, field, value):
    """Reject nonnumeric, nonfinite and inconsistent model values."""
    model[field] = value
    response = http.post(model_route, content=json.dumps(model),
                         headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text[:200]
    assert response.json()["detail"]


@pytest.mark.parametrize(("field", "value"), [
    ("source", float("nan")), ("source", float("inf")), ("source", None),
    ("v_th_v", True), ("v_th_v", float("-inf")), ("v_th_v", None),
    ("curves", []),
])
def test_invalid_embedded_transfer_proof_is_rejected_at_both_routes(model, field, value):
    """Require a valid extraction request when a caller sends calibration evidence."""
    proof = model["extraction_evidence"][0]["inputs"]
    proof[field] = value
    for path, payload in ((workbench, proof), (model_route, model)):
        response = http.post(path, content=json.dumps(payload),
                             headers={"content-type": "application/json"})
        assert response.status_code == 422, response.text[:200]
        assert response.json()["detail"]


def test_edited_transfer_keeps_matching_recovery_calibration(model):
    """An edited transfer coefficient changes only the transfer evidence status."""
    model["k_1"] *= 1.01
    model["extraction_evidence"][0]["status"] = "edited"
    response = http.post(model_route, json=model)
    assert response.status_code == 200, response.text
    transfer, recovery = response.json()["extraction_evidence"]
    assert [transfer["status"], recovery["status"]] == ["edited", "applied"]
    assert transfer["result"]["k_1"] == pytest.approx(2)
    model["q_oss"] *= 1.001
    assert http.post(model_route, json=model).status_code == 422


@pytest.mark.parametrize(("charge", "peak", "slope"), [
    (1e-310, 1e-160, 1e300),
    (1e-320, 1e10, 1.7e308),
])
def test_unrepresentable_recovery_lifetime_returns_422(model, charge, peak, slope):
    """Reject recovery constants outside the numerical range at either endpoint."""
    proof = model["extraction_evidence"][1]["inputs"]
    proof.update(q_rr_c=charge, q_oss_c=0, i_rr_a=peak, di_dt_a_s=slope)
    for path, payload in ((workbench, proof), (model_route, model)):
        response = http.post(path, json=payload)
        assert response.status_code == 422
        assert response.json()["detail"]
