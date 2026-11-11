"""Independent transport and provenance checks for transfer/recovery extraction."""
from copy import deepcopy
import csv
import io
import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.gui_web.backend.parameter_routes import (
    ExtractionEvidence, router, validate_biela_extraction_evidence,
)


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def transfer():
    """Return exact quadratic transfer measurements at one known condition."""
    return {"operation": "transfer-fit", "source": "independent transfer fixture",
            "conditions": {"v_ds_v": 10.0, "t_j_c": 25.0}, "v_th_v": 3.0,
            "curves": [{"v_g_v": float(index + 4), "t_j_c": 25.0,
                        "voltage_v": [10.0], "current_a": [current]}
                       for index, current in enumerate([2.1, 8.1, 18.1, 32.1, 50.1])]}


def recovery():
    """Return a same-condition recovery fixture in SI units."""
    return {"operation": "reverse-recovery-extraction", "source": "independent recovery fixture",
            "conditions": {"v_supply_v": 100.0, "t_j_c": 25.0,
                           "q_oss_v_ref_v": 100.0, "q_oss_t_j_c": 25.0},
            "q_rr_c": 100e-9, "i_rr_a": 10.0, "di_dt_a_s": 2.4e9,
            "q_oss_c": 12e-9, "i_forward_test_a": 20.0, "t_rr_s": 30e-9}


def calculate(payload):
    """Exercise the typed operation union."""
    return client.post("/api/parameters/calculate", json=payload)


def test_exact_transfer_samples_fit_and_shared_plot_values():
    """The solver retains source conditions, coverage, rank and exact plot rows."""
    payload = transfer()
    before = deepcopy(payload)
    response = calculate(payload)
    assert response.status_code == 200, response.text
    envelope = response.json()
    result = envelope["result"]
    assert [result[key] for key in ("k_1", "k_2", "x")] == pytest.approx([2, .1, 2])
    assert result["diagnostics"]["rank"] == 3
    assert result["diagnostics"]["rmse_a"] < 1e-8
    assert [result["current_min"], result["current_max"]] == [2.1, 50.1]
    assert envelope["context"] == {"source": payload["source"], "conditions": payload["conditions"]}
    assert envelope["plots"][0]["series"]["Fitted"] == result["fitted_current_a"]
    assert envelope["plots"][1]["series"]["Residual"] == result["residual_a"]
    assert payload == before


@pytest.mark.parametrize("source", ['=SUM(1,2)', '  @source', '\tsource', 'quoted,"source"\nline'])
def test_exact_transfer_csv_preserves_numeric_rows_and_safe_text(source):
    """CSV shares exact result rows and keeps source text from becoming a formula."""
    payload = transfer()
    payload["source"] = source
    envelope = calculate(payload).json()
    rows = list(csv.DictReader(io.StringIO(envelope["csv"])))
    result = envelope["result"]
    for index, row in enumerate(rows):
        for key in ("gate_voltage_v", "current_a", "fitted_current_a", "residual_a", "secant_gm_s"):
            assert float(row[key]) == result[key][index]
        assert float(row["v_ds_v"]) == payload["conditions"]["v_ds_v"]
    normalized = source.strip()
    assert rows[0]["source"] == ("'" + normalized if normalized.startswith(("=", "@")) else normalized)
    assert envelope["inputs"]["source"] == source.strip()


def test_recovery_csv_retains_test_charge_conditions_and_seconds():
    """The recovery row preserves SI inputs and exact calculated constants."""
    envelope = calculate(recovery()).json()
    row, = csv.DictReader(io.StringIO(envelope["csv"]))
    for key in ("q_rr_c", "q_oss_c", "i_forward_test_a", "di_dt_a_s"):
        assert float(row[key]) == envelope["inputs"][key]
    for key in ("tau_rr", "tau_c", "t_m"):
        assert float(row[key + "_s"]) == envelope["result"][key]
    assert float(row["q_rr_star"]) == envelope["result"]["q_rr_star"]
    assert float(row["v_supply_v"]) == float(row["q_oss_v_ref_v"]) == 100


def test_transfer_retains_bounded_output_curve_interpolation():
    """Multiple Vds samples use the accepted bounded interpolation."""
    payload = transfer()
    for curve in payload["curves"]:
        current = curve["current_a"][0]
        curve.update(voltage_v=[5.0, 15.0], current_a=[current - .5, current + .5])
    response = calculate(payload)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["k_1"] == pytest.approx(2)
    payload["conditions"]["v_ds_v"] = 20.0
    assert calculate(payload).status_code == 422


@pytest.mark.parametrize("change", [
    lambda p: p["conditions"].update(v_ds_v=11.0),
    lambda p: p["conditions"].update(t_j_c=None),
    lambda p: p["curves"][0].update(t_j_c=26.0),
    lambda p: p["curves"][0].update(v_g_v=3.0),
    lambda p: p["curves"][0].update(current_a=[2.1, 3.0]),
    lambda p: p["curves"][0].update(v_g_v=5.0),
    lambda p: p.update(curves=p["curves"][:3]),
    lambda p: [curve.update(current_a=[1.0]) for curve in p["curves"]],
    lambda p: p.update(v_th_v=True),
    lambda p: p["curves"][0].update(current_a=[True]),
])
def test_transfer_rejects_missing_mismatched_or_degenerate_records(change):
    """Invalid transfer data never becomes a fallback fit."""
    payload = transfer()
    change(payload)
    assert calculate(payload).status_code == 422


def test_recovery_constants_and_independent_charge_residual_identities():
    """Verify SI conversion, corrected charge and the implicit peak identity."""
    payload = recovery()
    response = calculate(payload)
    assert response.status_code == 200, response.text
    envelope = response.json()
    result = envelope["result"]
    assert result["q_rr_star"] == pytest.approx(88e-9)
    assert result["tau_rr"] == pytest.approx(6.716666666667e-9)
    assert result["tau_c"] == pytest.approx(13.669756359629e-9)
    assert result["t_m"] == pytest.approx(13.204949301187e-9)
    assert result["tau_rr"] * payload["i_rr_a"] + payload["i_rr_a"]**2 / (2 * payload["di_dt_a_s"]) == pytest.approx(88e-9)
    t1 = (payload["i_forward_test_a"] + payload["i_rr_a"]) / payload["di_dt_a_s"]
    peak = payload["di_dt_a_s"] * (result["tau_c"] - result["tau_rr"]) * -math.expm1(-t1 / result["tau_c"])
    assert peak == pytest.approx(payload["i_rr_a"], rel=1e-10)
    assert abs(result["peak_residual_a"]) < 1e-9
    assert result["t_rr_used"] is False
    assert envelope["inputs"] == payload
    payload["t_rr_s"] = 1e-3
    assert calculate(payload).json()["result"]["tau_rr"] == result["tau_rr"]


@pytest.mark.parametrize("change", [
    lambda p: p.update(q_rr_c=10e-9),
    lambda p: p.update(q_rr_c=20e-9),
    lambda p: p.update(i_forward_test_a=None),
    lambda p: p.update(i_forward_test_a=True),
    lambda p: p.update(di_dt_a_s=0.0),
    lambda p: p.update(q_oss_c=-1.0),
    lambda p: p.update(t_rr_s=-1.0),
    lambda p: p["conditions"].update(q_oss_v_ref_v=200.0),
    lambda p: p["conditions"].update(q_oss_t_j_c=26.0),
    lambda p: p["conditions"].update(t_j_c=None),
])
def test_recovery_rejects_invalid_charges_and_context(change):
    """Every extraction input has explicit physical and source constraints."""
    payload = recovery()
    change(payload)
    assert calculate(payload).status_code == 422


@pytest.mark.parametrize("payload,key", [(transfer(), "v_th_v"), (recovery(), "q_rr_c")])
def test_nonstandard_nan_errors_remain_json_safe(payload, key):
    """Nonstandard JSON NaN is rejected with a serializable 422 response."""
    import json
    data = deepcopy(payload)
    data[key] = float("nan")
    response = client.post("/api/parameters/calculate", content=json.dumps(data),
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_biela_evidence_preserves_separate_transfer_vds_and_rejects_recovery_conflict():
    """Source transfer voltage never silently replaces the model bus or Qoss."""
    fitted = calculate(transfer()).json()["result"]["biela_parameters"]
    parameters = {**fitted, "parameter_t_j": 25.0, "v_dc": 400.0,
                  "q_oss": 200e-9, "q_oss_v_ref": 400.0, "recovery_mode": "omit"}
    evidence = ExtractionEvidence(inputs=transfer(), status="applied")
    result = validate_biela_extraction_evidence([evidence], parameters)
    assert result[0]["context"]["conditions"]["v_ds_v"] == 10
    assert parameters["v_dc"] == 400
    parameters["k_1"] = 3.0
    with pytest.raises(ValueError, match="parameters changed"):
        validate_biela_extraction_evidence([evidence], parameters)
    evidence.status = "edited"
    assert validate_biela_extraction_evidence([evidence], parameters)[0]["status"] == "edited"
    rr_params = calculate(recovery()).json()["result"]["biela_parameters"]
    parameters.update(rr_params, recovery_mode="charge_control")
    evidence = ExtractionEvidence(inputs=recovery(), status="applied")
    with pytest.raises(ValueError, match="source Qoss"):
        validate_biela_extraction_evidence([evidence], parameters)
    parameters.update(q_oss=12e-9, q_oss_v_ref=100.0, v_dc=100.0)
    result = validate_biela_extraction_evidence([evidence], parameters)
    assert result[0]["result"]["q_rr_star"] == pytest.approx(88e-9)
    assert result[0]["inputs"]["i_forward_test_a"] == 20.0


def test_full_biela_http_boundary_preserves_recovery_evidence_and_rejects_conflicts():
    """The real Biela route enforces evidence before model calculation."""
    from transistordatabase.gui_web.backend.main import app as real_app
    http = TestClient(real_app)
    payload = {"model_kind": "christen_biela_estimate", "source": "algebra fixture with separate recovery source",
               "c_gs": 1e-9, "c_ds": 6e-11, "c_gd": 6e-11,
               "q_oss": 12e-9, "q_oss_v_ref": 100.0,
               "k_1": 10.0, "k_2": 0.0, "x": 1.0, "v_th": 3.0,
               "gm_i_min": 1.3, "gm_i_max": 50.0, "v_dc": 100.0,
               "i_load": 10.0, "v_driver_on": 15.0, "v_driver_off": 0.0,
               "r_g": 10.0, "l_s": 0.0, "l_d": 0.0, "t_j": 25.0,
               "parameter_t_j": 25.0, "recovery_mode": "charge_control",
               **calculate(recovery()).json()["result"]["biela_parameters"],
               "extraction_evidence": [{"inputs": recovery(), "status": "applied"}]}
    response = http.post("/api/analytical/biela", json=payload)
    assert response.status_code == 200, response.text
    evidence = response.json()["extraction_evidence"][0]
    assert evidence["inputs"] == recovery()
    assert evidence["result"]["q_rr_star"] == pytest.approx(88e-9)
    assert response.json()["parameter_inputs_si"]["q_oss"] == 12e-9
    payload["q_oss"] = 200e-9
    response = http.post("/api/analytical/biela", json=payload)
    assert response.status_code == 422
    assert "source Qoss" in response.json()["detail"]
    payload["q_oss"] = 12e-9
    payload["tau_rr"] *= 1.01
    assert http.post("/api/analytical/biela", json=payload).status_code == 422
    payload["extraction_evidence"][0]["inputs"]["q_rr_c"] = float("nan")
    import json
    response = http.post("/api/analytical/biela", content=json.dumps(payload),
                         headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]
