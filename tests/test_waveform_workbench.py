"""Independent exact-integral, event, spectrum and typed transport fixtures."""
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.core.models import LinearizedModel
from transistordatabase.gui_web.backend.waveform_routes import router
from transistordatabase.waveform_losses import (
    WaveformData, analyze_waveform, calc_conduction_loss, waveform_spectrum,
    waveform_statistics,
)


def linear(offset=0, resistance=.1):
    """Build an explicitly selected SI threshold/resistance fixture."""
    return {"mode": "linear-drop-estimate", "selected": True, "offset_v": offset,
            "resistance_ohm": resistance, "source": "Independent hand fixture", "conditions": {}}


def omitted(reason="Conduction-only independent fixture"):
    """Declare an intentional missing component with a reason."""
    return {"omit": True, "omission_reason": reason}


def request(time=(0, 1), current=(0, 10)):
    """Build a conduction-only request without inventing switching evidence."""
    return {"waveform": {"time_s": list(time), "current_a": list(current)},
            "conduction": {"positive": linear()},
            "events": {"mode": "omitted", "omission_reason": "No switching requested"},
            "switching": {"turn_on": omitted(), "turn_off": omitted()}}


def energy(voltage, multiplier, *, mode="voltage-scaling-estimate", zero=0):
    """Build independently conditioned bounded energy-current slices."""
    return {"mode": mode, **({"selected": True} if mode != "recorded" else {}),
            "current_a": [0, 10], "energy_j": [zero, zero + 10 * multiplier],
            "recorded_conditions": {"v_supply_v": voltage, "t_j_c": 25},
            "requested_conditions": {"v_supply_v": 200, "t_j_c": 25},
            "source": f"Independent {voltage}V energy slice"}


def switching_request():
    """Build the archive pre-edge identity fixture in SI units."""
    data = request([0, 1e-6, 2e-6, 3e-6, 4e-6], [0, 2, 9, 3, 0])
    data["waveform"].update(gate_v=[0, 0, 1, 1, 0], bus_voltage_v=200)
    data["events"] = {"mode": "gate", "threshold_v": .5}
    data["switching"] = {"turn_on": {"positive": energy(400, .001)},
                         "turn_off": {"positive": energy(600, .002)}}
    return data


@pytest.mark.parametrize("time,current,positive,negative,duty,expected", [
    ([0, 1], [10, 10], linear(1), None, 1, 20),
    ([0, 1, 4], [0, 10, 0], linear(), None, 1, 10 / 3),
    ([0, 2], [-10, 10], linear(1), linear(2, .2), 1, 12.5),
    ([0, 1], [0, 10], linear(), None, [0, 1], 2.5),
])
def test_exact_polynomial_fixtures(time, current, positive, negative, duty, expected):
    """Verify exact polynomial fixtures."""
    data = request(time, current)
    data["conduction"] = {"positive": positive, "negative": negative, "duty": duty}
    result = analyze_waveform(data)
    assert result["components"]["conduction"]["power_w"] == pytest.approx(expected)
    assert sum(s["energy_j"] for s in result["power_segments"]) == pytest.approx(expected * (time[-1] - time[0]))
    for segment in result["power_segments"]:
        assert len(segment["power_coefficients_w"]) == 4
        assert segment["energy_j"] == pytest.approx((segment["end_s"] - segment["start_s"]) * sum(c / (i + 1) for i, c in enumerate(segment["power_coefficients_w"])))


def test_crossed_table_knot_and_cumulative_evidence():
    """Verify crossed table knot and cumulative evidence."""
    data = request()
    data["conduction"]["positive"] = {"mode": "recorded", "source": "Measured hand curve",
        "current_a": [0, 5, 10], "drop_v": [0, .5, 2],
        "recorded_conditions": {"t_j_c": 25}, "requested_conditions": {"t_j_c": 25}}
    result = analyze_waveform(data)
    assert result["components"]["conduction"]["power_w"] == pytest.approx(65 / 12)
    assert [s["end_s"] for s in result["power_segments"]] == [.5, 1]
    assert result["conduction_plot"][0]["cumulative_j"] == 0
    assert result["conduction_plot"][-1]["cumulative_j"] == pytest.approx(65 / 12)


def test_preedge_identity_independent_reference_voltages_and_full_duration():
    """Verify preedge identity independent reference voltages and full duration."""
    result = analyze_waveform(switching_request())
    on, off = result["events"]
    assert (on["sample_index"], on["current_a"]) == (1, 2)
    assert (off["sample_index"], off["current_a"]) == (3, 3)
    assert on["bracket_s"] == [1e-6, 2e-6]
    assert on["energy_j"] == pytest.approx(.001)
    assert off["energy_j"] == pytest.approx(.002)
    assert result["components"]["turn_on"]["power_w"] == pytest.approx(250)
    assert result["components"]["turn_off"]["power_w"] == pytest.approx(500)
    assert result["components"]["turn_on"]["event_rate_hz"] == 250000
    assert result["total_status"] == "computed"


def test_finite_zero_current_energy_and_no_initial_or_wrapped_event():
    """Verify finite zero current energy and no initial or wrapped event."""
    data = switching_request()
    data["waveform"]["current_a"] = [0] * 5
    data["switching"]["turn_on"]["positive"] = energy(400, .001, zero=.002)
    assert analyze_waveform(data)["events"][0]["energy_j"] == pytest.approx(.001)
    data["waveform"]["gate_v"] = [1] * 5
    result = analyze_waveform(data)
    assert result["events"] == []
    assert result["components"]["turn_on"]["event_count"] == 0
    assert result["components"]["turn_on"]["power_w"] == 0


def test_explicit_masks_and_threshold_equality_rule():
    """Verify explicit masks and threshold equality rule."""
    data = switching_request()
    data["events"] = {"mode": "masks", "turn_on": [False, True, False, False],
                      "turn_off": [False, False, False, True]}
    assert [e["sample_index"] for e in analyze_waveform(data)["events"]] == [1, 3]
    data["events"] = {"mode": "gate", "threshold_v": 1}
    assert analyze_waveform(data)["events"] == []
    data["events"]["turn_on"] = [False] * 4
    with pytest.raises(ValueError, match="mutually exclusive"):
        analyze_waveform(data)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["waveform"].update(time_s=[0, 1, 1, 3, 4]), "increase strictly"),
    (lambda d: d["waveform"].update(current_a=[0, 1]), "align"),
    (lambda d: d["waveform"].update(gate_v=1), "1D"),
    (lambda d: d["waveform"].update(current_a=[0, 1, np.nan, 1, 0]), "finite"),
    (lambda d: d["waveform"].update(current_a=[-1] * 5), "Missing conduction"),
    (lambda d: d["conduction"].update(duty=1.1), "duty"),
    (lambda d: d["conduction"]["positive"].update(selected=False), "explicitly"),
    (lambda d: d["switching"]["turn_on"].update(positive=None), "energy model"),
    (lambda d: d["switching"]["turn_on"]["positive"].update(requested_conditions={"v_supply_v": 200, "t_j_c": 50}), "conditions"),
    (lambda d: d["switching"]["turn_on"]["positive"].update(current_a=[0, 1]), "supports"),
    (lambda d: d["switching"]["turn_on"]["positive"].update(mode="recorded"), "conditions"),
])
def test_rejects_instead_of_fallback(mutation, match):
    """Verify rejects instead of fallback."""
    data = switching_request()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        analyze_waveform(data)


@pytest.mark.parametrize("reason", [None, "", "  ", 123])
def test_omission_requires_actual_nonblank_string(reason):
    """Verify omission requires actual nonblank string."""
    data = request()
    data["events"]["omission_reason"] = reason
    with pytest.raises(ValueError, match="reason"):
        analyze_waveform(data)


def test_missing_omitted_and_zero_are_distinct_and_request_is_a_snapshot():
    """Verify missing omitted and zero are distinct and request is a snapshot."""
    data = request(current=[0, 0])
    data["waveform"]["current_a"] = np.array([0., 0.])
    result = analyze_waveform(data)
    assert result["components"]["conduction"]["power_w"] == 0
    assert result["components"]["turn_on"]["power_w"] is None
    assert result["total_power_w"] is None
    assert result["known_component_power_w"] == 0
    data["waveform"]["current_a"][0] = 7
    data["events"]["omission_reason"] = "changed later"
    assert result["request"]["waveform"]["current_a"] == [0, 0]
    assert result["event_policy"]["omission_reason"] == "No switching requested"


def test_gate_is_not_a_conduction_mask():
    """Verify gate is not a conduction mask."""
    data = request(current=[10, 10])
    data["waveform"]["gate_v"] = [0, 0]
    assert analyze_waveform(data)["components"]["conduction"]["power_w"] == 10


def test_irregular_time_statistics_and_derivatives():
    """Verify irregular time statistics and derivatives."""
    stats = waveform_statistics([0, 1, 4], [0, 10, 0])
    assert stats["mean"] == 5
    assert stats["rms"] == pytest.approx(np.sqrt(100 / 3))
    assert stats["interval_derivative"] == [10, -10 / 3]


@pytest.mark.parametrize("n", [15, 16])
def test_fft_dc_bin_sine_nyquist_and_parseval(n):
    """Verify fft dc bin sine nyquist and parseval."""
    time = np.arange(n) / n
    result = waveform_spectrum(time, 3 + 2 * np.sin(2 * np.pi * 3 * time))
    assert result["amplitude_a"][0] == pytest.approx(3)
    assert result["amplitude_a"][3] == pytest.approx(2)
    assert result["parseval_rms_a"] == pytest.approx(np.sqrt(11))
    if n % 2 == 0:
        assert waveform_spectrum(time, (-1.) ** np.arange(n))["amplitude_a"][-1] == 1


def test_fft_rejects_irregular_time_and_records_window_detrending():
    """Verify fft rejects irregular time and records window detrending."""
    with pytest.raises(ValueError, match="uniform"):
        waveform_spectrum([0, 1, 2.000001], [0, 1, 0])
    result = waveform_spectrum(np.arange(16), np.ones(16) * 3, window="hann", detrend="mean")
    assert result["window"] == "hann"
    assert result["discrete_input_rms_a"] == 0
    assert result["resampling"] == "none"


@pytest.mark.parametrize("operation", [
    lambda: waveform_statistics([0, 1e-250], [0, 1e100]),
    lambda: waveform_spectrum(np.arange(8), [1.7e308] * 4 + [-1.7e308] * 4),
    lambda: calc_conduction_loss(WaveformData(np.array([0, 1]), np.array([1e200, 1e200])), LinearizedModel(25, 1, 1, 0)),
])
def test_public_helpers_reject_finite_input_result_overflow(operation):
    """Verify public helpers reject finite input result overflow."""
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="finite|overflow"):
        operation()


def test_measured_table_validates_samples_once(monkeypatch):
    """Verify measured table validates samples once."""
    from transistordatabase import condition_tables
    table = condition_tables.MeasuredTable("energy", [0, 10], [0, 1], {}, "fixture")
    monkeypatch.setattr(condition_tables, "samples", lambda *args: pytest.fail("revalidated immutable table"))
    assert table.evaluate(5, {}) == .5
    with pytest.raises(ValueError, match="supports"):
        table.evaluate(11, {})


def test_tiny_signed_current_and_rms_do_not_underflow_to_zero():
    """Verify tiny signed current and rms do not underflow to zero."""
    data = request([0, 1], [-1e-170, 1e-170])
    data["conduction"] = {"positive": linear(1, 0), "negative": linear(1, 0)}
    result = analyze_waveform(data)
    assert result["components"]["conduction"]["power_w"] == pytest.approx(5e-171, rel=1e-14, abs=0)
    assert result["current_statistics"]["rms"] == pytest.approx(1e-170 / np.sqrt(3), rel=1e-14, abs=0)
    spectrum = waveform_spectrum([0, 1, 2, 3], [1e-170] * 4)
    assert spectrum["parseval_rms_a"] == pytest.approx(1e-170, rel=1e-14, abs=0)
    assert spectrum["discrete_input_rms_a"] == spectrum["parseval_rms_a"]


def test_resource_budget_rejects_before_interval_expansion(client):
    """Verify resource budget rejects before interval expansion."""
    data = request([0, 1, 2], [0, 9999, 0])
    data["conduction"]["positive"] = {"mode": "recorded", "source": "Dense measured fixture", "current_a": list(range(10000)), "drop_v": [1] * 10000, "recorded_conditions": {}, "requested_conditions": {}}
    response = client.post("/api/waveform/analyze", json=data)
    assert response.status_code == 413
    assert "Resource limit" in response.json()["detail"]


@pytest.fixture
def client():
    """Expose only the headless waveform router for transport tests."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_typed_transport_equals_science_and_rejects_nonfinite_extra_and_missing(client):
    """Verify typed transport equals science and rejects nonfinite extra and missing."""
    data = switching_request()
    response = client.post("/api/waveform/analyze", json=data)
    assert response.status_code == 200, response.text
    result = response.json()
    direct = analyze_waveform(data)
    assert result["components"] == direct["components"]
    assert result["events"] == direct["events"]
    data["surprise"] = True
    assert client.post("/api/waveform/analyze", json=data).status_code == 422
    assert client.post("/api/waveform/analyze", content='{"waveform":{"time_s":[0,NaN],"current_a":[0,1]}}', headers={"Content-Type": "application/json"}).status_code == 422
    data = switching_request()
    data["waveform"].pop("gate_v")
    assert client.post("/api/waveform/analyze", json=data).status_code == 422


def test_waveform_provenance_is_unknown_or_an_independent_user_snapshot(client):
    """Preserve declared source/calibration without fabricating measured identity."""
    data = request()
    assert analyze_waveform(data)["waveform_provenance"]["source"] is None
    data["provenance"] = {"source": "User supplied SI record", "calibration_notes": "Probe gain already applied", "imported_filename": "scope.csv", "edited_after_import": True}
    response = client.post("/api/waveform/analyze", json=data)
    assert response.status_code == 200
    assert response.json()["waveform_provenance"] == data["provenance"]
    result = analyze_waveform(data)
    data["provenance"]["source"] = "Changed after calculation"
    assert result["waveform_provenance"]["source"] == "User supplied SI record"
