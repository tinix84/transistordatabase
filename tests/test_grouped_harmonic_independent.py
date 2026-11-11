"""Analytic and transport oracles for grouped sample and harmonic evidence."""
import csv
import hashlib
import io
import json
import math
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase import grouped_series
from transistordatabase.grid_analysis import AnalysisResourceLimitError, evaluate_grid
from transistordatabase.gui_web.backend.analysis_routes import router
from transistordatabase.waveform_losses import waveform_spectrum, waveform_statistics


def grid(*points):
    """Define a collection with explicit loss omissions."""
    omitted = {"omit": True, "omission_reason": "Independent spectral fixture"}
    return {
        "configurations": [{"id": "C", "source": "Declared fixture", "conduction": omitted,
                            "switching": {"turn_on": omitted, "turn_off": omitted}}],
        "operating_points": list(points),
        "metric": {"name": "conduction_power_w", "unit": "W", "direction": "minimize"},
    }


def point(name, samples, **options):
    """Declare the sample interval, current, source and event policy."""
    return dict(id=name, source="  =source", weight=1,
                waveform={"time_s": list(range(len(samples))), "current_a": samples},
                events={"mode": "omitted", "omission_reason": "No switch events"}, **options)


def fingerprint(value):
    """Compute a canonical digest without calling the production hash helper."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def test_ascending_bin_order_includes_dc_and_excludes_stronger_nyquist():
    """Keep the first ten eligible frequencies when the last bin is strongest."""
    # The impulse gives every bin nonzero amplitude. The alternating term changes only Nyquist.
    values = [2 * (-1) ** j + int(j == 0) for j in range(32)]
    result = waveform_spectrum(range(32), values, relative_threshold=0)
    assert result["selected_bin_indices"] == list(range(10))
    assert result["amplitude_a"][0] == pytest.approx(1 / 32)
    assert result["amplitude_a"][1] == pytest.approx(1 / 16)
    assert result["amplitude_a"][-1] == pytest.approx(2 + 1 / 32)
    assert result["parseval_rms_a"] == pytest.approx(math.sqrt((9 + 31 * 4) / 32))


def test_odd_final_bin_and_nonzero_time_origin():
    """Use the sample sequence as the phase reference for the odd final bin."""
    result = waveform_spectrum([11, 12, 13], [0, 1, 0])
    assert result["frequency_hz"][-1] == pytest.approx(1 / 3)
    assert result["real_a"][-1] == pytest.approx(-1 / 3)
    assert result["imaginary_a"][-1] == pytest.approx(-math.sqrt(3) / 3)
    assert result["phase_rad"][-1] == pytest.approx(-2 * math.pi / 3)
    assert result["amplitude_a"][-1] == pytest.approx(2 / 3)


def test_strict_peak_product_retains_bin_just_above_rounded_cutoff():
    """Compare amplitudes with the declared cutoff at a one-ULP boundary."""
    threshold = 0.5652173913043478
    assert 11.5 * threshold == math.nextafter(6.5, 0)
    assert 6.5 / 11.5 == threshold
    result = waveform_spectrum([0, 1], [18, 5], relative_threshold=threshold)
    assert result["amplitude_a"] == [11.5, 6.5]
    assert result["selected_bin_indices"] == [0, 1]


def test_voltage_units_sample_statistics_and_null_manifests():
    """Keep physical units and distinguish unrequested, failed and empty spectra."""
    fft = {"window": "rectangular", "detrend": "none", "relative_threshold": .5}
    voltage = point("voltage", [0, 0, 0, 0], spectrum=fft)
    voltage["waveform"].update(device_voltage_v=[2, 2, 2, -2], gate_v=[0, 0, 0, 0],
                               bus_voltage_v=400)
    failed = point("irregular", [1, 2, 3], spectrum=fft)
    failed["waveform"]["time_s"] = [0, 1, 3]
    response = evaluate_grid(dict(grid(voltage, point("raw", [0, 2]), failed),
                                  grouped_series_exports=True))
    projection = response["grouped_series_exports"]
    freq = {(r["operating_point_id"], r["series"]): r for r in projection["frequency_rows"]}
    assert len(freq) == len(projection["frequency_rows"]) == 6
    observed = freq["voltage", "device_voltage_v"]
    assert observed["unit"] == "V"
    assert (observed["real"], observed["imaginary"], observed["magnitude"]) == (0, -2, 2)
    assert observed["frequency_hz"] == .25
    assert observed["phase_rad"] == pytest.approx(-math.pi / 2)
    assert freq["voltage", "bus_voltage_v"]["magnitude"] == 400
    for series in ("gate_v", "current_a"):
        empty = freq["voltage", series]
        assert empty["status"] == "computed" and empty["retained_bin_count"] == 0
        assert empty["phase_rad"] is None and empty["bin_index"] is None
    assert freq["raw", "current_a"]["status"] == "not_requested"
    assert freq["raw", "current_a"]["bin_index"] is None
    unsupported = freq["irregular", "current_a"]
    assert unsupported["pair_status"] == unsupported["status"] == "unsupported"
    assert unsupported["error"] and unsupported["magnitude"] is None
    stats = next(r for r in projection["time_rows"] if r["operating_point_id"] == "raw")
    assert stats["arithmetic_mean"] == stats["time_weighted_mean"] == 1
    assert stats["sample_standard_deviation"] == pytest.approx(math.sqrt(2))
    assert stats["discrete_rms"] == pytest.approx(math.sqrt(2))
    assert stats["time_weighted_rms"] == pytest.approx(2 / math.sqrt(3))
    csv_rows = list(csv.DictReader(io.StringIO(projection["frequency_csv"])))
    assert all(r["operating_point_source"] == "'  =source" for r in csv_rows)
    assert all(r["operating_point_source"] == "  =source" for r in freq.values())


def test_old_wire_evidence_omits_new_fields_with_false_or_absent_export_setting():
    """Keep legacy transport defaults and input digests with exports disabled."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    request = grid(point("p", [0, 1], spectrum={"window": "rectangular", "detrend": "none"}))
    for payload in (request, dict(request, grouped_series_exports=False)):
        reply = client.post("/api/analysis/grid", json=payload)
        assert reply.status_code == 200, reply.text
        response = reply.json()
        canonical = response["inputs"]
        assert "grouped_series_exports" not in canonical and "grouped_series_exports" not in response
        assert canonical["operating_points"][0]["spectrum"] == request["operating_points"][0]["spectrum"]
        assert response["input_sha256"] == fingerprint(canonical)
        pair = response["rows"][0]
        assert pair["input_sha256"] == fingerprint(pair["inputs"])
        assert pair["inputs"]["waveform_request"]["spectrum"] == request["operating_points"][0]["spectrum"]


def test_existing_piecewise_linear_extreme_gradient_is_representable():
    """Retain the legacy integral and derivative for representable gradients."""
    limit = sys.float_info.max
    result = waveform_statistics([0, 1, 2], [-limit, 0, limit])
    assert result["mean"] == 0
    assert result["rms"] / limit == pytest.approx(1 / math.sqrt(3))
    assert result["interval_derivative"] == [limit, limit]


def test_voltage_failures_stay_local_to_statistic_or_spectrum(monkeypatch):
    """Keep a successful electrical evaluation when one series operation fails."""
    data = point("p", [1, 1], spectrum={"window": "rectangular", "detrend": "none"})
    data["waveform"].update(gate_v=[-sys.float_info.max, sys.float_info.max], bus_voltage_v=400)
    original = grouped_series.series_spectrum

    def refuse_bus(time_s, values, **settings):
        if values[0] == 400:
            raise ValueError("Unsupported voltage FFT fixture")
        return original(time_s, values, **settings)

    monkeypatch.setattr(grouped_series, "series_spectrum", refuse_bus)
    with np.errstate(over="ignore"):
        response = evaluate_grid(dict(grid(data), grouped_series_exports=True))
    assert response["rows"][0]["status"] == "computed"
    exported = response["grouped_series_exports"]
    time = {row["series"]: row for row in exported["time_rows"]}
    freq = {row["series"]: row for row in exported["frequency_rows"]}
    assert time["gate_v"]["sample_standard_deviation"] is None
    assert time["gate_v"]["status"] == "unsupported"
    assert freq["gate_v"]["magnitude"] == sys.float_info.max
    assert time["bus_voltage_v"]["arithmetic_mean"] == 400
    assert freq["bus_voltage_v"]["status"] == "unsupported"
    assert freq["bus_voltage_v"]["pair_status"] == "computed"
    assert freq["bus_voltage_v"]["error"] == "Unsupported voltage FFT fixture"
    assert freq["current_a"]["magnitude"] == 1


def test_non_ascii_metadata_preflight_rejects_before_numerical_projection(monkeypatch):
    """Reject repeated escaped metadata before calculating export rows."""
    request = grid(point("p", [1, 1]))
    accepted = evaluate_grid(request)
    rows = accepted["rows"]
    grouped_series.project_grouped_series(rows, accepted["input_sha256"], 20000,
                                          AnalysisResourceLimitError)
    request["operating_points"][0]["source"] = 'µ"\n' * 1100
    accepted = evaluate_grid(request)
    monkeypatch.setattr(grouped_series, "waveform_scalar_statistics",
                        lambda *args, **kwargs: pytest.fail("projection began before preflight"))
    with pytest.raises(AnalysisResourceLimitError, match="byte budget"):
        grouped_series.project_grouped_series(accepted["rows"], accepted["input_sha256"],
                                              20000, AnalysisResourceLimitError)
