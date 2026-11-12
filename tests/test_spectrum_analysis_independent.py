"""Independent one-sided spectral checks with hand-solvable signals."""
from hashlib import sha256
import json
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.waveform_routes import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def waveform_request(values, **spectrum):
    omitted = {"omit": True, "omission_reason": "spectrum calibration fixture"}
    return {
        "waveform": {"time_s": [i / len(values) for i in range(len(values))], "current_a": values},
        "conduction": omitted,
        "events": {"mode": "omitted", "omission_reason": "spectrum only"},
        "switching": {"turn_on": omitted, "turn_off": omitted},
        "spectrum": {"window": "rectangular", "detrend": "none", **spectrum},
    }


def analyze(client, values, **spectrum):
    response = client.post("/api/waveform/analyze", json=waveform_request(values, **spectrum))
    assert response.status_code == 200, response.text
    return response.json()["spectrum"]


def mask(frequencies, limits, mode="exact"):
    return {"unit": "A_peak", "frequency_hz": frequencies, "limit_peak": limits,
            "interpolation": mode, "source": "independent numerical fixture"}


def band(name, lower, upper, endpoints="closed"):
    return {"name": name, "lower_hz": lower, "upper_hz": upper, "endpoints": endpoints}


def test_legacy_no_analysis_options_reproduce_merged_http_response(client):
    # HTTP 200 from e41cc8a using the same request, with every response field included.
    response = client.post("/api/waveform/analyze", json=waveform_request([1, -1] * 4))
    assert response.status_code == 200, response.text
    canonical = json.dumps(response.json(), sort_keys=True, separators=(",", ":")).encode()
    assert sha256(canonical).hexdigest() == "0bc2ed1d6c1f30dd63ecc8f32db611c4bf3d6aa80c5ee81a6c31a9e84006557d"


def test_two_harmonics_and_dc_have_distinct_peak_and_rms_units(client):
    x = [1 + 2 * math.cos(2 * math.pi * i / 8) + math.cos(4 * math.pi * i / 8)
         for i in range(8)]
    result = analyze(client, x, mask=mask([1, 2], [1.5, 2.0]),
                     bands=[band("harmonics", 1, 2), band("all", 0, 4)])
    screen = result["mask_screening"]
    assert screen["violating_bin_indices"] == [1]
    assert screen["uncovered_bin_indices"] == [0, 3, 4]
    assert screen["bins"][1]["amplitude_peak"] == pytest.approx(2)
    assert screen["bins"][1]["amplitude_peak"] - screen["bins"][1]["limit_peak"] == pytest.approx(0.5)
    assert screen["bins"][1]["margin_peak"] == pytest.approx(-0.5)
    assert screen["bins"][1]["excess_peak"] == pytest.approx(0.5)
    rows = result["band_rms"]
    assert [r["bin_count"] for r in rows] == [2, 5]
    assert [r["unit"] for r in rows] == ["A_rms", "A_rms"]
    assert [r["rms"] for r in rows] == pytest.approx([math.sqrt(2.5), math.sqrt(3.5)])


def test_strict_equality_and_a_harmonic_below_the_display_limit(client):
    # One nonzero sample gives the same real coefficient in all DFT bins.
    result = analyze(client, [16] + [0] * 31, mask=mask([0, 1, 15, 16], [0.5, 1, 0.75, 0.5]))
    screen = result["mask_screening"]
    assert 15 not in result["selected_bin_indices"]
    assert screen["violating_bin_indices"] == [15]
    assert [screen["bins"][i]["status"] for i in [0, 1, 16]] == ["pass"] * 3
    assert screen["bins"][16]["amplitude_peak"] == 0.5
    assert screen["bins"][15]["amplitude_peak"] == 1.0


def test_equal_row_counts_do_not_join_frequencies_by_position(client):
    result = analyze(client, [1] * 8, mask=mask([0.25, 1.25, 2.25, 3.25, 4], [0] * 5))
    screen = result["mask_screening"]
    assert screen["uncovered_bin_indices"] == [0, 1, 2, 3]
    assert screen["bins"][4]["status"] == "pass"
    assert screen["bins"][4]["amplitude_peak"] == 0
    assert not screen["coverage_complete"]


def test_returned_peak_amplitudes_used_as_exact_limits_all_pass(client):
    # The DC magnitude is 9/7 A. Separate evaluation orders can differ by one ULP.
    current = [-2, 5, -9, -6, -7, 6, 4]
    original = analyze(client, current)
    result = analyze(client, current, mask=mask(original["frequency_hz"], original["amplitude_a"]))
    assert result["amplitude_a"] == original["amplitude_a"]
    screen = result["mask_screening"]
    assert screen["coverage_complete"]
    assert screen["violating_bin_indices"] == []
    assert [row["amplitude_peak"] for row in screen["bins"]] == original["amplitude_a"]
    assert all(row["status"] == "pass" and row["margin_peak"] == 0 for row in screen["bins"])


def test_null_mask_splits_interpolation_without_extrapolation(client):
    result = analyze(client, [1] * 32, mask=mask([1, 3, 5, 7, 9], [1, 3, None, 7, 9], "linear-frequency"))
    rows = result["mask_screening"]["bins"]
    assert [i for i, row in enumerate(rows) if row["status"] == "uncovered"] == [0, 4, 5, 6, *range(10, 17)]
    assert [(rows[i]["coverage"], rows[i]["limit_peak"]) for i in (2, 8)] == [
        ("interpolated", 2.0), ("interpolated", 8.0)]
    assert rows[4]["limit_peak"] is None
    assert rows[4]["margin_peak"] is None
    assert rows[4]["excess_peak"] is None


def test_odd_final_bin_is_interior_and_phase_does_not_change_amplitude(client):
    for phi in (0.0, math.pi / 2):
        x = [2 * math.cos(2 * math.pi * 3 * n / 7 + phi) for n in range(7)]
        result = analyze(client, x, mask=mask([0, 3], [1, 1.5]), bands=[band("last", 2.5, 3.5)])
        screen = result["mask_screening"]
        assert screen["violating_bin_indices"] == [3]
        assert screen["bins"][3]["amplitude_peak"] == pytest.approx(2)
        assert result["band_rms"][0]["rms"] == pytest.approx(math.sqrt(2))


def test_hann_mask_uses_coherent_gain_but_band_rms_uses_the_windowed_signal(client):
    # The symmetric five-sample Hann window is exactly [0, 1/2, 1, 1/2, 0].
    options = {"window": "hann", "mask": mask([0, 2], [1, 1]),
               "bands": [band("dc", 0, 0.25), band("all", 0, 2)]}
    result = analyze(client, [1] * 5, **options)
    dc = result["mask_screening"]["bins"][0]
    assert dc["amplitude_peak"] == pytest.approx(1)
    assert dc["status"] == "pass"
    assert [r["rms"] for r in result["band_rms"]] == pytest.approx([0.4, math.sqrt(0.3)])
    detrended = analyze(client, [1] * 5, detrend="mean", **options)
    assert [r["rms"] for r in detrended["band_rms"]] == [0, 0]


def test_tiny_nyquist_rms_survives_squaring_and_missing_bins_remain_distinct(client):
    result = analyze(client, [1e-200, -1e-200] * 4,
                     bands=[band("last", 3.5, 4), band("empty", 1.1, 1.9), band("dc", 0, 0.25)])
    last, empty, dc = result["band_rms"]
    assert last["rms"] > 0 and math.isclose(last["rms"], 1e-200, rel_tol=1e-12)
    assert last["bin_count"] == 1
    assert empty["status"] == "no-bins" and empty["rms"] is None
    assert empty["bin_count"] == 0
    assert dc["status"] == "computed" and dc["rms"] == 0


def test_mask_only_rejects_nonzero_spectral_peaks_that_underflow_to_zero(client):
    # All four DFT coefficients of the impulse are nonzero before one-sided normalization.
    values = [math.ulp(0.0), 0, 0, 0]
    payload = waveform_request(values, mask=mask([0, 1, 2], [0, 0, 0]))
    response = client.post("/api/waveform/analyze", json=payload)
    assert response.status_code == 422, response.text


def test_band_endpoint_conventions_select_bins_by_frequency(client):
    conventions = ["closed", "left-closed", "right-closed", "open"]
    result = analyze(client, [16] + [0] * 31, bands=[band(kind, 1, 2, kind) for kind in conventions])
    rows = result["band_rms"]
    assert [(r["first_bin_index"], r["last_bin_index"]) for r in rows] == [
        (1, 2), (1, 1), (2, 2), (None, None)]
    assert [r["bin_count"] for r in rows] == [2, 1, 1, 0]


@pytest.mark.parametrize("wrong", [
    mask([1, 1], [1, 2]),
    mask([2, 1], [1, 2]),
    mask([0, 1], [1, 2]) | {"unit": "V_peak"},
    mask([0, 1], [1, 2], "log-frequency"),
])
def test_ambiguous_or_incompatible_limits_are_rejected(client, wrong):
    response = client.post("/api/waveform/analyze", json=waveform_request([1] * 8, mask=wrong))
    assert response.status_code == 422, response.text


def test_gate_feature_and_spectrum_keep_the_previous_combined_http_response(client):
    # HTTP 200 from the gap-1 service at 69442d4.
    payload = waveform_request([1, -1] * 4)
    payload["waveform"]["gate_v"] = [0, 8, 4, 6, 4, 2, 10, 10]
    payload["events"] = {"mode": "hysteresis", "digitizer": {
        "thresholds": {"scale": "absolute", "low_v": 2.5, "high_v": 7.5}}}
    response = client.post("/api/waveform/analyze", json=payload)
    assert response.status_code == 200, response.text
    canonical = json.dumps(response.json(), sort_keys=True, separators=(",", ":")).encode()
    assert sha256(canonical).hexdigest() == "0b870a33042f16789b94d3199eeb5dd424a18d9669948077a48ad2a59b04d3e8"


def test_log_frequency_interpolates_peak_amperes_at_the_geometric_midpoint(client):
    result = analyze(client, [16] + [0] * 31, mask=mask([1, 4], [0, 2], "log-frequency"))
    middle = result["mask_screening"]["bins"][2]
    assert middle["limit_peak"] == pytest.approx(1.0)
    assert middle["coverage"] == "interpolated"
    assert middle["status"] == "pass"


def test_mask_and_band_work_limits_are_checked_at_public_transport(client):
    too_many_points = mask(list(range(5002)), [0] * 5002)
    too_many_bands = [band(str(i), 0, 1) for i in range(33)]
    for options in ({"mask": too_many_points}, {"bands": too_many_bands}):
        response = client.post("/api/waveform/analyze", json=waveform_request([1] * 8, **options))
        assert response.status_code == 422, response.text


def test_band_power_matches_independent_direct_complex_sums(client):
    # An arbitrary odd-length trace exercises real and imaginary coefficients after mean subtraction.
    samples = [2, -1, 4, 0, 1, -3, 5]
    n = len(samples)
    windows = [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
    mean = math.fsum(samples) / n
    windowed = [(x - mean) * w for x, w in zip(samples, windows, strict=True)]
    power = 0.0
    for k in [1, 2]:
        real = math.fsum(y * math.cos(2 * math.pi * k * j / n) for j, y in enumerate(windowed))
        imag = math.fsum(y * math.sin(2 * math.pi * k * j / n) for j, y in enumerate(windowed))
        power += 2 * (real ** 2 + imag ** 2) / n ** 2
    result = analyze(client, samples, window="hann", detrend="mean", bands=[band("inner", 0.9, 2.1)])
    assert result["band_rms"][0]["rms"] == pytest.approx(math.sqrt(power), rel=1e-12)
    assert result["analysis_basis"]["window_power_gain"] == pytest.approx(math.fsum(w*w for w in windows) / n)


def test_exact_frequency_match_cannot_round_a_different_json_coordinate_into_place(client):
    data = waveform_request([1, -1] * 4, mask=mask([0, 4], [0, 0]))
    raw = json.dumps(data).replace('"frequency_hz": [0, 4]', '"frequency_hz": [0, 3.99999999999999999999]')
    response = client.post("/api/waveform/analyze", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
