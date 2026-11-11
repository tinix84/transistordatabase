"""Independent phasor, statistical, export, and compatibility fixtures."""
from copy import deepcopy
import csv
import io
import json
import math

import numpy as np
import pytest

from transistordatabase.grid_analysis import AnalysisResourceLimitError, evaluate_grid
from transistordatabase.gui_web.backend.analysis_routes import GridRequest
from transistordatabase.gui_web.backend.waveform_routes import Spectrum
from transistordatabase.waveform_losses import (
    series_spectrum, waveform_scalar_statistics, waveform_spectrum,
)


def grid_fixture():
    """Declare all four physical series without switching or direction assumptions."""
    omission = {"omit": True, "omission_reason": "Independent harmonic fixture"}
    return {
        "configurations": [{"id": "config", "source": "=independent source",
            "conduction": deepcopy(omission),
            "switching": {"turn_on": deepcopy(omission), "turn_off": deepcopy(omission)}}],
        "operating_points": [{"id": "point", "source": "Independent physical samples",
            "weight": 1, "waveform": {"time_s": [0, 1, 2, 3],
                "current_a": [2, 2, 2, -2], "gate_v": [0, 2, 0, 2],
                "device_voltage_v": [-2, -2, -2, -2], "bus_voltage_v": 400},
            "events": {"mode": "omitted", "omission_reason": "No switching heat"},
            "spectrum": {"window": "rectangular", "detrend": "none",
                         "relative_threshold": .5}}],
        "metric": {"name": "total_power_w", "unit": "W", "direction": "minimize"},
        "grouped_series_exports": True}


def test_four_sample_phasors_and_strict_half_threshold():
    """Check independently derived phasors and the strict threshold boundary."""
    spectrum = waveform_spectrum([0, 1, 2, 3], [2, 2, 2, -2], relative_threshold=.5)
    assert spectrum["amplitude_a"] == pytest.approx([1, 2, 1])
    assert spectrum["real_a"] == pytest.approx([1, 0, 1])
    assert spectrum["imaginary_a"] == pytest.approx([0, -2, 0])
    assert spectrum["phase_rad"] == pytest.approx([0, -math.pi / 2, 0])
    assert spectrum["selected_bin_indices"] == [1]
    assert spectrum["frequency_hz"][1] == .25
    shifted = waveform_spectrum([11, 12, 13, 14], [2, 2, 2, -2], relative_threshold=.5)
    assert shifted["phase_rad"] == spectrum["phase_rad"]
    boundary = waveform_spectrum([0, 1], [13, 10], relative_threshold=0.13043478260869562)
    assert boundary["selected_bin_indices"] == [0, 1]


@pytest.mark.parametrize("threshold", [True, False, -1, 1.01, math.nan, math.inf, "0.1"])
def test_threshold_rejects_invalid_and_boolean_values(threshold):
    """Reject nonnumeric, nonfinite, boolean, and out-of-range thresholds."""
    with pytest.raises(ValueError):
        waveform_spectrum([0, 1], [0, 2], relative_threshold=threshold)
    with pytest.raises(ValueError):
        Spectrum.model_validate({"window": "rectangular", "detrend": "none",
                                 "relative_threshold": threshold})


def test_dc_nyquist_odd_last_and_zero_phase():
    """Keep signed DC, odd-bin doubling, and null zero phase explicit."""
    negative = waveform_spectrum([0, 1], [-2, -2])
    assert negative["real_a"] == [-2, 0]
    assert negative["phase_rad"] == [math.pi, None]
    assert negative["selected_bin_indices"] == [0]
    odd = waveform_spectrum([0, 1, 2], [1, -.5, -.5])
    assert odd["amplitude_a"] == pytest.approx([0, 1])
    for threshold in (0, 1):
        assert waveform_spectrum([0, 1], [0, 0], relative_threshold=threshold)["selected_bin_indices"] == []
    assert waveform_spectrum([0, 1], [-2, -2], relative_threshold=1)["selected_bin_indices"] == []


def test_first_ten_bins_use_frequency_order_and_hann_parseval():
    """Check frequency order, coherent gain, and Parseval evidence."""
    # A unit impulse has equal nonzero interior peaks, independent of bin order.
    spectrum = waveform_spectrum(list(range(32)), [1] + [0] * 31, relative_threshold=0)
    assert spectrum["selected_bin_indices"] == list(range(10))
    time = np.arange(65) * .01
    values = 3 + np.sin(2 * np.pi * np.arange(65) * 4 / 65)
    hann = series_spectrum(time, values, window="hann", detrend="mean")
    assert hann["coherent_gain"] == pytest.approx(np.mean(np.hanning(65)))
    assert hann["parseval_rms"] == pytest.approx(hann["windowed_discrete_rms"])
    assert not any(key.endswith("_a") for key in hann)


def test_discrete_and_piecewise_linear_statistics_are_separate_and_stable():
    """Compare separate statistical definitions at normal and extreme scales."""
    stats = waveform_scalar_statistics([0, 1], [0, 2])
    assert stats["arithmetic_mean"] == 1
    assert stats["sample_standard_deviation"] == pytest.approx(math.sqrt(2))
    assert stats["discrete_rms"] == pytest.approx(math.sqrt(2))
    assert stats["rms"] == pytest.approx(2 / math.sqrt(3))
    assert "interval_derivative" not in stats
    tiny = waveform_scalar_statistics([0, 1], [0, 2e-250])
    assert tiny["discrete_rms"] == pytest.approx(math.sqrt(2) * 1e-250, abs=0)
    huge = waveform_scalar_statistics([0, 1], [0, 1e308])
    assert math.isfinite(huge["sample_standard_deviation"])


def test_grouped_projection_reuses_current_result_and_emits_raw_csv():
    """Preserve units, scalar bus alignment, hashes, and CSV numeric signs."""
    request = grid_fixture()
    before = deepcopy(request)
    response = evaluate_grid(request)
    assert request == before
    projection = response["grouped_series_exports"]
    assert json.loads(projection["time_json"]) == projection["time_rows"]
    assert json.loads(projection["frequency_json"]) == projection["frequency_rows"]
    assert len(projection["time_rows"]) == 4
    assert {row["series"]: row["unit"] for row in projection["time_rows"]} == {
        "current_a": "A", "gate_v": "V", "device_voltage_v": "V", "bus_voltage_v": "V"}
    current = next(r for r in projection["frequency_rows"] if r["series"] == "current_a")
    assert (current["bin_index"], current["frequency_hz"], current["real"], current["imaginary"]) == (1, .25, 0, -2)
    assert current["phase_rad"] == pytest.approx(-math.pi / 2)
    assert current["grid_input_sha256"] == response["input_sha256"]
    assert current["input_sha256"] == response["rows"][0]["input_sha256"]
    bus = next(r for r in projection["time_rows"] if r["series"] == "bus_voltage_v")
    assert bus["minimum"] == bus["maximum"] == bus["discrete_rms"] == 400
    parsed = list(csv.DictReader(io.StringIO(projection["frequency_csv"])))
    assert parsed[0]["configuration_source"] == "'=independent source"
    assert any(row["real"] == "-2.0" for row in parsed)
    assert projection["null_csv"] == "NULL"


def test_failed_unrequested_and_empty_spectra_have_null_bin_manifests():
    """Distinguish unsupported, unrequested, and computed-empty FFT outputs."""
    request = grid_fixture()
    absent = deepcopy(request["operating_points"][0])
    absent.update(id="absent", weight=0)
    absent.pop("spectrum")
    failed = deepcopy(request["operating_points"][0])
    failed.update(id="failed", weight=0)
    failed["waveform"]["time_s"] = [0, 1, 2, 4]
    empty = deepcopy(request["operating_points"][0])
    empty.update(id="empty", weight=0)
    empty["spectrum"]["relative_threshold"] = 1
    request["operating_points"].extend([absent, failed, empty])
    projection = evaluate_grid(request)["grouped_series_exports"]
    for point, status in (("absent", "not_requested"), ("failed", "unsupported"), ("empty", "computed")):
        rows = [r for r in projection["frequency_rows"] if r["operating_point_id"] == point]
        assert len(rows) == 4
        assert all(row["status"] == status and row["bin_index"] is None for row in rows)


def test_absent_default_fields_preserve_canonical_inputs_and_hashes():
    """Keep legacy request serialization and hashes unchanged by new defaults."""
    request = grid_fixture()
    request.pop("grouped_series_exports")
    request["operating_points"][0]["spectrum"].pop("relative_threshold")
    typed = GridRequest.model_validate(request).model_dump()
    assert "grouped_series_exports" not in typed
    assert "relative_threshold" not in typed["operating_points"][0]["spectrum"]
    legacy = evaluate_grid(typed)
    typed["grouped_series_exports"] = False
    explicit_false = evaluate_grid(typed)
    assert explicit_false["input_sha256"] == legacy["input_sha256"]
    assert "grouped_series_exports" not in explicit_false


def test_repeated_multibyte_metadata_budget_precedes_projection(monkeypatch):
    """Reject repeated escaped metadata before projected numerical work."""
    from transistordatabase import grouped_series
    request = grid_fixture()
    request["configurations"][0]["source"] = '界"' * 1000
    request["operating_points"] *= 8
    request["operating_points"] = [dict(point, id=str(index)) for index, point in enumerate(request["operating_points"])]
    monkeypatch.setattr("transistordatabase.grid_analysis.MAX_RESPONSE_BYTES", 200000)
    def must_not_compute(*args, **kwargs):
        raise AssertionError("Projection allocated before its metadata budget")
    monkeypatch.setattr(grouped_series, "waveform_scalar_statistics", must_not_compute)
    with pytest.raises(AnalysisResourceLimitError):
        evaluate_grid(request)
    assert json.dumps(request)
