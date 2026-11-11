"""Bounded all-bin peak masks and raw-windowed discrete band RMS."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction
import math
from numbers import Real

_UNITS = {"A": ("A_peak", "A_rms"), "V": ("V_peak", "V_rms")}
_ENDPOINTS = {"closed", "open", "left-closed", "right-closed"}


def _number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be an actual finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} exceeds the finite numeric range") from error
    if not math.isfinite(result) or (result == 0 and value != 0) or (nonnegative and result < 0):
        raise ValueError(f"{label} exceeds the supported finite range")
    return result


def _coordinate(value, label):
    result = _number(value, label, nonnegative=True)
    if value != result:
        raise ValueError(f"{label} exceeds accepted binary64 coordinate precision")
    return result


def _text(value, label, maximum=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"Supply a nonblank {label} of at most {maximum} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must be UTF-8 scalar text") from error
    return value.strip()


def _rescale(value, scale, denominator, label):
    if value == 0 or scale == 0:
        return 0.0
    a, ea = math.frexp(value)
    b, eb = math.frexp(scale)
    c, ec = math.frexp(denominator)
    try:
        result = math.ldexp(a * b / c, ea + eb - ec)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds the finite SI range") from error
    if not math.isfinite(result) or result == 0:
        raise ValueError(f"{label} exceeds the nonzero finite SI range")
    return result


def _interpolate(left, right, weight):
    exact_weight = weight if isinstance(weight, Fraction) else Fraction.from_float(weight)
    exact = (1 - exact_weight) * Fraction.from_float(left) + exact_weight * Fraction.from_float(right)
    value = float(exact)
    if not math.isfinite(value) or (value == 0 and exact != 0):
        raise ValueError("Interpolated mask limit exceeds the nonzero finite SI range")
    return value


def _log_ratio(high, low):
    ratio = (high - low) / low
    return math.log1p(ratio) if math.isfinite(ratio) else math.log(high) - math.log(low)


def _mask(frequency, peaks, spec, peak_unit):
    if not isinstance(spec, dict) or set(spec) - {"unit", "frequency_hz", "limit_peak", "interpolation", "source"}:
        raise ValueError("Supply only the declared peak-mask fields")
    if spec.get("unit") != peak_unit:
        raise ValueError(f"Mask unit must match the engine quantity {peak_unit}")
    source = _text(spec.get("source"), "mask source")
    x, limits = spec.get("frequency_hz"), spec.get("limit_peak")
    if not isinstance(x, (list, tuple)) or not isinstance(limits, (list, tuple)) or not 1 <= len(x) <= 5001 or len(x) != len(limits):
        raise ValueError("A mask needs 1 to 5001 aligned frequency and limit entries")
    x = [_coordinate(value, "mask frequency") for value in x]
    if any(right <= left for left, right in zip(x, x[1:], strict=False)):
        raise ValueError("Mask frequencies must be strictly increasing without duplicate or collapsed coordinates")
    limits = [None if value is None else _number(value, "mask limit", nonnegative=True) for value in limits]
    interpolation = spec.get("interpolation", "exact")
    if interpolation not in {"exact", "linear-frequency", "log-frequency"}:
        raise ValueError("Select exact, linear-frequency or log-frequency mask interpolation")
    if interpolation == "log-frequency" and any(
            x[i] <= 0 and limits[i] is not None and limits[i + 1] is not None
            for i in range(len(x) - 1)):
        raise ValueError("Log-frequency segments need positive endpoints. Isolate DC with a null gap or use exact mode")
    rows, failing, uncovered = [], [], []
    for index, (hz, amplitude) in enumerate(zip(frequency, peaks, strict=True)):
        position = bisect_left(x, hz)
        limit, coverage = None, "uncovered"
        if position < len(x) and x[position] == hz:
            limit = limits[position]
            if limit is not None:
                coverage = "exact"
        elif interpolation != "exact" and 0 < position < len(x):
            low, high = position - 1, position
            if limits[low] is not None and limits[high] is not None:
                if interpolation == "linear-frequency":
                    weight = ((Fraction.from_float(hz) - Fraction.from_float(x[low])) /
                              (Fraction.from_float(x[high]) - Fraction.from_float(x[low])))
                else:
                    left_distance = Fraction.from_float(_log_ratio(hz, x[low]))
                    right_distance = Fraction.from_float(_log_ratio(x[high], hz))
                    weight = left_distance / (left_distance + right_distance)
                limit = _interpolate(limits[low], limits[high], weight)
                coverage = "interpolated"
        status = "uncovered" if limit is None else "fail" if amplitude > limit else "pass"
        if status == "fail":
            failing.append(index)
        elif status == "uncovered":
            uncovered.append(index)
        rows.append({"bin_index": index, "frequency_hz": hz, "amplitude_peak": amplitude,
                     "limit_peak": limit, "margin_peak": None if limit is None else limit - amplitude,
                     "excess_peak": None if limit is None else max(amplitude - limit, 0.0),
                     "status": status, "coverage": coverage})
    return {"method": "coordinate-peak-mask-v1", "unit": peak_unit, "source": source,
            "interpolation": interpolation, "comparison": "strict amplitude > limit; equality passes",
            "margin_convention": "Limit minus observed amplitude; negative means violation",
            "excess_convention": "Maximum of observed amplitude minus limit and zero",
            "coordinate_basis": "Exact accepted binary64 FFT frequency coordinates; no nearest-bin or row-position matching",
            "status": "fail" if failing else "uncovered" if len(uncovered) == len(rows) else "incomplete" if uncovered else "pass",
            "coverage_complete": not uncovered, "violating_bin_indices": failing,
            "uncovered_bin_indices": uncovered, "covered_bin_count": len(rows) - len(uncovered),
            "observed_bin_count": len(rows), "profile_point_count": len(x), "bins": rows,
            "scope": "Only observed FFT bins are screened. Uncovered bins remain unknown. This is not physical EMC compliance."}


def _rms(coefficients, multiplicity, amplitude_scale, count):
    scale = max((max(abs(value.real), abs(value.imag)) for value in coefficients), default=0)
    if scale == 0 or amplitude_scale == 0:
        return 0.0
    terms = []
    for value, weight in zip(coefficients, multiplicity, strict=True):
        root = math.sqrt(weight)
        terms.extend((value.real / scale * root, value.imag / scale * root))
    normalized = math.hypot(*terms)
    return _rescale(scale, amplitude_scale, count / normalized, "Band RMS")


def analyze_spectrum_bins(frequency_hz, coefficients, *, sample_count, sample_interval_s,
                          amplitude_scale, peak_amplitudes, coherent_gain, window_power_gain, signal_unit,
                          window, detrend, mask=None, bands=None):
    """Analyze one existing normalized DFT without recomputing or filtering bins.

    Coefficients are the windowed DFT of samples divided by amplitude_scale.
    Masks use the engine-provided canonical one-sided peaks without recomputation.
    RMS uses the raw windowed
    DFT with one-sided multiplicity, without implicit power-gain correction.
    """
    if type(sample_count) is not int or not 2 <= sample_count <= 10000:
        raise ValueError("Spectrum analysis supports 2 to 10000 original samples")
    if signal_unit not in _UNITS:
        raise ValueError("Declare the engine signal unit A or V")
    frequency = [_number(value, "FFT frequency", nonnegative=True) for value in frequency_hz]
    coefficients = list(coefficients)
    if len(frequency) != sample_count // 2 + 1 or len(coefficients) != len(frequency):
        raise ValueError("All one-sided DFT bins must align with their frequency coordinates")
    if any(right <= left for left, right in zip(frequency, frequency[1:], strict=False)):
        raise ValueError("FFT frequencies are not distinct in the supported numeric range")
    if any(not math.isfinite(value.real) or not math.isfinite(value.imag) for value in coefficients):
        raise ValueError("DFT coefficients must be finite")
    amplitude_scale = _number(amplitude_scale, "amplitude scale", nonnegative=True)
    dt = _number(sample_interval_s, "sample interval")
    gain = _number(coherent_gain, "coherent gain")
    power_gain = _number(window_power_gain, "window power gain")
    if dt <= 0 or gain <= 0 or power_gain <= 0:
        raise ValueError("Sample interval and window gains must be positive")
    nyquist = _number(0.5 / dt, "physical Nyquist frequency")
    multiplicity = [1 if i == 0 or (sample_count % 2 == 0 and i == len(frequency) - 1) else 2
                    for i in range(len(frequency))]
    peak_unit, rms_unit = _UNITS[signal_unit]
    peaks = [_number(value, "canonical peak amplitude", nonnegative=True)
             for value in peak_amplitudes]
    if len(peaks) != len(frequency):
        raise ValueError("Canonical peak amplitudes must align with every FFT bin")
    if amplitude_scale > 0 and any(value != 0 and peak == 0
            for value, peak in zip(coefficients, peaks, strict=True)):
        raise ValueError("Canonical peak amplitude loses a nonzero FFT coefficient outside the supported SI range")
    result = {"analysis_basis": {"signal_unit": signal_unit, "peak_unit": peak_unit, "rms_unit": rms_unit,
              "window": window, "detrend": detrend, "coherent_gain": gain,
              "window_power_gain": power_gain, "sample_count": sample_count,
              "observed_bin_count": len(frequency), "physical_nyquist_hz": nyquist,
              "mask_basis": "Coherent-gain-corrected one-sided peak amplitude",
              "band_basis": "Raw windowed detrended discrete Parseval RMS; no power-gain correction",
              "selection_basis": "All observed bins, independent of display threshold or top-ten selection"}}
    if mask is not None:
        result["mask_screening"] = _mask(frequency, peaks, mask, peak_unit)
    if bands is not None:
        if not isinstance(bands, (list, tuple)) or not 1 <= len(bands) <= 32:
            raise ValueError("Declare 1 to 32 frequency bands")
        names, rows = set(), []
        axis = frequency.copy()
        if sample_count % 2 == 0:
            # This bin is the physical Nyquist endpoint despite product rounding.
            axis[-1] = nyquist
        for spec in bands:
            if not isinstance(spec, dict) or set(spec) - {"name", "lower_hz", "upper_hz", "endpoints"}:
                raise ValueError("Supply only the declared frequency-band fields")
            name = _text(spec.get("name"), "band name", 200)
            if name in names:
                raise ValueError("Frequency band names must be unique")
            names.add(name)
            low = _coordinate(spec.get("lower_hz"), "band lower frequency")
            high = _coordinate(spec.get("upper_hz"), "band upper frequency")
            if not 0 <= low <= high <= nyquist:
                raise ValueError("Band bounds must remain inside the physical Nyquist domain")
            endpoints = spec.get("endpoints", "closed")
            if endpoints not in _ENDPOINTS:
                raise ValueError("Select a declared open or closed band endpoint policy")
            first = bisect_left(axis, low) if endpoints in {"closed", "left-closed"} else bisect_right(axis, low)
            stop = bisect_right(axis, high) if endpoints in {"closed", "right-closed"} else bisect_left(axis, high)
            stop = max(first, stop)
            count = stop - first
            rms = _rms(coefficients[first:stop], multiplicity[first:stop], amplitude_scale, sample_count) if count else None
            rows.append({"name": name, "lower_hz": low, "upper_hz": high, "endpoints": endpoints,
                         "status": "computed" if count else "no-bins", "rms": rms, "unit": rms_unit,
                         "bin_count": count, "first_bin_index": first if count else None,
                         "last_bin_index": stop - 1 if count else None})
        result["band_rms"] = rows
        result["analysis_basis"]["band_overlap_policy"] = "Bands are independent and may overlap. Their RMS values are not additive."
    return result
