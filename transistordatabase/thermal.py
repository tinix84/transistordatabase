"""Explicit steady paths and fixed-case Foster heat response, in SI units.

No data lookup, device defaults or desktop dependencies live in this module.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math

import numpy as np


class ThermalResourceLimitError(ValueError):
    """An exact request exceeds the documented work budget."""


def _number(value, name, minimum=None, positive=False):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0 or minimum is not None and result < minimum:
        raise ValueError(f"{name} is outside its physical domain")
    return result


def _finite(value, label):
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} exceeds finite SI representation")
    return result


def _source(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonblank source is required")
    return value.strip()


def _vector(value, name, positive=False, minimum=None):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.ndim(value) != 1:
        raise ValueError(f"{name} needs a one-dimensional array")
    if not 1 <= len(value) <= 10000:
        raise ValueError(f"{name} requires 1 through 10000 values")
    return np.asarray(
        [_number(v, name, minimum, positive) for v in value], dtype=np.longdouble
    )


def steady_junction(power_w, resistance_k_w, sink_temperature_c, source):
    """Return one junction temperature for an explicit fixed sink and path.

    A sourced zero resistance is an ideal path. Missing paths have no fallback.
    """
    power = _number(power_w, "junction power", minimum=0)
    resistance = _number(resistance_k_w, "junction resistance", minimum=0)
    temperature = _number(sink_temperature_c, "sink temperature")
    _source(source)
    return _finite(_steady_temperature(power, resistance, temperature), "junction temperature")


def _steady_temperature(power, resistance, boundary):
    """Keep shared steady-path arithmetic in extended intermediate precision."""
    return np.longdouble(boundary) + np.longdouble(power) * resistance


def foster_network(model):
    """Validate complete positive R/tau (or R/C) poles without inferred poles."""
    r = _vector(model.get("r_th_vector"), "R poles", positive=True)
    tau_values = model.get("tau_vector")
    c_values = model.get("c_th_vector")
    if tau_values is None and c_values is None:
        raise ValueError(
            "Transient Foster response requires complete R/tau or R/C poles; total R is steady only"
        )
    c = None if c_values is None else _vector(c_values, "C poles", positive=True)
    if c is not None and len(c) != len(r):
        raise ValueError("R/C pole counts must match")
    tau = (
        r * c if tau_values is None else _vector(tau_values, "tau poles", positive=True)
    )
    if len(tau) != len(r) or not np.all(np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError("Complete positive aligned R/tau poles are required")
    if c is not None and tau_values is not None:
        if np.any(np.abs(tau - r * c) > np.longdouble(1e-9) * np.maximum(tau, r * c)):
            raise ValueError("Recorded tau and R*C disagree")
    total = np.sum(r)
    recorded = model.get("r_th_total")
    if recorded is not None:
        recorded = _number(recorded, "recorded total R", positive=True)
        if abs(total - recorded) > np.longdouble(1e-9) * max(total, recorded):
            raise ValueError(
                "Recorded total R disagrees with pole sum (relative tolerance 1e-9)"
            )
    _finite(total, "total thermal resistance")
    return r, tau


def thermal_impedance(model, time_s):
    """Return stable Foster step impedance, preserving scalar/array shape."""
    r, tau = foster_network(model)
    scalar = np.ndim(time_s) == 0
    values = [time_s] if scalar else time_s
    t = _vector(values, "time", minimum=0)
    if len(r) * len(t) > 1000000:
        raise ThermalResourceLimitError(
            "Foster impedance is limited to 1000000 pole/time products"
        )
    with np.errstate(over="ignore", under="ignore"):
        ratios = t[None, :] / tau[:, None]
    if np.any((t[None, :] > 0) & (ratios < np.finfo(np.longdouble).tiny)):
        raise ValueError(
            "Positive time/tau ratio underflows this platform's normal precision range; use a higher-precision runtime or supported numerical range"
        )
    z = np.sum(r[:, None] * -np.expm1(-ratios), axis=0)
    result = np.asarray([_finite(v, "thermal impedance") for v in z])
    return float(result[0]) if scalar else result


def _steady_interfaces(request, ts, pf, pr, rc, rf, rr):
    """Resolve explicit series contacts without merging their physical identities."""
    interfaces = request["interfaces"]
    if request.get("interface") is not None:
        raise ValueError("Select either one interface or the interfaces list")
    if not isinstance(interfaces, (list, tuple)) or not 1 <= len(interfaces) <= 32:
        raise ValueError("Supply 1 to 32 physical interfaces")
    paths = {"common": [], "forward": [], "reverse": []}
    rows, identifiers = [], set()

    def finite_ratio(value, label):
        try:
            result = float(value)
        except OverflowError as error:
            raise ValueError(f"{label} exceeds the finite SI range") from error
        if not math.isfinite(result) or (result == 0 and value != 0):
            raise ValueError(f"{label} exceeds the nonzero finite SI range")
        return result

    for interface in interfaces:
        if not isinstance(interface, dict) or set(interface) - {
                "interface_id", "path", "resistivity_k_m2_w", "area_m2", "source", "conditions"}:
            raise ValueError("Supply only the declared physical interface fields")
        identifier = _source(interface["interface_id"])
        if len(identifier) > 200 or identifier in identifiers:
            raise ValueError("Physical interface IDs must be unique and at most 200 characters")
        identifiers.add(identifier)
        path = interface["path"]
        if path not in paths:
            raise ValueError("Interface path must be common, forward or reverse")
        _source(interface["source"])
        conditions = interface["conditions"]
        if not isinstance(conditions, dict) or any(not isinstance(key, str) for key in conditions):
            raise ValueError("Interface conditions require named numbers or explicit unknowns")
        for value in conditions.values():
            if value is not None:
                _number(value, "interface condition")
        rho = _number(interface["resistivity_k_m2_w"], "interface resistivity", minimum=0)
        area = _number(interface["area_m2"], "interface area", positive=True)
        if rho == 0 and interface["resistivity_k_m2_w"] != 0:
            raise ValueError("Nonzero interface resistivity exceeds the supported SI range")
        resistance = Fraction.from_float(rho) / Fraction.from_float(area)
        resolved = finite_ratio(resistance, "Interface resistance")
        paths[path].append(resistance)
        rows.append({**deepcopy(interface), "resistance_k_w": resolved})
    added = {path: sum(values, Fraction()) for path, values in paths.items()}
    effective = {path: finite_ratio(Fraction.from_float(base) + added[path], "Effective path resistance")
                 for path, base in (("common", rc), ("forward", rf), ("reverse", rr))}
    shared = _steady_temperature(np.longdouble(pf) + pr, effective["common"], ts)
    return {"shared_temperature_c": _finite(shared, "shared temperature"),
            "forward_temperature_c": _finite(_steady_temperature(pf, effective["forward"], shared), "forward temperature"),
            "reverse_temperature_c": _finite(_steady_temperature(pr, effective["reverse"], shared), "reverse temperature"),
            "interface_contributions": rows,
            "added_path_resistances_k_w": {path: finite_ratio(value, "Added path resistance") for path, value in added.items()},
            "path_resistances_k_w": effective,
            "boundary": "steady fixed sink; explicit series contacts on common and independent branch paths; ideal zero paths allowed"}


def steady_paths(request):
    """Two independent branch paths sharing one explicitly supplied sink path."""
    ts = _number(request["sink_temperature_c"], "sink temperature")
    pf = _number(request["forward_power_w"], "forward power", minimum=0)
    pr = _number(request["reverse_power_w"], "reverse power", minimum=0)
    rc = _number(request["common_resistance_k_w"], "common resistance", minimum=0)
    rf = _number(request["forward_resistance_k_w"], "forward resistance", minimum=0)
    rr = _number(request["reverse_resistance_k_w"], "reverse resistance", minimum=0)
    if request.get("interfaces") is not None:
        return _steady_interfaces(request, ts, pf, pr, rc, rf, rr)
    interface = request.get("interface")
    added = 0.0
    if interface is not None:
        rho = _number(
            interface["resistivity_k_m2_w"], "interface resistivity", minimum=0
        )
        area = _number(interface["area_m2"], "interface area", positive=True)
        added = _finite(np.longdouble(rho) / area, "interface resistance")
        branch = interface["path"]
        if branch not in ("common", "forward", "reverse"):
            raise ValueError("Interface path must be common, forward or reverse")
        _source(interface["source"])
        if branch == "common":
            rc += added
        elif branch == "forward":
            rf += added
        else:
            rr += added
    total = np.longdouble(pf) + pr
    shared = _steady_temperature(total, rc, ts)
    tf = _steady_temperature(pf, rf, shared)
    tr = _steady_temperature(pr, rr, shared)
    return {
        "forward_temperature_c": _finite(tf, "forward temperature"),
        "reverse_temperature_c": _finite(tr, "reverse temperature"),
        "interface_resistance_k_w": added,
        "path_resistances_k_w": {"common": rc, "forward": rf, "reverse": rr},
        "boundary": "steady fixed sink; explicit ideal zero paths allowed",
    }


def _moments(h):
    """B0..B3 for exponential convolution; independent small-h series."""
    if h > 1:
        values = [-np.expm1(-h)]
        for k in range(1, 4):
            values.append(1 - k * values[-1] / h)
        return np.asarray(values, dtype=np.longdouble)
    values = []
    for k in range(4):
        term = h / (k + 1)
        total = term
        for m in range(100):
            term *= -h / (k + m + 2)
            updated = total + term
            if updated == total:
                break
            total = updated
        values.append(total)
    return np.asarray(values, dtype=np.longdouble)


def _endpoint_coefficients(
    coefficients, position=np.longdouble(1), width=np.longdouble(1)
):
    """Expand the original source about an endpoint before any cut translation."""
    result = []
    try:
        for k in range(4):
            terms = []
            for j in range(k, 4):
                value = _finite(
                    coefficients[j] * position ** (j - k), "endpoint heat term"
                )
                terms.extend([value] * math.comb(j, k))
            result.append(np.longdouble(math.fsum(terms)) * (-width) ** k)
    except OverflowError as exc:
        raise ValueError(
            "Endpoint heat coefficients exceed platform representation"
        ) from exc
    return np.asarray(result, dtype=np.longdouble)


def _convolution(coefficients, h, endpoint=None):
    """Integrate at the resolved endpoint basis, avoiding large-h cancellation."""
    if h <= 1:
        result = np.dot(coefficients, _moments(h))
        if result < 0:
            raise ValueError(
                "Nonnegative heat produced a negative convolution; polynomial precision is unresolved"
            )
        return result
    # P(1-v): compensated sums preserve exact endpoint zeros such as (1-u)^3.
    if endpoint is None:
        endpoint = _endpoint_coefficients(coefficients)
    decay = np.exp(-h)
    moments = [-np.expm1(-h)]
    for k in range(1, 4):
        moments.append((k / h) * moments[-1] - decay)
    for coefficient, moment in zip(endpoint, moments, strict=True):
        if coefficient != 0 and abs(moment) < np.finfo(np.longdouble).tiny:
            raise ValueError(
                "Endpoint forcing moment underflows platform precision; use a higher-precision runtime or supported numerical range"
            )
    result = np.dot(endpoint, np.asarray(moments, dtype=np.longdouble))
    if result < 0:
        raise ValueError(
            "Nonnegative heat produced a negative convolution; polynomial precision is unresolved"
        )
    return result


def _polynomial(coefficients):
    c = _vector(coefficients, "power coefficients")
    if len(c) > 4:
        raise ValueError(
            "Power polynomial must have at most four ascending coefficients"
        )
    c = np.pad(c, (0, 4 - len(c)))
    scale = np.max(np.abs(c))
    if scale == 0:
        scale = np.longdouble(1)
    normalized = c / scale
    roots = np.polynomial.polynomial.polyroots(
        np.asarray([normalized[1], 2 * normalized[2], 3 * normalized[3]], dtype=float)
    )
    points = [np.longdouble(0), np.longdouble(1)]
    points += [
        np.longdouble(v.real) for v in roots if abs(v.imag) <= 1e-12 and 0 < v.real < 1
    ]
    tolerance = np.longdouble(32 * np.finfo(float).eps) * np.sum(np.abs(normalized))
    if any(
        np.polynomial.polynomial.polyval(u, normalized) < -tolerance for u in points
    ):
        raise ValueError("Power polynomial is negative inside its interval")
    return c


def simulate_foster(request):
    """Exact cubic forcing at a fixed case, with explicit event deposition."""
    model = request["network"]
    _source(model["source"])
    owner = request["network_heat_owner"]
    if owner not in ("forward", "reverse", "shared-junction"):
        raise ValueError(
            "Declare the physical network heat owner: forward, reverse or shared-junction"
        )
    r, tau = foster_network(model)
    if len(r) > 100:
        raise ThermalResourceLimitError(
            "Transient response is limited to 100 Foster poles"
        )
    case = _number(request["case_temperature_c"], "case temperature")
    initial = request["initial_state"]
    if isinstance(initial, str) and initial == "cold":
        state = np.zeros(len(r), dtype=np.longdouble)
    elif isinstance(initial, str):
        raise ValueError("Initial state must be cold or one temperature rise per pole")
    else:
        state = _vector(initial, "initial pole rises", minimum=0)
        if len(state) != len(r):
            raise ValueError("Initial state needs one rise per Foster pole")
    raw_segments = request["segments"]
    if not raw_segments:
        raise ValueError("At least one explicit power segment is required")
    if len(raw_segments) > 10000:
        raise ThermalResourceLimitError(
            "Transient response is limited to 10000 exact segments"
        )
    events = request.get("events", [])
    if len(events) > 10000 or len(raw_segments) * len(events) > 2000000:
        raise ThermalResourceLimitError(
            "Exact event/segment processing exceeds 2000000 products / 10000 events"
        )
    segments = []
    energy = np.longdouble(0)
    previous = None
    for item in raw_segments:
        direction = item.get("direction")
        c = _polynomial(item["power_coefficients_w"])
        explicit_owner = item.get("heat_owner")
        if explicit_owner is not None:
            if explicit_owner not in ("forward", "reverse"):
                raise ValueError("Segment heat owner must be forward or reverse")
            if owner not in (explicit_owner, "shared-junction") and np.any(c != 0):
                raise ValueError("Segment heat owner differs from this Foster junction")
        elif direction in ("positive", "negative") and owner != "shared-junction":
            expected = "forward" if direction == "positive" else "reverse"
            if owner != expected:
                raise ValueError(
                    "Signed segment belongs to another physical junction; split branch heat explicitly"
                )
        start = _number(item["start_s"], "segment start", minimum=0)
        end = _number(item["end_s"], "segment end", minimum=0)
        if end <= start or previous is not None and start < previous:
            raise ValueError(
                "Segments must be ordered, positive-duration and nonoverlapping"
            )
        if previous is not None and start > previous:
            if request["gap_policy"] != "zero-power":
                raise ValueError("Power gaps require explicit zero-power convention")
            segments.append((previous, start, np.zeros(4, dtype=np.longdouble)))
        segment_energy = (np.longdouble(end) - start) * sum(
            c[k] / (k + 1) for k in range(4)
        )
        energy += segment_energy
        segments.append((start, end, c))
        previous = end
    start_time, end_time = segments[0][0], segments[-1][1]
    mode = request["event_mode"]
    if events and mode not in ("impulse", "interval-average"):
        raise ValueError(
            "Switching heat requires explicit impulse or interval-average deposition"
        )
    impulses = {}
    averages = []
    event_energy = np.longdouble(0)
    for event in events:
        t = _number(event["time_s"], "event time", minimum=0)
        e = _number(event["energy_j"], "event energy", minimum=0)
        event_owner = event["heat_owner"]
        if event_owner not in ("forward", "reverse"):
            raise ValueError(
                "Each event, including E(0), requires forward/reverse heat ownership"
            )
        if owner != "shared-junction" and event_owner != owner:
            raise ValueError(
                "Event heat owner differs from this Foster junction; use a separate network or explicitly declare a shared junction"
            )
        _source(event["source"])
        if not start_time <= t <= end_time:
            raise ValueError("Event outside observation interval")
        event_energy += e
        if mode == "impulse":
            impulses[t] = impulses.get(t, np.longdouble(0)) + e
        else:
            end = _number(event["end_s"], "event averaging end", minimum=0)
            if not t < end <= end_time:
                raise ValueError(
                    "Event averaging interval must have positive width within observation"
                )
            averages.append((t, end, np.longdouble(e) / (np.longdouble(end) - t)))
    sample_count = request["samples_per_segment"]
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= 100
    ):
        raise ValueError("samples_per_segment must be an integer from 1 to 100")
    budget = len(segments) * (sample_count + 1) + 4 * len(events)
    if budget > 100000 or len(events) > 10000:
        raise ThermalResourceLimitError(
            "Transient plotting expansion exceeds 100000 samples / 10000 events"
        )
    if budget * max(1, len(averages)) > 2000000 or budget * len(r) > 2000000:
        raise ThermalResourceLimitError(
            "Exact forcing work exceeds 2000000 sample/event or sample/pole products"
        )
    samples = []

    def emit(t, kind="continuous", values=None):
        pole_state = state if values is None else values
        samples.append(
            {
                "time_s": t,
                "temperature_c": _finite(
                    np.longdouble(case) + sum(pole_state), "junction temperature"
                ),
                "pole_rises_k": [_finite(v, "pole rise") for v in pole_state],
                "kind": kind,
            }
        )

    def impulse(t):
        nonlocal state
        if t in impulses:
            emit(t, "pre-event")
            state = state + r * impulses[t] / tau
            emit(t, "post-event")

    emit(start_time)
    impulse(start_time)

    def advance(initial, coefficients, dt, endpoint):
        with np.errstate(over="ignore", under="ignore"):
            h = dt / tau
        if np.any(h < 4 * np.finfo(np.longdouble).tiny):
            raise ValueError(
                "Positive interval/tau ratio underflows this platform's normal moment precision range; use a higher-precision runtime or supported numerical range"
            )
        forcing = np.asarray([_convolution(coefficients, v, endpoint) for v in h])
        updated = np.exp(-h) * initial + r * forcing
        if np.any(updated < 0):
            raise ValueError(
                "Nonnegative heat produced a negative Foster state; numerical precision is unresolved"
            )
        return updated

    for start, end, c in segments:
        # Global reversed source basis, formed once before any physical cuts.
        reversed_source = _endpoint_coefficients(c)
        cuts = {start, end}
        plot_times = sorted(
            {
                float(
                    np.longdouble(start)
                    + (np.longdouble(end) - start) * j / sample_count
                )
                for j in range(1, sample_count)
            }
        )
        cuts.update(t for t in impulses if start < t < end)
        cuts.update(t for a, b, _ in averages for t in (a, b) if start < t < end)
        cuts = sorted(cuts)
        duration = np.longdouble(end) - start

        def source_endpoint(
            target,
            span,
            end=end,
            duration=duration,
            reversed_source=reversed_source,
            c=c,
            start=start,
        ):
            remaining = (np.longdouble(end) - target) / duration
            if remaining <= 0.5:
                return _endpoint_coefficients(reversed_source, remaining, -span)
            return _endpoint_coefficients(
                c, (np.longdouble(target) - start) / duration, span
            )

        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            u = (np.longdouble(a) - start) / duration
            width = (np.longdouble(b) - a) / duration
            # Choose the original basis nearest the interval start; never form
            # tiny (1-u)^3 by subtracting near-unit rounded powers.
            remaining = (np.longdouble(end) - a) / duration
            local = (
                _endpoint_coefficients(reversed_source, remaining, width)
                if remaining <= 0.5
                else _endpoint_coefficients(c, u, -width)
            )
            midpoint = np.longdouble(a) + (np.longdouble(b) - a) / 2
            average_power = sum(p for x, y, p in averages if x <= midpoint < y)
            local[0] += average_power
            # Plot evaluations do not advance the physical state or change final heat.
            initial = state.copy()
            dt = np.longdouble(b) - a
            for t in plot_times:
                if not a < t < b:
                    continue
                fraction = (np.longdouble(t) - a) / dt
                partial = local * np.asarray([fraction**k for k in range(4)])
                endpoint = source_endpoint(t, (np.longdouble(t) - a) / duration)
                endpoint[0] += average_power
                emit(
                    t, values=advance(initial, partial, np.longdouble(t) - a, endpoint)
                )
            endpoint = source_endpoint(b, dt / duration)
            endpoint[0] += average_power
            state = advance(initial, local, dt, endpoint)
            emit(b)
            impulse(b)
    return {
        "samples": samples,
        "final_pole_rises_k": samples[-1]["pole_rises_k"],
        "final_temperature_c": samples[-1]["temperature_c"],
        "sampled_max_temperature_c": max(v["temperature_c"] for v in samples),
        "maximum_scope": "sampled only; no global transient-extrema claim",
        "conduction_energy_j": _finite(energy, "input energy"),
        "switching_energy_j": _finite(event_energy, "switching energy"),
        "event_mode": mode,
        "network_heat_owner": owner,
        "boundary": "one physical junction at fixed case; shared-junction explicitly co-locates both heat owners; no dynamic heatsink cascade",
        "polynomial_negative_tolerance": "32*float64 epsilon*sum(abs(coefficients)) W",
    }


def _loss_at(model, temperature):
    _source(model["source"])
    low = _number(model["temperature_min_c"], "minimum source temperature")
    high = _number(model["temperature_max_c"], "maximum source temperature")
    if high <= low:
        raise ValueError("Temperature source requires increasing coverage")
    if not low <= temperature <= high:
        return None
    if model["mode"] == "linear-temperature-estimate":
        if model.get("selected") is not True:
            raise ValueError("Select the linear temperature estimate explicitly")
        ref = _number(model["reference_temperature_c"], "reference temperature")
        p = _number(model["reference_power_w"], "reference power", minimum=0)
        alpha = _number(model["coefficient_per_k"], "temperature coefficient")
        value = np.longdouble(p) * (1 + np.longdouble(alpha) * (temperature - ref))
    elif model["mode"] == "recorded":
        t = _vector(model["temperature_c"], "recorded temperature")
        p = _vector(model["power_w"], "recorded power", minimum=0)
        if len(t) < 2 or len(t) != len(p) or np.any(np.diff(t) <= 0):
            raise ValueError(
                "Recorded temperature/loss pairs need aligned increasing coverage"
            )
        if low != t[0] or high != t[-1]:
            raise ValueError(
                "Recorded temperature coverage must match its table endpoints"
            )
        value = np.interp(
            temperature, np.asarray(t, dtype=float), np.asarray(p, dtype=float)
        )
    else:
        raise ValueError("Unknown temperature-loss source mode")
    return _number(
        _finite(value, "temperature-dependent loss"),
        "temperature-dependent loss",
        minimum=0,
    )


def electrothermal(request):
    """Bounded fixed-point solution with actual unrelaxed residual evidence."""
    models = request["loss_models"]
    if len(models) not in (1, 2):
        raise ValueError("Supply one or two independent branch loss models")
    sink = _number(request["sink_temperature_c"], "sink temperature")
    common = _number(request["common_resistance_k_w"], "common resistance", minimum=0)
    branch = _vector(request["branch_resistances_k_w"], "branch resistances", minimum=0)
    temperature = _vector(
        request["initial_temperatures_c"], "initial branch temperatures"
    )
    if len(branch) != len(models) or len(temperature) != len(models):
        raise ValueError(
            "One explicit initial temperature and path per branch is required"
        )
    tolerance = _number(request["tolerance_k"], "residual tolerance", positive=True)
    relaxation = _number(request["relaxation"], "relaxation", positive=True)
    maximum = request["max_iterations"]
    if (
        relaxation > 1
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= 10000
    ):
        raise ValueError(
            "Relaxation must be <=1 and max_iterations an integer from 1 to 10000"
        )
    matrix = np.full((len(models), len(models)), common, dtype=np.longdouble) + np.diag(
        branch
    )
    history = []
    status = "iteration-limit"
    accepted = None
    for iteration in range(maximum):
        powers = [
            _loss_at(model, _finite(t, "iterate temperature"))
            for model, t in zip(models, temperature, strict=True)
        ]
        if any(p is None for p in powers):
            history.append(
                {
                    "iteration": iteration,
                    "temperature_c": [
                        _finite(t, "iterate temperature") for t in temperature
                    ],
                    "power_w": None,
                    "residual_k": None,
                    "status": "source-domain-exit",
                }
            )
            status = "source-domain-exit"
            break
        target = sink + matrix @ np.asarray(powers, dtype=np.longdouble)
        residual = target - temperature
        entry = {
            "iteration": iteration,
            "temperature_c": [_finite(t, "iterate temperature") for t in temperature],
            "power_w": powers,
            "residual_k": [_finite(v, "fixed-point residual") for v in residual],
        }
        history.append(entry)
        if max(abs(residual)) <= tolerance:
            accepted = entry
            status = "converged"
            break
        temperature = temperature + relaxation * residual
    return {
        "status": status,
        "accepted_solution": accepted,
        "history": history,
        "tolerance_k": tolerance,
        "boundary": "fixed sink steady; bounded source losses at each branch temperature",
        "nonconvergence_scope": "Iteration failure does not prove no physical solution exists",
    }


def screen_current(request):
    """Explicit workbook-style reference-path screening; never a device rating."""
    maximum = _number(request["junction_limit_c"], "junction limit")
    reference = _number(request["reference_temperature_c"], "reference temperature")
    resistance = _number(
        request["path_resistance_k_w"], "path resistance", positive=True
    )
    v0 = _number(request["threshold_v"], "threshold", minimum=0)
    slope = _number(request["resistance_ohm"], "electrical resistance", minimum=0)
    if request["reference_kind"] not in ("case", "heatsink", "coolant"):
        raise ValueError("Reference kind must be case, heatsink or coolant")
    if maximum < reference:
        raise ValueError("Junction limit is below the selected reference temperature")
    budget = (np.longdouble(maximum) - reference) / resistance
    if slope == 0 and v0 == 0:
        current = None
        status = "no-finite-thermal-current-limit"
    elif budget == 0:
        current, status = 0.0, "screening-estimate"
    elif slope == 0:
        current, status = (
            _finite(budget / v0, "screening current"),
            "screening-estimate",
        )
    else:
        denominator = np.longdouble(v0) + np.sqrt(
            np.longdouble(v0) ** 2 + 4 * np.longdouble(slope) * budget
        )
        current, status = (
            _finite(2 * budget / denominator, "screening current"),
            "screening-estimate",
        )
    return {
        "status": status,
        "power_budget_w": _finite(budget, "power budget"),
        "thermal_screening_current_a": current,
        "package_limit": deepcopy(request.get("package_limit")),
        "scope": "fixed declared conditions; no self-heating solution and no datasheet rating",
    }


def analyze_thermal(request):
    """Canonical reproducible adapter shared by Python, HTTP and the workbench."""
    operations = {
        "steady": steady_paths,
        "transient": simulate_foster,
        "electrothermal": electrothermal,
        "screening": screen_current,
    }
    canonical = deepcopy(request)

    def validate_tree(value):
        if isinstance(value, dict):
            for item in value.values():
                validate_tree(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                validate_tree(item)
        elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
            raise ValueError("Request metadata and conditions must also be finite")

    validate_tree(canonical)
    _source(canonical["source"])
    operation = canonical["operation"]
    if operation not in operations:
        raise ValueError("Unsupported thermal operation")
    result = operations[operation](canonical)
    return {
        "operation": operation,
        "request": canonical,
        "result": result,
        "units": "s, W, J, K/W, J/K, m²; temperatures °C and temperature differences K",
    }
