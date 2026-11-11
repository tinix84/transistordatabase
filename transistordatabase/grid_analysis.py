"""Finite, isolated waveform evaluations with explicit weighted coverage."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from transistordatabase.grouped_series import project_grouped_series
from transistordatabase.grid_merit import build_merit, validate_objectives

from transistordatabase.waveform_losses import (
    WaveformResourceLimitError, _event_indices, _waveform, analyze_waveform,
)

METHOD_VERSION = "waveform-grid-v1"
MAX_PAIRS = 256
MAX_EXPANDED_SAMPLES = 200_000
MAX_RESPONSE_BYTES = 24_000_000
METRICS = {
    "total_power_w": None,
    "conduction_power_w": "conduction",
    "turn_on_power_w": "turn_on",
    "turn_off_power_w": "turn_off",
}


class AnalysisResourceLimitError(ValueError):
    """The complete analysis exceeds its declared resource budget."""


def snapshot(value):
    """Copy finite JSON evidence and reject non-JSON numerical values."""
    return json.loads(json.dumps(value, allow_nan=False))


def input_hash(value):
    """Identify the complete canonical input without depending on dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _identity(items, label):
    if not items:
        raise ValueError(f"Declare at least one {label}")
    ids = []
    for item in items:
        for field in ("id", "source"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Each {label} needs a nonblank {field}")
        ids.append(item["id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"Each {label} needs a unique ID")


def _metrics(result):
    output = {}
    for name, component in METRICS.items():
        value = (result["total_power_w"] if component is None else
                 result["components"][component]["power_w"])
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Power metrics must be finite and nonnegative")
        output[name] = {"value": value, "unit": "W",
                        "status": "available" if value is not None else "unavailable"}
    return output


def _weighted_mean(rows, metric, total_weight):
    """Sum nonnegative products at one exponent before the final SI scaling."""
    denominator, denominator_exponent = math.frexp(total_weight)
    terms = []
    for row in rows:
        value = row["metrics"][metric]["value"]
        if value == 0:
            continue
        magnitude, exponent = math.frexp(value)
        weight, weight_exponent = math.frexp(row["weight"])
        terms.append((magnitude * weight / denominator,
                      exponent + weight_exponent - denominator_exponent))
    if not terms:
        return 0.0
    common_exponent = max(exponent for _, exponent in terms)
    scaled_sum = math.fsum(math.ldexp(magnitude, exponent - common_exponent)
                           for magnitude, exponent in terms)
    result = math.ldexp(scaled_sum, common_exponent)
    if not math.isfinite(result) or result == 0:
        raise ValueError("Weighted mean is outside representable positive SI precision")
    return result


def _waveform_metadata_budget(request, available_bytes, *, handoff=False):
    """Bound repeated model metadata before the waveform engine builds its result.

    Event indices come from the accepted detector. The existing waveform engine
    separately limits numerical segment and plot expansion.
    """
    signals, _ = _waveform(request["waveform"])
    indices = _event_indices(signals, request["events"])
    projected = 4 * len(json.dumps(request, allow_nan=False).encode())
    if indices is not None:
        for name in ("turn_on", "turn_off"):
            component = request["switching"][name]
            if component.get("omit", False):
                continue
            for direction in ("positive", "negative"):
                model = component.get(direction)
                if model is None:
                    continue
                count = sum((signals["current_a"][index] < 0) == (direction == "negative")
                            for index in indices[name])
                metadata = {key: model.get(key) for key in (
                    "source", "mode", "recorded_conditions", "requested_conditions")}
                projected += count * (len(json.dumps(metadata, allow_nan=False).encode()) + 512) * (2 if handoff else 1)
    conduction = request["conduction"]
    if not conduction.get("omit", False):
        models = [model for direction in ("positive", "negative")
                  if (model := conduction.get(direction)) is not None]
        knot_count = sum(len(model.get("current_a", [])) for model in models)
        segment_bound = min(10000, (len(signals["time_s"]) - 1) * (2 + knot_count))
        source_bytes = max((len(json.dumps(model.get("source", "")).encode()) for model in models), default=0)
        projected += segment_bound * source_bytes * (2 if handoff else 1)
    if projected > available_bytes:
        raise AnalysisResourceLimitError("Repeated waveform metadata exceeds the complete response byte budget")


def evaluate_grid(request, *, _evaluator=None):
    """Evaluate immutable pairs through one supported operation, without cached state.

    The private evaluator seam supports independent aggregation fixtures. HTTP
    always uses the registered waveform operation.
    """
    request = snapshot(request)
    grouped_exports = request.get("grouped_series_exports", False)
    if not isinstance(grouped_exports, bool):
        raise ValueError("Grouped series export selection must be boolean")
    if not grouped_exports:
        request.pop("grouped_series_exports", None)
    objectives = request.get("objectives")
    if objectives is None:
        request.pop("objectives", None)
    else:
        objectives = validate_objectives(objectives, METRICS)
    configs, points = request["configurations"], request["operating_points"]
    _identity(configs, "configuration")
    _identity(points, "operating point")
    if len(configs) > 32 or len(points) > 64 or len(configs) * len(points) > MAX_PAIRS:
        raise AnalysisResourceLimitError(f"The grid supports at most {MAX_PAIRS} pairs")
    # Check complete expansion before constructing a Cartesian product.
    sample_count = sum(len(p["waveform"]["time_s"]) for p in points) * len(configs)
    if sample_count > MAX_EXPANDED_SAMPLES:
        raise AnalysisResourceLimitError("Expanded waveform samples exceed 200000")
    if len(json.dumps(request)) > 4_000_000:
        raise AnalysisResourceLimitError("Analysis input exceeds 4 MB")
    weights = [p["weight"] for p in points]
    if any(isinstance(w, bool) or not isinstance(w, (float, int)) or
           not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Operating-point weights must be finite and nonnegative")
    try:
        total_weight = math.fsum(weights)
    except OverflowError as exc:
        raise ValueError("Total operating-point weight overflow") from exc
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Total operating-point weight must be positive and finite")
    objective = request["metric"]
    if (objective["name"] not in METRICS or objective["unit"] != "W" or
            objective["direction"] not in {"minimize", "maximize"}):
        raise ValueError("Select a named power metric in W and its ranking direction")
    mode = request.get("summary_mode", "full")
    if mode not in {"full", "successful-subset"}:
        raise ValueError("Select full or successful-subset summary mode")
    include_zero = request.get("include_zero_weight_in_worst", False)
    if not isinstance(include_zero, bool):
        raise ValueError("Worst-point zero-weight selection must be boolean")
    evaluator = _evaluator or analyze_waveform
    rows, summaries, response_size = [], [], 0
    for config in configs:
        config_rows = []
        for point in points:
            inputs = {"waveform": point["waveform"], "events": point["events"],
                      "conduction": config["conduction"], "switching": config["switching"]}
            for name in ("spectrum", "provenance"):
                if point.get(name) is not None:
                    inputs[name] = point[name]
            evidence = {"configuration": config, "operating_point": point,
                        "waveform_request": inputs, "method_version": METHOD_VERSION}
            row = {"configuration_id": config["id"], "operating_point_id": point["id"],
                   "weight": point["weight"], "inputs": deepcopy(evidence),
                   "input_sha256": input_hash(evidence), "method_version": METHOD_VERSION}
            try:
                _waveform_metadata_budget(inputs, MAX_RESPONSE_BYTES - response_size)
                result = evaluator(deepcopy(inputs))
                row.update(status="computed", result=result, metrics=_metrics(result), error=None)
            except (ValueError, OverflowError, FloatingPointError) as exc:
                # A resource failure is global. Programmer errors remain visible.
                if isinstance(exc, (AnalysisResourceLimitError, WaveformResourceLimitError)):
                    raise AnalysisResourceLimitError(str(exc)) from exc
                row.update(status="unsupported", result=None, metrics={name: {
                    "value": None, "unit": "W", "status": "unavailable"}
                    for name in METRICS}, error=str(exc))
            response_size += len(json.dumps(row, allow_nan=False).encode())
            if response_size > MAX_RESPONSE_BYTES:
                raise AnalysisResourceLimitError("Analysis response exceeds 24 MB")
            rows.append(row)
            config_rows.append(row)
        metric_summaries = {}
        for name in METRICS:
            retained = [r for r in config_rows if r["weight"] > 0 and
                        r["metrics"][name]["value"] is not None]
            retained_weight = math.fsum(r["weight"] for r in retained)
            subset_mean = _weighted_mean(retained, name, retained_weight) if retained_weight else None
            complete = all(r["metrics"][name]["value"] is not None
                           for r in config_rows if r["weight"] > 0)
            candidates = [r for r in config_rows if (include_zero or r["weight"] > 0)
                          and r["metrics"][name]["value"] is not None]
            worst = (max if objective["direction"] == "minimize" else min)(
                (r["metrics"][name]["value"] for r in candidates), default=None)
            metric_summaries[name] = {
                "unit": "W", "full_mean": subset_mean if complete else None,
                "subset_mean": subset_mean, "retained_weight": retained_weight,
                "total_weight": total_weight, "coverage": retained_weight / total_weight,
                "selected_mean": subset_mean if mode == "successful-subset" or complete else None,
                "selected_summary": mode, "complete": complete, "worst_value": worst,
                "worst_point_ids": sorted(r["operating_point_id"] for r in candidates
                                          if r["metrics"][name]["value"] == worst),
                "worst_complete": all(r["metrics"][name]["value"] is not None
                                      for r in config_rows if include_zero or r["weight"] > 0),
                "zero_weight_in_worst": include_zero}
        summaries.append({"configuration_id": config["id"], "metrics": metric_summaries})
    ranking = [{"configuration_id": s["configuration_id"],
                "value": s["metrics"][objective["name"]]["selected_mean"], "unit": "W"}
               for s in summaries if s["metrics"][objective["name"]]["selected_mean"] is not None]
    ranking.sort(key=lambda r: ((-1 if objective["direction"] == "maximize" else 1)
                               * r["value"], r["configuration_id"]))
    output = {"method_version": METHOD_VERSION, "metric": objective, "summary_mode": mode,
            "rows": rows, "summaries": summaries, "ranking": ranking,
            "inputs": request, "input_sha256": input_hash(request),
            "limits": {"max_pairs": MAX_PAIRS, "max_expanded_samples": MAX_EXPANDED_SAMPLES,
                       "max_response_bytes": MAX_RESPONSE_BYTES}}
    if grouped_exports:
        output["grouped_series_exports"] = project_grouped_series(
            rows, output["input_sha256"],
            MAX_RESPONSE_BYTES - len(json.dumps(output, allow_nan=False).encode()) - 32,
            AnalysisResourceLimitError)
    if objectives is not None:
        output["merit"] = build_merit(
            rows, summaries, points, objectives, output["input_sha256"],
            MAX_RESPONSE_BYTES - len(json.dumps(output, allow_nan=False).encode()) - 16,
            AnalysisResourceLimitError, input_hash, _weighted_mean)
    if len(json.dumps(output, allow_nan=False).encode()) > MAX_RESPONSE_BYTES:
        raise AnalysisResourceLimitError("Analysis response exceeds 24 MB")
    return output


def thermal_handoff(request):
    """Stage exact known heat for an explicitly selected physical junction.

    A sign selects a user-declared owner. It does not identify a physical die.
    The caller must supply the thermal network and boundary separately.
    """
    request = snapshot(request)
    owners = request["direction_owners"]
    if set(owners) != {"positive", "negative", "zero"} or any(
            value not in {"forward", "reverse"} for value in owners.values()):
        raise ValueError("Declare the physical heat owner for each current direction")
    network_owner = request["network_heat_owner"]
    if network_owner not in {"forward", "reverse", "shared-junction"}:
        raise ValueError("Declare the physical thermal network owner")
    if request["event_mode"] != "impulse":
        raise ValueError("This handoff supports explicit impulse deposition only")
    if len(json.dumps(request, allow_nan=False).encode()) > 4_000_000:
        raise AnalysisResourceLimitError("Analysis input exceeds 4 MB")
    _waveform_metadata_budget(request["waveform_request"], MAX_RESPONSE_BYTES, handoff=True)
    result = analyze_waveform(request["waveform_request"])
    omitted = [name for name, component in result["components"].items()
               if component["status"] != "computed"]
    if omitted and request.get("allow_partial") is not True:
        raise ValueError("Acknowledge partial known heat before transferring omitted components")
    start, end = result["signals"]["time_s"][0], result["signals"]["time_s"][-1]
    if start < 0:
        raise ValueError("The thermal adapter requires a nonnegative waveform time origin")
    segments = deepcopy(result["power_segments"])
    segment_owners = []
    for index, segment in enumerate(segments):
        owner = owners[segment["direction"]]
        if any(coefficient != 0 for coefficient in segment["power_coefficients_w"]) and network_owner not in {owner, "shared-junction"}:
            raise ValueError("Conduction heat owner does not match the selected thermal network")
        segment["heat_owner"] = owner
        segment_owners.append({"segment_index": index, "heat_owner": owner})
    if not segments:
        segments = [{"start_s": start, "end_s": end, "power_coefficients_w": [0.0],
                     "heat_owner": network_owner if network_owner != "shared-junction" else owners["zero"],
                     "source": "Explicitly omitted conduction. Known switching heat only."}]
        segment_owners = [{"segment_index": 0, "heat_owner": segments[0]["heat_owner"]}]
    events = []
    for event in result["events"]:
        # A finite E(0) belongs to the explicitly declared zero-current owner.
        direction = "zero" if event["current_a"] == 0 else event["direction"]
        owner = owners[direction]
        if network_owner not in {owner, "shared-junction"}:
            raise ValueError("Switching heat owner does not match the selected thermal network")
        events.append({"time_s": event["time_s"], "energy_j": event["energy_j"],
                       "heat_owner": owner, "source": event["source"],
                       "conditions": event["requested_conditions"]})
    if len(segments) > 10000 or len(events) > 10000:
        raise AnalysisResourceLimitError("Thermal handoff exceeds 10000 segments or events")
    output = {"method_version": "waveform-thermal-handoff-v1",
            "source_request": request["waveform_request"], "input_sha256": input_hash(request),
            "source_input_sha256": input_hash(request["waveform_request"]),
            "direction_owners": owners, "segment_owners": segment_owners,
            "omitted_components": omitted, "scope": "known-components-only" if omitted else "complete-waveform-heat",
            "omission_reasons": {name: result["components"][name]["reason"] for name in omitted},
            "waveform_result": result,
            "thermal_input": {"segments": segments, "events": events,
                              "network_heat_owner": network_owner, "event_mode": "impulse",
                              "gap_policy": "reject"},
            "required_inputs": ["network", "case_temperature_c", "initial_state", "samples_per_segment"],
            "assumptions": ["Switching energy uses ideal instantaneous deposition",
                            "Absolute waveform times are preserved",
                            "Physical direction ownership is supplied explicitly"]}
    if len(json.dumps(output, allow_nan=False).encode()) > MAX_RESPONSE_BYTES:
        raise AnalysisResourceLimitError("Thermal handoff response exceeds 24 MB")
    return output
