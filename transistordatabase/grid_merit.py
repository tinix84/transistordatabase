"""Weighted merit scores on one complete, fixed configuration population."""
from __future__ import annotations

import json
import math

from transistordatabase.grouped_series import _csv_rows, _encoded_size

METHOD_VERSION = "waveform-grid-merit-v1"


def validate_objectives(objectives, metric_names):
    """Validate declared objective weights before any pair calculation."""
    if not isinstance(objectives, list) or not 1 <= len(objectives) <= 4:
        raise ValueError("Declare one to four unique merit objectives")
    names, weights = [], []
    for objective in objectives:
        if (not isinstance(objective, dict) or
                set(objective) != {"name", "unit", "direction", "weight"} or
                objective["name"] not in metric_names or objective["unit"] != "W" or
                objective["direction"] not in {"minimize", "maximize"}):
            raise ValueError("Select named merit power metrics in W and their ranking directions")
        weight = objective["weight"]
        if (isinstance(weight, bool) or not isinstance(weight, (float, int)) or
                not math.isfinite(weight) or weight < 0):
            raise ValueError("Merit objective weights must be finite and nonnegative")
        names.append(objective["name"])
        weights.append(weight)
    if len(set(names)) != len(names):
        raise ValueError("Merit objective names must be unique")
    try:
        total = math.fsum(weights)
    except OverflowError as exc:
        raise ValueError("Total merit objective weight overflow") from exc
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Total merit objective weight must be positive and finite")
    normalized = [weight / total for weight in weights]
    if any(weight > 0 and normalized_weight == 0
           for weight, normalized_weight in zip(weights, normalized, strict=True)):
        raise ValueError("A positive normalized merit weight is outside representable precision")
    return [{**objective, "normalized_weight": normalized_weight, "active": objective["weight"] > 0}
            for objective, normalized_weight in zip(objectives, normalized, strict=True)]


def _export_preflight(config_ids, point_ids, objectives, available_bytes, limit_error):
    """Bound repeated identities and all three export representations before expansion."""
    # Reserve every config and point ID on every audit row. A row has at most
    # 32 scalar fields, each shorter than this fixed diagnostic/numeric reserve.
    metadata = {"configuration_ids": config_ids, "operating_point_ids": point_ids,
                "objectives": objectives}
    trial = {**metadata, "configuration_id": max(config_ids, key=len),
             "minimum_configuration_ids": config_ids, "maximum_configuration_ids": config_ids,
             "missing_operating_point_ids": point_ids,
             "reserved_fields": "x" * 4096}
    json_text = json.dumps(trial, allow_nan=False)
    csv_text = _csv_rows([{key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict))
                            else value for key, value in trial.items()}], list(trial))
    per_row = _encoded_size(trial) + _encoded_size(json_text) + _encoded_size(csv_text)
    # Nested table evidence and top-level population/objective records also repeat.
    bound = per_row * len(config_ids) * len(objectives) * 2 + 8192
    if bound > available_bytes:
        raise limit_error("Merit metadata exceeds the complete response byte budget")


def build_merit(rows, summaries, points, objectives, grid_input_sha256,
                available_bytes, limit_error, hash_value, weighted_mean):
    """Use accepted full-population summaries without rerunning electrical models."""
    config_ids = sorted(summary["configuration_id"] for summary in summaries)
    positive_points = sorted(point["id"] for point in points if point["weight"] > 0)
    zero_points = sorted(point["id"] for point in points if point["weight"] == 0)
    _export_preflight(config_ids, positive_points + zero_points, objectives,
                      available_bytes, limit_error)
    by_config = {summary["configuration_id"]: summary["metrics"] for summary in summaries}
    active = [objective for objective in objectives if objective["active"]]
    eligible = [identity for identity in config_ids if all(
        by_config[identity][objective["name"]]["complete"] and
        by_config[identity][objective["name"]]["full_mean"] is not None
        for objective in active)]
    ranges = []
    for objective in objectives:
        name = objective["name"]
        population = eligible if objective["active"] else []
        values = [by_config[identity][name]["full_mean"] for identity in population]
        minimum, maximum = (min(values), max(values)) if values else (None, None)
        ranges.append({**objective, "minimum": minimum, "maximum": maximum,
            "minimum_configuration_ids": [identity for identity in population
                if by_config[identity][name]["full_mean"] == minimum],
            "maximum_configuration_ids": [identity for identity in population
                if by_config[identity][name]["full_mean"] == maximum]})
    population = {"configuration_ids": eligible, "operating_point_ids": positive_points,
                  "zero_weight_operating_point_ids": zero_points,
                  "configuration_count": len(config_ids), "eligible_configuration_count": len(eligible),
                  "configuration_coverage": len(eligible) / len(config_ids)}
    population_evidence = {"grid_input_sha256": grid_input_sha256,
        "population": population, "objectives": objectives,
        "raw_means": [{"configuration_id": identity,
                       "means": {objective["name"]: by_config[identity][objective["name"]]["full_mean"]
                                 for objective in objectives}}
                      for identity in config_ids]}
    population_sha256 = hash_value(population_evidence)
    merit_rows = []
    for identity in config_ids:
        objective_rows, reasons, score_rows = [], [], []
        for objective in ranges:
            name = objective["name"]
            summary = by_config[identity][name]
            missing = sorted(row["operating_point_id"] for row in rows
                             if row["configuration_id"] == identity and row["weight"] > 0
                             and row["metrics"][name]["value"] is None)
            penalty = None
            contribution = None if objective["active"] else 0.0
            if identity in eligible and objective["active"]:
                low, high, value = objective["minimum"], objective["maximum"], summary["full_mean"]
                numerator = ((value - low) if objective["direction"] == "minimize" else (high - value))
                penalty = 0.0 if high == low else numerator / (high - low)
                if numerator > 0 and penalty == 0:
                    raise ValueError("A positive merit penalty is outside representable precision")
                contribution = penalty * objective["normalized_weight"]
                if penalty > 0 and contribution == 0:
                    raise ValueError("A positive merit contribution is outside representable precision")
                score_rows.append({"weight": objective["weight"], "metrics": {"penalty": {"value": penalty}}})
            if objective["active"] and missing:
                reasons.append({"name": name, "missing_operating_point_ids": missing})
            objective_rows.append({**objective, "full_mean": summary["full_mean"],
                "complete": summary["complete"], "retained_weight": summary["retained_weight"],
                "total_weight": summary["total_weight"], "coverage": summary["coverage"],
                "missing_operating_point_ids": missing, "penalty": penalty, "contribution": contribution})
        score = (weighted_mean(score_rows, "penalty", math.fsum(objective["weight"] for objective in active))
                 if identity in eligible else None)
        if score is not None:
            penalties = [score_row["metrics"]["penalty"]["value"] for score_row in score_rows]
            # A convex mean cannot exceed its input extrema. Correct one-ulp
            # summation drift, including the all-zero and all-one endpoints.
            score = min(max(score, min(penalties)), max(penalties))
        merit_rows.append({"configuration_id": identity,
            "status": "computed" if identity in eligible else "excluded", "score": score,
            "rank": None, "objectives": objective_rows, "exclusion_reasons": reasons})
    ranked = sorted((row for row in merit_rows if row["score"] is not None),
                    key=lambda row: (row["score"], row["configuration_id"]))
    previous_score, rank = None, None
    for index, row in enumerate(ranked, 1):
        if row["score"] != previous_score:
            rank = index
        row["rank"] = rank
        previous_score = row["score"]
    ranking = [{key: row[key] for key in ("configuration_id", "score", "rank")} for row in ranked]
    export_rows = []
    for row in merit_rows:
        for objective, result in zip(ranges, row["objectives"], strict=True):
            export_rows.append({"method_version": METHOD_VERSION,
                "grid_input_sha256": grid_input_sha256, "population_sha256": population_sha256,
                "configuration_id": row["configuration_id"], "status": row["status"],
                "score": row["score"], "rank": row["rank"],
                "population_configuration_ids": eligible,
                "population_operating_point_ids": positive_points,
                "zero_weight_operating_point_ids": zero_points,
                **objective, **result})
    csv_rows = [{key: json.dumps(value, allow_nan=False, separators=(",", ":"))
                if isinstance(value, (list, dict)) else value for key, value in row.items()}
                for row in export_rows]
    output = {"method_version": METHOD_VERSION, "grid_input_sha256": grid_input_sha256,
        "population_sha256": population_sha256,
        "status": "computed" if eligible else "unavailable", "population": population,
        "objectives": ranges, "rows": merit_rows, "ranking": ranking, "export_rows": export_rows,
        "rows_json": json.dumps(export_rows, allow_nan=False, separators=(",", ":")),
        "rows_csv": _csv_rows(csv_rows, list(csv_rows[0])), "null_csv": "NULL"}
    if _encoded_size(output) > available_bytes:
        raise limit_error("Merit exceeds the complete response byte budget")
    return output
