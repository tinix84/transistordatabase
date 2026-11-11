"""Source-aware catalog screening, independent of GUI and transport code.

Catalog ratings are screening data, not measured operating-point loss curves.
Prices and their quantity breaks remain attached to each original source row.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
import json
import hashlib
from copy import deepcopy

from transistordatabase.catalog_sources import check_json_budget
from transistordatabase.workbook_fom import (
    COST_METRICS as WORKBOOK_COST_METRICS,
    METRICS as WORKBOOK_METRICS,
    derive_workbook_metrics,
    validate_mapping_evidence,
    validate_settings as validate_workbook_settings,
)


CATALOG_PATH = Path(__file__).parent / "data" / "catalogs" / "archive_sc.json"

# Values use SI except price, whose currency is recorded on the source.
METRICS = {
    "voltage_v": ("Blocking voltage", "V", "max"),
    "current_a": ("Catalog drain / collector current", "A", "max"),
    "power_w": ("Catalog power dissipation", "W", "max"),
    "rds_ohm": ("Rds(on), max", "Ω", "min"),
    "qg_c": ("Gate charge", "C", "min"),
    "ciss_f": ("Input capacitance", "F", "min"),
    "coss_f": ("Effective output capacitance", "F", "min"),
    "price": ("Quoted unit price", "source currency", "min"),
    "rds_qg": ("Rds(on) × Qg", "Ω·C", "min"),
    "rds_coss": ("Rds(on) × Coss(eff)", "Ω·F", "min"),
    "cost_fom": ("Price × Rds(on) / Vdss²", "currency·Ω/V²", "min"),
    "vi": ("Blocking voltage × catalog current", "V·A", "max"),
    "vi_per_price": ("Voltage × current / price", "V·A/currency", "max"),
    **WORKBOOK_METRICS,
}


def parse_quantity(value: object, unit: str = "") -> float | None:
    """Read a leading number with an SI prefix without stripping decimals.

    Conditions following a value (``41mOhm @ 32A, 8V``) are not numbers
    to concatenate. A bare numeric cell uses the requested base unit.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().replace("µ", "u").replace("μ", "u")
        match = re.match(
            r"^([+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-ZΩ]*)", text
        )
        if not match:
            return None
        number, suffix = match.groups()
        # A decimal comma is accepted; prices with thousands separators are
        # intentionally not guessed. The supplied prices are decimal numbers.
        result = float(number.replace(",", "."))
        if unit:
            suffix = suffix.replace("Ω", "Ohm")
            base = unit.replace("Ω", "Ohm")
            prefixes = {
                "": 1,
                "p": 1e-12,
                "n": 1e-9,
                "u": 1e-6,
                "m": 1e-3,
                "k": 1e3,
                "M": 1e6,
            }
            if suffix:
                if not suffix.lower().endswith(base.lower()):
                    return None
                prefix = suffix[: -len(base)]
                if prefix not in prefixes:
                    return None
                result *= prefixes[prefix]
        elif suffix:
            return None
    return result if math.isfinite(result) and result > 0 else None


DERIVED = {
    "rds_qg": (("rds_ohm", "qg_c"), ()),
    "rds_coss": (("rds_ohm", "coss_f"), ()),
    "cost_fom": (("price", "rds_ohm"), ("voltage_v", "voltage_v")),
    "vi": (("voltage_v", "current_a"), ()),
    "vi_per_price": (("voltage_v", "current_a"), ("price",)),
}
COST_METRICS = {"price", "cost_fom", "vi_per_price"} | WORKBOOK_COST_METRICS
SURVEY_PAIRS = [
    ("voltage_v", "current_a"), ("voltage_v", "power_w"),
    ("voltage_v", "price"), ("power_w", "current_a"),
    ("power_w", "price"), ("price", "rds_ohm"),
]


def _positive_reason(value: object) -> str | None:
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "not-numeric"
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        return "nonfinite"
    return "nonpositive" if value <= 0 else None


def _product_ratio(numerators: list[float], denominators: list[float]) -> float:
    """Avoid intermediate overflow/underflow for representable positive results."""
    mantissa, exponent = 1.0, 0
    for values, sign in ((numerators, 1), (denominators, -1)):
        for value in values:
            part, power = math.frexp(value)
            mantissa = mantissa * part if sign == 1 else mantissa / part
            exponent += sign * power
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as exc:
        raise ValueError("derived-overflow") from exc
    if not math.isfinite(result):
        raise ValueError("derived-overflow")
    if result == 0:
        raise ValueError("derived-underflow")
    return result


def metric_evidence(record: dict) -> tuple[dict, dict]:
    """Return positive SI metrics and explicit missing/invalid operand reasons."""
    values, reasons = {}, {}
    for key in METRICS.keys() - DERIVED.keys() - WORKBOOK_METRICS.keys():
        reason = _positive_reason(record.get(key))
        values[key] = None if reason else record[key]
        if reason:
            reasons[key] = reason
    for key, (numerators, denominators) in DERIVED.items():
        absent = [operand for operand in numerators + denominators
                  if values[operand] is None]
        if absent:
            values[key] = None
            reasons[key] = "invalid-operands: " + ", ".join(dict.fromkeys(absent))
            continue
        try:
            values[key] = _product_ratio(
                [values[k] for k in numerators], [values[k] for k in denominators]
            )
        except ValueError as exc:
            values[key], reasons[key] = None, str(exc)
    for key in WORKBOOK_METRICS:
        values[key] = None
        reasons[key] = "requires-workbook-source-derivation"
    return {key: values[key] for key in METRICS}, reasons


def metric_values(record: dict) -> dict[str, float | None]:
    """Compute screening figures only from finite, positive numeric operands."""
    return metric_evidence(record)[0]


def pareto_ids(points: list[dict], x_direction: str, y_direction: str) -> set[str]:
    """Return the exact two-objective non-dominated set, retaining equal ties.

    A device dominates another only if it is no worse on both axes and
    strictly better on at least one. Sorting makes this O(n log n).
    """
    if x_direction not in {"min", "max"} or y_direction not in {"min", "max"}:
        raise ValueError("Objective directions must be min or max")
    sx, sy = (1 if d == "min" else -1 for d in (x_direction, y_direction))
    ordered = sorted(points, key=lambda p: (sx * p["x"], sy * p["y"]))
    best_y = math.inf
    best_x = None
    ids = set()
    for p in ordered:
        x, y = sx * p["x"], sy * p["y"]
        if y < best_y:
            best_y, best_x = y, x
            ids.add(p["id"])
        elif y == best_y and x == best_x:
            ids.add(p["id"])
    return ids


def load_catalog() -> dict:
    """Load the portable snapshot without needing the archive at runtime."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def screen_catalog(
    catalog: dict,
    *,
    source_id: str | None = None,
    source_ids: list[str] | None = None,
    x_metric: str = "rds_ohm",
    y_metric: str = "cost_fom",
    x_direction: str = "min",
    y_direction: str = "min",
    technology: str = "",
    manufacturer: str = "",
    search: str = "",
    voltage_min: float | None = None,
    voltage_max: float | None = None,
    current_min: float | None = None,
    quote_cohort: dict | None = None,
    normalization: dict | None = None,
    panel_preset: str = "default",
    workbook_derivation: dict | None = None,
) -> dict:
    """Screen explicit source populations without losing unrelated panel coverage.

    A fixed normalization population is selected before display filters. Mixed
    source cost comparisons require an explicit recorded common quote cohort.
    """
    validate_catalog_snapshot(catalog)
    sources = {s["id"]: s for s in catalog["sources"]}
    if (source_id is None) == (source_ids is None):
        raise ValueError("Choose source_id or source_ids, exclusively")
    selected = [source_id] if source_ids is None else source_ids
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one distinct catalog source")
    if any(s not in sources for s in selected):
        raise ValueError("Unknown catalog source")
    workbook_settings = validate_workbook_settings(workbook_derivation, selected)
    if x_metric not in METRICS or y_metric not in METRICS:
        raise ValueError("Unknown screening metric")
    if x_metric == y_metric:
        raise ValueError("Choose two different screening metrics")
    if x_direction not in {"min", "max"} or y_direction not in {"min", "max"}:
        raise ValueError("Objective directions must be min or max")
    if panel_preset not in {"default", "survey"}:
        raise ValueError("Unknown panel preset")
    for value in (voltage_min, voltage_max, current_min):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0
        ):
            raise ValueError("Rating filters must be finite and nonnegative")
    if (
        voltage_min is not None
        and voltage_max is not None
        and voltage_min > voltage_max
    ):
        raise ValueError("Minimum voltage must not exceed maximum voltage")
    population = []
    for record in catalog["records"]:
        if record["source_id"] not in selected:
            continue
        values, reasons = metric_evidence(record)
        workbook_values, workbook_reasons, workbook_evidence = derive_workbook_metrics(
            record, sources[record["source_id"]], workbook_settings
        )
        values.update(workbook_values)
        for key in WORKBOOK_METRICS:
            reasons.pop(key, None)
        reasons.update(workbook_reasons)
        declared = workbook_evidence.get("declarations", {}).get("currency")
        population.append({
            **deepcopy(record), "source": deepcopy(sources[record["source_id"]]),
            "metrics": values, "metric_exclusions": reasons, "pareto": None,
            "workbook_evidence": workbook_evidence,
            **({"currency": declared, "currency_basis": "user declaration"} if declared else {}),
        })
    cost = _quote_policy(population, len(selected), quote_cohort)
    if cost["status"] == "unavailable":
        for record in population:
            for key in COST_METRICS:
                record["metrics"][key] = None
                record["metric_exclusions"][key] = cost["reason"]
                if key in record["workbook_evidence"]["metrics"]:
                    proof = record["workbook_evidence"]["metrics"][key]
                    proof.update({"status": "unavailable", "reason": cost["reason"]})
                    proof.pop("value", None)
    candidates = []
    for record in population:
        if technology and record.get("technology") != technology:
            continue
        if manufacturer and record.get("manufacturer") != manufacturer:
            continue
        if (
            search.casefold()
            not in (record.get("part_number", "") + " "
                    + record.get("manufacturer", "")).casefold()
        ):
            continue
        values = record["metrics"]
        v, i = values["voltage_v"], values["current_a"]
        if voltage_min is not None and (v is None or v < voltage_min):
            continue
        if voltage_max is not None and (v is None or v > voltage_max):
            continue
        if current_min is not None and (i is None or i < current_min):
            continue
        candidates.append(record)
    normalized = _normalize(population, candidates, normalization)
    points = [
        {**r, "x": r["metrics"][x_metric], "y": r["metrics"][y_metric]}
        for r in candidates
        if r["metrics"][x_metric] is not None and r["metrics"][y_metric] is not None
    ]
    frontier = pareto_ids(points, x_direction, y_direction)
    for candidate in candidates:
        if candidate["metrics"][x_metric] is not None and (
            candidate["metrics"][y_metric] is not None
        ):
            candidate["pareto"] = candidate["id"] in frontier
    for point in points:
        point["pareto"] = point["id"] in frontier
    points.sort(key=lambda p: p["y"], reverse=y_direction == "max")
    pairs = (
        SURVEY_PAIRS if panel_preset == "survey" else [
            (x_metric, y_metric), ("rds_ohm", "price"),
            ("voltage_v", "current_a"), ("rds_ohm", "rds_qg"),
        ]
    )
    panels = []
    for index, (x, y) in enumerate(pairs):
        complete, excluded = [], []
        for record in candidates:
            reasons = {key: record["metric_exclusions"][key] for key in (x, y)
                       if key in record["metric_exclusions"]}
            if reasons:
                excluded.append({"id": record["id"], "reasons": reasons})
            else:
                # Resolve IDs against candidates in adapters. Raw evidence exists once.
                complete.append({"id": record["id"], "x": record["metrics"][x],
                                 "y": record["metrics"][y]})
        panels.append({
            "id": f"{panel_preset}-{index}", "x_metric": x, "y_metric": y,
            "points": complete, "matched_count": len(candidates),
            "plotted_count": len(complete), "excluded_count": len(excluded),
            "exclusions": excluded,
            "normalization": {
                axis: normalized["metrics"].get(key) if normalized else None
                for axis, key in (("x", x), ("y", y))
            },
        })
    return {
        "source": deepcopy(sources[selected[0]]),
        "sources": [deepcopy(sources[s]) for s in selected],
        "candidates": candidates,
        "panels": panels,
        "cost_cohort": cost,
        "workbook_derivation": workbook_settings,
        "normalization": normalized,
        "points": points,
        "matched_count": len(candidates),
        "plotted_count": len(points),
        "excluded_count": len(candidates) - len(points),
        "pareto_count": len(frontier),
        "coverage": {
            key: sum(r["metrics"][key] is not None for r in candidates)
            for key in METRICS
        },
        "assumptions": [
            "Catalog screening uses quoted ratings, not operating-point loss predictions.",
            "Pareto objectives apply only to selected sources, filters and chosen axes.",
            "Missing or invalid operands are excluded, never replaced with zero.",
            "Measurement conditions and quote quantities differ between rows; inspect before selecting.",
            "Historical prices are not current quotes; no currency conversion is applied.",
            "V × I is a rating product, not usable continuous power.",
        ],
    }


def validate_catalog_snapshot(catalog: dict) -> dict:
    """Validate a portable supplied snapshot, never a caller-selected file path."""
    if not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int or catalog.get("schema_version") != 1:
        raise ValueError("Catalog snapshot schema_version must be 1")
    sources, records = catalog.get("sources"), catalog.get("records")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 64:
        raise ValueError("Catalog snapshot requires 1–64 sources")
    if not isinstance(records, list) or len(records) > 10000:
        raise ValueError("Catalog snapshot supports at most 10000 records")
    source_ids, record_ids = set(), set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise ValueError("Every catalog source requires a string id")
        if not source["id"] or source["id"] in source_ids:
            raise ValueError("Catalog source IDs must be nonempty and unique")
        if not isinstance(source.get("sha256"), str) or not re.fullmatch(
            r"[a-fA-F0-9]{64}", source["sha256"]
        ):
            raise ValueError("Every catalog source requires its SHA-256")
        source_ids.add(source["id"])
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("Every catalog record requires a string id")
        if not record["id"] or record["id"] in record_ids:
            raise ValueError("Catalog record IDs must be nonempty and unique")
        if not isinstance(record.get("source_id"), str) or record["source_id"] not in source_ids:
            raise ValueError("Catalog record references an unknown source")
        for key in ("part_number", "manufacturer", "technology"):
            if key in record and not isinstance(record[key], str):
                raise ValueError(f"Catalog record {key} must be text")
        for key in ("conditions", "raw", "formulas"):
            if key in record and not isinstance(record[key], dict):
                raise ValueError(f"Catalog record {key} must be an object")
        validate_mapping_evidence(record)
        record_ids.add(record["id"])
    check_json_budget(catalog)
    return catalog


def _quote_policy(population: list[dict], source_count: int, cohort: dict | None):
    if cohort is None and source_count == 1:
        return {"status": "historical-single-source", "reason": None,
                "currency": population[0].get("currency", population[0]["source"].get("currency"))
                if population else None, "quote_quantity": None}
    if cohort is None:
        return {"status": "unavailable", "reason":
                "mixed-source-cost-requires-explicit-currency-and-quote-quantity"}
    if set(cohort) != {"currency", "quote_quantity"}:
        raise ValueError("Quote cohort requires currency and quote_quantity only")
    currency, quantity = cohort["currency"], cohort["quote_quantity"]
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("Quote cohort currency must be a known three-letter code")
    if _positive_reason(quantity):
        raise ValueError("Quote cohort quantity must be finite and positive")
    mismatches = []
    for record in population:
        if record["metrics"]["price"] is None:
            continue
        actual_currency = record.get("currency", record["source"].get("currency"))
        actual_quantity = record.get("quote_quantity")
        if actual_currency != currency or _positive_reason(actual_quantity) or (
            actual_quantity != quantity
        ):
            mismatches.append(record["id"])
    return {"status": "unavailable" if mismatches else "compatible-explicit",
            "reason": "incompatible-or-unknown-recorded-quote-cohort"
            if mismatches else None, "currency": currency,
            "quote_quantity": quantity, "incompatible_ids": mismatches}


def _normalize(population: list[dict], candidates: list[dict], settings: dict | None):
    if settings is None:
        return None
    if set(settings) - {"mode", "metrics", "population_ids"}:
        raise ValueError("Unknown normalization setting")
    mode, keys = settings.get("mode"), settings.get("metrics", [])
    if mode not in {"fixed", "visible"}:
        raise ValueError("Normalization mode must be fixed or visible")
    if not keys or len(set(keys)) != len(keys) or any(k not in METRICS for k in keys):
        raise ValueError("Normalization requires distinct known metrics")
    requested = settings.get("population_ids")
    if requested is not None:
        if mode != "fixed":
            raise ValueError("population_ids applies only to fixed normalization")
        known = {record["id"] for record in population}
        if len(set(requested)) != len(requested) or any(i not in known for i in requested):
            raise ValueError("Normalization population IDs must be distinct selected rows")
    chosen = population if mode == "fixed" else candidates
    if requested is not None:
        chosen = [record for record in chosen if record["id"] in requested]
    ids = sorted(record["id"] for record in chosen)
    digest = hashlib.sha256(json.dumps([
        {"id": record["id"], "source_sha256": record["source"]["sha256"],
         "source_snapshot_id": record["source"].get("snapshot_id"),
         "effective_currency": record.get("currency", record["source"].get("currency")),
         "currency_basis": record.get("currency_basis", "recorded source"),
         **({"workbook_evidence": record["workbook_evidence"]}
            if any(key in WORKBOOK_METRICS for key in keys) else {}),
         "metrics": {key: record["metrics"][key] for key in keys}}
        for record in sorted(chosen, key=lambda row: row["id"])
    ], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {"mode": mode, "population_ids": ids, "population_hash": digest,
              "metrics": {}}
    chosen_ids = set(ids)
    for key in keys:
        valid = [r for r in chosen if r["metrics"][key] is not None and r["metrics"][key] > 0]
        denominator = min((r["metrics"][key] for r in valid), default=None)
        evidence = {
            "status": "available" if denominator is not None else "unavailable",
            "denominator": denominator,
            "denominator_ids": sorted(r["id"] for r in valid
                                      if r["metrics"][key] == denominator),
            "population_ids": ids, "population_hash": digest,
            "exclusions": [{"id": r["id"], "reason": r["metric_exclusions"].get(key, "zero-not-normalization-reference")}
                           for r in chosen if r["metrics"][key] is None or r["metrics"][key] == 0],
        }
        values = []
        for record in population:
            absolute, relative = record["metrics"][key], None
            reason = record["metric_exclusions"].get(key)
            if denominator is None:
                reason = reason or "no-positive-normalization-population"
            elif absolute is not None:
                try:
                    relative = 0.0 if absolute == 0 else _product_ratio([absolute], [denominator])
                except ValueError as exc:
                    reason = "normalization-" + str(exc)
            entry = {"absolute": absolute, "relative": relative, "reason": reason}
            record.setdefault("normalization", {})[key] = entry
            if record["id"] in chosen_ids:
                values.append({"id": record["id"], **entry})
        evidence["values"] = values
        result["metrics"][key] = evidence
    return result
