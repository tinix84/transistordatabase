"""Explicit workbook benchmarks from source inputs, without Excel execution.

The two archived profiles describe column meanings, never formula evaluation.
Declared estimates retain source conditions and cannot become device ratings.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re

from transistordatabase.thermal import screen_current
from transistordatabase.catalog_sources import coefficient_cache_evidence


METRICS = {
    "thermal_power_budget_w": ("Source 100 °C conduction heat budget estimate", "W", "max"),
    "thermal_current_a": ("Source constant-model thermal current estimate", "A", "max"),
    "thermal_current_per_price": ("Thermal current estimate / price", "A/currency", "max"),
    "thermal_vi_per_price": ("Blocking voltage × thermal current estimate / price", "V·A/currency", "max"),
    "voltage_energy_benchmark_hz": ("Test voltage / aggregate energy-current coefficient", "Hz", "max"),
    "voltage_energy_per_price": ("Voltage / energy-current coefficient / price", "Hz/currency", "max"),
    "rce_coes_s": ("IGBT differential rCE × Coes", "s", "min"),
    "rce_qg_vs": ("IGBT differential rCE × Qg", "V·s", "min"),
    "igbt_area_thermal_k_m2_w": ("IGBT area × declared RthJF", "K·m²/W", "min"),
    "diode_area_thermal_k_m2_w": ("Diode area × declared RthJF", "K·m²/W", "min"),
}
COST_METRICS = {
    "thermal_current_per_price", "thermal_vi_per_price", "voltage_energy_per_price",
}
CONVENTION = "source-constant-model-100C"
ASSUMPTIONS = [
    "Hold the recorded conduction model constant over temperature.",
    "Use a conduction-only heat budget with the source 100 °C reference boundary.",
    "MOSFET uses RthJC to case. IGBT uses RthJH to heatsink.",
    "The result is an estimate, not a measured rating or a self-heating solution.",
]
DESCRIPTIONS = {
    "voltage_energy_benchmark_hz": "A dimensional benchmark, not a maximum switching frequency. All three recorded J/A coefficients are required.",
    "voltage_energy_per_price": "The voltage/coefficient benchmark divided by a known historical quote. No currency conversion.",
    "rce_coes_s": "Differential IGBT resistance and source Coes. This is distinct from MOSFET Rds(on) × Coss.",
    "rce_qg_vs": "Differential IGBT resistance and source gate charge, with recorded source conditions.",
    "igbt_area_thermal_k_m2_w": "Explicitly declared area basis and IGBT junction-to-F thermal path. This is not a guaranteed material property.",
    "diode_area_thermal_k_m2_w": "Explicitly declared area basis and diode junction-to-F thermal path. This is not a guaranteed material property.",
    **{key: "Requires the explicit source constant-model estimate convention. " + " ".join(ASSUMPTIONS)
       for key in ("thermal_power_budget_w", "thermal_current_a", "thermal_current_per_price", "thermal_vi_per_price")},
}
EQUATIONS = {
    "thermal_power_budget_w": "P = (Tj_limit - 100 °C) / Rth_source_path",
    "thermal_current_a": "Solve V0 × I + R × I² = P for I >= 0, using the declared constant source model",
    "thermal_current_per_price": "I_thermal_estimate / quoted_unit_price",
    "thermal_vi_per_price": "V_blocking × I_thermal_estimate / quoted_unit_price",
    "voltage_energy_benchmark_hz": "V_switch_test / (Eon_per_A + Eoff_per_A + Err_per_A)",
    "voltage_energy_per_price": "voltage_energy_benchmark / quoted_unit_price",
    "rce_coes_s": "rCE_differential × Coes_source",
    "rce_qg_vs": "rCE_differential × Qg_source",
    "igbt_area_thermal_k_m2_w": "A_IGBT_declared_m2 × RthJF_IGBT",
    "diode_area_thermal_k_m2_w": "A_diode_declared_m2 × RthJF_diode",
}

# Only these two reviewed, immutable profiles have positional source semantics.
PROFILES = {
    "SemiconductorsData/MOSFET": {"kind": "mosfet", "path": "RthJC", "reference": "case"},
    "SemiconductorsData/IGBT": {"kind": "igbt", "path": "RthJH", "reference": "heatsink"},
}


def validate_settings(settings, selected):
    """Return bounded declarations, without changing any source snapshot."""
    if settings is None:
        return {"thermal_convention": None, "source_declarations": []}
    if not isinstance(settings, dict) or set(settings) - {"thermal_convention", "source_declarations"}:
        raise ValueError("Unknown workbook derivation setting")
    convention = settings.get("thermal_convention")
    if convention not in (None, CONVENTION):
        raise ValueError("Unknown workbook thermal convention")
    declarations = settings.get("source_declarations", [])
    if not isinstance(declarations, list) or len(declarations) > 64:
        raise ValueError("Workbook declarations support at most 64 sources")
    seen = set()
    for item in declarations:
        if not isinstance(item, dict) or set(item) - {"source_id", "currency", "area_unit"}:
            raise ValueError("Unknown workbook source declaration")
        identity = item.get("source_id")
        if not isinstance(identity, str) or identity not in selected or identity in seen:
            raise ValueError("Workbook declarations require distinct selected source IDs")
        seen.add(identity)
        currency, area = item.get("currency"), item.get("area_unit")
        if currency is not None and (not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency)):
            raise ValueError("Declared currency requires a three-letter code")
        if area not in (None, "m2", "mm2", "cm2"):
            raise ValueError("Declared area unit must be m2, mm2 or cm2")
    return {"thermal_convention": convention, "source_declarations": deepcopy(declarations)}


def _number(value, name, *, positive=False, allow_negative=False):
    if value is None:
        raise ValueError(f"missing-input: {name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"not-numeric-input: {name}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"nonfinite-input: {name}")
    if value < 0 and not allow_negative or positive and value == 0:
        raise ValueError(f"outside-domain-input: {name}")
    return float(value)


def _arithmetic(numerators, denominators=()):
    """Preserve representable products and ratios, including known zero."""
    if any(value == 0 for value in denominators):
        raise ValueError("zero-denominator")
    if any(value == 0 for value in numerators):
        return 0.0
    fraction, exponent = 1.0, 0
    for values, sign in ((numerators, 1), (denominators, -1)):
        for value in values:
            part, power = math.frexp(value)
            fraction = fraction * part if sign == 1 else fraction / part
            exponent += sign * power
    try:
        result = math.ldexp(fraction, exponent)
    except OverflowError as exc:
        raise ValueError("derived-overflow") from exc
    if not math.isfinite(result):
        raise ValueError("derived-overflow")
    if result == 0:
        raise ValueError("derived-underflow")
    return result


def validate_mapping_evidence(record):
    """Validate descriptive mapping metadata without making it an operand."""
    mapping = record.get("mapping_evidence", {})
    if not isinstance(mapping, dict):
        raise ValueError("Catalog mapping_evidence must be an object")
    for name, item in mapping.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError("Each catalog mapping_evidence field must be an object")
        for descriptor in ("column", "unit", "cell"):
            if descriptor in item and item[descriptor] is not None and not isinstance(item[descriptor], str):
                raise ValueError(f"Mapping evidence {descriptor} must be text or null")
    return mapping


def source_inputs(record, source):
    """Read fixed source columns or explicitly mapped canonical input fields."""
    mapping = validate_mapping_evidence(record)
    raw = record.get("raw", {})
    if not isinstance(raw, dict):
        raise ValueError("Catalog record raw must be an object")

    def profile_cell(column):
        cell = raw.get(str(column), {})
        if not isinstance(cell, dict):
            raise ValueError("Workbook profile raw cells must be objects with source values")
        return cell

    profile = PROFILES.get(record.get("source_id"))
    if profile and (source.get("file") != "SemiconductorsData.xlsm" or source.get("sheet") != profile["kind"].upper()):
        profile = None
    if profile:
        kind = profile["kind"]
        fields = {
            "model_gate_voltage_v": (20, "V", 1),
            "model_temperature_c": (21, "°C", 1),
            "model_resistance_ohm": (22 if kind == "mosfet" else 23, "mΩ" if kind == "mosfet" else "Ω", .001 if kind == "mosfet" else 1),
            "model_threshold_v": (None if kind == "mosfet" else 22, "V", 1),
            "junction_limit_c": (28, "°C", 1),
            "thermal_path_k_w": (32 if kind == "mosfet" else 34, "K/W", 1),
            "switch_voltage_v": (24, "V", 1),
            "eon_j_a": (25, "J/A", 1), "eoff_j_a": (26, "J/A", 1),
            "err_j_a": (27, "J/A", 1),
            "coes_f": (9, "pF", 1e-12), "source_qg_c": (10, "nC", 1e-9),
            "igbt_area_source": (30, "unknown", 1), "diode_area_source": (31, "unknown", 1),
            "igbt_rth_jf_k_w": (36, "K/W", 1), "diode_rth_jf_k_w": (37, "K/W", 1),
            "workbook_price": (39, "unknown currency", 1),
        }
        inputs = {}
        for name, (column, unit, factor) in fields.items():
            cell = {} if column is None else profile_cell(column)
            value = 0 if column is None else cell.get("value")
            converted = value
            validation = None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    converted = float(value) * factor
                except OverflowError:
                    converted = None
                    validation = "source-unit-conversion-overflow"
                if converted is not None and not math.isfinite(converted):
                    validation = "source-unit-conversion-overflow"
                    converted = None
                elif converted == 0 and value != 0:
                    validation = "source-unit-conversion-underflow"
                    converted = None
            inputs[name] = {"value": converted, "source_value": value, "source_unit": unit,
                            "column": column, "header": cell.get("header"),
                            "formula": record.get("formulas", {}).get(str(column)),
                            "basis": "profile constant" if column is None else "recorded source cell"}
            if validation:
                inputs[name]["validation"] = validation
            if name in ("eon_j_a", "eoff_j_a", "err_j_a"):
                expected = {"eon_j_a": "EON", "eoff_j_a": "EOFF", "err_j_a": "ERR"}[name]
                if cell.get("header") != expected + " (TJ=175°C)":
                    inputs[name]["value"] = None
                    inputs[name]["validation"] = "unrecognized-coefficient-header"
                    continue
                expression = inputs[name]["formula"] or value
                agreement = coefficient_cache_evidence(expression, value)
                literal = agreement["literal"]
                if agreement["literal_syntax"]:
                    if literal:
                        inputs[name]["coefficient_reference"] = literal
                    inputs[name]["coefficient_cache_status"] = agreement["status"]
                    if validation or agreement["validation"]:
                        inputs[name]["value"] = None
                        inputs[name]["validation"] = validation or agreement["validation"]
                    else:
                        inputs[name].update({"value": agreement["value"],
                                             "basis": "recognized literal J/A coefficient"})
                inputs[name]["conditions"] = {"temperature_c": 175, "test_voltage_v": profile_cell(24).get("value"),
                                               "reference_current_a": literal["reference_current_a"] if literal else None,
                                               "gate_resistance_ohm": None}
        origin = profile_cell(40).get("value")
        return profile, inputs, origin
    kind = record.get("workbook_profile")
    if kind not in ("mosfet", "igbt"):
        return None, {}, None
    profile = {"kind": kind, "path": "RthJC" if kind == "mosfet" else "RthJH",
               "reference": "case" if kind == "mosfet" else "heatsink"}
    units = {
        "model_gate_voltage_v": "V", "model_temperature_c": "°C", "model_resistance_ohm": "Ω",
        "model_threshold_v": "V", "junction_limit_c": "°C", "thermal_path_k_w": "K/W",
        "switch_voltage_v": "V", "eon_j_a": "J/A", "eoff_j_a": "J/A", "err_j_a": "J/A",
        "coes_f": "F", "source_qg_c": "C", "igbt_area_m2": "m2", "diode_area_m2": "m2",
        "igbt_rth_jf_k_w": "K/W", "diode_rth_jf_k_w": "K/W", "workbook_price": "source currency",
    }
    inputs = {}
    for name, unit in units.items():
        field = "price" if name == "workbook_price" else name
        metadata = mapping.get(field, {})
        column = metadata.get("column")
        inputs[name] = {
            "value": record.get(field), "unit": unit, "basis": "canonical mapped record field",
            "mapping_evidence": deepcopy(metadata), "classification": "A" if column else "recorded input",
            "source_cell": deepcopy(raw.get(column, {})) if column else {},
        }
    if kind == "mosfet" and record.get("model_threshold_v") is None:
        inputs["model_threshold_v"] = {"value": 0, "unit": "V", "basis": "declared MOSFET profile constant"}
    return profile, inputs, record.get("quote_origin")


def derive_workbook_metrics(record, source, settings):
    """Return calculated metrics and complete assumptions, leaving raw rows intact."""
    values = dict.fromkeys(METRICS)
    reasons = dict.fromkeys(METRICS, "no-declared-workbook-profile")
    profile, inputs, origin = source_inputs(record, source)
    declaration = next((d for d in settings["source_declarations"] if d["source_id"] == record["source_id"]), {})
    for recorded in (record.get("currency"), source.get("currency")):
        if (declaration.get("currency") and isinstance(recorded, str)
                and re.fullmatch(r"[A-Z]{3}", recorded) and recorded != declaration["currency"]):
            raise ValueError("A currency declaration cannot replace a known recorded currency")
    evidence = {"profile": profile, "source_inputs": inputs, "conditions": deepcopy(record.get("conditions", {})),
                "declarations": deepcopy(declaration), "thermal_convention": settings["thermal_convention"],
                "metrics": {}, "quote_origin": origin, "quote_quantity": record.get("quote_quantity"),
                "source_sha256": source.get("sha256"), "source_row": record.get("row")}
    if profile:
        evidence["conditions"].update({
            "conduction_model_temperature_c": inputs["model_temperature_c"]["value"],
            "conduction_model_gate_voltage_v": inputs["model_gate_voltage_v"]["value"],
            "thermal_path": profile["path"], "thermal_reference_kind": profile["reference"],
            "thermal_reference_temperature_c": 100,
            "switching_coefficient_temperature_c": 175 if record.get("source_id") in PROFILES else "not separately declared",
            "capacitance_and_charge_conditions": "Source test voltage and measurement conditions are not separately recorded unless present in source conditions.",
        })
    if profile is None:
        return values, reasons, evidence
    currency = declaration.get("currency") or record.get("currency", source.get("currency"))
    original_currency = record.get("currency", source.get("currency"))
    evidence["currency"] = currency
    evidence["original_currency"] = original_currency
    evidence["currency_basis"] = "user declaration" if declaration.get("currency") else "recorded source"
    evidence["currency_classification"] = "A" if declaration.get("currency") else "source"
    evidence["area_basis"] = {
        "source_unit": "unknown" if record.get("source_id") in PROFILES else "explicit source mapping",
        "declared_unit": declaration.get("area_unit"),
        "classification": "A" if declaration.get("area_unit") else "source mapping or unavailable",
    }

    def number(name, *, positive=False):
        item = inputs.get(name, {})
        if item.get("value") is None and item.get("validation"):
            raise ValueError(item["validation"] + ": " + name)
        diagnostic = item.get("mapping_evidence", {}).get("validation")
        if item.get("value") is None and isinstance(diagnostic, str):
            raise ValueError("mapped-source-input: " + diagnostic + ": " + name)
        return _number(item.get("value"), name, positive=positive)

    def calculate(key, operands, operation, *, estimate=False):
        proof = {"inputs": list(operands), "classification": "E" if estimate else "C",
                 "equation": EQUATIONS[key], "unit": METRICS[key][1]}
        if estimate:
            proof["assumptions"] = ASSUMPTIONS
        try:
            result = operation()
            if result is None:
                raise ValueError("no-finite-thermal-current-limit")
            result = _number(result, key)
            values[key] = result
            reasons.pop(key, None)
            proof["status"], proof["value"] = "available", result
        except ValueError as exc:
            reasons[key] = str(exc)
            proof["status"], proof["reason"] = "unavailable", str(exc)
        evidence["metrics"][key] = proof

    thermal = {}

    def heat_budget():
        if settings["thermal_convention"] != CONVENTION:
            raise ValueError("source-constant-model-estimate-acknowledgment-required")
        maximum = number("junction_limit_c")
        if maximum < 100:
            raise ValueError("junction-limit-below-source-reference-temperature")
        return _arithmetic([maximum - 100], [number("thermal_path_k_w", positive=True)])

    def budget_and_current():
        if settings["thermal_convention"] != CONVENTION:
            raise ValueError("source-constant-model-estimate-acknowledgment-required")
        # Require the actual source model conditions. They remain independent
        # of the temperature limit and the fixed reference boundary.
        number("model_gate_voltage_v", positive=True)
        _number(inputs["model_temperature_c"]["value"], "model_temperature_c", allow_negative=True)
        if not thermal:
            thermal.update(screen_current({
                "junction_limit_c": number("junction_limit_c"), "reference_temperature_c": 100,
                "path_resistance_k_w": number("thermal_path_k_w", positive=True),
                "threshold_v": number("model_threshold_v"), "resistance_ohm": number("model_resistance_ohm"),
                "reference_kind": profile["reference"],
            }))
            for key in ("power_budget_w", "thermal_screening_current_a"):
                value = thermal[key]
                if value == 0 and number("junction_limit_c") > 100:
                    raise ValueError("thermal-result-underflow")
        return thermal

    thermal_inputs = ("junction_limit_c", "thermal_path_k_w", "model_threshold_v", "model_resistance_ohm", "model_temperature_c", "model_gate_voltage_v")
    calculate("thermal_power_budget_w", ("junction_limit_c", "thermal_path_k_w"), heat_budget, estimate=True)
    calculate("thermal_current_a", thermal_inputs, lambda: budget_and_current()["thermal_screening_current_a"], estimate=True)

    def coefficient_benchmark():
        coefficients = [number(name) for name in ("eon_j_a", "eoff_j_a", "err_j_a")]
        maximum = max(coefficients)
        if maximum == 0:
            raise ValueError("zero-aggregate-energy-coefficient")
        total_scaled = math.fsum(value / maximum for value in coefficients)
        return _arithmetic([number("switch_voltage_v", positive=True)], [maximum, total_scaled])

    calculate("voltage_energy_benchmark_hz", ("switch_voltage_v", "eon_j_a", "eoff_j_a", "err_j_a"), coefficient_benchmark)

    def price():
        if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("unknown-currency-requires-source-declaration")
        _number(record.get("quote_quantity"), "quote_quantity", positive=True)
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("missing-input: quote_origin")
        return number("workbook_price", positive=True)

    def from_metric(key):
        if values[key] is None:
            raise ValueError(f"invalid-operand: {key}: {reasons[key]}")
        return values[key]

    calculate("thermal_current_per_price", ("thermal_current_a", "workbook_price"), lambda: _arithmetic([from_metric("thermal_current_a")], [price()]), estimate=True)
    calculate("thermal_vi_per_price", ("voltage_v", "thermal_current_a", "workbook_price"), lambda: _arithmetic([_number(record.get("voltage_v"), "voltage_v", positive=True), from_metric("thermal_current_a")], [price()]), estimate=True)
    calculate("voltage_energy_per_price", ("voltage_energy_benchmark_hz", "workbook_price"), lambda: _arithmetic([from_metric("voltage_energy_benchmark_hz")], [price()]))

    def igbt_product(name):
        if profile["kind"] != "igbt":
            raise ValueError("requires-igbt-profile")
        return _arithmetic([number("model_resistance_ohm"), number(name)])

    calculate("rce_coes_s", ("model_resistance_ohm", "coes_f"), lambda: igbt_product("coes_f"))
    calculate("rce_qg_vs", ("model_resistance_ohm", "source_qg_c"), lambda: igbt_product("source_qg_c"))

    def area_product(kind):
        if profile["kind"] != "igbt":
            raise ValueError("requires-igbt-profile")
        si_name = kind + "_area_m2"
        if si_name in inputs:
            area = number(si_name, positive=True)
        else:
            factors = {"m2": 1, "mm2": 1e-6, "cm2": 1e-4}
            unit = declaration.get("area_unit")
            if unit not in factors:
                raise ValueError("unknown-area-unit-requires-source-declaration")
            area = _arithmetic([number(kind + "_area_source", positive=True), factors[unit]])
        return _arithmetic([area, number(kind + "_rth_jf_k_w", positive=True)])

    for kind in ("igbt", "diode"):
        area_field = kind + "_area_m2" if kind + "_area_m2" in inputs else kind + "_area_source"
        metric = kind + "_area_thermal_k_m2_w"
        calculate(metric, (area_field, kind + "_rth_jf_k_w"), lambda kind=kind: area_product(kind))
        evidence["metrics"][metric]["area_conversion"] = {
            "input_field": area_field, "input_unit": "m2" if area_field.endswith("_m2") else "unknown",
            "declared_unit": None if area_field.endswith("_m2") else declaration.get("area_unit"),
            "factor_to_m2": 1 if area_field.endswith("_m2") else {"m2": 1, "mm2": 1e-6, "cm2": 1e-4}.get(declaration.get("area_unit")),
            "classification": "source mapping" if area_field.endswith("_m2") else "A",
        }
    evidence["thermal_result"] = thermal or None
    evidence["scope"] = "Explicit dimensional benchmarks and acknowledged conduction estimates. Source cached formulas are evidence only."
    return values, reasons, evidence
