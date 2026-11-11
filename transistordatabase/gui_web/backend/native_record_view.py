"""Preserve native core evidence through the established legacy editor interface."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace
import json

import numpy as np

from transistordatabase.core import adapters
from transistordatabase.core.serialization import core_to_json_document


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        if value.dtype.kind == "f":
            projected = float(value)
            if not np.isfinite(projected) or (value != 0 and projected == 0):
                raise ValueError("Native value exceeds the browser's finite numeric range")
            return projected
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Unsupported editor value: {type(value).__name__}")


def browser_projection(value):
    """Return finite browser numbers without changing the retained native value."""
    return json.loads(json.dumps(value, default=_json_value, allow_nan=False))


def _same(left, right):
    return json.dumps(left, default=_json_value, sort_keys=True) == json.dumps(
        right, default=_json_value, sort_keys=True
    )


class _CurveView(dict):
    """Retain the original observation when an editor changes selected fields."""

    def __init__(self, projection, source):
        projection = browser_projection(projection)
        super().__init__(projection)
        self.source = source
        self.original = deepcopy(projection)

    def materialize(self):
        result = deepcopy(self.source)
        for key, value in self.items():
            if key in self.original and _same(value, self.original[key]):
                continue
            if not hasattr(result, key):
                raise ValueError(f"Native observation cannot represent editor field {key}")
            if (key.startswith("graph_") or isinstance(getattr(result, key), np.ndarray)) and value is not None:
                value = np.asarray(value, dtype=float)
            setattr(result, key, value)
        return result


# These mappings reuse the compatibility adapters. Existing observations use
# field patches instead, so their native evidence and unedited values survive.
_LISTS = {
    "channel": ("channel_data", adapters._convert_channel_legacy_to_core),
    "e_on": ("e_on_data", adapters._convert_switching_legacy_to_core),
    "e_off": ("e_off_data", adapters._convert_switching_legacy_to_core),
    "e_rr": ("e_rr_data", adapters._convert_switching_legacy_to_core),
    "charge_curve": ("gate_charge_curves", adapters._convert_gate_charge_legacy_to_core),
    "soa": ("soa", adapters._convert_soa_legacy_to_core),
    "raw_measurement_data": ("raw_measurement_data", adapters._convert_raw_meas_legacy_to_core),
    "c_oss": ("c_oss", adapters._convert_vdc_legacy_to_core),
    "c_iss": ("c_iss", adapters._convert_vdc_legacy_to_core),
    "c_rss": ("c_rss", adapters._convert_vdc_legacy_to_core),
}


def _materialize_curve(value, converter):
    if isinstance(value, _CurveView):
        return value.materialize()
    values = dict(value)
    provenance = values.pop("import_provenance", None) if converter is adapters._convert_raw_meas_legacy_to_core else None
    result = converter(SimpleNamespace(**values))
    if provenance is not None:
        from transistordatabase.core.models import MeasurementEvidence

        trace_mapping = deepcopy(provenance.get("trace_mapping", {}))
        if "legacy_import_provenance" in trace_mapping:
            raise ValueError("RAW trace mapping uses the reserved legacy provenance key")
        trace_mapping["legacy_import_provenance"] = deepcopy(provenance)
        result.evidence = MeasurementEvidence(
            source_kind="ltspice-raw", filename=provenance.get("source_filename"),
            sha256=provenance.get("source_sha256"), parser=provenance.get("parser"),
            parser_version=provenance.get("parser_version"),
            trace_mapping=trace_mapping, method="raw event-window extraction",
            bus_voltage_v=values.get("v_supply"), t_j=values.get("t_j"),
            v_g_on=values.get("v_g"), v_g_off=values.get("v_g_off"),
            r_g_on=(values.get("r_g") if values.get("r_g") is not None and np.ndim(values["r_g"]) == 0 else None),
            r_g_off=(values.get("r_g_off") if values.get("r_g_off") is not None and np.ndim(values["r_g_off"]) == 0 else None),
            commutation_device=values.get("commutation_device"),
            disclosure="Original import provenance remains in trace_mapping.legacy_import_provenance. Native schema version remains 2.",
        )
    unsupported = set(values) - {field.name for field in fields(result)}
    if unsupported:
        raise ValueError("Native observation cannot represent editor fields: " + ", ".join(sorted(unsupported)))
    return result


class NativeRecordView(dict):
    """Expose mutable editor fields without converting the retained native record."""

    def __init__(self, transistor):
        self.source = transistor
        record, switch, diode = adapters.core_to_legacy_dicts(transistor)
        record.update(switch=switch, diode=diode)
        for component_name in ("switch", "diode"):
            component = getattr(transistor, component_name)
            section = record[component_name]
            section["raw_measurement_data"] = []
            for item in component.raw_measurement_data:
                projection = {field.name: getattr(item, field.name) for field in fields(item) if field.name != "evidence"}
                if item.evidence is not None and "legacy_import_provenance" in item.evidence.trace_mapping:
                    projection["import_provenance"] = deepcopy(item.evidence.trace_mapping["legacy_import_provenance"])
                section["raw_measurement_data"].append(projection)
        for name, component in ((None, transistor), ("switch", transistor.switch), ("diode", transistor.diode)):
            section = record if name is None else record[name]
            for legacy_key, (core_key, _converter) in _LISTS.items():
                originals = getattr(component, core_key, None)
                if legacy_key in section and isinstance(originals, list):
                    section[legacy_key] = [_CurveView(value, original) for value, original in zip(section[legacy_key], originals, strict=True)]
        # Legacy DPT records belong to the switch branch in the native model.
        record["raw_measurement_data"] = record["switch"]["raw_measurement_data"]
        super().__init__(record)
        self.original = deepcopy(record)

    def document(self):
        """Materialize edited fields while retaining untouched native evidence."""
        result = deepcopy(self.source)
        supported = {
            "name", "type", "author", "manufacturer", "housing_type", "comment",
            "datasheet_hyperlink", "datasheet_date", "datasheet_version",
            "v_abs_max", "i_abs_max", "i_cont", "housing_area", "cooling_area",
            "r_th_cs", "r_th_switch_cs", "r_th_diode_cs", "t_c_max",
            "c_oss", "c_iss", "c_rss", "raw_measurement_data", "switch", "diode",
        }
        for key in self.keys() | self.original.keys():
            if key not in supported and not _same(self.get(key), self.original.get(key)):
                raise ValueError(f"Native record cannot represent editor field {key}")
        for target, keys in (
            (result.metadata, ("type", "author", "manufacturer", "housing_type", "comment", "datasheet_hyperlink", "datasheet_date", "datasheet_version")),
            (result.electrical_ratings, ("v_abs_max", "i_abs_max", "i_cont")),
            (result.thermal_properties, ("housing_area", "cooling_area", "r_th_cs", "r_th_switch_cs", "r_th_diode_cs", "t_c_max")),
        ):
            for key in keys:
                if not _same(self[key], self.original[key]):
                    setattr(target, key, self[key])
        if self["name"] != self.source.metadata.name:
            raise ValueError("Native record identity cannot be changed by an editor")
        if not _same(self["switch"]["t_j_max"], self.original["switch"]["t_j_max"]):
            result.electrical_ratings.t_j_max = self["switch"]["t_j_max"]
        for name, component in ((None, result), ("switch", result.switch), ("diode", result.diode)):
            section = self if name is None else self[name]
            before = self.original if name is None else self.original[name]
            for legacy_key, (core_key, converter) in _LISTS.items():
                target = result.switch if name is None and legacy_key == "raw_measurement_data" else component
                if legacy_key in section and hasattr(target, core_key) and not _same(section[legacy_key], before.get(legacy_key)):
                    setattr(target, core_key, [_materialize_curve(value, converter) for value in section[legacy_key]])
            if name is not None:
                allowed = {key for key, (core_key, _) in _LISTS.items() if hasattr(component, core_key)} | {"thermal_foster"}
                if name == "switch":
                    allowed.add("t_j_max")
                for key in section.keys() | before.keys():
                    if key not in allowed and not _same(section.get(key), before.get(key)):
                        raise ValueError(f"Native {name} cannot represent editor field {key}")
            if name is not None and not _same(section["thermal_foster"], before["thermal_foster"]):
                from transistordatabase.core.models import FosterThermalModel

                unsupported = set(section["thermal_foster"]) - {field.name for field in fields(FosterThermalModel)}
                if unsupported:
                    raise ValueError("Native Foster model cannot represent editor fields: " + ", ".join(sorted(unsupported)))
                component.thermal_foster = adapters._convert_foster_legacy_to_core(SimpleNamespace(**section["thermal_foster"]))
        return core_to_json_document(result)
