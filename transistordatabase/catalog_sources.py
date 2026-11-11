"""Bounded, nonexecuting source import for portable catalog snapshots.

Original bytes and cells remain evidence. Formulas, VBA, queries and hyperlink
targets are never executed. Mapping is explicit; row identities do not depend
on potentially duplicated part numbers.
"""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import json
import math
import posixpath
import re
import zipfile
import zlib
from datetime import date, datetime, time
from xml.etree import ElementTree as ET

MAX_BYTES = 8_000_000
MAX_EXPANDED = 48_000_000
MAX_CELLS = 200_000
MAX_CELL_TEXT_BYTES = 16_000_000
MAX_SNAPSHOT_BYTES = 50_000_000
MAX_ROWS = 10_000
MAX_COLUMNS = 256
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
TEXT_FIELDS = {"part_number", "manufacturer", "technology", "package", "datasheet"}
UNITS = {
    "voltage_v": {"V": 1, "kV": 1000},
    "current_a": {"A": 1, "mA": .001},
    "power_w": {"W": 1, "kW": 1000},
    "rds_ohm": {"Ohm": 1, "mOhm": .001},
    "qg_c": {"C": 1, "nC": 1e-9, "uC": 1e-6},
    "coss_f": {"F": 1, "pF": 1e-12, "nF": 1e-9},
    "ciss_f": {"F": 1, "pF": 1e-12, "nF": 1e-9},
    "price": {"source currency": 1},
    "quote_quantity": {"pieces": 1},
    "model_gate_voltage_v": {"V": 1},
    "model_temperature_c": {"degC": 1},
    "model_resistance_ohm": {"Ohm": 1, "mOhm": .001},
    "model_threshold_v": {"V": 1},
    "junction_limit_c": {"degC": 1},
    "thermal_path_k_w": {"K/W": 1},
    "switch_voltage_v": {"V": 1, "kV": 1000},
    "eon_j_a": {"J/A": 1, "mJ/A": .001, "uJ/A": 1e-6},
    "eoff_j_a": {"J/A": 1, "mJ/A": .001, "uJ/A": 1e-6},
    "err_j_a": {"J/A": 1, "mJ/A": .001, "uJ/A": 1e-6},
    "coes_f": {"F": 1, "pF": 1e-12, "nF": 1e-9},
    "source_qg_c": {"C": 1, "nC": 1e-9, "uC": 1e-6},
    "igbt_area_m2": {"m2": 1, "mm2": 1e-6, "cm2": 1e-4},
    "diode_area_m2": {"m2": 1, "mm2": 1e-6, "cm2": 1e-4},
    "igbt_rth_jf_k_w": {"K/W": 1},
    "diode_rth_jf_k_w": {"K/W": 1},
}
TEXT_FIELDS |= {"quote_origin"}


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _nonzero_significand(token):
    return bool(re.search(r"[1-9]", token.lower().partition("e")[0]))


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def check_json_budget(value):
    """Limit the compact UTF-8 representation, including repeated references."""
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    size = 0
    try:
        for fragment in encoder.iterencode(value):
            size += len(fragment.encode("utf-8"))
            if size > MAX_SNAPSHOT_BYTES:
                raise ValueError("Catalog snapshot or comparison exceeds 50 MB")
    except (TypeError, RecursionError) as exc:
        raise ValueError("Catalog snapshot must contain finite JSON values") from exc


def _count_cell_text(size, *values):
    for value in values:
        if isinstance(value, str):
            size += len(value.encode("utf-8"))
            if size > MAX_CELL_TEXT_BYTES:
                raise ValueError("Expanded source cell text exceeds the 16 MB budget")
    return size


def decode_source(filename: str, content_base64: str) -> bytes:
    """Decode bounded uploaded bytes; a filename is metadata, never a path."""
    if not filename.strip() or len(filename) > 240:
        raise ValueError("Provide a filename of 1–240 characters")
    if len(content_base64) > 4 * ((MAX_BYTES + 2) // 3):
        raise ValueError("Source exceeds the 8 MB upload limit")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Source must be standard base64") from exc
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Source must contain 1–8,000,000 bytes")
    return data


class _WorkbookXML(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        """Reject a DTD before the parser can expand its entities."""
        raise ValueError("XML declarations containing entities are unsupported")


def _xml(archive, name):
    return ET.fromstring(archive.read(name), parser=ET.XMLParser(target=_WorkbookXML()))


def _zip(data):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid OOXML workbook") from exc
    members = archive.infolist()
    if len(members) > 2000 or sum(x.file_size for x in members) > MAX_EXPANDED:
        archive.close()
        raise ValueError("Workbook exceeds expanded ZIP resource limits")
    names = [x.filename for x in members]
    if len(set(names)) != len(names) or any(x.flag_bits & 1 for x in members):
        archive.close()
        raise ValueError("Duplicate ZIP members or encrypted workbooks unsupported")
    # Check before the workbook parser processes any XML, including shared strings.
    try:
        manifest = _xml(archive, "[Content_Types].xml")
        xml_names = {name for name in names if name.lower().endswith((".xml", ".rels"))}
        extensions = set()
        for entry in manifest:
            if not entry.get("ContentType", "").lower().endswith(("+xml", "/xml")):
                continue
            if entry.get("PartName"):
                xml_names.add(entry.attrib["PartName"].lstrip("/"))
            elif entry.get("Extension"):
                extensions.add(entry.attrib["Extension"].lower())
        xml_names.update(name for name in names if name.lower().rpartition(".")[2] in extensions)
        for name in xml_names - {"[Content_Types].xml"}:
            _xml(archive, name)
    except (ET.ParseError, KeyError, RuntimeError, ValueError) as exc:
        archive.close()
        raise ValueError("Invalid workbook XML") from exc
    return archive


def _json_cell(value):
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sheet_metadata(archive):
    from openpyxl.utils.cell import column_index_from_string, range_boundaries

    relations = {
        r.attrib["Id"]: r.attrib["Target"]
        for r in _xml(archive, "xl/_rels/workbook.xml.rels")
    }
    result = {}
    for sheet in _xml(archive, "xl/workbook.xml").findall("s:sheets/s:sheet", NS):
        target = relations[sheet.attrib[REL]]
        path = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        root = _xml(archive, path)
        stored_rows = root.findall("s:sheetData/s:row", NS)
        stored_cells = root.findall("s:sheetData/s:row/s:c", NS)
        row_numbers = [int(r.attrib["r"]) for r in stored_rows]
        if row_numbers != sorted(set(row_numbers)) or any(not 1 <= r <= MAX_ROWS for r in row_numbers):
            raise ValueError("Worksheet row identities are duplicated or outside limits")
        coordinates = [c.attrib["r"] for c in stored_cells]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("Worksheet contains duplicate cell coordinates")
        # openpyxl converts these lexical values to zero or infinity. Preserve
        # the source token so they cannot become a measured zero in a snapshot.
        numeric_errors = {}
        for cell in stored_cells:
            token = cell.findtext("s:v", None, NS)
            if cell.get("t", "n") != "n" or not token:
                continue
            value = float(token)
            if not math.isfinite(value):
                numeric_errors[cell.get("r")] = (token, "nonfinite-numeric-source-value")
            elif value == 0 and _nonzero_significand(token):
                numeric_errors[cell.get("r")] = (token, "numeric-underflow")
        actual_column, actual_row = 0, max(row_numbers, default=0)
        for coordinate in coordinates:
            match = re.fullmatch(r"([A-Z]{1,3})([1-9]\d*)", coordinate)
            if not match:
                raise ValueError("Invalid worksheet cell coordinate")
            column, number = column_index_from_string(match[1]), int(match[2])
            if column > MAX_COLUMNS or number > MAX_ROWS:
                raise ValueError("Worksheet cells exceed row/column limits")
            actual_column, actual_row = max(actual_column, column), max(actual_row, number)
        for row in stored_rows:
            if any(int(re.search(r"\d+$", c.attrib["r"])[0]) != int(row.attrib["r"])
                   for c in row.findall("s:c", NS)):
                raise ValueError("Worksheet cell coordinate disagrees with its parent row")
        dimension = root.find("s:dimension", NS)
        if dimension is not None:
            _, _, last_column, last_row = range_boundaries(dimension.attrib["ref"])
            if last_column < actual_column or last_row < actual_row:
                raise ValueError("Worksheet dimension excludes stored source cells")
        hidden = [int(r.attrib["r"]) for r in root.findall("s:sheetData/s:row", NS)
                  if r.attrib.get("hidden") in {"1", "true"}]
        if len(root.findall("s:sheetData/s:row/s:c", NS)) > MAX_CELLS:
            raise ValueError("Worksheet exceeds the 200,000 cell limit")
        relpath = posixpath.dirname(path) + "/_rels/" + posixpath.basename(path) + ".rels"
        rels = {r.attrib["Id"]: r.attrib for r in _xml(archive, relpath)} if relpath in archive.namelist() else {}
        links = []
        for link in root.findall("s:hyperlinks/s:hyperlink", NS):
            relationship = rels.get(link.attrib.get(REL), {})
            target = relationship.get("Target")
            location = link.attrib.get("location")
            links.append({"cell": link.attrib["ref"], "kind": "stored-link",
                          "target": target, "location": location,
                          "display": link.attrib.get("display"),
                          "tooltip": link.attrib.get("tooltip"),
                          "target_mode": relationship.get("TargetMode"),
                          "resolution": _target_kind(target, location)})
        result[sheet.attrib["name"]] = {"bounds": (actual_row, actual_column), "numeric_errors": numeric_errors,
                                         "hidden_rows": hidden, "links": links,
                                         "state": sheet.attrib.get("state", "visible")}
    return result


def _target_kind(target, location=None):
    if target is None:
        return "internal-workbook-location" if location else "missing-target"
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        return "absolute-target-no-fetch"
    if target.startswith("#"):
        return "internal-workbook-location"
    return "relative-to-original-workbook-no-server-resolution"


def _headers(row):
    counts = {}
    for cell in row:
        label = str(cell["value"]) if cell["value"] is not None else ""
        counts[label] = counts.get(label, 0) + 1
    return [{"column": c["column"], "label": str(c["value"]) if c["value"] is not None else "",
             "duplicate": counts[str(c["value"]) if c["value"] is not None else ""] > 1,
             "cell": c["cell"]} for c in row]


def read_source(filename, content_base64, *, encoding="utf-8-sig", delimiter=",", decimal="."):
    """Return all bounded source cells without evaluating formulas or queries."""
    data = decode_source(filename, content_base64)
    if decimal not in {".", ","} or delimiter not in {",", ";", "\t", "|"}:
        raise ValueError("Declare a supported delimiter and decimal convention")
    file_info = {"filename": filename, "sha256": _digest(data), "size_bytes": len(data)}
    sheets = []
    text_size = 0
    if filename.lower().endswith((".csv", ".tsv")):
        if encoding not in {"utf-8-sig", "utf-8", "cp1252", "utf-16"}:
            raise ValueError("Unsupported declared encoding")
        try:
            rows = []
            reader = csv.reader(io.StringIO(data.decode(encoding), newline=""), delimiter=delimiter, strict=True)
            count = 0
            for number, values in enumerate(reader, 1):
                count += len(values)
                if number > MAX_ROWS or len(values) > MAX_COLUMNS or count > MAX_CELLS:
                    raise ValueError("CSV exceeds row/column/cell resource limits")
                for value in values:
                    text_size = _count_cell_text(text_size, value, value)
                cells = [{"column": str(i), "cell": f"{i}:{number}", "value": v,
                          "cached_value": v, "formula": None, "error": None}
                         for i, v in enumerate(values, 1)]
                rows.append({"row": number, "hidden": False, "cells": cells})
        except (UnicodeError, csv.Error) as exc:
            raise ValueError("CSV does not match declared encoding/delimiter/quoting") from exc
        sheets.append({"name": "CSV", "state": "visible", "hidden_rows": [],
                       "links": [], "rows": rows})
    elif filename.lower().endswith((".xlsx", ".xlsm")):
        import openpyxl
        try:
            with _zip(data) as archive:
                metadata = _sheet_metadata(archive)
            formulas = openpyxl.load_workbook(io.BytesIO(data), read_only=True,
                                             data_only=False, keep_links=False)
            cached = openpyxl.load_workbook(io.BytesIO(data), read_only=True,
                                           data_only=True, keep_links=False)
            try:
                total = 0
                for sheet in formulas:
                    actual_row, actual_column = metadata[sheet.title].pop("bounds")
                    numeric_errors = metadata[sheet.title].pop("numeric_errors")
                    max_row = sheet.max_row or actual_row
                    max_column = sheet.max_column or actual_column
                    if max_row > MAX_ROWS or max_column > MAX_COLUMNS:
                        raise ValueError("Worksheet dimensions exceed row/column limits")
                    total += max_row * max_column
                    if total > MAX_CELLS:
                        raise ValueError("Workbook exceeds the 200,000 cell limit")
                    rows = []
                    bounds = {"max_row": max_row, "max_col": max_column}
                    hidden_rows = set(metadata[sheet.title]["hidden_rows"])
                    for number, (raw, cache) in enumerate(zip(sheet.iter_rows(**bounds), cached[sheet.title].iter_rows(**bounds), strict=True), 1):
                        cells = []
                        for i, (cell, saved) in enumerate(zip(raw, cache, strict=True), 1):
                            address = f"{openpyxl.utils.get_column_letter(i)}{number}"
                            formula = cell.value if cell.data_type == "f" else None
                            if formula is not None and not isinstance(formula, str):
                                raise ValueError("Unsupported OOXML formula representation")
                            value = saved.value if formula else cell.value
                            error = value if saved.data_type == "e" or cell.data_type == "e" else None
                            if isinstance(value, float) and not math.isfinite(value):
                                error = "nonfinite-numeric-source-value"
                            item = {"column": openpyxl.utils.get_column_letter(i),
                                          "cell": address,
                                          "value": _json_cell(value),
                                          "cached_value": _json_cell(saved.value),
                                          "formula": formula,
                                          "error": error}
                            if address in numeric_errors:
                                token, reason = numeric_errors[address]
                                item.update(value=None, cached_value=None, raw_numeric_token=token, error=reason)
                            text_size = _count_cell_text(text_size, item["value"], item["cached_value"],
                                                        formula, item["error"], item.get("raw_numeric_token"))
                            cells.append(item)
                        rows.append({"row": number, "hidden": number in hidden_rows, "cells": cells})
                    sheets.append({"name": sheet.title, **metadata[sheet.title], "rows": rows})
            finally:
                formulas.close()
                cached.close()
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError,
                zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, ET.ParseError) as exc:
            raise ValueError("Unsupported or invalid OOXML workbook structure") from exc
    else:
        raise ValueError("Supported source formats are CSV, TSV, XLSX and XLSM")
    for sheet in sheets:
        sheet["headers"] = _headers(sheet["rows"][0]["cells"]) if sheet["rows"] else []
    return {"source_file": file_info, "format": {"encoding": encoding, "delimiter": delimiter,
                                               "decimal": decimal}, "sheets": sheets}


def preview_source(filename, content_base64, *, preview_rows=20, **options):
    """Return worksheet census and a bounded row preview with stored evidence."""
    if isinstance(preview_rows, bool) or not isinstance(preview_rows, int) or not 1 <= preview_rows <= 200:
        raise ValueError("Preview rows must be 1–200")
    source = read_source(filename, content_base64, **options)
    for sheet in source["sheets"]:
        rows = sheet["rows"]
        sheet["row_count"] = len(rows)
        sheet["formula_count"] = sum(bool(c["formula"]) for r in rows for c in r["cells"])
        sheet["error_count"] = sum(bool(c["error"]) for r in rows for c in r["cells"])
        sheet["native_link_count"] = len(sheet["links"])
        sheet["rows"] = rows[:preview_rows]
    source["mapping_fields"] = {**{k: ["text"] for k in sorted(TEXT_FIELDS)},
                                **{k: list(v) for k, v in UNITS.items()}}
    return source


def _number(value, unit, units, decimal, *, allow_negative=False):
    if isinstance(value, bool) or value is None or value == "":
        return None, "missing-or-nonnumeric"
    if unit not in units:
        raise ValueError("Mapping unit is unsupported for its field")
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return None, "nonfinite-or-overflow"
    else:
        token = str(value).strip()
        point = re.escape(decimal)
        match = re.match(rf"^([+-]?(?:\d+(?:{point}\d*)?|{point}\d+)(?:[eE][+-]?\d+)?)", token)
        if not match:
            return None, "not-a-number-in-declared-unit"
        remaining = token[match.end():].strip()
        # Only enumerated SI units of this physical field are recognized. A
        # bare number uses the declared unit; suffixes supply their own unit.
        for suffix in sorted(units, key=len, reverse=True):
            if remaining.startswith(suffix):
                unit, remaining = suffix, remaining[len(suffix):].strip()
                break
        # Test-condition tails are retained in the original cell. They are
        # never concatenated into numbers or interpreted as measured models.
        if remaining and not (remaining.startswith(("@", "bei ")) or
                              (remaining.startswith("(") and remaining.endswith(")"))):
            return None, "unrecognized-unit-condition-or-decimal-convention"
        token = match[1].replace(",", ".")
        number = float(token)
        if number == 0 and _nonzero_significand(token):
            return None, "numeric-underflow"
    result = number * units[unit]
    if not math.isfinite(result):
        return None, "nonfinite-or-overflow"
    if result == 0 and number != 0:
        return None, "numeric-underflow"
    if result < 0 and not allow_negative:
        return None, "negative-catalog-rating-or-price"
    return result, None


def build_snapshot(filename, content_base64, *, sheet, mapping, metadata,
                   header_row=1, **options):
    """Map explicit columns into a new content-addressed, immutable snapshot."""
    mapping = json.loads(_canonical(mapping))
    metadata = {"currency": None, "snapshot": None, "quote_quantity": None,
                "technology": "Unknown", "label": None, "conditions": {},
                "workbook_profile": None,
                **json.loads(_canonical(metadata))}
    if metadata["workbook_profile"] not in (None, "mosfet", "igbt"):
        raise ValueError("Workbook profile must be mosfet or igbt")
    if metadata["quote_quantity"] is not None:
        if isinstance(metadata["quote_quantity"], bool) or not isinstance(metadata["quote_quantity"], (int, float)) or metadata["quote_quantity"] < 0:
            raise ValueError("Recorded quote quantity must be a nonnegative number")
        metadata["quote_quantity"] = float(metadata["quote_quantity"])
    if any(isinstance(v, bool) for v in metadata["conditions"].values()):
        raise ValueError("A condition must be a number, text or null")
    metadata["conditions"] = {
        k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        for k, v in metadata["conditions"].items()
    }
    source = read_source(filename, content_base64, **options)
    selected = next((s for s in source["sheets"] if s["name"] == sheet), None)
    if selected is None:
        raise ValueError("Unknown worksheet")
    if isinstance(header_row, bool) or not isinstance(header_row, int) or not 1 <= header_row <= len(selected["rows"]):
        raise ValueError("Header row is outside the worksheet")
    if "part_number" not in mapping or not mapping:
        raise ValueError("Map a part_number column explicitly")
    columns = {c["column"] for r in selected["rows"] for c in r["cells"]}
    for field, spec in mapping.items():
        if field not in TEXT_FIELDS | UNITS.keys() or spec.get("column") not in columns:
            raise ValueError("Unknown mapping field or source column")
        allowed = {"text": 1} if field in TEXT_FIELDS else UNITS[field]
        if spec.get("unit") not in allowed or set(spec) != {"column", "unit"}:
            raise ValueError("Each mapping needs a supported explicit column and unit")
    if set(metadata) - {"currency", "snapshot", "quote_quantity", "technology", "conditions", "label", "workbook_profile"}:
        raise ValueError("Unknown source metadata")
    canonical = {"parser": "catalog-sources/1", "source": source["source_file"], "sheet": sheet, "mapping": mapping,
                 "metadata": metadata, "header_row": header_row, "format": source["format"]}
    identity = _digest(_canonical(canonical))
    source_id = f"upload-{source['source_file']['sha256']}:{sheet}"
    records, errors = [], []
    header_cells = {cell["column"]: cell for cell in selected["rows"][header_row - 1]["cells"]}
    for row in selected["rows"][header_row:]:
        cells = {c["column"]: c for c in row["cells"]}
        record = {"id": f"{source_id}:{row['row']}", "source_id": source_id,
                  "row": row["row"], "part_number": "", "manufacturer": "Unknown",
                  "technology": metadata.get("technology", "Unknown"), "package": "",
                  "conditions": metadata.get("conditions", {}),
                  "workbook_profile": metadata["workbook_profile"],
                  "mapping_evidence": {},
                  "quote_quantity": metadata.get("quote_quantity"),
                  "quote_context": {"currency": metadata.get("currency") or "unspecified",
                                    "snapshot": metadata.get("snapshot"),
                                    "quote_quantity": metadata.get("quote_quantity")},
                  "raw": cells, "hidden": row["hidden"],
                  "provenance": {**source["source_file"], "sheet": sheet, "row": row["row"]}}
        row_errors = []
        for field, spec in mapping.items():
            cell = cells.get(spec["column"], {})
            record["mapping_evidence"][field] = {**spec, "cell": cell.get("cell")}
            value = cell.get("value")
            reason = None
            if cell.get("error"):
                reason = cell["error"] if "raw_numeric_token" in cell else (
                    "cached-formula-error" if cell["formula"] else "source-cell-error"
                )
            if field in TEXT_FIELDS:
                record[field] = str(value) if value is not None and not reason else ""
            else:
                if field in {"eon_j_a", "eoff_j_a", "err_j_a"}:
                    agreement = coefficient_cache_evidence(cell.get("formula"), value, error=cell.get("error"))
                    if agreement["literal_syntax"]:
                        value = agreement["value"]
                        reason = reason or agreement["validation"]
                        record["mapping_evidence"][field]["coefficient_cache_status"] = agreement["status"]
                        if agreement["literal"] is not None:
                            expected_header = {"eon_j_a": "EON", "eoff_j_a": "EOFF", "err_j_a": "ERR"}[field] + " (TJ=175°C)"
                            record["mapping_evidence"][field]["coefficient_reference"] = {
                                "reference_current_a": agreement["literal"]["reference_current_a"],
                                "coefficient_source_value": agreement["literal"]["coefficient_j_per_a"],
                                "literal_energy_source_value": agreement["literal"]["literal_energy_j"],
                                "literal_energy_source_unit": spec["unit"].removesuffix("/A"),
                                "source_unit": spec["unit"],
                                "factor_to_si": UNITS[field][spec["unit"]],
                                "reference_temperature_c": 175 if header_cells.get(spec["column"], {}).get("value") == expected_header else None,
                            }
                number, numeric_reason = _number(value, spec["unit"], UNITS[field], source["format"]["decimal"],
                                                 allow_negative=field in {"model_temperature_c", "junction_limit_c"})
                record[field] = None if reason else number
                reason = reason or numeric_reason
            if reason:
                record["mapping_evidence"][field]["validation"] = reason
                row_errors.append({"field": field, "cell": cell.get("cell"), "reason": reason})
        for item in record["mapping_evidence"].values():
            if "coefficient_reference" in item:
                item["coefficient_reference"]["reference_voltage_v"] = record.get("switch_voltage_v")
        record["quote_context"]["quote_quantity"] = record["quote_quantity"]
        if not record["part_number"].strip():
            row_errors.append({"field": "part_number", "reason": "missing-part-identifier-row-rejected"})
        else:
            records.append(record)
        if row_errors:
            errors.append({"row": row["row"], "record_id": record["id"], "errors": row_errors})
    snapshot = {"schema_version": 1, "snapshot_id": identity, "version": identity,
            "source_file": {**source["source_file"], "content_base64": content_base64},
            "mapping": canonical, "sources": [{"id": source_id, "file": filename, "snapshot_id": identity,
            "sheet": sheet, "sha256": source["source_file"]["sha256"],
            "currency": metadata.get("currency") or "unspecified",
            "snapshot": metadata.get("snapshot") or "supplied source; acquisition date unknown",
            "headers": _headers(selected["rows"][header_row - 1]["cells"]),
            "record_count": len(records), "notes": "Uploaded historical/source quote; no external refresh executed."}],
            "records": records, "row_errors": errors}
    check_json_budget(snapshot)
    return snapshot


def compare_snapshots(before, after, *, key_fields):
    """Compare explicit unique row keys, never silently merge duplicate parts."""
    if (not key_fields or len(set(key_fields)) != len(key_fields)
            or any(k not in TEXT_FIELDS | {"row"} for k in key_fields)):
        raise ValueError("Choose explicit supported key fields (include row for duplicate parts)")
    if any(len(snapshot["records"]) > MAX_ROWS for snapshot in (before, after)):
        raise ValueError("Snapshot comparison exceeds 10,000 records per side")
    check_json_budget({"before": before, "after": after, "key_fields": key_fields})
    def indexed(snapshot):
        result = {}
        for record in snapshot["records"]:
            if not isinstance(record.get("id"), str):
                raise ValueError("Every comparison record needs its original string id")
            key = tuple(record.get(k) for k in key_fields)
            if any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in key):
                raise ValueError("Comparison keys must be string identifiers or integer rows")
            if any(v is None or v == "" for v in key) or key in result:
                raise ValueError("Comparison keys must be present and unique; disambiguate duplicate rows")
            result[key] = record
        return result
    old, new = indexed(before), indexed(after)
    ignored = {"id", "source_id", "provenance"}
    changes = []
    for key in old:
        if key not in new:
            continue
        fields = [field for field in old[key].keys() | new[key].keys()
                  if field not in ignored and old[key].get(field) != new[key].get(field)]
        if fields:
            changes.append({"key": list(key), "before_id": old[key]["id"], "after_id": new[key]["id"], "fields": sorted(fields)})
    return {"key_fields": key_fields, "added": [new[k]["id"] for k in new if k not in old],
            "missing": [old[k]["id"] for k in old if k not in new], "changed": changes,
            "application": "comparison-only; no stored source overwritten"}


def static_links(filename, content_base64, *, cells=None, **options):
    """Inspect stored/literal/text links and GetURL formulas without execution."""
    source = read_source(filename, content_base64, **options)
    entries, formula_locations, unsupported = [], [], []
    literal = re.compile(r'^=HYPERLINK\(\s*"((?:[^"]|"")*)"\s*(?:[,;]\s*"((?:[^"]|"")*)"\s*)?\)$', re.I)
    for sheet in source["sheets"]:
        for link in sheet["links"]:
            entries.append({"sheet": sheet["name"], **link})
        for row in sheet["rows"]:
            for cell in row["cells"]:
                formula, value = cell["formula"], cell["value"]
                location = {"sheet": sheet["name"], "cell": cell["cell"]}
                if formula and re.match(r"^=GetURL\(", formula, re.I):
                    formula_locations.append({**location, "formula": formula, "status": "not-evaluated"})
                match = literal.fullmatch(formula or "")
                if match:
                    target = match[1].replace('""', '"')
                    entries.append({**location, "kind": "literal-HYPERLINK", "target": target,
                                    "display": match[2].replace('""', '"') if match[2] else None,
                                    "resolution": _target_kind(target)})
                elif formula and re.match(r"^=HYPERLINK\(", formula, re.I):
                    unsupported.append({**location, "formula": formula,
                                        "status": "nonliteral-target-not-evaluated"})
                elif not formula and isinstance(value, str) and re.fullmatch(r"(?:https?://|mailto:)[^\s]+", value):
                    entries.append({**location, "kind": "plain-URL-text", "target": value,
                                    "resolution": _target_kind(value)})
    native = sum(e["kind"] == "stored-link" for e in entries)
    if cells is not None:
        names = {sheet["name"] for sheet in source["sheets"]}
        if any(c["sheet"] not in names for c in cells):
            raise ValueError("Unknown worksheet in cell selection")
        entries = [{"sheet": c["sheet"], "cell": c["cell"],
                    "links": [e for e in entries if e["sheet"] == c["sheet"] and _covers(e["cell"], c["cell"])]}
                   for c in cells]
    return {"source_file": source["source_file"], "entries": entries,
            "native_link_count": native, "geturl_formula_locations": formula_locations,
            "unsupported_hyperlink_formulas": unsupported,
            "geturl_formula_count": len(formula_locations)}


def _covers(reference, coordinate):
    if reference == coordinate:
        return True
    if ":" not in reference or not re.fullmatch(r"[A-Z]{1,3}[1-9]\d*", coordinate):
        return False
    from openpyxl.utils.cell import range_boundaries
    try:
        left, top, right, bottom = range_boundaries(reference)
        column, row, _, _ = range_boundaries(coordinate)
        return left <= column <= right and top <= row <= bottom
    except (ValueError, TypeError):
        return False


def _literal_coefficient_division(expression):
    operand = r"((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    return re.fullmatch(rf"=?\s*{operand}\s*/\s*{operand}\s*", expression or "") if isinstance(expression, str) else None


def literal_energy_coefficient(expression):
    """Recognize only a nonnegative literal energy divided by positive current."""
    match = _literal_coefficient_division(expression)
    if not match:
        return None
    energy, current = float(match[1]), float(match[2])
    if (not math.isfinite(current) or not math.isfinite(energy) or current <= 0
            or energy == 0 and _nonzero_significand(match[1])):
        return None
    coefficient = energy / current
    if not math.isfinite(coefficient) or energy != 0 and coefficient <= 0:
        return None
    return {"coefficient_j_per_a": coefficient, "reference_current_a": current,
            "literal_energy_j": energy}


def coefficient_cache_evidence(expression, cache, *, error=None):
    """Check one named literal coefficient against its saved source cache."""
    literal = literal_energy_coefficient(expression)
    literal_syntax = _literal_coefficient_division(expression) is not None
    numeric = isinstance(cache, (int, float)) and not isinstance(cache, bool)
    if numeric:
        try:
            numeric = math.isfinite(cache)
        except OverflowError:
            numeric = False
    validation = None
    status = "unresolved-source-coefficient"
    value = cache
    if error or isinstance(cache, str) and cache.startswith("#"):
        validation, status = "cached-source-error", "cached-error"
    elif literal_syntax and literal is None:
        validation = status = "invalid-or-unrepresentable-coefficient-literal"
    elif literal is not None:
        expected = literal["coefficient_j_per_a"]
        if numeric and not math.isclose(cache, expected, rel_tol=1e-10, abs_tol=0):
            validation = status = "cache-formula-disagreement"
        elif cache is not None and not numeric and literal_energy_coefficient(cache) != literal:
            validation = status = "invalid-coefficient-cache-type"
        else:
            value = expected
            status = "literal-division-cache-missing" if cache is None else "literal-division"
    return {"literal": literal, "literal_syntax": literal_syntax,
            "numeric_cache": numeric, "validation": validation,
            "status": status, "value": None if validation else value}


def workbook_coefficients(filename, content_base64, **options):
    """Read documented MOSFET/IGBT J/A coefficient cells and their conditions."""
    source = read_source(filename, content_base64, **options)
    result = []
    for sheet in source["sheets"]:
        if sheet["name"] not in {"MOSFET", "IGBT"} or not sheet["rows"]:
            continue
        headers = {c["column"]: c for c in sheet["rows"][0]["cells"]}
        if headers.get("A", {}).get("value") != "Part-No.":
            continue
        for row in sheet["rows"][1:]:
            cells = {c["column"]: c for c in row["cells"]}
            if not cells.get("A", {}).get("value"):
                continue
            for column, component in (("Y", "e_on"), ("Z", "e_off"), ("AA", "e_rr")):
                cell = cells.get(column, {})
                header = headers.get(column, {})
                expected = {"Y": "EON", "Z": "EOFF", "AA": "ERR"}[column]
                if header.get("value") != expected + " (TJ=175°C)":
                    continue
                formula, cache = cell.get("formula"), cell.get("cached_value")
                agreement = coefficient_cache_evidence(formula, cache, error=cell.get("error"))
                literal = agreement["literal"]
                recognized = literal is not None
                current = literal["reference_current_a"] if literal else None
                energy = literal["literal_energy_j"] if literal else None
                coefficient = literal["coefficient_j_per_a"] if literal else None
                numeric_cache = agreement["numeric_cache"]
                bus = cells.get("X", {})
                bus_value = bus.get("value")
                bus_header = headers.get("X", {})
                expected_bus = {"MOSFET": "VDS switching losses", "IGBT": "VCE switch"}[sheet["name"]]
                bus_v = bus_value if bus_header.get("value") == expected_bus and isinstance(bus_value, (int, float)) and not isinstance(bus_value, bool) and math.isfinite(bus_value) and bus_value > 0 and not bus.get("error") else None
                status = agreement["status"]
                result.append({"sheet": sheet["name"], "row": row["row"],
                               "part_number": cells["A"]["value"], "component": component,
                               "cell": cell, "header": header, "unit": "J/A",
                               "recognized_literal_coefficient_j_per_a": coefficient,
                               "cached_coefficient_j_per_a": cache if numeric_cache and not cell.get("error") else None,
                               "reference_current_a": current if recognized else None,
                               "literal_energy_j": energy if recognized else None,
                               "reference_voltage_cell": bus, "reference_voltage_v": bus_v,
                               "reference_voltage_header_cell": bus_header,
                               "reference_temperature_c": 175.0,
                               "reference_temperature_header_cell": header["cell"],
                               "gate_resistance_ohm": None, "status": status})
    return {"source_file": source["source_file"], "coefficients": result,
            "scope": "Documented MOSFET/IGBT Y/Z/AA J/A fields only; literal divisions are recognized, never arbitrary formula evaluation. Numeric source cells have unresolved reference current."}
