"""Import the five sc archive catalogs into a portable screening snapshot.

Read source workbooks without executing formulas or macros. Store original
cells and cached formula values for inspection; recompute supported figures
from the quoted primary fields, not from legacy spreadsheet FOM columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import openpyxl

from transistordatabase.catalog_fom import CATALOG_PATH, parse_quantity


def technology(value: object) -> str:
    """Classify the source's explicit semiconductor description."""
    text = str(value).lower()
    if "igbt" in text:
        return "IGBT"
    if "sic" in text or "siliziumkarbid" in text:
        return "SiC"
    if "gan" in text or "gallium" in text:
        return "GaN"
    return "Si" if "mosfet" in text or "silicon" in text else "Unknown"


def normalize(
    headers: tuple | list,
    values: tuple | list,
    source: dict,
    row_number: int,
    formulas: tuple | list | None = None,
) -> dict | None:
    """Map the observed DigiKey/German/manual layouts with original units."""
    row = dict(zip(headers, values, strict=False))

    def field(*names: str) -> object:
        return next((row[n] for n in names if n in row), None)

    manual = "Part-No." in headers
    part = field("Mfr Part #", "Hersteller-Teilenummer", "Part-No.")
    if not part or str(part).startswith("="):
        return None
    q = parse_quantity
    if manual:
        # Manual MOSFET table: the first Rds(max) is mΩ at its explicit Vgs/Tj.
        # IGBT rCE must never be presented as MOSFET Rds(on).
        rds = q(values[14]) if source["sheet"] == "MOSFET" else None
        rds = rds * 1e-3 if rds else None
        qg = q(field("Qg(tot) typ"))
        qg = qg * 1e-9 if qg else None
        coss = q(field("COSS(effective)"))
        coss = coss * 1e-12 if coss else None
        conditions = {
            "resistance": f"max at Vgs={values[11]} V, Tj={values[12]} °C; source unit mΩ",
            "gate_charge": "typical; source unit nC; test conditions not separately recorded",
            "output_capacitance": "effective; source unit pF; test voltage not separately recorded",
            "current": str(values[6]) + " A at Tc=25 °C",
        }
        if source["sheet"] == "IGBT":
            conditions["resistance"] = (
                "IGBT VCE0/rCE fields are retained in original cells; "
                "they are not MOSFET Rds(on) and are excluded from its FOMs."
            )
    else:
        raw_r = field(
            "Rds On (Max) @ Id, Vgs",
            "Rds(On) (Max.) bei Id, Vgs",
            "Rds(On) (Max.) bei Id. Vgs",
        )
        raw_q = field("Gate Charge (Qg) (Max) @ Vgs", "Gateladung (Qg) (Max.) bei Vgs")
        rds, qg, coss = q(raw_r, "Ohm"), q(raw_q, "C"), None
        conditions = {
            "resistance": raw_r,
            "gate_charge": raw_q,
            "current": next(
                (
                    v
                    for k, v in row.items()
                    if k.startswith(("Current - Continuous", "Strom - Kontinuierlich"))
                ),
                None,
            ),
        }
    current_raw = values[6] if manual else conditions["current"]
    # Multi-condition strings must retain the first condition, not silently
    # choose a larger case-temperature rating later in the same cell.
    record = {
        "id": f"{source['id']}:{row_number}",
        "source_id": source["id"],
        "row": row_number,
        "part_number": str(part),
        "manufacturer": str(field("Mfr", "Hersteller", "Manufacturer") or "Unknown"),
        "technology": technology(field("Technology", "Technologie", "Semiconductor")),
        "package": str(
            field("Supplier Device Package", "Gehäusetyp vom Lieferanten", "Package")
            or ""
        ),
        "voltage_v": q(
            field(
                "Drain to Source Voltage (Vdss)",
                "Drain-Source-Spannung (Vdss)",
                "VDSS",
                "VCES",
            ),
            "V",
        ),
        "current_a": q(current_raw, "A"),
        "power_w": q(field("Power Dissipation (Max)", "Verlustleistung (max.)"), "W"),
        "rds_ohm": rds,
        "qg_c": qg,
        "coss_f": coss,
        "ciss_f": q(
            field(
                "Input Capacitance (Ciss) (Max) @ Vds",
                "Eingangskapazität (Ciss) (max.) bei Vds",
            ),
            "F",
        ),
        "price": q(field("Price", "Stückpreis (CHF)", "Cost at 1k pcs.")),
        "quote_quantity": field("@ qty") if not manual else 1000,
        "min_quantity": field("Min Qty", "Mindestbestellmenge ist"),
        "conditions": conditions,
        "datasheet": field("Datasheet", "Datenblätter"),
        "raw": {
            str(i + 1): {"header": str(h), "value": v}
            for i, (h, v) in enumerate(zip(headers, values, strict=False))
            if h is not None
        },
    }
    if formulas:
        record["formulas"] = {
            str(i + 1): v
            for i, v in enumerate(formulas)
            if isinstance(v, str) and v.startswith("=")
        }
    return record


def import_catalogs(archive: Path) -> dict:
    """Build records plus per-file hashes and worksheet provenance."""
    files = [
        "201210_sic_digikey.xlsx",
        "SiC_MOSFET_FOM.xlsx",
        "201210_mos_digikey.csv",
        "201210_gan_digikey.csv",
        "SemiconductorsData.xlsm",
    ]
    result = {"schema_version": 1, "sources": [], "records": []}
    for filename in files:
        path = archive / "db" / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".csv":
            # These historical exports contain Windows-1252 µ/° characters.
            with path.open(encoding="cp1252", newline="") as stream:
                rows = list(csv.reader(stream))
            sheets = [("CSV", rows, None)]
        else:
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            formula_wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
            sheets = [
                (ws.title, list(ws.values), list(formula_wb[ws.title].values))
                for ws in wb
            ]
            wb.close()
            formula_wb.close()
        for sheet, rows, formulas in sheets:
            if not rows or not any(
                h in rows[0]
                for h in ("Mfr Part #", "Hersteller-Teilenummer", "Part-No.")
            ):
                continue
            source = {
                "id": f"{path.stem}/{sheet}",
                "file": filename,
                "sheet": sheet,
                "original_path": str(path),
                "sha256": digest,
                "currency": "CHF"
                if filename == "SiC_MOSFET_FOM.xlsx"
                else "unspecified",
                "snapshot": "2020-12-10 (filename)"
                if filename.startswith("201210")
                else "historical; date unverified",
                "notes": (
                    "Prices at source quantity; @ qty=0 is not a valid confirmed quote tier. "
                    "Blank/error formula caches remain missing; macros are not executed."
                ),
            }
            records = [
                normalize(
                    rows[0], row, source, idx, formulas[idx - 1] if formulas else None
                )
                for idx, row in enumerate(rows[1:], start=2)
            ]
            records = [r for r in records if r is not None]
            source["record_count"] = len(records)
            result["sources"].append(source)
            result["records"].extend(records)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", type=Path, help="Read-only path to archive_ntbees2/sc"
    )
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    args = parser.parse_args()
    snapshot = import_catalogs(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sources": len(snapshot["sources"]),
                "records": len(snapshot["records"]),
                "output": str(args.output),
            }
        )
    )
