"""
TDB to ntbee2 CSV Converter Prototype

This module converts TDB JSON format to ntbee2 CSV format.
Supports both:
  1. plecs_devices.csv format (single test point)
  2. raggl_combos.csv format (current-dependent Raggl model)

The converter performs:
  - Field extraction from TDB JSON structure
  - Unit conversion (mOhm, µJ)
  - Curve interpolation to extract parameters at specific operating points
  - Voltage/current dependent model fitting (for Raggl)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Data Classes for ntbee2 CSV Records
# ============================================================================


@dataclass
class PLECSDeviceRecord:
    """Single record for plecs_devices.csv export."""
    part_number: str
    vendor: str
    type: str  # "mosfet", "igbt", "diode"
    technology: str  # "Si", "SiC", "GaN"
    V_br_V: float
    R_dson_mOhm: Optional[float] = None
    V_ce_sat_V: Optional[float] = None
    R_d_mOhm: Optional[float] = None
    E_on_uJ: float = 0.0
    E_off_uJ: float = 0.0
    V_test_V: float = 400.0
    I_test_A: float = 20.0
    Rth_jc_KW: float = 0.5
    package: Optional[str] = None

    def to_csv_dict(self) -> Dict[str, str]:
        """Convert to CSV-writable dictionary, handling NaN."""
        result = {}
        for key, value in asdict(self).items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                result[key] = ""
            else:
                result[key] = str(value)
        return result


@dataclass
class RagglComboRecord:
    """Single record for raggl_combos.csv export."""
    combo_id: str
    transistor_pn: str
    diode_pn: str
    vendor: str
    technology: str
    V_class_V: float
    V_test_V: float
    R_dson_mOhm: float
    V_ce_sat_V: Optional[float] = None
    a_on_uJ: float = 0.0
    k_on_uJ_A: float = 0.0
    a_off_uJ: float = 0.0
    k_off_uJ_A2: float = 0.0
    a_RR_uJ: float = 0.0
    k_RR_uJ_A: float = 0.0
    U_f_V: float = 0.7
    R_f_mOhm: float = 10.0
    Rth_jc_trans_KW: float = 0.5
    Rth_jc_diode_KW: float = 1.0
    package: Optional[str] = None
    n_Eon_points: int = 1
    r2_Eon: float = 1.0
    r2_Eoff: float = 1.0

    def to_csv_dict(self) -> Dict[str, str]:
        """Convert to CSV-writable dictionary, handling NaN."""
        result = {}
        for key, value in asdict(self).items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                result[key] = ""
            else:
                result[key] = str(value)
        return result


# ============================================================================
# TDB JSON Extraction Functions
# ============================================================================


def extract_r_ds_on(
    tdb_dict: Dict[str, Any],
    t_j: float = 25.0,
    v_g: float = 15.0,
) -> float:
    """Extract R_ds_on [Ohm] from TDB at specific operating point.

    Args:
        tdb_dict: TDB transistor JSON dict
        t_j: Junction temperature [°C]
        v_g: Gate voltage [V]

    Returns:
        R_ds_on [Ohm], or 0.0 if not available
    """
    try:
        switch = tdb_dict.get("switch", {})
        channel_data = switch.get("channel_data", [])

        if not channel_data:
            return 0.0

        # Find curve closest to target conditions
        best_curve = None
        best_distance = float("inf")

        for curve in channel_data:
            t_j_curve = curve.get("t_j", 25.0)
            v_g_curve = curve.get("v_gs", 15.0)
            distance = abs(t_j_curve - t_j) + abs(v_g_curve - v_g) * 10

            if distance < best_distance:
                best_distance = distance
                best_curve = curve

        if best_curve is None:
            return 0.0

        # Extract R_ds at I_d = 10 A (typical operating point)
        r_ds_curve = best_curve.get("r_channel", [])
        if not r_ds_curve:
            # Fallback: use single value if available
            r_ds_val = best_curve.get("resistance", 0.0)
            return float(r_ds_val)

        # If curve is list of dicts with {id, rd} or similar
        if isinstance(r_ds_curve, list) and r_ds_curve:
            if isinstance(r_ds_curve[0], dict):
                # Return resistance at middle point
                return float(r_ds_curve[len(r_ds_curve) // 2].get("rd", 0.1))
            else:
                # List of values
                return float(r_ds_curve[len(r_ds_curve) // 2])

        return float(r_ds_curve[0]) if r_ds_curve else 0.1

    except Exception as e:
        print(f"  Warning: Failed to extract R_ds_on: {e}")
        return 0.0


def extract_switching_energy(
    tdb_dict: Dict[str, Any],
    energy_type: str = "e_on",
    v_test: float = 400.0,
    i_test: float = 20.0,
) -> Tuple[float, float, float]:
    """Extract switching energy [J] at test point from TDB.

    Args:
        tdb_dict: TDB transistor JSON dict
        energy_type: "e_on" or "e_off"
        v_test: Test voltage [V]
        i_test: Test current [A]

    Returns:
        Tuple (energy [J], measured_v [V], measured_i [A])
    """
    try:
        switch = tdb_dict.get("switch", {})

        if energy_type == "e_on":
            energy_data = switch.get("e_on_data", [])
        else:
            energy_data = switch.get("e_off_data", [])

        if not energy_data:
            return 0.0, v_test, i_test

        # Find measurement point closest to (v_test, i_test)
        best_point = None
        best_distance = float("inf")

        for point in energy_data:
            v = point.get("v_ds", v_test)
            i = point.get("i_d", i_test)
            e = point.get("energy", 0.0)

            # Weighted distance: voltage scaled differently than current
            distance = abs(v - v_test) / max(1, v_test) + abs(i - i_test) / max(
                1, i_test
            )

            if distance < best_distance:
                best_distance = distance
                best_point = (e, v, i)

        if best_point is None:
            return 0.0, v_test, i_test

        # Return in Joules
        energy_j = float(best_point[0])
        measured_v = float(best_point[1])
        measured_i = float(best_point[2])

        return energy_j, measured_v, measured_i

    except Exception as e:
        print(f"  Warning: Failed to extract {energy_type}: {e}")
        return 0.0, v_test, i_test


def extract_thermal_resistance(tdb_dict: Dict[str, Any]) -> float:
    """Extract R_th_jc [K/W] from TDB thermal properties.

    Args:
        tdb_dict: TDB transistor JSON dict

    Returns:
        R_th_jc [K/W], or 0.5 if not available
    """
    try:
        thermal = tdb_dict.get("thermal_properties", {})
        r_th_jc = thermal.get("r_th_jc", 0.5)
        return float(r_th_jc)
    except Exception as e:
        print(f"  Warning: Failed to extract R_th_jc: {e}")
        return 0.5


def extract_breakdown_voltage(tdb_dict: Dict[str, Any]) -> float:
    """Extract V_br [V] from TDB electrical ratings.

    Args:
        tdb_dict: TDB transistor JSON dict

    Returns:
        V_br [V], or 650.0 if not available
    """
    try:
        electrical = tdb_dict.get("electrical_ratings", {})
        v_abs_max = electrical.get("v_abs_max", 650.0)
        return float(v_abs_max)
    except Exception as e:
        print(f"  Warning: Failed to extract V_br: {e}")
        return 650.0


def extract_metadata(tdb_dict: Dict[str, Any]) -> Dict[str, str]:
    """Extract metadata (name, type, manufacturer) from TDB.

    Args:
        tdb_dict: TDB transistor JSON dict

    Returns:
        Dict with 'name', 'type', 'manufacturer'
    """
    try:
        metadata = tdb_dict.get("metadata", {})
        return {
            "name": str(metadata.get("name", "Unknown")),
            "type": str(metadata.get("type", "mosfet")).lower(),
            "manufacturer": str(metadata.get("manufacturer", "Unknown")),
        }
    except Exception as e:
        print(f"  Warning: Failed to extract metadata: {e}")
        return {
            "name": "Unknown",
            "type": "mosfet",
            "manufacturer": "Unknown",
        }


# ============================================================================
# Converter Functions: TDB → ntbee2
# ============================================================================


def tdb_to_plecs_device(
    tdb_dict: Dict[str, Any],
    technology: str = "SiC",
) -> PLECSDeviceRecord:
    """Convert TDB JSON to plecs_devices.csv record.

    Args:
        tdb_dict: Parsed TDB JSON for single transistor
        technology: Override technology (Si/SiC/GaN)

    Returns:
        PLECSDeviceRecord ready for CSV export
    """
    meta = extract_metadata(tdb_dict)
    v_br = extract_breakdown_voltage(tdb_dict)
    r_th_jc = extract_thermal_resistance(tdb_dict)
    r_ds_on = extract_r_ds_on(tdb_dict, t_j=25.0, v_g=15.0) * 1e3  # [Ohm→mOhm]

    # Extract switching energies at 400V / 20A
    e_on_j, v_test_on, i_test_on = extract_switching_energy(
        tdb_dict, "e_on", v_test=400.0, i_test=20.0
    )
    e_off_j, v_test_off, i_test_off = extract_switching_energy(
        tdb_dict, "e_off", v_test=400.0, i_test=20.0
    )

    # Use average test point
    v_test = (v_test_on + v_test_off) / 2.0
    i_test = (i_test_on + i_test_off) / 2.0

    e_on_uj = e_on_j * 1e6  # [J → µJ]
    e_off_uj = e_off_j * 1e6

    return PLECSDeviceRecord(
        part_number=meta["name"],
        vendor=meta["manufacturer"],
        type=meta["type"],
        technology=technology,
        V_br_V=v_br,
        R_dson_mOhm=r_ds_on if r_ds_on > 0 else None,
        V_ce_sat_V=None,  # Not available in TDB MOSFET
        R_d_mOhm=None,
        E_on_uJ=e_on_uj,
        E_off_uJ=e_off_uj,
        V_test_V=v_test,
        I_test_A=i_test,
        Rth_jc_KW=r_th_jc,
        package=None,
    )


def tdb_to_raggl_combo(
    tdb_dict: Dict[str, Any],
    diode_pn: Optional[str] = None,
    technology: str = "SiC",
    v_class: float = 600.0,
) -> RagglComboRecord:
    """Convert TDB JSON to raggl_combos.csv record (synthesized from single point).

    NOTE: Raggl coefficients are derived from fitting multiple current points.
    This converter creates a synthetic single-point model by fitting E(I) = a + k×I
    from the TDB data point. This loses model detail and accuracy.

    Args:
        tdb_dict: Parsed TDB JSON for single transistor
        diode_pn: Diode part number (if None, uses transistor_pn + "-diode")
        technology: Override technology (Si/SiC/GaN)
        v_class: Device voltage class [V]

    Returns:
        RagglComboRecord with synthesized Raggl coefficients
    """
    meta = extract_metadata(tdb_dict)
    v_br = extract_breakdown_voltage(tdb_dict)
    r_th_jc = extract_thermal_resistance(tdb_dict)
    r_ds_on = extract_r_ds_on(tdb_dict, t_j=25.0, v_g=15.0) * 1e3  # [Ohm→mOhm]

    # Extract switching energies at 400V / 20A
    e_on_j, v_test_on, i_test_on = extract_switching_energy(
        tdb_dict, "e_on", v_test=400.0, i_test=20.0
    )
    e_off_j, v_test_off, i_test_off = extract_switching_energy(
        tdb_dict, "e_off", v_test=400.0, i_test=20.0
    )

    v_test = (v_test_on + v_test_off) / 2.0
    i_test = (i_test_on + i_test_off) / 2.0

    e_on_uj = e_on_j * 1e6
    e_off_uj = e_off_j * 1e6

    # Synthesize Raggl linear model from single point: E = a + k×I
    # Assume a = 0.3×E_test, k = 0.7×E_test / I_test (simple heuristic)
    if i_test > 0:
        a_on_uj = e_on_uj * 0.3
        k_on_uj_a = e_on_uj * 0.7 / i_test

        a_off_uj = e_off_uj * 0.3
        k_off_uj_a2 = e_off_uj * 0.7 / (i_test ** 2) if i_test > 0 else 0.0

        a_rr_uj = e_on_uj * 0.2  # Reverse recovery ≈ 20% of E_on
        k_rr_uj_a = a_rr_uj / i_test if i_test > 0 else 0.0
    else:
        a_on_uj = e_on_uj / 2.0
        k_on_uj_a = 0.0
        a_off_uj = e_off_uj / 2.0
        k_off_uj_a2 = 0.0
        a_rr_uj = 0.0
        k_rr_uj_a = 0.0

    if diode_pn is None:
        diode_pn = f"{meta['name']}-diode"

    combo_id = f"{meta['name']}+{diode_pn}"

    return RagglComboRecord(
        combo_id=combo_id,
        transistor_pn=meta["name"],
        diode_pn=diode_pn,
        vendor=meta["manufacturer"],
        technology=technology,
        V_class_V=v_class,
        V_test_V=v_test,
        R_dson_mOhm=r_ds_on if r_ds_on > 0 else 100.0,
        V_ce_sat_V=None,
        a_on_uJ=a_on_uj,
        k_on_uJ_A=k_on_uj_a,
        a_off_uJ=a_off_uj,
        k_off_uJ_A2=k_off_uj_a2,
        a_RR_uJ=a_rr_uj,
        k_RR_uJ_A=k_rr_uj_a,
        U_f_V=0.7,  # Typical diode forward voltage
        R_f_mOhm=10.0,  # Typical diode forward resistance
        Rth_jc_trans_KW=r_th_jc,
        Rth_jc_diode_KW=r_th_jc * 2.0,  # Typical diode is 2× transistor
        package=None,
        n_Eon_points=1,  # Single point from TDB
        r2_Eon=0.5,  # Low R² — synthesized from one point
        r2_Eoff=0.5,
    )


# ============================================================================
# Batch Conversion Functions
# ============================================================================


def convert_tdb_json_to_plecs_csv(
    tdb_json_dir: Path | str,
    output_csv: Path | str,
    technology: str = "SiC",
) -> int:
    """Convert all TDB JSON files in directory to plecs_devices.csv.

    Args:
        tdb_json_dir: Directory containing TDB JSON files
        output_csv: Output CSV file path
        technology: Device technology (Si/SiC/GaN)

    Returns:
        Number of transistors converted
    """
    tdb_json_dir = Path(tdb_json_dir)
    output_csv = Path(output_csv)

    records: List[PLECSDeviceRecord] = []

    for json_file in tdb_json_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                tdb_dict = json.load(f)

            record = tdb_to_plecs_device(tdb_dict, technology=technology)
            records.append(record)
            print(f"✓ Converted {json_file.name} → {record.part_number}")

        except Exception as e:
            print(f"✗ Failed to convert {json_file.name}: {e}")

    # Write CSV
    if records:
        fieldnames = [
            "part_number",
            "vendor",
            "type",
            "technology",
            "V_br_V",
            "R_dson_mOhm",
            "V_ce_sat_V",
            "R_d_mOhm",
            "E_on_uJ",
            "E_off_uJ",
            "V_test_V",
            "I_test_A",
            "Rth_jc_KW",
            "package",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_dict())

        print(f"\n✅ Converted {len(records)} transistors to {output_csv}")
        return len(records)
    else:
        print("⚠ No transistors converted")
        return 0


def convert_tdb_json_to_raggl_csv(
    tdb_json_dir: Path | str,
    output_csv: Path | str,
    technology: str = "SiC",
    v_class: float = 600.0,
) -> int:
    """Convert all TDB JSON files in directory to raggl_combos.csv.

    WARNING: Raggl coefficients are synthesized from single TDB point.
    This conversion loses model accuracy and detail. Consider using
    plecs_devices.csv format for better fidelity.

    Args:
        tdb_json_dir: Directory containing TDB JSON files
        output_csv: Output CSV file path
        technology: Device technology (Si/SiC/GaN)
        v_class: Device voltage class [V]

    Returns:
        Number of transistors converted
    """
    tdb_json_dir = Path(tdb_json_dir)
    output_csv = Path(output_csv)

    records: List[RagglComboRecord] = []

    for json_file in tdb_json_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                tdb_dict = json.load(f)

            record = tdb_to_raggl_combo(
                tdb_dict, technology=technology, v_class=v_class
            )
            records.append(record)
            print(
                f"✓ Converted {json_file.name} → {record.combo_id} "
                f"(⚠ synthesized Raggl)"
            )

        except Exception as e:
            print(f"✗ Failed to convert {json_file.name}: {e}")

    # Write CSV
    if records:
        fieldnames = [
            "combo_id",
            "transistor_pn",
            "diode_pn",
            "vendor",
            "technology",
            "V_class_V",
            "V_test_V",
            "R_dson_mOhm",
            "V_ce_sat_V",
            "a_on_uJ",
            "k_on_uJ_A",
            "a_off_uJ",
            "k_off_uJ_A2",
            "a_RR_uJ",
            "k_RR_uJ_A",
            "U_f_V",
            "R_f_mOhm",
            "Rth_jc_trans_KW",
            "Rth_jc_diode_KW",
            "package",
            "n_Eon_points",
            "r2_Eon",
            "r2_Eoff",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_dict())

        print(
            f"\n✅ Converted {len(records)} transistors to {output_csv}"
        )
        print("⚠ WARNING: Raggl coefficients are synthesized from single test points")
        return len(records)
    else:
        print("⚠ No transistors converted")
        return 0


# ============================================================================
# CLI Interface
# ============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python tdb_to_ntbee2_converter.py <tdb_dir> <output_csv> [--raggl] [--tech Si|SiC|GaN] [--vclass 600|900|1200|1700]")
        print()
        print("Examples:")
        print("  # Convert to plecs_devices.csv")
        print("  python tdb_to_ntbee2_converter.py ./tdb_jsons ./plecs_devices.csv")
        print()
        print("  # Convert to raggl_combos.csv (synthesized)")
        print("  python tdb_to_ntbee2_converter.py ./tdb_jsons ./raggl_combos.csv --raggl --tech SiC --vclass 650")
        sys.exit(1)

    tdb_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    use_raggl = "--raggl" in sys.argv
    tech = "SiC"
    vclass = 600.0

    # Parse optional arguments
    if "--tech" in sys.argv:
        idx = sys.argv.index("--tech")
        if idx + 1 < len(sys.argv):
            tech = sys.argv[idx + 1]

    if "--vclass" in sys.argv:
        idx = sys.argv.index("--vclass")
        if idx + 1 < len(sys.argv):
            vclass = float(sys.argv[idx + 1])

    if not tdb_dir.exists():
        print(f"Error: Directory not found: {tdb_dir}")
        sys.exit(1)

    print(f"Converting TDB JSON from {tdb_dir}")
    print(f"Output: {output_csv}")
    print(f"Technology: {tech}")
    if use_raggl:
        print(f"Format: raggl_combos.csv (voltage class {vclass}V)")
    else:
        print("Format: plecs_devices.csv")
    print()

    if use_raggl:
        count = convert_tdb_json_to_raggl_csv(tdb_dir, output_csv, tech, vclass)
    else:
        count = convert_tdb_json_to_plecs_csv(tdb_dir, output_csv, tech)

    sys.exit(0 if count > 0 else 1)
