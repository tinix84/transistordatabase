"""
Convert existing TDB JSON transistors to switching-pair format.

Each single device becomes a self-pair (device_with_device) where
high_side.device_id == low_side.device_id, with source='datasheet'.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def convert_tdb_to_self_pair(
    tdb_data: dict[str, Any], source_file: str = ""
) -> dict[str, Any]:
    """Convert a TDB JSON transistor to a self-pair switching pair dict.

    Args:
        tdb_data: Dictionary loaded from TDB JSON file
        source_file: Original source file path for migration tracking

    Returns:
        Dictionary suitable for SwitchingPair.from_dict()
    """
    # TDB format stores everything at root level (flat structure)
    device_id = tdb_data.get("name", "")
    device_type = tdb_data.get("type", "MOSFET")

    switch = tdb_data.get("switch", {})
    diode = tdb_data.get("diode", {})

    # Build conduction data from switch channel (TDB uses 'channel', not 'channel_data')
    conduction = _extract_conduction(switch.get("channel", []))

    # Build thermal model from foster data if available
    thermal_model = _extract_thermal_model(switch.get("thermal_foster", {}))

    # Build device side (shared structure for high and low)
    device_side = {
        "device_id": device_id,
        "count": 1,
        "metadata": {
            "name": device_id,
            "type": device_type,
            "manufacturer": tdb_data.get("manufacturer", ""),
            "housing_type": tdb_data.get("housing_type", ""),
            "datasheet_hyperlink": tdb_data.get("datasheet_hyperlink", ""),
            "datasheet_date": tdb_data.get("datasheet_date", ""),
            "datasheet_version": tdb_data.get("datasheet_version", ""),
            "cost": tdb_data.get("cost"),
            "weight": tdb_data.get("weight"),
        },
        "electrical_ratings": {
            "v_abs_max": tdb_data.get("v_abs_max", 0.0),
            "i_abs_max": tdb_data.get("i_abs_max", 0.0),
            "i_cont": tdb_data.get("i_cont", 0.0),
            "t_j_max": tdb_data.get("t_j_max", 175.0),
        },
        "thermal_properties": {
            "r_th_jc": tdb_data.get("r_th_jc", 0.0),
            "r_th_cs": tdb_data.get("r_th_cs", 0.0),
            "r_th_common": 0.0,
            "housing_area": tdb_data.get("housing_area", 0.0),
            "cooling_area": tdb_data.get("cooling_area", 0.0),
            "interface_area_forward": 0.0,
            "interface_area_reverse": 0.0,
            "thermal_model": thermal_model,
        },
        "conduction": conduction,
        "capacitance": {
            "c_oss": tdb_data.get("c_oss", []),
            "c_iss": tdb_data.get("c_iss", []),
            "c_rss": tdb_data.get("c_rss", []),
        },
        "gate_charge": switch.get("charge_curve", []),
        "soa": switch.get("soa", []),
    }

    # Build diode-side conduction (from diode.channel - TDB naming)
    diode_conduction = _extract_conduction(diode.get("channel", []))

    low_side = dict(device_side)
    low_side["role"] = "body_diode"
    if diode_conduction.get("curves"):
        low_side["conduction"] = diode_conduction

    # Build switching data from e_on, e_off, e_rr
    switching_data = _extract_switching_data(switch, diode)

    # Determine pair_type
    type_map = {
        "MOSFET": "mosfet_self",
        "SiC-MOSFET": "sic-mosfet_self",
        "IGBT": "igbt_self",
        "GaN": "gan_self",
        "GaN-Transistor": "gan_self",
    }
    pair_type = type_map.get(device_type, f"{device_type.lower()}_self")

    return {
        "$schema": "tdb-extended-v1.0",
        "pair_id": device_id,
        "pair_type": pair_type,
        "topology": "half_bridge",
        "high_side": {**device_side, "role": "switch"},
        "low_side": low_side,
        "switching_data": switching_data,
        "traces": {},
        "plecs_model": {
            "turn_on_formula": None,
            "turn_off_formula": None,
            "conduction_formula": None,
        },
        "migration": {
            "source_files": [source_file] if source_file else [],
            "migration_date": datetime.now().isoformat(),
            "schema_version": "1.0",
            "quality_score": 0.0,
        },
    }


def _extract_conduction(channel_data: list[dict] | None) -> dict[str, Any]:
    """Extract conduction curves from TDB channel_data format.

    Args:
        channel_data: List of channel characteristic dictionaries

    Returns:
        Dictionary with temperatures and curves
    """
    if not channel_data or channel_data is None:
        return {"temperatures": [], "curves": []}

    temperatures = []
    curves = []

    for ch in channel_data:
        t_j = ch.get("t_j", 25.0)
        temperatures.append(t_j)

        # TDB stores as graph_v_i: [[voltages], [currents]]
        graph = ch.get("graph_v_i", [[], []])
        if len(graph) >= 2:
            curves.append({
                "t_j": t_j,
                "on_state_voltage": graph[0] if isinstance(graph[0], list) else [],
                "on_state_current": graph[1] if isinstance(graph[1], list) else [],
            })
        else:
            curves.append({
                "t_j": t_j,
                "on_state_voltage": [],
                "on_state_current": [],
            })

    return {"temperatures": sorted(set(temperatures)), "curves": curves}


def _extract_thermal_model(
    foster_data: dict | None,
) -> dict[str, Any] | None:
    """Extract Foster thermal model from TDB format.

    Args:
        foster_data: Dictionary with 'r' and 'tau' arrays

    Returns:
        Thermal model dict or None
    """
    if not foster_data or foster_data is None:
        return None

    r_values = foster_data.get("r", [])
    tau_values = foster_data.get("tau", [])

    if not r_values:
        return None

    elements = []
    for i, r in enumerate(r_values):
        tau = tau_values[i] if i < len(tau_values) else 0.0
        c = tau / r if r > 0 else 0.0
        elements.append({"r": r, "c": c})

    return {"type": "foster", "elements": elements}


def _extract_switching_data(switch: dict, diode: dict) -> dict[str, Any]:
    """Extract pair-level switching data from TDB switch/diode data.

    Args:
        switch: Switch characteristics dict
        diode: Diode characteristics dict

    Returns:
        Switching data dict
    """
    # TDB uses 'e_on', 'e_off', etc. (not 'e_on_data')
    switching = {
        "source": "datasheet",
        "source_details": {
            "note": "Switching data from device datasheet (standard DPT conditions)",
        },
        "test_conditions": {
            "gate_resistance_on": 0.0,
            "gate_resistance_off": 0.0,
            "gate_voltage_on": 15.0,
            "gate_voltage_off": -5.0,
            "dc_bus_voltage": 0.0,
            "dead_time": 0.0,
            "parasitic_inductance": 0.0,
        },
        "turn_on": _extract_energy_data(switch.get("e_on", [])),
        "turn_off": _extract_energy_data(switch.get("e_off", [])),
        "reverse_recovery": _extract_energy_data(diode.get("e_rr", [])),
    }

    # Try to extract test conditions from first e_on entry
    e_on_data = switch.get("e_on", [])
    if e_on_data:
        first = e_on_data[0]
        switching["test_conditions"]["dc_bus_voltage"] = first.get("v_supply", 0.0)
        switching["test_conditions"]["gate_voltage_on"] = first.get("v_g", 15.0)
        switching["test_conditions"]["gate_resistance_on"] = first.get("r_g", 0.0)

    return switching


def _extract_energy_data(
    energy_list: list[dict] | None,
) -> dict[str, Any]:
    """Extract switching energy data from TDB e_on/e_off/e_rr format.

    Args:
        energy_list: List of energy data dicts at different temperatures

    Returns:
        Energy data dict with temperatures, current_axis, and energy curves
    """
    if not energy_list or energy_list is None:
        return {
            "temperatures": [],
            "current_axis": [],
            "voltage_axis": [],
            "energy": {},
        }

    temperatures = []
    energy_by_temp: dict[str, list[float]] = {}
    all_voltages = []

    # Find the reference current axis - use the entry with the most data points
    reference_currents: list[float] = []
    max_len = 0

    for entry in energy_list:
        t_j = entry.get("t_j", 25.0)
        if str(t_j) not in temperatures:
            temperatures.append(t_j)

        # TDB stores graph_i_e: [[currents], [energies]]
        graph = entry.get("graph_i_e")
        if graph and isinstance(graph, list) and len(graph) >= 2:
            currents = graph[0] if isinstance(graph[0], list) else []
            energies = graph[1] if isinstance(graph[1], list) else []
            if currents and energies and len(currents) == len(energies):
                # Store energies keyed by temperature
                if str(t_j) not in energy_by_temp:
                    energy_by_temp[str(t_j)] = energies
                    # Update reference if this has more points
                    if len(currents) > max_len:
                        reference_currents = currents
                        max_len = len(currents)
        elif entry.get("i_x") is not None and entry.get("e_x") is not None:
            # Single point data
            if str(t_j) not in energy_by_temp:
                energy_by_temp[str(t_j)] = [entry["e_x"]]

        v_supply = entry.get("v_supply", 0.0)
        if v_supply > 0 and v_supply not in all_voltages:
            all_voltages.append(v_supply)

    return {
        "temperatures": sorted(set(temperatures)),
        "current_axis": reference_currents,
        "voltage_axis": sorted(set(all_voltages)) if all_voltages else [],
        "energy": energy_by_temp,
    }


def migrate_tdb_to_pairs(
    source_dirs: list[Path], output_dir: Path
) -> dict[str, Any]:
    """Migrate all TDB JSON files to switching pair format.

    Args:
        source_dirs: List of directories containing TDB JSON files
        output_dir: Output directory for switching pair JSON files

    Returns:
        Statistics dict with conversion results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_files": 0,
        "converted": 0,
        "skipped": 0,
        "errors": 0,
        "duplicates": 0,
        "seen_ids": set(),
        "error_list": [],
    }

    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue

        for json_file in sorted(source_dir.rglob("*.json")):
            stats["total_files"] += 1

            try:
                with open(json_file, "r") as f:
                    tdb_data = json.load(f)

                # Check if it's a TDB-format file (has name key at root level)
                if "name" not in tdb_data:
                    logger.debug(f"Skipping non-TDB file: {json_file}")
                    stats["skipped"] += 1
                    continue

                device_id = tdb_data.get("name", "")

                if not device_id:
                    logger.warning(f"No device name in {json_file}")
                    stats["skipped"] += 1
                    continue

                # Deduplication
                if device_id in stats["seen_ids"]:
                    logger.debug(f"Duplicate device: {device_id}")
                    stats["duplicates"] += 1
                    continue

                stats["seen_ids"].add(device_id)

                # Convert
                pair_dict = convert_tdb_to_self_pair(
                    tdb_data,
                    source_file=str(
                        json_file.relative_to(json_file.parent.parent.parent)
                    )
                    if len(json_file.parts) > 3
                    else str(json_file),
                )

                # Calculate quality score using the model
                from transistordatabase.core.pair_models import SwitchingPair

                pair = SwitchingPair.from_dict(pair_dict)
                validation = pair.validate()
                pair_dict["migration"]["quality_score"] = validation["quality_score"]

                # Save
                output_path = output_dir / f"{device_id}.json"
                with open(output_path, "w") as f:
                    json.dump(pair_dict, f, indent=2)

                stats["converted"] += 1
                logger.info(
                    f"Converted: {device_id} "
                    f"(quality: {validation['quality_score']:.1f}%)"
                )

            except Exception as e:
                logger.error(f"Error converting {json_file}: {e}")
                stats["error_list"].append({"file": str(json_file), "error": str(e)})
                stats["errors"] += 1

    # Clean up for JSON serialization
    stats["seen_ids"] = list(stats["seen_ids"])
    return stats
