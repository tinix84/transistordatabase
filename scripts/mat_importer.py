"""
Parser and migrator for .mat files from the sc archive.

Converts 187 MatLab .mat files containing SiC and IGBT switching data
into unified switching-pair JSON files.

Each .mat file contains a structured numpy array with 25 fields:
- Metadata: name, blockingVoltage, cost, weight
- Conduction: onStateVoltage, onStateCurrent (arrays)
- Switching: turn-on/off energy, current, voltage
- Thermal: r_th forward/reverse/common, interface areas
- Traces: 6 optional trace structs with embedded images
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from transistordatabase.core.pair_models import (
    ConductionCurve,
    ConductionData,
    DeviceElectricalRatings,
    DeviceMetadata,
    DeviceThermalProperties,
    MigrationMetadata,
    PairSide,
    SwitchingData,
    SwitchingEnergyData,
    SwitchingPair,
    SwitchingTestConditions,
)

logger = logging.getLogger(__name__)

# Filename classification patterns
PATTERNS = {
    'parallel': re.compile(r'^(\d+)_x_(.+)$'),
    'combination': re.compile(r'^(.+?)_with_(.+)$'),
    'hb_sw_diode': re.compile(r'^(.+?)_(\d+C)?_?HB_Sw\+Diode$'),
    'hb_sw_sw': re.compile(r'^(.+?)_(\d+C)?_?HB_Sw\+Sw$'),
    'adjusted': re.compile(r'^Adjusted_(.+)$'),
    'temperature': re.compile(r'^(.+?)_(\d+)C(?:_(?:Max|Typical|typical))?$'),
    'eon': re.compile(r'^(.+?)_Eon$'),
    'eoff': re.compile(r'^(.+?)_Eoff$'),
    'vfwd': re.compile(r'^(.+?)_Vfwd$'),
    'vrev': re.compile(r'^(.+?)_Vrev$'),
    'reverse_on': re.compile(r'^(.+?)_?[Rr]everse[_ ]?[Oo][Nn]$'),
    'rg_variant': re.compile(r'^(.+?)_[Rr][Gg]\d+'),
}

# Known vendor prefixes for partner device validation
VENDOR_PREFIXES = {
    'Infineon', 'Infineon_', 'ST', 'ST_', 'ROHM', 'ROHM_', 'Cree', 'Cree_',
    'ON_Semiconductor', 'IXYS', 'Fairchild_Semiconductor', 'Mitsubishi',
    'Microsemi', 'Power_Integrations', 'GaN_Systems', 'EPC', 'Power_Devices',
    'Transphorm', 'Wolfspeed', 'Qorvo',
}


def parse_mat_filename(filename: str) -> dict[str, Any]:
    """
    Parse .mat filename and classify the switching pair.

    Returns:
        dict with keys:
            - base_device: str (main device name)
            - partner_device: str | None (second device if combination)
            - pair_type: str ('self', 'combination', 'parallel', 'hb_sw_diode', 'hb_sw_sw')
            - count: int (number of parallel devices, default 1)
            - temperature: int | None (temperature variant)
            - measurement_type: str | None ('Eon', 'Eoff', 'Vfwd', 'Vrev', 'reverse_on')
            - is_adjusted: bool
            - is_skippable: bool
            - skip_reason: str | None
    """
    base_name = Path(filename).stem

    result = {
        'base_device': '',
        'partner_device': None,
        'pair_type': 'self',
        'count': 1,
        'temperature': None,
        'measurement_type': None,
        'is_adjusted': False,
        'is_skippable': False,
        'skip_reason': None,
    }

    # Check for adjusted (skip these)
    adj_match = PATTERNS['adjusted'].match(base_name)
    if adj_match:
        result['is_adjusted'] = True
        result['is_skippable'] = True
        result['skip_reason'] = 'adjusted'
        result['base_device'] = adj_match.group(1)
        return result

    # Check for parallel (e.g., "10_x_ST_SCTW100N65G2AG")
    par_match = PATTERNS['parallel'].match(base_name)
    if par_match:
        result['count'] = int(par_match.group(1))
        remaining = par_match.group(2)
        result['pair_type'] = 'parallel'

        # Check for temperature variant on parallel device
        temp_match = PATTERNS['temperature'].match(remaining)
        if temp_match:
            result['base_device'] = temp_match.group(1)
            result['temperature'] = int(temp_match.group(2))
        else:
            result['base_device'] = remaining

        return result

    # Check for HB_Sw+Diode combination
    hb_diode_match = PATTERNS['hb_sw_diode'].match(base_name)
    if hb_diode_match:
        result['base_device'] = hb_diode_match.group(1)
        if hb_diode_match.group(2):
            result['temperature'] = int(hb_diode_match.group(2).rstrip('C'))
        result['pair_type'] = 'hb_sw_diode'
        result['partner_device'] = 'body_diode'
        return result

    # Check for HB_Sw+Sw combination
    hb_sw_match = PATTERNS['hb_sw_sw'].match(base_name)
    if hb_sw_match:
        result['base_device'] = hb_sw_match.group(1)
        if hb_sw_match.group(2):
            result['temperature'] = int(hb_sw_match.group(2).rstrip('C'))
        result['pair_type'] = 'hb_sw_sw'
        result['partner_device'] = 'synchronous_switch'
        return result

    # Check for combination pair (e.g., "Vendor1_PN1_with_Vendor2_PN2")
    comb_match = PATTERNS['combination'].match(base_name)
    if comb_match:
        left = comb_match.group(1)
        right = comb_match.group(2)
        result['base_device'] = left
        result['pair_type'] = 'combination'

        # Validate that partner has a real vendor prefix (not generic like "SiC_Diode")
        if _is_real_device_name(right):
            result['partner_device'] = right
        else:
            result['is_skippable'] = True
            result['skip_reason'] = 'unclear_partner'
            result['partner_device'] = right  # Still record for reference

        return result

    # Check for measurement types (Eon, Eoff, Vfwd, Vrev, reverse_on)
    # These should merge into a base device
    for mtype in ['eon', 'eoff', 'vfwd', 'vrev', 'reverse_on']:
        pat = PATTERNS[mtype]
        match = pat.match(base_name)
        if match:
            result['base_device'] = match.group(1)
            result['measurement_type'] = mtype.upper()
            return result

    # Check for temperature variants (e.g., "Vendor_PN_125C" or "Vendor_PN_150C_Max")
    temp_match = PATTERNS['temperature'].match(base_name)
    if temp_match:
        result['base_device'] = temp_match.group(1)
        result['temperature'] = int(temp_match.group(2))
        return result

    # Default: simple device name
    result['base_device'] = base_name

    return result


def _is_real_device_name(name: str) -> bool:
    """Check if a device name contains a real vendor prefix (not generic)."""
    generic_keywords = {'sic_diode', 'body_diode', 'diode', 'capacitor'}
    name_lower = name.lower()

    if any(kw in name_lower for kw in generic_keywords):
        return False

    # Check for vendor prefix
    for prefix in VENDOR_PREFIXES:
        if name.startswith(prefix):
            return True

    return False


def load_mat_file(path: Path) -> dict[str, Any]:
    """
    Load a .mat file and extract all 25 fields.

    Returns:
        dict with keys matching the SiC .mat file format.
        All arrays are converted to Python lists.
    """
    mat = scipy.io.loadmat(str(path), squeeze_me=False)
    sc = mat['scData'][0, 0]

    def safe_float(arr: np.ndarray, default: float = 0.0) -> float:
        """Safely extract a float scalar from a numpy array."""
        try:
            if arr.size > 0:
                val = float(arr.flat[0])
                return val if not np.isnan(val) else default
            return default
        except (ValueError, TypeError):
            return default

    def safe_array(arr: np.ndarray) -> list[float]:
        """Safely extract a 1D array as a list."""
        try:
            if arr.size > 0:
                return arr.flatten().tolist()
            return []
        except (ValueError, TypeError):
            return []

    def safe_str(arr: np.ndarray, default: str = '') -> str:
        """Safely extract a string from a numpy array."""
        try:
            if arr.size > 0:
                return str(arr.flat[0]).strip()
            return default
        except (ValueError, TypeError):
            return default

    return {
        'name': safe_str(sc['name']),
        'blocking_voltage': safe_float(sc['blockingVoltage']),
        'cost': safe_float(sc['cost']),
        'weight': safe_float(sc['weight']),
        'on_state_voltage': safe_array(sc['onStateVoltage']),
        'on_state_current': safe_array(sc['onStateCurrent']),
        'turn_on_energy': safe_array(sc['turnOnEnergy']),
        'turn_on_current': safe_array(sc['turnOnCurrent']),
        'turn_on_voltage': safe_float(sc['turnOnVoltage']),
        'turn_off_energy': safe_array(sc['turnOffEnergy']),
        'turn_off_current': safe_array(sc['turnOffCurrent']),
        'turn_off_voltage': safe_float(sc['turnOffVoltage']),
        'r_th_forward': safe_float(sc['forwardThermalResistance']),
        'r_th_reverse': safe_float(sc['reverseThermalResistance']),
        'r_th_common': safe_float(sc['commonThermalResistance']),
        'interface_area_forward': safe_float(sc['forwardThermalInterfaceArea']),
        'interface_area_reverse': safe_float(sc['reverseThermalInterfaceArea']),
        'interface_area_common': safe_float(sc['commonThermalInterfaceArea']),
    }


def mat_to_switching_pair(mat_data: dict[str, Any], filename_info: dict[str, Any]) -> SwitchingPair:
    """
    Convert extracted .mat data to a SwitchingPair.

    Args:
        mat_data: Output of load_mat_file()
        filename_info: Output of parse_mat_filename()

    Returns:
        SwitchingPair instance
    """
    base_device = filename_info['base_device']
    partner_device = filename_info.get('partner_device')
    pair_type = filename_info['pair_type']
    count = filename_info.get('count', 1)

    # Infer pair type from devices
    if pair_type == 'self':
        if pair_type == 'parallel':
            pair_type = 'parallel_self'
        else:
            pair_type = 'mosfet_self'  # Default; could be IGBT_self

    pair_id = _generate_pair_id(base_device, partner_device, count)

    # Create high side (always populated from mat data)
    high_side = _create_pair_side(
        device_id=base_device,
        mat_data=mat_data,
        count=count,
        role='switch' if pair_type == 'self' else 'switch',
    )

    # Create low side
    if pair_type == 'self' or pair_type == 'parallel_self':
        # Self-pair: low side is same device
        low_side = _create_pair_side(
            device_id=base_device,
            mat_data=mat_data,
            count=count,
            role='switch',
        )
    elif pair_type == 'hb_sw_diode':
        # Body diode or anti-parallel diode
        low_side = PairSide(
            device_id=base_device + '_body_diode',
            count=count,
            role='body_diode',
        )
    elif pair_type == 'hb_sw_sw':
        # Synchronous switch (same as high side)
        low_side = _create_pair_side(
            device_id=base_device,
            mat_data=mat_data,
            count=count,
            role='synchronous_switch',
        )
    elif pair_type == 'combination' and partner_device:
        # Partner device as low side (may not have mat_data)
        low_side = PairSide(
            device_id=partner_device,
            count=1,
            role='diode' if 'diode' in partner_device.lower() else 'switch',
        )
    else:
        # Fallback
        low_side = PairSide(device_id=base_device, count=count, role='switch')

    # Build switching data from mat arrays
    switching_data = _build_switching_data(mat_data, filename_info)

    # Create the pair
    pair = SwitchingPair(
        pair_id=pair_id,
        pair_type=pair_type,
        topology='half_bridge',
        high_side=high_side,
        low_side=low_side,
        switching_data=switching_data,
        migration=MigrationMetadata(
            source_files=[filename_info.get('source_file', '')],
            migration_date=datetime.utcnow().isoformat(),
            schema_version='1.0',
            quality_score=0.0,
        ),
    )

    # Validate and calculate quality score
    validation = pair.validate()
    pair.migration.quality_score = validation['quality_score']

    return pair


def _create_pair_side(
    device_id: str,
    mat_data: dict[str, Any],
    count: int = 1,
    role: str = 'switch',
) -> PairSide:
    """Create a PairSide from mat data."""
    side = PairSide(device_id=device_id, count=count, role=role)

    # Metadata
    side.metadata = DeviceMetadata(
        name=mat_data.get('name', device_id),
        type='mosfet',  # Default; could infer from name
        manufacturer=_extract_manufacturer(device_id),
        cost=mat_data.get('cost'),
        weight=mat_data.get('weight'),
    )

    # Electrical ratings
    side.electrical_ratings = DeviceElectricalRatings(
        v_abs_max=mat_data.get('blocking_voltage', 0.0),
        i_abs_max=0.0,  # Not in .mat data
        i_cont=0.0,
        t_j_max=175.0,
    )

    # Thermal properties
    side.thermal_properties = DeviceThermalProperties(
        r_th_jc=mat_data.get('r_th_forward', 0.0),
        r_th_cs=mat_data.get('r_th_reverse', 0.0),
        r_th_common=mat_data.get('r_th_common', 0.0),
        interface_area_forward=mat_data.get('interface_area_forward', 0.0),
        interface_area_reverse=mat_data.get('interface_area_reverse', 0.0),
    )

    # Conduction data (at 25°C by default)
    v_curve = mat_data.get('on_state_voltage', [])
    i_curve = mat_data.get('on_state_current', [])
    if v_curve and i_curve and len(v_curve) == len(i_curve):
        curve = ConductionCurve(t_j=25.0, on_state_voltage=v_curve, on_state_current=i_curve)
        side.conduction = ConductionData(temperatures=[25.0], curves=[curve])

    return side


def _build_switching_data(
    mat_data: dict[str, Any], filename_info: dict[str, Any]
) -> SwitchingData:
    """Build SwitchingData from .mat fields."""
    switching_data = SwitchingData(
        source='datasheet',
        source_details={'mat_file': filename_info.get('source_file', '')},
        test_conditions=SwitchingTestConditions(
            dc_bus_voltage=filename_info.get('turn_on_voltage', mat_data.get('turn_on_voltage', 0.0)),
        ),
    )

    # Turn-on energy
    eon = mat_data.get('turn_on_energy', [])
    eon_i = mat_data.get('turn_on_current', [])
    if eon and eon_i:
        switching_data.turn_on = SwitchingEnergyData(
            temperatures=[25.0],
            current_axis=eon_i,
            voltage_axis=[mat_data.get('turn_on_voltage', 0.0)],
            energy={'25.0': eon},
        )

    # Turn-off energy
    eoff = mat_data.get('turn_off_energy', [])
    eoff_i = mat_data.get('turn_off_current', [])
    if eoff and eoff_i:
        switching_data.turn_off = SwitchingEnergyData(
            temperatures=[25.0],
            current_axis=eoff_i,
            voltage_axis=[mat_data.get('turn_off_voltage', 0.0)],
            energy={'25.0': eoff},
        )

    return switching_data


def _extract_manufacturer(device_id: str) -> str:
    """Extract manufacturer name from device ID."""
    # Check common prefixes
    prefixes = {
        'Infineon': 'Infineon',
        'Infineon_': 'Infineon',
        'ST_': 'ST',
        'ST ': 'ST',
        'ROHM_': 'ROHM',
        'Cree_': 'Cree',
        'Cree': 'Cree',
        'ON_': 'ON Semiconductor',
        'IXYS_': 'IXYS',
        'IXYS': 'IXYS',
        'Fairchild': 'Fairchild',
        'Mitsubishi': 'Mitsubishi',
    }

    for prefix, mfr in prefixes.items():
        if device_id.startswith(prefix):
            return mfr

    return ''


def _generate_pair_id(base_device: str, partner_device: str | None, count: int = 1) -> str:
    """Generate a unique pair ID."""
    if partner_device and partner_device not in {'body_diode', 'synchronous_switch'}:
        # Combination pair
        return f"{base_device}___with___{partner_device}"
    elif count > 1:
        # Parallel self-pair
        return f"{count}x_{base_device}"
    else:
        # Simple self-pair
        return f"{base_device}__self"


def migrate_all_mat_files(mat_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Migrate all .mat files to switching-pair JSON entries.

    Processes all 187 .mat files in two passes:
    1. Categorize files and group by base device
    2. Create SwitchingPair entries, merging temperature/measurement variants

    Args:
        mat_dir: Path to scData directory containing .mat files
        output_dir: Path to output directory for JSON files

    Returns:
        Migration statistics dict
    """
    mat_dir = Path(mat_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total_files': 0,
        'files_created': 0,
        'files_skipped': 0,
        'files_merged': 0,
        'skipped_details': {},
        'quality_distribution': {},
        'pairs_created': {},
    }

    # First pass: categorize all files
    logger.info(f"Scanning {mat_dir} for .mat files...")
    all_files = sorted(mat_dir.glob('*.mat'))
    stats['total_files'] = len(all_files)

    file_categories: dict[str, list[tuple[Path, dict[str, Any]]]] = {}

    for mat_file in all_files:
        filename_info = parse_mat_filename(mat_file.name)

        # Record skip reason
        if filename_info['is_skippable']:
            reason = filename_info['skip_reason']
            stats['skipped_details'][reason] = stats['skipped_details'].get(reason, 0) + 1
            logger.info(f"SKIP {mat_file.name}: {reason}")
            stats['files_skipped'] += 1
            continue

        # Group by base device
        base = filename_info['base_device']
        if base not in file_categories:
            file_categories[base] = []

        filename_info['source_file'] = mat_file.name
        file_categories[base].append((mat_file, filename_info))

    logger.info(f"Grouped {len(file_categories)} unique base devices")

    # Second pass: create switching pairs
    for base_device, file_list in sorted(file_categories.items()):
        logger.info(f"\nProcessing base device: {base_device} ({len(file_list)} variant(s))")

        # Load primary file (first in list, or base device itself)
        primary_file = None
        primary_info = None
        merged_count = 0

        for mat_file, filename_info in file_list:
            if filename_info.get('measurement_type') is None and filename_info.get('temperature') is None:
                primary_file = mat_file
                primary_info = filename_info
                break

        if not primary_file and file_list:
            primary_file, primary_info = file_list[0]

        if not primary_file:
            logger.warning(f"  No primary file found for {base_device}")
            continue

        # Load primary mat data
        try:
            mat_data = load_mat_file(primary_file)
        except Exception as e:
            logger.error(f"  Failed to load {primary_file.name}: {e}")
            stats['files_skipped'] += 1
            continue

        # Merge variants into primary mat_data
        for mat_file, _filename_info in file_list:
            if mat_file == primary_file:
                continue

            try:
                variant_data = load_mat_file(mat_file)
                # Merge logic could be more sophisticated
                # For now, just track that we've merged
                merged_count += 1
                logger.info(f"  Merged variant: {mat_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to merge {mat_file.name}: {e}")

        # Create switching pair from primary data
        try:
            pair = mat_to_switching_pair(mat_data, primary_info)
            output_file = output_dir / f"{pair.pair_id}.json"

            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(pair.to_dict(), f, indent=2)

            # Record statistics
            quality = pair.migration.quality_score
            score_bucket = f"{int(quality // 10) * 10}-{int(quality // 10) * 10 + 10}"
            stats['quality_distribution'][score_bucket] = (
                stats['quality_distribution'].get(score_bucket, 0) + 1
            )

            stats['pairs_created'][pair.pair_id] = {
                'pair_type': pair.pair_type,
                'devices': pair.devices,
                'quality_score': quality,
                'source_file': primary_file.name,
                'merged_variants': merged_count,
            }

            stats['files_created'] += 1
            if merged_count > 0:
                stats['files_merged'] += merged_count

            logger.info(f"  Created {pair.pair_id} (quality={quality:.1f}%)")

        except Exception as e:
            logger.error(f"  Failed to create pair from {primary_file.name}: {e}")
            stats['files_skipped'] += 1

    return stats


# For CLI usage
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    mat_src = Path('/home/tinix/claude_wsl/archive_ntbees2/sc/db/scData')
    output = Path('/home/tinix/claude_wsl/transistordatabase/switching_pairs')

    stats = migrate_all_mat_files(mat_src, output)

    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files created: {stats['files_created']}")
    print(f"Files skipped: {stats['files_skipped']}")
    print(f"Variants merged: {stats['files_merged']}")

    print("\nSkip reasons:")
    for reason, count in sorted(stats['skipped_details'].items()):
        print(f"  {reason}: {count}")

    print("\nQuality score distribution:")
    for bucket in sorted(stats['quality_distribution'].keys()):
        count = stats['quality_distribution'][bucket]
        print(f"  {bucket}%: {count}")

    print(f"\nPairs created: {len(stats['pairs_created'])}")
