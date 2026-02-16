"""
PLECS XML parser and switching pair migration tool.

Parse PLECS semiconductor library XML files and create/merge switching pair entries.
Supports MOSFETs, IGBTs, SiC-MOSFETs, GaN, and Diodes.
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from transistordatabase.core.pair_models import (
    ConductionCurve,
    ConductionData,
    DeviceElectricalRatings,
    DeviceMetadata,
    DeviceThermalProperties,
    MigrationMetadata,
    PairSide,
    PLECSModel,
    SwitchingData,
    SwitchingEnergyData,
    SwitchingPair,
    SwitchingTestConditions,
    ThermalElement,
    ThermalModel,
)

logger = logging.getLogger(__name__)

# PLECS XML namespace
PLECS_NS = 'http://www.plexim.com/xml/semiconductors/'
NS = {'p': PLECS_NS}

# Device class to pair type mapping
DEVICE_CLASS_TO_PAIR_TYPE = {
    'MOSFET': 'mosfet_self',
    'IGBT': 'igbt_self',
    'SiC-MOSFET': 'sic-mosfet_self',
    'GaN': 'gan_self',
    'Diode': 'diode_self',
}


@dataclass
class PLECSDeviceData:
    """Parsed PLECS device data from XML."""

    vendor: str
    partnumber: str
    device_class: str  # MOSFET, IGBT, SiC-MOSFET, GaN, Diode
    turn_on: dict[str, Any] | None = None
    turn_off: dict[str, Any] | None = None
    conduction: dict[str, Any] | None = None
    thermal: dict[str, Any] | None = None
    formulas: dict[str, str | None] | None = None
    source_file: str = ""


def _parse_axis(parent: ET.Element, axis_name: str) -> list[float]:
    """
    Parse an axis (CurrentAxis, VoltageAxis, TemperatureAxis) from XML.

    Returns list of float values.
    """
    elem = parent.find(f'p:{axis_name}', NS)
    if elem is None:
        elem = parent.find(f'{axis_name}')

    if elem is None or elem.text is None:
        return []

    text = elem.text.strip()
    if not text:
        return []

    try:
        return [float(v) for v in text.split()]
    except (ValueError, AttributeError):
        logger.warning(f"Failed to parse {axis_name}: {text}")
        return []


def _extract_energy_table(
    package: ET.Element,
    loss_type: str,
) -> dict[str, Any] | None:
    """
    Extract TurnOnLoss or TurnOffLoss energy LUT.

    Returns dict with 'temperatures', 'current_axis', 'voltage_axis', 'energy'.
    """
    # Find the loss element (try with and without namespace)
    elem = package.find(f'.//p:{loss_type}', NS)
    if elem is None:
        elem = package.find(f'.//{loss_type}')
    if elem is None:
        return None

    current_axis = _parse_axis(elem, 'CurrentAxis')
    voltage_axis = _parse_axis(elem, 'VoltageAxis')
    temp_axis = _parse_axis(elem, 'TemperatureAxis')

    scale = 1.0
    energy_elem = elem.find('p:Energy', NS)
    if energy_elem is None:
        energy_elem = elem.find('Energy')

    if energy_elem is not None:
        scale_str = energy_elem.get('scale', '1.0')
        try:
            scale = float(scale_str)
        except (ValueError, TypeError):
            scale = 1.0

    # Parse energy values per temperature
    energy_by_temp = {}
    temp_elems = energy_elem.findall('p:Temperature', NS) if energy_elem is not None else []
    if not temp_elems:
        temp_elems = energy_elem.findall('Temperature') if energy_elem is not None else []

    for i, temp_elem in enumerate(temp_elems):
        temp_val = temp_axis[i] if i < len(temp_axis) else 25.0
        voltage_elems = temp_elem.findall('p:Voltage', NS)
        if not voltage_elems:
            voltage_elems = temp_elem.findall('Voltage')

        if voltage_elems:
            # Use first voltage level's values
            if voltage_elems[0].text:
                try:
                    values = [float(v) * scale for v in voltage_elems[0].text.split()]
                    energy_by_temp[str(temp_val)] = values
                except (ValueError, AttributeError):
                    logger.warning(f"Failed to parse energy values at temp {temp_val}")
        elif temp_elem.text:
            # Fallback: values directly in Temperature element
            try:
                values = [float(v) * scale for v in temp_elem.text.split()]
                energy_by_temp[str(temp_val)] = values
            except (ValueError, AttributeError):
                logger.warning(f"Failed to parse energy values at temp {temp_val}")

    return {
        'temperatures': temp_axis,
        'current_axis': current_axis,
        'voltage_axis': voltage_axis,
        'energy': energy_by_temp,
    }


def _extract_conduction_table(package: ET.Element) -> dict[str, Any] | None:
    """
    Extract ConductionLoss voltage-drop LUT.

    Returns dict with 'temperatures' and 'curves' (list of ConductionCurve dicts).
    """
    elem = package.find('.//p:ConductionLoss', NS)
    if elem is None:
        elem = package.find('.//ConductionLoss')
    if elem is None:
        return None

    current_axis = _parse_axis(elem, 'CurrentAxis')
    temp_axis = _parse_axis(elem, 'TemperatureAxis')

    scale = 1.0
    vdrop_elem = elem.find('p:VoltageDrop', NS)
    if vdrop_elem is None:
        vdrop_elem = elem.find('VoltageDrop')

    if vdrop_elem is not None:
        scale_str = vdrop_elem.get('scale', '1.0')
        try:
            scale = float(scale_str)
        except (ValueError, TypeError):
            scale = 1.0

    curves = []
    temp_elems = vdrop_elem.findall('p:Temperature', NS) if vdrop_elem is not None else []
    if not temp_elems:
        temp_elems = vdrop_elem.findall('Temperature') if vdrop_elem is not None else []

    for i, temp_elem in enumerate(temp_elems):
        temp_val = temp_axis[i] if i < len(temp_axis) else 25.0
        if temp_elem.text:
            try:
                voltages = [float(v) * scale for v in temp_elem.text.split()]
                curves.append({
                    't_j': temp_val,
                    'on_state_voltage': voltages,
                    'on_state_current': current_axis,
                })
            except (ValueError, AttributeError):
                logger.warning(f"Failed to parse conduction voltages at temp {temp_val}")

    return {
        'temperatures': temp_axis,
        'curves': curves,
    }


def _extract_thermal_model(package: ET.Element) -> dict[str, Any] | None:
    """
    Extract ThermalModel (Foster or Cauer RC network).

    Returns dict with 'type' and 'elements' (list of {r, c} dicts).
    """
    branch = package.find('.//p:Branch', NS)
    if branch is None:
        branch = package.find('.//Branch')
    if branch is None:
        return None

    branch_type = branch.get('type', 'foster').lower()
    elements = []

    rc_elems = branch.findall('p:RCElement', NS)
    if not rc_elems:
        rc_elems = branch.findall('RCElement')

    for rc_elem in rc_elems:
        try:
            r = float(rc_elem.get('R', '0.0'))
            c = float(rc_elem.get('C', '0.0'))
            elements.append({'r': r, 'c': c})
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse RC element: R={rc_elem.get('R')}, C={rc_elem.get('C')}")

    return {
        'type': branch_type,
        'elements': elements,
    }


def _extract_formulas(package: ET.Element) -> dict[str, str | None]:
    """
    Extract analytical formulas from ConductionLoss and switching loss elements.

    Returns dict with keys 'turn_on', 'turn_off', 'conduction'.
    """
    formulas = {
        'turn_on': None,
        'turn_off': None,
        'conduction': None,
    }

    # Try to extract formula from ConductionLoss
    cond_loss = package.find('.//p:ConductionLoss', NS)
    if cond_loss is None:
        cond_loss = package.find('.//ConductionLoss')
    if cond_loss is not None:
        formula_elem = cond_loss.find('p:Formula', NS)
        if formula_elem is None:
            formula_elem = cond_loss.find('Formula')
        if formula_elem is not None and formula_elem.text:
            formulas['conduction'] = formula_elem.text.strip()

    # Try to extract formulas from TurnOnLoss / TurnOffLoss if present
    for loss_type, key in [('TurnOnLoss', 'turn_on'), ('TurnOffLoss', 'turn_off')]:
        loss_elem = package.find(f'.//p:{loss_type}', NS)
        if loss_elem is None:
            loss_elem = package.find(f'.//{loss_type}')
        if loss_elem is not None:
            formula_elem = loss_elem.find('p:Formula', NS)
            if formula_elem is None:
                formula_elem = loss_elem.find('Formula')
            if formula_elem is not None and formula_elem.text:
                formulas[key] = formula_elem.text.strip()

    return formulas


def parse_plecs_xml(xml_path: Path) -> list[PLECSDeviceData]:
    """
    Parse one PLECS XML file and extract all devices.

    Returns list of PLECSDeviceData objects.
    """
    devices = []

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML {xml_path}: {e}")
        return devices

    # Find all Package elements (try with and without namespace)
    packages = root.findall('.//p:Package', NS)
    if not packages:
        packages = root.findall('.//Package')

    for package in packages:
        vendor = package.get('vendor', '')
        partnumber = package.get('partnumber', '')
        device_class = package.get('class', '')

        if not partnumber:
            logger.warning(f"Package in {xml_path} missing partnumber, skipping")
            continue

        device = PLECSDeviceData(
            vendor=vendor,
            partnumber=partnumber,
            device_class=device_class,
            turn_on=_extract_energy_table(package, 'TurnOnLoss'),
            turn_off=_extract_energy_table(package, 'TurnOffLoss'),
            conduction=_extract_conduction_table(package),
            thermal=_extract_thermal_model(package),
            formulas=_extract_formulas(package),
            source_file=str(xml_path),
        )
        devices.append(device)

    return devices


def plecs_to_switching_pair(device_data: PLECSDeviceData) -> SwitchingPair:
    """
    Convert parsed PLECS device data to a SwitchingPair (self-pair).

    Creates a pair_id from vendor and partnumber.
    """
    pair_id = f"{device_data.vendor.lower()}-{device_data.partnumber.lower()}".replace(' ', '_')
    pair_type = DEVICE_CLASS_TO_PAIR_TYPE.get(device_data.device_class, 'mosfet_self')

    # Create metadata for device
    metadata = DeviceMetadata(
        name=device_data.partnumber,
        type=device_data.device_class,
        manufacturer=device_data.vendor,
    )

    # Create thermal model if available
    thermal_model = None
    if device_data.thermal:
        thermal_model = ThermalModel(
            type=device_data.thermal.get('type', 'foster'),
            elements=[
                ThermalElement(r=elem['r'], c=elem['c'])
                for elem in device_data.thermal.get('elements', [])
            ],
        )

    thermal_properties = DeviceThermalProperties(thermal_model=thermal_model)

    # Create conduction data if available
    conduction = ConductionData()
    if device_data.conduction:
        conduction.temperatures = device_data.conduction.get('temperatures', [])
        conduction.curves = [
            ConductionCurve(
                t_j=curve['t_j'],
                on_state_voltage=curve.get('on_state_voltage', []),
                on_state_current=curve.get('on_state_current', []),
            )
            for curve in device_data.conduction.get('curves', [])
        ]

    # Create switching data (empty or with PLECS energy tables)
    switching_data = SwitchingData(source='plecs_model')

    if device_data.turn_on:
        switching_data.turn_on = SwitchingEnergyData(
            temperatures=device_data.turn_on.get('temperatures', []),
            current_axis=device_data.turn_on.get('current_axis', []),
            voltage_axis=device_data.turn_on.get('voltage_axis', []),
            energy=device_data.turn_on.get('energy', {}),
        )

    if device_data.turn_off:
        switching_data.turn_off = SwitchingEnergyData(
            temperatures=device_data.turn_off.get('temperatures', []),
            current_axis=device_data.turn_off.get('current_axis', []),
            voltage_axis=device_data.turn_off.get('voltage_axis', []),
            energy=device_data.turn_off.get('energy', {}),
        )

    # Create PLECS model with formulas
    plecs_model = PLECSModel(
        turn_on_formula=device_data.formulas.get('turn_on') if device_data.formulas else None,
        turn_off_formula=device_data.formulas.get('turn_off') if device_data.formulas else None,
        conduction_formula=device_data.formulas.get('conduction') if device_data.formulas else None,
    )

    # Create migration metadata
    migration = MigrationMetadata(
        source_files=[f"plecs:{device_data.source_file}"],
        migration_date=datetime.now().isoformat(),
        schema_version="1.0",
    )

    # Create side (same device for both high and low side in a self-pair)
    side = PairSide(
        device_id=pair_id,
        count=1,
        role='switch' if device_data.device_class != 'Diode' else 'diode',
        metadata=metadata,
        thermal_properties=thermal_properties,
        conduction=conduction,
    )

    # Create and return the switching pair
    pair = SwitchingPair(
        pair_id=pair_id,
        pair_type=pair_type,
        topology='half_bridge',
        high_side=side,
        low_side=side,
        switching_data=switching_data,
        plecs_model=plecs_model,
        migration=migration,
    )

    return pair


def merge_plecs_into_existing(existing_pair: SwitchingPair, plecs_device: PLECSDeviceData) -> None:
    """
    Merge PLECS data into an existing switching pair (modifies in place).

    - Adds PLECS formulas if not present
    - Merges conduction data if existing has none
    - Merges thermal model if existing has none
    - Adds PLECS as additional source
    """
    # Add PLECS formulas if available
    if plecs_device.formulas:
        if not existing_pair.plecs_model.turn_on_formula and plecs_device.formulas.get('turn_on'):
            existing_pair.plecs_model.turn_on_formula = plecs_device.formulas['turn_on']
        if not existing_pair.plecs_model.turn_off_formula and plecs_device.formulas.get('turn_off'):
            existing_pair.plecs_model.turn_off_formula = plecs_device.formulas['turn_off']
        if not existing_pair.plecs_model.conduction_formula and plecs_device.formulas.get('conduction'):
            existing_pair.plecs_model.conduction_formula = plecs_device.formulas['conduction']

    # Merge conduction data if existing has none
    if not existing_pair.high_side.conduction.curves and plecs_device.conduction:
        existing_pair.high_side.conduction.temperatures = plecs_device.conduction.get('temperatures', [])
        existing_pair.high_side.conduction.curves = [
            ConductionCurve(
                t_j=curve['t_j'],
                on_state_voltage=curve.get('on_state_voltage', []),
                on_state_current=curve.get('on_state_current', []),
            )
            for curve in plecs_device.conduction.get('curves', [])
        ]
        if existing_pair.low_side.device_id == existing_pair.high_side.device_id:
            # Copy to low side if it's a self-pair
            existing_pair.low_side.conduction = existing_pair.high_side.conduction

    # Merge thermal model if existing has none
    if not existing_pair.high_side.thermal_properties.thermal_model and plecs_device.thermal:
        thermal_model = ThermalModel(
            type=plecs_device.thermal.get('type', 'foster'),
            elements=[
                ThermalElement(r=elem['r'], c=elem['c'])
                for elem in plecs_device.thermal.get('elements', [])
            ],
        )
        existing_pair.high_side.thermal_properties.thermal_model = thermal_model
        if existing_pair.low_side.device_id == existing_pair.high_side.device_id:
            # Copy to low side if it's a self-pair
            existing_pair.low_side.thermal_properties.thermal_model = thermal_model

    # Add PLECS as additional source
    source_file = f"plecs:{plecs_device.source_file}"
    if source_file not in existing_pair.migration.source_files:
        existing_pair.migration.source_files.append(source_file)


def migrate_all_plecs(plecs_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Migrate all PLECS XML files to switching pairs.

    Scans all .xml files recursively under plecs_dir.
    For each XML, extracts all devices.
    If device already exists in output_dir (from .mat or TDB migration), MERGE the PLECS data in.
    If new device, create self-pair entry with PLECS data.

    Returns statistics dict.
    """
    plecs_dir = Path(plecs_dir)
    output_dir = Path(output_dir)

    if not plecs_dir.exists():
        logger.error(f"PLECS directory not found: {plecs_dir}")
        return {
            'status': 'error',
            'message': f"Directory not found: {plecs_dir}",
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_xml_files': 0,
        'total_devices_parsed': 0,
        'new_pairs_created': 0,
        'existing_pairs_merged': 0,
        'merge_errors': 0,
        'parse_errors': 0,
        'final_pair_count': 0,
        'timestamp': datetime.now().isoformat(),
    }

    # Find all XML files
    xml_files = sorted(plecs_dir.rglob('*.xml'))
    stats['total_xml_files'] = len(xml_files)

    logger.info(f"Found {len(xml_files)} PLECS XML files in {plecs_dir}")

    # Process each XML file
    for xml_file in xml_files:
        try:
            devices = parse_plecs_xml(xml_file)
            stats['total_devices_parsed'] += len(devices)

            for device in devices:
                pair_id = f"{device.vendor.lower()}-{device.partnumber.lower()}".replace(' ', '_')
                output_path = output_dir / f"{pair_id}.json"

                try:
                    if output_path.exists():
                        # Load existing pair and merge
                        with open(output_path, 'r') as f:
                            pair_dict = json.load(f)
                        existing_pair = SwitchingPair.from_dict(pair_dict)
                        merge_plecs_into_existing(existing_pair, device)
                        stats['existing_pairs_merged'] += 1

                        # Save merged pair
                        with open(output_path, 'w') as f:
                            json.dump(asdict(existing_pair), f, indent=2)
                    else:
                        # Create new self-pair
                        pair = plecs_to_switching_pair(device)
                        stats['new_pairs_created'] += 1

                        # Save new pair
                        with open(output_path, 'w') as f:
                            json.dump(asdict(pair), f, indent=2)

                except Exception as e:
                    logger.error(f"Failed to process device {device.partnumber}: {e}")
                    stats['merge_errors'] += 1

        except Exception as e:
            logger.error(f"Failed to parse {xml_file}: {e}")
            stats['parse_errors'] += 1

    # Count final pairs
    final_pairs = list(output_dir.glob('*.json'))
    stats['final_pair_count'] = len(final_pairs)

    logger.info(
        f"Migration complete: {stats['new_pairs_created']} new, "
        f"{stats['existing_pairs_merged']} merged, "
        f"{stats['final_pair_count']} total"
    )

    return stats


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Example usage
    plecs_dir = Path('/home/tinix/claude_wsl/archive_ntbees2/sc/data')
    output_dir = Path('/home/tinix/claude_wsl/transistordatabase/switching_pairs')

    result = migrate_all_plecs(plecs_dir, output_dir)
    print(json.dumps(result, indent=2))
