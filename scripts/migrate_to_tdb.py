#!/usr/bin/env python3
"""
Transistor Database Migration Script

Migrates transistors from multiple sources (File Exchange, sc archive) into TDB format.
Handles format conversion, duplicate detection, validation, and quality scoring.
"""

import json
import logging
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import traceback

# Setup logging
log_handlers = [
    logging.FileHandler('migration.log'),
    logging.StreamHandler(sys.stdout)
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)


@dataclass
class MigrationResult:
    """Result of migrating one transistor."""
    source_file: str
    source_format: str
    success: bool
    transistor_id: Optional[str]
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0


class TransistorMigrator:
    """Main migration orchestrator."""

    def __init__(self, dry_run: bool = False):
        self.logger = logging.getLogger(__name__)
        self.dry_run = dry_run
        self.results: List[MigrationResult] = []
        self.duplicates: Dict[str, List[str]] = defaultdict(list)
        self.conflicts: List[Dict] = []
        self.existing_ids: set = set()
        self.existing_data: Dict[str, Dict] = {}
        self.file_count = 0

        # Load field mapping
        try:
            mapping_path = Path('/home/tinix/claude_wsl/transistordatabase/FIELD_MAPPING.json')
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    self.field_mapping = json.load(f)
            else:
                self.field_mapping = {}
                self.logger.warning("FIELD_MAPPING.json not found, using direct mapping")
        except Exception as e:
            self.logger.warning(f"Error loading field mapping: {e}")
            self.field_mapping = {}

    def migrate_all(self, sources: List[Path], output_dir: Path):
        """Migrate all transistors from all sources."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scan existing TDB for duplicate detection
        self.scan_existing_tdb(output_dir)

        for source_path in sources:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing source: {source_path}")
            self.logger.info(f"{'='*60}")

            if not source_path.exists():
                self.logger.error(f"Source path does not exist: {source_path}")
                continue

            self.migrate_directory(source_path, output_dir)

        self.generate_report()

    def scan_existing_tdb(self, output_dir: Path):
        """Scan existing TDB to build duplicate detection set."""
        if output_dir.exists():
            for json_file in output_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    transistor_id = data.get('metadata', {}).get('name', json_file.stem)
                    self.existing_ids.add(transistor_id)
                    self.existing_data[transistor_id] = data
                    self.duplicates[transistor_id].append(str(json_file))
                except Exception as e:
                    self.logger.warning(f"Error scanning {json_file}: {e}")

        self.logger.info(f"Found {len(self.existing_ids)} existing transistors")

    def migrate_directory(self, source_dir: Path, output_dir: Path):
        """Migrate all files in a directory."""
        for file_path in source_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.json', '.xml', '.csv']:
                self.file_count += 1
                result = self.migrate_file(file_path, output_dir)
                self.results.append(result)

                # Progress update every 50 files
                if self.file_count % 50 == 0:
                    self.logger.info(f"Processed {self.file_count} files...")

        self.logger.info(f"Completed directory: {self.file_count} files processed")

    def migrate_file(self, file_path: Path, output_dir: Path) -> MigrationResult:
        """Migrate a single file."""
        try:
            self.logger.debug(f"Processing: {file_path}")

            # Load and convert based on format
            if file_path.suffix == '.json':
                converted = self.convert_json_file(file_path)
            elif file_path.suffix == '.xml':
                converted = self.convert_xml_file(file_path)
            elif file_path.suffix == '.csv':
                converted = self.convert_csv_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            if not converted:
                raise ValueError("Conversion returned None")

            # Extract transistor ID
            transistor_id = converted.get('metadata', {}).get('name', '')
            if not transistor_id:
                raise ValueError("No transistor ID found in converted data")

            # Validate
            validation_result = self.validate_tdb_data(converted)

            # Check for duplicates
            if transistor_id in self.existing_ids:
                conflict = self.handle_duplicate(transistor_id, converted, file_path, output_dir)
                quality_score = validation_result['score']
            else:
                # Save to output
                if not self.dry_run:
                    output_path = output_dir / f"{transistor_id}.json"
                    with open(output_path, 'w') as f:
                        json.dump(converted, f, indent=2)
                    self.logger.debug(f"Saved: {output_path}")

                self.existing_ids.add(transistor_id)
                self.existing_data[transistor_id] = converted
                self.duplicates[transistor_id].append(str(file_path))
                quality_score = validation_result['score']

            return MigrationResult(
                source_file=str(file_path),
                source_format=file_path.suffix,
                success=True,
                transistor_id=transistor_id,
                warnings=validation_result['warnings'],
                data_quality_score=quality_score
            )

        except Exception as e:
            self.logger.error(f"Failed to migrate {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return MigrationResult(
                source_file=str(file_path),
                source_format=file_path.suffix if hasattr(file_path, 'suffix') else 'unknown',
                success=False,
                transistor_id=None,
                error=str(e)
            )

    def convert_json_file(self, file_path: Path) -> Optional[Dict]:
        """Convert JSON file to TDB format."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check if already in TDB format
        if self.is_tdb_format(data):
            # Add migration metadata
            if 'metadata' not in data:
                data['metadata'] = {}
            data['metadata']['import_source'] = f'tdb_native:{file_path.name}'
            data['metadata']['import_date'] = datetime.now().isoformat()
            return data

        # Check if File Exchange format (flat structure with name/type/manufacturer at root)
        if self._is_file_exchange_format(data):
            converted = self.apply_file_exchange_mapping(data)
        else:
            # Unknown JSON format
            self.logger.debug(f"Unknown JSON format in {file_path}, treating as legacy File Exchange")
            converted = self.apply_file_exchange_mapping(data)

        # Add metadata
        converted.setdefault('metadata', {})
        converted['metadata']['import_source'] = f'file_exchange:{file_path.name}'
        converted['metadata']['import_date'] = datetime.now().isoformat()

        return converted

    def _is_file_exchange_format(self, data: Dict) -> bool:
        """Check if data is File Exchange format (flat structure)."""
        # File Exchange has name, type, manufacturer at root level
        return 'name' in data and 'type' in data and 'switch' in data

    def convert_xml_file(self, file_path: Path) -> Optional[Dict]:
        """Convert PLECS XML file to TDB format."""
        try:
            # Parse XML structure
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract namespace
            ns = {'sem': 'http://www.plexim.com/xml/semiconductors/'}

            # Get Package info
            package = root.find('.//sem:Package', ns)
            if package is None:
                package = root.find('Package')

            if package is None:
                self.logger.warning(f"No Package element found in {file_path}")
                return None

            class_type = package.get('class', 'unknown')
            vendor = package.get('vendor', 'Unknown')
            partnumber = package.get('partnumber', file_path.stem)

            # Create transistor ID
            transistor_id = f"{vendor}_{partnumber}".replace(' ', '_')

            # Initialize TDB structure
            converted = {
                'metadata': {
                    'name': transistor_id,
                    'type': class_type,
                    'manufacturer': vendor,
                    'housing_type': 'unknown',
                    'import_source': f'plecs_xml:{file_path.name}',
                    'import_date': datetime.now().isoformat()
                },
                'electrical_ratings': {
                    'v_abs_max': 0.0,
                    'i_abs_max': 0.0,
                    'i_cont': 0.0,
                    't_j_max': 150.0
                },
                'thermal_properties': {
                    'housing_area': 0.0,
                    'cooling_area': 0.0,
                    'r_th_jc': 0.0,
                    'r_th_cs': 0.0
                },
                'switch': {
                    'channel_data': [],
                    'e_on_data': [],
                    'e_off_data': [],
                    'gate_charge_curves': [],
                    'soa': [],
                    'thermal_foster': None
                },
                'diode': {
                    'channel_data': [],
                    'e_rr_data': [],
                    'thermal_foster': None
                },
                'c_oss': [],
                'c_iss': [],
                'c_rss': []
            }

            # Extract semiconductor data
            sem_data = package.find('.//sem:SemiconductorData', ns)
            if sem_data is None:
                sem_data = package.find('SemiconductorData')

            if sem_data is not None:
                # Extract switching losses
                turn_on = sem_data.find('.//sem:TurnOnLoss', ns) or sem_data.find('TurnOnLoss')
                if turn_on is not None:
                    e_on_data = self._parse_switching_loss(turn_on, 'E_on')
                    if e_on_data:
                        converted['switch']['e_on_data'].append(e_on_data)

                turn_off = sem_data.find('.//sem:TurnOffLoss', ns) or sem_data.find('TurnOffLoss')
                if turn_off is not None:
                    e_off_data = self._parse_switching_loss(turn_off, 'E_off')
                    if e_off_data:
                        converted['switch']['e_off_data'].append(e_off_data)

                # Extract conduction loss (Rds or Vf)
                conduction = sem_data.find('.//sem:ConductionLoss', ns) or sem_data.find('ConductionLoss')
                if conduction is not None:
                    channel_data = self._parse_conduction_loss(conduction, class_type)
                    if channel_data:
                        converted['switch']['channel_data'].append(channel_data)

            # Extract thermal model
            thermal = package.find('.//sem:ThermalModel', ns) or package.find('ThermalModel')
            if thermal is not None:
                foster = thermal.find('.//sem:Branch[@type="Foster"]', ns)
                if foster is None:
                    foster = thermal.find('.//Branch[@type="Foster"]')

                if foster is not None:
                    thermal_model = self._parse_foster_model(foster)
                    if thermal_model:
                        converted['switch']['thermal_foster'] = thermal_model
                        converted['diode']['thermal_foster'] = thermal_model

            return converted

        except Exception as e:
            self.logger.error(f"Error converting PLECS XML {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _parse_switching_loss(self, loss_elem: ET.Element, loss_type: str) -> Optional[Dict]:
        """Parse switching loss data from PLECS XML."""
        try:
            current_axis_text = loss_elem.findtext('.//CurrentAxis', '') or loss_elem.findtext('CurrentAxis', '')
            voltage_axis_text = loss_elem.findtext('.//VoltageAxis', '') or loss_elem.findtext('VoltageAxis', '')
            temp_axis_text = loss_elem.findtext('.//TemperatureAxis', '') or loss_elem.findtext('TemperatureAxis', '')

            if not current_axis_text or not voltage_axis_text:
                return None

            i_values = [float(x.strip()) for x in current_axis_text.split() if x.strip()]
            v_values = [float(x.strip()) for x in voltage_axis_text.split() if x.strip()]
            t_values = [float(x.strip()) for x in temp_axis_text.split() if x.strip()] if temp_axis_text else [25, 150]

            return {
                'loss_type': loss_type,
                'current_axis': i_values,
                'voltage_axis': v_values,
                'temperature_axis': t_values,
                'energy_data': self._extract_energy_data(loss_elem, len(t_values), len(v_values), len(i_values))
            }

        except Exception as e:
            self.logger.debug(f"Error parsing switching loss: {e}")
            return None

    def _extract_energy_data(self, elem: ET.Element, n_temps: int, n_volts: int, n_currents: int) -> List[List[List[float]]]:
        """Extract energy data from switching loss element."""
        try:
            energy_elem = elem.find('.//Energy', {'': 'http://www.plexim.com/xml/semiconductors/'})
            if energy_elem is None:
                energy_elem = elem.find('Energy')

            if energy_elem is None:
                return []

            data = []
            for temp_elem in energy_elem.findall('.//Temperature'):
                if temp_elem is None:
                    temp_elem = energy_elem.findall('Temperature')

                if isinstance(temp_elem, list):
                    for te in temp_elem:
                        temp_data = []
                        for volt_elem in te.findall('.//Voltage'):
                            if volt_elem is None:
                                volt_elem = te.findall('Voltage')

                            if isinstance(volt_elem, list):
                                for ve in volt_elem:
                                    values_text = ve.text or ''
                                    values = [float(x.strip()) for x in values_text.split() if x.strip()]
                                    temp_data.append(values)
                            else:
                                values_text = volt_elem.text or ''
                                values = [float(x.strip()) for x in values_text.split() if x.strip()]
                                temp_data.append(values)
                        data.append(temp_data)
                else:
                    temp_data = []
                    for volt_elem in temp_elem.findall('.//Voltage'):
                        if volt_elem is None:
                            volt_elem = temp_elem.findall('Voltage')

                        if isinstance(volt_elem, list):
                            for ve in volt_elem:
                                values_text = ve.text or ''
                                values = [float(x.strip()) for x in values_text.split() if x.strip()]
                                temp_data.append(values)
                        else:
                            values_text = volt_elem.text or ''
                            values = [float(x.strip()) for x in values_text.split() if x.strip()]
                            temp_data.append(values)
                    data.append(temp_data)

            return data
        except Exception as e:
            self.logger.debug(f"Error extracting energy data: {e}")
            return []

    def _parse_conduction_loss(self, conduction_elem: ET.Element, device_type: str) -> Optional[Dict]:
        """Parse conduction loss data from PLECS XML."""
        try:
            current_axis_text = conduction_elem.findtext('.//CurrentAxis', '') or conduction_elem.findtext('CurrentAxis', '')
            temp_axis_text = conduction_elem.findtext('.//TemperatureAxis', '') or conduction_elem.findtext('TemperatureAxis', '')

            if not current_axis_text:
                return None

            i_values = [float(x.strip()) for x in current_axis_text.split() if x.strip()]
            t_values = [float(x.strip()) for x in temp_axis_text.split() if x.strip()] if temp_axis_text else [25, 150]

            # Extract voltage drop or resistance
            volt_drop_elem = conduction_elem.find('.//VoltageDrop', {'': 'http://www.plexim.com/xml/semiconductors/'})
            if volt_drop_elem is None:
                volt_drop_elem = conduction_elem.find('VoltageDrop')

            volt_data = []
            if volt_drop_elem is not None:
                for temp_elem in volt_drop_elem.findall('.//Temperature'):
                    if temp_elem is None:
                        temp_elem = volt_drop_elem.findall('Temperature')

                    if isinstance(temp_elem, list):
                        for te in temp_elem:
                            values_text = te.text or ''
                            values = [float(x.strip()) for x in values_text.split() if x.strip()]
                            volt_data.append(values)
                    else:
                        values_text = temp_elem.text or ''
                        values = [float(x.strip()) for x in values_text.split() if x.strip()]
                        volt_data.append(values)

            return {
                'device_type': device_type,
                'current_axis': i_values,
                'temperature_axis': t_values,
                'voltage_drop_data': volt_data
            }

        except Exception as e:
            self.logger.debug(f"Error parsing conduction loss: {e}")
            return None

    def _parse_foster_model(self, foster_elem: ET.Element) -> Optional[Dict]:
        """Parse Foster thermal model from PLECS XML."""
        try:
            rtau_elements = foster_elem.findall('.//RTauElement', {'': 'http://www.plexim.com/xml/semiconductors/'})
            if not rtau_elements:
                rtau_elements = foster_elem.findall('.//RTauElement')

            if not rtau_elements:
                rtau_elements = foster_elem.findall('RTauElement')

            if not rtau_elements:
                return None

            branches = []
            for elem in rtau_elements:
                r = float(elem.get('R', 0))
                tau = float(elem.get('Tau', 0))
                branches.append({'R': r, 'tau': tau})

            return {'type': 'Foster', 'branches': branches}

        except Exception as e:
            self.logger.debug(f"Error parsing Foster model: {e}")
            return None

    def convert_csv_file(self, file_path: Path) -> Optional[Dict]:
        """Convert CSV file to TDB format."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Try to detect dialect
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except:
                    dialect = 'excel'

                f.seek(0)
                reader = csv.DictReader(f, dialect=dialect)
                rows = list(reader)

            if not rows:
                self.logger.warning(f"CSV file is empty: {file_path}")
                return None

            # Take first row (assuming one transistor per CSV or use filename)
            row = rows[0]

            # Create minimal TDB structure
            transistor_id = row.get('name') or row.get('part_number') or row.get('Part Number') or file_path.stem
            transistor_id = transistor_id.replace(' ', '_').strip()

            converted = {
                'metadata': {
                    'name': transistor_id,
                    'type': row.get('type', row.get('Type', 'MOSFET')),
                    'manufacturer': row.get('manufacturer', row.get('Manufacturer', 'Unknown')),
                    'housing_type': row.get('housing', row.get('Housing', 'unknown')),
                    'import_source': f'csv:{file_path.name}',
                    'import_date': datetime.now().isoformat()
                },
                'electrical_ratings': {
                    'v_abs_max': float(row.get('v_ds', row.get('V_DS', row.get('voltage', 0)))) or 0.0,
                    'i_abs_max': float(row.get('i_d', row.get('I_D', row.get('current', 0)))) or 0.0,
                    'i_cont': float(row.get('i_d_cont', row.get('I_D_cont', row.get('i_d', 0)))) or 0.0,
                    't_j_max': float(row.get('t_j_max', row.get('T_J_max', 175))) or 175.0
                },
                'thermal_properties': {
                    'housing_area': float(row.get('housing_area', 0)) or 0.0,
                    'cooling_area': float(row.get('cooling_area', 0)) or 0.0,
                    'r_th_jc': float(row.get('r_th_jc', row.get('R_th_JC', 0))) or 0.0,
                    'r_th_cs': float(row.get('r_th_cs', row.get('R_th_CS', 0))) or 0.0
                },
                'switch': {
                    'channel_data': [],
                    'e_on_data': [],
                    'e_off_data': [],
                    'gate_charge_curves': [],
                    'soa': []
                },
                'diode': {
                    'channel_data': [],
                    'e_rr_data': []
                },
                'c_oss': [],
                'c_iss': [],
                'c_rss': []
            }

            return converted

        except Exception as e:
            self.logger.error(f"Error converting CSV {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def is_tdb_format(self, data: Dict) -> bool:
        """Check if data is already in TDB format."""
        required_keys = ['metadata', 'electrical_ratings', 'switch', 'diode']
        return all(key in data for key in required_keys)

    def apply_file_exchange_mapping(self, data: Dict) -> Dict:
        """Apply File Exchange to TDB field mapping (95% compatible per Task 2)."""
        # File Exchange format is nearly identical to TDB, just ensure structure

        # Extract metadata fields from root level if present
        metadata = data.get('metadata', {})
        if not metadata:
            # Create metadata from root-level File Exchange fields
            metadata = {
                'name': data.get('name', 'Unknown'),
                'type': data.get('type', 'MOSFET'),
                'author': data.get('author', ''),
                'manufacturer': data.get('manufacturer', 'Unknown'),
                'housing_type': data.get('housing_type', ''),
                'technology': data.get('technology'),
                'datasheet_hyperlink': data.get('datasheet_hyperlink'),
                'datasheet_date': data.get('datasheet_date'),
                'datasheet_version': data.get('datasheet_version'),
            }

        # Extract electrical ratings from root or existing section
        electrical_ratings = data.get('electrical_ratings', {})
        if not electrical_ratings or len(electrical_ratings) == 0:
            electrical_ratings = {
                'v_abs_max': data.get('v_abs_max', 0.0),
                'i_abs_max': data.get('i_abs_max', 0.0),
                'i_cont': data.get('i_cont', 0.0),
                't_j_max': data.get('t_j_max', 150.0),
            }

        # Extract thermal properties from root or existing section
        thermal_properties = data.get('thermal_properties', {})
        if not thermal_properties or len(thermal_properties) == 0:
            thermal_properties = {
                'housing_area': data.get('housing_area', 0.0),
                'cooling_area': data.get('cooling_area', 0.0),
                'r_th_jc': data.get('r_th_jc', 0.0),
                'r_th_cs': data.get('r_th_cs', 0.0),
                'r_th_switch_cs': data.get('r_th_switch_cs', 0.0),
                'r_th_diode_cs': data.get('r_th_diode_cs', 0.0),
                't_c_max': data.get('t_c_max'),
            }

        # Build complete TDB structure
        converted = {
            'metadata': metadata,
            'electrical_ratings': electrical_ratings,
            'thermal_properties': thermal_properties,
            'switch': data.get('switch', {}),
            'diode': data.get('diode', {}),
            'c_oss': data.get('c_oss', []),
            'c_iss': data.get('c_iss', []),
            'c_rss': data.get('c_rss', []),
        }

        # Add optional capacitance fields if present
        if data.get('c_oss_er'):
            converted['c_oss_er'] = data.get('c_oss_er')
        if data.get('c_oss_tr'):
            converted['c_oss_tr'] = data.get('c_oss_tr')

        # Ensure all required sub-structures exist
        converted['switch'].setdefault('channel_data', [])
        converted['switch'].setdefault('e_on_data', [])
        converted['switch'].setdefault('e_off_data', [])
        converted['switch'].setdefault('gate_charge_curves', [])
        converted['switch'].setdefault('soa', [])

        converted['diode'].setdefault('channel_data', [])
        converted['diode'].setdefault('e_rr_data', [])

        return converted

    def handle_duplicate(self, transistor_id: str, new_data: Dict, source_file: Path, output_dir: Path) -> Dict:
        """Handle duplicate transistor - compare and flag for review."""
        existing_files = [f for f in self.duplicates[transistor_id]]

        if not existing_files:
            return {}

        # Load first existing file for comparison
        existing_data = self.existing_data.get(transistor_id, {})

        # Compare completeness
        new_score = self.calculate_completeness(new_data)
        existing_score = self.calculate_completeness(existing_data)

        conflict = {
            'transistor_id': transistor_id,
            'existing_file': existing_files[0] if existing_files else 'unknown',
            'existing_score': round(existing_score, 2),
            'new_file': str(source_file),
            'new_score': round(new_score, 2),
            'recommendation': 'keep_new' if new_score > existing_score + 5 else 'keep_existing',
            'score_difference': round(new_score - existing_score, 2)
        }

        self.conflicts.append(conflict)

        self.logger.warning(
            f"Duplicate found: {transistor_id} "
            f"(existing: {existing_score:.1f}%, new: {new_score:.1f}%)"
        )

        return conflict

    def calculate_completeness(self, data: Dict) -> float:
        """Calculate data completeness score (0-100)."""
        points = 0
        max_points = 0

        # Required fields (10 points each)
        required_fields = [
            ('metadata', 'name'),
            ('metadata', 'type'),
            ('metadata', 'manufacturer'),
            ('electrical_ratings', 'v_abs_max'),
            ('electrical_ratings', 'i_abs_max'),
        ]

        for section, field in required_fields:
            max_points += 10
            value = data.get(section, {}).get(field)
            if value and (isinstance(value, (int, float)) or (isinstance(value, str) and value.strip())):
                points += 10

        # Thermal properties (5 points each)
        thermal_fields = [
            ('thermal_properties', 'r_th_jc'),
            ('thermal_properties', 'r_th_cs'),
            ('thermal_properties', 'housing_area'),
        ]

        for section, field in thermal_fields:
            max_points += 5
            value = data.get(section, {}).get(field)
            if value and value != 0:
                points += 5

        # Curve data (15 points for each type present)
        curve_types = [
            ('switch', 'channel_data'),
            ('switch', 'e_on_data'),
            ('switch', 'e_off_data'),
            ('switch', 'gate_charge_curves'),
            ('diode', 'channel_data'),
        ]

        for section, field in curve_types:
            max_points += 15
            curves = data.get(section, {}).get(field, [])
            if curves and len(curves) > 0:
                points += 15

        # Capacitance data (5 points each)
        cap_types = ['c_oss', 'c_iss', 'c_rss']
        for cap_type in cap_types:
            max_points += 5
            caps = data.get(cap_type, [])
            if caps and len(caps) > 0:
                points += 5

        return (points / max_points * 100) if max_points > 0 else 0

    def validate_tdb_data(self, data: Dict) -> Dict:
        """Validate TDB data and generate warnings."""
        warnings = []

        # Check required fields
        if not data.get('metadata', {}).get('name'):
            warnings.append("Missing transistor name")

        if not data.get('metadata', {}).get('type'):
            warnings.append("Missing transistor type")

        # Check electrical ratings
        v_abs_max = data.get('electrical_ratings', {}).get('v_abs_max', 0)
        if v_abs_max and (v_abs_max < 5 or v_abs_max > 20000):
            warnings.append(f"Suspicious v_abs_max: {v_abs_max}V (expected 5-20000V)")

        i_abs_max = data.get('electrical_ratings', {}).get('i_abs_max', 0)
        if i_abs_max and (i_abs_max < 0.01 or i_abs_max > 10000):
            warnings.append(f"Suspicious i_abs_max: {i_abs_max}A (expected 0.01-10000A)")

        # Calculate score
        score = self.calculate_completeness(data)

        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'score': score
        }

    def generate_report(self):
        """Generate comprehensive migration report."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        avg_score = sum(r.data_quality_score for r in successful) / len(successful) if successful else 0
        median_score = sorted([r.data_quality_score for r in successful])[len(successful)//2] if successful else 0

        report = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'summary': {
                'total_processed': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'duplicates_found': len(self.conflicts),
                'unique_transistors': len(self.existing_ids),
                'average_quality_score': round(avg_score, 2),
                'median_quality_score': round(median_score, 2)
            },
            'by_format': self.get_format_statistics(),
            'conflicts': self.conflicts,
            'failed_files': [
                {
                    'file': r.source_file,
                    'format': r.source_format,
                    'error': r.error
                }
                for r in failed
            ],
            'quality_distribution': self.get_quality_distribution(successful),
            'sample_results': [
                {
                    'transistor_id': r.transistor_id,
                    'source_file': r.source_file,
                    'quality_score': round(r.data_quality_score, 2),
                    'warnings': r.warnings
                }
                for r in successful[:10]
            ]
        }

        # Save detailed report
        report_path = Path('/home/tinix/claude_wsl/transistordatabase/MIGRATION_REPORT.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save conflicts for review
        if self.conflicts:
            conflicts_path = Path('/home/tinix/claude_wsl/transistordatabase/CONFLICTS_FOR_REVIEW.json')
            with open(conflicts_path, 'w') as f:
                json.dump(self.conflicts, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"MIGRATION {'DRY RUN ' if self.dry_run else ''}COMPLETE")
        print(f"{'='*60}")
        print(f"Total files processed: {report['summary']['total_processed']}")
        print(f"Successful: {report['summary']['successful']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Unique transistors: {report['summary']['unique_transistors']}")
        print(f"Duplicates found: {report['summary']['duplicates_found']}")
        print(f"Average quality score: {report['summary']['average_quality_score']:.1f}%")
        print(f"Median quality score: {report['summary']['median_quality_score']:.1f}%")
        print(f"\nDetailed report saved to: {report_path}")
        if self.conflicts:
            print(f"Conflicts saved to: {conflicts_path}")
        print(f"{'='*60}\n")

    def get_format_statistics(self) -> Dict:
        """Get statistics by file format."""
        stats = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})

        for result in self.results:
            fmt = result.source_format
            stats[fmt]['total'] += 1
            if result.success:
                stats[fmt]['success'] += 1
            else:
                stats[fmt]['failed'] += 1

        return dict(stats)

    def get_quality_distribution(self, successful_results: List[MigrationResult]) -> Dict:
        """Get distribution of quality scores."""
        distribution = {
            'excellent (90-100%)': 0,
            'good (70-89%)': 0,
            'fair (50-69%)': 0,
            'poor (<50%)': 0
        }

        for result in successful_results:
            score = result.data_quality_score
            if score >= 90:
                distribution['excellent (90-100%)'] += 1
            elif score >= 70:
                distribution['good (70-89%)'] += 1
            elif score >= 50:
                distribution['fair (50-69%)'] += 1
            else:
                distribution['poor (<50%)'] += 1

        return distribution


# CLI Interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Migrate transistors to TDB format')
    parser.add_argument('--dry-run', action='store_true', help='Run without writing files')
    parser.add_argument('--output', default='transistors_merged', help='Output directory')
    args = parser.parse_args()

    # Setup paths
    base_path = Path('/home/tinix/claude_wsl')
    sources = [
        base_path / 'transistordatabase' / 'database',  # TDB native files
        base_path / 'file_exchange',  # File Exchange JSON
        base_path / 'archive_ntbees2' / 'sc',  # PLECS XML and CSV
    ]

    output_dir = Path('/home/tinix/claude_wsl/transistordatabase') / args.output

    # Run migration
    migrator = TransistorMigrator(dry_run=args.dry_run)
    migrator.migrate_all(sources, output_dir)


if __name__ == "__main__":
    main()
