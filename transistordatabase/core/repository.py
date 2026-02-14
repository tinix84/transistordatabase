"""Repository implementations and factory patterns for transistor data management."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .models import (
    Transistor, TransistorMetadata, ElectricalRatings, ThermalProperties,
    Switch, Diode, ChannelCharacteristics, SwitchingLossData
)
from .services import TransistorRepository, ITransistorLoader


# ---------------------------------------------------------------------------
# Helpers for JSON → numpy conversion
# ---------------------------------------------------------------------------

def _load_data_file(filename: str) -> list[str]:
    """Load lines from a data file in the transistordatabase/data/ directory."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / filename
    items: list[str] = []
    with open(file_path) as f:
        for line in f.read().splitlines():
            if line.startswith("#") or line.isspace() or not line:
                continue
            items.append(str(line))
    return items


def _convert_energy_arrays(items: list[dict]) -> None:
    """Convert graph arrays to numpy for a list of switching energy dicts."""
    for item in items:
        ds_type = item.get('dataset_type', '')
        if ds_type == 'graph_r_e' and 'graph_r_e' in item:
            item['graph_r_e'] = np.array(item['graph_r_e'])
        elif ds_type == 'graph_i_e' and 'graph_i_e' in item:
            item['graph_i_e'] = np.array(item['graph_i_e'])
        elif ds_type == 'graph_t_e' and 'graph_t_e' in item:
            item['graph_t_e'] = np.array(item['graph_t_e'])


def _convert_arrays_to_numpy(transistor_dict: dict) -> None:
    """Convert JSON list arrays to numpy arrays in-place.

    Replicates the conversion logic from
    ``DatabaseManager.convert_dict_to_transistor_object()``.
    """
    # Capacitances
    for cap_key in ('c_oss', 'c_iss', 'c_rss'):
        if cap_key in transistor_dict and transistor_dict[cap_key] is not None:
            for item in transistor_dict[cap_key]:
                item['graph_v_c'] = np.array(item['graph_v_c'])
    if 'graph_v_ecoss' in transistor_dict and transistor_dict['graph_v_ecoss'] is not None:
        transistor_dict['graph_v_ecoss'] = np.array(transistor_dict['graph_v_ecoss'])

    # Raw measurement data
    if 'raw_measurement_data' in transistor_dict:
        for item in transistor_dict['raw_measurement_data']:
            for key in ('dpt_on_vds', 'dpt_on_id', 'dpt_off_vds', 'dpt_off_id'):
                if key in item:
                    for u in range(len(item[key])):
                        item[key][u] = np.array(item[key][u])

    # Switch
    switch_args = transistor_dict.get('switch')
    if switch_args:
        if switch_args.get('thermal_foster', {}).get('graph_t_rthjc') is not None:
            switch_args['thermal_foster']['graph_t_rthjc'] = np.array(
                switch_args['thermal_foster']['graph_t_rthjc']
            )
        for ch in switch_args.get('channel', []):
            ch['graph_v_i'] = np.array(ch['graph_v_i'])
        for energy_key in ('e_on', 'e_off', 'e_on_meas', 'e_off_meas'):
            if energy_key in switch_args:
                _convert_energy_arrays(switch_args[energy_key])
        for gc in switch_args.get('charge_curve', []):
            gc['graph_q_v'] = np.array(gc['graph_q_v'])
        for tr in switch_args.get('r_channel_th', []):
            tr['graph_t_r'] = np.array(tr['graph_t_r'])
        for soa in switch_args.get('soa', []):
            soa['graph_i_v'] = np.array(soa['graph_i_v'])

    # Diode
    diode_args = transistor_dict.get('diode')
    if diode_args:
        if diode_args.get('thermal_foster', {}).get('graph_t_rthjc') is not None:
            diode_args['thermal_foster']['graph_t_rthjc'] = np.array(
                diode_args['thermal_foster']['graph_t_rthjc']
            )
        for ch in diode_args.get('channel', []):
            ch['graph_v_i'] = np.array(ch['graph_v_i'])
        if 'e_rr' in diode_args:
            _convert_energy_arrays(diode_args['e_rr'])
        for soa in diode_args.get('soa', []):
            soa['graph_i_v'] = np.array(soa['graph_i_v'])


def _json_dict_to_legacy_transistor(transistor_dict: dict):
    """Convert a raw JSON dict to a legacy Transistor object.

    Performs numpy array conversion, loads housing/manufacturer data,
    and constructs the legacy Transistor.
    """
    from transistordatabase.transistor import Transistor as LegacyTransistor

    _convert_arrays_to_numpy(transistor_dict)
    switch_args = transistor_dict.get('switch', {})
    diode_args = transistor_dict.get('diode', {})
    housing_types = _load_data_file("housing_types.txt")
    manufacturers = _load_data_file("module_manufacturers.txt")
    return LegacyTransistor(
        transistor_dict, switch_args, diode_args,
        housing_types, manufacturers,
    )


class JsonTransistorRepository(TransistorRepository):
    """File-based repository using JSON storage."""
    
    def __init__(self, data_directory: Path):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
    
    def get_by_name(self, name: str) -> Optional[Transistor]:
        """Get transistor by name from JSON file."""
        file_path = self.data_directory / f"{name}.json"
        if not file_path.exists():
            return None
        
        loader = JsonTransistorLoader()
        return loader.load_from_json(file_path)
    
    def save(self, transistor: Transistor) -> None:
        """Save transistor to JSON file."""
        file_path = self.data_directory / f"{transistor.metadata.name}.json"
        loader = JsonTransistorLoader()
        loader.save_to_json(transistor, file_path)
    
    def list_all(self) -> List[str]:
        """List all available transistor names."""
        json_files = self.data_directory.glob("*.json")
        return [f.stem for f in json_files]
    
    def delete(self, name: str) -> bool:
        """Delete transistor from repository."""
        file_path = self.data_directory / f"{name}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


class JsonTransistorLoader(ITransistorLoader):
    """JSON-based transistor loader.

    Uses the adapter bridge to load JSON files through the legacy Transistor
    constructor (which handles full numpy array conversion and validation),
    then converts the result to a core Transistor via ``legacy_to_core()``.
    """

    def load_from_json(self, file_path: Path) -> Transistor:
        """Load transistor from JSON file via adapter bridge.

        Pipeline: JSON dict -> numpy conversion -> legacy Transistor
        -> ``legacy_to_core()`` -> core Transistor.
        """
        from .adapters import legacy_to_core

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        legacy = _json_dict_to_legacy_transistor(data)
        return legacy_to_core(legacy)

    def save_to_json(self, transistor: Transistor, file_path: Path) -> None:
        """Save core transistor to JSON file via adapter bridge.

        Pipeline: core Transistor -> ``core_to_legacy_dicts()`` -> legacy
        Transistor -> ``convert_to_dict()`` -> JSON.
        """
        from .adapters import core_to_legacy_dicts
        from transistordatabase.transistor import Transistor as LegacyTransistor

        t_args, sw_args, di_args = core_to_legacy_dicts(transistor)
        housing_types = _load_data_file("housing_types.txt")
        manufacturers = _load_data_file("module_manufacturers.txt")
        legacy = LegacyTransistor(
            t_args, sw_args, di_args, housing_types, manufacturers,
        )
        legacy_dict = legacy.convert_to_dict()
        if "_id" in legacy_dict:
            del legacy_dict["_id"]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(legacy_dict, f, indent=2)

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Serialize special types to JSON-compatible format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class TransistorFactory:
    """Factory for creating transistor instances."""
    
    @staticmethod
    def create_empty_transistor(name: str, transistor_type: str = "IGBT") -> Transistor:
        """Create empty transistor with default values."""
        metadata = TransistorMetadata(
            name=name,
            type=transistor_type,
            author="",
            manufacturer="",
            housing_type=""
        )
        
        electrical = ElectricalRatings(
            v_abs_max=0.0,
            i_abs_max=0.0,
            i_cont=0.0,
            t_j_max=150.0
        )
        
        thermal = ThermalProperties(
            housing_area=0.0,
            cooling_area=0.0
        )
        
        return Transistor(metadata, electrical, thermal)
    
    @staticmethod
    def create_from_template(_template_name: str, new_name: str) -> Transistor:
        """Create transistor from existing template.

        :param _template_name: Template name (reserved for future implementation)
        """
        # Implementation would load template and create new instance
        return TransistorFactory.create_empty_transistor(new_name)
