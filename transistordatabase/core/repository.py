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


class RepositoryPathError(ValueError):
    """A preserved device identity cannot safely name a repository storage file."""


class JsonTransistorRepository(TransistorRepository):
    """File-based repository using JSON storage."""
    
    def __init__(self, data_directory: Path):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)

    def _storage_path(self, name: str) -> Path:
        """Reject unsafe names and resolved links without rewriting device identity."""
        if (not isinstance(name, str) or not name.strip() or name in (".", "..")
                or any(character in name for character in ("/", "\\", ":"))
                or any(ord(character) < 32 for character in name)):
            raise RepositoryPathError("Device identity must be a single relative storage name without path separators or traversal")
        try:
            root = self.data_directory.resolve()
            candidate = (self.data_directory / f"{name}.json").resolve()
        except (OSError, RuntimeError) as exc:
            raise RepositoryPathError("Device storage path cannot be resolved safely") from exc
        if candidate.parent != root:
            raise RepositoryPathError("Resolved device storage path must stay within the exact repository directory")
        return candidate
    
    def get_by_name(self, name: str) -> Optional[Transistor]:
        """Get transistor by name from JSON file."""
        file_path = self._storage_path(name)
        if not file_path.exists():
            return None
        
        loader = JsonTransistorLoader()
        return loader.load_from_json(file_path)
    
    def save(self, transistor: Transistor) -> None:
        """Save transistor to JSON file."""
        file_path = self._storage_path(transistor.metadata.name)
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

    Save complete versioned core data without legacy manufacturer/housing lists.
    Existing flat legacy JSON remains readable through the compatibility bridge.
    """

    def load_from_json(self, file_path: Path) -> Transistor:
        """Load native core JSON, or the established flat legacy representation."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self.load_from_dict(data)

    def load_from_dict(self, data: dict) -> Transistor:
        """Load decoded JSON in memory, rejecting unknown or malformed native schemas."""
        from copy import deepcopy
        from .adapters import legacy_to_core
        from .serialization import core_from_json_document

        if not isinstance(data, dict):
            raise ValueError("Transistor JSON must be an object")
        if "schema" in data or ("version" in data and "transistor" in data):
            return core_from_json_document(data)

        legacy = _json_dict_to_legacy_transistor(deepcopy(data))
        return legacy_to_core(legacy)

    def save_to_json(self, transistor: Transistor, file_path: Path) -> None:
        """Save all core values and source provenance without legacy coercion."""
        from .serialization import core_to_json_document

        document = core_to_json_document(transistor)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, allow_nan=False)

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
