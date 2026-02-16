"""
JSON repository for switching-pair data.

Handles loading, saving, searching, and listing switching pairs from JSON files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transistordatabase.core.pair_models import SwitchingPair

logger = logging.getLogger(__name__)


class JsonPairRepository:
    """Repository for switching-pair JSON files."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] | None = None

    def save_pair(self, pair: SwitchingPair) -> Path:
        """Save a switching pair to JSON file."""
        file_path = self.base_dir / f"{pair.pair_id}.json"
        data = pair.to_dict()

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Invalidate index cache
        self._index = None
        logger.debug(f"Saved pair: {pair.pair_id}")
        return file_path

    def load_pair(self, pair_id: str) -> SwitchingPair:
        """Load a switching pair from JSON file."""
        file_path = self.base_dir / f"{pair_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Switching pair not found: {pair_id}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        return SwitchingPair.from_dict(data)

    def delete_pair(self, pair_id: str) -> bool:
        """Delete a switching pair JSON file."""
        file_path = self.base_dir / f"{pair_id}.json"
        if file_path.exists():
            file_path.unlink()
            self._index = None
            return True
        return False

    def list_pairs(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """List all pairs with optional filtering. Returns summary dicts."""
        index = self._build_index()
        results = list(index.values())

        if filters:
            results = self._apply_filters(results, filters)

        return results

    def search_by_device(self, device_id: str) -> list[dict[str, Any]]:
        """Find all switching pairs containing a specific device."""
        index = self._build_index()
        return [
            entry for entry in index.values()
            if device_id in entry.get('devices', [])
        ]

    def get_unique_devices(self) -> list[dict[str, Any]]:
        """Get deduplicated list of unique devices across all pairs."""
        index = self._build_index()
        devices: dict[str, dict[str, Any]] = {}

        for entry in index.values():
            for side_key in ['high_side', 'low_side']:
                side = entry.get(side_key, {})
                dev_id = side.get('device_id', '')
                if dev_id and dev_id not in devices:
                    devices[dev_id] = {
                        'device_id': dev_id,
                        'type': side.get('type', ''),
                        'manufacturer': side.get('manufacturer', ''),
                        'v_abs_max': side.get('v_abs_max', 0),
                        'i_abs_max': side.get('i_abs_max', 0),
                        'pair_count': 0,
                    }
                if dev_id in devices:
                    devices[dev_id]['pair_count'] += 1

        return list(devices.values())

    def _build_index(self) -> dict[str, dict[str, Any]]:
        """Build or return cached index of all pairs."""
        if self._index is not None:
            return self._index

        self._index = {}
        for json_file in self.base_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                hs = data.get('high_side', {})
                ls = data.get('low_side', {})
                hs_meta = hs.get('metadata', {})
                ls_meta = ls.get('metadata', {})
                hs_elec = hs.get('electrical_ratings', {})

                self._index[json_file.stem] = {
                    'pair_id': data.get('pair_id', json_file.stem),
                    'pair_type': data.get('pair_type', ''),
                    'topology': data.get('topology', ''),
                    'devices': list({hs.get('device_id', ''), ls.get('device_id', '')} - {''}),
                    'high_side': {
                        'device_id': hs.get('device_id', ''),
                        'count': hs.get('count', 1),
                        'type': hs_meta.get('type', ''),
                        'manufacturer': hs_meta.get('manufacturer', ''),
                        'v_abs_max': hs_elec.get('v_abs_max', 0),
                        'i_abs_max': hs_elec.get('i_abs_max', 0),
                    },
                    'low_side': {
                        'device_id': ls.get('device_id', ''),
                        'count': ls.get('count', 1),
                        'type': ls_meta.get('type', ''),
                        'manufacturer': ls_meta.get('manufacturer', ''),
                    },
                    'source': data.get('switching_data', {}).get('source', ''),
                    'quality_score': data.get('migration', {}).get('quality_score', 0),
                }
            except Exception as e:
                logger.warning(f"Error indexing {json_file}: {e}")

        return self._index

    def _apply_filters(self, results: list[dict], filters: dict[str, Any]) -> list[dict]:
        """Apply filters to pair list."""
        filtered = results

        if 'pair_type' in filters:
            types = filters['pair_type']
            if isinstance(types, str):
                types = [t.strip() for t in types.split(',')]
            filtered = [r for r in filtered if r.get('pair_type') in types]

        if 'device' in filters:
            dev = filters['device']
            filtered = [r for r in filtered if dev in r.get('devices', [])]

        if 'manufacturer' in filters:
            mfr = filters['manufacturer']
            filtered = [r for r in filtered if r.get('high_side', {}).get('manufacturer', '') == mfr]

        if 'v_min' in filters:
            v_min = float(filters['v_min'])
            filtered = [r for r in filtered if r.get('high_side', {}).get('v_abs_max', 0) >= v_min]

        if 'v_max' in filters:
            v_max = float(filters['v_max'])
            filtered = [r for r in filtered if r.get('high_side', {}).get('v_abs_max', 0) <= v_max]

        if 'source' in filters:
            src = filters['source']
            filtered = [r for r in filtered if r.get('source') == src]

        return filtered

    def count(self) -> int:
        """Count total switching pairs."""
        return len(list(self.base_dir.glob('*.json')))
