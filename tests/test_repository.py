"""Tests for core repository implementations and DatabaseManager core shim."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from transistordatabase.core import models as core
from transistordatabase.core.repository import (
    JsonTransistorLoader,
    JsonTransistorRepository,
    _convert_arrays_to_numpy,
    _json_dict_to_legacy_transistor,
)
from transistordatabase.database_manager import DatabaseManager

# Paths to real test data
TEST_DIR = Path(__file__).parent / "test_data"
DATABASE_DIR = TEST_DIR / "database"
FIXED_TRANSISTOR_JSON = TEST_DIR / "CREE_C3M0060065J.json"
DATABASE_TRANSISTOR_JSON = DATABASE_DIR / "CREE_C3M0016120K.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir():
    """Create a temporary directory for repo tests, clean up after."""
    d = tempfile.mkdtemp(prefix="tdb_repo_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def loader():
    """Provide a JsonTransistorLoader instance."""
    return JsonTransistorLoader()


@pytest.fixture()
def database_json():
    """Fixture for DatabaseManager in JSON mode, restores state after test."""
    if not DATABASE_DIR.exists():
        raise Exception("The folder test_data/database is missing.")

    db = DatabaseManager()
    db.set_operation_mode_json(str(DATABASE_DIR))
    transistor = db.load_transistor("CREE_C3M0016120K")

    yield db

    # Restore database state
    transistor_names = db.get_transistor_names_list()
    if "CREE_C3M0016120K" not in transistor_names:
        with open(str(DATABASE_TRANSISTOR_JSON), "w") as fd:
            json.dump(transistor.convert_to_dict(), fd)


# ---------------------------------------------------------------------------
# _convert_arrays_to_numpy tests
# ---------------------------------------------------------------------------

class TestConvertArraysToNumpy:
    """Test the numpy array conversion helper."""

    def test_capacitance_conversion(self):
        """Verify capacitance graph arrays are converted to numpy."""
        d = {
            'c_oss': [{'graph_v_c': [[1, 2], [3, 4]]}],
            'c_iss': None,
            'switch': None,
            'diode': None,
        }
        _convert_arrays_to_numpy(d)
        assert isinstance(d['c_oss'][0]['graph_v_c'], np.ndarray)

    def test_switch_channel_conversion(self):
        """Verify switch channel graph_v_i is converted to numpy."""
        d = {
            'switch': {
                'thermal_foster': {'graph_t_rthjc': None},
                'channel': [{'graph_v_i': [[0, 1], [0, 10]]}],
            },
            'diode': None,
        }
        _convert_arrays_to_numpy(d)
        assert isinstance(d['switch']['channel'][0]['graph_v_i'], np.ndarray)

    def test_diode_e_rr_conversion(self):
        """Verify diode e_rr graph is converted based on dataset_type."""
        d = {
            'switch': None,
            'diode': {
                'thermal_foster': {'graph_t_rthjc': None},
                'channel': [],
                'e_rr': [{'dataset_type': 'graph_i_e', 'graph_i_e': [[1, 2], [3, 4]]}],
            },
        }
        _convert_arrays_to_numpy(d)
        assert isinstance(d['diode']['e_rr'][0]['graph_i_e'], np.ndarray)

    def test_handles_missing_keys(self):
        """Verify no crash when optional keys are missing."""
        d = {'switch': None, 'diode': None}
        _convert_arrays_to_numpy(d)  # Should not raise


# ---------------------------------------------------------------------------
# JsonTransistorLoader tests
# ---------------------------------------------------------------------------

class TestJsonTransistorLoader:
    """Test JsonTransistorLoader via adapter bridge."""

    def test_load_returns_core_transistor(self, loader):
        """Verify load_from_json returns a core.Transistor."""
        t = loader.load_from_json(DATABASE_TRANSISTOR_JSON)
        assert isinstance(t, core.Transistor)

    def test_load_metadata(self, loader):
        """Verify metadata fields are correctly loaded."""
        t = loader.load_from_json(DATABASE_TRANSISTOR_JSON)
        assert t.metadata.name == "CREE_C3M0016120K"
        assert t.metadata.type in ("SiC-MOSFET", "MOSFET", "IGBT", "GaN-Transistor")
        assert t.metadata.manufacturer != ""

    def test_load_electrical_ratings(self, loader):
        """Verify electrical ratings are loaded."""
        t = loader.load_from_json(DATABASE_TRANSISTOR_JSON)
        assert t.electrical_ratings.v_abs_max > 0
        assert t.electrical_ratings.i_abs_max > 0

    def test_load_switch_data(self, loader):
        """Verify switch channel data is loaded with numpy arrays."""
        t = loader.load_from_json(DATABASE_TRANSISTOR_JSON)
        assert len(t.switch.channel_data) > 0
        ch = t.switch.channel_data[0]
        assert isinstance(ch.graph_v_i, np.ndarray)
        assert ch.t_j is not None

    def test_load_switch_switching_loss(self, loader):
        """Verify switch switching loss data is loaded."""
        t = loader.load_from_json(DATABASE_TRANSISTOR_JSON)
        assert len(t.switch.e_on_data) > 0 or len(t.switch.e_off_data) > 0

    def test_load_diode_data(self, loader):
        """Verify diode channel data is loaded."""
        t = loader.load_from_json(DATABASE_TRANSISTOR_JSON)
        assert len(t.diode.channel_data) > 0
        ch = t.diode.channel_data[0]
        assert isinstance(ch.graph_v_i, np.ndarray)

    def test_load_second_transistor(self, loader):
        """Verify a second JSON file also loads correctly."""
        t = loader.load_from_json(FIXED_TRANSISTOR_JSON)
        assert isinstance(t, core.Transistor)
        assert t.metadata.name == "CREE_C3M0060065J"

    def test_load_capacitances(self, loader):
        """Verify capacitance data is loaded."""
        t = loader.load_from_json(DATABASE_TRANSISTOR_JSON)
        # At least one of c_oss, c_iss, c_rss should have data
        has_caps = (len(t.c_oss) > 0 or len(t.c_iss) > 0 or len(t.c_rss) > 0)
        assert has_caps


# ---------------------------------------------------------------------------
# JsonTransistorRepository tests
# ---------------------------------------------------------------------------

class TestJsonTransistorRepository:
    """Test the file-based repository using real JSON data."""

    def test_list_all(self):
        """Verify list_all returns available transistor names."""
        repo = JsonTransistorRepository(DATABASE_DIR)
        names = repo.list_all()
        assert "CREE_C3M0016120K" in names

    def test_get_by_name(self):
        """Verify get_by_name returns a core Transistor."""
        repo = JsonTransistorRepository(DATABASE_DIR)
        t = repo.get_by_name("CREE_C3M0016120K")
        assert t is not None
        assert isinstance(t, core.Transistor)
        assert t.metadata.name == "CREE_C3M0016120K"

    def test_get_nonexistent_returns_none(self, tmp_dir):
        """Verify get_by_name returns None for missing transistor."""
        repo = JsonTransistorRepository(tmp_dir)
        assert repo.get_by_name("NONEXISTENT") is None

    def test_save_and_reload(self, tmp_dir):
        """Verify save then get_by_name produces equivalent transistor."""
        # Load a real transistor
        src_repo = JsonTransistorRepository(DATABASE_DIR)
        original = src_repo.get_by_name("CREE_C3M0016120K")
        assert original is not None

        # Save to temp dir
        dest_repo = JsonTransistorRepository(tmp_dir)
        dest_repo.save(original)

        # Verify file exists
        assert (tmp_dir / "CREE_C3M0016120K.json").exists()

        # Reload and compare key fields
        reloaded = dest_repo.get_by_name("CREE_C3M0016120K")
        assert reloaded is not None
        assert reloaded.metadata.name == original.metadata.name
        assert reloaded.metadata.type == original.metadata.type
        assert reloaded.electrical_ratings.v_abs_max == original.electrical_ratings.v_abs_max
        assert reloaded.electrical_ratings.i_abs_max == original.electrical_ratings.i_abs_max
        assert len(reloaded.switch.channel_data) == len(original.switch.channel_data)
        assert len(reloaded.diode.channel_data) == len(original.diode.channel_data)

    def test_delete(self, tmp_dir):
        """Verify delete removes the transistor file."""
        src_repo = JsonTransistorRepository(DATABASE_DIR)
        original = src_repo.get_by_name("CREE_C3M0016120K")

        dest_repo = JsonTransistorRepository(tmp_dir)
        dest_repo.save(original)
        assert "CREE_C3M0016120K" in dest_repo.list_all()

        result = dest_repo.delete("CREE_C3M0016120K")
        assert result is True
        assert "CREE_C3M0016120K" not in dest_repo.list_all()

    def test_delete_nonexistent(self, tmp_dir):
        """Verify delete returns False for missing transistor."""
        repo = JsonTransistorRepository(tmp_dir)
        assert repo.delete("NONEXISTENT") is False


# ---------------------------------------------------------------------------
# DatabaseManager core shim tests
# ---------------------------------------------------------------------------

class TestDatabaseManagerCoreShim:
    """Test DatabaseManager core integration methods."""

    def test_load_transistor_core(self, database_json):
        """Verify load_transistor_core returns a core Transistor."""
        t = database_json.load_transistor_core("CREE_C3M0016120K")
        assert isinstance(t, core.Transistor)
        assert t.metadata.name == "CREE_C3M0016120K"

    def test_load_transistor_core_matches_legacy(self, database_json):
        """Verify core and legacy loads have consistent data."""
        legacy = database_json.load_transistor("CREE_C3M0016120K")
        core_t = database_json.load_transistor_core("CREE_C3M0016120K")

        # Compare key fields
        assert core_t.metadata.name == legacy.name
        assert core_t.metadata.type == legacy.type
        assert core_t.electrical_ratings.v_abs_max == legacy.v_abs_max
        assert core_t.electrical_ratings.i_abs_max == legacy.i_abs_max
        assert len(core_t.switch.channel_data) == len(legacy.switch.channel)
        assert len(core_t.diode.channel_data) == len(legacy.diode.channel)

    def test_load_transistor_core_nonexistent(self, database_json):
        """Verify load_transistor_core returns None for missing transistor."""
        t = database_json.load_transistor_core("NONEXISTENT_TRANSISTOR")
        assert t is None

    def test_core_repository_property(self, database_json):
        """Verify core_repository returns a JsonTransistorRepository."""
        repo = database_json.core_repository
        assert isinstance(repo, JsonTransistorRepository)
        names = repo.list_all()
        assert "CREE_C3M0016120K" in names

    def test_core_repository_cached(self, database_json):
        """Verify core_repository returns the same instance on repeated access."""
        repo1 = database_json.core_repository
        repo2 = database_json.core_repository
        assert repo1 is repo2

    def test_legacy_load_still_works(self, database_json):
        """Verify the legacy load_transistor method is unaffected."""
        from transistordatabase.transistor import Transistor as LegacyTransistor
        t = database_json.load_transistor("CREE_C3M0016120K")
        assert isinstance(t, LegacyTransistor)
