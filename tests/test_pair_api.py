"""Tests for switching-pair API endpoints."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from transistordatabase.core.pair_models import (
    SwitchingPair,
    PairSide,
    DeviceMetadata,
    DeviceElectricalRatings,
)
from transistordatabase.core.pair_repository import JsonPairRepository


@pytest.fixture
def pair_dir():
    """Create a temp dir with sample switching pairs."""
    tmp_path = Path(tempfile.mkdtemp())
    repo = JsonPairRepository(tmp_path)

    # Self-pair MOSFET
    repo.save_pair(SwitchingPair(
        pair_id="Test_MOSFET_A",
        pair_type="mosfet_self",
        high_side=PairSide(
            device_id="Test_MOSFET_A",
            role="switch",
            metadata=DeviceMetadata(name="Test_MOSFET_A", type="MOSFET", manufacturer="TestCo"),
            electrical_ratings=DeviceElectricalRatings(v_abs_max=650, i_abs_max=50),
        ),
        low_side=PairSide(
            device_id="Test_MOSFET_A",
            role="body_diode",
            metadata=DeviceMetadata(name="Test_MOSFET_A", type="MOSFET", manufacturer="TestCo"),
        ),
    ))

    # Combo pair
    repo.save_pair(SwitchingPair(
        pair_id="Test_MOSFET_A__Test_Diode_B",
        pair_type="mosfet_plus_diode",
        high_side=PairSide(
            device_id="Test_MOSFET_A",
            role="switch",
            metadata=DeviceMetadata(name="Test_MOSFET_A", type="MOSFET", manufacturer="TestCo"),
            electrical_ratings=DeviceElectricalRatings(v_abs_max=650, i_abs_max=50),
        ),
        low_side=PairSide(
            device_id="Test_Diode_B",
            role="diode",
            metadata=DeviceMetadata(name="Test_Diode_B", type="Diode", manufacturer="OtherCo"),
        ),
    ))

    return tmp_path


def test_list_pairs(pair_dir):
    """Test that list_pairs returns pair summaries."""
    repo = JsonPairRepository(pair_dir)
    pairs = repo.list_pairs()
    assert len(pairs) == 2


def test_filter_by_pair_type(pair_dir):
    """Test filtering by pair type."""
    repo = JsonPairRepository(pair_dir)
    results = repo.list_pairs(filters={'pair_type': 'mosfet_self'})
    assert len(results) == 1
    assert results[0]['pair_id'] == 'Test_MOSFET_A'


def test_search_by_device(pair_dir):
    """Test searching by device ID."""
    repo = JsonPairRepository(pair_dir)
    results = repo.search_by_device('Test_MOSFET_A')
    assert len(results) == 2  # appears in both self-pair and combo


def test_unique_devices(pair_dir):
    """Test getting unique devices."""
    repo = JsonPairRepository(pair_dir)
    devices = repo.get_unique_devices()
    device_ids = {d['device_id'] for d in devices}
    assert 'Test_MOSFET_A' in device_ids
    assert 'Test_Diode_B' in device_ids


def test_load_pair(pair_dir):
    """Test loading a full switching pair."""
    repo = JsonPairRepository(pair_dir)
    pair = repo.load_pair('Test_MOSFET_A')
    assert pair.pair_id == 'Test_MOSFET_A'
    assert pair.pair_type == 'mosfet_self'
    assert pair.high_side.device_id == 'Test_MOSFET_A'
    assert pair.low_side.device_id == 'Test_MOSFET_A'


def test_validate_pair(pair_dir):
    """Test pair validation."""
    repo = JsonPairRepository(pair_dir)
    pair = repo.load_pair('Test_MOSFET_A')
    result = pair.validate()
    assert isinstance(result, dict)
    assert 'valid' in result
    assert 'errors' in result
    assert 'warnings' in result
    assert 'quality_score' in result


def test_pair_properties(pair_dir):
    """Test pair property methods."""
    repo = JsonPairRepository(pair_dir)
    pair = repo.load_pair('Test_MOSFET_A')
    assert pair.is_self_pair is True
    assert pair.devices == ['Test_MOSFET_A']

    pair2 = repo.load_pair('Test_MOSFET_A__Test_Diode_B')
    assert pair2.is_self_pair is False
    assert set(pair2.devices) == {'Test_MOSFET_A', 'Test_Diode_B'}


def test_pair_serialization(pair_dir):
    """Test pair to_dict and from_dict round-trip."""
    repo = JsonPairRepository(pair_dir)
    original = repo.load_pair('Test_MOSFET_A')

    # Serialize to dict
    data = original.to_dict()
    assert isinstance(data, dict)
    assert data['pair_id'] == 'Test_MOSFET_A'

    # Deserialize from dict
    reconstructed = SwitchingPair.from_dict(data)
    assert reconstructed.pair_id == original.pair_id
    assert reconstructed.pair_type == original.pair_type
    assert reconstructed.high_side.device_id == original.high_side.device_id
    assert reconstructed.low_side.device_id == original.low_side.device_id


def test_delete_pair(pair_dir):
    """Test deleting a pair."""
    repo = JsonPairRepository(pair_dir)
    assert repo.count() == 2

    success = repo.delete_pair('Test_MOSFET_A')
    assert success is True
    assert repo.count() == 1

    # Try to load deleted pair
    with pytest.raises(FileNotFoundError):
        repo.load_pair('Test_MOSFET_A')


def test_voltage_filtering(pair_dir):
    """Test filtering by voltage range."""
    repo = JsonPairRepository(pair_dir)

    # Filter by v_min
    results = repo.list_pairs(filters={'v_min': 600})
    assert len(results) >= 1

    # Filter by v_max
    results = repo.list_pairs(filters={'v_max': 700})
    assert len(results) >= 1

    # Filter by both
    results = repo.list_pairs(filters={'v_min': 600, 'v_max': 700})
    assert len(results) >= 1


def test_manufacturer_filtering(pair_dir):
    """Test filtering by manufacturer."""
    repo = JsonPairRepository(pair_dir)
    results = repo.list_pairs(filters={'manufacturer': 'TestCo'})
    assert len(results) == 2

    # OtherCo is only on low_side, manufacturer filter checks high_side
    results = repo.list_pairs(filters={'manufacturer': 'OtherCo'})
    assert len(results) == 0


def test_device_filtering(pair_dir):
    """Test filtering by device."""
    repo = JsonPairRepository(pair_dir)
    results = repo.list_pairs(filters={'device': 'Test_Diode_B'})
    assert len(results) == 1
    assert results[0]['pair_id'] == 'Test_MOSFET_A__Test_Diode_B'
