"""Tests for switching-pair data models and repository."""
import json
import tempfile
from pathlib import Path

import pytest

from transistordatabase.core.pair_models import (
    SwitchingPair,
    PairSide,
    SwitchingData,
    DeviceMetadata,
    DeviceElectricalRatings,
    ConductionData,
    ConductionCurve,
    SwitchingEnergyData,
    SwitchingTestConditions,
    MigrationMetadata,
)
from transistordatabase.core.pair_repository import JsonPairRepository


class TestSwitchingPair:
    """Test SwitchingPair model."""

    def test_self_pair_detection(self):
        pair = SwitchingPair(
            pair_id="TestDevice",
            pair_type="mosfet_self",
            high_side=PairSide(device_id="TestDevice"),
            low_side=PairSide(device_id="TestDevice"),
        )
        assert pair.is_self_pair is True
        assert pair.devices == ["TestDevice"]

    def test_combo_pair_detection(self):
        pair = SwitchingPair(
            pair_id="DevA__DevB",
            pair_type="mosfet_plus_diode",
            high_side=PairSide(device_id="DevA"),
            low_side=PairSide(device_id="DevB"),
        )
        assert pair.is_self_pair is False
        assert set(pair.devices) == {"DevA", "DevB"}

    def test_parallel_detection(self):
        pair = SwitchingPair(
            pair_id="10x_DevA",
            pair_type="parallel_self",
            high_side=PairSide(device_id="DevA", count=10),
            low_side=PairSide(device_id="DevA", count=10),
        )
        assert pair.is_parallel is True

    def test_validation_valid_pair(self):
        pair = SwitchingPair(
            pair_id="Test",
            pair_type="mosfet_self",
            high_side=PairSide(
                device_id="Test",
                metadata=DeviceMetadata(name="Test", type="MOSFET", manufacturer="TestCo"),
                electrical_ratings=DeviceElectricalRatings(v_abs_max=650, i_abs_max=50),
                conduction=ConductionData(
                    temperatures=[25.0],
                    curves=[ConductionCurve(t_j=25.0, on_state_voltage=[0, 1, 2], on_state_current=[0, 10, 30])],
                ),
            ),
            low_side=PairSide(
                device_id="Test",
                metadata=DeviceMetadata(name="Test", type="MOSFET", manufacturer="TestCo"),
                electrical_ratings=DeviceElectricalRatings(v_abs_max=650, i_abs_max=50),
            ),
            switching_data=SwitchingData(
                turn_on=SwitchingEnergyData(
                    current_axis=[0, 10, 30],
                    energy={"25.0": [0, 100e-6, 500e-6]},
                ),
                turn_off=SwitchingEnergyData(
                    current_axis=[0, 10, 30],
                    energy={"25.0": [0, 80e-6, 400e-6]},
                ),
            ),
        )
        result = pair.validate()
        assert result['valid'] is True
        assert result['quality_score'] > 80

    def test_validation_missing_fields(self):
        pair = SwitchingPair()
        result = pair.validate()
        assert result['valid'] is False
        assert len(result['errors']) >= 3

    def test_validation_array_mismatch(self):
        pair = SwitchingPair(
            pair_id="Test",
            pair_type="mosfet_self",
            high_side=PairSide(device_id="Test"),
            low_side=PairSide(device_id="Test"),
            switching_data=SwitchingData(
                turn_on=SwitchingEnergyData(
                    current_axis=[0, 10, 30],
                    energy={"25.0": [0, 100e-6]},  # Wrong length!
                ),
            ),
        )
        result = pair.validate()
        assert any("array" in e.lower() or "values" in e.lower() or "length" in e.lower() for e in result['errors'])

    def test_serialization_roundtrip(self):
        pair = SwitchingPair(
            pair_id="Roundtrip_Test",
            pair_type="mosfet_self",
            high_side=PairSide(
                device_id="RT_Device",
                count=1,
                role="switch",
                metadata=DeviceMetadata(name="RT_Device", type="MOSFET", manufacturer="Test"),
                electrical_ratings=DeviceElectricalRatings(v_abs_max=650, i_abs_max=50),
            ),
            low_side=PairSide(
                device_id="RT_Device",
                count=1,
                role="body_diode",
                metadata=DeviceMetadata(name="RT_Device", type="MOSFET", manufacturer="Test"),
            ),
        )

        data = pair.to_dict()
        restored = SwitchingPair.from_dict(data)

        assert restored.pair_id == pair.pair_id
        assert restored.pair_type == pair.pair_type
        assert restored.high_side.device_id == pair.high_side.device_id
        assert restored.low_side.device_id == pair.low_side.device_id
        assert restored.is_self_pair is True


class TestJsonPairRepository:
    """Test JsonPairRepository."""

    def test_save_and_load(self, tmp_path):
        repo = JsonPairRepository(tmp_path)
        pair = SwitchingPair(
            pair_id="Save_Test",
            pair_type="mosfet_self",
            high_side=PairSide(device_id="Save_Test", role="switch"),
            low_side=PairSide(device_id="Save_Test", role="body_diode"),
        )
        repo.save_pair(pair)

        loaded = repo.load_pair("Save_Test")
        assert loaded.pair_id == "Save_Test"
        assert loaded.high_side.device_id == "Save_Test"

    def test_list_pairs(self, tmp_path):
        repo = JsonPairRepository(tmp_path)
        for i in range(3):
            pair = SwitchingPair(
                pair_id=f"Device_{i}",
                pair_type="mosfet_self",
                high_side=PairSide(device_id=f"Device_{i}"),
                low_side=PairSide(device_id=f"Device_{i}"),
            )
            repo.save_pair(pair)

        pairs = repo.list_pairs()
        assert len(pairs) == 3

    def test_search_by_device(self, tmp_path):
        repo = JsonPairRepository(tmp_path)

        # Self pair
        repo.save_pair(SwitchingPair(
            pair_id="DevA",
            pair_type="mosfet_self",
            high_side=PairSide(device_id="DevA"),
            low_side=PairSide(device_id="DevA"),
        ))

        # Combo pair
        repo.save_pair(SwitchingPair(
            pair_id="DevA__DevB",
            pair_type="mosfet_plus_diode",
            high_side=PairSide(device_id="DevA"),
            low_side=PairSide(device_id="DevB"),
        ))

        results = repo.search_by_device("DevA")
        assert len(results) == 2

        results = repo.search_by_device("DevB")
        assert len(results) == 1

    def test_filter_by_pair_type(self, tmp_path):
        repo = JsonPairRepository(tmp_path)
        repo.save_pair(SwitchingPair(pair_id="A", pair_type="mosfet_self", high_side=PairSide(device_id="A"), low_side=PairSide(device_id="A")))
        repo.save_pair(SwitchingPair(pair_id="B__C", pair_type="mosfet_plus_diode", high_side=PairSide(device_id="B"), low_side=PairSide(device_id="C")))

        results = repo.list_pairs(filters={'pair_type': 'mosfet_self'})
        assert len(results) == 1
        assert results[0]['pair_id'] == 'A'

    def test_unique_devices(self, tmp_path):
        repo = JsonPairRepository(tmp_path)
        repo.save_pair(SwitchingPair(pair_id="A", pair_type="mosfet_self", high_side=PairSide(device_id="A"), low_side=PairSide(device_id="A")))
        repo.save_pair(SwitchingPair(pair_id="A__B", pair_type="mosfet_plus_diode", high_side=PairSide(device_id="A"), low_side=PairSide(device_id="B")))

        devices = repo.get_unique_devices()
        device_ids = {d['device_id'] for d in devices}
        assert device_ids == {'A', 'B'}

    def test_count(self, tmp_path):
        repo = JsonPairRepository(tmp_path)
        assert repo.count() == 0
        repo.save_pair(SwitchingPair(pair_id="X", pair_type="mosfet_self", high_side=PairSide(device_id="X"), low_side=PairSide(device_id="X")))
        assert repo.count() == 1

    def test_delete_pair(self, tmp_path):
        repo = JsonPairRepository(tmp_path)
        repo.save_pair(SwitchingPair(pair_id="Del", pair_type="mosfet_self", high_side=PairSide(device_id="Del"), low_side=PairSide(device_id="Del")))
        assert repo.count() == 1
        repo.delete_pair("Del")
        assert repo.count() == 0

    def test_load_nonexistent(self, tmp_path):
        repo = JsonPairRepository(tmp_path)
        with pytest.raises(FileNotFoundError):
            repo.load_pair("nonexistent")
