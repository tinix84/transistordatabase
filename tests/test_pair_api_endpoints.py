"""Integration tests for switching-pair FastAPI endpoints."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transistordatabase.core.pair_models import (
    SwitchingPair,
    PairSide,
    DeviceMetadata,
    DeviceElectricalRatings,
)
from transistordatabase.core.pair_repository import JsonPairRepository
from transistordatabase.gui_web.backend.main import app


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
            electrical_ratings=DeviceElectricalRatings(v_abs_max=650, i_abs_max=50),
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
            electrical_ratings=DeviceElectricalRatings(v_abs_max=600, i_abs_max=30),
        ),
    ))

    return tmp_path


@pytest.fixture
def client(pair_dir, monkeypatch):
    """Patch the app to use temp pair directory and return test client."""
    import transistordatabase.gui_web.backend.main as main_module
    monkeypatch.setattr(main_module, '_pair_repo', JsonPairRepository(pair_dir))
    return TestClient(app)


def test_list_pairs_endpoint(client):
    """Test GET /api/pairs/ returns pair list."""
    response = client.get("/api/pairs/")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "page" in data
    assert "per_page" in data
    assert "pairs" in data
    assert data["total"] == 2


def test_list_pairs_pagination(client):
    """Test pagination on list_pairs endpoint."""
    response = client.get("/api/pairs/?page=1&per_page=1")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["page"] == 1
    assert data["per_page"] == 1
    assert len(data["pairs"]) == 1


def test_list_pairs_filter_by_type(client):
    """Test filtering by pair_type."""
    response = client.get("/api/pairs/?pair_type=mosfet_self")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["pairs"][0]["pair_id"] == "Test_MOSFET_A"


def test_list_pairs_filter_by_device(client):
    """Test filtering by device."""
    response = client.get("/api/pairs/?device=Test_Diode_B")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["pairs"][0]["pair_id"] == "Test_MOSFET_A__Test_Diode_B"


def test_list_pairs_filter_by_voltage(client):
    """Test filtering by voltage range."""
    response = client.get("/api/pairs/?v_min=600&v_max=700")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_get_pair_endpoint(client):
    """Test GET /api/pairs/{pair_id} returns full pair data."""
    response = client.get("/api/pairs/Test_MOSFET_A")
    assert response.status_code == 200
    data = response.json()
    assert data["pair_id"] == "Test_MOSFET_A"
    assert data["pair_type"] == "mosfet_self"
    assert data["high_side"]["device_id"] == "Test_MOSFET_A"


def test_get_pair_not_found(client):
    """Test GET /api/pairs/{pair_id} with non-existent pair."""
    response = client.get("/api/pairs/NonExistent")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_validate_pair_endpoint(client):
    """Test GET /api/pairs/{pair_id}/validate."""
    response = client.get("/api/pairs/Test_MOSFET_A/validate")
    assert response.status_code == 200
    data = response.json()
    assert "valid" in data
    assert "errors" in data
    assert "warnings" in data
    assert "quality_score" in data


def test_list_unique_devices_endpoint(client):
    """Test GET /api/devices/ returns unique devices."""
    response = client.get("/api/devices/")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "devices" in data
    assert data["total"] == 2
    device_ids = {d["device_id"] for d in data["devices"]}
    assert "Test_MOSFET_A" in device_ids
    assert "Test_Diode_B" in device_ids


def test_list_unique_devices_filter_by_type(client):
    """Test filtering devices by type."""
    response = client.get("/api/devices/?type=MOSFET")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["devices"][0]["type"] == "MOSFET"


def test_get_device_pairs_endpoint(client):
    """Test GET /api/devices/{device_id}/pairs."""
    response = client.get("/api/devices/Test_MOSFET_A/pairs")
    assert response.status_code == 200
    data = response.json()
    assert data["device_id"] == "Test_MOSFET_A"
    assert data["total"] == 2
    assert len(data["pairs"]) == 2


def test_export_pairs_csv_endpoint(client):
    """Test GET /api/pairs/export/csv returns CSV file."""
    response = client.get("/api/pairs/export/csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment" in response.headers["content-disposition"]

    # Parse CSV content
    csv_content = response.text
    lines = csv_content.strip().split('\n')
    assert len(lines) >= 3  # Header + 2 pairs


def test_export_pairs_csv_filter(client):
    """Test CSV export with filtering."""
    response = client.get("/api/pairs/export/csv?pair_type=mosfet_self")
    assert response.status_code == 200
    csv_content = response.text
    lines = csv_content.strip().split('\n')
    # Header + 1 pair (only mosfet_self)
    assert len(lines) >= 2


def test_export_pairs_csv_format(client):
    """Test CSV export has correct columns."""
    response = client.get("/api/pairs/export/csv")
    assert response.status_code == 200
    csv_content = response.text
    lines = csv_content.strip().split('\n')

    # Check header has expected columns
    header = lines[0]
    expected_columns = [
        'name', 'blockingVoltage', 'pair_type', 'high_side_device', 'low_side_device',
        'high_side_count', 'low_side_count',
        'forwardThermalResistance', 'reverseThermalResistance',
        'cost', 'weight', 'source', 'quality_score',
    ]
    for col in expected_columns:
        assert col in header
