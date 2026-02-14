"""Tests for the FastAPI backend REST API endpoints.

Uses FastAPI's TestClient (which does not require a running server).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from transistordatabase.core.models import (
    Transistor,
    TransistorMetadata,
    ElectricalRatings,
    ThermalProperties,
)
from transistordatabase.core.repository import JsonTransistorRepository

# Lazily import TestClient and the app to avoid import errors if
# httpx / starlette aren't installed.
try:
    from fastapi.testclient import TestClient
    _HAS_TEST_CLIENT = True
except ImportError:
    _HAS_TEST_CLIENT = False

# Path to real test data
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "database"

pytestmark = pytest.mark.skipif(
    not _HAS_TEST_CLIENT,
    reason="fastapi[testclient] or httpx not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Create a temporary database directory seeded with one test transistor."""
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    # Copy real test JSON into temp dir
    src = TEST_DATA_DIR / "CREE_C3M0016120K.json"
    if src.exists():
        shutil.copy(src, db_dir / "CREE_C3M0016120K.json")
    return db_dir


@pytest.fixture()
def client(tmp_db, monkeypatch):
    """Provide a FastAPI TestClient with the repo pointed at a temp directory."""
    import transistordatabase.gui_web.backend.main as backend_module

    # Swap the global repo to point at our temp dir
    original_repo = backend_module._repo
    test_repo = JsonTransistorRepository(tmp_db)
    monkeypatch.setattr(backend_module, "_repo", test_repo)

    client = TestClient(backend_module.app)
    yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCRUD:
    """Test basic CRUD endpoints."""

    def test_root(self, client):
        """Root endpoint returns version info."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0.0"

    def test_list_transistors(self, client):
        """GET /api/transistors returns a list."""
        resp = client.get("/api/transistors")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["metadata"]["name"] == "CREE_C3M0016120K"

    def test_get_transistor(self, client):
        """GET /api/transistors/{id} returns transistor data."""
        resp = client.get("/api/transistors/CREE_C3M0016120K")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["name"] == "CREE_C3M0016120K"
        assert "electrical" in data
        assert "thermal" in data
        assert "switch" in data
        assert "diode" in data

    def test_get_transistor_not_found(self, client):
        """GET /api/transistors/{id} returns 404 for missing transistor."""
        resp = client.get("/api/transistors/NONEXISTENT")
        assert resp.status_code == 404

    def test_delete_transistor(self, client, tmp_db):
        """DELETE removes a transistor."""
        # First verify it exists
        resp = client.get("/api/transistors/CREE_C3M0016120K")
        assert resp.status_code == 200

        # Delete it
        resp = client.delete("/api/transistors/CREE_C3M0016120K")
        assert resp.status_code == 200

        # Verify it's gone
        resp = client.get("/api/transistors/CREE_C3M0016120K")
        assert resp.status_code == 404

    def test_delete_not_found(self, client):
        """DELETE returns 404 for missing transistor."""
        resp = client.delete("/api/transistors/NONEXISTENT")
        assert resp.status_code == 404


class TestValidation:
    """Test validation endpoint."""

    def test_validate_transistor(self, client):
        """POST validate returns errors/warnings structure."""
        resp = client.post("/api/transistors/CREE_C3M0016120K/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" in data
        assert "warnings" in data
        assert isinstance(data["errors"], list)


class TestExport:
    """Test export endpoints."""

    def test_export_json(self, client):
        """Export in JSON format returns a file."""
        resp = client.post("/api/transistors/CREE_C3M0016120K/export/json")
        assert resp.status_code == 200

    def test_export_csv(self, client):
        """Export in CSV format returns a file."""
        resp = client.post("/api/transistors/CREE_C3M0016120K/export/csv")
        assert resp.status_code == 200

    def test_export_unsupported_format(self, client):
        """Export with unsupported format returns 400."""
        resp = client.post("/api/transistors/CREE_C3M0016120K/export/pdf")
        assert resp.status_code == 400

    def test_export_not_found(self, client):
        """Export for missing transistor returns 404."""
        resp = client.post("/api/transistors/NONEXISTENT/export/json")
        assert resp.status_code == 404


class TestPlots:
    """Test plot data endpoints."""

    def test_plot_channel(self, client):
        """GET /api/plots/channel returns plot data."""
        resp = client.get("/api/plots/channel/CREE_C3M0016120K")
        assert resp.status_code == 200
        data = resp.json()
        assert "curves" in data or "error" in data

    def test_plot_channel_diode(self, client):
        """GET /api/plots/channel with component=diode returns data."""
        resp = client.get(
            "/api/plots/channel/CREE_C3M0016120K",
            params={"component": "diode"},
        )
        assert resp.status_code == 200

    def test_plot_switching(self, client):
        """GET /api/plots/switching returns plot data."""
        resp = client.get("/api/plots/switching/CREE_C3M0016120K")
        assert resp.status_code == 200

    def test_plot_soa(self, client):
        """GET /api/plots/soa returns plot data."""
        resp = client.get("/api/plots/soa/CREE_C3M0016120K")
        assert resp.status_code == 200

    def test_plot_thermal(self, client):
        """GET /api/plots/thermal returns plot data."""
        resp = client.get("/api/plots/thermal/CREE_C3M0016120K")
        assert resp.status_code == 200

    def test_plot_gate_charge(self, client):
        """GET /api/plots/gate_charge returns plot data."""
        resp = client.get("/api/plots/gate_charge/CREE_C3M0016120K")
        assert resp.status_code == 200

    def test_plot_not_found(self, client):
        """Plot endpoints return 404 for missing transistor."""
        resp = client.get("/api/plots/channel/NONEXISTENT")
        assert resp.status_code == 404


class TestUpload:
    """Test upload endpoint."""

    def test_upload_json(self, client, tmp_db):
        """Upload a JSON file imports the transistor."""
        # Use the database transistor (known to load cleanly via legacy constructor)
        src = TEST_DATA_DIR / "CREE_C3M0016120K.json"
        if not src.exists():
            pytest.skip("Test JSON file not available")

        # Delete existing copy so we can test re-upload
        client.delete("/api/transistors/CREE_C3M0016120K")

        with open(src, "rb") as f:
            resp = client.post(
                "/api/transistors/upload",
                files={"file": ("CREE_C3M0016120K.json", f, "application/json")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "CREE_C3M0016120K"

        # Verify it's now in the database
        resp = client.get("/api/transistors/CREE_C3M0016120K")
        assert resp.status_code == 200

    def test_upload_non_json(self, client):
        """Upload a non-JSON file returns 400."""
        resp = client.post(
            "/api/transistors/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400
