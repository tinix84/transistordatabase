"""Isolated API tests for read-only DPT waveform serialization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.gui_web.backend import main


DEVICE_NAME = "DPT Waveform Fixture"


class TemporaryJsonRepository:
    """Repository facade whose records live only under pytest's temp path."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory

    def get_by_name(self, name: str) -> Any | None:
        """Load one isolated test record by its stored name."""
        path = self.data_directory / f"{name}.json"
        if not path.exists():
            return None
        return JsonTransistorLoader().load_from_json(path)


def _fixture_record() -> dict[str, Any]:
    source = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "test_data"
        / "CREE_C3M0060065J.json"
    )
    record = json.loads(source.read_text(encoding="utf-8"))
    record["name"] = DEVICE_NAME
    record["raw_measurement_data"] = [
        {
            "dataset_type": "dpt_u_i",
            "comment": "first measurement",
            "operator_note": "preserve available metadata",
            "measurement_testbench": "temporary fixture bench",
            "t_j": 25,
            "v_supply": 400,
            "v_g": 15,
            "v_g_off": -5,
            "r_g": 10,
            "load_inductance": 0.00075,
            "dpt_on_vds": [
                [[0.0, 400.0], [1e-6, 210.0]],
                [[0.0, 390.0], [1e-6, 200.0]],
            ],
            "dpt_on_id": [
                [[0.0, 0.0], [1e-6, 20.0]],
                [[0.0, 0.0], [1e-6, 22.0]],
            ],
            "dpt_off_vds": [],
            "dpt_off_id": [],
        },
        {
            "dataset_type": "dpt_u_i_r",
            "comment": "second measurement",
            "t_j": 125,
            "v_supply": 600,
            "dpt_on_vds": [],
            "dpt_on_id": [],
            "dpt_off_vds": [[[0.0, 600.0], [2e-6, 580.0]]],
            "dpt_off_id": [[[0.0, 40.0], [2e-6, 35.0]]],
        },
    ]
    return record


@pytest.fixture
def isolated_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path, dict[str, Any]]:
    """Create one test-owned record and point the API at its directory."""
    record = _fixture_record()
    record_path = tmp_path / f"{DEVICE_NAME}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(tmp_path))
    return TestClient(main.app), record_path, record


def test_dpt_data_includes_multiple_raw_records_and_acquisitions(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Keep summary fields and expose each raw waveform unchanged."""
    client, record_path, _record = isolated_database
    before = record_path.read_bytes()

    response = client.get(f"/api/transistors/{DEVICE_NAME}/dpt_data")

    assert response.status_code == 200
    payload = response.json()
    assert "waveform_preview" not in payload
    assert payload["has_e_on"] is True
    assert payload["has_e_off"] is True
    assert "has_e_rr" in payload
    assert "metadata" in payload
    raw_records = payload["raw_measurement_data"]
    assert len(raw_records) == 2
    assert raw_records[0]["metadata"]["comment"] == "first measurement"
    assert raw_records[0]["metadata"]["operator_note"] == "preserve available metadata"
    assert raw_records[0]["metadata"]["t_j"] == 25
    assert raw_records[0]["dpt_on_vds"] == [
        [[0.0, 400.0], [1e-6, 210.0]],
        [[0.0, 390.0], [1e-6, 200.0]],
    ]
    assert raw_records[0]["dpt_on_id"][1][-1] == [1e-6, 22.0]
    assert raw_records[0]["dpt_off_vds"] == []
    assert raw_records[0]["dpt_off_id"] == []
    assert raw_records[1]["dpt_on_vds"] == []
    assert raw_records[1]["dpt_on_id"] == []
    assert raw_records[1]["dpt_off_vds"] == [[[0.0, 600.0], [2e-6, 580.0]]]
    json.dumps(payload, allow_nan=False)
    assert record_path.read_bytes() == before


def test_default_dpt_get_keeps_full_response_without_preview_caps(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in preview budgets do not narrow legacy default GET behavior."""
    client, record_path, _record = isolated_database
    monkeypatch.setattr(main, "MAX_DPT_LEGACY_RECORD_FILE_BYTES", 16)
    monkeypatch.setattr(main, "MAX_DPT_RAW_MEASUREMENT_BYTES", 16)

    response = client.get(f"/api/transistors/{DEVICE_NAME}/dpt_data")

    assert record_path.stat().st_size > 16
    assert response.status_code == 200
    payload = response.json()
    assert "waveform_preview" not in payload
    assert payload["raw_measurement_data"][0]["dpt_on_vds"] == [
        [[0.0, 400.0], [1e-6, 210.0]],
        [[0.0, 390.0], [1e-6, 200.0]],
    ]


def test_bounded_dpt_preview_obeys_total_budget_and_preserves_source_json(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Repeated stored imports share one bounded, endpoint-preserving preview."""
    client, record_path, record = isolated_database
    fields = (
        "dpt_on_vds",
        "dpt_on_id",
        "dpt_off_vds",
        "dpt_off_id",
    )
    sample_count = 400
    raw_records = []
    for measurement_index in range(100):
        measurement = {
            "dataset_type": "dpt_u_i",
            "comment": f"measurement {measurement_index}",
        }
        for field in fields:
            measurement[field] = [[
                [sample_index * 1e-9, float(sample_index + measurement_index)]
                for sample_index in range(sample_count)
            ]]
        raw_records.append(measurement)
    record["raw_measurement_data"] = raw_records
    record_path.write_text(
        json.dumps(record, separators=(",", ":")), encoding="utf-8"
    )
    before = record_path.read_bytes()

    response = client.get(
        f"/api/transistors/{DEVICE_NAME}/dpt_data",
        params={"waveform_preview_points": 800},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["raw_measurement_data"]) == 100
    returned_points = 0
    for record_index, measurement in enumerate(payload["raw_measurement_data"]):
        for field in fields:
            acquisition = measurement[field][0]
            assert len(acquisition) == 100
            assert acquisition[0] == [0.0, float(record_index)]
            assert acquisition[-1] == [399e-9, float(399 + record_index)]
            returned_points += len(acquisition)
    assert returned_points == 40000
    assert payload["waveform_preview"] == {
        "max_points_per_acquisition": 800,
        "max_total_points": 40000,
        "returned_points": 40000,
        "sampling": "uniform_with_endpoints",
        "downsampled": True,
        "downsampled_acquisition_count": 400,
    }
    assert record_path.read_bytes() == before


def test_dpt_data_returns_empty_raw_list_without_changing_summary_fields(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Records without raw waveforms retain the old DPT summary contract."""
    client, record_path, record = isolated_database
    record["raw_measurement_data"] = []
    record_path.write_text(json.dumps(record), encoding="utf-8")

    response = client.get(f"/api/transistors/{DEVICE_NAME}/dpt_data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_measurement_data"] == []
    assert payload["has_e_on"] is True
    assert payload["has_e_off"] is True
    assert "has_e_rr" in payload
    assert "metadata" in payload


def test_dpt_data_normalizes_single_raw_measurement_dict_to_list(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Legacy singleton raw measurement records remain visible to the web API."""
    client, record_path, record = isolated_database
    singleton = record["raw_measurement_data"][0]
    record["raw_measurement_data"] = singleton
    record_path.write_text(json.dumps(record), encoding="utf-8")

    response = client.get(f"/api/transistors/{DEVICE_NAME}/dpt_data")

    assert response.status_code == 200
    raw_records = response.json()["raw_measurement_data"]
    assert len(raw_records) == 1
    assert raw_records[0]["metadata"]["comment"] == "first measurement"
    assert raw_records[0]["dpt_on_vds"][0][0] == [0.0, 400.0]


def test_raw_dpt_numpy_arrays_serialize_without_mutation() -> None:
    """NumPy samples are converted into JSON data without changing inputs."""
    vds = np.array([[0.0, 400.0], [1e-6, 210.0]])
    current = np.array([[0.0, 0.0], [1e-6, 20.0]])
    vds_before = vds.copy()
    current_before = current.copy()
    record = {
        "dataset_type": "dpt_u_i",
        "t_j": 25,
        "dpt_on_vds": [vds],
        "dpt_on_id": [current],
    }

    serialized = main._serialize_dpt_raw_measurement(record)

    assert serialized["dpt_on_vds"] == [vds_before.tolist()]
    assert serialized["dpt_on_id"] == [current_before.tolist()]
    assert serialized["dpt_off_vds"] == []
    assert serialized["dpt_off_id"] == []
    np.testing.assert_array_equal(vds, vds_before)
    np.testing.assert_array_equal(current, current_before)


def test_checked_in_infineon_non_finite_dpt_samples_are_json_safe_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the real checked-in -Infinity samples and source file unchanged."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "transistordatabase"
        / "examples"
        / "tdb_example"
        / "Infineon_IPW65R090CFD7.json"
    ).resolve()
    source_before = source_path.read_bytes()
    record_path = tmp_path / source_path.name
    shutil.copyfile(source_path, record_path)
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(tmp_path))

    response = TestClient(main.app).get(
        "/api/transistors/Infineon_IPW65R090CFD7/dpt_data"
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["raw_measurement_data"]) == 1
    off_current = payload["raw_measurement_data"][0]["dpt_off_id"]
    assert sum(sample[1] == "-Infinity" for acq in off_current for sample in acq) == 4
    json.dumps(payload, allow_nan=False)
    assert source_path.read_bytes() == source_before
    assert record_path.read_bytes() == source_before
