"""Raw-record API regressions for transient thermal Foster curves."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.gui_web.backend import main
from transistordatabase.gui_web.backend.schemas import ThermalCurveUpdate


DEVICE_NAME = "Thermal Curve Fixture"
GRAPH = [[0.0, 0.00001, 0.0001], [0.0, 0.12, 0.4]]
UPDATED_GRAPH = [[0.000002, 0.00002], [0.02, 0.2]]


class TemporaryRepository:
    """Load test-owned raw records through the production JSON reader."""

    def __init__(self, data_directory: Path) -> None:
        """Keep repository reads within the pytest temporary directory."""
        self.data_directory = data_directory

    def get_by_name(self, name: str) -> object | None:
        """Load one fixture record when the requested file exists."""
        path = self.data_directory / f"{name}.json"
        return JsonTransistorLoader().load_from_json(path) if path.exists() else None


@pytest.fixture
def thermal_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Configure FastAPI against a valid, isolated raw legacy device record."""
    fixture_path = Path(__file__).resolve().parent / "test_data" / "CREE_C3M0060065J.json"
    record = json.loads(fixture_path.read_text(encoding="utf-8"))
    record["name"] = DEVICE_NAME
    record["unknown_thermal_test_field"] = {"preserve": [1, 2, 3]}
    for component in ("switch", "diode"):
        record[component]["thermal_foster"]["graph_t_rthjc"] = None
        record[component]["thermal_foster"]["vendor_foster_key"] = f"{component}-keep"
    path = tmp_path / f"{DEVICE_NAME}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    repository = TemporaryRepository(tmp_path)
    monkeypatch.setattr(main, "_repo", repository)
    monkeypatch.setattr(main, "_pair_repo", object())
    return TestClient(main.app), path


@pytest.mark.parametrize("component", ["switch", "diode"])
def test_thermal_graph_crud_preserves_raw_foster_and_record_fields(thermal_client, component: str) -> None:
    """Round-trip exact graph points without replacing other stored fields."""
    client, record_path = thermal_client
    url = f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/thermal/{component}"
    other = "diode" if component == "switch" else "switch"

    loaded = client.get(f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves")
    assert loaded.status_code == 200, loaded.text
    thermal = loaded.json()["thermal_foster"][component]
    assert thermal["graph_t_rthjc"] is None
    assert len(thermal["_edit_token"]) == 64
    assert loaded.json()["thermal_foster"][other]["graph_t_rthjc"] is None

    added = client.post(url, json={"expected_token": thermal["_edit_token"], "graph_t_rthjc": GRAPH})
    assert added.status_code == 200, added.text
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored[component]["thermal_foster"]["graph_t_rthjc"] == GRAPH
    assert stored[component]["thermal_foster"]["vendor_foster_key"] == f"{component}-keep"
    assert stored[other]["thermal_foster"]["vendor_foster_key"] == f"{other}-keep"
    assert stored["unknown_thermal_test_field"] == {"preserve": [1, 2, 3]}

    current = client.get(f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves").json()
    thermal = current["thermal_foster"][component]
    assert thermal["graph_t_rthjc"] == GRAPH
    assert thermal["_edit_token"] != loaded.json()["thermal_foster"][component]["_edit_token"]

    stale_record = json.loads(record_path.read_text(encoding="utf-8"))
    stale_record[component]["thermal_foster"]["graph_t_rthjc"] = UPDATED_GRAPH
    record_path.write_text(json.dumps(stale_record), encoding="utf-8")
    stale = client.put(url, json={"expected_token": thermal["_edit_token"], "graph_t_rthjc": GRAPH})
    assert stale.status_code == 409
    assert json.loads(record_path.read_text(encoding="utf-8"))[component]["thermal_foster"]["graph_t_rthjc"] == UPDATED_GRAPH

    refreshed = client.get(f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves").json()["thermal_foster"][component]
    updated = client.put(url, json={"expected_token": refreshed["_edit_token"], "graph_t_rthjc": GRAPH})
    assert updated.status_code == 200, updated.text
    latest = client.get(f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves").json()["thermal_foster"][component]
    deleted = client.delete(url, params={"expected_token": latest["_edit_token"]})
    assert deleted.status_code == 200, deleted.text
    final = json.loads(record_path.read_text(encoding="utf-8"))
    assert final[component]["thermal_foster"]["graph_t_rthjc"] is None
    assert final[component]["thermal_foster"]["vendor_foster_key"] == f"{component}-keep"
    assert final["unknown_thermal_test_field"] == {"preserve": [1, 2, 3]}


@pytest.mark.parametrize("component", ["switch", "diode"])
@pytest.mark.parametrize(
    "invalid_graph",
    [
        [[1.0, 0.5], [0.1, 0.2]],
        [[1.0, 2.0], [-0.1, 0.2]],
        [[1.0, 2.0], [0.1]],
    ],
)
def test_thermal_graph_rejects_invalid_points_without_changing_record(
    thermal_client, component: str, invalid_graph: list[list[float]]
) -> None:
    """Reject malformed or physically invalid points before any file write."""
    client, record_path = thermal_client
    url = f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/thermal/{component}"
    token = client.get(f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves").json()["thermal_foster"][component]["_edit_token"]
    before = record_path.read_bytes()

    response = client.post(url, json={"expected_token": token, "graph_t_rthjc": invalid_graph})

    assert response.status_code == 422
    assert record_path.read_bytes() == before


def test_thermal_graph_schema_rejects_nonfinite_samples() -> None:
    """Reject NaN even when a caller bypasses strict JSON encoding."""
    with pytest.raises(ValidationError, match="finite"):
        ThermalCurveUpdate(
            expected_token="a" * 64,
            graph_t_rthjc=[[1.0, 2.0], [0.1, float("nan")]],
        )


@pytest.mark.parametrize("component", ["switch", "diode"])
@pytest.mark.parametrize("missing_foster", [True, False])
def test_missing_foster_reads_empty_and_add_initializes_only_that_entry(
    thermal_client, component: str, missing_foster: bool
) -> None:
    """Read absent/null Foster data safely, then initialize only the selected object."""
    client, record_path = thermal_client
    original = json.loads(record_path.read_text(encoding="utf-8"))
    if missing_foster:
        original[component].pop("thermal_foster")
    else:
        original[component]["thermal_foster"] = None
    record_path.write_text(json.dumps(original), encoding="utf-8")
    before_get = record_path.read_bytes()
    name = quote(DEVICE_NAME, safe="")

    loaded = client.get(f"/api/transistors/{name}/curves")

    assert loaded.status_code == 200, loaded.text
    thermal = loaded.json()["thermal_foster"][component]
    assert thermal["graph_t_rthjc"] is None
    assert len(thermal["_edit_token"]) == 64
    assert record_path.read_bytes() == before_get

    url = f"/api/transistors/{name}/curves/thermal/{component}"
    added = client.post(
        url,
        json={"expected_token": thermal["_edit_token"], "graph_t_rthjc": GRAPH},
    )

    assert added.status_code == 200, added.text
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored[component]["thermal_foster"]["graph_t_rthjc"] == GRAPH
    assert stored[component]["thermal_foster"]["r_th_total"] == 0
    assert stored["unknown_thermal_test_field"] == original["unknown_thermal_test_field"]
