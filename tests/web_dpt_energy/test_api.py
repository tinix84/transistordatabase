"""Focused API coverage for evidence-backed raw-window DPT energy curves."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.gui_web.backend import main


DEVICE_NAME = "DPT Energy Derivation Fixture"


class TemporaryJsonRepository:
    """Repository facade whose records stay inside pytest's temporary directory."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory

    def get_by_name(self, name: str) -> Any | None:
        """Load a fixture device from the test's private record directory."""
        path = self.data_directory / f"{name}.json"
        if not path.exists():
            return None
        return JsonTransistorLoader().load_from_json(path)


def _measurement(
    *,
    index: int,
    current_scale: float = 1.0,
    **overrides: Any,
) -> dict[str, Any]:
    times = [0.0, 1e-6, 2e-6]
    measurement = {
        "dataset_type": "dpt_u_i",
        "comment": f"Imported waveform {index}",
        "measurement_testbench": "raw DPT test bench",
        "measurement_date": "2026-10-01T00:00:00Z",
        "commutation_device": "DPT diode",
        "t_j": 25.0,
        "v_supply": 400.0,
        "v_g": 15.0,
        "v_g_off": -5.0,
        "r_g": 10.0,
        "r_g_off": 12.0,
        "load_inductance": 0.75,
        "commutation_inductance": 0.125,
        "import_provenance": {
            "parser": "PyLTSpice.RawRead",
            "source_filename": f"pulse-{index}.raw",
            "source_sha256": f"{index:064x}",
        },
        "dpt_on_vds": [[
            [times[0], 100.0], [times[1], 80.0], [times[2], 60.0],
        ]],
        "dpt_on_id": [[
            [times[0], 0.0],
            [times[1], 10.0 * current_scale],
            [times[2], 20.0 * current_scale],
        ]],
        "dpt_off_vds": [[
            [times[0], 50.0], [times[1], 100.0], [times[2], 150.0],
        ]],
        "dpt_off_id": [[
            [times[0], 20.0 * current_scale],
            [times[1], 15.0 * current_scale],
            [times[2], 10.0 * current_scale],
        ]],
    }
    measurement.update(overrides)
    return measurement


def _fixture_record() -> dict[str, Any]:
    source = Path(__file__).resolve().parents[2] / "tests" / "test_data" / "CREE_C3M0060065J.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    record["name"] = DEVICE_NAME
    record["switch"]["e_on"] = []
    record["switch"]["e_off"] = []
    record["raw_measurement_data"] = [
        _measurement(index=0),
        _measurement(index=1, current_scale=2.0),
        _measurement(index=2, current_scale=3.0),
        _measurement(index=3, current_scale=4.0),
    ]
    return record


@pytest.fixture
def isolated_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path, dict[str, Any]]:
    """Create an API client backed by an isolated legacy JSON record."""
    record = _fixture_record()
    record_path = tmp_path / f"{DEVICE_NAME}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(tmp_path))
    return TestClient(main.app), record_path, record


def _request(
    loss_type: str = "e_on",
    *,
    record_indices: tuple[int, int] = (0, 1),
    current_values: tuple[float, float] = (10.0, 20.0),
    vds_sign: int = 1,
    id_sign: int = 1,
) -> dict[str, Any]:
    return {
        "loss_type": loss_type,
        "points": [
            {
                "measurement_index": record_indices[0],
                "acquisition_index": 0,
                "i_x": current_values[0],
                "vds_sign": vds_sign,
                "id_sign": id_sign,
            },
            {
                "measurement_index": record_indices[1],
                "acquisition_index": 0,
                "i_x": current_values[1],
                "vds_sign": vds_sign,
                "id_sign": id_sign,
            },
        ],
    }


def _preview(client: TestClient, request: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=request,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _save_payload(
    request: dict[str, Any], preview: dict[str, Any]
) -> dict[str, Any]:
    """Bind save inputs to the exact reviewed source and target snapshot."""
    return {
        **request,
        "source_token": preview["source_token"],
        "request_token": preview["request_token"],
        "save_action": preview["save_action"],
        "existing_curve_index": preview["existing_curve_index"],
        "target_curve_count": preview["target_curve_count"],
        "target_edit_token": preview["target_edit_token"],
    }


def test_preview_uses_exact_signed_samples_and_discloses_raw_interval(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Preview integrates only exact source samples and returns its full interval."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()

    response = _preview(client, _request())

    assert response["event"] == "turn_on"
    assert response["graph_i_e"] == [[10.0, 20.0], [0.0014, 0.0028]]
    assert response["points"][0] == {
        "measurement_index": 0,
        "acquisition_index": 0,
        "event": "turn_on",
        "i_x": 10.0,
        "energy_j": 0.0014,
        "start_s": 0.0,
        "end_s": 2e-6,
        "sample_count": 3,
        "vds_sign": 1,
        "id_sign": 1,
    }
    assert response["integration"]["interval"] == "exact stored first-to-last event sample time"
    assert response["integration"]["boundary_detection"].startswith("none;")
    assert response["conditions"]["load_inductance"] == pytest.approx(0.75e-6)
    assert response["conditions"]["commutation_inductance"] == pytest.approx(0.125e-6)
    assert response["save_action"] == "add"
    assert len(response["source_token"]) == 64
    assert len(response["request_token"]) == 64
    assert record_path.read_bytes() == original_bytes


def test_polarity_controls_are_explicit_and_signed_energy_is_not_absoluted(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Polarity multipliers affect signed power without forcing positive energy."""
    client, _record_path, _record = isolated_database
    both_reversed = _preview(
        client,
        _request(vds_sign=-1, id_sign=-1),
    )
    assert both_reversed["graph_i_e"] == [[10.0, 20.0], [0.0014, 0.0028]]
    assert both_reversed["points"][0]["vds_sign"] == -1
    assert both_reversed["points"][0]["id_sign"] == -1

    negative = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(vds_sign=-1),
    )
    assert negative.status_code == 400
    assert "nonpositive signed raw-window energy" in negative.json()["detail"]
    assert "never made positive with an absolute value" in negative.json()["detail"]


def test_save_adds_graph_curve_and_preserves_every_raw_measurement(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Curve persistence stores supported fields and preserves all raw evidence."""
    client, record_path, original_record = isolated_database
    original_raw = copy.deepcopy(original_record["raw_measurement_data"])
    reviewed_request = _request()
    preview = _preview(client, reviewed_request)

    saved = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(reviewed_request, preview),
    )

    assert saved.status_code == 200, saved.text
    payload = saved.json()
    assert payload["action"] == "added"
    assert payload["loss_type"] == "e_on"
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored["raw_measurement_data"] == original_raw
    energy_curve = stored["switch"]["e_on"][-1]
    assert energy_curve["dataset_type"] == "graph_i_e"
    assert energy_curve["graph_i_e"] == [[10.0, 20.0], [0.0014, 0.0028]]
    assert energy_curve["v_g"] == 15
    assert energy_curve["v_g_off"] == -5
    assert energy_curve["r_g"] == 10
    assert energy_curve["load_inductance"] == pytest.approx(0.75e-6)
    assert energy_curve["commutation_inductance"] == pytest.approx(0.125e-6)
    assert "Vds sign +1, Id sign +1" in energy_curve["comment"]
    assert "not standards-normalized" in energy_curve["comment"]
    assert "DPT-ENERGY-PROVENANCE:" in energy_curve["comment"]


def test_same_context_curve_is_previewed_and_extended_safely(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """A provenance-verified same-context energy graph accepts new current points."""
    client, record_path, original_record = isolated_database
    raw_before = copy.deepcopy(original_record["raw_measurement_data"])
    first_request = _request(record_indices=(0, 1), current_values=(10.0, 20.0))
    first_preview = _preview(client, first_request)
    first_save = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(first_request, first_preview),
    )
    assert first_save.status_code == 200, first_save.text

    extension_request = _request(record_indices=(2, 3), current_values=(30.0, 40.0))
    extension_preview = _preview(client, extension_request)
    assert extension_preview["save_action"] == "extend"
    assert extension_preview["existing_curve_index"] == 0
    extension_save = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(extension_request, extension_preview),
    )
    assert extension_save.status_code == 200, extension_save.text
    assert extension_save.json()["action"] == "extended"

    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored["raw_measurement_data"] == raw_before
    curves = stored["switch"]["e_on"]
    assert len(curves) == 1
    assert curves[0]["graph_i_e"][0] == [10.0, 20.0, 30.0, 40.0]
    assert curves[0]["graph_i_e"][1] == pytest.approx([0.0014, 0.0028, 0.0042, 0.0056])
    assert curves[0]["comment"].count("DPT-ENERGY-PROVENANCE:") == 2


def _save_reviewed_request(client: TestClient, request: dict[str, Any]) -> dict[str, Any]:
    """Save one fixture request using the exact target returned from preview."""
    preview = _preview(client, request)
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(request, preview),
    )
    assert response.status_code == 200, response.text
    return preview


def test_preview_rejects_reusing_a_source_acquisition_at_a_different_current(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """One stored acquisition cannot create two current points in one curve."""
    client, _record_path, _record = isolated_database
    _save_reviewed_request(
        client,
        _request(record_indices=(0, 1), current_values=(10.0, 20.0)),
    )

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(record_indices=(1, 2), current_values=(30.0, 40.0)),
    )

    assert response.status_code == 409
    assert "already represented in the matching curve" in response.json()["detail"]


def test_preview_rejects_extension_over_the_fifty_window_curve_limit(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Extension checks the point limit across the full merged graph."""
    client, record_path, _record = isolated_database
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["raw_measurement_data"].extend(
        _measurement(index=index, current_scale=float(index + 1))
        for index in range(4, 51)
    )
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    initial_request = {
        "loss_type": "e_on",
        "points": [
            {
                "measurement_index": index,
                "acquisition_index": 0,
                "i_x": float(index + 1),
                "vds_sign": 1,
                "id_sign": 1,
            }
            for index in range(49)
        ],
    }
    _save_reviewed_request(client, initial_request)
    extension_request = {
        "loss_type": "e_on",
        "points": [
            {
                "measurement_index": index,
                "acquisition_index": 0,
                "i_x": float(index + 1),
                "vds_sign": 1,
                "id_sign": 1,
            }
            for index in (49, 50)
        ],
    }

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=extension_request,
    )

    assert response.status_code == 409
    assert "exceed the 50-window maximum" in response.json()["detail"]


def test_save_rejects_target_curve_edit_after_preview(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """A target graph edit after preview invalidates the reviewed target token."""
    client, record_path, _record = isolated_database
    _save_reviewed_request(client, _request())
    extension_request = _request(record_indices=(2, 3), current_values=(30.0, 40.0))
    extension_preview = _preview(client, extension_request)
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["switch"]["e_on"][0]["comment"] = (
        "Edited human note\n" + stored["switch"]["e_on"][0]["comment"]
    )
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(extension_request, extension_preview),
    )

    assert response.status_code == 409
    assert "target changed after preview" in response.json()["detail"]
    stored_after = json.loads(record_path.read_text(encoding="utf-8"))
    assert "Edited human note" in stored_after["switch"]["e_on"][0]["comment"]


def test_save_rejects_add_target_that_became_an_extend_target(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """A curve created after preview cannot silently become an extension target."""
    client, record_path, _record = isolated_database
    stale_request = _request(record_indices=(0, 1), current_values=(10.0, 20.0))
    stale_preview = _preview(client, stale_request)
    assert stale_preview["save_action"] == "add"
    _save_reviewed_request(
        client,
        _request(record_indices=(2, 3), current_values=(30.0, 40.0)),
    )

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(stale_request, stale_preview),
    )

    assert response.status_code == 409
    assert "target changed after preview" in response.json()["detail"]
    stored_after = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored_after["switch"]["e_on"][0]["graph_i_e"][0] == [30.0, 40.0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commutation_device", "edited diode"),
        ("load_inductance", 1.0),
        ("load_inductance", 0.75e-6 + 0.9e-9),
    ],
)
def test_preview_rejects_saved_curve_conditions_edited_after_provenance(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    field: str,
    value: Any,
) -> None:
    """The live SwitchingLossData conditions must still match its DPT marker."""
    client, record_path, _record = isolated_database
    _save_reviewed_request(client, _request())
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["switch"]["e_on"][0][field] = value
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(record_indices=(2, 3), current_values=(30.0, 40.0)),
    )

    assert response.status_code == 409
    assert "saved condition fields diverge" in response.json()["detail"]


def test_preview_rejects_a_single_stored_sample_pair_without_mutation(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """A one-sample event cannot define a trapezoidal integration interval."""
    client, record_path, _record = isolated_database
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["raw_measurement_data"][0]["dpt_on_vds"] = [[[0.0, 100.0]]]
    stored["raw_measurement_data"][0]["dpt_on_id"] = [[[0.0, 10.0]]]
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    before_preview = record_path.read_bytes()

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(),
    )

    assert response.status_code == 400
    assert "at least two stored event samples" in response.json()["detail"]
    assert record_path.read_bytes() == before_preview


def test_preview_rejects_multiple_tdb_curve_matches_without_mutation(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Two topology-lookup candidates remain ambiguous for extension."""
    client, record_path, _record = isolated_database
    _save_reviewed_request(client, _request())
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["switch"]["e_on"].append(copy.deepcopy(stored["switch"]["e_on"][0]))
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    before_preview = record_path.read_bytes()

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(record_indices=(2, 3), current_values=(30.0, 40.0)),
    )

    assert response.status_code == 409
    assert "Multiple existing e_on graph_i_e curves match" in response.json()["detail"]
    assert record_path.read_bytes() == before_preview


def test_preview_rejects_saved_graph_edit_even_when_provenance_is_unchanged(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """A retained marker cannot authorize edited persisted energy samples."""
    client, record_path, _record = isolated_database
    _save_reviewed_request(client, _request())
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["switch"]["e_on"][0]["graph_i_e"][1][0] = 0.0015
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(record_indices=(2, 3), current_values=(30.0, 40.0)),
    )

    assert response.status_code == 409
    assert "graph samples do not exactly match" in response.json()["detail"]


@pytest.mark.parametrize(
    ("current", "energy"),
    [(0.0, 0.0014), (10.0, 0.0)],
)
def test_preview_rejects_nonpositive_existing_graph_samples(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    current: float,
    energy: float,
) -> None:
    """Zero graph axes or energies cannot be extended as switching loss data."""
    client, record_path, _record = isolated_database
    _save_reviewed_request(client, _request())
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["switch"]["e_on"][0]["graph_i_e"][0][0] = current
    stored["switch"]["e_on"][0]["graph_i_e"][1][0] = energy
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(record_indices=(2, 3), current_values=(30.0, 40.0)),
    )

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("record_changes", "request_changes", "message"),
    [
        ({"t_j": 30.0}, {}, "source data changed after preview"),
        ({}, {"points": [{"measurement_index": 0, "acquisition_index": 0, "i_x": 11.0, "vds_sign": 1, "id_sign": 1}, {"measurement_index": 1, "acquisition_index": 0, "i_x": 20.0, "vds_sign": 1, "id_sign": 1}]}, "inputs changed after preview"),
        (
            {},
            {"points": [{"measurement_index": 99, "acquisition_index": 0, "i_x": 11.0, "vds_sign": 1, "id_sign": 1}, {"measurement_index": 1, "acquisition_index": 0, "i_x": 20.0, "vds_sign": 1, "id_sign": 1}]},
            "inputs changed after preview",
        ),
    ],
)
def test_save_rejects_stale_source_or_reviewed_request(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    record_changes: dict[str, Any],
    request_changes: dict[str, Any],
    message: str,
) -> None:
    """Save rejects changed conditions, sample sources, and reviewed inputs."""
    client, record_path, _record = isolated_database
    reviewed_request = _request()
    preview = _preview(client, reviewed_request)
    if record_changes:
        stored = json.loads(record_path.read_text(encoding="utf-8"))
        stored["raw_measurement_data"][0].update(record_changes)
        record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    save_request = {**reviewed_request, **request_changes}

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(save_request, preview),
    )

    assert response.status_code == 409
    assert message in response.json()["detail"]
    stored_after = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored_after["switch"]["e_on"] == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"t_j": None}, "missing required t_j"),
        ({"v_g_off": None}, "missing required v_g_off"),
        ({"r_g": None}, "missing r_g"),
        ({"r_g": [10.0, 20.0]}, "ambiguous r_g"),
        ({"import_provenance": None}, "verified PyLTSpice.RawRead importer provenance"),
    ],
)
def test_preview_rejects_missing_ambiguous_conditions_or_unknown_units(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    changes: dict[str, Any],
    message: str,
) -> None:
    """Incomplete conditions and unverified inductance units stop derivation."""
    client, record_path, _record = isolated_database
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["raw_measurement_data"][0].update(changes)
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(),
    )

    assert response.status_code == 400
    assert message in response.json()["detail"]


@pytest.mark.parametrize(
    ("record_changes", "request_changes", "message"),
    [
        ({"dpt_on_id": [[[0.0, 0.0], [1e-6, 10.0]]]}, {}, "misaligned"),
        (
            {
                "dpt_on_vds": [[[0.0, 100.0], [2e-6, 80.0], [1e-6, 60.0]]],
                "dpt_on_id": [[[0.0, 0.0], [2e-6, 10.0], [1e-6, 20.0]]],
            },
            {},
            "nonmonotonic",
        ),
        ({"dpt_on_vds": [[[0.0, 100.0], [1e-6, float("nan")], [2e-6, 60.0]]]}, {}, "non-finite"),
        ({}, {"current_values": (10.0, 10.0)}, "distinct i_x"),
    ],
)
def test_preview_rejects_malformed_event_samples_and_current_axis(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    record_changes: dict[str, Any],
    request_changes: dict[str, Any],
    message: str,
) -> None:
    """Invalid sample alignment, time order, finiteness, or current values fail."""
    client, record_path, _record = isolated_database
    if record_changes:
        stored = json.loads(record_path.read_text(encoding="utf-8"))
        stored["raw_measurement_data"][0].update(record_changes)
        record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(**request_changes),
    )

    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_preview_rejects_incompatible_commutation_context_and_preserves_source(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Different commutation devices cannot be combined in one energy curve."""
    client, record_path, _record = isolated_database
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    stored["raw_measurement_data"][1]["commutation_device"] = "Different diode"
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    original_bytes = record_path.read_bytes()

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(),
    )

    assert response.status_code == 400
    assert "incompatible test conditions" in response.json()["detail"]
    assert "commutation_device" in response.json()["detail"]
    assert record_path.read_bytes() == original_bytes


def test_eoff_uses_explicit_turn_off_gate_and_side_specific_resistance(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """Eoff persists its modern gate pair with the turn-off gate resistance."""
    client, record_path, _record = isolated_database
    original_raw = json.loads(record_path.read_text(encoding="utf-8"))["raw_measurement_data"]
    reviewed_request = _request("e_off")
    preview = _preview(client, reviewed_request)
    assert preview["event"] == "turn_off"
    assert preview["conditions"]["v_g"] == 15.0
    assert preview["conditions"]["v_g_off"] == -5.0
    assert preview["conditions"]["r_g"] == 12.0

    saved = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/save",
        json=_save_payload(reviewed_request, preview),
    )

    assert saved.status_code == 200, saved.text
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored["switch"]["e_off"][0]["r_g"] == 12.0
    assert stored["switch"]["e_off"][0]["v_g"] == 15.0
    assert stored["switch"]["e_off"][0]["v_g_off"] == -5.0
    assert stored["raw_measurement_data"] == original_raw


def test_legacy_matching_curve_without_dpt_context_is_rejected_as_ambiguous(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    """An unproven legacy graph is never merged with a DPT-derived curve."""
    client, record_path, _record = isolated_database
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    legacy_curve = {
        "dataset_type": "graph_i_e",
        "t_j": 25.0,
        "v_supply": 400.0,
        "v_g": 15.0,
        "v_g_off": -5.0,
        "r_g": 10.0,
        "graph_i_e": [[1.0, 2.0], [0.001, 0.002]],
    }
    stored["switch"]["e_on"] = [legacy_curve]
    record_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_energy/preview",
        json=_request(),
    )

    assert response.status_code == 409
    assert "commutation/test-bench context cannot be verified" in response.json()["detail"]
