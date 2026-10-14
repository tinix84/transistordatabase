"""Focused API coverage for additive native LTspice DPT RAW imports."""

from __future__ import annotations

import copy
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import PyLTSpice
import starlette.requests
from fastapi.testclient import TestClient
from PyLTSpice import RawWrite, Trace

from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.gui_web.backend import dpt_raw_import, main


DEVICE_NAME = "DPT RAW Import Fixture"


class TemporaryJsonRepository:
    """Repository facade whose records live only under pytest's temp path."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory

    def get_by_name(self, name: str) -> Any | None:
        """Load a fixture record from the temporary database directory."""
        path = self.data_directory / f"{name}.json"
        if not path.exists():
            return None
        return JsonTransistorLoader().load_from_json(path)


def _raw_bytes(
    tmp_path: Path,
    *,
    plot_name: str = "Transient Analysis",
    sample_count: int = 7,
    time_values: list[float] | None = None,
    vds_values: list[float] | None = None,
    id_values: list[float] | None = None,
) -> bytes:
    """Create a tiny native LTspice RAW file with reviewable traces."""
    path = tmp_path / "test.raw"
    time_values = (
        np.asarray(time_values, dtype=np.float64)
        if time_values is not None
        else np.arange(sample_count, dtype=np.float64) * 1e-6
    )
    sample_count = len(time_values)
    vds_values = (
        np.asarray(vds_values, dtype=np.float64)
        if vds_values is not None
        else 400.0 - time_values * 1e7
    )
    id_values = (
        np.asarray(id_values, dtype=np.float64)
        if id_values is not None
        else time_values * 1e7
    )
    writer = RawWrite(plot_name=plot_name, fastacces=False)
    writer.add_trace(Trace("time", time_values, whattype="time"))
    writer.add_trace(Trace("V(ds)", vds_values))
    writer.add_trace(Trace("I(M1)", id_values, whattype="current"))
    writer.save(str(path))
    return path.read_bytes()


def _mark_raw_plot_stepped(content: bytes) -> bytes:
    """Add LTspice's stepped flag while preserving the binary payload."""
    encoding = "utf_16_le" if content[1] == 0 else "utf-8"
    marker = "Binary:".encode(encoding)
    marker_index = content.index(marker)
    header = content[:marker_index].decode(encoding)
    assert "Flags: real\n" in header
    header = header.replace("Flags: real\n", "Flags: real stepped\n", 1)
    return header.encode(encoding) + content[marker_index:]


def _stepped_raw_bytes(
    tmp_path: Path,
    *,
    time_values: list[float] | None = None,
    vds_values: list[float] | None = None,
    id_values: list[float] | None = None,
) -> bytes:
    times = time_values or [0, 1, 2, 0, 1, 2, 3, 4]
    return _mark_raw_plot_stepped(
        _raw_bytes(
            tmp_path,
            time_values=[value * 1e-6 for value in times],
            vds_values=vds_values,
            id_values=id_values,
        )
    )


def _complex_raw_bytes(tmp_path: Path) -> bytes:
    """Create an AC plot with two complex128 vectors before a stepped plot."""
    path = tmp_path / "complex.raw"
    writer = RawWrite(plot_name="AC Analysis", fastacces=False, numtype="complex")
    writer.add_trace(
        Trace(
            "frequency",
            np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128),
            whattype="frequency",
            numerical_type="complex",
        )
    )
    writer.add_trace(
        Trace(
            "V(out)",
            np.asarray([5 + 6j, 7 + 8j], dtype=np.complex128),
            numerical_type="complex",
        )
    )
    writer.save(str(path))
    return path.read_bytes()


def _ascii_raw_bytes(tmp_path: Path) -> bytes:
    """Create a small native LTspice Values-section transient fixture."""
    binary = _raw_bytes(tmp_path, sample_count=7)
    encoding = "utf_16_le" if binary[1] == 0 else "utf-8"
    marker = "Binary:".encode(encoding)
    marker_index = binary.index(marker)
    header = binary[:marker_index].decode(encoding)
    newline = "\n".encode(encoding)
    data_start = binary.index(newline, marker_index) + len(newline)
    values = []
    for point_index in range(7):
        offset = data_start + point_index * 16
        time_value, vds_value, id_value = struct.unpack_from("<dff", binary, offset)
        values.extend(
            [
                f"{point_index}\t{time_value:.16e}",
                f"{vds_value:.9e}",
                f"{id_value:.9e}",
            ]
        )
    return (
        header.replace("No. Points:", "No. Points:", 1).encode(encoding)
        + "Values:\n".encode(encoding)
        + "\n".join(values).encode(encoding)
        + "\n".encode(encoding)
    )


def _stepped_ascii_raw_bytes(tmp_path: Path) -> bytes:
    """Create a stepped ASCII fixture whose later step starts at point index 3."""
    binary = _stepped_raw_bytes(tmp_path)
    encoding = "utf_16_le" if binary[1] == 0 else "utf-8"
    marker = "Binary:".encode(encoding)
    marker_index = binary.index(marker)
    header = binary[:marker_index].decode(encoding)
    newline = "\n".encode(encoding)
    data_start = binary.index(newline, marker_index) + len(newline)
    values = []
    for point_index in range(8):
        offset = data_start + point_index * 16
        time_value, vds_value, id_value = struct.unpack_from("<dff", binary, offset)
        values.extend(
            [
                f"{point_index}\t{time_value:.16e}",
                f"{vds_value:.9e}",
                f"{id_value:.9e}",
            ]
        )
    return (
        header.encode(encoding)
        + "Values:\n".encode(encoding)
        + "\n".join(values).encode(encoding)
        + "\n".encode(encoding)
    )


def _common_step_metadata() -> dict[str, Any]:
    return {
        "t_j": 25,
        "v_supply": 400,
        "v_g": 15,
        "v_g_off": -5,
        "r_g": 10,
        "r_g_off": 12,
        "load_inductance": 0.75,
        "commutation_inductance": 0.125,
        "measurement_testbench": "LTspice local test bench",
        "commutation_device": "fixture diode",
    }


def _one_step_import_selection() -> dict[str, Any]:
    return {
        **_trace_selection(),
        "plot_index": 0,
        "step_indices": [0],
        "step_selections": [
            {
                "step_index": 0,
                "i_x": 10,
                "turn_on": {"start_s": 0, "end_s": 1e-6},
                "turn_off": {"start_s": 2e-6, "end_s": 2e-6},
            }
        ],
        "same_context_confirmed": True,
        "metadata": _common_step_metadata(),
    }


@pytest.fixture
def isolated_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path, dict[str, Any]]:
    """Expose one copied transistor record through the FastAPI repository."""
    source = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "test_data"
        / "CREE_C3M0060065J.json"
    )
    record = json.loads(source.read_text(encoding="utf-8"))
    record["name"] = DEVICE_NAME
    record_path = tmp_path / f"{DEVICE_NAME}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(tmp_path))
    return TestClient(main.app), record_path, record


def _upload_file(content: bytes, filename: str = "measurement.raw") -> dict[str, Any]:
    return {"file": (filename, content, "application/octet-stream")}


def _trace_selection() -> dict[str, str]:
    return {"time_trace": "time", "vds_trace": "V(ds)", "id_trace": "I(M1)"}


def _import_selection() -> dict[str, Any]:
    return {
        **_trace_selection(),
        "turn_on": {"start_s": 1e-6, "end_s": 3e-6},
        "turn_off": {"start_s": 4e-6, "end_s": 6e-6},
        "metadata": {
            "t_j": 25,
            "v_supply": 400,
            "load_inductance": 0.75,
            "commutation_inductance": 0.125,
            "measurement_testbench": "LTspice local test bench",
            "comment": "Reviewed event boundaries",
        },
    }


def _attach_review_token(
    client: TestClient,
    content: bytes,
    selection: dict[str, Any],
) -> None:
    """Obtain the one-time server review token for a complete import selection."""
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/trace_preview",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selection)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["review_required"] is True
    assert response.json()["review_token"]
    selection["review_token"] = response.json()["review_token"]


def _assert_sample_pairs(
    actual: list[list[float]],
    expected: list[list[float]],
) -> None:
    assert len(actual) == len(expected)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-12,
        atol=1e-15,
    )


def test_preview_lists_traces_for_supported_ltspice_transient(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Show parser metadata without choosing or persisting a waveform."""
    client, _record_path, _record = isolated_database

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(_raw_bytes(tmp_path)),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "measurement.raw"
    assert payload["plot_name"] == "Transient Analysis"
    assert payload["sample_count"] == 7
    assert payload["time_trace_name"] == "time"
    assert payload["trace_names"] == ["time", "V(ds)", "I(M1)"]
    assert payload["plots"][0]["steps"][0]["sample_count"] == 7


def test_preview_and_trace_selection_handle_mixed_multi_plot_steps(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Inspect one selected transient among complex, stepped and other plots."""
    client, _record_path, _record = isolated_database
    content = b"".join(
        [
            _complex_raw_bytes(tmp_path),
            _stepped_raw_bytes(tmp_path),
            _raw_bytes(tmp_path, plot_name="Operating Point", sample_count=4),
        ]
    )

    header = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(content),
    )

    assert header.status_code == 200, header.text
    plots = header.json()["plots"]
    assert [item["plot_name"] for item in plots] == [
        "AC Analysis",
        "Transient Analysis",
        "Operating Point",
    ]
    assert plots[1]["is_stepped"] is True
    assert [item["sample_count"] for item in plots[1]["steps"]] == [3, 5]
    assert [item["time_range_s"] for item in plots[1]["steps"]] == [
        [0.0, 2e-6],
        [0.0, 4e-6],
    ]
    assert "step_values" not in header.text.casefold()

    selection = {
        **_trace_selection(),
        "plot_index": 1,
        "step_indices": [1],
    }
    traces = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/trace_preview",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selection)},
    )
    assert traces.status_code == 200, traces.text
    payload = traces.json()
    assert payload["plot_index"] == 1
    assert payload["step_indices"] == [1]
    assert [item["step_index"] for item in payload["steps"]] == [1]
    assert payload["steps"][0]["sample_count"] == 5
    assert len(payload["steps"][0]["traces"]["V(ds)"]) == 5


def test_stepped_import_appends_selected_steps_and_preserves_existing_raw_records(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Append only selected step windows with explicit shared metadata/current."""
    client, record_path, record = isolated_database
    record["raw_measurement_data"] = [{"comment": "preserve this record"}]
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    before = copy.deepcopy(record["raw_measurement_data"])
    selections = {
        **_trace_selection(),
        "plot_index": 0,
        "step_indices": [1, 0],
        "step_selections": [
            {
                "step_index": 1,
                "i_x": 22,
                "turn_on": {"start_s": 1e-6, "end_s": 2e-6},
                "turn_off": {"start_s": 3e-6, "end_s": 4e-6},
            },
            {
                "step_index": 0,
                "i_x": 10,
                "turn_on": {"start_s": 0, "end_s": 1e-6},
                "turn_off": {"start_s": 1e-6, "end_s": 2e-6},
            },
        ],
        "same_context_confirmed": True,
        "metadata": _common_step_metadata(),
    }
    content = _stepped_raw_bytes(tmp_path)
    _attach_review_token(client, content, selections)

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selections)},
    )

    assert response.status_code == 200, response.text
    assert response.json()["acquisition_count"] == 2
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert stored["raw_measurement_data"][0] == before[0]
    imported = stored["raw_measurement_data"][1]
    assert [len(item) for item in imported["dpt_on_vds"]] == [2, 2]
    provenance = imported["import_provenance"]
    assert provenance["source_plot_index"] == 0
    assert provenance["source_step_indices"] == [1, 0]
    assert provenance["current_axis_i_x_by_source_step_a"] == {"1": 22, "0": 10}
    assert provenance["step_parameters_imported"] is False
    assert provenance["conditions_source"] == "explicit user-entered common metadata"
    assert provenance["event_windows_by_source_step_s"]["1"]["turn_off"] == [
        3e-6,
        4e-6,
    ]


def test_multi_plot_unstepped_import_requires_reviewed_step_zero_context(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Require explicit current and common conditions for a selected multi-plot run."""
    client, record_path, _record = isolated_database
    before = record_path.read_bytes()
    content = _complex_raw_bytes(tmp_path) + _raw_bytes(tmp_path)
    selection = _one_step_import_selection()
    selection["plot_index"] = 1
    _attach_review_token(client, content, selection)

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selection)},
    )

    assert response.status_code == 200, response.text
    imported = json.loads(record_path.read_text(encoding="utf-8"))["raw_measurement_data"][0]
    assert imported["import_provenance"]["source_plot_index"] == 1
    assert imported["import_provenance"]["source_step_indices"] == [0]
    assert imported["import_provenance"]["current_axis_i_x_by_source_step_a"] == {"0": 10}
    assert imported["v_supply"] == 400
    assert record_path.read_bytes() != before


def test_review_token_rejects_missing_or_tampered_import_without_mutation(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Bind confirmation to exact file bytes, windows, current, and conditions."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    content = _stepped_raw_bytes(tmp_path)
    base = _one_step_import_selection()

    missing_token_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(base)},
    )
    assert missing_token_response.status_code == 409
    assert "Review this exact RAW file" in missing_token_response.json()["detail"]
    assert record_path.read_bytes() == original_bytes

    changed_content = _stepped_raw_bytes(
        tmp_path,
        id_values=[0.0, 3.0, 6.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    )
    for kind, submitted_content in [
        ("file", changed_content),
        ("window", content),
        ("current", content),
        ("conditions", content),
    ]:
        reviewed = copy.deepcopy(base)
        _attach_review_token(client, content, reviewed)
        tampered = copy.deepcopy(reviewed)
        if kind == "window":
            tampered["step_selections"][0]["turn_on"]["end_s"] = 2e-6
        elif kind == "current":
            tampered["step_selections"][0]["i_x"] = 11
        elif kind == "conditions":
            tampered["metadata"]["v_supply"] = 401
        response = client.post(
            f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
            files=_upload_file(submitted_content),
            data={"selection_json": json.dumps(tampered)},
        )
        assert response.status_code == 409, response.text
        assert "no longer matches" in response.json()["detail"]
        assert record_path.read_bytes() == original_bytes


def test_review_token_is_one_time_and_replay_does_not_mutate_record(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Reject replay after one successful append without changing stored data."""
    client, record_path, _record = isolated_database
    content = _stepped_raw_bytes(tmp_path)
    selection = _one_step_import_selection()
    _attach_review_token(client, content, selection)

    first = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selection)},
    )
    assert first.status_code == 200, first.text
    after_success = record_path.read_bytes()
    replay = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selection)},
    )

    assert replay.status_code == 409
    assert "already used" in replay.json()["detail"]
    assert record_path.read_bytes() == after_success


def test_review_token_expiry_and_store_capacity_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep server review state finite and reject expired tokens."""
    now = [100.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(main, "DPT_RAW_REVIEW_TOKEN_TTL_SECONDS", 5)
    expired_token = main._issue_dpt_raw_review_token("expired-signature")
    now[0] += 6
    with pytest.raises(main.HTTPException) as expired:
        main._consume_dpt_raw_review_token(expired_token, "expired-signature")
    assert expired.value.status_code == 409
    assert "expired" in expired.value.detail

    monkeypatch.setattr(main, "MAX_DPT_RAW_REVIEW_TOKENS", 1)
    live_token = main._issue_dpt_raw_review_token("live-signature")
    with pytest.raises(main.HTTPException) as full:
        main._issue_dpt_raw_review_token("another-signature")
    assert full.value.status_code == 503
    main._consume_dpt_raw_review_token(live_token, "live-signature")


def test_stepped_import_rejects_missing_plot_step_or_context_without_mutation(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Reject stale indexes, repeated step selection and missing conditions."""
    client, record_path, _record = isolated_database
    before = record_path.read_bytes()
    content = _stepped_raw_bytes(tmp_path)
    base = {
        **_trace_selection(),
        "plot_index": 0,
        "step_indices": [1],
        "step_selections": [
            {
                "step_index": 1,
                "i_x": 10,
                "turn_on": {"start_s": 0, "end_s": 1e-6},
                "turn_off": {"start_s": 1e-6, "end_s": 2e-6},
            }
        ],
        "same_context_confirmed": True,
        "metadata": _common_step_metadata(),
    }
    invalid_step = copy.deepcopy(base)
    invalid_step["step_selections"][0]["step_index"] = 10
    invalid_step["step_indices"] = [10]
    repeated_step = copy.deepcopy(base)
    repeated_step["step_indices"] = [1, 1]
    repeated_step["step_selections"] = [
        copy.deepcopy(base["step_selections"][0]),
        copy.deepcopy(base["step_selections"][0]),
    ]
    mismatched_step_indexes = copy.deepcopy(base)
    mismatched_step_indexes["step_indices"] = [0]
    stale_plot = {**base, "plot_index": 4}
    missing_context = {
        **base,
        "metadata": {},
    }
    no_plot_selection = copy.deepcopy(base)
    del no_plot_selection["plot_index"]
    no_plot_selection["step_selections"][0]["step_index"] = 0
    no_plot_selection["step_indices"] = [0]
    multi_plot_content = _complex_raw_bytes(tmp_path) + content
    unsupported_plot = {
        **copy.deepcopy(base),
        "plot_index": 0,
    }

    invalid_step_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(invalid_step)},
    )
    mismatched_indexes_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(mismatched_step_indexes)},
    )
    repeated_step_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(repeated_step)},
    )
    stale_plot_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(stale_plot)},
    )
    missing_context_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(missing_context)},
    )
    no_plot_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(multi_plot_content),
        data={"selection_json": json.dumps(no_plot_selection)},
    )
    unsupported_plot_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(multi_plot_content),
        data={"selection_json": json.dumps(unsupported_plot)},
    )

    assert invalid_step_response.status_code == 400
    assert "Source step 10" in invalid_step_response.json()["detail"]
    assert mismatched_indexes_response.status_code == 422
    assert "exactly match step_selections" in mismatched_indexes_response.json()["detail"]
    assert repeated_step_response.status_code == 422
    assert "step_indices must be unique" in repeated_step_response.json()["detail"]
    assert stale_plot_response.status_code == 400, stale_plot_response.text
    assert "not a transient analysis" in stale_plot_response.json()["detail"]
    assert missing_context_response.status_code == 400
    assert "explicit common values" in missing_context_response.json()["detail"]
    assert no_plot_response.status_code == 400
    assert "Select a transient plot" in no_plot_response.json()["detail"]
    assert unsupported_plot_response.status_code == 400
    assert "not a transient analysis" in unsupported_plot_response.json()["detail"]
    assert record_path.read_bytes() == before


def test_stepped_import_keeps_only_the_explicitly_selected_step(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Never append an unselected source step to the raw measurement record."""
    client, record_path, _record = isolated_database
    before = record_path.read_bytes()
    selections = {
        **_trace_selection(),
        "plot_index": 0,
        "step_indices": [1],
        "step_selections": [
            {
                "step_index": 1,
                "i_x": 20,
                "turn_on": {"start_s": 0, "end_s": 1e-6},
                "turn_off": {"start_s": 1e-6, "end_s": 2e-6},
            }
        ],
        "same_context_confirmed": True,
        "metadata": _common_step_metadata(),
    }
    content = _stepped_raw_bytes(tmp_path)
    _attach_review_token(client, content, selections)
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selections)},
    )

    assert response.status_code == 200, response.text
    assert response.json()["acquisition_count"] == 1
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    imported = stored["raw_measurement_data"][0]
    assert imported["import_provenance"]["source_step_indices"] == [1]
    assert len(imported["dpt_on_vds"]) == 1
    assert len(imported["dpt_off_vds"]) == 1
    assert record_path.read_bytes() != before


def test_later_step_only_stepped_ascii_import_renumbers_temporary_points(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Reindex temporary ASCII rows after selecting a step that starts later."""
    client, record_path, _record = isolated_database
    before = record_path.read_bytes()
    selections = {
        **_trace_selection(),
        "plot_index": 0,
        "step_indices": [1],
        "step_selections": [
            {
                "step_index": 1,
                "i_x": 20,
                "turn_on": {"start_s": 0, "end_s": 1e-6},
                "turn_off": {"start_s": 1e-6, "end_s": 2e-6},
            }
        ],
        "same_context_confirmed": True,
        "metadata": _common_step_metadata(),
    }
    content = _stepped_ascii_raw_bytes(tmp_path)
    _attach_review_token(client, content, selections)
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selections)},
    )

    assert response.status_code == 200, response.text
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    imported = stored["raw_measurement_data"][-1]
    assert imported["import_provenance"]["source_step_indices"] == [1]
    assert imported["dpt_on_vds"] == [[[0, 400], [1e-6, 390]]]
    assert imported["dpt_on_id"] == [[[0, 0], [1e-6, 10]]]
    assert imported["dpt_off_vds"] == [[[1e-6, 390], [2e-6, 380]]]
    assert imported["dpt_off_id"] == [[[1e-6, 10], [2e-6, 20]]]
    assert record_path.read_bytes() != before


def test_stepped_axis_without_repeated_origins_is_rejected(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Do not guess step indexes when the RAW axis has no run boundaries."""
    client, record_path, _record = isolated_database
    before = record_path.read_bytes()
    content = _stepped_raw_bytes(
        tmp_path,
        time_values=[0, 1, 2, 3, 4, 5],
    )
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(content),
    )
    assert response.status_code == 400
    assert "no repeated time-origin boundaries" in response.json()["detail"]
    assert record_path.read_bytes() == before


def test_invalid_and_unsupported_files_do_not_mutate_record(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Reject invalid inputs before changing the legacy JSON record."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()

    invalid = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(b"not a valid simulator file"),
    )
    unsupported_extension = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(b"time,value\n0,1\n", "measurement.csv"),
    )
    unsupported_analysis = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(_complex_raw_bytes(tmp_path)),
    )

    assert invalid.status_code == 400
    assert "valid LTspice .raw" in invalid.json()["detail"]
    assert unsupported_extension.status_code == 415
    assert "Choose an LTspice .raw file" in unsupported_extension.json()["detail"]
    assert unsupported_analysis.status_code == 400
    assert "no transient analysis" in unsupported_analysis.json()["detail"]
    assert record_path.read_bytes() == original_bytes


def test_trace_preview_is_display_only_and_keeps_source_record_unchanged(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Bound the selected trace preview and leave source data untouched."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    selection = _trace_selection()

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/trace_preview",
        files=_upload_file(_raw_bytes(tmp_path, sample_count=1200)),
        data={"selection_json": json.dumps(selection)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sample_count"] == 1200
    assert payload["preview_sample_count"] == 800
    assert payload["time_range_s"] == [0.0, 0.001199]
    assert len(payload["traces"]["V(ds)"]) == 800
    assert payload["traces"]["V(ds)"][0] == [0.0, 400.0]
    assert payload["traces"]["V(ds)"][-1] == [0.001199, -11590.0]
    assert record_path.read_bytes() == original_bytes


def test_import_appends_exact_window_samples_and_preserves_legacy_fields(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Append mapped samples while preserving existing raw and source fields."""
    client, record_path, record = isolated_database
    record["vendor_extension"] = {"keep": ["source", 17]}
    record["raw_measurement_data"] = [
        {
            "dataset_type": "dpt_u_i",
            "comment": "existing waveform stays first",
            "operator_extension": {"retain": True},
            "dpt_on_vds": [[[9e-6, 99.0]]],
            "dpt_on_id": [[[9e-6, 9.0]]],
            "dpt_off_vds": [],
            "dpt_off_id": [],
        }
    ]
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    existing_raw = copy.deepcopy(record["raw_measurement_data"])
    selection = _import_selection()

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(_raw_bytes(tmp_path)),
        data={"selection_json": json.dumps(selection)},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["success"] is True
    assert result["raw_measurement_count"] == 2
    assert result["on_sample_count"] == 3
    assert result["off_sample_count"] == 3
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert record_path.stat().st_size == main._encoded_json_size_up_to(
        stored,
        main.MAX_DPT_LEGACY_RECORD_FILE_BYTES,
        indent=2,
    )
    assert b"\r\n" not in record_path.read_bytes()
    assert stored["raw_measurement_data"][0] == existing_raw[0]
    assert stored["vendor_extension"] == {"keep": ["source", 17]}
    imported = stored["raw_measurement_data"][1]
    assert imported["dataset_type"] == "dpt_u_i"
    assert imported["comment"] == "Reviewed event boundaries"
    assert imported["load_inductance"] == 0.75
    assert imported["commutation_inductance"] == 0.125
    assert imported["measurement_testbench"] == "LTspice local test bench"
    _assert_sample_pairs(imported["dpt_on_vds"][0], [
        [1e-6, 390.0], [2e-6, 380.0], [3e-6, 370.0]
    ])
    _assert_sample_pairs(imported["dpt_on_id"][0], [
        [1e-6, 10.0], [2e-6, 20.0], [3e-6, 30.0]
    ])
    _assert_sample_pairs(imported["dpt_off_vds"][0], [
        [4e-6, 360.0], [5e-6, 350.0], [6e-6, 340.0]
    ])
    _assert_sample_pairs(imported["dpt_off_id"][0], [
        [4e-6, 40.0], [5e-6, 50.0], [6e-6, 60.0]
    ])
    assert imported["import_provenance"]["trace_mapping"] == {
        "time": "time", "vds": "V(ds)", "id": "I(M1)"
    }
    assert imported["import_provenance"]["event_windows_s"] == {
        "turn_on": [1e-6, 3e-6], "turn_off": [4e-6, 6e-6]
    }
    assert "source_sha256" in imported["import_provenance"]


def test_import_rejects_cumulative_raw_payload_over_limit_without_mutation(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not append when total stored raw DPT JSON exceeds its byte budget."""
    client, record_path, record = isolated_database
    record["raw_measurement_data"] = [{
        "dataset_type": "dpt_u_i",
        "comment": "existing measurement",
        "operator_note": "x" * 512,
    }]
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    before = record_path.read_bytes()
    monkeypatch.setattr(main, "MAX_DPT_RAW_MEASUREMENT_BYTES", 128)

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(_raw_bytes(tmp_path)),
        data={"selection_json": json.dumps(_import_selection())},
    )

    assert response.status_code == 413
    assert "128 bytes" in response.json()["detail"]
    assert "existing record was left unchanged" in response.json()["detail"]
    assert record_path.read_bytes() == before


def test_dpt_waveform_read_rejects_preexisting_raw_payload_over_limit(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a legacy oversized raw list before summary conversion/projection."""
    client, record_path, record = isolated_database
    record["raw_measurement_data"] = [{"comment": "x" * 512}]
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    before = record_path.read_bytes()
    monkeypatch.setattr(main, "MAX_DPT_RAW_MEASUREMENT_BYTES", 128)

    response = client.get(
        f"/api/transistors/{DEVICE_NAME}/dpt_data",
        params={"waveform_preview_points": 800},
    )

    assert response.status_code == 413
    assert "128 bytes" in response.json()["detail"]
    assert "stored record was left unchanged" in response.json()["detail"]
    assert record_path.read_bytes() == before


def test_dpt_waveform_read_bounds_file_before_json_parsing(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file above the cap returns 413 without attempting JSON decoding."""
    client, record_path, _record = isolated_database
    monkeypatch.setattr(main, "MAX_DPT_LEGACY_RECORD_FILE_BYTES", 32)
    record_path.write_bytes(b"{" + b" " * 32)

    response = client.get(
        f"/api/transistors/{DEVICE_NAME}/dpt_data",
        params={"waveform_preview_points": 800},
    )

    assert response.status_code == 413
    assert "DPT reads and imports stop before parsing" in response.json()["detail"]


def test_invalid_selection_and_empty_window_leave_record_bytes_unchanged(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Keep the stored record byte-for-byte unchanged on selection errors."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    content = _raw_bytes(tmp_path)

    empty_window = _import_selection()
    empty_window["turn_on"] = {"start_s": 10e-6, "end_s": 11e-6}
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(empty_window)},
    )
    reversed_window = _import_selection()
    reversed_window["turn_on"] = {"start_s": 3e-6, "end_s": 1e-6}
    reversed_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(reversed_window)},
    )

    assert response.status_code == 400
    assert "contains no source samples" in response.json()["detail"]
    assert reversed_response.status_code == 422
    assert record_path.read_bytes() == original_bytes


def test_upload_byte_limit_returns_actionable_error_without_mutation(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject oversized uploads with a readable size-limit error."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    monkeypatch.setattr(main, "MAX_RAW_UPLOAD_BYTES", 8)

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(b"larger than eight bytes"),
    )

    assert response.status_code == 413
    assert "Reduce the LTspice output" in response.json()["detail"]
    assert record_path.read_bytes() == original_bytes


@pytest.mark.parametrize("suffix", ["/preview", "/trace_preview", ""])
def test_request_body_content_length_limit_precedes_multipart_parser(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    """Reject oversized declared bodies before Starlette constructs its spool."""
    client, _record_path, _record = isolated_database
    monkeypatch.setattr(main, "MAX_RAW_REQUEST_BODY_BYTES", 32)

    def parser_must_not_start(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("multipart parsing started before the body cap")

    monkeypatch.setattr(starlette.requests, "MultiPartParser", parser_must_not_start)
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import{suffix}",
        content=b"x" * 64,
        headers={
            "content-type": "multipart/form-data; boundary=not-parsed",
            "content-length": "64",
        },
    )

    assert response.status_code == 413
    assert "request body exceeds" in response.json()["detail"]


@pytest.mark.anyio
async def test_request_body_chunk_limit_stops_before_multipart_parser_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count chunked bodies without buffering them before Starlette's parser."""
    monkeypatch.setattr(main, "MAX_RAW_REQUEST_BODY_BYTES", 10)
    forwarded: list[bytes] = []
    sent: list[dict[str, Any]] = []
    incoming = iter([
        {"type": "http.request", "body": b"123456", "more_body": True},
        {"type": "http.request", "body": b"78901", "more_body": False},
    ])

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def downstream(scope: dict[str, Any], receive_fn: Any, send_fn: Any) -> None:
        del scope, send_fn
        while True:
            message = await receive_fn()
            forwarded.append(message.get("body", b""))
            if not message.get("more_body", False):
                return

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = main.DptRawRequestBodyLimitMiddleware(downstream)
    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
            "headers": [(b"transfer-encoding", b"chunked")],
        },
        receive,
        send,
    )

    assert forwarded == [b"123456"]
    assert sent[0]["status"] == 413
    assert b"request body exceeds" in sent[1]["body"]


def test_parser_sample_limit_returns_actionable_error(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject oversized sample arrays with an actionable error."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    monkeypatch.setattr(dpt_raw_import, "MAX_RAW_POINTS", 2)

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(_raw_bytes(tmp_path, sample_count=3)),
    )

    assert response.status_code == 413
    assert "contains 3 samples; the limit is 2" in response.json()["detail"]
    assert record_path.read_bytes() == original_bytes


def test_bounded_ascii_transient_remains_importable(tmp_path: Path) -> None:
    """Keep the existing unstepped ASCII Values path within preflight caps."""
    parsed = dpt_raw_import.parse_ltspice_raw(
        _ascii_raw_bytes(tmp_path),
        "measurement.raw",
        ("time", "V(ds)", "I(M1)"),
    )

    assert parsed.sample_count == 7
    assert parsed.trace_values is not None
    assert parsed.trace_values["V(ds)"][0] == pytest.approx(400.0)
    assert parsed.trace_values["I(M1)"][-1] == pytest.approx(60.0)


def test_ascii_huge_count_rejects_before_rawread_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject hostile ASCII declarations before PyLTSpice allocates vectors."""
    valid = _ascii_raw_bytes(tmp_path)
    encoding = "utf_16_le" if valid[1] == 0 else "utf-8"
    marker = "Values:".encode(encoding)
    marker_index = valid.index(marker)
    header = valid[:marker_index].decode(encoding)
    malformed = (
        re.sub(r"(?m)^No\. Points:\s+\d+$", "No. Points: 1000001", header)
        .encode(encoding)
        + valid[marker_index:]
    )

    def forbidden_rawread(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("ASCII declarations must be bounded before RawRead allocation")

    monkeypatch.setattr(PyLTSpice, "RawRead", forbidden_rawread)
    with pytest.raises(dpt_raw_import.DptRawImportError, match="contains 1000001 samples"):
        dpt_raw_import.parse_ltspice_raw(
            malformed, "measurement.raw", ("time", "V(ds)", "I(M1)")
        )


def test_complex_transient_is_rejected_before_numeric_conversion(
    tmp_path: Path,
) -> None:
    """Never discard complex samples by casting them to real transient traces."""
    ac_content = _complex_raw_bytes(tmp_path)
    encoding = "utf_16_le" if ac_content[1] == 0 else "utf-8"
    marker = "Binary:".encode(encoding)
    marker_index = ac_content.index(marker)
    header = ac_content[:marker_index].decode(encoding)
    malformed = (
        header.replace("AC Analysis", "Transient Analysis", 1).encode(encoding)
        + ac_content[marker_index:]
    )

    with pytest.raises(dpt_raw_import.DptRawImportError, match="Complex-valued transient"):
        dpt_raw_import.parse_ltspice_raw(malformed, "measurement.raw")


def test_second_order_time_markers_decode_like_pyltspice(
    tmp_path: Path,
) -> None:
    """Decode LTspice negative time markers without changing signal samples."""
    raw_times = [0, 1e-6, -2e-6, 3e-6]
    vds_values = [100, 200, 300, 400]
    id_values = [1, 2, 3, 4]
    parsed = dpt_raw_import.parse_ltspice_raw(
        _raw_bytes(
            tmp_path,
            time_values=raw_times,
            vds_values=vds_values,
            id_values=id_values,
        ),
        "measurement.raw",
        ("time", "V(ds)", "I(M1)"),
    )

    assert parsed.time_values.tolist() == pytest.approx([0, 1e-6, 2e-6, 3e-6])
    assert parsed.trace_values is not None
    assert parsed.trace_values["V(ds)"].tolist() == pytest.approx(vds_values)
    assert parsed.trace_values["I(M1)"].tolist() == pytest.approx(id_values)


def test_data_marker_scans_inline_false_markers_with_bounded_slices() -> None:
    """Ignore inline marker text without repeatedly copying the file prefix."""

    class SliceTrackingBytes(bytes):
        def __new__(cls, value: bytes) -> "SliceTrackingBytes":
            instance = super().__new__(cls, value)
            instance.slice_lengths = []
            instance.newline_searches = 0
            return instance

        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, slice):
                start, stop, _step = key.indices(len(self))
                self.slice_lengths.append(max(0, stop - start))
            return super().__getitem__(key)

        def find(self, sub: bytes, start: int = 0, end: int | None = None) -> int:
            if sub == b"\n":
                self.newline_searches += 1
            if end is None:
                return super().find(sub, start)
            return super().find(sub, start, end)

    false_markers = b"prefix " + b" ".join([b"Binary:", b"Values:"] * 64)
    content = SliceTrackingBytes(false_markers + b"\nBinary:\n" + b"payload")

    marker_index, kind, data_start = dpt_raw_import._find_data_marker(
        content, 0, "utf-8"
    )

    assert kind == "binary"
    assert marker_index == len(false_markers) + 1
    assert data_start == marker_index + len(b"Binary:\n")
    assert content.find(b"payload", data_start) == data_start
    assert content.slice_lengths
    assert max(content.slice_lengths) <= len(b"\n")
    assert content.newline_searches == 1


def test_parser_aggregates_point_budget_across_plots_before_axis_reads(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce the file-wide point cap over every declared RAW result plot."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    monkeypatch.setattr(dpt_raw_import, "MAX_RAW_POINTS", 10)
    content = b"".join(
        [
            _complex_raw_bytes(tmp_path),
            _stepped_raw_bytes(tmp_path),
            _raw_bytes(tmp_path, plot_name="Operating Point", sample_count=4),
        ]
    )

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/preview",
        files=_upload_file(content),
    )

    assert response.status_code == 413
    assert "samples across its plots" in response.json()["detail"]
    assert "aggregate limit is 10" in response.json()["detail"]
    assert record_path.read_bytes() == original_bytes


def test_parser_checks_sample_limit_before_loading_selected_waveforms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject oversized declarations before constructing any RAW reader."""
    def forbidden_rawread(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("RawRead must not see data before point-budget preflight")

    monkeypatch.setattr(PyLTSpice, "RawRead", forbidden_rawread)
    monkeypatch.setattr(dpt_raw_import, "MAX_RAW_POINTS", 2)

    with pytest.raises(dpt_raw_import.DptRawImportError, match="contains 3 samples"):
        dpt_raw_import.parse_ltspice_raw(
            _raw_bytes(tmp_path, sample_count=3),
            "measurement.raw",
            ("time", "V(ds)", "I(M1)"),
        )


def test_parser_rejects_decreasing_time_axis_and_preserves_equal_timestamp_order(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Reject decreasing time and preserve repeated timestamps in source order."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    decreasing = _raw_bytes(
        tmp_path,
        time_values=[0.0, 2e-6, 1e-6, 3e-6],
    )
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/trace_preview",
        files=_upload_file(decreasing),
        data={"selection_json": json.dumps(_trace_selection())},
    )
    repeated = _raw_bytes(
        tmp_path,
        time_values=[0.0, 1e-6, 1e-6, 3e-6],
    )
    repeated_response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/trace_preview",
        files=_upload_file(repeated),
        data={"selection_json": json.dumps(_trace_selection())},
    )

    assert response.status_code == 400
    assert "time axis decreases" in response.json()["detail"]
    assert repeated_response.status_code == 200
    assert repeated_response.json()["traces"]["V(ds)"][1][0] == 1e-6
    assert repeated_response.json()["traces"]["V(ds)"][2][0] == 1e-6
    assert record_path.read_bytes() == original_bytes


def test_event_sample_mapping_preserves_input_order_and_inclusive_boundaries() -> None:
    """Keep source order and include both endpoints when selecting a window."""
    times = np.array([0.0, 1.0, 1.0, 3.0])
    vds = np.array([0.0, 10.0, 11.0, 30.0])
    ids = np.array([0.0, 1.0, 2.0, 3.0])

    selected_vds, selected_id = dpt_raw_import.select_event_samples(
        times, vds, ids, 1.0, 3.0
    )

    assert selected_vds == [[1.0, 10.0], [1.0, 11.0], [3.0, 30.0]]
    assert selected_id == [[1.0, 1.0], [1.0, 2.0], [3.0, 3.0]]


def test_nonfinite_preview_samples_are_json_gaps_and_clean_windows_still_import(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Mark non-finite preview points as gaps and allow clean event windows."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    content = _raw_bytes(
        tmp_path,
        time_values=[0.0, 1e-6, 2e-6, 3e-6],
        vds_values=[400.0, 390.0, float("nan"), 370.0],
        id_values=[0.0, 10.0, float("inf"), 30.0],
    )
    preview = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import/trace_preview",
        files=_upload_file(content),
        data={"selection_json": json.dumps(_trace_selection())},
    )

    assert preview.status_code == 200, preview.text
    preview_json = preview.json()
    assert preview_json["traces"]["V(ds)"][2] == [2e-6, None]
    assert preview_json["traces"]["I(M1)"][2] == [2e-6, None]
    assert preview_json["non_finite_preview_sample_counts"] == {
        "V(ds)": 1, "I(M1)": 1
    }

    selection = _import_selection()
    selection["turn_on"] = {"start_s": 1e-6, "end_s": 1e-6}
    selection["turn_off"] = {"start_s": 3e-6, "end_s": 3e-6}
    imported = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(content),
        data={"selection_json": json.dumps(selection)},
    )

    assert imported.status_code == 200, imported.text
    assert imported.json()["on_sample_count"] == 1
    assert imported.json()["off_sample_count"] == 1
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    assert len(stored["raw_measurement_data"]) == 1
    assert record_path.read_bytes() != original_bytes


def test_combined_event_sample_limit_rejects_without_mutation(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound pair-list allocation across both event windows together."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    monkeypatch.setattr(main, "MAX_TOTAL_EVENT_SAMPLES", 5)

    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(_raw_bytes(tmp_path)),
        data={"selection_json": json.dumps(_import_selection())},
    )

    assert response.status_code == 413
    assert "contain 6 samples combined; the limit is 5" in response.json()["detail"]
    assert record_path.read_bytes() == original_bytes


def test_stepped_aggregate_window_limit_preflights_before_signal_read(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count every selected step's event samples before loading Vds/Id arrays."""
    client, record_path, _record = isolated_database
    original_bytes = record_path.read_bytes()
    monkeypatch.setattr(main, "MAX_TOTAL_EVENT_SAMPLES", 10)
    original_parse = main.parse_ltspice_raw
    calls: list[tuple[str, ...] | None] = []

    def capture_trace_selection(content, filename, selected_traces=None, *, plot_index=None):
        calls.append(tuple(selected_traces) if selected_traces else None)
        if selected_traces is None:
            return original_parse(content, filename, plot_index=plot_index)
        return original_parse(
            content,
            filename,
            selected_traces,
            plot_index=plot_index,
        )

    monkeypatch.setattr(main, "parse_ltspice_raw", capture_trace_selection)
    selections = {
        **_trace_selection(),
        "plot_index": 0,
        "step_indices": [0, 1],
        "step_selections": [
            {
                "step_index": 0,
                "i_x": 10,
                "turn_on": {"start_s": 0, "end_s": 2e-6},
                "turn_off": {"start_s": 0, "end_s": 2e-6},
            },
            {
                "step_index": 1,
                "i_x": 20,
                "turn_on": {"start_s": 0, "end_s": 4e-6},
                "turn_off": {"start_s": 0, "end_s": 4e-6},
            },
        ],
        "same_context_confirmed": True,
        "metadata": _common_step_metadata(),
    }
    response = client.post(
        f"/api/transistors/{DEVICE_NAME}/dpt_raw_import",
        files=_upload_file(_stepped_raw_bytes(tmp_path)),
        data={"selection_json": json.dumps(selections)},
    )

    assert response.status_code == 413
    assert "samples combined; the limit is 10" in response.json()["detail"]
    assert calls
    assert all(selection is None for selection in calls)
    assert record_path.read_bytes() == original_bytes
