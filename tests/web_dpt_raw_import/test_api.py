"""Focused API coverage for additive native LTspice DPT RAW imports."""

from __future__ import annotations

import copy
import json
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
        files=_upload_file(_raw_bytes(tmp_path, plot_name="AC Analysis")),
    )

    assert invalid.status_code == 400
    assert "valid LTspice .raw" in invalid.json()["detail"]
    assert unsupported_extension.status_code == 415
    assert "Choose an LTspice .raw file" in unsupported_extension.json()["detail"]
    assert unsupported_analysis.status_code == 400
    assert "not a transient analysis" in unsupported_analysis.json()["detail"]
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


def test_parser_checks_sample_limit_before_loading_selected_waveforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sample-limit preflight happens before selected arrays are read."""
    calls: list[list[str] | None] = []

    class HeaderOnlyReader:
        dialect = "ltspice"
        has_axis = True
        nPoints = 3

        def __init__(
            self,
            _path: Path,
            traces_to_read: list[str] | None = None,
            verbose: bool = False,
        ) -> None:
            del verbose
            calls.append(traces_to_read)

        def get_nr_plots(self) -> int:
            return 1

        def get_plot_name(self) -> str:
            return "Transient Analysis"

        def get_raw_properties(self) -> dict[str, str]:
            return {"Flags": "real"}

        def get_steps(self) -> list[int]:
            return []

        def get_trace_names(self) -> list[str]:
            return ["time", "V(ds)", "I(M1)"]

    monkeypatch.setattr(PyLTSpice, "RawRead", HeaderOnlyReader)
    monkeypatch.setattr(dpt_raw_import, "MAX_RAW_POINTS", 2)

    with pytest.raises(dpt_raw_import.DptRawImportError, match="contains 3 samples"):
        dpt_raw_import.parse_ltspice_raw(
            b"header preflight fixture",
            "measurement.raw",
            ("time", "V(ds)", "I(M1)"),
        )

    assert calls == [None]


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
