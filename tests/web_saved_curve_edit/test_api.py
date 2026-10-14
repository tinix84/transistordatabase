"""Isolated regression coverage for editing raw legacy curve entries."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.gui_web.backend import main


DEVICE_NAME = "Saved Curve Edit Fixture"
NON_FINITE_DEVICE_NAME = "Infineon_IPW65R090CFD7"
CURVE_CASES = [
    ("switch", "channel", ("switch", "channel"), "graph_v_i"),
    ("diode", "channel", ("diode", "channel"), "graph_v_i"),
    ("switch", "e_on", ("switch", "e_on"), "graph_i_e"),
    ("switch", "e_off", ("switch", "e_off"), "graph_i_e"),
    ("diode", "e_rr", ("diode", "e_rr"), "graph_i_e"),
    ("switch", "gate_charge", ("switch", "charge_curve"), "graph_q_v"),
    ("switch", "soa", ("switch", "soa"), "graph_i_v"),
    ("capacitance", "c_oss", (None, "c_oss"), "graph_v_c"),
    ("capacitance", "c_iss", (None, "c_iss"), "graph_v_c"),
    ("capacitance", "c_rss", (None, "c_rss"), "graph_v_c"),
]


class TemporaryJsonRepository:
    """Repository facade whose only record is under pytest's temporary path."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory

    def get_by_name(self, name: str) -> Any:
        path = self.data_directory / f"{name}.json"
        if not path.exists():
            return None
        return JsonTransistorLoader().load_from_json(path)


def _fixture_record() -> dict[str, Any]:
    source = (
        Path(__file__).resolve().parents[2]
        / "transistordatabase"
        / "examples"
        / "tdb_example"
        / "Fuji_2MBI100XAA120-50.json"
    )
    record = json.loads(source.read_text(encoding="utf-8"))
    record["name"] = DEVICE_NAME
    record["unknown_future_field"] = {"keep": ["raw", 17, None]}
    return record


@pytest.fixture
def isolated_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path, dict[str, Any]]:
    record = _fixture_record()
    record_path = tmp_path / f"{DEVICE_NAME}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(tmp_path))
    return TestClient(main.app), record_path, record


@pytest.fixture
def isolated_infineon_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path, Path]:
    source = (
        Path(__file__).resolve().parents[2]
        / "transistordatabase"
        / "examples"
        / "tdb_example"
        / "Infineon_IPW65R090CFD7.json"
    )
    # Copy source bytes directly: fixture setup must not parse and duplicate the
    # multi-megabyte measurement payload just to put it in an isolated database.
    record_path = tmp_path / source.name
    shutil.copyfile(source, record_path)
    monkeypatch.setattr(main, "_repo", TemporaryJsonRepository(tmp_path))
    return TestClient(main.app), record_path, source


def _top_level_value_bytes(document: bytes, key: bytes) -> bytes:
    """Extract one top-level JSON array/object without materializing its value."""
    key_marker = b'"' + key + b'"'
    key_start = document.index(key_marker)
    value_start = document.index(b":", key_start + len(key_marker)) + 1
    while document[value_start] in b" \t\r\n":
        value_start += 1
    opening = document[value_start]
    if opening not in (ord("["), ord("{")):
        raise AssertionError(f"Top-level {key.decode()} value is not a container")
    closing_for = {ord("["): ord("]"), ord("{"): ord("}")}
    stack = [closing_for[opening]]
    in_string = False
    escaped = False
    for position in range(value_start + 1, len(document)):
        char = document[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == ord("\\"):
                escaped = True
            elif char == ord('"'):
                in_string = False
            continue
        if char == ord('"'):
            in_string = True
        elif char in closing_for:
            stack.append(closing_for[char])
        elif char in (ord("]"), ord("}")):
            if not stack or stack.pop() != char:
                raise AssertionError("Mismatched JSON container while extracting raw bytes")
            if not stack:
                return document[value_start : position + 1]
    raise AssertionError(f"Unterminated top-level {key.decode()} value")


def _patch_for(case: tuple[str, str, tuple[str | None, str], str], original: dict[str, Any]) -> dict[str, Any]:
    component, curve_type, _path, graph_key = case
    patch: dict[str, Any] = {}
    if graph_key != "graph_i_v":
        patch["t_j"] = original.get("t_j", 25)
    if graph_key == "graph_v_i":
        patch[graph_key] = [[0.0, 1.0], [0.0, 2.0]]
        if component == "switch":
            patch["v_g"] = original.get("v_g", 15)
    elif graph_key == "graph_i_e":
        patch.update(
            dataset_type=original.get("dataset_type", "graph_i_e"),
            v_supply=original.get("v_supply", 400),
            v_g=original.get("v_g", 15),
            graph_i_e=[[0.0, 10.0], [1e-6, 2e-6]],
        )
    elif graph_key == "graph_q_v":
        patch.update(
            v_supply=original["v_supply"],
            i_channel=original["i_channel"],
            i_g=original.get("i_g"),
            graph_q_v=[[0.0, 2e-9], [0.0, 2.0]],
        )
    elif graph_key == "graph_i_v":
        patch.update(t_c=original.get("t_c"), time_pulse=original.get("time_pulse"), graph_i_v=[[0.0, 10.0], [10.0, 1.0]])
    else:
        patch[graph_key] = [[0.0, 10.0], [1e-9, 5e-10]]
    return patch


@pytest.mark.parametrize("case", CURVE_CASES, ids=[f"{c}.{t}" for c, t, _p, _g in CURVE_CASES])
def test_update_patches_only_the_selected_raw_curve_and_preserves_unknown_values(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    case: tuple[str, str, tuple[str | None, str], str],
) -> None:
    client, record_path, before = isolated_database
    component, curve_type, (section_name, field), _graph_key = case
    curves = before[field] if section_name is None else before[section_name][field]
    original_curve = curves[0]
    original_curve["unknown_curve_field"] = {"preserve": True, "value": ["future"]}
    before["unknown_future_field"]["edit_check"] = "keep"
    if section_name is not None:
        before[section_name]["unknown_section_field"] = {"preserve": 22}
    record_path.write_text(json.dumps(before, indent=2), encoding="utf-8")
    baseline = copy.deepcopy(before)
    patch = _patch_for(case, original_curve)
    payload = {
        "expected_count": len(curves),
        "expected_token": main._curve_edit_token(original_curve),
        "curve": patch,
    }

    response = client.put(
        f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/{component}/{curve_type}/0",
        json=payload,
    )

    assert response.status_code == 200, response.text
    updated = json.loads(record_path.read_text(encoding="utf-8"))
    target_list = updated[field] if section_name is None else updated[section_name][field]
    expected = copy.deepcopy(baseline)
    expected_list = expected[field] if section_name is None else expected[section_name][field]
    expected_list[0].update(patch)
    assert updated == expected
    assert target_list[0]["unknown_curve_field"] == {"preserve": True, "value": ["future"]}
    assert updated["unknown_future_field"] == {"keep": ["raw", 17, None], "edit_check": "keep"}
    if section_name is not None:
        assert updated[section_name]["unknown_section_field"] == {"preserve": 22}


def test_curve_read_returns_raw_edit_token_and_exact_device_identity(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    client, _record_path, raw = isolated_database

    response = client.get(f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves")

    assert response.status_code == 200, response.text
    curve = response.json()["switch"]["channel_data"][0]
    expected_raw = raw["switch"]["channel"][0]
    assert curve["_edit_token"] == main._curve_edit_token(expected_raw)


def test_legacy_non_finite_sentinels_allow_curve_read_and_edit_without_changes(
    isolated_infineon_database: tuple[TestClient, Path, Path],
) -> None:
    client, record_path, source_path = isolated_infineon_database
    before_bytes = source_path.read_bytes()
    before_raw_measurements = _top_level_value_bytes(before_bytes, b"raw_measurement_data")
    assert before_raw_measurements.count(b"-Infinity") == 4

    curves_response = client.get(
        f"/api/transistors/{quote(NON_FINITE_DEVICE_NAME, safe='')}/curves"
    )

    assert curves_response.status_code == 200, curves_response.text
    api_curve = curves_response.json()["switch"]["channel_data"][0]
    patch = {"graph_v_i": [[0.0, 1.0], [0.0, 2.0]]}
    update_response = client.put(
        f"/api/transistors/{quote(NON_FINITE_DEVICE_NAME, safe='')}/curves/switch/channel/0",
        json={
            "expected_count": len(curves_response.json()["switch"]["channel_data"]),
            "expected_token": api_curve["_edit_token"],
            "curve": patch,
        },
    )

    assert update_response.status_code == 200, update_response.text
    after_bytes = record_path.read_bytes()
    after_raw_measurements = _top_level_value_bytes(after_bytes, b"raw_measurement_data")
    assert after_raw_measurements.count(b"-Infinity") == 4
    assert after_raw_measurements.replace(b"\r\n", b"\n") == before_raw_measurements.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_patch_values_are_rejected_without_writing(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    non_finite: float,
) -> None:
    client, record_path, record = isolated_database
    curves_response = client.get(
        f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves"
    )
    assert curves_response.status_code == 200, curves_response.text
    token = curves_response.json()["switch"]["channel_data"][0]["_edit_token"]
    original_bytes = record_path.read_bytes()

    response = client.put(
        f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/switch/channel/0",
        content=json.dumps(
            {
                "expected_count": len(record["switch"]["channel"]),
                "expected_token": token,
                "curve": {"graph_v_i": [[0.0, 1.0], [0.0, non_finite]]},
            },
            allow_nan=True,
        ),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400, response.text
    assert record_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "dataset_type,graph_key,graph_value,patch",
    [
        ("graph_r_e", "graph_r_e", [[1.0, 5.0], [0.001, 0.0005]], {"graph_r_e": [[1.0, 5.0], [0.0008, 0.0004]]}),
        ("graph_t_e", "graph_t_e", [[25.0, 125.0], [0.001, 0.002]], {"graph_t_e": [[25.0, 125.0], [0.0008, 0.0018]]}),
        ("single", None, None, {"i_x": 25.0, "e_x": 0.0008}),
    ],
)
def test_switching_graph_variants_patch_and_validate_as_raw_legacy_entries(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    dataset_type: str,
    graph_key: str | None,
    graph_value: list[list[float]] | None,
    patch: dict[str, Any],
) -> None:
    client, record_path, record = isolated_database
    raw_curve = record["switch"]["e_on"][0]
    raw_curve["dataset_type"] = dataset_type
    for key in ("graph_i_e", "graph_r_e", "graph_t_e"):
        raw_curve.pop(key, None)
    if graph_key:
        raw_curve[graph_key] = graph_value
        raw_curve["i_x"] = 20.0
        if dataset_type == "graph_t_e":
            raw_curve["r_g"] = 2.5
    else:
        raw_curve["i_x"] = 20.0
        raw_curve["e_x"] = 0.0007
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    before = copy.deepcopy(record)
    token = main._curve_edit_token(raw_curve)

    response = client.put(
        f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/switch/e_on/0",
        json={
            "expected_count": len(record["switch"]["e_on"]),
            "expected_token": token,
            "curve": patch,
        },
    )

    assert response.status_code == 200, response.text
    expected = copy.deepcopy(before)
    expected["switch"]["e_on"][0].update(patch)
    assert json.loads(record_path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    "precondition,index",
    [("count", 0), ("token", 0), ("index", 999)],
)
def test_stale_curve_preconditions_reject_without_changing_file_bytes(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
    precondition: str,
    index: int,
) -> None:
    client, record_path, record = isolated_database
    target = record["switch"]["channel"][0]
    payload: dict[str, Any] = {
        "expected_count": len(record["switch"]["channel"]),
        "expected_token": main._curve_edit_token(target),
        "curve": {"t_j": target["t_j"], "graph_v_i": [[0.0, 1.0], [0.0, 2.0]], "v_g": target["v_g"]},
    }
    if precondition == "count":
        payload["expected_count"] += 1
    elif precondition == "token":
        payload["expected_token"] = "0" * 64
    original_bytes = record_path.read_bytes()

    response = client.put(
        f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/switch/channel/{index}",
        json=payload,
    )

    assert response.status_code == 409, response.text
    assert record_path.read_bytes() == original_bytes


def test_invalid_curve_patch_leaves_file_unchanged(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    client, record_path, record = isolated_database
    target = record["switch"]["channel"][0]
    token = main._curve_edit_token(target)
    original_bytes = record_path.read_bytes()
    path = f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/switch/channel/0"
    invalid = client.put(
        path,
        json={
            "expected_count": len(record["switch"]["channel"]),
            "expected_token": token,
            "curve": {"t_j": 25, "v_g": 15, "graph_v_i": [[0, 1], [0]]},
        },
    )
    assert invalid.status_code == 400, invalid.text
    assert record_path.read_bytes() == original_bytes


def test_record_name_mismatch_never_targets_another_device_record(
    isolated_database: tuple[TestClient, Path, dict[str, Any]],
) -> None:
    client, record_path, record = isolated_database
    mismatched = copy.deepcopy(record)
    mismatched["name"] = "Different Device"
    record_path.write_text(json.dumps(mismatched, indent=2), encoding="utf-8")
    original_bytes = record_path.read_bytes()
    target = mismatched["switch"]["channel"][0]

    response = client.put(
        f"/api/transistors/{quote(DEVICE_NAME, safe='')}/curves/switch/channel/0",
        json={
            "expected_count": len(mismatched["switch"]["channel"]),
            "expected_token": main._curve_edit_token(target),
            "curve": {"t_j": 25, "v_g": target["v_g"], "graph_v_i": [[0, 1], [0, 2]]},
        },
    )

    assert response.status_code == 409, response.text
    assert record_path.read_bytes() == original_bytes
