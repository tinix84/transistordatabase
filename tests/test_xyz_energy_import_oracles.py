"""Check XYZ import identities against physical coordinates and source records."""
import base64
from hashlib import sha256
from itertools import permutations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.xyz_energy_routes import router
from transistordatabase.xyz_energy_import import import_xyz_energy_map


def declarations(order):
    """Declare the fixed conditions measured with the source."""
    return {
        "source_name": "measured.tsv",
        "input_kind": "file",
        "source": "test bench 1",
        "source_kind": "measured",
        "energy_kind": "e_rr",
        "conditions": {
            "t_j_c": 25, "v_g_on_v": None, "v_g_off_v": -4,
            "r_g_on_ohm": None, "r_g_off_ohm": 1.5,
            "commutation_device": None, "commutation_source": None,
            "energy_convention": "Recovery integral",
        },
        "options": {
            "delimiter": "tsv", "decimal": ".", "has_header": True,
            "columns": {field: index for index, field in enumerate(order)},
            "units": {"current": "mA", "voltage": "kV", "energy": "mJ"},
            "missing_energy_tokens": ["not measured"],
        },
    }


def source_bytes(order):
    """Write source observations in a chosen order, including BOM and blank line."""
    fields = ("current", "voltage", "energy")
    observations = [
        ("10000", "0.4", "2.125"), ("2500", "0", "not measured"),
        ("0", "0.4", ""), ("+2.500e3", "0.00", "0.10"),
        ("0", "0", "0"),
    ]
    lines = ["\t".join(order), ""]
    for record in observations:
        values = dict(zip(fields, record, strict=True))
        lines.append("\t".join(f'"{values[field]}"' for field in order))
    return ("\r\n".join(lines) + "\r\n").encode("utf-8-sig")


@pytest.mark.parametrize("order", permutations(("current", "voltage", "energy")))
def test_table_is_invariant_under_column_permutation(order):
    """Match every returned cell, raw observation, and physical line independently."""
    raw = source_bytes(order)
    result = import_xyz_energy_map(raw, declarations(order))
    assert result["source"]["sha256"] == sha256(raw).hexdigest()
    assert result["currents_a"] == [0, 2.5, 10]
    assert result["voltages_v"] == [0, 400]
    expected = [
        [("available", 0, [7]), ("ambiguous", None, [4, 6]), ("missing", None, [])],
        [("unknown", None, [5]), ("missing", None, []), ("available", .002125, [3])],
    ]
    lines = raw.decode("utf-8-sig").splitlines()
    rows = {row["id"]: row for row in result["rows"]}
    assert len(rows) == 5
    for cells, voltage, expected_row in zip(result["cells"], [0, 400], expected, strict=True):
        for cell, current, (status, energy, line_numbers) in zip(
            cells, [0, 2.5, 10], expected_row, strict=True
        ):
            assert (cell["current_a"], cell["voltage_v"]) == (current, voltage)
            assert (cell["status"], cell["energy_j"]) == (status, energy)
            assert [rows[key]["line_number"] for key in cell["row_ids"]] == line_numbers
            for key in cell["row_ids"]:
                row = rows[key]
                assert row["raw_record"] == lines[row["line_number"] - 1]
                assert row["cell_status"] == status
    assert rows["line-4"]["energy_j"] is None
    assert rows["line-6"]["energy_j"] == .0001


def test_normalized_identity_matches_over_http_and_python():
    """HTTP and Python must produce one normalized identity for one request."""
    order = ("voltage", "energy", "current")
    raw = source_bytes(order)
    metadata = declarations(order)
    direct = import_xyz_energy_map(raw, metadata)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as api:
        response = api.post("/api/analysis/energy-map/xyz", json={
            "content_base64": base64.b64encode(raw).decode("ascii"), **metadata,
        })
    assert response.status_code == 200, response.text
    transported = response.json()
    assert transported["request_sha256"] == direct["request_sha256"]
    assert transported == direct
