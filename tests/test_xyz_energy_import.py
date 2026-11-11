"""Independent coordinate, evidence, parser, and API import checks."""
import base64
from copy import deepcopy
import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.gui_web.backend.xyz_energy_routes import router
from transistordatabase.xyz_energy_import import (
    XyzEnergyLimitError, import_xyz_energy_map,
)

FIXTURE = b"9,600,90\n3,200,30\n0,600,12\n9,200,60\n0,200,0\n"


def options():
    """Declare every condition and syntax choice without source inference."""
    return dict(source_name="fixture.csv", input_kind="file", source="Supplied fixture",
                source_kind="manual", energy_kind="e_on", conditions=dict(
                    t_j_c=None, v_g_on_v=None, v_g_off_v=None,
                    r_g_on_ohm=None, r_g_off_ohm=None, commutation_device=None,
                    commutation_source=None, energy_convention="Supplied event integral"),
                options=dict(delimiter="csv", decimal=".", has_header=False,
                             columns=dict(current=0, voltage=1, energy=2),
                             units=dict(current="A", voltage="V", energy="µJ"),
                             missing_energy_tokens=["NA"]))


def client():
    """Isolate the public router from database and application fixtures."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def request(raw=FIXTURE, metadata=None):
    """Encode supplied bytes with the complete explicit import declaration."""
    return (metadata or options()) | {"content_base64": base64.b64encode(raw).decode()}


def test_shuffled_fixture_keeps_hole_zero_and_source():
    """Compare the unsorted source with the independent expected matrix."""
    supplied = options()
    before = deepcopy(supplied)
    result = import_xyz_energy_map(FIXTURE, supplied)
    assert supplied == before
    assert result["currents_a"] == [0, 3, 9]
    assert result["voltages_v"] == [200, 600]
    assert [[c["energy_j"] for c in row] for row in result["cells"]] == [
        [0, .000030, .000060], [.000012, None, .000090]]
    assert result["cells"][1][1]["status"] == "missing"
    assert result["source"]["sha256"] == hashlib.sha256(FIXTURE).hexdigest()
    assert result["census"] == dict(source_rows=5, available_rows=5, unknown_rows=0,
                                    available_cells=5, missing_cells=1, unknown_cells=0,
                                    ambiguous_cells=0, duplicate_coordinates=0,
                                    total_cells=6)
    assert result["rows"][0]["lexemes"] == ["9", "600", "90"]
    assert result["rows"][0]["raw_record"] == "9,600,90"
    assert result["rows"][0]["id"] == "line-1"


def test_completion_and_all_duplicates_stay_ambiguous():
    """Keep conflicting, identical, and unknown duplicate observations separate."""
    complete = import_xyz_energy_map(FIXTURE + b"3,600,45\n", options())
    assert complete["census"]["available_cells"] == 6
    for extra in (b"3,600,46\n", b"3,600,45\n", b"3,600,NA\n"):
        result = import_xyz_energy_map(FIXTURE + b"3,600,45\n" + extra, options())
        cell = result["cells"][1][1]
        assert cell["status"] == "ambiguous"
        assert cell["energy_j"] is None
        assert cell["row_ids"] == ["line-6", "line-7"]
        assert result["rows"][-2]["energy_j"] == .000045
        assert result["rows"][-1]["cell_status"] == "ambiguous"


@pytest.mark.parametrize("raw", [b"0,0,0", b"0,0,NA", b"0,0,"])
def test_single_zero_voltage_observation_and_explicit_unknown(raw):
    """Permit one real observation and distinguish known zero from unknown energy."""
    result = import_xyz_energy_map(raw, options())
    assert result["currents_a"] == [0]
    assert result["voltages_v"] == [0]
    assert result["cells"][0][0]["status"] == (
        "available" if raw == b"0,0,0" else "unknown")


@pytest.mark.parametrize("delimiter,raw", [
    ("csv", b' E , V , I \r\n"90","0.6","9000"\r\n'),
    ("tsv", b"E\tV\tI\n90\t0.6\t9000\n"),
    ("whitespace", b"E V I\n90 0.6 9000\n"),
])
def test_declared_header_permutation_and_exact_prefixes(delimiter, raw):
    """Apply only the declared columns and decimal unit powers."""
    config = options()
    config["options"].update(delimiter=delimiter, has_header=True,
                              columns=dict(current=2, voltage=1, energy=0),
                              units=dict(current="mA", voltage="kV", energy="uJ"))
    result = import_xyz_energy_map(raw, config)
    assert result["currents_a"] == [9]
    assert result["voltages_v"] == [600]
    assert result["cells"][0][0]["energy_j"] == .00009
    assert result["header"]["line_number"] == 1
    assert result["rows"][0]["line_number"] == 2
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("raw", [
    b"1,2,3,4", b"1,2", b'"1,2,3', b"1,2,=1+2", b"NaN,2,3",
    b"1,Infinity,3", b"-1,2,3", b"1,-2,3", b"1,2,-3", b"1,2,1_000",
    b"1,2,1e-400", b"1e400,2,3", b"1,2,1e99999999999999999",
    b"1,2," + b"1" * 81, b"1,2," + b"1" * 129,
    b"1,2,3\x00", b"\xff,2,3", b"1,2,3\n\n4,5",
    b'"1\n",2,3', b"1,2,?", b"", b"\n\n",
])
def test_bad_numeric_or_record_inputs_reject(raw):
    """Reject malformed records and numbers without coercion or execution."""
    with pytest.raises(ValueError):
        import_xyz_energy_map(raw, options())


@pytest.mark.parametrize("axis", [0, 1])
def test_distinct_decimal_coordinates_cannot_collapse_to_same_float(axis):
    """Detect precision loss before a false duplicate coordinate can form."""
    rows = [["1", "2", "3"], ["1", "2", "4"]]
    rows[1][axis] = "1.00000000000000001" if axis == 0 else "2.00000000000000001"
    raw = "\n".join(",".join(row) for row in rows).encode()
    with pytest.raises(ValueError, match="collapse"):
        import_xyz_energy_map(raw, options())


def test_equivalent_decimals_group_exactly_without_repair():
    """Equivalent decimal coordinates still represent two source observations."""
    result = import_xyz_energy_map(b"1,2,3\n1.0,2.00,3", options())
    assert result["census"]["ambiguous_cells"] == 1
    assert result["rows"][0]["current_decimal_key"] == result["rows"][1]["current_decimal_key"]


def test_hashes_cover_source_bytes_and_metadata_independently():
    """Bind row bytes and request choices to independent reproducible hashes."""
    config = options()
    first = import_xyz_energy_map(FIXTURE, config)
    config["conditions"]["t_j_c"] = 0
    second = import_xyz_energy_map(FIXTURE, config)
    assert first["source"]["sha256"] == second["source"]["sha256"]
    assert first["request_sha256"] != second["request_sha256"]
    config["input_kind"] = "paste"
    assert import_xyz_energy_map(FIXTURE, config)["source"]["hash_scope"] == "utf8-paste-bytes"
    assert first["source"]["hash_scope"] == "original-file-bytes"
    assert import_xyz_energy_map(FIXTURE.replace(b"\n", b"\r\n"), config)[
        "source"]["sha256"] != first["source"]["sha256"]


@pytest.mark.parametrize("edit", [
    lambda o: o["options"]["columns"].update(current=1),
    lambda o: o["options"]["columns"].update(current=True),
    lambda o: o["options"]["units"].update(energy="unknown"),
    lambda o: o["options"]["units"].update(current=[]),
    lambda o: o.update(source_kind={}),
    lambda o: o["options"].update(delimiter=[]),
    lambda o: o.update(conditions=[]),
    lambda o: o["options"].update(decimal=","),
    lambda o: o["options"].update(missing_energy_tokens=["0"]),
    lambda o: o["options"].update(missing_energy_tokens=[" NA"]),
    lambda o: o["conditions"].update(energy_convention=None),
    lambda o: o["conditions"].update(commutation_device="x" * 1001),
    lambda o: o.update(source=" "),
])
def test_invalid_declarations_reject_in_headless_and_api(edit):
    """Apply declaration constraints in both public entry points."""
    config = options()
    edit(config)
    with pytest.raises(ValueError):
        import_xyz_energy_map(FIXTURE, config)
    assert client().post("/api/analysis/energy-map/xyz", json=request(metadata=config)).status_code == 422


def test_axis_grid_row_and_raw_bounds(monkeypatch):
    """Reject resource excess before a large Cartesian matrix can form."""
    with pytest.raises(XyzEnergyLimitError, match="4 MB"):
        import_xyz_energy_map(b" " * 4_000_001, options())
    with pytest.raises(XyzEnergyLimitError, match="axis"):
        import_xyz_energy_map("\n".join(f"{i},1,1" for i in range(257)).encode(), options())
    with pytest.raises(XyzEnergyLimitError, match="axis"):
        import_xyz_energy_map("\n".join(f"1,{i},1" for i in range(129)).encode(), options())
    # A sparse diagonal exceeds the Cartesian grid budget before matrix allocation.
    raw = "\n".join([f"{i},0,1" for i in range(256)] +
                    [f"0,{i},1" for i in range(65)]).encode()
    with pytest.raises(XyzEnergyLimitError, match="grid product"):
        import_xyz_energy_map(raw, options())
    monkeypatch.setattr("transistordatabase.xyz_energy_import.MAX_ROWS", 2)
    with pytest.raises(XyzEnergyLimitError, match="data rows"):
        import_xyz_energy_map(b"1,2,3\n1,2,4\n1,2,5", options())


def test_response_bound_precedes_matrix_allocation(monkeypatch):
    """Enforce the evidence budget during source parsing."""
    monkeypatch.setattr("transistordatabase.xyz_energy_import.MAX_RESPONSE_BYTES", 6000)
    with pytest.raises(XyzEnergyLimitError, match="response evidence"):
        import_xyz_energy_map(FIXTURE, options())


def test_api_typed_response_and_request_failures():
    """Check source bounds, encoded bytes, extra fields, and the typed response."""
    api = client()
    reply = api.post("/api/analysis/energy-map/xyz", json=request())
    assert reply.status_code == 200, reply.text
    assert reply.json() == import_xyz_energy_map(FIXTURE, options())
    invalid = request() | {"content_base64": "not base64"}
    assert api.post("/api/analysis/energy-map/xyz", json=invalid).status_code == 422
    assert api.post("/api/analysis/energy-map/xyz", json=request() | {"path": "/etc/passwd"}).status_code == 422
    assert api.post("/api/analysis/energy-map/xyz", json=request(b" " * 4_000_001)).status_code == 413


def test_api_actual_body_bound_and_finite_condition_literals():
    """Ignore false length headers and reject nonzero JSON underflow."""
    api = client()
    encoded = json.dumps(request()).replace('"t_j_c": null', '"t_j_c": 1e-400')
    assert api.post("/api/analysis/energy-map/xyz", content=encoded,
                    headers={"content-type": "application/json"}).status_code == 422
    response = api.post("/api/analysis/energy-map/xyz", content=b" " * 6_000_001,
                        headers={"content-type": "application/json", "content-length": "1"})
    assert response.status_code == 413


def test_utf8_bom_and_exact_subnormal_or_large_finite_values():
    """Keep a UTF-8 byte order mark in the hash and permit finite extremes."""
    config = options()
    config["options"]["units"]["energy"] = "J"
    raw = b"\xef\xbb\xbf0,1e308,5e-324"
    result = import_xyz_energy_map(raw, config)
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["voltages_v"] == [1e308]
    assert result["rows"][0]["energy_j"] == 5e-324


def test_real_row_and_response_budgets_reject_without_large_matrix():
    """Exercise actual row and response limits with a one-cell duplicate source."""
    with pytest.raises(XyzEnergyLimitError, match="data rows"):
        # Short records fit the raw-byte cap and exceed the actual row count.
        # A low-evidence mock removes only the response estimate for this check.
        from unittest.mock import patch
        with patch("transistordatabase.xyz_energy_import.MAX_RESPONSE_BYTES", 100_000_000):
            import_xyz_energy_map(b"0,0,0\n" * 100_001, options())
    with pytest.raises(XyzEnergyLimitError, match="response evidence"):
        import_xyz_energy_map(b"0,0,0\n" * 100_000, options())


def test_request_identity_is_canonical_for_recursive_object_key_order():
    """Equivalent object field order preserves the pure and typed request identity."""
    def reversed_keys(value):
        if isinstance(value, dict):
            return {key: reversed_keys(item) for key, item in reversed(list(value.items()))}
        return value

    config = options()
    expected = import_xyz_energy_map(FIXTURE, config)["request_sha256"]
    reversed_config = reversed_keys(config)
    assert import_xyz_energy_map(FIXTURE, reversed_config)["request_sha256"] == expected
    reply = client().post("/api/analysis/energy-map/xyz",
                          json=reversed_keys(request(metadata=reversed_config)))
    assert reply.status_code == 200
    assert reply.json()["request_sha256"] == expected


@pytest.mark.parametrize("temperature", [0, 25])
def test_condition_number_identity_matches_direct_and_typed_entry_points(temperature):
    """Equivalent numeric condition types retain one normalized request identity."""
    config = options()
    config["conditions"]["t_j_c"] = temperature
    direct = import_xyz_energy_map(FIXTURE, config)
    reply = client().post("/api/analysis/energy-map/xyz", json=request(metadata=config))
    assert reply.status_code == 200
    assert reply.json()["request_sha256"] == direct["request_sha256"]
    config["conditions"]["t_j_c"] = float(temperature)
    assert import_xyz_energy_map(FIXTURE, config)["request_sha256"] == direct["request_sha256"]
