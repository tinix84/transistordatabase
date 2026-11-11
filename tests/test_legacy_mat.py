"""Independent format, physics-boundary and immutable source fixtures for MAT."""
import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import zlib

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from transistordatabase.core.serialization import core_from_json_document, core_to_json_document
from transistordatabase.gui_web.backend.mat_routes import router
from transistordatabase.legacy_mat import (
    MAX_EXPANDED_BYTES, MatFormatError, _read, archive_from_json, export_mat,
    finalize_mat, legacy_calibration_report, mat_array_blob, preview_mat,
)

SC = Path("/home/tinix/claude_wsl/archive_ntbees2/sc")


def mat(values, compressed=True):
    """Create an independent data-only MAT5 fixture."""
    buffer = BytesIO()
    savemat(buffer, values, do_compression=compressed, long_field_names=True)
    return buffer.getvalue()


def source(version="0.2"):
    """Keep a signed-current gap and independent original branch samples."""
    trace = {"xValues": np.array([[1., 2.]]), "yValues": np.array([[10., 20.]]),
             "xLabel": "Voltage (V)", "yLabel": "Current (A)"}
    return {"scData": {"name": np.empty((0, 0)), "fileFormatVersion": version,
            "onStateCurrent": np.array([[-20., -10., 10., 20.]]),
            "onStateVoltage": np.array([[-2., -1., 1., 2.]]),
            "turnOnCurrent": np.array([[-1e100, 0., 1e100]]),
            "turnOnEnergy": np.zeros((1, 3)), "turnOnVoltage": 1.,
            "forwardConductionTrace": trace, "reverseConductionTrace": trace,
            "unknownExtra": np.array([[0]], dtype=np.int16)}}


def choices():
    """Declare all inputs independently of the filename and original unknowns."""
    conditions = {"t_j_c": 80., "gate_voltage_v": 12., "gate_off_voltage_v": -3.,
                  "gate_resistance_ohm": None, "supply_voltage_v": None, "commutation_device": None}
    curves = [{"kind": "conduction", "branch": branch, "target": target,
               "source": "combined", "coordinate_convention": "signed-passive",
               "sample_indices": None, "sample_order": "sort-distinct-current",
               "zero_policy": "reject-zero", "conditions": dict(conditions)}
              for branch, target in (("forward", "switch"), ("reverse", "diode"))]
    for curve in curves:
        curve["evidence_reference"] = "Independent signed branch fixture with explicit conditions"
    return {"identity": dict(name="Explicit identity", type="MOSFET", manufacturer="Declared vendor",
                             author="Fixture", housing_type="Declared package"),
            "ratings": dict(v_abs_max=650., i_abs_max=50., i_cont=20., t_j_max=175.),
            "thermal_properties": dict(housing_area=.001, cooling_area=.002, r_th_cs=0.),
            "curves": curves, "units": "SC-editor-SI-confirmed", "declaration_source": "Independent fixture ratings and area declarations",
            "omission_reasons": {branch + kind: "No supported energy convention selected"
                                 for branch in ("forward", "reverse") for kind in ("TurnOnTrace", "TurnOffTrace")}}


def test_archive_unknown_empty_dtype_shape_no_filename_inference():
    """Keep source values, dimensions and original absence through JSON."""
    raw = mat(source())
    document = preview_mat(raw, "fictional_125C_1200V.mat")
    assert document["source"]["sha256"] == sha256(raw).hexdigest()
    assert document["variables"]["scData"]["fields"]["name"]["shape"] == [0, 0]
    assert document["variables"]["scData"]["fields"]["unknownExtra"]["dtype"] == "<i2"
    assert document["unknown_fields"] == ["unknownExtra"]
    assert archive_from_json(json.loads(json.dumps(document))) == document
    changed = deepcopy(document)
    changed["variables"]["scData"]["fields"]["unknownExtra"]["values"] = [[1]]
    with pytest.raises(ValueError, match="authoritative"):
        archive_from_json(changed)
    assert export_mat(raw, mode="source-preserved") == raw


@pytest.mark.parametrize("version", [None, "0.0", "0.1", "0.2"])
def test_old_version_migration_only_adds_empty_areas_and_version(version):
    """Version conversion never fills missing physical conditions or values."""
    values = source(version or "0.0")
    if version is None:
        values["scData"].pop("fileFormatVersion")
    raw = mat(values)
    document = preview_mat(raw)
    assert len(document["conversion_history"]) == (2 if version in (None, "0.0") else 1 if version == "0.1" else 0)
    converted, _ = _read(export_mat(raw, mode="legacy-mat5"))
    sc = converted["scData"]
    assert "".join(sc["fileFormatVersion"].flat) == "0.2"
    np.testing.assert_array_equal(sc["onStateCurrent"], values["scData"]["onStateCurrent"])
    if version in (None, "0.0"):
        assert all(sc[b + "ThermalInterfaceArea"].shape == (0, 0) for b in ("forward", "reverse", "common"))
    assert "t_j" not in sc and sc["name"].size == 0


@pytest.mark.parametrize("version", ["", "0.3", 0.2, np.empty((0, 0))])
def test_invalid_present_version_is_not_treated_as_missing(version):
    """Only an absent version starts the historical 0.0 conversion."""
    raw = mat(source(version))
    document = preview_mat(raw)
    assert not document["conversion_history"]
    assert "unsupported-source-version" in {d["code"] for d in document["diagnostics"]}
    with pytest.raises(ValueError, match="version"):
        export_mat(raw, mode="legacy-mat5")
    with pytest.raises(ValueError, match="version"):
        finalize_mat(raw, **choices())


def test_signed_domains_native_json_roundtrip_and_explicit_conditions():
    """Separate positive domains cannot interpolate through the source current gap."""
    raw = mat(source())
    transistor = finalize_mat(raw, **choices())
    restored = core_from_json_document(json.loads(json.dumps(core_to_json_document(transistor))))
    for component in (restored.switch, restored.diode):
        curve = component.channel_data[0]
        np.testing.assert_array_equal(curve.graph_v_i, [[1., 2.], [10., 20.]])
        assert curve.t_j == 80. and curve.v_g == 12.
        with pytest.raises(ValueError, match="outside"):
            curve.get_resistance_at_current(5.)
    provenance = restored.switch.metadata["legacy_mat_archive"]
    assert base64.b64decode(provenance["source_base64"]) == raw
    assert restored.thermal_properties.r_th_cs == 0.
    assert restored.metadata.name == "Explicit identity"


def test_placeholder_and_negative_energy_never_enter_passive_core():
    """Source preservation does not authorize a zero-loss or negative-heat model."""
    values = source()
    doc = preview_mat(mat(values))
    assert "synthetic-placeholder" in {d["code"] for d in doc["diagnostics"]}
    arguments = choices()
    curve = deepcopy(arguments["curves"][0])
    curve.update(kind="turn-on", conditions={"t_j_c": 25., "gate_voltage_v": 15., "gate_off_voltage_v": 0.,
                 "gate_resistance_ohm": 2., "supply_voltage_v": 1., "commutation_device": None})
    arguments["curves"] = [curve]
    with pytest.raises(ValueError, match="placeholder"):
        finalize_mat(mat(values), **arguments)
    values["scData"].update(turnOnCurrent=np.array([[1., 2.]]), turnOnEnergy=np.array([[-.001, .002]]))
    with pytest.raises(ValueError, match="Negative source energy"):
        finalize_mat(mat(values), **arguments)


def test_real_zero_curve_and_declared_ideal_scope_differ_from_placeholder():
    """An all-zero count alone does not establish a placeholder or a measured model."""
    values = source()
    values["scData"].update(turnOnCurrent=np.array([[1., 2.]]), turnOnEnergy=np.zeros((1, 2)), turnOnVoltage=400.)
    raw = mat(values)
    assert "synthetic-placeholder" not in {d["code"] for d in preview_mat(raw)["diagnostics"]}
    arguments = choices()
    curve = deepcopy(arguments["curves"][0])
    curve.update(kind="turn-on", conditions={"t_j_c": 25., "gate_voltage_v": 15., "gate_off_voltage_v": 0.,
                 "gate_resistance_ohm": 2., "supply_voltage_v": 400., "commutation_device": None})
    arguments["curves"] = [curve]
    arguments["omission_reasons"] = {b + k: "Not selected" for b in ("forward", "reverse")
                                     for k in ("ConductionTrace", "TurnOnTrace", "TurnOffTrace") if b + k != "forwardTurnOnTrace"}
    with pytest.raises(ValueError, match="All-zero combined"):
        finalize_mat(raw, **arguments)
    curve.update(data_status="source-zero-unverified", source_zero_acknowledgement="Source values are zero over 1 to 2 A. Measurement evidence is not established")
    unverified = finalize_mat(raw, **arguments)
    assert unverified.switch.metadata["legacy_mat_archive"]["declarations"]["curves"][0]["data_status"] == "source-zero-unverified"
    curve.pop("source_zero_acknowledgement")
    curve.update(data_status="declared-ideal-model", ideal_model_scope="Explicit zero switching model over 1 to 2 A at this condition")
    transistor = finalize_mat(raw, **arguments)
    np.testing.assert_array_equal(transistor.switch.e_on_data[0].graph_i_e, [[1., 2.], [0., 0.]])
    values["scData"].update(turnOnCurrent=np.array([[-1e100, 0., 1e100]]), turnOnEnergy=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="placeholder"):
        finalize_mat(mat(values), **arguments)


@pytest.mark.parametrize("compressed", [True, False])
def test_mat5_shapes_images_nonfinite_and_no_execution(compressed):
    """Images remain descriptors. Source strings are literal data."""
    values = source()
    values["scData"].update(column=np.array([[1], [2]], dtype=np.int32),
                             expression="__import__('os').system('THIS_MUST_NOT_RUN')",
                             bad=np.array([[float("nan"), float("inf")]]),
                             image=np.arange(9000, dtype=np.uint8).reshape(30, 100, 3))
    doc = preview_mat(mat(values, compressed))
    fields = doc["variables"]["scData"]["fields"]
    assert fields["column"]["shape"] == [2, 1]
    assert fields["bad"]["nonfinite"]
    image = fields["image"]
    assert image["encoding"] == "source-array-reference" and image["shape"] == [30, 100, 3]
    assert "values" not in image and "data_base64" not in image
    blob = mat_array_blob(mat(values, compressed), image["source_array_path"])
    decoded = np.frombuffer(base64.b64decode(blob["data_base64"]), dtype=blob["dtype"]).reshape(blob["shape"])
    np.testing.assert_array_equal(decoded, values["scData"]["image"])
    assert blob["sha256"] == image["sha256"]
    json.dumps(doc, allow_nan=False)


def test_unsupported_objects_cells_complex_hdf5_and_bad_compression():
    """Reject executable object families and resource abuse before SciPy decoding."""
    for values in ({"cell": np.array(["x", 3], dtype=object)}, {"complex": np.array([1 + 2j])}):
        with pytest.raises(MatFormatError, match="Unsupported MATLAB"):
            preview_mat(mat(values))
    with pytest.raises(MatFormatError, match="HDF5"):
        preview_mat(b"\x89HDF\r\n\x1a\n" + bytes(200))
    header = mat({"x": 1.})[:128]
    compressed = zlib.compress(bytes(MAX_EXPANDED_BYTES + 1))
    with pytest.raises(MatFormatError, match="expansion"):
        preview_mat(header + struct.pack("<II", 15, len(compressed)) + compressed)
    truncated = mat(source())[:-5]
    with pytest.raises(MatFormatError):
        preview_mat(truncated)


def independent_matrix(endian, *, shape=(1, 2), name=b"x", class_id=6):
    """Build MAT5 tags without relying on the reader or SciPy writer."""
    def tag(kind, payload):
        return struct.pack(endian + "II", kind, len(payload)) + payload + bytes(-len(payload) % 8)
    payload = tag(6, struct.pack(endian + "II", class_id, 0))
    payload += tag(5, struct.pack(endian + "i" * len(shape), *shape))
    # Exercise the small-element byte order independently.
    payload += struct.pack(endian + "HH", 1, len(name)) + name + bytes(4 - len(name)) if name else tag(1, b"")
    payload += tag(9, np.array([1.25, 2.5], dtype=endian + "f8").tobytes())
    return tag(14, payload)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_independent_endian_small_tags_and_numeric_values(endian):
    """Read independent endian fixtures without byte-swapping physical values."""
    header = mat({"x": 1.})[:124] + struct.pack(endian + "H", 0x100) + (b"MI" if endian == ">" else b"IM")
    raw = header + independent_matrix(endian)
    values, _ = _read(raw)
    np.testing.assert_array_equal(values["x"], [[1.25, 2.5]])
    assert preview_mat(raw)["variables"]["x"]["shape"] == [1, 2]


def test_invalid_preflight_rejects_before_scipy_materialization(monkeypatch):
    """Reject array, object, duplicate-variable and compressed corruption before loadmat."""
    import transistordatabase.legacy_mat as module
    monkeypatch.setattr(module, "loadmat", lambda *a, **k: pytest.fail("SciPy must not materialize invalid data"))
    header = mat({"x": 1.})[:128]
    cases = [header + independent_matrix("<", shape=(100_000_000, 2)),
             header + independent_matrix("<", shape=(8000, 8000, 0)),
             header + independent_matrix("<", shape=(1,) * 9),
             header + independent_matrix("<", class_id=3),
             header + independent_matrix("<") * 2,
             header + struct.pack("<II", 15, 8) + b"not-zlib"]
    def tag(kind, payload):
        return struct.pack("<II", kind, len(payload)) + payload + bytes(-len(payload) % 8)
    duplicate_fields = (tag(6, struct.pack("<II", 2, 0)) + tag(5, struct.pack("<ii", 1, 1)) +
                        tag(1, b"scData") + tag(5, struct.pack("<i", 2)) + tag(1, b"a\0a\0"))
    cases.append(header + tag(14, duplicate_fields))
    for raw in cases:
        with pytest.raises(MatFormatError):
            preview_mat(raw)


def test_duplicate_fields_nesting_and_octave_incomplete_payload():
    """Reject hidden overwrites and incomplete literal workspaces."""
    values = {"x": 1.}
    for _ in range(18):
        values = {"nested": values}
    with pytest.raises(ValueError, match="nesting"):
        preview_mat(mat(values))
    raw = octave_scalar_struct([b"# name: x\n# type: scalar\n1\n", b"# name: x\n# type: scalar\n2\n"])
    with pytest.raises(ValueError, match="Duplicate"):
        preview_mat(raw)
    for payload in (b"1 2", b"1+2 2 3", b"1 2 3 4"):
        raw = octave_scalar_struct([b"# name: x\n# type: matrix\n# rows: 1\n# columns: 3\n" + payload + b"\n"])
        with pytest.raises(ValueError, match="payload|row"):
            preview_mat(raw)


def octave_scalar_struct(fields):
    """Create the reviewed literal Octave text subset independently."""
    return b"# Created by Octave fixture\n# name: scData\n# type: scalar struct\n# ndims: 2\n 1 1\n# length: " + str(len(fields)).encode() + b"\n" + b"".join(fields)


def test_octave_utf8_multiline_metadata_literal_and_pixel_column_major():
    """Respect byte lengths and the original image channel order."""
    text = "Ω\n# name: literal string content"
    encoded = text.encode()
    raw = octave_scalar_struct([
        b"# name: name\n# type: sq_string\n# elements: 1\n# length: " + str(len(encoded)).encode() + b"\n" + encoded + b"\n\n",
        b"# name: image\n# type: uint8 matrix\n# ndims: 3\n 2 3 2\n" + b" ".join(str(i).encode() for i in range(12)) + b"\n",
        b"# name: empty\n# type: matrix\n# rows: 0\n# columns: 0\n\n",
    ])
    values, _ = _read(raw)
    assert "".join(values["scData"]["name"].flat) == text
    np.testing.assert_array_equal(values["scData"]["image"], np.arange(12).reshape((2, 3, 2), order="F"))
    assert values["scData"]["empty"].shape == (0, 0)
    assert export_mat(raw, mode="source-preserved") == raw
    bad = raw.replace(b"# ndims: 3\n 2 3 2", b"# ndims: 3\n 99999999 3 2")
    with pytest.raises(ValueError, match="budget"):
        preview_mat(bad)
    with pytest.raises(ValueError, match="Unsupported Octave type"):
        preview_mat(octave_scalar_struct([b"# name: x\n# type: function handle\n__import__('os')\n"]))


def test_linear_log_calibration_and_legacy_nonsquare_pixels():
    """Independent midpoint values retain inverted y and unusual pixel dimensions."""
    trace = {"image": np.zeros((6, 10, 3), dtype=np.uint8)}
    for axis, scale, a, b, low, high, midpoint in (("x", "log", 1., 100., 0., 1., 10.), ("y", "lin", 10., 30., 1., 0., 20.)):
        trace.update({axis + "Scale": np.array([list(scale)]), axis + "Locations": np.array([[.5]]),
                      axis + "Values": np.array([[midpoint]]),
                      axis + "LowerLeftCornerLocation": np.array([[low]]),
                      axis + "UpperRightCornerLocation": np.array([[high]]),
                      axis + "LowerLeftCornerValue": np.array([[a]]),
                      axis + "UpperRightCornerValue": np.array([[b]])})
    report = legacy_calibration_report(trace)
    assert report["axes"]["x"]["reconstructed_values"] == [10.]
    assert report["axes"]["y"]["reconstructed_values"] == [20.]
    assert report["axes"]["x"]["legacy_pixel_locations"] == [3.]
    assert report["axes"]["y"]["legacy_pixel_locations"] == [5.]
    trace["xLowerLeftCornerValue"] = np.array([[0.]])
    assert legacy_calibration_report(trace)["axes"]["x"]["status"] == "unsupported-calibration"
    trace["yLowerLeftCornerLocation"] = np.array([[-1e308]])
    trace["yUpperRightCornerLocation"] = np.array([[1e308]])
    assert legacy_calibration_report(trace)["axes"]["y"]["status"] == "unsupported-calibration"


def test_http_strict_inputs_preview_download_and_native_finalization():
    """Typed HTTP equals Python and rejects bool, nonfinite and server paths."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    raw = mat(source())
    payload = {"mat_base64": base64.b64encode(raw).decode(), "source_name": "125C.mat"}
    response = client.post("/api/mat/preview", json=payload)
    assert response.status_code == 200 and response.json() == preview_mat(raw, "125C.mat")
    assert client.post("/api/mat/export", json={**payload, "mode": "source-preserved"}).content == raw
    blob = client.post("/api/mat/array", json={**payload, "array_path": ["scData", "onStateCurrent"]})
    assert blob.status_code == 200 and blob.json() == mat_array_blob(raw, ["scData", "onStateCurrent"])
    assert client.post("/api/mat/array", json={**payload, "array_path": ["scData", "missing"]}).status_code == 422
    final = {**payload, **choices()}
    response = client.post("/api/mat/finalize", json=final)
    assert response.status_code == 200, response.text
    assert core_from_json_document(response.json()["core_json_document"]).switch.channel_data[0].t_j == 80.
    bad = deepcopy(final)
    bad["ratings"]["v_abs_max"] = True
    assert client.post("/api/mat/finalize", json=bad).status_code == 422
    encoded = json.dumps(final).replace('"t_j_c": 80.0', '"t_j_c": NaN')
    assert client.post("/api/mat/finalize", content=encoded, headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/api/mat/preview", json={**payload, "path": "/etc/passwd"}).status_code == 422
    assert client.post("/api/mat/preview", json={"mat_base64": "!"}).status_code == 422
    for location in ("declaration_source", "evidence_reference"):
        bad = deepcopy(final)
        if location == "declaration_source":
            bad[location] = " "
        else:
            bad["curves"][0][location] = " "
        assert client.post("/api/mat/finalize", json=bad).status_code == 422


def test_uploaded_mapping_bounded_calculation_uses_existing_core():
    """A selected 15 A point gives 4/15 ohm. A 5 A query remains unsupported."""
    values = source()
    values["scData"]["onStateVoltage"] = np.array([[-6., -2., 2., 6.]])
    raw = mat(values)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    payload = {"mat_base64": base64.b64encode(raw).decode(), **choices(), "target": "switch", "current_a": 15.}
    response = client.post("/api/mat/calculate", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["resistance_ohm"] == pytest.approx(4 / 15)
    assert result["voltage_v"] == pytest.approx(4.) and result["current_coverage_a"] == [10., 20.]
    assert result["resistance_ohm"] == finalize_mat(raw, **choices()).switch.channel_data[0].get_resistance_at_current(15.)
    assert result["source_sha256"] == sha256(raw).hexdigest()
    assert result["evidence_reference"] == choices()["curves"][0]["evidence_reference"]
    assert client.post("/api/mat/calculate", json={**payload, "current_a": 5.}).status_code == 422
    assert client.post("/api/mat/calculate", json={**payload, "current_a": True}).status_code == 422


@pytest.mark.skipif(not SC.is_dir(), reason="Read-only SC archive is not mounted")
def test_actual_198_census_all_previews_and_immutable_fingerprint():
    """Preview all actual workspaces and independently rehash original bytes."""
    before, after, formats, sc_count = [], [], {}, 0
    for path in sorted(SC.rglob("*.mat"), key=lambda p: p.relative_to(SC).as_posix()):
        raw = path.read_bytes()
        relative = path.relative_to(SC).as_posix()
        digest = sha256(raw).hexdigest()
        before.append(relative + "\0" + digest + "\n")
        document = preview_mat(raw, relative)
        assert document["source"]["sha256"] == digest
        formats[document["source"]["format"]] = formats.get(document["source"]["format"], 0) + 1
        if "scData" in document["variables"]:
            sc_count += 1
            assert len(document["known_fields"]) == 25
        # Descriptor references keep full decoded images out of JSON number lists.
        assert len(json.dumps(document)) < 2 * len(raw) + 300_000
        after.append(relative + "\0" + sha256(path.read_bytes()).hexdigest() + "\n")
    fingerprint = sha256("".join(before).encode()).hexdigest()
    assert fingerprint == "75fb1ff7951defd709175bd944fd7ea3284730d91387f7bd526a16da4b303206"
    assert before == after and len(before) == 198 and sc_count == 161
    assert formats == {"mat5": 174, "octave-text-subset": 24}


@pytest.mark.skipif(not SC.is_dir(), reason="Read-only SC archive is not mounted")
def test_actual_placeholder_conflicting_references_negative_trace_and_roundtrip():
    """Actual problematic physics stays visible despite successful format preview."""
    placeholder = (SC / "db/scData/Infineon_IPW65R048CFDA_125C_Max.mat").read_bytes()
    assert sum(d["code"] == "synthetic-placeholder" for d in preview_mat(placeholder)["diagnostics"]) == 2
    raw = (SC / "db/scData/Infineon IPX65R190CFDA_125C_HB_Sw+Sw.mat").read_bytes()
    document = preview_mat(raw)
    conflict = [d for d in document["diagnostics"] if d["code"] == "conflicting-reference-voltages"]
    assert any("175.0" in d["message"] and "400.0" in d["message"] for d in conflict)
    converted = export_mat(raw, mode="legacy-mat5")
    original_values = loadmat(BytesIO(raw), simplify_cells=True)["scData"]
    converted_values = loadmat(BytesIO(converted), simplify_cells=True)["scData"]
    assert original_values.keys() == converted_values.keys()
    for key in ("onStateVoltage", "onStateCurrent", "turnOnEnergy", "turnOffEnergy"):
        np.testing.assert_array_equal(original_values[key], converted_values[key])
    np.testing.assert_array_equal(original_values["forwardConductionTrace"]["image"], converted_values["forwardConductionTrace"]["image"])


@pytest.mark.skipif(not SC.is_dir(), reason="Read-only SC archive is not mounted")
def test_actual_octave_complete_semantic_mat5_roundtrip_and_largest_partial():
    """Convert the actual complete text record without altering arrays or images."""
    raw = (SC / "db/scData/Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat").read_bytes()
    before, _ = _read(raw)
    after, _ = _read(export_mat(raw, mode="legacy-mat5"))

    def equal_values(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_values(left[key], right[key])
        else:
            assert left.shape == right.shape
            assert left.dtype.kind == right.dtype.kind and left.dtype.itemsize == right.dtype.itemsize
            np.testing.assert_array_equal(left, right)

    equal_values(before, after)
    largest = (SC / "db/scData/ROHM_SCT3022AL_Eon.mat").read_bytes()
    assert len(largest) == 15_611_926
    document = preview_mat(largest)
    assert document["workspace_schema"] == "digitizer-traces"
    assert document["source"]["byte_length"] == len(largest)
    with pytest.raises(ValueError, match="scalar scData"):
        export_mat(largest, mode="legacy-mat5")
