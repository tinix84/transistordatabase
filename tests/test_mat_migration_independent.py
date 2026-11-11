"""Independent MAT shape, calibration, archive, and scientific-domain checks."""
import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import struct
import zlib

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from transistordatabase import legacy_mat
from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.core.serialization import core_from_json_document
from transistordatabase.gui_web.backend.mat_routes import router


ARCHIVE = Path("/home/tinix/claude_wsl/archive_ntbees2/sc")
TEXT_HEADER = b"# Created by Octave fixture\n"
MECHANISMS = {side + mechanism + "Trace" for side in ("forward", "reverse")
              for mechanism in ("Conduction", "TurnOn", "TurnOff")}


def binary(fields):
    """Write one scalar SC record with the independent SciPy writer."""
    stream = BytesIO()
    savemat(stream, {"scData": fields}, long_field_names=True, do_compression=True)
    return stream.getvalue()


def declaration(**changes):
    """Declare a single branch and identify each omitted source mechanism."""
    choice = {
        "kind": "conduction", "branch": "forward", "target": "switch",
        "evidence_reference": "Independent test declaration for this source and selected branch",
        "source": "original-trace", "coordinate_convention": "trace-positive-magnitudes",
        "sample_indices": None, "sample_order": "recorded-order", "zero_policy": "reject-zero",
        "conditions": {"t_j_c": 70., "gate_voltage_v": 15., "gate_off_voltage_v": None,
                       "gate_resistance_ohm": None, "supply_voltage_v": None, "commutation_device": None},
        **changes,
    }
    selected = choice["branch"] + {"conduction": "Conduction", "turn-on": "TurnOn", "turn-off": "TurnOff"}[choice["kind"]] + "Trace"
    return {
        "identity": {"name": "Review sample", "type": "MOSFET", "author": "Reviewer",
                     "manufacturer": "Unlisted vendor", "housing_type": "Declared package"},
        "ratings": {"v_abs_max": 400., "i_abs_max": 80., "i_cont": 40., "t_j_max": 150.},
        "thermal_properties": {"housing_area": 0.01, "cooling_area": 0.01},
        "units": "SC-editor-SI-confirmed", "curves": [choice],
        "declaration_source": "Independent MAT migration test fixture",
        "omission_reasons": {name: "Outside this review" for name in MECHANISMS - {selected}},
    }


def trace(current, voltage):
    """Create a current-first source with a trailing space in the axis label."""
    return {"xLabel": "Current (A) ", "xValues": np.asarray(current),
            "yLabel": "Voltage (V)", "yValues": np.asarray(voltage)}


def test_zero_sized_array_bounds_container_allocation(monkeypatch):
    """A zero trailing dimension must not hide unbounded leading containers."""
    monkeypatch.setattr(legacy_mat, "MAX_ARRAY_ELEMENTS", 16)
    raw = TEXT_HEADER + b"# name: x\n# type: matrix\n# rows: 65\n# columns: 0\n"
    with pytest.raises(ValueError):
        legacy_mat.preview_mat(raw)


def test_octave_2d_shape_includes_row_boundaries():
    """Reject a short row paired with a long row."""
    raw = TEXT_HEADER + b"# name: x\n# type: matrix\n# rows: 2\n# columns: 2\n1 2 3\n4\n"
    with pytest.raises(ValueError):
        legacy_mat.preview_mat(raw)


def test_nested_compression_rejected_before_second_inflation(monkeypatch):
    """Reject the inner compressed tag before starting another inflater."""
    raw = binary({"name": "source"})
    nested = zlib.compress(raw[128:])
    wrapped = raw[:128] + struct.pack("<II", 15, len(nested)) + nested
    original = zlib.decompressobj
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(zlib, "decompressobj", counted)
    with pytest.raises(ValueError):
        legacy_mat.preview_mat(wrapped)
    assert len(calls) == 1


def test_recorded_uint8_descending_current_is_invalid():
    """A descending current domain cannot enter bounded interpolation."""
    fields = {"forwardConductionTrace": trace(np.array([200, 100], dtype="u1"), [2., 1.])}
    with pytest.raises(ValueError):
        legacy_mat.finalize_mat(binary(fields), **declaration())


def test_reverse_int8_minimum_magnitude_is_128():
    """Convert a signed current magnitude without integer overflow."""
    fields = {"onStateCurrent": np.array([-50, -128], dtype="i1"),
              "onStateVoltage": np.array([-1., -2.])}
    transistor = legacy_mat.finalize_mat(binary(fields), **declaration(
        branch="reverse", source="combined", coordinate_convention="signed-passive"))
    np.testing.assert_array_equal(transistor.switch.channel_data[0].graph_v_i, [[1, 2], [50, 128]])


def test_exclude_original_samples_outside_calibration_before_finalization():
    """Retain extrapolated source points and require their explicit exclusion."""
    original = trace([1., 2., 3.], [0.1, 0.2, 0.3])
    original.update(xLocations=[0., 0.5, 1.5], xScale="lin",
                    xLowerLeftCornerLocation=0., xUpperRightCornerLocation=1.,
                    xLowerLeftCornerValue=1., xUpperRightCornerValue=3.)
    raw = binary({"forwardConductionTrace": original})
    report = legacy_mat.preview_mat(raw)["calibrations"]["scData.forwardConductionTrace"]
    assert report["axes"]["x"]["outside_calibration_indices"] == [2]
    with pytest.raises(ValueError):
        legacy_mat.finalize_mat(raw, **declaration())
    selected = legacy_mat.finalize_mat(raw, **declaration(sample_indices=[0, 1]))
    np.testing.assert_array_equal(selected.switch.channel_data[0].graph_v_i, [[0.1, 0.2], [1, 2]])


def test_unknown_source_fields_and_image_survive_repository_and_scipy_roundtrip(tmp_path):
    """Exercise HTTP, repository persistence, bounded interpolation, and compatibility export."""
    image = np.arange(3 * 79 * 47, dtype="u1").reshape((47, 79, 3))
    original = trace([10., 20.], [2., 6.])
    original.update(image=image, xLocations=[[0., 1.]], yLocations=[[1., 0.]])
    raw = binary({"forwardConductionTrace": original, "sensorCode": np.array([[17]], dtype="i2"),
                  "weight": np.empty((0, 1)), "cost": np.array([[0]], dtype="i4")})
    app = FastAPI()
    app.include_router(router)
    api = TestClient(app)
    upload = {"mat_base64": base64.b64encode(raw).decode(), "source_name": "unknown_125C.mat"}
    response = api.post("/api/mat/preview", json=upload)
    assert response.status_code == 200
    preview = response.json()
    assert legacy_mat.archive_from_json(preview) == preview
    fields = preview["variables"]["scData"]["fields"]
    assert fields["weight"]["shape"] == [0, 1] and fields["cost"]["values"] == [[0]]
    image_ref = fields["forwardConductionTrace"]["fields"]["image"]
    blob = api.post("/api/mat/array", json={**upload, "array_path": image_ref["source_array_path"]})
    assert base64.b64decode(blob.json()["data_base64"]) == image.tobytes()
    response = api.post("/api/mat/finalize", json={**upload, **declaration()})
    assert response.status_code == 200, response.text
    document = response.json()["core_json_document"]
    loader = JsonTransistorLoader()
    destination = tmp_path / "native-roundtrip.json"
    loader.save_to_json(core_from_json_document(document), destination)
    device = loader.load_from_json(destination)
    assert device.metadata.name == "Review sample"
    assert device.switch.channel_data[0].get_resistance_at_current(15) == pytest.approx(4 / 15)
    assert device.switch.channel_data[0].t_j == 70
    with pytest.raises(ValueError, match="outside"):
        device.switch.channel_data[0].get_resistance_at_current(5)
    archived = base64.b64decode(device.switch.metadata["legacy_mat_archive"]["source_base64"])
    assert sha256(archived).digest() == sha256(raw).digest()
    assert legacy_mat.export_mat(archived, mode="source-preserved") == raw
    converted = loadmat(BytesIO(legacy_mat.export_mat(archived, mode="legacy-mat5")), struct_as_record=False)["scData"][0, 0]
    np.testing.assert_array_equal(converted.forwardConductionTrace[0, 0].image, image)
    np.testing.assert_array_equal(converted.forwardConductionTrace[0, 0].yLocations, [[1, 0]])
    assert converted.sensorCode.dtype == "i2" and converted.sensorCode.item() == 17
    assert converted.weight.shape == (0, 1) and converted.cost.item() == 0


def test_semantic_export_must_preserve_empty_octave_string_shape_or_report_loss():
    """Prevent an undeclared shape change in a compatible source field."""
    raw = (TEXT_HEADER + b"# name: scData\n# type: scalar struct\n# ndims: 2\n1 1\n# length: 1\n"
           b"# name: note\n# type: sq_string\n# elements: 1\n# length: 0\n\n")
    def get_note(source):
        return legacy_mat.preview_mat(source)["variables"]["scData"]["fields"]["note"]
    assert get_note(raw)["shape"] == [1, 0]
    try:
        output = legacy_mat.export_mat(raw, mode="legacy-mat5")
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("shape", "character", "string"))
    else:
        assert get_note(output) == get_note(raw)
    assert legacy_mat.export_mat(raw, mode="source-preserved") == raw


@pytest.mark.parametrize("name", [b"_note", b"1note"])
def test_export_rejects_field_names_the_mat5_writer_would_skip(name):
    """Keep every source field or report an explicit export error."""
    raw = (TEXT_HEADER + b"# name: scData\n# type: scalar struct\n# ndims: 2\n1 1\n# length: 1\n"
           + b"# name: " + name + b"\n# type: scalar\n42\n")
    assert name.decode() in legacy_mat.preview_mat(raw)["unknown_fields"]
    assert legacy_mat.export_mat(raw, mode="source-preserved") == raw
    with pytest.raises(ValueError, match="field name"):
        legacy_mat.export_mat(raw, mode="legacy-mat5")


@pytest.mark.parametrize("fields", [{"onStateCurrent": "I"}, {"turnOnCurrent": "I", "turnOnEnergy": "E"}])
def test_wrong_typed_sc_fields_retain_archive_and_report_unsupported_science(fields):
    """Separate the typed source preview from unsupported scientific data."""
    raw = binary(fields)
    preview = legacy_mat.preview_mat(raw)
    assert base64.b64decode(preview["source_base64"]) == raw
    assert preview["diagnostics"]
    for key, value in fields.items():
        assert preview["variables"]["scData"]["fields"][key]["values"] == [[value]]
    with pytest.raises(ValueError):
        legacy_mat.finalize_mat(raw, **declaration())


def test_browser_archive_keeps_uint64_outside_exact_js_number_range():
    """Keep integer identity exact across browser JSON parsing."""
    value = 2**63 + 7
    raw = binary({"instrumentId": np.array([[value]], dtype="u8")})
    descriptor = legacy_mat.preview_mat(raw)["variables"]["scData"]["fields"]["instrumentId"]
    assert descriptor["encoding"] == "source-array-reference"
    resolved = legacy_mat.mat_array_blob(raw, descriptor["source_array_path"])
    assert np.frombuffer(base64.b64decode(resolved["data_base64"]), dtype=resolved["dtype"]).item() == value


@pytest.mark.skipif(not ARCHIVE.is_dir(), reason="The immutable SC archive is unavailable")
def test_actual_forward_reference_reverse_negative_energy_and_octave_calibration():
    """Compare the actual source arrays, voltage references, and calibration exclusions."""
    binary_source = (ARCHIVE / "db/scData/Infineon IPX65R190CFDA_125C_HB_Sw+Sw.mat").read_bytes()
    assert sha256(binary_source).hexdigest() == "e0e438c05b2e80583dcda4f777abd80096735e0bb5ad36fe86099e81100f555a"
    conditions = dict(t_j_c=73., gate_voltage_v=15., gate_off_voltage_v=-3., gate_resistance_ohm=2.,
                      supply_voltage_v=175., commutation_device="Declared partner")
    arguments = declaration(kind="turn-off", conditions=conditions)
    transistor = legacy_mat.finalize_mat(binary_source, **arguments)
    record = transistor.switch.e_off_data[0]
    stored = loadmat(BytesIO(binary_source), simplify_cells=True)["scData"]["forwardTurnOffTrace"]
    np.testing.assert_array_equal(record.graph_i_e, [stored["xValues"], stored["yValues"]])
    assert record.v_supply == 175. and record.t_j == 73.
    with pytest.raises(ValueError, match="reference"):
        legacy_mat.finalize_mat(binary_source, **declaration(kind="turn-off", conditions={**conditions, "supply_voltage_v": 400.}))
    with pytest.raises(ValueError, match="[Nn]egative"):
        legacy_mat.finalize_mat(binary_source, **declaration(kind="turn-off", branch="reverse", target="diode",
            conditions={**conditions, "supply_voltage_v": 400.}))
    octave_source = (ARCHIVE / "db/scData/Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat").read_bytes()
    assert sha256(octave_source).hexdigest() == "5b60413bae4f0289b1f44d3771e2cd78293684353cc91b7d000baeb009c849a6"
    with pytest.raises(ValueError):
        legacy_mat.finalize_mat(octave_source, **declaration())
    device = legacy_mat.finalize_mat(octave_source, **declaration(sample_indices=[0, 1]))
    np.testing.assert_array_equal(device.switch.channel_data[0].graph_v_i,
                                  [[0.0350263, 0.980736], [0.2834646, 10.77165]])


@pytest.mark.skipif(not ARCHIVE.is_dir(), reason="The immutable SC archive is unavailable")
def test_sc_source_manifest_fingerprint():
    """Check the path-sensitive hash of all immutable SC MAT files."""
    digest = sha256()
    size = 0
    distinct = set()
    paths = sorted(ARCHIVE.rglob("*.mat"), key=lambda path: path.relative_to(ARCHIVE).as_posix())
    for path in paths:
        source = path.read_bytes()
        fingerprint = sha256(source).hexdigest()
        size += len(source)
        distinct.add(fingerprint)
        digest.update(f"{path.relative_to(ARCHIVE).as_posix()}\0{fingerprint}\n".encode())
    assert len(paths) == 198 and len(distinct) < len(paths)
    assert size == 287_585_927
    assert digest.hexdigest() == "75fb1ff7951defd709175bd944fd7ea3284730d91387f7bd526a16da4b303206"
