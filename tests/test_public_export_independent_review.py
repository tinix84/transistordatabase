"""Independent numerical, format-consumer, and file ownership checks."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
from scipy.io import loadmat

from transistordatabase.backend.concrete_services import ExportService
from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, FosterThermalModel,
    MeasurementEvidence, RawMeasurementData, SwitchingLossData, ThermalProperties,
    Transistor, TransistorMetadata,
)
from transistordatabase.core.serialization import (
    core_from_json_document, core_to_json_document,
)
from transistordatabase.native_export import render_gecko, render_matlab


def part(scale=1):
    """Build a signed channel and two independent current sweeps in joules."""
    device = Transistor(
        TransistorMetadata("example", "SiC-MOSFET", "lab", "A", "B"),
        ElectricalRatings(900, 65, 45, 160), ThermalProperties(.01, .01),
    )
    curve = np.array([[1.5, -3, 2.5, 0, -2], [6, -12, 22, 0, -4]])
    curve[0] *= scale
    device.switch.channel_data = [ChannelCharacteristics(37, curve, 11)]
    device.diode.channel_data = [ChannelCharacteristics(37, curve.copy(), None)]
    for event, rg, amps, microjoules in [
        ("e_on_data", 4, [13, 5, 17, 1], [19, 15, 31, 3]),
        ("e_off_data", 7, [9, 16, 20, 3], [60, 81, 85, 24]),
    ]:
        record = SwitchingLossData(
            "graph_i_e", 37, 430, 11, v_g_off=-3, r_g=rg,
            graph_i_e=np.array([amps, np.array(microjoules) * 1e-6 * scale]),
        )
        decoy = deepcopy(record)
        decoy.v_g_off = None
        decoy.graph_i_e[1] += .125
        setattr(device.switch, event, [decoy, record])
    return device


def conditions(**overrides):
    """Return one explicitly selected source slice."""
    return {"component": "switch", "t_j": 37, "v_supply": 430, "v_g_on": 11,
            "v_g_off": -3, "r_g_on": 4, "r_g_off": 7} | overrides


@pytest.mark.parametrize("component,gate", [("switch", 11), ("diode", None)])
@pytest.mark.parametrize("i,slope,offset", [
    (-12, .125, -1.5), (-7.5, .125, -1.5), (-4, .5, 0), (-2, .5, 0),
    (0, .25, 0), (2.5, .25, 0), (6, .0625, 1.125), (22, .0625, 1.125),
])
def test_local_models_against_signed_nonuniform_segments(component, gate, i, slope, offset):
    """Check four distinct segment equations on both public model classes."""
    device = part(8)
    model = getattr(device, component).linearize_at_operating_point(37, gate, i)
    assert (model.r_channel, model.v0_channel) == (8 * slope, 8 * offset)
    assert (model.i_channel, model.v_g, model.t_j) == (i, gate, 37)


def test_multi_component_failure_preserves_old_point_and_source():
    """Reject missing gate evidence, duplicate records, and outside currents."""
    device = part()
    device.update_working_point(37, None, -4, "diode")
    original = device.working_point
    for args in [(37, 11, 6, "both"), (38, None, 6, "diode"),
                 (37, 11, 23, "switch"), (37, None, 6, "unknown")]:
        with pytest.raises(ValueError):
            device.update_working_point(*args)
        assert device.working_point is original
    device.switch.channel_data += deepcopy(device.switch.channel_data)
    with pytest.raises(ValueError):
        device.update_working_point(37, 11, 6, "switch")
    assert device.working_point is original


def scl_blocks(blob):
    """Read the block structure used by the repository's shipped SCL example."""
    source = iter(blob.decode("ascii").splitlines())
    blocks = []
    for count_key, tag, rows, conditions_in_block in (
        ("anzMesskurvenPvCOND", "LeitverlusteMesskurve", 2, ["tj"]),
        ("anzMesskurvenPvSWITCH", "SchaltverlusteMesskurve", 3, ["tj", "uBlock"]),
    ):
        key, count = next(source).split()
        assert key == count_key
        for _ in range(int(count)):
            assert next(source) == f"<{tag}>"
            key, r, c, *numbers = next(source).split()
            assert (key, int(r)) == ("data[][]", rows)
            table = np.asarray(numbers, dtype=float).reshape(rows, int(c))
            observed = {}
            for expected in conditions_in_block:
                key, value = next(source).split()
                assert key == expected
                observed[key] = float(value)
            assert next(source) == f"<\\{tag}>"
            blocks.append((table, observed))
    assert list(source) == []
    return blocks


@pytest.mark.parametrize("scale", [.0625, 8])
def test_scl_consumption_and_native_manifest(scale):
    """Check independently calculated overlap values and both signed branches."""
    device = part(scale)
    original = core_to_json_document(device)
    rendered = render_gecko(device, conditions())
    (channel, cond), (energy, switching_cond) = scl_blocks(rendered["example_Switch.scl"])
    assert (cond, switching_cond) == ({"tj": 37}, {"tj": 37, "uBlock": 430})
    np.testing.assert_array_equal(channel[0], np.array([-3, -2, 0, 1.5, 2.5]) * scale)
    np.testing.assert_array_equal(channel[1], [-12, -4, 0, 6, 22])
    np.testing.assert_array_equal(energy[0], [3, 5, 9, 13, 16, 17])
    np.testing.assert_allclose(energy[1:] / (scale * 1e-6),
                               [[9, 15, 17, 19, 28, 31], [24, 36, 60, 72, 81, 82]],
                               atol=0, rtol=1e-14)
    manifest = json.loads(rendered["example_Switch.manifest.json"])
    assert manifest["channel_current_coverage_a"] == [-12, 22]
    assert manifest["switching_current_coverage_a"] == [3, 17]
    assert manifest["selection"] == conditions()
    assert manifest["core_document"] == original == core_to_json_document(device)


def test_scl_parser_consumes_shipped_example():
    """Anchor the review reader to a pre-existing file, independent of rendering."""
    example = Path(__file__).parents[1] / "template_example/CREE_C3M0016120K_Switch.scl"
    blocks = scl_blocks(example.read_bytes())
    assert blocks[0][0].shape == (2, 35)
    assert blocks[0][1] == {"tj": -40}


def mat_sections(device):
    """Read actual MAT5 bytes with SciPy."""
    data, = render_matlab(device).values()
    assert data.startswith(b"MATLAB 5.0 MAT-file")
    mat = loadmat(BytesIO(data), simplify_cells=True)
    return mat["example"], json.loads(mat["core_json"])


def test_matlab_numeric_distinguishes_absent_and_recorded_zero_thermal_resistance():
    """Unknown thermal resistance must not look like a perfect thermal path."""
    device = part()
    device.diode.thermal_foster = FosterThermalModel(r_th_total=0)
    numeric, native = mat_sections(device)
    assert np.isnan(numeric["switch"]["thermal_foster"]["r_th_total"])
    assert numeric["diode"]["thermal_foster"]["r_th_total"] == 0
    assert native == core_to_json_document(device)


@pytest.mark.parametrize("empty_shape", [(2, 1), (2, 0, 3), (0, 3)])
def test_matlab_native_evidence_and_array_shapes(empty_shape):
    """Restore bytes, UTC, unknown conditions, large integer, dtype, and shape."""
    device = part()
    device.switch.metadata["owned"] = np.zeros(empty_shape, dtype=">u2")
    evidence = MeasurementEvidence(
        filename="∆trace.raw", original_base64="AAEC/w==", event="turn_off",
        step_values={"counter": 2**60 + 1}, v_g_off=None, window_s=(0., 1e-7),
    )
    device.switch.raw_measurement_data = [RawMeasurementData(
        "dpt", measurement_date=datetime(2025, 12, 1, 17, 26, tzinfo=timezone.utc),
        time_s=np.array([0., 1e-7]), evidence=evidence,
    )]
    original = core_to_json_document(device)
    numeric, native = mat_sections(device)
    assert native == original == core_to_json_document(device)
    np.testing.assert_array_equal(numeric["switch"]["channel"]["graph_v_i"],
                                  device.switch.channel_data[0].graph_v_i)
    restored = core_from_json_document(native)
    trace, = restored.switch.raw_measurement_data
    assert trace.evidence == evidence
    assert trace.measurement_date == device.switch.raw_measurement_data[0].measurement_date
    owned = restored.switch.metadata["owned"]
    assert (owned.dtype.str, owned.shape) == (">u2", empty_shape)


def test_exclusive_winner_writes_one_coherent_export(tmp_path, monkeypatch):
    """Race two distinct runs at the same paths and examine their actual files."""
    barrier = Barrier(2)
    previous = os.getcwd()

    def no_chdir(*_args):
        pytest.fail("Export changed the process working directory")

    monkeypatch.setattr(os, "chdir", no_chdir)

    def export(scale):
        device = part(scale)
        device.switch.metadata["run"] = scale
        barrier.wait(timeout=10)
        try:
            ExportService().export_to_gecko_circuits(device, conditions(output_path=tmp_path))
        except FileExistsError:
            return None
        return scale

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(export, [2, 4]))
    assert results.count(None) == 1
    winner, = (result for result in results if result is not None)
    data = (tmp_path / "example_Switch.scl").read_bytes()
    (channel, _), _ = scl_blocks(data)
    assert channel[0, -1] == 2.5 * winner
    manifest = json.loads((tmp_path / "example_Switch.manifest.json").read_bytes())
    assert core_from_json_document(manifest["core_document"]).switch.metadata["run"] == winner
    assert os.getcwd() == previous


@pytest.mark.parametrize("kind", ["bool", "complex", "unicode"])
def test_local_channel_requires_source_samples_with_real_numeric_type(kind):
    """Reject graph types that can produce fabricated real physical units."""
    device = part()
    graphs = {
        "bool": np.array([[False, True], [False, True]]),
        "complex": np.array([[1 + 64j, 2 + 32j], [0, 1]]),
        "unicode": np.array([["1", "2"], ["0", "1"]]),
    }
    device.switch.channel_data[0].graph_v_i = graphs[kind]
    with pytest.raises(ValueError):
        device.switch.linearize_at_operating_point(37, 11, .5)


def test_gecko_does_not_operate_on_boolean_energy_records():
    """Boolean energy arrays must never become a zero-to-one joule curve."""
    device = part()
    for record in (device.switch.e_on_data[1], device.switch.e_off_data[1]):
        record.graph_i_e = np.array([[False, True], [False, True]])
    with pytest.raises(ValueError):
        render_gecko(device, conditions())


def test_gecko_service_needs_destination_before_writing(monkeypatch):
    """A missing destination must fail before the filesystem adapter runs."""
    import transistordatabase.native_export as exporter

    def unexpected_write(*_args):
        pytest.fail("Missing output_path reached the filesystem writer")

    monkeypatch.setattr(exporter, "write_export_files", unexpected_write)
    with pytest.raises(ValueError, match="output_path"):
        ExportService().export_to_gecko_circuits(part(), conditions())


def test_boolean_recorded_gate_does_not_match_zero_volts():
    """A false source condition cannot select a measured zero-gate record."""
    device = part()
    device.switch.channel_data[0].v_g = False
    with pytest.raises(ValueError):
        device.switch.linearize_at_operating_point(37, 0, 1)


def test_boolean_numpy_query_is_not_an_operating_current():
    """Keep NumPy and Python booleans out of the public ampere input."""
    with pytest.raises(ValueError):
        part().switch.linearize_at_operating_point(37, 11, np.bool_(True))


def test_gecko_does_not_relabel_a_boolean_source_gate_as_zero_volts():
    """Validate measured event conditions before writing a numeric gate in SCL evidence."""
    device = part()
    for record in (device.switch.e_on_data[1], device.switch.e_off_data[1]):
        record.v_g_off = False
    with pytest.raises(ValueError):
        render_gecko(device, conditions(v_g_off=0))
