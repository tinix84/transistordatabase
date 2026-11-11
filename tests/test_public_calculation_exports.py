"""Independent numeric and consumer-schema checks for public headless adapters."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from io import BytesIO
import json
import os

import numpy as np
import pytest
from scipy.io import loadmat

from transistordatabase.backend.concrete_services import ExportService
from transistordatabase.core.models import (
    ChannelCharacteristics, Diode, ElectricalRatings, MeasurementEvidence,
    Switch, SwitchingLossData, ThermalProperties, Transistor, TransistorMetadata,
)
from transistordatabase.core.serialization import (
    core_from_json_document, core_to_json_document,
)
from transistordatabase.native_export import render_gecko, render_matlab


def device(name="test-part"):
    """Construct recorded curves with distinct bounded current domains."""
    transistor = Transistor(
        TransistorMetadata(name, "IGBT", "test", "source manufacturer", "source housing",
                           datasheet_date=datetime(2026, 1, 1)),
        ElectricalRatings(1200, 100, 50, 175), ThermalProperties(0.001, 0.001))
    transistor.switch.channel_data = [ChannelCharacteristics(
        25, np.array([[-3., -1., 1., 4.], [-20., -10., 0., 10.]]), 15)]
    transistor.diode.channel_data = [ChannelCharacteristics(
        25, np.array([[1., 2.], [0., 10.]]), None)]
    transistor.switch.e_on_data = [SwitchingLossData(
        "graph_i_e", 25, 600, 15, r_g=2, v_g_off=-4,
        graph_i_e=np.array([[1., 5., 10.], [0.001, 0.003, 0.010]]),
        comment="Recorded source on curve")]
    transistor.switch.e_off_data = [SwitchingLossData(
        "graph_i_e", 25, 600, 15, r_g=3, v_g_off=-4,
        graph_i_e=np.array([[2., 6., 12.], [0.002, 0.006, 0.012]]),
        comment="Recorded source off curve")]
    return transistor


def selection(**changes):
    """Declare every scientific condition for one supported switch slice."""
    return dict(component="switch", t_j=25, v_supply=600, v_g_on=15,
                v_g_off=-4, r_g_on=2, r_g_off=3, **changes)


@pytest.mark.parametrize("component", [Switch, Diode])
@pytest.mark.parametrize("current,resistance,offset", [
    (-20, .2, 1), (-15, .2, 1), (-10, .2, 1), (0, .3, 1), (10, .3, 1),
])
def test_signed_local_line_and_one_sided_endpoints(component, current, resistance, offset):
    """Check exact local slopes, signed intercepts, and one-sided endpoint derivatives."""
    item = component()
    # Deliberately unsorted, with distinct slopes at I=0.
    item.channel_data = [ChannelCharacteristics(
        25, np.array([[4., -1., 1., -3.], [10., -10., 0., -20.]]), 15)]
    result = item.linearize_at_operating_point(25, 15, current)
    assert result.r_channel == pytest.approx(resistance)
    assert result.v0_channel == pytest.approx(offset)
    assert result.v0_channel + result.r_channel * current == pytest.approx(
        {-20: -3, -15: -2, -10: -1, 0: 1, 10: 4}[current])


@pytest.mark.parametrize("component", [Switch, Diode])
def test_unknown_gate_is_not_a_wildcard_and_duplicates_fail(component):
    """Separate unknown and zero gate conditions and reject ambiguous evidence."""
    item = component()
    item.channel_data = [ChannelCharacteristics(25, np.array([[0., 1.], [0., 10.]]), gate)
                         for gate in (None, 0, 15)]
    assert [r.v_g for r in item.get_channel_data(25)] == [None]
    assert [r.v_g for r in item.get_channel_data(25, 0)] == [0]
    assert item.linearize_at_operating_point(25, None, 5).v_g is None
    item.channel_data.append(deepcopy(item.channel_data[-1]))
    with pytest.raises(ValueError, match="unambiguous"):
        item.linearize_at_operating_point(25, 15, 5)
    assert item.get_channel_data(26, 15) == []
    with pytest.raises(ValueError):
        item.linearize_at_operating_point(26, 15, 5)


@pytest.mark.parametrize("current", [-21, 11, float("nan"), float("inf"), True, None])
def test_local_line_rejects_outside_or_nonfinite_queries(current):
    """Reject clamping, booleans, and nonfinite operating currents."""
    with pytest.raises(ValueError):
        device().switch.linearize_at_operating_point(25, 15, current)


@pytest.mark.parametrize("graph", [
    [[1], [2]], [[1, 2], [2, 2]], [[1, float("nan")], [1, 2]],
    [[1, 2], [1, float("inf")]], [[1, 2, 3]],
])
def test_local_line_rejects_zero_span_and_invalid_samples(graph):
    """Reject duplicate currents, malformed shapes, and nonfinite source samples."""
    item = Switch()
    item.channel_data = [ChannelCharacteristics(25, np.array(graph), 15)]
    with pytest.raises(ValueError):
        item.linearize_at_operating_point(25, 15, 2)


@pytest.mark.parametrize("current,voltage", [
    ([-1e308, 1e308], [1, 2]), ([0, 1e308], [0, 1e-308]),
    ([0, 1], [-1e308, 1e308]),
])
def test_local_line_rejects_overflow_and_false_zero_slopes(current, voltage):
    """Reject unrepresentable spans, slopes, and offsets instead of false zero resistance."""
    item = Switch()
    item.channel_data = [ChannelCharacteristics(25, np.array([voltage, current]), 15)]
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite numeric range"):
            item.linearize_at_operating_point(25, 15, 0)


@pytest.mark.parametrize("temperature,gate", [(float("nan"), 15), (25, float("inf")), (True, 15), (25, False)])
def test_condition_inputs_must_be_finite(temperature, gate):
    """Reject invalid scientific conditions before record selection."""
    with pytest.raises(ValueError):
        device().switch.get_channel_data(temperature, gate)


def test_working_point_failure_is_atomic_and_component_is_validated():
    """Preserve a prior working point when a selected component fails."""
    transistor = device()
    transistor.update_working_point(25, 15, 5, "switch")
    previous = transistor.working_point
    # Switch succeeds, diode at the requested known gate is unavailable.
    for component in ("both", "bogus", "SWITCH", None):
        with pytest.raises(ValueError):
            transistor.update_working_point(25, 15, 5, component)
        assert transistor.working_point is previous
    transistor.update_working_point(25, None, 5, "diode")
    assert set(transistor.working_point) == {"diode"}


def test_mat5_standard_consumer_and_complete_native_roundtrip():
    """Read numeric fields with SciPy and restore complete native evidence."""
    transistor = device()
    transistor.switch.metadata = {"source": "fixture", "unknown": None, "shape": [1, 2]}
    transistor.switch.e_on_data[0].measurement_date = datetime(2026, 11, 10, 12, 34)
    transistor.switch.e_on_data[0].evidence = MeasurementEvidence(
        source_kind="simulation", filename="original.raw", sha256="a" * 64,
        original_base64="c291cmNl", parser="fixture", parser_version="1",
        window_s=(1e-6, 2e-6), event="turn_on", t_j=25,
        disclosure="Generic model. No measured device accuracy.")
    original = core_to_json_document(transistor)
    rendered = render_matlab(transistor)
    assert set(rendered) == {"test_part_Matlab.mat"}
    loaded = loadmat(BytesIO(next(iter(rendered.values()))), simplify_cells=True)
    numeric = loaded["test_part"]
    np.testing.assert_array_equal(numeric["switch"]["channel"]["graph_v_i"],
                                  transistor.switch.channel_data[0].graph_v_i)
    np.testing.assert_array_equal(numeric["switch"]["e_on"]["graph_i_e"],
                                  transistor.switch.e_on_data[0].graph_i_e)
    assert np.isnan(numeric["r_g_int"])
    assert np.isnan(numeric["diode"]["t_j_max"])
    assert np.isnan(numeric["diode"]["channel"]["v_g"])
    native = json.loads(loaded["core_json"])
    assert native == original
    assert core_to_json_document(core_from_json_document(native)) == original
    assert core_to_json_document(transistor) == original


def parse_scl(data):
    """Read the historical SCL block grammar independently of the renderer."""
    lines = data.decode().splitlines()
    assert lines[0] == "anzMesskurvenPvCOND 1"
    assert lines[1] == "<LeitverlusteMesskurve>"
    assert lines[4] == "<\\LeitverlusteMesskurve>"
    assert lines[5] == "anzMesskurvenPvSWITCH 1"
    assert lines[6] == "<SchaltverlusteMesskurve>"
    assert lines[10] == "<\\SchaltverlusteMesskurve>"
    arrays = []
    for index, rows in ((2, 2), (7, 3)):
        tokens = lines[index].split()
        assert tokens[0] == "data[][]"
        assert int(tokens[1]) == rows
        count = int(tokens[2])
        values = np.array([float(value) for value in tokens[3:]])
        assert len(values) == rows * count
        arrays.append(values.reshape(rows, count))
    assert float(lines[3].split()[1]) == 25
    assert float(lines[8].split()[1]) == 25
    assert float(lines[9].split()[1]) == 600
    return arrays


def test_gecko_schema_bounded_common_grid_precision_and_immutable_evidence():
    """Check SCL grammar and exact piecewise values without source mutation."""
    transistor = device()
    before = core_to_json_document(transistor)
    options = np.get_printoptions()
    rendered = render_gecko(transistor, selection())
    conduction, switching = parse_scl(rendered["test_part_Switch.scl"])
    np.testing.assert_array_equal(conduction, transistor.switch.channel_data[0].graph_v_i)
    np.testing.assert_array_equal(switching[0], [2, 5, 6, 10])
    # Independent piecewise equations for distinct on/off axes, with no I=0 extrapolation.
    np.testing.assert_allclose(switching[1], [.0015, .003, .0044, .010], rtol=1e-14)
    np.testing.assert_allclose(switching[2], [.002, .005, .006, .010], rtol=1e-14)
    manifest = json.loads(rendered["test_part_Switch.manifest.json"])
    assert manifest["selection"] == selection()
    assert manifest["switching_current_coverage_a"] == [2, 10]
    assert manifest["channel_current_coverage_a"] == [-20, 10]
    assert manifest["core_document"] == before
    assert core_to_json_document(transistor) == before
    assert np.get_printoptions() == options


@pytest.mark.parametrize("changes", [
    {"t_j": 26}, {"v_supply": 601}, {"v_g_on": None}, {"v_g_off": None},
    {"r_g_on": 0}, {"r_g_off": None}, {"recheck": True}, {"component": "diode"},
    {"t_j": float("nan")}, {"arbitrary": 1},
])
def test_gecko_requires_exact_complete_conditions(changes):
    """Reject missing evidence and nearest-condition export requests."""
    params = selection()
    params.update(changes)
    with pytest.raises(ValueError):
        render_gecko(device(), params)


def test_gecko_unknown_matches_only_unknown_and_real_zero_is_preserved():
    """Preserve actual zero-energy evidence at explicitly selected unknown conditions."""
    transistor = device()
    for record in transistor.switch.e_on_data + transistor.switch.e_off_data:
        record.v_g_off = None
        record.r_g = 0
        record.graph_i_e[1] = 0
    params = selection()
    params.update(v_g_off=None, r_g_on=0, r_g_off=0)
    _, switching = parse_scl(render_gecko(transistor, params)["test_part_Switch.scl"])
    np.testing.assert_array_equal(switching[1:], 0)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "disjoint", "negative", "single", "span"])
def test_gecko_incomplete_science_fails_before_any_write(tmp_path, failure):
    """Reject incomplete scientific inputs before creating destination files."""
    transistor = device()
    if failure == "missing":
        transistor.switch.e_on_data = []
    elif failure == "duplicate":
        transistor.switch.e_on_data *= 2
    elif failure == "disjoint":
        transistor.switch.e_off_data[0].graph_i_e[0] += 20
    elif failure == "negative":
        transistor.switch.e_on_data[0].graph_i_e[1, 0] = -1
    elif failure == "single":
        transistor.switch.e_on_data[0].dataset_type = "single"
    else:
        transistor.switch.channel_data[0].graph_v_i[1] = 0
    destination = tmp_path / "unwritten"
    with pytest.raises(ValueError):
        ExportService().export_to_gecko_circuits(transistor, selection(output_path=destination))
    assert not destination.exists()


def test_services_use_explicit_paths_concurrently_without_global_mutation(tmp_path, monkeypatch):
    """Run concurrent public exports with every process-wide directory change blocked."""
    previous = os.getcwd()
    def reject_chdir(*args):
        raise AssertionError("Export attempted process-wide chdir")
    monkeypatch.setattr(os, "chdir", reject_chdir)
    service = ExportService()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(service.export_to_matlab, device(), tmp_path / f"mat-{i}")
                   for i in range(2)]
        futures += [executor.submit(service.export_to_gecko_circuits, device(),
                                    selection(output_path=tmp_path / f"gecko-{i}"))
                    for i in range(2)]
        for future in futures:
            future.result()
    assert os.getcwd() == previous
    assert len(list(tmp_path.rglob("*.mat"))) == 2
    assert len(list(tmp_path.rglob("*.scl"))) == 2
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_all_existing_targets_are_checked_before_writing(tmp_path):
    """Protect existing source files and prevent collision-related partial exports."""
    manifest = tmp_path / "test_part_Switch.manifest.json"
    manifest.write_bytes(b"irreplaceable existing source")
    with pytest.raises(FileExistsError):
        ExportService().export_to_gecko_circuits(device(), selection(output_path=tmp_path))
    assert not (tmp_path / "test_part_Switch.scl").exists()
    assert manifest.read_bytes() == b"irreplaceable existing source"
    matlab = tmp_path / "test_part_Matlab.mat"
    matlab.write_bytes(b"source mat")
    with pytest.raises(FileExistsError):
        ExportService().export_to_matlab(device(), tmp_path)
    assert matlab.read_bytes() == b"source mat"


@pytest.mark.parametrize("name", ["../../outside", "C:\\source\\part", "CON", "_part", "core_json"])
def test_names_are_safe_matlab_identifiers_and_plain_filenames(name):
    """Preserve source names as data while using safe destination identifiers."""
    rendered = render_matlab(device(name))
    filename = next(iter(rendered))
    assert "/" not in filename and "\\" not in filename
    loaded = loadmat(BytesIO(rendered[filename]), simplify_cells=True)
    key = filename.removesuffix("_Matlab.mat")
    assert key[0].isalpha() and key != "core_json"
    assert loaded[key]["name"] == name
