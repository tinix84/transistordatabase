"""Independent source, numerical, transport, and native archive measurement checks."""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from spicelib.raw.raw_write import RawWrite, Trace

from transistordatabase.core.models import MeasurementEvidence, SwitchingLossData
from transistordatabase.core.repository import (
    JsonTransistorLoader, JsonTransistorRepository, TransistorFactory,
)
from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.gui_web.backend import main


def source_trace():
    """Two known commutations, different local currents, and a later larger pulse."""
    ns = np.arange(1001, dtype=float)
    knots = [0, 200, 250, 600, 650, 800, 850, 900, 1000]
    voltage = np.interp(ns, knots, [1, 1, 48, 48, 1, 1, 1, 48, 48])
    current = np.interp(ns, knots, [5, 5, 0, 0, 8, 18, 18, 0, 0])
    gate = np.interp(ns, knots, [12, 12, -2, -2, 12, 12, 12, -2, -2])
    return ns * 1e-9, voltage, current, gate


def csv_trace(*, gate_rails=(-2., 12.)):
    """Include BOM, non-ASCII text, quoted separators, multiline text, and reverse probe."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";", quoting=csv.QUOTE_ALL)
    writer.writerow(["time;ns", "Vds", 'probe "B"', "Vgs", "note"])
    for n, (_, v, i, g) in enumerate(zip(*source_trace(), strict=True)):
        low, high = gate_rails
        values = [n, v, -1000 * i, low + (g + 2) * (high - low) / 14]
        writer.writerow([format(x, ".15g").replace(".", ",") for x in values] +
                        ["observation; Δ\nquoted" if n == 0 else ""])
    return stream.getvalue().encode("utf-8-sig")


def raw_trace(tmp_path, *, steps=1, stepped_flag=False):
    """Use spicelib to write actual binary data, including deliberate reset-time sweeps."""
    writer = RawWrite()
    writer.flag_stepped = stepped_flag
    time, voltage, current, gate = source_trace()
    for name, samples, kind in [
        ("time", time, "time"), ("V(ds)", voltage, "voltage"),
        ("I(probe)", current, "current"), ("V(gs)", gate, "voltage"),
    ]:
        runs = [samples * (index + 1) if kind == "current" else samples
                for index in range(steps)]
        writer.add_trace(Trace(name, np.concatenate(runs), whattype=kind))
    path = tmp_path / "analytic.raw"
    assert not path.exists()
    writer.save(path)
    return path.read_bytes()


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Build a real native repository in a fresh test directory."""
    repo = JsonTransistorRepository(tmp_path / "isolated")
    monkeypatch.setattr(main, "_repo", repo)
    return TestClient(main.app, raise_server_exceptions=False), repo


def independent_energy(time, voltage, current, start, end):
    """Integrate individual sample intervals with clipped boundary voltages and currents."""
    total = 0.0
    for j in range(1, len(time)):
        left = max(start, time[j-1])
        right = min(end, time[j])
        if right <= left:
            continue
        v_l, v_r = np.interp([left, right], time[j-1:j+1], voltage[j-1:j+1])
        i_l, i_r = np.interp([left, right], time[j-1:j+1], current[j-1:j+1])
        total += (right - left) * (v_l * i_l + v_r * i_r) / 2
    return total


def request_for(fmt, source):
    """Declare the source encoding, columns, units, and signs for each test format."""
    request = {"format": fmt, "filename": f"fixture.{fmt}",
               "content_base64": base64.b64encode(source).decode(),
               "source_kind": "simulation", "disclosure": "Synthetic analytic source."}
    if fmt == "log":
        request["log"] = {"energy_off": "eoff", "current": "idc", "bus": "vdc",
                          "energy_unit": "J", "current_unit": "A", "bus_unit": "V"}
        return request
    if fmt == "csv":
        request["csv"] = {"delimiter": ";", "decimal": ",", "encoding": "utf-8-sig"}
        names = ["time;ns", "Vds", 'probe "B"', "Vgs"]
        units = ["ns", "V", "mA", "V"]
    else:
        names = ["time", "V(ds)", "I(probe)", "V(gs)"]
        units = ["s", "V", "A", "V"]
    roles = ["time", "voltage", "current", "gate"]
    request["mappings"] = {role: {"name": name, "unit": unit,
                           "polarity": -1 if role == "current" and fmt == "csv" else 1}
                           for role, name, unit in zip(roles, names, units, strict=True)}
    return request


CONDITIONS = {"bus_voltage_v": 48., "t_j": 73., "v_g_on": 12., "v_g_off": -2.,
              "r_g_on": 3., "r_g_off": 9., "commutation_device": "Independent fixture",
              "manual_override": "Temperature supplied by reviewer."}


def post(client, operation, payload, status=200):
    """Send JSON bytes without client-side conversion of nonfinite test inputs."""
    response = client.post(f"/api/measurement/{operation}", content=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == status, response.text[:700]
    return response.json()


@pytest.mark.parametrize("fmt", ["csv", "raw"])
def test_waveform_energy_and_native_archive(api, tmp_path, fmt):
    """Verify independent event integrals and every reloaded waveform sample."""
    client, repo = api
    source = csv_trace() if fmt == "csv" else raw_trace(tmp_path)
    request = request_for(fmt, source)
    time, voltage, current, gate = source_trace()
    preview = post(client, "preview", request)
    for key, values in zip(["time_s", "voltage_v", "current_a", "gate_v"],
                           [time, voltage, current, gate], strict=True):
        np.testing.assert_allclose(preview["waveform"][key], values, atol=1e-15, rtol=1e-7)
    device = TransistorFactory.create_empty_transistor("Independent-DPT")
    result = post(client, "extract", {**request, "conditions": CONDITIONS,
                                      "device_document": core_to_json_document(device)})
    assert repo.list_all() == []
    native = result["core_json_document"]
    archive = JsonTransistorLoader().load_from_dict(native)
    repo.save(archive)
    saved = repo.get_by_name(archive.metadata.name)
    export_path = tmp_path / "export.json"
    assert main._export.export_to_json(saved, export_path)
    final = JsonTransistorLoader().load_from_json(export_path)
    assert core_to_json_document(final) == native
    [raw] = final.switch.raw_measurement_data
    [on], [off] = final.switch.e_on_data, final.switch.e_off_data
    assert [on.i_x, off.i_x] == pytest.approx([7.92, 5])
    assert [on.r_g, off.r_g] == [3, 9]
    for array, name in [(raw.voltage_v, "voltage_v"), (raw.current_a, "current_a"),
                        (raw.gate_v, "gate_v"), (raw.time_s, "time_s")]:
        np.testing.assert_array_equal(array, preview["waveform"][name])
    for event, point, bounds in [
        ("turn_on", on, (600.495e-9, 649.5e-9)),
        ("turn_off", off, (200.5e-9, 249.5e-9)),
    ]:
        evidence = point.evidence
        assert evidence.event == event
        assert evidence.window_s == pytest.approx(bounds, abs=1e-15)
        assert evidence.thresholds and evidence.convention
        expected = independent_energy(time, voltage, current, *bounds)
        assert point.e_x == pytest.approx(expected, rel=3e-7)
        assert evidence.local_current_a == point.i_x
    for evidence in (raw.evidence, on.evidence, off.evidence):
        assert evidence.sha256 == hashlib.sha256(source).hexdigest()
        assert base64.b64decode(evidence.original_base64) == source
        assert evidence.parser_version and evidence.parser
        assert evidence.trace_mapping["signals"] == request["mappings"]
        assert evidence.source_kind == request["source_kind"]
        assert evidence.disclosure == request["disclosure"]
        assert evidence.manual_override == CONDITIONS["manual_override"]
        assert (evidence.t_j, evidence.v_g_off, evidence.r_g_off) == (73, -2, 9)


@pytest.mark.parametrize("change", ["missing_gate", "unknown_gate", "short_row", "overflow",
                                    "nonmonotone", "invalid_decimal", "blank_header"])
def test_csv_invalid_inputs_never_extract(api, change):
    """Reject malformed cells or missing evidence at the relevant operation."""
    client, _ = api
    source = csv_trace()
    if change == "short_row":
        source += b'"1001";"1";"0";"0"\r\n'
    elif change == "overflow":
        source += b'"1001";"1e999";"0";"0";""\r\n'
    elif change == "nonmonotone":
        source += b'"5";"1";"0";"0";""\r\n'
    elif change == "invalid_decimal":
        source += b'"1001";"2.5";"0";"0";""\r\n'
    elif change == "blank_header":
        source = source.replace(b'"note"', b'""', 1)
    request = request_for("csv", source)
    if change == "missing_gate":
        request["mappings"]["gate"] = None
        assert post(client, "preview", request)["waveform"]["gate_v"] is None
        post(client, "extract", {**request, "conditions": CONDITIONS}, 422)
    else:
        if change == "unknown_gate":
            request["mappings"]["gate"]["name"] = "absent"
        post(client, "preview", request, 422)


@pytest.mark.parametrize("key,value", [
    ("v_g_off", False), ("r_g_on", float("nan")), ("bus_voltage_v", float("inf")),
    ("r_g_off", -float("inf")),
])
def test_conditions_are_strict_finite_numbers(api, key, value):
    """Require serializable 422 responses for invalid physical conditions."""
    request = request_for("csv", csv_trace())
    post(api[0], "extract", {**request, "conditions": {**CONDITIONS, key: value}}, 422)


@pytest.mark.parametrize("stepped_flag", [True, False])
def test_reject_missing_companion_or_unmarked_time_reset(tmp_path, api, stepped_flag):
    """Prevent adjacent simulation steps from forming a single waveform."""
    source = raw_trace(tmp_path, steps=2, stepped_flag=stepped_flag)
    post(api[0], "preview", request_for("raw", source), 422)


def test_raw_rejects_missing_or_synthetic_step_metadata(api, tmp_path):
    """Reject an unrelated log even if the parser infers step boundaries."""
    request = request_for("raw", raw_trace(tmp_path, steps=2, stepped_flag=True))
    request["step_index"] = 1
    request["companion_log_base64"] = base64.b64encode(b"Circuit: unrelated\n\n").decode()
    post(api[0], "preview", request, 422)


def test_stepped_raw_uses_exact_selected_run(api, tmp_path):
    """Select only the declared step and preserve its companion identity."""
    request = request_for("raw", raw_trace(tmp_path, steps=2, stepped_flag=True))
    log = b"Circuit: independent\n.step idc=5 vdc=48\n.step idc=10 vdc=48\n"
    request.update(companion_log_base64=base64.b64encode(log).decode(), step_index=1)
    result = post(api[0], "preview", request)
    assert result["available"]["steps"] == [0, 1]
    assert result["source"]["step_values"] == {"idc": 10, "vdc": 48}
    assert result["source"]["companion_log_sha256"] == hashlib.sha256(log).hexdigest()
    assert result["waveform"]["current_a"][0] == 10
    assert len(result["waveform"]["time_s"]) == 1001
    post(api[0], "preview", {**request, "step_index": 2}, 422)


def test_original_log_30_steps_and_failed_integral(api):
    """Compare the original simulated integrals with known historical values."""
    source = (Path(__file__).resolve().parents[2] / "archive_ntbees2/sc/tests/"
              "DoublePulseTester_EPC8009_sync.log")
    if not source.is_file():
        pytest.skip("Historical SC log is absent on this checkout")
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == (
        "08e37487d73c74087d0303872093cbcda6fe316574e67f4fb57b35bd5e26e042")
    client, _ = api
    req = request_for("log", original)
    result = post(client, "preview", req)
    rows = result["log_records"]
    assert len(rows) == 30
    assert [row["current_a"] for row in rows[:3]] == [1, 3, 5]
    assert [row["energy_j"] for row in rows[:3]] == [1.0141e-7, 9.47901e-7, 2.69621e-6]
    assert all(row["window_s"] == [39e-6, 41e-6] and row["expression"] for row in rows)
    assert result["waveform"] is None
    extract = post(client, "extract", {**req, "step_index": 2, "conditions": CONDITIONS})
    [point] = extract["points"]["turn_off"]
    assert not extract["points"]["turn_on"] and extract["raw_record"] is None
    assert (point["e_x"], point["i_x"], point["v_supply"]) == (2.69621e-6, 5, 5)
    assert point["evidence"]["method"] == "supplied-log-integral"
    altered = original.replace(b"9.47901e-007", b"FAILED", 1)
    assert altered != original
    failed = post(client, "preview", request_for("log", altered))["log_records"]
    assert failed[0]["energy_j"] == rows[0]["energy_j"]
    assert failed[1]["status"] == "failed" and failed[1]["energy_j"] is None
    assert failed[2]["energy_j"] == rows[2]["energy_j"]
    assert source.read_bytes() == original


def test_reordered_measure_ordinals_never_attach_to_wrong_current(api):
    """Prevent reordered log rows from acquiring another step's current."""
    text = ("Circuit: sample\n.step idc=1 vdc=8\n.step idc=3 vdc=8\n\n"
            "Measurement: eoff\n step\tINTEG(V(ds)*I(dut))\tFROM\tTO\n"
            "    2\t3e-6\t1e-6\t2e-6\n    1\t1e-6\t1e-6\t2e-6\n")
    response = api[0].post("/api/measurement/preview", json=request_for("log", text.encode()))
    if response.status_code == 200:
        accepted = [(r["current_a"], r["energy_j"]) for r in response.json()["log_records"]
                    if r["status"] != "failed"]
        assert all(pair in [(1, 1e-6), (3, 3e-6)] for pair in accepted)
    else:
        assert response.status_code == 422, response.text[:600]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
@pytest.mark.parametrize("measure", ["eoff", "EOFF"])
def test_unstepped_log_preserves_original_integral_expression(api, encoding, measure):
    """Keep the expression and window from LTspice's one-line measurement syntax."""
    raw = ("Circuit: single\n" + measure +
           ":  INTEG(V(ds)*I(dut))=3e-6 FROM 1e-6 TO 2e-6\n").encode(encoding)
    request = request_for("log", raw)
    request["log"] = {"energy_off": "eoff", "energy_unit": "J"}
    result = post(api[0], "preview", request)
    [record] = result["log_records"]
    assert record["energy_j"] == 3e-6
    assert record["window_s"] == [1e-6, 2e-6]
    assert record["expression"] == "INTEG(V(ds)*I(dut))"


def test_waveform_uses_measured_gate_levels_and_preserves_unknown_resistance(api):
    """Extract from the declared gate rails with unknown gate resistance."""
    request = request_for("csv", csv_trace(gate_rails=(-4., 2.)))
    result = post(api[0], "extract", {**request, "conditions": {
        **CONDITIONS, "v_g_on": 2., "v_g_off": -4., "r_g_on": None, "r_g_off": None}})
    [on], [off] = result["points"]["turn_on"], result["points"]["turn_off"]
    assert (on["r_g"], off["r_g"]) == (None, None)
    assert on["evidence"]["v_g_on"] == 2.
    assert off["evidence"]["v_g_off"] == -4.
    assert on["e_x"] > 0 and off["e_x"] > 0


def test_step_number_bool_and_polarity_bool_never_coerce(api):
    """Keep Boolean values distinct from integer indices and signs."""
    request = request_for("csv", csv_trace())
    post(api[0], "preview", {**request, "step_index": False}, 422)
    request["mappings"]["gate"]["polarity"] = True
    post(api[0], "preview", request, 422)


def test_nonnumeric_step_metadata_cannot_trigger_response_encoding_failure(api):
    """Handle overflow in a retained step variable outside the selected mapping."""
    source = ("Circuit: sample\n.step idc=2 vdc=48 other=1e999\n\n"
              "Measurement: eoff\n step\tINTEG(V(ds)*I(dut))\tFROM\tTO\n"
              "    1\t3e-6\t1e-6\t2e-6\n")
    response = api[0].post("/api/measurement/preview", json=request_for("log", source.encode()))
    assert response.status_code in (200, 422), response.text[:500]
    if response.status_code == 200:
        record = response.json()["log_records"][0]
        assert record["status"] == "failed" or record["step_values"]["other"] == "1e999"


def test_csv_step_requires_existing_exact_step_selection(api):
    """Preserve each reset-time CSV step and reject absent step labels."""
    source = b"t,v,i,run\n0,32,1,first\n1,32,2,first\n0,24,4,second\n1,24,5,second\n"
    request = request_for("csv", source)
    request["csv"] = {"step_column": "run"}
    request["mappings"] = {
        role: {"name": name, "unit": unit, "polarity": 1}
        for role, name, unit in [("time", "t", "s"), ("voltage", "v", "V"), ("current", "i", "A")]}
    client = api[0]
    discovery = post(client, "preview", request)
    assert discovery["available"]["steps"] == ["first", "second"]
    assert discovery["waveform"] is None
    request["csv"]["step_value"] = "second"
    second = post(client, "preview", request)
    assert second["waveform"]["current_a"] == [4, 5]
    assert second["source"]["step_values"] == {"run": "second"}
    request["csv"]["step_value"] = "absent"
    post(client, "preview", request, 422)


def test_raw_declared_size_is_checked_before_library_allocation(api, monkeypatch):
    """Stop an undersized upload that claims billions of samples before parsing."""
    from spicelib.raw import raw_read

    calls = []
    monkeypatch.setattr(raw_read, "RawRead", lambda *args, **kwargs: calls.append(args))
    raw = (b"Title: independent\nPlotname: Transient Analysis\nFlags: real\n"
           b"No. Variables: 4\nNo. Points: 1999999999\nVariables:\nBinary:\n")
    post(api[0], "preview", request_for("raw", raw), 413)
    assert calls == []


def test_valid_large_finite_csv_value_stays_numeric_and_unchanged(api):
    """Use the original signed SI values in waveform statistics and preview."""
    source = b"t,v,i\n0,-1.23e300,2\n1,-1.23e300,3\n"
    request = request_for("csv", source)
    request["csv"] = {"delimiter": ",", "decimal": "."}
    request["mappings"] = {role: {"name": name, "unit": unit, "polarity": 1}
                           for role, name, unit in [("time", "t", "s"), ("voltage", "v", "V"),
                                                    ("current", "i", "A")]}
    result = post(api[0], "preview", request)
    assert result["waveform"]["voltage_v"] == [-1.23e300] * 2
    assert result["statistics"]["voltage_v"]["mean"] == -1.23e300


def test_legacy_curve_and_partial_log_do_not_invent_dpt_statistics(api):
    """Check partial source coverage and independent turn-off current statistics."""
    client, repo = api
    device = TransistorFactory.create_empty_transistor("Partial-log")
    device.switch.e_on_data.append(SwitchingLossData(
        dataset_type="graph_i_e", t_j=25., v_supply=80., v_g=5.,
        graph_i_e=np.array([[1., 20.], [.0001, .0007]])))
    repo.save(device)
    base = "/api/transistors/Partial-log"
    assert not client.get(base + "/dpt_data").json()["has_e_on"]
    legacy = client.post(base + "/dpt_validate")
    assert legacy.status_code == 200, legacy.text
    device.switch.e_off_data.append(SwitchingLossData(
        dataset_type="single", t_j=None, v_supply=None, v_g=None, i_x=32,
        e_x=3e-6, evidence=MeasurementEvidence(
            source_kind="simulation", method="supplied-log-integral", event="turn_off")))
    # Use the already loaded probe device for this subcase, without a second file write.
    base = "/api/transistors/Partial-both"
    device.metadata.name = "Partial-both"
    repo.save(device)
    status = client.get(base + "/dpt_data").json()
    assert not status["has_e_on"] and status["has_e_off"]
    validation = client.post(base + "/dpt_validate")
    assert validation.status_code == 200, validation.text
    data = validation.json()
    assert data["statistics"]["i_max_tested"] == 32
    assert data["errors"] or data["warnings"] or data.get("status") == "partial"


def test_fresh_ltspice_binary_complete_waveform_and_native_archive(api, tmp_path):
    """Exercise the preserved generic fixture when supplied by a verification checkout."""
    path_value = os.getenv("TDB_GENERIC_DPT_RAW")
    if not path_value:
        pytest.skip("Set TDB_GENERIC_DPT_RAW to the November 10 generic LTspice fixture")
    from spicelib.raw.raw_read import RawRead

    from transistordatabase.utils.ltspice_dpt import DPTConfig, LTspiceDPT
    path = Path(path_value)
    original = path.read_bytes()
    digest = "f851fb950bc03212585657b37d187d8ba9f18bd2ffd2aefe396ff8c0335dcca2"
    assert hashlib.sha256(original).hexdigest() == digest
    names = ["time", "V(sw)", "Id(M_dut)", "V(gate)"]
    reader = RawRead(path, traces_to_read=names, dialect="ltspice", verbose=False)
    time, v, i, g = [reader.get_trace(n).get_wave(0).astype(float) for n in names]
    assert len(time) == 18136 and time[0] == 0 and time[-1] == 18.04e-6
    assert np.all(np.diff(time) > 0)
    dpt = LTspiceDPT(DPTConfig())
    direct = dpt.analyze_waveforms(time, v, i, g)
    request = request_for("raw", original)
    request["filename"] = path.name
    request["disclosure"] = dpt.MODEL_DISCLOSURE + " The selected channel current is Id(M_dut)."
    for role, name in zip(["time", "voltage", "current", "gate"], names, strict=True):
        request["mappings"][role]["name"] = name
    device = TransistorFactory.create_empty_transistor("Generic-simulation")
    conditions = {"t_j": None, "v_g_on": 15., "v_g_off": -5., "r_g_on": 10.,
                  "r_g_off": 10., "commutation_device": "Idealized freewheel diode",
                  "bus_voltage_v": 400.}
    client, repo = api
    result = post(client, "extract", {**request, "conditions": conditions,
                                      "device_document": core_to_json_document(device)})
    assert repo.list_all() == []
    archive = JsonTransistorLoader().load_from_dict(result["core_json_document"])
    repo.save(archive)
    saved = repo.get_by_name("Generic-simulation")
    export = tmp_path / "verified-generic.json"
    assert main._export.export_to_json(saved, export)
    restored = JsonTransistorLoader().load_from_json(export)
    [raw] = restored.switch.raw_measurement_data
    [on], [off] = restored.switch.e_on_data, restored.switch.e_off_data
    for arr, expected in [(raw.time_s, time), (raw.voltage_v, v), (raw.current_a, i),
                          (raw.gate_v, g)]:
        np.testing.assert_array_equal(arr, expected)
    for point, bounds, check, source_i in [
        (on, direct.on_window, .0001855909028250532, 22.966204121569728),
        (off, direct.off_window, .00017960023618601573, 19.913280487060547),
    ]:
        assert point.evidence.window_s == bounds
        assert point.e_x == pytest.approx(independent_energy(time, v, i, *bounds), rel=1e-12)
        assert point.e_x == pytest.approx(check, rel=1e-12)
        assert point.i_x == pytest.approx(source_i, rel=1e-12)
    for evidence in [on.evidence, off.evidence, raw.evidence]:
        assert evidence.source_kind == "simulation"
        assert evidence.sha256 == digest
        assert evidence.disclosure == request["disclosure"]
        assert evidence.trace_mapping["signals"]["current"]["name"] == "Id(M_dut)"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
