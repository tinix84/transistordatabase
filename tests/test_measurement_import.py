"""Independent source, binary transport, and native evidence roundtrip checks."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from transistordatabase.core.models import (
    ElectricalRatings, ThermalProperties, Transistor, TransistorMetadata,
)
from transistordatabase.core.repository import JsonTransistorLoader, JsonTransistorRepository
from transistordatabase.core.serialization import core_from_json_document, core_to_json_document
from transistordatabase.gui_web.backend.measurement_routes import MeasurementRequest
from transistordatabase.measurement_import import extract_measurement, preview_measurement
from transistordatabase.utils.ltspice_dpt import DPTConfig, DPTResult, LTspiceDPT


def analytic():
    """Two complementary linear commutations with independently known overlap energy."""
    time = np.linspace(0, 8e-6, 1601)
    off = np.clip((time - 2e-6) / .5e-6, 0, 1)
    on = np.clip((time - 5e-6) / .5e-6, 0, 1)
    state = 1 - off + on
    return time, 100 * (1 - state), 10 * state, -5 + 20 * state


def request(data, fmt="csv"):
    """Declare every source field and probe mapping for the analytic fixture."""
    names = ("time", "V(ds)", "I(d)", "V(gs)")
    mappings = {key: {"name": name, "unit": unit, "polarity": 1}
                for key, name, unit in zip(("time", "voltage", "current", "gate"),
                                           names, ("s", "V", "A", "V"), strict=True)}
    return MeasurementRequest(format=fmt, filename="analytic." + fmt,
        content_base64=base64.b64encode(data).decode(), source_kind="simulation",
        disclosure="Analytic generic fixture. No device calibration.", mappings=mappings,
        step_index=0).model_dump()


def csv_bytes():
    """Export the analytic fixture through a BOM-encoded CSV transport."""
    return ("time,V(ds),I(d),V(gs)\n" + "\n".join(
        ",".join(format(value, ".17g") for value in row)
        for row in zip(*analytic(), strict=True))).encode("utf-8-sig")


CONDITIONS = {"t_j": None, "bus_voltage_v": 100., "v_g_on": 15.,
              "v_g_off": -5., "r_g_on": 2., "r_g_off": 7.,
              "commutation_device": None}


def test_csv_analytic_energy_and_core_evidence_roundtrip(tmp_path):
    """Native repository save and reload retain source bytes and separate event conditions."""
    data = csv_bytes()
    device = Transistor(TransistorMetadata("fixture", "MOSFET", "test", "unknown", "unknown"),
                        ElectricalRatings(200., 20., 10., 150.), ThermalProperties(.01, .01))
    result = extract_measurement(request(data), CONDITIONS, core_to_json_document(device))
    # Integral of 1000*x*(1-x) over x=.01 through .99, dt=.5 us.
    expected = 1000 * .5e-6 * ((.99**2/2 - .99**3/3) - (.01**2/2 - .01**3/3))
    assert result["analysis"]["e_on"] == pytest.approx(expected, rel=1e-4)
    assert result["analysis"]["e_off"] == pytest.approx(expected, rel=1e-4)
    core = core_from_json_document(result["core_json_document"])
    repository = JsonTransistorRepository(tmp_path / "isolated-repository")
    repository.save(core)
    reloaded = repository.get_by_name("fixture")
    loader = JsonTransistorLoader()
    destination = tmp_path / "evidence.json"
    loader.save_to_json(reloaded, destination)
    loaded = loader.load_from_json(destination)
    assert core_to_json_document(loaded) == result["core_json_document"]
    assert loaded.switch.e_on_data[0].r_g == 2
    assert loaded.switch.e_off_data[0].r_g == 7
    assert loaded.switch.e_on_data[0].evidence.window_s is not None
    raw = loaded.switch.raw_measurement_data[0]
    assert raw.evidence.sha256 == hashlib.sha256(data).hexdigest()
    assert base64.b64decode(raw.evidence.original_base64) == data
    assert np.array_equal(raw.gate_v, analytic()[3])
    assert loaded.switch.e_on_data[0].t_j is None


def test_binary_raw_and_csv_equal(tmp_path):
    """Installed spicelib binary writer and reader preserve the analytic trace transport."""
    from spicelib.raw.raw_write import RawWrite, Trace
    writer = RawWrite(numtype="double")
    for name, values, kind in zip(("time", "V(ds)", "I(d)", "V(gs)"), analytic(),
                                  ("time", "voltage", "current", "voltage"), strict=True):
        writer.add_trace(Trace(name, values, whattype=kind, numerical_type="double"))
    path = tmp_path / "analytic.raw"
    writer.save(path)
    csv_result = extract_measurement(request(csv_bytes()), CONDITIONS)
    raw_result = extract_measurement(request(path.read_bytes(), "raw"), CONDITIONS)
    for key, values in csv_result["waveform"].items():
        assert raw_result["waveform"][key] == pytest.approx(values, rel=1e-12, abs=1e-14)
    assert raw_result["analysis"]["e_on"] == pytest.approx(csv_result["analysis"]["e_on"])
    assert "absolute time" in raw_result["source"]["parser_normalizations"][0]


def test_historical_log_exact_values_and_scope():
    """The actual supplied log retains all 30 steps and original 39--41 us bounds."""
    path = Path("/home/tinix/claude_wsl/archive_ntbees2/sc/tests/DoublePulseTester_EPC8009_sync.log")
    if not path.exists():
        pytest.skip("Read-only historical SC log is unavailable")
    original = path.read_bytes()
    req = request(original, "log")
    req["log"].update(energy_off="eoff", current="idc", bus="vdc")
    result = preview_measurement(req)
    assert len(result["log_records"]) == 30
    assert [row["energy_j"] for row in result["log_records"][:3]] == [1.0141e-7, 9.47901e-7, 2.69621e-6]
    assert [row["current_a"] for row in result["log_records"][:3]] == [1, 3, 5]
    assert result["log_records"][0]["window_s"] == [39e-6, 41e-6]
    extracted = extract_measurement(req, CONDITIONS)
    assert extracted["analysis"] is None and extracted["raw_record"] is None
    assert extracted["points"]["turn_off"][0]["evidence"]["method"] == "supplied-log-integral"
    assert path.read_bytes() == original


def test_manual_result_is_explicit_and_cannot_be_measured():
    """Manual energies carry no fabricated gate events or measured local current."""
    dpt = LTspiceDPT(DPTConfig())
    on, off = dpt.extract_switching_loss_data(DPTResult(e_on=.001, e_off=.002))
    assert on.evidence.source_kind == "manual"
    assert off.evidence.method == "manual-point"
    assert on.evidence.local_current_a is None
    assert on.evidence.window_s is None
    with pytest.raises(ValueError, match="explicit finite"):
        dpt.extract_switching_loss_data(DPTResult())
    manual_zero, _ = dpt.extract_switching_loss_data(DPTResult(e_on=0., e_off=0.))
    assert manual_zero.evidence.source_kind == "manual" and manual_zero.e_x == 0.


@pytest.mark.parametrize("text", ["time,V(ds),I(d),V(gs)\n0,0,1,15\n0,1,0,-5",
                                  "time,V(ds),I(d),V(gs)\n0,0,nan,15\n1,1,0,-5",
                                  "time,V(ds),I(d),V(gs)\n0,0,1,15\n1,1,0"])
def test_invalid_csv_rejects(text):
    """Invalid alignment, time, and numeric values never produce plausible arrays."""
    with pytest.raises(ValueError):
        preview_measurement(request(text.encode()))
