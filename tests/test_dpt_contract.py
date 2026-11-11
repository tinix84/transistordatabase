"""Independent DPT topology, event and pure HTTP rendering contracts."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from transistordatabase.gui_web.backend import main
from transistordatabase.utils import ltspice_dpt as module
from transistordatabase.utils.ltspice_dpt import DPTConfig, LTspiceDPT


def transistor(name="Test label"):
    """Build a label-only object without a vendor model."""
    return SimpleNamespace(metadata=SimpleNamespace(name=name))


def test_exact_two_pulse_schedule_and_directed_paths():
    """Verify pulse plateaus, directed current loop and independent turn-off resistance."""
    dpt = LTspiceDPT(DPTConfig())
    schedule = dpt.pulse_schedule()
    t, voltage = np.array(schedule).T
    assert np.all(np.diff(t) > 0)
    high = voltage > 5
    assert np.sum(np.diff(high.astype(int)) == 1) == 2
    assert np.sum(np.diff(high.astype(int)) == -1) == 2
    assert t[3] - t[2] == pytest.approx(5e-6)
    assert t[5] - t[4] == pytest.approx(5e-6)
    assert t[7] - t[6] == pytest.approx(2e-6)
    deck = dpt.render_netlist()
    for line in ("Lload bus sw 0.0001", "D_freewheel sw bus GENERIC_FREEWHEEL",
                 "R_on drive on_path 10.0", "D_on on_path gate GATE_STEERING",
                 "D_off gate off_path GATE_STEERING", "R_off off_path drive 10.0",
                 ".options plotwinsize=0", "Id(M_dut)"):
        assert line in deck
    assert "PWL(" in deck and "PULSE(" not in deck
    assert "BV=600" not in deck
    assert "metadata only" in deck
    changed = LTspiceDPT(replace(dpt.config, r_g_off=42)).render_netlist()
    assert changed == deck.replace("R_off off_path drive 10.0", "R_off off_path drive 42")


def test_renderer_is_pure_and_label_cannot_inject_circuit(monkeypatch):
    """Rendering neither accesses storage nor promotes labels to circuit elements."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure renderer attempted an external side effect")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(LTspiceDPT, "run_simulation", forbidden)
    dpt = LTspiceDPT(DPTConfig())
    a = dpt.render_netlist(transistor("part\n.include secret.cir"))
    b = dpt.render_netlist(transistor("other"))
    assert a.splitlines()[1:] == b.splitlines()[1:]
    assert "\n.include" not in a
    assert dpt.netlist_filename(transistor("../bad/path")) == "dpt_.._bad_path.cir"


def test_file_writer_uses_renderer_and_preserves_existing_destination(tmp_path):
    """The optional writer preserves explicit paths and existing content."""
    dpt = LTspiceDPT(DPTConfig())
    destination = tmp_path / "explicit.asc"
    assert dpt.generate_netlist(destination) == destination
    assert destination.read_text() == dpt.render_netlist()
    with pytest.raises(FileExistsError):
        dpt.generate_netlist(destination)
    assert destination.read_text() == dpt.render_netlist()


@pytest.mark.parametrize("key", list(vars(DPTConfig()).keys())[:-1])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_config_rejects_nonfinite(key, value):
    """Every numeric configuration field must be finite."""
    with pytest.raises(ValueError, match="finite"):
        LTspiceDPT(replace(DPTConfig(), **{key: value})).render_netlist()


@pytest.mark.parametrize("changes", [
    {"v_dc": 0}, {"i_target": -1}, {"l_load": 0}, {"r_g_on": 0},
    {"r_g_off": -1}, {"t_dead": 0}, {"t_pulse2": 0},
    {"v_g_on": 3}, {"v_g_off": 3}, {"v_g_off": 2.5}, {"model_kind": "vendor"},
    {"i_target": 1e308, "l_load": 1e308}, {"t_dead": 1e308},
    {"i_target": 1e-300},
])
def test_rejects_invalid_or_unrepresentable_domains(changes):
    """Reject invalid physical domains and unrepresentable PWL schedules."""
    with pytest.raises(ValueError):
        LTspiceDPT(replace(DPTConfig(), **changes)).render_netlist()


def growing_two_pulse_trace():
    """Return simultaneous linear transitions with later second-pulse growth."""
    # First turn-off at 20 A, second turn-on near 20 A, later growth to 35 A.
    t = np.linspace(0, 15e-6, 15001)
    knots = [0, 5e-6, 5.05e-6, 10e-6, 10.05e-6, 12e-6, 12.05e-6, 15e-6]
    current = np.interp(t, knots, [0, 20, 0, 0, 20, 35, 0, 0])
    voltage = np.interp(t, knots, [.5, .5, 400, 400, .5, .5, 400, 400])
    gate = np.interp(t, knots, [15, 15, -5, -5, 15, 15, -5, -5])
    return t, voltage, current, gate


def test_growth_does_not_hide_first_turn_off_or_change_event_identity():
    """Compare local first-event thresholds and energy against analytic integrals."""
    dpt = LTspiceDPT(DPTConfig())
    t, v, i, g = growing_two_pulse_trace()
    result = dpt.analyze_waveforms(t, v, i, g)
    assert np.max(i) == 35
    assert result.i_off == pytest.approx(20, rel=.001)
    assert result.i_on == pytest.approx(19.8, rel=.002)
    assert result.t_f == pytest.approx(40e-9, rel=.005, abs=1e-12)
    assert result.t_r == pytest.approx(39.6e-9, rel=.005, abs=1e-12)
    # Simultaneous linear transitions: integral I*T*(Vlow/2+(Vhigh-Vlow)/6).
    exact_full = 20 * 50e-9 * (.5 / 2 + (400 - .5) / 6)
    assert result.e_on == pytest.approx(exact_full, rel=.003, abs=1e-9)
    assert result.e_off == pytest.approx(exact_full, rel=.003, abs=1e-9)
    assert 5e-6 < result.off_window[0] < result.off_window[1] < 5.051e-6
    assert 10e-6 < result.on_window[0] < result.on_window[1] < 10.051e-6
    assert "1%-99%" in result.energy_window_convention
    assert result.event_evidence["turn_off"]["local_current_high_a"] == result.i_off
    e_on, e_off = dpt.extract_switching_loss_data(result)
    assert e_on.i_x == result.i_on and e_off.i_x == result.i_off


@pytest.mark.parametrize("problem", ["missing_gate", "missing_edge", "zero_current", "nan", "shape", "time"])
def test_unresolved_measurement_is_an_error_not_zero_energy(problem):
    """Missing or invalid event evidence must not become zero losses."""
    t, v, i, g = growing_two_pulse_trace()
    if problem == "missing_gate":
        g = None
    elif problem == "missing_edge":
        g[:] = 15
    elif problem == "zero_current":
        i[:] = 0
    elif problem == "nan":
        v[10] = np.nan
    elif problem == "shape":
        i = i[:-1]
    elif problem == "time":
        t[10] = t[9]
    with pytest.raises(ValueError):
        LTspiceDPT(DPTConfig()).analyze_waveforms(t, v, i, g)


def test_simulation_waits_and_uses_actual_runner_output(monkeypatch, tmp_path):
    """Use the blocking runner contract and its actual output path."""
    actual = tmp_path / "actual.raw"
    actual.write_text("test fixture")
    calls = []
    class Runner:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def run_now(self, netlist):
            calls.append(netlist)
            return actual, tmp_path / "actual.log"
    monkeypatch.setattr(module, "HAS_PYLTSPICE", True)
    monkeypatch.setattr(module, "SimRunner", Runner, raising=False)
    input_path = tmp_path / "input.cir"
    assert LTspiceDPT(DPTConfig()).run_simulation(input_path) == actual
    assert calls[-1] == str(input_path)


def test_http_returns_real_text_and_explicit_model_without_writes(monkeypatch):
    """The HTTP endpoint renders disclosed circuit text without external effects."""
    monkeypatch.setattr(main, "_get_transistor", lambda identifier: transistor(identifier))
    monkeypatch.setattr(LTspiceDPT, "generate_netlist", lambda *args: pytest.fail("HTTP wrote a file"))
    monkeypatch.setattr(LTspiceDPT, "run_simulation", lambda *args: pytest.fail("HTTP ran simulation"))
    with TestClient(main.app) as client:
        response = client.post("/api/dpt/generate_netlist?transistor_id=TEST", json={"model_kind": "generic_nmos"})
        missing_choice = client.post("/api/dpt/generate_netlist?transistor_id=TEST", json={})
        invalid = client.post("/api/dpt/generate_netlist?transistor_id=TEST", json={"model_kind": "generic_nmos", "r_g_off": 0})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["netlist"].startswith("* Generic Double Pulse Test")
    assert result["filename"] == "dpt_TEST.cir"
    assert result["ideal_first_pulse_s"] == pytest.approx(5e-6)
    assert len(result["pulse_schedule"]) == 10
    assert "not a vendor" in result["model_disclosure"]
    assert missing_choice.status_code == invalid.status_code == 422
    assert "r_g_off must be positive" in invalid.json()["detail"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_http_nonfinite_has_actionable_422(monkeypatch, value):
    """Nonstandard nonfinite JSON inputs produce a serializable domain error."""
    monkeypatch.setattr(main, "_get_transistor", lambda identifier: transistor(identifier))
    with TestClient(main.app) as client:
        response = client.post("/api/dpt/generate_netlist?transistor_id=TEST",
                               content=json.dumps({"model_kind": "generic_nmos", "v_dc": value}),
                               headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "finite" in response.json()["detail"]
