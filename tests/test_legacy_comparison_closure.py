"""Independent physical fixtures for formerly fabricated legacy adapter results."""
from copy import deepcopy
import numpy as np
import pytest
from fastapi.testclient import TestClient
from transistordatabase.backend.concrete_services import (
    CalculationService, ComparisonService, ExportService,
)
from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, SwitchingLossData,
    ThermalProperties, Transistor, TransistorMetadata,
)
def device(name="bounded"):
    """Construct independent 2 V drop and 1/2 mJ energies at 10 A."""
    transistor = Transistor(
        metadata=TransistorMetadata(name=name, type="IGBT", author="test", manufacturer="fixture", housing_type="fixture"),
        electrical=ElectricalRatings(1200, 100, 50, 175),
        thermal=ThermalProperties(.001, .001, .1, .1, .1))
    transistor.switch.channel_data = [ChannelCharacteristics(25, np.array([[0., 2., 4.], [0., 10., 20.]]), 15)]
    for name, energy, rg in (("e_on_data", .001, 4), ("e_off_data", .002, 7)):
        setattr(transistor.switch, name, [SwitchingLossData(
            "graph_i_e", 25, 400, 15, r_g=rg, v_g_off=-3,
            graph_i_e=np.array([[0., 10., 20.], [0., energy, 2 * energy]]))])
    return transistor
def point(**changes):
    """Return explicit operating-point inputs."""
    return dict(t_j=25, i_channel=10, f_sw=100_000, duty=.25,
                v_supply=400, v_g=15, v_g_off=-3, r_g_on=4, r_g_off=7, **changes)
def config():
    """Return aligned per-device comparison inputs."""
    op = point()
    return {key: [value, value] if key in {"v_supply", "v_g", "v_g_off", "r_g_on", "r_g_off"} else value for key, value in op.items()} | {"parallel_count": [1, 1], "output_power_w": 1000}
def test_static_duty_physical_results_and_frequency():
    """Verify static duty physical results and frequency."""
    service = CalculationService()
    result = service.calculate_losses(device(), point())
    assert result["conduction_losses"] == pytest.approx(5)
    assert result["switching_losses_on"] == pytest.approx(100)
    assert result["switching_losses_off"] == pytest.approx(200)
    assert result["total_losses"] == pytest.approx(305)
    double = service.calculate_losses(device(), point() | {"f_sw": 200_000})
    assert double["total_losses"] == pytest.approx(605)
    assert result["evidence"]["intervals"][0]["energy_evidence"]["turn_off"]["recorded_conditions"]["r_g"] == 7
@pytest.mark.parametrize("key,value", [("t_j", 26), ("v_supply", 401), ("v_g", 14), ("r_g_on", 5), ("r_g_off", 6), ("v_g_off", None), ("i_channel", 21)])
def test_mismatched_conditions_or_current_never_fall_back(key, value):
    """Verify mismatched conditions or current never fall back."""
    with pytest.raises(ValueError):
        CalculationService().calculate_losses(device(), point() | {key: value})
def test_missing_conditions_and_missing_curves_do_not_become_zero():
    """Verify missing conditions and missing curves do not become zero."""
    with pytest.raises(ValueError, match="explicit"):
        CalculationService().calculate_losses(device(), {"i_channel": 10})
    with pytest.raises(ValueError, match="one unambiguous"):
        CalculationService().interpolate_switching_losses([], point() | {"r_g": 4})
    with pytest.raises(ValueError, match="v_g_off"):
        CalculationService().interpolate_switching_losses(device().switch.e_on_data, {"t_j": 25, "i_channel": 10})
@pytest.mark.parametrize("duty,frequency,conduction", [(0, 100_000, 0), (1, 100_000, 20), (.25, 0, 5)])
def test_explicit_inactive_switching_has_zero_without_unused_curves(duty, frequency, conduction):
    """Verify explicit inactive switching has zero without unused curves."""
    source = device()
    source.switch.e_on_data = []
    source.switch.e_off_data = []
    if duty == 0:
        source.switch.channel_data = []
    result = CalculationService().calculate_losses(source, point() | {"duty": duty, "f_sw": frequency})
    assert result["total_losses"] == pytest.approx(conduction)
    assert result["switching_status"].startswith("inactive")
def test_ambiguity_unknown_conditions_and_exact_single_point():
    """Verify ambiguity unknown conditions and exact single point."""
    source = device()
    service = CalculationService()
    conditions = point() | {"r_g": 4}
    source.switch.e_on_data.append(deepcopy(source.switch.e_on_data[0]))
    with pytest.raises(ValueError, match="unambiguous"):
        service.interpolate_switching_losses(source.switch.e_on_data, conditions)
    record = SwitchingLossData("single", 25, 400, 15, e_x=.003, r_g=None, i_x=10, v_g_off=None)
    assert service.interpolate_switching_losses([record], conditions | {"r_g": None, "v_g_off": None}) == .003
    with pytest.raises(ValueError):
        service.interpolate_switching_losses([record], conditions | {"r_g": None, "v_g_off": None, "i_channel": 9})
    with pytest.raises(ValueError):
        service.interpolate_switching_losses([record], conditions | {"v_g_off": None})
def test_advanced_missing_devices_and_actual_recorded_slices():
    """Verify advanced missing devices and actual recorded slices."""
    first = device()
    second = device("empty")
    second.switch.channel_data = []
    second.switch.e_on_data = []
    second.switch.e_off_data = []
    first.switch.channel_data.append(ChannelCharacteristics(125, np.array([[0., 3.], [0., 10.]]), 18))
    result = ComparisonService().generate_advanced_comparison([first, second], {})
    assert [curve["conditions"]["t_j"] for curve in result["channel_characteristics"]["curves"]] == [25, 125]
    assert all(row["total"] is None for row in result["loss_breakdown"]["rows"])
    assert all(row["efficiency_percent"] is None for row in result["efficiency"]["rows"])
    partial = ComparisonService().generate_advanced_comparison([first, second], config())
    assert partial["loss_breakdown"]["rows"][0]["total"] == pytest.approx(305)
    assert partial["loss_breakdown"]["rows"][1]["status"] == "unsupported"
    assert partial["efficiency"]["rows"][0]["efficiency_percent"] == pytest.approx(1000 / 1305 * 100)
    assert partial["efficiency"]["rows"][1]["efficiency_percent"] is None
    assert "Excludes magnetics" in partial["efficiency"]["scope"]
def test_parallel_current_and_count_keep_actual_physical_convention():
    """Verify parallel current and count keep actual physical convention."""
    settings = config() | {"parallel_count": [2, 1]}
    row = ComparisonService().generate_advanced_comparison([device(), device("b")], settings)["loss_breakdown"]["rows"][0]
    assert row["calculation"]["conditions"]["i_channel"] == 5
    assert row["conduction"] == pytest.approx(2.5)
    assert row["switching_on"] == pytest.approx(100)
    assert row["switching_off"] == pytest.approx(200)
def test_current_underflow_and_bad_observation_conditions_are_visible():
    """Verify current underflow and bad observation conditions are visible."""
    source = device()
    source.switch.channel_data[0].t_j = float("nan")
    result = ComparisonService().generate_advanced_comparison([source, device("valid")], config() | {"i_channel": 5e-324, "parallel_count": [2, 1]})
    assert len(result["channel_characteristics"]["curves"]) == 1
    assert result["channel_characteristics"]["diagnostics"][0]["status"] == "invalid-data"
    assert "precision" in result["loss_breakdown"]["rows"][0]["reason"]
@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_frequency_rejects(frequency):
    """Verify nonfinite or negative frequency rejects."""
    with pytest.raises(ValueError):
        CalculationService().calculate_losses(device(), point() | {"f_sw": frequency})
def test_inputs_immutable_and_unsupported_placeholders_write_nothing(tmp_path):
    """Verify inputs immutable and unsupported placeholders write nothing."""
    op = point()
    source = device()
    graph = source.switch.e_on_data[0].graph_i_e.copy()
    CalculationService().calculate_losses(source, op)
    assert op == point()
    np.testing.assert_equal(source.switch.e_on_data[0].graph_i_e, graph)
    with pytest.raises(NotImplementedError, match="optimum"):
        CalculationService().find_optimal_working_point(source, {})
    with pytest.raises(NotImplementedError, match="physical metric"):
        ComparisonService().rank_transistors([source], {})
    for method in (ExportService().export_to_spice, ExportService().export_to_ltspice):
        with pytest.raises(NotImplementedError):
            method(source, tmp_path / "unwritten.model")
    assert list(tmp_path.iterdir()) == []
def test_live_http_preserves_partial_panels_and_rejects_invalid_inputs(monkeypatch):
    """Verify live http preserves partial panels and rejects invalid inputs."""
    from transistordatabase.gui_web.backend import main
    devices = {"a": device(), "b": device("empty")}
    devices["b"].switch.e_off_data = []
    monkeypatch.setattr(main, "_get_transistor", devices.__getitem__)
    client = TestClient(main.app)
    response = client.post("/api/comparison/advanced", json={"transistor_ids": ["a", "b"], "config": config()})
    assert response.status_code == 200
    result = response.json()
    assert result["loss_breakdown"]["rows"][0]["total"] == pytest.approx(305)
    assert result["loss_breakdown"]["rows"][1]["total"] is None
    assert len(result["switching_losses_eon"]["curves"]) == 2
    invalid = client.post("/api/comparison/advanced", content='{"transistor_ids":["a","b"],"config":{"f_sw":NaN}}', headers={"content-type": "application/json"})
    assert invalid.status_code == 422
    bad_length = client.post("/api/comparison/advanced", json={"transistor_ids": ["a", "b"], "config": {"v_supply": [400]}})
    assert bad_length.status_code == 422
    for extension in ("spice", "ltspice"):
        for suffix in ("", "/preview"):
            endpoint = f"/api/transistors/a/export/{extension}{suffix}"
            reply = client.get(endpoint) if suffix else client.post(endpoint)
            assert reply.status_code == 422
            assert "explicit" in reply.json()["detail"]
