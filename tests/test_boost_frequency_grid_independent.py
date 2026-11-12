"""Independent closed forms and API compatibility for the Boost frequency study."""
from hashlib import sha256
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, SwitchingLossData,
    ThermalProperties, Transistor, TransistorMetadata,
)
from transistordatabase.core.repository import JsonTransistorRepository
from transistordatabase.gui_web.backend import main
from transistordatabase.gui_web.backend.main import app
from transistordatabase.topologies.operating_point import calculate_topology_operating_point, calculate_topology_sweep
from transistordatabase.topologies.semiconductor_pareto import calculate_boost_frequency_grid


def ideal_drop_device(name="Reference pair", on_energy=.001, off_energy=.002):
    """The two channel curves have zero drop and switching energies are constant."""
    device = Transistor(
        TransistorMetadata(name, "MOSFET", "Fixture", "Fixture", "Fixture"),
        ElectricalRatings(1200, 100, 50, 175), ThermalProperties(.001, .001))
    current = [0., 100.]
    device.switch.channel_data = [ChannelCharacteristics(25, np.array([[0., 0.], current]), 15)]
    device.diode.channel_data = [ChannelCharacteristics(25, np.array([[0., 0.], current]), None)]
    for key, energy in (("e_on_data", on_energy), ("e_off_data", off_energy)):
        curve = SwitchingLossData("graph_i_e", 25, 200, 15, r_g=10, v_g_off=-5,
                                  graph_i_e=np.array([current, [energy, energy]]),
                                  commutation_device="Fixture rectifier", measurement_testbench="Grid fixture")
        setattr(device.switch, key, [curve])
    return device


def boundary(**changes):
    return dict(mode="switching-loss-balanced", conduction_regime="CCM", v_in=100, v_out=200,
                p_out=1000, f_sw=20000, l_value=.001, c_out=.001, t_j=25, v_g=15, v_g_off=-5,
                r_g_on=10, r_g_off=10) | changes


def candidate(label, source):
    name = source.metadata.name
    return dict(candidate_id=label, label=label, switch_transistor_id=name, diode_transistor_id=name,
                switch_transistor=source, diode_transistor=source, parallel_switches=1,
                parallel_diodes=1, v_g=15, diode_v_g=None, v_g_off=-5, r_g_on=10, r_g_off=10)


def catalog_item(name, price):
    return dict(orderable_part=name, unit_cost_usd=price, length_mm=2, width_mm=3, height_mm=1,
                price_source_url="https://fixture.invalid/price", dimensions_source_url="https://fixture.invalid/body",
                cost_accessed_on="2026-10-01")


def check_scalar_equations(point, frequency, energy_sum):
    heat = frequency * energy_sum
    current = (1000 + heat) / 100
    duty = 1 - 5 / current
    ripple = 100 * duty / (.001 * frequency)
    voltage_residual = 100 - 200 * (1 - duty)
    calc = point["calculated"]
    assert point["losses"]["total_w"] == pytest.approx(heat, abs=1e-8)
    assert calc["i_in_avg_a"] == pytest.approx(current, rel=1e-7)
    assert calc["duty_cycle"] == pytest.approx(duty, abs=1e-8)
    assert calc["i_l_ripple_pp_a"] == pytest.approx(ripple, rel=1e-7)
    assert calc["i_l_valley_a"] == pytest.approx(current - ripple / 2, rel=1e-7)
    assert calc["i_l_peak_a"] == pytest.approx(current + ripple / 2, rel=1e-7)
    assert calc["power_balance_residual_w"] == pytest.approx(0, abs=1e-6)
    assert calc["inductor_volt_second_residual_v"] == pytest.approx(voltage_residual, abs=1e-6)
    assert point["losses"]["reverse_recovery_losses_included"] is False


@pytest.mark.parametrize("with_sweep,expected", [
    (False, "8045914116fef11afb03e3e85c3095bed94f05ea6127810b5cdf9a673d58b976"),
    (True, "73ed8f52b1eafbb59cbdae4ff800c7c84b97a646847ff28b84e9b1e26651f07c"),
])
def test_existing_estimate_and_frequency_line_keep_full_merged_response(with_sweep, expected):
    request = dict(topology="boost", mode="estimate", v_in=100, v_out=200, p_out=1000,
                   f_sw=20000, l_value=.001, c_out=.001)
    if with_sweep:
        request["sweep"] = dict(x_axis="f_sw", x_min=10000, x_max=40000,
                                x_points=3, metric="i_l_ripple_pp_a")
    response = TestClient(app).post("/api/topology/calculate", json=request)
    assert response.status_code == 200, response.text
    normalized = json.dumps(response.json(), sort_keys=True, separators=(",", ":")).encode()
    assert sha256(normalized).hexdigest() == expected


def test_existing_pareto_without_opt_in_preserves_complete_failure_population():
    request = dict(boundary=dict(v_in=100, v_out=200, p_out=1000, f_sw=20000,
                                 l_value=.001, c_out=.001, t_j=25),
                   candidates=[dict(candidate_id=name, label=name) for name in ("first", "second")],
                   catalog={})
    response = TestClient(app).post("/api/topology/pareto", json=request)
    assert response.status_code == 200, response.text
    normalized = json.dumps(response.json(), sort_keys=True, separators=(",", ":")).encode()
    assert sha256(normalized).hexdigest() == "84f7e942ecb1d3278c18142b153b5138923cdd48875a499de9a18c595c9e1916"


def test_mixed_line_retains_frequency_position_and_matches_independent_scalar_solves():
    device = ideal_drop_device()
    flags = boundary()
    sweep = dict(x_axis="f_sw", x_min=1000, x_max=19000, x_points=3, metric="p_loss_total_w")
    result = calculate_topology_sweep("boost", flags, sweep, device, device)
    assert result["x_values"] == [1000, 10000, 19000]
    assert result["voltage_law_enforced"] is False
    assert result["values"][0] is None
    assert result["voltage_law_residual_values_v"][0] is None
    assert result["power_balance_residual_values_w"][0] is None
    bad, *good = result["rows"]
    assert bad["frequency_hz"] == 1000 and bad["status"] == "unsupported"
    assert bad["operating_point"] is None and bad["diagnostic"]
    for index, row in enumerate(good, 1):
        f = result["x_values"][index]
        assert row["frequency_hz"] == f and row["status"] == "qualified"
        single = calculate_topology_operating_point("boost", flags | {"f_sw": f},
                                                     device, device, include_waveforms=False)
        assert row["operating_point"] == single
        check_scalar_equations(single, f, .003)
        assert result["values"][index] == pytest.approx(f * .003)
        assert result["voltage_law_residual_values_v"][index] == single["calculated"]["inductor_volt_second_residual_v"]
        assert result["power_balance_residual_values_w"][index] == single["calculated"]["power_balance_residual_w"]


def test_lossless_limit_is_a_supported_measured_zero_at_every_frequency():
    device = ideal_drop_device(on_energy=0, off_energy=0)
    sweep = dict(x_axis="f_sw", x_min=10000, x_max=40000, x_points=2, metric="p_loss_total_w")
    result = calculate_topology_sweep("boost", boundary(), sweep, device, device)
    assert result["values"] == [0, 0]
    for row in result["rows"]:
        assert row["status"] == "qualified"
        check_scalar_equations(row["operating_point"], row["frequency_hz"], 0)


def test_missing_switching_curve_keeps_gaps_distinct_from_zero_heat():
    device = ideal_drop_device()
    device.switch.e_off_data = []
    result = calculate_topology_sweep("boost", boundary(),
                                      dict(x_axis="f_sw", x_min=10000, x_max=20000,
                                           x_points=2, metric="p_loss_total_w"), device, device)
    assert result["values"] == [None, None]
    assert all(r["status"] == "unsupported" and r["diagnostic"] for r in result["rows"])


def test_each_frequency_has_its_own_complete_ranked_population():
    tied = ideal_drop_device("Tied")
    costly = ideal_drop_device("Costly")
    ideal = ideal_drop_device("Ideal", 0, 0)
    unknown = ideal_drop_device("No price")
    cases = [candidate("A", tied), candidate("B", tied), candidate("expensive", costly),
             candidate("ideal", ideal), candidate("unknown-price", unknown)]
    parts = {key: catalog_item(key, price) for key, price in (("Tied", 1), ("Costly", 2), ("Ideal", 3))}
    result = calculate_boost_frequency_grid(boundary(), cases, parts, [10000, 20000], voltage_residual_tolerance_v=4)
    assert result["frequencies_hz"] == [10000, 20000]
    assert result["scalar_call_count"] == 10
    for row in result["rows"]:
        f, pop = row["frequency_hz"], row["population"]
        assert pop["loss_scope"] == "channel_conduction_and_switch_turn_on_turn_off"
        assert pop["voltage_law_enforced"] is False
        items = {item["candidate_id"]: item for item in pop["candidates"]}
        assert list(items) == ["A", "B", "expensive", "ideal", "unknown-price"]
        assert items["unknown-price"]["eligible"] is False
        assert items["unknown-price"]["objectives"] is None and items["unknown-price"]["ineligible_reasons"]
        assert items["A"]["operating_point"]["calculated"]["inductor_volt_second_residual_v"] > 0
        assert items["B"]["operating_point"] == items["A"]["operating_point"]
        if f == 10000:
            assert set(pop["pareto_candidate_ids"]) == {"A", "B", "ideal"}
            assert items["A"]["objectives"] == items["B"]["objectives"]
            assert items["A"]["objectives"]["included_loss_w"] == pytest.approx(30)
            assert items["expensive"]["eligible"] is True and items["expensive"]["pareto"] is False
        else:
            assert pop["pareto_candidate_ids"] == ["ideal"]
            assert items["A"]["eligible"] is False and items["A"]["ineligible_reasons"]
            assert items["A"]["operating_point"]["losses"]["total_w"] == pytest.approx(60)


@pytest.mark.parametrize("frequencies", [[10000, 10000], [10000, 0], [True, 10000]])
def test_shared_grid_has_only_distinct_finite_positive_frequencies(frequencies):
    part = ideal_drop_device()
    with pytest.raises(ValueError):
        calculate_boost_frequency_grid(boundary(), [candidate("A", part)], {}, frequencies)


def test_resource_budget_applies_across_all_candidates_and_frequencies():
    part = ideal_drop_device()
    with pytest.raises(ValueError):
        calculate_boost_frequency_grid(boundary(), [candidate("A", part), candidate("B", part)],
                                       {}, list(range(10000, 27000, 1000)))


def test_api_snapshots_each_shared_device_and_missing_identity_once(tmp_path, monkeypatch):
    from transistordatabase.gui_web.backend import device_electrothermal_routes

    repository = JsonTransistorRepository(tmp_path / "source")
    part = ideal_drop_device()
    repository.save(part)
    stored = repository.data_directory / (part.metadata.name + ".json")
    original = stored.read_bytes()
    snapshots, paths = [], []
    read_snapshot = device_electrothermal_routes._snapshot
    resolve = repository._storage_path

    def count_snapshot(repo, name, **kwargs):
        snapshots.append(name)
        return read_snapshot(repo, name, **kwargs)

    def count_storage_paths(name):
        paths.append(name)
        return resolve(name)

    monkeypatch.setattr(device_electrothermal_routes, "_snapshot", count_snapshot)
    monkeypatch.setattr(repository, "_storage_path", count_storage_paths)
    monkeypatch.setattr(main, "_repo", repository)
    item = {k: v for k, v in candidate("A", part).items() if k not in ("switch_transistor", "diode_transistor")}
    absent = item | {"candidate_id": "unavailable", "switch_transistor_id": "Absent", "diode_transistor_id": "Absent"}
    request = dict(boundary=boundary(), frequencies_hz=[10000, 20000], catalog={part.metadata.name: catalog_item("Pair", 1)},
                   candidates=[item, item | {"candidate_id": "B"}, absent])
    response = TestClient(app).post("/api/topology/switching-frequency-pareto", json=request)
    assert response.status_code == 200, response.text
    assert snapshots == [part.metadata.name]
    assert paths.count("Absent") == 1
    assert response.json()["source_snapshots"][part.metadata.name]["sha256"] == sha256(original).hexdigest()
    for row in response.json()["rows"]:
        population = row["population"]
        assert set(population["pareto_candidate_ids"]) == {"A", "B"}
        missing = population["candidates"][2]
        assert missing["eligible"] is False and missing["operating_point"] is None
        assert missing["ineligible_reasons"]
    assert stored.read_bytes() == original


@pytest.mark.parametrize("with_energy", [True, False])
def test_normal_service_solves_exactly_listed_coordinates_and_retains_rejected_grid(monkeypatch, with_energy):
    from transistordatabase.gui_web.backend import topology_service
    from transistordatabase.topologies import operating_point

    device = ideal_drop_device()
    if not with_energy:
        device.switch.e_off_data = []
    actual_scalar = operating_point.calculate_topology_operating_point
    calls = []

    def counted(topology, parameters, *args, **kwargs):
        calls.append(parameters["f_sw"])
        return actual_scalar(topology, parameters, *args, **kwargs)

    monkeypatch.setattr(operating_point, "calculate_topology_operating_point", counted)
    monkeypatch.setattr(topology_service, "calculate_topology_operating_point", counted)
    request = boundary(f_sw=1000, sweep=dict(x_axis="f_sw", x_min=1000, x_max=19000,
                                            x_points=3, metric="p_loss_total_w"))
    result = topology_service.TopologyCalculatorService.calculate("boost", request, device)
    line = result["sweep"]
    assert calls == line["x_values"] == [1000, 10000, 19000]
    assert line["scalar_call_count"] == len(calls)
    assert line["rows"][0]["status"] == "unsupported"
    if with_energy:
        assert line["values"] == [None, pytest.approx(30), pytest.approx(57)]
    else:
        assert line["values"] == [None, None, None]
        assert all(row["diagnostic"] for row in line["rows"])


def test_bounded_core_rejects_unsized_frequency_iterables_without_enumeration():
    class UncountedFrequencies:
        def __iter__(self):
            raise AssertionError("The bounded API must reject an uncounted input before enumeration")

    part = ideal_drop_device()
    with pytest.raises(ValueError):
        calculate_boost_frequency_grid(boundary(), [candidate("A", part)], {}, UncountedFrequencies())


def test_list_backed_oversize_graph_rejects_before_a_scalar_solve(monkeypatch):
    from transistordatabase.topologies import operating_point
    part = ideal_drop_device()
    part.switch.channel_data[0].graph_v_i = [[0.] * 10001, list(range(10001))]

    def unexpected_scalar(*args, **kwargs):
        raise AssertionError("Source size validation must precede numerical solves")

    monkeypatch.setattr(operating_point, "calculate_topology_operating_point", unexpected_scalar)
    with pytest.raises(ValueError):
        calculate_topology_sweep("boost", boundary(), dict(x_axis="f_sw", x_min=10000,
                                  x_max=20000, x_points=2, metric="p_loss_total_w"), part, part)


@pytest.mark.parametrize("path", ["/api/topology/switching-frequency-grid", "/api/topology/switching-frequency-pareto"])
def test_dedicated_transport_enforces_four_mib_of_actual_json_before_parsing(path):
    # Trailing JSON whitespace gives a flat valid document one byte beyond the transport limit.
    size_limit = 4 * 1024**2
    body = b"{}" + b" " * (size_limit - 1)
    assert len(body) == size_limit + 1
    response = TestClient(app).post(path, content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_legacy_paths_identify_dedicated_transport_for_new_inputs():
    client = TestClient(app)
    line = boundary() | dict(topology="boost", sweep=dict(x_axis="f_sw", x_min=10000, x_max=20000,
                                                       x_points=2, metric="p_loss_total_w"))
    pareto = dict(boundary=boundary(), frequencies_hz=[10000, 20000], catalog={},
                  candidates=[dict(candidate_id=name, label=name) for name in ("first", "second")])
    for old, new, request in (("/api/topology/calculate", "/api/topology/switching-frequency-grid", line),
                              ("/api/topology/pareto", "/api/topology/switching-frequency-pareto", pareto)):
        response = client.post(old, json=request)
        assert response.status_code == 422
        assert new in response.text


def _http_switching_request(study, device_name):
    if study == "grid":
        return boundary() | dict(
            topology="boost", switch_transistor_id=device_name, diode_transistor_id=device_name,
            diode_v_g=None, sweep=dict(x_axis="f_sw", x_min=10000, x_max=20000,
                                      x_points=2, metric="p_loss_total_w"))
    entries = [
        dict(candidate_id=label, label=label, switch_transistor_id=device_name,
             diode_transistor_id=device_name, v_g=15, v_g_off=-5, diode_v_g=None,
             r_g_on=10, r_g_off=10)
        for label in ("0001", "25.0000000000000000001")
    ]
    return dict(boundary=boundary(voltage_residual_tolerance_v=0),
                frequencies_hz=[10000, 20000], candidates=entries,
                catalog={device_name: catalog_item("00123", 1)})


def test_existing_estimate_still_accepts_its_original_numeric_string_input():
    body = dict(topology="boost", mode="estimate", v_in="100", v_out=200,
                p_out=1000, f_sw=20000, l_value=.001, c_out=.001)
    response = TestClient(app).post("/api/topology/calculate", json=body)
    assert response.status_code == 200
    normalized = json.dumps(response.json(), sort_keys=True, separators=(",", ":")).encode()
    assert sha256(normalized).hexdigest() == "8045914116fef11afb03e3e85c3095bed94f05ea6127810b5cdf9a673d58b976"


@pytest.mark.parametrize("study,field,value", [
    pytest.param("grid", ("t_j",), "25.0000000000000000001", id="line-quoted-temperature"),
    pytest.param("grid", ("r_g_on",), "1e-400", id="line-quoted-underflow"),
    pytest.param("grid", ("r_g_off",), "0", id="line-quoted-zero"),
    pytest.param("grid", ("v_g",), True, id="line-boolean-gate"),
    pytest.param("grid", ("sweep", "x_points"), "2", id="line-quoted-count"),
    pytest.param("pareto", ("boundary", "t_j"), "25.0000000000000000001", id="population-quoted-temperature"),
    pytest.param("pareto", ("candidates", 0, "r_g_on"), "1e-400", id="population-quoted-underflow"),
    pytest.param("pareto", ("boundary", "voltage_residual_tolerance_v"), "0", id="population-quoted-zero"),
    pytest.param("pareto", ("candidates", 0, "v_g_off"), False, id="population-boolean-gate"),
    pytest.param("pareto", ("frequencies_hz", 0), "10000", id="population-quoted-frequency"),
])
def test_bounded_http_rejects_nonnumeric_tokens_before_source_access(tmp_path, monkeypatch, study, field, value):
    repo = JsonTransistorRepository(tmp_path / "source")
    repo.save(ideal_drop_device("00123"))
    reads = []
    original_path = repo._storage_path

    def observe_source(name):
        reads.append(name)
        return original_path(name)

    monkeypatch.setattr(repo, "_storage_path", observe_source)
    monkeypatch.setattr(main, "_repo", repo)
    body = _http_switching_request(study, "00123")
    position = body
    for key in field[:-1]:
        position = position[key]
    position[field[-1]] = value
    response = TestClient(app).post("/api/topology/switching-frequency-" + study, json=body)
    assert response.status_code == 422
    assert reads == []


@pytest.mark.parametrize("study", ["grid", "pareto"])
def test_bounded_http_preserves_numeric_identity_labels_and_known_zero(tmp_path, monkeypatch, study):
    repo = JsonTransistorRepository(tmp_path / "source")
    part = ideal_drop_device("00123", on_energy=0, off_energy=0)
    for curve in part.switch.e_on_data + part.switch.e_off_data:
        curve.r_g = 0
    repo.save(part)
    saved = repo._storage_path("00123")
    initial = saved.read_bytes()
    monkeypatch.setattr(main, "_repo", repo)
    body = _http_switching_request(study, "00123")
    conditions = [body] if study == "grid" else body["candidates"]
    for condition in conditions:
        condition.update(r_g_on=0, r_g_off=0, diode_v_g=None)
    response = TestClient(app).post("/api/topology/switching-frequency-" + study, json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["source_snapshots"]["00123"]["sha256"] == sha256(initial).hexdigest()
    if study == "grid":
        assert result["sweep"]["values"] == [0, 0]
        assert all(row["status"] == "qualified" for row in result["sweep"]["rows"])
    else:
        for row in result["rows"]:
            entries = row["population"]["candidates"]
            assert [entry["candidate_id"] for entry in entries] == ["0001", "25.0000000000000000001"]
            assert [entry["label"] for entry in entries] == ["0001", "25.0000000000000000001"]
            assert all(entry["eligible"] and entry["objectives"]["included_loss_w"] == 0 for entry in entries)
    assert saved.read_bytes() == initial





