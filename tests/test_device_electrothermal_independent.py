"""Independent fixed-current, temperature-domain and thermal-equilibrium oracles."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, MeasurementEvidence, SwitchingLossData,
    ThermalProperties, Transistor, TransistorMetadata,
)
from transistordatabase.core.repository import JsonTransistorLoader, JsonTransistorRepository
from transistordatabase.core.serialization import core_to_json_document
from transistordatabase.device_electrothermal import analyze_device_electrothermal, build_device_loss_table
from transistordatabase.gui_web.backend import main
from transistordatabase.gui_web.backend.device_electrothermal_routes import device_repository


def switch_device():
    device = Transistor(
        metadata=TransistorMetadata(name="Independent device", type="IGBT", author="Test",
                                    manufacturer="Fixture", housing_type="Fixture"),
        electrical=ElectricalRatings(1200, 100, 50, 175),
        thermal=ThermalProperties(.001, .001, .1, .1, .1))
    device.switch.channel_data = [
        ChannelCharacteristics(t, np.array([[0, 2 * v10], [0, 20]]), 15)
        for t, v10 in ((0, .2), (20, .4), (100, 1.2))]
    for attr, vg, rg, curve in (
        ("e_on_data", 15, 4, ((10, 1), (60, 4), (110, 5))),
        ("e_off_data", -15, 7, ((-10, 2), (40, 3), (80, 5))),
    ):
        # Interpolating at 10 A is required; zero-current energy is deliberately positive.
        records = [SwitchingLossData("graph_i_e", t, 400, vg, r_g=rg, v_g_off=None,
                                     graph_i_e=np.array([[0, 20], [.0005, .002 * energy_mj - .0005]]))
                   for t, energy_mj in curve]
        setattr(device.switch, attr, records)
    return device


def selection(gate, resistor=None, count=3):
    conditions = {"v_g": gate}
    if resistor is not None:
        conditions.update(v_g_off=None, r_g=resistor, v_supply=400)
    return {"indices": list(range(count)), "conditions": conditions}


def point(**changes):
    return dict(selected=True, switching_policy="held-current-pwm", source="Independent thermal fixture",
                current_a=10, duty=.5, frequency_hz=1000, sink_temperature_c=25, resistance_k_w=1,
                initial_temperature_c=25, tolerance_k=1e-10, max_iterations=100, relaxation=1,
                channel=selection(15), turn_on=selection(15, 4), turn_off=selection(-15, 7)) | changes


def test_independent_domains_kinks_and_fixed_point():
    data = analyze_device_electrothermal(switch_device(), point())
    table = data["table"]
    assert table["temperature_domain_c"] == [10, 80]
    assert table["current_domain_a"] == [0, 20]
    assert table["temperature_c"] == [10, 20, 40, 60, 80]
    assert table["component_power_w"]["channel"] == pytest.approx([1.5, 2, 3, 4, 5])
    assert table["component_power_w"]["turn_on"] == pytest.approx([1, 1.6, 2.8, 4, 4.4])
    assert table["component_power_w"]["turn_off"] == pytest.approx([2.4, 2.6, 3, 4, 5])
    assert table["total_power_w"] == pytest.approx([4.9, 6.2, 8.8, 12, 14.4])
    # Within the second interval P(T) = 3.6 + .13 T W. Solve T = 25 + P(T).
    expected = 28.6 / .87
    accepted = data["result"]["accepted_solution"]
    assert data["result"]["status"] == "converged"
    assert accepted["temperature_c"] == pytest.approx([expected], abs=1e-9)
    assert accepted["power_w"] == pytest.approx([expected - 25], abs=1e-9)
    assert abs(accepted["residual_k"][0]) <= 1e-10
    components = data["accepted_component_power_w"]
    assert components["sum_component_power_w"] == pytest.approx(accepted["power_w"][0])
    for row in table["record_evidence"]:
        family, conditions = row["family"], row["recorded_conditions"]
        assert conditions["v_g"] == (-15 if family == "turn_off" else 15)
        if family != "channel":
            assert conditions["v_g_off"] is None


def test_zero_current_switching_still_deposits_measured_energy():
    source = switch_device()
    source.switch.channel_data = []
    data = analyze_device_electrothermal(source, point(current_a=0, channel=None))
    table = data["table"]
    assert table["active_contributions"] == {"channel": False, "turn_on": True, "turn_off": True}
    assert table["total_power_w"] == pytest.approx([1] * len(table["temperature_c"]))
    assert data["result"]["accepted_solution"]["temperature_c"] == pytest.approx([26])
    assert data["accepted_component_power_w"]["channel"] == 0


def test_frequency_zero_uses_only_actual_channel_temperature_feedback():
    source = switch_device()
    source.switch.channel_data = [
        ChannelCharacteristics(t, np.array([[0, .2 * (1 + .004 * (t - 25))], [0, 20]]), 15)
        for t in (25, 125)]
    source.switch.e_on_data = []
    source.switch.e_off_data = []
    data = analyze_device_electrothermal(source, point(frequency_hz=0, resistance_k_w=10,
                                                       channel=selection(15, count=2), turn_on=None, turn_off=None))
    assert data["table"]["active_contributions"] == {"channel": True, "turn_on": False, "turn_off": False}
    assert data["table"]["total_power_w"] == pytest.approx([.5, .7])
    assert data["result"]["accepted_solution"]["temperature_c"][0] == pytest.approx(25 + 5 / .98, abs=1e-9)


@pytest.mark.parametrize("failure", ["missing-on", "same-temperature", "gate", "off-field", "current", "no-common-T"])
def test_active_missing_ambiguous_or_mismatched_sources_reject(failure):
    source = switch_device()
    request = point()
    if failure == "missing-on":
        source.switch.e_on_data = []
    elif failure == "same-temperature":
        source.switch.e_on_data[1].t_j = source.switch.e_on_data[0].t_j
    elif failure == "gate":
        request["turn_off"]["conditions"]["v_g"] = 15
    elif failure == "off-field":
        request["turn_off"]["conditions"].update(v_g=None, v_g_off=-15)
    elif failure == "current":
        request["current_a"] = 21
    else:
        for record in source.switch.e_off_data:
            record.t_j += 200
    with pytest.raises(ValueError):
        analyze_device_electrothermal(source, request)


@pytest.mark.parametrize("changes,status", [({"max_iterations": 1}, "iteration-limit"),
                                            ({"resistance_k_w": 100}, "source-domain-exit")])
def test_every_unaccepted_solution_keeps_residual_and_source_boundaries(changes, status):
    data = analyze_device_electrothermal(switch_device(), point(**changes))
    result = data["result"]
    assert result["status"] == status
    assert result["accepted_solution"] is None and data["accepted_component_power_w"] is None
    assert result["history"][0]["temperature_c"] == [25]
    assert result["history"][0]["power_w"] == pytest.approx([6.85])
    if status == "source-domain-exit":
        assert result["history"][-1]["temperature_c"][0] > 80
        assert result["history"][-1]["power_w"] is None


def test_provenance_digest_changes_only_with_selected_physics_and_all_sources_survive():
    source = switch_device()
    request = point()
    repository_source = {"sha256": "1" * 64}
    old_request = deepcopy(request)
    old_document = json.dumps(core_to_json_document(source), sort_keys=True)
    first = build_device_loss_table(source, request, repository_source=repository_source)
    assert old_document == json.dumps(core_to_json_document(source), sort_keys=True) and old_request == request
    first["source_identity"]["repository_source"]["sha256"] = "2" * 64
    assert repository_source["sha256"] == "1" * 64
    first_digest = first["source_identity"]["selected_dependency_sha256"]
    source.metadata.author = "A different fixture author"
    assert build_device_loss_table(source, request)["source_identity"]["selected_dependency_sha256"] == first_digest
    source.switch.e_on_data[0].graph_i_e[1, 1] += .0001
    assert build_device_loss_table(source, request)["source_identity"]["selected_dependency_sha256"] != first_digest


@pytest.mark.parametrize("mode,expected", [
    ("recorded", "5136a17e5dc27e78ed087cf54ae121f5baac36e17901545b73402aaaa5ce2e98"),
    ("linear-temperature-estimate", "9d075f707d8f2261da891479f21bee04de4731b722bea83457df6ec385192b05"),
])
def test_manual_thermal_route_matches_the_merged_main_response(mode, expected):
    model = dict(source="Independent calibrated source", conditions={"current_a": 10},
                 temperature_min_c=25, temperature_max_c=125, mode=mode)
    if mode == "recorded":
        model.update(temperature_c=[25, 125], power_w=[.5, .7])
    else:
        model.update(selected=True, reference_temperature_c=25, reference_power_w=.5, coefficient_per_k=.004)
    request = dict(operation="electrothermal", source="Independent manual thermal fixture", conditions={},
                   sink_temperature_c=25, common_resistance_k_w=0, branch_resistances_k_w=[10],
                   initial_temperatures_c=[25], tolerance_k=1e-9, relaxation=1.0, max_iterations=100,
                   loss_models=[model])
    response = TestClient(main.app).post("/api/thermal/analyze", json=request)
    assert response.status_code == 200, response.text
    canonical = json.dumps(response.json(), sort_keys=True, separators=(",", ":")).encode()
    assert sha256(canonical).hexdigest() == expected


def test_real_four_temperature_fuji_source_matches_hand_interpolated_equilibrium():
    path = Path(__file__).resolve().parents[1] / "database" / "Fuji_2MBI100XAA120-50.json"
    if not path.is_file():
        pytest.skip("Optional local four-temperature Fuji qualification record absent")
    fingerprint = sha256(path.read_bytes()).hexdigest()
    assert fingerprint == "4a4bc6bc5f5d287a843a4961af8591685b2b02cc33adcb036e9f9e6c4eef4a27"
    device = JsonTransistorLoader().load_from_json(path)
    request = point(current_a=50, resistance_k_w=.1, initial_temperature_c=45, sink_temperature_c=45,
                    channel=selection(15, count=4), turn_on=selection(15, 5.6, 4), turn_off=selection(-15, 5.6, 4))
    for family in ("turn_on", "turn_off"):
        request[family]["conditions"]["v_supply"] = 600
    data = analyze_device_electrothermal(device, request)
    assert data["table"]["temperature_c"] == [25, 125, 150, 175]
    assert data["table"]["total_power_w"] == pytest.approx([36.1497512815, 42.3810821848, 43.9218932858, 45.5661834884])
    accepted = data["result"]["accepted_solution"]
    assert accepted["temperature_c"][0] == pytest.approx(48.763050559454264, abs=1e-9)
    assert accepted["power_w"][0] == pytest.approx(37.63050559454264, abs=1e-9)
    assert sha256(path.read_bytes()).hexdigest() == fingerprint

@pytest.mark.parametrize("field,low,high", [
    ("commutation_device", "Rectifier A", "Rectifier B"),
    ("load_inductance", 1e-6, 2e-6),
    ("commutation_inductance", None, 30e-9),
    ("commutation_inductance", 1e-9, 1.5e-9),
    ("measurement_testbench", "Fixture A", "Fixture B"),
])
def test_mixing_recorded_commutation_setups_cannot_create_temperature_trends(field, low, high):
    device = switch_device()
    for row in device.switch.e_on_data:
        setattr(row, field, low)
    setattr(device.switch.e_on_data[-1], field, high)
    with pytest.raises(ValueError):
        build_device_loss_table(device, point())


@pytest.mark.parametrize("field,lib_a,lib_b", [
    ("method", "integrated", "datasheet"),
    ("convention", "signed", "positive"),
    ("event", "turn_on", "turn_off"),
])
def test_switching_measurement_convention_is_part_of_the_fixed_family(field, lib_a, lib_b):
    device = switch_device()
    for row in device.switch.e_on_data:
        row.evidence = MeasurementEvidence()
        setattr(row.evidence, field, lib_a)
    setattr(device.switch.e_on_data[-1].evidence, field, lib_b)
    with pytest.raises(ValueError):
        build_device_loss_table(device, point())


def test_a_consistent_but_different_commutation_device_changes_the_dependency_digest():
    device = switch_device()
    for row in device.switch.e_on_data:
        row.commutation_device = "Rectifier A"
        row.measurement_testbench = "Fixture A"
        row.evidence = MeasurementEvidence(event="turn_on", method="integrated", convention="signed")
    table = build_device_loss_table(device, point())
    selected_rows = [r for r in table["record_evidence"] if r["family"] == "turn_on"]
    for row in selected_rows:
        conditions = row["recorded_conditions"]
        assert conditions["measurement_testbench"] == "Fixture A"
        assert conditions["event"] == "turn_on" and conditions["method"] == "integrated"
        assert conditions["convention"] == "signed"
    first = table["source_identity"]["selected_dependency_sha256"]
    for row in device.switch.e_on_data:
        row.commutation_device = "Rectifier B"
    second = build_device_loss_table(device, point())["source_identity"]["selected_dependency_sha256"]
    assert first != second


def test_extended_precision_interior_duty_cannot_disable_both_events():
    interior = np.nextafter(np.longdouble(1), np.longdouble(0))
    if float(interior) < 1:
        pytest.skip("This platform's longdouble has binary64 precision")
    assert 0 < interior < 1
    device = switch_device()
    with pytest.raises(ValueError):
        build_device_loss_table(device, point(duty=interior))

def test_shuffled_native_curve_brackets_identify_the_original_sample_pair():
    device = switch_device()
    curve = np.array([[20, 0, 5], [.004, .001, .00175]])
    device.switch.e_on_data[0].graph_i_e = curve.copy()
    result = build_device_loss_table(device, point())
    row = next(item for item in result["record_evidence"] if item["record_key"] == "turn_on:0")
    assert row["current_sample_bracket"] == [2, 0]
    assert row["current_sorted_bracket"] == [1, 2]
    assert row["current_bracket_a"] == [5, 20]
    assert row["evaluated_energy_j"] == pytest.approx(.0025)
    np.testing.assert_array_equal(device.switch.e_on_data[0].graph_i_e, curve)

@pytest.fixture
def native_api(tmp_path, monkeypatch):
    """Use the actual application router and one temporary, serialized core record."""
    source = switch_device()
    source.switch.e_on_data[0].graph_i_e = np.array([[20, 0, 5], [.004, .001, .00175]])
    repository = JsonTransistorRepository(tmp_path / "devices")
    repository.save(source)
    path = repository.data_directory / (source.metadata.name + ".json")
    monkeypatch.setitem(main.app.dependency_overrides, device_repository, lambda: repository)
    return TestClient(main.app), source.metadata.name, path


def test_typed_snapshot_hash_and_source_brackets_refer_to_saved_bytes(native_api):
    client, name, path = native_api
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    inventory = client.get(f"/api/thermal/device/inventory/{name}")
    assert inventory.status_code == 200, inventory.text
    summary = inventory.json()
    assert summary["repository_source"]["sha256"] == digest
    assert summary["repository_source"]["byte_length"] == len(raw)
    row = summary["families"]["turn_on"][0]
    assert row["status"] == "available" and row["sample_count"] == 3
    req = point() | {"transistor_id": name, "expected_file_sha256": digest}
    response = client.post("/api/thermal/device/analyze", json=req)
    assert response.status_code == 200, response.text
    data = response.json()
    reference = data["table"]["source_identity"]["repository_source"]
    assert reference == summary["repository_source"]
    assert data["result"]["status"] == "converged"
    ev = next(item for item in data["table"]["record_evidence"] if item["record_key"] == "turn_on:0")
    assert ev["current_sample_bracket"] == [2, 0]
    assert ev["evaluated_energy_j"] == pytest.approx(.0025)
    assert path.read_bytes() == raw


def test_source_token_detects_a_changed_physical_file_even_if_values_are_identical(native_api):
    client, name, path = native_api
    original = path.read_bytes()
    with path.open("ab") as file:
        file.write(b"\n")
    after = path.read_bytes()
    req = point() | {"transistor_id": name, "expected_file_sha256": sha256(original).hexdigest()}
    response = client.post("/api/thermal/device/analyze", json=req)
    assert response.status_code == 409, response.text
    assert path.read_bytes() == after


@pytest.mark.parametrize("token", ["0.99999999999999999999", '"0.5"', "1e-500"])
def test_raw_duty_cannot_be_coerced_into_a_different_switching_policy(native_api, token):
    client, name, path = native_api
    digest = sha256(path.read_bytes()).hexdigest()
    request = point() | {"transistor_id": name, "expected_file_sha256": digest}
    body = json.dumps(request).replace('"duty": 0.5', '"duty": ' + token)
    response = client.post("/api/thermal/device/analyze", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("operation", ["inventory", "analyze"])
def test_repository_parser_depth_error_is_reported_without_changing_device(native_api, monkeypatch, operation):
    client, name, path = native_api
    original = path.read_bytes()

    def report_depth_limit(self, document):
        assert document["schema"] == "transistordatabase-core"
        raise RecursionError("Device nesting exceeds this parser's limit")

    monkeypatch.setattr(JsonTransistorLoader, "load_from_dict", report_depth_limit)
    if operation == "inventory":
        response = client.get(f"/api/thermal/device/inventory/{name}")
    else:
        response = client.post("/api/thermal/device/analyze", json=point() | {"transistor_id": name})
    assert response.status_code == 422, response.text
    assert "nesting" in response.json()["detail"]
    assert path.read_bytes() == original


def test_persisted_positive_energy_cannot_be_rounded_to_a_measured_zero(native_api):
    client, name, path = native_api
    original = path.read_bytes()
    # The fixture's first turn-on record has exactly one 0.00175 J ordinate at 5 A.
    ordinate = b"0.00175"
    assert original.count(ordinate) == 1
    request = point(current_a=5) | {"transistor_id": name}
    for literal in (b"0.0", b"1e-400"):
        raw = original.replace(ordinate, literal)
        path.write_bytes(raw)
        request["expected_file_sha256"] = sha256(raw).hexdigest()
        response = client.post("/api/thermal/device/analyze", json=request)
        assert path.read_bytes() == raw
        if literal == b"0.0":
            assert response.status_code == 200, response.text
            evidence = response.json()["table"]["record_evidence"]
            row = next(item for item in evidence if item["record_key"] == "turn_on:0")
            assert row["evaluated_energy_j"] == 0
        else:
            assert response.status_code == 422, response.text


def test_float32_source_distinguishes_its_smallest_positive_energy_from_underflow(native_api):
    client, name, path = native_api
    document = json.loads(path.read_bytes())
    records = document["transistor"]["fields"]["switch"]["fields"]["e_on_data"]["values"]
    graph = records[0]["fields"]["graph_i_e"]
    graph["dtype"] = "<f4"
    ordinates = graph["values"]["values"][1]["values"]
    ordinates[:] = [float(np.float32(value)) for value in ordinates]
    smallest = float(np.nextafter(np.float32(0), np.float32(1)))
    request = point(current_a=5) | {"transistor_id": name}
    for value in (smallest, 1e-50):
        ordinates[2] = value
        raw = json.dumps(document).encode()
        path.write_bytes(raw)
        request["expected_file_sha256"] = sha256(raw).hexdigest()
        response = client.post("/api/thermal/device/analyze", json=request)
        assert path.read_bytes() == raw
        if value == smallest:
            assert response.status_code == 200, response.text
            evidence = response.json()["table"]["record_evidence"]
            row = next(item for item in evidence if item["record_key"] == "turn_on:0")
            assert row["evaluated_energy_j"] == smallest
        else:
            assert response.status_code == 422, response.text







