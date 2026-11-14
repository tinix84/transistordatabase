"""Independent engineering and contract checks for the native waveform adapter."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from native_waveform_fixture import direct_request, inputs, omit_switching
from transistordatabase.native_waveform import (
    NativeWaveformError, evaluate_native_waveform, prepare_native_waveform_request,
)
from transistordatabase.waveform_losses import analyze_waveform


def canonical_hash(value):
    """Use the documented sorted compact JSON convention independently."""
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_closed_form_full_window_and_independent_engine_oracle():
    device, selection, case = inputs()
    record = evaluate_native_waveform(device, selection, case)
    result = record["analysis"]
    assert result["duration_s"] == .004
    assert result["total_status"] == "computed"
    for term, watts, joules in (
        ("conduction", 2., .008), ("turn_on", .5, .002), ("turn_off", .75, .003)
    ):
        assert result["components"][term]["status"] == "computed"
        assert result["components"][term]["power_w"] == pytest.approx(watts)
        assert result["components"][term]["energy_j"] == pytest.approx(joules)
    assert result["total_power_w"] == pytest.approx(3.25)
    assert result["total_energy_j"] == pytest.approx(.013)
    expected = analyze_waveform(direct_request())
    assert result["components"] == expected["components"]
    assert result["current_statistics"] == expected["current_statistics"]
    assert [event["sample_index"] for event in result["events"]] == [0, 2]
    assert result["events"][0]["bracket_s"] == [0., .001]
    assert result["events"][1]["current_a"] == 2.
    assert record["temperature"] == {"input_K": 298.15, "imposed_c": 25., "status": "imposed"}
    assert record["condition_matching_policy"] == {
        "numeric_atol": 1e-9, "numeric_rtol": 0.,
        "text": "exact", "null": "matches-only-null"}
    assert not any(key in record for key in ("efficiency", "feasible", "junction_temperature_K"))


def test_ramp_energy_catches_axis_swap_or_scalar_rms_substitution():
    device, selection, case = inputs()
    omit_switching(selection, case)
    case["waveform"] = {"time_s": [0., .001], "current_a": [0., 2.]}
    result = evaluate_native_waveform(device, selection, case)["analysis"]
    assert result["components"]["conduction"]["energy_j"] == pytest.approx(.5 * 4 * .001 / 3)
    assert result["total_power_w"] is None
    assert result["total_status"] == "partial-explicit-omissions"


def test_separately_selected_negative_branch_is_not_mirrored():
    device, selection, case = inputs()
    case["waveform"]["current_a"] = [-2.] * 5
    selection["conduction"]["negative"] = {
        "component": "diode", "record_index": 0,
        "requested_conditions": {"t_j_c": 25., "v_g_v": None}}
    for term in ("turn_on", "turn_off"):
        negative = deepcopy(selection[term]["positive"])
        negative["record_index"] = 1
        selection[term]["negative"] = negative
    result = evaluate_native_waveform(device, selection, case)["analysis"]
    assert result["components"]["conduction"]["power_w"] == pytest.approx(4.)
    assert result["total_power_w"] == pytest.approx(6.5)
    assert all(event["direction"] == "negative" for event in result["events"])
    selection["conduction"]["negative"] = None
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "missing-model"


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(input_temperature_K=323.15),
    lambda s: s["conduction"]["positive"]["requested_conditions"].update(v_g_v=None),
    lambda s: s["turn_on"]["positive"]["requested_conditions"].update(r_g_ohm=3.),
    lambda s: s["turn_off"]["positive"]["requested_conditions"].update(v_g_off_v=-5.),
    lambda s: s["turn_on"]["positive"]["requested_context"].update(method="different"),
    lambda s: s["turn_on"]["positive"]["requested_context"].update(commutation_device="known"),
])
def test_numeric_and_text_condition_mismatches_reject(mutation):
    device, selection, case = inputs()
    mutation(selection)
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "condition-mismatch"


def test_bus_voltage_queries_only_the_active_event_samples():
    device, selection, case = inputs()
    case["waveform"]["bus_voltage_v"] = [400., 410., 400., 420., 400.]
    assert evaluate_native_waveform(device, selection, case)["analysis"]["total_power_w"] == pytest.approx(3.25)
    case["waveform"]["bus_voltage_v"][2] = 420.
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code in {"condition-mismatch", "unsupported-domain"}


def test_scalar_and_vector_bus_have_same_normalized_identity():
    device, selection, case = inputs()
    scalar = evaluate_native_waveform(device, selection, case)
    case["waveform"]["bus_voltage_v"] = [400.] * 5
    vector = evaluate_native_waveform(device, selection, case)
    assert scalar == vector
    case["candidate_id"] = "other-candidate"
    other = evaluate_native_waveform(device, selection, case)
    assert other["waveform_sha256"] == scalar["waveform_sha256"]
    assert other["event_policy_sha256"] == scalar["event_policy_sha256"]
    assert other["request_sha256"] != scalar["request_sha256"]


def test_irregular_window_no_initial_or_wrapped_gate_event():
    device, selection, case = inputs()
    case["waveform"]["time_s"] = [0., .001, .003, .004, .010]
    case["waveform"]["gate_v"] = [15., 15., 0., 0., 0.]
    case["events"] = {"mode": "gate", "threshold_v": 7.5}
    record = evaluate_native_waveform(device, selection, case)
    result = record["analysis"]
    assert result["duration_s"] == .010
    assert len(result["events"]) == 1
    assert result["events"][0]["event"] == "turn_off"
    assert result["events"][0]["sample_index"] == 1
    assert result["components"]["turn_on"]["event_count"] == 0
    assert result["components"]["turn_off"]["power_w"] == pytest.approx(.3)


def test_hysteresis_keeps_explicit_initial_state_and_same_preedge_semantics():
    device, selection, case = inputs()
    case["waveform"]["gate_v"] = [5., 10., 5., 0., 10.]
    case["events"] = {"mode": "hysteresis", "digitizer": {
        "thresholds": {"scale": "absolute", "low_v": 2.5, "high_v": 7.5},
        "initial_state": False}}
    result = evaluate_native_waveform(device, selection, case)["analysis"]
    assert [(event["event"], event["sample_index"]) for event in result["events"]] == [
        ("turn_on", 0), ("turn_off", 2), ("turn_on", 3)]
    assert result["components"]["turn_on"]["event_count"] == 2


def test_known_zero_and_explicit_omissions_are_distinct():
    device, selection, case = inputs()
    for family in (device.switch.e_on_data, device.switch.e_off_data):
        family[0].graph_i_e[1] = 0.
    result = evaluate_native_waveform(device, selection, case)["analysis"]
    assert result["components"]["turn_on"]["power_w"] == 0.
    assert result["total_power_w"] == pytest.approx(2.)
    omit_switching(selection, case)
    omitted = evaluate_native_waveform(device, selection, case)["analysis"]
    assert omitted["components"]["turn_on"]["power_w"] is None
    assert omitted["known_component_power_w"] == pytest.approx(2.)
    assert omitted["total_power_w"] is None


def test_no_events_does_not_bypass_declared_model_requirement():
    device, selection, case = inputs()
    case["events"]["turn_on"] = [False] * 4
    case["events"]["turn_off"] = [False] * 4
    selection["turn_on"]["positive"] = None
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "missing-model"


def test_complete_hashes_and_input_snapshot_are_independently_reproducible():
    device, selection, case = inputs()
    record = evaluate_native_waveform(device, selection, case)
    assert record["request_sha256"] == canonical_hash(record["native_request_snapshot"])
    assert len(record["source_records"]) == 6
    assert record["source_records"][1] is None
    assert record["selected_model_sha256"] == canonical_hash(record["source_records"])
    unhashed = {key: value for key, value in record.items()
                if key not in {"evaluation_record_sha256", "evaluation_id"}}
    digest = canonical_hash(unhashed)
    assert record["evaluation_record_sha256"] == digest
    assert record["evaluation_id"] == "native-waveform-v1:" + digest
    changed = deepcopy(unhashed)
    changed["analysis"]["total_power_w"] += 1.
    assert canonical_hash(changed) != digest


def test_determinism_source_identity_and_caller_objects_remain_unchanged():
    device, selection, case = inputs()
    saved_selection, saved_case = deepcopy(selection), deepcopy(case)
    graph = device.switch.channel_data[0].graph_v_i.copy()
    first = evaluate_native_waveform(device, selection, case)
    assert first == evaluate_native_waveform(device, selection, case)
    assert selection == saved_selection and case == saved_case
    np.testing.assert_array_equal(device.switch.channel_data[0].graph_v_i, graph)
    device.switch.channel_data[0].graph_v_i[0, 1] = 2.2
    changed = evaluate_native_waveform(device, selection, case)
    assert changed["selected_model_sha256"] != first["selected_model_sha256"]
    assert changed["evaluation_id"] != first["evaluation_id"]


def test_prepared_request_reuses_the_numerical_engine():
    device, selection, case = inputs()
    prepared = prepare_native_waveform_request(device, selection, case)
    assert "native_request_snapshot" not in prepared
    assert analyze_waveform(prepared) == evaluate_native_waveform(device, selection, case)["analysis"]


@pytest.mark.parametrize("mutation", [
    lambda s, c: c.update(extra=True),
    lambda s, c: c["waveform"].update(duty=.5),
    lambda s, c: c["waveform"].update(current_a=[True] * 5),
    lambda s, c: c["waveform"].update(current_a=["2"] * 5),
    lambda s, c: c["waveform"].update(current_a=[float("nan")] * 5),
    lambda s, c: c["waveform"].update(time_s=[0., .001, .001, .003, .004]),
    lambda s, c: c["events"].update(turn_on=[1, 0, 0, 0]),
    lambda s, c: c["events"].update(turn_off=[True, False, False, False]),
    lambda s, c: s["conduction"]["positive"].update(record_index=True),
    lambda s, c: s["conduction"]["positive"].update(record_index=-1),
    lambda s, c: s["conduction"]["positive"].update(record_index=999),
    lambda s, c: s["turn_on"]["positive"].update(component="diode"),
    lambda s, c: s.update(input_temperature_K=float("inf")),
])
def test_invalid_shapes_types_extra_fields_and_indices_reject(mutation):
    device, selection, case = inputs()
    mutation(selection, case)
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code in {"invalid-input", "missing-model", "unsupported-domain"}


def test_iterators_are_rejected_without_consumption():
    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("Unbounded iterable was consumed")
    device, selection, case = inputs()
    case["waveform"]["current_a"] = MustNotIterate()
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_unsupported_energy_representation_is_not_fitted_or_extrapolated():
    device, selection, case = inputs()
    device.switch.e_on_data[0].dataset_type = "single"
    device.switch.e_on_data[0].e_x = .002
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "unsupported-domain"


def test_current_domain_exit_is_not_clamped():
    device, selection, case = inputs()
    case["waveform"]["current_a"] = [5.] * 5
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "unsupported-domain"


def test_signed_source_curve_is_not_silently_rectified():
    device, selection, case = inputs()
    device.switch.channel_data[0].graph_v_i[1, 0] = -1.
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_nonzero_extended_precision_source_is_not_converted_to_zero():
    tiny = np.longdouble("1e-4000")
    if tiny == 0:
        pytest.skip("Extended precision unavailable")
    device, selection, case = inputs()
    graph = device.switch.channel_data[0].graph_v_i.astype(np.longdouble)
    graph[0, 1] = tiny
    device.switch.channel_data[0].graph_v_i = graph
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_raw_measurement_payload_is_not_embedded_and_comment_is_hashed():
    device, selection, case = inputs()
    record = device.switch.e_on_data[0]
    record.evidence.original_base64 = "UNIQUE-RAW-PAYLOAD" * 10000
    record.evidence.companion_log_base64 = "UNIQUE-RAW-LOG" * 10000
    record.comment = "selected source comment " * 300
    result = evaluate_native_waveform(device, selection, case)
    encoded = json.dumps(result, allow_nan=False)
    assert "UNIQUE-RAW-PAYLOAD" not in encoded and "UNIQUE-RAW-LOG" not in encoded
    assert "excluded_payload_fields" in encoded
    assert sha256(record.comment.encode()).hexdigest() in encoded


def test_waveform_sample_boundary_and_one_beyond():
    device, selection, case = inputs()
    omit_switching(selection, case)
    selection["conduction"] = {"mode": "omitted", "reason": "Bound-only fixture"}
    case["waveform"] = {"time_s": np.arange(10000, dtype=float).tolist(),
                        "current_a": [0.] * 10000}
    assert evaluate_native_waveform(device, selection, case)["analysis"]["duration_s"] == 9999.
    case["waveform"]["time_s"].append(10000.)
    case["waveform"]["current_a"].append(0.)
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "resource-limit"


def test_documented_production_resource_limits():
    import transistordatabase.native_waveform as native
    assert native.MAX_REQUEST_BYTES == 4 * 1024 * 1024
    assert native.MAX_OUTPUT_BYTES == 16 * 1024 * 1024
    assert native.MAX_SAMPLES == native.MAX_CURVE_POINTS == 10000
    assert native.MAX_SELECTED_POINTS == 60000
    assert native.MAX_SELECTED_RECORDS == 6
    assert native.MAX_SEGMENTS == 10000
    assert native.MAX_PLOT_RECORDS == 90000
    assert native.MAX_METADATA_DEPTH == 8
    assert native.MAX_METADATA_SCALARS == 128
    assert native.MAX_METADATA_STRING == 2000


def test_request_byte_budget_exact_boundary_and_one_beyond(monkeypatch):
    import transistordatabase.native_waveform as native
    device, selection, case = inputs()
    original = evaluate_native_waveform(device, selection, case)
    snapshot = original["native_request_snapshot"]
    expected_bytes = len(json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode())
    monkeypatch.setattr(native, "MAX_REQUEST_BYTES", expected_bytes)
    assert evaluate_native_waveform(device, selection, case)["request_sha256"] == original["request_sha256"]
    monkeypatch.setattr(native, "MAX_REQUEST_BYTES", expected_bytes - 1)
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "resource-limit"


def test_output_byte_budget_exact_boundary_and_one_beyond(monkeypatch):
    import transistordatabase.native_waveform as native
    device, selection, case = inputs()
    original = evaluate_native_waveform(device, selection, case)
    expected_bytes = len(json.dumps(original, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode())
    monkeypatch.setattr(native, "MAX_OUTPUT_BYTES", expected_bytes)
    assert evaluate_native_waveform(device, selection, case) == original
    monkeypatch.setattr(native, "MAX_OUTPUT_BYTES", expected_bytes - 1)
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "resource-limit"


def test_metadata_text_boundary_and_nested_metadata_rejection():
    device, selection, case = inputs()
    case["provenance"]["calibration_notes"] = "Ω" * 2000
    assert evaluate_native_waveform(device, selection, case)["analysis"]["total_power_w"] == pytest.approx(3.25)
    case["provenance"]["calibration_notes"] += "Ω"
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "resource-limit"
    case["provenance"]["calibration_notes"] = {"nested": {"not": "admitted"}}
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_record_owned_measurement_date_and_raw_payload_exclusion_identity():
    from datetime import date
    device, selection, case = inputs()
    device.switch.e_on_data[0].measurement_date = date(2026, 11, 13)
    first = evaluate_native_waveform(device, selection, case)
    assert "2026-11-13" in json.dumps(first["source_records"])
    device.switch.e_on_data[0].evidence.original_base64 = "changed-opaque-payload"
    second = evaluate_native_waveform(device, selection, case)
    assert second["selected_model_sha256"] == first["selected_model_sha256"]
    assert second["evaluation_id"] == first["evaluation_id"]


@pytest.mark.parametrize("value", [None, [], {}, True])
def test_malformed_temperature_and_enum_fields_have_stable_errors(value):
    device, selection, case = inputs()
    selection["turn_on"]["positive"]["requested_conditions"]["t_j_c"] = value
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)
    device, selection, case = inputs()
    case["provenance"]["origin_kind"] = value
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_non_text_object_keys_reject_without_sorting_or_type_leaks():
    device, selection, case = inputs()
    case[123] = "bad"
    case[(1, 2)] = "bad"
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_source_comment_hash_budget_exact_boundary_and_one_beyond(monkeypatch):
    import transistordatabase.native_waveform as native
    device, selection, case = inputs()
    device.switch.e_on_data[0].comment = "1234567890"
    monkeypatch.setattr(native, "MAX_SOURCE_COMMENT_BYTES", 10)
    result = evaluate_native_waveform(device, selection, case)
    assert sha256(b"1234567890").hexdigest() in json.dumps(result)
    device.switch.e_on_data[0].comment += "1"
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "resource-limit"



def test_curve_sample_boundary_and_one_beyond():
    device, selection, case = inputs()
    x = np.linspace(0., 4., 10000)
    device.switch.channel_data[0].graph_v_i = np.array([x * .5, x])
    assert evaluate_native_waveform(device, selection, case)["analysis"]["total_power_w"] == pytest.approx(3.25)
    x = np.linspace(0., 4., 10001)
    device.switch.channel_data[0].graph_v_i = np.array([x * .5, x])
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "resource-limit"


def test_fresh_subprocess_rejects_gui_imports():
    repository = Path(__file__).resolve().parents[2]
    script = r"""
import importlib.abc, sys
class RejectGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("PyQt", "PySide", "transistordatabase.gui", "ntbees2", "fastapi")):
            raise AssertionError("Forbidden adapter import: " + fullname)
sys.meta_path.insert(0, RejectGui())
sys.path.insert(0, str(__import__("pathlib").Path("tests/headless").resolve()))
from native_waveform_fixture import inputs
from transistordatabase.native_waveform import evaluate_native_waveform
assert __import__("math").isclose(evaluate_native_waveform(*inputs())["analysis"]["total_power_w"], 3.25, rel_tol=1e-12)
print("headless PASS")
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=repository,
                            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headless PASS" in result.stdout


def test_returned_snapshots_do_not_alias_inputs_or_other_result_fields():
    device, selection, case = inputs()
    record = evaluate_native_waveform(device, selection, case)
    saved = deepcopy(case)
    record["native_request_snapshot"]["case"]["waveform"]["current_a"][0] = 999.
    assert case == saved
    assert record["analysis"]["signals"]["current_a"][0] == 2.
    record["analysis"]["request"]["waveform"]["current_a"][1] = 888.
    assert case == saved
    np.testing.assert_array_equal(device.switch.channel_data[0].graph_v_i,
                                  [[0., 2.], [0., 4.]])
    assert evaluate_native_waveform(device, selection, case)["analysis"]["total_power_w"] == pytest.approx(3.25)


@pytest.mark.parametrize("mutation", [
    lambda s, c: c.update(schema_version=True),
    lambda s, c: s.update(schema_version=True),
    lambda s, c: c["provenance"].update(extra="unexpected"),
    lambda s, c: c["provenance"].pop("source_id"),
    lambda s, c: s["turn_on"]["positive"]["requested_context"].update(extra=None),
    lambda s, c: s["turn_on"]["positive"]["requested_context"].pop("event"),
    lambda s, c: s["turn_on"].update(third=None),
    lambda s, c: s["turn_off"].update(mode="omitted", reason="test"),
])
def test_nested_schema_fields_and_boolean_versions_reject(mutation):
    device, selection, case = inputs()
    mutation(selection, case)
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_non_ascii_identity_and_waveform_policy_hash_sensitivity():
    device, selection, case = inputs()
    case["candidate_id"] = "München-半導体"
    case["provenance"]["source_id"] = "Messung-Ω"
    first = evaluate_native_waveform(device, selection, case)
    case["waveform"]["bus_voltage_v"] = [400.] * 5
    assert first == evaluate_native_waveform(device, selection, case)
    case["events"]["turn_on"] = [False, True, False, False]
    different_policy = evaluate_native_waveform(device, selection, case)
    assert first["waveform_sha256"] == different_policy["waveform_sha256"]
    assert first["event_policy_sha256"] != different_policy["event_policy_sha256"]
    case["waveform"]["current_a"][0] = 1.
    different_waveform = evaluate_native_waveform(device, selection, case)
    assert different_waveform["waveform_sha256"] != different_policy["waveform_sha256"]


def test_no_event_still_validates_selected_native_record():
    device, selection, case = inputs()
    case["events"]["turn_on"] = [False] * 4
    case["events"]["turn_off"] = [False] * 4
    device.switch.e_on_data[0].graph_i_e[0, 1] = 0.
    with pytest.raises(NativeWaveformError):
        evaluate_native_waveform(device, selection, case)


def test_all_six_directional_sources_accept_60000_total_curve_points():
    device, selection, case = inputs()
    x = np.linspace(0., 4., 10000)
    device.switch.channel_data[0].graph_v_i = np.array([x * .5, x])
    device.diode.channel_data[0].graph_v_i = np.array([x, x])
    selection["conduction"]["negative"] = {
        "component": "diode", "record_index": 0,
        "requested_conditions": {"t_j_c": 25., "v_g_v": None}}
    for term, family, joules in (
        ("turn_on", device.switch.e_on_data, .002),
        ("turn_off", device.switch.e_off_data, .003),
    ):
        for item in family:
            item.graph_i_e = np.array([x, np.full(10000, joules)])
        selection[term]["negative"] = deepcopy(selection[term]["positive"])
        selection[term]["negative"]["record_index"] = 1
    result = evaluate_native_waveform(device, selection, case)
    assert all(source is not None for source in result["source_records"])
    assert result["analysis"]["total_power_w"] == pytest.approx(3.25)


def test_exact_segment_and_plot_budget_and_one_beyond():
    device, selection, case = inputs()
    omit_switching(selection, case)
    x = np.arange(10000, dtype=float)
    device.switch.channel_data[0].graph_v_i = np.array([x * .5, x])
    device.diode.channel_data[0].graph_v_i = np.array([[0., 1.], [0., 1.]])
    selection["conduction"]["negative"] = {
        "component": "diode", "record_index": 0,
        "requested_conditions": {"t_j_c": 25., "v_g_v": None}}
    case["waveform"] = {"time_s": [0., 1.], "current_a": [-1., 9999.]}
    result = evaluate_native_waveform(device, selection, case)["analysis"]
    assert len(result["power_segments"]) == 10000
    assert len(result["conduction_plot"]) == 90000
    # Integral of R*i^2 along a linear ramp, integrated separately at zero.
    expected = (1. / 3. + .5 * 9999.**3 / 3.) / 10000.
    assert result["components"]["conduction"]["energy_j"] == pytest.approx(expected)
    device.diode.channel_data[0].graph_v_i = np.array([[0., 1., 2.], [0., 1., 2.]])
    case["waveform"]["current_a"][0] = -2.
    with pytest.raises(NativeWaveformError) as error:
        evaluate_native_waveform(device, selection, case)
    assert error.value.code == "resource-limit"


