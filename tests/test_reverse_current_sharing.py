"""Analytic and adversarial checks for bounded reverse current sharing."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from transistordatabase.reverse_current_sharing import solve_reverse_current_sharing
from transistordatabase.gui_web.backend.current_sharing_routes import router


URL = "/api/assemblies/reverse-sharing"


def branch(kind, current, voltage, gate=15):
    """Declare independent source conditions without a common gate bias assumption."""
    conditions = {"junction_temperature_c": 25, "gate_bias_v": gate, "frequency_hz": None}
    return {"kind": kind, "current_a": current, "voltage_v": voltage,
            "source": f"analytic {kind}", "isolation_evidence": f"independent {kind} fixture",
            "recorded_conditions": conditions, "requested_conditions": conditions.copy()}


def case(current=-10, mode="strict-recorded"):
    """Return Vc=.1 Ic and Vd=.6+.02 Id with isolated branch evidence."""
    return {"selected": True, "compatible_operating_state": True, "current_a": current,
            "junction_temperature_c": 25, "mode": mode,
            "channel": branch("channel-only", [0, 20], [0, 2]),
            "diode": branch("diode-only", [0, 20], [.6, 1], gate=-3)}


@pytest.fixture
def client():
    """Serve the route without project startup or database fixtures."""
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as application:
        yield application


def test_analytic_static_split_and_returned_float_evidence():
    """Check the analytic split and the evidence for returned floating point values."""
    result = solve_reverse_current_sharing(**case())["result"]
    assert result["status"] == "supported"
    assert result["support"] == "recorded"
    assert result["channel_current_a"] == pytest.approx(-20 / 3)
    assert result["diode_current_a"] == pytest.approx(-10 / 3)
    assert result["voltage_v"] == pytest.approx(-2 / 3)
    assert result["channel_static_power_w"] == pytest.approx(40 / 9)
    assert result["diode_static_power_w"] == pytest.approx(20 / 9)
    assert result["channel_static_power_w"] == result["voltage_v"] * result["channel_current_a"]
    assert result["diode_static_power_w"] == result["voltage_v"] * result["diode_current_a"]
    assert result["current_sum_a"] == math.fsum([result["channel_current_a"], result["diode_current_a"]])
    assert result["current_residual_a"] == float(Fraction(result["channel_current_a"]) + Fraction(result["diode_current_a"]) + 10)
    assert result["current_residual_a"] != 0
    assert result["root_evidence"]["voltage_residual_v"] != 0


@pytest.mark.parametrize("current,mode,status,support", [
    (-4, "strict-recorded", "unsupported", "none"),
    (-4, "ideal-diode-blocking", "supported", "estimate"),
    (0, "strict-recorded", "unsupported", "none"),
    (0, "ideal-diode-blocking", "supported", "estimate"),
    (-6, "ideal-diode-blocking", "supported", "recorded"),
    (-100, "ideal-diode-blocking", "unsupported", "none"),
])
def test_strict_domain_and_explicit_blocking(current, mode, status, support):
    """Distinguish recorded roots, unsupported points, and selected blocking estimates."""
    result = solve_reverse_current_sharing(**case(current, mode))["result"]
    assert (result["status"], result["support"]) == (status, support)
    if support == "estimate":
        assert result["channel_current_a"] == current
        assert result["diode_current_a"] == 0
        assert result["voltage_v"] == current / 10
        assert result["blocking_margin_v"] == pytest.approx(.6 + current / 10)
    if support == "none":
        assert result["channel_current_a"] is None
        assert result["diode_current_a"] is None
        assert result["voltage_v"] is None
        assert result["total_static_power_w"] is None


def test_recorded_zero_is_distinct_from_missing_zero():
    """Require recorded diode zero and a supported channel current."""
    inputs = case(-4, "ideal-diode-blocking")
    inputs["diode"]["current_a"] = [1, 20]
    assert solve_reverse_current_sharing(**inputs)["result"]["status"] == "unsupported"
    inputs = case(-4, "ideal-diode-blocking")
    inputs["channel"]["current_a"] = [5, 20]
    assert solve_reverse_current_sharing(**inputs)["result"]["status"] == "unsupported"


def test_domains_need_conserved_feasibility_not_equal_current_axes():
    """Accept nonoverlapping branch axes when their currents can conserve the total."""
    inputs = case(-10)
    inputs["channel"] = branch("channel-only", [6, 8], [.6, .8])
    inputs["diode"] = branch("diode-only", [2, 4], [.64, .68], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "supported"
    assert result["feasible_channel_current_magnitude_a"] == {"lower_a": 6, "upper_a": 8}
    assert result["channel_current_a"] == pytest.approx(-20 / 3)


def test_union_of_channel_and_transformed_diode_knots():
    """Evaluate both branch segment partitions in the conserved channel coordinate."""
    inputs = case(-10)
    inputs["channel"] = branch("channel-only", [0, 4, 7, 20], [0, .4, .7, 2])
    inputs["diode"] = branch("diode-only", [0, 2, 8, 20], [.6, .64, .76, 1], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["root_evidence"]["evaluated_knot_count"] == 6
    assert result["channel_current_a"] == pytest.approx(-20 / 3)


def test_single_feasible_endpoint_is_a_unique_root():
    """Accept a single domain endpoint without counting it twice."""
    inputs = case(-10)
    inputs["channel"] = branch("channel-only", [6, 8], [.6, .8])
    inputs["diode"] = branch("diode-only", [4, 8], [.6, .8], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "supported"
    assert result["channel_current_a"] == -6
    assert result["diode_current_a"] == -4
    assert result["root_evidence"]["isolated_root_count"] == 1


@pytest.mark.parametrize("mode", ["strict-recorded", "ideal-diode-blocking"])
def test_flat_common_voltage_is_ambiguous_even_at_blocking_endpoint(mode):
    """Reject a continuum even when a blocking endpoint also satisfies the model."""
    inputs = case(-10, mode)
    inputs["channel"]["voltage_v"] = [1, 1]
    inputs["diode"]["voltage_v"] = [1, 1]
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "ambiguous"
    assert result["root_evidence"]["flat_intervals_a"] == [[0, 10]]
    assert result["root_evidence"]["isolated_root_count"] == 0
    assert result["channel_current_a"] is None
    assert result["voltage_v"] is None


def test_flat_one_branch_can_still_give_a_unique_root():
    """Permit one flat branch when the other branch fixes a unique current."""
    inputs = case(-10)
    inputs["channel"]["voltage_v"] = [.7, .7]
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "supported"
    assert result["channel_current_a"] == -5


def test_arbitrarily_small_nonzero_slope_is_not_a_flat_interval():
    """Use exact classification rather than an arbitrary voltage tolerance."""
    inputs = case(-1)
    inputs["channel"] = branch("channel-only", [0, 1], [1, 1.000000000000001])
    inputs["diode"] = branch("diode-only", [0, 1], [1, 1.000000000000001], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "supported"
    assert result["channel_current_a"] == -.5


def test_root_at_shared_piecewise_knot_counts_once():
    """Count a segment boundary root once even when both branches share that knot."""
    inputs = case(-10)
    inputs["channel"] = branch("channel-only", [0, 5, 20], [0, .7, 2])
    inputs["diode"] = branch("diode-only", [0, 5, 20], [.6, .7, 1], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "supported"
    assert result["root_evidence"]["isolated_root_count"] == 1
    assert result["root_evidence"]["candidate_channel_current_magnitude_a"] == [5]


def test_recorded_zero_point_is_supported_when_both_branch_voltages_match():
    """Preserve a true zero operating point with two recorded zero endpoints."""
    inputs = case(0)
    inputs["diode"]["voltage_v"] = [0, 1]
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "supported"
    assert result["support"] == "recorded"
    assert result["channel_current_a"] == result["diode_current_a"] == 0
    assert result["voltage_v"] == result["total_static_power_w"] == 0


def test_maximum_length_flat_tables_form_one_continuum():
    """Classify the maximum supported table size without a quadratic root scan."""
    inputs = case(-9999)
    for key in ("channel", "diode"):
        inputs[key]["current_a"] = list(range(10000))
        inputs[key]["voltage_v"] = [1] * 10000
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "ambiguous"
    assert result["root_evidence"]["flat_intervals_a"] == [[0, 9999]]
    inputs["diode"]["current_a"].append(10000)
    inputs["diode"]["voltage_v"].append(1)
    with pytest.raises(ValueError):
        solve_reverse_current_sharing(**inputs)


@pytest.mark.parametrize("key,value", [
    ("selected", False), ("selected", 1), ("compatible_operating_state", False),
    ("current_a", 1), ("current_a", True), ("current_a", "-10"),
    ("current_a", float("nan")), ("junction_temperature_c", None),
    ("mode", "automatic"),
])
def test_reject_malformed_top_level_inputs(key, value):
    """Reject invalid policies, signs, types, and nonfinite quantities."""
    with pytest.raises(ValueError):
        solve_reverse_current_sharing(**(case() | {key: value}))


@pytest.mark.parametrize("key,value", [
    ("source", "  "), ("source", chr(0xD800)), ("isolation_evidence", ""),
    ("kind", "terminal-gate-on"), ("current_a", [0, 0]),
    ("current_a", [0, True]), ("current_a", [0, "20"]),
    ("current_a", [-1, 20]), ("current_a", [0, math.inf]),
    ("current_a", [20, 0]), ("current_a", [0]),
    ("voltage_v", [0, -1]), ("voltage_v", [1, 0]),
    ("voltage_v", [0, 1, 2]),
])
def test_reject_malformed_passive_branch(key, value):
    """Reject malformed branch source evidence and passive magnitude tables."""
    inputs = case()
    inputs["channel"][key] = value
    with pytest.raises(ValueError):
        solve_reverse_current_sharing(**inputs)


@pytest.mark.parametrize("change", [
    {"junction_temperature_c": None}, {"junction_temperature_c": 26},
    {"gate_bias_v": True}, {"gate_bias_v": "15"},
    {" ": 1}, {chr(0xD800): None},
])
def test_reject_unknown_mismatched_or_malformed_conditions(change):
    """Reject unknown temperatures and mismatched or malformed branch conditions."""
    inputs = case()
    inputs["channel"]["requested_conditions"].update(change)
    with pytest.raises(ValueError):
        solve_reverse_current_sharing(**inputs)


def test_common_temperature_mismatch_even_when_branch_conditions_match():
    """Require branch temperature to equal the common operating temperature."""
    inputs = case()
    inputs["junction_temperature_c"] = 26
    with pytest.raises(ValueError):
        solve_reverse_current_sharing(**inputs)


def test_explicit_unknown_gate_bias_is_preserved_without_inventing_zero():
    """Preserve a declared unknown gate condition but require its explicit key."""
    inputs = case()
    for key in ("recorded_conditions", "requested_conditions"):
        inputs["diode"][key]["gate_bias_v"] = None
    envelope = solve_reverse_current_sharing(**inputs)
    assert envelope["result"]["status"] == "supported"
    assert envelope["inputs"]["diode"]["recorded_conditions"]["gate_bias_v"] is None
    inputs["diode"]["recorded_conditions"].pop("gate_bias_v")
    inputs["diode"]["requested_conditions"].pop("gate_bias_v")
    with pytest.raises(ValueError):
        solve_reverse_current_sharing(**inputs)


def test_nonrepresentable_input_is_not_a_known_zero():
    """Reject input underflow before it becomes a fabricated zero."""
    with pytest.raises(ValueError):
        solve_reverse_current_sharing(**(case() | {"current_a": Fraction(-1, 10**400)}))


def test_overflowing_static_power_is_unsupported_with_null_outputs():
    """Report unsupported derived power instead of a nonfinite response."""
    inputs = case(-1e200)
    inputs["channel"] = branch("channel-only", [0, 1e200], [0, 1e200])
    inputs["diode"] = branch("diode-only", [0, 1e200], [0, 1e200], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "unsupported"
    assert "range" in result["diagnostic"]
    assert result["channel_current_a"] is None
    assert result["total_static_power_w"] is None


def test_underflowing_static_power_is_not_a_fabricated_zero():
    """Report unsupported power when a positive result cannot remain nonzero."""
    inputs = case(-1e-200)
    inputs["channel"] = branch("channel-only", [0, 1e-200], [0, 1e-200])
    inputs["diode"] = branch("diode-only", [0, 1e-200], [0, 1e-200], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "unsupported"
    assert result["channel_static_power_w"] is None


def test_nonrepresentable_derived_root_is_unsupported_instead_of_fabricated_zero():
    """Keep an unrepresentable root distinct from a recorded zero endpoint."""
    inputs = case(-1)
    inputs["channel"] = branch("channel-only", [0, 1], [0, 1e308])
    inputs["diode"] = branch("diode-only", [0, 1], [5e-324, 5e-324], -3)
    result = solve_reverse_current_sharing(**inputs)["result"]
    assert result["status"] == "unsupported"
    assert "range" in result["diagnostic"]
    assert result["channel_current_a"] is None
    assert result["root_evidence"]["isolated_root_count"] == 1
    assert result["root_evidence"]["candidate_channel_current_magnitude_a"] == []


def test_core_is_pure_and_canonical_hashes_replay(client):
    """Preserve inputs, reproduce both hashes, and replay the returned input object."""
    inputs = case()
    before = deepcopy(inputs)
    envelope = solve_reverse_current_sharing(**inputs)
    assert inputs == before
    assert envelope["inputs"]["channel"]["recorded_conditions"]["gate_bias_v"] == 15
    assert envelope["inputs"]["diode"]["recorded_conditions"]["gate_bias_v"] == -3
    assert solve_reverse_current_sharing(**envelope["inputs"]) == envelope
    response = client.post(URL, json=inputs)
    assert response.status_code == 200
    assert response.json() == envelope
    for key, hash_key in (("inputs", "input_sha256"), ("result", "result_sha256")):
        wire = json.dumps(envelope[key], sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert hashlib.sha256(wire.encode()).hexdigest() == envelope["evidence"][hash_key]
    changed = deepcopy(inputs)
    changed["channel"]["isolation_evidence"] += " revised"
    assert solve_reverse_current_sharing(**changed)["evidence"]["input_sha256"] != envelope["evidence"]["input_sha256"]


@pytest.mark.parametrize("current,mode,status", [(-4, "strict-recorded", "unsupported"),
                                                (-4, "ideal-diode-blocking", "supported")])
def test_http_valid_unsupported_is_200(client, current, mode, status):
    """Distinguish valid unsupported operating points from malformed requests."""
    response = client.post(URL, json=case(current, mode))
    assert response.status_code == 200
    assert response.json()["result"]["status"] == status


@pytest.mark.parametrize("literal", ["true", '"-10"', "NaN", "Infinity", "1e400", "-1e-400"])
def test_http_rejects_invalid_real_literals(client, literal):
    """Reject coercible values and numeric literals outside the supported range."""
    inputs = case()
    wire = json.dumps(inputs).replace('"current_a": -10', '"current_a": ' + literal, 1)
    assert client.post(URL, content=wire, headers={"Content-Type": "application/json"}).status_code == 422


def test_http_default_mode_extra_fields_and_unicode(client):
    """Check the strict default and reject extra fields or invalid Unicode."""
    inputs = case()
    inputs.pop("mode")
    response = client.post(URL, json=inputs)
    assert response.status_code == 200
    assert response.json()["inputs"]["mode"] == "strict-recorded"
    assert client.post(URL, json=inputs | {"surprise": True}).status_code == 422
    inputs["channel"]["source"] = chr(0xD800)
    assert client.post(URL, content=json.dumps(inputs), headers={"Content-Type": "application/json"}).status_code == 422


def test_http_stream_limit_does_not_trust_content_length(client):
    """Count actual bytes even when the declared request size is smaller."""
    response = client.post(URL, content=(b" " * 262144 for _ in range(17)),
                           headers={"Content-Length": "1"})
    assert response.status_code == 413
