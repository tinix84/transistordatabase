"""Independent physical algebra and mapping review for converter aggregation."""
from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transistordatabase.converter_losses import analyze_converter, converter_presets
from transistordatabase.gui_web.backend.converter_routes import ConverterRequest, router


def drop(offset=2, resistance=0):
    """Create a declared threshold/resistance estimate."""
    return {"mode": "linear-drop-estimate", "selected": True,
            "offset_v": offset, "resistance_ohm": resistance,
            "source": "Independent physical algebra", "conditions": {"t_j_c": None}}


def energy(value=.001, reference=400, target=200):
    """Create independently conditioned constant current-energy records."""
    return {"mode": "voltage-scaling-estimate", "selected": True,
            "current_a": [0, 100], "energy_j": [value, value],
            "recorded_conditions": {"v_supply_v": reference, "t_j_c": None},
            "requested_conditions": {"v_supply_v": target, "t_j_c": None},
            "source": f"Independent {reference} V energy"}


def model(offset=2, resistance=0):
    """Bind both signed directions without implicit abs borrowing."""
    return {"conduction": {"positive": drop(offset, resistance),
                           "negative": drop(offset, resistance)},
            "switching": {"turn_on": {"positive": energy(), "negative": energy()},
                          "turn_off": {"positive": energy(.002, 600),
                                       "negative": energy(.002, 600)}},
            "events": {"mode": "gate", "threshold_v": .5}}


def custom(operation="held-envelope-estimate", time=(0, 1), current=10, duty=.25):
    """Build one physical position with explicit independent signal bindings."""
    binding = {"role": "position", "members": ["Q1"], "multiplicity": 1,
               "conduction_current": {"signal": "i"},
               "switching_current": {"signal": "ic"},
               "bus_voltage": {"signal": "bus"}, "gate_signal": "g",
               "duty": {"signal": "d"}, "frequency_hz": {"signal": "f"}}
    return {"preset": "custom", "operation": operation, "time_s": list(time),
            "signals": {"i": deepcopy(current), "ic": deepcopy(current), "d": duty, "f": 1e5,
                        "bus": 200, "g": 0}, "bindings": [binding],
            "role_models": {"position": model()},
            "assumptions": {"envelope_selected": True, "symmetry_selected": True},
            "provenance": {"source": "Independent review"}}


def preset(identifier, time=(0, 1), signals=None):
    """Bind fixture models to the public immutable preset roles."""
    definition = next(p for p in converter_presets()["presets"] if p["id"] == identifier)
    return {"preset": identifier, "operation": definition["operation"],
            "time_s": list(time), "signals": signals or {},
            "role_models": {b["role"]: model() for b in definition["bindings"]
                            if not b.get("source_switching_omission")},
            "assumptions": {"symmetry_selected": True, "envelope_selected": True,
                            "source_omission_reason": "Source omits recovery; unknown loss",
                            "operating_power_w": 1}}


def powers(result):
    """Read physical total powers without using any scientific helper."""
    return [result["totals"]["components"][name]["power_w"]
            for name in ("conduction", "turn_on", "turn_off")]


@pytest.mark.parametrize("duty,frequency,expected", [
    (.25, 1e5, [5, 50, 200 / 3]), (0, 1e5, [0, 0, 0]),
    (1, 1e5, [20, 0, 0]), (.25, 0, [5, 0, 0]),
])
def test_physical_envelope_boundaries(duty, frequency, expected):
    """Check duty/frequency boundaries against direct physical algebra."""
    data = custom(duty=duty)
    data["signals"]["f"] = frequency
    if duty in (0, 1) or frequency == 0:
        data["role_models"]["position"].pop("switching")
    result = analyze_converter(data)
    assert powers(result) == pytest.approx(expected)
    assert result["totals"]["total_power_w"] == pytest.approx(sum(expected))
    evidence = result["branches"][0]["evidence"]
    assert "events" not in evidence
    assert "no resolved PWM" in evidence["switching_interpretation"]


def test_independent_random_held_integrals_and_split_invariance():
    """Compare random signed power histories with independent held integrals."""
    rng = np.random.default_rng(731)
    for _ in range(35):
        n = 7
        dt = rng.uniform(.01, 2, n)
        current = rng.uniform(-20, 20, n)
        duty = rng.uniform(0, 1, n)
        frequency = rng.uniform(0, 1e5, n)
        data = custom(time=np.r_[0, np.cumsum(dt)], current=current.tolist(), duty=duty.tolist())
        data["signals"].update(ic=(-current).tolist(), f=frequency.tolist())
        data["role_models"]["position"] = model(.7, .13)
        result = analyze_converter(data)
        duration = sum(dt)
        expected = [sum(dt * duty * np.abs(current) * (.7 + .13 * np.abs(current))) / duration,
                    sum(dt * frequency * .0005) / duration,
                    sum(dt * frequency * (.002 / 3)) / duration]
        assert powers(result) == pytest.approx(expected, rel=2e-13)
        split = deepcopy(data)
        split["time_s"] = np.r_[0, np.cumsum(np.repeat(dt / 2, 2))].tolist()
        for name in ("i", "ic", "d", "f"):
            split["signals"][name] = np.repeat(data["signals"][name], 2).tolist()
        assert powers(analyze_converter(split)) == pytest.approx(expected, rel=2e-13)


def test_irregular_weighting_not_sample_mean():
    """Use elapsed time rather than an unweighted interval count."""
    data = custom(time=[0, 1, 4], current=[5, 15], duty=1)
    assert powers(analyze_converter(data)) == pytest.approx([25, 0, 0])


def test_negative_zero_duty_needs_no_curves_and_omission_stays_unknown():
    """Distinguish provably inactive duty from an explicit unknown component."""
    data = custom(current=-10, duty=0)
    data["role_models"]["position"] = {}
    assert powers(analyze_converter(data)) == [0, 0, 0]
    data["role_models"]["position"]["conduction"] = {
        "omit": True, "omission_reason": "No conduction characterization"}
    result = analyze_converter(data)
    assert powers(result) == [None, 0, 0]
    assert result["totals"]["total_power_w"] is None


def test_explicit_switching_omission_requires_no_unused_signals():
    """Do not require gate/bus/commutation records for omitted mechanisms."""
    data = custom("waveform", current=[10, 10])
    data["role_models"]["position"]["switching"] = {
        name: {"omit": True, "omission_reason": "No switching characterization"}
        for name in ("turn_on", "turn_off")}
    for name in ("g", "bus", "ic"):
        data["signals"].pop(name)
    result = analyze_converter(data)
    assert powers(result) == [5, None, None]
    assert result["totals"]["total_power_w"] is None
    evidence = result["branches"][0]["evidence"]
    assert evidence["switching_signals"] == {}
    assert evidence["switching_signals_status"] == "not-evaluated-explicit-omissions"
    assert evidence["switching_cumulative_plot"] == []


def test_custom_twelve_individually_bound_positions_are_not_sc_dab():
    """Support explicit distinct circuits without changing the SC preset count."""
    data = custom(duty=1)
    data["bindings"] = []
    data["role_models"] = {}
    data["assumptions"]["symmetry_selected"] = False
    for i in range(1, 13):
        role = f"device_{i}"
        data["signals"][role] = i
        data["bindings"].append({"role": role, "members": [role], "multiplicity": 1,
                                 "conduction_current": {"signal": role},
                                 "duty": {"value": 1}, "frequency_hz": {"value": 0}})
        data["role_models"][role] = {"conduction": {"positive": drop()}}
    result = analyze_converter(data)
    assert result["physical_member_count"] == 12
    assert powers(result) == [156, 0, 0]
    assert result["preset"] == "custom"


def test_masks_need_no_gate_and_no_event_tail_changes_full_interval_power():
    """Preserve explicit mask evidence and normalize using full observation time."""
    data = custom("waveform", [0, 1, 2], current=[10, 10, 0], duty=1)
    data["signals"].pop("g")
    data["role_models"]["position"]["events"] = {
        "mode": "masks", "turn_on": [True, False], "turn_off": [False, True]}
    result = analyze_converter(data)
    assert len(result["branches"][0]["evidence"]["events"]) == 2
    data["time_s"].append(10)
    data["signals"]["i"].append(0)
    data["signals"]["ic"].append(0)
    for side in ("turn_on", "turn_off"):
        data["role_models"]["position"]["events"][side].append(False)
    extended = analyze_converter(data)
    for side in ("turn_on", "turn_off"):
        first = result["totals"]["components"][side]
        second = extended["totals"]["components"][side]
        assert second["energy_j"] == first["energy_j"]
        assert second["power_w"] == pytest.approx(first["power_w"] / 5)


def test_dab_exact_currents_members_and_preedge_energy():
    """Keep all role signs, distinct conduction signals and pre-edge currents."""
    currents = [0, 2, 9, 3, 0]
    signals = {"itp": currents, "its": currents, "vp": 200, "vs": 200}
    for number in (2, 4, 6, 8):
        signals[f"is{number}"] = number / 2
        signals[f"vgs{number}"] = [0, 0, 1, 1, 0]
    data = preset("sc-dab", [0, 1e-6, 2e-6, 3e-6, 4e-6], signals)
    for spec in data["role_models"].values():
        for name, multiplier in (("turn_on", .001), ("turn_off", .002)):
            for direction, extra in (("positive", 1), ("negative", 2)):
                spec["switching"][name][direction]["energy_j"] = [0, 100 * multiplier * extra]
    result = analyze_converter(data)
    assert result["physical_member_count"] == 8
    assert powers(result) == pytest.approx([40, 3000, 6000])
    assert [b["binding"]["switching_current"]["scale"] for b in result["branches"]] == [-1, 1, 1, -1]
    for branch, sign, cond in zip(result["branches"], [-1, 1, 1, -1], [1, 2, 3, 4], strict=True):
        assert branch["evidence"]["conduction_signals"]["current_a"] == [cond] * 5
        events = branch["evidence"]["events"]
        assert [e["current_a"] for e in events] == [2 * sign, 3 * sign]
        assert [e["sample_index"] for e in events] == [1, 3]
        factor = 1 if sign > 0 else 2
        assert [e["energy_j"] for e in events] == pytest.approx([.001 * factor, .002 * factor])


@pytest.mark.parametrize("name,count", [("sc-src-zcs", 2), ("sc-src-zvs", 4), ("sc-llc", 4)])
def test_src_actual_omissions_remain_partial(name, count):
    """Keep omitted secondary switching unknown despite a topology label."""
    signals = {"itp": [0, 2, 9, 3, 0], "vp": 200,
               "vgs1": [0, 0, 1, 1, 0], "vgs2": [0, 0, 1, 1, 0], "vgs4": [0, 0, 1, 1, 0]}
    data = preset(name, [0, 1e-6, 2e-6, 3e-6, 4e-6], signals)
    definition = next(p for p in converter_presets()["presets"] if p["id"] == name)
    for i, binding in enumerate(definition["bindings"], 1):
        data["signals"][binding["conduction_current"]["signal"]] = i / 2
        data["role_models"].setdefault(binding["role"], {"conduction": {"positive": drop()}})
    result = analyze_converter(data)
    assert result["physical_member_count"] == 8
    assert powers(result) == [20, None, None]
    assert result["totals"]["total_power_w"] is None
    assert sum(b["multiplicity"] for b in result["branches"] if b["components"]["turn_on"]["status"] == "computed") == count
    assert result["totals"]["known_component_power_w"] > 20


@pytest.mark.parametrize("name,expected", [("vr", 12), ("vrn", 30), ("ivr", 24), ("ivrn", 42), ("tvr", 0)])
def test_rectifier_members_count_with_unneeded_models_absent(name, expected):
    """Count actual rectifier positions without requiring absent positions."""
    data = preset(f"sc-{name}", signals={"ipfc": 0, "dpfc": 0, "fs": 0, "vdcp": 200, "idrl": 1, "idrn": 1.5})
    for binding in next(p for p in converter_presets()["presets"] if p["id"] == f"sc-{name}")["bindings"]:
        if binding["multiplicity"] == 0:
            data["role_models"].pop(binding["role"])
    assert powers(analyze_converter(data)) == pytest.approx([expected, 0, 0])


def test_neutral_three_times_current_is_not_device_multiplicity():
    """Square the actual neutral current before applying physical count."""
    data = preset("sc-six-switch", signals={"ia": 10, "dna": 0, "dpa": 0, "dpn": 1,
                                           "fs": 1e5, "fl": 50, "vdc": 200})
    data["role_models"]["neutral_igbts"] = model(0, .1)
    result = analyze_converter(data)
    neutral = result["branches"][2]
    assert neutral["components"]["conduction"]["power_w"] == pytest.approx(90)
    assert neutral["scaled_components"]["conduction"]["power_w"] == pytest.approx(180)
    data["signals"]["dpn"] = .5
    result = analyze_converter(data)
    assert result["branches"][2]["scaled_components"]["turn_on"]["power_w"] == pytest.approx(.05)
    assert result["branches"][2]["scaled_components"]["turn_off"]["power_w"] == pytest.approx(1 / 15)


def test_bridgeless_direction_activates_reverse_without_changing_sign():
    """Activate reverse devices explicitly while preserving their current sign."""
    data = preset("sc-bridgeless", signals={"iac": -10, "dn1": 0, "dp1": .25, "fs": 0, "vdc": 200})
    data["role_models"].pop("reverse_igbt")
    assert analyze_converter(data)["branches"][2]["total_power_w"] == 0
    data["assumptions"]["operating_power_w"] = -1
    with pytest.raises(ValueError, match="Missing required"):
        analyze_converter(data)
    data["role_models"]["reverse_igbt"] = {"conduction": {"positive": drop()}}
    reverse = analyze_converter(data)["branches"][2]
    assert reverse["total_power_w"] == pytest.approx(10)
    assert reverse["evidence"]["intervals"][0]["current_a"] == 10


def test_inverter_all_six_independent_positions_and_unbalanced_input():
    """Integrate six supplied positions without assuming phase balance."""
    n = 180
    theta = 2 * np.pi * (np.arange(n) + .5) / n
    data = preset("inverter-six", np.linspace(0, 1 / 50, n + 1), {"vdc": 400, "fs": 0})
    for phase, angle in zip("abc", [0, -2 * np.pi / 3, -4 * np.pi / 3], strict=True):
        data["signals"][f"i{phase}"] = (np.sqrt(150) * np.cos(theta + angle)).tolist()
        data["signals"][f"dp{phase}"] = (.5 + .3 * np.cos(theta + angle)).tolist()
        data["signals"][f"dn{phase}"] = (1 - np.array(data["signals"][f"dp{phase}"])).tolist()
        for position in ("upper", "lower"):
            data["role_models"][f"{phase}_{position}"] = {"conduction": {"positive": drop(0, .1), "negative": drop(0, .1)}}
    data["assumptions"]["symmetry_selected"] = False
    assert powers(analyze_converter(data)) == pytest.approx([22.5, 0, 0], rel=2e-13)
    data["signals"]["ib"] = (np.array(data["signals"]["ib"]) * 2).tolist()
    assert powers(analyze_converter(data)) == pytest.approx([45, 0, 0], rel=2e-13)


def test_zero_current_events_source_snapshot_and_no_mutation():
    """Retain finite E(0) and independent input/source snapshots."""
    data = custom("waveform", [0, 1, 2], current=[0, 0, 0], duty=1)
    data["signals"].update(g=[0, 1, 0])
    original = deepcopy(data)
    result = analyze_converter(data)
    assert data == original
    assert powers(result) == pytest.approx([0, .00025, .001 / 3])
    assert len(result["branches"][0]["evidence"]["events"]) == 2
    data["role_models"]["position"]["switching"]["turn_on"]["positive"]["source"] = "changed"
    assert result["request"] == original


@pytest.mark.parametrize("mutate", [
    lambda d: d["signals"].update(d=-.1), lambda d: d["signals"].update(d=1.1),
    lambda d: d["signals"].update(f=-1), lambda d: d["signals"].update(i=[1, 2]),
    lambda d: d["signals"].update(i=float("nan")),
    lambda d: d["assumptions"].update(envelope_selected=False),
    lambda d: d["bindings"][0].update(multiplicity=True),
    lambda d: d["role_models"]["position"]["switching"]["turn_on"]["positive"]["requested_conditions"].update(t_j_c=25),
])
def test_invalid_or_unknown_data_rejects_instead_of_fallback(mutate):
    """Reject invalid inputs and incompatible conditions explicitly."""
    data = custom()
    mutate(data)
    with pytest.raises((ValueError, KeyError)):
        analyze_converter(data)


def client():
    """Keep independent transport checks isolated from unrelated service startup."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_typed_canonical_api_equals_science_with_unknown_conditions_intact():
    """Transport the exact scientific result including unknown conditions."""
    data = custom()
    parsed = ConverterRequest(**data).model_dump(exclude_none=True)
    response = client().post("/api/converters/analyze", json=data)
    assert response.status_code == 200, response.text
    assert response.json() == analyze_converter(parsed)
    row = response.json()["branches"][0]["evidence"]["intervals"][0]
    assert row["energy_evidence"]["turn_on"]["recorded_conditions"]["t_j_c"] is None
    assert row["energy_evidence"]["turn_off"]["recorded_conditions"]["v_supply_v"] == 600
    assert response.json()["request"]["role_models"]["position"]["conduction"]["positive"]["conditions"] == {"t_j_c": None}
    assert client().get("/api/converters/presets").json() == converter_presets()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_raw_nonfinite_json_is_safe_typed_422(value):
    """Reject raw nonfinite JSON without failing error serialization."""
    import json

    raw = json.dumps(custom()).replace('"i": 10', f'"i": {value}')
    response = client().post("/api/converters/analyze", content=raw,
                             headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_typed_boolean_count_and_resource_budget_rejection():
    """Reject boolean counts and excessive exact expansion."""
    data = custom()
    data["bindings"][0]["multiplicity"] = True
    assert client().post("/api/converters/analyze", json=data).status_code == 422
    data = custom(time=np.linspace(0, 1, 10001).tolist())
    second = deepcopy(data["bindings"][0])
    second.update(role="second", members=["Q2"])
    data["bindings"].append(second)
    data["role_models"]["second"] = model()
    response = client().post("/api/converters/analyze", json=data)
    assert response.status_code == 413, response.text


def test_recorded_voltage_mismatch_and_extrapolation_are_not_scaled_or_clamped():
    """Require recorded coverage instead of silently scaling or clamping."""
    data = custom()
    on = data["role_models"]["position"]["switching"]["turn_on"]["positive"]
    on["mode"] = "recorded"
    on.pop("selected")
    on["requested_conditions"]["v_supply_v"] = 400
    response = client().post("/api/converters/analyze", json=data)
    assert response.status_code == 422
    data = custom(current=101)
    assert client().post("/api/converters/analyze", json=data).status_code == 422
