"""Exact core subset export corners and explicit unsupported conditions."""
from copy import deepcopy
from pathlib import Path

import pytest

from transistordatabase.plecs_core_export import core_selection_document, render_core_selection
from transistordatabase.plecs_document import finalize_plecs_package, parse_plecs_document


def fixture():
    """Use independently declared ratings and source unknown gates explicitly."""
    document = parse_plecs_document((Path(__file__).parent / "test_data/sample_plecs_switch.xml").read_bytes())
    transistor = finalize_plecs_package(document, package_id="0",
        ratings=dict(v_abs_max=1200., i_abs_max=75., i_cont=40., t_j_max=175.),
        thermal_properties=dict(housing_area=.0001, cooling_area=.0002),
        block_targets={"0":"switch"},
        gate_voltage_by_table={t.source_path:15. for t in document.packages[0].blocks[0].tables})
    transistor.metadata.comment = "Ordinary core record with no PLECS source dependency"
    selection = dict(component="switch", conduction=dict(gate_voltage_v=15.),
        turn_on=dict(gate_voltage_v=15.,gate_resistance_ohm=None),
        turn_off=dict(gate_voltage_v=15.,gate_resistance_ohm=None),
        include_thermal=True, omission_reasons={}, recovery_voltage_convention="preserve-signed")
    return transistor, selection


def test_generic_core_no_source_dependency_roundtrip_all_corners():
    """Recorded corners, fixed context and Foster values survive generic rendering."""
    transistor, selection = fixture()
    before = deepcopy(transistor.switch.e_on_data[0].graph_i_e)
    document, report = core_selection_document(transistor, selection)
    assert report["simulator_validation"] == "not-performed"
    assert document.packages[0].partnumber == "TEST_MOSFET_001"
    assert document.packages[0].fixed_conditions["selection"] == selection
    assert document.packages[0].fixed_conditions["electrical_ratings"]["v_abs_max"] == 1200
    assert document.packages[0].blocks[0].tables[1].slices[-1].plot_data[-1] == [50.,.0027]
    after = parse_plecs_document(render_core_selection(transistor, selection))
    assert after.packages == document.packages
    assert (before == transistor.switch.e_on_data[0].graph_i_e).all()


@pytest.mark.parametrize("defect,message", [
    ("missing-corner", "incomplete"), ("duplicate", "duplicate"),
    ("gate", "no actual"), ("rg", "no actual"),
    ("axis", "axes differ"), ("mixed-off", "mixed gate-off"),
    ("missing-thermal", "no Foster"), ("omit", "reason"),
])
def test_no_first_nearest_or_automatic_interpolation(defect, message):
    """Unsupported exact selections fail without silently choosing an alternative."""
    transistor, selection = fixture()
    if defect == "missing-corner":
        transistor.switch.e_on_data.pop()
    elif defect == "duplicate":
        transistor.switch.e_on_data.append(deepcopy(transistor.switch.e_on_data[0]))
    elif defect == "gate":
        selection["turn_on"]["gate_voltage_v"] = 15.001
    elif defect == "rg":
        selection["turn_on"]["gate_resistance_ohm"] = 0.
    elif defect == "axis":
        transistor.switch.e_on_data[0].graph_i_e[0][1] = 9.9
    elif defect == "mixed-off":
        transistor.switch.e_on_data[0].v_g_off = -3.
    elif defect == "missing-thermal":
        transistor.switch.thermal_foster = None
    else:
        selection["turn_on"] = None
    with pytest.raises(ValueError, match=message):
        core_selection_document(transistor, selection)


def test_deliberate_omission_is_not_exported_zero():
    """A requested subset retains unknown omissions instead of invented tables."""
    transistor, selection = fixture()
    selection["turn_on"] = None
    selection["omission_reasons"] = {"turn_on":"No applicable turn-on source for this scope"}
    document, report = core_selection_document(transistor, selection)
    assert [t.kind for t in document.packages[0].blocks[0].tables] == ["conduction","turn-off"]
    assert report["omitted_mechanisms"] == ["turn_on"]


def test_typed_http_selection_uses_injected_existing_repository(tmp_path):
    """Typed selection bytes and preview retain exact data without filesystem staging."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from transistordatabase.gui_web.backend.plecs_routes import plecs_repository, router
    from transistordatabase.backend.concrete_services import ExportService
    transistor, selection = fixture()

    class Repository:
        """A single existing repository fixture, not another path or loader."""

        def get_by_name(self, name):
            """Return only the fixture's actual identity."""
            return transistor if name == transistor.metadata.name else None

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[plecs_repository] = lambda: Repository()
    client = TestClient(app)
    route = "/api/plecs/core/" + transistor.metadata.name
    preview = client.post(route + "/preview", json=selection)
    assert preview.status_code == 200, preview.text
    assert preview.json()["report"]["evidence"][1]["curve_count"] == 4
    exported = client.post(route + "/export", json=selection)
    assert exported.status_code == 200, exported.text
    assert parse_plecs_document(exported.content).packages[0].fixed_conditions["selection"] == selection
    assert client.post("/api/plecs/core/missing/export", json=selection).status_code == 404
    unsupported = deepcopy(selection)
    unsupported["turn_on"]["gate_voltage_v"] = 14.999
    assert client.post(route + "/export", json=unsupported).status_code == 422
    ExportService().export_to_plecs(transistor, tmp_path, {"selection":selection})
    assert len(list(tmp_path.glob("*.xml"))) == 1
