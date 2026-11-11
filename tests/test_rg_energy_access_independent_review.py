"""Check branch provenance and replay against independent analytical curves."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from transistordatabase.core.models import (
    ElectricalRatings, MeasurementEvidence, SwitchingLossData, ThermalProperties,
    Transistor, TransistorMetadata,
)
from transistordatabase.gui_web.backend.rg_energy_routes import create_rg_energy_router
from transistordatabase.rg_energy_access import calculate_rg_energy, rg_energy_inventory


def specimen():
    """Use unequal current domains, unsorted samples and sparse source indices."""
    device = Transistor(
        TransistorMetadata("Bench µ", "SiC-MOSFET", "review", "A", "B"),
        ElectricalRatings(1200, 80, 70, 175), ThermalProperties(.1, .3),
    )
    sweep = SwitchingLossData(
        "graph_r_e", 45., 300., 0., i_x=8.,
        graph_r_e=np.array([[7., 1., 4.], [.015, .033, .024]]),
        commutation_device="other µ",
        evidence=MeasurementEvidence(method="unknown", filename="é.csv",
                                     original_base64="PRIVATE"),
    )
    device.switch.e_off_data = [
        replace(sweep, dataset_type="single", e_x=.01),
        replace(sweep, t_j=105., graph_r_e=sweep.graph_r_e * [[1.], [2.]]),
        replace(sweep, v_supply=300.0000000005),
    ]
    return device


def request():
    """Query the measured 45 °C slice with the neighboring recorded supply voltage."""
    return {
        "operation": "energy-at-rg", "branch": "e_off", "representation": "graph_r_e",
        "i_channel": 8., "t_j": 45., "r_g": 3.,
        "conditions": {"v_supply": 300., "v_g": 0., "v_g_off": None,
                       "load_inductance": None, "commutation_inductance": None,
                       "commutation_device": "other µ", "convention": None,
                       "event": None, "method": "unknown"},
    }


def test_repository_identity_and_tolerant_context_survive_transport():
    """Keep the storage key, canonical device name and actual curve conditions distinct."""
    device = specimen()
    lookups = []

    def lookup(storage_id):
        lookups.append(storage_id)
        return device

    app = FastAPI()
    app.include_router(create_rg_energy_router(lookup))
    with TestClient(app) as client:
        route = "/api/transistors/storage-alias/rg-energy"
        inventory = client.get(route).json()
        response = client.post(route, json=request())
    assert response.status_code == 200, response.text
    proof = json.loads(response.content)
    assert lookups == ["storage-alias"] * 2
    assert proof["device_id"] == inventory["device_id"] == "Bench µ"
    assert proof["query"]["conditions"]["v_supply"] == 300.
    assert proof["value"] == pytest.approx(.027)
    assert [r["source_id"] for r in proof["sources"]] == [
        "Bench µ:switch.e_off_data[1]", "Bench µ:switch.e_off_data[2]",
    ]
    row = proof["sources"][1]
    assert row["conditions"]["v_supply"] == 300.0000000005
    assert row["source_filename"] == "é.csv"
    assert "PRIVATE" not in response.text
    normalized = json.dumps(row["snapshot"], ensure_ascii=True, sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(normalized).hexdigest() == row["snapshot_sha256"]
    assert row["snapshot"]["axis"] == [1., 4., 7.]


def test_source_mutation_changes_digest_and_keeps_prior_result_replayable():
    """Change one measured sample and check both the old and new result proofs."""
    device = specimen()
    old = calculate_rg_energy(device, request())
    saved = deepcopy(old)
    device.switch.e_off_data[2].graph_r_e[1, 1] += .009
    new = calculate_rg_energy(device, request())
    assert old == saved
    assert new["value"] == pytest.approx(.030)
    assert old["sources"][1]["snapshot_sha256"] != new["sources"][1]["snapshot_sha256"]
    assert old["sources"][0]["snapshot_sha256"] == new["sources"][0]["snapshot_sha256"]


@pytest.mark.parametrize("bad_graph", [None, [[0., 1.], [2.]]])
def test_invalid_graph_in_another_temperature_slice_does_not_hide_a_valid_result(bad_graph):
    """Retain a diagnostic for the unusable row and the proof for the valid row."""
    device = specimen()
    device.switch.e_off_data[1].graph_r_e = bad_graph
    inventory = rg_energy_inventory(device)
    result = calculate_rg_energy(device, request())
    assert result["value"] == pytest.approx(.027)
    for rows in (inventory["curves"], result["sources"]):
        unavailable, available = rows
        assert unavailable["source_id"] == "Bench µ:switch.e_off_data[1]"
        assert unavailable["status"] == "unavailable"
        assert unavailable["diagnostic"]
        assert unavailable["preview"] is None
        assert available["source_id"] == "Bench µ:switch.e_off_data[2]"
        assert available["status"] == "available"
        assert available["snapshot_sha256"]


def test_unused_invalid_resistance_and_malformed_source_hash_preserve_math_proof():
    """A fixed current defines the resistance sweep even if ancillary fields are invalid."""
    device = specimen()
    active = device.switch.e_off_data[2]
    active.r_g = -1
    active.evidence.sha256 = "incomplete"
    result = calculate_rg_energy(device, request())
    row = result["sources"][1]
    assert result["value"] == pytest.approx(.027)
    assert row["status"] == "available"
    assert row["snapshot"]["axis"] == [1., 4., 7.]
    assert row["snapshot"]["i_x"] == 8.
    assert row["source_sha256"] is None
    assert {"r_g", "source_sha256"} <= set(row["unavailable_fields"])


def test_inverse_hole_and_duplicate_do_not_change_branch_coordinates():
    """Keep omitted current domains and duplicate resistance sweeps visible to the helper."""
    device = specimen()
    base = device.switch.e_off_data[2]

    def sweep(resistance, currents, energies):
        return replace(base, dataset_type="graph_i_e", i_x=None, r_g=resistance,
                       graph_i_e=np.array([currents, energies]))

    wide = sweep(1., [4., 12.], [.026, .034])
    narrow = sweep(5., [10., 12.], [.018, .020])
    distant = sweep(9., [4., 12.], [.010, .018])
    device.switch.e_off_data.extend([distant, narrow, wide])
    payload = request()
    payload.pop("r_g")
    payload.update(operation="rg-at-energy", representation="graph_i_e", e_target=.022)
    with pytest.raises(ValueError, match="supports"):
        calculate_rg_energy(device, payload)
    assert [row["index"] for row in rg_energy_inventory(device)["curves"]] == list(range(1, 6))
    device.switch.e_off_data[4] = replace(narrow, graph_i_e=np.array([[4., 12.], [.018, .026]]))
    assert calculate_rg_energy(device, payload)["value"] == pytest.approx(5.)
    device.switch.e_off_data.append(deepcopy(wide))
    with pytest.raises(ValueError):
        calculate_rg_energy(device, payload)


def test_invalid_selector_bytes_in_an_otherwise_complete_query_are_rejected():
    """Reject an invalid commutation device label before a lookup can succeed."""
    app = FastAPI()
    app.include_router(create_rg_energy_router(lambda _: specimen()))
    wire = json.dumps(request(), ensure_ascii=False).encode("utf-8")
    with TestClient(app) as client:
        response = client.post(
            "/api/transistors/part/rg-energy", content=wire.replace("other µ".encode(), b"other\xff"),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
