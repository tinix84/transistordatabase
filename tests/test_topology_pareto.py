"""Fixed-boundary semiconductor Pareto screening regressions."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from transistordatabase.core.models import (
    ChannelCharacteristics,
    ElectricalRatings,
    ThermalProperties,
    Transistor,
    TransistorMetadata,
)
from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.topologies.semiconductor_pareto import (
    calculate_boost_semiconductor_pareto,
)


BOUNDARY = {
    "v_in": 100.0,
    "v_out": 200.0,
    "p_out": 1000.0,
    "f_sw": 20_000.0,
    "l_value": 1e-3,
    "c_out": 1e-3,
    "t_j": 25.0,
    "t_case": 60.0,
}

CATALOG = {
    "CREE_C3M0060065J": {
        "orderable_part": "C3M0060065J",
        "unit_cost_usd": 9.64,
        "length_mm": 9.125,
        "width_mm": 10.23,
        "height_mm": 4.57,
        "price_source_url": "https://www.digikey.com/en/products/detail/wolfspeed-inc/C3M0060065J/12525328",
        "cost_accessed_on": date(2026, 10, 14),
        "dimensions_source_url": "https://assets.wolfspeed.com/uploads/2024/01/Wolfspeed_C3M0060065J_data_sheet.pdf",
    },
    "Infineon_IPBE65R050CFD7A": {
        "orderable_part": "IPBE65R050CFD7AATMA1",
        "unit_cost_usd": 10.78,
        "length_mm": 9.45,
        "width_mm": 10.20,
        "height_mm": 4.50,
        "price_source_url": "https://www.digikey.com/en/products/detail/infineon-technologies/IPBE65R050CFD7AATMA1/13212427",
        "cost_accessed_on": date(2026, 10, 14),
        "dimensions_source_url": "https://www.infineon.com/assets/row/public/documents/24/49/infineon-ipbe65r050cfd7a-datasheet-en.pdf",
    },
}


def notebook_candidates() -> list[dict[str, Any]]:
    """Return the notebook's three physical device configurations.

    :returns: Candidate request mappings in notebook order.
    :rtype: list[dict[str, Any]]
    """
    return [
        {
            "candidate_id": "cree-one",
            "label": "CREE ×1 + CREE rectifier ×1",
            "switch_transistor_id": "CREE_C3M0060065J",
            "diode_transistor_id": "CREE_C3M0060065J",
            "parallel_switches": 1,
            "parallel_diodes": 1,
            "v_g": 15.0,
            "diode_v_g": 0.0,
        },
        {
            "candidate_id": "infineon-one",
            "label": "Infineon ×1 + CREE rectifier ×1",
            "switch_transistor_id": "Infineon_IPBE65R050CFD7A",
            "diode_transistor_id": "CREE_C3M0060065J",
            "parallel_switches": 1,
            "parallel_diodes": 1,
            "v_g": 10.0,
            "diode_v_g": 0.0,
        },
        {
            "candidate_id": "infineon-two",
            "label": "Infineon ×2 + CREE rectifier ×1",
            "switch_transistor_id": "Infineon_IPBE65R050CFD7A",
            "diode_transistor_id": "CREE_C3M0060065J",
            "parallel_switches": 2,
            "parallel_diodes": 1,
            "v_g": 10.0,
            "diode_v_g": 0.0,
        },
    ]


def load_notebook_devices() -> dict[str, Transistor]:
    """Load the same example TDB records used by the notebook.

    :returns: CREE and Infineon devices keyed by their TDB names.
    :rtype: dict[str, Transistor]
    """
    examples = (
        Path(__file__).resolve().parents[1]
        / "transistordatabase"
        / "examples"
        / "tdb_example"
    )
    loader = JsonTransistorLoader()
    return {
        name: loader.load_from_json(examples / f"{name}.json")
        for name in CATALOG
    }


def resolve_candidates(
    candidates: list[dict[str, Any]], devices: dict[str, Transistor]
) -> list[dict[str, Any]]:
    """Attach core TDB objects to API-shaped candidate mappings.

    :param candidates: Candidate definitions with switch and diode TDB IDs.
    :type candidates: list[dict[str, Any]]
    :param devices: Devices keyed by TDB identity.
    :type devices: dict[str, Transistor]
    :returns: Candidate mappings with resolved switch and diode objects.
    :rtype: list[dict[str, Any]]
    """
    return [
        {
            **candidate,
            "switch_transistor": devices[candidate["switch_transistor_id"]],
            "diode_transistor": devices[candidate["diode_transistor_id"]],
        }
        for candidate in candidates
    ]


def curve_device(
    name: str, *, v_abs_max: float = 1200.0, i_abs_max: float = 100.0
) -> Transistor:
    """Build an exact-condition curve fixture with configurable ratings.

    :param name: TDB identity for the fixture device.
    :type name: str
    :param v_abs_max: Absolute voltage limit in volts.
    :type v_abs_max: float
    :param i_abs_max: Absolute current limit in amps.
    :type i_abs_max: float
    :returns: Device with switch and diode curves at 25 °C.
    :rtype: Transistor
    """
    transistor = Transistor(
        TransistorMetadata(name, "MOSFET", "test", "vendor", "package"),
        ElectricalRatings(v_abs_max, i_abs_max, i_abs_max, 175),
        ThermalProperties(0.001, 0.001),
    )
    currents = np.array([0.1, 100.0])
    transistor.switch.channel_data.append(
        ChannelCharacteristics(25, np.array([[0.01, 10.0], currents]), 15)
    )
    transistor.diode.channel_data.append(
        ChannelCharacteristics(25, np.array([[0.4, 5.4], currents]), 0)
    )
    return transistor


def fixture_catalog(
    part_id: str, *, unit_cost_usd: float = 5.0
) -> dict[str, dict[str, Any]]:
    """Build a complete sourced catalog entry for one fixture part.

    :param part_id: Part identity used as the catalog key and orderable label.
    :type part_id: str
    :param unit_cost_usd: One-unit USD snapshot price.
    :type unit_cost_usd: float
    :returns: Catalog mapping with price/date and dimensions provenance.
    :rtype: dict[str, dict[str, Any]]
    """
    return {
        part_id: {
            "orderable_part": part_id,
            "unit_cost_usd": unit_cost_usd,
            "length_mm": 10.0,
            "width_mm": 5.0,
            "height_mm": 2.0,
            "price_source_url": "https://example.com/price",
            "cost_accessed_on": date(2026, 10, 14),
            "dimensions_source_url": "https://example.com/datasheet",
        }
    }


def test_notebook_candidates_match_saved_three_objective_results() -> None:
    """Keep the Python operation numerically aligned with the saved example.

    :returns: None.
    :rtype: None
    """
    devices = load_notebook_devices()
    candidates = resolve_candidates(notebook_candidates(), devices)

    result = calculate_boost_semiconductor_pareto(BOUNDARY, candidates, CATALOG)

    assert result["topology"] == "boost"
    assert result["boundary"]["mode"] == "loss-balanced"
    assert result["boundary"]["conduction_regime"] == "CCM"
    assert result["loss_scope"] == "channel_conduction_only"
    assert result["objective_directions"] == {
        "channel_conduction_loss_w": "minimize",
        "package_body_volume_mm3": "minimize",
        "component_cost_usd": "minimize",
    }
    assert result["eligible_candidate_count"] == 3
    assert result["pareto_candidate_ids"] == [
        "cree-one", "infineon-one", "infineon-two"
    ]

    rows = result["candidates"]
    assert [row["objectives"]["channel_conduction_loss_w"] for row in rows] == pytest.approx(
        [18.849, 17.623, 16.650], abs=0.0005
    )
    assert [row["operating_point"]["calculated"]["inductor_volt_second_residual_v"] for row in rows] == pytest.approx(
        [0.004662, 0.004179, 0.003705], abs=5e-7
    )
    assert [row["objectives"]["package_body_volume_mm3"] for row in rows] == pytest.approx(
        [853.2, 860.4, 1294.1], abs=0.05
    )
    assert [row["objectives"]["component_cost_usd"] for row in rows] == pytest.approx(
        [19.28, 20.42, 31.20]
    )
    assert all(row["pareto"] for row in rows)
    assert all(row["ratings"]["pass"] for row in rows)
    assert all(row["operating_point"]["temperatures"]["switch_c"] is None for row in rows)
    assert all(
        row["catalog"]["switch"]["cost_accessed_on"] == "2026-10-14"
        for row in rows
    )
    assert "Exact objective ties remain non-dominated." in result["dominance_rule"]


def test_exact_objective_ties_both_remain_on_frontier() -> None:
    """Keep exact objective ties on the non-dominated frontier.

    :returns: None.
    :rtype: None
    """
    transistor = curve_device("same-part")
    candidate = {
        "candidate_id": "first",
        "label": "First",
        "switch_transistor_id": "same-part",
        "diode_transistor_id": "same-part",
        "parallel_switches": 1,
        "parallel_diodes": 1,
        "v_g": 15,
        "diode_v_g": 0,
        "switch_transistor": transistor,
        "diode_transistor": transistor,
    }
    other = {**candidate, "candidate_id": "second", "label": "Second"}

    result = calculate_boost_semiconductor_pareto(
        BOUNDARY, [candidate, other], fixture_catalog("same-part")
    )

    assert result["pareto_candidate_ids"] == ["first", "second"]
    assert result["candidates"][0]["objectives"] == result["candidates"][1]["objectives"]


def test_strictly_better_candidate_dominates_and_rating_failures_are_excluded() -> None:
    """Apply strict dominance and exclude absolute-rating failures.

    :returns: None.
    :rtype: None
    """
    low_cost = curve_device("low-cost")
    high_cost = curve_device("high-cost")
    over_voltage = curve_device("over-voltage", v_abs_max=150)
    candidates = [
        {
            "candidate_id": "low-cost",
            "label": "Low-cost device",
            "switch_transistor_id": "low-cost",
            "diode_transistor_id": "low-cost",
            "v_g": 15,
            "diode_v_g": 0,
            "switch_transistor": low_cost,
            "diode_transistor": low_cost,
        },
        {
            "candidate_id": "high-cost",
            "label": "Higher-cost device",
            "switch_transistor_id": "high-cost",
            "diode_transistor_id": "high-cost",
            "v_g": 15,
            "diode_v_g": 0,
            "switch_transistor": high_cost,
            "diode_transistor": high_cost,
        },
        {
            "candidate_id": "over-voltage",
            "label": "Insufficient voltage rating",
            "switch_transistor_id": "over-voltage",
            "diode_transistor_id": "over-voltage",
            "v_g": 15,
            "diode_v_g": 0,
            "switch_transistor": over_voltage,
            "diode_transistor": over_voltage,
        },
    ]
    catalog = {
        **fixture_catalog("low-cost", unit_cost_usd=4),
        **fixture_catalog("high-cost", unit_cost_usd=8),
        **fixture_catalog("over-voltage", unit_cost_usd=3),
    }

    result = calculate_boost_semiconductor_pareto(BOUNDARY, candidates, catalog)

    assert result["pareto_candidate_ids"] == ["low-cost"]
    assert result["candidates"][1]["eligible"] is True
    assert result["candidates"][1]["pareto"] is False
    assert result["candidates"][2]["eligible"] is False
    assert any("voltage rating" in reason for reason in result["candidates"][2]["ineligible_reasons"])


def test_curve_and_catalog_failures_are_returned_for_the_candidate() -> None:
    """Keep curve and catalog failures local to their candidates.

    :returns: None.
    :rtype: None
    """
    transistor = curve_device("curve-device")
    valid = {
        "candidate_id": "valid",
        "label": "Valid curves and catalog",
        "switch_transistor_id": "curve-device",
        "diode_transistor_id": "curve-device",
        "v_g": 15,
        "diode_v_g": 0,
        "switch_transistor": transistor,
        "diode_transistor": transistor,
    }
    wrong_gate = {
        **valid,
        "candidate_id": "wrong-gate",
        "label": "Wrong exact gate condition",
        "v_g": 14,
    }
    incomplete_catalog = {
        **valid,
        "candidate_id": "missing-market-data",
        "label": "Missing market data",
        "switch_transistor_id": "uncataloged-part",
    }
    catalog = fixture_catalog("curve-device")

    result = calculate_boost_semiconductor_pareto(
        BOUNDARY, [valid, wrong_gate, incomplete_catalog], catalog
    )

    assert result["pareto_candidate_ids"] == ["valid"]
    assert [candidate["eligible"] for candidate in result["candidates"]] == [
        True, False, False
    ]
    assert any("Missing required switch channel curve" in reason for reason in result["candidates"][1]["ineligible_reasons"])
    assert any("No sourced package and cost data" in reason for reason in result["candidates"][2]["ineligible_reasons"])


def test_direct_catalog_validation_is_candidate_local() -> None:
    """Reject malformed numeric, URL, and date metadata for one candidate.

    :returns: None.
    :rtype: None
    """
    valid_device = curve_device("valid-part")
    malformed_device = curve_device("malformed-part")
    candidates = [
        {
            "candidate_id": "valid",
            "label": "Valid catalog",
            "switch_transistor_id": "valid-part",
            "diode_transistor_id": "valid-part",
            "v_g": 15,
            "diode_v_g": 0,
            "switch_transistor": valid_device,
            "diode_transistor": valid_device,
        },
        {
            "candidate_id": "malformed",
            "label": "Malformed catalog",
            "switch_transistor_id": "malformed-part",
            "diode_transistor_id": "malformed-part",
            "v_g": 15,
            "diode_v_g": 0,
            "switch_transistor": malformed_device,
            "diode_transistor": malformed_device,
        },
    ]
    catalog = {
        **fixture_catalog("valid-part"),
        **fixture_catalog("malformed-part"),
    }
    catalog["malformed-part"].update(
        unit_cost_usd=0,
        length_mm=-1,
        price_source_url="not-a-url",
        cost_accessed_on="not-a-date",
        dimensions_source_url="ftp://example.com/datasheet",
    )

    result = calculate_boost_semiconductor_pareto(BOUNDARY, candidates, catalog)

    assert result["eligible_candidate_count"] == 1
    assert result["pareto_candidate_ids"] == ["valid"]
    malformed = result["candidates"][1]
    assert malformed["eligible"] is False
    assert malformed["objectives"] is None
    reasons = " ".join(malformed["ineligible_reasons"])
    assert "unit_cost_usd" in reasons
    assert "length_mm" in reasons
    assert "price_source_url" in reasons
    assert "cost_accessed_on" in reasons
    assert "dimensions_source_url" in reasons


def test_direct_catalog_rejects_untyped_and_nonfinite_values_locally() -> None:
    """Reject untyped and overflowing catalog values for one candidate.

    :returns: None.
    :rtype: None
    """
    valid_device = curve_device("valid-part")
    malformed_device = curve_device("malformed-part")
    candidates = [
        {
            "candidate_id": "valid",
            "label": "Valid catalog",
            "switch_transistor_id": "valid-part",
            "diode_transistor_id": "valid-part",
            "v_g": 15,
            "diode_v_g": 0,
            "switch_transistor": valid_device,
            "diode_transistor": valid_device,
        },
        {
            "candidate_id": "malformed",
            "label": "Malformed catalog values",
            "switch_transistor_id": "malformed-part",
            "diode_transistor_id": "malformed-part",
            "v_g": 15,
            "diode_v_g": 0,
            "switch_transistor": malformed_device,
            "diode_transistor": malformed_device,
        },
    ]
    catalog = {
        **fixture_catalog("valid-part"),
        **fixture_catalog("malformed-part"),
    }
    catalog["malformed-part"].update(
        unit_cost_usd="unknown",
        length_mm=1e309,
        width_mm=10**1000,
        price_source_url=123,
        dimensions_source_url="",
    )

    result = calculate_boost_semiconductor_pareto(BOUNDARY, candidates, catalog)

    assert result["eligible_candidate_count"] == 1
    assert result["pareto_candidate_ids"] == ["valid"]
    malformed = result["candidates"][1]
    assert malformed["eligible"] is False
    reasons = " ".join(malformed["ineligible_reasons"])
    assert "unit_cost_usd" in reasons
    assert "length_mm" in reasons
    assert "width_mm" in reasons
    assert "price_source_url" in reasons
    assert "package-dimensions source URL" in reasons
