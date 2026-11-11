"""Tests for gate resistance dependent switching energy."""
from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from transistordatabase.core.models import MeasurementEvidence, SwitchingLossData
from transistordatabase.rg_formula import (
    find_rg_for_target_energy,
    get_available_rg_values,
    interpolate_energy_at_rg,
    scale_energy_by_rg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def loss_data_with_graph_r_e() -> list[SwitchingLossData]:
    """SwitchingLossData entries that carry graph_r_e curves."""
    return [
        SwitchingLossData(
            dataset_type="graph_r_e",
            t_j=25.0,
            v_supply=800.0,
            v_g=15.0,
            r_g=5.0,
            i_x=20.0,
            graph_r_e=np.array([
                [1.0, 2.5, 5.0, 10.0, 20.0],   # resistance (Ohm)
                [0.5e-3, 0.8e-3, 1.2e-3, 2.0e-3, 3.5e-3],  # energy (J)
            ]),
        ),
    ]


@pytest.fixture()
def loss_data_with_graph_i_e() -> list[SwitchingLossData]:
    """SwitchingLossData entries with graph_i_e at different r_g values."""
    currents = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    return [
        SwitchingLossData(
            dataset_type="graph_i_e",
            t_j=25.0,
            v_supply=800.0,
            v_g=15.0,
            r_g=2.5,
            graph_i_e=np.array([
                currents,
                np.array([0.0, 0.1e-3, 0.3e-3, 0.6e-3, 1.0e-3, 1.5e-3]),
            ]),
        ),
        SwitchingLossData(
            dataset_type="graph_i_e",
            t_j=25.0,
            v_supply=800.0,
            v_g=15.0,
            r_g=5.0,
            graph_i_e=np.array([
                currents,
                np.array([0.0, 0.15e-3, 0.45e-3, 0.9e-3, 1.5e-3, 2.25e-3]),
            ]),
        ),
        SwitchingLossData(
            dataset_type="graph_i_e",
            t_j=25.0,
            v_supply=800.0,
            v_g=15.0,
            r_g=10.0,
            graph_i_e=np.array([
                currents,
                np.array([0.0, 0.2e-3, 0.6e-3, 1.2e-3, 2.0e-3, 3.0e-3]),
            ]),
        ),
        # A dataset at a different temperature (150 deg C)
        SwitchingLossData(
            dataset_type="graph_i_e",
            t_j=150.0,
            v_supply=800.0,
            v_g=15.0,
            r_g=5.0,
            graph_i_e=np.array([
                currents,
                np.array([0.0, 0.18e-3, 0.54e-3, 1.08e-3, 1.8e-3, 2.7e-3]),
            ]),
        ),
    ]


# ---------------------------------------------------------------------------
# TestInterpolateEnergyAtRg
# ---------------------------------------------------------------------------


class TestInterpolateEnergyAtRg:
    """Tests for interpolate_energy_at_rg."""

    def test_direct_interpolation_from_graph_r_e(
        self, loss_data_with_graph_r_e: list[SwitchingLossData]
    ) -> None:
        """Interpolate directly on the Rg-Energy curve."""
        energy = interpolate_energy_at_rg(
            loss_data_with_graph_r_e, r_g=5.0, i_channel=20.0, t_j=25.0,
        )
        assert energy == pytest.approx(1.2e-3, rel=1e-6)

    def test_direct_interpolation_midpoint(
        self, loss_data_with_graph_r_e: list[SwitchingLossData]
    ) -> None:
        """Interpolate at an Rg between data points on graph_r_e."""
        energy = interpolate_energy_at_rg(
            loss_data_with_graph_r_e, r_g=7.5, i_channel=20.0, t_j=25.0,
        )
        # Linear interp between (5.0, 1.2e-3) and (10.0, 2.0e-3)
        expected = 1.2e-3 + (2.0e-3 - 1.2e-3) * (7.5 - 5.0) / (10.0 - 5.0)
        assert energy == pytest.approx(expected, rel=1e-6)

    def test_unrecorded_rg_rejected(
        self, loss_data_with_graph_i_e: list[SwitchingLossData]
    ) -> None:
        """Reject an unrecorded Rg instead of selecting the nearest curve."""
        with pytest.raises(ValueError, match="No switching energy curve"):
            interpolate_energy_at_rg(
                loss_data_with_graph_i_e, r_g=4.0, i_channel=30.0, t_j=25.0,
            )

    def test_exact_rg_match(
        self, loss_data_with_graph_i_e: list[SwitchingLossData]
    ) -> None:
        """Exact Rg match on graph_i_e datasets."""
        energy = interpolate_energy_at_rg(
            loss_data_with_graph_i_e, r_g=10.0, i_channel=40.0, t_j=25.0,
        )
        assert energy == pytest.approx(2.0e-3, rel=1e-6)

    def test_no_data_rejected(self) -> None:
        """Reject absent evidence instead of manufacturing zero energy."""
        with pytest.raises(ValueError, match="No switching energy curve"):
            interpolate_energy_at_rg([], r_g=5.0, i_channel=20.0, t_j=25.0)

    def test_no_matching_temperature_rejected(
        self, loss_data_with_graph_i_e: list[SwitchingLossData]
    ) -> None:
        """Reject a temperature without recorded evidence."""
        with pytest.raises(ValueError, match="No switching energy curve"):
            interpolate_energy_at_rg(
                loss_data_with_graph_i_e, r_g=5.0, i_channel=20.0, t_j=75.0,
            )


# ---------------------------------------------------------------------------
# TestScaleEnergyByRg
# ---------------------------------------------------------------------------


class TestScaleEnergyByRg:
    """Tests for scale_energy_by_rg."""

    def test_linear_scaling(self) -> None:
        """Linear scaling: E_target = E_ref * r_g_target / r_g_ref."""
        result = scale_energy_by_rg(
            e_ref=1.0e-3, r_g_ref=5.0, r_g_target=10.0, scaling="linear",
        )
        assert result == pytest.approx(2.0e-3, rel=1e-9)

    def test_sqrt_scaling(self) -> None:
        """Square-root scaling: E_target = E_ref * sqrt(r_g_target / r_g_ref)."""
        result = scale_energy_by_rg(
            e_ref=1.0e-3, r_g_ref=4.0, r_g_target=16.0, scaling="sqrt",
        )
        # sqrt(16/4) = 2.0
        assert result == pytest.approx(2.0e-3, rel=1e-9)

    def test_identity_scaling(self) -> None:
        """Scaling with same Rg returns the reference energy."""
        result = scale_energy_by_rg(
            e_ref=1.5e-3, r_g_ref=5.0, r_g_target=5.0, scaling="linear",
        )
        assert result == pytest.approx(1.5e-3, rel=1e-9)

    def test_rg_ref_zero_raises(self) -> None:
        """Zero reference resistance raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            scale_energy_by_rg(
                e_ref=1.0e-3, r_g_ref=0.0, r_g_target=5.0, scaling="linear",
            )

    def test_unknown_scaling_raises(self) -> None:
        """Unknown scaling method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scaling method"):
            scale_energy_by_rg(
                e_ref=1.0e-3, r_g_ref=5.0, r_g_target=10.0, scaling="quadratic",
            )


# ---------------------------------------------------------------------------
# TestFindRgForTargetEnergy
# ---------------------------------------------------------------------------


class TestFindRgForTargetEnergy:
    """Tests for find_rg_for_target_energy."""

    def test_find_rg_from_graph_r_e(
        self, loss_data_with_graph_r_e: list[SwitchingLossData]
    ) -> None:
        """Find Rg from a graph_r_e curve."""
        rg = find_rg_for_target_energy(
            loss_data_with_graph_r_e,
            e_target=1.2e-3,
            i_channel=20.0,
            t_j=25.0,
        )
        assert rg is not None
        assert rg == pytest.approx(5.0, rel=1e-6)

    def test_find_rg_interpolated(
        self, loss_data_with_graph_r_e: list[SwitchingLossData]
    ) -> None:
        """Find an Rg value between data points on graph_r_e."""
        # Energy 1.6e-3 lies between (5.0, 1.2e-3) and (10.0, 2.0e-3)
        rg = find_rg_for_target_energy(
            loss_data_with_graph_r_e,
            e_target=1.6e-3,
            i_channel=20.0,
            t_j=25.0,
        )
        assert rg is not None
        # Linear interp: rg = 5.0 + (10.0-5.0)*(1.6e-3-1.2e-3)/(2.0e-3-1.2e-3) = 7.5
        assert rg == pytest.approx(7.5, rel=1e-3)

    def test_target_outside_range_rejected(
        self, loss_data_with_graph_r_e: list[SwitchingLossData]
    ) -> None:
        """Reject a target energy outside the recorded domain."""
        with pytest.raises(ValueError, match="inside recorded coverage"):
            find_rg_for_target_energy(
                loss_data_with_graph_r_e, e_target=10.0,
                i_channel=20.0, t_j=25.0,
            )

    def test_find_rg_from_graph_i_e_fallback(
        self, loss_data_with_graph_i_e: list[SwitchingLossData]
    ) -> None:
        """Fall back to building Rg-energy map from graph_i_e datasets."""
        # At i_channel=30, the three datasets give:
        #   r_g=2.5 -> 0.6e-3,  r_g=5.0 -> 0.9e-3,  r_g=10.0 -> 1.2e-3
        rg = find_rg_for_target_energy(
            loss_data_with_graph_i_e,
            e_target=0.9e-3,
            i_channel=30.0,
            t_j=25.0,
        )
        assert rg is not None
        assert rg == pytest.approx(5.0, rel=1e-3)

    def test_empty_data_rejected(self) -> None:
        """Reject an inverse request without recorded evidence."""
        with pytest.raises(ValueError, match="No switching energy curve"):
            find_rg_for_target_energy([], e_target=1.0e-3, i_channel=20.0, t_j=25.0)


# ---------------------------------------------------------------------------
# TestGetAvailableRgValues
# ---------------------------------------------------------------------------


class TestGetAvailableRgValues:
    """Tests for get_available_rg_values."""

    def test_extraction(
        self, loss_data_with_graph_i_e: list[SwitchingLossData]
    ) -> None:
        """Extract unique Rg values sorted in ascending order."""
        rg_values = get_available_rg_values(loss_data_with_graph_i_e)
        assert rg_values == [2.5, 5.0, 10.0]

    def test_empty_list(self) -> None:
        """Return empty list for empty input."""
        assert get_available_rg_values([]) == []

    def test_none_rg_excluded(self) -> None:
        """Datasets without r_g are excluded."""
        data = [
            SwitchingLossData(
                dataset_type="graph_i_e",
                t_j=25.0,
                v_supply=800.0,
                v_g=15.0,
                r_g=None,
            ),
        ]
        assert get_available_rg_values(data) == []

    def test_duplicates_collapsed(self) -> None:
        """Duplicate Rg values are collapsed to unique entries."""
        data = [
            SwitchingLossData(
                dataset_type="graph_i_e",
                t_j=25.0,
                v_supply=800.0,
                v_g=15.0,
                r_g=5.0,
            ),
            SwitchingLossData(
                dataset_type="graph_i_e",
                t_j=150.0,
                v_supply=800.0,
                v_g=15.0,
                r_g=5.0,
            ),
        ]
        assert get_available_rg_values(data) == [5.0]


def rg_curve(energies=(3.0, 5.0, 7.0), **changes):
    """Create E(Rg) = 1 + 2 Rg at a fully recorded operating point."""
    record = SwitchingLossData(
        dataset_type="graph_r_e", t_j=25.0, v_supply=800.0,
        v_g=15.0, v_g_off=-3.0, i_x=20.0,
        graph_r_e=np.array([[1.0, 2.0, 3.0], energies]),
    )
    return replace(record, **changes)


@pytest.mark.parametrize("operation", [interpolate_energy_at_rg, find_rg_for_target_energy])
@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("invalid", [True, np.bool_(False), 1 + 0j, "2", np.nan, np.inf])
def test_query_requires_real_finite_numbers(operation, index, invalid):
    """Reject invalid query values before selecting source curves."""
    arguments = [2.0, 20.0, 25.0]
    arguments[index] = invalid
    with pytest.raises(ValueError):
        operation([rg_curve()], *arguments)


@pytest.mark.parametrize("operation", [interpolate_energy_at_rg, find_rg_for_target_energy])
@pytest.mark.parametrize("arguments", [(-1.0, 20.0, 25.0), (2.0, -1.0, 25.0)])
def test_negative_query_magnitudes_rejected(operation, arguments):
    """Reject negative current, resistance and target energy."""
    with pytest.raises(ValueError, match="nonnegative"):
        operation([rg_curve()], *arguments)


@pytest.mark.parametrize("recorded_current", [None, 10.0, False, np.bool_(True), 20 + 0j])
@pytest.mark.parametrize("operation", [interpolate_energy_at_rg, find_rg_for_target_energy])
def test_rg_curve_requires_exact_recorded_current(operation, recorded_current):
    """Do not apply an Rg curve to unknown or different current conditions."""
    with pytest.raises(ValueError):
        operation([rg_curve(i_x=recorded_current)], 2.0, 20.0, 25.0)


@pytest.mark.parametrize("name,value", [
    ("v_supply", 400.0), ("v_g", 12.0), ("v_g_off", 0.0),
    ("commutation_device", "other diode"), ("load_inductance", 1e-3),
    ("commutation_inductance", 1e-8),
])
@pytest.mark.parametrize("operation", [interpolate_energy_at_rg, find_rg_for_target_energy])
def test_fixed_conditions_need_selection(operation, name, value):
    """Reject mixed physical contexts, then select the exact desired context."""
    base = rg_curve()
    changed = rg_curve((6.0, 10.0, 14.0), **{name: value})
    target = 2.0 if operation is interpolate_energy_at_rg else 10.0
    with pytest.raises(ValueError, match="physical switching energy context"):
        operation([base, changed], target, 20.0, 25.0)
    result = operation([base, changed], target, 20.0, 25.0,
                       conditions={name: value})
    assert result == pytest.approx(10.0 if target == 2.0 else 2.0)


@pytest.mark.parametrize("name", ["convention", "event", "method"])
def test_declared_evidence_conditions_prevent_mixing(name):
    """Keep recorded event definitions and energy conventions distinct."""
    first = rg_curve(evidence=MeasurementEvidence(**{name: "first"}))
    second = rg_curve((6.0, 10.0, 14.0), evidence=MeasurementEvidence(**{name: "second"}))
    with pytest.raises(ValueError, match="physical switching energy context"):
        find_rg_for_target_energy([first, second], 10.0, 20.0, 25.0)
    assert find_rg_for_target_energy(
        [first, second], 10.0, 20.0, 25.0, conditions={name: "second"},
    ) == 2.0


def test_unknown_condition_matches_only_explicit_unknown():
    """Do not substitute a specified gate voltage for unknown source conditions."""
    records = [rg_curve(v_g_off=None), rg_curve((6.0, 10.0, 14.0), v_g_off=0.0)]
    assert interpolate_energy_at_rg(
        records, 2.0, 20.0, 25.0, conditions={"v_g_off": None},
    ) == 5.0
    with pytest.raises(ValueError, match="No switching energy curve"):
        interpolate_energy_at_rg(records, 2.0, 20.0, 25.0,
                                 conditions={"v_g_off": -3.0})


@pytest.mark.parametrize("invalid", [False, np.bool_(False), "800", 800 + 0j, np.nan])
def test_invalid_recorded_condition_is_not_an_absent_slice(invalid):
    """Validate stored numeric evidence before exact-condition matching."""
    with pytest.raises(ValueError):
        interpolate_energy_at_rg([rg_curve(v_supply=invalid)], 2.0, 20.0, 25.0)


@pytest.mark.parametrize("graph", [
    np.array([[False, True], [False, True]]),
    np.array([[1 + 0j, 2 + 0j], [3 + 0j, 5 + 0j]]),
    np.array([["1", "2"], ["3", "5"]]),
    np.array([[1.0, 2.0], [3.0, np.nan]]),
    np.array([[1.0, 1.0], [3.0, 5.0]]),
    np.array([[-1.0, 2.0], [3.0, 5.0]]),
    np.array([[1.0, 2.0], [-3.0, 5.0]]),
    np.array([[1.0], [3.0]]),
    None,
])
@pytest.mark.parametrize("operation", [interpolate_energy_at_rg, find_rg_for_target_energy])
def test_invalid_source_samples_rejected(operation, graph):
    """Reject invalid samples before interpolation or inverse root search."""
    with pytest.raises(ValueError):
        operation([rg_curve(graph_r_e=graph)], 2.0, 20.0, 25.0)


@pytest.mark.parametrize("resistance", [0.0, 4.0])
def test_forward_resistance_coverage_is_bounded(resistance):
    """Do not clamp a requested Rg outside the measured curve."""
    with pytest.raises(ValueError, match="supports"):
        interpolate_energy_at_rg([rg_curve()], resistance, 20.0, 25.0)


@pytest.mark.parametrize("operation", [interpolate_energy_at_rg, find_rg_for_target_energy])
def test_current_curve_coverage_is_bounded(operation, loss_data_with_graph_i_e):
    """Do not clamp current on any source curve."""
    with pytest.raises(ValueError, match="supports"):
        operation(loss_data_with_graph_i_e, 5.0, 51.0, 25.0)


@pytest.mark.parametrize("energies,target,expected", [
    ((3.0, 5.0, 7.0), 4.0, 1.5),
    ((7.0, 5.0, 3.0), 4.0, 2.5),
    ((7.0, 3.0, 5.0), 6.0, 1.25),
    ((7.0, 3.0, 5.0), 3.0, 2.0),
    ((3.0, 5.0, 7.0), 3.0, 1.0),
    ((3.0, 5.0, 7.0), 7.0, 3.0),
])
def test_inverse_scans_every_segment_and_deduplicates_knots(energies, target, expected):
    """Recover analytic roots on increasing, decreasing and nonmonotonic data."""
    assert find_rg_for_target_energy([rg_curve(energies)], target, 20.0, 25.0) == expected


@pytest.mark.parametrize("energies,target", [
    ((7.0, 3.0, 5.0), 4.0),
    ((3.0, 5.0, 3.0), 3.0),
    ((3.0, 3.0, 5.0), 3.0),
    ((3.0, 3.0, 3.0), 3.0),
    ((3.0, 5.0, 7.0), 8.0),
])
def test_inverse_rejects_multiple_or_absent_roots(energies, target):
    """Reject multiple crossings, flat target intervals and uncovered energy."""
    with pytest.raises(ValueError):
        find_rg_for_target_energy([rg_curve(energies)], target, 20.0, 25.0)


def test_duplicate_records_and_representations_require_selection():
    """Never use the first record when more than one representation applies."""
    direct = rg_curve()
    current = SwitchingLossData(
        dataset_type="graph_i_e", t_j=25.0, v_supply=800.0,
        v_g=15.0, v_g_off=-3.0, r_g=2.0,
        graph_i_e=np.array([[10.0, 30.0], [3.0, 7.0]]),
    )
    for operation in (interpolate_energy_at_rg, find_rg_for_target_energy):
        with pytest.raises(ValueError, match="unambiguous"):
            operation([direct, direct], 2.0, 20.0, 25.0)
        with pytest.raises(ValueError, match="unambiguous"):
            operation([direct, current], 2.0, 20.0, 25.0)
    assert interpolate_energy_at_rg(
        [direct, current], 2.0, 20.0, 25.0, representation="graph_i_e",
    ) == 5.0
    assert find_rg_for_target_energy(
        [direct, current], 5.0, 20.0, 25.0, representation="graph_r_e",
    ) == 2.0
    with pytest.raises(ValueError, match="unique x"):
        find_rg_for_target_energy([current, current], 5.0, 20.0, 25.0)


def test_actual_zero_energy_and_source_immutability():
    """Preserve measured zero and leave unsorted source evidence unchanged."""
    record = rg_curve(graph_r_e=np.array([[3.0, 1.0, 2.0], [4.0, 0.0, 2.0]]))
    original = record.graph_r_e.copy()
    assert interpolate_energy_at_rg([record], 1.0, 20.0, 25.0) == 0.0
    assert find_rg_for_target_energy([record], 0.0, 20.0, 25.0) == 1.0
    np.testing.assert_array_equal(record.graph_r_e, original)


def test_single_point_is_not_an_interpolation_curve():
    """Do not infer an Rg interval from one scalar switching measurement."""
    record = rg_curve(dataset_type="single", graph_r_e=None, e_x=5.0, r_g=2.0)
    for operation in (interpolate_energy_at_rg, find_rg_for_target_energy):
        with pytest.raises(ValueError, match="No switching energy curve"):
            operation([record], 2.0, 20.0, 25.0)


def test_unrelated_current_and_representation_do_not_create_ambiguity():
    """Exclude inapplicable current slices before testing context uniqueness."""
    selected = rg_curve()
    unrelated = rg_curve(i_x=10.0, v_supply=400.0)
    assert interpolate_energy_at_rg([selected, unrelated], 2.0, 20.0, 25.0) == 5.0
    assert find_rg_for_target_energy([selected, unrelated], 5.0, 20.0, 25.0) == 2.0
    other_representation = SwitchingLossData(
        dataset_type="graph_i_e", t_j=25.0, v_supply=400.0,
        v_g=15.0, r_g=8.0, graph_i_e=np.array([[10.0, 30.0], [3.0, 7.0]]),
    )
    assert interpolate_energy_at_rg(
        [selected, other_representation], 2.0, 20.0, 25.0,
    ) == 5.0
    assert find_rg_for_target_energy(
        [selected, other_representation], 5.0, 20.0, 25.0,
        representation="graph_r_e",
    ) == 2.0


@pytest.mark.parametrize("energies,target,expected", [
    ((3.0, 5.0, 7.0), 4.0, 1.5),
    ((7.0, 5.0, 3.0), 4.0, 2.5),
    ((7.0, 3.0, 5.0), 6.0, 1.25),
])
def test_inverse_current_curves_build_conditioned_rg_mapping(energies, target, expected):
    """Recover independent analytic roots after bounded current interpolation."""
    records = [SwitchingLossData(
        dataset_type="graph_i_e", t_j=25.0, v_supply=800.0, v_g=15.0,
        r_g=rg, graph_i_e=np.array([[10.0, 30.0], [energy - 1.0, energy + 1.0]]),
    ) for rg, energy in zip((1.0, 2.0, 3.0), energies, strict=True)]
    assert find_rg_for_target_energy(records, target, 20.0, 25.0) == expected


def test_incomplete_inverse_current_coverage_rejected():
    """Do not silently omit an Rg measurement outside its current coverage."""
    records = [SwitchingLossData(
        dataset_type="graph_i_e", t_j=25.0, v_supply=800.0, v_g=15.0,
        r_g=rg, graph_i_e=np.array([currents, [3.0, 7.0]]),
    ) for rg, currents in ((1.0, [10.0, 30.0]), (2.0, [30.0, 50.0]))]
    with pytest.raises(ValueError, match="supports"):
        find_rg_for_target_energy(records, 5.0, 20.0, 25.0)


def test_unrelated_current_coverage_does_not_select_another_context():
    """Ignore a physical context with no current coverage at the requested point."""
    chosen = [SwitchingLossData(
        dataset_type="graph_i_e", t_j=25.0, v_supply=800.0, v_g=15.0,
        r_g=rg, graph_i_e=np.array([[10.0, 30.0], [energy - 1.0, energy + 1.0]]),
    ) for rg, energy in ((1.0, 3.0), (2.0, 5.0))]
    unrelated = replace(chosen[0], v_supply=400.0,
                        graph_i_e=np.array([[40.0, 60.0], [3.0, 7.0]]))
    assert interpolate_energy_at_rg([*chosen, unrelated], 1.0, 20.0, 25.0) == 3.0
    assert find_rg_for_target_energy([*chosen, unrelated], 4.0, 20.0, 25.0) == 1.5


def test_target_energy_does_not_select_physical_context():
    """Reject two eligible physical contexts even if only one has a target root."""
    records = [rg_curve(), rg_curve((10.0, 20.0, 30.0), v_supply=400.0)]
    with pytest.raises(ValueError, match="physical switching energy context"):
        find_rg_for_target_energy(records, 5.0, 20.0, 25.0)


def test_forward_ignores_a_representation_without_rg_coverage():
    """Use the unique covered representation at the same physical context."""
    current = SwitchingLossData(
        dataset_type="graph_i_e", t_j=25.0, v_supply=800.0, v_g=15.0,
        v_g_off=-3.0, r_g=4.0, graph_i_e=np.array([[10.0, 30.0], [6.0, 8.0]]),
    )
    assert interpolate_energy_at_rg([rg_curve(), current], 4.0, 20.0, 25.0) == 7.0


def test_uncovered_tolerance_bridge_is_rejected_in_every_source_order():
    """Reject an uncovered context that matches only part of the covered context."""
    records = [SwitchingLossData(
        dataset_type="graph_i_e", t_j=25.0, v_supply=800.0, v_g=15.0,
        r_g=rg, commutation_inductance=inductance,
        graph_i_e=np.array([currents, [energy - 1.0, energy + 1.0]]),
    ) for rg, inductance, currents, energy in (
        (1.0, 10e-9, [30.0, 50.0], 12.0),
        (5.0, 10.75e-9, [10.0, 30.0], 8.0),
        (9.0, 11.5e-9, [10.0, 30.0], 4.0),
    )]
    for order in permutations(records):
        with pytest.raises(ValueError, match="physical switching energy context"):
            find_rg_for_target_energy(list(order), 6.0, 20.0, 25.0)
