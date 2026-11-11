"""Tests for analytical switching loss models."""
from __future__ import annotations

import numpy as np
import pytest

from transistordatabase.analytical_models import (
    ChristenBielaModel,
    HalfBridgeParams,
    TransconductanceParams,
    ReverseRecoveryParams,
    SwitchingEnergyResult,
    GateChargeModel,
    GateChargeModelParams,
    IgbtModel,
    IgbtModelParams,
    calc_charge_equivalent_capacitance,
    calc_charge_stored,
    calc_device_capacitances,
    get_package_inductance,
)


class TestChristenBielaModel:
    """Public import compatibility; detailed physical fixtures are in test_biela_contract."""

    def test_corrected_public_model(self):
        """Keep the public model import numerically tied to the independent fixture."""
        model = ChristenBielaModel(1e-9, 1e-9, 1e-9, 200e-9,
                                  TransconductanceParams(10, 0, 1, 3))
        result = model.calc_switching_energy(HalfBridgeParams(100, 10, 15, 0, 10, 0, 0))
        assert isinstance(result, SwitchingEnergyResult)
        assert result.e_total == pytest.approx(178.7589668316306e-6)

    def test_charge_control_constants_are_not_arbitrary(self):
        """Reject inconsistent lifetime/transit constants instead of silently using them."""
        with pytest.raises(ValueError, match='satisfy'):
            ChristenBielaModel(1e-9, 1e-9, 1e-9, 200e-9,
                              TransconductanceParams(10, 0, 1, 3),
                              ReverseRecoveryParams(8.6e-9, 16e-9, 18.6e-9, 88e-9))


class TestCapacitanceFunctions:
    """Tests for capacitance calculation functions."""

    def test_charge_equivalent_capacitance(self) -> None:
        """Known C(V) curve returns expected value."""
        v_axis = np.array([0, 100, 200])
        c_axis = np.array([1000e-12, 500e-12, 200e-12])
        c_eq = calc_charge_equivalent_capacitance(v_axis, c_axis, 200.0)
        # Approximate integral / V_0
        assert 500e-12 < c_eq < 600e-12

    def test_charge_stored(self) -> None:
        """Integral of C(V) returns expected charge."""
        v_axis = np.array([0, 100, 200])
        c_axis = np.array([1000e-12, 500e-12, 200e-12])
        q = calc_charge_stored(v_axis, c_axis, 200.0)
        # Trapz integration
        assert 100e-9 < q < 150e-9

    def test_device_capacitances(self) -> None:
        """C_gs, C_ds, C_gd computed correctly."""
        c_gs, c_ds, c_gd = calc_device_capacitances(1100e-12, 150e-12, 15e-12)
        assert c_gs == pytest.approx(1085e-12)
        assert c_ds == pytest.approx(135e-12)
        assert c_gd == pytest.approx(15e-12)


class TestPackageLookup:
    """Tests for package inductance lookup."""

    def test_known_packages(self) -> None:
        """TO-247, TO-220, QFN return expected values."""
        assert get_package_inductance("TO-247") == (4e-9, 10e-9)
        assert get_package_inductance("TO-220") == (5e-9, 12e-9)
        assert get_package_inductance("QFN") == (0.5e-9, 1e-9)

    def test_unknown_package(self) -> None:
        """Returns default (4e-9, 10e-9)."""
        assert get_package_inductance("unknown") == (4e-9, 10e-9)


class TestGateChargeModel:
    """Tests for the gate charge switching time model."""

    @pytest.fixture
    def model(self) -> GateChargeModel:
        """Create a standard gate charge model for testing."""
        params = GateChargeModelParams(
            q_gs=10e-9,      # 10nC
            q_gd=15e-9,      # 15nC
            q_g=50e-9,       # 50nC
            v_th=3.0,
            v_plateau=4.5,
            r_g=10.0,
            r_g_int=1.0,
        )
        return GateChargeModel(params)

    def test_turn_on_times_positive(self, model: GateChargeModel) -> None:
        """Verify all turn-on times are positive."""
        times = model.calc_turn_on_times(15.0)
        assert times['t_delay'] >= 0
        assert times['t_rise'] >= 0
        assert times['t_fall_v'] >= 0
        assert times['t_total'] > 0

    def test_turn_off_times_positive(self, model: GateChargeModel) -> None:
        """Verify all turn-off times are positive."""
        times = model.calc_turn_off_times(15.0, 0.0)
        assert times['t_delay'] >= 0
        assert times['t_rise_v'] >= 0
        assert times['t_fall_i'] >= 0
        assert times['t_total'] > 0

    def test_higher_rg_slower(self) -> None:
        """Verify higher gate resistance gives longer switching times."""
        params_low = GateChargeModelParams(
            q_gs=10e-9, q_gd=15e-9, q_g=50e-9,
            v_th=3.0, v_plateau=4.5, r_g=5.0,
        )
        params_high = GateChargeModelParams(
            q_gs=10e-9, q_gd=15e-9, q_g=50e-9,
            v_th=3.0, v_plateau=4.5, r_g=20.0,
        )
        t_low = GateChargeModel(params_low).calc_turn_on_times()['t_total']
        t_high = GateChargeModel(params_high).calc_turn_on_times()['t_total']
        assert t_high > t_low

    def test_total_is_sum_of_phases(self, model: GateChargeModel) -> None:
        """Verify total time is sum of individual phases."""
        times = model.calc_turn_on_times()
        assert times['t_total'] == pytest.approx(
            times['t_delay'] + times['t_rise'] + times['t_fall_v']
        )


class TestIgbtModel:
    """Tests for the IGBT tail current model."""

    @pytest.fixture
    def model(self) -> IgbtModel:
        """Create a standard IGBT model for testing."""
        return IgbtModel(IgbtModelParams(
            v_ce_sat=1.5,
            i_tail_factor=0.1,
            tau_tail=1e-6,
            t_fall=200e-9,
        ))

    def test_tail_energy_positive(self, model: IgbtModel) -> None:
        """Verify tail energy is positive."""
        e_tail = model.calc_tail_energy(600.0, 100.0)
        assert e_tail > 0

    def test_tail_increases_with_current(self, model: IgbtModel) -> None:
        """Verify tail energy increases with load current."""
        e_50 = model.calc_tail_energy(600.0, 50.0)
        e_200 = model.calc_tail_energy(600.0, 200.0)
        assert e_200 > e_50

    def test_e_off_total_exceeds_overlap(self, model: IgbtModel) -> None:
        """Verify total off energy exceeds overlap-only energy."""
        e_overlap = 0.5 * 600.0 * 100.0 * 200e-9
        e_total = model.calc_e_off_total(600.0, 100.0, e_overlap)
        assert e_total > e_overlap

    def test_conduction_loss(self, model: IgbtModel) -> None:
        """Verify conduction loss calculation."""
        p_cond = model.calc_conduction_loss(i_avg=50.0, _i_rms=60.0)
        assert p_cond == pytest.approx(1.5 * 50.0)
