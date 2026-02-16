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
    """Tests for the Christen-Biela half-bridge switching loss model."""

    @pytest.fixture
    def c2m_model(self) -> ChristenBielaModel:
        """C2M0080120D-like model from paper Section III."""
        # Approximate paper parameters for C2M0080120D
        return ChristenBielaModel(
            c_gs=1080e-12,
            c_ds=130e-12,
            c_gd=14.5e-12,
            q_oss=86.56e-9,
            gm_params=TransconductanceParams(k_1=0.1319, k_2=-0.076, x=3.80, v_th=4.5),
            rr_params=ReverseRecoveryParams(
                tau_rr=8.6e-9, tau_c=16e-9, t_m=18.6e-9, q_rr_star=88e-9
            ),
        )

    @pytest.fixture
    def default_params(self) -> HalfBridgeParams:
        """C2M0080120D operating point from paper Table I."""
        return HalfBridgeParams(
            v_0=600, i_0=20, v_g_on=20, v_g_off=-5, r_g=7.1, l_s=4e-9, l_d=10e-9
        )

    def test_e_off_reference_point(
        self, c2m_model: ChristenBielaModel, default_params: HalfBridgeParams
    ) -> None:
        """E_T,off should be in reasonable range (paper: 14.1uJ)."""
        result = c2m_model.calc_switching_energy(default_params)
        # Wide tolerance due to parameter uncertainty
        assert 5e-6 < result.e_off < 30e-6

    def test_e_on_reference_point(
        self, c2m_model: ChristenBielaModel, default_params: HalfBridgeParams
    ) -> None:
        """E_T,on should be in reasonable range (paper: 274uJ)."""
        result = c2m_model.calc_switching_energy(default_params)
        assert 150e-6 < result.e_on < 600e-6

    def test_e_on_much_greater_than_e_off(
        self, c2m_model: ChristenBielaModel, default_params: HalfBridgeParams
    ) -> None:
        """E_on / E_off > 5 for SiC MOSFETs (paper shows ~20x)."""
        result = c2m_model.calc_switching_energy(default_params)
        assert result.e_on / result.e_off > 5

    def test_energy_increases_with_voltage(
        self, c2m_model: ChristenBielaModel
    ) -> None:
        """E_total increases with DC bus voltage."""
        params_400v = HalfBridgeParams(
            v_0=400, i_0=20, v_g_on=20, v_g_off=-5, r_g=7.1
        )
        params_800v = HalfBridgeParams(
            v_0=800, i_0=20, v_g_on=20, v_g_off=-5, r_g=7.1
        )
        e_400 = c2m_model.calc_switching_energy(params_400v).e_total
        e_800 = c2m_model.calc_switching_energy(params_800v).e_total
        assert e_800 > e_400

    def test_energy_positive_nonzero_current(
        self, c2m_model: ChristenBielaModel
    ) -> None:
        """E_total > 0 for nonzero current."""
        params_10a = HalfBridgeParams(
            v_0=600, i_0=10, v_g_on=20, v_g_off=-5, r_g=7.1
        )
        params_30a = HalfBridgeParams(
            v_0=600, i_0=30, v_g_on=20, v_g_off=-5, r_g=7.1
        )
        e_10 = c2m_model.calc_switching_energy(params_10a).e_total
        e_30 = c2m_model.calc_switching_energy(params_30a).e_total
        # Both should be positive
        assert e_10 > 0
        assert e_30 > 0

    def test_zero_current(self, c2m_model: ChristenBielaModel) -> None:
        """E_on = 0, E_off = 0 for I_0 = 0."""
        params = HalfBridgeParams(v_0=600, i_0=0, v_g_on=20, v_g_off=-5, r_g=7.1)
        result = c2m_model.calc_switching_energy(params)
        assert result.e_on == 0.0
        assert result.e_off == 0.0

    def test_zvs_boundary(
        self, c2m_model: ChristenBielaModel, default_params: HalfBridgeParams
    ) -> None:
        """I_0_zvs > 0 and reasonable."""
        result = c2m_model.calc_switching_energy(default_params)
        assert 0 < result.i_0_zvs < 20

    def test_detailed_result(
        self, c2m_model: ChristenBielaModel, default_params: HalfBridgeParams
    ) -> None:
        """Detailed result returns SwitchingEnergyResult."""
        result = c2m_model.calc_switching_energy(default_params)
        assert isinstance(result, SwitchingEnergyResult)
        assert result.t_rv > 0
        assert result.t_fi >= 0
        assert result.t_ri > 0
        assert result.t_fv > 0
        assert result.i_oss_off > 0
        assert result.i_oss_on <= 0  # Zero or negative for turn-on

    def test_switching_loss_curve(self, c2m_model: ChristenBielaModel) -> None:
        """Curve returns arrays of correct length."""
        currents = np.linspace(5, 50, 10)
        e_on, e_off, e_total = c2m_model.calc_switching_loss_curve(
            v_0=600, currents=currents, v_g_on=20, v_g_off=-5, r_g=7.1
        )
        assert len(e_on) == 10
        assert len(e_off) == 10
        assert len(e_total) == 10
        # All values should be non-negative
        assert all(e_on >= 0)
        assert all(e_off >= 0)
        assert all(e_total >= 0)

    def test_no_reverse_recovery(self) -> None:
        """Model without rr_params works and has smaller E_on."""
        model_no_rr = ChristenBielaModel(
            c_gs=1080e-12,
            c_ds=130e-12,
            c_gd=14.5e-12,
            q_oss=86.56e-9,
            gm_params=TransconductanceParams(k_1=0.1319, k_2=-0.076, x=3.80, v_th=4.5),
            rr_params=None,
        )
        params = HalfBridgeParams(v_0=600, i_0=20, v_g_on=20, v_g_off=-5, r_g=7.1)
        result = model_no_rr.calc_switching_energy(params)
        assert result.e_on > 0
        assert result.e_rr_s2 == 0.0


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
