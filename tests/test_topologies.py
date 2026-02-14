"""Tests for simple PWM converter topologies (Buck, Boost, Buck-Boost).

Basic smoke tests to verify topology-specific duty cycle and voltage calculations.
"""
from __future__ import annotations

import numpy as np
import pytest

from transistordatabase.topologies import BuckConverter, BoostConverter, BuckBoostConverter


class TestBuckConverter:
    """Tests for Buck converter topology."""

    def test_duty_cycle_ccm_basic(self):
        """Test Buck CCM duty cycle calculation with ideal switches."""
        v_in = 12.0
        v_out = 5.0
        v_ch1 = 0.1  # switch voltage drop
        v_ch2 = 0.3  # diode voltage drop

        duty = BuckConverter.duty_cycle_ccm_mesh(v_in, v_out, v_ch1, v_ch2)

        # Expected: (V_out + V_ch2) / (V_in - V_ch1 + V_ch2)
        expected = (5.0 + 0.3) / (12.0 - 0.1 + 0.3)
        assert np.isclose(duty, expected), f"Expected {expected}, got {duty}"
        assert 0 < duty < 1, "Duty cycle should be between 0 and 1"

    def test_duty_cycle_ccm_mesh_array(self):
        """Test Buck CCM duty cycle with numpy arrays."""
        v_in = np.array([12.0, 15.0, 20.0])
        v_out = np.array([5.0, 5.0, 5.0])
        v_ch1 = np.array([0.1, 0.1, 0.1])
        v_ch2 = np.array([0.3, 0.3, 0.3])

        duty = BuckConverter.duty_cycle_ccm_mesh(v_in, v_out, v_ch1, v_ch2)

        assert duty.shape == v_in.shape
        assert np.all(duty > 0) and np.all(duty < 1)
        # Higher V_in should give lower duty cycle for same V_out
        assert duty[0] > duty[1] > duty[2]

    def test_inductor_voltage(self):
        """Test Buck inductor voltage calculation."""
        v_in = 12.0
        v_out = 5.0
        v_ch1 = 0.2

        v_l = BuckConverter.inductor_voltage(v_in, v_out, v_ch1)

        # Expected: V_in - V_out - V_ch1
        expected = 12.0 - 5.0 - 0.2
        assert np.isclose(v_l, expected)
        assert v_l > 0, "Inductor voltage should be positive during on-phase"

    def test_blocking_voltage(self):
        """Test Buck blocking voltage calculation."""
        v_in = 48.0
        v_out = 12.0

        v_block = BuckConverter.blocking_voltage(v_in, v_out)

        assert v_block == v_in, "Buck blocking voltage should equal input voltage"


class TestBoostConverter:
    """Tests for Boost converter topology."""

    def test_duty_cycle_ccm_basic(self):
        """Test Boost CCM duty cycle calculation with ideal switches."""
        v_in = 12.0
        v_out = 24.0
        v_ch1 = 0.1
        v_ch2 = 0.3

        duty = BoostConverter.duty_cycle_ccm_mesh(v_in, v_out, v_ch1, v_ch2)

        # Expected: (V_out - V_in + V_ch2) / (V_out - V_ch1 + V_ch2)
        expected = (24.0 - 12.0 + 0.3) / (24.0 - 0.1 + 0.3)
        assert np.isclose(duty, expected), f"Expected {expected}, got {duty}"
        assert 0 < duty < 1, "Duty cycle should be between 0 and 1"

    def test_duty_cycle_ccm_mesh_array(self):
        """Test Boost CCM duty cycle with numpy arrays."""
        v_in = np.array([12.0, 12.0, 12.0])
        v_out = np.array([24.0, 30.0, 48.0])
        v_ch1 = np.array([0.1, 0.1, 0.1])
        v_ch2 = np.array([0.3, 0.3, 0.3])

        duty = BoostConverter.duty_cycle_ccm_mesh(v_in, v_out, v_ch1, v_ch2)

        assert duty.shape == v_in.shape
        assert np.all(duty > 0) and np.all(duty < 1)
        # Higher V_out should give higher duty cycle for same V_in
        assert duty[0] < duty[1] < duty[2]

    def test_inductor_voltage(self):
        """Test Boost inductor voltage calculation."""
        v_in = 12.0
        v_out = 24.0
        v_ch1 = 0.2

        v_l = BoostConverter.inductor_voltage(v_in, v_out, v_ch1)

        # Expected: V_in - V_ch1
        expected = 12.0 - 0.2
        assert np.isclose(v_l, expected)
        assert v_l > 0, "Inductor voltage should be positive during on-phase"

    def test_blocking_voltage(self):
        """Test Boost blocking voltage calculation."""
        v_in = 12.0
        v_out = 48.0

        v_block = BoostConverter.blocking_voltage(v_in, v_out)

        assert v_block == v_out, "Boost blocking voltage should equal output voltage"


class TestBuckBoostConverter:
    """Tests for Buck-Boost converter topology."""

    def test_duty_cycle_ccm_basic(self):
        """Test Buck-Boost CCM duty cycle calculation with ideal switches."""
        v_in = 12.0
        v_out = 15.0  # can be higher or lower than V_in
        v_ch1 = 0.1
        v_ch2 = 0.3

        duty = BuckBoostConverter.duty_cycle_ccm_mesh(v_in, v_out, v_ch1, v_ch2)

        # Expected: (V_out + V_ch2) / (V_in + V_out - V_ch1 + V_ch2)
        expected = (15.0 + 0.3) / (12.0 + 15.0 - 0.1 + 0.3)
        assert np.isclose(duty, expected), f"Expected {expected}, got {duty}"
        assert 0 < duty < 1, "Duty cycle should be between 0 and 1"

    def test_duty_cycle_ccm_mesh_array(self):
        """Test Buck-Boost CCM duty cycle with numpy arrays."""
        v_in = np.array([12.0, 12.0, 12.0])
        v_out = np.array([8.0, 12.0, 20.0])  # lower, equal, higher than V_in
        v_ch1 = np.array([0.1, 0.1, 0.1])
        v_ch2 = np.array([0.3, 0.3, 0.3])

        duty = BuckBoostConverter.duty_cycle_ccm_mesh(v_in, v_out, v_ch1, v_ch2)

        assert duty.shape == v_in.shape
        assert np.all(duty > 0) and np.all(duty < 1)
        # Higher V_out should give higher duty cycle
        assert duty[0] < duty[1] < duty[2]

    def test_inductor_voltage(self):
        """Test Buck-Boost inductor voltage calculation."""
        v_in = 12.0
        v_out = 15.0
        v_ch1 = 0.2

        v_l = BuckBoostConverter.inductor_voltage(v_in, v_out, v_ch1)

        # Expected: V_in - V_ch1
        expected = 12.0 - 0.2
        assert np.isclose(v_l, expected)
        assert v_l > 0, "Inductor voltage should be positive during on-phase"

    def test_blocking_voltage(self):
        """Test Buck-Boost blocking voltage calculation."""
        v_in = 12.0
        v_out = 15.0

        v_block = BuckBoostConverter.blocking_voltage(v_in, v_out)

        # Expected: V_in + V_out
        expected = 12.0 + 15.0
        assert v_block == expected, "Buck-Boost blocking voltage should be V_in + V_out"


class TestConverterComparison:
    """Comparative tests across all three topologies."""

    def test_all_converters_exist(self):
        """Verify all three converter classes are importable."""
        assert BuckConverter is not None
        assert BoostConverter is not None
        assert BuckBoostConverter is not None

    def test_duty_cycle_range(self):
        """Verify all converters produce valid duty cycles (0 < D < 1) for typical values."""
        v_in = 12.0
        v_out_buck = 5.0
        v_out_boost = 24.0
        v_out_buckboost = 15.0
        v_ch1 = 0.1
        v_ch2 = 0.3

        d_buck = BuckConverter.duty_cycle_ccm_mesh(v_in, v_out_buck, v_ch1, v_ch2)
        d_boost = BoostConverter.duty_cycle_ccm_mesh(v_in, v_out_boost, v_ch1, v_ch2)
        d_buckboost = BuckBoostConverter.duty_cycle_ccm_mesh(v_in, v_out_buckboost, v_ch1, v_ch2)

        assert 0 < d_buck < 1, f"Buck duty cycle {d_buck} out of range"
        assert 0 < d_boost < 1, f"Boost duty cycle {d_boost} out of range"
        assert 0 < d_buckboost < 1, f"Buck-Boost duty cycle {d_buckboost} out of range"

    def test_blocking_voltage_relationships(self):
        """Verify blocking voltage relationships across topologies."""
        v_in = 12.0
        v_out = 24.0

        v_block_buck = BuckConverter.blocking_voltage(v_in, v_out)
        v_block_boost = BoostConverter.blocking_voltage(v_in, v_out)
        v_block_buckboost = BuckBoostConverter.blocking_voltage(v_in, v_out)

        # Buck: V_in, Boost: V_out, Buck-Boost: V_in + V_out
        assert v_block_buck == v_in
        assert v_block_boost == v_out
        assert v_block_buckboost == v_in + v_out
        assert v_block_buckboost > v_block_boost > v_block_buck
