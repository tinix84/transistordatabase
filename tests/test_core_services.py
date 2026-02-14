"""Tests for backend concrete service implementations.

Exercises the real service implementations from
``transistordatabase.backend.concrete_services``.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from transistordatabase.backend.concrete_services import (
    CalculationService,
    ComparisonService,
    ConcreteServiceFactory,
    ExportService,
    PlottingService,
    ValidationService,
)
from transistordatabase.core.models import (
    ChannelCharacteristics,
    Diode,
    ElectricalRatings,
    FosterThermalModel,
    GateChargeCurve,
    SOA,
    Switch,
    SwitchingLossData,
    ThermalProperties,
    Transistor,
    TransistorMetadata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def transistor() -> Transistor:
    """Build a realistic core Transistor for testing services."""
    t = Transistor(
        metadata=TransistorMetadata(
            name="TEST_SVC", type="IGBT", author="test",
            manufacturer="Infineon", housing_type="TO247",
        ),
        electrical=ElectricalRatings(
            v_abs_max=1200.0, i_abs_max=200.0, i_cont=100.0, t_j_max=175.0,
        ),
        thermal=ThermalProperties(
            housing_area=0.002, cooling_area=0.002,
            r_th_cs=0.5, r_th_switch_cs=0.3, r_th_diode_cs=0.4,
        ),
    )

    # Switch channel data
    graph_v_i = np.array([
        [0, 0.5, 1.0, 1.5, 2.0],
        [0, 5, 20, 50, 100],
    ])
    t.switch.channel_data = [
        ChannelCharacteristics(t_j=25, graph_v_i=graph_v_i, v_g=15),
        ChannelCharacteristics(t_j=150, graph_v_i=graph_v_i * 1.1, v_g=15),
    ]

    # Switch e_on data
    graph_i_e = np.array([
        [10, 50, 100, 150, 200],
        [0.001, 0.005, 0.012, 0.02, 0.035],
    ])
    t.switch.e_on_data = [
        SwitchingLossData(
            dataset_type="graph_i_e", t_j=25, v_supply=600, v_g=15,
            r_g=10, graph_i_e=graph_i_e,
        ),
    ]
    t.switch.e_off_data = [
        SwitchingLossData(
            dataset_type="graph_i_e", t_j=25, v_supply=600, v_g=15,
            r_g=10, graph_i_e=graph_i_e * 0.8,
        ),
    ]

    # Foster thermal model
    t.switch.thermal_foster = FosterThermalModel(
        r_th_total=0.5,
        r_th_vector=[0.1, 0.15, 0.25],
        c_th_vector=[0.001, 0.01, 0.1],
        c_th_total=0.111,
        tau_vector=[0.0001, 0.0015, 0.025],
        tau_total=None,
        graph_t_rthjc=np.array([
            [0.001, 0.01, 0.1, 1.0],
            [0.05, 0.2, 0.4, 0.5],
        ]),
    )

    # Gate charge
    t.switch.gate_charge_curves = [
        GateChargeCurve(
            v_supply=600, t_j=25, i_channel=100, i_g=1.0,
            graph_q_v=np.array([[0, 10e-9, 30e-9, 60e-9], [0, 5, 10, 15]]),
        ),
    ]

    # SOA
    t.switch.soa = [
        SOA(
            t_c=25, time_pulse=0.01,
            graph_i_v=np.array([[0, 200, 600, 1200], [200, 200, 100, 10]]),
        ),
    ]

    # Diode channel
    t.diode.channel_data = [
        ChannelCharacteristics(t_j=25, graph_v_i=graph_v_i, v_g=0),
    ]
    t.diode.e_rr_data = [
        SwitchingLossData(
            dataset_type="graph_i_e", t_j=25, v_supply=600, v_g=0,
            r_g=10, graph_i_e=graph_i_e * 0.5,
        ),
    ]

    return t


@pytest.fixture()
def empty_transistor() -> Transistor:
    """Build a minimal transistor with no curve data."""
    return Transistor(
        metadata=TransistorMetadata(
            name="EMPTY", type="MOSFET", author="test",
            manufacturer="Cree", housing_type="TO247",
        ),
        electrical=ElectricalRatings(
            v_abs_max=650.0, i_abs_max=50.0, i_cont=30.0, t_j_max=175.0,
        ),
        thermal=ThermalProperties(housing_area=0.001, cooling_area=0.001),
    )


# ---------------------------------------------------------------------------
# PlottingService
# ---------------------------------------------------------------------------

class TestPlottingService:
    """Test PlottingService with real transistor data."""

    def setup_method(self):
        self.svc = PlottingService()

    def test_plot_channel_switch(self, transistor):
        """Channel plot returns curves for switch component."""
        result = self.svc.plot_channel_characteristics(transistor, "switch")
        assert "curves" in result
        assert len(result["curves"]) == 2  # 2 temperatures
        assert "x_data" in result["curves"][0]
        assert "y_data" in result["curves"][0]

    def test_plot_channel_diode(self, transistor):
        """Channel plot returns curves for diode component."""
        result = self.svc.plot_channel_characteristics(transistor, "diode")
        assert len(result["curves"]) == 1

    def test_plot_channel_empty(self, empty_transistor):
        """Channel plot returns error for empty transistor."""
        result = self.svc.plot_channel_characteristics(empty_transistor, "switch")
        assert "error" in result

    def test_plot_switching_losses(self, transistor):
        """Switching loss plot returns curves."""
        result = self.svc.plot_switching_losses(transistor, "e_on", "i_e")
        assert "curves" in result
        assert len(result["curves"]) >= 1

    def test_plot_switching_e_rr(self, transistor):
        """Reverse recovery loss plot returns curves."""
        result = self.svc.plot_switching_losses(transistor, "e_rr", "i_e")
        assert "curves" in result

    def test_plot_soa(self, transistor):
        """SOA plot returns curves."""
        result = self.svc.plot_safe_operating_area(transistor)
        assert "curves" in result
        assert len(result["curves"]) >= 1

    def test_plot_soa_empty(self, empty_transistor):
        """SOA plot returns error for empty transistor."""
        result = self.svc.plot_safe_operating_area(empty_transistor)
        assert "error" in result

    def test_plot_thermal(self, transistor):
        """Thermal impedance plot returns data."""
        result = self.svc.plot_thermal_impedance(transistor)
        assert "curves" in result
        assert len(result["curves"]) == 1

    def test_plot_gate_charge(self, transistor):
        """Gate charge plot returns curves."""
        result = self.svc.plot_gate_charge(transistor)
        assert "curves" in result
        assert len(result["curves"]) >= 1


# ---------------------------------------------------------------------------
# CalculationService
# ---------------------------------------------------------------------------

class TestCalculationService:
    """Test CalculationService calculations."""

    def setup_method(self):
        self.svc = CalculationService()

    def test_interpolate_switching_losses(self, transistor):
        """Interpolation returns reasonable value."""
        result = self.svc.interpolate_switching_losses(
            transistor.switch.e_on_data,
            {"t_j": 25.0, "i_channel": 100.0},
        )
        assert result > 0
        assert result == pytest.approx(0.012, abs=0.001)

    def test_interpolate_no_data(self):
        """Interpolation with no data returns 0."""
        result = self.svc.interpolate_switching_losses([], {"t_j": 25.0})
        assert result == 0.0

    def test_calculate_thermal_impedance(self, transistor):
        """Thermal impedance calculation returns array."""
        foster = transistor.switch.thermal_foster
        time_points = np.array([0.001, 0.01, 0.1, 1.0])
        model = {
            "r_th_vector": foster.r_th_vector,
            "tau_vector": foster.tau_vector,
            "r_th_total": foster.r_th_total,
        }
        result = self.svc.calculate_thermal_impedance(model, time_points)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        # Impedance should increase with time
        assert result[-1] > result[0]

    def test_calculate_losses(self, transistor):
        """Loss calculation returns all expected keys."""
        result = self.svc.calculate_losses(
            transistor,
            {"t_j": 25.0, "v_g": 15.0, "i_channel": 100.0, "f_sw": 10000.0},
        )
        assert "switching_losses_on" in result
        assert "switching_losses_off" in result
        assert "total_losses" in result
        assert result["total_losses"] >= 0

    def test_calculate_thermal_resistance(self, transistor):
        """Thermal resistance calculation returns reasonable values."""
        result = self.svc.calculate_thermal_resistance(
            transistor, [10.0, 20.0, 15.0],
        )
        assert "r_th_jc" in result
        assert "t_j_rise" in result
        assert "t_j_avg" in result


# ---------------------------------------------------------------------------
# ValidationService
# ---------------------------------------------------------------------------

class TestValidationService:
    """Test ValidationService validation logic."""

    def setup_method(self):
        self.svc = ValidationService()

    def test_valid_transistor(self, transistor):
        """Valid transistor passes with no errors."""
        result = self.svc.validate_transistor(transistor)
        assert len(result["errors"]) == 0

    def test_invalid_voltage(self):
        """Zero voltage produces an error."""
        t = Transistor(
            metadata=TransistorMetadata(
                name="BAD", type="IGBT", author="test",
                manufacturer="", housing_type="",
            ),
            electrical=ElectricalRatings(
                v_abs_max=0, i_abs_max=0, i_cont=0, t_j_max=175.0,
            ),
            thermal=ThermalProperties(housing_area=0, cooling_area=0),
        )
        result = self.svc.validate_transistor(t)
        assert len(result["errors"]) > 0

    def test_validate_channel_data(self, transistor):
        """Channel data validation works."""
        result = self.svc.validate_channel_data(transistor.switch.channel_data)
        assert isinstance(result, list)

    def test_validate_switching_data(self, transistor):
        """Switching data validation works."""
        result = self.svc.validate_switching_data(transistor.switch.e_on_data)
        assert isinstance(result, list)

    def test_check_completeness(self, transistor):
        """Completeness check returns percentages."""
        result = self.svc.check_data_completeness(transistor)
        assert "overall" in result
        assert result["overall"] > 0


# ---------------------------------------------------------------------------
# ComparisonService
# ---------------------------------------------------------------------------

class TestComparisonService:
    """Test ComparisonService comparison logic."""

    def setup_method(self):
        self.svc = ComparisonService()

    def test_compare_two(self, transistor, empty_transistor):
        """Comparison of two transistors returns structured results."""
        result = self.svc.compare_transistors([transistor, empty_transistor])
        assert result["transistor_count"] == 2
        assert "electrical_comparison" in result

    def test_compare_too_few(self, transistor):
        """Comparison with <2 transistors returns error."""
        result = self.svc.compare_transistors([transistor])
        assert "error" in result

    def test_find_similar(self, transistor, empty_transistor):
        """find_similar_transistors returns matching transistors."""
        result = self.svc.find_similar_transistors(
            transistor, [empty_transistor], tolerance=1.0,
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ExportService
# ---------------------------------------------------------------------------

class TestExportService:
    """Test ExportService export methods."""

    def setup_method(self):
        self.svc = ExportService()

    def test_export_json(self, transistor, tmp_path):
        """JSON export creates a valid file."""
        out = tmp_path / "test.json"
        result = self.svc.export_to_json(transistor, out)
        assert result is True
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["metadata"]["name"] == "TEST_SVC"

    def test_export_csv(self, transistor, tmp_path):
        """CSV export creates a file."""
        out = tmp_path / "test.csv"
        result = self.svc.export_to_csv([transistor], out)
        assert result is True
        assert out.exists()
        content = out.read_text()
        assert "TEST_SVC" in content

    def test_export_spice(self, transistor, tmp_path):
        """SPICE export creates a file."""
        out = tmp_path / "test.spice"
        result = self.svc.export_to_spice(transistor, out)
        assert result is True
        assert out.exists()


# ---------------------------------------------------------------------------
# ConcreteServiceFactory
# ---------------------------------------------------------------------------

class TestConcreteServiceFactory:
    """Test the service factory."""

    def test_create_all(self):
        """Factory creates all service types."""
        services = ConcreteServiceFactory.create_all_services()
        assert "plotting" in services
        assert "calculation" in services
        assert "export" in services
        assert "comparison" in services
        assert "validation" in services

    def test_individual_creation(self):
        """Individual factory methods return correct types."""
        assert isinstance(ConcreteServiceFactory.create_plotting_service(), PlottingService)
        assert isinstance(ConcreteServiceFactory.create_calculation_service(), CalculationService)
        assert isinstance(ConcreteServiceFactory.create_export_service(), ExportService)
        assert isinstance(ConcreteServiceFactory.create_comparison_service(), ComparisonService)
        assert isinstance(ConcreteServiceFactory.create_validation_service(), ValidationService)
