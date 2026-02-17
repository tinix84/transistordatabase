"""
Core domain layer for the transistor database.

This package contains the core business models, services, and interfaces
that define the domain logic separated from presentation and infrastructure concerns.
"""

__version__ = "1.0.1"

from .models import (
    Transistor,
    Switch,
    Diode,
    TransistorMetadata,
    ElectricalRatings,
    ThermalProperties,
    ChannelCharacteristics,
    SwitchingLossData,
    ITransistorComponent,
    FosterThermalModel,
    GateChargeCurve,
    SOA,
    VoltageDependentCapacitance,
    EffectiveOutputCapacitance,
    TemperatureDependResistance,
    RawMeasurementData,
    LinearizedModel,
)

from .services import (
    ITransistorLoader,
    ICalculationService,
    IExportService,
    IValidationService,
    IPlottingService,
    IComparisonService,
    TransistorRepository,
    TransistorService,
)

from .repository import (
    JsonTransistorRepository,
    JsonTransistorLoader,
    TransistorFactory,
)

from .adapters import (
    legacy_to_core,
    core_to_legacy_dicts,
)

__all__ = [
    # Models
    'Transistor',
    'Switch',
    'Diode',
    'TransistorMetadata',
    'ElectricalRatings',
    'ThermalProperties',
    'ChannelCharacteristics',
    'SwitchingLossData',
    'ITransistorComponent',
    'FosterThermalModel',
    'GateChargeCurve',
    'SOA',
    'VoltageDependentCapacitance',
    'EffectiveOutputCapacitance',
    'TemperatureDependResistance',
    'RawMeasurementData',
    'LinearizedModel',
    # Services
    'ITransistorLoader',
    'ICalculationService',
    'IExportService',
    'IValidationService',
    'IPlottingService',
    'IComparisonService',
    'TransistorRepository',
    'TransistorService',
    # Repository
    'JsonTransistorRepository',
    'JsonTransistorLoader',
    'TransistorFactory',
    # Adapters
    'legacy_to_core',
    'core_to_legacy_dicts',
]

__version__ = '1.0.1'
