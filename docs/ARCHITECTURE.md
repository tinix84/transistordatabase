# Architecture Reference

*Auto-generated from codebase structure on 2026-02-17*

This document provides an overview of the transistordatabase architecture based on the actual code structure.

## Core Modules


### Domain Models (`core/models.py`)

**Classes:**

- `TransistorMetadata` — Core metadata for a transistor.
- `ElectricalRatings` — Electrical ratings and limits.
- `ThermalProperties` — Thermal properties of the transistor.
- `FosterThermalModel` — Foster thermal RC network model for transient thermal behavior.
- `GateChargeCurve` — Gate charge characteristics of a switch.
- `SOA` — Safe Operating Area characteristics.
- `VoltageDependentCapacitance` — Voltage-dependent capacitance data (C_oss, C_iss, C_rss).
- `EffectiveOutputCapacitance` — Energy-related or time-related effective output capacitance.
- `TemperatureDependResistance` — Temperature-dependent on-resistance curve.
- `RawMeasurementData` — RAW measurement data, e.g. from double pulse test.
- `LinearizedModel` — Linearized Switch/Diode model at a specific operating point.
- `ChannelCharacteristics` — V-I characteristics for a channel at specific conditions.
- `SwitchingLossData` — Switching loss characteristics.
- `ITransistorComponent` — Interface for transistor components (Switch/Diode).
- `Switch` — Switch component with all switching characteristics.
- `Diode` — Diode component with reverse characteristics.
- `Transistor` — Main transistor aggregate containing all components and metadata.


### Service Interfaces (`core/services.py`)

**Classes:**

- `ITransistorLoader` — Interface for loading transistor data from various sources.
- `ICalculationService` — Interface for transistor calculations and analysis.
- `IExportService` — Interface for exporting transistor data to various formats.
- `IValidationService` — Interface for validating transistor data.
- `IPlottingService` — Interface for plotting transistor characteristics.
- `IComparisonService` — Interface for comparing multiple transistors.
- `TransistorRepository` — Repository interface for transistor data persistence.
- `TransistorService` — High-level service for transistor operations.


### Data Persistence (`core/repository.py`)

**Classes:**

- `JsonTransistorRepository` — File-based repository using JSON storage.
- `JsonTransistorLoader` — JSON-based transistor loader.
- `TransistorFactory` — Factory for creating transistor instances.

**Functions:**

- `_load_data_file()` — Load lines from a data file in the transistordatabase/data/ directory.
- `_convert_energy_arrays()` — Convert graph arrays to numpy for a list of switching energy dicts.
- `_convert_arrays_to_numpy()` — Convert JSON list arrays to numpy arrays in-place.
- `_json_dict_to_legacy_transistor()` — Convert a raw JSON dict to a legacy Transistor object.


### Legacy Bridge (`core/adapters.py`)

**Functions:**

- `_np()` — Convert a value to a numpy array, or return None.
- `_list_or_none()` — Convert numpy array to list, pass through lists, return None for None.
- `_convert_foster_legacy_to_core()` — Convert a legacy FosterThermalModel to a core FosterThermalModel.
- `_convert_channel_legacy_to_core()` — Convert a legacy ChannelData to a core ChannelCharacteristics.
- `_convert_switching_legacy_to_core()` — Convert a legacy SwitchEnergyData to a core SwitchingLossData.
- `_convert_linearized_legacy_to_core()` — Convert a legacy LinearizedModel to a core LinearizedModel.
- `_convert_gate_charge_legacy_to_core()` — Convert a legacy GateChargeCurve to a core GateChargeCurve.
- `_convert_soa_legacy_to_core()` — Convert a legacy SOA to a core SOA.
- `_convert_temp_resist_legacy_to_core()` — Convert a legacy TemperatureDependResistance to core.
- `_convert_raw_meas_legacy_to_core()` — Convert a legacy RawMeasurementData to core.
- `_convert_vdc_legacy_to_core()` — Convert a legacy VoltageDependentCapacitance to core.
- `_convert_eoc_legacy_to_core()` — Convert a legacy EffectiveOutputCapacitance to core.
- `legacy_to_core()` — Convert a legacy ``transistor.Transistor`` to a ``core.models.Transistor``.
- `_convert_foster_core_to_dict()` — Convert a core FosterThermalModel to a dict for legacy construction.
- `_convert_channel_core_to_dict()` — Convert a core ChannelCharacteristics to dict for legacy ChannelData.
- `_convert_switching_core_to_dict()` — Convert a core SwitchingLossData to dict for legacy SwitchEnergyData.
- `_convert_linearized_core_to_dict()` — Convert a core LinearizedModel to dict for legacy construction.
- `_convert_gate_charge_core_to_dict()` — Convert a core GateChargeCurve to dict for legacy construction.
- `_convert_soa_core_to_dict()` — Convert a core SOA to dict for legacy construction.
- `_convert_temp_resist_core_to_dict()` — Convert a core TemperatureDependResistance to dict for legacy.
- `_convert_vdc_core_to_dict()` — Convert a core VoltageDependentCapacitance to dict for legacy.
- `_convert_eoc_core_to_dict()` — Convert a core EffectiveOutputCapacitance to dict for legacy.
- `core_to_legacy_dicts()` — Convert a core Transistor to the 3 dicts needed by the legacy constructor.


### Backend Services (`backend/concrete_services.py`)

**Service Implementations:**

- `PlottingService` — Matplotlib-based plotting service returning structured plot data.
- `CalculationService` — Transistor calculation and analysis service.
- `ExportService` — Service for exporting transistor data to various formats.
- `ComparisonService` — Service for comparing transistors with advanced plot generation.
- `ValidationService` — Service for validating transistor data.
- `ConcreteServiceFactory` — Factory for creating concrete service implementations.


### Analytical Models (`analytical_models.py`)

**Models:**

- `GateChargeModelParams` — Parameters for gate charge switching time model.
- `GateChargeModel` — Gate charge based switching time estimation model.
- `IgbtModelParams` — Parameters for IGBT turn-off tail current model.
- `IgbtModel` — IGBT-specific model accounting for minority carrier tail current.
- `HalfBridgeParams` — Operating conditions and circuit parameters for half-bridge switching.
- `TransconductanceParams` — Transconductance model parameters from transfer characteristic fit.
- `ReverseRecoveryParams` — Body diode reverse recovery time constants.
- `SwitchingEnergyResult` — Detailed switching energy breakdown from Christen-Biela model.
- `ChristenBielaModel` — Full Christen-Biela analytical switching loss model for half-bridge MOSFETs.


## Module Dependencies

```
transistordatabase/
├── core/                  # Domain layer (models, interfaces)
│   ├── models.py         # Transistor, Switch, Diode entities
│   ├── services.py       # ABC interfaces for all services
│   ├── repository.py     # JSON persistence
│   └── adapters.py       # Legacy ↔ core converters
├── backend/              # Service implementations
│   └── concrete_services.py
├── frontend/             # UI layer
│   ├── interfaces.py     # UI widget ABCs
│   └── pyqt5_impl.py     # PyQt5 implementations
├── gui/                  # PyQt5 GUI (mixin-based)
│   └── mixins/           # 8 mixin modules
├── gui_web/              # Vue 3 + FastAPI web interface
│   ├── backend/          # FastAPI REST API
│   └── src/              # Vue 3 frontend
└── analytical_models.py  # Switching loss models
```

## Key Patterns

### Repository Pattern
`JsonTransistorRepository` provides CRUD operations for transistor data persistence.

### Adapter Pattern
`legacy_to_core()` and `core_to_legacy_dicts()` bridge legacy and core models.

### Service Layer
All business logic is behind ABC interfaces in `core/services.py`, implemented in `backend/concrete_services.py`.

### Mixin Architecture (GUI)
PyQt5 MainWindow uses multiple inheritance from 8 mixins for modularity.

---

*This file is auto-generated by the pre-commit hook. Manual edits will be overwritten.*
