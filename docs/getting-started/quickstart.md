# Quick Start Guide

## Loading a Transistor

```python
from transistordatabase import DatabaseManager

# Initialize database
db = DatabaseManager()

# Load a transistor
transistor = db.load_transistor("C2M0080120D")

# Access properties
print(f"Name: {transistor.metadata.name}")
print(f"Type: {transistor.metadata.type}")
print(f"Manufacturer: {transistor.metadata.manufacturer}")
print(f"Max Voltage: {transistor.electrical_ratings.v_abs_max} V")
print(f"Max Current: {transistor.electrical_ratings.i_abs_max} A")
```

## Using Analytical Models

```python
from transistordatabase.analytical_models import ChristenBielaModel, HalfBridgeParams

# Create model from transistor data
model = ChristenBielaModel.from_transistor(transistor)

# Calculate switching energy
params = HalfBridgeParams(
    v_0=600,      # DC bus voltage (V)
    i_0=20,       # Load current (A)
    v_g_on=20,    # Gate-on voltage (V)
    v_g_off=-5,   # Gate-off voltage (V)
    r_g=10        # Gate resistance (Ω)
)

result = model.calc_switching_energy(params)
print(f"Turn-on energy: {result.e_on*1e6:.2f} µJ")
print(f"Turn-off energy: {result.e_off*1e6:.2f} µJ")
```

## Exporting to Simulation Tools

```python
from transistordatabase.backend.concrete_services import ConcreteServiceFactory

# Create export service
factory = ConcreteServiceFactory()
exporter = factory.create_export_service()

# Export to PLECS
from pathlib import Path
exporter.export_to_plecs(transistor, Path("transistor.xml"))

# Export to GeckoCIRCUITS
export_params = {"working_directory": "output/"}
exporter.export_to_gecko_circuits(transistor, export_params)
```

## Next Steps

- [Examples](examples.md) - More detailed examples
- [User Guide](../guide/overview.md) - Comprehensive guide
- [API Reference](../api/core/models.md) - Full API documentation
