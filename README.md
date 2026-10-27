<!-- Last modified: 2026-10-27 -->
<!-- Version: 1.0.1 -->

# Transistor Database

[![CI](https://github.com/upb-lea/transistordatabase/actions/workflows/python-package.yml/badge.svg)](https://github.com/upb-lea/transistordatabase/actions)
[![PyPI version](https://badge.fury.io/py/transistordatabase.svg)](https://pypi.org/project/transistordatabase/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)

The Transistor Database (TDB) is a tool developed by LEA at the University of Paderborn that helps with choosing transistors for electronics development. Transistors can be saved in a database, making them easy to interchange between platforms. Export transistor data to various simulation software and share it with colleagues using a `.json` file.

![Why transistor database?](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/Why_transistordatabase.png)


## Functionality Overview

![Workflow](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/Workflow.png)

- Digitize transistor datasheet parameters and graphs and save to TDB
- Use data for calculations in Python (e.g. loss calculations for a boost converter)
- Export data to MATLAB for calculations
- Export transistors to GeckoCIRCUITS simulation program
- Export transistors to Simulink simulation program
- Export transistors to PLECS simulation program

> **Note:** Development status: Alpha


## Features

Three main features are implemented in the Transistor Database: a Python interface for managing the database and implementing optimization routines, a GUI for managing and visualizing transistor properties, and an online database for automatic updates.

### Python Interface

Use the transistor data in your own optimization program:

- Automatically calculate converter losses with many different transistors
- Search the database for usable transistors for your application
- Functions provided to search for closest operating point
- Functions provided to linearize transistor/diode conduction behaviour
- Functions provided to calculate the output capacitance energy from the C_oss curve

![optimization](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/documentation/workflow_wp.png)

### GUI

- Manage, store and search transistors
- Export transistor models to GeckoCIRCUITS, PLECS, MATLAB/Octave
- Compare transistors (interpolate switch loss data for new gate resistors and temperatures)

![gui_database](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/gui_database.png)
![gui_comparison](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/gui_comparison.png)
![gui_create_transistor](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/gui_create_transistor.png)

### Web Interface

The modern web interface provides:

- **Vue 3** frontend with responsive design
- **FastAPI** backend with automatic API documentation
- Real-time search and filtering
- Interactive comparison and topology calculator
- Export to PLECS, MATLAB, Simulink, GeckoCIRCUITS

Quick start:

```bash
# Terminal 1: Start backend
uvicorn transistordatabase.gui_web.backend.main:app --port 8002

# Terminal 2: Start frontend
cd transistordatabase/gui_web
npm install  # First time only
npm run dev

# Open browser: http://localhost:5174
```

### Transistor Database File Exchange

The [File Exchange repository](https://github.com/upb-lea/transistordatabase_File_Exchange) contains transistors currently added to the database. Every transistor from this repository can be automatically downloaded to your local database. You can publish your own transistors by creating a pull request, or send the `.json` file to [tdb@lea.upb.de](mailto:tdb@lea.upb.de).


## Installation

### Requirements

- Python >= 3.10

### pip

```bash
pip install transistordatabase
```

### From Source

```bash
git clone https://github.com/upb-lea/transistordatabase.git
cd transistordatabase
pip install -e .
```

### Linux (Ubuntu)

```bash
sudo apt install python3 python3-pip
pip install transistordatabase
```


## Documentation

Full documentation: [https://upb-lea.github.io/transistordatabase/](https://upb-lea.github.io/transistordatabase/)

- [Quick Start Guide](docs/QUICK_START.md) — Get running in 5 minutes
- [User Guide](docs/USER_GUIDE.md) — Complete feature documentation
- [Tutorial](docs/TUTORIAL.md) — Hands-on tutorials from first search to converter design
- [FAQ](docs/FAQ.md) — Frequently asked questions with troubleshooting tips
- [API Documentation](docs/API_DOCUMENTATION.md) — Complete REST API reference


## Usage

### Minimal Python Example

```python
import os
from transistordatabase.database_manager import DatabaseManager

# Path for json files
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdb_example")

# Create DatabaseManager instance and set it to json format
tdb_json = DatabaseManager()
tdb_json.set_operation_mode_json(path)

# Print database
tdb_json.print_tdb()

# Load a transistor from the database
transistor_loaded = tdb_json.load_transistor('CREE_C3M0016120K')
```

### Transistor Object Hierarchy

```
Transistor
├── Metadata
├── Switch
│   ├── Switch Metadata
│   ├── Channel Data
│   └── Switching Data
└── Diode
    ├── Diode Metadata
    ├── Channel Data
    └── Switching Data
```

### Reading Curves from Datasheets

Use [WebPlotDigitizer](https://apps.automeris.io/wpd/) to digitize datasheet curves. Channel data for switch and diode always needs to be positive.

![diode channel data negative](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/Diode_channel_data_negative.png)

Use the [template](/template_example/template_example.py) to generate a new transistor object and store it to the database.

Reference lists:
- [List of manufacturers](https://github.com/upb-lea/transistordatabase/blob/main/transistordatabase/module_manufacturers.txt)
- [List of housing types](https://github.com/upb-lea/transistordatabase/blob/main/transistordatabase/housing_types.txt)
- [Fuji housing overview](https://www.fujielectric.com/products/semiconductor/model/igbt/2pack.html)

### Working Point (wp) Calculations

**Full-automated** — quickstart method fills wp with closest operating point:
```python
transistor_loaded.quickstart_wp()
```

**Half-automated** — provide a working point and let the algorithm find the closest match:
```python
transistor_loaded.update_wp(125, 15, 50)
```

**Manual** — look up all operating points explicitly:
```python
transistor_loaded.wp.e_oss = transistor_loaded.calc_v_eoss()
transistor_loaded.wp.q_oss = transistor_loaded.calc_v_qoss()

# Switch: linearize channel and search for loss curves
transistor_loaded.wp.switch_v_channel, transistor_loaded.wp.switch_r_channel = \
    transistor_loaded.calc_lin_channel(25, 15, 150, 'switch')
transistor_loaded.wp.e_on = transistor_loaded.get_object_i_e('e_on', 25, 15, 600, 2.5).graph_i_e
transistor_loaded.wp.e_off = transistor_loaded.get_object_i_e('e_off', 25, -4, 600, 2.5).graph_i_e

# Diode: linearize channel and search for loss curves
transistor_loaded.wp.diode_v_channel, transistor_loaded.wp.diode_r_channel = \
    transistor_loaded.calc_lin_channel(25, -4, 150, 'diode')
```

### Parallel Transistors

```python
# Parallel 3 transistors
parallel_transistorobject = tdb_json.parallel_transistors(transistor_loaded, 3)
```


## Export Transistor Objects

### Virtual Datasheet

```python
# Linux: export datasheet as HTML
html_str = transistor_loaded.export_datasheet(build_collection=True)
with open(f"{transistor_loaded.name}.html", "w") as f:
    f.write(html_str)
```

![Generated virtual datasheet example](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/Virtual_Datasheet.png)

### Export to GeckoCIRCUITS

GeckoCIRCUITS is an open source multi-platform schematic simulator ([download](http://gecko-simulations.com/GeckoCIRCUITS/GeckoCIRCUITS.zip)).

```python
transistor_loaded.export_geckocircuits(True, 600, 15, -4, 2.5, 2.5)
```

![GeckoExporter usage example](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/Example_Gecko_Exporter.png)

> **Hint:** It is also possible to control GeckoCIRCUITS from Python to sweep transistors. Linux users may want to use [this version](https://github.com/tinix84/gecko/releases/tag/v1.1) of GeckoCIRCUITS (port to OpenJDK).

### Export to PLECS

```python
transistor_loaded.export_plecs()

# With custom gate voltages: [Vg_on, Vg_off, Vd_on, Vd_off]
transistor_loaded.export_plecs([15, -15, 15, 0])
```

![PLECS thermal exporter usage example](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/PLECS_thermal_editor.png)

### Export to Simulink

```python
transistor_loaded = db.load_transistor('Infineon_FF200R12KE3')
transistor_loaded.export_simulink_loss_model()
```

![Simulink exporter usage example](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/Example_Simulink_Exporter.png)

### Export to MATLAB/Octave

```python
transistor_loaded = db.load_transistor('Fuji_2MBI100XAA120-50')
transistor_loaded.export_matlab()
```

![Matlab .mat exporter usage example](https://raw.githubusercontent.com/upb-lea/transistordatabase/main/sphinx/images/Matlab.png)


## Architecture (v1.0.1)

```
transistordatabase/
├── core/              — Domain models, ABC interfaces, repository, adapters
├── backend/           — Concrete service implementations (Plotting, Calculation, Export)
├── frontend/          — UI widget ABCs and PyQt5 implementations
├── gui/               — PyQt5 desktop GUI (mixin-based architecture)
├── gui_web/           — Vue 3 + FastAPI web interface
├── topologies/        — Power converter topology analyzers (Buck, Boost, Buck-Boost)
└── utils/             — LTspice DPT netlist generation and analysis
```

See [Architecture Documentation](https://upb-lea.github.io/transistordatabase/) for full details.


## Development

### Running Tests

```bash
pytest tests/ -q                   # Run all tests (~283 tests)
pytest tests/test_core_services.py # Core backend services
pytest tests/test_rest_api.py      # FastAPI endpoints
pytest tests/test_gui_playwright.py -v --headed  # E2E browser tests
```

### Linting

```bash
ruff check transistordatabase/
```

### Contributing

Pull requests are welcome. For major changes, please open an issue first. See [Contributing.rst](Contributing.rst) for details.

Bug reports: [GitHub Issues](https://github.com/upb-lea/transistordatabase/issues)


## Changelog

See [CHANGELOG.md](CHANGELOG.md).


## License

Licensed under [GPLv3](https://choosealicense.com/licenses/gpl-3.0/).


## About

This project started in 2020 as a side project and was initially written in MATLAB. In January 2021 a very early alpha status was reached. First pip package was provided in May 2021. First GUI was provided in June 2022. The project is developed and maintained by [LEA, University of Paderborn](https://github.com/upb-lea).
