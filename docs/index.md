# Transistor Database

Welcome to the Transistor Database (TDB) documentation!

## Overview

Transistor Database is a Python package for managing power semiconductor transistor data (MOSFETs, SiC-MOSFETs, IGBTs, GaN) and exporting to simulation tools (GeckoCIRCUITS, PLECS, Simulink, MATLAB, LTSpice).

Developed by LEA (Laboratory for Power Electronics and Electrical Drives) at the University of Paderborn.

## Key Features

- **Comprehensive Data Model**: Store complete transistor characteristics including channel data, switching losses, thermal models, and more
- **Analytical Models**: Built-in switching loss models (Christen-Biela, gate charge, IGBT) for quick performance estimation
- **Multiple Export Formats**: Export to PLECS, GeckoCIRCUITS, MATLAB, LTSpice, and other simulation tools
- **Web & Desktop GUI**: Both PyQt5 desktop application and Vue 3 web interface
- **Database Management**: JSON-based repository with search, filtering, and comparison tools
- **Clean Architecture**: Core domain models with pluggable service implementations

## Quick Links

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [API Reference](api/core/models.md)
- [GitHub Repository](https://github.com/tinix84/transistordatabase)

## Getting Started

```bash
# Install from PyPI
pip install transistordatabase

# Or install from source
git clone https://github.com/tinix84/transistordatabase.git
cd transistordatabase
pip install -e .
```

```python
# Quick example
from transistordatabase import DatabaseManager

db = DatabaseManager()
transistor = db.load_transistor("C2M0080120D")
print(f"Loaded: {transistor.metadata.name}")
```

## Project Status

Current version: **1.0.0** (Stable)

See [Changelog](CHANGELOG.md) for release history.

## License

This project is licensed under the GPL-3.0 License.

## Citation

If you use this software in your research, please cite:

```
@software{transistordatabase,
  title = {Transistor Database},
  author = {LEA, University of Paderborn},
  url = {https://github.com/tinix84/transistordatabase},
  year = {2024}
}
```
