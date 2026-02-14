# Quick Start Guide

Get up and running with Transistor Database in 5 minutes! 🚀

## Prerequisites

- **Python 3.10+** (check: `python --version`)
- **Node.js 18+** (check: `node --version`)
- **Git** (check: `git --version`)

## Installation (2 minutes)

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/tinix84/transistordatabase.git
cd transistordatabase

# Install Python package
pip install -e .
```

### 2. Choose Your Interface

#### Option A: Web Interface (Recommended)

```bash
# Terminal 1: Start backend (port 8002)
uvicorn transistordatabase.gui_web.backend.main:app --port 8002

# Terminal 2: Start frontend
cd transistordatabase/gui_web
npm install  # First time only
npm run dev

# Open browser
# → http://localhost:5173
```

#### Option B: Desktop Interface

```bash
# Install PyQt5 dependencies
pip install PyQt5

# Launch desktop app
python -m transistordatabase.gui.gui
```

## Your First Actions (3 minutes)

### 1. Search for Transistors (30 seconds)

**Web Interface:**
1. Click **"Search Database"** tab
2. Filter by **Type**: Select "SiC-MOSFET"
3. See results instantly!

**Desktop Interface:**
1. Go to **"Search Database"** tab
2. Enable **"Type"** checkbox
3. Select "SiC-MOSFET" from dropdown
4. Click **"Search"**

### 2. View Transistor Details (30 seconds)

- Click on any transistor name in the results
- See full specifications:
  - Voltage/current ratings
  - Thermal resistance
  - Switching characteristics
  - Capacitance curves

### 3. Compare Two Transistors (1 minute)

1. Go to **"Comparison Tools"** tab
2. Select **First Transistor**: CREE_C3M0060065J
3. Select **Second Transistor**: Infineon_IPT60R028G7
4. View side-by-side comparison with charts!

### 4. Export to Simulation Tool (1 minute)

1. Go to **"Exporting Tools"** tab
2. Select a transistor
3. Choose format:
   - **PLECS** → For thermal modeling
   - **MATLAB** → For custom scripts
   - **Simulink** → For Simscape models
4. Click **"Export"**
5. Import file into your simulation tool

## Quick Commands Cheat Sheet

```bash
# Backend operations
uvicorn transistordatabase.gui_web.backend.main:app --port 8002  # Start API
pytest tests/ -q                                                  # Run tests
ruff check transistordatabase/                                    # Lint code

# Frontend operations (from transistordatabase/gui_web/)
npm run dev          # Start development server
npm run build        # Production build
npm test             # Run unit tests
npm run test:e2e     # Run E2E tests

# Desktop GUI
python -m transistordatabase.gui.gui  # Launch PyQt5 interface

# Python API usage
python3 -c "from transistordatabase import DatabaseManager; dm = DatabaseManager(); print(len(dm.get_all_transistor_names()))"
```

## Project Structure Overview

```
transistordatabase/
├── core/                  # Core business logic (clean architecture)
│   ├── models.py         # Domain entities
│   ├── services.py       # ABC interfaces
│   ├── repository.py     # Data persistence
│   └── adapters.py       # Legacy bridge
├── backend/              # Service implementations
│   └── concrete_services.py
├── gui/                  # PyQt5 desktop interface
│   └── api_client.py     # REST client
├── gui_web/              # Vue 3 web interface
│   ├── backend/main.py   # FastAPI REST API
│   └── src/              # Vue frontend code
├── database/             # Transistor data (JSON files)
├── docs/                 # Documentation
└── tests/                # Test suite (370 tests)
```

## Database Location

Transistor data is stored in JSON files:

```
transistordatabase/database/
├── CREE_C3M0060065J.json
├── Infineon_IPT60R028G7.json
├── Infineon_FF300R12KE3.json
└── ...
```

**Custom database path:**
```bash
export TDB_DATABASE_PATH="/path/to/my/database"
```

## Common Issues & Solutions

### Issue: "0 transistors" in web GUI

**Solution:**
```bash
# Check backend is running on port 8002
lsof -i:8002

# If nothing, start backend:
uvicorn transistordatabase.gui_web.backend.main:app --port 8002

# Hard refresh browser
Ctrl+Shift+R
```

### Issue: Import error "No module named 'transistordatabase'"

**Solution:**
```bash
cd transistordatabase
pip install -e .
```

### Issue: Web frontend won't start

**Solution:**
```bash
cd transistordatabase/gui_web
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Issue: Backend crashes on startup

**Solution:**
```bash
# Check Python version (need 3.10+)
python --version

# Reinstall dependencies
pip install -e .

# Check for port conflicts
lsof -i:8002  # Kill process if needed
```

## Sample Workflows

### Workflow 1: Find 1200V IGBTs

```bash
# 1. Start web interface
uvicorn transistordatabase.gui_web.backend.main:app --port 8002 &
cd transistordatabase/gui_web && npm run dev

# 2. In browser (http://localhost:5173):
#    - Search Database → Type: IGBT
#    - Voltage Rating: Min = 1000V, Max = 1400V
#    - View results
```

### Workflow 2: Design Buck Converter

```bash
# 1. Launch desktop GUI
python -m transistordatabase.gui.gui

# 2. In GUI:
#    - Go to Topology Calculator
#    - Select: Buck Converter
#    - Choose transistor: CREE_C3M0060065J
#    - Input: V_in = 400V, V_out = 12V, I_out = 50A
#    - Set frequency: 100 kHz
#    - View efficiency and thermal results
```

### Workflow 3: Add Custom Transistor

```bash
# 1. Get datasheet from manufacturer
# 2. Launch web interface
# 3. Go to "Create Transistor" tab
# 4. Fill in:
#    - Name: Manufacturer_PartNumber
#    - Type: MOSFET/SiC-MOSFET/IGBT/GaN
#    - Max voltage, current, temperature
#    - R_ds_on, thermal resistance
#    - Switching loss curves (from datasheet graphs)
# 5. Save!
```

## Python API Quick Example

```python
from transistordatabase import DatabaseManager
from transistordatabase.core import Transistor

# Initialize database manager
db = DatabaseManager()

# Load a transistor (returns core model)
transistor = db.load_transistor_core("CREE_C3M0060065J")

# Access properties
print(f"Name: {transistor.metadata.name}")
print(f"Type: {transistor.metadata.type}")
print(f"Max Voltage: {transistor.electrical_ratings.v_abs_max}V")
print(f"Max Current: {transistor.electrical_ratings.i_abs_max}A")
print(f"R_th_JC: {transistor.thermal_properties.r_th_junction_case}K/W")

# List all transistors
all_names = db.get_all_transistor_names()
print(f"Database contains {len(all_names)} transistors")

# Filter by type
sic_mosfets = [t for t in all_names if "SiC" in t]
print(f"SiC MOSFETs: {len(sic_mosfets)}")
```

## REST API Quick Example

```bash
# Get all transistors
curl http://localhost:8002/api/transistors

# Get specific transistor
curl http://localhost:8002/api/transistors/CREE_C3M0060065J

# Create new transistor
curl -X POST http://localhost:8002/api/transistors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test_Transistor",
    "type": "MOSFET",
    "manufacturer": "Generic",
    "v_abs_max": 600,
    "i_abs_max": 100
  }'

# Export to PLECS
curl http://localhost:8002/api/export/CREE_C3M0060065J/plecs > output.xml

# API documentation
# → http://localhost:8002/docs (Swagger UI)
# → http://localhost:8002/redoc (ReDoc)
```

## Next Steps

Now that you're running, dive deeper:

1. **📖 [User Guide](USER_GUIDE.md)** - Complete feature documentation
2. **🎓 [Tutorials](TUTORIAL.md)** - 5 hands-on tutorials
3. **❓ [FAQ](FAQ.md)** - Common questions answered
4. **🔧 [API Documentation](API_DOCUMENTATION.md)** - REST API reference

## Support

- **Documentation**: Check USER_GUIDE.md and TUTORIAL.md
- **GitHub Issues**: https://github.com/tinix84/transistordatabase/issues
- **API Docs**: http://localhost:8002/docs (when backend is running)

## Tips for Success

✅ **Always start backend before frontend** (web interface)
✅ **Use filters to narrow results** - don't view entire database at once
✅ **Export before editing** - backup your transistor data
✅ **Check datasheet accuracy** - garbage in = garbage out
✅ **Use virtual datasheet** - preview before exporting
✅ **Test in simulation** - verify exported data works

## Performance Tips

- **Database too slow?** Use filters to limit results
- **Export fails?** Check browser console (F12) for errors
- **Frontend sluggish?** Use desktop app for better performance
- **Backend timeout?** Increase uvicorn timeout: `--timeout-keep-alive 75`

## Development Setup (Optional)

For contributors who want to modify code:

```bash
# Install development dependencies
pip install -e ".[dev]"
cd transistordatabase/gui_web
npm install

# Run tests
pytest tests/ -v
cd transistordatabase/gui_web && npm test

# Lint code
ruff check transistordatabase/
cd transistordatabase/gui_web && npm run lint

# Build documentation
cd docs && make html

# Format code
ruff format transistordatabase/
```

---

**You're all set! 🎉**

Start exploring the database, compare transistors, and export to your favorite simulation tool!

For detailed tutorials, see [TUTORIAL.md](TUTORIAL.md).
For complete feature reference, see [USER_GUIDE.md](USER_GUIDE.md).
