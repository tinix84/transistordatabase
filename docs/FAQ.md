# Transistor Database - Frequently Asked Questions (FAQ)

Quick answers to common questions! 💡

## Table of Contents

- [General](#general)
- [Getting Started](#getting-started)
- [Search & Filtering](#search--filtering)
- [Data Entry](#data-entry)
- [Exporting](#exporting)
- [Comparison](#comparison)
- [Topology Calculator](#topology-calculator)
- [Troubleshooting](#troubleshooting)
- [Technical](#technical)

---

## General

### What is Transistor Database?

Transistor Database is a tool for managing power semiconductor data and analyzing power converter designs. It helps engineers select transistors, export data to simulation tools, and calculate converter performance.

### Who is it for?

- Power electronics engineers
- Circuit designers
- Researchers
- Students learning power electronics
- Simulation engineers

### What transistor types are supported?

- **MOSFETs** - Silicon MOSFETs
- **SiC-MOSFETs** - Silicon Carbide MOSFETs
- **IGBTs** - Insulated Gate Bipolar Transistors
- **GaN Transistors** - Gallium Nitride transistors

### Is it free?

Yes! Transistor Database is open-source and free to use.

### Where can I get support?

- **Documentation**: Check USER_GUIDE.md and TUTORIAL.md
- **GitHub Issues**: https://github.com/tinix84/transistordatabase/issues
- **Discord**: [Join our community]
- **Email**: [Contact information]

---

## Getting Started

### How do I install it?

```bash
# Clone repository
git clone https://github.com/tinix84/transistordatabase.git
cd transistordatabase

# Install dependencies
pip install -e .

# For web interface
cd transistordatabase/gui_web
npm install
```

### How do I run it?

**Desktop (PyQt5):**
```bash
python -m transistordatabase.gui.gui
```

**Web Interface:**
```bash
# Terminal 1: Backend
uvicorn transistordatabase.gui_web.backend.main:app --port 8002

# Terminal 2: Frontend
cd transistordatabase/gui_web
npm run dev

# Open: http://localhost:5173
```

### Which version should I use - desktop or web?

**Use Desktop if:**
- You prefer native applications
- You work offline frequently
- You need maximum performance

**Use Web if:**
- You want modern UI
- You work in a browser
- You need remote access
- You want the latest features

Both have the same core functionality!

### Where is my data stored?

Data is stored in JSON files in the `database/` directory:
```
transistordatabase/
└── database/
    ├── CREE_C3M0060065J.json
    ├── Infineon_FF300R12KE3.json
    └── ...
```

### Can I use my own database location?

Yes! Set the database path in settings or via environment variable:
```bash
export TDB_DATABASE_PATH="/path/to/my/database"
```

---

## Search & Filtering

### How do I search for a specific transistor?

1. Go to Search Database tab
2. Enable "Name Contains" filter
3. Type part of the name
4. Results update automatically

### Can I save my filter settings?

Currently, filters are not saved between sessions. This feature is planned for a future release.

**Workaround**: Export filtered results to reapply later.

### Why don't I see any results?

**Common reasons:**
1. **Filters too restrictive** - Try resetting filters
2. **Database empty** - Check database/ directory has JSON files
3. **Name typo** - Check spelling in name filter

**Solution**: Click "Reset Filters" button and try again.

### How many transistors can I search at once?

The database can handle thousands of transistors. Performance depends on your computer hardware.

**Current database**: ~25-30 transistors (sample set)

### Can I search by package type?

Yes! Use the "Housing Type" filter:
1. Enable "Housing Type" checkbox
2. Select from dropdown (TO-220, TO-247, etc.)

### How do I filter by power rating?

Power rating isn't directly filterable, but you can filter by:
- **Voltage rating** (V_abs_max)
- **Current rating** (I_cont)
- Then calculate: P = V × I

---

## Data Entry

### How do I add a new transistor?

1. Go to "Create Transistor" tab
2. Fill in metadata (name, type, manufacturer)
3. Add electrical ratings
4. Add curves and characteristics
5. Click "Save"

See [Tutorial 3](TUTORIAL.md#tutorial-3-adding-a-new-transistor) for detailed steps.

### What data is required?

**Minimum required:**
- Name (must be unique)
- Type (MOSFET, IGBT, etc.)
- Manufacturer
- Max voltage (V_abs_max)
- Max current (I_abs_max)

**Recommended for calculations:**
- Continuous current (I_cont)
- Max junction temperature (T_j_max)
- Thermal resistance (R_th)
- Switching losses (E_on, E_off)

### Where do I get transistor data?

**Primary source**: Manufacturer datasheets
- Download from manufacturer website
- Look for "electrical characteristics"
- Extract graphs and tables

**Data to extract:**
- Absolute maximum ratings
- Electrical characteristics
- Switching characteristics graphs
- Thermal characteristics
- Capacitance curves

### How do I enter graph data?

For curves (I-V, C-V, E-I), enter arrays of points:

**Example: Output Characteristics**
```
V_ds: [0, 0.5, 1.0, 1.5, 2.0]
I_d:  [0, 10, 20, 30, 35]
```

**Tips:**
- Use 5-10 points for smooth curves
- Include key regions (low current, saturation)
- Match units (V, A, not mV, mA)

### Can I import data from datasheets automatically?

Not currently. Automatic datasheet parsing is planned but not yet implemented.

**Current workflow**: Manual entry from datasheets

### How do I edit an existing transistor?

1. Find transistor in Search Database
2. Click "Edit" button (or transistor name)
3. Modify fields
4. Click "Save" to update

### Can I delete a transistor?

Yes:
1. Find transistor in Search Database
2. Click "Delete" button
3. Confirm deletion
4. Transistor is permanently removed

**Warning**: This cannot be undone! Export first if unsure.

### What if I make a mistake?

**Before saving**: Just reload the page or click "Cancel"

**After saving**:
1. Export database first (backup)
2. Edit the transistor to fix
3. Or delete and re-create

---

## Exporting

### What export formats are supported?

1. **JSON** - Data backup, sharing
2. **MATLAB** - MATLAB scripts (.m files)
3. **PLECS** - PLECS thermal models (.xml)
4. **Simulink** - Simulink/Simscape (.mat)
5. **GeckoCIRCUITS** - GeckoCIRCUITS models (.ipes)
6. **PDF** - Virtual datasheet with plots

### How do I export to PLECS?

1. Go to "Exporting Tools" tab
2. Select transistor
3. Click "PLECS" button
4. Click "Export"
5. Import XML file in PLECS Thermal Library

See [Tutorial 4](TUTORIAL.md#tutorial-4-exporting-for-simulation) for details.

### Can I export multiple transistors at once?

Yes! Bulk export:
1. Go to Search Database
2. Apply filters to select transistors
3. Click "Export Results"
4. Choose format
5. Downloads ZIP with all files

### Why does my export fail?

**Common reasons:**
1. **Incomplete data** - Missing required fields
2. **Invalid format** - Corrupted transistor data
3. **Disk space** - Not enough space to write file
4. **Permissions** - Can't write to download folder

**Check**: Browser console (F12) for error messages

### Can I customize export format?

Export formats are standardized for tool compatibility. Custom formats are not currently supported.

**Workaround**: Export JSON and convert with custom script.

### Do I need special software to use exports?

**Yes, for simulation formats:**
- PLECS export → Needs PLECS software
- MATLAB export → Needs MATLAB
- Simulink export → Needs MATLAB + Simulink
- GeckoCIRCUITS → Needs GeckoCIRCUITS

**No special software needed:**
- JSON → Any text editor
- PDF → Any PDF viewer

---

## Comparison

### How many transistors can I compare?

**2 to 3 transistors** at once.

This is optimal for side-by-side comparison. More than 3 becomes hard to read.

### Can I compare different technologies?

Yes! Compare any combination:
- SiC vs Si MOSFETs
- MOSFET vs IGBT
- Different manufacturers
- Different voltage classes

This helps understand technology trade-offs!

### What metrics are compared?

**Electrical:**
- Voltage rating
- Current rating
- On-state resistance

**Thermal:**
- Thermal resistance
- Package size

**Switching:**
- Turn-on/off energies
- Reverse recovery

**Capacitance:**
- Output capacitance
- Input capacitance
- Gate charge

### Can I export comparison results?

Yes! Click "Export Comparison" to save:
- PDF report with charts
- Excel spreadsheet with table
- PNG images of charts

### How do I interpret the comparison?

**Color coding:**
- 🟢 Green = Best in category
- 🟡 Yellow = Medium
- 🔴 Red = Worst in category

**Lower is better for:**
- R_ds_on, R_th (losses)
- E_on, E_off (switching)
- C_oss (switching speed)

**Higher is better for:**
- Voltage rating
- Current rating
- Temperature rating

---

## Topology Calculator

### What topologies are supported?

Currently supported:
1. **Buck Converter** (step-down)
2. **Boost Converter** (step-up)
3. **Buck-Boost Converter** (step-up/down)

All operate in Continuous Conduction Mode (CCM).

### Can I add more topologies?

Advanced topologies (Bridgeless PFC, DAB, LLC, SRC-ZVS) were removed to simplify the codebase. The focus is on fundamental PWM converters.

**Workaround**: Export transistor data and use in external simulation tools.

### Why is my efficiency unrealistic?

**If efficiency > 100%**: Input parameters are wrong
- Check voltage values
- Verify current values
- Ensure realistic operating point

**If efficiency < 70%**: Check design
- Frequency too high?
- Poor transistor choice?
- Operating point mismatch?

**Typical efficiencies:**
- Buck/Boost: 90-97%
- Buck-Boost: 88-95%

### What is gate resistance (R_g)?

Gate resistance controls switching speed:
- **Low R_g (2-5Ω)**: Fast switching, lower loss, more ringing
- **High R_g (15-30Ω)**: Slow switching, higher loss, cleaner waveforms
- **Typical: 8-12Ω**: Good balance

Use the slider to find optimal value!

### How accurate are the calculations?

**Accuracy depends on:**
- Quality of transistor data (from datasheet)
- Number of data points in curves
- Operating conditions match datasheet
- Thermal model accuracy

**Typical accuracy:** ±5-10% vs hardware

**Best accuracy when:**
- Operating point is well within SOA
- Temperature is measured/known
- Parasitics are minimized in layout

### Can I simulate DCM operation?

Currently only CCM (Continuous Conduction Mode) is supported. The calculator checks if DCM occurs and shows a warning.

**To stay in CCM:**
- Increase inductance
- Increase switching frequency
- Increase minimum load current

### Why does temperature exceed T_j_max?

**Reasons:**
1. **Insufficient cooling** - Add heatsink or increase airflow
2. **Too much loss** - Reduce switching frequency or improve efficiency
3. **Wrong R_th value** - Check thermal resistance data

**Solutions:**
- Select larger transistor (lower R_ds_on)
- Reduce switching frequency
- Improve thermal design
- Add forced cooling (fan)

---

## Troubleshooting

### The web GUI shows "0 transistors"

**Problem**: Frontend can't reach backend

**Solutions:**
1. Check backend is running on port 8002:
   ```bash
   lsof -i:8002
   ```

2. Check frontend is on port 5173:
   ```bash
   lsof -i:5173
   ```

3. Verify backend URL in browser console (F12):
   - Should call `http://localhost:8002/api/transistors`

4. Check CORS settings in backend:
   - Must allow origin `http://localhost:5173`

5. Hard refresh browser: `Ctrl+Shift+R`

See [WEB_GUI_DEBUG_REPORT.md](../WEB_GUI_DEBUG_REPORT.md) for detailed debugging.

### Tests are failing

**Unit tests failing:**
```bash
npm test
```

**Common causes:**
- Mock data mismatch
- Component not fully implemented
- Props changed but tests not updated

**E2E tests failing:**
```bash
npm run test:e2e
```

**Common causes:**
- Backend not running
- Frontend not running
- Ports changed
- Selectors changed in UI

**Solution**: See [TEST_README.md](../transistordatabase/gui_web/TEST_README.md)

### Import error in Python

**Error**: `ModuleNotFoundError: No module named 'transistordatabase'`

**Solution**:
```bash
cd transistordatabase
pip install -e .
```

### Database not found

**Error**: "Database directory not found"

**Solution**:
1. Check `database/` directory exists
2. Check it contains `.json` files
3. Set correct path in settings
4. Or set environment variable:
   ```bash
   export TDB_DATABASE_PATH="/path/to/database"
   ```

### Application won't start

**Desktop app**:
```bash
# Check Python version (need 3.10+)
python --version

# Check dependencies
pip install -e .

# Try running with full path
python -m transistordatabase.gui.gui
```

**Web app**:
```bash
# Check Node version (need 18+)
node --version

# Install dependencies
cd transistordatabase/gui_web
npm install

# Start backend first
uvicorn transistordatabase.gui_web.backend.main:app --port 8002

# Then frontend
npm run dev
```

### Slow performance

**Cause**: Large database or slow computer

**Solutions:**
1. Use filters to limit results
2. Increase pagination (show fewer per page)
3. Close other applications
4. Use desktop app (faster than web)

### Export fails silently

**Check browser downloads:**
1. Look in Downloads folder
2. Check browser download settings
3. Check popup blocker
4. Check disk space

**Check browser console** (F12):
- Look for error messages
- Check network tab for failed requests

---

## Technical

### What Python version is required?

**Minimum: Python 3.10**

The project uses features introduced in Python 3.10:
- Type hints with `|` operator
- `match` statements
- Improved error messages

### What Node version is required?

**Minimum: Node.js 18**

The Vue.js frontend requires modern Node.js:
- ES modules support
- Modern JavaScript features
- Vite build tool compatibility

### Can I use it with Python 3.9?

No. Python 3.10+ is required for type hints and other features.

**Upgrade Python:**
```bash
# Ubuntu/Debian
sudo apt install python3.10

# macOS
brew install python@3.10

# Windows
# Download from python.org
```

### How is data stored?

**Format**: JSON files
**Location**: `database/` directory
**One file per transistor**: `Manufacturer_PartNumber.json`

**Example**:
```json
{
  "metadata": {
    "name": "CREE_C3M0060065J",
    "type": "SiC-MOSFET",
    "manufacturer": "CREE"
  },
  "electrical": {
    "v_abs_max": 650,
    "i_abs_max": 120,
    "i_cont": 60
  },
  ...
}
```

### Is there a REST API?

Yes! The web backend exposes a REST API:

**Base URL**: `http://localhost:8002`

**Endpoints**:
- `GET /api/transistors` - List all
- `GET /api/transistors/{name}` - Get one
- `POST /api/transistors` - Create
- `PUT /api/transistors/{name}` - Update
- `DELETE /api/transistors/{name}` - Delete

**Documentation**: `http://localhost:8002/docs` (Swagger UI)

### Can I contribute?

Yes! Contributions welcome:

**Ways to contribute:**
1. Add transistor data
2. Report bugs
3. Suggest features
4. Improve documentation
5. Submit pull requests

**See**: CONTRIBUTING.md (if available)

### What license is it?

Check the LICENSE file in the repository.

Typically open-source (MIT, BSD, or GPL).

### How do I cite this in papers?

```bibtex
@software{transistordatabase,
  title = {Transistor Database},
  author = {[Authors]},
  year = {2024},
  url = {https://github.com/tinix84/transistordatabase}
}
```

Check repository for official citation.

---

## Still Have Questions?

**Check documentation:**
- [User Guide](USER_GUIDE.md) - Complete reference
- [Tutorial](TUTORIAL.md) - Step-by-step examples
- [Quick Start](QUICK_START.md) - Get running fast

**Get help:**
- [GitHub Issues](https://github.com/tinix84/transistordatabase/issues)
- [Discord Community](#)
- [Email Support](#)

**Found a bug?**
Report it on GitHub Issues with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Screenshots if applicable
- Your OS and versions

We're here to help! 🤝
