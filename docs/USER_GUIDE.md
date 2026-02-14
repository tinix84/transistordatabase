# Transistor Database - User Guide

## Welcome to Transistor Database! 🔌

This guide will help you make the most of the Transistor Database application for managing and analyzing power semiconductor transistor data.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Search Database](#search-database)
4. [Create Transistor](#create-transistor)
5. [Exporting Tools](#exporting-tools)
6. [Comparison Tools](#comparison-tools)
7. [Topology Calculator](#topology-calculator)
8. [Tips & Best Practices](#tips--best-practices)

---

## Overview

### What is Transistor Database?

Transistor Database is a comprehensive tool for managing power semiconductor data including:
- **MOSFETs** - Metal-Oxide-Semiconductor Field-Effect Transistors
- **SiC-MOSFETs** - Silicon Carbide MOSFETs
- **IGBTs** - Insulated Gate Bipolar Transistors
- **GaN Transistors** - Gallium Nitride Transistors

### Key Features

✅ **Search & Filter** - Find transistors by specifications
✅ **Data Management** - Create, edit, and delete transistor entries
✅ **Export** - Export to MATLAB, PLECS, Simulink, GeckoCIRCUITS, and more
✅ **Comparison** - Compare up to 3 transistors side-by-side
✅ **Topology Calculator** - Analyze Buck, Boost, and Buck-Boost converters
✅ **Visual Datasheets** - Generate PDF datasheets

### Supported Data

Each transistor entry includes:
- Metadata (name, type, manufacturer, housing)
- Electrical ratings (voltage, current, temperature)
- Thermal properties (thermal resistance, cooling area)
- Switch characteristics (on-state, switching losses)
- Diode characteristics (forward voltage, reverse recovery)
- Capacitances (Coss, Ciss, Crss)
- Foster thermal models
- Gate charge curves
- Safe Operating Area (SOA)

---

## Getting Started

### Launching the Application

**Desktop Application (PyQt5):**
```bash
python -m transistordatabase.gui.gui
```

**Web Application:**
```bash
# Start backend
uvicorn transistordatabase.gui_web.backend.main:app --port 8002

# In another terminal, start frontend
cd transistordatabase/gui_web
npm run dev

# Open browser to http://localhost:5173
```

### Interface Overview

The application has 5 main tabs:

1. **🔍 Search Database** - Find and browse transistors
2. **➕ Create Transistor** - Add or edit transistor data
3. **📤 Exporting Tools** - Export to simulation tools
4. **🔍 Comparison Tools** - Compare transistor specifications
5. **🧮 Topology Calculator** - Analyze converter topologies

### Navigation

- Click tabs at the top to switch views
- Your current view is highlighted in blue
- The header shows total transistor count
- Use the theme toggle (☀️/🌙) to switch between light/dark mode

---

## Search Database

### Basic Search

1. **Navigate** to the "Search Database" tab
2. **View Results** - All transistors are displayed by default
3. **Scan the list** to find transistors of interest

### Advanced Filtering

Enable filters to narrow down results:

#### Name Filter
- ☑️ Check "Name Contains"
- Type part of the transistor name (e.g., "CREE", "Infineon")
- Case-insensitive search

#### Type Filter
- ☑️ Check "Type"
- Select from dropdown:
  - IGBT
  - SiC-MOSFET
  - MOSFET
  - GaN-Transistor

#### Manufacturer Filter
- ☑️ Check "Manufacturer"
- Select from available manufacturers:
  - CREE
  - Infineon
  - Fuji Electric
  - GaN Systems
  - And more...

#### Voltage Range Filter
- ☑️ Check "Max Voltage (V)"
- Set minimum and/or maximum values
- Example: 600V to 1200V

#### Current Range Filter
- ☑️ Check "Continuous Current (A)"
- Set min/max current ratings
- Example: 50A to 300A

#### Temperature Range Filter
- ☑️ Check "Max Junction Temperature (°C)"
- Set operating temperature range
- Example: 125°C to 175°C

### Multiple Filters

Combine multiple filters for precise searches:

**Example 1: High-voltage SiC MOSFETs**
- Type: SiC-MOSFET
- Max Voltage: 1000V to 1500V
- Manufacturer: CREE

**Example 2: High-current IGBTs**
- Type: IGBT
- Continuous Current: 200A minimum
- Max Junction Temp: 150°C minimum

### Working with Results

#### View Modes
- **Table View** - Detailed list with all specifications
- **Card View** - Visual cards with key information

#### Sorting
- Click column headers to sort
- Click again to reverse sort order
- Sortable fields: Name, Voltage, Current, etc.

#### Pagination
- Default: 20 items per page
- Use page controls at bottom
- Jump to specific page

#### Actions on Results
- **View Details** - Click transistor name
- **Load to Exporting** - Send to export tools
- **Load to Comparison** - Add to comparison
- **Load to Topology** - Use in topology calculator
- **Edit** - Modify transistor data
- **Delete** - Remove from database

### Reset Filters

Click **"🔄 Reset Filters"** to clear all filters and show all transistors.

### Export Results

Click **"💾 Export Results"** to export filtered results to CSV or JSON.

---

## Create Transistor

### Creating a New Transistor

1. Navigate to **"➕ Create Transistor"** tab
2. Fill in the required fields (marked with *)
3. Add optional data for better accuracy
4. Click **"Save"** to create

### Editing an Existing Transistor

1. Find transistor in Search Database
2. Click **"Edit"** button
3. Modify fields as needed
4. Click **"Save"** to update

### Form Sections

#### Metadata (Required)
- **Name*** - Unique identifier (e.g., "CREE_C3M0060065J")
- **Type*** - IGBT, SiC-MOSFET, MOSFET, or GaN
- **Manufacturer*** - Company name
- **Housing Type** - Package type (TO-247, TO-220, etc.)
- **Author** - Who created this entry
- **Comment** - Additional notes

#### Electrical Ratings
- **Max Voltage (V_abs_max)** - Maximum drain-source voltage
- **Max Current (I_abs_max)** - Peak current rating
- **Continuous Current (I_cont)** - Continuous current rating
- **Max Junction Temp (T_j_max)** - Maximum operating temperature

#### Thermal Properties
- **R_th_cs** - Case-to-sink thermal resistance
- **R_th_total** - Total thermal resistance
- **Housing Area** - Physical package area
- **Cooling Area** - Effective cooling surface area

#### Switch Characteristics
- **Channel Data** - I-V characteristics at different temperatures
- **E_on Data** - Turn-on energy losses vs current
- **E_off Data** - Turn-off energy losses vs current
- **Gate Charge Curves** - Q_g vs V_gs
- **Foster Thermal Model** - RC thermal network

#### Diode Characteristics
- **Channel Data** - Forward I-V characteristics
- **E_rr Data** - Reverse recovery losses
- **Foster Thermal Model** - Diode thermal network

#### Capacitances
- **C_oss** - Output capacitance vs voltage
- **C_iss** - Input capacitance
- **C_rss** - Reverse transfer capacitance

### Data Entry Tips

**Best Practices:**
1. Use manufacturer datasheets as primary source
2. Include measurement conditions (temperature, gate voltage)
3. Add multiple operating points for better interpolation
4. Use consistent units throughout
5. Add comments to document data sources

**Naming Convention:**
```
Manufacturer_PartNumber
Examples:
- CREE_C3M0060065J
- Infineon_FF300R12KE3
- GaNSystems_GS66506T
```

### Validation

The form validates:
- ✅ Required fields are filled
- ✅ Numeric fields contain valid numbers
- ✅ Voltage/current values are positive
- ✅ Name is unique in database

### Saving Options

- **Save** - Save and return to search
- **Save & New** - Save and create another
- **Cancel** - Discard changes and return

---

## Exporting Tools

Export transistor data to simulation tools and other formats.

### Supported Export Formats

#### 1. JSON Export
- **Use Case**: Data backup, sharing, or custom processing
- **Contents**: Complete transistor data in JSON format
- **Extension**: `.json`

#### 2. MATLAB Export
- **Use Case**: MATLAB simulation and analysis
- **Contents**: MATLAB structure with all data
- **Extension**: `.m`

#### 3. PLECS Export
- **Use Case**: PLECS power electronics simulation
- **Contents**: PLECS thermal model XML
- **Extension**: `.xml`

#### 4. Simulink Export
- **Use Case**: MATLAB/Simulink models
- **Contents**: Simulink block parameters
- **Extension**: `.mat`

#### 5. GeckoCIRCUITS Export
- **Use Case**: GeckoCIRCUITS power electronics simulation
- **Contents**: GeckoCIRCUITS model file
- **Extension**: `.ipes`

#### 6. Virtual Datasheet (PDF)
- **Use Case**: Documentation and sharing
- **Contents**: Professional PDF datasheet with plots
- **Extension**: `.pdf`

### How to Export

1. Navigate to **"📤 Exporting Tools"** tab
2. **Select Transistor** from dropdown
   - Or come from Search with pre-selected transistor
3. **Choose Export Format** (click format button)
4. **Click "Export"**
5. **Save File** when prompted by browser

### Bulk Export

To export multiple transistors:
1. Go to Search Database
2. Filter to desired transistors
3. Click "Export Results"
4. Choose format
5. Downloads ZIP with all files

### Export Options

Some formats have additional options:

**JSON Export:**
- Pretty Print (indented)
- Compact (minified)

**MATLAB Export:**
- MATLAB Version (R2020a+, Legacy)
- Include Comments

**PDF Datasheet:**
- Include Plots
- Include SOA Curves
- Color or Black & White

---

## Comparison Tools

Compare up to 3 transistors side-by-side.

### How to Compare

1. Navigate to **"🔍 Comparison Tools"** tab
2. **Select First Transistor** from dropdown 1
3. **Select Second Transistor** from dropdown 2
4. *Optional:* **Select Third Transistor** from dropdown 3
5. View comparison table and charts

### Comparison Table

The table shows key specifications:

**Electrical:**
- Max Voltage (V_abs_max)
- Max Current (I_abs_max)
- Continuous Current (I_cont)
- Max Junction Temperature

**Thermal:**
- R_th_cs (Case-to-sink)
- R_th_total (Total)
- Housing Area
- Cooling Area

**Switch:**
- On-state Resistance (R_ds_on)
- Turn-on Energy (E_on)
- Turn-off Energy (E_off)

**Diode:**
- Forward Voltage Drop
- Reverse Recovery Energy (E_rr)

**Metadata:**
- Manufacturer
- Housing Type
- Transistor Type

### Comparison Charts

Visual comparison with bar charts:

1. **Voltage Rating Chart** - Max voltage comparison
2. **Current Rating Chart** - Current capabilities
3. **Thermal Resistance Chart** - Thermal performance
4. **Switching Losses Chart** - E_on and E_off
5. **Capacitance Chart** - C_oss, C_iss, C_rss

### Interpreting Results

**Color Coding:**
- 🟢 **Green** - Best performance in category
- 🟡 **Yellow** - Medium performance
- 🔴 **Red** - Lowest performance

**Key Insights:**

**For High Efficiency:**
- Lower R_ds_on (conduction losses)
- Lower E_on and E_off (switching losses)
- Better thermal resistance

**For High Power:**
- Higher voltage rating
- Higher current rating
- Larger package (better cooling)

**For High Frequency:**
- Lower capacitances (C_oss, C_iss)
- Lower switching energies
- Fast reverse recovery (diode)

### Comparison Actions

- **Pop Out Charts** - Open chart in new window
- **Export Comparison** - Save comparison as PDF/Excel
- **Clear** - Reset all selections
- **Print** - Print comparison table

### Use Cases

**Application 1: Design Selection**
- Compare candidates for your converter
- Evaluate trade-offs (cost, performance, availability)
- Choose optimal device

**Application 2: Replacement Search**
- Find alternatives for obsolete parts
- Match specifications closely
- Verify pin compatibility (housing type)

**Application 3: Portfolio Analysis**
- Compare manufacturer offerings
- Understand technology differences (Si vs SiC vs GaN)
- Benchmark performance

---

## Topology Calculator

Analyze converter performance with real transistor data.

### Supported Topologies

1. **Buck Converter** (Step-down)
   - Input > Output voltage
   - Common in DC-DC converters
   - Example: 48V to 12V

2. **Boost Converter** (Step-up)
   - Output > Input voltage
   - Used in PFC, battery systems
   - Example: 12V to 400V

3. **Buck-Boost Converter** (Step-up/down)
   - Flexible voltage conversion
   - Inverted output
   - Example: 12V to 24V or 12V to 5V

### How to Calculate

1. Navigate to **"🧮 Topology Calculator"** tab
2. **Select Topology** (Buck, Boost, or Buck-Boost)
3. **Select Transistor** from dropdown
4. **Enter Operating Parameters:**
   - Input Voltage (V_in)
   - Output Voltage (V_out)
   - Output Current (I_out)
   - Switching Frequency (f_sw)
   - Gate Resistance (R_g)
5. **View Results** automatically

### Input Parameters

#### Electrical Parameters
- **V_in** - Input voltage (DC)
  - Range: 10V to 1000V
  - Example: 400V (DC bus)

- **V_out** - Output voltage (DC)
  - Range: 3V to 1000V
  - Example: 48V (battery)

- **I_out** - Output current (DC)
  - Range: 0.1A to 1000A
  - Example: 50A (load)

- **f_sw** - Switching frequency
  - Range: 1kHz to 1MHz
  - Example: 100kHz
  - Higher = smaller passives, more switching loss

#### Component Parameters
- **R_g** - Gate resistance
  - Range: 1Ω to 100Ω
  - Example: 10Ω
  - Lower = faster switching, more ringing
  - Use slider for real-time update

- **L** - Inductance (optional)
  - Auto-calculated for CCM operation
  - Can override for custom design

- **C** - Capacitance (optional)
  - Auto-calculated for ripple spec
  - Can override for custom design

### Results Display

#### Calculated Values

**Operating Point:**
- Duty Cycle (D) - Percentage on-time
- RMS Current - Through inductor
- Peak Current - Maximum inductor current
- Current Ripple - ΔI_L

**Losses:**
- Conduction Loss (Switch) - I²R losses
- Switching Loss (Switch) - E_on + E_off
- Conduction Loss (Diode) - Forward voltage loss
- Total Loss - Sum of all losses

**Efficiency:**
- η (Efficiency) - P_out / P_in × 100%
- Power Loss - Total dissipated power

**Thermal:**
- Junction Temperature (T_j) - Calculated from losses
- Temperature Rise - ΔT above ambient
- Warning if T_j > T_j_max

#### Waveform Plots

**Plot 1: Current Waveforms**
- Inductor current i_L(t)
- Switch current i_sw(t)
- Diode current i_d(t)

**Plot 2: Voltage Waveforms**
- Switch voltage v_sw(t)
- Diode voltage v_d(t)
- Inductor voltage v_L(t)

**Plot 3: Loss Breakdown**
- Pie chart of loss distribution
- Conduction vs switching
- Switch vs diode

**Plot 4: Efficiency vs Load**
- Efficiency curve
- Optimal load point
- Loss breakdown at different loads

### Gate Resistance Optimization

Use the **R_g slider** to find optimal gate resistance:

**Effects of R_g:**
- **Lower R_g (< 5Ω)**
  - ✅ Faster switching
  - ✅ Lower switching losses
  - ❌ More ringing/EMI
  - ❌ Higher gate drive stress

- **Higher R_g (> 20Ω)**
  - ✅ Cleaner waveforms
  - ✅ Less EMI
  - ❌ Slower switching
  - ❌ Higher switching losses

**Optimal Range:**
- Most applications: 5Ω to 15Ω
- High-frequency (>200kHz): 3Ω to 8Ω
- High-power (>10kW): 10Ω to 20Ω

### Design Guidelines

#### Buck Converter
```
Duty Cycle: D = V_out / V_in
Example: 12V / 48V = 0.25 (25%)

Inductance: L = (V_in - V_out) × D / (ΔI_L × f_sw)
For 10% ripple: ΔI_L = 0.1 × I_out

Capacitance: C = ΔI_L / (8 × f_sw × ΔV_out)
For 1% ripple: ΔV_out = 0.01 × V_out
```

#### Boost Converter
```
Duty Cycle: D = 1 - (V_in / V_out)
Example: 1 - (12V / 48V) = 0.75 (75%)

Inductance: L = V_in × D / (ΔI_L × f_sw)
Capacitance: C = I_out × D / (f_sw × ΔV_out)
```

#### Buck-Boost Converter
```
Duty Cycle: D = V_out / (V_in + V_out)
Example: 48V / (12V + 48V) = 0.8 (80%)

Note: Output is inverted (negative)
```

### Warnings and Errors

The calculator shows warnings for:

⚠️ **Overvoltage** - V_ds > V_abs_max
⚠️ **Overcurrent** - I_peak > I_abs_max
⚠️ **Overtemperature** - T_j > T_j_max
⚠️ **Discontinuous Mode** - Inductor current reaches zero
⚠️ **Unrealistic Efficiency** - Check input parameters

### Export Calculation

**Export Options:**
- PDF Report - Complete calculation report
- Excel Spreadsheet - Results table
- MATLAB Script - Reproduce calculation
- Plot Images - PNG of all plots

---

## Tips & Best Practices

### Data Quality

**Good Data = Accurate Results**

1. **Use Datasheet Values**
   - Extract from manufacturer datasheets
   - Document datasheet revision
   - Note test conditions

2. **Include Temperature Effects**
   - Add data at multiple temperatures (25°C, 125°C, 175°C)
   - Critical for accurate loss calculation

3. **Complete Switching Data**
   - E_on and E_off at different currents
   - Include gate resistance used in measurements
   - Note supply voltage

### Search Efficiency

**Find What You Need Faster**

1. **Start Broad, Then Filter**
   - First: Select transistor type
   - Then: Add voltage range
   - Finally: Add other filters

2. **Use Name Search for Known Parts**
   - Type manufacturer name
   - Or part number fragment

3. **Bookmark Frequent Searches**
   - Save filter combinations
   - Export filtered lists

### Exporting Workflow

**Streamline Your Workflow**

1. **Batch Export Similar Parts**
   - Filter by type/voltage range
   - Export all at once
   - Organize in simulation library

2. **Standardize Export Settings**
   - Use same format for project
   - Consistent naming convention
   - Version control exported files

### Comparison Strategy

**Make Better Decisions**

1. **Compare Apples to Apples**
   - Same voltage class
   - Similar current rating
   - Same technology (all SiC or all Si)

2. **Consider Full System Cost**
   - Device cost
   - Cooling requirements
   - Gate driver complexity
   - PCB area

3. **Look Beyond One Metric**
   - Don't just minimize R_ds_on
   - Consider switching losses
   - Evaluate thermal performance
   - Check availability and lead time

### Topology Calculator Usage

**Get Accurate Results**

1. **Verify Operating Mode**
   - Check CCM/DCM mode indicator
   - CCM is typical for power converters
   - Adjust inductance if needed

2. **Thermal Design Margin**
   - Target T_j < 0.8 × T_j_max
   - Account for ambient temperature variation
   - Consider heatsink thermal resistance

3. **Frequency Selection**
   - Higher frequency = smaller passives
   - But more switching loss
   - Consider EMI requirements
   - 50-200kHz is common range

4. **Validation**
   - Cross-check with hand calculations
   - Compare with similar designs
   - Verify efficiency is realistic (70-98%)

### Data Backup

**Protect Your Work**

1. **Regular Exports**
   - Export entire database monthly
   - JSON format preserves all data
   - Store in version control

2. **Before Major Changes**
   - Export before bulk edits
   - Export before database migration
   - Keep previous versions

### Collaboration

**Share Knowledge**

1. **Document Sources**
   - Add author name
   - Include datasheet links in comments
   - Note measurement conditions

2. **Consistent Naming**
   - Follow team conventions
   - Use full part numbers
   - Include manufacturer prefix

3. **Review Process**
   - Peer review new entries
   - Validate against datasheets
   - Test in topology calculator

---

## Keyboard Shortcuts

Speed up your workflow:

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + F` | Focus search field |
| `Ctrl/Cmd + N` | New transistor |
| `Ctrl/Cmd + S` | Save current form |
| `Ctrl/Cmd + E` | Export current |
| `Escape` | Cancel/Close dialog |
| `Tab` | Navigate between fields |
| `Shift + Tab` | Navigate backwards |

---

## Troubleshooting

### "No transistors found"
- Check that filters aren't too restrictive
- Reset filters and try again
- Verify database directory is correct

### "Export failed"
- Check disk space
- Verify write permissions
- Try different export format

### "Calculation error in topology"
- Verify all input parameters are positive
- Check voltage/current are within transistor limits
- Ensure realistic operating point

### "Cannot save transistor"
- Name might already exist (must be unique)
- Check required fields are filled
- Verify numeric fields have valid numbers

---

## Getting Help

### Documentation
- **User Guide** (this document)
- **API Documentation** - For developers
- **FAQ** - Common questions
- **Tutorial** - Step-by-step walkthrough

### Community
- **GitHub Issues** - Report bugs or request features
- **Discord** - Chat with users and developers
- **LinkedIn** - Professional network

### Support
- **Email**: [Your contact]
- **Issue Tracker**: https://github.com/tinix84/transistordatabase/issues

---

## Appendix

### Glossary

**CCM** - Continuous Conduction Mode (inductor current never reaches zero)
**DCM** - Discontinuous Conduction Mode (inductor current reaches zero)
**R_ds_on** - On-state resistance of MOSFET
**E_on** - Turn-on switching energy
**E_off** - Turn-off switching energy
**E_rr** - Reverse recovery energy (diode)
**C_oss** - Output capacitance
**R_th** - Thermal resistance
**T_j** - Junction temperature
**SOA** - Safe Operating Area

### Units

| Parameter | Unit | Symbol |
|-----------|------|--------|
| Voltage | Volts | V |
| Current | Amperes | A |
| Resistance | Ohms | Ω |
| Capacitance | Farads | F |
| Inductance | Henrys | H |
| Energy | Joules | J |
| Power | Watts | W |
| Frequency | Hertz | Hz |
| Temperature | Celsius | °C |
| Thermal Resistance | K/W | K/W |

---

**Need more help? Check out the [Tutorial](TUTORIAL.md) for a step-by-step walkthrough!**
