# Dashboard Quick Start Guide

## Installation (One-Time Setup)

```bash
# Run setup script
./setup_dashboard.sh

# Or manual installation
pip install jupyter jupyterlab ipywidgets plotly numpy pandas matplotlib scipy
jupyter nbextension enable --py widgetsnbextension
```

## Launch Dashboard

```bash
jupyter notebook transistordatabase_performance_dashboard.ipynb
```

## 5-Minute Walkthrough

### 1. Run Setup Cells (1-3)
- Cell 1: Import libraries (30 seconds)
- Cell 2: Load database (20 seconds, loads 245 devices)
- Cell 3: View summary statistics

**Expected Output**:
```
✓ Successfully loaded 245 transistors
📊 Total Devices: 245
🔌 Device Types:
  MOSFET        120
  SiC-MOSFET     68
  IGBT           45
  GaN            12
```

### 2. Configure Filters (Cell 4)
**Example: Find 1200V SiC MOSFETs with good data**

1. Select Device Type: `SiC-MOSFET`
2. Set Min Voltage: `1000V`
3. Set Max Voltage: `1500V`
4. Check: ✓ Require capacitance curves
5. Check: ✓ Require channel data
6. Click: **Apply Filters**

**Result**: ~15-25 devices

### 3. View Performance Plots

#### FOM Plot (Cell 5)
**What it shows**: R_ds(on) vs Q_g scatter with FOM contours

**How to use**:
- Hover over points to see device name and specs
- Lower and left = better (low R_ds, low Q_g)
- Devices below 100 mΩ·nC line are excellent

#### Capacitance Overlay (Cell 7)
**What it shows**: C_oss(V), C_iss(V), C_rss(V) curves

**How to use**:
1. Select 2-4 devices from dropdown
2. Click **Plot Capacitance**
3. Compare curves at your operating voltage

**Tip**: Lower C_oss → lower switching loss

### 4. Run Analytical Model (Cell 9)

**Best devices for analytical model**:
- Has C_oss curve ✓
- Has ≥3 channel curves ✓
- Has switching loss data (for validation)

**Example Configuration**:
```
DC Bus Voltage: 600V
Load Current: 20A
V_g,on: 20V
V_g,off: -5V
Gate Resistance: 10Ω
```

**Click**: **Run Analytical Model**

**Expected Output**:
```
[Turn-On Results]
  E_on = 145.32 µJ
  t_ri = 12.5 ns (current rise)
  t_fv = 18.3 ns (voltage fall)

[Turn-Off Results]
  E_off = 98.67 µJ
  t_rv = 15.2 ns (voltage rise)
  t_fi = 10.8 ns (current fall)

[Total Switching Energy]
  E_total = 243.99 µJ/cycle
```

### 5. Current Sweep (Cell 10)

**What it does**: Generates E_on(I), E_off(I) curves from 5-50A

**How to use**:
1. Set I_min = 5A, I_max = 50A
2. Select same device from step 4
3. Click **Run Current Sweep**
4. Compare model vs measured data (if available)

**Interpretation**:
- Model line = analytical prediction
- Markers = datasheet measurements
- Close match = model accurate for this device

### 6. Export Results (Cell 11)

```
Filename: my_filtered_devices.csv
Click: Export to CSV
```

**Output**: CSV file with all 15-25 filtered devices

## Common Use Cases

### Use Case 1: Device Selection
**Goal**: Find best device for 600V/20A buck converter

1. Filter: `600-900V`, `≥15A`, `MOSFET or SiC-MOSFET`
2. View FOM plot → identify top 5
3. Run analytical model at 600V/20A → compare E_total
4. Check capacitance curves → lower C_oss better
5. Export finalists to CSV

### Use Case 2: Model Validation
**Goal**: Verify analytical model accuracy

1. Filter: devices with `E_on` and `E_off` data
2. Select device, configure operating conditions
3. Run current sweep
4. Check model vs measured error
5. Expected: ±10-30% typical

### Use Case 3: Technology Comparison
**Goal**: Si MOSFET vs SiC MOSFET at 1200V

1. Run with Type = `MOSFET`, 1000-1200V → note FOM
2. Run with Type = `SiC-MOSFET`, 1000-1200V → note FOM
3. Compare FOM distributions
4. Result: SiC typically 5-10× better FOM

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No devices after filtering | Expand voltage range or uncheck data requirements |
| "Insufficient channel data" | Use different device with ≥3 V_gs curves |
| Widgets not interactive | Run: `jupyter nbextension enable --py widgetsnbextension` |
| Slow loading | Normal for 245 devices. First load: ~30 sec |
| Plot not showing | Click cell, Shift+Enter to re-run |

## Key Metrics Explained

| Metric | Unit | Meaning | Good Value |
|--------|------|---------|------------|
| R_ds(on) | mΩ | On-resistance | Lower = less conduction loss |
| Q_g | nC | Gate charge | Lower = faster switching |
| FOM | mΩ·nC | Figure of merit | <100 excellent, <500 good |
| E_on | µJ | Turn-on energy | Depends on V, I, R_g |
| E_off | µJ | Turn-off energy | Usually E_off < E_on |
| C_oss | pF | Output capacitance | Lower = less E_oss |

## Next Steps

- **Full Documentation**: See `DASHBOARD_README.md`
- **Model Details**: See `transistordatabase/analytical_models.py`
- **Example Scripts**: See `demo_christen_biela.py`

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Shift+Enter | Run cell and move to next |
| Ctrl+Enter | Run cell (stay) |
| A | Insert cell above |
| B | Insert cell below |
| DD | Delete cell |
| M | Convert to markdown |
| Y | Convert to code |

## Support

Questions? Open an issue: https://github.com/upb-lea/transistordatabase/issues
