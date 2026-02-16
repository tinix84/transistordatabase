# Transistor Database Performance Dashboard

## Overview

This interactive Jupyter notebook provides a comprehensive analysis and visualization platform for the transistor database, featuring live plotting, advanced filtering, and integrated analytical modeling using the **Christen-Biela switching loss model**.

**File**: `transistordatabase_performance_dashboard.ipynb`

## Features

### 1. Database Overview Dashboard
- **Summary Statistics**: Total devices, device type distribution, manufacturer breakdown
- **Voltage Classes**: Distribution across voltage ratings (600V, 1200V, 1700V, etc.)
- **Current Ratings**: Device current capability analysis
- **Data Completeness Metrics**: Percentage of devices with:
  - Channel characteristics (I-V curves)
  - Switching loss data (E_on, E_off, E_rr)
  - Capacitance curves (C_oss, C_iss, C_rss)
  - Gate charge curves (Q_g)

### 2. Interactive Filtering
Filter devices by multiple criteria:
- **Device Type**: MOSFET, SiC-MOSFET, IGBT, GaN
- **Voltage Range**: Min/Max voltage sliders (0-10,000V)
- **Current Range**: Minimum current rating (0-1,000A)
- **Manufacturer**: Multi-select from all manufacturers
- **Package Type**: Housing selection
- **Data Completeness**: Require specific measurement data

### 3. Live Performance Plots

#### Figure of Merit (FOM) Analysis
- **R_ds(on) vs Q_g scatter plot** with log-log scale
- **FOM contour lines**: Constant R×Q_g values (10, 50, 100, 500, 1000, 5000 mΩ·nC)
- Device type color coding
- Bubble size by voltage rating
- Interactive hover with device details

#### Voltage vs Current Rating
- Distribution of devices across V-I space
- Log-log scatter with device type markers
- FOM-weighted sizing

#### Capacitance Curves Overlay
- **Multi-device comparison**: C_oss(V), C_iss(V), C_rss(V)
- Side-by-side subplot layout
- Log-scale axes for wide voltage range
- Interactive legend

#### Channel Characteristics
- **Output characteristics**: I_ds vs V_ds at multiple V_gs
- Temperature-dependent curves
- Single device detailed view

### 4. Christen-Biela Analytical Model Integration

The notebook integrates the state-of-the-art **Christen-Biela analytical switching loss model** (IEEE TPEL 2019):

#### Features:
- **Turn-on energy calculation**: E_on with phase breakdown
- **Turn-off energy calculation**: E_off with timing analysis
- **ZVS boundary**: Zero-voltage switching threshold current
- **Detailed breakdown**:
  - I_oss: Output capacitance charging current
  - V_mil: Miller plateau voltage
  - t_rv, t_fi: Voltage rise and current fall times
  - g_m: Operating point transconductance

#### Operating Conditions:
- DC bus voltage (V_0)
- Load current (I_0)
- Gate drive voltages (V_g,on, V_g,off)
- Gate resistance (R_g)
- Package parasitics (L_s, L_d) - auto-detected from housing type

#### Model Inputs (Extracted from Database):
- **Capacitances**: C_gs, C_ds, C_gd from C(V) curves
- **Q_oss**: Charge stored in output capacitance
- **Transconductance model**: Fitted from transfer characteristics
- **Package inductances**: Lookup table by housing type

#### Current Sweep Analysis:
- Generate E_on(I), E_off(I), E_total(I) curves
- Overlay measured data (if available) for validation
- Compare analytical predictions vs datasheet values
- Configurable current range and number of points

### 5. Export & Save Capabilities
- **CSV Export**: Save filtered device lists with all metrics
- **Plot Export**: High-resolution figure saving (Plotly)
- **Results Tables**: Christen-Biela detailed breakdown

## Installation

### Prerequisites

```bash
# Core dependencies
pip install jupyter jupyterlab

# Data science stack
pip install numpy pandas matplotlib plotly scipy

# Interactive widgets
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension

# Transistor database (if not already installed)
pip install -e .
```

### For JupyterLab (recommended):
```bash
pip install jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Usage

### 1. Launch Notebook

```bash
# From transistordatabase root directory
jupyter notebook transistordatabase_performance_dashboard.ipynb

# Or with JupyterLab
jupyter lab transistordatabase_performance_dashboard.ipynb
```

### 2. Run Cells Sequentially

1. **Setup & Database Loading** (Cells 1-2)
   - Imports libraries
   - Loads all 245 transistors from `transistors_merged/`
   - Displays load status

2. **Database Summary** (Cell 3)
   - Automatic statistics generation
   - No user input required

3. **Interactive Filters** (Cell 4)
   - Configure filter widgets
   - Click "Apply Filters" to update dataset
   - `df_filtered` variable updated globally

4. **Performance Plots** (Cells 5-8)
   - Run cells to generate plots
   - FOM scatter, V-I distribution, capacitance overlay, I-V curves
   - All plots use filtered dataset

5. **Christen-Biela Model** (Cells 9-10)
   - Configure operating conditions
   - Select device with good data coverage
   - Click "Run Analytical Model"
   - Run current sweep for E(I) curves

6. **Export Results** (Cell 11)
   - Specify CSV filename
   - Click "Export to CSV"

### 3. Best Practices

#### Device Selection for Analytical Model:
- **Minimum requirements**:
  - C_oss curve (mandatory)
  - Channel data with ≥3 V_gs points (for transconductance fitting)
- **Recommended**:
  - C_iss and C_rss curves
  - Switching loss data (E_on, E_off) for validation
  - Gate charge curve

#### Filter Settings for High-Quality Devices:
```python
# Example filter configuration
Device Type: ['MOSFET', 'SiC-MOSFET']
Min Voltage: 600V
Max Voltage: 1200V
✓ Require capacitance curves
✓ Require channel data
```

#### Operating Conditions:
- **V_dc**: Match device voltage rating (e.g., 600V for 1200V device)
- **I_0**: 10-50% of device current rating
- **R_g**: 5-20Ω typical for MOSFETs, adjust based on application
- **V_g,on**: +15V to +20V for MOSFETs, +15V for IGBTs
- **V_g,off**: -5V to 0V for MOSFETs, -15V for IGBTs

## Technical Details

### Data Extraction

The notebook extracts the following from each transistor:

```python
# Metadata
- name, type, manufacturer, housing_type

# Electrical ratings
- v_abs_max, i_abs_max, i_cont

# Performance metrics
- r_ds_on: Extracted from channel data linear region
- q_g: Total gate charge at V_gs=15V
- fom: R_ds(on) × Q_g figure of merit

# Data completeness flags
- has_channel_sw, has_e_on, has_e_off, has_c_oss, etc.
```

### Christen-Biela Model Implementation

The model follows the 2019 IEEE TPEL paper by Christen & Biela with errata corrections:

**Reference**: T. Christen and J. Biela, "Analytical Switching Loss Modeling Based on Datasheet Parameters for MOSFETs in a Half-Bridge," IEEE Transactions on Power Electronics, vol. 34, no. 4, pp. 3700-3710, 2019.

#### Model Pipeline:

1. **Capacitance extraction**:
   ```python
   C_eq = (1/V_0) * ∫[0,V_0] C(v) dv  # Charge-equivalent capacitance
   C_gs = C_iss_eq - C_rss_eq
   C_ds = C_oss_eq - C_rss_eq
   C_gd = C_rss_eq
   ```

2. **Transconductance fitting**:
   ```python
   I_ch = k_1 × (V_gs - V_th)^x + k_2  # Transfer characteristic
   g_m(I_ch) = (k_1 × I_ch^x / (I_ch - k_2))^(1/x)
   ```

3. **Turn-off energy** (Phases 2a-2c):
   - Phase 2a: Voltage rise (I_oss charging C_oss)
   - Phase 2b: Current fall (Miller plateau)
   - Phase 2c: Post-switching

4. **Turn-on energy** (Phases 1a-1d):
   - Phase 1a: Current rise
   - Phase 1b: Voltage fall (initial)
   - Phase 1c: C_oss discharge (I_oss)
   - Phase 1d: Final voltage fall

5. **ZVS boundary**:
   ```python
   I_0,ZVS = 2 × I_oss  # Threshold for lossless turn-off
   ```

### Accuracy & Limitations

**Model Accuracy**:
- ±10-30% for switching energy (typical)
- Better for hard-switching (high I_0)
- Less accurate near ZVS boundary

**Limitations**:
- Assumes hard-switched half-bridge topology
- Neglects dead-time effects
- Simplified body diode reverse recovery
- No temperature dependence
- Linear gate driver (constant R_g)

**Best Use Cases**:
- Device comparison at same operating point
- Trend analysis (E vs I, E vs R_g, E vs V_dc)
- FOM validation (R_ds × Q_g vs switching loss)
- Quick estimates when measurement data unavailable

## Example Workflow

### Scenario: Find best SiC MOSFET for 600V, 20A application

1. **Filter devices**:
   ```
   Device Type: SiC-MOSFET
   Min Voltage: 600V
   Max Voltage: 1200V
   Min Current: 15A
   ✓ Require channel data
   ✓ Require capacitance curves
   ```

2. **View FOM plot**:
   - Identify devices with low R_ds × Q_g
   - Note top 3-5 candidates

3. **Run analytical model**:
   - V_dc = 600V
   - I_0 = 20A
   - R_g = 10Ω
   - V_g,on = +20V, V_g,off = -5V

4. **Compare switching losses**:
   - Run current sweep (5A - 50A)
   - Compare E_total curves
   - Validate with measured data

5. **Check capacitance curves**:
   - Overlay C_oss(V) for finalists
   - Lower C_oss → lower E_oss

6. **Export results**:
   - Save filtered list as CSV
   - Document best device

## Troubleshooting

### Issue: "No devices with capacitance curves"
**Solution**: Uncheck "Require capacitance curves" or expand voltage/current range

### Issue: "Insufficient channel data for transconductance fitting"
**Solution**: Model uses default g_m parameters. Select different device with ≥3 V_gs curves.

### Issue: "Negative capacitance warning"
**Solution**: C_iss < C_rss in some devices. Model clamps to zero. Results may be inaccurate.

### Issue: Plots not interactive
**Solution**:
```bash
# Install/update ipywidgets
pip install --upgrade ipywidgets
jupyter nbextension enable --py widgetsnbextension

# For JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Issue: Slow loading (>30 seconds)
**Solution**: Normal for 245 devices. Consider filtering in Cell 2:
```python
# Load only first 100 devices for testing
transistor_names = repository.list_all()[:100]
```

## Output Files

### CSV Export Format
```csv
name,type,manufacturer,housing,v_max,i_max,i_cont,r_ds_on,q_g,fom,has_channel_sw,...
C2M0080120D,MOSFET,Cree,TO-247,1200,36,36,0.080,53e-9,4.24e-9,True,...
```

### Columns:
- **name**: Transistor identifier
- **type**: MOSFET, SiC-MOSFET, IGBT, GaN
- **manufacturer**: Vendor name
- **housing**: Package type (TO-247, TO-220, etc.)
- **v_max, i_max, i_cont**: Electrical ratings (V, A)
- **r_ds_on**: On-resistance (Ω)
- **q_g**: Gate charge (C)
- **fom**: Figure of merit R×Q_g (Ω·C)
- **has_***: Boolean flags for data availability

## Performance

- **Load time**: ~10-30 seconds (245 devices)
- **Filter update**: <1 second
- **Plot generation**: 1-3 seconds
- **Analytical model**: <100ms per operating point
- **Current sweep**: ~2 seconds (20 points)

## Future Enhancements

Potential additions:
- Multi-device current sweep comparison
- Thermal resistance analysis (R_thJC, R_thCA)
- Reverse recovery analysis (E_rr model)
- Gate charge curve comparison
- SOA (Safe Operating Area) plots
- Pareto frontier analysis (multi-objective optimization)
- Export to PDF report

## Contact & Support

For questions, bug reports, or feature requests:
- **Repository**: https://github.com/upb-lea/transistordatabase
- **Issues**: https://github.com/upb-lea/transistordatabase/issues
- **Documentation**: https://transistordatabase.readthedocs.io

## Citation

If you use this dashboard in your research, please cite:

```bibtex
@software{transistordatabase,
  title = {Transistor Database},
  author = {LEA, University of Paderborn},
  year = {2024},
  url = {https://github.com/upb-lea/transistordatabase}
}

@article{christen2019analytical,
  title={Analytical Switching Loss Modeling Based on Datasheet Parameters for MOSFETs in a Half-Bridge},
  author={Christen, Thomas and Biela, J{\"u}rg},
  journal={IEEE Transactions on Power Electronics},
  volume={34},
  number={4},
  pages={3700--3710},
  year={2019},
  publisher={IEEE}
}
```

## License

This notebook is part of the transistordatabase project. See main repository for license details.
