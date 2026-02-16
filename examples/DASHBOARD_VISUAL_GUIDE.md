# Dashboard Visual Guide

**Visual reference for expected notebook outputs and plots**

This guide shows what each section of the dashboard produces, helping users understand what to expect.

## Section 1: Database Loading

### Expected Output
```
✓ Imports successful
Working directory: /home/tinix/claude_wsl/transistordatabase

✓ Database directory found: /home/tinix/claude_wsl/transistordatabase/transistors_merged

📊 Found 245 transistors in database

Loading transistors...
  Loaded 50/245...
  Loaded 100/245...
  Loaded 150/245...
  Loaded 200/245...

✓ Successfully loaded 245 transistors
⚠ Failed to load 0 transistors (see load_errors list)
```

**Time**: 20-30 seconds

---

## Section 2: Database Summary

### Expected Output
```
======================================================================
DATABASE SUMMARY STATISTICS
======================================================================

📊 Total Devices: 245

🔌 Device Types:
MOSFET        120
SiC-MOSFET     68
IGBT           45
GaN            12

🏭 Top 10 Manufacturers:
Infineon       52
Wolfspeed      38
ROHM           24
STMicroel...   19
ON Semi...     15
Vishay         12
Toshiba        10
Microsemi       8
GaN Syste...    7
Transphorm      6

📦 Top 10 Package Types:
TO-247         89
TO-220         45
SOT-227        23
Module         18
TO-263         15
QFN            12
D2PAK          10
DFN             8
TO-3P           7
unknown        18

⚡ Voltage Classes (V_max):
<100V            8
100-300V        24
300-600V        42
600-900V        56
900-1200V       78
1200-1700V      32
1700-3500V       5
>3500V           0

📈 Data Completeness:
  Switch channel data:      198 (80.8%)
  Diode channel data:       145 (59.2%)
  E_on data:                132 (53.9%)
  E_off data:               128 (52.2%)
  E_rr data:                 89 (36.3%)
  C_oss curves:             189 (77.1%)
  C_iss curves:             156 (63.7%)
  C_rss curves:             142 (58.0%)
  Gate charge curves:       167 (68.2%)

⚙️  Performance Metrics (where available):
  R_ds(on) available:       198 devices
    Range: 5.20 - 850.50 mΩ
    Median: 45.30 mΩ
  Q_g available:            167 devices
    Range: 8.5 - 320.0 nC
    Median: 52.0 nC
  FOM (R*Q_g) available:    145 devices
    Median FOM: 234.5 mΩ·nC

======================================================================
```

**Insight**: ~77% of devices have capacitance data, making them suitable for analytical modeling.

---

## Section 3: Interactive Filters

### Widget Layout
```
┌─────────────────────────────────────────┐
│         Filter Controls                 │
├─────────────────────────────────────────┤
│ Device Type:         [MOSFET         ]↕ │
│                      [SiC-MOSFET     ]  │
│                      [IGBT           ]  │
│                                          │
│ Min Voltage (V):     [====|=========] 0 │
│ Max Voltage (V):     [===========|] 10k │
│ Min Current (A):     [|==============] 0│
│                                          │
│ Manufacturer:        [All            ]↕ │
│                                          │
│ Data Completeness Requirements           │
│ [ ] Require channel data                 │
│ [ ] Require switching loss data          │
│ [✓] Require capacitance curves           │
│ [ ] Require gate charge curves           │
│                                          │
│         [ Apply Filters ]                │
└─────────────────────────────────────────┘
```

### After Clicking "Apply Filters"
```
✓ Filtered to 189 devices (from 245 total)

Device type breakdown:
MOSFET        95
SiC-MOSFET    52
IGBT          34
GaN            8

Top manufacturers:
Infineon      45
Wolfspeed     32
ROHM          19
STMicroel...  15
ON Semi...    12
```

---

## Section 4.1: FOM Scatter Plot

### Plot Description
**Interactive Plotly scatter plot with:**
- **X-axis**: Gate charge Q_g (log scale, nC)
- **Y-axis**: R_ds(on) (log scale, mΩ)
- **Colors**: Device type (MOSFET=blue, SiC-MOSFET=red, IGBT=green, GaN=purple)
- **Size**: Voltage rating (larger = higher voltage)
- **Contours**: Diagonal dashed lines for FOM=10, 50, 100, 500, 1000, 5000 mΩ·nC

### Visual Representation
```
R_ds(on) (mΩ)
1000 ┤                     ●
     │               FOM=1000
     │          ●    ╱
 100 ┤     ●    ╱ ● ╱ FOM=100
     │   ● ╱ ● ╱  ●
     │ ● ╱ ●╱  ╱
  10 ┤ ╱●  ╱ ● ╱ FOM=10
     │╱  ●  ╱
   1 ┤  ●  ╱
     └──────────────────────
      10  50 100   500 1000
           Q_g (nC)

● MOSFET    ● SiC-MOSFET
● IGBT      ● GaN
```

### Hover Tooltip Example
```
┌─────────────────────────────┐
│ Device: C2M0080120D         │
│ Type: SiC-MOSFET            │
│ Manufacturer: Wolfspeed     │
│ V_max: 1200 V               │
│ R_ds(on): 80 mΩ             │
│ Q_g: 53 nC                  │
│ FOM: 424 mΩ·nC              │
└─────────────────────────────┘
```

### Expected Output Text
```
✓ Plotted 145 devices with FOM data
  Best FOM: 45.2 mΩ·nC (EPC2036_GaN_100V)
  Worst FOM: 5240.8 mΩ·nC (IGBT_legacy_device)
```

---

## Section 4.3: Capacitance Overlay

### Plot Description
**Three-panel subplot showing:**
1. **Left**: C_oss(V) for selected devices
2. **Center**: C_iss(V) for selected devices
3. **Right**: C_rss(V) for selected devices

### Visual Representation
```
C_oss(V)              C_iss(V)              C_rss(V)

5000 pF ┤───╲        5000 pF ┤───╲        500 pF ┤───╲
        │    ╲──               │    ╲──             │    ╲──
1000 pF ┤      ───╲  1000 pF ┤      ───╲   100 pF ┤      ───╲
        │         ╲─           │         ╲─          │         ╲─
 100 pF ┤           ───       100 pF ┤           ──  10 pF ┤           ──
        └─────────────        └─────────────        └─────────────
         10  100 1000V         10  100 1000V         10  100 1000V

Legend:
── Device 1  ── Device 2  ── Device 3
```

### Key Observations
- **C_oss decreases with voltage** (voltage-dependent capacitance)
- **C_iss relatively flat** (mostly C_gs, less voltage-dependent)
- **C_rss tracks with C_oss** (Miller capacitance)

---

## Section 5: ChristenBielaModel Results

### Single Operating Point Output

```
======================================================================
CHRISTEN-BIELA MODEL: C2M0080120D
======================================================================

[Capacitances at V_0=600V]
  C_oss_eq = 189.45 pF
  C_iss_eq = 2150.32 pF
  C_rss_eq = 52.18 pF
  C_gs = 2098.14 pF
  C_ds = 137.27 pF
  C_gd = 52.18 pF
  Q_oss = 53.24 nC

[Transconductance Model]
  k_1 = 0.458
  k_2 = 0.012
  x = 2.134
  V_th = 3.52 V

[Operating Conditions]
  V_0      = 600 V
  I_0      = 20.0 A
  V_g,on   = 20 V
  V_g,off  = -5 V
  R_g      = 10.0 Ω
  L_s      = 4.0 nH (package: TO-247)
  L_d      = 10.0 nH

[Turn-On Results]
  E_on = 145.32 µJ
  I_oss = 3.45 A
  V_mil = 8.32 V
  t_ri = 12.5 ns (current rise)
  t_fv = 18.3 ns (voltage fall)
  g_m,1b = 2.845 S
  g_m,3b = 1.234 S

[Turn-Off Results]
  E_off = 98.67 µJ
  I_oss = 3.45 A
  V_mil = 8.32 V
  t_rv = 15.2 ns (voltage rise)
  t_fi = 10.8 ns (current fall)
  g_m = 1.987 S

[ZVS Boundary]
  I_0,ZVS = 6.90 A
  ✗ Hard switching (I_0 > I_0,ZVS): Losses occur

[Total Switching Energy]
  E_total = 243.99 µJ/cycle
  E_on / E_total = 59.6%
  E_off / E_total = 40.4%

[Comparison with Measured Data]
  E_on: Measured=152.00 µJ, Model=145.32 µJ, Error=-4.4%
  E_off: Measured=105.50 µJ, Model=98.67 µJ, Error=-6.5%

======================================================================
```

### Interpretation
- **E_on > E_off**: Typical for MOSFETs (body diode recovery in complementary switch)
- **Error < 10%**: Excellent model accuracy
- **I_0 > I_0,ZVS**: Hard switching regime, losses expected

---

## Section 5.1: Current Sweep Plot

### Plot Description
**Interactive line plot with:**
- **X-axis**: Load current (A)
- **Y-axis**: Switching energy (µJ)
- **Blue line**: E_on (model)
- **Red line**: E_off (model)
- **Green line**: E_total (model)
- **Blue squares**: E_on (measured)
- **Red squares**: E_off (measured)

### Visual Representation
```
E (µJ)
  500 ┤                                    ╱●─ E_total
      │                              ╱●───╱
  400 ┤                        ╱●───╱
      │                  ╱●───╱
  300 ┤            ╱●───╱
      │      ╱●───╱                  ●─── E_on (model)
  200 ┤╱●───╱                   ■─── E_on (meas)
      │                    ●─── E_off (model)
  100 ┤──●───●───●───●    ■─── E_off (meas)
      │
    0 └──────────────────────────────────
      5   10   15   20   25   30   35  40
                 I_0 (A)
```

### Key Observations
1. **E_on and E_off increase roughly linearly with current**
2. **Model tracks measured data closely** (squares on lines)
3. **E_total dominated by E_on at high current**

### Expected Output Text
```
✓ Current sweep complete (20 points)
```

---

## Section 6: CSV Export

### Output Format
```csv
name,type,manufacturer,housing,v_max,i_max,i_cont,r_ds_on,q_g,fom,has_channel_sw,...
C2M0080120D,SiC-MOSFET,Wolfspeed,TO-247,1200,36,36,0.080,5.3e-08,4.24e-09,True,...
IMZA120R007M1H,SiC-MOSFET,Infineon,TO-247,1200,202,202,0.007,2.8e-07,1.96e-09,True,...
IPW60R045CP,MOSFET,Infineon,TO-247,600,60,60,0.045,1.2e-07,5.40e-09,True,...
```

### File Created
```
✓ Exported 189 devices to: /path/to/filtered_devices.csv
```

---

## Common Visual Patterns

### High-Quality Device Indicators
1. **FOM plot**: Device in lower-left quadrant (low R_ds, low Q_g)
2. **Capacitance**: C_oss drops steeply with voltage
3. **Model accuracy**: Error < 15% vs measured data
4. **Data completeness**: All checkboxes ticked

### Technology Comparison
| Technology | FOM Range | C_oss Behavior | E_on/E_off Ratio |
|------------|-----------|----------------|------------------|
| Si MOSFET | 200-1000 | Moderate slope | 1.2-1.5 |
| SiC MOSFET | 50-500 | Steep drop | 1.4-1.8 |
| GaN | 10-100 | Very steep | 1.1-1.3 |
| IGBT | 500-5000 | Flat | 0.5-0.8 (E_off > E_on) |

### Voltage Class Patterns
- **600V devices**: Dense FOM cluster at 100-500 mΩ·nC
- **1200V devices**: Broader spread, 200-1000 mΩ·nC
- **1700V+ devices**: Sparse, 500-3000 mΩ·nC

---

## Interactive Features Demo

### Plotly Toolbar (Top Right of Plots)
```
🔍 Zoom  📷 Download  🏠 Reset  ↕️ Pan  📊 Autoscale
```

**Actions**:
- **Click + Drag**: Zoom to region
- **Double-Click**: Reset view
- **Hover**: Show tooltip
- **Click Legend**: Toggle trace visibility
- **Camera Icon**: Save as PNG

### Widget Interactions
```
Filter Update Flow:
1. Adjust sliders/dropdowns
2. Click "Apply Filters"
3. df_filtered updated
4. Re-run plot cells to see changes
```

---

## Performance Expectations

| Operation | Time | Note |
|-----------|------|------|
| Load database | 20-30s | One-time per session |
| Apply filters | <1s | Instant update |
| Generate FOM plot | 1-2s | 145 devices |
| Capacitance overlay | 1-2s | 3 devices × 3 panels |
| Analytical model | <0.1s | Single point |
| Current sweep | 2-3s | 20 points |
| CSV export | <0.5s | 189 devices |

---

## Troubleshooting Visual Issues

### Issue: Plots show but not interactive
**Solution**: Install/enable ipywidgets
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Issue: Empty plots
**Solution**: Check filtered dataset size
```python
print(f"Filtered devices: {len(df_filtered)}")
```

### Issue: Analytical model error
**Solution**: Select device with complete data
- Check: "has_c_oss" = True
- Check: "has_channel_sw" = True

---

## Best Visual Results

### For Publication-Quality Plots:
1. Filter to 5-10 devices for comparison
2. Use capacitance overlay with distinct voltage classes
3. Export Plotly plots as SVG (vector graphics)
4. Current sweep with measured data overlay

### For Device Selection:
1. Start with FOM plot (identify low-FOM cluster)
2. Click 3-5 devices in cluster
3. Compare capacitance curves
4. Run analytical model on finalists
5. Export filtered list with rankings

---

**End of Visual Guide**
