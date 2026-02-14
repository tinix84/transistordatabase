# Transistor Database - Interactive Tutorial

Learn Transistor Database through hands-on examples! 🎓

## Table of Contents

1. [Tutorial 1: Your First Search](#tutorial-1-your-first-search)
2. [Tutorial 2: Comparing Transistors](#tutorial-2-comparing-transistors)
3. [Tutorial 3: Adding a New Transistor](#tutorial-3-adding-a-new-transistor)
4. [Tutorial 4: Exporting for Simulation](#tutorial-4-exporting-for-simulation)
5. [Tutorial 5: Designing a Buck Converter](#tutorial-5-designing-a-buck-converter)

---

## Tutorial 1: Your First Search

**Goal:** Find a suitable SiC MOSFET for a 650V, 30A application.

**Time:** 5 minutes

### Step 1: Open the Application

Launch the web interface and navigate to the **Search Database** tab.

You should see all available transistors listed.

### Step 2: Filter by Type

1. Find the "Type" filter section
2. Check the ☑️ checkbox next to "Type:"
3. Select **"SiC-MOSFET"** from the dropdown

The list now shows only SiC MOSFETs.

### Step 3: Add Voltage Range

1. Find the "Max Voltage" filter section
2. Check the ☑️ checkbox
3. Enter:
   - **Min:** 600
   - **Max:** 700

The list now shows SiC MOSFETs rated 600-700V.

### Step 4: Add Current Range

1. Find the "Continuous Current" filter
2. Check the ☑️ checkbox
3. Enter:
   - **Min:** 25

The list now shows devices that can handle at least 25A continuously.

### Step 5: Review Results

You should see several transistors that match:
- ✅ SiC-MOSFET technology
- ✅ 600-700V voltage rating
- ✅ ≥25A current rating

**Example Results:**
- CREE_C3M0060065J (650V, 60A)
- UnitedSiC_UF3SC065007K4S (650V, 70A)

### Step 6: View Details

Click on a transistor name to see:
- Complete electrical specifications
- Thermal properties
- Switching characteristics
- Available curves and models

### ✅ Checkpoint

You've learned to:
- Navigate to Search Database
- Apply type filters
- Set voltage/current ranges
- Review search results

**Next:** Learn to compare transistors side-by-side!

---

## Tutorial 2: Comparing Transistors

**Goal:** Compare three 1200V IGBTs to find the best for your application.

**Time:** 10 minutes

### Step 1: Navigate to Comparison Tools

Click the **"🔍 Comparison Tools"** tab at the top.

### Step 2: Select First Transistor

1. Find the first dropdown (Transistor 1)
2. Click to open the list
3. Select **"Infineon_FF300R12KE3"**

You'll see its specifications appear in the first column.

### Step 3: Select Second Transistor

1. Find the second dropdown (Transistor 2)
2. Select **"Infineon_FF200R12KE3"**

Now you see both transistors side-by-side!

### Step 4: Select Third Transistor (Optional)

1. Find the third dropdown (Transistor 3)
2. Select **"Semikron_SKM400GB12T4"**

All three transistors are now compared.

### Step 5: Analyze Electrical Ratings

Look at the comparison table:

**Max Voltage (V_abs_max):**
- Infineon FF300: 1200V
- Infineon FF200: 1200V
- Semikron SKM400: 1200V
→ All equivalent ✅

**Continuous Current (I_cont):**
- Infineon FF300: **300A** 🏆
- Infineon FF200: 200A
- Semikron SKM400: 400A 🏆🏆
→ Semikron has highest current rating

### Step 6: Compare Thermal Performance

**R_th_cs (Case-to-Sink):**
- Infineon FF300: 0.05 K/W 🏆
- Infineon FF200: 0.08 K/W
- Semikron SKM400: 0.06 K/W
→ FF300 has best thermal performance

### Step 7: Review Charts

Scroll down to see visual comparisons:

**Chart 1: Voltage Rating**
- Bar chart showing voltage capabilities
- All equal in this case

**Chart 2: Current Rating**
- Clear visual of current differences
- SKM400 leads

**Chart 3: Thermal Resistance**
- FF300 shows advantage
- Lower is better for cooling

### Step 8: Interpret Results

**For your application:**

If you need **maximum current** (400A):
→ Choose **Semikron SKM400GB12T4**

If you need **best thermal performance** (limited cooling):
→ Choose **Infineon FF300R12KE3**

If you need **cost-effectiveness** (200A is enough):
→ Choose **Infineon FF200R12KE3**

### Step 9: Export Comparison

1. Click **"Export Comparison"** button
2. Choose format (PDF or Excel)
3. Save the comparison report

Use this report in design documentation!

### ✅ Checkpoint

You've learned to:
- Select multiple transistors for comparison
- Read comparison tables
- Interpret visual charts
- Make data-driven device selection
- Export comparison reports

**Next:** Learn to add your own transistor data!

---

## Tutorial 3: Adding a New Transistor

**Goal:** Add a new transistor from a datasheet to the database.

**Time:** 20 minutes

**You'll need:** A transistor datasheet (we'll use a hypothetical example)

### Step 1: Navigate to Create Form

Click the **"➕ Create Transistor"** tab.

You'll see an empty form ready for data entry.

### Step 2: Enter Metadata

Start with basic information:

**Name:** `Example_NewTransistor_100V`
- Use format: Manufacturer_PartNumber
- Must be unique

**Type:** Select **"MOSFET"** from dropdown

**Manufacturer:** `Example Semiconductors`

**Housing Type:** `TO-220`
- Common packages: TO-220, TO-247, D2PAK, etc.

**Author:** `Your Name`

**Comment:** `Added from datasheet rev 1.2, 2024-01-15`
- Document your source!

### Step 3: Enter Electrical Ratings

From the datasheet "Absolute Maximum Ratings" section:

**V_abs_max:** `100` (V)
- Drain-Source Breakdown Voltage: 100V

**I_abs_max:** `80` (A)
- Pulsed Drain Current: 80A

**I_cont:** `40` (A)
- Continuous Drain Current at 25°C: 40A

**T_j_max:** `175` (°C)
- Maximum Junction Temperature: 175°C

### Step 4: Enter Thermal Properties

From the datasheet "Thermal Characteristics" section:

**R_th_cs:** `0.5` (K/W)
- Case-to-Sink Thermal Resistance: 0.5 K/W

**R_th_total:** `1.2` (K/W)
- Junction-to-Ambient (with heatsink): 1.2 K/W

**Housing Area:** `0.0012` (m²)
- Measured from package dimensions

**Cooling Area:** `0.001` (m²)
- Effective cooling surface

### Step 5: Add Switch Channel Data

From the datasheet "Output Characteristics" graph:

**At 25°C, V_gs = 10V:**

Create a new channel entry:
- **V_g:** `10` (V)
- **T_j:** `25` (°C)
- **Dataset Type:** `graph_i_e`

**I-V Points** (read from graph):
```
V_ds (V)    I_d (A)
0.0         0
0.1         10
0.2         20
0.5         35
1.0         40
```

Enter these as arrays:
- **Voltage array:** `[0, 0.1, 0.2, 0.5, 1.0]`
- **Current array:** `[0, 10, 20, 35, 40]`

### Step 6: Add Switching Loss Data

From the datasheet "Switching Characteristics" graph:

**Turn-On Energy at V_supply = 80V, V_gs = 10V, R_g = 10Ω:**

Create E_on entry:
- **V_supply:** `80` (V)
- **V_g:** `10` (V)
- **T_j:** `25` (°C)
- **R_g:** `10` (Ω)

**Energy vs Current Points:**
```
I_d (A)     E_on (µJ)
0           0
10          50
20          95
30          135
40          170
```

Enter as arrays:
- **Current array:** `[0, 10, 20, 30, 40]`
- **Energy array:** `[0, 50e-6, 95e-6, 135e-6, 170e-6]`

**Repeat for E_off** (Turn-Off Energy)

### Step 7: Add Capacitance Data

From the datasheet "Capacitance" graph:

**C_oss vs V_ds:**

Create C_oss entry:
- **V_g:** `0` (V) - Capacitance measured with gate shorted
- **Dataset Type:** `graph_c_v`

**Points from graph:**
```
V_ds (V)    C_oss (pF)
0           2000
10          1200
25          800
50          500
100         350
```

Enter as arrays:
- **Voltage array:** `[0, 10, 25, 50, 100]`
- **Capacitance array:** `[2000e-12, 1200e-12, 800e-12, 500e-12, 350e-12]`

### Step 8: Validate Entry

Before saving, check:
- ✅ All required fields filled
- ✅ Units are correct
- ✅ Arrays have same length
- ✅ Values are reasonable (positive, in expected range)
- ✅ Name is unique

### Step 9: Save

Click **"Save"** button.

If successful:
- ✅ Confirmation message appears
- You're returned to Search Database
- Your new transistor is in the list!

If error:
- ❌ Red error message shows problem
- Fix the indicated field
- Try saving again

### Step 10: Verify

1. Go to **Search Database**
2. Type your transistor name in search
3. Click to view details
4. Verify all data looks correct

### ✅ Checkpoint

You've learned to:
- Navigate to Create form
- Enter complete transistor data
- Add electrical ratings
- Add thermal properties
- Add characteristic curves
- Validate and save
- Verify the new entry

**Next:** Learn to export for simulation tools!

---

## Tutorial 4: Exporting for Simulation

**Goal:** Export a transistor to PLECS for power electronics simulation.

**Time:** 10 minutes

### Step 1: Select Transistor

**Option A: From Search**
1. Go to Search Database
2. Find transistor: **"CREE_C3M0060065J"**
3. Click **"Load to Exporting"** button
4. You're taken to Exporting Tools with transistor pre-selected

**Option B: Direct Selection**
1. Go to Exporting Tools tab
2. Select **"CREE_C3M0060065J"** from dropdown

### Step 2: Choose Export Format

You'll see format buttons:
- 📄 JSON
- 📊 MATLAB
- ⚡ PLECS
- 📈 Simulink
- 🦎 GeckoCIRCUITS
- 📋 Datasheet (PDF)

Click **"⚡ PLECS"** button.

### Step 3: Configure Export Options

PLECS export dialog appears:

**Options:**
- ☑️ Include thermal model
- ☑️ Include switching losses
- ☑️ Include capacitances
- Format: PLECS Thermal Model XML

Leave all checked for complete model.

### Step 4: Export

Click **"Export"** button.

Your browser downloads:
```
CREE_C3M0060065J_PLECS.xml
```

### Step 5: Import to PLECS

Now in PLECS:

1. Open PLECS Standalone or Blockset
2. Go to **"Thermal"** → **"Thermal Description Library"**
3. Click **"Import"**
4. Select downloaded `CREE_C3M0060065J_PLECS.xml`
5. The transistor appears in your library!

### Step 6: Use in Simulation

In your PLECS circuit:

1. Add **"MOSFET"** block
2. Right-click → **"Specify..."**
3. Click **"Load from library"**
4. Select **"CREE_C3M0060065J"**
5. The block now uses real transistor data!

### Step 7: Run Simulation

Your simulation now includes:
- ✅ Actual R_ds_on vs temperature
- ✅ Real switching losses (E_on, E_off)
- ✅ Thermal RC network
- ✅ Accurate capacitances

Results will be much more realistic than generic MOSFET model!

### Step 8: Export Other Formats

Try exporting to other tools:

**For MATLAB:**
1. Select **"📊 MATLAB"**
2. Export creates `.m` file
3. Load in MATLAB: `transistor = load('CREE_C3M0060065J.m')`
4. Access data: `transistor.v_abs_max`, etc.

**For Simulink:**
1. Select **"📈 Simulink"**
2. Export creates `.mat` file
3. Load in Simulink model
4. Use in Simscape Power Systems blocks

**For PDF Datasheet:**
1. Select **"📋 Datasheet"**
2. Export creates professional PDF
3. Includes all plots and specifications
4. Perfect for documentation!

### ✅ Checkpoint

You've learned to:
- Select transistor for export
- Choose appropriate export format
- Configure export options
- Import to simulation tool
- Use exported data in circuits

**Next:** Design a complete power converter!

---

## Tutorial 5: Designing a Buck Converter

**Goal:** Design a 400V to 48V, 25A Buck converter and verify performance.

**Time:** 25 minutes

### Step 1: Specify Requirements

**Design Specifications:**
- Input Voltage: **V_in = 400V** (DC bus)
- Output Voltage: **V_out = 48V** (battery)
- Output Current: **I_out = 25A** (load)
- Switching Frequency: **f_sw = 100kHz**
- Target Efficiency: **>95%**

### Step 2: Select Transistor

Go to **Search Database** and filter:
- **Type:** MOSFET or SiC-MOSFET
- **Max Voltage:** 500V to 700V (safety margin)
- **Continuous Current:** >30A (margin above 25A)

**Good Options:**
- CREE_C3M0060065J (650V, 60A, SiC)
- Infineon_IPBE65R050CFD7A (650V, 50A)

Choose **CREE_C3M0060065J** for high efficiency.

### Step 3: Navigate to Topology Calculator

Click **"🧮 Topology Calculator"** tab.

### Step 4: Configure Calculator

**Select Topology:**
Choose **"Buck Converter"** from dropdown.

**Select Transistor:**
Choose **"CREE_C3M0060065J"**.

### Step 5: Enter Operating Point

Fill in parameters:

**Input Voltage (V_in):** `400` V

**Output Voltage (V_out):** `48` V

**Output Current (I_out):** `25` A

**Switching Frequency (f_sw):** `100000` Hz (100 kHz)

**Gate Resistance (R_g):** `10` Ω
- Start with 10Ω (typical)
- Will optimize later with slider

### Step 6: Review Calculated Results

The calculator shows:

**Operating Point:**
- **Duty Cycle:** D = 48/400 = **12%**
  - Switch is ON only 12% of the time
  - Makes sense: big voltage step-down

- **Peak Inductor Current:** **27.5 A**
  - Slightly above average (25A)
  - Need to check I_abs_max

- **RMS Inductor Current:** **25.2 A**
  - Close to average, good for CCM

**Automatically Calculated Components:**
- **Inductance:** **L = 180 µH**
  - For 10% current ripple at 100kHz

- **Capacitance:** **C = 47 µF**
  - For 1% voltage ripple

### Step 7: Analyze Losses

**Switch Losses:**
- **Conduction Loss:** 3.2 W
  - From R_ds_on × I²_rms × D
  - Low due to low duty cycle and SiC R_ds_on

- **Turn-On Loss:** 8.5 W
  - From E_on × f_sw
  - Main loss component at 100kHz

- **Turn-Off Loss:** 6.3 W
  - From E_off × f_sw

- **Total Switch Loss:** **18.0 W**

**Diode Losses:**
- **Conduction Loss:** 24.5 W
  - V_f × I_avg × (1-D)
  - Higher due to high (1-D) = 88%

- **Reverse Recovery:** 2.2 W
  - Minimal with SiC MOSFET body diode

- **Total Diode Loss:** **26.7 W**

**Total Loss:** **44.7 W**

### Step 8: Calculate Efficiency

**Output Power:** P_out = 48V × 25A = **1200 W**

**Input Power:** P_in = 1200W + 44.7W = **1244.7 W**

**Efficiency:** η = 1200 / 1244.7 = **96.4%** ✅

**Exceeds our 95% target!**

### Step 9: Check Thermal Performance

**Junction Temperature:**

Calculator shows: **T_j = 85°C**

Calculation:
```
T_j = T_ambient + P_loss × R_th_total
T_j = 25°C + 44.7W × 0.5K/W
T_j = 25°C + 22.4°C
T_j = 47.4°C
```

With heatsink R_th_sa = 1.3 K/W:
```
T_j = 25°C + 44.7W × (0.5 + 1.3)K/W
T_j = 25°C + 80°C
T_j = 105°C
```

**Safety Check:**
- T_j = 105°C
- T_j_max = 175°C
- Margin = **70°C** ✅ Safe!

### Step 10: Optimize Gate Resistance

Use the **R_g slider** to optimize:

**Try R_g = 5Ω:**
- Switching losses decrease to 12W
- Efficiency improves to **96.9%**
- But expect more ringing

**Try R_g = 15Ω:**
- Switching losses increase to 19W
- Efficiency drops to **95.8%**
- Cleaner waveforms

**Optimal: R_g = 8Ω**
- Switching losses: 14.2W
- Total losses: 40.9W
- Efficiency: **96.7%**
- Good trade-off

### Step 11: Verify Component Stresses

**Switch Stress:**
- V_ds_max = 400V (< 650V rating) ✅
- I_d_peak = 27.5A (< 60A rating) ✅
- T_j = 105°C (< 175°C max) ✅

**Diode Stress:**
- V_r_max = 400V (same device) ✅
- I_f_avg = 25A × 0.88 = 22A ✅

**Inductor:**
- I_L_rms = 25.2A
- I_L_peak = 27.5A
- Choose inductor rated >30A

**Capacitor:**
- V_rated > 60V (safety margin)
- I_ripple_rms ≈ 2.5A
- Choose low-ESR capacitor

### Step 12: Review Waveforms

**Plot 1: Current Waveforms**
- Inductor current: triangular, 25A ±10%
- Switch current: 27.5A peak during ON time
- Diode current: 27.5A peak during OFF time

**Plot 2: Voltage Waveforms**
- Switch voltage: 0V (ON) / 400V (OFF)
- Clean switching with R_g = 8Ω
- Some ringing visible (normal)

**Plot 3: Loss Breakdown**
- Switch: 16W (35%)
- Diode: 25W (55%)
- Capacitor ESR: 4W (10%)

**Insight:** Diode loss is dominant!
→ Consider synchronous rectification (replace diode with another MOSFET)

### Step 13: Export Design

Click **"Export Calculation"**

Choose format:
- **PDF Report** - Complete design documentation
- **Excel** - Design calculations table
- **MATLAB** - Script to reproduce

Save as: `Buck_400to48V_25A_Design.pdf`

### Step 14: Design Summary

**Final Design:**

**Topology:** Non-isolated Buck Converter

**Components:**
- Switch: CREE C3M0060065J (SiC MOSFET, 650V, 60A)
- Diode: Body diode (or external Schottky for lower loss)
- Inductor: 180µH, 30A rated, <50mΩ DCR
- Capacitor: 47µF, 100V, low-ESR
- Gate Driver: 10A peak, 8Ω gate resistance

**Performance:**
- Efficiency: 96.7% at full load
- Power Loss: 40.9W
- Junction Temperature: 105°C (70°C margin)
- Switching Frequency: 100kHz
- Output Ripple: <1% (480mV p-p)

**Next Steps:**
1. Select specific inductor and capacitor parts
2. Design gate driver circuit
3. PCB layout (minimize loop area)
4. Prototype and test
5. Validate efficiency curve

### ✅ Checkpoint

You've learned to:
- Specify converter requirements
- Select appropriate transistor
- Use Topology Calculator
- Analyze losses comprehensively
- Calculate efficiency and temperature
- Optimize gate resistance
- Review waveforms
- Verify component stresses
- Export complete design report
- **Design a complete power converter!** 🎉

---

## What's Next?

### Continue Learning

1. **Try Different Topologies**
   - Design a Boost converter (12V to 400V)
   - Design a Buck-Boost converter

2. **Explore Advanced Features**
   - Multi-temperature comparison
   - Foster thermal model analysis
   - SOA verification

3. **Optimize Designs**
   - Trade-off studies (frequency vs size vs loss)
   - Cost optimization
   - Thermal management strategies

### Real-World Projects

**Project 1: Solar MPPT**
- Boost converter
- 30V to 400V
- Variable input
- High efficiency required

**Project 2: Battery Charger**
- Buck converter
- 400V to 14.4V
- Current-controlled
- Thermal management critical

**Project 3: Bidirectional DC-DC**
- Buck-Boost
- Battery to DC bus
- Four-quadrant operation
- Efficiency in both directions

### Share Your Work

- Export and document your designs
- Contribute transistor data to the database
- Share efficiency results
- Report bugs or suggest features

---

## Feedback

**Completed the tutorial?** Great! 🎉

We'd love to hear:
- What did you find most helpful?
- What could be clearer?
- What tutorials would you like to see?

**Contribute:**
- Add your transistor data
- Share design examples
- Improve documentation
- Report issues on GitHub

---

**Happy designing! ⚡**
