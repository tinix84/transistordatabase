# Transistor Database - Vue.js Web Interface

This is the modern web interface for the Transistor Database, built with Vue 3 + Vite and FastAPI backend.

## Features

🌐 **Modern Web Interface** with comprehensive transistor database management:

- **🔍 Advanced Search Database**: Multi-criteria filtering, range-based property selection, manufacturer/type filters
- **📤 Device data downloads**: Complete native JSON, ratings summary CSV, MATLAB MAT5, and conditioned GeckoCIRCUITS SCL archives
- **🧮 Topology Calculator**: Buck/Boost/Buck-Boost converter design with power loss analysis  
- **🔍 Comparison Tools**: Side-by-side transistor analysis with interactive charts
- **📊 Database Management**: Import/export, statistics, data validation
- **🌙 Dark/Light Theme**: Customizable interface with localStorage persistence
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

## Install the headless backend

Use the existing Conda environment. Run these commands from the repository root:

```bash
conda activate ntbees2
python -m pip install -e .
python -c "import openpyxl, spicelib; from transistordatabase.gui_web.backend.main import app"
```

The backend dependencies include `openpyxl` for workbook import and `spicelib` for LTspice RAW and log import.
Qt is an optional legacy desktop dependency under the `legacy-qt` extra. The local web interface requires no Qt installation.

## Quick Start

1. **Start Backend** with the repository's Conda environment (from the repository root):
```bash
conda activate ntbees2
python -m uvicorn transistordatabase.gui_web.backend.main:app --host 127.0.0.1 --port 8002
```
The backend launcher scripts use this full FastAPI application, including the topology routes. They do not create an environment or install dependencies.
`dev-server.bat` uses `pushd` for directory changes so the backend and frontend can also start from a UNC checkout.

2. **Start Frontend** (from root `gui_web/` directory):
```bash
npm run dev
```
Install the frontend dependencies separately before the first run.

3. **Access Application:**
- Web Interface: http://localhost:5174
- API Documentation: http://localhost:8002/docs

## Import a catalog source

Open **Catalog source import**, or use `/index.html?view=catalog-sources` on the local backend.

1. Select a CSV, TSV, XLSX or XLSM file.
2. Declare the CSV encoding, separator and decimal convention.
3. Select **Preview source**. Inspect sheets, hidden rows, formulas, saved caches and errors.
4. Select the worksheet and header row. Map the part number and each required field with an explicit unit.
5. Enter recorded quote and test conditions. Leave unknown values empty.
6. Select **Create catalog snapshot**. Download the catalog JSON or select **Explore snapshot in FOM**.

The catalog JSON retains exact source bytes, source hash, mapping and row errors.
**Download exact source bytes** returns the original file.
Changing the file, mapping, units or context invalidates the current snapshot.
The browser does not modify the source file or refresh external data.

Use **Inspect static links** to inspect link evidence without opening targets.
An optional cell batch selects cells on the selected worksheet.
Relative targets refer to the original workbook. GetURL formulas remain unevaluated metadata.
Use **Inspect J/A coefficients** for the documented MOSFET and IGBT layouts.
Each coefficient retains its reference conditions, literal formula, saved cache and status.

To compare a refresh, upload before and after catalog JSON snapshots.
Select explicit comparison keys, then select **Compare snapshots**.
Use the row key to disambiguate repeated part numbers when source row identity is appropriate.
The report lists added, changed and missing records. Comparison does not overwrite a catalog.
Select **Explore after snapshot in FOM** to inspect the supplied after population explicitly.

## Explore workbook benchmarks

Open **FOM & Pareto**, or use `/index.html?view=catalog-fom` on the local backend.

1. Select the catalog sources and objective metrics.
2. Open **Workbook estimates and source declarations**.
3. For thermal estimates, read and accept the constant source model and conduction-only assumptions.
4. Declare a source currency only when you have evidence for that currency. No currency conversion applies.
5. Declare the source chip area unit before selecting an area–thermal metric. The archived area header has no unit.
6. Select a part to inspect the metric values, missing-input reasons, source conditions and calculation evidence.
7. Export JSON or CSV to retain declarations, source evidence and normalization populations.

The thermal model uses a 100 °C case boundary for MOSFETs and a 100 °C heatsink boundary for IGBTs.
It holds the recorded conduction model constant at its source temperature and gate voltage.
Thermal current is a screening estimate, not a measured rating or actual power capability.
The test-voltage / aggregate energy-current coefficient benchmark is not a maximum switching frequency.
All three switching coefficients must be known. A blank coefficient remains missing.
IGBT differential rCE products remain separate from MOSFET Rds(on) products.
Chip area products require declared units and retain the source thermal path. They are not guaranteed material properties.

For a supplied source, select **Conduction model profile** in Catalog source import.
Select each numeric unit explicitly, including area units, and map the model conditions and quote origin.
The default profile is unknown. Selecting a profile does not supply missing measurements or coefficients.
Original source bytes and stored catalog data remain unchanged.

## Estimate gate resistance from a target slew

1. Open **More**, **Analytical Models**, then **Device Parameter Operations**.
2. Select **Gate resistance from target voltage slew**.
3. Enter the positive target slew magnitude in V/s and minimum reverse-transfer capacitance in F.
4. Enter the driver rails and Miller plateau interval. Replace the illustrative source with your evidence.
5. Select the estimate acknowledgment and calculate the result.

Each selected result is the minimum total series resistance that keeps the predicted slew within the target over the plateau interval.
Greater resistance reduces the predicted slew under the constant-capacitance model.
The min/max fields are extrema of the per-plateau resistance requirements, not allowable resistance limits.
Total resistance includes driver, internal and external resistance. The result does not directly specify an external resistor.
The source equation reference is SC `mfiles/calcRg.m`, evidence ID `sc-92afe9737e47b855`.

## Inspect and map a legacy MAT file

Select **Legacy MAT** in the navigation bar, or use `/index.html?view=legacy-mat` on the local backend.

1. Select an original `.mat` file within the 24 MiB source limit.
2. Inspect the source hash, schema, diagnostics, units, and original field tree.
3. Expand a trace image field. Select **View original image** to inspect the original pixels.
4. Inspect the Python calibration comparisons. Preserve the historical `x/image-height,y/image-width` coordinate convention.
5. Enter independent identity, ratings, areas, and their source citation.
6. Select each supported branch mechanism and its core target. Prefer original branch traces over synthetic combined arrays.
7. Declare each branch's conditions, evidence reference, sample order, zero policy, and coordinate convention.
8. Exclude outside-calibration samples with original indices, or explicitly select a calibration extrapolation estimate.
9. Give a reason for each omitted mechanism. Confirm the proposed SI units.
10. Select **Validate explicit core model**. Inspect the Python finalized records.
11. Select a mapped conduction target. Enter a positive current within its recorded coverage.
12. Select **Calculate bounded point**. Inspect the Python voltage, resistance, conditions, and coverage.
13. Download the point CSV, point JSON, or native core JSON.

The native core document retains the original MAT bytes and declarations. The operation does not write to the database.
Changing declarations invalidates the core result and calculated point. Replacing the source invalidates the preview and declarations.
The filename does not supply identity, ratings, temperature, or test conditions.

Use **Download source archive JSON** for the versioned preview and exact source bytes.
Use **Download exact original MAT** for byte-identical archival preservation.
Use **Download semantic MAT5 v0.2** for supported SC compatibility conversion.
Keep new scientific declarations in native core JSON. MAT5 compatibility conversion does not encode them.

Heat-sink, individual trace, and numeric workspaces support preview and exact archival preservation only.
The core mapping supports explicit SC branch subsets. It rejects synthetic energy placeholders and unresolved negative energy.
A finite all-zero combined energy curve requires an unverified-source acknowledgement or a scoped ideal-model declaration.
Neither option establishes measured status. The exact empty-trace placeholder remains unsupported.

## Import measurement waveforms or supplied integrals

Select **Measurement import**, or open `/index.html?view=measurements` on the local backend.

1. Select CSV, LTspice RAW, or LTspice log as the source format.
2. Upload the original source file. For stepped RAW, upload its matching companion log.
3. Declare the source type and model disclosure. For CSV, declare the encoding, separator, and decimal convention.
4. Select **Inspect source columns and steps**.
5. Map each waveform trace and its recorded unit. Declare each probe direction explicitly.
6. For RAW, select one step and declare stored SI units. For stepped CSV, select the exact step value.
7. For logs, map supplied energy measurements and current/bus variables with units. Select the step for derived core points.
8. Declare known test conditions. Leave unknown conditions as **Unknown**. A blank **Known** value is invalid.
9. Optionally upload an existing native device JSON to attach the imported raw record and energy points.
10. Select **Preview mapped SI data**. Inspect the source hash, aligned samples, statistics, or supplied integral rows.
11. Select **Analyze and prepare review**. Inspect each local event, integration window, threshold times, and energy.
12. Download the full measurement review JSON, energy CSV, or native device JSON.

Waveform extraction requires a gate trace and declared bus/gate rails.
Its event window uses local 1–99% voltage/current transitions without conduction subtraction.
Log results remain simulator-supplied integrals with original expressions and FROM/TO bounds. They contain no reconstructed gate evidence.
The log preview includes all source records. Analysis creates core points for the selected step only.
Failed log measurements retain their failure status and reason. Missing values remain unknown.

The review JSON preserves source bytes, hashes, parser versions, mappings, conditions, and raw/derived records.
Native device JSON requires an existing device archive. It preserves exact native integer representations during browser download.
Review and export do not save data to the user database or start a simulator.
Generic simulation does not establish measured device accuracy.
Changes to files, mappings, units, steps, disclosures, or conditions invalidate analysis and exports.
Preview the mapping again after each change. Display range changes affect all waveform time axes together.
The server parser capability report gives the active source and sample limits.

## Public Python calculations and data exports

Core `Switch` and `Diode` methods select exact recorded temperatures and gate voltages.
`get_channel_data(t_j, None)` selects an explicitly unknown gate voltage.
It does not select every gate voltage. Zero volts is a separate known condition.
`linearize_at_operating_point(t_j, v_g, current)` requires one matching curve and current within its coverage.
The local line is `V = V0 + R I`. Interior knots use the right segment.
Endpoints use the one-sided recorded segment. Signed current and voltage remain signed.
`update_working_point(..., component="switch")` accepts `switch`, `diode`, or `both`.
If any selected calculation fails, the previous working point remains unchanged.

```python
from pathlib import Path
from transistordatabase.backend.concrete_services import ExportService

service = ExportService()
# transistor is an existing core Transistor with recorded source data.
service.export_to_matlab(transistor, Path("fresh-mat-export"))
service.export_to_gecko_circuits(transistor, {
    "component": "switch",
    "t_j": 25,
    "v_supply": 600,
    "v_g_on": 15,
    "v_g_off": -4,
    "r_g_on": 2,
    "r_g_off": 3,
    "output_path": Path("fresh-gecko-export"),
})
```

These example conditions must match the device records. They are not device defaults.
MATLAB export writes `<safe_name>_Matlab.mat` as MAT5.
Its named device structure uses the legacy transistor database dictionary fields.
Unknown values use `NaN` in this numeric view.
The supplementary `core_json` variable preserves complete native evidence, nulls, dates, and array shapes.
This format differs from SC `scData` and the Simulink loss-model format.

```python
import json
from scipy.io import loadmat
from transistordatabase.core.serialization import core_from_json_document

archive = loadmat("fresh-mat-export/test_part_Matlab.mat", simplify_cells=True)
restored = core_from_json_document(json.loads(archive["core_json"]))
```

Gecko export writes `<safe_name>_Switch.scl` and a source-evidence JSON manifest.
The supported subset contains one switch channel slice and recorded turn-on/turn-off energy-current curves.
`v_g_on` matches core `v_g` for both events. `v_g_off` matches their separate core `v_g_off` field.
Explicit `None` gate or resistance conditions match only recorded unknown values.
The shared energy grid retains source knots within the common current interval.
The exporter rejects duplicate conditions, disjoint coverage, missing curves, and nearest-condition `recheck` requests.
It does not mirror channel branches or add zero-energy curves.
Diode SCL export requires forward-recovery energy that the core diode model does not provide.
Single-point energies, inferred conditions, and Simulink models are outside this export subset.
Format checks do not establish acceptance by MATLAB applications or GeckoCIRCUITS simulations.

Both exporters use explicit paths and refuse existing target files.
They preserve the source device and do not change the process working directory.
Use a fresh destination for each export. Preserve the Gecko manifest with its SCL file.
Filesystem failures can leave reserved or partially written files. Existing files are never replaced.

## Technology Stack

- **Frontend**: Vue 3, Vite, Chart.js, Axios
- **Backend**: FastAPI, Python 3.8+
- **Database**: JSON file-based with 25+ transistors
- **Styling**: CSS Variables for theming, responsive design

## Vue 3 + Vite

This template uses Vue 3 `<script setup>` SFCs. Learn more about [script setup docs](https://v3.vuejs.org/api/sfc-script-setup.html#sfc-script-setup) and [IDE Support for Vue](https://vuejs.org/guide/scaling-up/tooling.html#ide-support).

## Project Structure

```
gui_web/
├── src/
│   ├── components/          # Vue components
│   │   ├── SearchDatabase.vue      # Advanced search interface
│   │   ├── ExportingTools.vue      # Export functionality  
│   │   ├── TopologyCalculator.vue  # Converter calculations
│   │   ├── TransistorComparison.vue # Comparison tools
│   │   └── ...
│   ├── services/            # API services
│   ├── App.vue             # Main application
│   └── main.js             # Application entry
├── backend/                 # FastAPI backend package
│   └── main.py             # Full API server, including topology routes
└── package.json            # Dependencies
```

## Device data downloads

1. Open Exporting Tools.
2. Select the devices with the checkboxes or the database search.
3. Select Native JSON, ratings summary CSV, MATLAB MAT5, or GeckoCIRCUITS.
4. For GeckoCIRCUITS, enter the exact junction temperature, supply voltage, gate voltages, and gate resistances.
5. Enter `null` for a gate condition only when the source records that condition as unknown.
6. Download the selected devices. Multiple devices produce one ZIP archive.

Native JSON retains the complete core record, array shapes, integer precision, conditions, and evidence. The browser downloads the original Python bytes. CSV contains ratings only.

MAT5 contains a numeric legacy view and a complete `core_json` source archive. It is not SC `scData` or a Simulink model.

GeckoCIRCUITS downloads contain the switch SCL file and its JSON evidence manifest. The source must contain one exact switch slice. No nearest-condition selection, mirrored conduction, or inferred switching losses apply. Simulator acceptance is not verified.

Use the PLECS data workbench for source table selection and explicit omissions. Use the DPT workbench for explicit operating conditions and a selected generic template. Generic SPICE and Simulink device models are unavailable.

The HTTP export routes reject filesystem parameters and undeclared options. Batch Gecko exports accept `options.selection` for a common slice or `options.selections` for an exact device-ID mapping. Each slice declares `component`, `t_j`, `v_supply`, `v_g_on`, `v_g_off`, `r_g_on`, and `r_g_off`. Unknown gate conditions require explicit JSON null values.
