# sc archive capability review and FOM migration

Reviewed on 2026-10-15 against the user's archive at
`\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc`.

The current application did not provide equivalent catalog/FOM exploration.
The new FOM & Pareto view addresses that concrete gap. Full archive parity is
still incomplete. A menu entry or an existing Python class is not evidence
that the corresponding browser workflow works.

## Evidence and boundaries

`sc-archive-function-inventory.json` inventories all 134 Python, MATLAB and
notebook files found recursively, including duplicate copies and notebook
checkpoints. It records 314 definitions, 32 Python stub definitions and five
Python files with syntax errors. These are definition counts, not 314 distinct
working capabilities. The inventory records source hashes and line locations;
it is structural evidence, not execution or numerical equivalence evidence.

The two requested notebooks were inspected cell by cell, including their saved
output types. `tests/pareto_sc.ipynb` contains shared-source Bokeh market-survey
scatter plots. It does not calculate a non-dominated set. Its
`calc_calc_relative_cost` helper assigns `Price` to `rdson`, and its regex
cleaning can strip decimal points and SI prefixes. The replacement recomputes
figures from primary source cells and uses an actual two-objective dominance
calculation.

`docs/Mosfet_switching_losses.ipynb` documents capacitance decomposition, gate
RC charging, transconductance, Miller plateau intervals and a current plot.
It mixes Python and Octave syntax, references `Vpl` before its defining cell,
and ends with a turn-off heading without a corresponding implementation.
The companion `semiconductors/mosfet.py` contains undefined variables and
inconsistent waveform interval expressions. The scientific intent is valuable;
copying those cells is not sufficient validation of a new switching simulator.

## Catalog workflow now available

Open the app's **FOM & Pareto** navigation entry, or use
`http://localhost:8002/index.html?view=catalog-fom` while the local FastAPI service runs.
The Vite development URL is `http://localhost:5174/?view=catalog-fom`.

| Source | Worksheet | Imported rows |
| --- | --- | ---: |
| 201210_sic_digikey.xlsx | in | 286 |
| SiC_MOSFET_FOM.xlsx | 1200V | 174 |
| SiC_MOSFET_FOM.xlsx | 600V | 1,137 |
| SiC_MOSFET_FOM.xlsx | Tabelle8 | 233 |
| 201210_mos_digikey.csv | CSV | 500 |
| 201210_gan_digikey.csv | CSV | 99 |
| SemiconductorsData.xlsm | MOSFET | 23 |
| SemiconductorsData.xlsm | IGBT | 38 |

The 2,490 rows remain separate quotes/observations. Part numbers appearing in
different sources, package/quantity offers or worksheet rows are not merged.
Summary-only sheets and blank sheets are not treated as device catalogs.

- Technology, manufacturer, part text, voltage range and current filters.
- Configurable X/Y objectives with independent minimize/maximize directions.
- Rds × Qg, Rds × effective Coss, price × Rds / Vdss², voltage × current,
  voltage × current / price and primary catalog parameters.
- Four linked scatter views, hover details, click selection, logarithmic
  axes, zoom/pan, reset and PNG export.
- Exact Pareto candidate highlighting, including equal ties, and a candidate
  filter. Highlighting in companion plots refers to the chosen objectives.
- A paginated table, source-row details, original conditions and cells,
  cached spreadsheet values and original formulas.
- CSV export with SI metrics, currency, snapshot, quantity and conditions;
  JSON export with request, full result, assumptions and provenance.
- Missing-operand exclusion counts and per-metric coverage.
- Invalid filters clear previous results; outdated responses cannot replace
  the latest selection.

### Calculation and source rules

The workbook's cost FOM is recomputed as `price * Rds(on) / Vdss**2` using
each record's own blocking voltage. For `SiC_MOSFET_FOM.xlsx`, `1200V!AF2`,
the original formula is `=I2/(1200^2/U2)` and the independently recomputed
value is `2.926217777777778e-7 CHF·Ω/V²`.

Bare numeric manual MOSFET Rds(max) cells use mΩ and the first explicitly
recorded gate voltage/temperature pair. Qg(typ) uses nC, and effective Coss
uses pF. Missing maximum resistance is not filled with a typical value or a
different-temperature extrapolation. IGBT rCE is not treated as MOSFET Rds.
Original cells retain other operating-condition groups and thermal/cost
formulas, but those formulas are not executed or presented as new validated
thermal predictions.

The German workbook explicitly labels CHF. Other supplied sources do not
establish currency, so they are labeled unspecified. `@ qty=0` is retained as
an unconfirmed quote tier. Comparisons are isolated to one source/worksheet;
no currency conversion, price refresh, quantity normalization or condition
normalization is implied. Current/power strings use their first explicitly
recorded condition, which remains visible. The voltage-current product is a
rating product, not usable converter power.

Source workbooks remain unchanged. No macros or arbitrary formulas execute.
The portable snapshot includes original paths, SHA-256 hashes, worksheet and
row identifiers, so normal use does not depend on the archive remaining at
its original location. It does not create curve-complete TDB devices from
catalog ratings.

## Wider capability comparison

| Legacy evidence | Current evidence | Remaining parity gap |
| --- | --- | --- |
| `tests/pareto_sc.ipynb`, five catalog sources | New tested Python FOM operation, FastAPI routes and browser explorer | Cross-source quote/condition normalization and brush/lasso selection are not implemented. |
| `docs/Mosfet_switching_losses.ipynb`, `semiconductors/mosfet.py`, `gate_driver.py` | `analytical_models.py` contains gate-charge timing and semiconductor parameter models | The notebook's interval waveforms and explanatory parameter exploration are not available as an equivalent validated browser workflow. |
| `semiconductors/biela_swlosses.py` and example code | `ChristenBielaModel`, capacitance integration, transconductance fitting and reverse-recovery extraction exist in Python | `/api/analytical/biela` imports `BielaModel`/`BielaModelParams`, which no longer exist. Its form does not supply the full current model parameters. It needs a deliberate contract repair and browser validation. |
| `semiconductors/igbt.py`, diode fitting and `extract_linear_model_IGBT.m` | `IgbtModel` and core channel/linearized models | Equivalent fitting, slopes, condition handling and parameter visualizations are not demonstrated in the browser. |
| `semiconductors/helpers.py`, `calc_Pcond_Semi.m`, `calc_Psw_Semi*.m` | `waveform_losses.py`, topology calculations and existing headless tests | Arbitrary waveform upload, gate-edge inspection and linked loss visualization are not demonstrated as one browser workflow. |
| `SemiconductorTemperature.m`, thermal/interface properties and derating helpers | Core Foster models and partial topology steady-state estimates | Shared thermal networks, transient temperatures and current/frequency derating require independent fixtures and a browser workflow. |
| `ParallelSemiconductors.m`, `ParallelIGBTandDiode.m` | Topology equal-sharing switch/diode counts | General device-pair combinations and exact legacy current-sharing semantics remain unverified. |
| `scplecs.py`, PLECS conversion/editor MATLAB functions | `plecs_importer.py`, export services, browser import/export components | Python import tests pass; complete archive-library round trips and browser editing parity have not been established here. |
| `utils/dpt`, LTspice log/raw helpers, `helpers_plot.py` waveform/surface viewers | `utils/ltspice_dpt.py` and DPT Manager component | The web route calls `LTspiceDPT()` without its required config, then calls the file-generating API as if it returned text. DPT waveform/surface viewer parity remains open. |
| DAB, SRC-ZCS, SRC-ZVS, LLC, bridgeless PFC, six-switch PFC, interleaved/VR PFC and inverter MATLAB functions | Current Buck, Boost, inverting Buck-Boost and Flyback web/topology operations | These converter families are not covered by the current four-topology workflow. Their equations and operating regimes need separate numerical acceptance cases. |
| MATLAB data pickers, editors and format conversion utilities | Browser curve editor/uploader, TDB repositories and adapters | File-format migrations, digitization, legacy data editing and export equivalence remain to be verified individually. |

This review did not run Qt, MATLAB, Octave, SPICE, or archive simulations.
The archive's stubs and broken files do not establish complete prior execution
coverage. Conversely, passing current Python tests does not establish browser
parity for these open workflows.

## Reproduction

Use the existing Conda `ntbees2` environment. The read-only workbook importer
also requires `openpyxl`, already available in the inspected environment.

```bash
conda run -n ntbees2 python -m scripts.import_sc_catalogs /home/tinix/claude_wsl/archive_ntbees2/sc
conda run -n ntbees2 python scripts/audit_sc_archive.py /home/tinix/claude_wsl/archive_ntbees2/sc docs/sprints/sc-archive-function-inventory.json
```

The portable data is in `transistordatabase/data/catalogs/archive_sc.json`.
Science is in `transistordatabase/catalog_fom.py`; transport is in
`transistordatabase/gui_web/backend/catalog_routes.py`; Vue owns rendering.

## Validation

- 181 headless Python tests pass across catalog FOM/import, analytical models,
  waveform losses, gate resistance interpolation, PLECS import, LTspice DPT
  helpers and topology operations/API. Run with `--noconftest` to avoid the
  repository's legacy Qt fixture.
- Pareto tests compare all four objective-direction combinations against an
  independent brute-force definition, including ties. Unit cases cover
  decimal commas, mA, kV, mΩ, nC, pF, missing values and invalid domains.
- Real-source tests verify the workbook formula/cached-value agreement,
  10,000-unit quote context, manual MOSFET units and IGBT separation.
- Three Chromium browser workflows pass against the real FastAPI service,
  including source switching, filters, four charts, source-cell inspection,
  log-axis controls, JSON/CSV/PNG content, empty results and invalid-filter
  handling. The compiled app also opens directly from FastAPI on a narrow
  viewport with four rendered charts.
- The FastAPI `/` route retains its API metadata response. The compiled
  browser entry point is `/index.html`, covered by the browser test.
- The shared app shell now constrains its flex width and reduces padding on
  narrow screens so the explorer's responsive chart grid fits the viewport.
- Production Vite build passes with the existing large-bundle warning.
  `--emptyOutDir false` preserves existing build files.
- Browser screenshots of the populated overview and selected device details
  are visually inspected. No Qt runtime is used for migration validation.
