# Remaining semiconductor data and visualization contracts

Prepared read-only on 2026-10-16 for ordered step 8 of
`sprint-semiconductor-function-parity.md`. This document is preparation, not phase
acceptance or implementation evidence. Steps 1–4 are accepted in that plan;
step 5 is active, and steps 6/7 have their own thermal/converter contracts.
Their engines must be accepted before step-8 adapters depend on them.

Reference roots, preserved exactly:

```text
\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc
D:\OneDrive\claude\bhy\enco
```

All relative source references below identify one of those roots or this
repository explicitly. No reference script, macro, Power Query, simulator model
or application was executed. Existing current Python readers were exercised
against three read-only reference XML files and one reference log; those probes
are described separately from proposed acceptance tests. ENCO bodies are not
copied or used as runtime dependencies. Workbook details come from the completed
`semiconductor-workbook-function-audit.md`, including its read-only VBA census.

## Architectural boundary and existing dependencies

Use narrow headless Python operations, typed FastAPI transport and Vue views.
Keep reusable calculations independent of HTTP, chart libraries and local paths.
One calculation must supply table, plot and export values. Do not introduce a
general ENCO-style monorepo, a new numerical engine for each chart, automatic
simulator execution, converter waveform synthesis, or another thermal solver.

Reusable interfaces inspected:

| Current interface | Reuse and limit |
| --- | --- |
| `condition_tables.py:15 samples`, `:29 bounded_interpolate`, `:55 require_conditions`, `:64 MeasuredTable`, `:104 separable_energy_estimate` | Validated copied samples, bounded 1-D lookup, exact known/unknown conditions and a named separable estimate. No measured multidimensional interpolation is currently promised. |
| `device_parameters.py:20 fit_linear_vi`, `:58 integrate_capacitance`, `:94 energy_current_coefficient`, `:108 resistance_temperature_estimate` | Already accepted parameter operations; route through these instead of reproducing workbook arithmetic in JavaScript. |
| `biela.py:161 extract_reverse_recovery_params`, `:197 fit_transconductance` | Accepted headless extraction with physical domains, fit coverage and diagnostics. Direct browser extraction is still absent. |
| `waveform_losses.py:316 waveform_statistics`, `:337 waveform_spectrum`, `:407 analyze_waveform` | Active step-5 implementation, not accepted evidence at preparation time. Use its accepted contract later for spectra, waveform evidence and losses. |
| `topologies/operating_point.py` and `gui_web/backend/topology_service.py:20 calculate` | Bounded existing buck/boost/buck-boost/flyback operating-point evaluation. A configuration evaluator may call a supported calculation; it must not invent solutions for other converters. |
| `catalog_fom.py:35 parse_quantity`, `:80 metric_values`, `:100 pareto_ids`, `:128 screen_catalog` | Historical source-isolated screening and exact 2-D Pareto. Normalization is an additional operation, not an alteration of physical metrics. |
| `utils/ltspice_dpt.py:255 analyze_waveforms` | Accepted gate-edge/local-reference extraction with explicit windows. `:386 extract_switching_loss_data` still loses evidence. |
| `gui_web/backend/parameter_routes.py:167 calculate` | Existing typed operation dispatch and context/plot envelope. Extend this small adapter for transfer/recovery rather than add a parallel calculator framework. Line numbers may move during implementation. |

Installed Windows Conda `ntbees2` was inspected without installation: numpy
2.4.6, scipy 1.17.1, xmltodict 1.0.4, Pillow 12.3.0, PyLTSpice 6.0.1 and
spicelib 1.6.4 are present; pandas and openpyxl are absent there. The bundled
Python runtime returned by `load_workspace_dependencies` supplies openpyxl and
was already used for the workbook audit. Do not assume the Windows and WSL
Conda environments contain identical optional parsers. Missing optional readers
need a useful unsupported-format response; no environment creation or installation
is necessary for preparation. A separate WSL `ntbees2` probe confirms installed
openpyxl3.1.5 and pandas; it has no PyLTSpice/spicelib. Runtime spreadsheet import
can therefore stay native in the WSL service. Do not bridge to a Windows Python
process from HTTP merely because the development verification used its optional
RAW/log reader; RAW support needs an explicit optional capability until the
server has an approved parser. Current Vue dependencies include Chart.js and its
zoom plugin; a heatmap can use a small SVG/canvas view plus Python-supplied grid
values instead of adding a large plotting dependency for a single surface.

## 8A. Close live adapters before claiming advanced comparison

There are real defects in supported current adapters, not just missing charts.

| Inspected source | Actual behavior / reachability | Required closure |
| --- | --- | --- |
| `backend/concrete_services.py:296 CalculationService.interpolate_switching_losses` | Nearest T fallback, first matching dataset, clamped `np.interp`, no supplied-voltage/gate-resistance selection, absent data returns 0. Used by `:344 calculate_losses`; the public/headless controller calls that service at `frontend/interfaces.py:433–439`. No direct `gui_web` call was found. | Delegate to accepted bounded selectors with explicit conditions; reject unknown requests. Do not credit this public path as validated because the new parameter workbench is correct. |
| `backend/concrete_services.py:344 calculate_losses` | Hidden defaults and missing conduction suppressed as zero; switching conditions lose voltage and gate context. | Delegate to the accepted static-duty/waveform calculation as appropriate, preserving duties, energy conditions and omission policy. Existing arguments insufficient for a scientifically complete result should fail with a migration message. |
| `backend/concrete_services.py:382 find_optimal_working_point` | Returns constraint values without evaluation or optimization. Declaration/ABC has no other call site found. | Mark unsupported clearly or implement finite declared candidate evaluation through 8B. Do not label returned maxima optimal. |
| `backend/concrete_services.py:595 rank_transistors` | All scores equal 1, criteria unused; no production caller found by symbol search. | Unsupported until a named, unit-consistent objective is supplied; do not count as optimizer parity. |
| `backend/concrete_services.py:629 generate_advanced_comparison` and `gui_web/backend/main.py:307 POST /api/comparison/advanced` | This is a live browser/API path. `:904 _plot_loss_breakdown` uses nearest/tolerated conditions, hardcodes 100 kHz, ignores requested Rg, makes missing values zero. `:986 _plot_efficiency_comparison` fabricates simplified switching loss and a conduction fallback. | Preserve available recorded curves as explicitly conditioned observations. Loss/efficiency panels must call accepted calculations with explicit inputs or return structured unsupported/missing-data state. Never display invented near-100% efficiency. |
| `backend/concrete_services.py:721 _plot_switching_comparison` | First curve within approximately 25 °C /100 V /5 Ω, unknown Rg bypasses check. Other comparison helpers use nearest temperature and label it approximately. | Exact-selection controls or visibly separate recorded slices with their actual conditions. A different measurement is not a result at the requested point. |
| `backend/concrete_services.py:482 export_to_plecs` | Legacy bridge changes process-wide cwd to generate files. Web export routes use this service. | Pure text/bytes rendering or explicit destination paths; no process-wide cwd mutation in concurrent HTTP handlers. |

Thermal adapter defects are already assigned to step 6; do not duplicate that
work here. Generic SPICE/other export routes also require honest model scope:
`export_to_spice:463` currently emits an NPN header for arbitrary device types,
and `export_to_ltspice:528` chooses operating conditions from ratings without
the explicit generic-template choice required by the accepted DPT web route.
Route these to the disclosed supported export contract or fail clearly.

Independent acceptance: two devices with absent switching curves must yield
missing-data states, never zero loss / perfect efficiency; a mismatched T/V/Rg
must not reuse a nearby curve; requests outside current coverage must fail;
explicit fs must change switching W proportionally; advanced-comparison HTTP
responses must preserve individual unsupported panels and actual conditions.
No Qt execution is required to test the public controller/service boundary.

## 8B. Configuration × operating-point evaluation

ENCO evidence: `enco/packages/enco-analysis/src/enco_analysis/circuit/`
`circuit_group/__init__.py:166 generate_configuration_space`,
`:232 generate_operating_point_space`, `:306 filter`, `:348–365` data-frame
adapters and `:378 save_excel`; `optimizer/list.py:25 ListOptimizer`,
`:38 optimize`; `circuit_group/dataframe.py:89 get_indicator_dataframe` and
`:188 get_indicator_merit_dataframe`. `library-design/dcdc/cllc/src/project/`
`optimizer.py:279 normalize_indicators`, `:292 aggregate_operating_points`,
`:310 compute_merit` demonstrates design-grid aggregation, not a universally
validated semiconductor optimizer. The inspected `ListOptimizer` has shared
class-level invalid configurations and invalidates whole configurations from
one failed pre/post point; these defects are not parity requirements.

Lean contract: a finite declared collection of immutable configurations, finite
declared operating points, explicit nonnegative weights and one supported
evaluator. Return one row per pair with configuration/point IDs, full SI inputs,
source IDs, method/version, status, diagnostics, and typed metrics with units.
Known invalid input is a per-pair unsupported/error row; a resource-limit or
unexpected programmer error must not silently become an invalid device. Put
limits on generated pair counts before constructing the full Cartesian product.
Reject duplicate IDs/empty populations and missing objective metrics.

Provide per-configuration weighted means and maxima, keeping the worst-point ID
and per-metric coverage. The default full-population summary is unavailable if
any positive-weight point lacks that metric. Optional successful-subset means
must be explicitly selected and labeled with retained/total weight; never
renormalize silently. A zero-weight failure remains visible without making the
weighted mean incomplete; worst-case reports state whether zero-weight points
are included. A raw weighted sum and a normalized weighted mean are different
operations. Rank by a named physical metric/direction or explicit dimensionless
normalization; do not add watts, euros and volume into an opaque score.

Independent fixture: config A losses [10,30] W at points p1/p2, weights [1,3],
gives weighted mean25 W and worst30 W at p2; B [20,20] gives mean/worst20 W.
If A/p2 fails, A/p1 and all B rows survive; A's full weighted summary is
unavailable, successful-subset selection returns10 W with coverage1/4. Reverse
the input order and confirm stable IDs/results. Run two independent evaluations
to prove no leaked invalid-state list. Include all-zero/negative/nonfinite
weights, excessive products, finite derived-value overflow, and immutable-source
tests. Browser matrix, filters, loss bars and CSV/JSON export must show failures
and coverage without hiding them. A cached evaluation, if implemented, must key
complete inputs/model/source hash/version, not just device/configuration name.

## 8C. Energy grids/surfaces and linked plots

SC evidence: `semiconductors/helpers_plot.py:17 DoublePulseViewerBokeh`,
`:32 add_time_wfm`, `:41 DoublePulseViewerPlotly_old`, `:57 add_time_wfm`,
`:67 add_surface`, `:117 DoublePulseViewerPlotly`; `semiconductors/scplecs.py:219`
`calc_Esw_mesh` hardcodes a 5×6 reshape and returns nothing. Its incomplete body
is not a numerical target. ENCO plotting evidence:
`enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/plot.py:48`
stacked bars, `:295` time series, `:379` frequency series, `:482` grouped bars.
`enco/packages/enco-models/src/enco_models/circuit/plot/pareto.py:22 ParetoPlot`
is only a schema; it does not add a working Pareto renderer to ENCO.

Energy-map contract: view recorded Eon/Eoff/Err slices separately at explicit
Tj, Vg/on/off, Rg, commutation-device/source and energy convention. Build a
rectangular I×V grid only where every selected slice supports each displayed
current and fixed conditions match; reuse bounded current interpolation.
Expose missing cells/multiple ambiguous records, never reshape by assumed size
or fill from nearest T. An optional interpolated I/V surface is a named estimate,
bounded inside a fully covered rectangular cell, with its interpolation method
and original points visible. Do not silently promote separable temperature/Rg
estimates into measured multi-D data. Heatmap + slice selector + table achieves
the useful surface workflow; rotatable 3-D rendering is presentation only.

Independent grid fixture: I=[0,10] A, V=[400,800] V, E at400=[0,.001] J and
at800=[0,.002] J with all other conditions equal. Recorded corners must remain
exact; bounded bilinear estimate at5 A/600 V is.00075 J; missing one corner or
query900 V must produce unsupported status. Retain finite E(0), zero/nonzero
records and duplicate-condition conflicts. Verify no source-array mutation and
no implicit voltage proportionality beyond explicitly chosen interpolation.

Linked time/spectrum/loss contract: reuse step-5 returned samples/statistics/FFT;
select a configuration/point in 8B to display those exact traces plus branch
stacked watts and the same exported rows. Record FFT detrending/window, sampling
and frequency units; reject nonuniform samples unless an explicit resampling
estimate is supplied. Independent coherent sinusoid fixture verifies DC,
amplitude/frequency and power units, not merely that a line appears. Shared
cursor/time zoom must retain axis units and distinguish time-domain selections
from spectral frequency zoom. Source traces and displayed subsets retain their
identity. Step 7 owns converter multiplicities/branch science.

## 8D. PLECS supported import/export and real round trips

SC evidence: `matlab_func/convertPlecsXmlToscData02.m:1`,
`semiconductors/scdata.py:95 load_plecs_xml`,
`semiconductors/scplecs.py:58 load_plecs_xml_to_dict`, `:63` variables and
`:104–210` conduction/energy axis/scale/lookup properties. ENCO's substantive
reader is `enco/packages/enco-components/src/enco_components/components/mosfet/`
`plecs.py:22 xml_to_enco`; its `:246 enco_to_xml` writes example100V/50A
placeholder metadata and is not a valid exporter reference.

Current `plecs_importer.py:28 import_plecs_xml`, `:60 _parse_package`,
`:134 _parse_conduction_loss`, `:178 _parse_switching_loss` handles the simple
default-namespace unit fixture, but does not currently guarantee general PLECS
parity. Specific defects: repeated SemiconductorData becomes a list but the
reader expects a dict; prefixed namespace keys are not normalized; it ignores
ComputationMethod/Formula/Variables/CustomTables; shape mismatches are skipped;
missing axis conditions default to25 °C/0 V/0 Vgate; measured axis extrema are
turned into absolute ratings and continuous current is invented as half maximum
at `:331 _infer_electrical_ratings`. Unknown conditions must remain unknown.

Read-only probes of the actual current importer on 2026-10-16:

| Supplied exact fixture relative to its reference root | Observed result |
| --- | --- |
| SC `db/plecs/Infineon_IGBT/IGW60T120_IGBT.xml` | Two channel curves, four Eon and four Eoff; invented ratings600 V/240 A/120 A continuous. XML lines6–8/21–23 are table coverage, not absolute ratings. |
| SC `db/plecs/Infineon_IGBT/IKW20N60TA_Diode.xml` | Two diode curves, four Err and one zero Eon placeholder; voltage rating becomes0 because recovery supply axis at lines17 is negative/zero. Signed PLECS voltage is a convention, not a negative absolute rating. |
| ENCO `enco/packages/enco-components/tests/mosfet/_mock/G3R40MT12K.xml` | `AttributeError: 'list' object has no attribute 'get'`. Correction verified 2026-10-16: lines119–175 contain one SemiconductorData block with repeated ConductionLoss elements (gate on/off); lines121–122/135–136 are Table and formula; lines149–160 are Formula only. Treating the base tables as a complete imported model would be wrong even after fixing the list crash. |

Lean closure: parse a supported table-only subset into a typed intermediate
document with package identity, axes, scale, signed-voltage convention, conditions,
thermal branch, source SHA and unknown fields. Validate rectangular axis/row
counts, finite values, unique axes, actual namespace URI and multiple packages/
semiconductor blocks. Empty/no tables is explicit missing data. Preserve original
formula/custom-table/variable metadata as uninterpreted source, and either reject
scientific evaluation/export of unsupported formulas or convert only an explicitly
recognized named separable estimate with diagnostics. Never evaluate arbitrary
formula strings, import reference Python, or substitute guessed gate parameters.
Mapping signed recovery voltage to the positive-bus core convention must be
explicit and reversible in provenance. Table current coverage must not become
datasheet current or voltage limits. Ratings remain unknown unless separately
provided. Different blocks representing a switch/body diode need a deliberate
aggregate mapping, not accidental overwrites.

Pure renderer returns namespaced PLECS XML bytes with real identity and supported
table/thermal contents. Round trip a supported source to intermediate→core→XML→
intermediate, comparing semantic arrays/conditions/scales/source mapping rather
than XML whitespace. Unsupported structure should return a structured report
instead of silently dropping it. Do not claim a simulator accepted the exported
model without a separate simulator test. Existing legacy-export helper may be
retained only where the equivalence suite proves its contract; HTTP must not
change process cwd or rely on hidden nearest-condition choices.

Independent acceptance: the two supplied SC files above plus the current
`tests/test_data/sample_plecs_switch.xml`, prefixed/default/no-namespace variants,
single/multiple voltage rows, scaled energies, diode reverse voltage, unknown
gate/Rg, duplicate conditions, malformed row lengths, nonfinite values and a
separate formula-only rejection fixture. G3R40MT12K must give a useful unsupported
formula/custom-table report rather than a crash or plausible false model. Verify
all supported numeric corners and Foster parameters survive round trip, sources
are unchanged, live upload/export/download works, and no placeholder device data
appear. Use read-only source fixtures; generated exports go to fresh destinations.

## 8E. DPT CSV / RAW / log import and provenance

SC `utils/dpt/ltspice_dpt.py:52 import_ltspice_log` constructs stepped Eon/Eoff
records but uses fragile regular expressions, truncating zip semantics and removed
pandas DataFrame.append; `semiconductors/helpers.py:48 calc_sw_losses_ltspice`
has fixed windows, vector resistance subtraction and clipping. Their input/output
intent is substantive; their parsing/integration defects must not be reproduced.
SC contains genuine stepped measurement logs, e.g.
`tests/DoublePulseTester_EPC8009_sync.log` (also duplicated under utils/dpt).
Installed `spicelib.log.ltsteps.LTSpiceLogReader` was read and exercised against
that file without running LTspice. Supported APIs include `get_measure_names()`,
`get_step_vars()`, indexing `reader[name]`, `get_measure_value(...)` and
`get_measure_values_at_steps(...)`. The source contains30 Eoff entries and
`idc`/`vdc` step columns; first three Eoff values are
[1.0141e-7,9.47901e-7,2.69621e-6] J with first three idc=[1,3,5] A at5 V.
Its Eoff integration interval is39–41 µs (`eoff_from/eoff_to`). Those are
historical simulated measurements, not present measured-device accuracy.

Use installed `PyLTSpice.RawRead` / spicelib parser for binary/ascii RAW rather
than writing another binary parser. APIs inspected: get_trace_names(),
get_time_axis(step=0), get_trace(name).get_wave(step=0), get_steps(). Require
explicit trace mapping and per-step selection/conditions. Stepped RAW often
requires companion log; if absent, do not concatenate reset-time steps into one
monotonic waveform. A missing parser in the server runtime is a supported-format
capability response; CSV remains available. Do not launch a simulator on upload.
ENCO `enco/packages/enco-simulation/src/enco_simulation/ltspice/loader.py:32`
is another RAW reader, not a reason to reproduce its parser.

CSV contract: explicitly mapped time, Vds/Vce, Id/Ic and gate columns with units,
separator/decimal convention and sign/probe direction. Use a standard CSV parser,
respect quoted fields, then validate aligned finite increasing samples through
the common engine. No magnitude-based unit guessing, silent sorting, clipping,
polarity correction or resampling. Preview each mapping before extraction.

Log contract: explicit measurement/step variable mapping and known units; retain
FROM/TO and expression text when available. A failed measure remains failed and
is not zero. Without traces the imported result is a supplied integral, not a
re-extracted local-edge energy; do not invent current/gate waveform evidence.

Provenance envelope must retain original filename/hash, parser/version, step,
trace names/probe polarity/unit conversions, source vs simulated status, actual
event identity, absolute integration bounds, integration/edge convention,
thresholds, local current/bus, Tj/on/off gate/Rg/commutation conditions and any
manual override. Generic simulation must retain its generic-model disclosure.
Unknown metadata is explicit. Core `SwitchingLossData:204` currently lacks a
structured provenance field; `RawMeasurementData:139` also lacks gate trace,
source hash and extraction-window fields. Introduce a narrowly typed evidence
record or additive schema field, version serialization/adapter handling, and
associate both raw records and extracted single points with it. Test a complete
save→reload→export, not merely construction of an in-memory dataclass.

`LTspiceDPT.extract_switching_loss_data:386` currently drops DPTResult event
evidence and falls back to target current if measured current is absent. The
new path must reject provenance-free manual results as measured extraction or
label them manual explicitly. Preserve the separate on/off currents and Rg.
The live `GET .../dpt_data` route (`main.py:1496`) calls every switching curve
DPT data, and `POST .../dpt_validate` (`:1526`) tests numpy arrays as booleans
at line1559: ambiguous truth values can raise instead of validating. Repair both
while adding import; existing datasheet data alone is not evidence of DPT.

Independent acceptance: generate a small analytic two-event trace with known
V*I integrals; CSV and RAW mappings of the same trace give equal SI arrays and
event energies. Use the preserved generic RAW from accepted step3 for a real
binary-parser transport regression, retaining generic scope. Use the supplied
30-step EPC log to check exact first values, count, steps and39–41 µs windows.
Failed/partial rows, UTF BOM, missing gate/step/log, reversed current probe,
multiple pulses, unequal sample counts, nonmonotonic time, scientific notation
and overflow need clear outcomes. Browser preview/export must display provenance
and distinguish log integral / waveform extraction / manual point.

## 8F. Calibrated image digitization

SC `mfiles/analyze_image.m:4` calibrates linear/log axes at two coordinate
points; `:51–64` transforms selected pixels. `mfiles/DataPicker.m:1` adds point
editing, labels and saved image/calibration state; `:320 updateValues` applies
linear/log maps. Its mouse-position normalization divides x by image height and
y by width at lines208–209/244–245; preserve intent, not that dimensional bug.
Current web components have no calibrated image-digitization path. Existing
`checker_functions.py:80ff` handles digitizer text data, not image calibration.

Small independent axis maps suffice: each axis has its own two pixel anchors,
two physical values, linear/log scale and unit. X and Y anchors need not be the
same corner pair. For log axes require positive physical anchors and interpolate
in log10 space. Pixel separation must be nonzero; physical anchors distinct;
inputs/outputs finite; image width/height/hash known; origin top-left with x
right/y down recorded explicitly. Reversed pixel or physical direction is valid
and follows the anchor map. Points outside calibrated intervals require explicit
extrapolation selection or rejection; never silently extrapolate.

Browser scales display pixels to original natural-image pixels independently of
devicePixelRatio/CSS zoom. Calibration belongs to that exact image hash/dimensions;
replacing image invalidates it unless identical. Preserve selection order and
pixel coordinates, allow explicit undo/remove/edit, then export physical CSV
plus JSON calibration/image/source/units. No automated curve recognition,
perspective/skew correction or precision claim is needed for this parity scope.
Do not edit the uploaded source image. A curve destination needs explicit core
field, conditions and unit conversion, not guessed capacitance/current axes.

Independent fixture:1000×600 image, x pixel100→900 maps1→100 on log scale;
pixel500 gives10. Independent y pixel500→100 maps0→20 V linearly; pixel300
gives10 V despite downward-positive image coordinates. Change x only and verify
y unchanged. Test linear-linear/log-linear/linear-log/log-log, inverted axes,
equal pixels/nonpositive log values/out-of-image pixels/output overflow, CSS
resizing and image replacement. Browser must demonstrate that a known click
at natural pixel500/300 exports10/10 with calibration identity intact.

## 8G. Direct transfer fitting and recovery extraction adapters

Extend the existing typed parameter operation union and view with two operations;
science remains `biela.py`. Transfer inputs are channel curves at common declared
Tj/Vds and known Vth, with at least four distinct above-threshold gate voltages.
Output fitted k1/k2/x/Vth, current/gate coverage, rank/condition number, residual
plot, RMSE/max residual and exact source conditions. Bounded Vds interpolation
is already inside the core helper. Do not allow a field called gm to substitute
for the nonlinear transfer model. Browser result becomes stale after any input
or condition edit, and use in Biela remains explicitly selected.

Independent transfer fixture already validated headlessly in
`tests/test_biela_contract.py:241–255`: at Vds10 V/Tj25 °C/Vth3 V,
Vg=[4,5,6,7,8] and I=[2.1,8.1,18.1,32.1,50.1] gives k1=2,k2=.1,x=2,
RMSE<1e-8 A, rank3 and supported current2.1–50.1 A. The API/browser must reproduce
those diagnostics and reject missing conditions/flat currents/insufficient rank.

Recovery inputs: positive Qrr, peak Irr, positive fall-slope magnitude, Qoss>=0
and forward test current at the same declared condition; Trr is compatibility
metadata, not an invented exponential-tail cutoff. Return corrected Qrr and
tau_rr/tau_c/Tm with units, equations/scope and source/test metadata. Existing
core fixture (`tests/test_biela_contract.py:221–233`) Qrr100 nC,Irr10 A,
di/dt2.4 GA/s,Qoss12 nC,If20 A gives corrected88 nC,
tau_rr6.716666666667 ns,tau_c13.669756359629 ns,Tm13.204949301187 ns.
Verify the independent charge/residual identity, invalid corrected/fall charges,
transport unit conversion and complete browser JSON export. Do not infer these
parameters from total Eon or a generic device name.

## 8H. Declared-population FOM, typed workbook records and source refresh

The workbook audit lists exact cell families. Summary `SiC_MOSFET_FOM.xlsx`
`Tabelle1!C2:C311` divides each stored B value by the minimum of B. Its A/B
values lack formula-linked part identities; retain the standalone summary source,
not fabricated joins. Add a declared-population normalization operation over
explicit eligible record IDs/source/filter snapshot, metric and direction.
For minimum normalization, every eligible metric must be finite/positive;
excluded records and reasons are reported. Return denominator, denominator IDs,
population hash/list and absolute metric alongside relative values. Decide
explicitly whether filters change the population or only hide displayed points.
Never normalize cross-currency/tier/source quotes without an explicit common
unit/source selection. Existing Pareto results remain the same for positive
scalar normalization of an axis.

Independent FOM fixture: values[2,4,8] ->[1,2,4]; fixed population remains that
after hiding value2; selecting visible-population normalization gives[1,2] for
[4,8] with new population/denominator. Empty/no positive population is unavailable,
missing/zero/negative metrics are excluded with counts, tied minima all identified.

Typed workbook coefficient import is still distinct from the accepted operation
that computes a supplied J/A coefficient. `scripts/import_sc_catalogs.py:33`
currently preserves formulas/raw/cache but normalizes only selected catalog fields.
Define reusable row extraction for explicitly supported `SemiconductorsData.xlsm`
MOSFET/IGBT layouts: preserve cell addresses, literal formula, cached value/error,
units, condition-header references, Eon/Eoff/Err reference current and bus/Tj,
VCE0/rCE vs Rds and historical quote tier/currency unknowns. Recognize only
documented literal divisions/headers; never evaluate arbitrary Excel formulas or
promote J/A to joules. Formula `MOSFET!Y2=.000632/36` gives coefficient
1.7555555556e-5 J/A, not .000632 J at every current. Cache/formula disagreement
must be reported; a missing cache is missing, not recalculation. Thermal budgets/
area screening belongs to step6, not this import helper. Row-specific arbitrary
thermal additions or price multipliers remain source assumptions.

Reusable refresh is an explicit supplied-file import, not execution of the old
Power Query pointing at another person's Downloads folder. Keep the existing
`scripts/import_sc_catalogs.py:146 import_catalogs` as a special known-layout
adapter, factor reusable CSV/XLSX row reading/column mapping into headless code,
and expose preview→validated snapshot/export. Standard CSV quoting, BOM/declared
encoding, explicit delimiter/decimal convention, duplicated-header disambiguation,
source hash/sheet/row/formula/cache and per-row rejected-value reasons are needed.
No silent duplicate-part collapse: source+sheet+row remains stable provenance.
Fresh imports create versioned snapshots; no automatic overwriting of existing
user files or refreshing unknown external locations. Compare added/changed/missing
source rows before an explicit application step. Historical quotes stay historical.

Static hyperlink reader covers verified VBA `Modul1:2 ExtractHyperlinks` and
`:17 GetURL` intent without running macros. Read OOXML hyperlink relationships
and internal target/location/tooltip/display text; select one cell or a batch
with exact sheet/cell IDs; return absent links explicitly. Distinguish stored
link objects, plain URL text and supported literal HYPERLINK formulas; do not
evaluate `GetURL`, arbitrary formulas, web addresses, external workbooks or
embedded scripts. External targets are data and are not fetched. A returned
relative relationship target must retain both raw target and workbook-relative
resolution semantics, not silently become a URL on the current server.

Independent acceptance: current known archive source counts/hash snapshots;
CSV with embedded commas/newlines/quoted quotes; XLSX duplicate headers,
formula/error/cache and hidden rows; literal hyperlink, internal location,
missing target/multiple links/batch cell IDs. Verify all59 native XLSM link
objects (23 MOSFET+36 IGBT) are listed and all63 GetURL formula locations remain
formula metadata, not63 fabricated hyperlinks. Fixture-built workbooks must pass
without Excel, network, VBA or Power Query. Verify each read-only source hash is
unchanged. Browser preview/export must expose source/conditions/errors and never
label a refresh as current pricing.

## Final ledger overlay and completion rules

Keep `semiconductor-function-gap-ledger.json` as the immutable structural
pre-implementation baseline. Add a separate versioned capability/evidence overlay
keyed by exact baseline definition IDs and explicit workbook cell/procedure IDs.
Every overlay row has capability ID, baseline IDs, current entry points,
supported/excluded regimes, status, numerical/API/browser tests and artifact
paths, source hashes and residual limitations. A status is evidence about a
defined contract, not a count of how many source methods share its name.

Suggested remaining-function mapping groups (expand to exact baseline IDs using
the baseline, including aliases/helpers as subordinate members):

| Capability ID | Baseline anchors / coverage decision |
| --- | --- |
| `evaluation-grid` | ENCO `enco-1add2a28bb168830`, `enco-29f66a64514ba8d1`, `enco-25d22625814aa494`, CircuitGroup data-frame/export rows and `optimizer/list.py` evaluator methods. Old monorepo generations are equivalent-reference candidates until reviewed, not separate mandatory engines. |
| `linked-plot-data` | SC `sc-94c45f5c2ac963d6`, `sc-67594472598798ed`, `sc-0c9501bd111cf4da`; ENCO `enco-770709e5b596763a`, `enco-cdcfbdcc9da53c76`, `enco-b487eaa3ab6e6707`, `enco-29e85749082b06b0`. Phase5 supplies time/spectrum science; 8C supplies selection/bars/maps. |
| `energy-grid` | SC `sc-ebc372d2bd62ac36` incomplete fixed reshape records source intent, not validated source output; substantive SC PLECS axis/LUT properties map to strict grid construction. |
| `plecs-roundtrip` | SC `sc-aff249cc865946a3`, `sc-303f1e3d5b7a8037`, ScDataPlecs reader/properties; ENCO `enco-648141e2fac703f9`. ENCO `enco-806c8487b2f85d4b` exporter remains source-placeholder, with independently implemented current replacement evidence. |
| `measurement-import` | SC `sc-7b10bd773106e5ee` log import and `sc-aa1cb8c51ad0ec40` DPT integration, plus ENCO `enco-fd50d6ef7310d2ee` RAW loader and methods. Parser internals may be satisfied by tested installed-library reuse with a named unsupported-runtime state; simulator adapters stay reference-only unless separately implemented. |
| `image-digitization` | SC `sc-6dd251b385e740b2`, `sc-59dcb71cb79973a7`, `sc-077ea85e0452003e`, `sc-6c9e629b2a2e1e61`, `sc-e640474f5446e949`, `sc-6181616afd5ee44e`; one calibrated editor replaces these UI/helper functions. |
| `transfer-recovery-browser` | Current baseline extraction helpers and their SC parameter intents; accepted core scientific evidence remains separate from new transport/browser evidence. |
| `catalog-normalization-refresh` | Workbook `Tabelle1!C2:C311`, typed coefficient cells, five recorded Power Query queries, VBA `Modul1.ExtractHyperlinks/GetURL`; these need explicit supplemental source IDs because the Python/MATLAB definition baseline alone cannot enumerate workbook procedures/cells. |
| `legacy-live-adapters` | Current baseline CalculationService/ComparisonService/ExportService rows; rejected fallbacks/placeholder optimizers get explicit repaired/unsupported status with caller evidence, not inherited new-workbench status. |

Do not mark unrelated magnetics, PLECS process orchestration, general circuit
simulators, repository context utilities, examples, tests, stubs, duplicate bodies
or domain-specific converter synthesis as semiconductor functions implemented by
this effort. Preserve them as reviewed context/reference-only/duplicate/stub
with reasons. MATLAB `save7`, generic `xml2struct`/`struct2xml`, file lookup and
version migration helpers need explicit format/identity decisions in the overlay;
JSON/core serialization and PLECS roundtrips do not automatically prove arbitrary
legacy `.mat` schema migration. Remaining unsupported legacy formats must be
listed honestly, with actual input fixtures required before implementation.

Acceptance gates are independent numeric fixtures, source/format roundtrips,
typed HTTP equality/error states, real browser workflow/export and headless/lint/
build regressions. Never promote every definition to implemented because the
family has one successful test. An unresolved unsupported format, formula or
domain stays visible in the final gap report.

## Scoped implementation ownership after steps 5–7 are accepted

Work in dependency order, with one owner per mutable file boundary. Suggested
vertical slices are: (1) pure data readers/PLECS intermediate and renderer plus
provenance-preserving core serialization; (2) extraction/digitization operations
and parameter-browser adapters; (3) finite evaluation/grid/normalization and
linked-view operations; (4) live legacy adapter delegation/fail-closed behavior
and final overlay reconciliation. A browser owner may consume accepted typed
operation outputs, but must not own a competing numerical path. `main.py`,
`schemas.py`, `parameter_routes.py`, shared API-client/App navigation and
`core/models.py` must be assigned explicitly rather than edited concurrently.
Separate independent review should run numeric/format fixtures and browser
exports against each slice before the next is marked accepted. The immutable
baseline, workbook audit and phase plan remain parent-owned. No phase advancement
or production implementation occurred during this contract preparation.
