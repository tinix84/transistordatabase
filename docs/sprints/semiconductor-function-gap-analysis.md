# Semiconductor function gap analysis

Phase 1 of [the execution plan](sprint-semiconductor-function-parity.md#1-complete-the-function-ledger). Static discovery baseline; no new parity is claimed.

Reference roots (read-only):

- `\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc`
- `D:\OneDrive\claude\bhy\enco`

The JSON ledger retains every function/class, source root, exact header, line, notebook cell, body/definition/file hash, duplicate identity, role, candidate target, gap, status and sequence step. Classes are included so schema-only features cannot masquerade as functions. Notebook line numbers are cell-local and always include a cell index. MATLAB body hashes conservatively stop at the next declaration; Python hashes use normalized ASTs. Duplication requires identical definition content, never a matching name alone.

## Parser runtime and reproduction

Python **3.12.14** (cpython); interpreter `C:\Users\tinix\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.

Use Python **3.12 or newer** for this inventory generator: ENCO uses PEP 695 generic declarations. An older parser can report valid reference files as syntax errors and change body hashes, duplicate links and classifications.

The current-package index is a **pre-implementation baseline**. Scientific modules may change during the subsequent numbered implementation steps; their execution evidence belongs in a separate reviewed overlay. The baseline does not assert that stored current-source hashes describe those later edits.

Reproduce from this repository in PowerShell (regeneration refreshes the current index):

```powershell
& 'C:\Users\tinix\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/build_semiconductor_gap_ledger.py
```

## Scope and status meanings

- `missing`: no reviewed current counterpart selected; investigate before implementation.
- `candidate-match-needs-validation`: related current code exists; units, conditions, regime and behavior remain unproven.
- `Python-only`: current source exists; this baseline has no function-level numerical/API/browser proof.
- `broken-adapter`: source inspection finds a caller/callee contract mismatch.
- `stub`, `duplicate`, `reference-only`: preserve evidence without counting each as an independent missing capability.
- `browser-verified`: reserved for function-specific numerical, API and browser evidence; no baseline row receives it.

General ENCO CLI, repositories, documentation, optimizers, magnetic/passive design and simulation orchestration are comparison context, not missing semiconductor features to transplant. Source tests remain test references. Existing Qt code remains a screenshot/reference source and is outside migration targets.

## Family overview

| Family | SC definitions | ENCO definitions | Current definitions | Execution step |
| --- | ---: | ---: | ---: | --- |
| Gate charge, drive and interval waveforms | 32 | 0 | 13 | 2 |
| Double pulse, raw/log extraction and energy maps | 54 | 20 | 9 | 3 |
| Device fitting, capacitances and signed slew | 24 | 0 | 17 | 4 |
| Analytical switching model | 38 | 0 | 21 | 4 |
| Signed waveform conduction and event energies | 42 | 7 | 20 | 5 |
| Conditioned regression and lookup tables | 0 | 95 | 0 | 5 |
| Time and frequency series operations | 0 | 434 | 0 | 5 |
| Parallel/composite devices and thermal paths | 21 | 3 | 28 | 6 |
| Converter branch loss aggregation | 12 | 105 | 68 | 7 |
| Circuit indicators and aggregation | 0 | 74 | 0 | 7 |
| Catalog, FOM and Pareto exploration | 5 | 4 | 22 | 8 |
| Linked plots, maps and surfaces | 14 | 60 | 125 | 8 |
| Import/export and database conversion | 74 | 35 | 94 | 8 |
| Image calibration and curve digitization | 6 | 0 | 0 | 8 |
| Device and condition data contracts | 63 | 317 | 596 | 8 |
| Non-semiconductor orchestration/context | 0 | 592 | 0 | context only |

Counts include classes, notebook functions and duplicate copies; they are not unique working product features.

## ENCO findings and implementation boundaries

- The ENCO root has several generations: `enco/packages`, standalone `enco-core`, `enco-analysis`, `enco-circuits`, `enco-library`, `enco-database`, and `library-design`. All source sets are inventoried, including the project circuit examples. Identical copies are linked by definition hash; different bodies remain distinct.
- MOSFET regressors expose conduction resistance, forward diode voltage/current, switching energies, transitions and waveform loss indices. Separable LUTs are condition-normalized estimates; their training and error metrics do not establish arbitrary multidimensional empirical coverage. Current adapters must retain measured conditions and explicit missing/estimate scope.
- Inspected scalar `MOSFET.get_vf/get_switching_energy` wrappers pass NumPy arrays to regressors whose `predict` methods select Polars columns. These wrappers are broken-adapter evidence even where low-level regressor tests exist.
- PLECS XML import extracts substantive conduction and switching datasets. `enco_to_xml` writes fixed example XML and is a placeholder, so bidirectional round-trip parity is unproven.
- `ParetoPlot` is a schema. The inspected plot dispatcher lacks a Pareto branch. Its linked time/frequency plots and grouped/stacked indicator bars are substantive visual references; do not credit a supported Pareto renderer.
- TimeSeries/SeriesContainer supply arithmetic, RMS, interpolation, FFT and transition operations. Preserve explicit sample alignment, signed current, event conventions, exact conditions and distinct zero/missing states when implementing the lean shared Python engine.
- Circuit indicator attachment aggregates RMS, maximum voltage, cost/volume and loss indices, transition statistics and circuit/group totals. Indices and fixed-temperature loss sums are not a validated electrothermal solver. Future iteration must expose convergence status.
- ENCO copyright headers reserve rights to BRUSA. Inspect locally for behavior; write independent implementations here and do not publish reference bodies.

## Validation and inventory reconciliation

```json
{
  "old_archive_definitions": 314,
  "accounted_old_definitions": 314,
  "missing_old_definitions": [],
  "additional_non_notebook_archive_functions": [
    [
      "matlab_func/Bridgeless_PFC/calc_bridgeless_semiconductor_loss.py",
      1
    ],
    [
      "semiconductors/biela_swlosses.py",
      14
    ],
    [
      "semiconductors/biela_swlosses.py",
      31
    ],
    [
      "semiconductors/biela_swlosses.py",
      45
    ],
    [
      "semiconductors/biela_swlosses.py",
      72
    ],
    [
      "semiconductors/biela_swlosses.py",
      81
    ],
    [
      "semiconductors/biela_swlosses.py",
      102
    ],
    [
      "semiconductors/biela_swlosses.py",
      107
    ],
    [
      "semiconductors/biela_swlosses.py",
      110
    ],
    [
      "semiconductors/biela_swlosses.py",
      130
    ],
    [
      "semiconductors/biela_swlosses.py",
      143
    ],
    [
      "semiconductors/biela_swlosses.py",
      163
    ],
    [
      "semiconductors/biela_swlosses.py",
      168
    ],
    [
      "semiconductors/biela_swlosses.py",
      178
    ],
    [
      "semiconductors/biela_swlosses.py",
      203
    ],
    [
      "tests/biela_swlosses.py",
      15
    ],
    [
      "tests/biela_swlosses.py",
      32
    ],
    [
      "tests/biela_swlosses.py",
      46
    ],
    [
      "tests/biela_swlosses.py",
      73
    ],
    [
      "tests/biela_swlosses.py",
      82
    ],
    [
      "tests/biela_swlosses.py",
      103
    ],
    [
      "tests/biela_swlosses.py",
      108
    ],
    [
      "tests/biela_swlosses.py",
      111
    ],
    [
      "tests/biela_swlosses.py",
      131
    ],
    [
      "tests/biela_swlosses.py",
      144
    ],
    [
      "tests/biela_swlosses.py",
      164
    ],
    [
      "tests/biela_swlosses.py",
      169
    ],
    [
      "tests/biela_swlosses.py",
      179
    ],
    [
      "tests/biela_swlosses.py",
      204
    ],
    [
      "tests/test_igbt.py",
      5
    ],
    [
      "tests/test_igbt.py",
      18
    ]
  ],
  "source_locations_verified": 3144,
  "unique_ids": 3144,
  "files": 505,
  "method": "Old inventory path/line multiset reconciliation plus direct source-line checks, separate from extraction."
}
```

The original archive multiset is reconciled in full. Additional notebook definitions/classes are supplemental. Source syntax errors and unresolved literal example data below are retained, never imported or executed. A literal may require a caller working directory or dynamic path, so unresolved references are not all proven missing files.

## Syntax errors

- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 0` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 1` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 6` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 14` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 18` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 20` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 22` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 24` — invalid syntax
- `sc:Doc/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 27` — invalid syntax. Perhaps you forgot a comma?
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 0` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 1` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 6` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 14` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 18` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 20` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 22` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 24` — invalid syntax
- `sc:docs/.ipynb_checkpoints/Mosfet_switching_losses-checkpoint.ipynb:1 cell 27` — invalid syntax. Perhaps you forgot a comma?
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 0` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 1` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 6` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 14` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 18` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 20` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 22` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 24` — invalid syntax
- `sc:docs/Mosfet_switching_losses.ipynb:1 cell 27` — invalid syntax. Perhaps you forgot a comma?
- `sc:matlab_func/Bridgeless_PFC/calc_bridgeless_semiconductor_loss.py:1` — expected '('
- `sc:matlab_func/Bridgeless_PFC/test_calc_bridgeless_semiconductor_loss.py:1` — invalid syntax
- `sc:semiconductors/biela_swlosses.py:56` — expected argument value expression
- `sc:tests/biela_swlosses.py:37` — invalid syntax
- `sc:tests/test_igbt.py:71` — expected an indented block after 'if' statement on line 70

## Workbook and catalog metric capabilities

The five user-supplied catalog files are represented by eight imported source sheets / CSVs (2,490 rows) in the existing portable snapshot. This function census does not execute spreadsheet formulas or macros. A full census of formulas on unimported/summary sheets and VBA procedures remains **open**. Repeated formula cells do not count as distinct functions.

`SemiconductorsData.xlsm` contains temperature-dependent resistance estimates, Coss×Rds and Qg×Rds FOMs, maximum loss at 100 °C, a derived DC current, current/cost, voltage×current/cost, voltage/total-switching-energy and its cost ratio. These require independent unit/condition review: its MOSFET FOM references the later 150 °C resistance column while the current browser's primary Rds uses the first explicit 25 °C condition. The workbook DC current also uses typical resistance while another FOM uses maximum resistance. Preserve these distinctions before enabling workbook-equivalent comparisons.

Each following representative formula family is **reference-needs-validation**. Formula text and source-cell references are provenance, not validated thermal/current predictions. Existing catalog FOM numerical/browser evidence is documented separately in `sc-archive-parity-review.md`.

- `201210_sic_digikey.xlsx` / `in`: 286 imported records; No formula families in the imported row snapshot; source catalog ratings and cost remain condition-specific.
- `SiC_MOSFET_FOM.xlsx` / `1200V`: 174 imported records; `Rdson (Ohm)` (`1200V!U2`, `=MID(download_3[[#This Row],[Rds(On) (Max.) bei Id, Vgs]],1,FIND("mOhm",download_3[[#This Row],[Rds(On) (Max.) bei Id, Vgs]])-1)*0.001`); `Spalte1` (`1200V!AF2`, `=I2/(1200^2/U2)`)
- `SiC_MOSFET_FOM.xlsx` / `600V`: 1137 imported records; `Rdson (Ohm)` (`600V!U2`, `=MID(download_6[[#This Row],[Rds(On) (Max.) bei Id, Vgs]],1,FIND("mOhm",download_6[[#This Row],[Rds(On) (Max.) bei Id, Vgs]])-1)*0.001`); `Spalte1` (`600V!AF2`, `=I2/(MID(download_6[[#This Row],[Drain-Source-Spannung (Vdss)]],1,3)^2/U2)`)
- `SiC_MOSFET_FOM.xlsx` / `Tabelle8`: 233 imported records; `Spalte3` (`Tabelle8!R2`, `=MID(download_7[[#This Row],[Drain-Source-Spannung (Vdss)]],1,FIND("V",download_7[[#This Row],[Drain-Source-Spannung (Vdss)]])-1)`); `Spalte1` (`Tabelle8!V2`, `=MID(download_7[[#This Row],[Rds(On) (Max.) bei Id. Vgs]],1,FIND("mOhm",download_7[[#This Row],[Rds(On) (Max.) bei Id. Vgs]])-1)*0.001`); `Spalte2` (`Tabelle8!AG2`, `=I2/(R2^2/V2)`)
- `201210_mos_digikey.csv` / `CSV`: 500 imported records; No formula families in the imported row snapshot; source catalog ratings and cost remain condition-specific.
- `201210_gan_digikey.csv` / `CSV`: 99 imported records; No formula families in the imported row snapshot; source catalog ratings and cost remain condition-specific.
- `SemiconductorsData.xlsm` / `MOSFET`: 23 imported records; `RDSON (max)` (`MOSFET!S2`, `=R2*((O2/N2))`); `RDSON (typ)` (`MOSFET!V2`, `=R2+(U2-Q2)*((R2-N2)/(Q2-M2))`); `EON (TJ=175°C)` (`MOSFET!Y2`, `=0.000632/36`); `EOFF (TJ=175°C)` (`MOSFET!Z2`, `=0.000243/36`); `Weblink` (`MOSFET!AR2`, `=GetURL(A2)`); `FOM(Coss*Rdson_max)` (`MOSFET!AT2`, `=W2*0.001*I2`); `FOM(Qg*Rdson_max)` (`MOSFET!AU2`, `=W2*0.001*J2`); `max. Ploss @ 100°C` (`MOSFET!AV2`, `=(AB2-100)/AF2`); `max. DC Current` (`MOSFET!AW2`, `=SQRT(AV2/(V2*0.001))`); `Current per Cost` (`MOSFET!AX2`, `=AW2/AM2`); `V*I per Cost` (`MOSFET!AY2`, `=AX2*F2`); `V per ETS` (`MOSFET!AZ2`, `=X2/(Y2+Z2+AA2)`); `V/ETS per Cost` (`MOSFET!BA2`, `=AZ2/AM2`); `Cost at 1k pcs.` (`MOSFET!AM19`, `=5.9*0.85`)
- `SemiconductorsData.xlsm` / `IGBT`: 38 imported records; `rCE (typ)` (`IGBT!W2`, `=1/65`); `EON (TJ=175°C)` (`IGBT!Y2`, `=0.003/80`); `EOFF (TJ=175°C)` (`IGBT!Z2`, `=0.003/80`); `max. RthJH IGBT` (`IGBT!AH2`, `=AF2+0.66`); `Weblink` (`IGBT!AR2`, `=GetURL(A2)`); `FOM(Coes*Rce_max)` (`IGBT!AT2`, `=W2*I2`); `FOM(Qg*Rdson_max)` (`IGBT!AU2`, `=J2*W2`); `max. Ploss @ 100°C` (`IGBT!AV2`, `=(AB2-100)/AH2`); `max. DC Current` (`IGBT!AW2`, `=(SQRT(4*AV2*W2+V2^2)-V2)/(2*W2)`); `V x I per Cost` (`IGBT!AX2`, `=(AW2*F2)/AM2`); `V per ETS` (`IGBT!AY2`, `=X2/(Y2+Z2+AA2)`); `V/ETS per Cost` (`IGBT!AZ2`, `=AY2/AM2`); `max. RthJH Diode` (`IGBT!AI6`, `=AG6+AG6*1.9`); `Chiparea / IGBT` (`IGBT!AD23`, `=99.54*4`); `Cooling perf. IGBT` (`IGBT!BA23`, `=AD23*AJ23`); `ERR (TJ=175°C)` (`IGBT!AA24`, `=0.021/300`); `Chiparea / Diode` (`IGBT!AE25`, `=12.4*6*2`); `Cooling perf. Diode` (`IGBT!BB25`, `=AE25*AK25`)

## Unresolved example-data references

- `sc:data/scData/Source_Data/edit_scdata_structs.m`: `Infineon_IPW65R080CFDA_scaled.mat` (line 9); `IPW65R110CFDA.mat` (line 10); `STPSC40065CWY.mat` (line 11); `SCS220AE2HR.mat` (line 12); `FGH40T65SPD-F085.mat` (line 13); `Infineon_IPW65R080CFDA_scaled.mat` (line 19); `Infineon_IPW65R110CFDA_scaled.mat` (line 23)
- `sc:db/scData/src_data/edit_scdata_structs.m`: `Infineon_IPW65R080CFDA_scaled.mat` (line 9); `IPW65R110CFDA.mat` (line 10); `STPSC40065CWY.mat` (line 11); `SCS220AE2HR.mat` (line 12); `FGH40T65SPD-F085.mat` (line 13); `Infineon_IPW65R080CFDA_scaled.mat` (line 19); `Infineon_IPW65R110CFDA_scaled.mat` (line 23)
- `sc:matlab_func/Bridgeless_PFC/test_calc_bridgeless_semiconductor_loss.m`: `waveforms.mat` (line 10); `scData/Infineon_IPW65R048CFDA.mat` (line 12); `scData/ROHM_SCS220AE2HR.mat` (line 14); `scData/AIGW40N65F5.mat` (line 16)
- `sc:matlab_func/Bridgeless_PFC/test_calc_bridgeless_semiconductor_loss.py`: `waveforms.mat` (line 10); `scData/Infineon_IPW65R048CFDA.mat` (line 12); `scData/ROHM_SCS220AE2HR.mat` (line 14); `scData/AIGW40N65F5.mat` (line 16)
- `sc:matlab_func/ConvertscDataFileFormatFrom00To01.m`: `*.mat` (line 4)
- `sc:matlab_func/ConvertscDataFileFormatFrom01To02.m`: `*.mat` (line 4)
- `sc:matlab_func/DCDC_DAB/test_calc_DAB_Psemi.m`: `inputData/waveforms_DAB_Example_VaryP_Vp420_Vs275.mat` (line 18); `inputData/dab_waveforms.mat` (line 20); `scData/Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat` (line 28)
- `sc:matlab_func/DCDC_SRC_ZCS/test_calc_SRCZCS_Psemi.m`: `inputData/srczcs_mains_waveforms.mat` (line 18); `inputData/waveforms_SRCZCS_Example_VaryP_Vp425_Vs275.mat` (line 21); `scData/Infineon_AIKW50N65DF5_Eon60uJ.mat` (line 28); `scData/ST_STTH60RQ06WY_diode_150C_Typical.mat` (line 32)
- `sc:matlab_func/DCDC_SRC_ZVS/test_calc_SRCZVS_Psemi.m`: `inputData/waveforms_example_P3600_Vp420_Vs275.mat` (line 18); `scData/Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat` (line 25); `scData/ST_STTH60RQ06WY_diode_150C_Typical.mat` (line 28)
- `sc:matlab_func/Doc/EXAMPLE_TwoLevelInverter.m`: `../db/Infineon_AIGW50N65F5.mat` (line 17); `../db/Infineon_IDW75D65D1.mat` (line 19); `../db/hsData.mat` (line 26)
- `sc:matlab_func/EditHsData.m`: `hsData.mat` (line 13)
- `sc:matlab_func/EditInverterSemiconductorPack.m`: `Infineon_AIKW50N65DF5.mat` (line 10); `Infineon_AIKW50N65DF5_slow.mat` (line 14); `Infineon_AIKW50N65DF5.mat` (line 18); `Infineon_AIKW50N65DF5_slow.mat` (line 22); `STGYA120M65DF2AG.mat` (line 26); `STGYA120M65DF2AG_slow.mat` (line 30); `STGYA120M65DF2AG.mat` (line 34); `STGYA120M65DF2AG_slow.mat` (line 38); `ST_SCTW100N65G2AG.mat` (line 42); `ST_SCTW100N65G2AG_slow.mat` (line 46); `ST_SCTW100N65G2AG.mat` (line 50); `ST_SCTW100N65G2AG_slow.mat` (line 54)
- `sc:matlab_func/LLC/test_calc_llc_Psemi.m`: `inputData/waveforms_LLC_Example_P3600_Vp425_Vs195_470.mat` (line 18); `../../02_Waveform_Simulation/LLC/Output/llc_mains_waveforms.mat` (line 19); `scData/Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat` (line 26); `scData/ST_STTH60RQ06WY_diode_150C_Typical.mat` (line 29)
- `sc:matlab_func/PFC-SixSwitch/test_three_phase_totempole_semiconductor_loss.m`: `waveforms.mat` (line 10); `scData/SCT3080KL.mat` (line 12); `scData/VS-45APS12LHM3.mat` (line 14)
- `sc:matlab_func/PFC_IVR/test_calc_vr_semiconductor_loss.m`: `waveforms.mat` (line 12)
- `sc:matlab_func/scDataByName.m`: `/*.mat` (line 28)
- `sc:matlab_func/test_convertPlecsXmlToscData02.m`: `d:\OneDrive - NTB\105\06_Simulationen\ntbees\classes\sc\db\PLECS\WS_Modue_PLECS_07_27_18\CAS120M12BM2.xml` (line 6)
- `sc:mfiles/ConvertscDataFileFormatFrom00To01.m`: `*.mat` (line 4)
- `sc:mfiles/ConvertscDataFileFormatFrom01To02.m`: `*.mat` (line 4)
- `sc:mfiles/Doc/EXAMPLE_TwoLevelInverter.m`: `../db/Infineon_AIGW50N65F5.mat` (line 17); `../db/Infineon_IDW75D65D1.mat` (line 19); `../db/hsData.mat` (line 26)
- `sc:mfiles/edit_scdata_structs.m`: `Infineon_IPW65R080CFDA_scaled.mat` (line 9); `IPW65R110CFDA.mat` (line 10); `STPSC40065CWY.mat` (line 11); `SCS220AE2HR.mat` (line 12); `FGH40T65SPD-F085.mat` (line 13); `Infineon_IPW65R080CFDA_scaled.mat` (line 19); `Infineon_IPW65R110CFDA_scaled.mat` (line 23)
- `sc:mfiles/EditHsData.m`: `hsData.mat` (line 13)
- `sc:mfiles/EditInverterSemiconductorPack.m`: `Infineon_AIKW50N65DF5.mat` (line 10); `Infineon_AIKW50N65DF5_slow.mat` (line 14); `Infineon_AIKW50N65DF5.mat` (line 18); `Infineon_AIKW50N65DF5_slow.mat` (line 22); `STGYA120M65DF2AG.mat` (line 26); `STGYA120M65DF2AG_slow.mat` (line 30); `STGYA120M65DF2AG.mat` (line 34); `STGYA120M65DF2AG_slow.mat` (line 38); `ST_SCTW100N65G2AG.mat` (line 42); `ST_SCTW100N65G2AG_slow.mat` (line 46); `ST_SCTW100N65G2AG.mat` (line 50); `ST_SCTW100N65G2AG_slow.mat` (line 54)
- `sc:mfiles/EXAMPLE_TwoLevelInverter.m`: `hsData.mat` (line 29)
- `sc:mfiles/scDataByName.m`: `/*.mat` (line 28)
- `sc:mfiles/SemiconductorEditor.m`: `*.mat` (line 92); `*.mat` (line 94); `*.mat` (line 109); `*.mat` (line 111); `*.mat` (line 178); `*.mat` (line 189); `*.mat` (line 249); `*.mat` (line 260); `*.mat` (line 325); `*.mat` (line 336); `*.mat` (line 405); `*.mat` (line 416); `*.mat` (line 485); `*.mat` (line 496); `*.mat` (line 565); `*.mat` (line 576); `*.mat` (line 724)
- `sc:mfiles/test_convertPlecsXmlToscData02.m`: `d:\OneDrive - NTB\105\06_Simulationen\ntbees\classes\sc\db\PLECS\WS_Modue_PLECS_07_27_18\CAS120M12BM2.xml` (line 6)
- `sc:mfiles/test_save7.m`: `*.mat` (line 7)
- `sc:mfiles/test_scDataValues.m`: `*.mat` (line 14)
- `sc:semiconductors/helpers.py`: `DoublePulseTester_EPC8009_sync.raw` (line 19)
- `sc:tests/mfiles/test_scDataValues.m`: `C3M0075120K.mat` (line 9)
- `sc:tests/pareto_sc.py`: `/LTSpice/DoublePulseTester_EPC8009_sync.log` (line 17); `/../db/SiC_MOSFET_FOM.xlsx` (line 18)
- `sc:tests/read_raw_upgrade.py`: `DoublePulseTester_EPC8009_sync.raw` (line 158)
- `sc:tests/test_basic.py`: `/../db/plecs/Wolfspeed_MOSFET_PLECS_10_2019/C2M0025120D.xml` (line 23); `/../db/plecs/Infineon_IGBT/IKFW50N60DH3E_IGBT.xml` (line 32); `/../db/plecs/Wolfspeed_MOSFET_PLECS_10_2019/C2M0025120D.xml` (line 58); `/../db/plecs/Wolfspeed_MOSFET_PLECS_10_2019/C2M0025120D.xml` (line 63); `/../db/plecs/Wolfspeed_MOSFET_PLECS_10_2019/test_multi_vars_C2M0025120D.xml` (line 68)
- `sc:tests/test_convertPlecsXmlToscData02.m`: `d:\OneDrive - NTB\105\06_Simulationen\ntbees\classes\sc\db\PLECS\WS_Modue_PLECS_07_27_18\CAS120M12BM2.xml` (line 6)
- `sc:tests/test_xmlplecs.py`: `C:\Users\tinix\OneDrive - NTB\105\06_Simulationen\ntbees\classes\sc\data\cree_plecs\Wolfspeed_MOSFET_PLECS_10_2019\C2M0025120D.xml` (line 98)
- `sc:utils/dpt/mfiles/runme_switching_losses.m`: `./DoublePulseTester_EPC8009_sync.log` (line 8); `Eloss.mat` (line 32); `Eloss.mat` (line 34); `Eloss.mat` (line 41); `Eloss.mat` (line 61)
- `sc:utils/dpt/OLD/ltspice_switching_losses.py`: `DoublePulseTester_EPC8009_sync.log` (line 294); `./LTSpice/DoublePulseTester_EPC8009_sync.raw` (line 305)
- `sc:utils/dpt/read_raw_upgrade.py`: `./LTSpice/DoublePulseTester_EPC8009_sync.raw` (line 67)
- `sc:utils/dpt/tests_basic.py`: `/LTSpice/DoublePulseTester_EPC8009_sync.log` (line 57); `/LTSpice/DoublePulseTester_EPC8009_sync.raw` (line 58)
- `sc:utils/semiconductor.m`: `*.mat` (line 114)
- `enco:enco/_release/release.py`: `data.json` (line 133)
- `enco:enco/packages/enco-cli/src/enco_cli/cli/docs.py`: `settings.json` (line 175)
- `enco:enco/packages/enco-components/tests/mosfet/test_boost.py`: `G3R40MT12K.xml` (line 32); `G3R40MT12K.json` (line 109); `SCT070HU120G3AG.json` (line 123); `au_switching_energy.csv` (line 202); `al_switching_energy.csv` (line 211)
- `enco:enco/packages/enco-components/tests/mosfet/test_plecs_conversion.py`: `G3R40MT12K.xml` (line 25); `G3R40MT12K.json` (line 28)
- `enco:enco/packages/enco-core/src/enco_core/log.py`: `enco.log` (line 33)
- `enco:enco/packages/enco-simulation/src/enco_simulation/simulation.py`: `*.xml` (line 186)
- `enco:enco-circuits/tests/test_boost.py`: `boost.json` (line 117); `boost_group.json` (line 133)
- `enco:enco-circuits/tests/test_circuit_group.py`: `test.json` (line 133); `test.xlsx` (line 135); `test.json` (line 137)
- `enco:enco-circuits/tests/test_coupled_boost.py`: `coupled_boost_group.json` (line 127)
- `enco:enco-circuits/tests/test_load.py`: `pfc.json` (line 9)
- `enco:enco-circuits/tests/test_pfc_boost_dab.py`: `boost.json` (line 78); `boost_dab_group.json` (line 94)
- `enco:enco-circuits/tests/test_pfc_universal.py`: `pfc.json` (line 127); `pfc.xlsx` (line 129)
- `enco:enco-core/src/enco_core/log.py`: `enco.log` (line 33)
- `enco:library-design/dcdc/cllc/src/project/cllc.py`: `cllc.xlsx` (line 50); `cllc.json` (line 53)
- `enco:library-design/dcdc/cllc/src/project/electrical_mosfet.py`: `rds_on.json` (line 97)
- `enco:library-design/dcdc/cllc/src/project/electrical_transformer.py`: `kpi_debug.json` (line 251); `kpi_tensor.json` (line 264); `kpi_tensor_normalized.json` (line 272); `kpi_tensor_reduced_op_weights.json` (line 289); `kpi_tensor_reduced_kpi_weights.json` (line 309); `results.json` (line 351)
- `enco:library-design/dcdc/cllc/src/project/optimizer.py`: `entry_indicators.json` (line 260); `initial_space.csv` (line 336); `final_space.csv` (line 342); `merit.csv` (line 370)
- `enco:library-design/dcdc/cllc_3p/src/project/main.py`: `converter.json` (line 59); `converter.xlsx` (line 66)
- `enco:library-design/dcdc/cllc_3p/src/project/optimization_electrical.py`: `optimization_electrical.xlsx` (line 141)
- `enco:library-design/dcdc/d3ab/src/project/main.py`: `converter.json` (line 79); `converter.xlsx` (line 86)
- `enco:library-design/dcdc/d3ab/src/project/optimization_electrical.py`: `optimization_electrical.xlsx` (line 146)
- `enco:library-design/dcdc/dab_cf/src/project/dab.py`: `dab.xlsx` (line 58); `dab.json` (line 61)
- `enco:library-design/dcdc/dab_cf/src/project/electrical_mosfet.py`: `rds_on.json` (line 89)
- `enco:library-design/dcdc/dab_cf/src/project/optimizer.py`: `entry_indicators.json` (line 264); `initial_space.csv` (line 340); `final_space.csv` (line 346); `merit.csv` (line 374)
- `enco:library-design/dcdc/psfb/src/psfb/main.py`: `operating_points.xlsx` (line 177); `psfb.xlsx` (line 227); `psfb.json` (line 228)
- `enco:library-design/dcdc/psfb/src/psfb/mosfet.py`: `operating_points.xlsx` (line 177); `psfb_mosfet.xlsx` (line 232); `psfb_mosfet.json` (line 233)
- `enco:library-design/project/OBC7/src/project/llc.py`: `llc.json` (line 51); `llc.xlsx` (line 58)
- `enco:library-design/project/OBC7/src/project/pfc.py`: `pfc.json` (line 51); `pfc.xlsx` (line 58)
- `enco:library-design/uncertainty/chain/src/project/main.py`: `MeasurementChain.asc` (line 37)
- `current:backend/concrete_services.py`: `housing_types.txt` (line 56); `module_manufacturers.txt` (line 57)
- `current:catalog_fom.py`: `archive_sc.json` (line 15)
- `current:core/pair_repository.py`: `*.json` (line 106); `*.json` (line 178)
- `current:core/repository.py`: `housing_types.txt` (line 114); `module_manufacturers.txt` (line 115); `*.json` (line 146); `housing_types.txt` (line 190); `module_manufacturers.txt` (line 191)
- `current:database_manager.py`: `housing_types.txt` (line 52); `module_manufacturers.txt` (line 58); `https://raw.githubusercontent.com/upb-lea/transistordatabase_File_Exchange/main/index.txt` (line 75); `https://raw.githubusercontent.com/upb-lea/transistordatabase_File_Exchange/main/index.txt` (line 325); `https://raw.githubusercontent.com/upb-lea/transistordatabase_File_Exchange/main/module_manufacturers.txt` (line 327); `https://raw.githubusercontent.com/upb-lea/transistordatabase_File_Exchange/main/housing_types.txt` (line 329); `C:/Users/.../GaN-Systems/400V/*.csv` (line 776)
- `current:examples/first_example.py`: `diff.json` (line 63)
- `current:gui/mixins/curve_mixin.py`: `/*.csv` (line 101)
- `current:gui/mixins/settings_mixin.py`: `/settings.json` (line 229); `settings.json` (line 244); `settings.json` (line 271)
- `current:gui_web/backend/main.py`: `*.xml` (line 370); `*.mat` (line 377); `*.xml` (line 449); `*.xml` (line 534); `*.mat` (line 540); `attachment; filename=switching_pairs.csv` (line 1796)
- `current:gui_web/backend/main_simple.py`: `*.json` (line 35)
- `current:gui_web/tests/debug_upload.py`: `CREE_C3M0016120K.json` (line 68); `minimal_test.json` (line 143)
- `current:gui_web/tests/test_file_upload.py`: `CREE_C3M0016120K.json` (line 85); `minimal.json` (line 131); `tc_fallback.json` (line 165); `large_device.json` (line 225); `invalid.json` (line 246); `text.txt` (line 264); `empty.json` (line 282); `duplicate1.json` (line 310); `duplicate2.json` (line 320); `integration_test.json` (line 363)
- `current:gui_web/tests/test_main.py`: `test.json` (line 431); `test.txt` (line 458); `invalid.json` (line 477); `workflow_test.json` (line 653)
- `current:helper_functions.py`: `housing_types.txt` (line 176); `module_manufacturers.txt` (line 180); `transistor_c_oss.csv` (line 664); `transistor_c_oss_detail.csv` (line 665)
- `current:transistor.py`: `housing.txt` (line 119); `module_manufacturer.txt` (line 119); `_Simulink_lossmodel.mat` (line 1662); `_Matlab.mat` (line 1689); `PLECS_Exporter_template_Diode.txt` (line 2113); `_diode.xml` (line 2115); `PLECS_Exporter_template_Switch.txt` (line 2125); `_switch.xml` (line 2128)

## Individual reference functions by implementation family

Every SC/ENCO definition is listed below. Repeated files are retained so source coverage is auditable. Headers and body hashes are in the JSON. Candidate targets identify areas to investigate, never asserted parity.

### Gate charge, drive and interval waveforms

Execution: step 2 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**sc: `matlab_func/mosfet_switching_losses_refactor.py`**

- `Mosfet.calc_Tg` (function, line 60) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-cecd8420589d8707`.
- `Mosfet.calc_forcing_voltage` (function, line 65) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-10f1d94771b3f99a`.
- `Mosfet.calc_t1` (function, line 70) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-31ee3fc1323735d7`.
- `Mosfet.calc_t2` (function, line 75) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-399e718b604b0512`.
- `Mosfet.calc_t3` (function, line 92) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-b72fdc6bfc1999bd`.
- `Mosfet.calc_turnon_interval_0t1` (function, line 100) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-af6f217bff65a537`.
- `Mosfet.calc_turnon_interval_t1t2` (function, line 110) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-8e8ba7f078687f7e`.
- `Mosfet.calc_turnon_interval_t2t3` (function, line 119) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-fb022133e7148d65`.
- `Mosfet.calc_turnon_interval_t3tend` (function, line 129) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-c80bf13d1ab64d22`.
- `Mosfet.calc_miller_plateau_static` (function, line 140) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-c3a556e28964093c`.
- `Mosfet.calc_miller_plateau_from_t2` (function, line 144) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-4b4377aaaafa2065`.
- `Mosfet.calc_turnon_transition` (function, line 149) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-42677ba89c74ce85`.
- `Mosfet.calc_gate_current_waveform` (function, line 176) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-2e1f198744dbb194`.
- `Mosfet.calc_exp_approx_Vgs` (function, line 181) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-3d3bf8752087e466`.
- `Mosfet.plot_turnon_transition_matlplotlib` (function, line 185) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-f7ac8fd74a69f458`.

**sc: `semiconductors/gate_driver.py`**

- `GateDriver` (class, line 1) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-b96b93cd54a84527`.
- `GateDriver.__init__` (function, line 2) — **duplicate**; canonical `sc-6e38e5930a879393`. ID `sc-115924df7e8098e7`.

**sc: `semiconductors/mosfet.py`**

- `Mosfet.calc_Tg` (function, line 44) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-b674395011ebe219`.
- `Mosfet.calc_forcing_voltage` (function, line 49) — **duplicate**; canonical `sc-10f1d94771b3f99a`. ID `sc-3e08c08a6c4eb039`.
- `Mosfet.calc_t1` (function, line 54) — **duplicate**; canonical `sc-31ee3fc1323735d7`. ID `sc-286da512145bb2e3`.
- `Mosfet.calc_t2` (function, line 59) — **duplicate**; canonical `sc-399e718b604b0512`. ID `sc-3ec4b60414a8fee1`.
- `Mosfet.calc_t3` (function, line 76) — **duplicate**; canonical `sc-b72fdc6bfc1999bd`. ID `sc-6225503d9bb6265b`.
- `Mosfet.calc_turnon_interval_0t1` (function, line 84) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-4203f3e509a56fa0`.
- `Mosfet.calc_turnon_interval_t1t2` (function, line 94) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-0614711e4b9dea2a`.
- `Mosfet.calc_turnon_interval_t2t3` (function, line 103) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-ca5897dddd111395`.
- `Mosfet.calc_turnon_interval_t3tend` (function, line 113) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-4a918516adc31129`.
- `Mosfet.calc_miller_plateau_static` (function, line 124) — **duplicate**; canonical `sc-c3a556e28964093c`. ID `sc-027fc4b9043e1f71`.
- `Mosfet.calc_miller_plateau_from_t2` (function, line 128) — **duplicate**; canonical `sc-4b4377aaaafa2065`. ID `sc-6c79e0ee6d16d0c6`.
- `Mosfet.calc_turnon_transition` (function, line 133) — **candidate-match-needs-validation**; candidate `analytical_models.py:88 GateChargeModel.calc_turn_on_times`. ID `sc-a099571637ed4224`.
- `Mosfet.calc_gate_current_waveform` (function, line 160) — **duplicate**; canonical `sc-2e1f198744dbb194`. ID `sc-8535fe1633702dc3`.
- `Mosfet.calc_exp_approx_Vgs` (function, line 165) — **duplicate**; canonical `sc-3d3bf8752087e466`. ID `sc-146b05b7349c2a07`.
- `Mosfet.plot_turnon_transition_matlplotlib` (function, line 169) — **duplicate**; canonical `sc-f7ac8fd74a69f458`. ID `sc-7b162a7d64df604f`.


### Double pulse, raw/log extraction and energy maps

Execution: step 3 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco/packages/enco-simulation/src/enco_simulation/ltspice/adapter.py`**

- `LTSpiceAdapter` (class, line 32) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-18311858a5c1ef73`.
- `LTSpiceAdapter.__init__` (function, line 37) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-fe413c152808d9ef`.
- `LTSpiceAdapter._generate_netlist` (function, line 69) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-43aa5edf20bb4ac2`.
- `LTSpiceAdapter._before_run` (function, line 182) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-0fdf1d55e510c907`.
- `LTSpiceAdapter._run` (function, line 190) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-67bbd03435606c68`.
- `LTSpiceAdapter._after_run` (function, line 204) — **stub**. ID `enco-782cd20fb5273236`.
- `LTSpiceAdapter._load` (function, line 207) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-48ba649ce6ff6607`.
- `LTSpiceAdapter.__repr__` (function, line 226) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-cd0e162030832f3e`.
- `LTSpiceAdapter.__str__` (function, line 229) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-0efd14a5d44778ad`.
- `engineering_suffix` (function, line 235) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-9fa905652fda8aec`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/ltspice/loader.py`**

- `LTSpiceRawLoader` (class, line 32) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-fd50d6ef7310d2ee`.
- `LTSpiceRawLoader.__init__` (function, line 33) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-f255070493037515`.
- `LTSpiceRawLoader.load` (function, line 45) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-69a0d6a0af6c8487`.
- `LTSpiceRawLoader.__load` (function, line 55) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-a933dc508e868f66`.
- `LTSpiceRawLoader.__read_binary` (function, line 97) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-7253219193ec6559`.
- `LTSpiceRawLoader.__parse_header` (function, line 320) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-3577c52aa25b70cd`.
- `LTSpiceRawLoader.__getitem__` (function, line 385) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-052fac610cadf5ec`.
- `LTSpiceRawLoader.__repr__` (function, line 397) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-bebe995e00cc60df`.
- `LTSpiceRawLoader.__str__` (function, line 403) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-9dbf1764cc58af9f`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/ltspice/type.py`**

- `LTSpiceSimulationType` (class, line 20) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `enco-7a15a4bb6ae93253`.

**sc: `matlab_func/double_pulse/importfile_Eoff.m`**

- `importfile_Eoff` (function, line 1) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-6612aebd798460d8`.

**sc: `matlab_func/double_pulse/importfile_Eon.m`**

- `importfile_Eon` (function, line 1) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-8b7a7456a3088c78`.

**sc: `matlab_func/double_pulse/importfile_step_descr.m`**

- `importfile_step_descr` (function, line 1) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-2010484ad06198ef`.

**sc: `matlab_func/double_pulse/xyz2grid.m`**

- `xyz2grid` (function, line 1) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-4d62fb1625d590c7`.

**sc: `matlab_func/double_pulse/xyzread.m`**

- `xyzread` (function, line 1) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-7496258f50c32467`.

**sc: `tests/read_raw_upgrade.py`**

- `find_nearest` (function, line 24) — **duplicate**; canonical `sc-81fc707fd885ac42`. ID `sc-f30d6530487ad6a6`.
- `calc_sw_losses_ltspice` (function, line 29) — **duplicate**; canonical `sc-aa1cb8c51ad0ec40`. ID `sc-9444848b52c7ca93`.
- `DoublePulseViewerBokeh` (class, line 55) — **duplicate**; canonical `sc-94c45f5c2ac963d6`. ID `sc-0471648fc987c705`.
- `DoublePulseViewerBokeh.__init__` (function, line 56) — **duplicate**; canonical `sc-1f61e57d7f0bf5ff`. ID `sc-2f85ebb8d28d24ec`.
- `DoublePulseViewerBokeh.add_time_wfm` (function, line 71) — **duplicate**; canonical `sc-67594472598798ed`. ID `sc-8802901ac644a1c5`.
- `DoublePulseViewerBokeh.show` (function, line 76) — **duplicate**; canonical `sc-705bf23273a886cd`. ID `sc-eec5e3b2de88cc19`.
- `DoublePulseViewerPlotly` (class, line 79) — **reference-only**. ID `sc-26835ea0d1cde9c5`.
- `DoublePulseViewerPlotly.__init__` (function, line 80) — **duplicate**; canonical `sc-b5c020835c28151e`. ID `sc-23b93df1681544b8`.
- `DoublePulseViewerPlotly.add_time_wfm` (function, line 95) — **duplicate**; canonical `sc-ce5c7005524063c6`. ID `sc-2cbf3d0da6280376`.
- `DoublePulseViewerPlotly.add_surface` (function, line 105) — **duplicate**; canonical `sc-0c9501bd111cf4da`. ID `sc-a5369e4c595cff0c`.
- `DoublePulseViewerPlotly.show` (function, line 133) — **duplicate**; canonical `sc-589a48ec6fabc7de`. ID `sc-edd8d19821cebb2b`.

**sc: `utils/dpt/OLD/ltspice_switching_losses.py`**

- `find_nearest` (function, line 33) — **duplicate**; canonical `sc-81fc707fd885ac42`. ID `sc-b334b4c2855c9aa3`.
- `calc_sw_losses_ltspice` (function, line 39) — **duplicate**; canonical `sc-aa1cb8c51ad0ec40`. ID `sc-09d35f55d0056589`.
- `DoublePulseViewerBokeh` (class, line 68) — **duplicate**; canonical `sc-94c45f5c2ac963d6`. ID `sc-1c7556996e76d511`.
- `DoublePulseViewerBokeh.__init__` (function, line 69) — **duplicate**; canonical `sc-1f61e57d7f0bf5ff`. ID `sc-97336cf68100465a`.
- `DoublePulseViewerBokeh.add_time_wfm` (function, line 83) — **duplicate**; canonical `sc-67594472598798ed`. ID `sc-939b44f9eb26c974`.
- `DoublePulseViewerBokeh.show` (function, line 88) — **duplicate**; canonical `sc-705bf23273a886cd`. ID `sc-9873744899efb8e0`.
- `DoublePulseViewerPlotly` (class, line 92) — **duplicate**; canonical `sc-a7dc7164ca6fbe25`. ID `sc-ec49c73f7da57705`.
- `DoublePulseViewerPlotly.__init__` (function, line 93) — **duplicate**; canonical `sc-b5c020835c28151e`. ID `sc-18e3fdce228009cb`.
- `DoublePulseViewerPlotly.add_time_wfm` (function, line 108) — **duplicate**; canonical `sc-ce5c7005524063c6`. ID `sc-604268de7742e69d`.
- `DoublePulseViewerPlotly.show` (function, line 118) — **duplicate**; canonical `sc-f7926913c80d84cf`. ID `sc-8a891eb07763d316`.
- `plot_switching_energy_map` (function, line 126) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-85bb2aa704f442bb`.
- `import_ltspice_log` (function, line 177) — **duplicate**; canonical `sc-7b10bd773106e5ee`. ID `sc-1c9b55ac73c0bcee`.
- `plot_switching_energy_map_scatter` (function, line 202) — **duplicate**; canonical `sc-217d191b91bf2329`. ID `sc-5cd85e22d14b8a49`.
- `test_extract_switching_losses_from_wfm` (function, line 256) — **reference-only**. ID `sc-b3195a4f5caf2cb1`.
- `test_extract_switching_losses_from_log_file` (function, line 291) — **reference-only**. ID `sc-c72a668971ca284f`.

**sc: `utils/dpt/helpers_plot.py`**

- `DoublePulseViewerBokeh` (class, line 22) — **duplicate**; canonical `sc-94c45f5c2ac963d6`. ID `sc-416fb6fdfc3967f9`.
- `DoublePulseViewerBokeh.__init__` (function, line 23) — **duplicate**; canonical `sc-1f61e57d7f0bf5ff`. ID `sc-f498183474130132`.
- `DoublePulseViewerBokeh.add_time_wfm` (function, line 37) — **duplicate**; canonical `sc-67594472598798ed`. ID `sc-ed448de130460fe5`.
- `DoublePulseViewerBokeh.show` (function, line 42) — **duplicate**; canonical `sc-705bf23273a886cd`. ID `sc-f1cffa4e2c5c19f9`.
- `DoublePulseViewerPlotly` (class, line 46) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-a7dc7164ca6fbe25`.
- `DoublePulseViewerPlotly.__init__` (function, line 47) — **duplicate**; canonical `sc-b5c020835c28151e`. ID `sc-e9a10cfdda0cc0ae`.
- `DoublePulseViewerPlotly.add_time_wfm` (function, line 62) — **duplicate**; canonical `sc-ce5c7005524063c6`. ID `sc-3111ba9c69dd02e7`.
- `DoublePulseViewerPlotly.show` (function, line 72) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-f7926913c80d84cf`.
- `plot_single_switching_energy_map` (function, line 81) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-c9cfcdf14fc276c8`.
- `plot_switching_energy_map` (function, line 121) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-402bc439da69cdaa`.
- `plot_switching_energy_map_scatter` (function, line 173) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-217d191b91bf2329`.

**sc: `utils/dpt/ltspice_dpt.py`**

- `find_nearest` (function, line 17) — **duplicate**; canonical `sc-81fc707fd885ac42`. ID `sc-c0df2a59aef38ff1`.
- `calc_sw_losses_ltspice` (function, line 23) — **duplicate**; canonical `sc-aa1cb8c51ad0ec40`. ID `sc-db306c492c6ba014`.
- `import_ltspice_log` (function, line 52) — **candidate-match-needs-validation**; candidate `utils/ltspice_dpt.py:180 LTspiceDPT.analyze_waveforms`. ID `sc-7b10bd773106e5ee`.

**sc: `utils/dpt/mfiles/importfile_Eoff.m`**

- `importfile_Eoff` (function, line 1) — **duplicate**; canonical `sc-6612aebd798460d8`. ID `sc-c0d13f4c2ac91578`.

**sc: `utils/dpt/mfiles/importfile_Eon.m`**

- `importfile_Eon` (function, line 1) — **duplicate**; canonical `sc-8b7a7456a3088c78`. ID `sc-781cff09bf3241c4`.

**sc: `utils/dpt/mfiles/importfile_step_descr.m`**

- `importfile_step_descr` (function, line 1) — **duplicate**; canonical `sc-2010484ad06198ef`. ID `sc-b2c9c0ee83348415`.

**sc: `utils/dpt/mfiles/xyz2grid.m`**

- `xyz2grid` (function, line 1) — **duplicate**; canonical `sc-4d62fb1625d590c7`. ID `sc-44d85bd5bc635833`.

**sc: `utils/dpt/mfiles/xyzread.m`**

- `xyzread` (function, line 1) — **duplicate**; canonical `sc-7496258f50c32467`. ID `sc-320f07f005227466`.

**sc: `utils/dpt/read_raw_upgrade.py`**

- `find_nearest` (function, line 26) — **duplicate**; canonical `sc-81fc707fd885ac42`. ID `sc-cad4c9d4194b7a31`.
- `calc_sw_losses_ltspice` (function, line 31) — **duplicate**; canonical `sc-aa1cb8c51ad0ec40`. ID `sc-d6b261d2c650b4b5`.

**sc: `utils/dpt/tests_basic.py`**

- `test_extract_switching_losses_from_wfm` (function, line 13) — **reference-only**. ID `sc-b7e1233d8aab1521`.
- `test_extract_switching_losses_from_log_file` (function, line 47) — **reference-only**. ID `sc-1277c15b9a29461a`.


### Device fitting, capacitances and signed slew

Execution: step 4 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**sc: `matlab_func/extract_linear_model_IGBT.m`**

- `extract_linear_model_IGBT` (function, line 1) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-7c3ab7c15378c562`.

**sc: `matlab_func/igbt_utils.py`**

- `dicdt_at_turnoff` (function, line 12) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-975dee050d3e39d7`.
- `dicdt_at_turnon` (function, line 18) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-36542c8d0e7dcc00`.
- `dvcedt_at_turnoff` (function, line 24) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-d7f6b2fd37d723d8`.
- `dvcedt_at_turnon` (function, line 30) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-6fb9af7f555e044f`.

**sc: `matlab_func/mosfet_switching_losses_refactor.py`**

- `Mosfet.calc_GE_mosfet_capacitances` (function, line 39) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-637b8015cc7781d0`.
- `Mosfet.calc_terminal_capacitors` (function, line 50) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-4f4e1826b5ec3b49`.

**sc: `mfiles/extract_linear_model_IGBT.m`**

- `extract_linear_model_IGBT` (function, line 1) — **duplicate**; canonical `sc-7c3ab7c15378c562`. ID `sc-9ea4771c40d8f6f3`.

**sc: `mfiles/transistor/calc_dicdt_at_turnoff.m`**

- `calc_dicdt_at_turnoff` (function, line 1) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-215991ec5210dbd0`.

**sc: `mfiles/transistor/calc_dicdt_at_turnon.m`**

- `calc_dicdt_at_turnon` (function, line 1) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-247c959e62360917`.

**sc: `mfiles/transistor/calc_dvcedt_at_turnoff.m`**

- `calc_dvcedt_at_turnoff` (function, line 1) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-5527ffb10d7041ad`.

**sc: `mfiles/transistor/calc_dvcedt_at_turnon.m`**

- `calc_dvcedt_at_turnon` (function, line 1) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-f490b19c2d4205c1`.

**sc: `mfiles/transistor/transistor.m`**

- `get.capacity` (function, line 63) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-72497474ea547b85`.
- `set.capacity` (function, line 166) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-e11112a0827c44c4`.

**sc: `semiconductors/diode.py`**

- `DiodeElectrical.fit_diode_params` (function, line 7) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-8f511124fe96b1fe`.

**sc: `semiconductors/igbt.py`**

- `extract_linear_model` (function, line 4) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-45b403aacdd153fb`.
- `dicdt_at_turnoff` (function, line 17) — **duplicate**; canonical `sc-975dee050d3e39d7`. ID `sc-fbe4d5f7b0837bc4`.
- `dicdt_at_turnon` (function, line 28) — **duplicate**; canonical `sc-36542c8d0e7dcc00`. ID `sc-2c5c836c07973678`.
- `dvcedt_at_turnoff` (function, line 39) — **duplicate**; canonical `sc-d7f6b2fd37d723d8`. ID `sc-aec022bdb8a56c49`.
- `dvcedt_at_turnon` (function, line 50) — **duplicate**; canonical `sc-6fb9af7f555e044f`. ID `sc-d557c3120bae8617`.

**sc: `semiconductors/mosfet.py`**

- `Mosfet.calc_GE_mosfet_capacitances` (function, line 22) — **duplicate**; canonical `sc-637b8015cc7781d0`. ID `sc-1c686a459310c347`.
- `Mosfet.calc_terminal_capacitors` (function, line 33) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-40fe304df5c1bfb0`.

**sc: `semiconductors/transistor.py`**

- `calc_dicdt_at_turnoff` (function, line 4) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-8df1e3c59fb2b521`.
- `Transistor.capacity` (function, line 106) — **candidate-match-needs-validation**; candidate `analytical_models.py:431 calc_device_capacitances`, `core/models.py:272 Switch.linearize_at_operating_point`. ID `sc-38246ff8d4b817b2`.


### Analytical switching model

Execution: step 4 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**sc: `semiconductors/biela_swlosses.py`**

- `SwitchingLosses` (class, line 13) — **reference-only**. ID `sc-57d89374475bf66f`.
- `SwitchingLosses.__init__` (function, line 14) — **reference-only**. ID `sc-9f694d4de236ea48`.
- `Datasheet_C2M0080120D` (class, line 30) — **reference-only**. ID `sc-17690125051ce0a1`.
- `Datasheet_C2M0080120D.__init__` (function, line 31) — **reference-only**. ID `sc-c0e84e8655acbe32`.
- `Device` (class, line 44) — **reference-only**. ID `sc-bc6707dc266ec112`.
- `Device.__init__` (function, line 45) — **reference-only**. ID `sc-a3144d94d7ff6e5d`.
- `Device.calc_reverse_recovery_params` (function, line 72) — **reference-only**. ID `sc-e43b733a1ea4694c`.
- `Device.calc_mosfet_capacitances` (function, line 81) — **reference-only**. ID `sc-ab9c6d13704feedb`.
- `Device.calc_tau_c` (function, line 102) — **reference-only**. ID `sc-4b8bfa0bac76664f`.
- `Device.func_ich_vgs` (function, line 107) — **reference-only**. ID `sc-a1bda99de89e07a6`.
- `Device.calc_transconductance_parameters` (function, line 110) — **reference-only**. ID `sc-ad7983bbef4ee3a3`.
- `Turnoff` (class, line 129) — **reference-only**. ID `sc-18ac352c082e0fc0`.
- `Turnoff.__init__` (function, line 130) — **reference-only**. ID `sc-fa30eacd8d896423`.
- `Turnon` (class, line 142) — **reference-only**. ID `sc-2982bc4fd160cb62`.
- `Turnon.__init__` (function, line 143) — **reference-only**. ID `sc-6d7c711a1d361d8c`.
- `Turnon.__init__.calc_Ioss` (function, line 163) — **reference-only**. ID `sc-b7aa76bbeb4c914c`.
- `Turnon.__init__.calc_ETon` (function, line 168) — **reference-only**. ID `sc-36e1d7afc4a112d8`.
- `test_fit_ich_vgs` (function, line 178) — **reference-only**. ID `sc-98ac7a79f27bf12d`.
- `run_example_device_C2M0080120D` (function, line 203) — **reference-only**. ID `sc-b1a5c0f1f3775262`.

**sc: `tests/biela_swlosses.py`**

- `SwitchingLosses` (class, line 14) — **reference-only**. ID `sc-b61a8461922a7587`.
- `SwitchingLosses.__init__` (function, line 15) — **reference-only**. ID `sc-b191dddc222a8286`.
- `Datasheet_C2M0080120D` (class, line 31) — **reference-only**. ID `sc-9494d05ec9b238e5`.
- `Datasheet_C2M0080120D.__init__` (function, line 32) — **reference-only**. ID `sc-5473b1eb9c4a96cd`.
- `Device` (class, line 45) — **reference-only**. ID `sc-f18ef7a38d4747d8`.
- `Device.__init__` (function, line 46) — **reference-only**. ID `sc-a97a415634bfb65f`.
- `Device.calc_reverse_recovery_params` (function, line 73) — **reference-only**. ID `sc-cc702097c930945e`.
- `Device.calc_mosfet_capacitances` (function, line 82) — **reference-only**. ID `sc-ac36e2929471ade9`.
- `Device.calc_tau_c` (function, line 103) — **reference-only**. ID `sc-9daceb807399839f`.
- `Device.func_ich_vgs` (function, line 108) — **reference-only**. ID `sc-1ffcacc4520a3364`.
- `Device.calc_transconductance_parameters` (function, line 111) — **reference-only**. ID `sc-d5813c4f011f1caf`.
- `Turnoff` (class, line 130) — **reference-only**. ID `sc-c304fe4dc0b8d383`.
- `Turnoff.__init__` (function, line 131) — **reference-only**. ID `sc-8e988896f9dfbd69`.
- `Turnon` (class, line 143) — **reference-only**. ID `sc-51ec5c4d144aae1f`.
- `Turnon.__init__` (function, line 144) — **reference-only**. ID `sc-15b80a0a8e75f43d`.
- `Turnon.__init__.calc_Ioss` (function, line 164) — **reference-only**. ID `sc-1f32e052a97556b9`.
- `Turnon.__init__.calc_ETon` (function, line 169) — **reference-only**. ID `sc-6fbeb2a3d4bdb601`.
- `test_fit_ich_vgs` (function, line 179) — **reference-only**. ID `sc-83add025c011fc00`.
- `run_example_device_C2M0080120D` (function, line 204) — **reference-only**. ID `sc-5e4d3145eebbd579`.


### Signed waveform conduction and event energies

Execution: step 5 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/__init__.py`**

- `MOSFET.get_switching_energy` (function, line 129) — **broken-adapter**; candidate `waveform_losses.py:238 calc_total_loss`. ID `enco-b69363eaa42dc78a`.
- `MOSFET.get_conduction_loss` (function, line 182) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `enco-d8cb7fba5c4178bb`.
- `MOSFET.get_switching_energy_dataframe` (function, line 217) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `enco-b4a21a0822b6e01e`.
- `MOSFET.get_switching_loss` (function, line 308) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `enco-eb4ff3ea88c23a7e`.
- `MOSFET.get_transitions` (function, line 599) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `enco-4d80b7c566e9a639`.
- `MOSFET.get_conduction_loss_index` (function, line 719) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `enco-1708782b7484876c`.
- `MOSFET.get_switching_loss_index` (function, line 724) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `enco-827a8a60d4655502`.

**sc: `matlab_func/HardSwitchedSemiconductorLoss.m`**

- `HardSwitchedSemiconductorLoss` (function, line 1) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-692a550289448210`.

**sc: `matlab_func/LLC/internalFunctions/calc_Psw_Semi_v2.m`**

- `calc_Psw_Semi_v2` (function, line 27) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-82eff1ca17f59992`.

**sc: `matlab_func/calc_Pcond_Semi.m`**

- `calc_Pcond_Semi` (function, line 15) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-e89fae5a286227f3`.

**sc: `matlab_func/calc_Psw_Semi.m`**

- `calc_Psw_Semi` (function, line 25) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-1689c86fae58bc43`.

**sc: `matlab_func/calc_switching_voltage_current.m`**

- `calc_switching_voltage_current` (function, line 1) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-dd6c793e8041f322`.

**sc: `matlab_func/mosfet_switching_losses_refactor.py`**

- `GateDriver` (class, line 13) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-90b0cf3785e8b738`.
- `GateDriver.__init__` (function, line 14) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-6e38e5930a879393`.
- `SwitchingPoint` (class, line 21) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-d2b7fc3fb15ee775`.
- `SwitchingPoint.__init__` (function, line 22) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-4da5d2718f68d8e6`.
- `Mosfet` (class, line 28) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-b1284d5bc2eac591`.
- `Mosfet.__init__` (function, line 29) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-16503f5eefbc5442`.
- `Mosfet.calc_on_voltage` (function, line 56) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-f175b4d0115c1d8f`.

**sc: `mfiles/HardSwitchedSemiconductorLoss.m`**

- `HardSwitchedSemiconductorLoss` (function, line 1) — **duplicate**; canonical `sc-692a550289448210`. ID `sc-84100dbc8c55fc2b`.

**sc: `mfiles/SemiconductorEditor.m`**

- `newConductionTrace` (function, line 742) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-ec82e01c12e412e6`.

**sc: `mfiles/calc_Pcond_Semi.m`**

- `calc_Pcond_Semi` (function, line 15) — **duplicate**; canonical `sc-e89fae5a286227f3`. ID `sc-add0076434fce68a`.

**sc: `mfiles/calc_Psw_Semi.m`**

- `calc_Psw_Semi` (function, line 25) — **duplicate**; canonical `sc-1689c86fae58bc43`. ID `sc-5069ee43b1a54072`.

**sc: `mfiles/calc_Psw_Semi_v2.m`**

- `calc_Psw_Semi_v2` (function, line 27) — **duplicate**; canonical `sc-82eff1ca17f59992`. ID `sc-4a25f4c6676cc880`.

**sc: `mfiles/calc_switching_voltage_current.m`**

- `calc_switching_voltage_current` (function, line 1) — **duplicate**; canonical `sc-dd6c793e8041f322`. ID `sc-011e1927f5d0ae7e`.

**sc: `mfiles/transistor/calc_conducting_power.m`**

- `calc_conducting_power` (function, line 1) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-3bf9486791683477`.
- `calc_single_conduction_power` (function, line 31) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-7fe7e7861425d481`.

**sc: `mfiles/transistor/calc_switching_energy.m`**

- `calc_switching_energy` (function, line 1) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-64073c6d372435ef`.

**sc: `mfiles/transistor/transistor.m`**

- `get.calc_switching_energy` (function, line 85) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-e511e16fbe85d55f`.
- `get.calc_conducting_power` (function, line 89) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-0afdfb72bbf534f6`.

**sc: `semiconductors/helpers.py`**

- `rms` (function, line 7) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-f2b26f1f95aa883e`.
- `calc_sw_losses_ltspice` (function, line 48) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-aa1cb8c51ad0ec40`.
- `calc_Pcond_Semi` (function, line 77) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-f9d8de221d3f175e`.
- `calc_Psw_Semi_v2` (function, line 137) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-e6c2d2df2d275a44`.
- `HardSwitchSemicondutorLoss` (function, line 173) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-70495a5f84dce932`.

**sc: `semiconductors/scdata.py`**

- `ScDataGeneric.calc_conducting_power` (function, line 39) — **stub**. ID `sc-ca71f3cdd1ddbe8d`.
- `ScDataGeneric.calc_switching_voltage_current` (function, line 47) — **stub**. ID `sc-2d54046c0875faec`.
- `ScDataGeneric._extract_switching_losses` (function, line 62) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-a43696a96c3a9713`.

**sc: `semiconductors/scplecs.py`**

- `ScDataPlecs.temperature_axis_conduction_loss` (function, line 86) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-ada056e722e4f828`.
- `ScDataPlecs.current_axis_conduction_loss` (function, line 92) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-44f2d6f518924b59`.
- `ScDataPlecs.scale_conduction_loss` (function, line 98) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-c9ce4afe8b269db5`.
- `ScDataPlecs.conduction_loss` (function, line 113) — **stub**. ID `sc-bea2a13cfb4bd3b2`.

**sc: `semiconductors/transistor.py`**

- `calc_pcond_semi` (function, line 19) — **candidate-match-needs-validation**; candidate `waveform_losses.py:238 calc_total_loss`. ID `sc-e212b7edf2d0e919`.
- `Transistor.calc_switching_energy` (function, line 123) — **stub**. ID `sc-d3adad76ad2f628b`.
- `Transistor.calc_conducting_power` (function, line 127) — **stub**. ID `sc-41e124839b5c5a34`.

**sc: `tests/test_basic.py`**

- `test_ScDataPlecs_temperature_axis_conduction_loss` (function, line 107) — **stub**. ID `sc-3da1c0d6115072c1`.
- `test_ScDataPlecs_current_axis_conduction_loss` (function, line 110) — **stub**. ID `sc-78c7fa49fee1e75d`.
- `test_ScDataPlecs_voltage_conduction_loss` (function, line 113) — **stub**. ID `sc-e2587b3ee98b7900`.
- `test_ScDataPlecs_conduction_loss` (function, line 119) — **stub**. ID `sc-889ead96f6bab5ef`.


### Conditioned regression and lookup tables

Execution: step 5 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco-core/src/enco_core/lut/__init__.py`**

- `LookupTable2D` (class, line 29) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-54142f01e93e50f0`.
- `LookupTable2D.__init__` (function, line 30) — **duplicate**; canonical `enco-ceeba07b4e4eab58`. ID `enco-640a9b6b64524e5f`.
- `LookupTable2D.from_dataframe` (function, line 48) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-6a2e769b11c4fb1c`.
- `LookupTable2D.from_dataframe.get_cluster_range` (function, line 69) — **duplicate**; canonical `enco-9a03e13af31c8589`. ID `enco-25f95c90dd74316e`.
- `LookupTable2D.clean_data` (function, line 169) — **duplicate**; canonical `enco-84c86c2d14f0b7c3`. ID `enco-fb277103f360e746`.
- `LookupTable2D.sort_data` (function, line 173) — **duplicate**; canonical `enco-6edc695b9654940b`. ID `enco-dfb632a46df49664`.
- `LookupTable2D.length` (function, line 177) — **duplicate**; canonical `enco-576e47154eadef15`. ID `enco-d167ce0416169b62`.
- `LookupTable2D.add_point` (function, line 180) — **duplicate**; canonical `enco-f9490e294a968c5d`. ID `enco-3d6a947284568e7c`.
- `LookupTable2D.interpolate` (function, line 184) — **duplicate**; canonical `enco-e487babf029e6451`. ID `enco-a3a2bbdadc7fda4c`.
- `LookupTable2D.get_point` (function, line 194) — **duplicate**; canonical `enco-3e2afa3ec63617d2`. ID `enco-803c0e464c4ba618`.
- `LookupTable2D.get_points` (function, line 198) — **duplicate**; canonical `enco-858ab90e3bafacbd`. ID `enco-51a0265f08bb8d11`.
- `LookupTable2D.__repr__` (function, line 240) — **duplicate**; canonical `enco-581364e1a03cbddf`. ID `enco-eaf018ae4dda4455`.
- `LookupTable2D.__str__` (function, line 243) — **duplicate**; canonical `enco-55a7d5f014d84cf3`. ID `enco-6978fbdc7c3872a0`.
- `LookupTableND` (class, line 248) — **duplicate**; canonical `enco-4dbb45b881d59715`. ID `enco-bedd60966a7a4ad8`.
- `LookupTableND.__init__` (function, line 249) — **duplicate**; canonical `enco-3eae2cc8ebc64796`. ID `enco-f1a4fc35acbffb50`.
- `LookupTableND._validate_luts` (function, line 257) — **duplicate**; canonical `enco-b8fc2f5a367b5411`. ID `enco-dc4098f52108b25e`.
- `LookupTableND.get_columns` (function, line 268) — **duplicate**; canonical `enco-97d63fab9b38bd94`. ID `enco-cc92ccbeff1f06d5`.
- `LookupTableND.get_values` (function, line 271) — **duplicate**; canonical `enco-d59e90fe9b74aa6c`. ID `enco-625011ce30d49630`.
- `LookupTableND.to_dataframe` (function, line 324) — **duplicate**; canonical `enco-83505649a5960f2c`. ID `enco-f1d34893d200273a`.
- `TestLookupTable2D` (class, line 352) — **duplicate**; canonical `enco-de63c073a94ae957`. ID `enco-cf0a20816c9e5ddb`.
- `TestLookupTable2D.setUp` (function, line 353) — **duplicate**; canonical `enco-6b215b3e71815995`. ID `enco-d9e3789a25a07943`.
- `TestLookupTable2D.test_sort_data` (function, line 357) — **duplicate**; canonical `enco-5705405c2c1c7e73`. ID `enco-13eeeb7cc2e43a4c`.
- `TestLookupTable2D.test_interpolate` (function, line 361) — **duplicate**; canonical `enco-38a896d5a14f2d83`. ID `enco-c46f2be15695eac7`.
- `TestLookupTable2D.test_get_value` (function, line 365) — **duplicate**; canonical `enco-84c5f07df8aa7152`. ID `enco-a0e8c9513d5ec1b0`.
- `TestLookupTable2D.test_get_values` (function, line 368) — **duplicate**; canonical `enco-d9bd36518af7b6a9`. ID `enco-cc27e285c53242ef`.
- `TestLookupTable2D.test_dataframe` (function, line 373) — **duplicate**; canonical `enco-a55851448085a325`. ID `enco-5b2693083cf12fc3`.
- `TestLookupTableND` (class, line 396) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-4a42ad400197b002`.
- `TestLookupTableND.setUp` (function, line 397) — **duplicate**; canonical `enco-e9c195d050b56bd6`. ID `enco-467ed6716869e9c4`.
- `TestLookupTableND.test_validate_luts` (function, line 430) — **duplicate**; canonical `enco-bcdd013de2b549a0`. ID `enco-7b11621f4260fd32`.
- `TestLookupTableND.test_get_columns` (function, line 441) — **duplicate**; canonical `enco-81d55872bddf5ad2`. ID `enco-39a4582a664ccb74`.
- `TestLookupTableND.test_get_values` (function, line 444) — **reference-only**. ID `enco-aa5cf4985efaf042`.

**enco: `enco/packages/enco-components/src/enco_components/components/diode/regressor_diode.py`**

- `_direct_feature_map` (function, line 58) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-6f7bd54bcd745e35`.
- `get_transformer` (function, line 88) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-3cf1325f1b1fc533`.
- `get_model` (function, line 102) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-8988ca08489d149d`.
- `get_trained_model` (function, line 113) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-9c727f42c115c727`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/regressor_diode_if.py`**

- `_direct_feature_map` (function, line 35) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-d5b2bc7c5ef98d81`.
- `get_transformer` (function, line 58) — **duplicate**; canonical `enco-3cf1325f1b1fc533`. ID `enco-2af734b1fb5c3060`.
- `get_model` (function, line 72) — **duplicate**; canonical `enco-8988ca08489d149d`. ID `enco-170e22a7b87aae1e`.
- `get_trained_model` (function, line 83) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-b340fe16ac4ea5d2`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/regressor_diode_vf.py`**

- `_direct_feature_map` (function, line 35) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-bab18d5c6c077766`.
- `get_transformer` (function, line 60) — **duplicate**; canonical `enco-3cf1325f1b1fc533`. ID `enco-855cb4607ce49408`.
- `get_model` (function, line 74) — **duplicate**; canonical `enco-8988ca08489d149d`. ID `enco-b34f1520fa335212`.
- `get_trained_model` (function, line 85) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-efa374abcdd83d03`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/regressor_energy.py`**

- `EnergyRegressor` (class, line 31) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-063c1f751f14429d`.
- `EnergyRegressor.__init__` (function, line 32) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-3e8a86084500bef8`.
- `EnergyRegressor.fit` (function, line 42) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-f1716ce1915dc7fd`.
- `EnergyRegressor.predict` (function, line 74) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-9946ffd3452b0579`.
- `EnergyRegressor.rmse` (function, line 83) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-cfddec9c175640d4`.
- `EnergyRegressor.r2_score` (function, line 88) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-785acc0270427985`.
- `get_model` (function, line 96) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-c4b7424ae80f165f`.
- `get_trained_model` (function, line 100) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-d39c19ad591c971f`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/regressor_eoff.py`**

- `_direct_feature_map` (function, line 35) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-e87a8649a0681e4e`.
- `get_transformer` (function, line 63) — **duplicate**; canonical `enco-3cf1325f1b1fc533`. ID `enco-c7873163593e0c5b`.
- `get_model` (function, line 77) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-8e809d82ef0921e5`.
- `get_trained_model` (function, line 88) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-c32c22c8b7e53a80`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/regressor_eon.py`**

- `_direct_feature_map` (function, line 35) — **duplicate**; canonical `enco-e87a8649a0681e4e`. ID `enco-324abcf5341c6aeb`.
- `get_transformer` (function, line 63) — **duplicate**; canonical `enco-3cf1325f1b1fc533`. ID `enco-8fa0bae43108b49f`.
- `get_model` (function, line 77) — **duplicate**; canonical `enco-8e809d82ef0921e5`. ID `enco-87bbe63d644df962`.
- `get_trained_model` (function, line 88) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-36e02ae53b5ea7d1`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/regressor_ron.py`**

- `get_model` (function, line 29) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-42b4d756af6a58ee`.
- `get_trained_model` (function, line 40) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-130c2a2161139100`.

**enco: `enco/packages/enco-components/tests/mosfet/test_regressor.py`**

- `energy_regressor_data` (function, line 26) — **reference-only**. ID `enco-4721e37c70f30a8a`.
- `test_fit` (function, line 286) — **reference-only**. ID `enco-1686bc460b39acc5`.
- `test_score` (function, line 302) — **reference-only**. ID `enco-bf0149826376bb17`.
- `test_soft_switching` (function, line 313) — **reference-only**. ID `enco-01f43d60ebd7e06d`.

**enco: `enco/packages/enco-core/src/enco_core/lut/__init__.py`**

- `LookupTable2D` (class, line 29) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-1a9ff40b88397dbe`.
- `LookupTable2D.__init__` (function, line 30) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-ceeba07b4e4eab58`.
- `LookupTable2D.from_dataframe` (function, line 48) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-c22fee084afa79dc`.
- `LookupTable2D.from_dataframe.get_cluster_range` (function, line 69) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-9a03e13af31c8589`.
- `LookupTable2D.clean_data` (function, line 171) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-84c86c2d14f0b7c3`.
- `LookupTable2D.sort_data` (function, line 175) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-6edc695b9654940b`.
- `LookupTable2D.length` (function, line 179) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-576e47154eadef15`.
- `LookupTable2D.add_point` (function, line 182) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-f9490e294a968c5d`.
- `LookupTable2D.interpolate` (function, line 186) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-e487babf029e6451`.
- `LookupTable2D.get_point` (function, line 196) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-3e2afa3ec63617d2`.
- `LookupTable2D.get_points` (function, line 200) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-858ab90e3bafacbd`.
- `LookupTable2D.__repr__` (function, line 242) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-581364e1a03cbddf`.
- `LookupTable2D.__str__` (function, line 245) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-55a7d5f014d84cf3`.
- `LookupTableND` (class, line 250) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-4dbb45b881d59715`.
- `LookupTableND.__init__` (function, line 251) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-3eae2cc8ebc64796`.
- `LookupTableND._validate_luts` (function, line 259) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-b8fc2f5a367b5411`.
- `LookupTableND.get_columns` (function, line 270) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-97d63fab9b38bd94`.
- `LookupTableND.get_values` (function, line 273) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-d59e90fe9b74aa6c`.
- `LookupTableND.to_dataframe` (function, line 328) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-83505649a5960f2c`.
- `TestLookupTable2D` (class, line 356) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-de63c073a94ae957`.
- `TestLookupTable2D.setUp` (function, line 357) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-6b215b3e71815995`.
- `TestLookupTable2D.test_sort_data` (function, line 361) — **reference-only**. ID `enco-5705405c2c1c7e73`.
- `TestLookupTable2D.test_interpolate` (function, line 365) — **reference-only**. ID `enco-38a896d5a14f2d83`.
- `TestLookupTable2D.test_get_value` (function, line 369) — **reference-only**. ID `enco-84c5f07df8aa7152`.
- `TestLookupTable2D.test_get_values` (function, line 372) — **reference-only**. ID `enco-d9bd36518af7b6a9`.
- `TestLookupTable2D.test_dataframe` (function, line 377) — **reference-only**. ID `enco-a55851448085a325`.
- `TestLookupTableND` (class, line 400) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-5c045b24002767c4`.
- `TestLookupTableND.setUp` (function, line 401) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:129 _switching_curve`. ID `enco-e9c195d050b56bd6`.
- `TestLookupTableND.test_validate_luts` (function, line 434) — **reference-only**. ID `enco-bcdd013de2b549a0`.
- `TestLookupTableND.test_get_columns` (function, line 445) — **reference-only**. ID `enco-81d55872bddf5ad2`.


### Time and frequency series operations

Execution: step 5 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco-analysis/tests/series/test_frequency_series.py`**

- `TestTimeSeries` (class, line 25) — **duplicate**; canonical `enco-eda0cf47e67b41b4`. ID `enco-f78d7dfc1e123ae9`.
- `TestTimeSeries.setUp` (function, line 26) — **stub**; canonical `enco-aadbfde5235cf72b`. ID `enco-b0957fa74a155213`.
- `TestTimeSeries.test_constructor` (function, line 29) — **duplicate**; canonical `enco-df83725f168dae3c`. ID `enco-1c558ded165404d3`.
- `TestTimeSeries.test_ladd` (function, line 47) — **duplicate**; canonical `enco-c933b9314f4b4356`. ID `enco-121b6bb76567bb1d`.
- `TestTimeSeries.test_radd` (function, line 76) — **duplicate**; canonical `enco-0990e17f4fbdc0c1`. ID `enco-eb236fe9b4fb1d9e`.
- `TestTimeSeries.test_sub` (function, line 91) — **duplicate**; canonical `enco-c1eeb6e1380e911e`. ID `enco-ea8f660053f49f46`.
- `TestTimeSeries.test_rsub` (function, line 111) — **duplicate**; canonical `enco-df3e00a16f96b26a`. ID `enco-6fec72a805b65219`.
- `TestTimeSeries.test_mul` (function, line 124) — **duplicate**; canonical `enco-9d2a194e1282e0b2`. ID `enco-7c74c3af5972c106`.
- `TestTimeSeries.test_truediv` (function, line 144) — **duplicate**; canonical `enco-3c0d63713042ccff`. ID `enco-a0f343e268058bec`.

**enco: `enco-analysis/tests/series/test_series_container.py`**

- `TestSeriesContainer` (class, line 27) — **reference-only**. ID `enco-0f9fbeaaf8a84c75`.
- `TestSeriesContainer.setUp` (function, line 28) — **reference-only**. ID `enco-244bdeba0041f229`.
- `TestSeriesContainer.test_sample_time` (function, line 53) — **duplicate**; canonical `enco-57b05874603f1c70`. ID `enco-ebb160585c6d32cc`.
- `TestSeriesContainer.test_correlation` (function, line 58) — **duplicate**; canonical `enco-bf820e1752f63628`. ID `enco-e160c10c01579054`.
- `TestSeriesContainer.test_derivative` (function, line 70) — **duplicate**; canonical `enco-a3ba9730ea493b69`. ID `enco-a56a42e4aa9f81b6`.
- `TestSeriesContainer.test_integral` (function, line 93) — **duplicate**; canonical `enco-33ea56fef8bba6c7`. ID `enco-083a27b0c80ee2db`.
- `TestSeriesContainer.test_abs` (function, line 119) — **duplicate**; canonical `enco-063471091703743b`. ID `enco-924364688d6b94c1`.
- `TestSeriesContainer.test_time_to_frequency` (function, line 127) — **reference-only**. ID `enco-fdade57a0b6fc811`.
- `TestSeriesContainer.test_get_harmonics` (function, line 138) — **reference-only**. ID `enco-0958c122556a4c6b`.
- `TestSeriesContainer.test_get_transitions` (function, line 154) — **duplicate**; canonical `enco-326e2ed8c5ac867a`. ID `enco-0389199f43b4b2de`.
- `TestSeriesContainer.test_frequency_to_time` (function, line 176) — **reference-only**. ID `enco-c5da3b9aeff13948`.
- `TestSeriesContainer.test_interpolation` (function, line 207) — **duplicate**; canonical `enco-ff80d21866043cf0`. ID `enco-552ca851e84a65cd`.
- `TestSeriesContainer.test_digital` (function, line 217) — **reference-only**. ID `enco-52b8752d7c0b8858`.

**enco: `enco-analysis/tests/series/test_time_series.py`**

- `TestTimeSeries` (class, line 25) — **reference-only**. ID `enco-16899dd641c81d97`.
- `TestTimeSeries.setUp` (function, line 26) — **stub**; canonical `enco-aadbfde5235cf72b`. ID `enco-8b67943862da8d9e`.
- `TestTimeSeries.test_rms` (function, line 29) — **duplicate**; canonical `enco-eeaef05502b605e2`. ID `enco-e0b6e27ba5c5c0bb`.
- `TestTimeSeries.test_integral` (function, line 49) — **reference-only**. ID `enco-8d4987f4542901ff`.
- `TestTimeSeries.test_ladd` (function, line 80) — **duplicate**; canonical `enco-f9fc52190d5a852c`. ID `enco-ad0f65f37f79c36a`.
- `TestTimeSeries.test_radd` (function, line 114) — **duplicate**; canonical `enco-ef78977d75a9f6bc`. ID `enco-58b59ebdc274375c`.
- `TestTimeSeries.test_sub` (function, line 128) — **duplicate**; canonical `enco-beaac877c56c6a1c`. ID `enco-c16d29b1c289322f`.
- `TestTimeSeries.test_rsub` (function, line 162) — **duplicate**; canonical `enco-edba52af22193c70`. ID `enco-f0fe8d480f51b943`.
- `TestTimeSeries.test_mul` (function, line 176) — **duplicate**; canonical `enco-7161b35be71e2909`. ID `enco-70d27cc164fc442e`.
- `TestTimeSeries.test_rmul` (function, line 210) — **duplicate**; canonical `enco-ff9850eb2391020a`. ID `enco-b3b15655f4f63cb8`.
- `TestTimeSeries.test_div` (function, line 224) — **duplicate**; canonical `enco-dfa229e8e3d3d393`. ID `enco-f0c6e29e1a2e5056`.
- `TestTimeSeries.test_rdiv` (function, line 258) — **duplicate**; canonical `enco-605d98ccd3d299ff`. ID `enco-c115ea7fd8a3e58f`.
- `TestTimeSeries.test_pow` (function, line 272) — **duplicate**; canonical `enco-c727359823133fba`. ID `enco-a7b7f5e1cfb1c74a`.

**enco: `enco-core/src/enco_core/series/frequency_series.py`**

- `FrequencySeries` (class, line 29) — **duplicate**; canonical `enco-53407944aad8c5e6`. ID `enco-c1edc65b27c0595a`.
- `FrequencySeries.__init__` (function, line 34) — **duplicate**; canonical `enco-afb003db899c8ce7`. ID `enco-0fe6f3601db49c36`.
- `FrequencySeries.new` (function, line 38) — **duplicate**; canonical `enco-ea898fee5064de30`. ID `enco-c22a265c652c1f64`.
- `FrequencySeries.container` (function, line 68) — **duplicate**; canonical `enco-b0bdcf53756e34c9`. ID `enco-236ddefbfe6371fe`.
- `FrequencySeries.identifier` (function, line 72) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-fe204cc9189e0d13`.
- `FrequencySeries.identifier` (function, line 76) — **duplicate**; canonical `enco-300df15206860ea6`. ID `enco-5bd3367468828161`.
- `FrequencySeries.label` (function, line 93) — **duplicate**; canonical `enco-e025b51efa904eef`. ID `enco-ff4da224d15a633a`.
- `FrequencySeries.label` (function, line 97) — **duplicate**; canonical `enco-44e0adcb0d35abf2`. ID `enco-9431d6132f46931a`.
- `FrequencySeries.description` (function, line 101) — **duplicate**; canonical `enco-e9aa639d99ff5e29`. ID `enco-fae7cd7103edf444`.
- `FrequencySeries.description` (function, line 105) — **duplicate**; canonical `enco-13c3f236bd8a50f5`. ID `enco-9d98b04dd72ee16c`.
- `FrequencySeries.unit` (function, line 109) — **duplicate**; canonical `enco-141315bf788bf13b`. ID `enco-7ced9f8ad0835cde`.
- `FrequencySeries.unit` (function, line 113) — **duplicate**; canonical `enco-f003dc236522084b`. ID `enco-f57a225f158267c4`.
- `FrequencySeries.set_identifier` (function, line 119) — **duplicate**; canonical `enco-9c3660f201f6a9a3`. ID `enco-20a6a7b36fcfaf20`.
- `FrequencySeries.set_frequency` (function, line 137) — **duplicate**; canonical `enco-3ed656a5b986fd39`. ID `enco-392599856809c304`.
- `FrequencySeries.set_real` (function, line 141) — **duplicate**; canonical `enco-ab2b522e5b6a4455`. ID `enco-e220e4b6fa136832`.
- `FrequencySeries.set_imag` (function, line 145) — **duplicate**; canonical `enco-447b008d0dbe4c83`. ID `enco-9cae5fca3caf2cbb`.
- `FrequencySeries.generate_magnitude` (function, line 149) — **duplicate**; canonical `enco-fd9122606f7dd484`. ID `enco-1aa36829259c1a9b`.
- `FrequencySeries.generate_phase` (function, line 163) — **duplicate**; canonical `enco-e5e57268e49c2385`. ID `enco-66490f702914d403`.
- `FrequencySeries.get_identifier` (function, line 178) — **duplicate**; canonical `enco-35350ab0bd8d3862`. ID `enco-cb961b85697c4442`.
- `FrequencySeries.get_frequency` (function, line 181) — **duplicate**; canonical `enco-b60cc03ea2cf13cc`. ID `enco-c04948ee403da259`.
- `FrequencySeries.get_real` (function, line 186) — **duplicate**; canonical `enco-3ee1212a034ffdd2`. ID `enco-e2f630d436ba5089`.
- `FrequencySeries.get_imag` (function, line 191) — **duplicate**; canonical `enco-96060f54894f823b`. ID `enco-29fb81789786ba7b`.
- `FrequencySeries.get_magnitude` (function, line 196) — **duplicate**; canonical `enco-00fb15d8ad8895c7`. ID `enco-a7fb033fd617b38c`.
- `FrequencySeries.get_phase` (function, line 213) — **duplicate**; canonical `enco-a24babfb8445098b`. ID `enco-1eb87b8304607e42`.
- `FrequencySeries.get_group_delay` (function, line 223) — **duplicate**; canonical `enco-70ec30107b9b2475`. ID `enco-4b0cd40809d6f6e6`.
- `FrequencySeries.get_violations` (function, line 228) — **duplicate**; canonical `enco-e4a3722b969ba6c9`. ID `enco-8823d0d69fc0849c`.
- `FrequencySeries.length` (function, line 261) — **duplicate**; canonical `enco-d8d77dfec56adf18`. ID `enco-41bf19cd2e01484d`.
- `FrequencySeries.min` (function, line 264) — **duplicate**; canonical `enco-3bfccb5c4e56ba41`. ID `enco-1bbc9be46b95242b`.
- `FrequencySeries.max` (function, line 268) — **duplicate**; canonical `enco-8b744851a274ac4e`. ID `enco-5d9b9fd61eafd4e7`.
- `FrequencySeries.mean` (function, line 272) — **duplicate**; canonical `enco-0fc1e4b649f2e0d9`. ID `enco-adde9b134c600a64`.
- `FrequencySeries.std` (function, line 276) — **duplicate**; canonical `enco-f6c3f7c1808b2c3f`. ID `enco-c5f0d32b22a81043`.
- `FrequencySeries.median` (function, line 280) — **duplicate**; canonical `enco-7e98da276cf257cb`. ID `enco-e31a321719ea88fa`.
- `FrequencySeries.rms` (function, line 284) — **duplicate**; canonical `enco-5385655f21d3c9ef`. ID `enco-bbae504e026cb40c`.
- `FrequencySeries.harmonics` (function, line 288) — **duplicate**; canonical `enco-f29f7269e0740345`. ID `enco-a18aea7c2f8ff361`.
- `FrequencySeries.clone` (function, line 295) — **duplicate**; canonical `enco-d749b517f444d8a0`. ID `enco-ce6ae0576fbdf68f`.
- `FrequencySeries.slice` (function, line 315) — **duplicate**; canonical `enco-633ec1f6d55c7eb3`. ID `enco-6f9b622cb19aac7f`.
- `FrequencySeries.__repr__` (function, line 320) — **duplicate**; canonical `enco-604c181a7b78b40c`. ID `enco-5f2747864fa9fe2c`.
- `FrequencySeries.__str__` (function, line 323) — **duplicate**; canonical `enco-9ae7b064bcbd5ece`. ID `enco-ed2cfdf9556ce5c3`.
- `FrequencySeries.__add__` (function, line 327) — **duplicate**; canonical `enco-07f216fd5e2f4f9f`. ID `enco-7b453ecfdf4dc053`.
- `FrequencySeries.__radd__` (function, line 393) — **duplicate**; canonical `enco-9ca9963306ebbc07`. ID `enco-52f5b5e0c89ecbcb`.
- `FrequencySeries.__sub__` (function, line 459) — **duplicate**; canonical `enco-546b8c9bc1fe064c`. ID `enco-db37f8f7f1634724`.
- `FrequencySeries.__rsub__` (function, line 525) — **duplicate**; canonical `enco-7cdd867c7e7235cf`. ID `enco-26ab3cc238ba68eb`.
- `FrequencySeries.__mul__` (function, line 591) — **duplicate**; canonical `enco-d390a0e90fc62ca6`. ID `enco-57428fd88c1676b7`.
- `FrequencySeries.__truediv__` (function, line 660) — **duplicate**; canonical `enco-3e60649198fd7c0c`. ID `enco-8d7e63edb5b7bb15`.

**enco: `enco-core/src/enco_core/series/series_container.py`**

- `SeriesDomain` (class, line 30) — **duplicate**; canonical `enco-069bd4ae7f0272fd`. ID `enco-25b1ae4792a0fc65`.
- `domain_handler` (function, line 44) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-8474362ff6d75668`.
- `domain_handler.wrapper` (function, line 46) — **duplicate**; canonical `enco-2f542bfc6b607703`. ID `enco-c4a46ebab4626ac2`.
- `SeriesContainer` (class, line 58) — **duplicate**; canonical `enco-b0c2a2ec3906d11b`. ID `enco-ac6400fe2d098f92`.
- `SeriesContainer.__init__` (function, line 63) — **duplicate**; canonical `enco-aeaab5b33fef77de`. ID `enco-df930f80f8976c26`.
- `SeriesContainer.new` (function, line 67) — **duplicate**; canonical `enco-48e6517ae0445fa4`. ID `enco-612996a51c9d116d`.
- `SeriesContainer.time_data` (function, line 77) — **duplicate**; canonical `enco-458701762eb5f319`. ID `enco-23e30171b99cf8ce`.
- `SeriesContainer.time_data` (function, line 81) — **duplicate**; canonical `enco-873e51fea57a9767`. ID `enco-ef75e870f60e96f5`.
- `SeriesContainer._requires_update` (function, line 96) — **duplicate**; canonical `enco-779dd44d6df0c93b`. ID `enco-e2c8c8a23b53ad5a`.
- `SeriesContainer._requires_update.update_hash` (function, line 118) — **duplicate**; canonical `enco-92a9c5e1c5e06ae0`. ID `enco-4eabe78709502049`.
- `SeriesContainer.with_columns` (function, line 155) — **duplicate**; canonical `enco-e023663decebc420`. ID `enco-7fa73b45b133b01e`.
- `SeriesContainer.get_columns` (function, line 163) — **duplicate**; canonical `enco-b7c712cd54768019`. ID `enco-aceef9cd22776e0b`.
- `SeriesContainer.has_column` (function, line 172) — **duplicate**; canonical `enco-f26e5f9906359a59`. ID `enco-9877808d6224a293`.
- `SeriesContainer.get_identifiers` (function, line 175) — **duplicate**; canonical `enco-3e4b6e8005673322`. ID `enco-8814055a5915ea55`.
- `SeriesContainer.join` (function, line 201) — **duplicate**; canonical `enco-b1f5ac73901175b7`. ID `enco-b9523108a899ab28`.
- `SeriesContainer.rename` (function, line 245) — **duplicate**; canonical `enco-70e3b3b91d43e7be`. ID `enco-689cfc33d0bf693e`.
- `SeriesContainer.generate_x` (function, line 302) — **duplicate**; canonical `enco-2ce9d578fca973d8`. ID `enco-129bd1cf05c28e03`.
- `SeriesContainer.generate_magnitude` (function, line 311) — **duplicate**; canonical `enco-20a214fae0d6fadf`. ID `enco-bf991fdec9d0e01a`.
- `SeriesContainer.generate_phase` (function, line 340) — **duplicate**; canonical `enco-78d511225078c0b0`. ID `enco-fdccfac5fe59fa3a`.
- `SeriesContainer.plot` (function, line 370) — **duplicate**; canonical `enco-d348fe39878e3912`. ID `enco-f2f61d2fd0ac3930`.
- `SeriesContainer.set_data` (function, line 389) — **duplicate**; canonical `enco-f95b4983b4411be3`. ID `enco-b684ff282789edba`.
- `SeriesContainer.set_metadata` (function, line 397) — **duplicate**; canonical `enco-5fb35a9d19b51c35`. ID `enco-196b69a20238d894`.
- `SeriesContainer.set_resolution` (function, line 405) — **duplicate**; canonical `enco-687d2abab993c78b`. ID `enco-c495e4ad9df70ae2`.
- `SeriesContainer.set_column` (function, line 413) — **duplicate**; canonical `enco-414dd328e9bf3e23`. ID `enco-9714a8fadb4baf35`.
- `SeriesContainer.get_data` (function, line 503) — **duplicate**; canonical `enco-0fd5b1aa19429598`. ID `enco-1074e085786f6d7a`.
- `SeriesContainer.get_metadata` (function, line 506) — **duplicate**; canonical `enco-15652d397c1e7826`. ID `enco-e5f4e811f5c83029`.
- `SeriesContainer.get_step` (function, line 521) — **duplicate**; canonical `enco-d5d359902a764be9`. ID `enco-0e7580ee16514bf7`.
- `SeriesContainer.get_column` (function, line 525) — **duplicate**; canonical `enco-4989c81d69bbd739`. ID `enco-02bdabcbc7d8e947`.
- `SeriesContainer.length` (function, line 538) — **duplicate**; canonical `enco-186b350b07714b2e`. ID `enco-169d19be253a5a44`.
- `SeriesContainer.mean` (function, line 542) — **duplicate**; canonical `enco-d9ed88a4e873cf46`. ID `enco-64d011262d425e97`.
- `SeriesContainer.std` (function, line 546) — **duplicate**; canonical `enco-c97b0c80096bf358`. ID `enco-5d73919eeb680d2f`.
- `SeriesContainer.min` (function, line 550) — **duplicate**; canonical `enco-807f9c31b25a2313`. ID `enco-610e0cf04779eacf`.
- `SeriesContainer.max` (function, line 554) — **duplicate**; canonical `enco-241741cdf8c56469`. ID `enco-4722c64c1fb13236`.
- `SeriesContainer.median` (function, line 558) — **duplicate**; canonical `enco-c353741255894641`. ID `enco-8c4c0c75ac8f0cdb`.
- `SeriesContainer.rms` (function, line 562) — **duplicate**; canonical `enco-bcf491e73df0f2bc`. ID `enco-7d832ee821406b71`.
- `SeriesContainer.count` (function, line 567) — **duplicate**; canonical `enco-9c2390c782079060`. ID `enco-c6da81359f7b314d`.
- `SeriesContainer.energy` (function, line 571) — **duplicate**; canonical `enco-2e5bc271472deac3`. ID `enco-51b2e0b8cf032a06`.
- `SeriesContainer.rmse` (function, line 574) — **duplicate**; canonical `enco-1ed9cddee5381e80`. ID `enco-6f0c62273cb7120f`.
- `SeriesContainer.correlation` (function, line 591) — **duplicate**; canonical `enco-8af8dfcc4cc9e57a`. ID `enco-e276c77262bac1e0`.
- `SeriesContainer.harmonics` (function, line 603) — **duplicate**; canonical `enco-6918ddd45dbe2531`. ID `enco-ddd6f746eeaf949f`.
- `SeriesContainer.transitions` (function, line 624) — **duplicate**; canonical `enco-ff3af9682cbb1fc0`. ID `enco-da5033cc0e7732e3`.
- `SeriesContainer.clear` (function, line 660) — **duplicate**; canonical `enco-37e937564b4605f5`. ID `enco-700f1c85792df7bf`.
- `SeriesContainer.to_csv` (function, line 671) — **duplicate**; canonical `enco-ae66362236f92fd3`. ID `enco-dffe2aa66236db39`.
- `SeriesContainer.clone` (function, line 694) — **duplicate**; canonical `enco-7d58c8f0f6fe7f0b`. ID `enco-9ac6403773f515ac`.
- `SeriesContainer.slice` (function, line 714) — **duplicate**; canonical `enco-cf5378cd519bf488`. ID `enco-79101136ecfeec03`.
- `SeriesContainer.derivative` (function, line 726) — **duplicate**; canonical `enco-28ccf411a17e82c7`. ID `enco-25f0ceb04311360b`.
- `SeriesContainer.integral` (function, line 752) — **duplicate**; canonical `enco-8abb252486b4973b`. ID `enco-67c9317ae8579c24`.
- `SeriesContainer.abs` (function, line 778) — **duplicate**; canonical `enco-32e6699949868237`. ID `enco-f72aace05efa0929`.
- `SeriesContainer.shift` (function, line 803) — **duplicate**; canonical `enco-fe1afddfe116f4b0`. ID `enco-7aec97dd08363ff0`.
- `SeriesContainer.detrend` (function, line 842) — **duplicate**; canonical `enco-58c47e746a9d0395`. ID `enco-537a00c33ff517ec`.
- `SeriesContainer.digital` (function, line 879) — **duplicate**; canonical `enco-cc098d72d654f83f`. ID `enco-bcb864e3410de56a`.
- `SeriesContainer.invert` (function, line 920) — **duplicate**; canonical `enco-75be16ec8e2167b6`. ID `enco-653de714e9a95046`.
- `SeriesContainer.time_to_frequency` (function, line 948) — **duplicate**; canonical `enco-c12d331bf0910806`. ID `enco-b65296ba2d72f80c`.
- `SeriesContainer.frequency_to_time` (function, line 1033) — **duplicate**; canonical `enco-ca1eb4d835f4dfd1`. ID `enco-f1c941e9febc2e50`.
- `SeriesContainer.sum` (function, line 1102) — **duplicate**; canonical `enco-c09ba3560ba91dd7`. ID `enco-3d74e1e583fb8f38`.
- `SeriesContainer.sub` (function, line 1110) — **duplicate**; canonical `enco-c2ce41ffd9077d71`. ID `enco-f0992d646d528caa`.
- `SeriesContainer.mul` (function, line 1118) — **duplicate**; canonical `enco-7d21679f1e2516e9`. ID `enco-bb592a9179ab45ba`.
- `SeriesContainer.div` (function, line 1126) — **duplicate**; canonical `enco-6683ead143c2d023`. ID `enco-f3b6ca1f0aed6023`.
- `SeriesContainer.pow` (function, line 1134) — **duplicate**; canonical `enco-63461a8260ea3d5e`. ID `enco-058f3e6b4171f9a8`.
- `SeriesContainer.gt` (function, line 1142) — **duplicate**; canonical `enco-d1f4ebcd38f583fd`. ID `enco-88e45ca9248bb8b8`.
- `SeriesContainer.lt` (function, line 1150) — **duplicate**; canonical `enco-db53e8cf3c971f52`. ID `enco-0b1666db8d438cf7`.
- `SeriesContainer.gte` (function, line 1158) — **duplicate**; canonical `enco-3bf66ac9e84e24c1`. ID `enco-ffa952e01cdac4ce`.
- `SeriesContainer.lte` (function, line 1166) — **duplicate**; canonical `enco-d69d6f38ed05970c`. ID `enco-cce74948283c64ff`.
- `SeriesContainer.__str__` (function, line 1175) — **duplicate**; canonical `enco-c6a6e44a9705cbfd`. ID `enco-a37500303fc8dd94`.
- `SeriesContainer.__repr__` (function, line 1178) — **duplicate**; canonical `enco-5d5d53d1172083d8`. ID `enco-037e27a137b88d15`.

**enco: `enco-core/src/enco_core/series/time_series.py`**

- `TimeSeries` (class, line 31) — **duplicate**; canonical `enco-4b69bb2321229bed`. ID `enco-cfe2b167bc5f2daa`.
- `TimeSeries.__init__` (function, line 34) — **duplicate**; canonical `enco-7751b79b2fd746ba`. ID `enco-f72e50291ee448f3`.
- `TimeSeries.new` (function, line 38) — **duplicate**; canonical `enco-c3c540446cf0f321`. ID `enco-b4316c7d6893f5bf`.
- `TimeSeries.container` (function, line 63) — **duplicate**; canonical `enco-b0bdcf53756e34c9`. ID `enco-bbf90f4d94e132e9`.
- `TimeSeries.identifier` (function, line 67) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-8fe032250085086d`.
- `TimeSeries.identifier` (function, line 71) — **duplicate**; canonical `enco-264e3a7fed8c66cf`. ID `enco-aa391b339eeeecfe`.
- `TimeSeries.label` (function, line 76) — **duplicate**; canonical `enco-8c3c0faf6da8be7b`. ID `enco-cc3e9e946c2c52a9`.
- `TimeSeries.label` (function, line 80) — **duplicate**; canonical `enco-44e0adcb0d35abf2`. ID `enco-7ebd17b359cca0d0`.
- `TimeSeries.description` (function, line 84) — **duplicate**; canonical `enco-83ba50cc2374a0d7`. ID `enco-30c5c646c9d437f6`.
- `TimeSeries.description` (function, line 88) — **duplicate**; canonical `enco-13c3f236bd8a50f5`. ID `enco-d6a97d53baaa4967`.
- `TimeSeries.unit` (function, line 92) — **duplicate**; canonical `enco-141315bf788bf13b`. ID `enco-c67d3307f23d05a1`.
- `TimeSeries.unit` (function, line 96) — **duplicate**; canonical `enco-f003dc236522084b`. ID `enco-e20a6dc5cb87f0c1`.
- `TimeSeries.__getitem__` (function, line 100) — **duplicate**; canonical `enco-b83ab39a9071f8f8`. ID `enco-40255ebbec03b129`.
- `TimeSeries.__setitem__` (function, line 109) — **duplicate**; canonical `enco-89a07f6fb168802c`. ID `enco-4710c585b4fab1aa`.
- `TimeSeries.set_identifier` (function, line 114) — **duplicate**; canonical `enco-7d5a2c357530a02a`. ID `enco-049df9744053a5b5`.
- `TimeSeries.set_x` (function, line 119) — **duplicate**; canonical `enco-b071b0bf22e870c4`. ID `enco-c437d616482d9866`.
- `TimeSeries.set_y` (function, line 122) — **duplicate**; canonical `enco-61bf017b92ba7d6a`. ID `enco-dd50290256b3df68`.
- `TimeSeries.get_identifier` (function, line 127) — **duplicate**; canonical `enco-35350ab0bd8d3862`. ID `enco-fcb3daf406bafdc9`.
- `TimeSeries.get_x` (function, line 130) — **duplicate**; canonical `enco-ec63d568f5e6859d`. ID `enco-16e3c19503e29c8f`.
- `TimeSeries.get_y` (function, line 135) — **duplicate**; canonical `enco-f640835ea25fa124`. ID `enco-50a7759fb6ecd40e`.
- `TimeSeries.length` (function, line 141) — **duplicate**; canonical `enco-a9af9cf390f3f380`. ID `enco-a9600f5c52bdcc47`.
- `TimeSeries.min` (function, line 144) — **duplicate**; canonical `enco-d2d7a1cb728209fb`. ID `enco-7e3fc7fbf61611b7`.
- `TimeSeries.max` (function, line 148) — **duplicate**; canonical `enco-be943fcc7ab42f01`. ID `enco-f56e7dedb2ceac64`.
- `TimeSeries.mean` (function, line 152) — **duplicate**; canonical `enco-e0cf986eaf38319d`. ID `enco-051b7e3663a45ac3`.
- `TimeSeries.std` (function, line 156) — **duplicate**; canonical `enco-ca9f358ffd96dbcb`. ID `enco-28d4860e3ccac55f`.
- `TimeSeries.median` (function, line 160) — **duplicate**; canonical `enco-5d251877286fba80`. ID `enco-0ea6757917517406`.
- `TimeSeries.rms` (function, line 164) — **duplicate**; canonical `enco-de89633749046286`. ID `enco-452d124db9ab295f`.
- `TimeSeries.harmonics` (function, line 168) — **duplicate**; canonical `enco-f29f7269e0740345`. ID `enco-bd6402077735914f`.
- `TimeSeries.transitions` (function, line 173) — **duplicate**; canonical `enco-d231e1c72ecf02d8`. ID `enco-44a039ccc18cecff`.
- `TimeSeries.clone` (function, line 180) — **duplicate**; canonical `enco-db1dc39a3a841937`. ID `enco-736e765ce98677e1`.
- `TimeSeries.slice` (function, line 192) — **duplicate**; canonical `enco-1633fec56ccf0027`. ID `enco-705edff9bca84435`.
- `TimeSeries.integral` (function, line 196) — **duplicate**; canonical `enco-086ac1f71ab38cf5`. ID `enco-cc02d3360a1dc885`.
- `TimeSeries.derivative` (function, line 208) — **duplicate**; canonical `enco-ffd6d21447021585`. ID `enco-0575a4a4802143bd`.
- `TimeSeries.abs` (function, line 220) — **duplicate**; canonical `enco-6d26267e0d596ed1`. ID `enco-b2bd7acb389f5e55`.
- `TimeSeries.shift` (function, line 232) — **duplicate**; canonical `enco-59ea8bb3cf7f68bf`. ID `enco-6054453cd81b1be6`.
- `TimeSeries.detrend` (function, line 249) — **duplicate**; canonical `enco-fc8d3f8a475a36d6`. ID `enco-a11c3d74c8f5c139`.
- `TimeSeries.digital` (function, line 261) — **duplicate**; canonical `enco-146b5eb9e4fd7616`. ID `enco-30409efecb0a6506`.
- `TimeSeries.to_csv` (function, line 278) — **duplicate**; canonical `enco-9e6dfb5521229c00`. ID `enco-d3e336b039afb427`.
- `TimeSeries.__repr__` (function, line 288) — **duplicate**; canonical `enco-65f506b890e6bb58`. ID `enco-cbfe9082067ca7f7`.
- `TimeSeries.__str__` (function, line 291) — **duplicate**; canonical `enco-360511b5fd6c4137`. ID `enco-e34bf6037609b5e2`.
- `TimeSeries.to_frequency_series` (function, line 295) — **duplicate**; canonical `enco-678f6803e4a59336`. ID `enco-f79dc0ad3706770f`.
- `TimeSeries.__add__` (function, line 307) — **duplicate**; canonical `enco-c96501236d03d9b9`. ID `enco-29df97ee40599d5c`.
- `TimeSeries.__radd__` (function, line 345) — **duplicate**; canonical `enco-ec1b572dbe484c73`. ID `enco-d901e7d46264d32a`.
- `TimeSeries.__sub__` (function, line 383) — **duplicate**; canonical `enco-1d53efc78661dc43`. ID `enco-d86d4dcb6d3795bf`.
- `TimeSeries.__rsub__` (function, line 421) — **duplicate**; canonical `enco-438fae5f92387958`. ID `enco-23eea778fd0c3d7b`.
- `TimeSeries.__mul__` (function, line 459) — **duplicate**; canonical `enco-fb4d85cefe65f5d3`. ID `enco-5e1cfe5e9f483de7`.
- `TimeSeries.__rmul__` (function, line 497) — **duplicate**; canonical `enco-ea25ce0b19b41a4b`. ID `enco-8106bbef171f17fe`.
- `TimeSeries.__truediv__` (function, line 535) — **duplicate**; canonical `enco-c6fc41756db1477a`. ID `enco-4729532b79466388`.
- `TimeSeries.__rtruediv__` (function, line 573) — **duplicate**; canonical `enco-55c5fea453d36496`. ID `enco-6ddbb827287648dd`.
- `TimeSeries.__pow__` (function, line 611) — **duplicate**; canonical `enco-d620afcdd7b6fed2`. ID `enco-b6a836d7d883f12d`.
- `TimeSeries.__invert__` (function, line 649) — **duplicate**; canonical `enco-d5dc0cd035db58b2`. ID `enco-5140d3d159553015`.

**enco: `enco-core/tests/series/test_frequency_series.py`**

- `TestTimeSeries` (class, line 25) — **duplicate**; canonical `enco-eda0cf47e67b41b4`. ID `enco-98feac1d84945953`.
- `TestTimeSeries.setUp` (function, line 26) — **stub**; canonical `enco-aadbfde5235cf72b`. ID `enco-d3b2da703329222e`.
- `TestTimeSeries.test_constructor` (function, line 29) — **duplicate**; canonical `enco-df83725f168dae3c`. ID `enco-68316473140bdc62`.
- `TestTimeSeries.test_ladd` (function, line 47) — **duplicate**; canonical `enco-c933b9314f4b4356`. ID `enco-b1c2d1239e2009ba`.
- `TestTimeSeries.test_radd` (function, line 76) — **duplicate**; canonical `enco-0990e17f4fbdc0c1`. ID `enco-f534ae9e29425058`.
- `TestTimeSeries.test_sub` (function, line 91) — **duplicate**; canonical `enco-c1eeb6e1380e911e`. ID `enco-07304f8e433ae4d1`.
- `TestTimeSeries.test_rsub` (function, line 111) — **duplicate**; canonical `enco-df3e00a16f96b26a`. ID `enco-e6ba352726c70b89`.
- `TestTimeSeries.test_mul` (function, line 124) — **duplicate**; canonical `enco-9d2a194e1282e0b2`. ID `enco-98e202234cf2e4cf`.
- `TestTimeSeries.test_truediv` (function, line 144) — **duplicate**; canonical `enco-3c0d63713042ccff`. ID `enco-9923b5ef17725979`.

**enco: `enco-core/tests/series/test_series_container.py`**

- `TestSeriesContainer` (class, line 27) — **duplicate**; canonical `enco-0f9fbeaaf8a84c75`. ID `enco-f33b7744378d04fc`.
- `TestSeriesContainer.setUp` (function, line 28) — **duplicate**; canonical `enco-244bdeba0041f229`. ID `enco-24ebe8f888c83fef`.
- `TestSeriesContainer.test_sample_time` (function, line 53) — **duplicate**; canonical `enco-57b05874603f1c70`. ID `enco-ddb84f034644b928`.
- `TestSeriesContainer.test_correlation` (function, line 58) — **duplicate**; canonical `enco-bf820e1752f63628`. ID `enco-2ac83013ac98251f`.
- `TestSeriesContainer.test_derivative` (function, line 70) — **duplicate**; canonical `enco-a3ba9730ea493b69`. ID `enco-e7bfabe3b5cda847`.
- `TestSeriesContainer.test_integral` (function, line 93) — **duplicate**; canonical `enco-33ea56fef8bba6c7`. ID `enco-7b4fe7d663340e7d`.
- `TestSeriesContainer.test_abs` (function, line 119) — **duplicate**; canonical `enco-063471091703743b`. ID `enco-8993b8485e9ff0ed`.
- `TestSeriesContainer.test_time_to_frequency` (function, line 127) — **duplicate**; canonical `enco-fdade57a0b6fc811`. ID `enco-d1fdc3dda6c173a2`.
- `TestSeriesContainer.test_get_harmonics` (function, line 138) — **duplicate**; canonical `enco-0958c122556a4c6b`. ID `enco-117721b91d38ab63`.
- `TestSeriesContainer.test_get_transitions` (function, line 154) — **duplicate**; canonical `enco-326e2ed8c5ac867a`. ID `enco-c87bf1727b284f4a`.
- `TestSeriesContainer.test_frequency_to_time` (function, line 176) — **duplicate**; canonical `enco-c5da3b9aeff13948`. ID `enco-9cd15e89af9cc020`.
- `TestSeriesContainer.test_interpolation` (function, line 207) — **duplicate**; canonical `enco-ff80d21866043cf0`. ID `enco-94bc810f1ffbf1b8`.
- `TestSeriesContainer.test_digital` (function, line 217) — **duplicate**; canonical `enco-52b8752d7c0b8858`. ID `enco-0185036bba607b4b`.

**enco: `enco-core/tests/series/test_time_series.py`**

- `TestTimeSeries` (class, line 25) — **duplicate**; canonical `enco-16899dd641c81d97`. ID `enco-bf0afcc857937a1e`.
- `TestTimeSeries.setUp` (function, line 26) — **stub**; canonical `enco-aadbfde5235cf72b`. ID `enco-d729a93b3878dfe9`.
- `TestTimeSeries.test_rms` (function, line 29) — **duplicate**; canonical `enco-eeaef05502b605e2`. ID `enco-a380905989368b13`.
- `TestTimeSeries.test_integral` (function, line 49) — **duplicate**; canonical `enco-8d4987f4542901ff`. ID `enco-d00251ea65ae6b4e`.
- `TestTimeSeries.test_ladd` (function, line 80) — **duplicate**; canonical `enco-f9fc52190d5a852c`. ID `enco-7653aca55b9ec7fb`.
- `TestTimeSeries.test_radd` (function, line 114) — **duplicate**; canonical `enco-ef78977d75a9f6bc`. ID `enco-86011803d122dfa0`.
- `TestTimeSeries.test_sub` (function, line 128) — **duplicate**; canonical `enco-beaac877c56c6a1c`. ID `enco-94dc164cc36f6192`.
- `TestTimeSeries.test_rsub` (function, line 162) — **duplicate**; canonical `enco-edba52af22193c70`. ID `enco-bd4dfcd484b65491`.
- `TestTimeSeries.test_mul` (function, line 176) — **duplicate**; canonical `enco-7161b35be71e2909`. ID `enco-0b864077ad804c93`.
- `TestTimeSeries.test_rmul` (function, line 210) — **duplicate**; canonical `enco-ff9850eb2391020a`. ID `enco-6148de5b95454e88`.
- `TestTimeSeries.test_div` (function, line 224) — **duplicate**; canonical `enco-dfa229e8e3d3d393`. ID `enco-4f9b3ef2eb04c94d`.
- `TestTimeSeries.test_rdiv` (function, line 258) — **duplicate**; canonical `enco-605d98ccd3d299ff`. ID `enco-e15eae57fe1e12bb`.
- `TestTimeSeries.test_pow` (function, line 272) — **duplicate**; canonical `enco-c727359823133fba`. ID `enco-c0b90bdb0c167ebd`.

**enco: `enco/packages/enco-core/src/enco_core/series/frequency_series.py`**

- `FrequencySeries` (class, line 29) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-53407944aad8c5e6`.
- `FrequencySeries.__init__` (function, line 34) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-afb003db899c8ce7`.
- `FrequencySeries.new` (function, line 38) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ea898fee5064de30`.
- `FrequencySeries.container` (function, line 68) — **duplicate**; canonical `enco-b0bdcf53756e34c9`. ID `enco-325433b12bbaf3a6`.
- `FrequencySeries.identifier` (function, line 72) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-5e52e2b3dade8f33`.
- `FrequencySeries.identifier` (function, line 76) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-300df15206860ea6`.
- `FrequencySeries.label` (function, line 93) — **duplicate**; canonical `enco-e025b51efa904eef`. ID `enco-d6651c52a189ecb4`.
- `FrequencySeries.label` (function, line 97) — **duplicate**; canonical `enco-44e0adcb0d35abf2`. ID `enco-49e9b369822582f1`.
- `FrequencySeries.description` (function, line 101) — **duplicate**; canonical `enco-e9aa639d99ff5e29`. ID `enco-a8a1731f1d2779e1`.
- `FrequencySeries.description` (function, line 105) — **duplicate**; canonical `enco-13c3f236bd8a50f5`. ID `enco-74c525a71bd56784`.
- `FrequencySeries.unit` (function, line 109) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-141315bf788bf13b`.
- `FrequencySeries.unit` (function, line 113) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-f003dc236522084b`.
- `FrequencySeries.set_identifier` (function, line 119) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-9c3660f201f6a9a3`.
- `FrequencySeries.set_frequency` (function, line 137) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-3ed656a5b986fd39`.
- `FrequencySeries.set_real` (function, line 141) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ab2b522e5b6a4455`.
- `FrequencySeries.set_imag` (function, line 145) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-447b008d0dbe4c83`.
- `FrequencySeries.generate_magnitude` (function, line 149) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-fd9122606f7dd484`.
- `FrequencySeries.generate_phase` (function, line 163) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-e5e57268e49c2385`.
- `FrequencySeries.get_identifier` (function, line 178) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-35350ab0bd8d3862`.
- `FrequencySeries.get_frequency` (function, line 181) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-b60cc03ea2cf13cc`.
- `FrequencySeries.get_real` (function, line 186) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-3ee1212a034ffdd2`.
- `FrequencySeries.get_imag` (function, line 191) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-96060f54894f823b`.
- `FrequencySeries.get_magnitude` (function, line 196) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-00fb15d8ad8895c7`.
- `FrequencySeries.get_phase` (function, line 213) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-a24babfb8445098b`.
- `FrequencySeries.get_group_delay` (function, line 223) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-70ec30107b9b2475`.
- `FrequencySeries.get_violations` (function, line 228) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-e4a3722b969ba6c9`.
- `FrequencySeries.length` (function, line 261) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d8d77dfec56adf18`.
- `FrequencySeries.min` (function, line 264) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-3bfccb5c4e56ba41`.
- `FrequencySeries.max` (function, line 268) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-8b744851a274ac4e`.
- `FrequencySeries.mean` (function, line 272) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-0fc1e4b649f2e0d9`.
- `FrequencySeries.std` (function, line 276) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-f6c3f7c1808b2c3f`.
- `FrequencySeries.median` (function, line 280) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-7e98da276cf257cb`.
- `FrequencySeries.rms` (function, line 284) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-5385655f21d3c9ef`.
- `FrequencySeries.harmonics` (function, line 288) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-f29f7269e0740345`.
- `FrequencySeries.clone` (function, line 295) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d749b517f444d8a0`.
- `FrequencySeries.slice` (function, line 315) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-633ec1f6d55c7eb3`.
- `FrequencySeries.__repr__` (function, line 320) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-604c181a7b78b40c`.
- `FrequencySeries.__str__` (function, line 323) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-9ae7b064bcbd5ece`.
- `FrequencySeries.__add__` (function, line 327) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-07f216fd5e2f4f9f`.
- `FrequencySeries.__radd__` (function, line 393) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-9ca9963306ebbc07`.
- `FrequencySeries.__sub__` (function, line 459) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-546b8c9bc1fe064c`.
- `FrequencySeries.__rsub__` (function, line 525) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-7cdd867c7e7235cf`.
- `FrequencySeries.__mul__` (function, line 591) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d390a0e90fc62ca6`.
- `FrequencySeries.__truediv__` (function, line 660) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-3e60649198fd7c0c`.

**enco: `enco/packages/enco-core/src/enco_core/series/series_container.py`**

- `SeriesDomain` (class, line 30) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-069bd4ae7f0272fd`.
- `domain_handler` (function, line 44) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-a7110f8da44a8947`.
- `domain_handler.wrapper` (function, line 46) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-2f542bfc6b607703`.
- `SeriesContainer` (class, line 58) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-b0c2a2ec3906d11b`.
- `SeriesContainer.__init__` (function, line 63) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-aeaab5b33fef77de`.
- `SeriesContainer.new` (function, line 67) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-48e6517ae0445fa4`.
- `SeriesContainer.time_data` (function, line 77) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-458701762eb5f319`.
- `SeriesContainer.time_data` (function, line 81) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-873e51fea57a9767`.
- `SeriesContainer._requires_update` (function, line 96) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-779dd44d6df0c93b`.
- `SeriesContainer._requires_update.update_hash` (function, line 118) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-92a9c5e1c5e06ae0`.
- `SeriesContainer.with_columns` (function, line 155) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-e023663decebc420`.
- `SeriesContainer.get_columns` (function, line 163) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-b7c712cd54768019`.
- `SeriesContainer.has_column` (function, line 172) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-f26e5f9906359a59`.
- `SeriesContainer.get_identifiers` (function, line 175) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-3e4b6e8005673322`.
- `SeriesContainer.join` (function, line 201) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-b1f5ac73901175b7`.
- `SeriesContainer.rename` (function, line 245) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-70e3b3b91d43e7be`.
- `SeriesContainer.generate_x` (function, line 302) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-2ce9d578fca973d8`.
- `SeriesContainer.generate_magnitude` (function, line 311) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-20a214fae0d6fadf`.
- `SeriesContainer.generate_phase` (function, line 340) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-78d511225078c0b0`.
- `SeriesContainer.plot` (function, line 370) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d348fe39878e3912`.
- `SeriesContainer.set_data` (function, line 389) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-f95b4983b4411be3`.
- `SeriesContainer.set_metadata` (function, line 397) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-5fb35a9d19b51c35`.
- `SeriesContainer.set_resolution` (function, line 405) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-687d2abab993c78b`.
- `SeriesContainer.set_column` (function, line 413) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-414dd328e9bf3e23`.
- `SeriesContainer.get_data` (function, line 503) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-0fd5b1aa19429598`.
- `SeriesContainer.get_metadata` (function, line 506) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-15652d397c1e7826`.
- `SeriesContainer.get_step` (function, line 521) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d5d359902a764be9`.
- `SeriesContainer.get_column` (function, line 525) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-4989c81d69bbd739`.
- `SeriesContainer.length` (function, line 538) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-186b350b07714b2e`.
- `SeriesContainer.mean` (function, line 542) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d9ed88a4e873cf46`.
- `SeriesContainer.std` (function, line 546) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c97b0c80096bf358`.
- `SeriesContainer.min` (function, line 550) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-807f9c31b25a2313`.
- `SeriesContainer.max` (function, line 554) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-241741cdf8c56469`.
- `SeriesContainer.median` (function, line 558) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c353741255894641`.
- `SeriesContainer.rms` (function, line 562) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-bcf491e73df0f2bc`.
- `SeriesContainer.count` (function, line 567) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-9c2390c782079060`.
- `SeriesContainer.energy` (function, line 571) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-2e5bc271472deac3`.
- `SeriesContainer.rmse` (function, line 574) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-1ed9cddee5381e80`.
- `SeriesContainer.correlation` (function, line 591) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-8af8dfcc4cc9e57a`.
- `SeriesContainer.harmonics` (function, line 603) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-6918ddd45dbe2531`.
- `SeriesContainer.transitions` (function, line 624) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ff3af9682cbb1fc0`.
- `SeriesContainer.clear` (function, line 660) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-37e937564b4605f5`.
- `SeriesContainer.to_csv` (function, line 671) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ae66362236f92fd3`.
- `SeriesContainer.clone` (function, line 694) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-7d58c8f0f6fe7f0b`.
- `SeriesContainer.slice` (function, line 714) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-cf5378cd519bf488`.
- `SeriesContainer.derivative` (function, line 726) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-28ccf411a17e82c7`.
- `SeriesContainer.integral` (function, line 752) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-8abb252486b4973b`.
- `SeriesContainer.abs` (function, line 778) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-32e6699949868237`.
- `SeriesContainer.shift` (function, line 803) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-fe1afddfe116f4b0`.
- `SeriesContainer.detrend` (function, line 842) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-58c47e746a9d0395`.
- `SeriesContainer.digital` (function, line 879) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-cc098d72d654f83f`.
- `SeriesContainer.invert` (function, line 920) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-75be16ec8e2167b6`.
- `SeriesContainer.time_to_frequency` (function, line 948) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c12d331bf0910806`.
- `SeriesContainer.frequency_to_time` (function, line 1033) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ca1eb4d835f4dfd1`.
- `SeriesContainer.sum` (function, line 1102) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c09ba3560ba91dd7`.
- `SeriesContainer.sub` (function, line 1110) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c2ce41ffd9077d71`.
- `SeriesContainer.mul` (function, line 1118) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-7d21679f1e2516e9`.
- `SeriesContainer.div` (function, line 1126) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-6683ead143c2d023`.
- `SeriesContainer.pow` (function, line 1134) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-63461a8260ea3d5e`.
- `SeriesContainer.gt` (function, line 1142) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d1f4ebcd38f583fd`.
- `SeriesContainer.lt` (function, line 1150) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-db53e8cf3c971f52`.
- `SeriesContainer.gte` (function, line 1158) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-3bf66ac9e84e24c1`.
- `SeriesContainer.lte` (function, line 1166) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d69d6f38ed05970c`.
- `SeriesContainer.__str__` (function, line 1175) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c6a6e44a9705cbfd`.
- `SeriesContainer.__repr__` (function, line 1178) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-5d5d53d1172083d8`.

**enco: `enco/packages/enco-core/src/enco_core/series/time_series.py`**

- `TimeSeries` (class, line 31) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-4b69bb2321229bed`.
- `TimeSeries.__init__` (function, line 34) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-7751b79b2fd746ba`.
- `TimeSeries.new` (function, line 38) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c3c540446cf0f321`.
- `TimeSeries.container` (function, line 63) — **duplicate**; canonical `enco-b0bdcf53756e34c9`. ID `enco-1a58e82c556efa88`.
- `TimeSeries.identifier` (function, line 67) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-ed8a498b4e7d701b`.
- `TimeSeries.identifier` (function, line 71) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-264e3a7fed8c66cf`.
- `TimeSeries.label` (function, line 76) — **duplicate**; canonical `enco-8c3c0faf6da8be7b`. ID `enco-5ef7d71e3a23f1e5`.
- `TimeSeries.label` (function, line 80) — **duplicate**; canonical `enco-44e0adcb0d35abf2`. ID `enco-fbbac894c4ce8345`.
- `TimeSeries.description` (function, line 84) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-83ba50cc2374a0d7`.
- `TimeSeries.description` (function, line 88) — **duplicate**; canonical `enco-13c3f236bd8a50f5`. ID `enco-a05856870167862a`.
- `TimeSeries.unit` (function, line 92) — **duplicate**; canonical `enco-141315bf788bf13b`. ID `enco-f66513ff6e85a11f`.
- `TimeSeries.unit` (function, line 96) — **duplicate**; canonical `enco-f003dc236522084b`. ID `enco-c5ec62f543f52ee3`.
- `TimeSeries.__getitem__` (function, line 100) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-b83ab39a9071f8f8`.
- `TimeSeries.__setitem__` (function, line 109) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-89a07f6fb168802c`.
- `TimeSeries.set_identifier` (function, line 114) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-7d5a2c357530a02a`.
- `TimeSeries.set_x` (function, line 119) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-b071b0bf22e870c4`.
- `TimeSeries.set_y` (function, line 122) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-61bf017b92ba7d6a`.
- `TimeSeries.get_identifier` (function, line 127) — **duplicate**; canonical `enco-35350ab0bd8d3862`. ID `enco-61d1904f97c19894`.
- `TimeSeries.get_x` (function, line 130) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ec63d568f5e6859d`.
- `TimeSeries.get_y` (function, line 135) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-f640835ea25fa124`.
- `TimeSeries.length` (function, line 141) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-a9af9cf390f3f380`.
- `TimeSeries.min` (function, line 144) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d2d7a1cb728209fb`.
- `TimeSeries.max` (function, line 148) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-be943fcc7ab42f01`.
- `TimeSeries.mean` (function, line 152) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-e0cf986eaf38319d`.
- `TimeSeries.std` (function, line 156) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ca9f358ffd96dbcb`.
- `TimeSeries.median` (function, line 160) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-5d251877286fba80`.
- `TimeSeries.rms` (function, line 164) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-de89633749046286`.
- `TimeSeries.harmonics` (function, line 168) — **duplicate**; canonical `enco-f29f7269e0740345`. ID `enco-161f6aa8a312b57c`.
- `TimeSeries.transitions` (function, line 173) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d231e1c72ecf02d8`.
- `TimeSeries.clone` (function, line 180) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-db1dc39a3a841937`.
- `TimeSeries.slice` (function, line 192) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-1633fec56ccf0027`.
- `TimeSeries.integral` (function, line 196) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-086ac1f71ab38cf5`.
- `TimeSeries.derivative` (function, line 208) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ffd6d21447021585`.
- `TimeSeries.abs` (function, line 220) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-6d26267e0d596ed1`.
- `TimeSeries.shift` (function, line 232) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-59ea8bb3cf7f68bf`.
- `TimeSeries.detrend` (function, line 249) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-fc8d3f8a475a36d6`.
- `TimeSeries.digital` (function, line 261) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-146b5eb9e4fd7616`.
- `TimeSeries.to_csv` (function, line 278) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-9e6dfb5521229c00`.
- `TimeSeries.__repr__` (function, line 288) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-65f506b890e6bb58`.
- `TimeSeries.__str__` (function, line 291) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-360511b5fd6c4137`.
- `TimeSeries.to_frequency_series` (function, line 295) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-678f6803e4a59336`.
- `TimeSeries.__add__` (function, line 307) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c96501236d03d9b9`.
- `TimeSeries.__radd__` (function, line 345) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ec1b572dbe484c73`.
- `TimeSeries.__sub__` (function, line 383) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-1d53efc78661dc43`.
- `TimeSeries.__rsub__` (function, line 421) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-438fae5f92387958`.
- `TimeSeries.__mul__` (function, line 459) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-fb4d85cefe65f5d3`.
- `TimeSeries.__rmul__` (function, line 497) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-ea25ce0b19b41a4b`.
- `TimeSeries.__truediv__` (function, line 535) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-c6fc41756db1477a`.
- `TimeSeries.__rtruediv__` (function, line 573) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-55c5fea453d36496`.
- `TimeSeries.__pow__` (function, line 611) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d620afcdd7b6fed2`.
- `TimeSeries.__invert__` (function, line 649) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d5dc0cd035db58b2`.

**enco: `enco/packages/enco-core/tests/series/test_frequency_series.py`**

- `TestTimeSeries` (class, line 25) — **reference-only**. ID `enco-eda0cf47e67b41b4`.
- `TestTimeSeries.setUp` (function, line 26) — **stub**. ID `enco-aadbfde5235cf72b`.
- `TestTimeSeries.test_constructor` (function, line 29) — **reference-only**. ID `enco-df83725f168dae3c`.
- `TestTimeSeries.test_ladd` (function, line 47) — **reference-only**. ID `enco-c933b9314f4b4356`.
- `TestTimeSeries.test_radd` (function, line 76) — **reference-only**. ID `enco-0990e17f4fbdc0c1`.
- `TestTimeSeries.test_sub` (function, line 91) — **reference-only**. ID `enco-c1eeb6e1380e911e`.
- `TestTimeSeries.test_rsub` (function, line 111) — **reference-only**. ID `enco-df3e00a16f96b26a`.
- `TestTimeSeries.test_mul` (function, line 124) — **reference-only**. ID `enco-9d2a194e1282e0b2`.
- `TestTimeSeries.test_truediv` (function, line 144) — **reference-only**. ID `enco-3c0d63713042ccff`.

**enco: `enco/packages/enco-core/tests/series/test_series_container.py`**

- `TestSeriesContainer` (class, line 31) — **reference-only**. ID `enco-75a12f7d0fd03162`.
- `TestSeriesContainer.setUp` (function, line 32) — **reference-only**. ID `enco-9c9385cf9e5364c7`.
- `TestSeriesContainer.test_sample_time` (function, line 59) — **reference-only**. ID `enco-57b05874603f1c70`.
- `TestSeriesContainer.test_correlation` (function, line 64) — **reference-only**. ID `enco-bf820e1752f63628`.
- `TestSeriesContainer.test_derivative` (function, line 76) — **reference-only**. ID `enco-a3ba9730ea493b69`.
- `TestSeriesContainer.test_integral` (function, line 99) — **reference-only**. ID `enco-33ea56fef8bba6c7`.
- `TestSeriesContainer.test_abs` (function, line 125) — **reference-only**. ID `enco-063471091703743b`.
- `TestSeriesContainer.test_time_to_frequency` (function, line 133) — **reference-only**. ID `enco-a090b0d47aaa3ea3`.
- `TestSeriesContainer.test_get_harmonics` (function, line 145) — **reference-only**. ID `enco-00ee389c11834a35`.
- `TestSeriesContainer.test_get_transitions` (function, line 159) — **reference-only**. ID `enco-326e2ed8c5ac867a`.
- `TestSeriesContainer.test_frequency_to_time` (function, line 181) — **reference-only**. ID `enco-9120e1cd7e620d19`.
- `TestSeriesContainer.test_interpolation` (function, line 217) — **reference-only**. ID `enco-ff80d21866043cf0`.
- `TestSeriesContainer.test_digital` (function, line 227) — **reference-only**. ID `enco-5ba574f7aef61ae2`.

**enco: `enco/packages/enco-core/tests/series/test_time_series.py`**

- `TestTimeSeries` (class, line 25) — **reference-only**. ID `enco-09d70ac7b6619250`.
- `TestTimeSeries.setUp` (function, line 26) — **stub**; canonical `enco-aadbfde5235cf72b`. ID `enco-71d41e1c8faee31f`.
- `TestTimeSeries.test_rms` (function, line 29) — **reference-only**. ID `enco-eeaef05502b605e2`.
- `TestTimeSeries.test_integral` (function, line 49) — **reference-only**. ID `enco-20e0a6593ecfbb00`.
- `TestTimeSeries.test_ladd` (function, line 80) — **reference-only**. ID `enco-f9fc52190d5a852c`.
- `TestTimeSeries.test_radd` (function, line 114) — **reference-only**. ID `enco-ef78977d75a9f6bc`.
- `TestTimeSeries.test_sub` (function, line 128) — **reference-only**. ID `enco-beaac877c56c6a1c`.
- `TestTimeSeries.test_rsub` (function, line 162) — **reference-only**. ID `enco-edba52af22193c70`.
- `TestTimeSeries.test_mul` (function, line 176) — **reference-only**. ID `enco-7161b35be71e2909`.
- `TestTimeSeries.test_rmul` (function, line 210) — **reference-only**. ID `enco-ff9850eb2391020a`.
- `TestTimeSeries.test_div` (function, line 224) — **reference-only**. ID `enco-dfa229e8e3d3d393`.
- `TestTimeSeries.test_rdiv` (function, line 258) — **reference-only**. ID `enco-605d98ccd3d299ff`.
- `TestTimeSeries.test_pow` (function, line 272) — **reference-only**. ID `enco-c727359823133fba`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/plot/frequency_series.py`**

- `FrequencySeriesPlot` (class, line 22) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-f729543ab3dce8db`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/plot/time_series.py`**

- `TimeSeriesPlot` (class, line 22) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-96c0323c60aaba2b`.

**enco: `enco/packages/enco-models/src/enco_models/series/frequency_series.py`**

- `FrequencySeriesModel` (class, line 23) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-e2731489d3d96efb`.

**enco: `enco/packages/enco-models/src/enco_models/series/series_container.py`**

- `SeriesHashModel` (class, line 25) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-d375897ab3324a47`.
- `SeriesContainerModel` (class, line 34) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-9641af1748677819`.
- `SeriesContainerModel.__init__` (function, line 50) — **duplicate**; canonical `enco-ac8341be148d8de4`. ID `enco-473a063c018903ed`.
- `SeriesContainerModel.dump` (function, line 57) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-48f5a966a9167ac0`.
- `SeriesContainerModel.load` (function, line 127) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-826cea5588ab0547`.

**enco: `enco/packages/enco-models/src/enco_models/series/time_series.py`**

- `TimeSeriesModel` (class, line 23) — **candidate-match-needs-validation**; candidate `waveform_losses.py:25 WaveformData`. ID `enco-491d246b54c2b626`.


### Parallel/composite devices and thermal paths

Execution: step 6 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco-circuits/enco_circuits/library/converter/pfc_boost_dab.py`**

- `parallel` (function, line 35) — **duplicate**; canonical `enco-338a6ab045282e4b`. ID `enco-dbd0d6af840f90ce`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/__init__.py`**

- `MOSFET.get_temperature_delta` (function, line 365) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `enco-4e88504bb2cf271e`.
- `MOSFET.get_temperature_delta.compute_delta` (function, line 389) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `enco-40fbd32bf0f00c77`.

**sc: `matlab_func/ParallelIGBTandDiode.m`**

- `ParallelIGBTandDiode` (function, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-679e1dd50075f925`.

**sc: `matlab_func/ParallelSemiconductors.m`**

- `ParallelSemiconductors` (function, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-c58e6169b11c66cd`.

**sc: `matlab_func/SemiconductorTemperature.m`**

- `SemiconductorTemperature` (function, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-ae25213abbd6b0bf`.

**sc: `matlab_func/reverseTurnedOnMOSFET.m`**

- `reverseTurnedOnMOSFET` (function, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-690e0248306e4f30`.

**sc: `mfiles/ParallelIGBTandDiode.m`**

- `ParallelIGBTandDiode` (function, line 1) — **duplicate**; canonical `sc-679e1dd50075f925`. ID `sc-3f960fd512b5adcf`.

**sc: `mfiles/ParallelSemiconductors.m`**

- `ParallelSemiconductors` (function, line 1) — **duplicate**; canonical `sc-c58e6169b11c66cd`. ID `sc-65053de8e9205cb8`.

**sc: `mfiles/SemiconductorTemperature.m`**

- `SemiconductorTemperature` (function, line 1) — **duplicate**; canonical `sc-ae25213abbd6b0bf`. ID `sc-ac4a029737f1d352`.

**sc: `mfiles/reverseTurnedOnMOSFET.m`**

- `reverseTurnedOnMOSFET` (function, line 1) — **duplicate**; canonical `sc-690e0248306e4f30`. ID `sc-b359b5e81c77859e`.

**sc: `mfiles/transistor/transistor.m`**

- `get.thermal_data` (function, line 77) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-944795e08f68a4b8`.
- `set.thermal_data` (function, line 182) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-a330f65c85af42cb`.

**sc: `semiconductors/helpers.py`**

- `calc_derating_current_frequency` (function, line 187) — **stub**. ID `sc-dbdff9aaf8d8cc3d`.

**sc: `semiconductors/scdata.py`**

- `ScDataGeneric.parallel_mosfet_diode` (function, line 50) — **stub**. ID `sc-bfb3712ed7cbc8ae`.
- `ScDataGeneric.parallel_igbt_diode` (function, line 53) — **stub**. ID `sc-959d96bd1acfb0e0`.
- `ScDataGeneric.parallel_semicondutors` (function, line 56) — **stub**. ID `sc-281fc6f2a03dc7db`.

**sc: `semiconductors/scplecs.py`**

- `ScDataPlecs.forwardThermalResistance` (function, line 117) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-aaa6c5dab7f2373d`.
- `ScDataPlecs.forwardThermalResistance_at_fsw` (function, line 124) — **stub**. ID `sc-209bcdae5a983d6f`.
- `ScDataPlecs.reverseThermalResistance` (function, line 128) — **stub**. ID `sc-7171ca02ffe7eb37`.
- `ScDataPlecs.forwardThermalInterfaceArea` (function, line 132) — **stub**. ID `sc-1a7182734e5738dd`.
- `ScDataPlecs.reverseThermalInterfaceArea` (function, line 136) — **stub**. ID `sc-d6225e73aef2108d`.
- `ScDataPlecs.commonThermalInterfaceArea` (function, line 140) — **stub**. ID `sc-75b7459ec468a214`.

**sc: `semiconductors/transistor.py`**

- `Transistor.thermal_data` (function, line 118) — **candidate-match-needs-validation**; candidate `core/models.py:71 FosterThermalModel.get_thermal_impedance`. ID `sc-dea41fb771084724`.


### Converter branch loss aggregation

Execution: step 7 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco-circuits/enco_circuits/library/converter/boost.py`**

- `BoostConverter` (class, line 33) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-ec6b5d56572b661d`.
- `BoostConverter.new` (function, line 35) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-d5d36579767a44ae`.
- `BoostConverter.initialize` (function, line 59) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7209b94ef37b71e7`.
- `BoostConverter._update_time_series` (function, line 133) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-565631f5d3eff739`.
- `BoostConverter._update_time_series.transfer_function_dm` (function, line 332) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-5da6c991c9d6cb88`.
- `BoostConverter._update_time_series.transfer_function_emc` (function, line 369) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-4d865b52caa5a22d`.
- `BoostConverter.update` (function, line 664) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c295532aa4e4fc76`.

**enco: `enco-circuits/enco_circuits/library/converter/boost_coupled.py`**

- `CoupledBoostConverter` (class, line 33) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-f40ffdce192ae870`.
- `CoupledBoostConverter.new` (function, line 35) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-72eff43ce15c6ff6`.
- `CoupledBoostConverter.initialize` (function, line 58) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7d016373be73b58f`.
- `CoupledBoostConverter._update_time_series` (function, line 129) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-e34257e1358a1399`.
- `CoupledBoostConverter.update` (function, line 361) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-71e08125d7154bdc`.

**enco: `enco-circuits/enco_circuits/library/converter/pfc_boost_dab.py`**

- `to_dbuv` (function, line 46) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-de0c935f391c6722`.
- `get_cm_voltage` (function, line 50) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-dfe31904028336a9`.
- `PFC_BoostDAB` (class, line 104) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-b711c053a0f21773`.
- `PFC_BoostDAB.new` (function, line 106) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-a65c7287210e27bd`.
- `PFC_BoostDAB.initialize` (function, line 140) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-78db2e6e79b8a479`.
- `PFC_BoostDAB._update_time_series` (function, line 314) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-50a880b4e5c933a3`.
- `PFC_BoostDAB._update_time_series.apply_rise_fall_time` (function, line 595) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-5aa2269a6917f450`.
- `PFC_BoostDAB._update_time_series.tf_lisn_ac` (function, line 805) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c7fb2a526a24d208`.
- `PFC_BoostDAB._update_time_series.tf_lisn_dc` (function, line 929) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-05f96bb671004837`.
- `PFC_BoostDAB.update` (function, line 1089) — **duplicate**; canonical `enco-c295532aa4e4fc76`. ID `enco-68eb8f1ff2b6b440`.

**enco: `enco-circuits/enco_circuits/library/converter/pfc_universal.py`**

- `PFC_Universal_V0` (class, line 31) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-82cc1416192c2e01`.
- `PFC_Universal_V0.new` (function, line 33) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7fbd8cc21589d70f`.
- `PFC_Universal_V0.initialize` (function, line 64) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-21c1b41d8eb0e5ef`.
- `PFC_Universal_V0.initialize.get_mosfet` (function, line 161) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-07bcea5b99366922`.
- `PFC_Universal_V0._update_time_series` (function, line 202) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7a8217103646f754`.
- `PFC_Universal_V0.update` (function, line 924) — **duplicate**; canonical `enco-c295532aa4e4fc76`. ID `enco-5b148e678308ef3b`.

**enco: `enco-circuits/tests/test_boost.py`**

- `TestBoostConverter` (class, line 28) — **reference-only**. ID `enco-429f49505f4a09d6`.
- `TestBoostConverter.setUp` (function, line 29) — **reference-only**. ID `enco-53161fee467917fb`.
- `TestBoostConverter.test_boost` (function, line 36) — **reference-only**. ID `enco-4cb2163f2374e267`.

**enco: `enco-circuits/tests/test_coupled_boost.py`**

- `TestCoupledBoostConverter` (class, line 27) — **reference-only**. ID `enco-a99d6cd954bdf99a`.
- `TestCoupledBoostConverter.setUp` (function, line 28) — **duplicate**; canonical `enco-53161fee467917fb`. ID `enco-87acc677c467992a`.
- `TestCoupledBoostConverter.test_boost` (function, line 35) — **reference-only**. ID `enco-66a9ce8cd3096bcc`.

**enco: `enco-circuits/tests/test_pfc_boost_dab.py`**

- `TestBoostConverter` (class, line 28) — **reference-only**. ID `enco-3b5f6101925973d1`.
- `TestBoostConverter.setUp` (function, line 29) — **duplicate**; canonical `enco-53161fee467917fb`. ID `enco-bb050998c961e43e`.
- `TestBoostConverter.test_boost` (function, line 36) — **reference-only**. ID `enco-d4e340e32e3ed196`.

**enco: `enco-circuits/tests/test_pfc_universal.py`**

- `TestCoupledBoostConverter` (class, line 31) — **reference-only**. ID `enco-70d1b5163aa06f49`.
- `TestCoupledBoostConverter.setUp` (function, line 32) — **duplicate**; canonical `enco-6e9bbfe0bd525bb9`. ID `enco-e1b65cf7c00788f8`.
- `TestCoupledBoostConverter.test_boost` (function, line 42) — **reference-only**. ID `enco-ea4c78a1dceb9149`.

**enco: `library-design/dcdc/cllc/src/project/models/cllc.py`**

- `CLLC` (class, line 27) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-4fcb7d826ac03c63`.
- `CLLC.initialize` (function, line 28) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8309ca7f9192a7ad`.
- `CLLC.update` (function, line 142) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-a83d935e4ca078ce`.
- `CLLC.finalize` (function, line 158) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-0807ff525daa8270`.

**enco: `library-design/dcdc/cllc_3p/src/project/models/converter.py`**

- `CLLC` (class, line 25) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8cb36dc9511517ec`.
- `CLLC.initialize` (function, line 26) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-63ea1c8f0b339d8a`.
- `CLLC.update` (function, line 182) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-fd1fefd0612e8e33`.
- `CLLC.finalize` (function, line 199) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-283d0d44ae68a6ad`.

**enco: `library-design/dcdc/d3ab/src/project/models/converter.py`**

- `DAB` (class, line 27) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-ba207a4d4869fe53`.
- `DAB.initialize` (function, line 28) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-0a6276f27766664c`.
- `DAB.update` (function, line 191) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-ae00b1a611b79c99`.
- `DAB.finalize` (function, line 224) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-4cd0c8dc56329f5d`.

**enco: `library-design/dcdc/dab_cf/src/project/models/dab.py`**

- `DAB` (class, line 26) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-f5b63d298c82ec9f`.
- `DAB.initialize` (function, line 27) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-214b6bb5c4c643a6`.
- `DAB.update` (function, line 151) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-acb44a97d17ce704`.
- `DAB.finalize` (function, line 170) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-074131aa5a5fc53b`.

**enco: `library-design/dcdc/psfb/src/psfb/main.py`**

- `OperatingPoint` (class, line 28) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-06ea86cc89092808`.
- `OperatingPoint.__init__` (function, line 29) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-fa74b649298a9715`.
- `OperatingPoint.__repr__` (function, line 47) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-040eea723ea6cde4`.
- `OperatingPoint.__str__` (function, line 50) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-468e3bf969b0658f`.
- `load_operating_points` (function, line 54) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8f8665eac453356d`.
- `run` (function, line 79) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-a1fd182c5ef13a72`.

**enco: `library-design/dcdc/psfb/src/psfb/mosfet.py`**

- `OperatingPoint` (class, line 28) — **duplicate**; canonical `enco-06ea86cc89092808`. ID `enco-b08e1bb47c7d21ef`.
- `OperatingPoint.__init__` (function, line 29) — **duplicate**; canonical `enco-fa74b649298a9715`. ID `enco-4dee9334cff3d2cd`.
- `OperatingPoint.__repr__` (function, line 47) — **duplicate**; canonical `enco-040eea723ea6cde4`. ID `enco-e5d1cfd54d09f47e`.
- `OperatingPoint.__str__` (function, line 50) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-a89a266e938787d3`.
- `load_operating_points` (function, line 54) — **duplicate**; canonical `enco-8f8665eac453356d`. ID `enco-7acda1603735607b`.
- `run` (function, line 79) — **duplicate**; canonical `enco-a1fd182c5ef13a72`. ID `enco-9649c97b4b49baa8`.

**enco: `library-design/dcdc/psfb/src/psfb/psfb.py`**

- `PSFB` (class, line 23) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-4ad135c72703d8a1`.
- `PSFB.initialize` (function, line 26) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-54d231cb8b3b982f`.
- `PSFB.update` (function, line 57) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-5f99dd70ba1316d5`.
- `PSFB.finalize` (function, line 68) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-291ad387c358012f`.

**enco: `library-design/dcdc/staedler_llc/src/project/components/pfc_choke.py`**

- `OBC7_PFC_Choke_Model` (class, line 24) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-4a5b4ef756386d55`.
- `OBC7_PFC_Choke` (class, line 39) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-74cc4778db474d39`.
- `OBC7_PFC_Choke.inductance` (function, line 44) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8b06cbebd73c3ea2`.
- `OBC7_PFC_Choke.inductance` (function, line 48) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-16c641b9b6575a95`.
- `OBC7_PFC_Choke.validate` (function, line 52) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-4b19cebd0b4a6bc0`.
- `OBC7_PFC_Choke.train` (function, line 56) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-3ae3681eecd15f94`.
- `OBC7_PFC_Choke.winding_loss` (function, line 60) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-222e19152f1396da`.
- `OBC7_PFC_Choke.core_loss` (function, line 67) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-24b248624805cdfe`.
- `OBC7_PFC_Choke.total_loss` (function, line 114) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-cd35d9efbf22aa29`.
- `OBC7_PFC_Choke.get_inductance` (function, line 125) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-5cfcbda6ef7d8871`.

**enco: `library-design/dcdc/staedler_llc/src/project/models/llc.py`**

- `LLC` (class, line 25) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-fab08a42286ed59f`.
- `LLC.initialize` (function, line 26) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-ef8cefe835d7a897`.
- `LLC.update` (function, line 153) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-574325d3e3d955a6`.
- `LLC.finalize` (function, line 156) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c4952fbc63e7312a`.

**enco: `library-design/project/OBC7/src/project/components/pfc_choke.py`**

- `OBC7_PFC_Choke_Model` (class, line 24) — **duplicate**; canonical `enco-4a5b4ef756386d55`. ID `enco-ac8d12ff8c531822`.
- `OBC7_PFC_Choke` (class, line 39) — **duplicate**; canonical `enco-74cc4778db474d39`. ID `enco-82cf3330681f766e`.
- `OBC7_PFC_Choke.inductance` (function, line 44) — **duplicate**; canonical `enco-8b06cbebd73c3ea2`. ID `enco-49363acc83666bca`.
- `OBC7_PFC_Choke.inductance` (function, line 48) — **duplicate**; canonical `enco-16c641b9b6575a95`. ID `enco-2f00f8385099b46f`.
- `OBC7_PFC_Choke.validate` (function, line 52) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-a227476a0a0b428b`.
- `OBC7_PFC_Choke.train` (function, line 56) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-687b7a0e9ba46d07`.
- `OBC7_PFC_Choke.winding_loss` (function, line 60) — **duplicate**; canonical `enco-222e19152f1396da`. ID `enco-5d5a288fce78a13a`.
- `OBC7_PFC_Choke.core_loss` (function, line 67) — **duplicate**; canonical `enco-24b248624805cdfe`. ID `enco-fb8a3bcd708b21f2`.
- `OBC7_PFC_Choke.total_loss` (function, line 114) — **duplicate**; canonical `enco-cd35d9efbf22aa29`. ID `enco-72ed2ce6a4b68f67`.
- `OBC7_PFC_Choke.get_inductance` (function, line 125) — **duplicate**; canonical `enco-5cfcbda6ef7d8871`. ID `enco-4e279210abf2c4a0`.

**enco: `library-design/project/OBC7/src/project/models/llc.py`**

- `LLC` (class, line 24) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-f6c1ad430d2fec30`.
- `LLC.initialize` (function, line 25) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8818ccb882c5f8ac`.
- `LLC.update` (function, line 152) — **duplicate**; canonical `enco-574325d3e3d955a6`. ID `enco-6d971340df78f084`.
- `LLC.finalize` (function, line 155) — **duplicate**; canonical `enco-c4952fbc63e7312a`. ID `enco-ccbd473071f6b04b`.

**enco: `library-design/project/OBC7/src/project/models/pfc.py`**

- `PFC` (class, line 24) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-cc6e9b17b3b55a1e`.
- `PFC.initialize` (function, line 25) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-94686fa2d4ecac9a`.
- `PFC.update` (function, line 131) — **duplicate**; canonical `enco-574325d3e3d955a6`. ID `enco-a24c13fc40554f6b`.
- `PFC.finalize` (function, line 134) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-aecc47fc35a781d9`.

**enco: `library-design/project/OBC7/src/project/pfc.py`**

- `main` (function, line 24) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-e047edd34612add6`.

**sc: `matlab_func/Bridgeless_PFC/calc_bridgeless_semiconductor_loss.m`**

- `calc_bridgeless_semiconductor_loss` (function, line 1) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-23fcb8242eb20e0e`.

**sc: `matlab_func/Bridgeless_PFC/calc_bridgeless_semiconductor_loss.py`**

- `SCLosses` (function, line 1) — **reference-only**. ID `sc-b49c7a95ee0fe10d`.

**sc: `matlab_func/DCDC_DAB/internalFunctions/calc_DAB_Psemi.m`**

- `calc_DAB_Psemi` (function, line 16) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-554047b592f977f3`.

**sc: `matlab_func/DCDC_DAB/internalFunctions/calc_Pcond_Semi.m`**

- `calc_Pcond_Semi` (function, line 15) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-ffec36cdd818c20c`.

**sc: `matlab_func/DCDC_DAB/internalFunctions/calc_Psw_Semi.m`**

- `calc_Psw_Semi` (function, line 25) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-b97c4f3eddd3d902`.

**sc: `matlab_func/DCDC_SRC_ZCS/internalFunctions/calc_Psw_Semi_v2.m`**

- `calc_Psw_Semi_v2` (function, line 27) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-4b330c95c0e9932c`.

**sc: `matlab_func/DCDC_SRC_ZCS/internalFunctions/calc_SRCZCS_Psemi.m`**

- `calc_SRCZCS_Psemi` (function, line 17) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-e61930dd6f1d52e4`.

**sc: `matlab_func/DCDC_SRC_ZVS/internalFunctions/calc_Psw_Semi.m`**

- `calc_Psw_Semi` (function, line 25) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-4e4278465740be50`.

**sc: `matlab_func/DCDC_SRC_ZVS/internalFunctions/calc_SRCZVS_Psemi.m`**

- `calc_SRCZVS_Psemi` (function, line 17) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-7fabd52d835912e5`.

**sc: `matlab_func/LLC/internalFunctions/calc_llc_Psemi.m`**

- `calc_llc_Psemi` (function, line 17) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-bac4784630d33e43`.

**sc: `matlab_func/PFC-SixSwitch/three_phase_totempole_pfc_semiconductor_loss.m`**

- `three_phase_totempole_pfc_semiconductor_loss` (function, line 1) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-f6624ccac96c07fd`.

**sc: `matlab_func/PFC_IVR/calc_vr_semiconductor_loss.m`**

- `calc_vr_semiconductor_loss` (function, line 1) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `sc-49383e7671acc4b1`.


### Circuit indicators and aggregation

Execution: step 7 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco-analysis/src/enco_analysis/circuit/circuit.py`**

- `Circuit.set_indicator` (function, line 122) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-a045d6db9ce6c35a`.
- `Circuit.get_indicator` (function, line 126) — **duplicate**; canonical `enco-c44ebd1e92f54c16`. ID `enco-ac787b17da5fbaa0`.
- `Circuit.remove_indicator` (function, line 133) — **duplicate**; canonical `enco-796c36a913d60494`. ID `enco-5572685c314708bf`.

**enco: `enco-analysis/src/enco_analysis/circuit/circuit_group.py`**

- `CircuitGroup.get_indicators` (function, line 803) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-5b1411b641d90cac`.

**enco: `enco-circuits/enco_circuits/circuit.py`**

- `Circuit.set_indicator` (function, line 102) — **duplicate**; canonical `enco-a045d6db9ce6c35a`. ID `enco-e6da2639176dea56`.
- `Circuit.get_indicator` (function, line 106) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c8049612a6a56895`.
- `Circuit.remove_indicator` (function, line 112) — **duplicate**; canonical `enco-796c36a913d60494`. ID `enco-e924e79ae65e0025`.

**enco: `enco-circuits/enco_circuits/circuit_group.py`**

- `CircuitGroup.get_indicators` (function, line 791) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-08b7b5e5825d7046`.

**enco: `enco-circuits/enco_circuits/library/converter/boost.py`**

- `BoostConverter._update_indicators` (function, line 505) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-201e819b0f1b83f4`.

**enco: `enco-circuits/enco_circuits/library/converter/boost_coupled.py`**

- `CoupledBoostConverter._update_indicators` (function, line 317) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7e6174c05633cadb`.

**enco: `enco-circuits/enco_circuits/library/converter/pfc_boost_dab.py`**

- `PFC_BoostDAB._update_indicators` (function, line 1048) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-37eed79ed0c4a181`.

**enco: `enco-circuits/enco_circuits/library/converter/pfc_universal.py`**

- `PFC_Universal_V0._update_indicators` (function, line 600) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-766dbd8954119823`.

**enco: `enco-circuits/enco_circuits/library/utility/emc_receiver.py`**

- `EMC_Receiver._update_indicators` (function, line 181) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-71cd337b390435ad`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit/circuit.py`**

- `Circuit` (class, line 43) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-262afd78eae21025`.
- `Circuit.__init__` (function, line 46) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-55d7590866a86e3b`.
- `Circuit.clone` (function, line 52) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-3d482192bd996ba9`.
- `Circuit.new` (function, line 56) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7f41a0f9f7ea554d`.
- `Circuit.container` (function, line 61) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-b0bdcf53756e34c9`.
- `Circuit.identifier` (function, line 65) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-39744b66cd1e3e46`.
- `Circuit.identifier` (function, line 69) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-6460a295b8c210a1`.
- `Circuit.label` (function, line 73) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8c3c0faf6da8be7b`.
- `Circuit.label` (function, line 77) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-44e0adcb0d35abf2`.
- `Circuit.description` (function, line 81) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-e9aa639d99ff5e29`.
- `Circuit.description` (function, line 85) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-13c3f236bd8a50f5`.
- `Circuit.container` (function, line 89) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-43880dc78405c440`.
- `Circuit.configuration` (function, line 93) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-dfa550236bc14b98`.
- `Circuit.configuration` (function, line 97) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c36e13aad0702b75`.
- `Circuit.operating_point` (function, line 101) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-81844228e4cf76ca`.
- `Circuit.operating_point` (function, line 105) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-ae50373f6e556b3c`.
- `Circuit.indicators` (function, line 109) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-a25645538b724d73`.
- `Circuit.indicators` (function, line 113) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-0f54aa8686ac2872`.
- `Circuit.components` (function, line 117) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-418395b026fc0f0a`.
- `Circuit.components` (function, line 127) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-363e4bde7bc0ff08`.
- `Circuit.attributes` (function, line 132) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-b27a8663d0e08aa2`.
- `Circuit.attributes` (function, line 136) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-977da1a9562bba2a`.
- `Circuit.set_initialize_function` (function, line 141) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-47672bd14988d247`.
- `Circuit.set_update_function` (function, line 144) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-123869d7f82edd38`.
- `Circuit.set_finalize_function` (function, line 147) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-4fb6515af98289a4`.
- `Circuit.initialize` (function, line 150) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-24f8ba814598af7a`.
- `Circuit.update` (function, line 155) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-a699b6543ab81c85`.
- `Circuit.finalize` (function, line 162) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8b3dbbb0af182014`.
- `Circuit.set_indicator` (function, line 211) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-9edc5ff247d57ee7`.
- `Circuit.get_indicator` (function, line 216) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c44ebd1e92f54c16`.
- `Circuit.remove_indicator` (function, line 225) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-796c36a913d60494`.
- `Circuit.clear_series` (function, line 232) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-ef4cec14aeeaf9f7`.
- `Circuit.clear_components` (function, line 235) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-e509417f648d06f0`.
- `Circuit.clear_indicators` (function, line 238) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-8adefa71d744ef51`.
- `Circuit.clear` (function, line 241) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-634455e2e0989966`.
- `Circuit.generate_time` (function, line 248) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-40bab6520dabac50`.
- `Circuit.get_time_series` (function, line 252) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-f7b2f5e43540e9f4`.
- `Circuit.set_time_series` (function, line 269) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c386e6f0bee0a9d7`.
- `Circuit.generate_frequency` (function, line 287) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7a3ec50a1e85775d`.
- `Circuit.get_frequency_series` (function, line 291) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-74b2b0fce9c3e1b5`.
- `Circuit.set_frequency_series` (function, line 321) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-cce985f672e5c3f9`.
- `Circuit.slice_time` (function, line 348) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-d9e7d7d992e69bc6`.
- `Circuit.slice_frequency` (function, line 351) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-d9d1b4c7d447b0d3`.
- `Circuit.set_component` (function, line 355) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-215ec3f4c84a0d50`.
- `Circuit.get_component` (function, line 359) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-d460cec200fae6a6`.
- `Circuit.remove_component` (function, line 367) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-530ec1af95fe22c2`.
- `Circuit.get_components_by_type` (function, line 373) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-12715058edc21de8`.
- `Circuit.save` (function, line 382) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-0011cc0285aa6bb1`.
- `Circuit.load` (function, line 405) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-d8aadf41df93f13e`.
- `Circuit.get_hash` (function, line 424) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-7a68d14c09c422f3`.
- `Circuit.__repr__` (function, line 445) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-6061275903fb4343`.
- `Circuit.__str__` (function, line 448) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-bef780acb28bc8a1`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit/frequency.py`**

- `attach_fs_indicators` (function, line 21) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-c1e1bfaaf2f90136`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit/mosfet.py`**

- `attach_electrical_indicators` (function, line 22) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-029676362daf9552`.
- `attach_physical_indicators` (function, line 271) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-6201203722e7f211`.
- `attach_mosfet_indicators` (function, line 448) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-5b96f894b25965da`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/__init__.py`**

- `CircuitGroup.get_indicator_dataframe` (function, line 348) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-2d4b741bac60a4e9`.
- `CircuitGroup._add_indicator_worksheet` (function, line 368) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-78194f30b2c8b85f`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/dataframe.py`**

- `_get_indicator_keys` (function, line 50) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-e3c5fd69c6e431d9`.
- `get_indicator_dataframe` (function, line 89) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-35deec4ca946d64d`.
- `get_indicator_merit_dataframe` (function, line 188) — **candidate-match-needs-validation**; candidate `topologies/operating_point.py:257 calculate_topology_operating_point`. ID `enco-ff0e2cb8fd29ffd5`.


### Catalog, FOM and Pareto exploration

Execution: step 8 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/__init__.py`**

- `MOSFET.get_volume_index` (function, line 709) — **candidate-match-needs-validation**; candidate `catalog_fom.py:128 screen_catalog`. ID `enco-e5bb43a4c901c9c4`.
- `MOSFET.get_cost_index` (function, line 755) — **candidate-match-needs-validation**; candidate `catalog_fom.py:128 screen_catalog`. ID `enco-f9693f97fd609cb8`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/plot/pareto.py`**

- `ParetoPlot` (class, line 22) — **reference-only**. ID `enco-8bb1af3924dc6655`.

**enco: `library-design/dcdc/cllc_3p/src/project/components/cllc_trasnformer.py`**

- `CLLC_Transformer.get_volume_index` (function, line 121) — **duplicate**; canonical `enco-670ad65f12c0dfde`. ID `enco-6a13ade3e1a69d5e`.

**sc: `tests/pareto_sc.ipynb`**

- `calc_calc_relative_cost` (function, line 1 / cell 1) — **reference-only**. ID `sc-b15e0159c1df7f0c`.
- `calc_utilization_factor` (function, line 8 / cell 1) — **reference-only**. ID `sc-ee2e90e015ccf811`.
- `plot_pareto1` (function, line 1 / cell 2) — **reference-only**. ID `sc-fbff7b2bb33c24f0`.
- `plot_pareto2` (function, line 28 / cell 2) — **reference-only**. ID `sc-e101922646dfc164`.

**sc: `tests/pareto_sc.py`**

- `test1` (function, line 35) — **reference-only**. ID `sc-6374f28e441bb91e`.


### Linked plots, maps and surfaces

Execution: step 8 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco-analysis/src/enco_analysis/circuit/circuit_group.py`**

- `CircuitGroup.add_plot` (function, line 167) — **duplicate**; canonical `enco-88b6dd2406937328`. ID `enco-cfe693ca091f6748`.
- `CircuitGroup.add_custom_plot` (function, line 170) — **duplicate**; canonical `enco-af130be2389fa805`. ID `enco-7062a1580831f1a3`.
- `CircuitGroup.remove_plot` (function, line 173) — **duplicate**; canonical `enco-4fb49a29e9c80faf`. ID `enco-d63a534ef3d16fe3`.
- `CircuitGroup.get_plots` (function, line 179) — **duplicate**; canonical `enco-c92f6a416ee56049`. ID `enco-dcdad46f91792d84`.
- `CircuitGroup.plot` (function, line 182) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-878be3af9aedd243`.
- `CircuitGroup._group_plots` (function, line 196) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-5b9dd56d9153c419`.
- `CircuitGroup._generate_plots` (function, line 215) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-9945abe81edae64b`.
- `CircuitGroup._generate_time_series_plot` (function, line 343) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-282650277af80fa4`.
- `CircuitGroup._generate_frequency_series_plot` (function, line 426) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-08d1a13075e554c8`.
- `CircuitGroup._generate_vbar_grouped_plot` (function, line 551) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-9effa7b5a35da9b9`.
- `CircuitGroup._generate_vbar_stacked_plot` (function, line 640) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-338b96e5db9d01fb`.

**enco: `enco-circuits/enco_circuits/circuit_group.py`**

- `CircuitGroup.add_plot` (function, line 161) — **duplicate**; canonical `enco-88b6dd2406937328`. ID `enco-753636acb637c931`.
- `CircuitGroup.add_custom_plot` (function, line 164) — **duplicate**; canonical `enco-af130be2389fa805`. ID `enco-8709af8aff224252`.
- `CircuitGroup.remove_plot` (function, line 167) — **duplicate**; canonical `enco-4fb49a29e9c80faf`. ID `enco-35a42d09f21482a5`.
- `CircuitGroup.get_plots` (function, line 173) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-23b950303d318557`.
- `CircuitGroup.plot` (function, line 176) — **duplicate**; canonical `enco-878be3af9aedd243`. ID `enco-6373abc263a29ad3`.
- `CircuitGroup._group_plots` (function, line 190) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-962261955c0dbd94`.
- `CircuitGroup._generate_plots` (function, line 209) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-138594e94c841902`.
- `CircuitGroup._generate_time_series_plot` (function, line 327) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-48d473d167c43872`.
- `CircuitGroup._generate_frequency_series_plot` (function, line 411) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-a6971bae91b685d8`.
- `CircuitGroup._generate_vbar_grouped_plot` (function, line 531) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-7d0ce9c5a3efc7a9`.
- `CircuitGroup._generate_vbar_stacked_plot` (function, line 624) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-2590f7ce54e02bc0`.

**enco: `enco-circuits/enco_circuits/library/converter/boost.py`**

- `BoostConverter.attach_default_plots` (function, line 681) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-d770cb85dc02ec41`.

**enco: `enco-circuits/enco_circuits/library/converter/boost_coupled.py`**

- `CoupledBoostConverter.attach_default_plots` (function, line 378) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-9c67bc37da23214b`.

**enco: `enco-circuits/enco_circuits/library/converter/pfc_boost_dab.py`**

- `PFC_BoostDAB.attach_default_plots` (function, line 1106) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-e162c063bc60a1e6`.

**enco: `enco-circuits/enco_circuits/library/converter/pfc_universal.py`**

- `PFC_Universal_V0.attach_default_plots` (function, line 941) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-9264eca439cc8f5d`.

**enco: `enco-circuits/enco_circuits/library/utility/emc_receiver.py`**

- `EMC_Receiver.attach_default_plots` (function, line 213) — **stub**. ID `enco-735d01ed2e7e2914`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/__init__.py`**

- `CircuitGroup.plots` (function, line 104) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-0f47e765203b4d60`.
- `CircuitGroup.plots` (function, line 108) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-aa5bb4844749451f`.
- `CircuitGroup.add_plot` (function, line 311) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-88b6dd2406937328`.
- `CircuitGroup.add_custom_plot` (function, line 314) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-af130be2389fa805`.
- `CircuitGroup.remove_plot` (function, line 317) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-4fb49a29e9c80faf`.
- `CircuitGroup.get_plots` (function, line 323) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-c92f6a416ee56049`.
- `CircuitGroup.plot` (function, line 326) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-965577f061d1cdcd`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/plot.py`**

- `generate_vbar_stacked_plot` (function, line 48) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-770709e5b596763a`.
- `group_plots` (function, line 144) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-dbc171cce395b2e5`.
- `generate_plots` (function, line 165) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-83fe63815ef185d8`.
- `generate_time_series_plot` (function, line 295) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-cdcfbdcc9da53c76`.
- `generate_frequency_series_plot` (function, line 379) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-b487eaa3ab6e6707`.
- `generate_vbar_grouped_plot` (function, line 482) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-29e85749082b06b0`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/__init__.py`**

- `MOSFET.plot_regressors` (function, line 515) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-b7d91944a862d149`.
- `MOSFET.plot_regressors.plot_energy` (function, line 524) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-b83aa2765bd7d74c`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/plot/plot.py`**

- `PlotLine` (class, line 23) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-4359f76f3362547f`.
- `PlotSpecification` (class, line 31) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-8adcca0cb918eded`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/plot/vbar_grouped.py`**

- `VBarGroupedPlot` (class, line 22) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-3c7b10a2a5b4086d`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/plot/vbar_stacked.py`**

- `VBarStackedPlot` (class, line 22) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-34f3d9e16baddeef`.

**enco: `library-design/component/body_diode/main.py`**

- `plot_results` (function, line 69) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-79c5424e13e27a0d`.

**enco: `library-design/dcdc/cllc/src/project/analysis/maximum_power.py`**

- `plot_constant_input` (function, line 48) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-e4572f780efe1a9b`.

**enco: `library-design/dcdc/cllc/src/project/models/cllc.py`**

- `CLLC.attach_plots` (function, line 458) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-3e9ac6c9b6c86438`.

**enco: `library-design/dcdc/cllc_3p/src/project/analysis/maximum_power.py`**

- `plot_constant_input` (function, line 48) — **duplicate**; canonical `enco-e4572f780efe1a9b`. ID `enco-78da035b6426b325`.

**enco: `library-design/dcdc/cllc_3p/src/project/models/converter.py`**

- `CLLC.attach_plots` (function, line 357) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-664c50b1b3fd1da7`.

**enco: `library-design/dcdc/d3ab/src/project/analysis/maximum_power.py`**

- `plot_constant_input` (function, line 50) — **duplicate**; canonical `enco-e4572f780efe1a9b`. ID `enco-96c1c682eff38c54`.

**enco: `library-design/dcdc/d3ab/src/project/models/converter.py`**

- `DAB.attach_plots` (function, line 562) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-fc84a047b033dd2e`.

**enco: `library-design/dcdc/dab_cf/src/project/analysis/maximum_power.py`**

- `plot_constant_input` (function, line 62) — **duplicate**; canonical `enco-e4572f780efe1a9b`. ID `enco-7723bebd0db33866`.

**enco: `library-design/dcdc/dab_cf/src/project/models/dab.py`**

- `DAB.attach_plots` (function, line 464) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-c3b90529c1c660e8`.

**enco: `library-design/dcdc/psfb/src/psfb/psfb.py`**

- `PSFB.attach_plots` (function, line 129) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-a00e491bac0fcdbd`.

**enco: `library-design/dcdc/staedler_llc/src/project/models/llc.py`**

- `LLC.attach_plots` (function, line 286) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-656f2090870e7df3`.

**enco: `library-design/project/OBC7/src/project/models/llc.py`**

- `LLC.attach_plots` (function, line 285) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-8972ea19ac4b4b7d`.

**enco: `library-design/project/OBC7/src/project/models/pfc.py`**

- `PFC.attach_plots` (function, line 360) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-b276c1cb33973694`.

**enco: `library-design/uncertainty/chain/src/project/main.py`**

- `plot_results` (function, line 312) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `enco-ab23b7fc0075f121`.

**sc: `semiconductors/helpers_plot.py`**

- `DoublePulseViewerBokeh` (class, line 17) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-94c45f5c2ac963d6`.
- `DoublePulseViewerBokeh.__init__` (function, line 18) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-1f61e57d7f0bf5ff`.
- `DoublePulseViewerBokeh.add_time_wfm` (function, line 32) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-67594472598798ed`.
- `DoublePulseViewerBokeh.show` (function, line 37) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-705bf23273a886cd`.
- `DoublePulseViewerPlotly_old` (class, line 41) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-d3c4bff0c63fa8f4`.
- `DoublePulseViewerPlotly_old.__init__` (function, line 42) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-b5c020835c28151e`.
- `DoublePulseViewerPlotly_old.add_time_wfm` (function, line 57) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-ce5c7005524063c6`.
- `DoublePulseViewerPlotly_old.add_surface` (function, line 67) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-0c9501bd111cf4da`.
- `DoublePulseViewerPlotly_old.show` (function, line 96) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-589a48ec6fabc7de`.
- `DoublePulseViewerPlotly` (class, line 117) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-524ab084a5201696`.
- `DoublePulseViewerPlotly.__init__` (function, line 118) — **candidate-match-needs-validation**; candidate `plot_functions.py:5 plot_soa_lib`. ID `sc-463acd8e40622f50`.
- `DoublePulseViewerPlotly.add_time_wfm` (function, line 132) — **duplicate**; canonical `sc-ce5c7005524063c6`. ID `sc-5207368422095956`.
- `DoublePulseViewerPlotly.add_surface` (function, line 142) — **duplicate**; canonical `sc-0c9501bd111cf4da`. ID `sc-590b2b5a9059abd0`.
- `DoublePulseViewerPlotly.show` (function, line 171) — **duplicate**; canonical `sc-589a48ec6fabc7de`. ID `sc-7859baaf7c013ca5`.


### Import/export and database conversion

Execution: step 8 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/__init__.py`**

- `MOSFET.to_xml` (function, line 774) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-08c541fa5cc4ab5d`.
- `MOSFET.from_xml` (function, line 780) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-71a1cead4d500fe2`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/plecs.py`**

- `xml_to_enco` (function, line 22) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-648141e2fac703f9`.
- `enco_to_xml` (function, line 246) — **stub**. ID `enco-806c8487b2f85d4b`.

**enco: `enco/packages/enco-components/src/enco_components/database/__init__.py`**

- `get_database_library` (function, line 19) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-70be41cca05711c7`.

**enco: `enco/packages/enco-components/src/enco_components/database/database.py`**

- `get_database_library` (function, line 24) — **duplicate**; canonical `enco-70be41cca05711c7`. ID `enco-29ac49fc5c8a555c`.
- `Database` (class, line 70) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-d1cef41e1a64eac1`.
- `Database.__init__` (function, line 87) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-41ae73e2834a105b`.
- `Database.update` (function, line 91) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-897b6faac138a5be`.
- `Database.get_component` (function, line 98) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-07d1bdede5ffc8fc`.

**enco: `enco/packages/enco-components/tests/mosfet/test_plecs_conversion.py`**

- `test_from_plecs` (function, line 24) — **reference-only**. ID `enco-9d709651527e8f7e`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/plecs/adapter.py`**

- `PlecsAdapter` (class, line 38) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-e368fa9f5fc65c9a`.
- `PlecsAdapter.__init__` (function, line 39) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-c243855af0e12be6`.
- `PlecsAdapter._load_executable` (function, line 62) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-969167d1d7cdb2e3`.
- `PlecsAdapter._start_executable` (function, line 115) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-72361b93b39f4f0c`.
- `PlecsAdapter._stop_executable` (function, line 144) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-5b9139deed323165`.
- `PlecsAdapter._load_parameters` (function, line 159) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-61a28c525a953699`.
- `PlecsAdapter._check_parameters` (function, line 193) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-f73039858c2f9237`.
- `PlecsAdapter._before_run` (function, line 203) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-a4562974b6df5854`.
- `PlecsAdapter._run` (function, line 211) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-325879d153f1467f`.
- `PlecsAdapter._after_run` (function, line 247) — **stub**; canonical `enco-782cd20fb5273236`. ID `enco-5146eda77c6bac3a`.
- `PlecsAdapter._load` (function, line 251) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-55663d7c9b42fb0b`.
- `PlecsAdapter.__repr__` (function, line 297) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-09a0a6e113d35aad`.
- `PlecsAdapter.__str__` (function, line 300) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-9d06fdae31b1b8c8`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/plecs/loader.py`**

- `PlecsCsvLoader` (class, line 20) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-85ec25983f19361d`.
- `PlecsCsvLoader.__init__` (function, line 21) — **stub**. ID `enco-814e7d5a52d9752b`.
- `PlecsCsvLoader.load` (function, line 24) — **stub**. ID `enco-3813b3d0f0ae59f7`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/plecs/type.py`**

- `PlecsSimulationType` (class, line 20) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `enco-0b9023bbd7dc9bde`.

**enco: `enco/packages/enco/tests/test_imports.py`**

- `test_analysis` (function, line 14) — **reference-only**. ID `enco-deebef8e67f731e3`.
- `test_components` (function, line 21) — **reference-only**. ID `enco-0d990ea02bfb56c7`.
- `test_core` (function, line 30) — **reference-only**. ID `enco-df68f0cd22e3ea9b`.
- `test_documentation` (function, line 50) — **reference-only**. ID `enco-71e66ca643b14fe9`.
- `test_models` (function, line 59) — **reference-only**. ID `enco-8e40c3e739324997`.
- `test_simulation` (function, line 69) — **reference-only**. ID `enco-f051bf883b960138`.
- `test_standards` (function, line 80) — **reference-only**. ID `enco-3c9b4ffd71087391`.

**sc: `matlab_func/ConvertscDataFileFormatFrom00To01.m`**

- `ConvertscDataFileFormatFrom00To01` (function, line 1) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-12d981e33f867021`.

**sc: `matlab_func/ConvertscDataFileFormatFrom01To02.m`**

- `ConvertscDataFileFormatFrom01To02` (function, line 1) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-cbad2e1690e07594`.

**sc: `matlab_func/convertPlecsXmlToscData02.m`**

- `convertPlecsXmlToscData02` (function, line 1) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-aff249cc865946a3`.

**sc: `matlab_func/save7.m`**

- `save7` (function, line 1) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-a6a3fecc21c6372c`.

**sc: `matlab_func/scDataByName.m`**

- `scDataByName` (function, line 1) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-378a41d28330cf60`.
- `SinglescDataByName` (function, line 26) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-9d268920806e9664`.

**sc: `matlab_func/struct2xml.m`**

- `struct2xml` (function, line 1) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-6e1f8ae0367ce5df`.
- `parseStruct` (function, line 85) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-ede2a0c133d1f38c`.
- `val2str` (function, line 164) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-35ab9391685ed869`.

**sc: `matlab_func/xml2struct.m`**

- `xml2struct` (function, line 1) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-722d13ce9d3d1e1f`.
- `parseChildNodes` (function, line 51) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-4d03cae7fff14372`.
- `getNodeData` (function, line 133) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-db7c0c6f9c5dbe12`.
- `parseAttributes` (function, line 164) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-0116fb851893c405`.
- `xmlFromString` (function, line 185) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-896e411f150519aa`.

**sc: `mfiles/ConvertscDataFileFormatFrom00To01.m`**

- `ConvertscDataFileFormatFrom00To01` (function, line 1) — **duplicate**; canonical `sc-12d981e33f867021`. ID `sc-3fbf2cafdd04aeb7`.

**sc: `mfiles/ConvertscDataFileFormatFrom01To02.m`**

- `ConvertscDataFileFormatFrom01To02` (function, line 1) — **duplicate**; canonical `sc-cbad2e1690e07594`. ID `sc-0bf8b3f58a49af46`.

**sc: `mfiles/convertPlecsXmlToscData02.m`**

- `convertPlecsXmlToscData02` (function, line 1) — **duplicate**; canonical `sc-aff249cc865946a3`. ID `sc-92dca3dc7aadf37b`.

**sc: `mfiles/save7.m`**

- `save7` (function, line 1) — **duplicate**; canonical `sc-a6a3fecc21c6372c`. ID `sc-3d28e166013d22d7`.

**sc: `mfiles/scDataByName.m`**

- `scDataByName` (function, line 1) — **duplicate**; canonical `sc-378a41d28330cf60`. ID `sc-8580c80788a011b0`.
- `SinglescDataByName` (function, line 26) — **duplicate**; canonical `sc-9d268920806e9664`. ID `sc-80d6c3e29d33d03c`.

**sc: `mfiles/struct2xml.m`**

- `struct2xml` (function, line 1) — **duplicate**; canonical `sc-6e1f8ae0367ce5df`. ID `sc-c1994d4617ba13f0`.
- `parseStruct` (function, line 85) — **duplicate**; canonical `sc-ede2a0c133d1f38c`. ID `sc-4a0717866215cbb2`.
- `val2str` (function, line 164) — **duplicate**; canonical `sc-35ab9391685ed869`. ID `sc-40173ba3ab5903f9`.

**sc: `mfiles/xml2struct.m`**

- `xml2struct` (function, line 1) — **duplicate**; canonical `sc-722d13ce9d3d1e1f`. ID `sc-bcadee091f1d7672`.
- `parseChildNodes` (function, line 51) — **duplicate**; canonical `sc-4d03cae7fff14372`. ID `sc-78f3e59402a4aaa5`.
- `getNodeData` (function, line 133) — **duplicate**; canonical `sc-db7c0c6f9c5dbe12`. ID `sc-dd9ca1a260077256`.
- `parseAttributes` (function, line 164) — **duplicate**; canonical `sc-0116fb851893c405`. ID `sc-431a46878acab021`.
- `xmlFromString` (function, line 185) — **duplicate**; canonical `sc-896e411f150519aa`. ID `sc-91bf78259f7beeda`.

**sc: `semiconductors/scdata.py`**

- `ScDataGeneric.save_plecs_xml` (function, line 35) — **stub**. ID `sc-8c1c034335850db8`.
- `ScDataGeneric.scData_by_name` (function, line 59) — **stub**. ID `sc-5112a11b2ad76818`.
- `ScDataGeneric.load_plecs_xml` (function, line 95) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-303f1e3d5b7a8037`.
- `ScData01.convertscDataFileFormatFrom00To01` (function, line 129) — **stub**. ID `sc-43a30a3be7f165e5`.
- `ScData02.convertscDataFileFormatFrom01To02` (function, line 151) — **stub**. ID `sc-fbf66ef495299cd4`.

**sc: `semiconductors/scplecs.py`**

- `ScDataPlecs` (class, line 41) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-83dcace18dbb7c79`.
- `ScDataPlecs.__init__` (function, line 42) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-9e8e1808611d5a19`.
- `ScDataPlecs.load_plecs_xml_to_dict` (function, line 58) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-f28ddc9154cb8b59`.
- `ScDataPlecs.load_plecs_xml_variables` (function, line 63) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-f14cb946305656e4`.
- `ScDataPlecs.partnumber` (function, line 70) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-e441acf000baf01e`.
- `ScDataPlecs.vendor` (function, line 74) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-864fc4c4594c375b`.
- `ScDataPlecs.package_class` (function, line 78) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-f136019130af8576`.
- `ScDataPlecs.semiconductor_type` (function, line 82) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-903258cfba3a8fab`.
- `ScDataPlecs.voltage_drop_lut` (function, line 104) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-7037ce8bb335052a`.
- `ScDataPlecs.temperature_axis_Eon` (function, line 144) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-52e123173cbaab74`.
- `ScDataPlecs.current_axis_Eon` (function, line 150) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-9e8dec9ee4354d3a`.
- `ScDataPlecs.voltage_axis_Eon` (function, line 156) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-356adb86d8c756cd`.
- `ScDataPlecs.temperature_axis_Eoff` (function, line 162) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-8da5ef74548cce9c`.
- `ScDataPlecs.current_axis_Eoff` (function, line 168) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-1fe1f7399d8c875e`.
- `ScDataPlecs.voltage_axis_Eoff` (function, line 174) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-92bf5587335a2960`.
- `ScDataPlecs.formula_Eon` (function, line 180) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-1abf99667f78bd59`.
- `ScDataPlecs.formula_Eoff` (function, line 185) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-0efa93799c500a21`.
- `ScDataPlecs.scale_Eon` (function, line 190) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-4471bd8ce0b8f6e4`.
- `ScDataPlecs.scale_Eoff` (function, line 195) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-08a6621fa48c9f7b`.
- `ScDataPlecs.Eon_lut` (function, line 200) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-4403e1caded7d15b`.
- `ScDataPlecs.Eoff_lut` (function, line 210) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-daab4deb46251f82`.
- `calc_Esw_mesh` (function, line 219) — **candidate-match-needs-validation**; candidate `plecs_importer.py:28 import_plecs_xml`. ID `sc-ebc372d2bd62ac36`.

**sc: `tests/test_basic.py`**

- `test_xmlplecs_to_scdata02` (function, line 22) — **reference-only**. ID `sc-a3b33c3d70a6a816`.
- `test_xmlplecs_igbt_to_scdata02` (function, line 31) — **reference-only**. ID `sc-ae9e3917ac6cd8ff`.
- `test_ScData01_convertscDataFileFormatFrom00To01` (function, line 45) — **stub**. ID `sc-250d7562da0ea8c4`.
- `test_ScData02_convertscDataFileFormatFrom01To02` (function, line 51) — **stub**. ID `sc-5180575300ada0d7`.
- `test_ScDataPlecsobj_C2M0025120D` (function, line 58) — **reference-only**. ID `sc-9ebea0961e8de225`.
- `test_ScDataPlecs_load_plecs_xml_to_dict` (function, line 63) — **reference-only**. ID `sc-70e00c333a50b307`.
- `test_ScDataPlecs_C2M0025120D_parameters` (function, line 68) — **reference-only**. ID `sc-c45c114d8018dc17`.
- `test_ScDataPlecs_vendor` (function, line 98) — **stub**. ID `sc-ee7f39e88e8ac420`.
- `test_ScDataPlecs_package_class` (function, line 101) — **stub**. ID `sc-401c9d752c2f355e`.
- `test_ScDataPlecs_semiconductor_type` (function, line 104) — **stub**. ID `sc-3da6f6cb12b0aae3`.
- `test_ScDataPlecs_dict_to_numpy` (function, line 116) — **stub**. ID `sc-1e376cd7a1f1f5df`.
- `test_ScDataPlecs_ScDataPlecs2scData0x` (function, line 122) — **stub**. ID `sc-bc83a5eeb4159ef9`.
- `TestXmlPlecsConversion` (class, line 145) — **reference-only**. ID `sc-5cc6eff5bf4b896c`.
- `TestXmlPlecsConversion.test_xmlplecs_to_scdata02` (function, line 147) — **reference-only**. ID `sc-b665e76ad378b6f0`.

**sc: `tests/test_xmlplecs.py`**

- `find_nearest` (function, line 5) — **duplicate**; canonical `sc-81fc707fd885ac42`. ID `sc-f535e47b0f5c24d9`.
- `ScData02` (class, line 11) — **reference-only**. ID `sc-78c81e47f46543ba`.
- `ScData02.__init__` (function, line 13) — **reference-only**. ID `sc-a3bb14946e2cd389`.
- `ScData02._extract_swithching_losses` (function, line 22) — **reference-only**. ID `sc-f08002244ba62b1e`.
- `ScData02.load_plecs_xml` (function, line 55) — **reference-only**. ID `sc-6c6179f66c5c31e1`.


### Image calibration and curve digitization

Execution: step 8 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**sc: `mfiles/DataPicker.m`**

- `DataPicker` (function, line 1) — **missing**. ID `sc-59dcb71cb79973a7`.
- `plotImage` (function, line 287) — **missing**. ID `sc-077ea85e0452003e`.
- `plotDataPoints` (function, line 300) — **missing**. ID `sc-6c9e629b2a2e1e61`.
- `plotCornerLocations` (function, line 309) — **missing**. ID `sc-e640474f5446e949`.
- `updateValues` (function, line 320) — **missing**. ID `sc-6181616afd5ee44e`.

**sc: `mfiles/analyze_image.m`**

- `analyze_image` (function, line 4) — **missing**. ID `sc-6dd251b385e740b2`.


### Device and condition data contracts

Execution: step 8 of [sprint-semiconductor-function-parity.md](sprint-semiconductor-function-parity.md)

**enco: `enco-analysis/src/enco_analysis/circuit/analysis.py`**

- `Analysis` (class, line 20) — **duplicate**; canonical `enco-bc7c9201af65bd07`. ID `enco-1a7d941499d096b1`.

**enco: `enco-analysis/src/enco_analysis/circuit/circuit.py`**

- `Circuit` (class, line 36) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c776173674119837`.
- `Circuit.__init__` (function, line 39) — **duplicate**; canonical `enco-55d7590866a86e3b`. ID `enco-53a50f38a01102c0`.
- `Circuit.new` (function, line 46) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-40e3225a28f1bb06`.
- `Circuit.container` (function, line 59) — **duplicate**; canonical `enco-b0bdcf53756e34c9`. ID `enco-1874df5a65dd63c7`.
- `Circuit.identifier` (function, line 63) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-fa29e8d719e92958`.
- `Circuit.identifier` (function, line 67) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b743522b6a2373d6`.
- `Circuit.label` (function, line 72) — **duplicate**; canonical `enco-8c3c0faf6da8be7b`. ID `enco-b4814e5ead2b2e1b`.
- `Circuit.label` (function, line 76) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e72b952097ebeb2b`.
- `Circuit.description` (function, line 80) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7f500b96ebea6b6f`.
- `Circuit.description` (function, line 84) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4c53483874538e9c`.
- `Circuit.container` (function, line 88) — **duplicate**; canonical `enco-43880dc78405c440`. ID `enco-6b233b14947463c1`.
- `Circuit.set_initialize_function` (function, line 93) — **duplicate**; canonical `enco-47672bd14988d247`. ID `enco-fb5feeb5e594f229`.
- `Circuit.set_update_function` (function, line 96) — **duplicate**; canonical `enco-123869d7f82edd38`. ID `enco-e47e44b2121e3c1f`.
- `Circuit.set_finalize_function` (function, line 99) — **duplicate**; canonical `enco-4fb6515af98289a4`. ID `enco-7c3c23af88bc8b47`.
- `Circuit.initialize` (function, line 102) — **duplicate**; canonical `enco-24f8ba814598af7a`. ID `enco-e2021079884c1bc7`.
- `Circuit.update` (function, line 107) — **duplicate**; canonical `enco-a699b6543ab81c85`. ID `enco-23c8a02aba84cf54`.
- `Circuit.finalize` (function, line 112) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-605345e3692ca1ba`.
- `Circuit.generate_time` (function, line 141) — **duplicate**; canonical `enco-40bab6520dabac50`. ID `enco-ddae9082ef0e8095`.
- `Circuit.get_time_series` (function, line 145) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-0c7145ffbfaf8414`.
- `Circuit.set_time_series` (function, line 154) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8024cc7785f6286b`.
- `Circuit.generate_frequency` (function, line 169) — **duplicate**; canonical `enco-7a3ec50a1e85775d`. ID `enco-1b8cc3d7f1642cfc`.
- `Circuit.get_frequency_series` (function, line 173) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-be3e05d6dd85c93e`.
- `Circuit.set_frequency_series` (function, line 192) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-17d2eb82ba71bdbe`.
- `Circuit.slice_time` (function, line 212) — **duplicate**; canonical `enco-d9e7d7d992e69bc6`. ID `enco-cc94c18b22684ced`.
- `Circuit.slice_frequency` (function, line 215) — **duplicate**; canonical `enco-d9d1b4c7d447b0d3`. ID `enco-ecef703cd348832b`.
- `Circuit.set_component` (function, line 219) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-9b5bf94085d24a2b`.
- `Circuit.get_component` (function, line 223) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-138f42bb21b1e46e`.
- `Circuit.remove_component` (function, line 239) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-08df8d44e0d1c184`.
- `Circuit.get_components` (function, line 245) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-92bf4e77822f9791`.
- `Circuit.__repr__` (function, line 308) — **duplicate**; canonical `enco-6061275903fb4343`. ID `enco-b75fe7fb7c604b82`.
- `Circuit.__str__` (function, line 311) — **duplicate**; canonical `enco-bef780acb28bc8a1`. ID `enco-fc0bf8f354204a8d`.

**enco: `enco-analysis/src/enco_analysis/circuit/circuit_group.py`**

- `palette` (function, line 57) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-68e66ce8a2bfd2bd`.
- `CircuitGroup` (class, line 66) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8b65d3f50635bed2`.
- `CircuitGroup.__init__` (function, line 69) — **duplicate**; canonical `enco-ef51e04562bdefef`. ID `enco-ee8403472590744c`.
- `CircuitGroup.new` (function, line 76) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-cb048043ae155c6c`.
- `CircuitGroup.identifier` (function, line 86) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-f14fa0ac5810157a`.
- `CircuitGroup.identifier` (function, line 90) — **duplicate**; canonical `enco-4ee80a55c8a3e9aa`. ID `enco-67815fcd7bca7241`.
- `CircuitGroup.label` (function, line 94) — **duplicate**; canonical `enco-8c3c0faf6da8be7b`. ID `enco-d69235d6dc876c9d`.
- `CircuitGroup.description` (function, line 98) — **duplicate**; canonical `enco-d4e3a0ffb866ee63`. ID `enco-86a16d88e86b3dc7`.
- `CircuitGroup.label` (function, line 102) — **duplicate**; canonical `enco-5627556312e8063d`. ID `enco-b1f784a4805eb529`.
- `CircuitGroup.description` (function, line 106) — **duplicate**; canonical `enco-6a0302adb2684087`. ID `enco-c80422d7ed22c5cc`.
- `CircuitGroup.timestamp` (function, line 110) — **duplicate**; canonical `enco-c10b8b3cb02dfe98`. ID `enco-4d962549a6aed455`.
- `CircuitGroup.add_circuit` (function, line 114) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-adc56c37a3bf38c7`.
- `CircuitGroup.remove_circuit` (function, line 117) — **duplicate**; canonical `enco-7668cc83bd49599c`. ID `enco-66c0e5183c846937`.
- `CircuitGroup.get_circuits` (function, line 123) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-9f9884d4a9d3776e`.
- `CircuitGroup._generate_group_information` (function, line 322) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-a179a5f8262d4f5b`.
- `CircuitGroup.save_excel` (function, line 736) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-41693cd1df199844`.

**enco: `enco-circuits/enco_circuits/__init__.py`**

- `get_design_library` (function, line 37) — **duplicate**; canonical `enco-6b30d3bb1a98d3b1`. ID `enco-16821a286828e6cc`.

**enco: `enco-circuits/enco_circuits/analysis.py`**

- `Analysis` (class, line 20) — **duplicate**; canonical `enco-bc7c9201af65bd07`. ID `enco-c574e4c05d4ed6dc`.

**enco: `enco-circuits/enco_circuits/circuit.py`**

- `Circuit` (class, line 32) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7eb5e8ba9b02a728`.
- `Circuit.__init__` (function, line 35) — **duplicate**; canonical `enco-55d7590866a86e3b`. ID `enco-7bcb09a822325c7f`.
- `Circuit.new` (function, line 42) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-21ef642b6c58695a`.
- `Circuit.identifier` (function, line 47) — **duplicate**; canonical `enco-4dcc3306cc37108f`. ID `enco-df24647f49671e0e`.
- `Circuit.label` (function, line 51) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-a7e4af44ee5b52fd`.
- `Circuit.description` (function, line 55) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e34f5ffc41c40b6d`.
- `Circuit.identifier` (function, line 59) — **duplicate**; canonical `enco-279868a2f0e08f4e`. ID `enco-4059796d95aa03c8`.
- `Circuit.label` (function, line 63) — **duplicate**; canonical `enco-bebbf72da8c316d4`. ID `enco-064492d6e9f1ab09`.
- `Circuit.description` (function, line 67) — **duplicate**; canonical `enco-58bccdca965b8b8e`. ID `enco-813ed3ce063ce9e5`.
- `Circuit.set_initialize_function` (function, line 72) — **duplicate**; canonical `enco-47672bd14988d247`. ID `enco-b6cb188e7b482040`.
- `Circuit.set_update_function` (function, line 75) — **duplicate**; canonical `enco-123869d7f82edd38`. ID `enco-f988e2228f18fb6b`.
- `Circuit.set_finalize_function` (function, line 78) — **duplicate**; canonical `enco-4fb6515af98289a4`. ID `enco-a91fd513a26012fd`.
- `Circuit.initialize` (function, line 81) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-eb4820900b13b4e6`.
- `Circuit.update` (function, line 86) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4f98417573d47e25`.
- `Circuit.finalize` (function, line 91) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-3e2a8b4eb1ecfd45`.
- `Circuit.get_time_series` (function, line 119) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2284ba226561723c`.
- `Circuit.set_time_series` (function, line 134) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e8e3f2c88830f335`.
- `Circuit.set_frequency_series` (function, line 150) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c80c94d09507a66e`.
- `Circuit.get_frequency_series` (function, line 165) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b8601029ee0cef3b`.
- `Circuit.set_component` (function, line 183) — **duplicate**; canonical `enco-9b5bf94085d24a2b`. ID `enco-73bc789fa23dc538`.
- `Circuit.get_component` (function, line 187) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8a30b8935baefbaa`.
- `Circuit.remove_component` (function, line 203) — **duplicate**; canonical `enco-08df8d44e0d1c184`. ID `enco-5536c33171419e9f`.
- `Circuit.get_components` (function, line 209) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-5a644c8eb7f87444`.
- `Circuit.save` (function, line 227) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1ba8637a5d9819c3`.
- `Circuit.load` (function, line 250) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7d9ba1339421d4a4`.
- `Circuit.__repr__` (function, line 272) — **duplicate**; canonical `enco-6061275903fb4343`. ID `enco-4c461c5a973d12a4`.
- `Circuit.__str__` (function, line 275) — **duplicate**; canonical `enco-bef780acb28bc8a1`. ID `enco-5565d17ac0248d12`.

**enco: `enco-circuits/enco_circuits/circuit_group.py`**

- `palette` (function, line 54) — **duplicate**; canonical `enco-68e66ce8a2bfd2bd`. ID `enco-40cb9ee2f9278b31`.
- `CircuitGroup` (class, line 63) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-92936315ae3ce842`.
- `CircuitGroup.__init__` (function, line 66) — **duplicate**; canonical `enco-ef51e04562bdefef`. ID `enco-79c9df0f5ffaac18`.
- `CircuitGroup.new` (function, line 73) — **duplicate**; canonical `enco-2f89036cc389f346`. ID `enco-817e8209486f504b`.
- `CircuitGroup.identifier` (function, line 78) — **duplicate**; canonical `enco-4dcc3306cc37108f`. ID `enco-954a1ac0a6bf1f56`.
- `CircuitGroup.label` (function, line 82) — **duplicate**; canonical `enco-a7e4af44ee5b52fd`. ID `enco-46148513d83eabb0`.
- `CircuitGroup.description` (function, line 86) — **duplicate**; canonical `enco-e34f5ffc41c40b6d`. ID `enco-f159c68cc0bea290`.
- `CircuitGroup.identifier` (function, line 90) — **duplicate**; canonical `enco-279868a2f0e08f4e`. ID `enco-70ac9d034921dab3`.
- `CircuitGroup.label` (function, line 94) — **duplicate**; canonical `enco-bebbf72da8c316d4`. ID `enco-e7268e44b9b99f8e`.
- `CircuitGroup.description` (function, line 98) — **duplicate**; canonical `enco-58bccdca965b8b8e`. ID `enco-c3eb89e09c55b957`.
- `CircuitGroup.add_circuit` (function, line 102) — **duplicate**; canonical `enco-adc56c37a3bf38c7`. ID `enco-b23b2f883463f405`.
- `CircuitGroup.remove_circuit` (function, line 105) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-85e4b89c2cb76573`.
- `CircuitGroup.get_circuits` (function, line 111) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4011e7b3c82f772a`.
- `CircuitGroup.save` (function, line 115) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-83f694242c942b76`.
- `CircuitGroup.load` (function, line 138) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-d06faacc8dcac719`.
- `CircuitGroup._generate_group_information` (function, line 306) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f4150d023049e89c`.
- `CircuitGroup.save_excel` (function, line 724) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c0e2058aa90e10c6`.

**enco: `enco-circuits/enco_circuits/configuration.py`**

- `update_configuration` (function, line 28) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-cc60cd27cfabfc62`.

**enco: `enco-circuits/enco_circuits/library/utility/emc_receiver.py`**

- `EMC_Receiver` (class, line 56) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-6167eebd40909adf`.
- `EMC_Receiver.new` (function, line 58) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b54e2db27aa33b53`.
- `EMC_Receiver.initialize` (function, line 63) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-812e915d89ea5fc8`.
- `EMC_Receiver.set_input` (function, line 79) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e7b879faff0c5dce`.
- `EMC_Receiver.get_average_response` (function, line 84) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-5086f855d7fa8981`.
- `EMC_Receiver.get_peak_response` (function, line 87) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-505db6b42f6fd6e3`.
- `EMC_Receiver._update_time_series` (function, line 90) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-71d684b048166ad4`.
- `EMC_Receiver.update` (function, line 196) — **duplicate**; canonical `enco-c295532aa4e4fc76`. ID `enco-25bf24a6a68d286c`.

**enco: `enco-circuits/tests/test_circuit_group.py`**

- `TestCircuitGroup` (class, line 34) — **reference-only**. ID `enco-8ccd6b38fef59e04`.
- `TestCircuitGroup.setUp` (function, line 36) — **reference-only**. ID `enco-6e9bbfe0bd525bb9`.
- `TestCircuitGroup.test_circuit_group` (function, line 46) — **reference-only**. ID `enco-bd7a82e8ab1fa61e`.

**enco: `enco-core/src/enco_core/structures/indicator.py`**

- `Indicator` (class, line 16) — **duplicate**; canonical `enco-28da68501b8c1a91`. ID `enco-cac617e47003c219`.
- `Indicator.__init__` (function, line 29) — **duplicate**; canonical `enco-42e256fe66175d55`. ID `enco-38364b81085cf821`.
- `Indicator.new` (function, line 33) — **duplicate**; canonical `enco-c2ef27d1eea2052f`. ID `enco-639f67688dc01440`.
- `Indicator.identifier` (function, line 38) — **duplicate**; canonical `enco-4dcc3306cc37108f`. ID `enco-68adc691ba79946c`.
- `Indicator.label` (function, line 42) — **duplicate**; canonical `enco-8be544a37b27c11e`. ID `enco-dec73e68bc6f1d37`.
- `Indicator.description` (function, line 46) — **duplicate**; canonical `enco-f59eec9467f13cec`. ID `enco-d62dc4f9e46ae9cf`.
- `Indicator.value` (function, line 50) — **duplicate**; canonical `enco-19a7728dfd8ba123`. ID `enco-b8af36da96e7cea8`.
- `Indicator.unit` (function, line 54) — **duplicate**; canonical `enco-141315bf788bf13b`. ID `enco-e8f8871f91d4d4e6`.
- `Indicator.identifier` (function, line 58) — **duplicate**; canonical `enco-279868a2f0e08f4e`. ID `enco-6a29ef0a8f78d07c`.
- `Indicator.label` (function, line 62) — **duplicate**; canonical `enco-bebbf72da8c316d4`. ID `enco-16ac3cdf0d941d78`.
- `Indicator.description` (function, line 66) — **duplicate**; canonical `enco-58bccdca965b8b8e`. ID `enco-505bab1061ecad40`.
- `Indicator.value` (function, line 70) — **duplicate**; canonical `enco-8e946185c8ae80d7`. ID `enco-f351dbc4c4b9e047`.
- `Indicator.unit` (function, line 74) — **duplicate**; canonical `enco-d207010c1433fa68`. ID `enco-7f3c93d57540b849`.
- `Indicator.__str__` (function, line 78) — **duplicate**; canonical `enco-b9cb3f48855f93f8`. ID `enco-05418380f3f0544d`.
- `Indicator.__repr__` (function, line 81) — **duplicate**; canonical `enco-b82645714eb7725d`. ID `enco-a7e2cde6d29be47c`.
- `Indicator.__float__` (function, line 85) — **duplicate**; canonical `enco-07b9f599d508efb9`. ID `enco-7339f7ae9598e809`.
- `Indicator.__int__` (function, line 88) — **duplicate**; canonical `enco-34d88f5b5c00a276`. ID `enco-3db90715bd9dbb26`.
- `Indicator.__bool__` (function, line 91) — **duplicate**; canonical `enco-d9587da7fdaffa34`. ID `enco-77c5a9bf569881f0`.
- `Indicator.__add__` (function, line 95) — **duplicate**; canonical `enco-990b722a6891dff0`. ID `enco-915c5cace1e32b55`.
- `Indicator.__radd__` (function, line 109) — **duplicate**; canonical `enco-a9667e396dfafb6b`. ID `enco-866af8bdf2069166`.
- `Indicator.__sub__` (function, line 112) — **duplicate**; canonical `enco-7adc064065a945dc`. ID `enco-5c03d4b066b0b168`.
- `Indicator.__rsub__` (function, line 126) — **duplicate**; canonical `enco-5c47282e2f83716c`. ID `enco-df2bbdb089a40eb9`.
- `Indicator.__mul__` (function, line 129) — **duplicate**; canonical `enco-6a500030d2f151f1`. ID `enco-42131a02cf19b7b2`.
- `Indicator.__rmul__` (function, line 143) — **duplicate**; canonical `enco-2f5ca1879ee60bc6`. ID `enco-36f3d0518e09dd56`.
- `Indicator.__truediv__` (function, line 146) — **duplicate**; canonical `enco-bb56513d7c1d57df`. ID `enco-f310caf408e00772`.
- `Indicator.__rtruediv__` (function, line 160) — **duplicate**; canonical `enco-d5e2ca7816381546`. ID `enco-05242f3640b941dc`.
- `Indicator.__eq__` (function, line 164) — **duplicate**; canonical `enco-c659c3b216a2be6b`. ID `enco-6b6ff62ab71cdfd9`.
- `Indicator.__ne__` (function, line 167) — **duplicate**; canonical `enco-b3a1db75d0f36ea5`. ID `enco-970df8d98611e146`.
- `Indicator.__lt__` (function, line 170) — **duplicate**; canonical `enco-18465a64533a40fe`. ID `enco-e7638e36b4e0f389`.
- `Indicator.__le__` (function, line 173) — **duplicate**; canonical `enco-80976df612328380`. ID `enco-abf815d5429997f6`.
- `Indicator.__gt__` (function, line 176) — **duplicate**; canonical `enco-69f535870c281ce9`. ID `enco-9d2b35801681d06e`.
- `Indicator.__ge__` (function, line 179) — **duplicate**; canonical `enco-7a4e0baa8d51b3e6`. ID `enco-ad3831a077dc17ca`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/analysis.py`**

- `Analysis` (class, line 21) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-bc7c9201af65bd07`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/__init__.py`**

- `CircuitGroup` (class, line 49) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-a0dd9bf0f224bf99`.
- `CircuitGroup.__init__` (function, line 52) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-ef51e04562bdefef`.
- `CircuitGroup.clone` (function, line 58) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-6e5c591f056eccc6`.
- `CircuitGroup.new` (function, line 62) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2f89036cc389f346`.
- `CircuitGroup.identifier` (function, line 67) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-7eca66c2472f8527`.
- `CircuitGroup.identifier` (function, line 71) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4ee80a55c8a3e9aa`.
- `CircuitGroup.label` (function, line 75) — **duplicate**; canonical `enco-8c3c0faf6da8be7b`. ID `enco-cfffbe973dbcaf18`.
- `CircuitGroup.description` (function, line 79) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-d4e3a0ffb866ee63`.
- `CircuitGroup.label` (function, line 83) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-5627556312e8063d`.
- `CircuitGroup.description` (function, line 87) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-6a0302adb2684087`.
- `CircuitGroup.timestamp` (function, line 91) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c10b8b3cb02dfe98`.
- `CircuitGroup.circuits` (function, line 95) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-067a21fd4038607e`.
- `CircuitGroup.circuits` (function, line 100) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1781aaf10b4dacea`.
- `CircuitGroup.add_circuit` (function, line 112) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-6d171dd4e2e2aa25`.
- `CircuitGroup.remove_circuit` (function, line 116) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7668cc83bd49599c`.
- `CircuitGroup.__len__` (function, line 123) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-078a694d1ffccdc0`.
- `CircuitGroup.save` (function, line 127) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-544c5b9ebc740efb`.
- `CircuitGroup.load` (function, line 146) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-42abe11b358c760d`.
- `CircuitGroup.generate_configuration_space` (function, line 166) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1add2a28bb168830`.
- `CircuitGroup.generate_configuration_space.engineering_format` (function, line 181) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-9225bbc71aec6271`.
- `CircuitGroup.generate_operating_point_space` (function, line 232) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-29f66a64514ba8d1`.
- `CircuitGroup.generate_operating_point_space.engineering_format` (function, line 245) — **duplicate**; canonical `enco-9225bbc71aec6271`. ID `enco-7e49e85a507b33fa`.
- `CircuitGroup.filter` (function, line 306) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-25d22625814aa494`.
- `CircuitGroup.get_time_dataframe` (function, line 357) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b0c0f7e54fc9522e`.
- `CircuitGroup.get_frequency_dataframe` (function, line 362) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2919b5c561a46dba`.
- `CircuitGroup.save_excel` (function, line 378) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4af2da6a4c4c6365`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/dataframe.py`**

- `_get_configuration_keys` (function, line 24) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-194f095c8aa226b1`.
- `_get_operating_point_keys` (function, line 37) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-27bc51e61ae69489`.
- `_get_time_keys` (function, line 63) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-df63da5c3b05dc6f`.
- `_get_frequency_keys` (function, line 76) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2da83235ef863ed0`.
- `get_time_dataframe` (function, line 214) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1c6b96e97d0a62f2`.
- `get_frequency_dataframe` (function, line 302) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4b36c6af1581b780`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit_group/plot.py`**

- `_palette` (function, line 41) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-77ab4c82c46b5e56`.
- `generate_group_information` (function, line 273) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-12d614e375a409b4`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/configuration.py`**

- `Configuration` (class, line 26) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4b52407523743d48`.
- `Configuration.__init__` (function, line 29) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8dc31586f272b196`.
- `Configuration.new` (function, line 33) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-abed652a91a3a2af`.
- `Configuration.identifier` (function, line 40) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-d59864b97af2f1e9`.
- `Configuration.identifier` (function, line 44) — **duplicate**; canonical `enco-4ee80a55c8a3e9aa`. ID `enco-6488fde2fc100a09`.
- `Configuration.label` (function, line 49) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e025b51efa904eef`.
- `Configuration.label` (function, line 53) — **duplicate**; canonical `enco-5627556312e8063d`. ID `enco-e18a93ae95246afa`.
- `Configuration.with_values` (function, line 57) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-280e1ac5d8b23228`.
- `Configuration.get_hash` (function, line 63) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-41036008c2eb199e`.
- `Configuration.__getitem__` (function, line 71) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7a11662a76233863`.
- `Configuration.__setitem__` (function, line 77) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1ff0269ef25ce91c`.
- `Configuration.__eq__` (function, line 80) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4cde6fc1da57c7d0`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/operating_point.py`**

- `OperatingPoint` (class, line 26) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-efb832220c8c5118`.
- `OperatingPoint.__init__` (function, line 29) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-dc2ed781f9d34050`.
- `OperatingPoint.new` (function, line 33) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-ef902291353d3963`.
- `OperatingPoint.identifier` (function, line 40) — **duplicate**; canonical `enco-39744b66cd1e3e46`. ID `enco-ef2d66dad3b63d1b`.
- `OperatingPoint.identifier` (function, line 44) — **duplicate**; canonical `enco-4ee80a55c8a3e9aa`. ID `enco-ed295dfdd0f7d127`.
- `OperatingPoint.label` (function, line 49) — **duplicate**; canonical `enco-e025b51efa904eef`. ID `enco-6ad2cddf86ecea4f`.
- `OperatingPoint.label` (function, line 53) — **duplicate**; canonical `enco-5627556312e8063d`. ID `enco-b0e4d459bf011d06`.
- `OperatingPoint.weight` (function, line 58) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-36931d2675fde256`.
- `OperatingPoint.weight` (function, line 62) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-ebdf413594f004fe`.
- `OperatingPoint.with_values` (function, line 66) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1c1276d24009da55`.
- `OperatingPoint.get_hash` (function, line 72) — **duplicate**; canonical `enco-41036008c2eb199e`. ID `enco-ff21bc9a0b73adaa`.
- `OperatingPoint.__getitem__` (function, line 80) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-06898411a4905e37`.
- `OperatingPoint.__setitem__` (function, line 88) — **duplicate**; canonical `enco-1ff0269ef25ce91c`. ID `enco-f36d58510b769a2e`.
- `OperatingPoint.__eq__` (function, line 91) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-d370b504c95e846a`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/optimizer/list.py`**

- `ListOptimizer` (class, line 25) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-5b9c153e997e3f87`.
- `ListOptimizer.__init__` (function, line 26) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f8c0d4e4278a4737`.
- `ListOptimizer.optimize` (function, line 38) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-9e1c1b9f8b030bf6`.

**enco: `enco/packages/enco-components/src/enco_components/components/__init__.py`**

- `register_component_model` (function, line 27) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-629cb955480c405b`.
- `register_component` (function, line 40) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-19b190468e07f0b3`.

**enco: `enco/packages/enco-components/src/enco_components/components/component.py`**

- `Component` (class, line 27) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1bb0fad51e55dd4e`.
- `Component.__getattr__` (function, line 31) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-9dd93e55f6df4c46`.
- `Component.__setattr__` (function, line 44) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-cd6028e45b613000`.
- `Component.__init__` (function, line 56) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-60947397a8b06631`.
- `Component.new` (function, line 65) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e04073e7894ba3b8`.
- `Component.clone` (function, line 81) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7e24f5e7f32aceb6`.
- `Component.attach_validation_criterion` (function, line 86) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b00b15fe69ed7d95`.
- `Component.validate` (function, line 97) — **stub**. ID `enco-340198dc5a47cd52`.
- `Component.get_component_classes` (function, line 103) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c05b960da8a90558`.
- `Component.get_class` (function, line 111) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-a45630bf70a4b40e`.
- `Component.get_parameter` (function, line 118) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b9fa5577ee4f9419`.
- `Component.get_data_set` (function, line 132) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-19e2642f017655bb`.
- `Component.set_parameter` (function, line 204) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c372771e56890a06`.
- `Component.save` (function, line 212) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f79c33c4cb5818e1`.
- `Component.load` (function, line 234) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-02dd1375bfb00f66`.
- `Component.set_regressor` (function, line 256) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-1b0ca1e95642c124`.
- `Component.get_regressor` (function, line 263) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-3a77e4447277a34f`.
- `Component.get_regressed_values` (function, line 273) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-ad519af300c25ffb`.
- `Component.rename_dataset` (function, line 287) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-892e5ebf7da2d982`.
- `Component.train` (function, line 307) — **stub**. ID `enco-308ce6ff677a6dbd`.

**enco: `enco/packages/enco-components/src/enco_components/components/diode/__init__.py`**

- `DiodeModel` (class, line 29) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2ff6451119a23023`.
- `Diode` (class, line 35) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-6b73e0dbf5a73608`.
- `Diode.__init__` (function, line 39) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-53e529aab3f7c5fb`.
- `Diode.validate` (function, line 49) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-ce0e5d861324f6ad`.
- `Diode.train` (function, line 55) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c874229cdc9d1e0e`.
- `Diode.get_vf` (function, line 62) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f7ce91c01c4aecfc`.

**enco: `enco/packages/enco-components/src/enco_components/components/mosfet/__init__.py`**

- `MosfetModel` (class, line 37) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-6d1451b4b981aae9`.
- `MOSFET` (class, line 41) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-670c35eeac957c62`.
- `MOSFET.attributes` (function, line 46) — **duplicate**; canonical `enco-b27a8663d0e08aa2`. ID `enco-55f7f2f666821d46`.
- `MOSFET.attributes` (function, line 50) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8379b365c6426ba0`.
- `MOSFET.__init__` (function, line 54) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-9bd2c455a5079d6f`.
- `MOSFET.validate` (function, line 58) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-c5510b8080d96ae5`.
- `MOSFET.train` (function, line 64) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b1ff841464ea2d09`.
- `MOSFET.get_ron` (function, line 105) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4ab7c8720e70ec78`.
- `MOSFET.get_vf` (function, line 117) — **broken-adapter**; candidate `core/models.py:384 Transistor`. ID `enco-0ea4305913236e49`.
- `MOSFET.get_channel_diode_currents` (function, line 145) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e11ddebd7e4b83f8`.
- `MOSFET._get_vf` (function, line 438) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4656fc677f86737f`.
- `MOSFET._get_rdson` (function, line 463) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-74722bec0e53b9b6`.
- `MOSFET.get_diode_voltage` (function, line 478) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4afd19715f93c7da`.
- `MOSFET.get_diode_loss` (function, line 495) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f69eafac5246d656`.
- `MOSFET.get_total_diode_loss` (function, line 508) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b868774f2b1d6557`.
- `MOSFET.get_rdson` (function, line 693) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2550243d8f236648`.
- `MOSFET.get_diode_current` (function, line 764) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-396842419c21b91e`.
- `MOSFET.get_mosfet_current` (function, line 769) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-d4607f37d0a79c85`.

**enco: `enco/packages/enco-components/src/enco_components/configuration.py`**

- `update_configuration` (function, line 28) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c4d6b746d79391b2`.

**enco: `enco/packages/enco-components/tests/mosfet/test_boost.py`**

- `mosfet` (function, line 31) — **reference-only**. ID `enco-e09c054e20ff7607`.
- `test_boost` (function, line 115) — **reference-only**. ID `enco-ec78077425d59374`.

**enco: `enco/packages/enco-core/src/enco_core/structures/indicator.py`**

- `Indicator` (class, line 16) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-28da68501b8c1a91`.
- `Indicator.__init__` (function, line 29) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-42e256fe66175d55`.
- `Indicator.new` (function, line 33) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c2ef27d1eea2052f`.
- `Indicator.identifier` (function, line 38) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4dcc3306cc37108f`.
- `Indicator.label` (function, line 42) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8be544a37b27c11e`.
- `Indicator.description` (function, line 46) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f59eec9467f13cec`.
- `Indicator.value` (function, line 50) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-19a7728dfd8ba123`.
- `Indicator.unit` (function, line 54) — **duplicate**; canonical `enco-141315bf788bf13b`. ID `enco-35429e921fbb9217`.
- `Indicator.identifier` (function, line 58) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-279868a2f0e08f4e`.
- `Indicator.label` (function, line 62) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-bebbf72da8c316d4`.
- `Indicator.description` (function, line 66) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-58bccdca965b8b8e`.
- `Indicator.value` (function, line 70) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8e946185c8ae80d7`.
- `Indicator.unit` (function, line 74) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-d207010c1433fa68`.
- `Indicator.__str__` (function, line 78) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b9cb3f48855f93f8`.
- `Indicator.__repr__` (function, line 81) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b82645714eb7725d`.
- `Indicator.__float__` (function, line 85) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-07b9f599d508efb9`.
- `Indicator.__int__` (function, line 88) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-34d88f5b5c00a276`.
- `Indicator.__bool__` (function, line 91) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-d9587da7fdaffa34`.
- `Indicator.__add__` (function, line 95) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-990b722a6891dff0`.
- `Indicator.__radd__` (function, line 109) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-a9667e396dfafb6b`.
- `Indicator.__sub__` (function, line 112) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7adc064065a945dc`.
- `Indicator.__rsub__` (function, line 126) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-5c47282e2f83716c`.
- `Indicator.__mul__` (function, line 129) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-6a500030d2f151f1`.
- `Indicator.__rmul__` (function, line 143) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2f5ca1879ee60bc6`.
- `Indicator.__truediv__` (function, line 146) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-bb56513d7c1d57df`.
- `Indicator.__rtruediv__` (function, line 160) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-d5e2ca7816381546`.
- `Indicator.__eq__` (function, line 164) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-c659c3b216a2be6b`.
- `Indicator.__ne__` (function, line 167) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b3a1db75d0f36ea5`.
- `Indicator.__lt__` (function, line 170) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-18465a64533a40fe`.
- `Indicator.__le__` (function, line 173) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-80976df612328380`.
- `Indicator.__gt__` (function, line 176) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-69f535870c281ce9`.
- `Indicator.__ge__` (function, line 179) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-7a4e0baa8d51b3e6`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/circuit.py`**

- `OperatingPointModel` (class, line 30) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2e2edc66de87e1a7`.
- `OperatingPointModel.__init__` (function, line 35) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-ac8341be148d8de4`.
- `ConfigurationModel` (class, line 43) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-11502ace961f5e42`.
- `ConfigurationModel.__init__` (function, line 47) — **duplicate**; canonical `enco-ac8341be148d8de4`. ID `enco-a4cc523c89319fed`.
- `CircuitModel` (class, line 55) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-65e976aca00d2678`.
- `CircuitModel.__init__` (function, line 66) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-2abe4a6de36f8d36`.
- `CircuitModel.dump` (function, line 110) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-280084b6849ef848`.
- `CircuitModel.load` (function, line 133) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-49317b3040a36831`.

**enco: `enco/packages/enco-models/src/enco_models/circuit/circuit_group.py`**

- `CircuitGroupModel` (class, line 32) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-721916c5329911d8`.
- `CircuitGroupModel.__init__` (function, line 43) — **duplicate**; canonical `enco-ac8341be148d8de4`. ID `enco-f318a33d40cd7d64`.
- `CircuitGroupModel.dump` (function, line 50) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-46bf54a74b8bc508`.
- `CircuitGroupModel.load` (function, line 78) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f5a8f2bc1d1366bb`.

**enco: `enco/packages/enco-models/src/enco_models/component.py`**

- `DatasheetParameterModel` (class, line 28) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-22d4839e92beeb1a`.
- `BaseComponentModel` (class, line 35) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-b0a39b30d83b496d`.
- `BaseComponentModel.__init__` (function, line 54) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-8c1167b21ae5fd23`.
- `BaseComponentModel.dump` (function, line 61) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-bc73fb51d645683f`.
- `BaseComponentModel.load` (function, line 72) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-fe6ad14946e65a4b`.

**enco: `enco/packages/enco-models/src/enco_models/indicator.py`**

- `IndicatorModel` (class, line 19) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-07d64fb525434600`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/results.py`**

- `SimulationResults.populate_circuit` (function, line 57) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-977bb923562f8c5d`.

**enco: `library-design/component/body_diode/main.py`**

- `get_mosfet` (function, line 27) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-ac871508a83dc4fc`.

**enco: `library-design/dcdc/cllc/src/project/optimizer.py`**

- `validate_indicators` (function, line 114) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e99c1bf3a98e04e0`.
- `normalize_indicators` (function, line 279) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-5089c53b90a43fda`.

**enco: `library-design/dcdc/cllc_3p/src/project/components/cllc_trasnformer.py`**

- `DAB_Transformer_Model` (class, line 23) — **duplicate**; canonical `enco-a964116cc7748783`. ID `enco-4ede4d4d1e30d860`.
- `CLLC_Transformer` (class, line 44) — **duplicate**; canonical `enco-bd256d1066a2ec3e`. ID `enco-3427aec880c6b9b5`.
- `CLLC_Transformer.validate` (function, line 48) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-c23fd3c5afcb3907`.
- `CLLC_Transformer.train` (function, line 52) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-b907c49cf5d1becb`.
- `CLLC_Transformer.winding_loss` (function, line 56) — **duplicate**; canonical `enco-eebbc89e7848e95f`. ID `enco-7c499215704c2a54`.
- `CLLC_Transformer.core_loss` (function, line 88) — **duplicate**; canonical `enco-16c63550a6dfa11c`. ID `enco-a678df11a8b85fb1`.
- `CLLC_Transformer.get_loss_index` (function, line 132) — **duplicate**; canonical `enco-c671c19460e669da`. ID `enco-961e4acade1e413d`.

**enco: `library-design/dcdc/cllc_3p/src/project/utility/simulation.py`**

- `update_circuit` (function, line 35) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-705da29e555c2cef`.

**enco: `library-design/dcdc/d3ab/src/project/utility/simulation.py`**

- `update_circuit` (function, line 37) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-f71ece07cdfd3d02`.

**enco: `library-design/dcdc/dab_cf/src/project/optimizer.py`**

- `validate_indicators` (function, line 111) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-4f92d66f47850163`.
- `normalize_indicators` (function, line 283) — **duplicate**; canonical `enco-5089c53b90a43fda`. ID `enco-336838bdca024713`.

**enco: `library-design/dcdc/staedler_llc/src/project/simulation.py`**

- `update_circuit` (function, line 41) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `enco-e1cc75b635449071`.

**sc: `mfiles/SemiconductorEditor.m`**

- `SemiconductorEditor` (function, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-d1d21aad6c4afe85`.
- `newScData` (function, line 687) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-e072fbc569319779`.
- `updateScData` (function, line 716) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-54c64f494cc6ea4e`.
- `openScData` (function, line 723) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-ff9488323bcd26ac`.
- `newSwitchTrace` (function, line 762) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-bec3a3cfce420006`.
- `combineTraces` (function, line 783) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-4d8f91a52aae3b86`.

**sc: `mfiles/calcRg.m`**

- `calcRg` (function, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-92afe9737e47b855`.

**sc: `mfiles/transistor/display.m`**

- `display` (function, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-d597b1cbd3e4fe95`.

**sc: `mfiles/transistor/transistor.m`**

- `transistor` (function, line 23) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-7b099074440519c2`.
- `get.name` (function, line 29) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-8f904c818aa8ddd0`.
- `get.type` (function, line 38) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-3541e24fec1363a2`.
- `get.switching_data` (function, line 49) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-64587de026333b08`.
- `get.conducting_data` (function, line 56) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-6194b454ea34a4b1`.
- `get.break_down_voltage` (function, line 70) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-258f78970d1296e6`.
- `set.name` (function, line 93) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-d02190752c6eb58e`.
- `set.type` (function, line 100) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-10d8ce71d6122377`.
- `set.switching_data` (function, line 107) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-1a43cb79a0253033`.
- `set.conducting_data` (function, line 136) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-50d4f673266e8afd`.
- `set.break_down_voltage` (function, line 174) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-6c5db867d203d28c`.

**sc: `semiconductors/diode.py`**

- `DiodeElectrical` (class, line 1) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-08f68223700d65ba`.
- `DiodeElectrical.__init__` (function, line 2) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-0157fafa4dcf3d34`.

**sc: `semiconductors/helpers.py`**

- `find_nearest` (function, line 10) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-81fc707fd885ac42`.
- `run_double_pulse_sim` (function, line 15) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-5539173cea0a3f22`.
- `SwitchingPoint` (class, line 191) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-2b6b94fbdd2399f2`.
- `SwitchingPoint.__init__` (function, line 192) — **duplicate**; canonical `sc-4da5d2718f68d8e6`. ID `sc-ae63c9fc75595950`.
- `SwitchElectrical` (class, line 198) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-5c48eefae7d1f8a6`.
- `SwitchElectrical.__init__` (function, line 199) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-a42ca40265a38de0`.

**sc: `semiconductors/mosfet.py`**

- `Mosfet` (class, line 11) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-8eec5975ddce4186`.
- `Mosfet.__init__` (function, line 12) — **duplicate**; canonical `sc-16503f5eefbc5442`. ID `sc-85f47fafc1826ccf`.
- `Mosfet.calc_on_voltage` (function, line 40) — **duplicate**; canonical `sc-f175b4d0115c1d8f`. ID `sc-1f801cb2196651f6`.

**sc: `semiconductors/scdata.py`**

- `ScDataGeneric` (class, line 7) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-765fe8f266050441`.
- `ScDataGeneric.__init__` (function, line 8) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-4db7a50103f8fb0d`.
- `ScDataGeneric.__repr__` (function, line 29) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-01d403db133bdf06`.
- `ScDataGeneric.calc_switching_power` (function, line 43) — **stub**. ID `sc-fca303c6cbeba540`.
- `ScData01` (class, line 110) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-9ea3c17956d39088`.
- `ScData01.__init__` (function, line 111) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-6bbd57fee9c93f91`.
- `ScData02` (class, line 132) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-1f90493c400fdc91`.
- `ScData02.__init__` (function, line 133) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-41b834faa0021681`.
- `ScData03` (class, line 154) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-cc53d8794578e8a2`.
- `ScData03.__init__` (function, line 155) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-450aa5f356b0615c`.

**sc: `semiconductors/switch.py`**

- `SwitchElectrical` (class, line 1) — **duplicate**; canonical `sc-5c48eefae7d1f8a6`. ID `sc-5f9a9e660291421e`.
- `SwitchElectrical.__init__` (function, line 2) — **duplicate**; canonical `sc-a42ca40265a38de0`. ID `sc-2416cb7ae1af8273`.

**sc: `semiconductors/transistor.py`**

- `Transistor` (class, line 59) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-90e80bc09b3ad12e`.
- `Transistor.__init__` (function, line 60) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-e148fb812e6fffa7`.
- `Transistor.name` (function, line 70) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-891d4d3fc49dea76`.
- `Transistor.name` (function, line 76) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-dd88de4c88c46343`.
- `Transistor.type` (function, line 82) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-a571949f6bdd651d`.
- `Transistor.type` (function, line 88) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-f63d74a380422dee`.
- `Transistor.switching_data` (function, line 94) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-a72d2c32bdff00fa`.
- `Transistor.conducting_data` (function, line 100) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-89fc7bd55ac0e116`.
- `Transistor.break_down_voltage` (function, line 112) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-a4cee2b78297b051`.

**sc: `tests/test_basic.py`**

- `test_ScDataGeneric` (function, line 39) — **reference-only**. ID `sc-dbd8674cca6ed4d4`.
- `test_ScData01` (function, line 42) — **reference-only**. ID `sc-6685e062e81eb025`.
- `test_ScData02` (function, line 48) — **stub**. ID `sc-6d6f1036e6881352`.
- `test_ScData03` (function, line 54) — **stub**. ID `sc-af018450cf1f67f7`.

**sc: `tests/test_igbt.py`**

- `test_lobsiger_approximations` (function, line 5) — **reference-only**. ID `sc-e36adc8a53e760f3`.
- `test_switch_event_GE` (function, line 18) — **reference-only**. ID `sc-098592d2fb3dcda9`.

**sc: `utils/semiconductor.m`**

- `mosfet` (function, line 86) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-bdbf8079d1e9c8e7`.
- `loadPropertiesFromFilev01` (function, line 113) — **candidate-match-needs-validation**; candidate `core/models.py:384 Transistor`. ID `sc-1a078b101c005ca7`.
- `generateEonTable` (function, line 186) — **stub**. ID `sc-ede84a4a54b458a9`.
- `generateEoffTable` (function, line 187) — **stub**. ID `sc-0bfd1e204ed82736`.
- `generateVonTable` (function, line 188) — **stub**. ID `sc-3afe09d1e0c152fa`.
- `generateRonTable` (function, line 189) — **stub**. ID `sc-4611e2e5762df57f`.


### Non-semiconductor orchestration/context

Execution: context only; no automatic port.

**enco: `enco-analysis/src/enco_analysis/configuration.py`**

- `update_configuration` (function, line 28) — **duplicate**; canonical `enco-f21ef71cb7fad4a3`. ID `enco-aefdc4291d54a0b8`.

**enco: `enco-analysis/tests/test_version.py`**

- `TestVersions` (class, line 21) — **duplicate**; canonical `enco-d19ee100004ffd4a`. ID `enco-a58a032f67efc038`.
- `TestVersions.test_get_versions` (function, line 22) — **duplicate**; canonical `enco-8c246d9b0d630211`. ID `enco-b98ca0e756b1c3fa`.

**enco: `enco-core/src/enco_core/configuration/__init__.py`**

- `update_configuration` (function, line 27) — **duplicate**; canonical `enco-6a7ff9fd7bbac6a1`. ID `enco-685caf3a5a2d70dd`.

**enco: `enco-core/src/enco_core/configuration/configuration.py`**

- `totoml` (function, line 21) — **duplicate**; canonical `enco-e158cd324b0536ff`. ID `enco-11f9b130031b004d`.
- `ConfigurationSnippet` (class, line 33) — **duplicate**; canonical `enco-997a95d671314bc7`. ID `enco-b8cac5ab555e1901`.
- `ConfigurationSnippet.__init__` (function, line 38) — **duplicate**; canonical `enco-8c741c629895a028`. ID `enco-bf64cfc6f5723232`.
- `ConfigurationSnippet.render` (function, line 42) — **duplicate**; canonical `enco-00f75d5e9c867497`. ID `enco-4e3bab4df5ba7e23`.
- `ConfigurationSnippet.__eq__` (function, line 55) — **duplicate**; canonical `enco-d42470f168b8c5b9`. ID `enco-9a389f1a7171dc64`.
- `Configuration` (class, line 62) — **duplicate**; canonical `enco-e34098139f47035d`. ID `enco-6fd7b5deec0567a9`.
- `Configuration.__init__` (function, line 80) — **duplicate**; canonical `enco-a1e1dd6a51b38af0`. ID `enco-542d2286a1f8e03f`.
- `Configuration.save` (function, line 85) — **duplicate**; canonical `enco-832baa8620892a6b`. ID `enco-088f0dbf4e4a76cb`.
- `Configuration.load` (function, line 138) — **duplicate**; canonical `enco-0ea99be878cb7688`. ID `enco-9e5fa079857e45f7`.
- `Configuration.exists` (function, line 146) — **duplicate**; canonical `enco-1b693a7880ff6892`. ID `enco-abc7fbed49f56730`.
- `Configuration.get` (function, line 151) — **duplicate**; canonical `enco-5a1151a4925c1381`. ID `enco-0ef2fdf661c66096`.
- `Configuration.set` (function, line 161) — **duplicate**; canonical `enco-42c74edc8d3be065`. ID `enco-3a4a5c27fd9fbbf2`.
- `Configuration.__getitem__` (function, line 171) — **duplicate**; canonical `enco-8b32266f803613d1`. ID `enco-d16a050f81e2fde6`.
- `Configuration.__setitem__` (function, line 174) — **duplicate**; canonical `enco-5efee7999814d66f`. ID `enco-1d122f727da0df08`.
- `Configuration.__delitem__` (function, line 177) — **duplicate**; canonical `enco-7355a23c8443e5df`. ID `enco-f24d47735bf23a40`.
- `Configuration.__contains__` (function, line 180) — **duplicate**; canonical `enco-0b844059d5e44e6a`. ID `enco-8502dbc0c5acde33`.
- `Configuration.__str__` (function, line 183) — **duplicate**; canonical `enco-c97fe77668d598c7`. ID `enco-a7889672f4bbb560`.
- `Configuration.__repr__` (function, line 186) — **duplicate**; canonical `enco-0a91279d81e9e1bb`. ID `enco-dcb09a1a2646f180`.
- `Configuration.__iter__` (function, line 189) — **duplicate**; canonical `enco-30a0ddb23983cd20`. ID `enco-fccbed2617a5fab4`.
- `Configuration.__len__` (function, line 192) — **duplicate**; canonical `enco-043e83a0da1df30a`. ID `enco-8e087ae4370d5e89`.

**enco: `enco-core/src/enco_core/e_series.py`**

- `ESeries` (class, line 400) — **duplicate**; canonical `enco-06993c04b039e8e3`. ID `enco-86b3be4000477091`.
- `get_closest_series_value` (function, line 410) — **duplicate**; canonical `enco-7f20218e4400303f`. ID `enco-e0954961a27e2b87`.
- `get_series_values` (function, line 420) — **duplicate**; canonical `enco-daaf18b627abd047`. ID `enco-4405ad96332c0a5e`.

**enco: `enco-core/src/enco_core/library/library.py`**

- `filter_json` (function, line 24) — **duplicate**; canonical `enco-13ce3e3d00f6c6c7`. ID `enco-eacab7463f979538`.
- `quote` (function, line 36) — **duplicate**; canonical `enco-82d47858af1b39d4`. ID `enco-eb876c5c5034b205`.
- `Library` (class, line 44) — **duplicate**; canonical `enco-6704e160cc755dad`. ID `enco-2659f663e62c570e`.
- `Library.__init__` (function, line 50) — **duplicate**; canonical `enco-a1e355de823ffac2`. ID `enco-c7136700d721eadc`.
- `Library.delete` (function, line 76) — **duplicate**; canonical `enco-22f67c2a9db76831`. ID `enco-48f108c7087572f5`.
- `Library.update` (function, line 88) — **duplicate**; canonical `enco-38f75dc1bd2f539c`. ID `enco-320bfea683e6b2f1`.
- `Library.get` (function, line 107) — **duplicate**; canonical `enco-da09eae32b060785`. ID `enco-9ae55fdcfe13c8a4`.
- `Library.search` (function, line 119) — **duplicate**; canonical `enco-6e8b42b8049fae8b`. ID `enco-49dc229740a55485`.
- `Library._load_templates` (function, line 169) — **duplicate**; canonical `enco-a9284f5904d4acb4`. ID `enco-354f61b2ff12ac4d`.

**enco: `enco-core/src/enco_core/library/template.py`**

- `Template` (class, line 25) — **duplicate**; canonical `enco-29d365b611d4ee2e`. ID `enco-0b577245818c7e02`.
- `Template.__init__` (function, line 26) — **duplicate**; canonical `enco-0bb32722a0c9043d`. ID `enco-22398c6e01d6ef5d`.
- `Template._load_identifier` (function, line 68) — **duplicate**; canonical `enco-053cd41ed86aa2bc`. ID `enco-ccc1bcad20da5124`.
- `Template._validate_manifest` (function, line 90) — **duplicate**; canonical `enco-e0272bf432bb17cb`. ID `enco-965de01d0c309f64`.
- `Template._run_prompt_script` (function, line 117) — **duplicate**; canonical `enco-c83a6f501ea64347`. ID `enco-af1183bdd6f34232`.
- `Template.render` (function, line 170) — **duplicate**; canonical `enco-1728c65c0b0b3248`. ID `enco-fc872a1346a4ed5f`.
- `Template.__repr__` (function, line 246) — **duplicate**; canonical `enco-7a41154397455742`. ID `enco-984a59930fc113dd`.
- `Template.__str__` (function, line 249) — **duplicate**; canonical `enco-1178cadd920c62b0`. ID `enco-1111f3e71e7faf96`.

**enco: `enco-core/src/enco_core/path.py`**

- `get_data_path` (function, line 20) — **duplicate**; canonical `enco-f9946dc86ca2bc95`. ID `enco-feafd0e0f775ffbd`.

**enco: `enco-core/src/enco_core/repository/git.py`**

- `GitRepository` (class, line 20) — **reference-only**. ID `enco-26e10f4aadac6972`.
- `GitRepository.__init__` (function, line 21) — **duplicate**; canonical `enco-c843548e716196fb`. ID `enco-aad325aa3e05daea`.
- `GitRepository.delete` (function, line 35) — **duplicate**; canonical `enco-8aca20a8a182d386`. ID `enco-e0e3934bf7c5a52b`.
- `GitRepository.delete.remove_readonly` (function, line 41) — **duplicate**; canonical `enco-f3fdc559a66ac806`. ID `enco-8482d19599ca93e7`.
- `GitRepository.update` (function, line 54) — **reference-only**. ID `enco-b705725c97eb9bd4`.
- `GitRepository.information` (function, line 129) — **duplicate**; canonical `enco-34e8b629b8ac5ab8`. ID `enco-b992b74149a11069`.
- `GitRepository.commit` (function, line 225) — **duplicate**; canonical `enco-7386edb491fd4b2c`. ID `enco-79b9c0f2cc1fbc3b`.
- `GitRepository.push` (function, line 234) — **duplicate**; canonical `enco-8fcaa1284cd12829`. ID `enco-a1c159a680c64854`.

**enco: `enco-core/src/enco_core/repository/svn.py`**

- `SubversionRepository` (class, line 16) — **duplicate**; canonical `enco-847540b937f27ca8`. ID `enco-a09fcc9415351ac5`.
- `SubversionRepository.__init__` (function, line 17) — **duplicate**; canonical `enco-040dcdd1ec3f30a7`. ID `enco-33ab9eea452c396a`.
- `SubversionRepository.information` (function, line 28) — **duplicate**; canonical `enco-fd4512878db2ddfc`. ID `enco-b31ca779b7ac3ffd`.
- `SubversionRepository.update` (function, line 92) — **duplicate**; canonical `enco-8f3db7f9740c6e37`. ID `enco-ded704286551e7f6`.

**enco: `enco-core/src/enco_core/structures/metadata.py`**

- `MetadataBase` (class, line 22) — **duplicate**; canonical `enco-d45b13c561187177`. ID `enco-ad1c05253a8c56c7`.
- `MetadataBase.__init__` (function, line 34) — **duplicate**; canonical `enco-a4d161f8954546f2`. ID `enco-d781553c1b8d7e91`.
- `MetadataBase.get_uuid` (function, line 43) — **duplicate**; canonical `enco-2b2288cc93e43209`. ID `enco-8711aa5a15ecbb82`.
- `MetadataBase.get_timestamp` (function, line 46) — **duplicate**; canonical `enco-a71ddd2a4e2b05d8`. ID `enco-5cd4cf3ec42babfe`.
- `MetadataBase.get_identifier` (function, line 49) — **duplicate**; canonical `enco-35350ab0bd8d3862`. ID `enco-a097da34b18211c9`.
- `MetadataBase.get_label` (function, line 52) — **duplicate**; canonical `enco-65ad8bb63503d8e2`. ID `enco-6bd9a6ad663e72f0`.
- `MetadataBase.get_description` (function, line 55) — **duplicate**; canonical `enco-8b34f480d4c65cce`. ID `enco-7df04f66488c6826`.
- `MetadataBase.get_tags` (function, line 58) — **duplicate**; canonical `enco-7f68b863b191b65c`. ID `enco-167d37e78013311b`.
- `MetadataBase.get_attributes` (function, line 61) — **duplicate**; canonical `enco-ecf9d4be844a2267`. ID `enco-3206efbbe5ca8208`.
- `MetadataBase.set_label` (function, line 64) — **duplicate**; canonical `enco-43d6a924d85c320e`. ID `enco-bf348147547b98c7`.
- `MetadataBase.set_description` (function, line 67) — **duplicate**; canonical `enco-a60f94c9cbd20b00`. ID `enco-d6d52c43df895584`.
- `MetadataBase.set_tags` (function, line 70) — **duplicate**; canonical `enco-390b9baa6ade39dc`. ID `enco-b6037ca4c4f30e59`.
- `MetadataBase.set_attributes` (function, line 73) — **duplicate**; canonical `enco-cf6ad8304f47b6ae`. ID `enco-e6c909fc6cc9cf03`.
- `MetadataBase.add_tag` (function, line 76) — **duplicate**; canonical `enco-40489a30045c3858`. ID `enco-3d6372044ede9f8f`.
- `MetadataBase.add_attribute` (function, line 79) — **duplicate**; canonical `enco-e9c41d86f0267a81`. ID `enco-dcfcf5697aa3fede`.
- `MetadataBase.remove_tag` (function, line 82) — **duplicate**; canonical `enco-28f0e0be6c0cd394`. ID `enco-b0ef5b333eeafed6`.
- `MetadataBase.remove_attribute` (function, line 85) — **duplicate**; canonical `enco-77ad015fb5d10d23`. ID `enco-95a5f562690fa46d`.

**enco: `enco-core/src/enco_core/utility/__init__.py`**

- `SerializableObject` (class, line 24) — **duplicate**; canonical `enco-51d54dc4e47f77d6`. ID `enco-d330f0f301348800`.
- `SerializableObject.to_dict` (function, line 30) — **stub**; canonical `enco-f2e7d0d0fbd49ed1`. ID `enco-7478005ffc394401`.
- `SerializableObject.from_dict` (function, line 34) — **stub**; canonical `enco-d262dae53eb9e219`. ID `enco-2aaec69562c64d53`.
- `SerializableObject.to_json` (function, line 38) — **stub**; canonical `enco-97dad1a079a053c2`. ID `enco-1e558bd31f531b89`.
- `SerializableObject.from_json` (function, line 42) — **stub**; canonical `enco-6be539611c4f981d`. ID `enco-e6b7b3e211ef0aa4`.
- `NamedObject` (class, line 46) — **duplicate**; canonical `enco-baa132c8a320dda6`. ID `enco-9bba7062e7d826b9`.
- `NamedObject.__init__` (function, line 51) — **duplicate**; canonical `enco-e8bc10dd2deded35`. ID `enco-efb1c1d146ca2c03`.
- `NamedObject.__eq__` (function, line 57) — **duplicate**; canonical `enco-b9a2df0895d85fb2`. ID `enco-f90b7bd0f550ae68`.
- `NamedObject.__hash__` (function, line 63) — **duplicate**; canonical `enco-0eaf68814bf7ee44`. ID `enco-0eeec57298909fc0`.
- `NamedObject.__str__` (function, line 68) — **stub**; canonical `enco-cd1bf077afc8bcb8`. ID `enco-7cf109673c435478`.
- `NamedObject.__repr__` (function, line 73) — **stub**; canonical `enco-856031dc0a16fb97`. ID `enco-614a232576ab95b3`.
- `NamedObject.to_dict` (function, line 77) — **duplicate**; canonical `enco-cd95442a7d0da899`. ID `enco-d11a35cf3080df37`.
- `NamedObject.from_dict` (function, line 83) — **duplicate**; canonical `enco-d8caadf70700aa2f`. ID `enco-353076e0a248cc64`.
- `NamedObject.to_json` (function, line 87) — **duplicate**; canonical `enco-b8d3a58d145623be`. ID `enco-37a90743dda55a4f`.
- `NamedObject.from_json` (function, line 90) — **duplicate**; canonical `enco-e7c45ba0bc726dfd`. ID `enco-e66ad1c420abda12`.

**enco: `enco-core/src/enco_core/utility/context.py`**

- `Context` (class, line 21) — **duplicate**; canonical `enco-f0bbfb262b4c4265`. ID `enco-30f69bd86321b4de`.
- `Context.__init__` (function, line 22) — **duplicate**; canonical `enco-7e77a4bacb10092a`. ID `enco-921ee289087a6b6b`.
- `Context.load` (function, line 43) — **duplicate**; canonical `enco-613d342a79828679`. ID `enco-d7d101efa8ee9bd9`.
- `Context.extend` (function, line 61) — **duplicate**; canonical `enco-d015868bc14ff5b5`. ID `enco-3050c9b1fadbde61`.
- `Context.to_file` (function, line 72) — **stub**; canonical `enco-7aafe152f1feca5b`. ID `enco-d55860b481df8ff6`.
- `Context._get_context_git` (function, line 89) — **duplicate**; canonical `enco-215bdaba2d2c75f6`. ID `enco-d18ca5c31d2b9cb7`.
- `Context._get_context_git.return_and_print` (function, line 98) — **duplicate**; canonical `enco-0ea6dc1a2fc375a5`. ID `enco-b39195b1d75880a3`.
- `Context._get_context_svn` (function, line 144) — **duplicate**; canonical `enco-817b2df6eaa53a86`. ID `enco-8e3f3e1e05f455b1`.
- `Context._get_context_svn.return_and_print` (function, line 153) — **duplicate**; canonical `enco-809852c77b12cb4f`. ID `enco-7be65d28f7f746db`.
- `Context._get_context_system` (function, line 168) — **duplicate**; canonical `enco-b381890cd4605d33`. ID `enco-6861f4c18edefd29`.
- `Context._get_context_system.return_and_print` (function, line 176) — **duplicate**; canonical `enco-ca00ec0f379b2966`. ID `enco-46bfde0a07b98b5c`.

**enco: `enco-core/src/enco_core/utility/dataset.py`**

- `Dataset` (class, line 8) — **duplicate**; canonical `enco-4ad8eb7cec3c64c6`. ID `enco-83134fb316fd1d9e`.
- `Dataset.__init__` (function, line 9) — **duplicate**; canonical `enco-8f120746cadf20eb`. ID `enco-f5aec6c8323c625e`.
- `Dataset.load` (function, line 22) — **duplicate**; canonical `enco-407c29aab0c94e50`. ID `enco-55de2d21dcc948b3`.
- `Dataset.to_dataframe` (function, line 33) — **duplicate**; canonical `enco-3ca0f40326085a5a`. ID `enco-43366e6bf6546081`.

**enco: `enco-core/src/enco_core/utility/metadata.py`**

- `PackageMetadata` (class, line 14) — **duplicate**; canonical `enco-c09a72cb39cd793d`. ID `enco-177316cb1f72866f`.
- `PackageMetadata.__init__` (function, line 15) — **duplicate**; canonical `enco-e76c61062382b7b8`. ID `enco-ffae9a46758a1fef`.
- `PackageMetadata.version` (function, line 18) — **duplicate**; canonical `enco-3fda3b7dcc2cf2d3`. ID `enco-5429fe7600d7210c`.
- `PackageMetadata.authors` (function, line 33) — **duplicate**; canonical `enco-7f5f14f84d55ec84`. ID `enco-5f52c9a063ad49c7`.

**enco: `enco-core/src/enco_core/utility/path.py`**

- `get_data_folder` (function, line 14) — **duplicate**; canonical `enco-25c00624ed3cec59`. ID `enco-1407e8e46a7c1f46`.

**enco: `enco-core/src/enco_core/utility/version.py`**

- `get_versions` (function, line 20) — **duplicate**; canonical `enco-ad84800278540340`. ID `enco-31f43fc0efecbcce`.
- `VersionInformation` (class, line 68) — **duplicate**; canonical `enco-afd9ec7a0dec2fff`. ID `enco-81fe51e44a9f4bfa`.
- `VersionInformation.__init__` (function, line 69) — **duplicate**; canonical `enco-177857b73fb88ae4`. ID `enco-79d9dff519fd27ed`.
- `VersionInformation.__str__` (function, line 81) — **duplicate**; canonical `enco-ecafd85d521503b3`. ID `enco-0eb16db3879f6e39`.
- `VersionInformation.__repr__` (function, line 84) — **duplicate**; canonical `enco-d80766ec12ef5802`. ID `enco-894d19e2211e03ad`.
- `get_version_information` (function, line 88) — **duplicate**; canonical `enco-37a964beb7de6cef`. ID `enco-3dfabaf7ab72e367`.

**enco: `enco-core/tests/test_version.py`**

- `TestVersions` (class, line 21) — **duplicate**; canonical `enco-d19ee100004ffd4a`. ID `enco-194dfe4e6d42072d`.
- `TestVersions.test_get_versions` (function, line 22) — **duplicate**; canonical `enco-8c246d9b0d630211`. ID `enco-96e7946fc49b8875`.

**enco: `enco-library/src/enco_library/main.py`**

- `main` (function, line 1) — **reference-only**. ID `enco-511771623816a083`.

**enco: `enco/_release/release.py`**

- `CommitInformation` (class, line 23) — **reference-only**. ID `enco-c4b6b2ecb8e03d2f`.
- `get_commits_since_last_tag` (function, line 30) — **reference-only**. ID `enco-580604a955489946`.
- `get_changed_files` (function, line 62) — **reference-only**. ID `enco-41a179bd02dbd3bc`.
- `get_last_version` (function, line 89) — **reference-only**. ID `enco-f7ed49ea8e6c0d39`.
- `get_current_version` (function, line 100) — **reference-only**. ID `enco-b9ee06eb1d274cbe`.
- `generate_release_report` (function, line 122) — **reference-only**. ID `enco-23464f6162736da4`.

**enco: `enco/_scripts/update_gitignore.py`**

- `update_gitignore` (function, line 5) — **reference-only**. ID `enco-a8041b0636ca77f7`.

**enco: `enco/_scripts/version.py`**

- `run_command` (function, line 26) — **reference-only**. ID `enco-dfa10e7ff8a372b0`.
- `has_changes_between_branches` (function, line 39) — **reference-only**. ID `enco-ab33597797505228`.
- `parse_arguments` (function, line 51) — **reference-only**. ID `enco-2976d0b01edb7878`.
- `process_repo` (function, line 60) — **reference-only**. ID `enco-186ff6efaa7f6d6d`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit/capacitor.py`**

- `attach_capacitor_indicators` (function, line 21) — **reference-only**. ID `enco-dd32656b6c4a27e8`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/circuit/circuit/inductor.py`**

- `attach_inductor_indicators` (function, line 20) — **reference-only**. ID `enco-b877ee6ad59aaa5b`.

**enco: `enco/packages/enco-analysis/src/enco_analysis/configuration.py`**

- `update_configuration` (function, line 28) — **reference-only**. ID `enco-f21ef71cb7fad4a3`.

**enco: `enco/packages/enco-cli/src/enco_cli/__init__.py`**

- `get_test_library` (function, line 38) — **reference-only**. ID `enco-1878b0529b63bf06`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/__init__.py`**

- `version` (function, line 77) — **reference-only**. ID `enco-be93c759a28d5ab3`.
- `entry` (function, line 86) — **reference-only**. ID `enco-af74310c95bc1333`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/callback.py`**

- `callback` (function, line 22) — **reference-only**. ID `enco-ff4ffa868eccd971`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/config.py`**

- `print_dictionary` (function, line 20) — **reference-only**. ID `enco-c50d21ba0257d526`.
- `list` (function, line 37) — **reference-only**. ID `enco-051f1c2d93d88ff6`.
- `get` (function, line 50) — **reference-only**. ID `enco-2428705b222bcbfd`.
- `set` (function, line 66) — **reference-only**. ID `enco-34bc4afc2cc7e190`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/design.py`**

- `get_design_library` (function, line 23) — **reference-only**. ID `enco-6b30d3bb1a98d3b1`.
- `clone` (function, line 77) — **reference-only**. ID `enco-978b89f7270996a9`.
- `list` (function, line 129) — **reference-only**. ID `enco-799e8b29419024c8`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/docs.py`**

- `clone` (function, line 24) — **reference-only**. ID `enco-871f7b7b749fdac4`.
- `list` (function, line 60) — **reference-only**. ID `enco-05129447c1d2abc6`.
- `build` (function, line 83) — **reference-only**. ID `enco-0d5c21aa29974be1`.
- `watch` (function, line 109) — **reference-only**. ID `enco-a8c7c3174febafe6`.
- `fonts` (function, line 138) — **reference-only**. ID `enco-15a02487f432f667`.
- `sync` (function, line 149) — **reference-only**. ID `enco-bf8bbf9fee733287`.
- `entry` (function, line 195) — **reference-only**. ID `enco-1ff5e728a35cb3d3`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/instrument.py`**

- `list` (function, line 26) — **reference-only**. ID `enco-dc7bc52e234ea92b`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/simulation.py`**

- `clean` (function, line 19) — **reference-only**. ID `enco-d83a420f990b582a`.

**enco: `enco/packages/enco-cli/src/enco_cli/cli/test.py`**

- `Environment` (class, line 24) — **reference-only**. ID `enco-129d4e2945d0f8e9`.
- `detect_environment` (function, line 29) — **reference-only**. ID `enco-5c02a672719a9ae7`.
- `function_prompt` (function, line 66) — **reference-only**. ID `enco-d45654dc6b095a16`.
- `clone` (function, line 93) — **reference-only**. ID `enco-4027ea8df1078fdf`.
- `list` (function, line 152) — **reference-only**. ID `enco-b2c2c3e018d3593e`.
- `init` (function, line 176) — **reference-only**. ID `enco-309745224be7893d`.

**enco: `enco/packages/enco-cli/src/enco_cli/configuration.py`**

- `update_configuration` (function, line 26) — **reference-only**. ID `enco-baa5c909a29ddb94`.

**enco: `enco/packages/enco-cli/src/enco_cli/version/__init__.py`**

- `upgrade` (function, line 16) — **reference-only**. ID `enco-50043e1e7c224fa5`.

**enco: `enco/packages/enco-components/src/enco_components/components/capacitor/__init__.py`**

- `CapacitorModel` (class, line 40) — **reference-only**. ID `enco-628ebfc50593b54d`.
- `Capacitor` (class, line 45) — **reference-only**. ID `enco-5d8bc6411dbb536f`.
- `Capacitor.__init__` (function, line 49) — **reference-only**. ID `enco-0398b62e22c61ea3`.
- `Capacitor.validate` (function, line 55) — **reference-only**. ID `enco-b689305b9bb1cf6a`.
- `Capacitor.train` (function, line 59) — **stub**. ID `enco-a19e0c3275118534`.
- `Capacitor.get_volume_index` (function, line 64) — **reference-only**. ID `enco-ae71851e8f987206`.
- `Capacitor.get_loss_index` (function, line 70) — **reference-only**. ID `enco-d1e942f1d23717e3`.

**enco: `enco/packages/enco-components/src/enco_components/components/capacitor/ceramic_capacitor.py`**

- `CeramicCapacitorModel` (class, line 22) — **reference-only**. ID `enco-114489fd2475f73d`.
- `CeramicCapacitor` (class, line 26) — **reference-only**. ID `enco-f3c24f0c7170e71c`.
- `CeramicCapacitor.capacitance` (function, line 31) — **reference-only**. ID `enco-a27e80043030e946`.
- `CeramicCapacitor.capacitance` (function, line 35) — **reference-only**. ID `enco-73b3613053c168c1`.
- `CeramicCapacitor.validate` (function, line 39) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-715bf40d31b83790`.
- `CeramicCapacitor.train` (function, line 43) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-f2e0b74f2b871abc`.
- `CeramicCapacitor.get_volume_index` (function, line 48) — **reference-only**. ID `enco-e12561fe59fd14dd`.

**enco: `enco/packages/enco-components/src/enco_components/components/capacitor/film_capacitor.py`**

- `FilmCapacitorModel` (class, line 22) — **reference-only**. ID `enco-dd76c1f342f0963e`.
- `FilmCapacitor` (class, line 26) — **reference-only**. ID `enco-3ded0e52f3e69b4b`.
- `FilmCapacitor.capacitance` (function, line 31) — **duplicate**; canonical `enco-a27e80043030e946`. ID `enco-fe61822f3d5e72f3`.
- `FilmCapacitor.capacitance` (function, line 35) — **duplicate**; canonical `enco-73b3613053c168c1`. ID `enco-d6b624639259cc1e`.
- `FilmCapacitor.validate` (function, line 39) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-b2de086535e05f38`.
- `FilmCapacitor.train` (function, line 43) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-ee7e8222d7ff52f4`.
- `FilmCapacitor.get_volume_index` (function, line 48) — **duplicate**; canonical `enco-e12561fe59fd14dd`. ID `enco-d72b09d676ca3b5c`.

**enco: `enco/packages/enco-components/src/enco_components/components/capacitor/ideal.py`**

- `IdealCapacitorModel` (class, line 22) — **reference-only**. ID `enco-412cb82938d3bf23`.
- `IdealCapacitor` (class, line 26) — **reference-only**. ID `enco-152d5eed93192698`.
- `IdealCapacitor.capacitance` (function, line 31) — **duplicate**; canonical `enco-a27e80043030e946`. ID `enco-25e556bdb84d11cc`.
- `IdealCapacitor.capacitance` (function, line 35) — **duplicate**; canonical `enco-73b3613053c168c1`. ID `enco-3826828564921091`.
- `IdealCapacitor.validate` (function, line 41) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-b7235680c90f98bd`.
- `IdealCapacitor.train` (function, line 47) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-3ad7caaf3597d42a`.
- `IdealCapacitor.get_volume_index` (function, line 52) — **duplicate**; canonical `enco-e12561fe59fd14dd`. ID `enco-9e1141dd18e9b984`.

**enco: `enco/packages/enco-components/src/enco_components/components/inductor/__init__.py`**

- `InductorModel` (class, line 26) — **reference-only**. ID `enco-9bd08cd0a291b9da`.
- `Inductor` (class, line 31) — **reference-only**. ID `enco-47bdf34268f8099d`.
- `Inductor.__init__` (function, line 35) — **reference-only**. ID `enco-d2705e5e21941e83`.
- `Inductor.validate` (function, line 41) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-9ebfc5b776454f06`.
- `Inductor.train` (function, line 45) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-48778ef29787cdd4`.
- `Inductor.get_volume_index` (function, line 50) — **reference-only**. ID `enco-946d5d3f66e772c6`.
- `Inductor.get_loss_index` (function, line 59) — **reference-only**. ID `enco-0b6464fb972d95c7`.

**enco: `enco/packages/enco-components/src/enco_components/components/resistor/__init__.py`**

- `ResistorModel` (class, line 32) — **reference-only**. ID `enco-b370175a84626691`.
- `Resistor` (class, line 46) — **reference-only**. ID `enco-6f5e3b731cc0bae3`.
- `Resistor.__init__` (function, line 50) — **reference-only**. ID `enco-c099227a495556dc`.
- `Resistor.validate` (function, line 54) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-480c0572ebe5bcc1`.
- `Resistor.train` (function, line 58) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-9bb13c63ab07eb6c`.
- `Resistor.lifetime` (function, line 62) — **reference-only**. ID `enco-0b220c0660a2d42e`.
- `Resistor.tolerance` (function, line 69) — **reference-only**. ID `enco-5fe53e341422c778`.
- `Resistor.get_volume_index` (function, line 78) — **reference-only**. ID `enco-48f8f15a803cf775`.
- `Resistor.get_loss_index` (function, line 82) — **reference-only**. ID `enco-62dbb787b23bc4a7`.

**enco: `enco/packages/enco-components/src/enco_components/components/resistor/thick_film.py`**

- `ThickFilmResistorModel` (class, line 20) — **reference-only**. ID `enco-c79fc35a553cb8f8`.
- `ThickFilmResistor` (class, line 25) — **reference-only**. ID `enco-8b2ed6b05459c9f3`.
- `ThickFilmResistor.__init__` (function, line 29) — **reference-only**. ID `enco-69a3dd1ce75dfc0b`.
- `ThickFilmResistor.validate` (function, line 33) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-4bd99282790f8f1c`.
- `ThickFilmResistor.train` (function, line 39) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-666cfd46ed5e91c4`.

**enco: `enco/packages/enco-components/src/enco_components/components/resistor/thin_film.py`**

- `ThinFilmResistorModel` (class, line 20) — **reference-only**. ID `enco-80c16f05f8ddcba5`.
- `ThinFilmResistor` (class, line 25) — **reference-only**. ID `enco-4df20fe05a466dc1`.
- `ThinFilmResistor.__init__` (function, line 29) — **reference-only**. ID `enco-447d37fd6204e028`.
- `ThinFilmResistor.validate` (function, line 33) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-f562530916a78768`.
- `ThinFilmResistor.train` (function, line 39) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-bf9f5338fff71c6a`.

**enco: `enco/packages/enco-core/src/enco_core/configuration/__init__.py`**

- `update_configuration` (function, line 27) — **reference-only**. ID `enco-6a7ff9fd7bbac6a1`.

**enco: `enco/packages/enco-core/src/enco_core/configuration/configuration.py`**

- `totoml` (function, line 21) — **reference-only**. ID `enco-e158cd324b0536ff`.
- `ConfigurationSnippet` (class, line 33) — **reference-only**. ID `enco-997a95d671314bc7`.
- `ConfigurationSnippet.__init__` (function, line 38) — **reference-only**. ID `enco-8c741c629895a028`.
- `ConfigurationSnippet.render` (function, line 42) — **reference-only**. ID `enco-00f75d5e9c867497`.
- `ConfigurationSnippet.__eq__` (function, line 55) — **reference-only**. ID `enco-d42470f168b8c5b9`.
- `Configuration` (class, line 62) — **reference-only**. ID `enco-e34098139f47035d`.
- `Configuration.__init__` (function, line 80) — **reference-only**. ID `enco-a1e1dd6a51b38af0`.
- `Configuration.save` (function, line 85) — **reference-only**. ID `enco-832baa8620892a6b`.
- `Configuration.load` (function, line 138) — **reference-only**. ID `enco-0ea99be878cb7688`.
- `Configuration.exists` (function, line 146) — **reference-only**. ID `enco-1b693a7880ff6892`.
- `Configuration.get` (function, line 151) — **reference-only**. ID `enco-5a1151a4925c1381`.
- `Configuration.set` (function, line 161) — **reference-only**. ID `enco-42c74edc8d3be065`.
- `Configuration.__getitem__` (function, line 171) — **reference-only**. ID `enco-8b32266f803613d1`.
- `Configuration.__setitem__` (function, line 174) — **reference-only**. ID `enco-5efee7999814d66f`.
- `Configuration.__delitem__` (function, line 177) — **reference-only**. ID `enco-7355a23c8443e5df`.
- `Configuration.__contains__` (function, line 180) — **reference-only**. ID `enco-0b844059d5e44e6a`.
- `Configuration.__str__` (function, line 183) — **reference-only**. ID `enco-c97fe77668d598c7`.
- `Configuration.__repr__` (function, line 186) — **reference-only**. ID `enco-0a91279d81e9e1bb`.
- `Configuration.__iter__` (function, line 189) — **reference-only**. ID `enco-30a0ddb23983cd20`.
- `Configuration.__len__` (function, line 192) — **reference-only**. ID `enco-043e83a0da1df30a`.

**enco: `enco/packages/enco-core/src/enco_core/e_series.py`**

- `ESeries` (class, line 400) — **reference-only**. ID `enco-06993c04b039e8e3`.
- `get_closest_series_value` (function, line 410) — **reference-only**. ID `enco-7f20218e4400303f`.
- `get_series_values` (function, line 420) — **reference-only**. ID `enco-daaf18b627abd047`.

**enco: `enco/packages/enco-core/src/enco_core/library/library.py`**

- `filter_json` (function, line 24) — **reference-only**. ID `enco-13ce3e3d00f6c6c7`.
- `quote` (function, line 36) — **reference-only**. ID `enco-82d47858af1b39d4`.
- `Library` (class, line 44) — **reference-only**. ID `enco-6704e160cc755dad`.
- `Library.__init__` (function, line 50) — **reference-only**. ID `enco-a1e355de823ffac2`.
- `Library.delete` (function, line 76) — **reference-only**. ID `enco-22f67c2a9db76831`.
- `Library.update` (function, line 88) — **reference-only**. ID `enco-38f75dc1bd2f539c`.
- `Library.get` (function, line 107) — **reference-only**. ID `enco-da09eae32b060785`.
- `Library.search` (function, line 119) — **reference-only**. ID `enco-6e8b42b8049fae8b`.
- `Library._load_templates` (function, line 169) — **reference-only**. ID `enco-a9284f5904d4acb4`.

**enco: `enco/packages/enco-core/src/enco_core/library/template.py`**

- `Template` (class, line 25) — **reference-only**. ID `enco-29d365b611d4ee2e`.
- `Template.__init__` (function, line 26) — **reference-only**. ID `enco-0bb32722a0c9043d`.
- `Template._load_identifier` (function, line 68) — **reference-only**. ID `enco-053cd41ed86aa2bc`.
- `Template._validate_manifest` (function, line 90) — **reference-only**. ID `enco-e0272bf432bb17cb`.
- `Template._run_prompt_script` (function, line 117) — **reference-only**. ID `enco-c83a6f501ea64347`.
- `Template.render` (function, line 170) — **reference-only**. ID `enco-1728c65c0b0b3248`.
- `Template.__repr__` (function, line 246) — **reference-only**. ID `enco-7a41154397455742`.
- `Template.__str__` (function, line 249) — **reference-only**. ID `enco-1178cadd920c62b0`.

**enco: `enco/packages/enco-core/src/enco_core/path.py`**

- `get_data_path` (function, line 20) — **reference-only**. ID `enco-f9946dc86ca2bc95`.

**enco: `enco/packages/enco-core/src/enco_core/repository/git.py`**

- `GitRepository` (class, line 20) — **reference-only**. ID `enco-614a10c097fd05a8`.
- `GitRepository.__init__` (function, line 21) — **reference-only**. ID `enco-c843548e716196fb`.
- `GitRepository.delete` (function, line 35) — **reference-only**. ID `enco-8aca20a8a182d386`.
- `GitRepository.delete.remove_readonly` (function, line 41) — **reference-only**. ID `enco-f3fdc559a66ac806`.
- `GitRepository.update` (function, line 54) — **reference-only**. ID `enco-66228283c7faecd1`.
- `GitRepository.information` (function, line 128) — **reference-only**. ID `enco-34e8b629b8ac5ab8`.
- `GitRepository.commit` (function, line 224) — **reference-only**. ID `enco-7386edb491fd4b2c`.
- `GitRepository.push` (function, line 233) — **reference-only**. ID `enco-8fcaa1284cd12829`.

**enco: `enco/packages/enco-core/src/enco_core/repository/svn.py`**

- `SubversionRepository` (class, line 16) — **reference-only**. ID `enco-847540b937f27ca8`.
- `SubversionRepository.__init__` (function, line 17) — **reference-only**. ID `enco-040dcdd1ec3f30a7`.
- `SubversionRepository.information` (function, line 28) — **reference-only**. ID `enco-fd4512878db2ddfc`.
- `SubversionRepository.update` (function, line 92) — **reference-only**. ID `enco-8f3db7f9740c6e37`.

**enco: `enco/packages/enco-core/src/enco_core/structures/metadata.py`**

- `MetadataBase` (class, line 22) — **reference-only**. ID `enco-d45b13c561187177`.
- `MetadataBase.__init__` (function, line 34) — **reference-only**. ID `enco-a4d161f8954546f2`.
- `MetadataBase.get_uuid` (function, line 43) — **reference-only**. ID `enco-2b2288cc93e43209`.
- `MetadataBase.get_timestamp` (function, line 46) — **reference-only**. ID `enco-a71ddd2a4e2b05d8`.
- `MetadataBase.get_identifier` (function, line 49) — **duplicate**; canonical `enco-35350ab0bd8d3862`. ID `enco-6f6cee9e7166f447`.
- `MetadataBase.get_label` (function, line 52) — **reference-only**. ID `enco-65ad8bb63503d8e2`.
- `MetadataBase.get_description` (function, line 55) — **reference-only**. ID `enco-8b34f480d4c65cce`.
- `MetadataBase.get_tags` (function, line 58) — **reference-only**. ID `enco-7f68b863b191b65c`.
- `MetadataBase.get_attributes` (function, line 61) — **reference-only**. ID `enco-ecf9d4be844a2267`.
- `MetadataBase.set_label` (function, line 64) — **reference-only**. ID `enco-43d6a924d85c320e`.
- `MetadataBase.set_description` (function, line 67) — **reference-only**. ID `enco-a60f94c9cbd20b00`.
- `MetadataBase.set_tags` (function, line 70) — **reference-only**. ID `enco-390b9baa6ade39dc`.
- `MetadataBase.set_attributes` (function, line 73) — **reference-only**. ID `enco-cf6ad8304f47b6ae`.
- `MetadataBase.add_tag` (function, line 76) — **reference-only**. ID `enco-40489a30045c3858`.
- `MetadataBase.add_attribute` (function, line 79) — **reference-only**. ID `enco-e9c41d86f0267a81`.
- `MetadataBase.remove_tag` (function, line 82) — **reference-only**. ID `enco-28f0e0be6c0cd394`.
- `MetadataBase.remove_attribute` (function, line 85) — **reference-only**. ID `enco-77ad015fb5d10d23`.

**enco: `enco/packages/enco-core/src/enco_core/utility/__init__.py`**

- `SerializableObject` (class, line 24) — **reference-only**. ID `enco-51d54dc4e47f77d6`.
- `SerializableObject.to_dict` (function, line 30) — **stub**. ID `enco-f2e7d0d0fbd49ed1`.
- `SerializableObject.from_dict` (function, line 34) — **stub**. ID `enco-d262dae53eb9e219`.
- `SerializableObject.to_json` (function, line 38) — **stub**. ID `enco-97dad1a079a053c2`.
- `SerializableObject.from_json` (function, line 42) — **stub**. ID `enco-6be539611c4f981d`.
- `NamedObject` (class, line 46) — **reference-only**. ID `enco-baa132c8a320dda6`.
- `NamedObject.__init__` (function, line 51) — **reference-only**. ID `enco-e8bc10dd2deded35`.
- `NamedObject.__eq__` (function, line 57) — **reference-only**. ID `enco-b9a2df0895d85fb2`.
- `NamedObject.__hash__` (function, line 63) — **reference-only**. ID `enco-0eaf68814bf7ee44`.
- `NamedObject.__str__` (function, line 68) — **stub**. ID `enco-cd1bf077afc8bcb8`.
- `NamedObject.__repr__` (function, line 73) — **stub**. ID `enco-856031dc0a16fb97`.
- `NamedObject.to_dict` (function, line 77) — **reference-only**. ID `enco-cd95442a7d0da899`.
- `NamedObject.from_dict` (function, line 83) — **reference-only**. ID `enco-d8caadf70700aa2f`.
- `NamedObject.to_json` (function, line 87) — **reference-only**. ID `enco-b8d3a58d145623be`.
- `NamedObject.from_json` (function, line 90) — **reference-only**. ID `enco-e7c45ba0bc726dfd`.

**enco: `enco/packages/enco-core/src/enco_core/utility/context.py`**

- `Context` (class, line 21) — **reference-only**. ID `enco-f0bbfb262b4c4265`.
- `Context.__init__` (function, line 22) — **reference-only**. ID `enco-7e77a4bacb10092a`.
- `Context.load` (function, line 43) — **reference-only**. ID `enco-613d342a79828679`.
- `Context.extend` (function, line 61) — **reference-only**. ID `enco-d015868bc14ff5b5`.
- `Context.to_file` (function, line 72) — **stub**. ID `enco-7aafe152f1feca5b`.
- `Context._get_context_git` (function, line 89) — **reference-only**. ID `enco-215bdaba2d2c75f6`.
- `Context._get_context_git.return_and_print` (function, line 98) — **reference-only**. ID `enco-0ea6dc1a2fc375a5`.
- `Context._get_context_svn` (function, line 144) — **reference-only**. ID `enco-817b2df6eaa53a86`.
- `Context._get_context_svn.return_and_print` (function, line 153) — **reference-only**. ID `enco-809852c77b12cb4f`.
- `Context._get_context_system` (function, line 168) — **reference-only**. ID `enco-b381890cd4605d33`.
- `Context._get_context_system.return_and_print` (function, line 176) — **reference-only**. ID `enco-ca00ec0f379b2966`.

**enco: `enco/packages/enco-core/src/enco_core/utility/dataset.py`**

- `Dataset` (class, line 8) — **reference-only**. ID `enco-4ad8eb7cec3c64c6`.
- `Dataset.__init__` (function, line 9) — **reference-only**. ID `enco-8f120746cadf20eb`.
- `Dataset.load` (function, line 22) — **reference-only**. ID `enco-407c29aab0c94e50`.
- `Dataset.to_dataframe` (function, line 33) — **reference-only**. ID `enco-3ca0f40326085a5a`.

**enco: `enco/packages/enco-core/src/enco_core/utility/metadata.py`**

- `PackageMetadata` (class, line 14) — **reference-only**. ID `enco-c09a72cb39cd793d`.
- `PackageMetadata.__init__` (function, line 15) — **reference-only**. ID `enco-e76c61062382b7b8`.
- `PackageMetadata.version` (function, line 18) — **reference-only**. ID `enco-3fda3b7dcc2cf2d3`.
- `PackageMetadata.authors` (function, line 33) — **reference-only**. ID `enco-7f5f14f84d55ec84`.

**enco: `enco/packages/enco-core/src/enco_core/utility/path.py`**

- `get_data_folder` (function, line 14) — **reference-only**. ID `enco-25c00624ed3cec59`.

**enco: `enco/packages/enco-core/src/enco_core/utility/version.py`**

- `get_versions` (function, line 20) — **reference-only**. ID `enco-ad84800278540340`.
- `VersionInformation` (class, line 68) — **reference-only**. ID `enco-afd9ec7a0dec2fff`.
- `VersionInformation.__init__` (function, line 69) — **reference-only**. ID `enco-177857b73fb88ae4`.
- `VersionInformation.__str__` (function, line 81) — **reference-only**. ID `enco-ecafd85d521503b3`.
- `VersionInformation.__repr__` (function, line 84) — **reference-only**. ID `enco-d80766ec12ef5802`.
- `get_version_information` (function, line 88) — **reference-only**. ID `enco-37a964beb7de6cef`.

**enco: `enco/packages/enco-core/tests/test_version.py`**

- `TestVersions` (class, line 21) — **reference-only**. ID `enco-d19ee100004ffd4a`.
- `TestVersions.test_get_versions` (function, line 22) — **reference-only**. ID `enco-8c246d9b0d630211`.

**enco: `enco/packages/enco-documentation/src/enco_documentation/compiler.py`**

- `build` (function, line 27) — **reference-only**. ID `enco-e799c5afecf8dee3`.
- `watch` (function, line 140) — **reference-only**. ID `enco-0be956aff42fb074`.
- `watch.cleanup` (function, line 240) — **reference-only**. ID `enco-80f418d625e4c816`.
- `fonts` (function, line 269) — **reference-only**. ID `enco-08d496ce6115b57e`.

**enco: `enco/packages/enco-documentation/src/enco_documentation/configuration.py`**

- `update_configuration` (function, line 17) — **reference-only**. ID `enco-25b24b53dfc91832`.

**enco: `enco/packages/enco-documentation/src/enco_documentation/packages.py`**

- `update_local_packages` (function, line 18) — **reference-only**. ID `enco-f06ad119b51149dc`.

**enco: `enco/packages/enco-documentation/src/enco_documentation/templates.py`**

- `get_template_library` (function, line 20) — **reference-only**. ID `enco-7ab72ce3f2c326e7`.
- `get_local_package_folder` (function, line 62) — **reference-only**. ID `enco-1989cc07f2ea771b`.

**enco: `enco/packages/enco-documentation/src/enco_documentation/typst.py`**

- `is_typst_installed` (function, line 21) — **reference-only**. ID `enco-9345501774d72b1a`.
- `install_typst` (function, line 30) — **reference-only**. ID `enco-7cb1e25ec35912ba`.
- `upgrade_typst` (function, line 40) — **reference-only**. ID `enco-6b90eb503c300884`.

**enco: `enco/packages/enco-documentation/tests/test_compiler.py`**

- `test_build` (function, line 24) — **reference-only**. ID `enco-badf4e0c24689a77`.
- `test_watch` (function, line 31) — **reference-only**. ID `enco-11b1ed8c4c81508e`.

**enco: `enco/packages/enco-models/src/enco_models/metadata.py`**

- `MetadataModel` (class, line 23) — **reference-only**. ID `enco-c1151287c5cc4a41`.
- `MetadataModel.__init__` (function, line 35) — **reference-only**. ID `enco-d063e56265cd4986`.

**enco: `enco/packages/enco-models/src/enco_models/standards/IPC9592.py`**

- `DeviceType` (class, line 22) — **reference-only**. ID `enco-181561b003719caf`.
- `Model_IPC9592A` (class, line 49) — **reference-only**. ID `enco-07cfc8c34b6d82e7`.

**enco: `enco/packages/enco-models/src/enco_models/standards/standard.py`**

- `StandardModel` (class, line 19) — **reference-only**. ID `enco-3e99ec01f4d5f371`.

**enco: `enco/packages/enco-models/src/enco_models/validation.py`**

- `ValidationCriterionModel` (class, line 21) — **reference-only**. ID `enco-06f5966031008614`.
- `ValidationMaskModel` (class, line 29) — **reference-only**. ID `enco-46e5f9a3aa3513c0`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/adapter.py`**

- `Adapter` (class, line 23) — **reference-only**. ID `enco-a38eaeb791268f4f`.
- `Adapter.__init__` (function, line 28) — **reference-only**. ID `enco-5993e894f61cb5e0`.
- `Adapter._before_run` (function, line 35) — **stub**. ID `enco-45cf66fe98eeeed1`.
- `Adapter._run` (function, line 40) — **stub**. ID `enco-13d59bca93f69dd0`.
- `Adapter._after_run` (function, line 44) — **stub**. ID `enco-fecc43fffed3a149`.
- `Adapter._load` (function, line 49) — **stub**. ID `enco-0c2f8e618e927f8f`.
- `Adapter.run` (function, line 52) — **reference-only**. ID `enco-0343b8e152c9a00d`.
- `Adapter.__repr__` (function, line 69) — **reference-only**. ID `enco-aceaab85f11b691a`.
- `Adapter.__str__` (function, line 72) — **reference-only**. ID `enco-0416f2db02e62fd3`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/configuration.py`**

- `update_configuration` (function, line 32) — **reference-only**. ID `enco-d9ed8a10d9582b6f`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/handlers.py`**

- `slice` (function, line 24) — **reference-only**. ID `enco-cb5816970a15557d`.
- `slice.handler` (function, line 25) — **reference-only**. ID `enco-7b4af6a0cf358236`.
- `rename` (function, line 55) — **reference-only**. ID `enco-933ca7f2f59eee4c`.
- `rename.handler` (function, line 58) — **reference-only**. ID `enco-363a5b6da05c8b55`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/pool.py`**

- `SimulationPool` (class, line 15) — **reference-only**. ID `enco-fa2a50a7ddffafa4`.
- `SimulationPool.__init__` (function, line 16) — **reference-only**. ID `enco-e2413d96dd388258`.
- `SimulationPool.add` (function, line 26) — **reference-only**. ID `enco-48725d12063708b2`.
- `SimulationPool.get` (function, line 32) — **reference-only**. ID `enco-83afc39a9f20e0b8`.
- `SimulationPool.run` (function, line 39) — **reference-only**. ID `enco-73b6093b9d8e8b1b`.
- `SimulationPool.__repr__` (function, line 75) — **reference-only**. ID `enco-282d537824db754c`.
- `SimulationPool.__str__` (function, line 78) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-25ce796d9966504b`.
- `SimulationPool.__len__` (function, line 81) — **reference-only**. ID `enco-445a2ea580d5cef2`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/qspice/adapter.py`**

- `QSpiceAdapter` (class, line 32) — **reference-only**. ID `enco-25236e7d59604ada`.
- `QSpiceAdapter.__init__` (function, line 37) — **reference-only**. ID `enco-24654deeff184146`.
- `QSpiceAdapter._generate_netlist` (function, line 68) — **reference-only**. ID `enco-7fa4f857fde83dab`.
- `QSpiceAdapter._before_run` (function, line 167) — **duplicate**; canonical `enco-0fdf1d55e510c907`. ID `enco-1fa7d30bdbf9f2e9`.
- `QSpiceAdapter._after_run` (function, line 175) — **stub**; canonical `enco-782cd20fb5273236`. ID `enco-26c2a8bb3ffe52e9`.
- `QSpiceAdapter._operating_point` (function, line 179) — **reference-only**. ID `enco-46bac55a91369683`.
- `QSpiceAdapter._load_operating_point` (function, line 198) — **reference-only**. ID `enco-21b7d4ace52a1320`.
- `QSpiceAdapter._transient` (function, line 217) — **reference-only**. ID `enco-51ae6858df42759f`.
- `QSpiceAdapter._load_transient` (function, line 236) — **reference-only**. ID `enco-7b29f1c79202b3df`.
- `QSpiceAdapter._ac` (function, line 257) — **reference-only**. ID `enco-aa6047916889f4cf`.
- `QSpiceAdapter._load_ac` (function, line 276) — **reference-only**. ID `enco-42ed5113fab90c4c`.
- `QSpiceAdapter.__repr__` (function, line 292) — **reference-only**. ID `enco-7f57a0fa029c8873`.
- `QSpiceAdapter.__str__` (function, line 295) — **reference-only**. ID `enco-34ff85589e499104`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/qspice/loader.py`**

- `QSpiceRawLoader` (class, line 33) — **reference-only**. ID `enco-f30e613e143a71f7`.
- `QSpiceRawLoader.__init__` (function, line 34) — **duplicate**; canonical `enco-f255070493037515`. ID `enco-39ac9db29a75656d`.
- `QSpiceRawLoader.load` (function, line 46) — **duplicate**; canonical `enco-69a0d6a0af6c8487`. ID `enco-bb374def62ec1774`.
- `QSpiceRawLoader.__load` (function, line 56) — **reference-only**. ID `enco-8c4bee0ce021fc4e`.
- `QSpiceRawLoader.__read_binary` (function, line 92) — **reference-only**. ID `enco-78890674cc590c19`.
- `QSpiceRawLoader.__parse_header` (function, line 241) — **reference-only**. ID `enco-d69f08c5bf42a149`.
- `QSpiceRawLoader.__getitem__` (function, line 302) — **duplicate**; canonical `enco-052fac610cadf5ec`. ID `enco-f6457cdf58187157`.
- `QSpiceRawLoader.__repr__` (function, line 314) — **reference-only**. ID `enco-4f91cadc2dd54a08`.
- `QSpiceRawLoader.__str__` (function, line 320) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-cc69bd006ac631bb`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/qspice/type.py`**

- `QSpiceSimulationType` (class, line 21) — **reference-only**. ID `enco-8567f8ce682fd371`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/results.py`**

- `SimulationResults` (class, line 19) — **reference-only**. ID `enco-28622b8f1f6fbe0d`.
- `SimulationResults.__init__` (function, line 25) — **reference-only**. ID `enco-c820c2e39c7b9352`.
- `SimulationResults.get_dataframe` (function, line 138) — **reference-only**. ID `enco-0fc50357f2300215`.
- `SimulationResults.__repr__` (function, line 147) — **reference-only**. ID `enco-b2e55d1a9ce0d949`.
- `SimulationResults.__str__` (function, line 150) — **reference-only**. ID `enco-e83dd97e064eaf88`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/simulation.py`**

- `Simulation` (class, line 29) — **reference-only**. ID `enco-e30bc18d770ff294`.
- `Simulation.__init__` (function, line 42) — **reference-only**. ID `enco-4554109d421a1015`.
- `Simulation.get_parameter` (function, line 87) — **reference-only**. ID `enco-5a2b556900d67b04`.
- `Simulation.set_parameter` (function, line 100) — **reference-only**. ID `enco-1dde7d5b96652bb0`.
- `Simulation.get_parameters` (function, line 111) — **reference-only**. ID `enco-eb0844020e2f59f8`.
- `Simulation.add_handler` (function, line 118) — **reference-only**. ID `enco-6fecd1550a191757`.
- `Simulation._save_results` (function, line 125) — **reference-only**. ID `enco-7998c1385715ae83`.
- `Simulation._load_results` (function, line 137) — **reference-only**. ID `enco-f4ecb21c2428e0ab`.
- `Simulation._results_exist` (function, line 151) — **reference-only**. ID `enco-292b345e6fccf612`.
- `Simulation.run` (function, line 162) — **reference-only**. ID `enco-a07bb634a5fa4b5a`.
- `Simulation.hash` (function, line 246) — **reference-only**. ID `enco-b0009c8abd3bcbea`.
- `Simulation.__repr__` (function, line 294) — **reference-only**. ID `enco-559d5a1781f826b5`.
- `Simulation.__str__` (function, line 297) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-7be337939dd30535`.
- `clear_results` (function, line 304) — **reference-only**. ID `enco-4a35a0af6969a807`.
- `clear_results.compute_size` (function, line 329) — **reference-only**. ID `enco-38965b2c9e12f2b4`.
- `clear_results.delete_oldest_file` (function, line 343) — **reference-only**. ID `enco-2e0f737d0e671a9d`.
- `clear_results.delete_old_files` (function, line 368) — **reference-only**. ID `enco-4497b856025bb9e4`.

**enco: `enco/packages/enco-simulation/src/enco_simulation/type.py`**

- `SimulationType` (class, line 14) — **reference-only**. ID `enco-7d42c627d93ea502`.
- `OperatingPoint` (class, line 18) — **reference-only**. ID `enco-67260cb2c2e70fa7`.
- `OperatingPoint.__repr__` (function, line 20) — **reference-only**. ID `enco-fe7eda678c363bf8`.
- `OperatingPoint.__str__` (function, line 23) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-2c480f0dadd8a465`.
- `Transient` (class, line 27) — **reference-only**. ID `enco-c8cec6012b1e365a`.
- `Transient.__init__` (function, line 28) — **reference-only**. ID `enco-26dae91af5dc4c1f`.
- `Transient.__repr__` (function, line 34) — **reference-only**. ID `enco-949539ce5c3cb951`.
- `Transient.__str__` (function, line 37) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-a6ff8d619a2859c3`.
- `AC` (class, line 41) — **reference-only**. ID `enco-39c07cf130bf192a`.
- `AC.__init__` (function, line 42) — **reference-only**. ID `enco-1ed7ae6d2cc85394`.
- `AC.__repr__` (function, line 48) — **reference-only**. ID `enco-73a759ec8adfb370`.
- `AC.__str__` (function, line 51) — **duplicate**; canonical `enco-0416f2db02e62fd3`. ID `enco-ed040ee3ddf1f2cb`.

**enco: `enco/packages/enco-standards/src/enco_standards/cispr25.py`**

- `lisn_dm_transfer_function` (function, line 23) — **reference-only**. ID `enco-3840dc3851e23e3f`.
- `lisn_cm_transfer_function` (function, line 45) — **reference-only**. ID `enco-7c6c981960e68d34`.
- `lisn_cm_transfer_function.parallel` (function, line 66) — **reference-only**. ID `enco-338a6ab045282e4b`.

**enco: `enco/packages/enco-standards/src/enco_standards/limits.py`**

- `EMC_Limit` (class, line 13) — **reference-only**. ID `enco-2cb10fe39ca76fcb`.
- `EMC_Limit.__init__` (function, line 14) — **reference-only**. ID `enco-c93a5dc3e88f5e1b`.
- `EMC_Limit.set_mask` (function, line 19) — **reference-only**. ID `enco-e30eea043c0d34de`.
- `EMC_Limit.remove_mask` (function, line 33) — **reference-only**. ID `enco-b7ae3cfdb30b2ba0`.
- `EMC_Limit.get_mask` (function, line 42) — **reference-only**. ID `enco-10f250e11fcd3a19`.
- `EMC_Limit.get_mask_identifiers` (function, line 51) — **reference-only**. ID `enco-54221f389586d20b`.
- `EMC_Limit.interpolate_mask` (function, line 61) — **reference-only**. ID `enco-9880bbb128379ae8`.
- `EMC_Limit.compute_gain` (function, line 84) — **reference-only**. ID `enco-71d38cf6fa934fe0`.
- `EMC_Limit.get_ec_r10_ac` (function, line 110) — **reference-only**. ID `enco-3de3abfcff7b0b30`.
- `EMC_Limit.get_ec_r10_dc` (function, line 130) — **reference-only**. ID `enco-c86a533e999ae950`.
- `EMC_Limit.get_tl81000_2018_03_an` (function, line 150) — **reference-only**. ID `enco-2d10b66770dd5538`.
- `EMC_Limit.get_cispr_22_a` (function, line 167) — **reference-only**. ID `enco-78d0ded36f4d50d4`.
- `EMC_Limit.get_cispr_22_b` (function, line 187) — **reference-only**. ID `enco-1858a943b8646b62`.
- `EMC_Limit.get_cispr_25` (function, line 211) — **reference-only**. ID `enco-7a49e27d13afeb70`.
- `EMC_Limit.combine` (function, line 406) — **reference-only**. ID `enco-589edd782eb24863`.
- `TestEMCLimit` (class, line 433) — **reference-only**. ID `enco-68d3dd08a19fffe1`.
- `TestEMCLimit.setUp` (function, line 434) — **reference-only**. ID `enco-0c92663cb69d417a`.
- `TestEMCLimit.test_set_mask` (function, line 437) — **reference-only**. ID `enco-bd9834c278415cfd`.
- `TestEMCLimit.test_remove_mask` (function, line 445) — **reference-only**. ID `enco-78c28c7f578cc1c7`.
- `TestEMCLimit.test_get_mask` (function, line 454) — **reference-only**. ID `enco-ed28fc9edfac1ef5`.
- `TestEMCLimit.test_interpolate_mask` (function, line 463) — **reference-only**. ID `enco-c6c334f0fb4e1583`.
- `TestEMCLimit.test_compute_gain` (function, line 474) — **reference-only**. ID `enco-d19bfbb2f837a7a0`.
- `TestEMCLimit.test_interpolation` (function, line 486) — **reference-only**. ID `enco-5b57b8b5b828435b`.

**enco: `enco/packages/enco-standards/src/enco_standards/standard.py`**

- `Standard` (class, line 21) — **reference-only**. ID `enco-85de2e46309e02df`.
- `Standard.attach_verification_criteria` (function, line 23) — **stub**. ID `enco-755993af3416b905`.

**enco: `library-design/dcdc/cllc/src/project/analysis/maximum_power.py`**

- `compute_max_power` (function, line 32) — **reference-only**. ID `enco-b39110fa92c85453`.

**enco: `library-design/dcdc/cllc/src/project/cllc.py`**

- `main` (function, line 24) — **reference-only**. ID `enco-1be33686bcf3bb10`.

**enco: `library-design/dcdc/cllc/src/project/components/cllc_transformer.py`**

- `DAB_Transformer_Model` (class, line 23) — **reference-only**. ID `enco-a964116cc7748783`.
- `CLLC_Transformer` (class, line 44) — **reference-only**. ID `enco-bd256d1066a2ec3e`.
- `CLLC_Transformer.validate` (function, line 48) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-d50308caaaf033cf`.
- `CLLC_Transformer.train` (function, line 52) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-0a299d20621a5d3b`.
- `CLLC_Transformer.winding_loss` (function, line 56) — **reference-only**. ID `enco-eebbc89e7848e95f`.
- `CLLC_Transformer.core_loss` (function, line 88) — **reference-only**. ID `enco-16c63550a6dfa11c`.
- `CLLC_Transformer.get_volume_index` (function, line 121) — **reference-only**. ID `enco-670ad65f12c0dfde`.
- `CLLC_Transformer.get_loss_index` (function, line 132) — **reference-only**. ID `enco-c671c19460e669da`.

**enco: `library-design/dcdc/cllc/src/project/electrical_transformer.py`**

- `plot` (function, line 355) — **reference-only**. ID `enco-54eb4160146e38a1`.

**enco: `library-design/dcdc/cllc/src/project/operating_space.py`**

- `Configuration` (class, line 29) — **reference-only**. ID `enco-0deae796d88bc696`.
- `Configuration.__init__` (function, line 56) — **reference-only**. ID `enco-b34246dac1bea58a`.
- `Configuration.get_hash` (function, line 79) — **reference-only**. ID `enco-d0be91ba4edffcae`.
- `OperatingPoint` (class, line 88) — **reference-only**. ID `enco-b5455ef02750f42d`.
- `OperatingPoint.__init__` (function, line 93) — **reference-only**. ID `enco-e8b9e22f4a63dc90`.
- `OperatingPoint.get_hash` (function, line 99) — **duplicate**; canonical `enco-d0be91ba4edffcae`. ID `enco-7a7491c212d14f0b`.
- `Entry` (class, line 108) — **reference-only**. ID `enco-f4aaa9a077c6aa36`.
- `Entry.get_hash` (function, line 114) — **duplicate**; canonical `enco-d0be91ba4edffcae`. ID `enco-7f02a3a6c18cbe6c`.
- `Entry.with_operating_point` (function, line 122) — **reference-only**. ID `enco-c4f40a387fdec078`.
- `Entry.with_configuration` (function, line 139) — **reference-only**. ID `enco-73129207cbff87a4`.
- `load_entries` (function, line 158) — **reference-only**. ID `enco-bd26b328b2f45513`.
- `get_nominal_entry` (function, line 164) — **reference-only**. ID `enco-3011c56fe65e5809`.

**enco: `library-design/dcdc/cllc/src/project/optimizer.py`**

- `generate_design_space` (function, line 31) — **reference-only**. ID `enco-8d17ceb8b98f13e5`.
- `generate_configuration_df` (function, line 76) — **reference-only**. ID `enco-4fe573868404ef9c`.
- `validate_entry` (function, line 97) — **reference-only**. ID `enco-68f09a87887102c6`.
- `generate_dataframe` (function, line 136) — **reference-only**. ID `enco-1fdcc0e43fd9553c`.
- `aggregate_operating_points` (function, line 292) — **reference-only**. ID `enco-030550d6d901934f`.
- `compute_merit` (function, line 310) — **reference-only**. ID `enco-1345987de071cc7d`.
- `main` (function, line 321) — **reference-only**. ID `enco-41371741cb05f21a`.

**enco: `library-design/dcdc/cllc/src/project/simulation.py`**

- `SimulationType` (class, line 33) — **reference-only**. ID `enco-332cadca6637d52b`.
- `run` (function, line 37) — **reference-only**. ID `enco-50d6c0143f2598c6`.

**enco: `library-design/dcdc/cllc_3p/src/project/analysis/maximum_power.py`**

- `compute_max_power` (function, line 32) — **duplicate**; canonical `enco-b39110fa92c85453`. ID `enco-da6c030300c9a30e`.

**enco: `library-design/dcdc/cllc_3p/src/project/main.py`**

- `main` (function, line 27) — **reference-only**. ID `enco-8279e270d87290e0`.

**enco: `library-design/dcdc/cllc_3p/src/project/optimization_electrical.py`**

- `main` (function, line 30) — **reference-only**. ID `enco-833a69257027a41d`.
- `main.pre_validation` (function, line 50) — **reference-only**. ID `enco-b4da0486a2ab1594`.
- `main.run` (function, line 72) — **reference-only**. ID `enco-ddf20742360f09fd`.
- `main.post_validation` (function, line 87) — **reference-only**. ID `enco-b03b86b9a8d1c3d7`.

**enco: `library-design/dcdc/cllc_3p/src/project/utility/simulation.py`**

- `SimulationFile` (class, line 31) — **reference-only**. ID `enco-7683b9ab421ca07d`.

**enco: `library-design/dcdc/d3ab/src/project/analysis/frequency_computation.py`**

- `calculate_required_frequency` (function, line 21) — **reference-only**. ID `enco-0a2e19bd75e15982`.

**enco: `library-design/dcdc/d3ab/src/project/analysis/maximum_power.py`**

- `compute_max_power` (function, line 32) — **reference-only**. ID `enco-222825afa21b8084`.

**enco: `library-design/dcdc/d3ab/src/project/components/transformer.py`**

- `DAB_3PH_Transformer_Model` (class, line 23) — **reference-only**. ID `enco-ec3d246ed6694373`.
- `DAB_Transformer` (class, line 35) — **reference-only**. ID `enco-9ba230ec690efb92`.
- `DAB_Transformer.validate` (function, line 39) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-f5faf65192e401bc`.
- `DAB_Transformer.train` (function, line 43) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-7d2a319b5deed3f1`.
- `DAB_Transformer.get_flux` (function, line 48) — **reference-only**. ID `enco-592361abdec1819d`.
- `DAB_Transformer.get_minimum_core_cross_section` (function, line 52) — **reference-only**. ID `enco-3f0e5e444e94dd3e`.
- `DAB_Transformer.get_minimum_wire_cross_section` (function, line 55) — **reference-only**. ID `enco-185a3aa9edef71b4`.
- `DAB_Transformer.get_minimum_al` (function, line 58) — **reference-only**. ID `enco-5438058f5494db39`.
- `DAB_Transformer.get_minimum_mlt` (function, line 63) — **reference-only**. ID `enco-192b301c247db007`.
- `DAB_Transformer.get_maximum_turns` (function, line 67) — **reference-only**. ID `enco-9336c1a9bb16222b`.
- `DAB_Transformer.get_copper_resistivity` (function, line 74) — **reference-only**. ID `enco-2c6617ee60236d44`.
- `DAB_Transformer.get_maximum_resistance` (function, line 79) — **reference-only**. ID `enco-90388093a7d01435`.
- `DAB_Transformer.get_area_product` (function, line 91) — **reference-only**. ID `enco-768cb11b3be727a4`.
- `DAB_Transformer.get_core_loss_index` (function, line 97) — **reference-only**. ID `enco-5890eb065156e5cb`.
- `DAB_Transformer.get_winding_loss_index` (function, line 105) — **reference-only**. ID `enco-dd22842dc8dd43d6`.

**enco: `library-design/dcdc/d3ab/src/project/components/transformer_materials/steinmetz_coefficients.py`**

- `load_steinmetz_data` (function, line 11) — **reference-only**. ID `enco-b47749411f19c706`.

**enco: `library-design/dcdc/d3ab/src/project/components/transformer_old.py`**

- `DAB_3PH_Transformer_Model` (class, line 24) — **reference-only**. ID `enco-0e481b074f9421a9`.
- `DAB_Transformer` (class, line 60) — **reference-only**. ID `enco-2809d3f4fece962d`.
- `DAB_Transformer.validate` (function, line 64) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-b7eec77fa48c0318`.
- `DAB_Transformer.train` (function, line 68) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-a086b1b0902b5274`.
- `DAB_Transformer.winding_loss` (function, line 72) — **reference-only**. ID `enco-1736a8057e409ff8`.
- `DAB_Transformer.core_loss` (function, line 105) — **reference-only**. ID `enco-3534ef74c6a7d78d`.
- `DAB_Transformer.get_volume_index` (function, line 136) — **reference-only**. ID `enco-c897231543dd1f39`.
- `DAB_Transformer.get_loss_index` (function, line 143) — **duplicate**; canonical `enco-c671c19460e669da`. ID `enco-5662efe2b99432b2`.
- `DAB_Transformer.check_winding_area_feasibility` (function, line 147) — **reference-only**. ID `enco-27d4f7c675a3d803`.
- `DAB_Transformer.check_core_saturation` (function, line 163) — **reference-only**. ID `enco-747fa95755fe0a09`.

**enco: `library-design/dcdc/d3ab/src/project/main.py`**

- `main` (function, line 27) — **reference-only**. ID `enco-0c5a873650f465d0`.

**enco: `library-design/dcdc/d3ab/src/project/optimization_electrical.py`**

- `main` (function, line 30) — **reference-only**. ID `enco-a9407371abbfcf69`.
- `main.pre_validation` (function, line 46) — **duplicate**; canonical `enco-b4da0486a2ab1594`. ID `enco-a75541425a81b1a4`.
- `main.run` (function, line 68) — **reference-only**. ID `enco-c920e50e6ad73938`.
- `main.post_validation` (function, line 88) — **reference-only**. ID `enco-8655e3253478853a`.

**enco: `library-design/dcdc/d3ab/src/project/utility/simulation.py`**

- `SimulationFile` (class, line 33) — **reference-only**. ID `enco-b663240411b807cb`.

**enco: `library-design/dcdc/dab_cf/src/project/analysis/maximum_power.py`**

- `compute_max_power` (function, line 28) — **reference-only**. ID `enco-809c15dea389fb95`.

**enco: `library-design/dcdc/dab_cf/src/project/components/dab_transformer.py`**

- `DAB_Transformer_Model` (class, line 24) — **reference-only**. ID `enco-2475e86c839597d6`.
- `DAB_Transformer` (class, line 45) — **reference-only**. ID `enco-e9d155800e2b44ce`.
- `DAB_Transformer.validate` (function, line 49) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-ecb571f2c0e48f9a`.
- `DAB_Transformer.train` (function, line 53) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-aa1848800d3939a9`.
- `DAB_Transformer.winding_loss` (function, line 57) — **duplicate**; canonical `enco-eebbc89e7848e95f`. ID `enco-0ef1d03e0773f936`.
- `DAB_Transformer.core_loss` (function, line 85) — **duplicate**; canonical `enco-16c63550a6dfa11c`. ID `enco-684721f65bc34176`.
- `DAB_Transformer.get_volume_index` (function, line 116) — **duplicate**; canonical `enco-670ad65f12c0dfde`. ID `enco-78a388de1c09afee`.
- `DAB_Transformer.get_loss_index` (function, line 121) — **duplicate**; canonical `enco-c671c19460e669da`. ID `enco-cc86d75942fa31c0`.

**enco: `library-design/dcdc/dab_cf/src/project/dab.py`**

- `main` (function, line 24) — **reference-only**. ID `enco-ccaf98b07ca7826b`.

**enco: `library-design/dcdc/dab_cf/src/project/operating_space.py`**

- `Configuration` (class, line 32) — **reference-only**. ID `enco-72835badf82acaca`.
- `Configuration.__init__` (function, line 58) — **reference-only**. ID `enco-318f13926886ec1e`.
- `Configuration.get_hash` (function, line 80) — **duplicate**; canonical `enco-d0be91ba4edffcae`. ID `enco-1bddd9eeebeac275`.
- `OperatingPoint` (class, line 89) — **duplicate**; canonical `enco-b5455ef02750f42d`. ID `enco-1064b075ee090bae`.
- `OperatingPoint.__init__` (function, line 96) — **duplicate**; canonical `enco-e8b9e22f4a63dc90`. ID `enco-4d066b32ba669cc8`.
- `OperatingPoint.get_hash` (function, line 102) — **duplicate**; canonical `enco-d0be91ba4edffcae`. ID `enco-7583c1c71dabd710`.
- `Entry` (class, line 111) — **duplicate**; canonical `enco-f4aaa9a077c6aa36`. ID `enco-993f439bed8a1219`.
- `Entry.get_hash` (function, line 117) — **duplicate**; canonical `enco-d0be91ba4edffcae`. ID `enco-0976fa1ec5bed75c`.
- `Entry.with_operating_point` (function, line 125) — **duplicate**; canonical `enco-c4f40a387fdec078`. ID `enco-80abb99406776392`.
- `Entry.with_configuration` (function, line 142) — **duplicate**; canonical `enco-73129207cbff87a4`. ID `enco-0ff7f9c615da2a54`.
- `load_entries` (function, line 161) — **duplicate**; canonical `enco-bd26b328b2f45513`. ID `enco-f1938d96db7f5d78`.
- `get_nominal_entry` (function, line 167) — **duplicate**; canonical `enco-3011c56fe65e5809`. ID `enco-efc89671ab1b2557`.

**enco: `library-design/dcdc/dab_cf/src/project/optimizer.py`**

- `generate_design_space` (function, line 32) — **reference-only**. ID `enco-fcbf78215d496e58`.
- `generate_configuration_df` (function, line 62) — **duplicate**; canonical `enco-4fe573868404ef9c`. ID `enco-0ee85dcc1176aa38`.
- `validate_entry` (function, line 83) — **reference-only**. ID `enco-1b20aec8bdd8eece`.
- `generate_dataframe` (function, line 140) — **reference-only**. ID `enco-4690600ceb76e95a`.
- `aggregate_operating_points` (function, line 296) — **duplicate**; canonical `enco-030550d6d901934f`. ID `enco-351bdb3d3d552764`.
- `compute_merit` (function, line 314) — **duplicate**; canonical `enco-1345987de071cc7d`. ID `enco-e0a2a8db604b2c59`.
- `main` (function, line 325) — **reference-only**. ID `enco-a9800f02522a7f16`.

**enco: `library-design/dcdc/dab_cf/src/project/simulation.py`**

- `SimulationType` (class, line 33) — **reference-only**. ID `enco-6bc3f62eb3a02b6c`.
- `run` (function, line 37) — **reference-only**. ID `enco-afd6ed5c860009f3`.

**enco: `library-design/dcdc/staedler_llc/src/project/components/llc_transformer.py`**

- `LLC_Transformer_Model` (class, line 22) — **reference-only**. ID `enco-2fe18923857850d7`.
- `OBC7_LLC_Transformer` (class, line 39) — **reference-only**. ID `enco-7f2ade6b09aa1f55`.
- `OBC7_LLC_Transformer.validate` (function, line 43) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-6a22f3a797006448`.
- `OBC7_LLC_Transformer.train` (function, line 47) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-8fafbeafb7531602`.
- `OBC7_LLC_Transformer.winding_loss` (function, line 51) — **reference-only**. ID `enco-528a6153305e680c`.
- `OBC7_LLC_Transformer.core_loss` (function, line 65) — **reference-only**. ID `enco-01c451a043dec24c`.

**enco: `library-design/dcdc/staedler_llc/src/project/llc.py`**

- `simulate_group` (function, line 28) — **reference-only**. ID `enco-a0a8e6ab83839a0a`.

**enco: `library-design/dcdc/staedler_llc/src/project/simulation.py`**

- `SimulationFile` (class, line 37) — **reference-only**. ID `enco-74d377257641f002`.

**enco: `library-design/project/OBC7/src/project/components/llc_transformer.py`**

- `LLC_Transformer_Model` (class, line 22) — **duplicate**; canonical `enco-2fe18923857850d7`. ID `enco-ccfbd23d449f7389`.
- `OBC7_LLC_Transformer` (class, line 39) — **duplicate**; canonical `enco-7f2ade6b09aa1f55`. ID `enco-5dd00c5a83a76af3`.
- `OBC7_LLC_Transformer.validate` (function, line 43) — **duplicate**; canonical `enco-b689305b9bb1cf6a`. ID `enco-5872c7123b7eb80d`.
- `OBC7_LLC_Transformer.train` (function, line 47) — **stub**; canonical `enco-a19e0c3275118534`. ID `enco-49f273ea35883a00`.
- `OBC7_LLC_Transformer.winding_loss` (function, line 51) — **duplicate**; canonical `enco-528a6153305e680c`. ID `enco-19f646a6f65fd273`.
- `OBC7_LLC_Transformer.core_loss` (function, line 65) — **duplicate**; canonical `enco-01c451a043dec24c`. ID `enco-09062640bc98879e`.

**enco: `library-design/project/OBC7/src/project/llc.py`**

- `main` (function, line 24) — **reference-only**. ID `enco-11653378998948eb`.

**enco: `library-design/project/OBC7/src/project/operating_space.py`**

- `Configuration` (class, line 34) — **reference-only**. ID `enco-ba6d0943ea5f4267`.
- `Configuration.__str__` (function, line 89) — **reference-only**. ID `enco-01c31848482b11da`.
- `Configuration.__repr__` (function, line 92) — **duplicate**; canonical `enco-5d5d53d1172083d8`. ID `enco-af75b3086d4d47d4`.
- `OperatingPoint` (class, line 96) — **reference-only**. ID `enco-590c3867df758603`.
- `OperatingPoint.__str__` (function, line 115) — **duplicate**; canonical `enco-01c31848482b11da`. ID `enco-2765ef833c4db9b0`.
- `OperatingPoint.__repr__` (function, line 118) — **duplicate**; canonical `enco-5d5d53d1172083d8`. ID `enco-242cb9a3db460361`.
- `Entry` (class, line 122) — **reference-only**. ID `enco-17798d52f4e6f9c4`.
- `Entry.__str__` (function, line 137) — **duplicate**; canonical `enco-01c31848482b11da`. ID `enco-dc7340e6304ea78f`.
- `Entry.__repr__` (function, line 140) — **duplicate**; canonical `enco-5d5d53d1172083d8`. ID `enco-56361a50c6495f97`.
- `load_entries` (function, line 147) — **reference-only**. ID `enco-cf0fa9ae1bd378e0`.
- `get_nominal_entry` (function, line 243) — **reference-only**. ID `enco-2ef0a1839aeaf8f7`.

**enco: `library-design/project/OBC7/src/project/simulation.py`**

- `SimulationType` (class, line 34) — **reference-only**. ID `enco-2271119e1a3e7d38`.
- `run` (function, line 39) — **reference-only**. ID `enco-8b282d0915daf430`.

**enco: `library-design/uncertainty/chain/src/project/main.py`**

- `adc_transfer_function` (function, line 104) — **reference-only**. ID `enco-12a1e66307f28a74`.
- `simulation_result_handler` (function, line 143) — **reference-only**. ID `enco-158a833a7fd19e8a`.
- `generate_voltage_simulations` (function, line 154) — **reference-only**. ID `enco-caf997dac06ffff0`.
- `generate_lifetime_simulations` (function, line 207) — **reference-only**. ID `enco-21d02a8b439fff37`.
- `generate_temperature_simulations` (function, line 260) — **reference-only**. ID `enco-bf0407872d73197b`.

## Current package source index

Current definitions remain in the JSON individually. This grouped index lists actual names for navigation; no browser parity is inferred.

- `analytical_models.py`: `get_package_inductance`:47 (Python-only); `GateChargeModelParams`:57 (Python-only); `GateChargeModel`:78 (Python-only); `GateChargeModel.__init__`:85 (Python-only); `GateChargeModel.calc_turn_on_times`:88 (Python-only); `GateChargeModel.calc_turn_off_times`:118 (Python-only); `IgbtModelParams`:149 (Python-only); `IgbtModel`:164 (Python-only); `IgbtModel.__init__`:172 (Python-only); `IgbtModel.calc_tail_energy`:175 (Python-only); `IgbtModel.calc_e_off_total`:187 (Python-only); `IgbtModel.calc_conduction_loss`:207 (Python-only); `HalfBridgeParams`:224 (Python-only); `TransconductanceParams`:246 (Python-only); `TransconductanceParams.calc_gm`:263 (Python-only); `ReverseRecoveryParams`:277 (Python-only); `SwitchingEnergyResult`:293 (Python-only); `SwitchingEnergyResult.to_dict`:354 (Python-only); `calc_charge_equivalent_capacitance`:367 (Python-only); `calc_charge_stored`:400 (Python-only); `calc_device_capacitances`:431 (Python-only); `extract_reverse_recovery_params`:464 (Python-only); `extract_reverse_recovery_params.equation_40`:514 (Python-only); `fit_transconductance`:550 (Python-only); `fit_transconductance.transfer_func`:602 (Python-only); `ChristenBielaModel`:633 (Python-only); `ChristenBielaModel.__init__`:647 (Python-only); `ChristenBielaModel._solve_i_oss`:673 (Python-only); `ChristenBielaModel.calc_turn_off_energy`:762 (Python-only); `ChristenBielaModel.calc_i_0_zvs`:835 (Python-only); `ChristenBielaModel._calc_reverse_recovery`:866 (Python-only); `ChristenBielaModel.calc_turn_on_energy`:945 (Python-only); `ChristenBielaModel.calc_switching_energy`:1041 (Python-only); `ChristenBielaModel.from_transistor`:1094 (Python-only); `ChristenBielaModel.from_transistor.find_closest_temp`:1117 (Python-only); `ChristenBielaModel.calc_switching_loss_curve`:1181 (Python-only)
- `backend/concrete_services.py`: `_load_data_file`:34 (Python-only); `_build_legacy_transistor`:47 (Python-only); `PlottingService`:68 (Python-only); `PlottingService.__init__`:71 (Python-only); `PlottingService.plot_channel_characteristics`:74 (Python-only); `PlottingService.plot_switching_losses`:122 (Python-only); `PlottingService.plot_safe_operating_area`:184 (Python-only); `PlottingService.plot_thermal_impedance`:218 (Python-only); `PlottingService.plot_gate_charge`:248 (Python-only); `CalculationService`:287 (Python-only); `CalculationService.calculate_channel_resistance`:290 (Python-only); `CalculationService.interpolate_switching_losses`:296 (Python-only); `CalculationService.calculate_thermal_impedance`:327 (Python-only); `CalculationService.calculate_losses`:344 (Python-only); `CalculationService.find_optimal_working_point`:382 (Python-only); `CalculationService.calculate_thermal_resistance`:392 (Python-only); `ExportService`:410 (Python-only); `ExportService.export_to_json`:413 (Python-only); `ExportService.export_to_csv`:436 (Python-only); `ExportService.export_to_spice`:463 (Python-only); `ExportService.export_to_plecs`:482 (Python-only); `ExportService.export_to_matlab`:512 (Python-only); `ExportService.export_to_ltspice`:528 (Python-only); `ExportService.export_to_gecko_circuits`:546 (Python-only); `ComparisonService`:579 (Python-only); `ComparisonService.__init__`:582 (Python-only); `ComparisonService.compare_characteristics`:586 (Python-only); `ComparisonService.rank_transistors`:595 (Python-only); `ComparisonService.compare_transistors`:604 (Python-only); `ComparisonService.generate_advanced_comparison`:629 (Python-only); `ComparisonService._plot_channel_comparison`:685 (Python-only); `ComparisonService._plot_switching_comparison`:721 (Python-only); `ComparisonService._plot_gate_charge_comparison`:773 (Python-only); `ComparisonService._plot_soa_comparison`:806 (Python-only); `ComparisonService._plot_thermal_comparison`:834 (Python-only); `ComparisonService._plot_capacitance_comparison`:865 (Python-only); `ComparisonService._plot_loss_breakdown`:904 (Python-only); `ComparisonService._plot_efficiency_comparison`:986 (Python-only); `ComparisonService.find_similar_transistors`:1043 (Python-only); `ValidationService`:1070 (Python-only); `ValidationService.validate_transistor`:1073 (Python-only); `ValidationService.validate_channel_data`:1102 (Python-only); `ValidationService.validate_switching_data`:1112 (Python-only); `ValidationService.check_data_consistency`:1120 (Python-only); `ValidationService.check_data_completeness`:1131 (Python-only); `ConcreteServiceFactory`:1163 (Python-only); `ConcreteServiceFactory.create_plotting_service`:1167 (Python-only); `ConcreteServiceFactory.create_calculation_service`:1172 (Python-only); `ConcreteServiceFactory.create_export_service`:1177 (Python-only); `ConcreteServiceFactory.create_comparison_service`:1182 (Python-only); `ConcreteServiceFactory.create_validation_service`:1187 (Python-only); `ConcreteServiceFactory.create_all_services`:1192 (Python-only)
- `catalog_fom.py`: `parse_quantity`:35 (Python-only); `metric_values`:80 (Python-only); `pareto_ids`:100 (Python-only); `load_catalog`:123 (Python-only); `screen_catalog`:128 (Python-only)
- `catalog_importer.py`: `CatalogEntry`:23 (Python-only); `CatalogEntry.calc_fom_rds_qg`:49 (Python-only); `CatalogEntry.calc_fom_rds_area`:60 (Python-only); `parse_catalog_csv`:71 (Python-only); `catalog_to_transistors`:124 (Python-only); `rank_by_fom`:158 (Python-only); `_parse_row`:182 (Python-only); `_get_str`:208 (Python-only); `_parse_float_safe`:221 (Python-only); `_parse_voltage`:240 (Python-only); `_parse_current`:250 (Python-only); `_parse_resistance`:260 (Python-only); `_parse_charge`:282 (Python-only)
- `checker_functions.py`: `check_realnum`:7 (Python-only); `check_2d_dataset`:27 (Python-only); `check_str`:52 (Python-only); `csv2array`:71 (Python-only); `check_float`:131 (Python-only); `check_keys`:148 (Python-only); `check_duplicates`:180 (Python-only)
- `core/adapters.py`: `_np`:19 (Python-only); `_list_or_none`:28 (Python-only); `_convert_foster_legacy_to_core`:41 (Python-only); `_convert_channel_legacy_to_core`:58 (Python-only); `_convert_switching_legacy_to_core`:67 (Python-only); `_convert_linearized_legacy_to_core`:90 (Python-only); `_convert_gate_charge_legacy_to_core`:101 (Python-only); `_convert_soa_legacy_to_core`:112 (Python-only); `_convert_temp_resist_legacy_to_core`:121 (Python-only); `_convert_raw_meas_legacy_to_core`:132 (Python-only); `_convert_vdc_legacy_to_core`:155 (Python-only); `_convert_eoc_legacy_to_core`:163 (Python-only); `legacy_to_core`:174 (Python-only); `_convert_foster_core_to_dict`:253 (Python-only); `_convert_channel_core_to_dict`:278 (Python-only); `_convert_switching_core_to_dict`:287 (Python-only); `_convert_linearized_core_to_dict`:319 (Python-only); `_convert_gate_charge_core_to_dict`:330 (Python-only); `_convert_soa_core_to_dict`:341 (Python-only); `_convert_temp_resist_core_to_dict`:350 (Python-only); `_convert_vdc_core_to_dict`:361 (Python-only); `_convert_eoc_core_to_dict`:369 (Python-only); `core_to_legacy_dicts`:380 (Python-only)
- `core/models.py`: `TransistorMetadata`:18 (Python-only); `ElectricalRatings`:33 (Python-only); `ThermalProperties`:43 (Python-only); `FosterThermalModel`:55 (Python-only); `FosterThermalModel.get_thermal_impedance`:71 (Python-only); `GateChargeCurve`:89 (Python-only); `SOA`:100 (Python-only); `VoltageDependentCapacitance`:111 (Python-only); `EffectiveOutputCapacitance`:119 (Python-only); `TemperatureDependResistance`:128 (Python-only); `RawMeasurementData`:139 (Python-only); `LinearizedModel`:162 (Python-only); `ChannelCharacteristics`:173 (Python-only); `ChannelCharacteristics.get_resistance_at_current`:180 (Python-only); `SwitchingLossData`:204 (Python-only); `ITransistorComponent`:232 (Python-only); `ITransistorComponent.get_channel_data`:236 (stub); `ITransistorComponent.linearize_at_operating_point`:242 (stub); `Switch`:248 (Python-only); `Switch.__init__`:251 (Python-only); `Switch.get_channel_data`:263 (Python-only); `Switch.linearize_at_operating_point`:272 (Python-only); `Diode`:321 (Python-only); `Diode.__init__`:324 (Python-only); `Diode.get_channel_data`:338 (Python-only); `Diode.linearize_at_operating_point`:344 (Python-only); `Transistor`:384 (Python-only); `Transistor.__init__`:387 (Python-only); `Transistor.working_point`:406 (Python-only); `Transistor.update_working_point`:410 (Python-only); `Transistor.get_component`:431 (Python-only)
- `core/pair_models.py`: `ConductionCurve`:17 (Python-only); `ConductionData`:26 (Python-only); `ThermalElement`:34 (Python-only); `ThermalModel`:42 (Python-only); `DeviceThermalProperties`:50 (Python-only); `DeviceMetadata`:64 (Python-only); `DeviceElectricalRatings`:79 (Python-only); `CapacitanceData`:89 (Python-only); `PairSide`:98 (Python-only); `SwitchingTestConditions`:115 (Python-only); `SwitchingEnergyData`:128 (Python-only); `SwitchingData`:139 (Python-only); `PLECSModel`:151 (Python-only); `MigrationMetadata`:160 (Python-only); `SwitchingPair`:170 (Python-only); `SwitchingPair.is_self_pair`:191 (Python-only); `SwitchingPair.is_parallel`:196 (Python-only); `SwitchingPair.devices`:201 (Python-only); `SwitchingPair.validate`:208 (Python-only); `SwitchingPair._calculate_quality_score`:272 (Python-only); `SwitchingPair.to_dict`:313 (Python-only); `SwitchingPair.from_dict`:319 (Python-only); `_parse_pair_side`:358 (Python-only); `_parse_switching_data`:431 (Python-only)
- `core/pair_repository.py`: `JsonPairRepository`:18 (Python-only); `JsonPairRepository.__init__`:21 (Python-only); `JsonPairRepository.save_pair`:26 (Python-only); `JsonPairRepository.load_pair`:39 (Python-only); `JsonPairRepository.delete_pair`:50 (Python-only); `JsonPairRepository.list_pairs`:59 (Python-only); `JsonPairRepository.search_by_device`:69 (Python-only); `JsonPairRepository.get_unique_devices`:77 (Python-only); `JsonPairRepository._build_index`:100 (Python-only); `JsonPairRepository._apply_filters`:144 (Python-only); `JsonPairRepository.count`:176 (Python-only)
- `core/repository.py`: `_load_data_file`:22 (duplicate); `_convert_energy_arrays`:35 (Python-only); `_convert_arrays_to_numpy`:47 (Python-only); `_json_dict_to_legacy_transistor`:103 (Python-only); `JsonTransistorRepository`:122 (Python-only); `JsonTransistorRepository.__init__`:125 (Python-only); `JsonTransistorRepository.get_by_name`:129 (Python-only); `JsonTransistorRepository.save`:138 (Python-only); `JsonTransistorRepository.list_all`:144 (Python-only); `JsonTransistorRepository.delete`:149 (Python-only); `JsonTransistorLoader`:158 (Python-only); `JsonTransistorLoader.load_from_json`:166 (Python-only); `JsonTransistorLoader.save_to_json`:180 (Python-only); `JsonTransistorLoader._json_serializer`:203 (Python-only); `TransistorFactory`:212 (Python-only); `TransistorFactory.create_empty_transistor`:216 (Python-only); `TransistorFactory.create_from_template`:241 (Python-only)
- `core/services.py`: `ITransistorLoader`:22 (Python-only); `ITransistorLoader.load_from_json`:26 (stub); `ITransistorLoader.save_to_json`:30 (stub); `ICalculationService`:34 (Python-only); `ICalculationService.calculate_channel_resistance`:38 (stub); `ICalculationService.interpolate_switching_losses`:44 (stub); `ICalculationService.calculate_thermal_impedance`:50 (stub); `ICalculationService.calculate_losses`:57 (stub); `ICalculationService.find_optimal_working_point`:63 (stub); `ICalculationService.calculate_thermal_resistance`:69 (stub); `IExportService`:75 (Python-only); `IExportService.export_to_plecs`:79 (stub); `IExportService.export_to_matlab`:86 (stub); `IExportService.export_to_ltspice`:92 (stub); `IExportService.export_to_gecko_circuits`:98 (stub); `IExportService.export_to_json`:104 (stub); `IExportService.export_to_csv`:110 (stub); `IExportService.export_to_spice`:116 (stub); `IValidationService`:122 (Python-only); `IValidationService.validate_transistor`:126 (stub); `IValidationService.validate_channel_data`:130 (stub); `IValidationService.validate_switching_data`:136 (stub); `IValidationService.check_data_consistency`:142 (stub); `IValidationService.check_data_completeness`:146 (stub); `IPlottingService`:150 (Python-only); `IPlottingService.plot_channel_characteristics`:154 (stub); `IPlottingService.plot_switching_losses`:163 (stub); `IPlottingService.plot_safe_operating_area`:171 (stub); `IPlottingService.plot_thermal_impedance`:175 (stub); `IPlottingService.plot_gate_charge`:179 (stub); `IComparisonService`:183 (Python-only); `IComparisonService.compare_characteristics`:187 (stub); `IComparisonService.rank_transistors`:196 (stub); `IComparisonService.compare_transistors`:205 (stub); `IComparisonService.find_similar_transistors`:211 (stub); `TransistorRepository`:218 (Python-only); `TransistorRepository.get_by_name`:222 (stub); `TransistorRepository.save`:226 (stub); `TransistorRepository.list_all`:230 (stub); `TransistorRepository.delete`:234 (stub); `TransistorService`:238 (Python-only); `TransistorService.__init__`:241 (Python-only); `TransistorService.load_transistor`:257 (Python-only); `TransistorService.validate_and_save`:268 (Python-only); `TransistorService.analyze_transistor`:276 (Python-only); `TransistorService._analyze_channel_data`:293 (Python-only); `TransistorService._analyze_switching_data`:297 (Python-only); `TransistorService._analyze_thermal_data`:301 (Python-only)
- `data_classes.py`: `_obj_to_dict`:17 (Python-only); `GateChargeCurve`:27 (Python-only); `GateChargeCurve.__init__`:36 (Python-only); `GateChargeCurve.convert_to_dict`:50 (Python-only); `GateChargeCurve.get_plots`:59 (Python-only); `SOA`:81 (Python-only); `SOA.__init__`:88 (Python-only); `SOA.convert_to_dict`:100 (Python-only); `SOA.get_plots`:109 (Python-only); `TemperatureDependResistance`:132 (Python-only); `TemperatureDependResistance.__init__`:141 (Python-only); `TemperatureDependResistance.convert_to_dict`:155 (Python-only); `TemperatureDependResistance.get_plots`:164 (Python-only); `EffectiveOutputCapacitance`:187 (Python-only); `EffectiveOutputCapacitance.__init__`:194 (Python-only); `EffectiveOutputCapacitance.convert_to_dict`:206 (Python-only); `EffectiveOutputCapacitance.collect_data`:216 (Python-only); `SwitchEnergyData`:227 (Python-only); `SwitchEnergyData.__init__`:273 (Python-only); `SwitchEnergyData.convert_to_dict`:328 (Python-only); `SwitchEnergyData.copy`:337 (Python-only); `ChannelData`:361 (Python-only); `ChannelData.__init__`:377 (Python-only); `ChannelData.convert_to_dict`:389 (Python-only); `LinearizedModel`:398 (Python-only); `LinearizedModel.__init__`:411 (Python-only); `LinearizedModel.convert_to_dict`:423 (Python-only); `VoltageDependentCapacitance`:433 (Python-only); `VoltageDependentCapacitance.__init__`:446 (Python-only); `VoltageDependentCapacitance.convert_to_dict`:457 (Python-only); `VoltageDependentCapacitance.get_plots`:466 (Python-only); `FosterThermalModel`:490 (Python-only); `FosterThermalModel.__init__`:522 (Python-only); `FosterThermalModel.convert_to_dict`:548 (Python-only); `FosterThermalModel.get_plots`:557 (Python-only); `FosterThermalModel.collect_data`:588 (Python-only); `RawMeasurementData`:604 (Python-only); `RawMeasurementData.__init__`:631 (Python-only); `RawMeasurementData.convert_to_dict`:661 (Python-only)
- `database_manager.py`: `OperationMode`:20 (Python-only); `DatabaseManager`:26 (Python-only); `DatabaseManager.__init__`:45 (Python-only); `DatabaseManager.set_operation_mode_json`:63 (Python-only); `DatabaseManager.set_operation_mode_mongodb`:102 (Python-only); `DatabaseManager.save_transistor`:118 (Python-only); `DatabaseManager.delete_transistor`:170 (Python-only); `DatabaseManager.load_transistor`:193 (Python-only); `DatabaseManager.core_repository`:218 (Python-only); `DatabaseManager.load_transistor_core`:235 (Python-only); `DatabaseManager.get_transistor_names_list`:253 (Python-only); `DatabaseManager.print_tdb`:281 (Python-only); `DatabaseManager.update_from_fileexchange`:324 (Python-only); `DatabaseManager.compare_with_fileexchange`:389 (Python-only); `DatabaseManager.export_all_datasheets`:440 (Python-only); `DatabaseManager.convert_dict_to_transistor_object`:472 (Python-only); `DatabaseManager.parallel_transistors`:573 (Python-only); `DatabaseManager.import_xml_data`:678 (Python-only); `DatabaseManager.export_single_transistor_to_json`:743 (Python-only); `DatabaseManager.dpt_save_data`:761 (Python-only)
- `diode.py`: `Diode`:16 (Python-only); `Diode.__init__`:31 (Python-only); `Diode.convert_to_dict`:143 (Python-only); `Diode.find_next_gate_voltage`:158 (Python-only); `Diode.find_approx_wp`:211 (Python-only); `Diode.plot_all_channel_data`:253 (Python-only); `Diode.plot_energy_data`:322 (Python-only); `Diode.plot_energy_data_r`:365 (Python-only); `Diode.plot_soa`:408 (Python-only); `Diode.collect_data`:418 (Python-only)
- `examples/first_example.py`: `extract_from_mongodb_to_json`:8 (Python-only); `insert_mongodb_from_json`:25 (Python-only); `example_json_database`:41 (Python-only); `example_mongodb_database`:48 (Python-only); `example_update_from_online_database`:54 (Python-only)
- `exceptions.py`: `MissingDataError`:2 (Python-only)
- `exporter.py`: `dict2matlab`:6 (Python-only)
- `frontend/interfaces.py`: `IPlotWidget`:19 (reference-only); `IPlotWidget.clear_plot`:23 (stub); `IPlotWidget.render_plot_data`:28 (stub); `IPlotWidget.set_title`:37 (stub); `IPlotWidget.set_labels`:42 (stub); `IPlotWidget.add_legend`:47 (stub); `IPlotWidget.save_plot`:52 (stub); `ITransistorView`:63 (reference-only); `ITransistorView.load_transistor`:67 (stub); `ITransistorView.get_transistor`:72 (stub); `ITransistorView.validate_input`:77 (stub); `ITransistorView.clear_view`:86 (stub); `ITransistorView.set_read_only`:91 (stub); `IMainWindow`:96 (reference-only); `IMainWindow.show_message`:100 (stub); `IMainWindow.show_progress`:110 (stub); `IMainWindow.hide_progress`:120 (stub); `IMainWindow.get_active_transistor`:125 (stub); `IMainWindow.set_active_transistor`:130 (stub); `IMainWindow.refresh_transistor_list`:135 (stub); `IDialogFactory`:140 (reference-only); `IDialogFactory.create_file_dialog`:144 (stub); `IDialogFactory.create_input_dialog`:158 (stub); `IDialogFactory.create_confirmation_dialog`:172 (stub); `PlotController`:185 (reference-only); `PlotController.__init__`:188 (reference-only); `PlotController.plot_channel_characteristics`:200 (reference-only); `PlotController.plot_switching_losses`:243 (reference-only); `PlotController.plot_safe_operating_area`:286 (reference-only); `PlotController.plot_thermal_impedance`:314 (reference-only); `PlotController.plot_gate_charge`:342 (reference-only); `PlotController.save_current_plot`:370 (reference-only); `PlotController.get_current_plot_data`:374 (reference-only); `TransistorController`:379 (reference-only); `TransistorController.__init__`:382 (reference-only); `TransistorController.load_transistor`:397 (reference-only); `TransistorController.save_transistor`:407 (reference-only); `TransistorController.calculate_losses`:433 (reference-only); `TransistorController.get_current_transistor`:446 (reference-only); `ApplicationController`:451 (reference-only); `ApplicationController.handle_error`:454 (reference-only); `ApplicationController.__init__`:458 (reference-only); `ApplicationController.register_plot_controller`:475 (reference-only); `ApplicationController.register_transistor_controller`:479 (reference-only); `ApplicationController.load_transistor_from_file`:483 (reference-only); `ApplicationController.save_transistor_to_file`:514 (reference-only); `ApplicationController.plot_transistor_characteristic`:549 (reference-only)
- `frontend/pyqt5_impl.py`: `QWidget`:23 (reference-only); `QMainWindow`:26 (reference-only); `QVBoxLayout`:29 (reference-only); `FigureCanvas`:32 (reference-only); `Figure`:35 (reference-only); `MatplotlibPlotWidget`:44 (reference-only); `MatplotlibPlotWidget.__init__`:47 (reference-only); `MatplotlibPlotWidget.clear_plot`:67 (reference-only); `MatplotlibPlotWidget.render_plot_data`:72 (reference-only); `MatplotlibPlotWidget.set_title`:96 (reference-only); `MatplotlibPlotWidget.set_labels`:101 (reference-only); `MatplotlibPlotWidget.add_legend`:107 (reference-only); `MatplotlibPlotWidget.save_plot`:117 (reference-only); `TransistorEditWidget`:127 (reference-only); `TransistorEditWidget.__init__`:130 (reference-only); `TransistorEditWidget._setup_ui`:140 (reference-only); `TransistorEditWidget._create_metadata_group`:165 (reference-only); `TransistorEditWidget._create_electrical_group`:217 (reference-only); `TransistorEditWidget._create_thermal_group`:252 (reference-only); `TransistorEditWidget.load_transistor`:279 (reference-only); `TransistorEditWidget.get_transistor`:303 (reference-only); `TransistorEditWidget.validate_input`:342 (reference-only); `TransistorEditWidget.clear_view`:367 (reference-only); `TransistorEditWidget.set_read_only`:385 (reference-only); `MainWindowImpl`:402 (reference-only); `MainWindowImpl.__init__`:405 (reference-only); `MainWindowImpl._setup_ui`:415 (reference-only); `MainWindowImpl.show_message`:433 (reference-only); `MainWindowImpl.show_progress`:447 (reference-only); `MainWindowImpl.hide_progress`:453 (reference-only); `MainWindowImpl.get_active_transistor`:458 (reference-only); `MainWindowImpl.set_active_transistor`:462 (reference-only); `MainWindowImpl.refresh_transistor_list`:467 (stub); `PyQt5DialogFactory`:473 (reference-only); `PyQt5DialogFactory.__init__`:476 (reference-only); `PyQt5DialogFactory.create_file_dialog`:483 (reference-only); `PyQt5DialogFactory.create_input_dialog`:508 (reference-only); `PyQt5DialogFactory.create_confirmation_dialog`:517 (reference-only); `create_pyqt5_implementations`:529 (reference-only)
- `generalplotsettings.py`: `global_plot_settings_font_latex`:6 (Python-only); `global_plot_settings_font_sansserif`:20 (Python-only); `update_font_size`:32 (Python-only)
- `gui/_utils.py`: `resource_path`:6 (reference-only)
- `gui/_widgets.py`: `MatplotlibWidget`:13 (reference-only); `MatplotlibWidget.__init__`:16 (reference-only); `PopOutPlotWindow`:31 (reference-only); `PopOutPlotWindow.__init__`:34 (reference-only); `ViewCurveWindow`:43 (reference-only); `ViewCurveWindow.__init__`:46 (reference-only); `ViewCurveWindow.update_curve`:60 (reference-only); `ViewCurveWindow.view_curve`:95 (reference-only); `EmittingStream`:127 (reference-only); `EmittingStream.write`:132 (reference-only)
- `gui/api_client.py`: `ApiError`:16 (reference-only); `ApiError.__init__`:19 (reference-only); `TransistorApiClient`:25 (reference-only); `TransistorApiClient.__init__`:33 (reference-only); `TransistorApiClient._url`:46 (reference-only); `TransistorApiClient._handle`:49 (reference-only); `TransistorApiClient.list_transistors`:65 (reference-only); `TransistorApiClient.get_transistor`:70 (reference-only); `TransistorApiClient.create_transistor`:77 (reference-only); `TransistorApiClient.update_transistor`:84 (reference-only); `TransistorApiClient.delete_transistor`:91 (reference-only); `TransistorApiClient.upload_json`:102 (reference-only); `TransistorApiClient.validate`:117 (reference-only); `TransistorApiClient.compare`:124 (reference-only); `TransistorApiClient.export`:137 (reference-only); `TransistorApiClient.plot_channel`:172 (reference-only); `TransistorApiClient.plot_switching`:183 (reference-only); `TransistorApiClient.plot_soa`:197 (reference-only); `TransistorApiClient.plot_thermal`:204 (reference-only); `TransistorApiClient.plot_gate_charge`:211 (reference-only); `TransistorApiClient.ping`:222 (reference-only)
- `gui/comparison_tools_functions.py`: `_get_main_window`:9 (reference-only); `new_annotation`:15 (reference-only); `click_event`:28 (reference-only); `plot_all_energy_data`:53 (reference-only); `plot_all_energy_data.clicked`:67 (reference-only); `plot_all_energy_data_r_g`:160 (reference-only); `plot_all_energy_data_r_g.clicked`:173 (duplicate); `plot_all_channel_data`:267 (reference-only); `plot_all_channel_data.clicked`:278 (duplicate); `plot_e_on`:401 (reference-only); `plot_e_on.clicked`:425 (duplicate); `plot_e_off`:541 (reference-only); `plot_e_off.clicked`:565 (duplicate); `plot_e_rr`:684 (reference-only); `plot_e_rr.clicked`:708 (duplicate); `plot_channel`:817 (reference-only); `plot_channel.clicked`:842 (duplicate); `plot_v_eoss`:996 (reference-only); `plot_v_eoss.clicked`:1008 (duplicate); `plot_v_qoss`:1040 (reference-only); `plot_v_qoss.clicked`:1053 (duplicate)
- `gui/gui.py`: `MainWindow`:31 (reference-only); `MainWindow.__init__`:50 (reference-only); `MainWindow.closeEvent`:702 (reference-only); `MainWindow.__del__`:715 (reference-only); `CurveCheckerWindow`:724 (reference-only); `CurveCheckerWindow.__init__`:731 (reference-only); `CurveCheckerWindow.save_curve`:751 (reference-only); `CurveCheckerWindow.update_curve`:813 (reference-only); `CurveCheckerWindow.run_curve_checker`:851 (reference-only); `CurveCheckerWindow.closeEvent`:885 (reference-only); `InformationWindow`:895 (reference-only); `InformationWindow.__init__`:902 (reference-only); `InformationWindow.button_yes_pressed`:910 (reference-only); `InformationWindow.button_cancel_pressed`:919 (reference-only); `InformationWindow.run_information_window`:929 (reference-only)
- `gui/mixins/comparison_mixin.py`: `ComparisonToolsMixin`:12 (reference-only); `ComparisonToolsMixin.clear_comparison_tools`:15 (reference-only); `ComparisonToolsMixin.compare_create_plot`:49 (reference-only); `ComparisonToolsMixin.compare_update_plots`:239 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot1`:273 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot2`:288 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot3`:303 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot4`:318 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot5`:333 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot6`:348 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot7`:363 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot8`:378 (reference-only); `ComparisonToolsMixin.compare_pop_out_plot9`:393 (reference-only); `ComparisonToolsMixin.comboBox_compare_transistor_changed`:408 (reference-only); `ComparisonToolsMixin.comboBox_compare_transistor1_changed`:539 (reference-only); `ComparisonToolsMixin.comboBox_compare_transistor2_changed`:554 (reference-only); `ComparisonToolsMixin.comboBox_compare_transistor3_changed`:569 (reference-only); `ComparisonToolsMixin.slider_compare_r_g_value_changed`:584 (reference-only)
- `gui/mixins/creation_mixin.py`: `_get_information_window`:16 (reference-only); `TransistorCreationMixin`:22 (reference-only); `TransistorCreationMixin.comboBox_create_transistor_switch_add_switching_losses_curve_type_changed`:25 (reference-only); `TransistorCreationMixin.comboBox_create_transistor_diode_add_switching_losses_curve_type_changed`:32 (reference-only); `TransistorCreationMixin.fill_comboBoxes_create_transistor`:39 (reference-only); `TransistorCreationMixin.clear_create_transistor`:65 (reference-only); `TransistorCreationMixin.get_all_items_text_from_comboBox`:76 (reference-only); `TransistorCreationMixin.get_all_items_data_from_comboBox`:89 (reference-only); `TransistorCreationMixin.create_transistor`:102 (reference-only); `TransistorCreationMixin.load_transistor_into_local_database`:445 (reference-only); `TransistorCreationMixin.overwrite_transistor`:480 (reference-only); `TransistorCreationMixin.preview_transistor_on_virtual_datasheet`:502 (reference-only); `TransistorCreationMixin.convert_graph_key_in_list_to_array`:521 (reference-only)
- `gui/mixins/curve_mixin.py`: `_get_curve_checker_window`:8 (reference-only); `CurveManagementMixin`:14 (reference-only); `CurveManagementMixin.add_curve_switch_channel`:17 (reference-only); `CurveManagementMixin.add_dpt_measurement_data_directory`:62 (reference-only); `CurveManagementMixin.view_dpt_measurement_data`:152 (reference-only); `CurveManagementMixin.delete_dpt_measurement_data`:180 (reference-only); `CurveManagementMixin.view_curve_switch_channel`:195 (reference-only); `CurveManagementMixin.delete_curve_switch_channel`:207 (reference-only); `CurveManagementMixin.add_curve_switch_switching_losses`:217 (reference-only); `CurveManagementMixin.view_curve_switch_switching_losses`:288 (reference-only); `CurveManagementMixin.delete_curve_switch_switching_losses`:307 (reference-only); `CurveManagementMixin.add_curve_switch_gate_charge`:313 (reference-only); `CurveManagementMixin.view_curve_switch_gate_charge`:363 (reference-only); `CurveManagementMixin.delete_curve_switch_gate_charge`:376 (reference-only); `CurveManagementMixin.add_curve_switch_r_on`:382 (reference-only); `CurveManagementMixin.view_curve_switch_r_on`:426 (reference-only); `CurveManagementMixin.delete_curve_switch_r_on`:439 (reference-only); `CurveManagementMixin.add_curve_diode_channel`:445 (reference-only); `CurveManagementMixin.view_curve_diode_channel`:482 (reference-only); `CurveManagementMixin.delete_curve_diode_channel`:495 (reference-only); `CurveManagementMixin.add_curve_diode_switching_losses`:501 (reference-only); `CurveManagementMixin.view_curve_diode_switching_losses`:569 (reference-only); `CurveManagementMixin.delete_curve_diode_switching_losses`:589 (reference-only); `CurveManagementMixin.add_curve_switch_soa_t_pulse`:595 (reference-only); `CurveManagementMixin.view_curve_switch_soa_t_pulse`:632 (reference-only); `CurveManagementMixin.delete_curve_switch_soa_t_pulse`:641 (reference-only); `CurveManagementMixin.add_curve_diode_soa_t_pulse`:647 (reference-only); `CurveManagementMixin.view_curve_diode_soa_t_pulse`:684 (reference-only); `CurveManagementMixin.delete_curve_diode_soa_t_pulse`:693 (reference-only); `CurveManagementMixin.add_curve_switch_t_rthjc`:699 (reference-only); `CurveManagementMixin.view_curve_switch_t_rthjc`:721 (reference-only); `CurveManagementMixin.delete_curve_switch_t_rthjc`:730 (reference-only); `CurveManagementMixin.add_curve_diode_t_rthjc`:735 (reference-only); `CurveManagementMixin.view_curve_diode_t_rthjc`:757 (reference-only); `CurveManagementMixin.delete_curve_diode_t_rthjc`:766 (reference-only); `CurveManagementMixin.add_curve_v_ecoss`:771 (reference-only); `CurveManagementMixin.view_curve_v_ecoss`:793 (reference-only); `CurveManagementMixin.delete_curve_v_ecoss`:802 (reference-only); `CurveManagementMixin.add_curve_c_iss_normal`:807 (reference-only); `CurveManagementMixin.view_curve_c_iss_normal`:833 (reference-only); `CurveManagementMixin.delete_curve_c_iss_normal`:842 (reference-only); `CurveManagementMixin.add_curve_c_iss_detail`:847 (reference-only); `CurveManagementMixin.view_curve_c_iss_detail`:875 (reference-only); `CurveManagementMixin.delete_curve_c_iss_detail`:884 (reference-only); `CurveManagementMixin.add_curve_c_oss_normal`:889 (reference-only); `CurveManagementMixin.view_curve_c_oss_normal`:916 (reference-only); `CurveManagementMixin.delete_curve_c_oss_normal`:925 (reference-only); `CurveManagementMixin.add_curve_c_oss_detail`:930 (reference-only); `CurveManagementMixin.view_curve_c_oss_detail`:958 (reference-only); `CurveManagementMixin.delete_curve_c_oss_detail`:967 (reference-only); `CurveManagementMixin.add_curve_c_rss_normal`:972 (reference-only); `CurveManagementMixin.view_curve_c_rss_normal`:999 (reference-only); `CurveManagementMixin.delete_curve_c_rss_normal`:1008 (reference-only); `CurveManagementMixin.add_curve_c_rss_detail`:1013 (reference-only); `CurveManagementMixin.view_curve_c_rss_detail`:1041 (reference-only); `CurveManagementMixin.delete_curve_c_rss_detail`:1050 (reference-only)
- `gui/mixins/export_mixin.py`: `ExportToolsMixin`:6 (reference-only); `ExportToolsMixin.export_datasheet`:9 (reference-only); `ExportToolsMixin.export_json`:22 (reference-only); `ExportToolsMixin.export_matlab`:36 (reference-only); `ExportToolsMixin.export_simulink`:52 (reference-only); `ExportToolsMixin.export_plecs`:76 (reference-only); `ExportToolsMixin.export_gecko`:98 (reference-only)
- `gui/mixins/search_mixin.py`: `SearchDatabaseMixin`:9 (reference-only); `SearchDatabaseMixin.update_database_from_fileexchange`:12 (reference-only); `SearchDatabaseMixin.reset_filter_search_database`:22 (reference-only); `SearchDatabaseMixin.get_marked_transistor`:32 (reference-only); `SearchDatabaseMixin.delete_marked_transistor_search_database_from_local_tdb`:51 (reference-only); `SearchDatabaseMixin.load_from_search_database_into_create_transistor`:57 (reference-only); `SearchDatabaseMixin.load_from_search_database_into_exporting_tools`:391 (reference-only); `SearchDatabaseMixin.load_from_search_database_into_comparison_tools`:409 (reference-only); `SearchDatabaseMixin.load_from_search_database_into_topology_calculator`:438 (reference-only); `SearchDatabaseMixin.check_value_is_in_between`:465 (reference-only); `SearchDatabaseMixin.search_database_load_data`:472 (reference-only)
- `gui/mixins/settings_mixin.py`: `SettingsMixin`:10 (reference-only); `SettingsMixin.get_settings_dict`:13 (reference-only); `SettingsMixin.export_settings`:220 (reference-only); `SettingsMixin.save_settings`:236 (reference-only); `SettingsMixin.load_settings`:248 (reference-only); `SettingsMixin.load_local_settings`:264 (reference-only); `SettingsMixin.set_all_settings`:277 (reference-only)
- `gui/mixins/topology_mixin.py`: `TopologyCalculatorMixin`:20 (reference-only); `TopologyCalculatorMixin.clear_topology_calculator`:23 (reference-only); `TopologyCalculatorMixin.comboBox_topology_topology_changed`:53 (reference-only); `TopologyCalculatorMixin.comboBox_topology_plot_line_contour_changed`:69 (reference-only); `TopologyCalculatorMixin.comboBox_topology_plots_line_contour_changed`:87 (reference-only); `TopologyCalculatorMixin.comboBox_topology_plot_x_axis_changed`:112 (reference-only); `TopologyCalculatorMixin.comboBox_topology_plots_x_axis_changed`:159 (reference-only); `TopologyCalculatorMixin.comboBox_topology_transistor_changed`:190 (reference-only); `TopologyCalculatorMixin.comboBox_topology_transistor1_changed`:279 (reference-only); `TopologyCalculatorMixin.slider_topology_r_g_value_changed`:293 (reference-only); `TopologyCalculatorMixin.new_annotation`:304 (reference-only); `TopologyCalculatorMixin.click_event`:316 (reference-only); `TopologyCalculatorMixin.topology_create_plot`:339 (reference-only); `TopologyCalculatorMixin.topology_create_plot.clicked`:358 (reference-only); `TopologyCalculatorMixin.topology_update_plots`:872 (reference-only); `TopologyCalculatorMixin.get_converter`:928 (reference-only); `TopologyCalculatorMixin.topology_pop_out_plot1`:943 (reference-only); `TopologyCalculatorMixin.topology_pop_out_plot2`:963 (reference-only); `TopologyCalculatorMixin.topology_pop_out_plot3`:983 (reference-only); `TopologyCalculatorMixin.topology_pop_out_plot4`:1003 (reference-only); `TopologyCalculatorMixin.topology_pop_out_plot5`:1023 (reference-only); `TopologyCalculatorMixin.topology_pop_out_plot6`:1043 (reference-only)
- `gui/mixins/utils_mixin.py`: `UtilitiesMixin`:10 (reference-only); `UtilitiesMixin.show_popup_message`:13 (reference-only); `UtilitiesMixin.browse_file_csv`:27 (reference-only); `UtilitiesMixin.webbrowser_original_datasheet`:36 (reference-only); `UtilitiesMixin.webbrowser_virtual_datasheet`:41 (reference-only); `UtilitiesMixin.email_add_transistor_to_transistordatabase_file_exchange`:50 (reference-only); `UtilitiesMixin.webbrowser_contribute`:71 (reference-only); `UtilitiesMixin.webbrowser_bugreport`:75 (reference-only); `UtilitiesMixin.webbrowser_documentation`:79 (reference-only); `UtilitiesMixin.standard_output_written`:83 (reference-only); `UtilitiesMixin.show_stdout`:91 (reference-only)
- `gui_web/api/main.py`: `health_check`:23 (Python-only); `root`:29 (Python-only)
- `gui_web/api/topology.py`: `calculate_topology`:11 (Python-only); `calculate_buck_converter`:29 (Python-only); `calculate_boost_converter`:64 (Python-only); `calculate_buck_boost_converter`:97 (Python-only)
- `gui_web/api/transistors.py`: `get_transistors`:55 (Python-only); `get_transistor`:60 (Python-only); `search_transistors`:67 (Python-only); `upload_transistor`:93 (Python-only); `export_transistors`:115 (Python-only)
- `gui_web/backend/catalog_routes.py`: `catalog`:16 (Python-only); `ScreeningRequest`:21 (Python-only); `sources`:39 (Python-only); `screen`:61 (Python-only)
- `gui_web/backend/main.py`: `_get_transistor`:112 (Python-only); `transistor_to_dict`:124 (Python-only); `dict_to_transistor`:168 (Python-only); `root`:229 (Python-only); `get_transistors`:235 (Python-only); `get_transistor`:247 (Python-only); `create_transistor`:253 (Python-only); `update_transistor`:264 (Python-only); `delete_transistor`:276 (Python-only); `validate_transistor`:288 (Python-only); `compare_transistors`:295 (Python-only); `advanced_comparison`:304 (Python-only); `export_transistor`:338 (Python-only); `export_preview`:401 (Python-only); `batch_export`:483 (Python-only); `upload_transistor`:581 (Python-only); `plot_channel`:613 (Python-only); `plot_switching`:623 (Python-only); `plot_soa`:634 (Python-only); `plot_thermal`:641 (Python-only); `plot_gate_charge`:648 (Python-only); `get_all_curves`:659 (Python-only); `get_all_curves.array_to_list`:663 (Python-only); `add_channel_curve`:769 (Python-only); `add_switching_loss_curve`:811 (Python-only); `add_gate_charge_curve`:880 (Python-only); `add_soa_curve`:909 (Python-only); `add_capacitance_curve`:936 (Python-only); `delete_curve`:980 (Python-only); `validate_curves`:1040 (Python-only); `calculate_topology`:1084 (Python-only); `compare_topologies`:1110 (Python-only); `topology_recommendations`:1133 (Python-only); `get_user_settings`:1195 (Python-only); `update_user_settings`:1203 (Python-only); `reset_user_settings`:1210 (Python-only); `advanced_search`:1217 (Python-only); `add_favorite`:1281 (Python-only); `remove_favorite`:1291 (Python-only); `list_favorites`:1301 (Python-only); `import_plecs`:1311 (Python-only); `calculate_biela`:1360 (broken-adapter); `calculate_gate_charge`:1415 (Python-only); `generate_dpt_netlist`:1466 (broken-adapter); `get_dpt_data`:1510 (Python-only); `validate_dpt_data`:1540 (Python-only); `list_pairs`:1613 (Python-only); `get_pair`:1668 (Python-only); `validate_pair`:1682 (Python-only); `list_unique_devices`:1695 (Python-only); `get_device_pairs`:1718 (Python-only); `export_pairs_csv`:1728 (Python-only); `_frontend_dist`:1809 (Python-only)
- `gui_web/backend/main_simple.py`: `load_example_data`:31 (Python-only); `normalize_transistor_data`:44 (Python-only); `startup_event`:76 (Python-only); `root`:82 (duplicate); `get_transistors`:88 (Python-only); `get_transistor`:94 (Python-only); `create_transistor`:102 (Python-only); `update_transistor`:114 (Python-only); `delete_transistor`:128 (Python-only); `upload_transistor`:138 (Python-only); `compare_transistors`:167 (Python-only); `export_transistor`:190 (Python-only)
- `gui_web/backend/schemas.py`: `ChannelCurveCreate`:14 (Python-only); `ChannelCurveCreate.check_length_match`:24 (Python-only); `ChannelCurveCreate.check_temperature_range`:38 (Python-only); `SwitchingLossCreate`:45 (Python-only); `SwitchingLossCreate.check_dataset_type`:84 (Python-only); `GateChargeCurveCreate`:92 (Python-only); `GateChargeCurveCreate.check_length_match`:104 (Python-only); `SOACreate`:115 (Python-only); `SOACreate.check_length_match`:125 (Python-only); `CapacitanceCurveCreate`:136 (Python-only); `CapacitanceCurveCreate.check_length_match`:145 (Python-only); `CurveValidationResult`:156 (Python-only); `ExportOptions`:165 (Python-only); `PLECSExportOptions`:172 (Python-only); `MATLABExportOptions`:184 (Python-only); `BatchExportRequest`:191 (Python-only); `TopologySweepRequest`:199 (Python-only); `TopologySweepRequest.validate_axis_ranges`:217 (Python-only); `TopologyCalculationRequest`:229 (Python-only); `AdvancedSearchRequest`:272 (Python-only); `UserSettings`:285 (Python-only); `BielaModelRequest`:302 (Python-only); `GateChargeModelRequest`:324 (Python-only); `DPTConfigRequest`:338 (Python-only); `PLECSImportResponse`:352 (Python-only)
- `gui_web/backend/topology_service.py`: `TopologyCalculatorService`:14 (Python-only); `TopologyCalculatorService.calculate`:20 (Python-only); `TopologyCalculatorService.compare_topologies`:48 (Python-only)
- `gui_web/tests/conftest.py`: `test_data_dir`:12 (reference-only); `sample_transistor_flat`:17 (reference-only); `sample_transistor_nested`:53 (reference-only); `multiple_transistors_data`:78 (reference-only)
- `gui_web/tests/debug_upload.py`: `test_upload_endpoint`:10 (reference-only); `test_minimal_upload`:124 (reference-only)
- `gui_web/tests/run_tests.py`: `run_tests`:12 (reference-only); `run_specific_test`:96 (reference-only); `check_dependencies`:110 (reference-only)
- `gui_web/tests/test_file_upload.py`: `TestFileUpload`:24 (reference-only); `TestFileUpload.setup_method`:27 (reference-only); `TestFileUpload.test_upload_cree_format`:31 (reference-only); `TestFileUpload.test_upload_minimal_format`:113 (reference-only); `TestFileUpload.test_upload_with_t_c_max_fallback`:146 (reference-only); `TestFileUpload.test_upload_large_file`:175 (reference-only); `TestFileUpload.test_upload_error_handling`:235 (reference-only); `TestFileUpload.test_upload_duplicate_handling`:291 (reference-only); `TestFileUploadIntegration`:330 (reference-only); `TestFileUploadIntegration.setup_method`:333 (duplicate); `TestFileUploadIntegration.test_upload_then_operations`:337 (reference-only)
- `gui_web/tests/test_main.py`: `TestMainFunctions`:25 (reference-only); `TestMainFunctions.setup_method`:28 (duplicate); `TestMainFunctions.test_dict_to_transistor_nested_format`:33 (reference-only); `TestMainFunctions.test_dict_to_transistor_flat_format`:73 (reference-only); `TestMainFunctions.test_dict_to_transistor_flat_format_with_missing_fields`:103 (reference-only); `TestMainFunctions.test_dict_to_transistor_with_t_c_max_fallback`:128 (reference-only); `TestMainFunctions.test_transistor_to_dict`:144 (reference-only); `TestAPIEndpoints`:179 (reference-only); `TestAPIEndpoints.setup_method`:182 (duplicate); `TestAPIEndpoints.test_root_endpoint`:187 (reference-only); `TestAPIEndpoints.test_get_transistors_empty`:195 (reference-only); `TestAPIEndpoints.test_get_transistors_with_data`:201 (reference-only); `TestAPIEndpoints.test_get_transistor_by_id_existing`:235 (reference-only); `TestAPIEndpoints.test_get_transistor_by_id_not_found`:259 (reference-only); `TestAPIEndpoints.test_create_transistor_success`:265 (reference-only); `TestAPIEndpoints.test_create_transistor_invalid_data`:298 (reference-only); `TestAPIEndpoints.test_update_transistor_success`:316 (reference-only); `TestAPIEndpoints.test_update_transistor_not_found`:365 (reference-only); `TestAPIEndpoints.test_delete_transistor_success`:376 (reference-only); `TestAPIEndpoints.test_delete_transistor_not_found`:397 (reference-only); `TestAPIEndpoints.test_upload_transistor_success`:402 (reference-only); `TestAPIEndpoints.test_upload_transistor_invalid_file_type`:448 (reference-only); `TestAPIEndpoints.test_upload_transistor_invalid_json`:467 (reference-only); `TestAPIEndpoints.test_compare_transistors_success`:486 (reference-only); `TestAPIEndpoints.test_compare_transistors_insufficient_devices`:512 (reference-only); `TestAPIEndpoints.test_compare_transistors_not_found`:521 (reference-only); `TestAPIEndpoints.test_validate_transistor`:526 (reference-only); `TestAPIEndpoints.test_export_transistor_json`:542 (reference-only); `TestAPIEndpoints.test_export_transistor_invalid_format`:558 (reference-only); `TestIntegration`:568 (reference-only); `TestIntegration.setup_method`:571 (duplicate); `TestIntegration.test_full_transistor_lifecycle`:575 (reference-only); `TestIntegration.test_upload_and_use_workflow`:628 (reference-only)
- `helper_functions.py`: `isvalid_transistor_name`:24 (Python-only); `isvalid_dict`:28 (Python-only); `matlab_compatibility_test`:245 (Python-only); `get_xml_data`:272 (Python-only); `read_data_file`:375 (Python-only); `html_to_pdf`:396 (Python-only); `get_vc_plots`:416 (Python-only); `compare_plot`:439 (Python-only); `get_gatedefaults`:500 (Python-only); `negate_and_append`:517 (Python-only); `get_loss_curves`:544 (Python-only); `get_channel_data`:586 (Python-only); `gen_exp_func`:624 (Python-only); `merge_curve`:647 (Python-only); `compare_list`:681 (Python-only); `get_copy_transistor_name`:688 (Python-only); `get_img_raw_data`:705 (Python-only)
- `helper_pdf.py`: `html_to_pdf`:13 (Python-only); `html_to_pdf.fetch_next`:34 (Python-only); `html_to_pdf.handle_print_finished`:44 (Python-only); `html_to_pdf.handle_load_finished`:50 (Python-only)
- `mongodb_handling.py`: `connect_tdb`:6 (Python-only); `connect_local_tdb`:21 (Python-only); `drop_local_tdb`:47 (Python-only); `MissingServerConnection`:65 (Python-only)
- `plecs_importer.py`: `import_plecs_xml`:28 (Python-only); `_parse_package`:60 (Python-only); `_parse_conduction_loss`:134 (Python-only); `_parse_switching_loss`:178 (Python-only); `_extract_rg`:243 (Python-only); `_parse_thermal_model`:261 (Python-only); `_parse_float_list`:295 (Python-only); `_map_plecs_type`:306 (Python-only); `_is_diode_type`:322 (Python-only); `_infer_electrical_ratings`:331 (Python-only)
- `plot_functions.py`: `plot_soa_lib`:5 (Python-only)
- `rg_formula.py`: `interpolate_energy_at_rg`:16 (Python-only); `scale_energy_by_rg`:64 (Python-only); `find_rg_for_target_energy`:97 (Python-only); `get_available_rg_values`:160 (Python-only)
- `scripts/tdb_backend.py`: `main`:5 (Python-only)
- `scripts/tdb_pyqt.py`: `main`:7 (Python-only)
- `scripts/tdb_webgui.py`: `_open_browser`:8 (Python-only); `main`:13 (Python-only)
- `switch.py`: `append_valid_datasets`:18 (Python-only); `Switch`:44 (Python-only); `Switch.__init__`:68 (Python-only); `Switch.convert_to_dict`:130 (Python-only); `Switch.find_next_gate_voltage`:150 (Python-only); `Switch.find_approx_wp`:227 (Python-only); `Switch.plot_channel_data_vge`:278 (Python-only); `Switch.plot_channel_data_temp`:301 (Python-only); `Switch.plot_all_channel_data`:324 (Python-only); `Switch.plot_energy_data`:394 (Python-only); `Switch.plot_energy_data_r`:440 (Python-only); `Switch.plot_energy_data_t`:484 (Python-only); `Switch.plot_all_on_resistance_curves`:528 (Python-only); `Switch.plot_all_charge_curves`:561 (Python-only); `Switch.plot_soa`:597 (Python-only); `Switch.collect_data`:607 (Python-only)
- `topologies/converter_common.py`: `ConverterBase`:13 (Python-only); `ConverterBase.duty_cycle_ccm_mesh`:26 (stub); `ConverterBase.duty_cycle_dcm1_mesh`:31 (stub); `ConverterBase.duty_cycle_dcm2_mesh`:36 (stub); `ConverterBase.inductor_voltage`:41 (stub); `ConverterBase.blocking_voltage`:46 (stub); `ConverterBase.f_m_calc_channel`:54 (Python-only); `ConverterBase.f_m_i_peak`:110 (Python-only); `ConverterBase.f_m_i1_rms`:138 (Python-only); `ConverterBase.f_m_i1_mean`:166 (Python-only); `ConverterBase.f_m_i2_rms`:194 (Python-only); `ConverterBase.f_m_i2_mean`:223 (Python-only); `ConverterBase.f_m_i_l_rms`:252 (Python-only); `ConverterBase.f_m_i_l_mean`:283 (Python-only); `ConverterBase.f_m_conduction_losses1`:291 (Python-only); `ConverterBase.f_m_conduction_losses2`:301 (Python-only); `ConverterBase.f_m_p_on1`:313 (Python-only); `ConverterBase.f_m_p_off1`:350 (Python-only); `ConverterBase.f_m_p_rr2`:374 (Python-only); `ConverterBase.f_m_conduction_losses`:414 (Python-only); `ConverterBase.f_m_p_on_off1`:419 (Python-only); `ConverterBase.f_m_p_on_off_rr_1_2`:424 (Python-only); `ConverterBase.f_m_p1`:429 (Python-only); `ConverterBase.f_m_p2`:434 (Python-only); `ConverterBase.f_m_t_switch1`:439 (Python-only); `ConverterBase.f_m_t_diode2`:447 (Python-only); `ConverterBase.f_vec_calc_channel`:460 (Python-only); `ConverterBase.f_vec_i_peak`:517 (Python-only); `ConverterBase.f_vec_i1_rms`:546 (Python-only); `ConverterBase.f_vec_i1_mean`:580 (Python-only); `ConverterBase.f_vec_i2_rms`:613 (Python-only); `ConverterBase.f_vec_i2_mean`:648 (Python-only); `ConverterBase.f_vec_i_l_rms`:683 (Python-only); `ConverterBase.f_vec_i_l_mean`:721 (Python-only); `ConverterBase.f_vec_conduction_losses1`:734 (Python-only); `ConverterBase.f_vec_conduction_losses2`:751 (Python-only); `ConverterBase.f_vec_p_on1`:770 (Python-only); `ConverterBase.f_vec_p_off1`:816 (Python-only); `ConverterBase.f_vec_p_rr2`:846 (Python-only); `ConverterBase.f_vec_conduction_losses`:896 (Python-only); `ConverterBase.f_vec_p_on_off1`:901 (Python-only); `ConverterBase.f_vec_p_on_off_rr_1_2`:906 (Python-only); `ConverterBase.f_vec_p1`:911 (Python-only); `ConverterBase.f_vec_p2`:916 (Python-only); `ConverterBase.f_vec_t_switch1`:921 (Python-only); `ConverterBase.f_vec_t_diode2`:930 (Python-only); `BoostConverter`:945 (Python-only); `BoostConverter.duty_cycle_ccm_mesh`:949 (Python-only); `BoostConverter.duty_cycle_dcm1_mesh`:954 (Python-only); `BoostConverter.duty_cycle_dcm2_mesh`:960 (Python-only); `BoostConverter.inductor_voltage`:965 (Python-only); `BoostConverter.blocking_voltage`:970 (Python-only); `BuckConverter`:975 (Python-only); `BuckConverter.duty_cycle_ccm_mesh`:979 (Python-only); `BuckConverter.duty_cycle_dcm1_mesh`:984 (Python-only); `BuckConverter.duty_cycle_dcm2_mesh`:990 (Python-only); `BuckConverter.inductor_voltage`:995 (Python-only); `BuckConverter.blocking_voltage`:1000 (Python-only); `BuckBoostConverter`:1005 (Python-only); `BuckBoostConverter.duty_cycle_ccm_mesh`:1009 (Python-only); `BuckBoostConverter.duty_cycle_dcm1_mesh`:1014 (Python-only); `BuckBoostConverter.duty_cycle_dcm2_mesh`:1020 (Python-only); `BuckBoostConverter.inductor_voltage`:1025 (Python-only); `BuckBoostConverter.blocking_voltage`:1030 (Python-only)
- `topologies/operating_point.py`: `_finite_positive`:42 (Python-only); `_optional_finite`:52 (Python-only); `_same_condition`:64 (Python-only); `_curve_arrays`:70 (Python-only); `_interpolate_curve`:88 (Python-only); `_channel_curve`:98 (Python-only); `_switching_curve`:129 (Python-only); `_energy_at_current`:155 (Python-only); `_mean_conduction_power`:175 (Python-only); `_dcm_output_ripple`:186 (Python-only); `_temperature_result`:221 (Python-only); `calculate_topology_operating_point`:257 (Python-only); `calculate_topology_operating_point.commutation_device_status`:581 (Python-only); `calculate_topology_sweep`:678 (Python-only); `calculate_topology_sweep.evaluate`:714 (Python-only)
- `transistor.py`: `Transistor`:44 (Python-only); `Transistor.__init__`:111 (Python-only); `Transistor.convert_raw_measurement_data`:229 (Python-only); `Transistor.convert_voltage_dependent_capacitance`:265 (Python-only); `Transistor.__eq__`:301 (Python-only); `Transistor.__repr__`:327 (Python-only); `Transistor.convert_to_dict`:331 (Python-only); `Transistor.update_wp`:358 (Python-only); `Transistor.init_loss_matrices`:413 (Python-only); `Transistor.init_switch_channel_matrix`:417 (Python-only); `Transistor.quickstart_wp`:464 (Python-only); `Transistor.calc_v_eoss`:477 (Python-only); `Transistor.calc_v_qoss`:489 (Python-only); `Transistor.plot_v_eoss`:501 (Python-only); `Transistor.plot_v_qoss`:521 (Python-only); `Transistor.plot_v_coss`:542 (Python-only); `Transistor.plot_half_bridge_equivalent_coss`:562 (Python-only); `Transistor.plot_half_bridge_equivalent_eoss`:607 (Python-only); `Transistor.calc_energy_object_voltage_correction`:673 (Python-only); `Transistor.calc_real_on_off_loss`:688 (Python-only); `Transistor.plot_energy_objects`:717 (Python-only); `Transistor.get_object_v_i`:757 (Python-only); `Transistor.get_object_i_e`:820 (Python-only); `Transistor.get_object_i_e_simplified`:885 (Python-only); `Transistor.get_object_r_e_simplified`:941 (Python-only); `Transistor.calc_object_i_e`:992 (Python-only); `Transistor.calc_i_e_curve_using_r_e_curve`:1050 (Python-only); `Transistor.calc_lin_channel`:1083 (Python-only); `Transistor.calc_thermal_params`:1187 (Python-only); `Transistor.calc_thermal_params.upper_limit`:1235 (Python-only); `Transistor.compare_channel_linearized`:1281 (Python-only); `Transistor.raw_measurement_data_plots`:1345 (Python-only); `Transistor.plot_curves`:1390 (Python-only); `Transistor.export_datasheet`:1416 (Python-only); `Transistor.export_simulink_loss_model`:1476 (Python-only); `Transistor.export_matlab`:1667 (Python-only); `Transistor.collect_i_e_and_r_e_combination`:1692 (Python-only); `Transistor.export_geckocircuits`:1726 (Python-only); `Transistor.export_geckocircuits_coss`:2040 (Python-only); `Transistor.export_plecs`:2083 (Python-only); `Transistor.WP`:2132 (Python-only); `Transistor.WP.__init__`:2162 (Python-only); `Transistor.validate_transistor`:2177 (Python-only); `Transistor.get_curve_data`:2207 (Python-only); `Transistor.add_dpt_measurement`:2371 (Python-only); `Transistor.add_soa_data`:2438 (Python-only); `Transistor.add_gate_charge_data`:2503 (Python-only); `Transistor.add_temp_depend_resistor_data`:2556 (Python-only); `Transistor.compare_measurement_datasheet`:2609 (Python-only); `attach_units`:2666 (Python-only)
- `utils/ltspice_dpt.py`: `DPTConfig`:31 (Python-only); `DPTConfig.t_pulse1`:56 (Python-only); `DPTResult`:70 (Python-only); `LTspiceDPT`:93 (Python-only); `LTspiceDPT.__init__`:101 (Python-only); `LTspiceDPT.generate_netlist`:108 (Python-only); `LTspiceDPT.run_simulation`:160 (Python-only); `LTspiceDPT.analyze_waveforms`:180 (Python-only); `LTspiceDPT.extract_switching_loss_data`:314 (Python-only)
- `waveform_losses.py`: `WaveformData`:25 (Python-only); `LossResult`:41 (Python-only); `calc_conduction_loss`:60 (Python-only); `calc_conduction_loss_from_channel`:93 (Python-only); `detect_switching_edges`:124 (Python-only); `detect_zvs`:152 (Python-only); `calc_switching_loss`:185 (Python-only); `calc_total_loss`:238 (Python-only); `_interpolate_energy`:281 (Python-only)

## Completion rule

Update a row only with an exact current callable, independent numerical fixture, API equality evidence where exposed, and exercised browser behavior where required by the plan. Re-run this generator for source inventories; preserve subsequent execution evidence in a separately reviewed overlay before regeneration. This baseline does not overwrite the existing FOM report or count its prior aggregate test totals as proof for unrelated functions.
