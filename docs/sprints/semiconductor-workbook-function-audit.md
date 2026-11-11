# Semiconductor workbook function audit

Audited 2026-10-16. This is a read-only census of all five files named by the user, including sheets omitted from the catalog import. ZIP/XML inspection and bundled Python/openpyxl reads did not execute Excel, VBA, Power Query, formulas, or reference-project code. Cached results below are stored source values, not recalculation evidence. No reference file was changed.

## Sources and freshness

The exact source directory is `\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc\db`.

| Exact filename in that directory | Bytes | SHA-256 | Evidence of age |
| --- | ---: | --- | --- |
| `201210_sic_digikey.xlsx` | 64,893 | `aea21eb6f1f44267f8c454393be63d052f5f4d0b9b6cc6183151edd4df7c2581` | Filename suggests 2020-12-10. `docProps/core.xml` says created 2023-11-24T13:09:10Z, modified 2023-11-24T13:09:21Z; these document timestamps do not establish quote acquisition time. |
| `SiC_MOSFET_FOM.xlsx` | 419,663 | `1470636e91595cf4f0a48a8a30fe50e3558710cde3d4b7bf5abac59bd99bc641` | Core metadata created 2020-09-11T11:44:46Z, modified 2020-12-09T23:20:01Z. |
| `201210_mos_digikey.csv` | 267,820 | `a56d5f7319caa70f1b2b43954c60111a29359f800810e9db16f6e745c7f5ad4f` | Filename suggests 2020-12-10; CSV has no workbook creation or quote timestamp. |
| `201210_gan_digikey.csv` | 50,064 | `7d97978c7015b45f4fdfe4f1dbfb1bc75ea44547b4fbb5321231f6221cece28f` | Same filename-only dating limitation. |
| `SemiconductorsData.xlsm` | 57,237 | `f7b4c0424f1329c469dbffd96b4ab44fe6ac448ccb7ff0e6f331159c8a0b3ab0` | Core metadata created 2006-09-16T00:00:00Z, modified 2017-11-21T23:18:16Z. |

These are historical prices and ratings, not current purchasing information. The CHF currency is explicit in `SiC_MOSFET_FOM.xlsx` cell I1 on its three catalog sheets. `Price` in the English exports and `Cost at 1k pcs.` in the manual workbook do not establish currency.

All five hashes match the per-source hashes currently stored in `transistordatabase/data/catalogs/archive_sc.json`. Its eight imported sources contain 2,490 part records: 286 + 174 + 1,137 + 233 + 500 + 99 + 23 + 38. Summary rows without part identifiers are additional evidence, not additional catalog devices.

## Complete sheet and native-feature census

Formula-cell counts include shared-formula followers. Sheet dimensions describe stored cells, not usable device counts.

| Workbook | Sheet | Stored dimension | Formula cells, by column | Role |
| --- | --- | --- | --- | --- |
| `201210_sic_digikey.xlsx` | `in` | A1:AD287 | 0 | 286-row English catalog export. |
| `SiC_MOSFET_FOM.xlsx` | `1200V` | A1:AF175 | U:174, AF:174 = **348** | Catalog, unit parsing, cost FOM. |
| Same | `600V` | A1:AF1138 | U:1,137, AF:1,137 = **2,274** | Same, with row-specific voltage text parsing. |
| Same | `Tabelle8` | A1:AG234 | R:233, V:233, AG:233 = **699** | Mixed-voltage catalog and cost FOM. |
| Same | `Tabelle7` | A1 | 0 | Empty worksheet. |
| Same | `Tabelle1` | A1:C311 | C:310 = **310** | Summary, minimum-normalized relative cost and scatter-chart inputs. |
| `SemiconductorsData.xlsm` | `MOSFET` | A1:BA25 | R:10, S:9, V:10, W:9, Y:12, Z:12, AM:1, AR:23, AT:24, AU:24, AV:17, AW:17, AX:16, AY:17, AZ:12, BA:11 = **224** | Device/condition data and derived screening estimates. |
| Same | `IGBT` | A1:BB46 | W:19, Y:25, Z:25, AA:6, AD:5, AE:2, AH:24, AI:6, AR:40, AT:40, AU:40, AV:28, AW:25, AX:20, AY:29, AZ:21, BA:4, BB:2 = **361** | Threshold-plus-resistance device estimates and comparison metrics. |
| Same | `Diode` | A1 | 0 | Empty worksheet. |

The two CSV files contain 501 and 100 records including their header, with 30 fields per header. No CSV field starts with `=`. CSV does not contain workbook-native sheets, charts, names, pivots or VBA.

There are **4,216 formula cells** across the three OOXML workbooks: 3,631 in the FOM workbook and 585 in the XLSM. All nine worksheets have visible sheet state. Separately, the FOM workbook stores 1,077 hidden rows on `600V`, and hidden columns J:T on `1200V`. These are visibility choices, not new algorithms; a migration must not silently discard hidden records.

Native feature inventory, based on every ZIP member plus workbook/sheet XML:

- No PivotTable/pivot-cache parts, worksheet data validations, conditional-format rules, or `xl/externalLinks/` workbook-link parts occur in these three workbooks. External data connections do exist as described below; their absence from `externalLinks` is not absence of external dependencies.
- `SiC_MOSFET_FOM.xlsx` has three query-backed Excel tables: `download_3` at `1200V!A1:AF175`, `download_6` at `600V!A1:AF1138`, and `download_7` at `Tabelle8!A1:AG234`. Each table has an AutoFilter and calculated columns. `SemiconductorsData.xlsm` instead has a worksheet AutoFilter at `MOSFET!A1:BA1`, which covers the header only.
- The FOM workbook has three hidden, sheet-local names: `ExterneDaten_1` refers to `'1200V'!$A$1:$AE$175`; `ExterneDaten_3` to `'600V'!$A$1:$AE$1138`; another `ExterneDaten_1` to `Tabelle8!$A$1:$AF$234`. These imported-data ranges omit the final added FOM column. The XLSM has `_xlnm._FilterDatabase` at `MOSFET!$A$1:$BA$1`. The plain SiC export has no defined names.
- The FOM workbook has **one scatter chart**, `xl/charts/chart1.xml`, using series label `Tabelle1!$C$1`, x-values `Tabelle1!$A$2:$A$1047499`, and y-values `Tabelle1!$C$2:$C$1047499`. Its axes are “Blocking Voltage (V)” and “Relative Semiconductor Cost”; y is base-10 logarithmic, maximum 10. Actual summary data occupy only rows 2:311. This is a relative-cost plot, not an implemented Pareto-front calculation.
- No chart parts occur in the other two workbooks. The XLSM has 23 cell hyperlinks on `MOSFET` and 36 on `IGBT`; its `GetURL` formulas are separate from those hyperlink objects.

## Calculation families and exact examples

Cells and formulas in this table are from the named reference workbooks. Repeated rows with different literal device values belong to the same family; they are not independent functions. “Gap” means absent from the current catalog calculation contract, not proof that no unrelated Python helper exists elsewhere.

| Workflow | Source evidence and exact formula example | Conditions and current gap |
| --- | --- | --- |
| Parse resistance into ohms | FOM workbook `1200V!U2`: `=MID(download_3[[#This Row],[Rds(On) (Max.) bei Id, Vgs]],1,FIND("mOhm",download_3[[#This Row],[Rds(On) (Max.) bei Id, Vgs]])-1)*0.001`; corresponding U column on `600V`, V on `Tabelle8`. | Current `catalog_fom.parse_quantity` covers primary-field SI parsing and also handles ohms; it does not execute workbook formulas. Preserve source current/gate-voltage conditions. |
| Parse blocking voltage | FOM workbook `Tabelle8!R2`: `=MID(download_7[[#This Row],[Drain-Source-Spannung (Vdss)]],1,FIND("V",download_7[[#This Row],[Drain-Source-Spannung (Vdss)]])-1)`. `600V!AF2` instead uses the first three characters of voltage text. | Current SI parser covers voltage from the original field; it should retain this safer behavior rather than adopt fixed-length parsing. |
| Historical cost FOM | `1200V!AF2`: `=I2/(1200^2/U2)`; `600V!AF2`: `=I2/(MID(download_6[[#This Row],[Drain-Source-Spannung (Vdss)]],1,3)^2/U2)`; `Tabelle8!AG2`: `=I2/(R2^2/V2)`. | Current metric `cost_fom=price*rds_ohm/voltage_v**2` covers the intended family using each actual rating and source quote. The 1200V formula assumes the whole sheet is 1200 V. |
| Relative-cost summary | `Tabelle1!A1:C1` = voltage, `Spalte1`, `Cost`; `C2`: `=B2/MIN(B:B)`, repeated through C311. A/B are stored values, with no formula links back to catalog sheets. | **Gap:** minimum-normalized FOM view and its declared normalization population. Summary lineage cannot be reconstructed solely from formulas; do not invent part IDs or join to catalog by value. |
| Carry typical-to-maximum resistance ratio between temperatures | XLSM `MOSFET!S2`: `=R2*((O2/N2))`; `W2`: `=V2*((S2/R2))`. | M/Q/U headers define temperatures; L/P/T define gate bias. **Gap:** explicit selectable estimate carrying a max/typ ratio between conditions. This is not measured maximum resistance at every temperature. |
| Linear temperature interpolation/extrapolation | `MOSFET!V2`: `=R2+(U2-Q2)*((R2-N2)/(Q2-M2))`; row2 M=25, Q=125, U=150 °C and all three VGS=18 V. `R14`: `=V14-25*((V14-N14)/125)`; `V16`: `=7.5-25*((7.5-N16)/150)`. | **Gap:** catalog-level condition-aware resistance estimate. Row2 is extrapolation from 25/125 to 150 °C, not bounded interpolation. Literal assumptions must be exposed rather than generalized silently. |
| Switching-energy divided by test current | `MOSFET!Y2`: `=0.000632/36`; `Z2`: `=0.000243/36`; `IGBT!Y2` and `Z2`: `=0.003/80`; `IGBT!AA24`: `=0.021/300`. | The headers say EON/EOFF/ERR at TJ=175 °C, but these formulas numerically store **energy-per-current coefficients**, not total event energy. X stores reference switching voltage. **Gap:** typed import/estimate that retains the test current, voltage, temperature and coefficient units; never feed these directly into E(I) as joules. Gate resistance is not a dedicated column. |
| IGBT on-state slope from a point and threshold | `IGBT!W2`: `=1/65`; `W6`: `=(3-V6)/210`; `W21`: `=(3.5-V21)/175`. V/W headers = VCE0/rCE at U's temperature. | **Gap:** catalog import of explicit threshold-plus-resistance models with condition metadata. This is different from fitting a measured VCE(I) curve and from MOSFET Rds(on). |
| Additional thermal path estimates | `IGBT!AH2`: `=AF2+0.66`; `AH6`: `=AF6+AF6*1.9`; `AI6`: `=AG6+AG6*1.9`; `AH26`: `=0.1+0.14`. | AF/AG = junction-to-case, AH/AI = junction-to-heatsink, AJ/AK = junction-to-fluid. **Gap:** explicit documented network/assumptions; literal additions/scalings are row-specific estimates, not universal thermal laws. |
| Thermal power budget at a 100 °C reference | `MOSFET!AV2`: `=(AB2-100)/AF2`; `IGBT!AV2`: `=(AB2-100)/AH2`; AB=maxTJ. | **Gap:** thermal budget metric with a selected reference location and correct path. MOSFET uses RthJC, IGBT uses RthJH. Neither is a transient thermal simulation. |
| MOSFET thermal DC current estimate | `MOSFET!AW2`: `=SQRT(AV2/(V2*0.001))`. | **Gap:** explicitly labeled thermal screening estimate, using typical resistance V2 at the row's U temperature. It does not include package/current limits or solve self-heating. It is not the datasheet current rating. |
| IGBT thermal DC current estimate | `IGBT!AW2`: `=(SQRT(4*AV2*W2+V2^2)-V2)/(2*W2)`. | Positive root of `rCE*I²+VCE0*I=Pbudget`. **Gap:** same explicit thermal screening workflow, with zero-slope handling and domain validation in any new implementation. |
| Current and voltage-current per historical cost | `MOSFET!AX2`: `=AW2/AM2`; `AY2`: `=AX2*F2`; `IGBT!AX2`: `=(AW2*F2)/AM2`. | Current `vi_per_price` uses the quoted catalog current, **not** AW's thermal estimate. Therefore these workbook workflows are not covered by that similarly named metric. AM is cost at 1,000 pieces with unspecified currency. |
| Voltage divided by switching coefficient, optionally cost | `MOSFET!AZ2`: `=X2/(Y2+Z2+AA2)`; `BA2`: `=AZ2/AM2`; equivalent `IGBT!AY2`, `AZ2`. | **Gap:** typed switching-coefficient screening metric. Do not label these ratios efficiencies, losses or device energy limits. Zero/missing component coefficients need an explicit policy. |
| Coss/Qg and resistance products | `MOSFET!AT2`: `=W2*0.001*I2`; `AU2`: `=W2*0.001*J2`. `IGBT!AT2`: `=W2*I2`; `AU2`: `=J2*W2`. | Current `rds_coss`/`rds_qg` cover MOSFET screening, but current import deliberately selects first-condition Rds(max) O, not derived W. Thus family parity is not numerical equality to row2. IGBT rCE must remain a separate metric/type. Later source rows contain broken references, below. |
| Chip area and cooling index | `IGBT!AD25`: `=12.64*11.08*2`; `AE25`: `=12.4*6*2`; `BA23`: `=AD23*AJ23`; `BB25`: `=AE25*AK25`. | **Gap:** structured area, path and their product. Dimensions/counts are literal row assumptions. Area-times-Rth is a screening index, not complete cooling geometry or transient behavior. |
| Price assumption | `MOSFET!AM19`: `=5.9*0.85`. | Row-specific historical price adjustment. Current import retains the formula/cache. No documented meaning for the factor establishes a general currency-conversion workflow. |
| Hyperlink extraction macros | `MOSFET!AR2`: `=GetURL(A2)`, repeated through AR24; IGBT also uses it, for 63 formula cells total. Verified module declarations: `Modul1` line2 `Sub ExtractHyperlinks()` and line17 `Function GetURL(pWorkRng As Range) As String`. | Hyperlink extraction is a substantive data-import convenience. **Gap:** a reusable static hyperlink reader/bulk workflow; macro execution is unnecessary. The source procedure census is now complete below, without claiming execution parity. |

## Power Query and VBA

The FOM workbook's `xl/connections.xml` contains **five** Microsoft.Mashup.OleDb.1 connections: `download(3)` through `download(7)`. `xl/queryTables/queryTable1.xml`, `queryTable2.xml`, and `queryTable3.xml` bind the three loaded tables to connection IDs 1, 4, and 5 respectively. Connection IDs 2 and 3 have no loaded query-table part.

Its `customXml/item1.xml` is a base64 DataMashup payload. Read-only decoding of its declared package section (1,539 bytes) yielded the embedded ZIP and `Formulas/Section1.m`. Each of the five `shared` queries has three steps: `Csv.Document(File.Contents(...),[Delimiter=",", Columns=30, Encoding=65001, QuoteStyle=QuoteStyle.None])`, `Table.PromoteHeaders`, and `Table.TransformColumnTypes`. The recorded file paths are `C:\Users\michael.leibl\Downloads\download(3).csv` through `C:\Users\michael.leibl\Downloads\download(7).csv`. None was opened or refreshed. The queries classify stock/quantity fields as integers and most other columns, including CHF price and electrical values, as text. This is a refresh/import workflow, not an additional switching or loss algorithm. New code should use supplied source paths and robust CSV quoting rather than reproduce machine-specific paths or `QuoteStyle.None`.

The XLSM **does contain a VBA project**, `xl/vbaProject.bin`, 17,408 bytes, with OLE compound-file signature `d0cf11e0a1b11ae1`. A follow-up located already-installed `olefile` 0.47 at `C:\Users\tinix\miniconda3\envs\social_env\Lib\site-packages`. The bundled and Windows/WSL `ntbees2` environments lacked this reader; no installation was required. `scripts/audit_vba_metadata.py` uses that existing reader with `write_mode=False`, then a bounded [Microsoft MS-OVBA decompression implementation](https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/492124cc-5afc-48c8-b439-b42ad7087a7b). It reads ordered directory records from byte0, rejects unsupported record forms, validates stream bounds and each decompressed `Attribute VB_Name`, and reconciles every module against the independent `PROJECT` list. Module source starts at the directory's declared offset, rather than searching the performance cache for plausible text. [Microsoft's MODULE record specification](https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/f6475ef5-0436-47c6-b31f-ea8f23810dc6) defines that mapping.

The source census found **five modules and exactly two source procedures**. Four document modules contain no procedure declarations and therefore no source event handlers. Counts reconcile procedure declarations with their `End Sub`/`End Function` statements. Source bodies were neither executed nor copied into this report or the helper.

| Module / stream under `VBA/` | PROJECT kind | Directory record offset | Declared source offset in stream | Stream bytes / compressed source bytes / decompressed source bytes | Procedure signatures and source lines |
| --- | --- | ---: | ---: | --- | --- |
| `DieseArbeitsmappe` | Document | 478 | 821 | 1,007 /186 /310 | None |
| `Tabelle1` | Document | 656 | 821 | 995 /174 /301 | None |
| `Tabelle2` | Document | 780 | 821 | 995 /174 /301 | None |
| `Modul1` | Module | 904 | 2,063 | 2,451 /388 /552 | Line2 `Sub ExtractHyperlinks()`; line17 `Function GetURL(pWorkRng As Range) As String` |
| `Tabelle3` | Document | 1,016 | 805 | 979 /174 /301 | None |

The compound file contains 13 streams. `VBA/dir` is 627 compressed bytes and 1,146 decompressed bytes, declaring code page1252. Its reference section, parsed through the declared module-list boundary, contains two registered references: `stdole` at directory offset122, GUID `{00020430-0000-0000-C000-000000000046}`, version2.0, recorded path `C:\Windows\SysWOW64\stdole2.tlb`; and `Office` at offset262, GUID `{2DF8D04C-5BFA-101B-BDE5-00AA0044DE52}`, version2.0, description Microsoft Office14.0 Object Library, recorded path `C:\Program Files (x86)\Common Files\Microsoft Shared\OFFICE14\MSO.DLL`. These are recorded metadata, not proof those libraries are installed or loaded. The remaining compound-file streams include compiled/performance caches; this is a complete **source** census, not a disassembly or guarantee that cached executable code matches source. No Excel, macro, or external reference was loaded. The `.xlsx` files contain no `vbaProject` part.

Verification passed 23 byte-level decompression fixtures, including Microsoft's canonical [no-compression example](https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/515cce92-0cbb-4588-b273-1d93ec583ba6), [normal-compression example](https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/b6b8c482-a6bb-45e5-a476-098897a334cb) and [maximum-compression example](https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-ovba/22374b4a-8092-4281-b5af-66f56a27073c), plus raw/multiple chunks, overlapping copies, changing copy-offset bit widths and invalid bounds/signatures. Ruff passed on the helper. The final XLSM SHA-256 remains `f7b4c0424f1329c469dbffd96b4ab44fe6ac448ccb7ff0e6f331159c8a0b3ab0`.

To repeat the metadata-only audit from this repository in PowerShell:

```powershell
& 'C:\Users\tinix\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B scripts/audit_vba_metadata.py --self-test --workbook '\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc\db\SemiconductorsData.xlsm' --olefile-site-packages 'C:\Users\tinix\miniconda3\envs\social_env\Lib\site-packages'
```

## Source defects that must not become parity requirements

- The FOM workbook stores **385 cached `#VALUE!` cells**: 46 on `1200V`, 211 on `600V`, 128 on `Tabelle8`. Examples: `1200V!I141` is `Obsolet`, so `AF141` divides text; `600V!I951` is `Digi-Reel`; `Tabelle8!U133` is `3.3Ohm bei 50mA. 5V`, so the `FIND("mOhm",...)` formula in V133 fails. These are actual saved errors, not newly evaluated failures.
- The hardcoded 1200V denominator assumes every row's rating equals 1200 V. That assumption is consistent with the observed Q-column values: 148 entries `1200V` and 26 entries `1,2kV`. It is not evidence of an existing numerical error on this sheet. A reusable implementation should still parse each original field, including SI prefixes, rather than inherit the sheet-specific assumption or the other sheet's fixed character length.
- XLSM MOSFET FOM formulas change semantics after row2. `AT3=V3*J3` and its shared followers multiply typical resistance by typical Qg, despite AT1 saying `FOM(Coss*Rdson_max)`; `AU3=W3*K3` uses maximum Qg with a derived maximum resistance. The latter is not the same condition/product as row2 AU2. Copying these formulas would reproduce misleading labels and inconsistent scaling.
- XLSM IGBT `AT3=Q3*E3` multiplies a temperature cell by the text qualification field (`E2` is `automotive`), and `AU3=R3*F3` is not the stated Qg/resistance product. The source saves 39 `#VALUE!` and two `#DIV/0!` errors on `IGBT`, 41 errors total. MOSFET caches contain no standard Excel error tokens, which does not establish correct formulas or units.
- Spreadsheet sums treat blank energy coefficients as zero, and some zero results have incomplete operands. A new engine must retain missing data explicitly. Saved zeros, zero formula errors, or an old chart cannot establish measured-data coverage.

## Migration implications

The current catalog module already recomputes primary-field MOSFET FOMs in SI, preserves raw/cached/formula evidence, isolates source/quote context, exposes missing operands, and calculates an actual two-objective Pareto frontier. This comparison is backed by `catalog_fom.py` functions `parse_quantity`, `metric_values`, `screen_catalog`, and `pareto_ids`, and `scripts/import_sc_catalogs.py` function `normalize`. It does not prove the remaining workbook-derived workflows are implemented.

Remaining substantive work is: (1) minimum-normalized summary view with declared population and traceable source rows; (2) typed condition-specific MOSFET/IGBT screening estimates, including threshold-plus-resistance, temperature estimates and their provenance; (3) explicit thermal budgets/current estimates with the chosen reference location and package limits; (4) energy-per-current coefficient import and ratios with reference voltage/temperature/current; (5) structured area/cooling and historical cost assumptions; (6) reusable CSV/worksheet refresh plus static hyperlink extraction. A lean implementation should share validated Python primitives with FastAPI and the web UI, while preserving estimates separately from measured curves. Excel formatting, empty sheets, obsolete paths and broken copied formulas are not separate feature gaps.

The compressed VBA source census is now complete for this exact workbook hash: five modules, two procedures, no source event handlers, and two registered references. Runtime macro behavior and compiled cache/source equivalence were deliberately not tested. No claim of implemented macro parity is made. The scientific and web migration checklist remains owned by `sprint-semiconductor-function-parity.md`; this report does not mark an implementation step complete.
