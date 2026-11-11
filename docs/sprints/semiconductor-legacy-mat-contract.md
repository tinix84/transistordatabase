# Legacy SC MAT import and compatibility contract

Read-only preparation, 2026-10-16. This supplements step 8 of
`sprint-semiconductor-function-parity.md` and `semiconductor-data-workflow-contract.md`.
It is not a parser implementation, phase acceptance or measured-device validation.
Only this document was added. No MATLAB/Octave/reference Python, GUI, macro or
embedded expression was executed. Existing WSL Conda `ntbees2` SciPy read binary
data; Octave text was inspected as text. Both reference roots remain read-only:

```text
\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc
D:\OneDrive\claude\bhy\enco
```

## Verified format census and source semantics

SC contains 198 `.mat` files totaling 287,585,927 bytes: 174 MATLAB 5-family
binary files and 24 Octave text workspaces. SciPy successfully read all binary
files. Binary top-level schemas are 160 `scData`, four `hsData`, four individual
digitizer traces, and six energy-grid/raw measurement workspaces. Counts include
duplicate source files. Every binary `scData` declares `fileFormatVersion='0.2'`
and has the same 25 field names. There are actual supported migration fixtures;
MAT import cannot be deferred on the premise that none exist.

The complete source-census fingerprint is SHA-256
`75fb1ff7951defd709175bd944fd7ea3284730d91387f7bd526a16da4b303206`.
Definition: sort archive-relative POSIX paths lexically; concatenate, per file,
UTF-8 `relative_path + NUL + lowercase_sha256(file_bytes) + LF`, then hash.
Retain a full per-file manifest as implementation evidence later; this fingerprint
does not replace per-file diagnostics. A final independent reread after preparation
reproduced this fingerprint for all198 files unchanged. The largest supplied file
is the15,611,926-byte text `db/scData/ROHM_SCT3022AL_Eon.mat`.

| Read-only SC source | Observed contract, not behavior to copy blindly |
| --- | --- |
| `matlab_func/save7.m:1–9` / duplicate `mfiles/save7.m` | Octave `-mat7-binary`, MATLAB `-v7`: these use the SciPy-readable MAT5 family, not HDF5 v7.3. Preserve save intent without calling `evalin`, MATLAB or Octave. |
| `matlab_func/ConvertscDataFileFormatFrom00To01.m:12–16` | Missing version is treated as 0.0; conversion adds three empty thermal-interface-area fields and version0.1. Its single-file branch has an indexing defect; GUI overwrite behavior is not a requirement. |
| `matlab_func/ConvertscDataFileFormatFrom01To02.m:12–21` | Changes only version0.1 to0.2 and uses `save7`. It does not add measurement conditions, repair values or alter units. No actual0.0/0.1 fixture was found in this binary census; synthetic compatibility tests are required. |
| `mfiles/SemiconductorEditor.m:687–720` | Identity/rating/cost/weight; signed combined conduction and switching arrays; independent on/off reference voltages; three thermal resistances/areas; six original branch traces. |
| `mfiles/SemiconductorEditor.m:742–780` | Conduction traces have17 fields, switching traces18 including `testVoltage`; retain values, labels/scales, calibration anchors, point locations and image. Whole trace fields can instead be empty arrays:102 such fields occur across160 records. |
| `mfiles/SemiconductorEditor.m:783–848` | `combineTraces` swaps axes by an exact label comparison, inserts ±1e−6A extrapolations, supplies ±1e100 zero placeholders, sorts/uniques and combines signed branches. Its implicit voltage normalization is not a measured-data contract. |
| `mfiles/DataPicker.m:139–158,288–351` | Both linear and logarithmic calibration exist. Locations use the historical unusual x/image-height and y/image-width normalization. Preserve this coordinate convention; do not reinterpret old locations as ordinary x/width,y/height. |
| `matlab_func/convertPlecsXmlToscData02.m:14–60` | Nearest-condition selection and evaluated XML formulas can produce version0.2 files without trace/condition evidence; version alone never establishes measurement completeness. `break_down_voltage` in that writer differs from editor `blockingVoltage`; preserve unexpected aliases and require an explicit conflict rule. |

An ENCO component-source search found no corresponding MAT reader/writer to reuse.
This claim is scoped to that inspected component tree; it is not proof that every
ENCO project workspace has no MATLAB data. ENCO PLECS/formula work belongs to the
separate data-workflow contract. Implement independent adapters, not copied bodies.

## Typed preview first; scientific finalization is separate

Provide headless bytes→preview and preview+explicit choices→finalized model
operations, with thin typed API/Vue adapters. Never load by an arbitrary browser-
supplied filesystem path. Preview all160 actual binary `scData` fixtures even when
scientific evaluation is unavailable. Return parser/schema version, original
filename, SHA-256, variable/field paths, dtype/shape, original units/labels,
unknown metadata, diagnostics and preserved source values. A valid data container
can contain an unusable physical curve; distinguish these outcomes. Large images
need descriptors/hash and a bounded preview/blob reference, not millions of JSON
numbers. Never drop the original image or calibration when exporting an archive.

Use the installed SciPy reader for MAT5; retain original dimensions/dtypes in the
intermediate representation before deliberate singleton normalization. Reject
arbitrary MATLAB objects, function handles, opaque data and unsupported structures
with specific diagnostics; do not deserialize executable code or import modules
named by a file. HDF5 v7.3 and general MATLAB/Octave workspaces are separate,
explicitly unsupported formats until a real schema/reader is reviewed. File-size,
expanded-array/image, nesting and variable-count budgets must precede expensive
materialization; never silently truncate. The declared supported budgets must
accommodate the actual supplied fixtures, with safe errors beyond them.

The intermediate model preserves all25 known fields plus unknown extras. Empty
arrays mean unknown/missing, not0; zero scalar thermal resistance is a distinct
recorded ideal-path value. Preserve an empty name and let a user supply a separate
identity; a filename is a provenance label, not an inferred part number/rating.
Do not infer temperature from `125C` in a filename, or Rg, gate bias, commutation
partner, max/typical status, energy integration convention or measurement accuracy
from a file version. Do not invent voltage/current ratings from curve coverage.
Cost currency and weight units require explicit metadata where absent.

The editor labels A,V,J and thermal K/W,m² establish a proposed schema unit map,
which preview displays with its provenance. Trace labels can reverse current and
voltage axes and can contain trailing spaces (two observed). Parse only recognized
unit/axis labels; display ambiguous labels for explicit mapping. Unknown or
conflicting labels must not become guessed SI. Supply conditions independently
for conduction, Eon and Eoff, including branch-specific metadata. Retain each
new user declaration alongside the original absence and conversion history.

Split combined conduction by current sign, mapping reverse signed I,V to positive
magnitude coordinates only under an explicit signed-passive convention. Energy
is not voltage: do not negate energy merely because its current is negative.
Keep positive and negative branch domains separate. No interpolation across zero,
borrowed branch, inferred E(0), extension to zero, sorting away conflicts or hidden
extrapolation. At exactly zero, use explicitly recorded branch values/conventions
or require a choice; conflicting one-sided limits stay visible. Existing accepted
bounded tables/waveform/thermal engines consume finalized data, not raw preview.

Prefer original branch traces for scientific finalization when combined arrays
have synthetic points, altered normalization or conflicts. Preserve both versions
and report their differences; do not silently regenerate the combined arrays and
declare them repaired. If a branch trace is absent, original combined values can
still be previewed and explicitly finalized over their genuine recorded domain
when units/conditions/convention are supplied. Unknown metadata remains unknown.

## Placeholders, negative energy and actual conflicting references

`db/scData/Infineon_IPW65R048CFDA_125C_Max.mat` has empty identity/rating/cost/
weight/thermal fields, real signed conduction, and both switching currents
`[-1e100,-0,1e100]`, all-zero energies, reference1V, with empty switching traces.
This matches the editor's empty-trace construction: expose an identified synthetic
placeholder candidate with evidence, not a zero-loss measurement over an enormous
domain. The exact signature plus empty/source trace evidence distinguishes it
from an independently measured/explicit ideal zero curve. Across160 binary files,
71 Eon arrays and70 Eoff arrays are entirely zero; those counts alone do not
classify placeholders. Deliberately ideal zero data need a declared ideal-model
status and scope, never a fabricated measurement source.

Ten binary records contain negative Eon samples: nine only at negative current,
one (`db/scData/STPSC40065CWY.mat`) at positive current, including5A. Thirteen
records contain negative Eoff samples; all13 negative values occur at +1e−6A,
the editor's inserted near-zero location. Example
`Infineon_IHW40N120R5_with_Infineon_IHW40N120R5.mat` has a negative combined
near-zero Eoff even though its forward trace has only positive measured energies;
this is consistent with extrapolation. Other tiny negative values already occur
in stored traces. Reverse Eon traces may contain substantial negative values,
and the combined arrays do not universally reconcile with the present editor
algorithm: signed conventions, earlier transformations and edits are unresolved.
The text half-bridge fixture below also contains negative reverse Eon values.

These are observed sign/location facts, not proof of defective measurements or
recoverable returned energy. Do not clamp, take absolute values or assume that
negative-current energy is always signed-negative. Preserve original values and
source convention; unresolved negative energy is unsupported for passive heat
evaluation. A future explicit net-electrical-energy workflow must be distinguished
from junction heat and cannot be enabled merely by relabeling these curves.

Real conflict: `db/scData/Infineon IPX65R190CFDA_125C_HB_Sw+Sw.mat` stores forward
turn-off trace175V, reverse turn-off trace400V, and combined `turnOffVoltage=175V`.
The editor scales each branch by its original V/min(V), opposite the usual named
linear E∝V normalization to a lower voltage. Never apply that transformation as
measured parity or infer one shared original reference from the combined field.
Return both trace references plus the combined declaration and a conflict report;
any new linear-voltage estimate must be explicitly selected and retain original
branch values. No physical interpretation of this file's negative reverse trace
is established by the source inspection.

## Bounded Octave text subset

The24 text files contain23 single digitizer workspaces plus one complete `scData`:
`db/scData/Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat`. Observed token counts are
30 `scalar struct`,166 `matrix`,225 `scalar`,112 `sq_string`,25 `uint8 matrix`.
This small data-only subset is sufficient to target these actual fixtures without
launching Octave or implementing a general interpreter.

Specify a bounded reader over the literal `# name:`, `# type:`, dimensions/counts
and payload records. Scalar struct declares2-D1×1 plus field count; recursively
consume exactly that number of named fields. Numeric2-D matrices declare rows/
columns and row payloads; n-D uint8 images declare `# ndims: 3` then H,W,C and
scalar values in Octave's column-major order. Verify0–255 integer pixels and
reshape in Fortran order. Strings declare element/length metadata; preserve
literal text/line breaks using a tested encoding/count convention, never run
string content. Require exact dimensions/counts, unique field names and complete
consumption, except recognized creator comments/whitespace. Unknown types,
function handles, expressions, complex/sparse/cell/object data or malformed counts
produce a useful unsupported/invalid report. Do not use `eval`, `load` via Octave,
pickle, arbitrary XML formulas or source-file imports.

Test linear/log calibration independently: location fractionq maps to
`a+q*(b−a)` or `10**(log10(a)+q*(log10(b)−log10(a)))`; log anchors must be
positive and location anchors distinct. Preserve y-axis inversion naturally from
anchor order. Old x/height,y/width coordinates map back to pixels by multiplying
those exact legacy dimensions. Compare reconstructed coordinates with stored
values and disclose residuals; manual value edits can legitimately disagree, so
never overwrite source values. Out-of-calibration points are visible unsupported
or an explicitly selected extrapolation estimate, not silently accepted samples.

## Other actual workspace schemas and compatibility export

`hsData` contains only common/forward/reverse thermal-interface resistance
coefficients; source naming is not sufficient to treat these as K/W.
`mfiles/SemiconductorTemperature.m:10,19,28` divides them by the corresponding
interface area: the dimensional contract is K·m²/W divided by an explicitly supplied
interface area. Retain all three branches. `hsData_ideal.mat` actually stores
0.000133 for each coefficient: its filename does not mean ideal zero.

`utils/dpt/data/Eloss_Infineon_IPW65R190CFDA.mat` contains matching8×7 I,V,Eon,
Eoff meshes, current increasing across columns and voltage descending400→50V
down rows. Preview them unchanged. A canonical grid may reorder whole rows with
all arrays together after validating coordinate pairs; never assume5×6, transpose
energies alone or infer unknown Tj/Rg. `utils/dpt/data/EPC2001.mat` has48 Eon/Eoff
values,48×2 `IV_mat`, and a filename string. A referenced log path is provenance,
not authority to open arbitrary local files. Validate pair counts/order before
forming a measured grid; the existence of a filename is not DPT window evidence.

Provide canonical JSON preview/finalized exports first. A supported legacy MAT5
writer can use SciPy `savemat` with an explicit25-field0.2 compatibility mapping;
unsupported conditions/new fields require a sidecar/canonical document and a
visible loss report, never silent metadata loss or false historical equivalence.
Return bytes/downloads or write a fresh user-selected destination without overwrite.
No original file mutation, GUI migration, version-number-only physical repair or
claim of arbitrary `.mat` compatibility. Original source bytes remain the authority
for lossless archival preservation, including unsupported extras.

## Required implementation and independent acceptance evidence

1. Retain a per-file source-hash/format/schema census; preview all160 binary records
   and the14 other binary workspaces without execution. Exercise all24 text files
   against the declared subset. Separate duplicate identity from distinct paths.
2. Confirm raw→preview→canonical JSON→preview preserves every scalar/array/shape,
   empty/zero distinction, conditions/extras and source identity. Where legacy
   MAT export is selected, independently inspect its fields/dtypes/shapes with
   SciPy and report intentionally unrepresentable metadata. Semantic equivalence
   is required, not binary-identical compression/header timestamps.
3. Test actual files below plus independent synthetic0.0/0.1 versions, singleton/
   row/column/empty arrays, two-sided zero limits, differing branch references,
   trailing/unknown labels, genuine all-zero measurements versus placeholder
   candidates, negative-source energy, nonfinite/complex values, duplicate fields,
   malformed dimensions, over-budget nesting/images and unsupported v7.3.
4. Calibration corner/center fixtures: linear10→30 gives20 at midpoint; log1→100
   gives10; inverted-y coordinates and nonsquare RGB images must retain pixel
   identity/channel ordering. Test UTF-8/newline/metadata-looking string content
   using byte lengths supported by the documented subset. Reject incomplete
   payloads without partial scientific success.
5. Prove a signed conduction fixture never interpolates through the unsupported
   current gap, a source-only filename does not invent Tj, placeholders do not
   give zero switching loss, and unresolved negative energy never heats/cools a
   passive Foster model. Browser preview→explicit mapping/conditions→bounded
   calculation→download must match Python values and invalidate stale edits.
6. Recompute the complete census fingerprint after read-only tests; preserve source
   hashes. Record implemented/supported-preview/unsupported-evaluation states
   separately in the reviewed step-8 overlay; do not regenerate the baseline ledger.

Selected immutable fixture hashes (archive-relative paths):

| Fixture | SHA-256 |
| --- | --- |
| `db/scData/Infineon_IPW65R048CFDA_125C_Max.mat` | `8d2cffb487a246d550a069b891267944c25347c7420928541aba24568d3b20c1` |
| `db/scData/Infineon IPX65R190CFDA_125C_HB_Sw+Sw.mat` | `e0e438c05b2e80583dcda4f777abd80096735e0bb5ad36fe86099e81100f555a` |
| `db/scData/Infineon_IPW65R048CFDA_125C_HB_Sw+Sw.mat` (text) | `5b60413bae4f0289b1f44d3771e2cd78293684353cc91b7d000baeb009c849a6` |
| `db/scData/STGYA120M65DF2AG_Eon.mat` (binary trace) | `7f656104426baa4da8149a6629b091e3b9a50b47e7c75a598033c8fee462b15a` |
| `db/scData/Cree_C3M0075120K_Eoff.mat` (text RGB817×1040×3) | `d1002fc1960406edf196eff890dc9cd393caf91842bf265bd64386d3497d8c11` |
| `db/scData/hsData.mat` | `4b7778bd2a8dee0896374c23f3071c6ebe385d999649eb8caefc4e30f203b244` |
| `db/scData/hsData_ideal.mat` | `17d72c31de007f98ec475de8f36976150707e2fbab71d7e4188e38f9fc2e7882` |
| `utils/dpt/data/Eloss_Infineon_IPW65R190CFDA.mat` | `40dcd2d37811a96d6c1ecede546c2e98fda428d7ddb4ba84587c8f5f9af983ac` |
| `utils/dpt/data/EPC2001.mat` | `81600471157f15e6c75e7bcf13b1848bdc5077263ddc8e1a318452c0753a8629` |
