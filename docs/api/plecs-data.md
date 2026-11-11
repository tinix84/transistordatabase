# PLECS source documents and explicit core export

The headless modules `plecs_document.py` and `plecs_core_export.py` replace the
legacy importer’s guessed ratings/default conditions and the exporter’s process
working-directory changes. No reference application, formula, macro or simulator
is executed. A successful XML round trip establishes data-format equivalence,
not simulator acceptance or measured-device accuracy.

## Source preview and download

`POST /api/plecs/preview` accepts exactly one source representation:

```json
{"xml_base64":"...original file bytes...","source_name":"device.xml"}
```

or `{"xml_text":"<SemiconductorLibrary ...>","source_name":"pasted XML"}`.
Uploaded file bytes use base64 so the SHA-256, encoding, BOM and archival download
refer to the actual file. Unicode text submissions have the separately labelled
`source.hash_scope="submitted-utf8-text"`. Text is parsed as Unicode, including
when an old declaration still names Latin-1; supported model rendering writes a
consistent UTF-8 declaration. A source-preserved text download retains submitted
text bytes. Non-ASCII text with a non-UTF-8 declaration must explicitly correct
its declaration before source archive export, or use supported UTF-8 rendering;
for non-UTF-8 file identity use byte upload.

The typed document contains:

- `source`: name, SHA-256, encoding and hash scope; `source_base64` retains bytes.
- `source_xml`: decoded source, `namespace_uri`, original library version.
- All packages and semiconductor blocks with distinct IDs and actual identity.
- Every conduction/Eon/Eoff table: original axes, raw values and scale, method,
  explicit known/unknown gate/Rg conditions, source path and attributes.
- Python-generated `slices`: actual SI current/voltage or current/energy chart
  pairs, actual Tj/supply conditions and units. Formula placeholder tables have
  no scientific chart slices.
- Foster, Cauer and state-space thermal source metadata with explicit support
  statuses. Cauer R/C elements are never treated as Foster R/tau poles.
- Original variables, constants, custom tables, comments and other unknown
  content as uninterpreted XML, alongside readable diagnostics.
- `status`: `supported-tables`, `unsupported-content` or `missing-data`;
  `can_export` means supported table-model rendering is available. `can_archive`
  means source preservation is available, including unsupported formulas.

Absent ratings remain `null`. A table’s maximum current/voltage, a variable’s
maximum, and a human comment never establish a device’s absolute ratings or gate
conditions. Nonempty variables/constants/custom tables and all formula methods
are preserved and reported as unsupported scientific content. Independent
table-only sections remain visible even when another section is unsupported.

`POST /api/plecs/export` accepts the same source plus required `mode`:

- `supported-tables`: validates and renders the supported numeric document as
  namespaced UTF-8 XML; unsupported structures return HTTP 422 with diagnostics.
- `source-preserved`: downloads exactly the supplied source bytes. This is an
  explicit archive of original content, including formulas and unsupported
  thermal topology, without a claim that those models were evaluated.

All malformed axes, duplicate/overlapping conditions, rectangular shape failures,
nonfinite data or unsafe XML return HTTP 422 `{ "detail": "..." }`. Distinct Rg
contexts remain distinct; duplicate conditions do not silently choose a first
record. Default, prefixed and no-namespace PLECS inputs are supported; foreign or
mixed namespaces, DTDs/entities and UTF-16/32 bytes are rejected. Supported byte
encodings are UTF-8, Latin-1 and Windows-1252. Bounds are 8 MB source, 100,000 XML
nodes, 64 nesting levels, 128 packages, 512 blocks per package and 100,000 numeric
table cells across the whole document. The complete numeric/plot workload is
checked before constructing SI plot copies. No samples are silently truncated.

## Explicit source-to-core mapping

`POST /api/plecs/finalize` adds:

```json
{
  "package_id":"0",
  "ratings":{"v_abs_max":1200.0,"i_abs_max":75.0,"i_cont":40.0,"t_j_max":175.0},
  "thermal_properties":{"housing_area":0.0001,"cooling_area":0.0002},
  "block_targets":{"0":"switch"},
  "gate_voltage_by_table":{"Package[0]/SemiconductorData[0]/TurnOnLoss[0]":15.0},
  "recovery_voltage_convention":"preserve-signed",
  "package_thermal_target":null
}
```

All required gate conditions must be declared for actual selected tables; the
example above is illustrative and a file with additional switch tables needs
their corresponding paths. Existing numeric source gate conditions cannot be
overridden. Every block is deliberately mapped to a distinct `switch` or `diode`
target. Multiple blocks sharing one core component are unsupported. A package
thermal branch with multiple core components needs an explicit owner.

`negative-to-positive-bus` reversibly negates nonpositive PLECS diode recovery
supply voltages for the core positive-bus convention. Energies are not negated.
`preserve-signed` retains original signed voltages. An explicit all-zero diode
turn-on placeholder stays in source provenance and does not become a fabricated
switch Eon curve; nonzero diode turn-on has no supported core representation.

The response has `status="finalized-not-persisted"`, declared metadata, full core
arrays, `roundtrip_xml_base64` and `core_json_document`. The last field is the
complete reloadable archive used by the browser JSON download; the readable
`core` field is a display view. No device is implicitly saved or overwritten.
Original bytes and absence of metadata are preserved separately from declarations
in the core metadata provenance comment. Source-backed core export verifies that
identity, ratings, thermal information, arrays and conditions remain unchanged;
editing core values cannot emit stale original XML. Generic core selections use
the separate explicit operation below.

The existing `import_plecs_xml(path, finalizations=[...])` Python adapter now requires
the same explicit mapping arguments. The legacy upload `/api/import/plecs` safely
reports that metadata finalization is required and saves nothing. Its old
inference of ratings from curve extrema, 25 °C defaults, zero gate voltage and
half-maximum continuous current has been removed.

## Complete native core JSON persistence

`core/serialization.py` defines the explicit `transistordatabase-core` version 1
document consumed by `JsonTransistorLoader` and emitted by its save operation and
`ExportService.export_to_json`. Saving no longer constructs a legacy device or
requires a vendor/housing lookup list. An actual unlisted manufacturer and unknown
empty housing remain unchanged. Existing flat legacy JSON continues to load
through the established compatibility branch; source files are not rewritten.

The archive retains all existing core aggregate/dataclass fields: identity and
source provenance, independently declared ratings/areas, component metadata,
curves and conditions, Foster data, raw nested numpy arrays, diode recovery
parameters, timezone-bearing datetimes, explicit `None` and working-point models.
Tagged numpy arrays retain dtype/shape, including empty arrays and extended
precision without narrowing them to Python floats. Mapping data is independently
tagged, so user metadata keys cannot masquerade as serializer instructions.

Only registered core models are reconstructible; archive tags never import a
module or evaluate code. Unsupported versions, unknown classes, incompatible
field sets, malformed shapes and nonfinite values fail clearly. Serialization
validates the entire object before opening a destination. This schema serves
core persistence, rather than a claim that legacy applications understand the
new native archive. Tests use fresh output destinations and preserve reference
JSON/XML files unchanged.

The existing multipart `POST /api/transistors/upload` accepts the complete native
archive file unchanged, parses it in memory and saves it through the shared real
repository. `POST /api/transistors` and the existing dictionary update helper
recognize native archives before considering nested forms or complete flat legacy
JSON. An explicit unsupported schema/version, including a malformed native
document missing its schema marker, is rejected; it never becomes an `Unknown`
device. The active browser file uploader forwards the original file without
rebuilding its fields. Existing flat legacy files retain their complete curves,
and explicitly supplied nested metadata forms retain their existing contract.

`tests/test_native_json_api_import.py` verifies native file upload → isolated
real repository save/reload → existing JSON download → native reload → semantic
PLECS export, plus unknown-schema rejection and existing legacy-file compatibility.
No user database is written by these tests and no staging-file cleanup is needed.

Repository persistence separately constrains the storage filename: absolute,
drive, separator, traversal and control-character names are rejected without
rewriting actual device identity. Both save and lookup resolve the candidate,
including any existing link, and require its parent to be the exact resolved
repository directory. Boundary failures return HTTP 422 through native import
and device/core lookup routes. Source preview and document export retain actual
identity independently of this filesystem storage requirement. Focused tests
verify rejected imports, writer noninvocation and preservation of an existing
outside link target; they do not remove any files or links.

## Exact selection from existing core records

`core_selection_document(transistor, selection)` / `render_core_selection(...)`
support ordinary core records, independently of PLECS source provenance. Selection
must explicitly name one component, each included mechanism’s exact gate/Rg
conditions, thermal inclusion, omissions with reasons and voltage convention:

```json
{
  "component":"switch",
  "conduction":{"gate_voltage_v":15.0},
  "turn_on":{"gate_voltage_v":15.0,"gate_resistance_ohm":2.0},
  "turn_off":{"gate_voltage_v":-3.0,"gate_resistance_ohm":5.0},
  "include_thermal":true,
  "omission_reasons":{},
  "recovery_voltage_convention":"preserve-signed"
}
```

The typed HTTP operations are `POST /api/plecs/core/{transistor_id}/preview`
and `POST /api/plecs/core/{transistor_id}/export`, with that selection object as
the request body. Preview returns `{document, report}`; export returns XML bytes
directly without temporary files. Both use the existing application repository
through an injected dependency. Missing device IDs return 404 and unsupported
selections return 422. The concrete ExportService accepts
`template_config={"selection": selection}` for an explicit filesystem destination;
existing files are never overwritten. A call without selection supports only
unchanged source-backed imports and otherwise fails with the workbench direction.

`GET /api/plecs/core/{transistor_id}/contexts` lists actual recorded contexts
under `components.switch` / `components.diode`: conduction, turn-on and turn-off
groups, curve counts, actual temperature/supply sets, dataset types, additional
conditions and thermal availability. It never selects a context automatically.
An invalid record produces a top-level diagnostic with component, mechanism,
record index and reason while valid independent groups remain available. Boolean
and nonfinite physical conditions are invalid. An invalid record does not become
an inferred unknown value; exact selected export remains strict.

An explicitly selected `null` condition matches actual unknown data, never any
gate/Rg value. A `null` mechanism is an omission and requires a nonblank reason;
it does not produce a zero-loss table. At least one table must be included. Core
diode Eon is unsupported and must be omitted explicitly. Selected energy records
must have a single consistent additional gate-off/commutation-device/inductance
context, complete Tj×supply rectangular coverage, identical current axes and no
duplicate corners. Interpolation, nearest selection and automatic extrapolation
are absent. Negative energy requires a resolved source convention.

The returned report lists actual curve counts, current axes, temperature/voltage
coverage, contexts, omissions and `simulator_validation="not-performed"`. The
exact fixed conditions and actual separately known ratings are retained in a
machine-readable `TDB_FIXED_CONDITIONS_JSON=` Comment line. The intermediate
reader exposes it as `package.fixed_conditions`; it never mines human comments
for physical metadata. Numeric PLECS table axes remain current/Tj/supply voltage.
The Comment record is adapter metadata, not a newly implemented PLECS gate axis.

## Source and verification evidence

The tests read the two supplied SC sources `IGW60T120_IGBT.xml` and
`IKW20N60TA_Diode.xml` without modifying them, verify supported table/Foster
corners and source SHA-256 before/after, and exercise signed diode recovery.
The actual ENCO `G3R40MT12K.xml` has **one** SemiconductorData block and two
ConductionLoss elements (`gate="on"`/`gate="off"`), plus formula/custom-table/
Cauer/state-space content. It now yields a useful unsupported-content preview
and byte-exact source archive, rather than an exception or plausible incomplete
model. Multiple packages/blocks are separately tested synthetic format cases.

Tests: `tests/test_plecs_document.py`, `tests/test_plecs_core_export.py`, and
updated explicit-mapping `tests/test_plecs_import.py`. Browser workflow evidence
is recorded by the PLECS workbench tests. No simulator was launched for this work.

## Browser workflow

Open `http://localhost:8002/index.html?view=plecs-data`, or choose **PLECS data
workbench** under More Tools. Upload XML or paste source text, then inspect the
document. File uploads retain original bytes for SHA-256 and archival downloads;
editing the XML text explicitly switches to a new UTF-8 text identity. Source
name edits do not change the XML bytes. Results and pending responses are
invalidated when their source changes.

Each package and semiconductor block has its own recorded-table evidence,
unexecuted formulas and preserved metadata. Supported slices show Python-supplied
SI points and matching tables; original axes, values and scale remain inspectable.
The signed supply label does not silently change reverse-recovery conventions.
Thermal networks remain attached to the correct block or package. Cauer R/C
elements and package state-space source are visibly distinct from Foster poles.
Unsupported models permit canonical document JSON and exact original-source XML
downloads, while the supported table export remains disabled.

The optional core-mapping form starts with blank datasheet ratings and areas,
unselected block targets and an unselected recovery convention. Gate declarations
and package thermal ownership are explicit. Successful mapping shows actual curve
counts, separately declared limits and `finalized-not-persisted`; its JSON download
uses the versioned native `core_json_document` for the existing loader. Its XML
download is the verified source-backed core round trip. No database save is
implicit.

The existing-database export form uses the current device list and the exact
recorded contexts endpoint. It never selects the first device, physical component
or context automatically. Inclusion flags and nonblank omission reasons define the
selected subset. Actual unknown gate/Rg values remain null, not zero. The preview
shows context/coverage evidence, Python-produced SI plots and points, omissions and
simulator-validation status. Evidence JSON and semantic XML downloads share that
selection; editing it clears prior results.

Frontend regressions are in `PlecsWorkbench.test.js`, `PlecsCoreMapping.test.js`,
`PlecsCoreExport.test.js` and `e2e/plecs-data-workbench.spec.js`. A retained Canvas
draw probe converts the actual rendered 50 A endpoint back to energy using drawn
axis ticks, checks 0.0027 J and verifies unrelated downloads/form edits do not
restart unchanged plot animations. Stable computed datasets prevent that source
of misleading intermediate screenshots. Numeric tables scroll within the mobile
viewport while source names and paths wrap without changing their contents.

Owner verification on 2026-10-16 passed 17 Vue tests across those three focused
suites and five Chromium workflows. The three unaffected real-source/mobile/error
cases passed in `/tmp/tdb-plecs-owner-browser-20261016-01`; the final source/native
archive and existing-core cases passed in
`/tmp/tdb-plecs-owner-source-generic-final-20261016-04`. The affected generic
canvas probe and its final settled screenshot passed in
`/tmp/tdb-plecs-owner-generic-render-final-20261016-05`. Its actual drawn endpoints
are 4 V, 0.002 J and 0.004 J at 20 A, matching the in-memory production API's
independent recorded fixture. The source endpoint remains 0.0027 J at 50 A after
XML downloads and mapping edits. Both final desktop and 390px screenshots were
visually inspected. Existing stored database data was not changed by these tests.

The preserved production build contains 146 modules; the existing bundle-size
warning remains. General formulas, Cauer-to-Foster conversion and simulator
acceptance remain outside the supported numeric-table export contract. Current
selection rejects mixed additional gate-off/commutation contexts even when their
Vg/Rg values match; the context list displays that source evidence and the preview
reports the unsupported ambiguity.
