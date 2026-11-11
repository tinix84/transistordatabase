# Catalog source files

`transistordatabase/catalog_sources.py` owns bounded CSV/OOXML reading, explicit
column mapping, immutable snapshot creation, source comparisons, static links and
documented workbook switching coefficients. It is headless. The typed adapter is
`transistordatabase/gui_web/backend/catalog_source_routes.py`.

All endpoints are POST under `/api/catalog-sources`. Files arrive as exact bytes
in standard `content_base64`, with a metadata `filename`. Filenames are never
opened as server paths. The server writes no files and refreshes no stored catalog.
The caller downloads returned JSON and may pass its catalog `sources`/`records`
to the FOM screening adapter. Existing bundled catalogs remain unchanged.

## Request and result contract

Every file request has:

```json
{
  "filename": "survey.csv",
  "content_base64": "UGFydDtWDQpQMTs2MDBWDQo=",
  "encoding": "utf-8-sig",
  "delimiter": ";",
  "decimal": "."
}
```

CSV supports declared UTF-8 (with optional BOM), Windows-1252 or UTF-16 encoding;
comma, semicolon, tab or pipe separators; and a declared dot/comma decimal.
Standard quoted separators, embedded newlines and doubled quotes are preserved.
There is no sorting, deduplication, delimiter sniffing, thousands-separator guess,
magnitude-based unit inference, formula evaluation or external query refresh.
OOXML supports `.xlsx` and `.xlsm`; VBA, Power Query, external workbook references
and hyperlink targets are retained as source bytes and are never executed/fetched.
Legacy binary `.xls` is explicitly unsupported.

- `/preview`: optionally `preview_rows` (1–200, default20). Returns source
  SHA-256/size, complete worksheet census, headers with stable column IDs and
  duplicate flags, hidden-row IDs, formula/error/native-link counts, mapping
  fields/units, and the first requested rows. Each cell has `column`, `cell`,
  `value`, `cached_value`, `formula`, and `error`. Missing formula caches remain
  missing; saved errors are source evidence, not recalculated results.
- `/snapshot`: add `sheet`, `header_row` (default1), `mapping`, and `metadata`.
  Mapping keys are catalog fields; values are `{column,unit}`. Use Excel letters
  for OOXML and one-based numeric strings for CSV, so duplicate labels are harmless.
  `part_number` is required. Text fields use unit `text`. Numeric fields declare
  a unit from the preview's supported list. Bare numbers use that unit; text
  suffixes can name an enumerated SI unit of the same field (e.g. `1200V`,
  `1.2kV`, `41mOhm @ 32A, 8V`). Original condition tails remain in raw cell
  evidence; they do not establish an operating-point model. Wrong decimal
  convention, unknown units, missing values and negative ratings/prices receive
  per-row reasons and `null`. A real stored zero remains zero. A missing part ID
  rejects the row with a reason. Hidden rows are retained.
- `/diff`: `{before:{records:[...]}, after:{records:[...]}, key_fields:[...]}`.
  Returns added/missing record IDs and changed fields for common keys. Choose
  fields explicitly from text catalog fields or `row`; duplicate or absent keys
  reject instead of silently collapsing parts. Include `row` to disambiguate
  repeated part numbers. This compares supplied records only; it applies nothing.
- `/links`: optional `cells:[{sheet,cell}]`; omitted means all static links.
  Returns stored relationship links, internal locations, literal `HYPERLINK`
  strings and plain URL text as distinct kinds. Selected cells have `links:[]`
  when absent, and multiple links remain a list. Relationship ranges cover their
  cells. Relative targets remain raw and explicitly relative to the original
  workbook; they are never resolved against localhost. Literal display text,
  tooltip and target mode are retained. `GetURL` locations are separately listed
  with `not-evaluated` status; nonliteral HYPERLINK formulas are separately
  unsupported. Neither is promoted to an observed link.
- `/coefficients`: returns documented `MOSFET`/`IGBT` layouts with `Part-No.`
  header and exact Y/Z/AA energy headers. Each entry is **J/A**, with original
  cell formula/cache/error, source header/address, X-column switching voltage,
  175°C header condition, and unknown gate resistance. Supported literal
  positive-denominator divisions expose independent numerator energy and reference
  current. For `MOSFET!Y2 = .000632/36`, the coefficient is
  `1.7555555555555556e-5 J/A`, reference36 A, and numerator.000632 J.
  No arbitrary formula is calculated. Numeric cells remain source coefficients
  with unknown reference current. Cache absence and cache/formula disagreement
  have explicit statuses; neither is silently recalculated or overwritten.

Example mapping:

```json
{
  "sheet": "CSV",
  "mapping": {
    "part_number": {"column": "1", "unit": "text"},
    "voltage_v": {"column": "2", "unit": "V"},
    "rds_ohm": {"column": "3", "unit": "mOhm"}
  },
  "metadata": {
    "currency": "CHF",
    "snapshot": "Historical quote; acquisition date unknown",
    "quote_quantity": 1000,
    "technology": "SiC",
    "conditions": {"rds_vgs_v": 18, "rds_tj_c": 25}
  }
}
```

Metadata is supplied, not guessed from timestamps or device names. Unknown
currency stays `unspecified`; quote tier0 is retained and is not a confirmed
purchasing tier. Source-specific historical quotations do not become current prices.

## Snapshot identity and limits

The returned `schema_version:1` catalog uses the existing `sources`/`records`
shape. Extra top-level evidence is `snapshot_id`, `version`, `mapping`,
`source_file`, and `row_errors`. The source bytes are embedded exactly once as
`source_file.content_base64`, with their SHA-256. A snapshot ID hashes parser
version, file identity, sheet, explicit mapping, header row, parsing conventions
and declared metadata. New bytes or choices create a new snapshot. Records retain
`source_id`, row, original cell evidence, hidden state and source provenance;
their IDs include original-byte hash+sheet+row. Duplicate part numbers stay
distinct. No part-number-only merge is performed.

Limits are 8 MB uploaded bytes, 2,000 ZIP members, 48 MB total expanded OOXML,
200,000 workbook cells, 10,000 rows and 256 columns. ZIP/XML checks precede
workbook parsing; duplicate members, encrypted files, entities and false worksheet
dimensions that exclude stored cells reject explicitly. Comparisons permit at
most10,000 records per side. Exceeding supported limits produces422, not a partial
successful import. Preview does not emit images or the original byte blob.

## Evidence and current limits

Focused tests cover quoted/BOM CSV, SI suffixes and condition tails, declared
decimals, missing versus zero, duplicate headers/parts, hidden rows, caches,
formula errors/disagreement, relative/internal/literal/plain/missing hyperlinks,
source comparisons, unsupported formats, resource guards and HTTP equality.
They read all five actual supplied files without changes and reconcile all eight
catalog populations:286/174/1137/233/500/99/23/38. The FOM workbook retains
3,631 formulas and1,077 hidden rows. XLSM static inspection finds59 native links
(23MOSFET+36IGBT) and separately63 GetURL formulas. Every original hash is checked
before and after inspection.

This backend does not author/recalculate Excel workbooks, execute macros or
Power Query, evaluate general formulas, infer unknown conditions, convert these
J/A coefficients into measured E(I) tables, or perform automatic source refresh.
Browser mapping/preview/export acceptance belongs to the subsequent UI slice.
