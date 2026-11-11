# Catalog screening API usage

Use `GET /api/catalog/sources` to select source IDs and metric names. Submit
`POST /api/catalog/screen` with either `source_id` or `source_ids`. Do not send
both selection forms.

```json
{
  "source_ids": ["201210_gan_digikey/CSV", "SiC_MOSFET_FOM/1200V"],
  "x_metric": "voltage_v",
  "y_metric": "current_a",
  "x_direction": "max",
  "y_direction": "max",
  "panel_preset": "survey",
  "normalization": {
    "mode": "fixed",
    "metrics": ["voltage_v", "current_a"]
  }
}
```

For an imported source, supply `snapshot` with `schema_version: 1`, `sources`,
and `records` from the catalog source snapshot response. Select IDs from that
snapshot. This request does not change the bundled catalog. Retain source
SHA-256, source snapshot ID, sheet, and row when you export the result.

Read `candidates` for all filtered rows. Read `points` for complete chosen
objective pairs. Each panel contains compact point references with `id`, `x`,
and `y`. Resolve each ID against `candidates` to inspect original source cells,
conditions, and normalization values. A candidate with incomplete objectives
has `pareto: null`. A panel reports its own exclusions independently.

For mixed sources, price axes require an explicit quote cohort:

```json
"quote_cohort": {"currency": "CHF", "quote_quantity": 10000}
```

Use only a common known currency and matching recorded row quantity tiers.
Minimum order quantity does not establish a quote tier. Inspect `cost_cohort`
for unavailable comparisons and incompatible row IDs. Physical panels retain
their complete pairs when cost comparison is unavailable. Historical single
source prices retain their recorded limitations. No currency conversion occurs.

Choose normalization deliberately:

- `fixed` uses selected source records before search and rating filters.
  Optional `population_ids` restricts this denominator population to explicit IDs.
- `visible` uses candidates after search and rating filters.
- Omit `normalization` for absolute values.

Read `normalization.metrics` for each denominator, all tied minimum IDs,
population IDs, population hash, and exclusions. Candidate normalization entries
contain absolute and relative values. Missing or unrepresentable derived values
remain unavailable with a reason. Legend visibility and the Pareto display
filter do not change either declared denominator population.

Use the explorer's original survey layout for the six historical scatter pairs.
Use JSON export for the request, source evidence, panel populations, and visible
IDs. CSV export contains source identity, original condition text, cohort, and
denominator evidence. Its first row contains source descriptors, normalization
exclusions, and the full population and denominator ID lists. Later rows
reference that evidence through the population hash.
An empty export contains one `evidence_only` row. A PNG records the visible chart. Catalog ratings
and rating products are screening quantities, not operating point loss estimates.
