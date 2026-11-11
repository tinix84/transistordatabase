# FOM visual comparison contract

Read-only preparation, 2026-10-16. This supplements section 8H of
`semiconductor-data-workflow-contract.md`; it does not advance the implementation
plan or replace the immutable structural baseline. No notebook was executed.
Cell numbers below are zero-based JSON cell indices; line numbers are within
the cell source.

## Evidence and actual gaps

The reference root is
`\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc`.

| Reference | Actual visual capability / limit |
| --- | --- |
| `tests\pareto_sc.ipynb`, cell 2 lines 1–25 and 28–51 | Two three-panel Bokeh grids: voltage/current, voltage/power, voltage/price; power/current, power/price, price/Rds. Shared `ColumnDataSource` and part/index tooltips. These are scatter surveys, not a computed Pareto frontier. |
| Same, cell 7 lines 2–17 and cells 11–12; cell 14 lines 2–25 and cells 15–16 | GaN and SiC are loaded into separate surveys, each replacing `source`. Stored Bokeh HTML/JavaScript outputs exist for both grids. There is no implemented combined GaN/SiC/Si overlay, three-dimensional chart, or gate-charge chart in this notebook. Cell 5's earlier two-plot example is commented out. |
| Same, cell 1 lines 1–6; cells 7/14 parsing | The unused relative-cost helper mistakenly assigns resistance from price and uses an incorrect expression. Regex parsing also loses decimals/units or produces missing values. Preserve visual intent, not these defects. |
| `docs\Mosfet_switching_losses.ipynb`, cell 27 lines 1–5 | A single current-versus-time turn-on plot has a stored PNG. Cells 17–26 explain gate/threshold/Miller intervals; cell 28 is only a turn-off heading. No catalog/FOM overlay exists here. This intent belongs to the accepted gate workbench and its separate scientific evidence. |
| FOM workbook `Tabelle1!C2:C311` and `xl/charts/chart1.xml` | Actual minimum-normalized relative cost versus blocking voltage; logarithmic cost axis. Population/standalone-summary lineage and source error limitations are already specified in 8H and the workbook audit. |

Notebook SHA-256 values: Pareto
`986d0459a34b87f043a7e43580b8056025609d3a82cadde6d1c6cd7c3c8effc6`;
switching notebook
`a8c1ce8cc7b16df267feef82f9fa24823da7ebf926759069bd2d2ebe3ff6cacf`.

Current `catalog_fom.py:128` accepts one `source_id`; line 166 isolates it.
Selectable metrics already reproduce all six old scatter pairs individually.
The four current panels, linked point inspection, log axes, zoom, PNG and
CSV/JSON exports improve on those old static survey groups. However,
`catalog_fom.py:186` removes rows missing either chosen objective, and
`CatalogFom.vue:13,109–112` supplies that reduced list to every panel. Thus a row
with missing price disappears even from a complete voltage/current panel. This
is a real coverage gap. The fixed panels also omit the original voltage/power
and power/current overview pairing, although both remain selectable axes.

The current technology-colored renderer (`CatalogScatter.vue`) can display
mixed technologies already present in one source. Combining selected source
surveys would be a useful extension consistent with the user's requested
improvement, rather than proof of a feature implemented in the old notebook.

## Lean extension within phase 8

1. Return filtered **candidate records** independently from chosen-objective
   pairs. Each panel derives its own complete pairs, excluded count and reason
   from candidates. Retain exact chosen-objective Pareto membership; a candidate
   missing an objective has `pareto=null`/unavailable, never "dominated".
   Label an explicit Pareto-only display filter as a restriction to that
   objective population. Add an optional original-survey preset for the six
   pairs using the same scatter component; no second plotting engine.
2. Allow an explicit list of source IDs for physical axes (voltage, catalog
   current/power, Rds, Qg, Ciss/Coss and their physical products). Reuse the
   existing screening/Pareto engine. Keep source+sheet+row identities, hashes,
   snapshot labels, technology and original condition text on every point and
   export. Do not merge duplicate part numbers or promote catalog ratings to
   operating-point loss. Unknown conditions remain unknown; unequal Rds/Qg
   test conditions stay visible. An optional comparable-condition cohort must
   require explicitly matching known conditions rather than treating unknown
   values as equal measured conditions. No derived Vloss metric is justified
   without a declared operating point/model.
3. Price-related panels/axes use a separately declared quote cohort. A single
   source preserves today's behavior. Across sources require known common
   currency and explicitly compatible recorded quantity tier, plus visible
   acquisition/snapshot age; missing currency or quantity cannot establish
   compatibility. Reject an unsupported mixed cohort with an actionable reason,
   or keep separately labeled source panels. Never silently convert currency,
   equate minimum order with quote tier, or borrow one source's currency for
   another row. Cost exclusions must not remove physical-only points.
4. Apply 8H's positive minimum normalization to a declared fixed population or
   an explicitly selected visible population. Return absolute and relative
   values, denominator and its record IDs, population IDs/hash and exclusions.
   Hiding points or selecting a chart legend must not silently change a fixed
   denominator. Normalizing compatible physical values across selected sources
   is allowed; cost values first require the quote policy above. Normalization
   does not change Pareto membership for the same eligible population.

## Independent acceptance

- Synthetic sources A/B contain physical points A1=(1,4), B1=(2,2), B2=(4,1),
  and A2=(3,5), minimizing both axes: combined frontier is A1/B1/B2, A2 is
  dominated. Two identical coordinate rows retain distinct IDs and both ties.
  All per-point source hashes/conditions survive JSON and CSV.
- Add a row with voltage=650 V, current=20 A, Rds=0.05 ohm, missing price/Qg.
  It is absent from Rds/price and Rds/Qg, present in voltage/current with
  objective Pareto unavailable. Each panel reports its own correct coverage.
- Mixed CHF/EUR and unknown-currency/unknown-tier cost comparisons fail clearly;
  the same records remain usable for physical axes. Compatible explicit CHF
  tiers succeed without conversion. Differing source snapshots remain visible.
- Values [2,4,8] normalize to [1,2,4]. Hiding 2 under fixed population retains
  [2,4]; explicitly selecting visible population produces [1,2] with a new
  denominator/population. Verify ties, empty populations and invalid operands.
- Browser: select GaN+SiC physical comparison, inspect a point across panels,
  change axes/log scales, use the original-survey preset, enforce a cost-cohort
  error, change normalization population and export JSON/CSV/PNG. Assert stale
  responses cannot overwrite new filters; verify mobile overflow and chart
  teardown. Test actual point counts and exported identities/denominator rather
  than screenshots alone. Reference notebook/workbook hashes stay unchanged.

Implementation should extend the existing core/API/components and tests, not
introduce a generic dashboard framework or copy reference notebook bodies.
