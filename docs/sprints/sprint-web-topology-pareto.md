# Sprint: fixed-boundary Boost semiconductor Pareto screen

Status: implemented in the browser workflow; focused validation passed; awaiting
independent review.

## Goal

Make the existing three-candidate semiconductor tradeoff in
[`topology_semiconductor_pareto.ipynb`](../../examples/topology_semiconductor_pareto.ipynb)
usable through the local web GUI. Evaluate every switch/rectifier configuration
at one shared Boost CCM boundary and compare channel-conduction loss,
package-body volume, and sourced one-unit cost.

## Scope delivered

- Added the reusable Python operation
  `calculate_boost_semiconductor_pareto` in
  `transistordatabase/topologies/semiconductor_pareto.py`. It delegates each
  row's electrical solve to the existing `loss-balanced` operating-point
  operation and does not mutate TDB objects.
- Added `POST /api/topology/pareto` with a typed request. It resolves each TDB
  device independently and returns lookup, curve-condition, rating, or catalog
  problems on the affected candidate while keeping valid candidates available.
- Added the Pareto panel to the Vue topology workflow. It sends a single SI
  boundary for all candidates, displays results and model limits, and downloads
  the boundary, candidate configurations, market sources, provenance, and
  outcomes as JSON.
- Kept price and package information as request/catalog metadata. No cost or
  dimension values are added to TDB records.
- Left the existing notebook and its saved outputs unchanged. Its three
  candidates provide a numerical regression for the web operation.

The initial browser values reproduce the notebook's fixed 100 V to 200 V,
1 kW, 20 kHz Boost case, with 1 mH, 1 mF, 25 °C junction temperature, and 60 °C
fixed case/sink reference. The same editable boundary is used for each row.
Frequency, inductance, and capacitance are converted from displayed kHz, µH,
and µF to Hz, H, and F before the API call. The API accepts 2–20 candidate
configurations with unique nonblank IDs and limits each parallel count to
1–100.

## Pareto and data rules

- The objectives are `channel_conduction_loss_w`,
  `package_body_volume_mm3`, and `component_cost_usd`; all are minimized.
- An eligible candidate is dominated only when another eligible candidate is
  no worse in every objective and strictly better in at least one. Exact
  objective ties remain non-dominated. Ineligible candidates do not affect the
  frontier.
- Component cost is one-unit USD list price per orderable physical package,
  multiplied by the switch and rectifier counts. The price URL and access date
  travel with the request and result. The defaults are the notebook's U.S.
  DigiKey snapshot accessed 2026-10-14.
- Package volume is `length × width × height` in mm³ per physical package,
  multiplied by the same counts. The dimensions source URL is retained.
- Catalog data must include a nonblank orderable part, positive finite one-unit
  cost, three positive finite dimensions, price source URL and date, and
  dimensions source URL. Missing or unusable market data makes that candidate
  ineligible; it does not insert a zero objective.
- Electrical calculations use exact TDB curve conditions and the existing
  no-extrapolation behavior. A missing device or unsupported curve condition is
  reported on that candidate. A curve failure does not become zero loss.
- The basic eligibility screen compares absolute blocking voltage and peak
  current per device against TDB absolute ratings. It does not establish SOA,
  thermal safety, derating, or design margin.

## Model limits shown to the user

Loss includes switch and rectifier channel conduction only. Switching,
reverse-recovery, gate-drive, passive, and magnetic losses are excluded. The
CCM solve power-balances the included channel losses and returns the averaged
inductor volt-second residual separately; nonlinear channel curves may leave a
nonzero residual. Parallel devices assume identical behavior and equal current
sharing. Temperature is a one-pass estimate at the selected curve condition
and remains unavailable when required thermal resistances are absent or
nonpositive. The fixed case/sink temperature is user input. Package volume is a
body bounding-box estimate and omits leads, board clearance, cooling, and
assembly. These are screening outputs, not a design recommendation.

## Acceptance and focused evidence

- The checked-in CREE and Infineon TDB records reproduce notebook losses
  18.849 W, 17.623 W, and 16.650 W; volt-second residuals 0.004662 V,
  0.004179 V, and 0.003705 V; volumes 853.2, 860.4, and 1294.1 mm³; and costs
  $19.28, $20.42, and $31.20. All three remain on the three-objective
  frontier.
- Python coverage checks exact objective ties, strict domination, voltage
  rating exclusion, missing exact gate curves, and candidate-local invalid price,
  dimensions, source URLs, and access dates. Malformed field types and non-finite
  JSON numbers are screened inside the reusable operation, so valid alternatives
  remain eligible.
- FastAPI coverage checks a mixed valid/invalid candidate request, missing
  TDB lookup, malformed row-level catalog values including string prices,
  overflowing numbers, and numeric source values, shared-boundary validation,
  and normalized duplicate IDs.
- Vue coverage checks SI conversion, catalog/source submission, tie-rule
  display, candidate-specific errors, assumptions, and JSON export.
- Chromium coverage checks the notebook defaults and result table, residual,
  source/access-date display, unavailable temperatures, limitation text,
  objective plots, and exported request/provenance/results. A separate flow
  sends the browser request to the running FastAPI endpoint and verifies its
  candidate-level response.

Focused validation on the implementation branch:

- `conda run -n ntbees2 python -m pytest --import-mode=importlib --noconftest tests/test_topology_pareto.py tests/test_topology_api.py -q`: 38 passed, including malformed catalog types, non-finite JSON numbers, URL/date validation, and mixed-candidate API behavior.
- `npm run test:run -- src/tests/unit/TopologyParetoWorkflow.test.js`: 2 passed.
- `npx playwright test src/tests/e2e/topology-pareto-workflows.spec.js --project=chromium`: 2 passed. One keeps deterministic results to verify the UI and export; the other sends a real browser request to FastAPI and verifies its response. FastAPI was started with `conda run -n ntbees2 python -m uvicorn ... --port 8002`; Playwright reused it because its Windows child-shell configuration could not resolve `conda` on PATH.
- `npm run build -- --outDir <new-empty-temp-directory>`: passed with Vite's existing >500 kB JavaScript chunk warning. The validation build used an inspected empty temporary output directory so it did not overwrite the repository's ignored `gui_web/dist` contents.
- `conda run -n ntbees2 ruff check --ignore B905 ...`: passed. The ignored `B905` findings are three pre-existing `zip()` calls elsewhere in `gui_web/backend/main.py`.

## Remaining gaps

This sprint provides only the browser Pareto workflow for the notebook's Boost
CCM boundary. It does not add total switching/recovery or passive loss
objectives, nonlinear exact volt-second closure, temperature iteration, SOA or
thermal qualification, cost-feed integration, or freshness checks for user
catalog snapshots. The checked-in example devices still lack matched
switching/recovery curves and usable positive interface thermal resistances,
so those terms and temperatures remain unavailable. The wider local-web
capability migration and screenshot parity remain open in the master plan.
