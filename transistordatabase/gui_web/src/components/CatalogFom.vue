<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { catalogApi } from '../services/api.js'
import CatalogScatter from './CatalogScatter.vue'
import { catalogCsv } from './catalogPresentation.js'
const props = defineProps({ suppliedCatalog: { type: Object, default: null } })
const sources = ref([]), metrics = ref({}), facets = ref({})
const result = ref(null), error = ref(''), loading = ref(false), selectedId = ref('')
const frontierOnly = ref(false), page = ref(1)
const normalizationMode = ref('none'), quoteCurrency = ref(''), quoteQuantity = ref('')
const thermalConvention = ref(false), sourceDeclarations = ref({})
const query = ref({ source_ids: [], panel_preset: 'default', x_metric: 'rds_ohm', y_metric: 'cost_fom', x_direction: 'min', y_direction: 'min', technology: '', manufacturer: '', search: '', voltage_min: '', voltage_max: '', current_min: '' })
let requestId = 0, timer
onBeforeUnmount(() => { ++requestId; clearTimeout(timer) })
const facet = computed(() => Object.fromEntries(['technology', 'manufacturer'].map(key => [key, [...new Set(query.value.source_ids.flatMap(id => facets.value[id]?.[key] || []))].sort()])))
const candidates = computed(() => result.value?.candidates || result.value?.points || [])
const points = computed(() => candidates.value.filter(p => !frontierOnly.value || p.pareto === true))
const rows = computed(() => points.value.slice((page.value - 1) * 40, page.value * 40))
const pages = computed(() => Math.max(1, Math.ceil(points.value.length / 40)))
const selected = computed(() => candidates.value.find(p => p.id === selectedId.value))
const selectedSource = computed(() => selected.value?.source || result.value?.sources?.find(s => s.id === selected.value?.source_id) || result.value?.source)
const resultSources = computed(() => result.value?.sources || (result.value ? [result.value.source] : []))
const declarationSources = computed(() => sources.value.filter(source => query.value.source_ids.includes(source.id)))
const workbookMetric = key => /^(thermal_|voltage_energy_|rce_|igbt_area_|diode_area_)/.test(key)
const workbookSelected = computed(() => [query.value.x_metric, query.value.y_metric].some(workbookMetric))
const currency = computed(() => result.value?.cost_cohort?.currency || (resultSources.value.length === 1 ? resultSources.value[0].currency === 'unspecified' ? 'unverified currency' : resultSources.value[0].currency : 'recorded currency'))
const panels = computed(() => {
  const byId = new Map(candidates.value.map(p => [p.id, p]))
  return (result.value?.panels || []).map(panel => ({ ...panel, points: panel.points.map(p => byId.get(p.id)).filter(p => p && (!frontierOnly.value || p.pareto === true)) }))
})
const fmt = value => value == null ? 'Missing' : value === 0 ? '0' : Math.abs(value) < 0.001 ? value.toExponential(3) : Number(value.toPrecision(5)).toLocaleString()
const label = key => `${metrics.value[key]?.label || key} (${metrics.value[key]?.unit || ''})`
const cellHeader = (cell, col) => cell.header ?? selectedSource.value?.headers?.find(header => header.column === col)?.label ?? ''
function request() {
  const payload = Object.fromEntries(Object.entries(query.value).map(([k, v]) => [k, ['voltage_min', 'voltage_max', 'current_min'].includes(k) ? (v === '' ? null : Number(v)) : v]))
  if (normalizationMode.value !== 'none') payload.normalization = { mode: normalizationMode.value, metrics: Object.keys(metrics.value) }
  if (quoteCurrency.value || quoteQuantity.value !== '') payload.quote_cohort = { currency: quoteCurrency.value, quote_quantity: Number(quoteQuantity.value) }
  const declarations = declarationSources.value.map(source => ({ source_id: source.id, currency: sourceDeclarations.value[source.id]?.currency?.trim().toUpperCase() || null, area_unit: sourceDeclarations.value[source.id]?.area_unit || null })).filter(entry => entry.currency || entry.area_unit)
  if (thermalConvention.value || declarations.length) payload.workbook_derivation = { thermal_convention: thermalConvention.value ? 'source-constant-model-100C' : null, source_declarations: declarations }
  if (props.suppliedCatalog) payload.snapshot = props.suppliedCatalog
  return payload
}
async function screen() {
  if (!query.value.source_ids.length) { loading.value = false; return }
  const id = ++requestId
  loading.value = true; error.value = ''
  try {
    const data = await catalogApi.screen(request())
    if (id !== requestId) return
    result.value = data; page.value = 1
  } catch (e) {
    if (id === requestId) { result.value = null; error.value = typeof e.response?.data?.detail === 'string' ? e.response.data.detail : e.message }
  } finally { if (id === requestId) loading.value = false }
}
watch([query, normalizationMode, quoteCurrency, quoteQuantity, thermalConvention, sourceDeclarations], () => {
  ++requestId; clearTimeout(timer); result.value = null; selectedId.value = ''; error.value = ''; loading.value = false
  timer = setTimeout(screen, 180)
}, { deep: true })
watch(frontierOnly, () => { page.value = 1 })
function useSuppliedCatalog(data) {
  thermalConvention.value = false; sourceDeclarations.value = {}
  sources.value = data.sources
  facets.value = Object.fromEntries(data.sources.map(source => [source.id, Object.fromEntries(['technology', 'manufacturer'].map(key => [key, [...new Set(data.records.filter(r => r.source_id === source.id).map(r => r[key]).filter(Boolean))].sort()]))]))
  query.value.source_ids = data.sources.map(s => s.id)
  sourceChanged()
}
watch(() => props.suppliedCatalog, data => { if (data) useSuppliedCatalog(data) })
onMounted(async () => {
  try {
    const data = await catalogApi.sources()
    sources.value = data.sources; metrics.value = data.metrics; facets.value = data.facets
    if (props.suppliedCatalog) useSuppliedCatalog(props.suppliedCatalog)
    else query.value.source_ids = [data.sources.find(s => s.id === 'SiC_MOSFET_FOM/1200V')?.id || data.sources[0]?.id].filter(Boolean)
  } catch (e) { error.value = `Catalog could not be loaded: ${e.message}` }
})
function sourceChanged() { query.value.technology = ''; query.value.manufacturer = '' }
function axisChanged(axis) { query.value[`${axis}_direction`] = metrics.value[query.value[`${axis}_metric`]].direction }
function selectRow(id) { selectedId.value = id }
function setDeclaration(sourceId, field, value) { sourceDeclarations.value = { ...sourceDeclarations.value, [sourceId]: { ...sourceDeclarations.value[sourceId], [field]: value } } }
const evidenceText = value => value == null ? 'Unknown / missing' : typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)
function download(format) {
  const data = format === 'json' ? JSON.stringify({ request: request(), result: result.value, visible_ids: points.value.map(p => p.id), display: { frontier_only: frontierOnly.value, legend_visibility_only: true } }, null, 2) : catalogCsv(result.value, metrics.value, points.value)
  const url = URL.createObjectURL(new Blob([data], { type: format === 'json' ? 'application/json' : 'text/csv;charset=utf-8' })), a = document.createElement('a')
  a.href = url; a.download = `catalog-fom.${format}`; a.click(); URL.revokeObjectURL(url)
}
const safeLink = url => typeof url === 'string' && /^https?:\/\//.test(url) ? url : undefined
const panelTitle = panel => panel.id === 'default-0' ? 'Chosen objectives' : `${metrics.value[panel.x_metric]?.label} and ${metrics.value[panel.y_metric]?.label}`
</script>

<template>
  <section class="fom-workspace">
    <header class="fom-title"><div><p class="eyebrow">SEMICONDUCTOR SELECTION</p><h2>FOM &amp; Pareto explorer</h2><p>Compare recorded catalog surveys with traceable figures of merit.</p></div><div class="exports"><button :disabled="!result" @click="download('csv')">Export CSV</button><button :disabled="!result" @click="download('json')">Export JSON</button></div></header>
    <div class="filter-panel">
      <p v-if="suppliedCatalog" class="wide">Supplied source snapshot. Original source hashes and rows remain attached. Bundled catalog data is unchanged.</p>
      <label class="wide">Catalog sources · select one or more<select v-model="query.source_ids" multiple size="5" aria-label="Catalog sources" @change="sourceChanged"><option v-for="s in sources" :key="s.id" :value="s.id">{{ s.file }} · {{ s.sheet }} ({{ s.record_count }})</option></select><span>Ctrl / Command + click selects multiple surveys. Physical points retain every source identity.</span></label>
      <label>Technology<select v-model="query.technology" aria-label="Technology"><option value="">All technologies</option><option v-for="t in facet.technology" :key="t">{{ t }}</option></select></label>
      <label>Manufacturer<select v-model="query.manufacturer" aria-label="Manufacturer"><option value="">All manufacturers</option><option v-for="m in facet.manufacturer" :key="m">{{ m }}</option></select></label>
      <label>Part or manufacturer<input v-model="query.search" aria-label="Part or manufacturer" placeholder="Search part number"></label>
      <label>Min voltage (V)<input v-model="query.voltage_min" aria-label="Min voltage" type="number" min="0"></label><label>Max voltage (V)<input v-model="query.voltage_max" aria-label="Max voltage" type="number" min="0"></label><label>Min catalog current (A)<input v-model="query.current_min" aria-label="Min catalog current" type="number" min="0" step="any"></label>
      <label>Survey layout<select v-model="query.panel_preset" aria-label="Survey layout"><option value="default">FOM overview · four panels</option><option value="survey">Original survey · six panels</option></select></label>
      <label>Axis values<select v-model="normalizationMode" aria-label="Axis values"><option value="none">Absolute SI values</option><option value="fixed">Relative · fixed selected-source population</option><option value="visible">Relative · filtered candidate population</option></select></label>
      <label>Comparable quote currency<input v-model="quoteCurrency" aria-label="Comparable quote currency" placeholder="e.g. CHF"></label><label>Recorded quote quantity<input v-model="quoteQuantity" aria-label="Recorded quote quantity" type="number" min="1" placeholder="Explicit quantity tier"></label>
    </div>
    <div class="objectives"><label>X metric<select v-model="query.x_metric" aria-label="X metric" @change="axisChanged('x')"><option v-for="(m,key) in metrics" :key="key" :value="key">{{ label(key) }}</option></select></label><label>X objective<select v-model="query.x_direction" aria-label="X objective"><option value="min">Minimize</option><option value="max">Maximize</option></select></label><label>Y metric<select v-model="query.y_metric" aria-label="Y metric" @change="axisChanged('y')"><option v-for="(m,key) in metrics" :key="key" :value="key">{{ label(key) }}</option></select></label><label>Y objective<select v-model="query.y_direction" aria-label="Y objective"><option value="min">Minimize</option><option value="max">Maximize</option></select></label></div>
    <details class="method workbook-controls" :open="workbookSelected" aria-label="Workbook derivation controls">
      <summary>Workbook estimates and source declarations</summary>
      <p>Thermal current is a conduction-only screening estimate. The model holds the recorded conduction resistance and threshold voltage constant at their source conditions.</p>
      <p>The thermal boundary is 100 °C. MOSFET estimates use junction-to-case resistance. IGBT estimates use junction-to-heatsink resistance. These estimates are not measured current ratings.</p>
      <label class="check"><input v-model="thermalConvention" type="checkbox" aria-label="Acknowledge thermal screening assumptions">I accept the constant source model and conduction-only assumptions for the 100 °C boundary.</label>
      <p>Source currency and chip area units remain unknown until declared. Declarations apply only to this calculation and remain separate from original cells.</p>
      <div v-for="source in declarationSources" :key="source.id" class="source-declaration">
        <strong>{{ source.file }} · {{ source.sheet }}</strong><span>Source: {{ source.id }} · Recorded currency: {{ source.currency === 'unspecified' || !source.currency ? 'Unknown' : source.currency }}</span>
        <div class="declaration-fields"><label>Declared quote currency<input :value="sourceDeclarations[source.id]?.currency || ''" :aria-label="`Declared currency ${source.id}`" maxlength="3" placeholder="Unknown · e.g. CHF" @input="setDeclaration(source.id, 'currency', $event.target.value)"></label><label>Declared source chip area unit<select :value="sourceDeclarations[source.id]?.area_unit || ''" :aria-label="`Declared area unit ${source.id}`" @change="setDeclaration(source.id, 'area_unit', $event.target.value)"><option value="">Unknown · no SI area metric</option><option value="m2">m²</option><option value="mm2">mm²</option><option value="cm2">cm²</option></select></label></div>
      </div>
      <p>The switching voltage / energy coefficient benchmark requires recorded turn-on, turn-off and recovery coefficients. A blank coefficient remains missing.</p>
      <p>This dimensional benchmark is not a maximum switching frequency. Test voltage differs from rated blocking voltage. IGBT rCE remains distinct from MOSFET Rds(on).</p>
    </details>
    <div v-if="workbookSelected" class="metric-notes" aria-label="Selected workbook metric definitions"><p v-for="key in [query.x_metric, query.y_metric].filter(workbookMetric)" :key="key"><strong>{{ label(key) }}</strong> · {{ metrics[key]?.description || metrics[key]?.basis || 'Calculated source benchmark. Inspect the selected row evidence and missing-input reasons.' }}</p></div>
    <p v-if="!query.source_ids.length">Select at least one catalog source.</p><p v-if="loading" role="status">Calculating catalog frontier…</p><p v-if="error" role="alert" class="error">{{ error }}</p>
    <template v-if="result">
      <div class="summary-cards" aria-live="polite"><div><strong>{{ result.matched_count }}</strong><span>matching catalog rows</span></div><div><strong>{{ result.plotted_count }}</strong><span>complete objective pairs</span></div><div><strong>{{ result.pareto_count }}</strong><span>Pareto candidates</span></div><div><strong>{{ result.excluded_count }}</strong><span>unavailable objective pairs</span></div></div>
      <div v-for="source in resultSources" :key="source.id" class="source-note"><strong>{{ source.file }} · {{ source.sheet }}</strong><span>{{ source.snapshot }}. Prices: {{ source.currency === 'unspecified' ? 'unverified currency' : source.currency }}. {{ source.notes }}</span><span>Source ID: {{ source.id }} · SHA-256: <code>{{ source.sha256 }}</code></span></div>
      <p v-if="result.cost_cohort?.status === 'unavailable'" class="cohort-warning" role="status">Cost comparison unavailable: {{ result.cost_cohort.reason }} Physical panels retain their own complete pairs.</p>
      <div class="plot-help"><label><input v-model="frontierOnly" type="checkbox"> Pareto candidates only</label><span>This display filter restricts every panel to the chosen objective frontier. Missing objectives have unavailable Pareto membership. Click a point or part to inspect across panels. Chart legends change visibility only; neither normalization population changes when a legend is hidden.</span></div>
      <div class="plot-grid"><CatalogScatter v-for="panel in panels" :key="panel.id" :title="panelTitle(panel)" :points="panel.points" :x-key="panel.x_metric" :y-key="panel.y_metric" :metrics="metrics" :selected-id="selectedId" :currency="currency" :relative="normalizationMode !== 'none'" :matched-count="panel.matched_count" :excluded-count="panel.excluded_count" :exclusions="panel.exclusions" @select="selectRow" /></div>
      <details v-if="result.normalization" class="method" open><summary>Normalization population and denominators</summary><p>{{ normalizationMode === 'fixed' ? 'Fixed population: all selected-source records before screen filters.' : 'Filtered population: candidates after the explicit screen filters.' }} Only positive valid operands contribute. Legend visibility and Pareto display do not change the denominator.</p><div class="raw-table"><table><thead><tr><th>Metric</th><th>Denominator · absolute SI</th><th>Population</th><th>Evidence</th></tr></thead><tbody><tr v-for="(e,key) in result.normalization.metrics" :key="key"><td>{{ label(key) }}</td><td>{{ fmt(e.denominator) }} · {{ e.status }}</td><td>{{ e.population_ids?.length }} rows</td><td><details><summary>IDs and hash</summary><p>{{ e.population_hash }}</p><p>Minimum record IDs: {{ e.denominator_ids?.join(', ') }}</p><p>Population IDs: {{ e.population_ids?.join(', ') }}</p><p v-for="ex in e.exclusions" :key="ex.id">{{ ex.id }}: {{ ex.reason }}</p></details></td></tr></tbody></table></div></details>
      <aside v-if="selected" class="device-detail" aria-label="Selected catalog device"><h3>{{ selected.part_number }} <small>{{ selected.manufacturer }} · {{ selected.technology }}</small></h3><p>{{ selected.package }} · {{ selected.source_id }} · {{ selectedSource?.sheet }} · Source row {{ selected.row }} · Quote quantity: {{ selected.quote_quantity ?? 'unrecorded' }} · Minimum order: {{ selected.min_quantity ?? 'unrecorded' }}</p><p>Snapshot: {{ selectedSource?.snapshot }} · SHA-256: <code>{{ selectedSource?.sha256 }}</code></p><p>Pareto: {{ selected.pareto == null ? 'Unavailable for chosen objectives' : selected.pareto ? 'Frontier' : 'Dominated' }}</p><dl><template v-for="(value,key) in selected.conditions" :key="key"><dt>{{ key.replaceAll('_',' ') }}</dt><dd>{{ value ?? 'Unrecorded' }}</dd></template></dl><a v-if="safeLink(selected.datasheet)" :href="safeLink(selected.datasheet)" target="_blank" rel="noopener noreferrer">Source datasheet</a><details><summary>Original cells and cached values</summary><div class="raw-table"><table><thead><tr><th>Column</th><th>Header</th><th>Value</th><th>Original formula</th></tr></thead><tbody><tr v-for="(cell,col) in selected.raw" :key="col"><td>{{ col }}</td><td>{{ cellHeader(cell, col) }}</td><td>{{ cell.error ?? cell.value ?? 'Missing' }}</td><td>{{ selected.formulas?.[col] ?? cell.formula ?? '' }}</td></tr></tbody></table></div></details></aside>
      <div class="table-heading"><h3>Catalog results</h3><span>All filtered candidates · unavailable objective values stay visible</span></div><div class="results-table"><table><thead><tr><th>Part number</th><th>Technology</th><th>Manufacturer</th><th>{{ label(query.x_metric) }} · absolute</th><th>{{ label(query.y_metric) }} · absolute</th><th v-if="normalizationMode !== 'none'">Relative X / Y</th><th>Pareto</th><th>Source / row</th></tr></thead><tbody><tr v-for="p in rows" :key="p.id" :class="{selected:selectedId===p.id}"><td><button class="part-button" @click="selectRow(p.id)">{{ p.part_number }}</button></td><td>{{ p.technology }}</td><td>{{ p.manufacturer }}</td><td>{{ fmt(p.metrics[query.x_metric]) }}</td><td>{{ fmt(p.metrics[query.y_metric]) }}</td><td v-if="normalizationMode !== 'none'">{{ fmt(p.normalization?.[query.x_metric]?.relative) }} / {{ fmt(p.normalization?.[query.y_metric]?.relative) }}</td><td>{{ p.pareto == null ? 'Unavailable' : p.pareto ? 'Yes' : 'Dominated' }}</td><td>{{ p.source_id }} / {{ p.row }}</td></tr></tbody></table><p v-if="!points.length">No candidate rows match these filters. Choose another metric or source.</p></div><div class="pagination"><button :disabled="page === 1" @click="page--">Previous</button><span>Page {{ page }} of {{ pages }}</span><button :disabled="page === pages" @click="page++">Next</button></div>
      <aside v-if="selected" class="device-detail workbook-evidence" aria-label="Selected metric evidence"><h3>Selected metric values and input evidence</h3><div class="raw-table"><table><thead><tr><th>Metric · absolute SI</th><th>Value</th><th>Availability / missing input</th></tr></thead><tbody><tr v-for="key in [query.x_metric, query.y_metric]" :key="key"><td>{{ label(key) }}</td><td>{{ fmt(selected.metrics[key]) }}</td><td>{{ selected.metric_exclusions?.[key] || (selected.metrics[key] == null ? 'Unavailable · inspect source evidence' : 'Available · source conditions apply') }}</td></tr></tbody></table></div><details v-if="selected.workbook_evidence" open><summary>Workbook inputs, declarations and calculation evidence</summary><dl class="evidence-list"><template v-for="(value,key) in selected.workbook_evidence" :key="key"><dt>{{ key.replaceAll('_', ' ') }}</dt><dd><pre>{{ evidenceText(value) }}</pre></dd></template></dl></details></aside>
      <details class="method"><summary>Metric definitions, data coverage and assumptions</summary><p>Cost FOM = price × Rds(on) / Vdss², recomputed using each row’s blocking voltage. Rds × Qg and Rds × Coss require both recorded operands. Pareto membership uses absolute objectives and retains tied points. Catalog ratings are screening data. Unequal or unknown test conditions remain visible.</p><p v-for="a in result.assumptions" :key="a">{{ a }}</p><p v-for="(count,key) in result.coverage" :key="key">{{ label(key) }}: {{ count }} / {{ result.matched_count }} rows</p></details>
    </template>
  </section>
</template>
<style scoped>
.fom-workspace { --fom-bg: #f5f7fb; --fom-card: #fff; --fom-line: #dbe2ec; --fom-text: #1b2941; --fom-muted: #52647c; color: var(--fom-text); background: var(--fom-bg); padding: 28px; border-radius: 14px; }
:global(.dark-theme) .fom-workspace { --fom-bg: #161f2c; --fom-card: #202d40; --fom-line: #3c4c65; --fom-text: #e5edf9; --fom-muted: #b1c1d8; }
.fom-title, .exports, .plot-help, .table-heading, .pagination { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.eyebrow { font-size: 11px; letter-spacing: 1.8px; font-weight: 700; color: #0b8278; margin: 0; }
h2 { font-size: 28px; margin: 6px 0; } h3 { font-size: 17px; } p { line-height: 1.55; }
.fom-title p:last-child { color: var(--fom-muted); margin-top: 8px; }
button { border: 1px solid var(--fom-line); border-radius: 6px; padding: 8px 13px; background: var(--fom-card); color: var(--fom-text); cursor: pointer; }
button:disabled { opacity: .45; cursor: default; } button:hover:enabled { border-color: #088275; }
.filter-panel, .objectives { display: grid; gap: 14px; padding: 20px; border: 1px solid var(--fom-line); border-radius: 10px; background: var(--fom-card); margin: 18px 0; grid-template-columns: repeat(4, minmax(0, 1fr)); }
.wide { grid-column: span 2; } label { display: flex; flex-direction: column; gap: 6px; font-size: 12px; font-weight: 600; }
select, input:not([type=checkbox]) { width: 100%; box-sizing: border-box; border: 1px solid var(--fom-line); border-radius: 5px; padding: 9px; color: var(--fom-text); background: var(--fom-card); min-width: 0; }
.objectives { grid-template-columns: minmax(0, 2fr) minmax(0, 1fr) minmax(0, 2fr) minmax(0, 1fr); }
.summary-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.summary-cards div { padding: 18px; border: 1px solid var(--fom-line); border-radius: 10px; background: var(--fom-card); }
.summary-cards strong { font-size: 30px; display: block; font-variant-numeric: tabular-nums; }
.summary-cards span { color: var(--fom-muted); font-size: 12px; }
.source-note { padding: 14px 0; font-size: 12px; color: var(--fom-muted); line-height: 1.6; min-width: 0; overflow-wrap: anywhere; }.source-note span { display: block; }
.plot-help { align-items: start; font-size: 12px; margin-bottom: 16px; color: var(--fom-muted); }.plot-help label { flex-direction: row; align-items: center; white-space: nowrap; }.plot-help span { max-width: 700px; line-height: 1.5; }
.plot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.device-detail, .method { background: var(--fom-card); border: 1px solid var(--fom-line); border-radius: 10px; padding: 20px; margin-top: 20px; overflow-wrap: anywhere; }
.device-detail h3 { margin-top: 0; }.device-detail small { display: block; font-size: 13px; font-weight: 400; padding-top: 6px; }
dl { display: grid; grid-template-columns: 150px 1fr; gap: 8px; font-size: 13px; } dt { text-transform: capitalize; } dd { margin: 0; }
details { margin-top: 14px; font-size: 13px; } summary { cursor: pointer; font-weight: 600; } a { color: #078b82; }
.table-heading { margin-top: 24px; }.table-heading span { font-size: 12px; color: var(--fom-muted); }
.results-table, .raw-table { overflow: auto; max-height: 520px; border: 1px solid var(--fom-line); border-radius: 8px; background: var(--fom-card); }
table { border-collapse: collapse; width: 100%; font-size: 12px; } th { background: var(--fom-bg); text-align: left; position: sticky; top: 0; } td, th { padding: 12px; border-bottom: 1px solid var(--fom-line); } tr.selected { background: #d6eee9; color: #142e2a; }
.part-button { border: 0; background: transparent; color: inherit; padding: 0; text-align: left; text-decoration: underline; font-family: monospace; }
.pagination { justify-content: flex-end; margin-top: 12px; font-size: 12px; }.error { color: #b42332; }.method p { color: var(--fom-muted); }
.check { flex-direction: row; align-items: flex-start; line-height: 1.5; }.check input { flex: 0 0 auto; margin-top: 3px; }.source-declaration { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--fom-line); }.source-declaration > span { display: block; font-size: 12px; margin-top: 6px; }.declaration-fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 12px; }.metric-notes { padding: 0 8px; font-size: 13px; overflow-wrap: anywhere; }.evidence-list { grid-template-columns: minmax(100px, 1fr) minmax(0, 3fr); }.evidence-list pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }.workbook-controls { margin-bottom: 20px; }
@media (max-width: 850px) { .fom-workspace { padding: 14px; }.filter-panel, .objectives { grid-template-columns: repeat(2, minmax(0, 1fr)); }.plot-grid { grid-template-columns: 1fr; }.summary-cards { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 500px) { .declaration-fields, .evidence-list { grid-template-columns: minmax(0, 1fr); }.evidence-list dt { margin-top: 12px; } }
</style>

