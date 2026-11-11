<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { catalogSourceApi } from '../services/api.js'

const emit = defineEmits(['explore-catalog'])
const MAX_SNAPSHOT_BYTES = 50000000
const file = ref(null), preview = ref(null), snapshot = ref(null), links = ref(null), coefficients = ref(null)
const encoding = ref('utf-8-sig'), delimiter = ref(','), decimal = ref('.')
const sheetName = ref(''), headerRow = ref(1), mapping = ref({}), previewPage = ref(1)
const metadata = ref({ label: '', currency: '', snapshot: '', quote_quantity: '', technology: 'Unknown', workbook_profile: null })
const conditions = ref([{ name: 'junction_temperature_c', value: '' }, { name: 'gate_voltage_v', value: '' }, { name: 'gate_resistance_ohm', value: '' }])
const busy = ref(''), error = ref(''), notice = ref(''), linkCells = ref('')
const before = ref(null), after = ref(null), difference = ref(null), diffError = ref(''), keys = ref([])
let fileVersion = 0, requestVersion = 0, diffVersion = 0, beforeVersion = 0, afterVersion = 0
const selectedSheet = computed(() => preview.value?.sheets.find(s => s.name === sheetName.value))
const header = computed(() => selectedSheet.value?.rows.find(r => r.row === Number(headerRow.value))?.cells || [])
const columns = computed(() => {
  const labels = header.value.map(c => String(c.value ?? ''))
  return header.value.map(c => ({ ...c, duplicate: labels.filter(v => v === String(c.value ?? '')).length > 1 }))
})
const previewRows = computed(() => selectedSheet.value?.rows.slice((previewPage.value - 1) * 20, previewPage.value * 20) || [])
const previewPages = computed(() => Math.max(1, Math.ceil((selectedSheet.value?.rows.length || 0) / 20)))
const fields = computed(() => Object.keys(preview.value?.mapping_fields || {}))
const mappingReady = computed(() => !!mapping.value.part_number && Object.values(mapping.value).every(spec => !!spec.unit))
const comparisonKeys = ['part_number', 'manufacturer', 'technology', 'package', 'datasheet', 'row']
const display = value => value === null || value === undefined || value === '' ? 'Unknown / missing' : String(value)
const label = value => value.replaceAll('_', ' ')
const message = e => typeof e.response?.data?.detail === 'string' ? e.response.data.detail : e.message

function invalidate() {
  ++requestVersion
  snapshot.value = null; busy.value = ''; error.value = ''; notice.value = ''
}
watch([encoding, delimiter, decimal], () => { invalidate(); preview.value = null; links.value = null; coefficients.value = null; mapping.value = {} }, { flush: 'sync' })
watch([sheetName, headerRow], () => { invalidate(); mapping.value = {}; previewPage.value = 1 }, { flush: 'sync' })
watch([mapping, metadata, conditions], invalidate, { deep: true, flush: 'sync' })
watch(linkCells, () => { ++requestVersion; links.value = null; busy.value = '' }, { flush: 'sync' })
watch(keys, () => { ++diffVersion; difference.value = null; diffError.value = '' }, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { ++fileVersion; ++requestVersion; ++diffVersion; ++beforeVersion; ++afterVersion })

function readFile(selected) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).split(',')[1])
    reader.onerror = () => reject(new Error('Cannot read the selected file.'))
    reader.readAsDataURL(selected)
  })
}
async function selectFile(event) {
  const token = ++fileVersion, selected = event.target.files?.[0]
  invalidate(); file.value = null; preview.value = null; links.value = null; coefficients.value = null; mapping.value = {}
  if (!selected) return
  if (selected.size > 8000000) { error.value = 'Select a source file of at most 8,000,000 bytes.'; return }
  busy.value = 'Reading source bytes'
  try {
    const content_base64 = await readFile(selected)
    if (token === fileVersion) file.value = { filename: selected.name, content_base64 }
  } catch (e) { if (token === fileVersion) error.value = message(e) }
  finally { if (token === fileVersion) busy.value = '' }
}
const sourceRequest = () => ({ ...file.value, encoding: encoding.value, delimiter: delimiter.value, decimal: decimal.value })
async function run(operation, payload, accept) {
  const token = ++requestVersion
  error.value = ''; busy.value = operation
  try {
    const result = await catalogSourceApi[operation](payload)
    if (token === requestVersion) accept(result)
  } catch (e) { if (token === requestVersion) error.value = message(e) }
  finally { if (token === requestVersion) busy.value = '' }
}
function inspect() {
  preview.value = null; snapshot.value = null; links.value = null; coefficients.value = null
  return run('preview', { ...sourceRequest(), preview_rows: 200 }, result => {
    preview.value = result
    sheetName.value = result.sheets[0]?.name || ''; headerRow.value = 1; mapping.value = {}
  })
}
function setColumn(field, column) {
  if (!column) { const next = { ...mapping.value }; delete next[field]; mapping.value = next; return }
  const units = preview.value.mapping_fields[field]
  mapping.value = { ...mapping.value, [field]: { column, unit: units[0] === 'text' ? 'text' : '' } }
}
function createSnapshot() {
  const context = Object.fromEntries(conditions.value.filter(c => c.name.trim()).map(c => [c.name.trim(), c.value.trim() || null]))
  const meta = { ...metadata.value, currency: metadata.value.currency.trim() || null,
    snapshot: metadata.value.snapshot.trim() || null, label: metadata.value.label.trim() || null,
    quote_quantity: metadata.value.quote_quantity === '' ? null : Number(metadata.value.quote_quantity), conditions: context }
  snapshot.value = null
  return run('snapshot', { ...sourceRequest(), sheet: sheetName.value, header_row: Number(headerRow.value), mapping: mapping.value, metadata: meta }, result => { snapshot.value = result })
}
function inspectLinks() {
  links.value = null
  const cells = linkCells.value.trim() ? linkCells.value.split(/[\s,]+/).filter(Boolean).map(cell => ({ sheet: sheetName.value, cell })) : undefined
  return run('links', { ...sourceRequest(), ...(cells ? { cells } : {}) }, result => { links.value = result })
}
function inspectCoefficients() {
  coefficients.value = null
  return run('coefficients', sourceRequest(), result => { coefficients.value = result })
}
function download(value, filename, type = 'application/json') {
  const blob = value instanceof Blob ? value : new Blob([JSON.stringify(value)], { type })
  const url = URL.createObjectURL(blob), anchor = document.createElement('a')
  anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url)
}
function downloadSource() {
  const source = snapshot.value?.source_file || file.value
  const bytes = Uint8Array.from(atob(source.content_base64), c => c.charCodeAt(0))
  download(new Blob([bytes]), source.filename)
}
function explore(value = snapshot.value) {
  emit('explore-catalog', { schema_version: 1, sources: value.sources, records: value.records })
}
function useSnapshot(side) {
  if (side === 'before') { ++beforeVersion; before.value = snapshot.value }
  else { ++afterVersion; after.value = snapshot.value }
  ++diffVersion; difference.value = null; diffError.value = ''
}
async function uploadSnapshot(event, side) {
  const token = side === 'before' ? ++beforeVersion : ++afterVersion, selected = event.target.files?.[0]
  ++diffVersion; difference.value = null; diffError.value = ''
  if (side === 'before') before.value = null
  else after.value = null
  if (!selected) return
  try {
    if (selected.size > MAX_SNAPSHOT_BYTES) throw new Error('Select a snapshot of at most 50,000,000 bytes.')
    const value = JSON.parse(await selected.text(), (_key, item) => {
      if (typeof item === 'number' && !Number.isFinite(item)) throw new Error('The snapshot contains a nonfinite numeric value.')
      return item
    })
    if (token !== (side === 'before' ? beforeVersion : afterVersion)) return
    const catalog = value.catalog || value
    if (catalog.schema_version !== 1 || !Array.isArray(catalog.records) || !Array.isArray(catalog.sources)) throw new Error('Select a version 1 catalog snapshot with sources and records.')
    if (side === 'before') before.value = catalog
    else after.value = catalog
  } catch (e) { if (token === (side === 'before' ? beforeVersion : afterVersion)) diffError.value = message(e) }
}
async function compare() {
  const token = ++diffVersion
  difference.value = null; diffError.value = ''
  try {
    const payload = { before: { records: before.value.records }, after: { records: after.value.records }, key_fields: [...keys.value] }
    if (new TextEncoder().encode(JSON.stringify(payload)).byteLength > MAX_SNAPSHOT_BYTES) throw new Error('The combined comparison exceeds 50,000,000 bytes. Select smaller snapshots.')
    const result = await catalogSourceApi.diff(payload)
    if (token === diffVersion) difference.value = result
  } catch (e) { if (token === diffVersion) diffError.value = message(e) }
}
</script>

<template>
  <section class="source-workspace">
    <header><p class="eyebrow">SOURCE IMPORT BROWSER</p><h2>Catalog source import</h2><p>Preview a supplied file. Map columns and units. Create a separate, versioned catalog snapshot.</p></header>
    <p class="source-note">Source files stay unchanged. Macros, Power Query, formulas and external targets are never executed or fetched. Quotes retain their recorded context.</p>
    <section class="card" aria-label="Source file controls">
      <h3>1. Select source bytes</h3>
      <div class="grid">
        <label class="wide">Source file · CSV, TSV, XLSX or XLSM<input aria-label="Catalog source file" type="file" accept=".csv,.tsv,.xlsx,.xlsm" @change="selectFile"></label>
        <label>CSV encoding<select v-model="encoding" aria-label="CSV encoding"><option value="utf-8-sig">UTF-8 with optional BOM</option><option value="utf-8">UTF-8</option><option value="cp1252">Windows-1252</option><option value="utf-16">UTF-16</option></select></label>
        <label>CSV separator<select v-model="delimiter" aria-label="CSV separator"><option value=",">Comma</option><option value=";">Semicolon</option><option :value="'\t'">Tab</option><option value="|">Pipe</option></select></label>
        <label>Decimal convention<select v-model="decimal" aria-label="Decimal convention"><option value=".">Dot</option><option value=",">Comma</option></select></label>
      </div>
      <div class="actions"><button :disabled="!file || !!busy" @click="inspect">Preview source</button><button :disabled="!file || !!busy" @click="downloadSource">Download exact source bytes</button></div>
    </section>
    <p v-if="busy" role="status">{{ busy }}…</p><p v-if="error" role="alert" class="error">{{ error }}</p>
    <template v-if="preview">
      <section class="card">
        <h3>2. Inspect sheets and original cells</h3>
        <p>{{ preview.source_file.filename }} · {{ preview.source_file.size_bytes }} bytes · SHA-256: <code>{{ preview.source_file.sha256 }}</code></p>
        <div class="table-wrap" tabindex="0" aria-label="Worksheet census"><table><thead><tr><th>Sheet</th><th>State</th><th>Rows</th><th>Hidden row IDs</th><th>Formulas / errors / stored links</th></tr></thead><tbody><tr v-for="s in preview.sheets" :key="s.name"><td>{{ s.name }}</td><td>{{ s.state }}</td><td>{{ s.row_count }}</td><td>{{ s.hidden_rows.join(', ') || 'None' }}</td><td>{{ s.formula_count }} / {{ s.error_count }} / {{ s.native_link_count }}</td></tr></tbody></table></div>
        <div class="grid"><label>Worksheet<select v-model="sheetName" aria-label="Source worksheet"><option v-for="s in preview.sheets" :key="s.name">{{ s.name }}</option></select></label><label>Header row · preview range 1–200<input v-model.number="headerRow" aria-label="Header row" type="number" min="1" :max="Math.min(200, selectedSheet?.row_count || 1)"></label></div>
        <p>Stable column IDs distinguish duplicate headers. The preview contains at most 200 rows. Snapshot mapping includes every source row.</p>
        <div class="table-wrap preview-table" tabindex="0" aria-label="Source cell preview"><table><thead><tr><th>Row</th><th v-for="c in columns" :key="c.column">{{ c.column }} · {{ display(c.value) }} <strong v-if="c.duplicate">(duplicate header)</strong></th></tr></thead><tbody><tr v-for="row in previewRows" :key="row.row"><th>{{ row.row }}{{ row.hidden ? ' · hidden' : '' }}</th><td v-for="cell in row.cells" :key="cell.column"><span class="cell-value">{{ display(cell.value) }}</span><small>{{ cell.cell }}</small><small v-if="cell.formula">Formula: {{ cell.formula }}</small><small v-if="cell.formula">Saved cache: {{ display(cell.cached_value) }}</small><strong v-if="cell.error" class="error">Source error: {{ cell.error }}</strong></td></tr></tbody></table></div>
        <div class="actions"><button :disabled="previewPage === 1" @click="previewPage--">Previous preview rows</button><span>{{ previewPage }} / {{ previewPages }}</span><button :disabled="previewPage === previewPages" @click="previewPage++">Next preview rows</button></div>
      </section>
      <section class="card">
        <h3>3. Map columns and declare source context</h3>
        <p>Map the part number. Select each numeric unit explicitly. Unmapped fields and empty context values remain unknown.</p>
        <div class="mapping-grid"><div v-for="field in fields" :key="field" class="mapping-row"><label>{{ label(field) }}{{ field === 'part_number' ? ' · required' : '' }}<select :value="mapping[field]?.column || ''" :aria-label="`Map ${field}`" @change="setColumn(field, $event.target.value)"><option value="">Unmapped</option><option v-for="c in columns" :key="c.column" :value="c.column">{{ c.column }} · {{ display(c.value) }}{{ c.duplicate ? ' (duplicate)' : '' }}</option></select></label><label>Unit<select v-if="mapping[field]" v-model="mapping[field].unit" :aria-label="`Unit ${field}`"><option v-if="preview.mapping_fields[field][0] !== 'text'" value="">Select source unit</option><option v-for="unit in preview.mapping_fields[field]" :key="unit">{{ unit }}</option></select><span v-else>—</span></label></div></div>
        <div class="grid">
          <label>Source label<input v-model="metadata.label" aria-label="Source label" placeholder="Optional source description"></label><label>Technology<input v-model="metadata.technology" aria-label="Source technology"></label>
          <label class="wide">Conduction model profile<select v-model="metadata.workbook_profile" aria-label="Workbook conduction profile"><option :value="null">Unknown · no workbook derivation</option><option value="mosfet">MOSFET · RthJC to case</option><option value="igbt">IGBT · RthJH to heatsink</option></select><span>Select a profile only when the mapped source fields support that model. Source temperature and gate voltage remain unchanged.</span></label>
          <label>Recorded currency<input v-model="metadata.currency" aria-label="Recorded currency" placeholder="Unknown unless supplied"></label><label>Recorded quote quantity (pieces)<input v-model="metadata.quote_quantity" aria-label="Recorded quote quantity" type="number" min="0" step="any" placeholder="Unknown"></label>
          <label class="wide">Snapshot / quote date and provenance<input v-model="metadata.snapshot" aria-label="Source snapshot context" placeholder="Acquisition date unknown unless supplied"></label>
        </div>
        <h4>Source test conditions</h4><div v-for="(condition, index) in conditions" :key="index" class="condition"><label>Condition name<input v-model="condition.name" :aria-label="`Condition name ${index + 1}`"></label><label>Recorded value and unit<input v-model="condition.value" :aria-label="`Condition value ${index + 1}`" placeholder="Unknown"></label></div><button @click="conditions.push({ name: '', value: '' })">Add condition</button>
        <div class="actions"><button :disabled="!mappingReady || !!busy || !header.length" @click="createSnapshot">Create catalog snapshot</button></div>
      </section>
      <section v-if="snapshot" class="card" aria-label="Created catalog snapshot">
        <h3>Versioned snapshot</h3><p>{{ snapshot.records.length }} accepted rows · {{ snapshot.row_errors.length }} rows with rejected values or identifiers</p><p>Version: <code>{{ snapshot.version }}</code></p>
        <div class="actions"><button @click="download(snapshot, `catalog-${snapshot.version}.json`)">Download catalog JSON</button><button :disabled="!snapshot.records.length" @click="explore()">Explore snapshot in FOM</button><button @click="useSnapshot('before')">Use as before snapshot</button><button @click="useSnapshot('after')">Use as after snapshot</button></div>
        <div v-if="snapshot.row_errors.length" class="table-wrap" tabindex="0" aria-label="Snapshot row errors"><table><thead><tr><th>Source row</th><th>Field / cell</th><th>Reason</th></tr></thead><tbody><template v-for="row in snapshot.row_errors" :key="row.row"><tr v-for="(e,i) in row.errors" :key="`${row.row}-${i}`"><td>{{ row.row }}</td><td>{{ e.field }} · {{ e.cell || 'Missing cell' }}</td><td>{{ e.reason }}</td></tr></template></tbody></table></div>
      </section>
      <section class="card">
        <h3>Static links and switching coefficients</h3><p>Link targets remain text. Relative targets refer to the original workbook, never this server.</p>
        <label>Optional cell batch · selected worksheet<input v-model="linkCells" aria-label="Link cell batch" placeholder="A2, B3 · leave empty for all links"></label><div class="actions"><button :disabled="!!busy" @click="inspectLinks">Inspect static links</button><button :disabled="!!busy" @click="inspectCoefficients">Inspect J/A coefficients</button></div>
        <template v-if="links"><p>{{ links.native_link_count }} stored link objects · {{ links.geturl_formula_count }} GetURL formulas (not evaluated)</p><button @click="download(links, 'catalog-static-links.json')">Download link evidence</button><div class="table-wrap" tabindex="0" aria-label="Static link evidence"><table><thead><tr><th>Sheet / cell</th><th>Kind</th><th>Target / location</th><th>Resolution / mode</th><th>Display / tooltip</th></tr></thead><tbody><template v-for="(entry,i) in links.entries" :key="i"><tr v-for="(link,j) in (entry.links === undefined ? [entry] : entry.links.length ? entry.links : [{ kind: 'absent', target: null }])" :key="j"><td>{{ entry.sheet }}!{{ entry.cell }}</td><td>{{ link.kind }}</td><td>{{ display(link.target) }}<br v-if="link.location">{{ link.location }}</td><td>{{ link.resolution }} · {{ link.target_mode }}</td><td>{{ display(link.display) }} · {{ display(link.tooltip) }}</td></tr></template></tbody></table></div><details><summary>Unevaluated link formulas</summary><p v-for="entry in [...links.geturl_formula_locations, ...links.unsupported_hyperlink_formulas]" :key="`${entry.sheet}-${entry.cell}`">{{ entry.sheet }}!{{ entry.cell }} · {{ entry.formula }} · {{ entry.status }}</p></details></template>
        <template v-if="coefficients"><p>{{ coefficients.scope }}</p><p>{{ coefficients.coefficients.length }} coefficient records · unit J/A. A coefficient is not an energy at every current.</p><button @click="download(coefficients, 'catalog-switching-coefficients.json')">Download coefficient evidence</button><div class="table-wrap" tabindex="0" aria-label="Switching coefficient evidence"><table><thead><tr><th>Part / cell / component</th><th>Coefficient (J/A)</th><th>Saved cache (J/A)</th><th>Reference I / V / T</th><th>Numerator energy / Rg</th><th>Formula / header / status</th></tr></thead><tbody><tr v-for="c in coefficients.coefficients" :key="`${c.sheet}-${c.cell.cell}`"><td>{{ c.part_number }}<br>{{ c.sheet }}!{{ c.cell.cell }} · {{ c.component }}</td><td>{{ display(c.recognized_literal_coefficient_j_per_a) }}</td><td>{{ display(c.cached_coefficient_j_per_a) }}<br>{{ display(c.cell.error) }}</td><td>{{ display(c.reference_current_a) }} A<br>{{ display(c.reference_voltage_v) }} V · {{ c.reference_voltage_cell?.cell }}<br>{{ display(c.reference_temperature_c) }} °C · {{ c.reference_temperature_header_cell }}</td><td>{{ display(c.literal_energy_j) }} J<br>Rg: {{ display(c.gate_resistance_ohm) }}</td><td>{{ display(c.cell.formula) }}<br>{{ c.header.value }}<br>{{ c.status }}</td></tr></tbody></table></div></template>
      </section>
    </template>
    <section class="card" aria-label="Snapshot comparison">
      <h3>Compare supplied snapshots before applying a refresh</h3><p>Upload two catalog JSON snapshots or use the snapshot buttons above. Choose unique comparison keys explicitly. Comparison changes no stored catalog.</p>
      <div class="grid"><label>Before snapshot JSON<input aria-label="Before snapshot JSON" type="file" accept=".json,application/json" @change="uploadSnapshot($event, 'before')"><span>{{ before ? `${before.records.length} records loaded` : 'No snapshot' }}</span></label><label>After snapshot JSON<input aria-label="After snapshot JSON" type="file" accept=".json,application/json" @change="uploadSnapshot($event, 'after')"><span>{{ after ? `${after.records.length} records loaded` : 'No snapshot' }}</span></label></div>
      <fieldset><legend>Explicit comparison keys</legend><label v-for="key in comparisonKeys" :key="key" class="check"><input v-model="keys" type="checkbox" :value="key" :aria-label="`Compare key ${key}`">{{ label(key) }}</label></fieldset><p>Repeated part numbers need an additional key, such as source row. Duplicate or missing keys produce an error.</p>
      <div class="actions"><button :disabled="!before || !after || !keys.length" @click="compare">Compare snapshots</button></div><p v-if="diffError" role="alert" class="error">{{ diffError }}</p>
      <template v-if="difference"><p>{{ difference.added.length }} added · {{ difference.changed.length }} changed · {{ difference.missing.length }} missing</p><p>{{ difference.application }}</p><button @click="download(difference, 'catalog-refresh-comparison.json')">Download comparison</button><div class="table-wrap" tabindex="0" aria-label="Snapshot differences"><table><thead><tr><th>Change</th><th>Record / comparison key</th><th>Fields</th></tr></thead><tbody><tr v-for="id in difference.added" :key="`a-${id}`"><td>Added</td><td>{{ id }}</td><td>New source row</td></tr><tr v-for="c in difference.changed" :key="c.before_id"><td>Changed</td><td>{{ c.key.join(' · ') }}<br>{{ c.before_id }} → {{ c.after_id }}</td><td>{{ c.fields.join(', ') }}</td></tr><tr v-for="id in difference.missing" :key="`m-${id}`"><td>Missing</td><td>{{ id }}</td><td>Absent from supplied after snapshot</td></tr></tbody></table></div><button :disabled="!after.records.length" @click="explore(after)">Explore after snapshot in FOM</button></template>
    </section>
  </section>
</template>

<style scoped>
.source-workspace { max-width: 100%; min-width: 0; color: var(--text-primary); padding: 24px; background: var(--bg-primary); border-radius: 12px; overflow-wrap: anywhere; }
.source-workspace h2 { font-size: 28px; margin: 6px 0; }.source-workspace h3 { margin-bottom: 12px; }.source-workspace h4 { margin: 16px 0 8px; }.source-workspace p { margin: 10px 0; }.eyebrow { color: #0b8278; font-size: 12px; letter-spacing: 1.5px; font-weight: 700; }.source-note { color: var(--text-secondary); }
.card { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 10px; padding: 20px; margin: 18px 0; min-width: 0; }.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 14px 0; }.wide { grid-column: span 2; }label { display: flex; flex-direction: column; gap: 6px; font-size: 13px; font-weight: 600; min-width: 0; }
input, select { min-width: 0; max-width: 100%; width: 100%; padding: 9px; border: 1px solid var(--border-color); border-radius: 5px; background: var(--bg-secondary); color: var(--text-primary); }input[type=file] { font-size: 12px; }button { border: 1px solid var(--border-color); padding: 9px 13px; border-radius: 6px; background: var(--bg-secondary); color: var(--text-primary); cursor: pointer; }button:disabled { opacity: .5; cursor: default; }.actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
.mapping-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 14px; }.mapping-row { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0,1fr); gap: 8px; }.condition { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; margin: 10px 0; }
.table-wrap { max-width: 100%; overflow: auto; max-height: 480px; border: 1px solid var(--border-color); margin: 14px 0; }.table-wrap:focus { outline: 2px solid var(--accent-blue); }table { border-collapse: collapse; width: 100%; min-width: 640px; font-size: 12px; }td, th { text-align: left; vertical-align: top; border-bottom: 1px solid var(--border-color); padding: 10px; min-width: 110px; max-width: 360px; }th { background: var(--bg-tertiary); position: sticky; top: 0; }.preview-table td { min-width: 180px; }.cell-value { white-space: pre-wrap; }small { display: block; margin-top: 6px; color: var(--text-secondary); }code { overflow-wrap: anywhere; }.error { color: #b42332; }.check { flex-direction: row; display: inline-flex; align-items: center; margin: 8px 14px 8px 0; }.check input { width: auto; }fieldset { padding: 12px; border: 1px solid var(--border-color); }details { margin-top: 12px; }summary { cursor: pointer; }
@media(max-width: 750px) { .source-workspace { padding: 10px; }.card { padding: 12px; }.grid,.mapping-grid,.condition { grid-template-columns: minmax(0,1fr); }.wide { grid-column: span 1; }.source-workspace h2 { font-size: 24px; }button { max-width: 100%; white-space: normal; } }
</style>
