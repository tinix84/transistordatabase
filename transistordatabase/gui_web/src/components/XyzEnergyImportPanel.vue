<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { analysisApi } from '../services/api.js'
import EnergyConditionsEditor from './EnergyConditionsEditor.vue'
import EnergyScene3D from './EnergyScene3D.vue'
import { analysisCsv, analysisError, downloadAnalysis, finiteDecimal } from './analysisFiles.js'

const MAX_BYTES = 4_000_000
const input = reactive({ input_kind: 'paste', text: '', source: '', source_kind: 'unknown', energy_kind: 'e_on', common: false,
  delimiter: 'csv', has_header: false, current_column: 0, voltage_column: 1, energy_column: 2,
  current_unit: 'A', voltage_unit: 'V', energy_unit: 'J', missing_tokens: '',
  conditions: { t_j_c: null, v_g_on_v: null, v_g_off_v: null, r_g_on_ohm: null, r_g_off_ohm: null,
    commutation_device: null, commutation_source: null, energy_convention: '' } })
const file = ref(null), result = ref(null), rawJson = ref(''), acceptedBytes = ref(null)
const error = ref(''), stale = ref(false), busy = ref(false), selectedCell = ref(null), rowPage = ref(0), selectedPage = ref(0), fullEvidence = ref(false)
const currentPage = ref(0), voltagePage = ref(0)
watch(result, () => { currentPage.value = 0; voltagePage.value = 0 })
let revision = 0, disposed = false
function invalidate() {
  revision++; stale.value = !!result.value || busy.value; result.value = null; rawJson.value = ''; acceptedBytes.value = null
  error.value = ''; busy.value = false; selectedCell.value = null; rowPage.value = 0; fullEvidence.value = false
}
watch(input, invalidate, { deep: true, flush: 'sync' })
watch(file, invalidate, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
function selectFile(event) { file.value = event.target.files?.[0] ?? null }
function base64(bytes) {
  let binary = ''
  for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768))
  return btoa(binary)
}
async function parse() {
  const token = ++revision; error.value = ''; result.value = null; rawJson.value = ''; acceptedBytes.value = null; busy.value = true; stale.value = false
  selectedCell.value = null; rowPage.value = 0; fullEvidence.value = false
  try {
    if (!input.common) throw new Error('Confirm that the declared conditions apply to every source row.')
    if (!input.source.trim()) throw new Error('Enter a source identifier.')
    if (!input.conditions.energy_convention.trim()) throw new Error('Enter the energy convention.')
    if (new Set([input.current_column, input.voltage_column, input.energy_column]).size !== 3) throw new Error('Assign a different column to each axis.')
    const { text, ...configuration } = input
    if (input.input_kind === 'paste' && text.length > MAX_BYTES) throw new Error('The source exceeds the 4 MB byte limit.')
    const snapshot = JSON.parse(JSON.stringify(configuration)), sourceFile = file.value
    for (const key of ['t_j_c','v_g_on_v','v_g_off_v','r_g_on_ohm','r_g_off_ohm']) if (snapshot.conditions[key] !== null) snapshot.conditions[key] = finiteDecimal(snapshot.conditions[key], key)
    if (snapshot.input_kind === 'file' && !sourceFile) throw new Error('Select an XYZ source file.')
    if (snapshot.input_kind === 'file' && sourceFile.size > MAX_BYTES) throw new Error('The source exceeds the 4 MB byte limit.')
    const bytes = snapshot.input_kind === 'file' ? new Uint8Array(await sourceFile.arrayBuffer()) : new TextEncoder().encode(text)
    if (disposed || token !== revision) return
    if (!bytes.length) throw new Error('Enter source triples or select a file.')
    if (bytes.length > MAX_BYTES) throw new Error('The source exceeds the 4 MB byte limit.')
    const payload = { content_base64: base64(bytes), source_name: snapshot.input_kind === 'file' ? sourceFile.name : 'pasted-xyz.txt',
      input_kind: snapshot.input_kind, source: snapshot.source, source_kind: snapshot.source_kind, energy_kind: snapshot.energy_kind,
      conditions: snapshot.conditions, options: { delimiter: snapshot.delimiter, decimal: '.', has_header: snapshot.has_header,
        columns: { current: snapshot.current_column, voltage: snapshot.voltage_column, energy: snapshot.energy_column },
        units: { current: snapshot.current_unit, voltage: snapshot.voltage_unit, energy: snapshot.energy_unit },
        missing_energy_tokens: snapshot.missing_tokens === '' ? [] : snapshot.missing_tokens.split(',').map(value => value.trim()) } }
    const response = await analysisApi.importXyzEnergy(payload)
    if (disposed || token !== revision) return
    result.value = response.data; rawJson.value = response.rawJson; acceptedBytes.value = bytes
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
const maximum = computed(() => result.value ? result.value.cells.reduce((max, row) => row.reduce((value, cell) => cell.status === 'available' ? Math.max(value, cell.energy_j) : value, max), 0) : 0)
const voltageIndices = computed(() => result.value ? result.value.voltages_v.map((_, index) => index).reverse().slice(voltagePage.value * 16, (voltagePage.value + 1) * 16) : [])
const matrixCurrents = computed(() => result.value?.currents_a.slice(currentPage.value * 16, (currentPage.value + 1) * 16) ?? [])
const visibleRows = computed(() => result.value?.rows.slice(rowPage.value * 100, (rowPage.value + 1) * 100) ?? [])
const pages = computed(() => Math.ceil((result.value?.rows.length ?? 0) / 100))
const rowIndex = computed(() => new Map((result.value?.rows ?? []).map(row => [row.id, row])))
const selectedRows = computed(() => selectedCell.value?.row_ids.slice(selectedPage.value * 100, (selectedPage.value + 1) * 100).map(id => rowIndex.value.get(id)) ?? [])
watch(selectedCell, () => { selectedPage.value = 0 })
function number(value) { return value === null || value === undefined ? 'NULL' : String(value) }
function color(cell) { return cell.status === 'available' ? { backgroundColor: `rgba(13,148,180,${maximum.value === 0 ? .12 : .12 + .68 * (cell.energy_j / maximum.value)})` } : {} }
function csv(kind) {
  const evidence = result.value
  const metadata = { source_sha256: evidence.source.sha256, request_sha256: evidence.request_sha256, source: evidence.source.source,
    source_kind: evidence.source.source_kind, energy_kind: evidence.energy_kind, conditions: evidence.conditions, options: evidence.options, null_convention: 'NULL means unknown or unavailable; 0 is recorded zero' }
  const rows = kind === 'cells' ? evidence.cells.flat() : evidence.rows
  // Metadata appears on the first data record. Later records keep its columns blank.
  const blankMetadata = Object.fromEntries(Object.keys(metadata).map(key => [key, '']))
  const parts = [], encoder = new TextEncoder(); let bytes = 0
  for (let index = 0; index < rows.length; index++) {
    const record = analysisCsv([{ ...(index === 0 ? metadata : blankMetadata), ...rows[index], energy_j: rows[index].energy_j === null ? 'NULL' : rows[index].energy_j }])
    const part = index === 0 ? record : '\r\n' + record.slice(record.indexOf('\r\n') + 2)
    bytes += encoder.encode(part).length
    if (bytes > 24_000_000) { error.value = 'The CSV exceeds the 24 MB download limit. Use the exact response JSON and original source downloads.'; return }
    parts.push(part)
  }
  downloadAnalysis(parts.join(''), `xyz-energy-${kind}.csv`, 'text/csv')
}
function downloadSource() { downloadAnalysis(acceptedBytes.value, result.value.source.name, 'application/octet-stream') }
</script>

<template>
  <section class="xyz-panel" data-testid="xyz-energy">
    <h3>Import XYZ energy observations</h3>
    <p>Import three columns for magnitude current, voltage and energy. Python validates the source and preserves every original row.</p>
    <p>No interpolation applies. A blank energy or a declared missing token is unknown. A missing coordinate is a hole. Recorded zero remains zero.</p>
    <form @submit.prevent="parse">
      <section class="panel"><h4>Original source</h4>
        <label>Source input<select v-model="input.input_kind" aria-label="XYZ source input"><option value="paste">Paste UTF-8 text</option><option value="file">Original file bytes</option></select></label>
        <label v-if="input.input_kind === 'paste'">XYZ source text<textarea v-model="input.text" rows="7" aria-label="XYZ source text" placeholder="9,600,90&#10;3,200,30&#10;0,600,12&#10;9,200,60&#10;0,200,0"></textarea></label>
        <label v-else>XYZ source file<input type="file" aria-label="XYZ source file" @change="selectFile"><span v-if="file">{{ file.name }} · {{ file.size }} bytes</span></label>
        <p>File hashes cover the original bytes. Paste hashes cover the UTF-8 encoding of the entered text. Files must contain UTF-8 text.</p>
        <div class="fields"><label>Source identifier<input v-model="input.source" aria-label="XYZ source identifier" required></label>
          <label>Source kind<select v-model="input.source_kind" aria-label="XYZ source kind"><option v-for="kind in ['unknown','manual','measured','datasheet','simulated']" :key="kind" :value="kind">{{ kind }}</option></select></label>
          <label>Energy kind<select v-model="input.energy_kind" aria-label="XYZ energy kind"><option value="e_on">Turn-on E_on</option><option value="e_off">Turn-off E_off</option><option value="e_rr">Reverse recovery E_rr</option></select></label></div>
      </section>
      <section class="panel"><h4>Explicit parser options and units</h4>
        <div class="fields"><label>Delimiter<select v-model="input.delimiter" aria-label="XYZ delimiter"><option value="csv">CSV (comma)</option><option value="tsv">TSV (tab)</option><option value="whitespace">Whitespace</option></select></label>
          <label>Declared missing energy tokens (comma separated)<input v-model="input.missing_tokens" aria-label="XYZ missing energy tokens" placeholder="NA"></label></div>
        <label class="check"><input v-model="input.has_header" type="checkbox" aria-label="XYZ has header">The source has one header record</label>
        <p>The decimal separator is a period (.). Each data record must have exactly three columns. No unit inference or sign conversion applies.</p>
        <div class="fields"><div v-for="axis in ['current','voltage','energy']" :key="axis">
          <label>{{ axis }} column<select v-model="input[`${axis}_column`]" :aria-label="`XYZ ${axis} column`"><option v-for="column in [0,1,2]" :key="column" :value="column">Column {{ column + 1 }}</option></select></label>
          <label>{{ axis }} source unit<select v-model="input[`${axis}_unit`]" :aria-label="`XYZ ${axis} unit`"><option v-for="unit in ({current:['A','mA'],voltage:['V','kV'],energy:['J','mJ','µJ','uJ']})[axis]" :key="unit" :value="unit">{{ unit }}</option></select></label>
        </div></div>
      </section>
      <section class="panel"><h4>Common source conditions</h4><EnergyConditionsEditor v-model="input.conditions" prefix="XYZ" preserve-numeric-text/>
        <p>Blank conditions remain unknown. Enter the energy convention. Declare one common condition set for all rows.</p>
        <label class="check"><input v-model="input.common" type="checkbox" aria-label="XYZ common conditions">These conditions apply to every source row</label>
      </section>
      <p>Limits: 4 MB source bytes, 100,000 rows, 256 current coordinates, 128 voltage coordinates, 16,384 cells and 24 MB response.</p>
      <button type="submit" :disabled="busy" data-testid="xyz-parse">{{ busy ? 'Parsing source…' : 'Import XYZ observations' }}</button>
    </form>
    <p v-if="error" role="alert" data-testid="xyz-error">{{ error }}</p>
    <p v-if="stale" role="status">The source or options changed. Import again to restore the result and downloads.</p>
    <section v-if="result" data-testid="xyz-results">
      <h3>Accepted {{ result.energy_kind }} observations</h3>
      <p class="hash">Source SHA-256: {{ result.source.sha256 }} · {{ result.source.hash_scope }} · {{ result.source.byte_count }} bytes</p>
      <p class="hash">Request SHA-256: {{ result.request_sha256 }} · {{ result.method_version }}</p>
      <div class="actions"><button @click="downloadAnalysis(rawJson, 'xyz-energy-review.json', 'application/json;charset=utf-8')" data-testid="xyz-json">Download exact response JSON</button>
        <button @click="csv('rows')" data-testid="xyz-rows-csv">Download normalized source CSV</button><button @click="csv('cells')" data-testid="xyz-cells-csv">Download cell CSV</button>
        <button @click="downloadSource" data-testid="xyz-source-download">Download exact original source</button></div>
      <p data-testid="xyz-census">{{ result.census.source_rows }} source rows · {{ result.census.available_cells }} available cells · {{ result.census.missing_cells }} holes · {{ result.census.unknown_cells }} unknown cells · {{ result.census.ambiguous_cells }} ambiguous cells · {{ result.census.total_cells }} total cells</p>
      <EnergyScene3D :result="result" mode="xyz"/>
      <h4>Observed energy matrix</h4><p>I [A] increases to the right. V [V] increases upward. Each cell shows energy [J]. Rows use the returned cells[V index][I index].</p>
      <div class="legend" aria-label="Linear energy color legend"><span>0 J</span><span class="gradient"></span><span>{{ number(maximum) }} J</span></div>
      <p>Color scale: linear. Hole = no source row. Unknown = source row without energy. Ambiguous = repeated coordinate, including identical values.</p>
      <p v-if="result.currents_a.length > 16 || result.voltages_v.length > 16">The matrix shows 16 current coordinates and 16 voltage coordinates per page. All cells remain in the downloads.</p>
      <div class="actions"><template v-if="result.currents_a.length > 16"><button :disabled="currentPage === 0" @click="currentPage--">Previous current columns</button><span>Current page {{ currentPage + 1 }} / {{ Math.ceil(result.currents_a.length / 16) }}</span><button :disabled="(currentPage + 1) * 16 >= result.currents_a.length" @click="currentPage++">Next current columns</button></template><template v-if="result.voltages_v.length > 16"><button :disabled="voltagePage === 0" @click="voltagePage--">Previous voltage rows</button><span>Voltage page {{ voltagePage + 1 }} / {{ Math.ceil(result.voltages_v.length / 16) }}</span><button :disabled="(voltagePage + 1) * 16 >= result.voltages_v.length" @click="voltagePage++">Next voltage rows</button></template></div>
      <div class="table-scroll"><table class="heatmap" aria-label="Observed XYZ energy matrix" data-testid="xyz-matrix"><thead><tr><th scope="col">V [V] / I [A]</th><th v-for="(current,index) in matrixCurrents" :key="index" scope="col">{{ number(current) }} A</th></tr></thead>
        <tbody><tr v-for="vIndex in voltageIndices" :key="vIndex" :data-voltage="result.voltages_v[vIndex]"><th scope="row">{{ number(result.voltages_v[vIndex]) }} V</th>
          <td v-for="(cell,iIndex) in result.cells[vIndex].slice(currentPage * 16, (currentPage + 1) * 16)" :key="iIndex" :class="cell.status" :style="color(cell)" :data-current="cell.current_a" :data-voltage="cell.voltage_v" :data-energy="number(cell.energy_j)" :data-status="cell.status">
            <button @click="selectedCell = cell" :aria-label="`${cell.current_a} A, ${cell.voltage_v} V: ${number(cell.energy_j)} J, ${cell.status}`">{{ number(cell.energy_j) }} J<br>{{ cell.status === 'missing' ? 'hole' : cell.status }}<br><small>{{ cell.row_ids.slice(0,3).join(', ') }}{{ cell.row_ids.length > 3 ? ` · ${cell.row_ids.length} rows` : '' }}</small></button>
          </td></tr></tbody></table></div>
      <section v-if="selectedCell" class="panel" data-testid="xyz-cell-evidence"><h4>Cell evidence: {{ selectedCell.current_a }} A, {{ selectedCell.voltage_v }} V</h4><p>{{ selectedCell.status }} · {{ number(selectedCell.energy_j) }} J · {{ selectedCell.row_ids.length }} source rows · {{ selectedCell.reason }}</p>
        <p>Each page displays at most 100 original rows. All row IDs and values remain in the exact response download.</p>
        <div v-if="selectedCell.row_ids.length > 100" class="actions"><button :disabled="selectedPage === 0" @click="selectedPage--">Previous cell rows</button><span>Page {{ selectedPage + 1 }} / {{ Math.ceil(selectedCell.row_ids.length / 100) }}</span><button :disabled="(selectedPage + 1) * 100 >= selectedCell.row_ids.length" @click="selectedPage++">Next cell rows</button></div>
        <pre v-for="row in selectedRows" :key="row.id">{{ JSON.stringify(row, null, 2) }}</pre></section>
      <details><summary>Accepted source, options, units, conditions and limits</summary><pre>{{ JSON.stringify({ source: result.source, header: result.header, options: result.options, units: result.units, conditions: result.conditions, census: result.census, limits: result.limits }, null, 2) }}</pre></details>
      <h4>Original row evidence</h4><p>All {{ result.rows.length }} rows remain in the accepted response and downloads. This table displays at most 100 rows per page.</p>
      <div v-if="pages > 1" class="actions"><button :disabled="rowPage === 0" @click="rowPage--">Previous rows</button><span>Page {{ rowPage + 1 }} / {{ pages }}</span><button :disabled="rowPage + 1 >= pages" @click="rowPage++">Next rows</button></div>
      <div class="table-scroll"><table aria-label="Original XYZ row evidence"><thead><tr><th>Row / line</th><th>Original record and lexemes</th><th>I [A]</th><th>V [V]</th><th>E [J]</th><th>Row / cell status</th></tr></thead><tbody><tr v-for="row in visibleRows" :key="row.id"><th>{{ row.id }} / {{ row.line_number }}</th><td><code>{{ row.raw_record }}</code><br><code>{{ JSON.stringify(row.lexemes) }}</code></td><td>{{ number(row.current_a) }}</td><td>{{ number(row.voltage_v) }}</td><td>{{ number(row.energy_j) }}</td><td>{{ row.status }} / {{ row.cell_status }}<br>{{ row.reason }}</td></tr></tbody></table></div>
      <p v-for="assumption in result.assumptions" :key="assumption">{{ assumption }}</p>
      <button @click="fullEvidence = !fullEvidence">{{ fullEvidence ? 'Hide' : 'Show' }} full accepted evidence</button><pre v-if="fullEvidence">{{ rawJson }}</pre>
    </section>
  </section>
</template>

<style scoped>
.xyz-panel { min-width:0; }h3,h4 { margin:.8rem 0; }p { line-height:1.6; }.panel { border:1px solid var(--border-color,#aaa); border-radius:8px; padding:1rem; margin:1rem 0; background:var(--bg-secondary); min-width:0; }.fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:1rem; }label { display:grid; gap:.4rem; margin:.7rem 0; }.check { display:flex; align-items:baseline; gap:.5rem; }input:not([type=checkbox]),textarea,select { width:100%; min-width:0; padding:.5rem; border:1px solid var(--border-color,#aaa); background:var(--bg-secondary); color:inherit; font:inherit; }textarea,code { font-family:monospace; }button { padding:.6rem; margin:.25rem; cursor:pointer; border:1px solid #7b97ab; border-radius:5px; background:var(--bg-secondary); color:inherit; }button:disabled { opacity:.6; cursor:default; }.actions { display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; }.hash,pre,code { overflow-wrap:anywhere; }pre { white-space:pre-wrap; }.table-scroll { overflow-x:auto; max-width:100%; }table { border-collapse:collapse; width:100%; }th,td { border:1px solid var(--border-color,#aaa); padding:.55rem; text-align:left; vertical-align:top; }.heatmap { min-width:440px; }.heatmap button { width:100%; margin:0; min-width:110px; background:transparent; font:inherit; }.heatmap td.ambiguous { background:#e6bd81; color:#31200a; }.heatmap td.unknown { background:repeating-linear-gradient(45deg,#ddd,#ddd 5px,#eee 5px,#eee 10px); color:#333; }.heatmap td.missing { background:#eee; color:#444; border-style:dashed; }.legend { display:flex; align-items:center; gap:.5rem; }.gradient { display:inline-block; max-width:300px; flex:1; height:1.2rem; border:1px solid #527780; background:linear-gradient(to right,rgba(13,148,180,.12),rgba(13,148,180,.8)); }[role=alert] { color:#b02828; }summary { cursor:pointer; font-weight:600; }
@media(max-width:600px) { .panel { padding:.6rem; } }
</style>
