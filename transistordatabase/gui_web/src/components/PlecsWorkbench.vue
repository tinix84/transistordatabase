<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import PlecsThermalEvidence from './PlecsThermalEvidence.vue'
import PlecsCoreMapping from './PlecsCoreMapping.vue'
import PlecsCoreExport from './PlecsCoreExport.vue'
import { plecsDataApi } from '../services/api.js'

const input = reactive({ xml: '', name: '' })
const originalBase64 = ref(null)
const result = ref(null), error = ref(''), busy = ref(false), exporting = ref(false), stale = ref(false)
const selectedSlices = reactive({})
let revision = 0, disposed = false
watch(input, () => {
  revision++; stale.value = !!result.value || busy.value || exporting.value
  result.value = null; error.value = ''
}, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
watch(() => input.xml, () => { originalBase64.value = null }, { flush: 'sync' })
const display = value => value === null || value === undefined ? 'Unknown' : String(value)
const request = () => ({ ...(originalBase64.value === null ? { xml_text: input.xml } : { xml_base64: originalBase64.value }), source_name: input.name.trim() || null })
function message(failure) {
  const detail = failure.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map(row => row.message ?? row.msg ?? JSON.stringify(row)).join('; ')
  return detail?.message ?? failure.message
}
async function upload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  input.name = file.name; input.xml = ''
  const token = ++revision
  try {
    const bytes = new Uint8Array(await file.arrayBuffer())
    const declaration = new TextDecoder('ascii').decode(bytes.slice(0, 200)).match(/<\?xml[^>]*encoding\s*=\s*['"]([^'"]+)['"]/i)
    const encoding = bytes[0] === 255 && bytes[1] === 254 ? 'utf-16le' : bytes[0] === 254 && bytes[1] === 255 ? 'utf-16be' : declaration?.[1] ?? 'utf-8'
    const text = new TextDecoder(encoding, { fatal: true }).decode(bytes)
    let binary = ''
    for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768))
    if (!disposed && token === revision) { input.xml = text; originalBase64.value = btoa(binary) }
  } catch (failure) { if (!disposed && token === revision) error.value = `Cannot read XML: ${failure.message}` }
}
async function preview() {
  result.value = null; error.value = ''; stale.value = false
  const token = ++revision; busy.value = true
  try {
    if (!input.xml.trim()) throw new Error('Choose an XML file or enter XML text first')
    const response = await plecsDataApi.preview(request())
    if (disposed || token !== revision) return
    Object.keys(selectedSlices).forEach(key => { delete selectedSlices[key] })
    result.value = response
  } catch (failure) { if (!disposed && token === revision) error.value = message(failure) }
  finally { if (!disposed) busy.value = false }
}
function save(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url)
}
function downloadJson() {
  if (result.value) save(new Blob([JSON.stringify(result.value, null, 2)], { type: 'application/json' }), 'plecs-document.json')
}
async function downloadXml(mode) {
  if (!result.value) return
  error.value = ''; const token = revision; exporting.value = true
  try {
    const blob = await plecsDataApi.export({ ...request(), mode })
    if (disposed || token !== revision) return
    save(blob, mode === 'source-preserved' ? 'plecs-preserved-source.xml' : 'plecs-supported-tables.xml')
  } catch (failure) {
    if (disposed || token !== revision) return
    if (failure.response?.data instanceof Blob) {
      try { failure.response.data = JSON.parse(await failure.response.data.text()) } catch { /* Non-JSON error body. */ }
    }
    if (!disposed && token === revision) error.value = message(failure)
  } finally { if (!disposed) exporting.value = false }
}
function slice(table) { return table.slices?.[selectedSlices[table.source_path] ?? 0] }
const plottedDatasets = computed(() => Object.fromEntries((result.value?.packages ?? []).flatMap(pkg => pkg.blocks.flatMap(block => block.tables.map(table => {
  const selected = slice(table)
  return [table.source_path, selected ? [{ label: selected.label, data: selected.plot_data.map(([x, y]) => ({ x, y })), borderColor: '#2469b2' }] : []]
})))))
</script>

<template>
  <section class="plecs-workbench">
    <h2>PLECS data workbench</h2>
    <p>Inspect recorded tables and model structure before export. Unknown ratings and conditions remain unknown. Formulas are preserved as source text.</p>
    <form @submit.prevent="preview" class="input-card">
      <label>PLECS XML file <input data-testid="plecs-file" type="file" accept=".xml,text/xml,application/xml" @change="upload"></label>
      <label>Source name <input v-model="input.name" data-testid="plecs-source" placeholder="Original file name or source reference"></label>
      <label>XML source <textarea v-model="input.xml" data-testid="plecs-xml" rows="10" spellcheck="false" placeholder="Paste SemiconductorLibrary XML here"></textarea></label>
      <p>{{ originalBase64 === null ? 'Text input: editing creates a UTF-8 source identity.' : 'Uploaded original bytes retained. Editing XML text creates a new UTF-8 source identity.' }}</p>
      <button type="submit" :disabled="busy">{{ busy ? 'Inspecting…' : 'Inspect PLECS document' }}</button>
    </form>
    <p v-if="stale" role="status">Source changed. Inspect again to refresh the document and exports.</p>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <section v-if="result" data-testid="plecs-results" class="results">
      <h3>Document evidence</h3>
      <dl class="evidence">
        <dt>Support status</dt><dd>{{ result.status }}</dd>
        <dt>Source</dt><dd>{{ display(result.source.name) }}</dd>
        <dt>Source SHA-256</dt><dd data-testid="plecs-source-hash">{{ result.source.sha256 }}</dd>
        <dt>Hash scope</dt><dd>{{ result.source.hash_scope }}</dd>
        <dt>Encoding</dt><dd>{{ display(result.source.encoding) }}</dd>
        <dt>Namespace</dt><dd>{{ display(result.namespace_uri) }}</dd>
        <dt>Library version</dt><dd>{{ display(result.version) }}</dd>
      </dl>
      <p>Uploaded source identity covers the original file bytes, including encoding and any byte order mark. Pasted or edited XML text is submitted as UTF-8.</p>
      <div class="actions">
        <button @click="downloadJson">Download canonical document JSON</button>
        <button :disabled="!result.can_export || exporting" @click="downloadXml('supported-tables')">Download supported table XML</button>
        <button :disabled="!result.can_archive || exporting" @click="downloadXml('source-preserved')">Download preserved source XML</button>
      </div>
      <p>Preserved source XML retains unsupported content for archival use. Supported table export is available only when the Python reader validates the complete supported structure. Simulator acceptance is not established by this preview.</p>
      <PlecsCoreMapping v-if="result.can_export" :key="result.source.sha256" :document="result" :source-request="request()" />
      <section v-if="result.diagnostics.length" class="diagnostics" aria-label="Document diagnostics">
        <h4>Support report</h4>
        <ul><li v-for="(item, index) in result.diagnostics" :key="index"><strong>{{ item.code }}</strong>: {{ item.message }} <small>{{ item.path }}</small></li></ul>
      </section>
      <details v-for="(metadata, index) in result.preserved_metadata" :key="index"><summary>Preserved library {{ metadata.kind }} source</summary><pre>{{ metadata.xml }}</pre></details>
      <article v-for="pkg in result.packages" :key="pkg.id" class="package-card" data-testid="plecs-package">
        <h3>{{ display(pkg.partnumber) }}</h3>
        <p>Package {{ pkg.id }} · {{ display(pkg.device_class) }} · Vendor {{ display(pkg.vendor) }}</p>
        <p class="unknown-ratings">Absolute voltage/current ratings: Unknown. Table coverage does not establish datasheet ratings.</p>
        <section v-for="block in pkg.blocks" :key="block.id" class="block-card">
          <h4>Semiconductor block {{ block.id }} · {{ display(block.device_type) }}</h4>
          <p v-if="!block.tables.length">No recorded loss tables in this block.</p>
          <section v-for="table in block.tables" :key="table.source_path" class="table-card" data-testid="plecs-table">
            <h4>{{ table.kind }} · {{ table.status }}</h4>
            <p>{{ table.method }} · Source scale {{ display(table.scale) }} · Gate {{ display(table.gate_voltage_v) }} V · Rg {{ display(table.gate_resistance_ohm) }} Ω</p>
            <p class="source-path">{{ table.source_path }}</p>
            <div v-if="table.formula" class="formula"><strong>Unexecuted formula</strong><pre>{{ table.formula }}</pre></div>
            <template v-if="table.status === 'supported-table' && table.slices?.length">
              <label>Recorded slice
                <select v-model="selectedSlices[table.source_path]" data-testid="plecs-slice">
                  <option v-for="(record, index) in table.slices" :key="index" :value="index">{{ record.label }}</option>
                </select>
              </label>
              <p>Tj {{ display(slice(table).temperature_c) }} °C · Signed supply {{ display(slice(table).supply_voltage_v) }} V</p>
              <CurvePlotter :title="`${table.kind}: ${slice(table).label}`" :datasets="plottedDatasets[table.source_path]" x-label="Current (A)" :y-label="table.kind === 'conduction' ? 'Voltage drop (V)' : 'Energy (J)'" />
              <div class="table-scroll"><table><caption>Recorded slice in SI units, supplied by Python</caption><thead><tr><th>Current (A)</th><th>{{ table.kind === 'conduction' ? 'Voltage drop (V)' : 'Energy (J)' }}</th></tr></thead><tbody><tr v-for="(point, index) in slice(table).plot_data" :key="index"><td>{{ point[0] }}</td><td>{{ point[1] }}</td></tr></tbody></table></div>
            </template>
            <p v-else>Source data is retained below; this table is not an evaluated loss model.</p>
            <details><summary>Original table axes and values</summary>
              <dl><dt>Current axis (A)</dt><dd>{{ table.current_a }}</dd><dt>Temperature axis (°C)</dt><dd>{{ table.temperature_c }}</dd><dt>Signed supply axis (V)</dt><dd>{{ display(table.voltage_v) }}</dd></dl>
              <pre>{{ JSON.stringify(table.values, null, 2) }}</pre>
            </details>
          </section>
          <details v-for="(metadata, index) in block.preserved_metadata" :key="index"><summary>Preserved {{ metadata.kind }} source</summary><pre>{{ metadata.xml }}</pre></details>
          <PlecsThermalEvidence :networks="block.thermal ?? []" :scope="`Block ${block.id}`" />
        </section>
        <PlecsThermalEvidence :networks="pkg.thermal" scope="Package" />
        <details v-for="(metadata, index) in pkg.preserved_metadata" :key="index"><summary>Preserved {{ metadata.kind }} source</summary><pre>{{ metadata.xml }}</pre></details>
      </article>
    </section>
    <PlecsCoreExport />
  </section>
</template>

<style scoped>
.plecs-workbench { max-width: 1240px; margin: auto; overflow-wrap: anywhere; }
.input-card, .package-card { border: 1px solid var(--border-color, #d4dbe5); border-radius: 10px; padding: 1.1rem; margin: 1rem 0; }
label { display: flex; flex-direction: column; gap: .35rem; margin-bottom: .85rem; font-weight: 600; }
input, textarea, select { box-sizing: border-box; width: 100%; min-width: 0; max-width: 100%; padding: .6rem; color: inherit; background: var(--bg-primary, white); border: 1px solid #aeb9c8; border-radius: 5px; font: inherit; }
textarea { font-family: monospace; font-size: .85rem; }
button { padding: .6rem .85rem; border: 1px solid #8ba0b9; border-radius: 5px; background: var(--bg-secondary, #eef4fa); color: inherit; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
.actions { display: flex; flex-wrap: wrap; gap: .6rem; }
.evidence { display: grid; grid-template-columns: minmax(90px, 150px) minmax(0, 1fr); gap: .4rem .8rem; }
dt { font-weight: 600; } dd { margin: 0 0 .4rem; }
.block-card { margin-top: 1.5rem; }
.table-card, .thermal-card { padding: .8rem 0; border-top: 1px solid #d4dbe5; }
.table-scroll { overflow-x: auto; max-width: 100%; margin: .7rem 0; }
table { border-collapse: collapse; min-width: 500px; width: 100%; }
th, td { padding: .6rem; border: 1px solid #c6cfdb; text-align: left; }
caption { text-align: left; margin-bottom: .4rem; } pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: .6rem 0; }
details { padding: .6rem 0; } summary { cursor: pointer; font-weight: 600; }
.diagnostics, .unknown-ratings { background: var(--bg-secondary, #fff7e7); padding: .8rem; border-left: 3px solid #b48017; }
.diagnostics small { display: block; } .error { color: #b52222; } .source-path { font-size: .85rem; }
@media(max-width: 500px) { .input-card, .package-card { padding: .75rem; } .evidence { grid-template-columns: 1fr; gap: .1rem; } }
</style>
