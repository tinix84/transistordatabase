<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { rgEnergyApi, transistorApi } from '../services/api.js'
import CurvePlotter from './CurvePlotter.vue'
import { analysisCsv, analysisError, displayNumber, downloadAnalysis, finiteDecimal } from './analysisFiles.js'
import { returnedPlotBounds } from './returnedPlotBounds.js'

const conditionKeys = ['v_supply', 'v_g', 'v_g_off', 'load_inductance', 'commutation_inductance', 'commutation_device', 'convention', 'event', 'method']
const conditionLabels = { v_supply: 'Supply voltage (V)', v_g: 'Gate voltage (V)', v_g_off: 'Off gate voltage (V)', load_inductance: 'Load inductance (H)', commutation_inductance: 'Commutation inductance (H)', commutation_device: 'Commutation device', convention: 'Convention', event: 'Event', method: 'Method' }
const input = reactive({ device: '', branch: 'e_on', representation: 'graph_r_e', context: '', operation: 'energy-at-rg', current: '', temperature: '', query: '' })
const devices = ref([]), inventory = ref(null), inventoryBusy = ref(false), listBusy = ref(false), inventoryError = ref('')
const result = ref(null), rawJson = ref(''), busy = ref(false), error = ref(''), stale = ref(false)
let revision = 0, inventoryRevision = 0, listRevision = 0, disposed = false
function invalidate() {
  revision++; stale.value = result.value !== null || busy.value || stale.value
  result.value = null; rawJson.value = ''; error.value = ''; busy.value = false
}
watch(input, invalidate, { deep: true, flush: 'sync' })
watch(() => [input.device, input.branch, input.representation], () => { input.context = '' }, { flush: 'sync' })
watch(() => input.device, loadInventory, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++; inventoryRevision++; listRevision++ })
async function loadDevices() {
  const token = ++listRevision; listBusy.value = true; inventoryError.value = ''
  try { const rows = await transistorApi.getAll(); if (!disposed && token === listRevision) devices.value = rows }
  catch (failure) { if (!disposed && token === listRevision) inventoryError.value = analysisError(failure) }
  finally { if (!disposed && token === listRevision) listBusy.value = false }
}
async function loadInventory() {
  const token = ++inventoryRevision, device = input.device
  invalidate(); input.context = ''; inventory.value = null; inventoryError.value = ''; inventoryBusy.value = !!device
  if (!device) return
  try { const data = await rgEnergyApi.inventory(device); if (!disposed && token === inventoryRevision) inventory.value = data }
  catch (failure) { if (!disposed && token === inventoryRevision) inventoryError.value = analysisError(failure) }
  finally { if (!disposed && token === inventoryRevision) inventoryBusy.value = false }
}
onMounted(loadDevices)
const branchCurves = computed(() => (inventory.value?.curves || []).filter(curve => curve.branch === input.branch))
const contexts = computed(() => {
  const groups = new Map()
  for (const curve of branchCurves.value.filter(curve => curve.representation === input.representation && curve.status !== 'unavailable' && curve.conditions)) {
    const key = JSON.stringify(conditionKeys.map(name => curve.conditions[name]))
    if (!groups.has(key)) groups.set(key, { key, conditions: curve.conditions, curves: [] })
    groups.get(key).curves.push(curve)
  }
  return [...groups.values()]
})
const selected = computed(() => contexts.value.find(context => context.key === input.context))
const displayCurves = computed(() => result.value ? result.value.sources.filter(curve => curve.branch === result.value.query.branch
  && curve.representation === result.value.query.representation) : selected.value?.curves || [])
const unavailableCurves = computed(() => (result.value?.sources || inventory.value?.curves || []).filter(curve => curve.branch === input.branch
  && curve.representation === input.representation && curve.status === 'unavailable'))
function conditionValue(value) { return value === null ? 'Unknown / not recorded (null)' : typeof value === 'string' ? JSON.stringify(value) : String(value) }
function deviceId(device) { return device.id ?? device.metadata.name }
function contextLabel(context, index) {
  return `Context ${index + 1}: V=${conditionValue(context.conditions.v_supply)}, Vg=${conditionValue(context.conditions.v_g)}, method=${conditionValue(context.conditions.method)}; ${context.curves.length} record(s)`
}
function request() {
  if (!input.device || !selected.value) throw new Error('Select a device and an exact recorded context.')
  const value = { operation: input.operation, branch: input.branch, representation: input.representation,
    i_channel: finiteDecimal(input.current, 'Requested current'), t_j: finiteDecimal(input.temperature, 'Requested junction temperature'),
    conditions: Object.fromEntries(conditionKeys.map(key => [key, selected.value.conditions[key]])) }
  value[input.operation === 'energy-at-rg' ? 'r_g' : 'e_target'] = finiteDecimal(input.query, input.operation === 'energy-at-rg' ? 'Gate resistance' : 'Target energy')
  return value
}
async function calculate() {
  const token = ++revision; result.value = null; rawJson.value = ''; error.value = ''; stale.value = false; busy.value = true
  try {
    const response = await rgEnergyApi.calculate(input.device, request())
    if (disposed || token !== revision) return
    result.value = response.data; rawJson.value = response.rawJson
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
const datasets = computed(() => {
  const data = displayCurves.value.filter(curve => curve.status !== 'unavailable' && curve.preview).map((curve, index) => ({
    label: `${curve.source_id}; Tj=${conditionValue(curve.t_j)}°C; ${input.representation === 'graph_r_e' ? `I=${conditionValue(curve.i_x)}A` : `Rg=${conditionValue(curve.r_g)}Ω`}`,
    data: curve.preview.axis.map((x, i) => ({ x, y: curve.preview.energy_j[i] })),
    borderColor: ['#0369a1', '#b45309', '#6d28d9', '#047857'][index % 4], pointRadius: 3,
    showLine: !curve.preview.truncated,
  }))
  if (result.value) {
    const response = result.value, query = response.query
    data.push({ label: 'Returned bounded query estimate', showLine: false, pointRadius: 7, borderColor: '#be123c', backgroundColor: '#be123c',
      data: [{ x: query.representation === 'graph_i_e' ? query.i_channel : query.operation === 'energy-at-rg' ? query.r_g : response.value,
        y: query.operation === 'energy-at-rg' ? response.value : query.e_target }] })
  }
  return data
})
function paddedBounds(data) {
  const { min, max } = returnedPlotBounds(data)
  if (min === null || max === null) return { min, max }
  const span = max - min, padding = (Number.isFinite(span) ? span : Math.max(Math.abs(min), Math.abs(max))) * .1
  return { min: Number.isFinite(min - padding) ? min - padding : min, max: Number.isFinite(max + padding) ? max + padding : max }
}
const xBounds = computed(() => paddedBounds(datasets.value.map(dataset => ({ data: dataset.data.map(point => ({ y: point.x })) }))))
const yBounds = computed(() => paddedBounds(datasets.value))
const csv = computed(() => result.value ? analysisCsv([{ device_id: result.value.device_id, operation: result.value.operation, value: result.value.value, unit: result.value.unit,
  evidence_tag: result.value.evidence_tag, query: result.value.query, accepted_conditions: result.value.conditions, source_scope: result.value.source_scope, sources: result.value.sources, method: result.value.method }]) : '')
</script>

<template>
  <section id="rg-energy" class="rg-energy" data-testid="rg-energy">
    <h4>Repository Rg–energy queries</h4>
    <p>Evaluate switching energy at a gate resistance, or estimate gate resistance at a target energy. Select an exact recorded context.</p>
    <p>Python applies bounded interpolation and rejects ambiguous inverse crossings. No nearest context, extrapolation, or resistance scaling applies.</p>
    <form @submit.prevent="calculate">
      <div class="columns">
        <label>Repository device<select v-model="input.device" data-testid="rg-device"><option value="">Select a device</option><option v-for="device in devices" :key="deviceId(device)" :value="deviceId(device)">{{ device.name || device.metadata?.name || device.id }}</option></select></label>
        <label>Energy branch<select v-model="input.branch" data-testid="rg-branch"><option value="e_on">Turn-on energy (e_on)</option><option value="e_off">Turn-off energy (e_off)</option><option value="e_rr">Reverse recovery energy (e_rr)</option></select></label>
        <label>Recorded representation<select v-model="input.representation" data-testid="rg-representation"><option value="graph_r_e">Energy versus gate resistance (graph_r_e)</option><option value="graph_i_e">Energy versus current at recorded Rg (graph_i_e)</option></select></label>
      </div>
      <p v-if="listBusy || inventoryBusy" role="status">Loading repository data…</p>
      <p v-if="inventoryError" role="alert">{{ inventoryError }}</p>
      <button type="button" @click="input.device ? loadInventory() : loadDevices()">Refresh repository data</button>
      <p v-if="inventory && !branchCurves.length" role="status">No recorded energy curves exist for this branch.</p>
      <p v-else-if="inventory && !contexts.length" role="status">No selectable context exists for this representation. Review unavailable records or select another representation or branch.</p>
      <section v-if="unavailableCurves.length" aria-label="Unavailable repository records" data-testid="rg-unavailable-records">
        <h5>Unavailable repository records</h5>
        <p>These records remain in the source proof. They cannot supply a selectable context or preview.</p>
        <div v-for="curve in unavailableCurves" :key="curve.source_id">
          <p><code>{{ curve.source_id }}</code>: {{ curve.diagnostic }}</p>
          <details><summary>Recorded metadata and unavailable fields</summary><pre>{{ JSON.stringify(curve, null, 2) }}</pre></details>
        </div>
      </section>
      <label v-if="contexts.length">Exact recorded context<select v-model="input.context" data-testid="rg-context"><option value="">Select a context explicitly</option><option v-for="(context, index) in contexts" :key="context.key" :value="context.key">{{ contextLabel(context, index) }}</option></select></label>
      <template v-if="selected">
        <h5>Accepted context conditions</h5>
        <dl data-testid="rg-conditions"><template v-for="key in conditionKeys" :key="key"><dt>{{ conditionLabels[key] }}</dt><dd>{{ conditionValue(selected.conditions[key]) }}</dd></template></dl>
        <p>Zero is a known value. Null means unknown. The text "unknown" is a recorded method value, distinct from null.</p>
        <h5>{{ result ? 'All considered records in this representation' : 'All records in this context' }}</h5>
        <p>Records remain separate. Duplicate coverage and holes can prevent a result. The server checks the full branch.</p>
        <p v-for="curve in displayCurves.filter(item => item.status !== 'unavailable' && item.diagnostic)" :key="`diagnostic-${curve.source_id}`">
          Source metadata diagnostic for {{ curve.source_id }}: {{ curve.diagnostic }}. Unavailable fields: {{ curve.unavailable_fields?.join(', ') }}.
        </p>
        <div class="table-scroll"><table aria-label="Recorded Rg energy coverage"><thead><tr><th>Repository source / evidence</th><th>Tj (°C)</th><th>Fixed I (A) / Rg (Ω)</th><th>Recorded domain</th><th>Energy domain (J)</th></tr></thead><tbody>
          <tr v-for="curve in displayCurves" :key="curve.source_id"><td><code>{{ curve.source_id }}</code><br>{{ curve.source_kind }}<br>File: {{ curve.source_filename ?? 'Not recorded' }}<br>Source SHA-256: {{ curve.source_sha256 ?? 'Not recorded' }}<br>Repository snapshot SHA-256: <code>{{ curve.snapshot_sha256 ?? 'Unavailable' }}</code><details><summary>Recorded conditions for this source</summary><dl v-if="curve.conditions"><template v-for="key in conditionKeys" :key="key"><dt>{{ conditionLabels[key] }}</dt><dd>{{ conditionValue(curve.conditions[key]) }}</dd></template></dl><p v-else>Context unavailable. This is not a set of nine unknown selectors.</p></details><p v-if="curve.status === 'unavailable'">Unavailable: {{ curve.diagnostic }}</p></td><td>{{ conditionValue(curve.t_j) }}</td><td>{{ input.representation === 'graph_r_e' ? conditionValue(curve.i_x) : conditionValue(curve.r_g) }}</td><td v-if="curve.coverage">{{ displayNumber(curve.coverage.lower) }} to {{ displayNumber(curve.coverage.upper) }} {{ curve.coverage.axis_unit }} ({{ curve.coverage.sample_count }} points)</td><td v-else>Unavailable</td><td>{{ curve.coverage ? `${displayNumber(curve.coverage.energy_min)} to ${displayNumber(curve.coverage.energy_max)}` : 'Unavailable' }}</td></tr>
        </tbody></table></div>
        <p v-if="displayCurves.some(curve => curve.preview?.truncated)">Some previews contain selected source points only. Downloaded results retain the server source evidence.</p>
        <p v-if="result">This table and plot use all considered source snapshots in this representation. Each source retains its recorded conditions.</p>
      </template>
      <div class="columns">
        <label>Query operation<select v-model="input.operation" data-testid="rg-operation"><option value="energy-at-rg">Energy at gate resistance</option><option value="rg-at-energy">Gate resistance at target energy</option></select></label>
        <label>Requested current (A)<input :value="input.current" @input="input.current = $event.target.value" inputmode="decimal" data-testid="rg-current"></label>
        <label>Requested junction temperature (°C)<input :value="input.temperature" @input="input.temperature = $event.target.value" inputmode="decimal" data-testid="rg-temperature"></label>
        <label>{{ input.operation === 'energy-at-rg' ? 'Gate resistance (Ω)' : 'Target energy (J)' }}<input :value="input.query" @input="input.query = $event.target.value" inputmode="decimal" data-testid="rg-query"></label>
      </div>
      <p v-if="input.representation === 'graph_i_e'">Forward queries require an exact recorded Rg. Inverse estimates can interpolate between supported recorded Rg values.</p>
      <button type="submit" :disabled="busy || inventoryBusy" data-testid="rg-calculate">{{ busy ? 'Calculating…' : 'Calculate bounded Rg–energy query' }}</button>
      <p v-if="error" role="alert" data-testid="rg-error">{{ error }}</p>
      <p v-if="stale && !result" role="status">Selection or query changed. Calculate again to show current results and exports.</p>
    </form>
    <section v-if="result" data-testid="rg-results" aria-label="Bounded Rg energy result">
      <h5>Bounded interpolation estimate</h5>
      <p data-testid="rg-value">{{ displayNumber(result.value) }} {{ result.unit }}</p>
      <p>{{ result.evidence_tag }}. {{ result.method }}.</p>
      <p>This result estimates the recorded curves. Device accuracy and interactions outside the selected context are not validated.</p>
      <p>Accepted query: {{ result.query.branch }}, {{ result.query.representation }}, I = {{ displayNumber(result.query.i_channel) }} A,
        Tj = {{ displayNumber(result.query.t_j) }} °C,
        {{ result.operation === 'energy-at-rg' ? `Rg = ${displayNumber(result.query.r_g)} Ω` : `target E = ${displayNumber(result.query.e_target)} J` }}.</p>
      <p>Source scope: {{ result.source_scope }}. Source evidence lists records considered in the branch. It does not select one arbitrary record.</p>
      <details><summary>Accepted query, context, source IDs and domains</summary><pre>{{ JSON.stringify(result, null, 2) }}</pre></details>
      <button type="button" data-testid="rg-json" @click="downloadAnalysis(rawJson, 'rg-energy.json', 'application/octet-stream')">Download exact server JSON</button>
      <button type="button" data-testid="rg-csv" @click="downloadAnalysis(csv, 'rg-energy.csv', 'text/csv')">Download query + source proof CSV</button>
    </section>
    <template v-if="selected && datasets.length">
      <p>Preview: {{ result ? 'all considered source curves in this representation' : 'all recorded curves in the selected context' }}. Source IDs identify the table conditions. The red point is the returned query.</p>
      <CurvePlotter title="Recorded context curves and returned query" :datasets="datasets" :x-label="input.representation === 'graph_r_e' ? 'Gate resistance (Ω)' : 'Current (A)'" y-label="Energy (J)" :x-min="xBounds.min" :x-max="xBounds.max" :y-min="yBounds.min" :y-max="yBounds.max" :allow-zoom="false" />
    </template>
  </section>
</template>

<style scoped>
.rg-energy { margin-top: 1.5rem; padding: 1rem; min-width: 0; border: 1px solid #cbd5e1; border-radius: 8px; background: #f8fafc; }
h4 { font-size: 1.15rem; } h5 { font-size: 1rem; } p { line-height: 1.6; overflow-wrap: anywhere; } label { display: grid; gap: .4rem; margin: .7rem 0; min-width: 0; }
.columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .8rem; } input, select { width: 100%; min-width: 0; box-sizing: border-box; padding: .6rem; }
button { padding: .7rem; margin: .5rem .5rem .5rem 0; max-width: 100%; } dl { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: .5rem; } dt, dd { margin: 0; overflow-wrap: anywhere; }
.table-scroll { overflow-x: auto; } table { min-width: 650px; width: 100%; border-collapse: collapse; } th, td { padding: .6rem; text-align: left; border-bottom: 1px solid #cbd5e1; vertical-align: top; overflow-wrap: anywhere; max-width: 400px; }
pre, code { overflow-wrap: anywhere; white-space: pre-wrap; } pre { max-width: 100%; overflow: auto; } [role=alert] { color: #a11; }
@media(max-width: 760px) { .columns { grid-template-columns: 1fr; } .rg-energy { padding: .6rem; } }
</style>
