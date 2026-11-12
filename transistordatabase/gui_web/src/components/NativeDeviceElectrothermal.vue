<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import { transistorApi, deviceThermalApi } from '../services/api.js'
import { binary64Decimal, displayNumber, analysisCsv, downloadAnalysis, analysisError } from './analysisFiles.js'
const input = reactive({ deviceId: '', source: '', selected: false, current: '50', duty: '0.5', frequency: '1000',
  sink: '45', resistance: '0.1', initial: '45', tolerance: '1e-8', iterations: '200', relaxation: '1' })
const families = ['channel', 'turn_on', 'turn_off']
const selected = reactive({ channel: [], turn_on: [], turn_off: [] })
const inventory = ref(null), devices = ref([]), result = ref(null), error = ref(''), loading = ref(false), busy = ref(false), stale = ref(false)
const visible = reactive({ channel: 100, turn_on: 100, turn_off: 100 })
let revision = 0, disposed = false
watch([input, selected], () => { revision++; stale.value = result.value !== null || busy.value; result.value = null; error.value = '' }, { deep: true, flush: 'sync' })
watch(() => input.deviceId, () => { inventory.value = null; for (const family of families) selected[family] = [] }, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
onMounted(async () => {
  try {
    const records = await transistorApi.getAll()
    if (!disposed) devices.value = records.map(row => typeof row === 'string' ? row : row.id ?? row.name ?? row.metadata?.name).filter(Boolean)
  } catch { /* An explicit device ID can still use the injected repository. */ }
})
const accepted = computed(() => result.value?.result.accepted_solution)
function conditionsKey(row) { return JSON.stringify(Object.fromEntries(Object.keys(row.recorded_conditions ?? {}).sort().map(key => [key, row.recorded_conditions[key]]))) }
async function loadInventory() {
  result.value = null; error.value = ''; inventory.value = null
  for (const family of families) selected[family] = []
  const token = ++revision, id = input.deviceId
  if (!id.trim()) { error.value = 'Select an exact repository device ID.'; return }
  loading.value = true
  try {
    const response = await deviceThermalApi.inventory(id)
    if (disposed || token !== revision || id !== input.deviceId) return
    inventory.value = response
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed) loading.value = false }
}
function selectGroup(family, row) {
  const chosen = inventory.value.families[family].filter(item => item.status === 'available' && conditionsKey(item) === conditionsKey(row))
  if (chosen.length > 32) { error.value = 'This group exceeds32 records. Select at most32 unique temperature records explicitly.'; return }
  selected[family] = chosen.map(item => item.index)
}
function familyRequest(family) {
  const rows = selected[family].map(index => inventory.value.families[family].find(row => row.index === index))
  if (rows.length < 2 || rows.some(row => !row || row.status !== 'available')) throw new Error(`${family} needs at least two available temperature records.`)
  if (new Set(rows.map(row => row.temperature_c)).size !== rows.length) throw new Error(`${family} needs unique temperatures. Disambiguate duplicate source rows.`)
  if (rows.some(row => conditionsKey(row) !== conditionsKey(rows[0]))) throw new Error(`${family} mixes fixed conditions. Select one hardware, gate, event and method context.`)
  return { indices: selected[family].slice(), conditions: { ...rows[0].recorded_conditions } }
}
function request() {
  if (!input.selected) throw new Error('Select the held-current PWM estimate explicitly.')
  if (!inventory.value || inventory.value.transistor_id !== input.deviceId) throw new Error('Refresh the selected device source inventory.')
  if (!input.source.trim()) throw new Error('Declare the junction-to-fixed-sink path source and assumptions.')
  const current = binary64Decimal(input.current, 'Held current'), duty = binary64Decimal(input.duty, 'Duty'), frequency = binary64Decimal(input.frequency, 'Frequency')
  const iterations = binary64Decimal(input.iterations, 'Iteration limit')
  if (!Number.isInteger(iterations)) throw new Error('The iteration limit must be an integer.')
  const switching = frequency > 0 && duty > 0 && duty < 1
  return { transistor_id: input.deviceId, expected_file_sha256: inventory.value.repository_source?.sha256 ?? null,
    selected: true, switching_policy: 'held-current-pwm', source: input.source, current_a: current, duty, frequency_hz: frequency,
    sink_temperature_c: binary64Decimal(input.sink, 'Sink temperature'), resistance_k_w: binary64Decimal(input.resistance, 'Thermal path'),
    initial_temperature_c: binary64Decimal(input.initial, 'Initial temperature'), tolerance_k: binary64Decimal(input.tolerance, 'Residual tolerance'),
    max_iterations: iterations, relaxation: binary64Decimal(input.relaxation, 'Relaxation'),
    channel: current > 0 && duty > 0 ? familyRequest('channel') : null,
    turn_on: switching ? familyRequest('turn_on') : null, turn_off: switching ? familyRequest('turn_off') : null }
}
async function calculate() {
  result.value = null; error.value = ''; stale.value = false
  const token = ++revision
  busy.value = true
  try {
    const response = await deviceThermalApi.analyze(request())
    if (!disposed && token === revision) result.value = response
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed) busy.value = false }
}
defineExpose({ calculate })
const lossDatasets = computed(() => {
  if (!result.value) return []
  const table = result.value.table
  return [['total', table.total_power_w], ...Object.entries(table.component_power_w)].map(([name, values], i) => ({
    label: `${name} power (W)`, data: table.temperature_c.map((t, j) => ({ x: t, y: values[j] })),
    borderColor: ['#0f172a', '#0369a1', '#b45309', '#9333ea'][i], pointRadius: 3 }))
})
function tableCsv() {
  const response = result.value, table = response.table, identity = table.source_identity
  const rows = table.temperature_c.map((temperature, i) => ({ temperature_c: temperature,
    conduction_power_w: table.component_power_w.channel[i], turn_on_power_w: table.component_power_w.turn_on[i],
    turn_off_power_w: table.component_power_w.turn_off[i], total_power_w: table.total_power_w[i],
    current_a: response.request.current_a, duty: response.request.duty, frequency_hz: response.request.frequency_hz,
    device_id: table.device_name, selected_dependency_sha256: identity.selected_dependency_sha256,
    repository_file_sha256: identity.repository_source?.sha256 ?? null, scope: table.scope }))
  downloadAnalysis(analysisCsv(rows), 'native-device-temperature-loss-table.csv', 'text/csv')
}
</script>

<template>
  <section class="native-device" data-testid="native-device-panel">
    <h3>Native device curves → one-junction electrothermal feedback</h3>
    <p>One switch junction at a fixed sink. This held-current estimate includes selected channel and switching heat. It does not infer diode, package, or shared-junction heat.</p>
    <div class="fields"><label>Exact repository device ID<input v-model="input.deviceId" list="native-thermal-devices" data-testid="native-device-id"><datalist id="native-thermal-devices"><option v-for="id in devices" :key="id" :value="id"/></datalist></label>
      <label>Junction-to-fixed-sink path source / assumptions<input v-model="input.source" data-testid="native-thermal-source"></label></div>
    <button type="button" :disabled="loading" @click="loadInventory" data-testid="native-inventory-refresh">{{ loading ? 'Reading source…' : 'Refresh source inventory' }}</button>
    <div class="fields"><label v-for="field in [{key:'current',label:'Held switch current (A)'},{key:'duty',label:'Conduction duty [0,1]'},{key:'frequency',label:'Declared PWM frequency (Hz)'},{key:'sink',label:'Fixed sink temperature (C)'},{key:'resistance',label:'Junction-to-fixed-sink resistance (K/W)'},{key:'initial',label:'Initial junction temperature (C)'},{key:'tolerance',label:'Actual residual tolerance (K)'},{key:'iterations',label:'Maximum iterations'},{key:'relaxation',label:'Relaxation (0,1]'}]" :key="field.key">{{ field.label }}<input v-model="input[field.key]" inputmode="decimal" :data-testid="`native-${field.key}`"></label></div>
    <label class="check"><input v-model="input.selected" type="checkbox" data-testid="native-estimate-selected">Select the held-current PWM estimate. One on/off pair per declared period applies only for0&lt;duty&lt;1; endpoint duty has no events.</label>
    <p>Zero frequency disables switching. Active switching at zero current still requires recorded E(0). Each active contribution needs at least two supported temperatures. All-inactive inputs have no inferred source domain.</p>
    <template v-if="inventory">
      <p data-testid="native-source-identity">Persisted-file SHA: {{ inventory.repository_source?.sha256 || 'Unavailable for this repository provider' }}. This differs from the selected curve dependency digest.</p>
      <section v-for="family in families" :key="family" class="family" :data-testid="`native-family-${family}`">
        <h4>{{ family }}: select explicit temperature records</h4><p>{{ selected[family].length }} selected. Independent on/off gate fields retain their recorded meanings. Group selection does not drop duplicate temperatures.</p>
        <div class="table-scroll"><table><thead><tr><th>Select</th><th>Source index / temperature</th><th>Current support</th><th>Fixed recorded conditions</th><th>Availability</th></tr></thead><tbody><tr v-for="row in inventory.families[family].slice(0, visible[family])" :key="row.index">
          <td><input type="checkbox" v-model="selected[family]" :value="row.index" :disabled="row.status !== 'available' || (selected[family].length >= 32 && !selected[family].includes(row.index))" :data-testid="`native-record-${family}-${row.index}`"></td>
          <td>{{ row.index }} / {{ displayNumber(row.temperature_c) }} C<br>{{ row.representation }}</td><td>{{ row.current_domain_a?.map(displayNumber).join(' to ') || 'Unavailable' }} A</td>
          <td><details><summary>Exact gate / hardware / method context</summary><pre>{{ JSON.stringify(row.recorded_conditions, null, 2) }}</pre></details><button v-if="row.status === 'available'" type="button" @click="selectGroup(family, row)" :data-testid="`native-group-${family}-${row.index}`">Select this fixed-condition group</button></td><td>{{ row.status }} {{ row.diagnostic || '' }}</td>
        </tr></tbody></table></div>
        <button v-if="visible[family] < inventory.families[family].length" type="button" @click="visible[family] += 100">Show the next100 records</button>
      </section>
    </template>
    <button type="button" :disabled="busy || !inventory || !input.selected" @click="calculate" data-testid="native-calculate">{{ busy ? 'Solving…' : 'Calculate native-device feedback' }}</button>
    <p v-if="error" role="alert">{{ error }}</p><p v-if="stale && !result" role="status">Inputs or source selection changed. Recalculate for current results and exports.</p>
    <section v-if="result" data-testid="native-thermal-result">
      <h4>Solver status: {{ result.result.status }}</h4>
      <p>{{ result.table.scope }}. Temperature support {{ result.table.temperature_domain_c.map(displayNumber).join(' to ') }} C; current support {{ result.table.current_domain_a.map(displayNumber).join(' to ') }} A.</p>
      <p>Selected dependency SHA: {{ result.table.source_identity.selected_dependency_sha256 }}. {{ result.table.source_identity.selected_dependency_scope }}.</p>
      <p v-if="accepted" data-testid="native-accepted">Accepted junction: {{ displayNumber(accepted.temperature_c[0]) }} C; included loss: {{ displayNumber(accepted.power_w[0]) }} W; actual unrelaxed residual: {{ displayNumber(accepted.residual_k[0]) }} K.</p>
      <p v-else data-testid="native-no-solution">No accepted solution. The last iterate is not a physical solution. {{ result.result.nonconvergence_scope }}</p>
      <p v-if="result.accepted_component_power_w">Accepted components: channel {{ displayNumber(result.accepted_component_power_w.channel) }} W; turn-on {{ displayNumber(result.accepted_component_power_w.turn_on) }} W; turn-off {{ displayNumber(result.accepted_component_power_w.turn_off) }} W.</p>
      <p>Active contributions: {{ JSON.stringify(result.table.active_contributions) }}.</p>
      <ul><li v-for="note in result.table.assumptions" :key="note">{{ note }}</li></ul>
      <CurvePlotter title="Included native-device loss versus temperature" :datasets="lossDatasets" x-label="Junction temperature (C)" y-label="Included power (W)" :allow-zoom="false" />
      <div class="table-scroll"><table><thead><tr><th>Temperature (C)</th><th>Channel (W)</th><th>Turn-on (W)</th><th>Turn-off (W)</th><th>Total (W)</th></tr></thead><tbody><tr v-for="(temperature, i) in result.table.temperature_c" :key="temperature"><td>{{ displayNumber(temperature) }}</td><td>{{ displayNumber(result.table.component_power_w.channel[i]) }}</td><td>{{ displayNumber(result.table.component_power_w.turn_on[i]) }}</td><td>{{ displayNumber(result.table.component_power_w.turn_off[i]) }}</td><td>{{ displayNumber(result.table.total_power_w[i]) }}</td></tr></tbody></table></div>
      <button type="button" @click="downloadAnalysis(result, 'native-device-electrothermal-evidence.json')" data-testid="native-thermal-json">Export full request and result JSON</button>
      <button type="button" @click="tableCsv" data-testid="native-thermal-csv">Export temperature loss table CSV</button>
      <details><summary>Selected source record and solver history evidence</summary><pre>{{ JSON.stringify({ records: result.table.record_evidence, history: result.result.history }, null, 2) }}</pre></details>
    </section>
  </section>
</template>

<style scoped>
.native-device, .family { min-width:0; max-width:100%; } .fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr)); gap:.8rem; }
label { display:grid; gap:.4rem; margin:.6rem 0; } .check { display:flex; align-items:baseline; gap:.5rem; }.check input { width:auto; }
input:not([type=checkbox]), select { box-sizing:border-box; width:100%; min-width:0; padding:.6rem; font:inherit; }
p,li,pre { overflow-wrap:anywhere; } pre { white-space:pre-wrap; } .table-scroll { min-width:0; max-width:100%; overflow-x:auto; }
table { width:100%; border-collapse:collapse; } th,td { padding:.6rem; text-align:left; border-bottom:1px solid #cbd5e1; vertical-align:top; }
button { margin:.5rem .5rem .5rem 0; padding:.6rem; font:inherit; cursor:pointer; } .family { margin:1rem 0; padding:.7rem; box-sizing:border-box; border:1px solid #94a3b8; }
[role=alert] { color:#991b1b; padding:.8rem; background:#fee2e2; }
</style>
