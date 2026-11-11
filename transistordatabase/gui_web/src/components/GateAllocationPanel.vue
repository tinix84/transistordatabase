<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import { waveformApi } from '../services/api.js'
import { analysisCsv, analysisError, downloadAnalysis, finiteDecimal as finite } from './analysisFiles.js'
import { returnedPlotBounds } from './returnedPlotBounds.js'

const props = defineProps({ parseSamples: { type: Function, required: true }, sampleRevision: { type: String, required: true }, provenance: { type: Object, required: true } })
const input = reactive({ mode: 'binary', threshold: .5, selected: false, source: '', temperature: '', gateVoltage: '' })
const result = ref(null), rawJson = ref(''), error = ref(''), busy = ref(false), stale = ref(false)
let revision = 0, disposed = false
watch([input, () => props.sampleRevision, () => props.provenance], () => {
  revision++; stale.value = result.value !== null || busy.value; result.value = null; rawJson.value = ''; error.value = ''
}, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
const rows = computed(() => result.value?.result.samples ?? [])
const datasets = computed(() => [
  ['current_a', 'Signed device current', '#475569'], ['gate_on_current_a', 'Gate-on channel allocation', '#0369a1'],
  ['gate_off_current_a', 'Gate-off diode allocation', '#b45309'],
].map(([key, label, color]) => ({ label, borderColor: color, backgroundColor: key === 'current_a' ? 'transparent' : color,
  showLine: false, pointRadius: key === 'current_a' ? 6 : 4,
  data: rows.value.map(row => ({ x: row.time_s, y: row[key] })) })))
const bounds = computed(() => {
  const { min, max } = returnedPlotBounds(datasets.value)
  if (min === null || max === null) return { min, max }
  const span = max - min
  const padding = (Number.isFinite(span) ? span : Math.max(Math.abs(min), Math.abs(max))) * .05
  return { min: Number.isFinite(min - padding) ? min - padding : min,
    max: Number.isFinite(max + padding) ? max + padding : max }
})
const gateDataset = computed(() => [{ label: 'Gate state (0 = low, 1 = high)', showLine: false, pointRadius: 4,
  data: rows.value.map(row => ({ x: row.time_s, y: Number(row.gate_on) })) }])
const csv = computed(() => analysisCsv(rows.value.map(row => ({ ...row, source: result.value.inputs.source,
  conditions: result.value.inputs.conditions, provenance: result.value.inputs.provenance,
  gate_policy: result.value.inputs.gate.mode, threshold_v: result.value.inputs.gate.threshold_v ?? null,
  semantics: result.value.result.semantics, scope: result.value.result.scope,
  input_sha256: result.value.evidence.input_sha256, result_sha256: result.value.evidence.result_sha256 }))))
function request() {
  if (!input.selected) throw new Error('Explicitly select the gate-state allocation assumption')
  if (!input.source.trim()) throw new Error('Supply a nonblank allocation source')
  const columns = props.parseSamples(true, input.mode === 'binary')
  if (!columns.gate_v) throw new Error('Gate allocation requires a gate_v sample at every time')
  const gate = input.mode === 'binary' ? { mode: 'binary', state: columns.gate_v }
    : { mode: 'threshold', gate_v: columns.gate_v, threshold_v: finite(input.threshold, 'Allocation gate threshold') }
  if (input.mode === 'binary' && columns.gate_v.some(value => value !== 0 && value !== 1)) throw new Error('Binary gate state requires numeric 0/1 samples')
  return { time_s: columns.time_s, current_a: columns.current_a, gate, selected: true, source: input.source.trim(),
    conditions: { junction_temperature_c: input.temperature === '' ? null : finite(input.temperature, 'Junction temperature'),
      gate_on_voltage_v: input.gateVoltage === '' ? null : finite(input.gateVoltage, 'Gate-on voltage') }, provenance: { ...props.provenance } }
}
async function calculate() {
  const token = ++revision
  result.value = null; rawJson.value = ''; error.value = ''; stale.value = false; busy.value = true
  try {
    const response = await waveformApi.allocate(request())
    if (disposed || token !== revision) return
    result.value = response.data; rawJson.value = response.rawJson
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed) busy.value = false }
}
</script>

<template>
  <section class="gate-allocation" data-testid="gate-allocation">
    <details>
      <summary>Gate-state current allocation</summary>
      <p>This panel reads the waveform CSV above, including when the loss calculation uses advanced JSON. Gate-on current is the channel allocation. Gate-off current is the diode allocation.</p>
      <p>These command-state allocations do not identify physical current paths or simultaneous sharing. They do not change the waveform loss calculation.</p>
      <form @submit.prevent="calculate">
        <label>Allocation gate policy<select v-model="input.mode" data-testid="allocation-mode"><option value="binary">Binary gate_v: numeric 0/1 only</option><option value="threshold">Voltage gate_v &gt; declared threshold; equality is low</option></select></label>
        <label v-if="input.mode === 'threshold'">Allocation threshold (V)<input :value="input.threshold" @input="input.threshold = $event.target.value" type="number" step="any" data-testid="allocation-threshold"></label>
        <label>Allocation source / assumption<input v-model="input.source" data-testid="allocation-source" placeholder="State the source and the command-state assumption"></label>
        <div class="columns"><label>Junction temperature (°C), blank = unknown<input :value="input.temperature" @input="input.temperature = $event.target.value" type="number" step="any"></label><label>Gate-on voltage (V), blank = unknown<input :value="input.gateVoltage" @input="input.gateVoltage = $event.target.value" type="number" step="any"></label></div>
        <label class="check"><input v-model="input.selected" type="checkbox" data-testid="allocation-selected"> I select the command-state allocation assumption.</label>
        <p>Binary masks accept actual booleans or numeric 0/1 through the API. This CSV uses numeric 0/1. Every sample keeps its signed current. No state or interpolation is defined between samples.</p>
        <button type="submit" :disabled="busy" data-testid="allocation-calculate">{{ busy ? 'Allocating…' : 'Allocate gate-state current' }}</button>
        <p v-if="error" role="alert">{{ error }}</p><p v-if="stale && !result" role="status">Inputs changed. Allocate again to show current results.</p>
      </form>
      <section v-if="result" data-testid="allocation-results">
        <h3>{{ result.result.sample_count }} original signed samples</h3>
        <p>{{ result.result.semantics }} {{ result.result.scope }}</p>
        <p>Source: {{ result.inputs.source }}. Waveform source: {{ result.inputs.provenance?.source ?? 'Unknown' }}.</p>
        <CurvePlotter title="Signed sample allocations" :datasets="datasets" x-label="Time (s)" y-label="Current (A)" :y-min="bounds.min" :y-max="bounds.max" :allow-zoom="false"/>
        <CurvePlotter title="Original sample gate state" :datasets="gateDataset" x-label="Time (s)" y-label="Gate state (0/1)" :y-min="-.1" :y-max="1.1" :allow-zoom="false"/>
        <p>Plots show original sample markers only. The table previews the first 100 samples. CSV and JSON contain every sample.</p>
        <div class="table-scroll"><table><thead><tr><th>Time (s)</th><th>Device current (A)</th><th>Gate state</th><th>Gate-on channel allocation (A)</th><th>Gate-off diode allocation (A)</th></tr></thead><tbody><tr v-for="(row, index) in rows.slice(0, 100)" :key="index"><td>{{ row.time_s }}</td><td>{{ row.current_a }}</td><td>{{ row.gate_on ? 1 : 0 }}</td><td>{{ row.gate_on_current_a }}</td><td>{{ row.gate_off_current_a }}</td></tr></tbody></table></div>
        <div class="actions"><button type="button" data-testid="allocation-json" @click="downloadAnalysis(rawJson, 'gate-state-allocation.json', 'application/octet-stream')">Download request + result JSON</button><button type="button" data-testid="allocation-csv" @click="downloadAnalysis(csv, 'gate-state-allocation.csv', 'text/csv')">Download sample CSV</button></div>
        <details><summary>Review exact response JSON</summary><pre>{{ rawJson }}</pre></details>
      </section>
    </details>
  </section>
</template>

<style scoped>
.gate-allocation { margin: 1rem 0; padding: 1rem; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; min-width: 0; }
summary { cursor: pointer; font-weight: 700; } p { line-height: 1.6; overflow-wrap: anywhere; }
label { display: grid; gap: .4rem; margin: .7rem 0; } .check { display: flex; gap: .5rem; }
input:not([type=checkbox]), select { width: 100%; min-width: 0; box-sizing: border-box; padding: .6rem; font: inherit; border: 1px solid #94a3b8; border-radius: 6px; }
.columns { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 1rem; }
button { padding: .65rem .9rem; font: inherit; cursor: pointer; border: 1px solid #94a3b8; border-radius: 6px; background: white; } .actions { display: flex; flex-wrap: wrap; gap: .7rem; margin: 1rem 0; }
[role=alert] { background: #fee2e2; color: #991b1b; padding: 1rem; } pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.table-scroll { max-width: 100%; overflow-x: auto; } table { width: 100%; border-collapse: collapse; } td, th { padding: .6rem; border-bottom: 1px solid #cbd5e1; text-align: left; }
@media(max-width: 780px) { .columns { grid-template-columns: 1fr; } .gate-allocation { padding: .8rem; } }
</style>
