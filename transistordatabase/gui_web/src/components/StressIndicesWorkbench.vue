<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { stressApi } from '../services/api.js'
import { analysisCsv, analysisError, displayNumber as number, downloadAnalysis, finiteDecimal as finite } from './analysisFiles.js'

const conditionLabels = { junction_temperature_c: 'Junction temperature (°C)', gate_on_voltage_v: 'Gate on voltage (V)', gate_off_voltage_v: 'Gate off voltage (V)', dc_bus_voltage_v: 'DC bus voltage (V)' }
const input = reactive({ source: '', samples: '', transitions: '', transitionMode: 'unknown', resistance: '',
  conditions: Object.fromEntries(Object.keys(conditionLabels).map(key => [key, ''])) })
const result = ref(null), reviewJson = ref(null), error = ref(''), busy = ref(false), stale = ref(false)
let revision = 0, disposed = false
watch(input, () => { revision++; stale.value = stale.value || !!result.value || busy.value; result.value = null; reviewJson.value = null; error.value = ''; busy.value = false }, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
const metricInfo = {
  conduction: { title: 'Conduction stress index', equation: 'I_rms²', unit: 'A²' },
  volume: { title: 'Volume stress index', equation: 'I_rms² × max(V)²', unit: 'A² V²' },
  switching: { title: 'Switching stress index', equation: 'Σ f_event × I(event)² × V(event)', unit: 'Hz A² V' },
  cost: { title: 'Cost stress index', equation: '1 / (R_DS(on) × max(V)²)', unit: '1/(ohm V²)' },
}
function csv(text, header) {
  const lines = text.trim().split(/\r?\n/)
  if (lines.shift().split(',').map(value => value.trim()).join(',') !== header.join(',')) throw new Error(`Use the exact CSV header: ${header.join(',')}.`)
  return lines.map((line, index) => {
    const values = line.split(',').map(value => value.trim())
    if (values.length !== header.length) throw new Error(`CSV row ${index + 2} does not match the header.`)
    return Object.fromEntries(header.map((key, col) => [key, key === 'kind' ? values[col] : finite(values[col], `CSV row ${index + 2} ${key}`)]))
  })
}
function request() {
  if (!input.source.trim()) throw new Error('Declare the waveform source.')
  const samples = csv(input.samples, ['time_s', 'current_a', 'voltage_v'])
  if (samples.length < 2) throw new Error('Supply at least two SI waveform samples.')
  const transitions = input.transitionMode === 'unknown' ? null : input.transitionMode === 'none' ? [] : csv(input.transitions, ['time_s', 'frequency_hz', 'kind'])
  if (input.transitionMode === 'provided' && !transitions.length) throw new Error('Supply at least one transition record, or select explicitly no transitions.')
  return { source: input.source, conditions: Object.fromEntries(Object.entries(input.conditions).map(([key, value]) => [key, String(value).trim() === '' ? null : finite(value, conditionLabels[key])])),
    time_s: samples.map(row => row.time_s), current_a: samples.map(row => row.current_a), voltage_v: samples.map(row => row.voltage_v),
    rds_on_ohm: String(input.resistance).trim() === '' ? null : finite(input.resistance, 'On resistance'),
    transitions }
}
function example() {
  input.source = 'Independent linear-current hand fixture. No measured device or calibrated simulation data.'
  input.samples = 'time_s,current_a,voltage_v\n0,0,400\n1,6,400'
  input.resistance = '0.1'; input.transitionMode = 'provided'
  input.transitions = 'time_s,frequency_hz,kind\n0.5,1000,turn_on'
  for (const key of Object.keys(input.conditions)) input.conditions[key] = ''
}
async function calculate() {
  const token = ++revision; error.value = ''; result.value = null; reviewJson.value = null; busy.value = true; stale.value = false
  try {
    const payload = request(), response = await stressApi.calculate(payload)
    if (disposed || token !== revision) return
    result.value = response.data; reviewJson.value = response.rawJson
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
const metrics = computed(() => Object.entries(metricInfo).map(([key, info]) => ({ key, ...info, metric: result.value?.result.metrics[key] })))
function exportCsv() {
  const response = result.value
  downloadAnalysis(analysisCsv(metrics.value.map(item => ({ index: item.key, value: item.metric.value, unit: item.metric.unit,
    status: item.metric.status, reason: item.metric.reason ?? null, equation: item.equation, source: response.inputs.source,
    conditions: response.inputs.conditions, time_s: response.inputs.time_s, current_a: response.inputs.current_a, voltage_v: response.inputs.voltage_v,
    rds_on_ohm: response.inputs.rds_on_ohm, transitions: response.inputs.transitions,
    method: response.evidence.method, input_sha256: response.evidence.input_sha256, result_sha256: response.evidence.result_sha256,
    hash_verification: 'Verify hashes with the exact review JSON download. CSV numeric formatting is not hash evidence.' }))), 'semiconductor-stress-indices.csv', 'text/csv')
}
</script>

<template>
  <section class="stress-workbench" data-testid="stress-workbench">
    <header><h2>Semiconductor stress indices</h2><p>Calculate the four ENCO MOSFET stress proxies from explicit SI waveform samples and transition records.</p>
      <p>These indices describe electrical stress. They do not calculate loss power, physical volume, or purchase price.</p></header>
    <form @submit.prevent="calculate">
      <section class="panel"><h3>1. Source and waveform</h3>
        <button type="button" @click="example" data-testid="stress-example">Load hand fixture</button>
        <label>Waveform source<textarea v-model="input.source" rows="2" data-testid="stress-source" placeholder="Declare the measurement, simulation, or manual source and its limits."></textarea></label>
        <label>Waveform CSV · time (s), signed current (A), signed voltage (V)<textarea v-model="input.samples" rows="7" spellcheck="false" data-testid="stress-samples" placeholder="time_s,current_a,voltage_v"></textarea></label>
        <p>Use the exact header time_s,current_a,voltage_v. Supply 2–20,000 aligned samples with strictly increasing time.</p>
        <p>Python integrates squared piecewise-linear current over the full supplied interval. Peak voltage is the maximum signed sample.</p>
      </section>
      <section class="panel"><h3>2. Conditions and optional evidence</h3><p>Blank condition values and blank on resistance remain unknown. Record the conditions that apply to the supplied waveform.</p>
        <div class="fields"><label v-for="(label, key) in conditionLabels" :key="key">{{ label }}<input v-model="input.conditions[key]" type="number" step="any" :data-testid="'stress-condition-' + key" placeholder="Unknown"></label>
          <label>On resistance R_DS(on) (ohm)<input v-model="input.resistance" type="number" step="any" data-testid="stress-resistance" placeholder="Unknown"></label>
          <label>Transition evidence<select v-model="input.transitionMode" data-testid="stress-transition-mode"><option value="unknown">Unknown · no transition evidence</option><option value="none">Explicitly no transitions · zero switching index</option><option value="provided">Supplied transition records</option></select></label>
        </div>
        <label v-if="input.transitionMode === 'provided'">Transition CSV · time (s), frequency (Hz), kind<textarea v-model="input.transitions" rows="5" spellcheck="false" data-testid="stress-transitions" placeholder="time_s,frequency_hz,kind"></textarea></label>
        <p>Each transition time must lie inside the waveform. Kinds are turn_on and turn_off. Frequencies are explicit nonnegative event weights.</p>
        <p>The switching index uses signed voltage and squared interpolated current. Python does not infer events from the waveform.</p>
      </section>
      <button type="submit" :disabled="busy" data-testid="stress-calculate">{{ busy ? 'Calculating…' : 'Calculate stress indices' }}</button>
    </form>
    <p v-if="busy" role="status">Calculating the supplied waveform in Python…</p>
    <p v-if="stale" data-testid="stress-stale">Inputs changed. Calculate again before review or download.</p>
    <p v-if="error" role="alert">{{ error }}</p>
    <section v-if="result" class="panel" data-testid="stress-results"><h3>Calculated stress proxies</h3>
      <p>{{ result.result.scope }}</p>
      <div class="metrics"><article v-for="item in metrics" :key="item.key" :data-testid="'stress-metric-' + item.key"><h4>{{ item.title }}</h4><p class="equation">{{ item.equation }}</p>
        <p class="metric-value">{{ number(item.metric.value) }} <span>{{ item.unit }}</span></p><p>Status: {{ item.metric.status }}</p><p v-if="item.metric.reason">{{ item.metric.reason }}</p></article></div>
      <p>RMS current: {{ number(result.result.current_rms_a) }} A · Mean square current: {{ number(result.result.current_mean_square_a2) }} A²</p>
      <p>Maximum signed voltage: {{ number(result.result.peak_voltage_v) }} V · Duration: {{ number(result.result.duration_s) }} s · Samples: {{ result.result.sample_count }}</p>
      <h4>Accepted source and conditions</h4><p data-testid="stress-accepted-source">{{ result.inputs.source }}</p>
      <dl><template v-for="(value, key) in result.inputs.conditions" :key="key"><dt>{{ conditionLabels[key] ?? key }}</dt><dd>{{ value === null ? 'Unknown' : value }}</dd></template><dt>On resistance (ohm)</dt><dd>{{ result.inputs.rds_on_ohm === null ? 'Unknown' : number(result.inputs.rds_on_ohm) }}</dd><dt>Transition evidence</dt><dd>{{ result.inputs.transitions === null ? 'Unknown' : `${result.inputs.transitions.length} explicit records` }}</dd></dl>
      <div v-if="result.result.transition_samples.length" class="table-wrap"><table data-testid="stress-transition-results"><thead><tr><th>Kind</th><th>Time (s)</th><th>Frequency (Hz)</th><th>Current (A)</th><th>Voltage (V)</th><th>Contribution (Hz A² V)</th></tr></thead><tbody><tr v-for="(row, index) in result.result.transition_samples" :key="index"><td>{{ row.kind }}</td><td>{{ number(row.time_s) }}</td><td>{{ number(row.frequency_hz) }}</td><td>{{ number(row.current_a) }}</td><td>{{ number(row.voltage_v) }}</td><td>{{ number(row.contribution) }}</td></tr></tbody></table></div>
      <h4>Calculation provenance</h4><p>Method: {{ result.evidence.method }}</p><p class="hash" data-testid="stress-hash">Input SHA-256: {{ result.evidence.input_sha256 }}</p><p class="hash" data-testid="stress-result-hash">Result SHA-256: {{ result.evidence.result_sha256 }}</p>
      <details><summary>Python definitions and accepted SI input</summary><pre>{{ JSON.stringify({ definitions: result.result.definitions, inputs: result.inputs }, null, 2) }}</pre></details>
      <p>The review JSON preserves the server text for hash verification. CSV provides a tabular view.</p>
      <div class="downloads"><button type="button" @click="downloadAnalysis(reviewJson, 'semiconductor-stress-indices.json', 'application/json;charset=utf-8')" data-testid="stress-json">Download review JSON</button><button type="button" @click="exportCsv" data-testid="stress-csv">Download indices CSV</button></div>
    </section>
  </section>
</template>

<style scoped>
.stress-workbench { max-width: 1180px; margin: auto; padding: 1rem; color: var(--text-color, #263444); }
.panel { border: 1px solid var(--border-color, #cbd5df); border-radius: 8px; padding: 1rem; margin: 1rem 0; background: var(--card-bg, #fff); }
.fields,.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: 1rem; }
label { display: flex; flex-direction: column; gap: .4rem; margin: .8rem 0; }
input,select,textarea { width: 100%; min-width: 0; box-sizing: border-box; padding: .6rem; font: inherit; border: 1px solid #aab8c5; border-radius: 4px; background: var(--input-bg, #fff); color: inherit; }
textarea { resize: vertical; } button { padding: .65rem 1rem; cursor: pointer; max-width: 100%; } button:disabled { cursor: wait; }
article { padding: 1rem; background: var(--bg-secondary, #f2f6f9); border-radius: 6px; min-width: 0; }
h4 { margin-top: .3rem; } .equation { font-family: monospace; overflow-wrap: anywhere; }.metric-value { font-size: 1.3rem; font-weight: 700; }.metric-value span { font-size: 1rem; }
.table-wrap { overflow-x: auto; } table { border-collapse: collapse; width: 100%; } th,td { padding: .6rem; border-bottom: 1px solid #cbd5df; text-align: left; white-space: nowrap; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; } .hash,p,dd { overflow-wrap: anywhere; } dl { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: .5rem; } dd { margin: 0; }
.downloads { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1rem; } [role=alert] { color: #af2b24; } [data-testid=stress-stale] { padding: .8rem; background: #fff0cf; color: #5b4307; }
@media (max-width: 600px) { .stress-workbench { padding: .5rem; } .panel { padding: .75rem; } }
</style>
