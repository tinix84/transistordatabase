<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { assemblyApi } from '../services/api.js'
import CurvePlotter from './CurvePlotter.vue'
import { analysisCsv, analysisError, displayNumber, downloadAnalysis, finiteDecimal } from './analysisFiles.js'
import { returnedPlotBounds } from './returnedPlotBounds.js'

const branch = () => ({ csv: '', source: '', isolation: '', recordedTemperature: '', requestedTemperature: '', recordedGate: '', requestedGate: '' })
const input = reactive({ current: '', temperature: '', mode: 'strict-recorded', selected: false, compatible: false, channel: branch(), diode: branch() })
const result = ref(null), rawJson = ref(''), error = ref(''), busy = ref(false), stale = ref(false)
let revision = 0, disposed = false
watch(input, () => {
  revision++; stale.value = result.value !== null || busy.value || stale.value
  result.value = null; rawJson.value = ''; error.value = ''; busy.value = false
}, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })

function loadFixture() {
  Object.assign(input, { current: '-10', temperature: '25', mode: 'strict-recorded', selected: false, compatible: false,
    channel: { csv: 'current_a,voltage_v\n0,0\n10,1', source: 'Hand fixture: Vc = 0.1 Ic. Not a measured device.', isolation: 'Constructed channel-only function. No diode contribution.', recordedTemperature: '25', requestedTemperature: '25', recordedGate: '15', requestedGate: '15' },
    diode: { csv: 'current_a,voltage_v\n0,0.6\n10,0.8', source: 'Hand fixture: Vd = 0.6 + 0.02 Id. Not a measured device.', isolation: 'Constructed diode-only function. No channel contribution.', recordedTemperature: '25', requestedTemperature: '25', recordedGate: '0', requestedGate: '0' } })
}
function curve(key) {
  const value = input[key], label = key === 'channel' ? 'Channel' : 'Diode'
  const lines = value.csv.trim().split(/\r?\n/).map(line => line.trim())
  if (lines.shift() !== 'current_a,voltage_v') throw new Error(`${label} CSV requires the header current_a,voltage_v.`)
  if (lines.length < 2) throw new Error(`${label} CSV requires at least two points.`)
  const points = lines.map((line, index) => {
    const cells = line.split(',')
    if (cells.length !== 2) throw new Error(`${label} CSV row ${index + 2} requires two unquoted SI values.`)
    return cells.map((cell, column) => finiteDecimal(cell, `${label} row ${index + 2} ${column ? 'voltage' : 'current'}`))
  })
  if (!value.source.trim() || !value.isolation.trim()) throw new Error(`${label} requires a source and source-only isolation evidence.`)
  const conditions = prefix => ({ junction_temperature_c: finiteDecimal(value[`${prefix}Temperature`], `${label} ${prefix} junction temperature`),
    gate_bias_v: value[`${prefix}Gate`].trim() === '' ? null : finiteDecimal(value[`${prefix}Gate`], `${label} ${prefix} gate bias`) })
  return { kind: `${key}-only`, current_a: points.map(point => point[0]), voltage_v: points.map(point => point[1]), source: value.source.trim(),
    isolation_evidence: value.isolation.trim(), recorded_conditions: conditions('recorded'), requested_conditions: conditions('requested') }
}
function request() {
  if (!input.selected) throw new Error('Acknowledge the source curves and isolation evidence.')
  if (!input.compatible) throw new Error('Confirm the compatible operating state of both branches.')
  return { selected: true, mode: input.mode, current_a: finiteDecimal(input.current, 'Signed total reverse current'),
    junction_temperature_c: finiteDecimal(input.temperature, 'Common junction temperature'), compatible_operating_state: true,
    channel: curve('channel'), diode: curve('diode') }
}
async function calculate() {
  const token = ++revision
  result.value = null; rawJson.value = ''; error.value = ''; stale.value = false; busy.value = true
  try {
    const response = await assemblyApi.reverseSharing(request())
    if (disposed || token !== revision) return
    result.value = response.data; rawJson.value = response.rawJson
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
const solved = computed(() => result.value?.result)
const datasets = computed(() => {
  if (!result.value) return []
  const data = ['channel', 'diode'].map((key, index) => {
    const source = result.value.inputs[key]
    return { label: `${key} source-only curve`, borderColor: index ? '#b45309' : '#0369a1', pointRadius: 3,
      data: source.current_a.map((x, i) => ({ x, y: source.voltage_v[i] })) }
  })
  if (solved.value.status === 'supported') {
    for (const [key, color] of [['channel', '#0369a1'], ['diode', '#b45309']]) {
      const current = solved.value[`${key}_current_a`], voltage = solved.value.voltage_v
      if (Number.isFinite(current) && Number.isFinite(voltage)) data.push({ label: `${key} solved point${solved.value.support === 'estimate' && key === 'diode' ? ' (ideal blocking estimate, off source curve)' : ''}`,
        showLine: false, pointRadius: 7, borderColor: color, backgroundColor: color, data: [{ x: -current, y: -voltage }] })
    }
  }
  return data
})
function paddedBounds(data) {
  const { min, max } = returnedPlotBounds(data)
  if (min === null || max === null) return { min, max }
  const span = max - min, padding = (Number.isFinite(span) ? span : Math.max(Math.abs(min), Math.abs(max))) * .07
  return { min: Number.isFinite(min - padding) ? min - padding : min, max: Number.isFinite(max + padding) ? max + padding : max }
}
const yBounds = computed(() => paddedBounds(datasets.value))
const xBounds = computed(() => paddedBounds(datasets.value.map(dataset => ({ data: dataset.data.map(point => ({ y: point.x })) }))))
const values = [ ['channel_current_a', 'Signed channel current (A)'], ['diode_current_a', 'Signed diode current (A)'], ['current_sum_a', 'Signed current sum (A)'],
  ['current_residual_a', 'Current residual (A)'], ['voltage_v', 'Common signed Vds (V)'], ['channel_static_power_w', 'Channel static power (W)'],
  ['diode_static_power_w', 'Diode static power (W)'], ['total_static_power_w', 'Total static power (W)'], ['blocking_margin_v', 'Ideal diode blocking margin (V)'] ]
const csv = computed(() => result.value ? analysisCsv([{ ...solved.value, conditions_and_sources: result.value.inputs, evidence: result.value.evidence }]) : '')
</script>

<template>
  <section id="reverse-current-sharing" class="reverse-sharing" data-testid="reverse-sharing">
    <details>
      <summary>Physical reverse-current sharing at one operating point</summary>
      <p>The assembly envelope above selects the lower drop. This separate solver finds simultaneous channel and diode currents at one common voltage.</p>
      <p>Supply isolated source-only curves. A reverse-on terminal curve that already includes diode current is unsuitable.</p>
      <p>Static power estimates use the declared curves and conditions. They do not predict switching losses, thermal feedback, or device accuracy.</p>
      <button type="button" data-testid="sharing-fixture" @click="loadFixture">Load hand-checkable fixture (not measured device data)</button>
      <form @submit.prevent="calculate">
        <div class="columns">
          <label>Signed total reverse current (A), zero allowed<input :value="input.current" @input="input.current = $event.target.value" inputmode="decimal" data-testid="sharing-current"></label>
          <label>Common requested junction temperature (°C)<input :value="input.temperature" @input="input.temperature = $event.target.value" inputmode="decimal" data-testid="sharing-temperature"></label>
        </div>
        <label>Support policy<select v-model="input.mode" data-testid="sharing-mode"><option value="strict-recorded">Strict recorded support (default)</option><option value="ideal-diode-blocking">Explicit ideal diode blocking estimate</option></select></label>
        <p v-if="input.mode === 'ideal-diode-blocking'">This selected estimate permits an ideal blocked diode below its recorded zero-current drop. It does not extrapolate a conducting diode curve.</p>
        <div class="columns">
          <fieldset v-for="key in ['channel', 'diode']" :key="key">
            <legend>{{ key === 'channel' ? 'Channel-only branch' : 'Diode-only branch' }}</legend>
            <label>Source<input v-model="input[key].source" :data-testid="`sharing-${key}-source`"></label>
            <label>Source-only isolation evidence<textarea v-model="input[key].isolation" rows="2" :data-testid="`sharing-${key}-isolation`" /></label>
            <label>Source CSV: magnitude current (A), positive drop (V)<textarea v-model="input[key].csv" rows="6" :data-testid="`sharing-${key}-csv`" placeholder="current_a,voltage_v" spellcheck="false" /></label>
            <p>Use unquoted decimal SI values. No blank values become zero. No interpolation occurs outside the recorded current range.</p>
            <div v-for="prefix in ['recorded', 'requested']" :key="prefix" class="conditions">
              <label>{{ prefix }} junction temperature (°C)<input :value="input[key][`${prefix}Temperature`]" @input="input[key][`${prefix}Temperature`] = $event.target.value" inputmode="decimal" :data-testid="`sharing-${key}-${prefix}-temperature`"></label>
              <label>{{ prefix }} gate bias (V), blank = unknown / not applicable<input :value="input[key][`${prefix}Gate`]" @input="input[key][`${prefix}Gate`] = $event.target.value" inputmode="decimal" :data-testid="`sharing-${key}-${prefix}-gate`"></label>
            </div>
          </fieldset>
        </div>
        <p>Both curves require the same known junction temperature as the requested point. Declare each gate bias independently. Blank gate bias remains unknown / not applicable. No condition scaling applies.</p>
        <label class="check"><input v-model="input.selected" type="checkbox" data-testid="sharing-selected"> I reviewed the source-only curves and their isolation evidence.</label>
        <label class="check"><input v-model="input.compatible" type="checkbox" data-testid="sharing-compatible"> I confirm both branch conditions describe a compatible simultaneous operating state.</label>
        <button type="submit" :disabled="busy" data-testid="sharing-calculate">{{ busy ? 'Solving…' : 'Solve reverse-current sharing' }}</button>
        <p v-if="error" role="alert">{{ error }}</p>
        <p v-if="stale && !result" role="status">Inputs changed. Solve again to show current results and exports.</p>
      </form>
      <section v-if="result" data-testid="sharing-results">
        <h3>{{ solved.status === 'supported' ? 'Calculated static split' : `Split unavailable: ${solved.status}` }}</h3>
        <p>Basis: {{ solved.support === 'recorded' ? 'Recorded branch curves' : solved.support === 'estimate' ? 'Selected blocking estimate' : 'No supported split' }}.</p>
        <p>This is a piecewise linear curve and root calculation. The calculated split is not a measured current split.</p>
        <p>{{ solved.diagnostic }}</p><p>{{ solved.scope }}</p>
        <p v-if="solved.status !== 'supported'">The split is unavailable. No solved point is plotted. Unavailable values remain unknown.</p>
        <div class="table-scroll"><table><tbody><tr v-for="[key, label] in values" :key="key"><th>{{ label }}</th><td>{{ displayNumber(solved[key]) }}</td></tr></tbody></table></div>
        <p>Feasible channel magnitude bounds (A): {{ displayNumber(solved.feasible_channel_current_magnitude_a?.lower_a) }} to {{ displayNumber(solved.feasible_channel_current_magnitude_a?.upper_a) }}.</p>
        <p>Signed currents are negative in reverse conduction. The plot uses current magnitude and positive drop. Solved points share one drop.</p>
        <CurvePlotter title="Accepted source-only curves and solved branch points" :datasets="datasets" x-label="Branch current magnitude (A)" y-label="Positive branch drop (V)" :x-min="xBounds.min" :x-max="xBounds.max" :y-min="yBounds.min" :y-max="yBounds.max" :allow-zoom="false" />
        <details open><summary>Root proof, equations and units</summary><pre>{{ JSON.stringify({ root_evidence: solved.root_evidence, equations: solved.equations, units: solved.units }, null, 2) }}</pre></details>
        <div class="actions"><button type="button" data-testid="sharing-json" @click="downloadAnalysis(rawJson, 'reverse-current-sharing.json', 'application/octet-stream')">Download exact request + result JSON</button><button type="button" data-testid="sharing-csv" @click="downloadAnalysis(csv, 'reverse-current-sharing.csv', 'text/csv')">Download conditions + proof CSV</button></div>
        <details><summary>Review exact server response</summary><pre>{{ rawJson }}</pre></details>
      </section>
    </details>
  </section>
</template>

<style scoped>
.reverse-sharing { padding: 1rem; margin-top: 1rem; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; min-width: 0; }
summary { font-weight: 700; cursor: pointer; } p { line-height: 1.6; overflow-wrap: anywhere; } label { display: block; margin: .7rem 0; }
input:not([type=checkbox]), textarea, select { width: 100%; min-width: 0; box-sizing: border-box; padding: .45rem; } textarea { font-family: monospace; }
.columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; } fieldset { min-width: 0; border: 1px solid #cbd5e1; }
button { padding: .6rem .8rem; margin: .5rem .4rem .5rem 0; max-width: 100%; } .table-scroll { overflow-x: auto; } table { width: 100%; border-collapse: collapse; } th, td { padding: .5rem; text-align: left; border-bottom: 1px solid #cbd5e1; overflow-wrap: anywhere; }
pre { overflow: auto; max-width: 100%; white-space: pre-wrap; overflow-wrap: anywhere; } [role=alert] { color: #a11; } @media(max-width: 760px) { .columns { grid-template-columns: 1fr; } .reverse-sharing { padding: .6rem; } }
</style>
