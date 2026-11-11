<template>
  <section class="assembly-workbench">
    <h2>Semiconductor assemblies</h2>
    <p>The curve transformations below use ideal equal sharing, separate signed branches, or an explicitly selected reverse-on envelope. These curve transformations do not solve current balancing or simultaneous channel/diode sharing.</p>
    <p><a href="#reverse-current-sharing">Go to reverse-current sharing</a> to calculate a simultaneous static split from isolated branch curves.</p>
    <form @submit.prevent="calculate">
      <label>Assembly operation <select v-model="operation" aria-label="Assembly operation"><option value="parallel-bank">Ideal equal-sharing parallel bank</option><option value="signed-composite">Signed IGBT and diode composite</option><option value="reverse-envelope">Ideal reverse-on MOSFET envelope</option></select></label>
      <p>Changing the operation loads illustrative values. Replace sources and samples before using a device result. Blank optional fields remain unknown.</p>
      <template v-if="operation === 'parallel-bank'">
        <label>Parallel device count <input v-model.number="input.count" type="number" min="1" step="1" aria-label="Parallel device count" /></label>
        <AssemblyDeviceEditor v-model="input.device" title="Single device" @invalid="invalidInput" />
      </template>
      <template v-else-if="operation === 'signed-composite'">
        <label>Explicit common thermal resistance (K/W) <input v-model.number="input.common_resistance_k_per_w" type="number" step="any" aria-label="Common thermal resistance" /></label>
        <label><input v-model="voltageSelected" type="checkbox" aria-label="Select linear voltage estimate" /> Select linear energy-versus-voltage estimate for differing test voltages</label>
        <label v-if="voltageSelected">Target supply voltage (V) <input v-model.number="targetVoltage" type="number" step="any" aria-label="Target supply voltage" /></label>
        <div class="devices"><AssemblyDeviceEditor v-model="input.positive" title="Positive IGBT" @invalid="invalidInput" /><AssemblyDeviceEditor v-model="input.negative" title="Negative diode" @invalid="invalidInput" /></div>
      </template>
      <template v-else>
        <label><input v-model="input.selected" type="checkbox" aria-label="Select ideal envelope" /> Select the ideal lower-drop envelope estimate</label>
        <div class="devices"><AssemblyTableEditor v-model="input.channel" title="Reverse-on channel" @invalid="invalidInput" /><AssemblyTableEditor v-model="input.diode" title="Reverse diode" @invalid="invalidInput" /></div>
      </template>
      <details><summary>Advanced reproducible input (JSON; SI units)</summary><label>Canonical assembly input<textarea v-model="advancedText" aria-label="Assembly JSON" rows="15" @input="advancedDirty = true" @change="applyAdvanced" /></label><button type="button" @click="applyAdvanced">Apply JSON input</button></details>
      <button type="submit" :disabled="busy || invalid">{{ busy ? 'Calculating…' : 'Calculate assembly' }}</button>
    </form>
    <p v-if="error" role="alert">{{ error }}</p>
    <section v-if="result" data-testid="assembly-results" aria-label="Assembly results">
      <h3>Result in SI units</h3>
      <table><tbody><tr v-for="row in rows" :key="row[0]"><th>{{ row[0] }}</th><td>{{ format(row[1]) }}</td></tr></tbody></table>
      <p v-for="assumption in result.assumptions" :key="assumption">{{ assumption }}</p>
      <p v-for="estimate in result.estimates || []" :key="estimate.branch + estimate.event">{{ estimate.branch }} {{ estimate.event }}: linear-voltage-estimate factor {{ format(estimate.factor) }}; recorded {{ JSON.stringify(estimate.recorded_conditions) }}; requested {{ JSON.stringify(estimate.requested_conditions) }}</p>
      <div class="plots"><CurvePlotter v-for="plot in plots" :key="plot.title" :title="plot.title" x-label="Signed current (A)" :y-label="plot.yLabel" :datasets="plot.datasets" /></div>
      <details><summary>All result values, conditions and sources</summary><pre>{{ JSON.stringify(result, null, 2) }}</pre></details>
      <button @click="download">Download assembly JSON</button>
    </section>
    <ReverseSharingPanel />
  </section>
</template>
<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { assemblyApi } from '../services/api.js'
import AssemblyDeviceEditor from './AssemblyDeviceEditor.vue'
import AssemblyTableEditor from './AssemblyTableEditor.vue'
import CurvePlotter from './CurvePlotter.vue'
import ReverseSharingPanel from './ReverseSharingPanel.vue'
const clone = value => JSON.parse(JSON.stringify(value))
const table = (kind, y, source, extra = {}) => ({ kind, x: [0, 10], y, conditions: { t_j_c: 25, v_g_v: 15, ...extra }, source })
const device = (source = 'Illustrative single device — replace source') => ({ source, blocking_voltage_v: 1200, current_rating_a: 10, area_m2: .01, cost: 2, currency: 'CHF', weight_kg: .02, internal_resistance_k_per_w: 1, shared_sink_resistance_k_per_w: .2, foster: null, conduction: table('channel', [0, 2], source), turn_on: table('energy', [.0001, .001], source, { v_supply_v: 400, r_g_ohm: null }), turn_off: table('energy', [.0002, .002], source, { v_supply_v: 400, r_g_ohm: null }) })
function example(operation) {
  if (operation === 'parallel-bank') return { operation, count: 3, device: device() }
  if (operation === 'signed-composite') { const negative = device('Illustrative negative diode — replace source'); negative.conduction.kind = 'diode'; return { operation, positive: device('Illustrative positive IGBT — replace source'), negative, common_resistance_k_per_w: .2, voltage_estimate: null } }
  return { operation, selected: false, channel: table('channel', [0, 2], 'Illustrative reverse-on channel'), diode: table('diode', [.8, 1], 'Illustrative reverse diode', { v_g_v: 0 }) }
}
const operation = ref('parallel-bank'), input = ref(example('parallel-bank')), result = ref(null), error = ref(''), busy = ref(false), invalid = ref(false), voltageSelected = ref(false), targetVoltage = ref(400), advancedText = ref(JSON.stringify(input.value, null, 2))
let version = 0, disposed = false, applying = false
const invalidFields = new Set(), advancedDirty = ref(false)
function invalidate() { version++; result.value = null; error.value = '' }
function invalidInput(value) { if (value.invalid) invalidFields.add(value.key); else invalidFields.delete(value.key); invalid.value = invalidFields.size > 0; invalidate() }
watch(operation, value => { if (!applying) { input.value = example(value); voltageSelected.value = false; invalid.value = false; invalidFields.clear() }; invalidate() }, { flush: 'sync' })
watch([input, voltageSelected, targetVoltage], () => { invalidate(); advancedText.value = JSON.stringify(request(), null, 2) }, { deep: true, flush: 'sync' })
watch(advancedText, () => invalidate())
function request() { const value = clone(input.value); if (value.operation === 'signed-composite') value.voltage_estimate = voltageSelected.value ? { selected: true, target_voltage_v: targetVoltage.value } : null; return value }
function applyAdvanced() {
  try {
    const parsed = JSON.parse(advancedText.value)
    if (!['parallel-bank', 'signed-composite', 'reverse-envelope'].includes(parsed.operation)) throw new Error('Choose a supported operation')
    const devices = parsed.operation === 'parallel-bank' ? [parsed.device] : parsed.operation === 'signed-composite' ? [parsed.positive, parsed.negative] : []
    if (devices.some(device => !device || typeof device !== 'object')) throw new Error('Supply each device object')
    const tables = parsed.operation === 'reverse-envelope' ? [parsed.channel, parsed.diode] : devices.flatMap(device => ['conduction', 'turn_on', 'turn_off'].map(key => device[key]).filter(Boolean))
    if (tables.some(table => !table || !Array.isArray(table.x) || !Array.isArray(table.y) || !table.conditions || typeof table.conditions !== 'object')) throw new Error('Each table needs x/y arrays and an explicit conditions object')
    applying = true; operation.value = parsed.operation; input.value = parsed; voltageSelected.value = !!parsed.voltage_estimate?.selected; targetVoltage.value = parsed.voltage_estimate?.target_voltage_v ?? 400; applying = false; invalid.value = false; invalidFields.clear(); advancedDirty.value = false
  } catch (exc) { applying = false; error.value = `Invalid input: ${exc.message}`; invalid.value = true }
}
async function calculate() {
  if (advancedDirty.value) { applyAdvanced(); if (invalid.value) return }
  invalidate(); const token = version; busy.value = true
  try { const response = await assemblyApi.analyze(request()); if (!disposed && token === version) result.value = response }
  catch (exc) { if (!disposed && token === version) { const detail = exc.response?.data?.detail; error.value = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : exc.message } }
  finally { if (!disposed) busy.value = false }
}
onBeforeUnmount(() => { disposed = true; version++ })
const labels = { blocking_voltage_v: 'Blocking rating (V)', current_rating_a: 'Current rating (A)', area_m2: 'Contact area (m²)', cost: 'Total cost', known_cost_sum: 'Known cost sum (incomplete total remains unknown)', currency: 'Currency', weight_kg: 'Total weight (kg)', known_weight_kg_sum: 'Known weight sum (kg)', internal_resistance_k_per_w: 'Internal thermal resistance (K/W)', shared_sink_resistance_k_per_w: 'Shared sink resistance (K/W)' }
const rows = computed(() => { if (!result.value) return []; if (result.value.operation === 'reverse-envelope') return [['Overlap current magnitude (A)', result.value.overlap_magnitude_a], ['Crossings (A)', result.value.crossings_magnitude_a]]; const data = result.value.device || result.value.summary; return Object.entries(data).filter(([key]) => labels[key]).map(([key, value]) => [labels[key], value]) })
const plots = computed(() => {
  if (!result.value) return []
  if (result.value.operation === 'reverse-envelope') return [{ title: 'Ideal reverse-on signed envelope', yLabel: 'Signed voltage (V)', datasets: [{ label: 'Python exact envelope', data: result.value.current_a.map((x, i) => ({ x, y: result.value.voltage_v[i] })), borderColor: '#2455b8', pointRadius: 3 }] }]
  const branches = result.value.device ? { positive: result.value.device } : result.value.branches
  return ['conduction', 'turn_on', 'turn_off'].map(key => ({ title: { conduction: 'Signed conduction branches', turn_on: 'Independent turn-on energy', turn_off: 'Independent turn-off energy' }[key], yLabel: key === 'conduction' ? 'Signed voltage (V)' : 'Event energy (J)', datasets: Object.entries(branches).filter(([, value]) => value[key]).map(([direction, value]) => { const curve = value[key], sign = direction === 'negative' ? -1 : 1; return { label: `${direction}: ${curve.source}`, data: curve.x.map((x, i) => ({ x: sign * x, y: (key === 'conduction' ? sign : 1) * curve.y[i] })).sort((a, b) => a.x - b.x), borderColor: direction === 'negative' ? '#c14c25' : '#2455b8', pointRadius: 3 } }) })).filter(plot => plot.datasets.length)
})
function format(value) { return value === null ? 'Unknown' : typeof value === 'number' ? value.toPrecision(8) : Array.isArray(value) ? value.map(format).join(', ') : String(value) }
function download() { const blob = new Blob([JSON.stringify(result.value, null, 2)], { type: 'application/json' }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = 'semiconductor-assembly.json'; link.click(); URL.revokeObjectURL(url) }
</script>
<style scoped>
.assembly-workbench { max-width: 1250px; min-width: 0; margin: auto; padding: 1rem; } label { display: block; margin: .7rem 0; } select, input:not([type=checkbox]), textarea { box-sizing: border-box; max-width: 100%; padding: .45rem; } textarea { width: 100%; font-family: monospace; } .devices, .plots { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; } button { padding: .6rem 1rem; margin: .7rem 0; } table { border-collapse: collapse; width: 100%; } th, td { text-align: left; padding: .5rem; border-bottom: 1px solid #aab6c2; overflow-wrap: anywhere; } pre { overflow: auto; max-width: 100%; } [role=alert] { color: #b12b2b; } @media(max-width: 760px) { .devices, .plots { grid-template-columns: 1fr; } .assembly-workbench { padding: .5rem; } }
</style>
