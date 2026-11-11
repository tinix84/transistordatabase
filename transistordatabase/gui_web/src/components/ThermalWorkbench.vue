<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import { thermalApi } from '../services/api.js'

const props = defineProps({ stagedHandoff: { type: Object, default: null } })
const handoffReviewed = ref(false)

const input = reactive({ operation: 'steady', source: 'Editable thermal hand fixture; replace with recorded source',
  sink: 25, pf: 10, pr: 0, rc: .2, rf: 1, rr: 2,
  interfaceEnabled: false, interfacePath: 'forward', rho: .001, area: .01,
  poles: '.2,.01\n.3,.1', poleMode: 'tau', total: '', initialMode: 'cold', initial: '0,0',
  gap: 'reject', eventMode: 'none', events: '', samples: 20, networkOwner: 'forward',
  segments: '0,0.01,100,0,0,0\n0.01,0.037,0,0,0,0\n0.037,0.1,50,0,0,0',
  branchCount: 1, initialTemperatures: '25', branchResistances: '.5', tolerance: 1e-8, maximum: 200, relaxation: 1,
  models: [model(10, .004), model(5, .02)],
  tjmax: 125, referenceKind: 'case', path: 1, v0: 1, resistance: .1, gate: '', modelTemperature: 125,
  packageEnabled: false, packageCurrent: 20, packageApplicability: '',
  advanced: false, json: '' })
function model(power, alpha) { return { mode: 'linear-temperature-estimate', selected: false,
  low: 25, high: 175, reference: 25, power, alpha, pairs: '25,10\n125,14' } }
const result = ref(null), error = ref(''), busy = ref(false), stale = ref(false)
let revision = 0, disposed = false
watch(input, () => { revision++; stale.value = result.value !== null || busy.value; result.value = null; error.value = ''; if (props.stagedHandoff) handoffReviewed.value = false }, { deep: true, flush: 'sync' })
watch(() => props.stagedHandoff, value => {
  if (!value) return
  input.operation = 'transient'; input.advanced = false
  input.source = `Waveform heat ${value.source_input_sha256}; editable thermal network hand fixture`
  input.networkOwner = value.thermal_input.network_heat_owner
  input.eventMode = value.thermal_input.event_mode
  input.gap = value.thermal_input.gap_policy
  handoffReviewed.value = false
}, { immediate: true })
onBeforeUnmount(() => { disposed = true; revision++ })
function number(value, name) {
  if (value === '' || !Number.isFinite(Number(value))) throw new Error(`${name} needs a finite number`)
  return Number(value)
}
function rows(text, count, name) {
  const parsed = text.trim().split(/\r?\n/).filter(row => row.trim()).map((row, i) => {
    const cells = row.trim().split(/[,;\t ]+/)
    if (cells.length !== count) throw new Error(`${name} row ${i + 1} needs ${count} values`)
    return cells.map(cell => number(cell, name))
  })
  if (!parsed.length) throw new Error(`${name} needs data`)
  return parsed
}
function vector(text, name) { return text.trim().split(/[,;\t ]+/).map(cell => number(cell, name)) }
function request() {
  if (props.stagedHandoff && !handoffReviewed.value) throw new Error('Review the thermal source, network, fixed case, initial state and event convention before calculation.')
  if (input.advanced) return JSON.parse(input.json)
  const context = { operation: input.operation, source: input.source, conditions: {} }
  const sink = number(input.sink, 'Reference temperature')
  if (input.operation === 'steady') return { ...context, sink_temperature_c: sink,
    forward_power_w: number(input.pf, 'Forward power'), reverse_power_w: number(input.pr, 'Reverse power'),
    common_resistance_k_w: number(input.rc, 'Common resistance'), forward_resistance_k_w: number(input.rf, 'Forward resistance'),
    reverse_resistance_k_w: number(input.rr, 'Reverse resistance'),
    interface: input.interfaceEnabled ? { source: input.source, conditions: {}, path: input.interfacePath,
      resistivity_k_m2_w: number(input.rho, 'Interface resistivity'), area_m2: number(input.area, 'Interface area') } : null }
  if (input.operation === 'transient') {
    const poles = rows(input.poles, 2, 'Foster poles')
    const events = props.stagedHandoff ? props.stagedHandoff.thermal_input.events : input.events.trim() ? input.events.trim().split(/\r?\n/).map((row, i) => {
      const cells = row.split(',').map(cell => cell.trim())
      const expected = input.eventMode === 'interval-average' ? 4 : 3
      if (cells.length !== expected) throw new Error(`Event row ${i + 1} needs ${expected} values`)
      return { time_s: number(cells[0], 'Event time'), energy_j: number(cells[1], 'Event energy'), heat_owner: cells[2],
        source: input.source, conditions: {}, ...(expected === 4 ? { end_s: number(cells[3], 'Averaging end') } : {}) }
    }) : []
    return { ...context, network: { source: input.source, conditions: {}, r_th_vector: poles.map(row => row[0]),
      [input.poleMode === 'tau' ? 'tau_vector' : 'c_th_vector']: poles.map(row => row[1]),
      r_th_total: input.total === '' ? null : number(input.total, 'Recorded total R') },
      network_heat_owner: input.networkOwner, case_temperature_c: sink, initial_state: input.initialMode === 'cold' ? 'cold' : vector(input.initial, 'Initial pole rises'),
      gap_policy: input.gap, event_mode: input.eventMode, events, samples_per_segment: number(input.samples, 'Samples per segment'),
      segments: props.stagedHandoff ? props.stagedHandoff.thermal_input.segments : rows(input.segments, 6, 'Power segments').map(row => ({ start_s: row[0], end_s: row[1], power_coefficients_w: row.slice(2) })) }
  }
  if (input.operation === 'electrothermal') return { ...context, sink_temperature_c: sink,
    common_resistance_k_w: number(input.rc, 'Common resistance'), branch_resistances_k_w: vector(input.branchResistances, 'Branch paths'),
    initial_temperatures_c: vector(input.initialTemperatures, 'Initial branch temperatures'), tolerance_k: number(input.tolerance, 'Residual tolerance'),
    max_iterations: number(input.maximum, 'Iteration limit'), relaxation: number(input.relaxation, 'Relaxation'),
    loss_models: input.models.slice(0, input.branchCount).map(value => {
      const common = { source: input.source, conditions: {}, mode: value.mode,
        temperature_min_c: number(value.low, 'Minimum source temperature'), temperature_max_c: number(value.high, 'Maximum source temperature') }
      if (value.mode === 'linear-temperature-estimate') return { ...common, selected: value.selected,
        reference_temperature_c: number(value.reference, 'Model reference temperature'), reference_power_w: number(value.power, 'Reference power'),
        coefficient_per_k: number(value.alpha, 'Temperature coefficient') }
      const pairs = rows(value.pairs, 2, 'Recorded temperature/loss')
      return { ...common, temperature_c: pairs.map(row => row[0]), power_w: pairs.map(row => row[1]) }
    }) }
  return { ...context, conditions: { model_temperature_c: number(input.modelTemperature, 'Conduction model temperature'),
    gate_v: input.gate === '' ? null : number(input.gate, 'Gate condition') },
    junction_limit_c: number(input.tjmax, 'Junction limit'), reference_kind: input.referenceKind,
    reference_temperature_c: sink, path_resistance_k_w: number(input.path, 'Reference path'),
    threshold_v: number(input.v0, 'Threshold'), resistance_ohm: number(input.resistance, 'Electrical resistance'),
    package_limit: input.packageEnabled ? { current_a: number(input.packageCurrent, 'Package limit'), source: input.source,
      conditions: {}, applicability: input.packageApplicability } : null }
}
async function calculate() {
  const token = ++revision; busy.value = true; result.value = null; error.value = ''; stale.value = false
  try {
    const value = await thermalApi.analyze(request())
    if (!disposed && token === revision) result.value = value
  } catch (err) {
    if (!disposed && token === revision) {
      const detail = err.response?.data?.detail
      error.value = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(item => item.msg).join('; ') : err.message
    }
  } finally { if (!disposed) busy.value = false }
}
function prepareJson() { input.json = JSON.stringify(request(), null, 2); input.advanced = true }
function download(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type })), anchor = document.createElement('a')
  anchor.href = url; anchor.download = name; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
}
function exportJson() { download(JSON.stringify({ ...result.value, ...(props.stagedHandoff ? { waveform_handoff: props.stagedHandoff } : {}) }, null, 2), 'thermal-request-result.json', 'application/json') }
function exportCsv() {
  const samples = result.value.result.samples || result.value.result.history
  const keys = result.value.result.samples ? ['time_s', 'temperature_c', 'pole_rises_k', 'kind'] : ['iteration', 'temperature_c', 'power_w', 'residual_k', 'status']
  const quote = value => `"${(Array.isArray(value) ? value.join(';') : value ?? '').toString().replaceAll('"', '""')}"`
  download([keys.join(','), ...samples.map(sample => keys.map(key => quote(sample[key])).join(','))].join('\n'), 'thermal-evidence.csv', 'text/csv')
}
const chart = computed(() => {
  if (!result.value) return []
  const out = result.value.result
  if (out.samples) return [{ label: 'Junction temperature (sampled)', data: out.samples.map(row => ({ x: row.time_s, y: row.temperature_c })), borderColor: '#d04a36', pointRadius: 0 }]
  if (out.history) return Array.from({ length: result.value.request.loss_models.length }, (_, i) => ({ label: `Branch ${i + 1} temperature`, data: out.history.map(row => ({ x: row.iteration, y: row.temperature_c[i] })), borderColor: i ? '#b34a96' : '#257da3' }))
  return []
})
function display(value) {
  if (value === null || value === undefined) return 'Unavailable'
  if (typeof value !== 'number') return String(value).replaceAll('-', ' ')
  return Number(value.toPrecision(7)).toString()
}
const summary = computed(() => {
  if (!result.value) return []
  const out = result.value.result
  const labels = { forward_temperature_c: 'Forward junction temperature (°C)', reverse_temperature_c: 'Reverse junction temperature (°C)',
    interface_resistance_k_w: 'Added interface resistance (K/W)', final_temperature_c: 'Final junction temperature (°C)',
    sampled_max_temperature_c: 'Sampled maximum temperature (°C)', conduction_energy_j: 'Conduction heat energy (J)',
    switching_energy_j: 'Switching heat energy (J)', event_mode: 'Event deposition convention', network_heat_owner: 'Physical junction heat owner',
    status: 'Calculation status', tolerance_k: 'Actual residual tolerance (K)', power_budget_w: 'Reference-path power budget (W)',
    thermal_screening_current_a: 'Thermal screening current (A)' }
  const entries = Object.entries(labels).filter(([key]) => key in out).map(([key, label]) => [label, display(out[key])])
  if (out.history) entries.push(['Loss / residual evaluations', out.history.length])
  if (out.path_resistances_k_w) for (const [path, value] of Object.entries(out.path_resistances_k_w)) entries.push([`${path[0].toUpperCase() + path.slice(1)} path resistance (K/W)`, display(value)])
  if (out.package_limit) { entries.push(['Separate package current limit (A)', display(out.package_limit.current_a)]); entries.push(['Package applicability', out.package_limit.applicability]) }
  return entries
})
const branchResult = computed(() => result.value?.result.accepted_solution || result.value?.result.history?.at(-1) || null)
</script>

<template>
  <section class="thermal-workbench">
    <h2>Thermal paths and fixed-case response</h2>
    <p>Editable hand fixtures demonstrate the equations. Supply your device's source and physical paths. Foster transients hold the case temperature fixed.</p>
    <section v-if="stagedHandoff" class="handoff" data-testid="staged-thermal-handoff">
      <h3>Exact waveform heat staged for review</h3><p><strong>{{ stagedHandoff.scope }}</strong>. Omitted components: {{ stagedHandoff.omitted_components.join(', ') || 'none' }}.</p>
      <p v-if="stagedHandoff.scope === 'known-components-only'">This response uses known components only. Missing heat remains unknown. This calculation does not establish the full junction temperature.</p>
      <p>Exact Python coefficients and event timing are retained. Supply and review the physical thermal network before calculation.</p>
      <p>Waveform request SHA-256: {{ stagedHandoff.source_input_sha256 }}<br>Ownership handoff SHA-256: {{ stagedHandoff.input_sha256 }}</p>
      <details><summary>Source, omissions and exact handoff evidence</summary><pre>{{ JSON.stringify(stagedHandoff, null, 2) }}</pre></details>
    </section>
    <form @submit.prevent="calculate">
      <div class="fields"><label>Calculation<select v-model="input.operation" :disabled="!!stagedHandoff" aria-label="Thermal operation"><option value="steady">Steady common and branch paths</option><option value="transient">Fixed-case Foster transient</option><option value="electrothermal">Bounded steady electrothermal iteration</option><option value="screening">Reference-path current screening</option></select></label>
        <label>Source / assumptions<input v-model="input.source" aria-label="Thermal source" required></label></div>
      <template v-if="!input.advanced">
        <label>{{ input.operation === 'transient' ? 'Fixed case temperature' : 'Sink / selected reference temperature' }} (°C)<input v-model="input.sink" type="number" step="any" aria-label="Reference temperature"></label>
        <div v-if="input.operation === 'steady'" class="fields">
          <label>Forward branch heat (W)<input v-model="input.pf" type="number" step="any" min="0" aria-label="Forward heat"></label>
          <label>Reverse branch heat (W)<input v-model="input.pr" type="number" step="any" min="0"></label>
          <label>Common path (K/W)<input v-model="input.rc" type="number" step="any" min="0"></label>
          <label>Forward branch path (K/W)<input v-model="input.rf" type="number" step="any" min="0"></label>
          <label>Reverse branch path (K/W)<input v-model="input.rr" type="number" step="any" min="0"></label>
          <label class="check"><input v-model="input.interfaceEnabled" type="checkbox">Add an explicit interface term</label>
          <template v-if="input.interfaceEnabled"><label>Interface path<select v-model="input.interfacePath"><option>common</option><option>forward</option><option>reverse</option></select></label><label>Area-specific resistivity (K·m²/W)<input v-model="input.rho" type="number" step="any"></label><label>Contact area (m²)<input v-model="input.area" type="number" step="any"></label></template>
        </div>
        <template v-else-if="input.operation === 'transient'">
          <div class="fields"><label>Physical junction / heat ownership<select v-model="input.networkOwner" :disabled="!!stagedHandoff"><option value="forward">Forward junction only</option><option value="reverse">Reverse junction only</option><option value="shared-junction">Both branches explicitly share this one junction</option></select></label><label>Second pole column<select v-model="input.poleMode"><option value="tau">Time constant τ (s)</option><option value="capacity">Capacity C (J/K), τ=R·C</option></select></label><label>Foster poles: R (K/W), τ (s) or C (J/K)<textarea v-model="input.poles" rows="4" aria-label="Foster poles"></textarea></label><label>Recorded total R (K/W), optional<input v-model="input.total" type="number" step="any"></label>
          <label>Initial state at first segment start<select v-model="input.initialMode"><option value="cold">Cold at the fixed case</option><option value="poles">Explicit per-pole rises (K)</option></select></label><label v-if="input.initialMode === 'poles'">Pole rises, comma separated (K)<input v-model="input.initial"></label><label>Gaps in forcing<select v-model="input.gap"><option value="reject">Reject missing intervals</option><option value="zero-power">Explicitly zero power in gaps</option></select></label><label>Plot samples per segment<input v-model="input.samples" type="number" min="1" max="100"></label></div>
          <label v-if="!stagedHandoff">Exact heat segments: start_s,end_s,c0,c1,c2,c3 (W)<textarea v-model="input.segments" rows="6" aria-label="Heat segments"></textarea></label><p v-else>{{ stagedHandoff.thermal_input.segments.length }} exact Python heat segments and {{ stagedHandoff.thermal_input.events.length }} events are staged without resampling.</p><p>P(u)=c0+c1·u+c2·u²+c3·u³, u=(t−start)/(end−start). The full polynomial is integrated; plot sampling does not change heat.</p>
          <label>Switching heat convention<select v-model="input.eventMode" :disabled="!!stagedHandoff" aria-label="Event heat convention"><option value="none">No switching events supplied</option><option value="impulse">Ideal instantaneous energy deposition</option><option value="interval-average">Uniform heat over explicit intervals</option></select></label>
          <label v-if="!stagedHandoff">Events: time_s,energy_j,heat_owner{{ input.eventMode === 'interval-average' ? ',end_s' : '' }}<textarea v-model="input.events" rows="3" placeholder="0.037,0.1,forward" aria-label="Thermal events"></textarea></label><p>Use forward or reverse ownership, including zero-current events. Each supplied event deposits energy once into this network.</p>
        </template>
        <template v-else-if="input.operation === 'electrothermal'">
          <div class="fields"><label>Branches<select v-model.number="input.branchCount"><option :value="1">One</option><option :value="2">Two coupled</option></select></label><label>Common path (K/W)<input v-model="input.rc" type="number" step="any" min="0"></label><label>Branch paths, comma separated (K/W)<input v-model="input.branchResistances"></label><label>Initial temperatures, comma separated (°C)<input v-model="input.initialTemperatures"></label><label>Actual residual tolerance (K)<input v-model="input.tolerance" type="number" step="any"></label><label>Maximum iterations<input v-model="input.maximum" type="number"></label><label>Relaxation (0,1]<input v-model="input.relaxation" type="number" step="any"></label></div>
          <fieldset v-for="(value, i) in input.models.slice(0, input.branchCount)" :key="i"><legend>Branch {{ i + 1 }} loss source</legend><div class="fields"><label>Loss model<select v-model="value.mode"><option value="recorded">Recorded bounded temperature / loss table</option><option value="linear-temperature-estimate">Explicit linear temperature estimate</option></select></label><label>Minimum source temperature (°C)<input v-model="value.low" type="number" step="any"></label><label>Maximum source temperature (°C)<input v-model="value.high" type="number" step="any"></label><template v-if="value.mode === 'linear-temperature-estimate'"><label>Reference temperature (°C)<input v-model="value.reference" type="number" step="any"></label><label>Reference branch loss (W)<input v-model="value.power" type="number" step="any"></label><label>Temperature coefficient (1/K)<input v-model="value.alpha" type="number" step="any"></label><label class="check"><input v-model="value.selected" type="checkbox" :aria-label="`Select branch ${i + 1} temperature estimate`">Select this temperature estimate</label></template><label v-else>Temperature (°C), loss (W)<textarea v-model="value.pairs" rows="4"></textarea></label></div></fieldset>
          <p>Sources are bounded at each branch's own temperature. Only a converged result is an accepted solution.</p>
        </template>
        <template v-else><div class="fields"><label>Reference boundary<select v-model="input.referenceKind"><option>case</option><option>heatsink</option><option>coolant</option></select></label><label>Junction limit (°C)<input v-model="input.tjmax" type="number" step="any"></label><label>Complete reference path (K/W)<input v-model="input.path" type="number" step="any"></label><label>Conduction threshold (V)<input v-model="input.v0" type="number" step="any"></label><label>Conduction resistance (Ω)<input v-model="input.resistance" type="number" step="any"></label><label>Model temperature condition (°C)<input v-model="input.modelTemperature" type="number" step="any"></label><label>Gate condition (V), blank = unknown<input v-model="input.gate" type="number" step="any"></label><label class="check"><input v-model="input.packageEnabled" type="checkbox">Report a separate package limit</label><template v-if="input.packageEnabled"><label>Package current limit (A)<input v-model="input.packageCurrent" type="number" step="any"></label><label>Package applicability / conditions<input v-model="input.packageApplicability"></label></template></div><p>This is a fixed-condition screening estimate; it does not solve self-heating or establish a datasheet current rating.</p></template>
      </template>
      <details v-if="!stagedHandoff"><summary>Advanced reproducible request</summary><button v-if="!input.advanced" type="button" @click="prepareJson">Prepare JSON from these fields</button><label class="check"><input v-model="input.advanced" type="checkbox">Use advanced JSON</label><textarea v-if="input.advanced" v-model="input.json" rows="14" aria-label="Thermal JSON"></textarea></details>
      <label v-if="stagedHandoff" class="check"><input v-model="handoffReviewed" type="checkbox">I reviewed the thermal source, network, fixed case, initial state and instantaneous event convention.</label>
      <button type="submit" :disabled="busy">{{ busy ? 'Calculating…' : 'Calculate thermal result' }}</button>
    </form>
    <p v-if="error" role="alert">{{ error }}</p><p v-if="stale" role="status">Inputs changed. Calculate again to restore results and exports.</p>
    <section v-if="result" aria-label="Thermal result"><h3>Python thermal result</h3><p>Source: {{ result.request.source }}</p><table><tbody><tr v-for="[label, value] in summary" :key="label"><th>{{ label }}</th><td>{{ value }}</td></tr></tbody></table>
      <template v-if="branchResult"><h4>{{ result.result.accepted_solution ? 'Accepted branch solution' : 'Last iterate — not an accepted solution' }}</h4><table class="branch-results"><thead><tr><th>Branch</th><th>Junction (°C)</th><th>Loss (W)</th><th>Residual (K)</th></tr></thead><tbody><tr v-for="(temperature, i) in branchResult.temperature_c" :key="i"><th>{{ i + 1 }}</th><td>{{ display(temperature) }}</td><td>{{ display(branchResult.power_w?.[i]) }}</td><td>{{ display(branchResult.residual_k?.[i]) }}</td></tr></tbody></table></template>
      <table v-if="result.result.final_pole_rises_k"><thead><tr><th>Foster pole</th><th>Final rise above case (K)</th></tr></thead><tbody><tr v-for="(rise, i) in result.result.final_pole_rises_k" :key="i"><th>{{ i + 1 }}</th><td>{{ display(rise) }}</td></tr></tbody></table>
      <p v-for="key in ['boundary', 'scope', 'maximum_scope', 'nonconvergence_scope']" :key="key">{{ result.result[key] || '' }}</p>
      <CurvePlotter v-if="chart.length" :datasets="chart" title="Thermal response evidence" :x-label="result.operation === 'transient' ? 'Time (s)' : 'Iteration'" y-label="Temperature (°C)" :allow-zoom="false" />
      <p v-if="result.operation === 'transient'">Maximum shown is sampled; it does not establish the global transient maximum. Event jumps retain pre/post samples.</p>
      <button @click="exportJson">Export thermal request + result JSON</button><button v-if="result.result.samples || result.result.history" @click="exportCsv">Export thermal evidence CSV</button>
      <details><summary>Advanced result diagnostics</summary><pre>{{ JSON.stringify(result.result, null, 2) }}</pre></details>
    </section>
  </section>
</template>

<style scoped>
.thermal-workbench { max-width:1100px; margin:auto; padding:1rem; min-width:0; }
.handoff { padding:1rem; border:1px solid #638399; border-radius:6px; overflow-wrap:anywhere; }
.fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr)); gap:.8rem; margin:.8rem 0; }
label { display:flex; flex-direction:column; gap:.3rem; margin:.5rem 0; font-size:.93rem; }
input,select,textarea { box-sizing:border-box; width:100%; min-width:0; padding:.5rem; font:inherit; border:1px solid #aab8c5; border-radius:4px; background:var(--bg-primary,#fff); color:inherit; }
textarea { font-family:monospace; resize:vertical; }
.check { flex-direction:row; align-items:center; }.check input { width:auto; }
button { margin:.5rem .5rem .5rem 0; padding:.65rem 1rem; border:1px solid #638399; border-radius:4px; cursor:pointer; }button:disabled { cursor:wait; opacity:.6; }
fieldset { min-width:0; margin:1rem 0; border:1px solid #aab8c5; }p { line-height:1.5; }[role=alert] { color:#b02828; }
table { width:100%; border-collapse:collapse; table-layout:fixed; }th,td { text-align:left; vertical-align:top; border-bottom:1px solid #bbb; padding:.5rem; overflow-wrap:anywhere; }th { width:40%; }
.branch-results th { width:auto; }.branch-results th,.branch-results td { padding:.45rem .25rem; font-size:.88rem; }pre { max-width:100%; overflow:auto; font-size:.8rem; }
</style>
