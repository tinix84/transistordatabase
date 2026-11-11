<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import ConverterRoleEditor from './ConverterRoleEditor.vue'
import CircuitPhysicalSummary from './CircuitPhysicalSummary.vue'
import { converterApi } from '../services/api.js'
import { finite, vector, roleModel, serializeRole } from './converterForm.js'

const presets = ref([]), revisionLabel = ref(''), result = ref(null), error = ref(''), busy = ref(false), stale = ref(false)
const input = reactive({ preset: 'sc-dab', operation: 'waveform', time: '0,0.000001,0.000002,0.000003,0.000004',
  signals: [], roles: {}, bindings: [], symmetry: false, envelope: false, sourceOmission: false, omissionReason: '', power: 1000,
  source: '', notes: '', advanced: false, json: '' })
const selectedRole = ref(''), rangeStart = ref(null), rangeEnd = ref(null)
let revision = 0, disposed = false
watch(input, () => { revision++; stale.value = result.value !== null || busy.value; result.value = null; error.value = '' }, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
const selectedPreset = computed(() => presets.value.find(value => value.id === input.preset))
const isEnvelope = computed(() => input.operation === 'held-envelope-estimate')
const hasSourceOmission = computed(() => input.bindings.some(binding => binding.source_switching_omission))

function signalText(value) { return typeof value === 'number' ? String(value) : value.join(',') }
function setSignals(values) { input.signals = Object.entries(values).map(([name, value]) => ({ name, values: signalText(value) })) }
function requiredNames(bindings) {
  const names = new Set()
  for (const binding of bindings) {
    for (const key of ['conduction_current', 'switching_current', 'bus_voltage', 'duty', 'frequency_hz']) if (binding[key]?.signal) names.add(binding[key].signal)
    if (binding.gate_signal) names.add(binding.gate_signal)
  }
  return [...names]
}
function selectPreset() {
  const preset = selectedPreset.value
  if (!preset) return
  input.operation = preset.operation
  input.bindings = JSON.parse(JSON.stringify(preset.bindings))
  input.roles = Object.fromEntries(preset.bindings.map(binding => [binding.role, roleModel()]))
  setSignals(Object.fromEntries(requiredNames(preset.bindings).map(name => [name, []])))
  input.symmetry = false; input.envelope = false; input.sourceOmission = false
  input.omissionReason = ''; input.advanced = false
}
onMounted(async () => {
  try {
    const data = await converterApi.presets()
    if (disposed) return
    presets.value = data.presets; revisionLabel.value = data.revision; selectPreset()
  } catch (failure) { if (!disposed) error.value = `Cannot load topology mappings: ${failure.message}` }
})
function fixture(kind) {
  input.preset = kind === 'envelope' ? 'sc-six-switch' : kind === 'llc' ? 'sc-llc' : 'sc-dab'; selectPreset()
  input.source = 'Independent algebraic hand fixture; not measured device or converter data'
  input.notes = 'SI values supplied directly; no operating-point solver or physical soft-switching claim'
  if (kind === 'envelope') {
    input.time = '0,1,4'; setSignals({ ia: [10, 10], dna: [.25, .25], dpa: [.75, .75], dpn: [1, 1], fs: [100000, 100000], fl: [50, 50], vdc: [200, 200] })
  } else {
    input.time = '0,0.000001,0.000002,0.000003,0.000004'
    setSignals({ is2: [1, 1, 1, 1, 1], is4: [2, 2, 2, 2, 2], is6: [3, 3, 3, 3, 3], is8: [4, 4, 4, 4, 4],
      itp: [0, 2, 9, 3, 0], its: [0, -4, -8, -5, 0], vgs2: [0, 0, 1, 1, 0], vgs4: [0, 0, 1, 1, 0],
      vgs6: [0, 0, 1, 1, 0], vgs8: [0, 0, 1, 1, 0], vp: 200, vs: 200 })
  }
  input.symmetry = false; input.envelope = false
  input.omissionReason = kind === 'llc' ? 'Selected archive LLC source scope omits secondary switching and recovery; these losses remain unknown' : ''
}
function circuitFixture() {
  input.preset = 'custom'; input.operation = 'waveform'; input.time = '0,1'
  input.bindings = [{ role: 'switch_pair', members: ['Q1', 'Q2'], multiplicity: 2,
    conduction_current: { signal: 'current', scale: 1 }, switching_current: { signal: 'current', scale: 1 },
    gate_signal: 'gate', bus_voltage: { value: 200 }, duty: { value: 1 }, frequency_hz: { value: 0 } }]
  const model = roleModel()
  model.positive.pairs = '0,1\n10,1'; model.negative.pairs = '0,1\n10,1'
  input.roles = { switch_pair: model }; setSignals({ current: [1, 1], gate: [0, 0] })
  input.source = 'Independent 1 A, 1 V, 1 s algebraic hand fixture; not measured device data'
  input.notes = 'Two identical physical members. No switching edges. Explicit per-device current.'
  input.symmetry = true; input.envelope = false; input.sourceOmission = false; input.advanced = false
}
function customize() { input.preset = 'custom' }
function addBranch() {
  let suffix = input.bindings.length + 1
  while (input.roles[`custom_${suffix}`]) suffix++
  const role = `custom_${suffix}`
  input.bindings.push({ role, members: [role], multiplicity: 1,
    conduction_current: { signal: '', scale: 1 }, switching_current: { signal: '', scale: 1 },
    gate_signal: '', bus_voltage: { value: 200 }, duty: { value: 1 }, frequency_hz: { value: 0 } })
  input.roles[role] = roleModel()
}
function request() {
  if (input.advanced) return JSON.parse(input.json)
  const signals = {}
  for (const row of input.signals) {
    const name = row.name.trim()
    if (!name || Object.hasOwn(signals, name)) throw new Error('Signal names must be nonblank and unique')
    const values = vector(row.values, `Signal ${name}`)
    signals[name] = values.length === 1 ? values[0] : values
  }
  return { preset: input.preset, operation: input.operation, time_s: vector(input.time, 'Time'), signals,
    role_models: Object.fromEntries(input.bindings.map(binding => {
      const model = serializeRole(input.roles[binding.role], binding.source_switching_omission)
      return [binding.role, model]
    })),
    assumptions: { symmetry_selected: input.symmetry, envelope_selected: input.envelope,
      ...(input.sourceOmission ? { source_omission_reason: input.omissionReason } : {}), operating_power_w: finite(input.power, 'Operating power') },
    provenance: { source: input.source.trim() || null, calibration_notes: input.notes.trim() || null },
    ...(input.preset === 'custom' ? { bindings: JSON.parse(JSON.stringify(input.bindings)) } : {}) }
}
async function calculate() {
  result.value = null; error.value = ''; stale.value = false; const token = ++revision; busy.value = true
  try {
    const data = await converterApi.analyze(request())
    if (token !== revision || disposed) return
    result.value = data; selectedRole.value = data.branches[0]?.role ?? ''
    rangeStart.value = data.request.time_s[0]; rangeEnd.value = data.request.time_s.at(-1)
  } catch (failure) {
    if (token !== revision || disposed) return
    const detail = failure.response?.data?.detail
    error.value = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : failure.message
  } finally { if (!disposed) busy.value = false }
}
function showRequest() { try { input.json = JSON.stringify(request(), null, 2) } catch (failure) { error.value = failure.message } }
const activeBranch = computed(() => result.value?.branches.find(branch => branch.role === selectedRole.value))
const rangeValid = computed(() => Number.isFinite(rangeStart.value) && Number.isFinite(rangeEnd.value) && rangeStart.value < rangeEnd.value)
const xMin = computed(() => rangeValid.value ? rangeStart.value : null), xMax = computed(() => rangeValid.value ? rangeEnd.value : null)
function points(rows, x, y, label, extra = {}) { return rows?.length ? [{ label, pointRadius: 0, data: rows.map(row => ({ x: row[x], y: row[y] })), ...extra }] : [] }
// These views consume only returned Python evidence; no losses are recomputed here.
const evidence = computed(() => activeBranch.value?.evidence ?? {})
function sampled(signals, key, label) {
  return signals?.[key] ? [{ label, pointRadius: 0, data: signals.time_s.map((time, i) => ({ x: time, y: signals[key][i] })) }] : []
}
function held(rows, key, label) {
  return rows?.some(row => row[key] !== null && row[key] !== undefined)
    ? [{ label, pointRadius: 0, data: rows.flatMap(row => [{ x: row.start_s, y: row[key] }, { x: row.end_s, y: row[key] }]) }] : []
}
const plots = computed(() => {
  const value = evidence.value, rows = value.intervals
  const switchingKnown = ['turn_on', 'turn_off'].some(key => activeBranch.value?.components[key]?.status === 'computed')
  const candidates = rows ? [
    ['Conduction current', 'A', held(rows, 'current_a', 'Conduction current')],
    ['Commutation current', 'A', held(rows, 'switching_current_a', 'Commutation current')],
    ['Held duty', 'Duty (0–1)', held(rows, 'duty', 'Explicit duty')],
    ['Selected switching frequency', 'Hz', held(rows, 'frequency_hz', 'Envelope switching frequency')],
    ['Average loss power envelope', 'W', ['conduction', 'turn_on', 'turn_off'].flatMap(key => held(rows, `${key}_power_w`, key.replaceAll('_', ' ')))],
  ] : [
    ['Conduction current', 'A', sampled(value.conduction_signals, 'current_a', 'Conduction current')],
    ['Commutation current', 'A', switchingKnown ? sampled(value.switching_signals, 'current_a', 'Commutation current') : []],
    ['Gate event evidence', 'V', switchingKnown ? sampled(value.switching_signals, 'gate_v', 'Physical gate evidence') : []],
    ['Conduction power', 'W', points(value.conduction_plot, 'time_s', 'power_w', 'Exact conduction power')],
    ['Cumulative known energy', 'J', [...points(value.conduction_plot, 'time_s', 'cumulative_j', 'Conduction'), ...(switchingKnown ? points(value.switching_cumulative_plot, 'time_s', 'energy_j', 'Known switching energy') : [])]],
    ['Switching event energy', 'J', ['turn_on', 'turn_off'].map(key => ({ label: key.replaceAll('_', ' '), showLine: false, pointRadius: 5, data: (value.events ?? []).filter(event => event.event === key).map(event => ({ x: event.time_s, y: event.energy_j })) }))],
  ]
  return Object.fromEntries(candidates.filter(([, , data]) => data.length && data.some(series => series.data.length)).map(([title, y_label, datasets]) => [title, { title, y_label, datasets }]))
})
const components = ['conduction', 'turn_on', 'turn_off']
const labels = { conduction: 'Conduction', turn_on: 'Turn-on', turn_off: 'Turn-off' }
const barMaximum = computed(() => Math.max(0, ...(result.value?.branches ?? []).map(branch => branch.known_component_power_w)))
function number(value) { return value === null || value === undefined ? 'Unknown' : Number(value).toPrecision(6) }
function conditions(value) {
  if (!value) return 'Unknown conditions'
  const names = { t_j_c: ['Tj', '°C'], v_g_v: ['Vg', 'V'], r_g_ohm: ['Rg', 'Ω'], v_supply_v: ['Supply', 'V'] }
  return Object.entries(value).map(([key, amount]) => `${names[key]?.[0] ?? key}: ${amount === null ? 'unknown' : `${number(amount)} ${names[key]?.[1] ?? ''}`}`).join(' · ')
}
const modelSources = computed(() => {
  const model = activeBranch.value?.model_bindings
  if (!model) return []
  return [['Conduction', model.conduction], ['Turn-on', model.switching?.turn_on], ['Turn-off', model.switching?.turn_off]].flatMap(([label, spec]) => {
    if (!spec || spec.omit) return [{ label, direction: '—', source: spec?.omission_reason ?? 'Not supplied in this role', mode: 'Omitted / unavailable' }]
    return ['positive', 'negative'].filter(direction => spec[direction]).map(direction => ({ label, direction, ...spec[direction] }))
  })
})
function mapping(value) {
  if (value === null || value === undefined) return 'Not supplied'
  if (typeof value === 'string') return value
  if (Object.hasOwn(value, 'value')) return String(value.value)
  return `${value.complement ? '1 − ' : ''}${value.scale === undefined || value.scale === 1 ? '' : `${value.scale} × `}${value.signal}`
}
function download(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type })); const link = document.createElement('a')
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 0)
}
function exportBranches() {
  const columns = ['role', 'members', 'multiplicity', 'conduction_w', 'turn_on_w', 'turn_off_w', 'known_subtotal_w', 'total_w', 'status', 'source']
  const escape = value => `"${String(value ?? '').replaceAll('"', '""')}"`
  const rows = result.value.branches.map(branch => [branch.role, branch.members.join(';'), branch.multiplicity,
    ...components.map(key => branch.scaled_components[key]?.power_w), branch.known_component_power_w, branch.total_power_w, branch.total_status, branch.model_bindings?.source ?? ''])
  download([columns, ...rows].map(row => row.map(escape).join(',')).join('\r\n'), 'converter-branches.csv', 'text/csv')
}
</script>

<template>
  <section class="converter-workbench">
    <header><p class="eyebrow">Supplied waveforms and explicit circuit mappings</p><h2>Converter loss workbench</h2><p>Aggregate device losses by physical role. Conduction current and commutation current are independently bound, with signed models and independent on/off conditions.</p></header>
    <p class="notice">This workbench postprocesses supplied SI signals. Waveform mode counts recorded gate transitions. The interval-held envelope mode is an explicitly selected duty/frequency estimate; its averaged powers are not PWM transients.</p>
    <form @submit.prevent="calculate">
      <section class="panel"><h3>1. Circuit and supplied signals</h3>
        <div class="actions"><button type="button" :disabled="!presets.length" @click="fixture('dab')">Load independent DAB fixture</button><button type="button" :disabled="!presets.length" @click="fixture('envelope')">Load independent envelope fixture</button><button type="button" :disabled="!presets.length" @click="fixture('llc')">Load LLC omission fixture</button></div>
        <label>Topology preset<select v-model="input.preset" data-testid="converter-preset" @change="selectPreset"><option v-for="preset in presets" :key="preset.id" :value="preset.id">{{ preset.label }}</option><option value="custom">Custom supplied branch mapping</option></select></label>
        <p v-if="selectedPreset">Mapping revision {{ revisionLabel }} · {{ selectedPreset.source }}</p>
        <label>Operation<select v-model="input.operation" :disabled="input.preset !== 'custom'"><option value="waveform">Recorded edge waveform aggregation</option><option value="held-envelope-estimate">Interval-held duty / frequency estimate</option></select></label>
        <p>{{ isEnvelope ? 'N increasing time edges define N−1 held intervals. Supply N−1 signal values or a single explicit constant. No gate events or FFT are inferred.' : 'N increasing times define N aligned samples. Gate evidence selects transitions; it does not mask conduction. Switching uses the sample before each edge bracket.' }}</p>
        <label>Time {{ isEnvelope ? 'edges' : 'samples' }} (s), comma or whitespace separated<textarea v-model="input.time" rows="2" data-testid="converter-time"></textarea></label>
        <div v-for="(row, index) in input.signals" :key="index" class="signal-row"><label>Signal name<input v-model="row.name" :data-testid="`signal-name-${index}`"></label><label>{{ row.name || 'Signal' }} values (SI)<input v-model="row.values" :data-testid="`signal-${row.name}`" placeholder="Explicit scalar or aligned values"></label><button type="button" @click="input.signals.splice(index, 1)">Remove row</button></div>
        <button type="button" @click="input.signals.push({ name: '', values: '' })">Add supplied signal</button>
        <div class="columns"><label>Signal source (blank means unknown)<input v-model="input.source" data-testid="converter-source"></label><label>Orientation / calibration / solver notes<input v-model="input.notes"></label></div>
      </section>
      <section class="panel"><h3>2. Role mapping and selected assumptions</h3>
        <p>Multiplicity represents identical physical members under a selected symmetry assumption. It does not divide current among parallel dies. ENCO's twelve-device DAB is a different circuit from this eight-device SC preset.</p>
        <div class="table-scroll"><table><thead><tr><th>Role / members</th><th>Count</th><th>Conduction current</th><th>Commutation current</th><th>Gate / bus</th><th>Duty / frequency</th></tr></thead><tbody><tr v-for="binding in input.bindings" :key="binding.role"><th>{{ binding.role }}<small>{{ binding.members.join(', ') }}</small></th><td>{{ binding.multiplicity }}</td><td>{{ mapping(binding.conduction_current) }}</td><td>{{ mapping(binding.switching_current) }}</td><td>{{ binding.gate_signal ?? 'No gate' }} / {{ mapping(binding.bus_voltage) }}</td><td>{{ mapping(binding.duty) }} / {{ mapping(binding.frequency_hz) }} Hz</td></tr></tbody></table></div>
        <label class="check"><input v-model="input.symmetry" type="checkbox" data-testid="converter-symmetry"> I select the displayed physical-member symmetry and multiplicities</label>
        <label v-if="isEnvelope" class="check"><input v-model="input.envelope" type="checkbox" data-testid="converter-envelope"> I select the interval-held duty/frequency loss estimate</label>
        <label v-if="hasSourceOmission" class="check"><input v-model="input.sourceOmission" type="checkbox" data-testid="converter-omission"> I select the source's omitted switching scope, leaving the full total unknown</label>
        <label v-if="hasSourceOmission && input.sourceOmission">Source omission reason<input v-model="input.omissionReason"></label>
        <label v-if="input.preset === 'sc-bridgeless'">Signed operating power (W), controls reverse IGBT activation<input v-model="input.power" type="number" step="any"></label>
        <button v-if="input.preset !== 'custom'" type="button" @click="customize">Customize displayed branch mappings</button>
        <template v-if="input.preset === 'custom'"><p>Change the supplied branch references below. Named signals and scalar/complement mappings are explicit; arbitrary expressions are unsupported.</p>
          <fieldset v-for="binding in input.bindings" :key="binding.role" class="custom-binding"><legend>{{ binding.role }}</legend><div class="columns"><label>Multiplicity<input v-model.number="binding.multiplicity" type="number" min="0" step="1"></label><label>Physical members (comma separated)<input :value="binding.members.join(',')" @input="binding.members = $event.target.value.split(',').map(value => value.trim()).filter(Boolean)"></label></div><div v-for="key in ['conduction_current', 'switching_current', 'bus_voltage', 'duty', 'frequency_hz']" :key="key" class="columns"><template v-if="binding[key]"><label>{{ key.replaceAll('_', ' ') }} {{ Object.hasOwn(binding[key], 'value') ? 'constant' : 'signal' }}<input v-if="Object.hasOwn(binding[key], 'value')" v-model.number="binding[key].value" type="number" step="any"><input v-else v-model="binding[key].signal"></label><label v-if="binding[key].signal">Scalar multiplier<input v-model.number="binding[key].scale" type="number" step="any"></label><label v-if="key === 'duty'" class="check"><input v-model="binding[key].complement" type="checkbox"> Use one minus the scaled duty</label></template></div><label v-if="!isEnvelope">Gate signal (waveform)<input v-model="binding.gate_signal"></label><label class="check"><input v-model="binding.source_switching_omission" type="checkbox"> Preserve explicit source switching omission for this role</label><label v-if="binding.multiplicity === 0">Inactive role reason<input v-model="binding.inactive_reason"></label></fieldset><button type="button" @click="addBranch">Add explicit custom branch</button>
        </template>
      </section>
      <section class="panel"><h3>3. Independent signed device models</h3><p>The editable hand curves are algebraic fixtures, not manufacturer data. Replace each role's source and curves with the intended device or assembly. Unknown directions are unavailable unless explicitly supplied. Expand a role to edit its bounded curves and source conditions.</p><ConverterRoleEditor v-for="binding in input.bindings" :key="binding.role" v-model="input.roles[binding.role]" :role="binding.role" :source-omission="binding.source_switching_omission" :envelope="isEnvelope"/></section>
      <section class="panel"><details><summary>Advanced canonical request import / inspection</summary><button type="button" @click="showRequest">Copy form into JSON editor</button><label class="check"><input v-model="input.advanced" type="checkbox"> Use JSON editor as calculation input</label><label>Canonical request<textarea v-model="input.json" rows="12" spellcheck="false" data-testid="converter-json"></textarea></label></details></section>
      <button type="submit" class="primary" :disabled="busy || !input.bindings.length">{{ busy ? 'Calculating…' : 'Calculate converter losses' }}</button><p v-if="error" role="alert" class="error">{{ error }}</p><p v-if="stale && !result" role="status">Inputs changed. Calculate again to show current results.</p>
    </form>
    <CircuitPhysicalSummary :converter-input="input" :bindings="input.bindings" :converter-request="request" @load-fixture="circuitFixture"/>
    <section v-if="result" class="results" data-testid="converter-results">
      <h3>Converter losses over {{ number(result.duration_s) }} s</h3><p>Known component subtotal: <strong>{{ number(result.totals.known_component_power_w) }} W</strong>. Complete total: <strong>{{ number(result.totals.total_power_w) }} W</strong> · {{ result.totals.total_status }}.</p><p v-if="result.totals.total_power_w === null" class="notice">The known subtotal excludes explicitly omitted losses. It is not a complete converter loss or efficiency prediction.</p>
      <p>{{ result.operation === 'held-envelope-estimate' ? 'Selected interval-held envelope estimate. No resolved PWM event count is claimed.' : 'Supplied waveform aggregation with explicit pre-edge event evidence.' }}</p>
      <div class="actions"><button @click="download(JSON.stringify({ request: result.request, response: result }, null, 2), 'converter-analysis.json', 'application/json')">Download request + result JSON</button><button @click="exportBranches">Download branch CSV</button></div>
      <h4>Known branch powers, after physical multiplicity (W)</h4><div class="legend"><span v-for="key in components" :key="key" :class="key">{{ labels[key] }}</span></div>
      <div v-for="branch in result.branches" :key="branch.role" class="bar-row"><strong>{{ branch.role }} × {{ branch.multiplicity }}</strong><div class="bar-track"><span v-for="key in components" :key="key" :class="key" :style="{ width: `${barMaximum ? (branch.scaled_components[key]?.power_w ?? 0) / barMaximum * 100 : 0}%` }" :title="`${labels[key]}: ${number(branch.scaled_components[key]?.power_w)} W`"></span></div><span>{{ number(branch.known_component_power_w) }} W · {{ branch.total_status }}</span></div>
      <div class="table-scroll"><table><thead><tr><th>Role / members</th><th>Component</th><th>Per device (W)</th><th>Physical group (W)</th><th>Status / reason</th></tr></thead><tbody><template v-for="branch in result.branches" :key="branch.role"><tr v-for="key in components" :key="key"><th>{{ branch.role }}<small>{{ branch.members.join(', ') }}</small></th><td>{{ labels[key] }}</td><td>{{ number(branch.components[key]?.power_w) }}</td><td>{{ number(branch.scaled_components[key]?.power_w) }}</td><td>{{ branch.scaled_components[key]?.status }} {{ branch.scaled_components[key]?.reason ?? branch.scaled_components[key]?.inactive_reason ?? '' }}</td></tr></template></tbody></table></div>
      <section class="panel"><h4>Linked branch evidence</h4><label>Inspect branch<select v-model="selectedRole" data-testid="converter-branch"><option v-for="branch in result.branches" :key="branch.role" :value="branch.role">{{ branch.role }} — {{ branch.members.join(', ') }}</option></select></label><p>Role source: {{ activeBranch?.model_bindings?.source ?? 'Unknown' }}. Traces and event evidence below are per physical device; the branch table applies multiplicity.</p><div class="columns"><label>Visual start (s)<input v-model.number="rangeStart" type="number" step="any" data-testid="converter-range-start"></label><label>Visual end (s)<input v-model.number="rangeEnd" type="number" step="any" data-testid="converter-range-end"></label></div><p>This shared visual range does not change the full observation interval used for energy or average power.</p><p v-if="!rangeValid" role="alert">Visual start must precede visual end.</p>
        <p v-if="evidence.inactive_reason" class="notice">Inactive role: {{ evidence.inactive_reason }}. Its supported contribution is computed zero.</p>
        <div class="plots"><CurvePlotter v-for="(plot, name) in plots" :key="selectedRole + name" :title="plot.title ?? name" :datasets="plot.datasets" x-label="Time (s)" :y-label="plot.y_label ?? ''" :x-min="xMin" :x-max="xMax" :allow-zoom="false"/></div>
        <div v-if="evidence.events?.length" class="table-scroll"><table><thead><tr><th>Event</th><th>Bracket start (s)</th><th>Pre-edge current (A)</th><th>Energy (J)</th><th>Source / conditions</th></tr></thead><tbody><tr v-for="(event, i) in evidence.events" :key="i"><td>{{ event.event }}</td><td>{{ number(event.time_s) }}</td><td>{{ number(event.current_a) }}</td><td>{{ number(event.energy_j) }}</td><td>{{ event.source }}<small>Recorded: {{ conditions(event.recorded_conditions) }}</small><small>Requested: {{ conditions(event.requested_conditions) }}</small></td></tr></tbody></table></div>
        <h4>Bound model sources and independent conditions</h4><div class="table-scroll"><table><thead><tr><th>Component / direction</th><th>Source / mode</th><th>Recorded conditions</th><th>Requested conditions</th></tr></thead><tbody><tr v-for="(model, i) in modelSources" :key="i"><th>{{ model.label }} / {{ model.direction }}</th><td>{{ model.source }}<small>{{ model.mode }}</small></td><td>{{ conditions(model.recorded_conditions ?? model.conditions) }}</td><td>{{ conditions(model.requested_conditions ?? model.conditions) }}</td></tr></tbody></table></div>
      </section>
    </section>
  </section>
</template>

<style scoped>
.converter-workbench { max-width:1250px; margin:auto; min-width:0; color:#172033; overflow-wrap:anywhere; }
h2 { margin:.25rem 0; }.eyebrow { text-transform:uppercase; font-size:.8rem; letter-spacing:.08em; color:#24647b; }
.notice { border-left:4px solid #31829c; background:#eef8fb; padding:1rem; line-height:1.5; }
.panel { border:1px solid #d1dbe6; background:#f8fafc; border-radius:10px; padding:1rem; margin:1rem 0; min-width:0; }
label { display:grid; gap:.4rem; margin:.8rem 0; } .check { display:flex; align-items:baseline; gap:.5rem; }
input:not([type=checkbox]), select, textarea { width:100%; min-width:0; box-sizing:border-box; border:1px solid #94a3b8; border-radius:5px; font:inherit; padding:.6rem; }
textarea { font-family:monospace; } button { padding:.6rem .9rem; border:1px solid #94a3b8; background:white; color:#172033; border-radius:5px; cursor:pointer; } button:disabled { opacity:.5; cursor:default; }.primary { background:#155e75; color:white; }
.actions { display:flex; flex-wrap:wrap; gap:.6rem; }.columns,.plots { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }
.plots :deep(.curve-plotter) { min-width:0; max-width:100%; box-sizing:border-box; }
.plots :deep(canvas) { max-width:100%; }
.signal-row { display:grid; grid-template-columns:minmax(100px,1fr) minmax(180px,3fr) auto; gap:.8rem; align-items:center; }.custom-binding { min-width:0; margin:1rem 0; }
.table-scroll { max-width:100%; overflow:auto; } table { width:100%; min-width:650px; border-collapse:collapse; text-align:left; font-size:.9rem; } th,td { padding:.6rem; border-bottom:1px solid #cbd5e1; vertical-align:top; overflow-wrap:anywhere; } small { display:block; font-weight:400; color:#475569; }.error { background:#fef2f2; color:#991b1b; padding:1rem; overflow-wrap:anywhere; }
.bar-row { display:grid; grid-template-columns:140px minmax(0,1fr) 230px; gap:.8rem; align-items:center; margin:1rem 0; font-size:.9rem; }.bar-track { display:flex; min-height:22px; background:#e2e8f0; }.bar-track span { display:block; }.conduction { background:#147d92; }.turn_on { background:#df9b32; }.turn_off { background:#8058ab; }.legend { display:flex; flex-wrap:wrap; gap:.7rem; }.legend span { padding:.3rem .6rem; color:white; border-radius:3px; } summary { cursor:pointer; }
@media(max-width:700px) { .columns,.plots,.signal-row,.bar-row { grid-template-columns:1fr; }.panel { padding:.8rem; } .bar-row { gap:.35rem; } }
</style>
