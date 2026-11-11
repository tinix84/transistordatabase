<script setup>
import { onBeforeUnmount, reactive, ref, watch } from 'vue'
import WaveformAnalysisView from './WaveformAnalysisView.vue'
import GateAllocationPanel from './GateAllocationPanel.vue'
import HysteresisGateControls from './HysteresisGateControls.vue'
import { defaultHysteresis, hysteresisRequest } from './gateHysteresis.js'
import SpectrumAnalysisControls from './SpectrumAnalysisControls.vue'
import { defaultSpectrumAnalysis, spectrumAnalysisRequest } from './spectrumAnalysis.js'
import { binaryDecimal, finiteDecimal } from './analysisFiles.js'
import WaveformModelEditor from './WaveformModelEditor.vue'
import { model, finite, branch } from './waveformModels.js'
import { waveformApi } from '../services/api.js'

const input = reactive({
  csv: 'time_s,current_a,gate_v,device_voltage_v\n0,0,0,400\n0.000001,2,0,400\n0.000002,9,1,2\n0.000003,3,1,2\n0.000004,0,0,400',
  conductionOmit: false, conductionReason: '', positive: model(), negative: model(false, false), duty: 1,
  eventMode: 'gate', threshold: .5, hysteresis: defaultHysteresis(), eventReason: '', bus: '',
  onOmit: true, offOmit: true, onReason: 'Explicit conduction-only evaluation', offReason: 'Explicit conduction-only evaluation',
  onPositive: model(true), onNegative: model(true, false), offPositive: model(true), offNegative: model(true, false),
  fft: false, window: 'rectangular', detrend: 'none', harmonicThreshold: .1, spectralAnalysis: defaultSpectrumAnalysis(), advanced: false, json: '',
  waveformSource: '', calibrationNotes: '', importedFilename: '', importedEdited: false,
})
const result = ref(null), error = ref(''), busy = ref(false), stale = ref(false)
const emit = defineEmits(['thermal-handoff'])
let revision = 0, disposed = false, importing = false
watch(input, () => { revision++; stale.value = result.value !== null || busy.value; result.value = null; error.value = '' }, { deep: true, flush: 'sync' })
watch(() => input.csv, () => { if (input.importedFilename && !importing) input.importedEdited = true }, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })

function parseCsv(strictNumbers = false, binaryGate = false) {
  if (input.csv.includes('"')) throw new Error('Use numeric unquoted CSV/TSV fields; quoted CSV is not supported in this workbench')
  const lines = input.csv.split(/\r?\n/).filter(line => line.trim())
  if (lines.length < 3) throw new Error('Waveform CSV needs a header and at least two sample rows')
  const delimiter = lines[0].includes(',') ? ',' : lines[0].includes(';') ? ';' : '\t'
  const header = lines[0].trim().split(delimiter).map(value => value.trim())
  const allowed = ['time_s', 'current_a', 'gate_v', 'device_voltage_v', 'bus_voltage_v', 'duty', 'turn_on', 'turn_off']
  if (new Set(header).size !== header.length || header.some(name => !allowed.includes(name))) throw new Error('Use the documented unique SI column names')
  if (!header.includes('time_s') || !header.includes('current_a')) throw new Error('CSV requires time_s and current_a')
  const columns = Object.fromEntries(header.map(name => [name, []]))
  lines.slice(1).forEach((line, row) => {
    const cells = line.split(delimiter).map(value => value.trim())
    if (cells.length !== header.length) throw new Error(`Waveform row ${row + 2} does not match its header`)
    header.forEach((name, col) => columns[name].push(cells[col] === '' ? null : (binaryGate && name === 'gate_v' ? binaryDecimal : strictNumbers ? finiteDecimal : finite)(cells[col], `${name} row ${row + 2}`)))
  })
  for (const [name, values] of Object.entries(columns)) {
    if (values.every(value => value === null)) { delete columns[name]; continue }
    if (name === 'turn_on' || name === 'turn_off') {
      if (values.at(-1) === null) values.pop()
      if (values.some(value => value !== 0 && value !== 1)) throw new Error('Event-mask columns require 0 or 1, with the last row blank')
      columns[name] = values.map(Boolean)
    } else if (values.includes(null)) throw new Error(`${name} has missing samples`)
  }
  return columns
}
function request() {
  if (input.advanced) {
    JSON.parse(input.json)
    return input.json
  }
  const columns = parseCsv()
  const { duty, turn_on, turn_off, ...waveform } = columns
  if (input.bus !== '' && !waveform.bus_voltage_v) waveform.bus_voltage_v = finite(input.bus, 'Bus voltage')
  const omitted = reason => ({ omit: true, omission_reason: reason })
  const eventPolicy = input.eventMode === 'gate' ? { mode: 'gate', threshold_v: finite(input.threshold, 'Gate threshold') }
    : input.eventMode === 'hysteresis' ? { mode: 'hysteresis', digitizer: hysteresisRequest(input.hysteresis) }
    : input.eventMode === 'masks' ? { mode: 'masks', turn_on, turn_off }
    : { mode: 'omitted', omission_reason: input.eventReason }
  return { waveform,
    provenance: { source: input.waveformSource.trim() || null, calibration_notes: input.calibrationNotes.trim() || null,
      imported_filename: input.importedFilename || null, edited_after_import: input.importedEdited },
    conduction: input.conductionOmit ? omitted(input.conductionReason) : { positive: branch(input.positive), negative: branch(input.negative), duty: duty ?? finite(input.duty, 'Duty') },
    events: eventPolicy,
    switching: { turn_on: input.onOmit ? omitted(input.onReason) : { positive: branch(input.onPositive, true), negative: branch(input.onNegative, true) },
      turn_off: input.offOmit ? omitted(input.offReason) : { positive: branch(input.offPositive, true), negative: branch(input.offNegative, true) } },
    ...(input.fft ? { spectrum: { window: input.window, detrend: input.detrend, ...harmonicThreshold(input.harmonicThreshold), ...spectrumAnalysisRequest(input.spectralAnalysis) } } : {}) }
}
function harmonicThreshold(value) {
  const threshold = finiteDecimal(value, 'Harmonic threshold')
  if (threshold < 0 || threshold > 1) throw new Error('Harmonic threshold must be between 0 and 1.')
  return threshold === .1 ? {} : { relative_threshold: threshold }
}
async function calculate() {
  result.value = null; error.value = ''; stale.value = false
  const token = ++revision
  busy.value = true
  try {
    const data = await waveformApi.analyze(request())
    if (token !== revision || disposed) return
    result.value = data
  } catch (failure) {
    if (token !== revision || disposed) return
    const detail = failure.response?.data?.detail
    error.value = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : failure.message
  } finally { if (!disposed) busy.value = false }
}
async function importCsv(event) {
  const file = event.target.files?.[0]
  if (file) {
    const text = await file.text()
    importing = true
    input.importedFilename = file.name; input.importedEdited = false; input.csv = text
    importing = false
  }
}
function loadSwitchingFixture() {
  input.conductionOmit = false; input.positive = model(); input.negative = model(false, false)
  input.positive.source = 'Explicit independent threshold/resistance fixture'
  input.onOmit = false; input.offOmit = false; input.eventMode = 'gate'; input.threshold = .5; input.bus = 200
  input.csv = 'time_s,current_a,gate_v,device_voltage_v\n0,0,0,400\n0.000001,2,0,400\n0.000002,9,1,2\n0.000003,3,1,2\n0.000004,0,0,400'
  for (const [key, voltage, end] of [['onPositive', 400, .01], ['offPositive', 600, .02]]) {
    input[key] = model(true); input[key].mode = 'voltage-scaling-estimate'; input[key].pairs = `0,0\n10,${end}`
    input[key].recorded.v_supply_v = voltage; input[key].requested.v_supply_v = 200
    input[key].source = `Independent ${voltage} V hand fixture, not device measurements`
  }
  input.fft = false; input.advanced = false
  input.importedFilename = ''; input.importedEdited = false
  input.waveformSource = 'Independent sampled hand fixture; no device measurement identity'
  input.calibrationNotes = 'Supplied SI values; no digitization or sensor calibration applied'
}
function showRequest() {
  try { const payload = request(); input.json = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2) } catch (failure) { error.value = failure.message }
}
</script>

<template>
  <section class="waveform-workbench">
    <header><p class="eyebrow">Signed time-domain analysis</p><h2>Waveform workbench</h2>
      <p>Integrate arbitrary sampled current with explicit forward/reverse models, then inspect each switching event and its source conditions.</p></header>
    <div class="intro"><strong>All data use SI units.</strong> Current and duty are piecewise linear. Gate signals detect events; they do not mask conduction. Missing direction data rejects the calculation. Estimates require your explicit selection.</div>
    <form @submit.prevent="calculate">
      <section class="panel">
        <h3>1. Waveform samples</h3>
        <p>Use numeric unquoted CSV/TSV fields. Required columns: <code>time_s,current_a</code>. Optional: <code>gate_v,device_voltage_v,bus_voltage_v,duty,turn_on,turn_off</code>. Device voltage and switching bus voltage are separate quantities. Mask columns contain N−1 boolean intervals (0/1); leave their final row blank.</p>
        <div class="actions"><label>Import numeric CSV<input type="file" accept=".csv,.tsv,.txt" @change="importCsv"></label><button type="button" @click="loadSwitchingFixture">Load independent switching fixture</button></div>
        <label>Waveform CSV<textarea v-model="input.csv" rows="8" spellcheck="false" data-testid="waveform-csv"></textarea></label>
        <p v-if="input.importedFilename">Imported file: {{ input.importedFilename }} <strong>{{ input.importedEdited ? '(edited after import)' : '(original imported samples)' }}</strong></p>
        <div class="columns"><label>Waveform source (blank means unknown)<input v-model="input.waveformSource" placeholder="Measurement file, simulation, or explicit assumption"></label><label>SI calibration / scaling notes (blank means unknown)<input v-model="input.calibrationNotes" placeholder="Probe gain, current orientation, time calibration"></label></div>
        <div class="columns"><label>Constant switching bus (V), if no bus column<input v-model="input.bus" type="number" step="any" placeholder="Unknown / not supplied"></label>
          <label>Constant conduction duty, if no duty column<input v-model.number="input.duty" type="number" min="0" max="1" step="any"></label></div>
      </section>
      <section class="panel">
        <h3>2. Conduction by current direction</h3>
        <label class="check"><input v-model="input.conductionOmit" type="checkbox"> Explicitly omit conduction</label>
        <label v-if="input.conductionOmit">Reason for omitting conduction<input v-model="input.conductionReason"></label>
        <div v-else class="columns"><WaveformModelEditor v-model="input.positive" title="Positive-current conduction"/><WaveformModelEditor v-model="input.negative" title="Negative-current conduction"/></div>
      </section>
      <section class="panel">
        <h3>3. Switching evidence and independent energy models</h3>
        <label>Event policy<select v-model="input.eventMode" aria-label="Event policy"><option value="gate">Gate &gt; threshold (equality is low)</option><option value="hysteresis">Hysteresis: strict crossings and deadband hold</option><option value="masks">Explicit interval masks from CSV</option><option value="omitted">Explicitly omit event detection</option></select></label>
        <label v-if="input.eventMode === 'gate'">Gate threshold (V)<input v-model.number="input.threshold" type="number" step="any"></label>
        <HysteresisGateControls v-if="input.eventMode === 'hysteresis'" v-model="input.hysteresis" label="Switching event"/>
        <label v-if="input.eventMode === 'omitted'">Reason for omitting event detection<input v-model="input.eventReason"></label>
        <p>Every event uses sample i before the transition bracket [i, i+1]. No initial or wrapped event is invented. E(0) remains finite when recorded. No nearest-temperature lookup, clamping or automatic ZVS energy reduction is applied.</p>
        <template v-for="side in ['on', 'off']" :key="side">
          <h4>{{ side === 'on' ? 'Turn-on energy' : 'Turn-off energy' }}</h4>
          <label class="check"><input v-model="input[side + 'Omit']" type="checkbox"> Explicitly omit {{ side === 'on' ? 'turn-on' : 'turn-off' }} losses</label>
          <label v-if="input[side + 'Omit']">Omission reason<input v-model="input[side + 'Reason']"></label>
          <div v-else class="columns"><WaveformModelEditor v-model="input[side + 'Positive']" :title="`Positive-current turn-${side}`" energy/><WaveformModelEditor v-model="input[side + 'Negative']" :title="`Negative-current turn-${side}`" energy/></div>
        </template>
      </section>
      <section class="panel">
        <h3>4. Optional spectrum and reproducible request</h3>
        <label class="check"><input v-model="input.fft" type="checkbox"> Compute uniform-sample FFT (reject irregular sampling)</label>
        <div v-if="input.fft" class="columns"><label>FFT window<select v-model="input.window"><option value="rectangular">Rectangular</option><option value="hann">Hann</option></select></label><label>Detrending<select v-model="input.detrend"><option value="none">None</option><option value="mean">Subtract sample mean</option></select></label></div>
        <label v-if="input.fft">Current harmonic threshold (fraction of maximum amplitude)<input v-model="input.harmonicThreshold" type="number" min="0" max="1" step="any" aria-label="Current harmonic threshold"></label>
        <SpectrumAnalysisControls v-if="input.fft" v-model="input.spectralAnalysis"/>
        <details><summary>Advanced: inspect or supply the canonical JSON request</summary><button type="button" @click="showRequest">Copy form into request editor</button><label class="check"><input v-model="input.advanced" type="checkbox"> Use JSON editor as calculation input</label><label>Canonical request<textarea v-model="input.json" rows="12" spellcheck="false"></textarea></label></details>
      </section>
      <button type="submit" class="primary" :disabled="busy">{{ busy ? 'Calculating…' : 'Calculate waveform losses' }}</button>
      <p v-if="error" role="alert" class="error">{{ error }}</p><p v-if="stale && !result" role="status">Inputs changed. Calculate again to show current results.</p>
    </form>
    <GateAllocationPanel :parse-samples="parseCsv" :sample-revision="input.csv" :provenance="{ source: input.waveformSource || null, calibration_notes: input.calibrationNotes || null, imported_filename: input.importedFilename || null, edited_after_import: input.importedEdited }"/>
    <WaveformAnalysisView v-if="result" :result="result" @thermal-handoff="emit('thermal-handoff', $event)"/>
  </section>
</template>

<style scoped>
.waveform-workbench { max-width: 1440px; margin: auto; padding: 1.25rem; color: #1e293b; }
h2 { font-size: 2rem; margin: .3rem 0; } .eyebrow { color: #0369a1; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
p { line-height: 1.6; } .intro { background: #e0f2fe; border-left: 4px solid #0284c7; padding: 1rem; border-radius: 5px; }
.panel { background: #f8fafc; border: 1px solid #cbd5e1; padding: 1.2rem; border-radius: 12px; margin: 1rem 0; min-width: 0; }
label { display: grid; gap: .4rem; margin: .6rem 0; } label.check { display: flex; align-items: baseline; gap: .5rem; }
input:not([type=checkbox]), select, textarea { width: 100%; box-sizing: border-box; min-width: 0; border: 1px solid #94a3b8; padding: .6rem; border-radius: 6px; font: inherit; }
textarea { font-family: ui-monospace, monospace; } .columns, .plots { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 1rem; }
button { border: 1px solid #94a3b8; border-radius: 6px; padding: .65rem .9rem; background: #fff; color: #0f172a; cursor: pointer; font: inherit; } button:disabled { opacity: .5; cursor: default; }
.primary { background: #0369a1; color: white; border-color: #0369a1; font-weight: 700; }.actions { display: flex; flex-wrap: wrap; align-items: center; gap: .7rem; }
.error { color: #991b1b; background: #fee2e2; padding: 1rem; overflow-wrap: anywhere; } .table-scroll { overflow-x: auto; max-width: 100%; }
table { width: 100%; border-collapse: collapse; background: white; } th, td { text-align: left; padding: .7rem; border-bottom: 1px solid #cbd5e1; vertical-align: top; } pre { white-space: pre-wrap; overflow-wrap: anywhere; } code { overflow-wrap: anywhere; } summary { cursor: pointer; padding: .5rem 0; }
@media(max-width: 780px) { .columns, .plots { grid-template-columns: 1fr; } .waveform-workbench { padding: .5rem; } .panel { padding: .8rem; } h2 { font-size: 1.6rem; } }
</style>
