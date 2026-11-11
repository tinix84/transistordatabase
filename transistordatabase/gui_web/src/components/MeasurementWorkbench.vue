<script setup>
import { computed, onBeforeUnmount, reactive, ref, shallowRef, watch } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import { measurementApi } from '../services/api.js'

const form = reactive({ format: 'csv', encoding: 'utf-8-sig', delimiter: ',', decimal: '.', sourceKind: 'unknown', disclosure: '', stepIndex: '', stepColumn: '', stepValue: '',
  mappings: Object.fromEntries(['time', 'voltage', 'current', 'gate'].map(key => [key, { name: '', unit: '', polarity: 1 }])),
  log: { energy_on: '', energy_off: '', current: '', bus: '', energy_unit: '', current_unit: '', bus_unit: '' },
  conditions: Object.fromEntries(['t_j', 'bus_voltage_v', 'v_g_on', 'v_g_off', 'r_g_on', 'r_g_off'].map(key => [key, { known: false, value: '' }])), commutation: '', manualOverride: '',
})
const conditionLabels = { t_j: 'Junction temperature (°C)', bus_voltage_v: 'DC bus voltage (V)', v_g_on: 'Gate on voltage (V)', v_g_off: 'Gate off voltage (V)', r_g_on: 'Turn-on gate resistance (Ω)', r_g_off: 'Turn-off gate resistance (Ω)' }
const quantityLabels = { time: 'Time', voltage: 'Device voltage Vds / Vce', current: 'Device current Id / Ic', gate: 'Gate voltage Vgs / Vge' }
const units = { time: ['s', 'ms', 'us', 'ns'], voltage: ['V', 'mV', 'kV'], current: ['A', 'mA', 'kA'], gate: ['V', 'mV'] }
const source = shallowRef(null), companion = shallowRef(null), nativeDocument = shallowRef(null), discovery = shallowRef(null), preview = shallowRef(null), result = shallowRef(null)
const inspectedInterpretation = ref('')
const interpretation = () => JSON.stringify([form.format, form.sourceKind, form.disclosure])
const capabilities = shallowRef(null), busy = ref(false), error = ref(''), stale = ref(false), rangeStart = ref(null), rangeEnd = ref(null)
let revision = 0, disposed = false
function invalidate() { revision++; stale.value = stale.value || !!preview.value || !!result.value || busy.value; preview.value = null; result.value = null; error.value = ''; busy.value = false }
watch(form, invalidate, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
measurementApi.capabilities().then(value => { if (!disposed) capabilities.value = value }).catch(() => {})
function encode(bytes) { const chunks = []; for (let i = 0; i < bytes.length; i += 32768) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 32768))); return btoa(chunks.join('')) }
function finite(value, name) { if (String(value).trim() === '' || !Number.isFinite(Number(value))) throw new Error(`${name} requires a finite value.`); return Number(value) }
function failureMessage(failure) { const detail = failure.response?.data?.detail; return typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : failure.message }
async function upload(event, kind = 'source') {
  invalidate(); const token = revision, file = event.target.files?.[0]
  if (kind === 'source') { source.value = null; discovery.value = null } else if (kind === 'companion') companion.value = null; else nativeDocument.value = null
  if (!file) return
  busy.value = true
  try {
    const limit = kind === 'native' ? 40_000_000 : 8_000_000
    if (file.size === 0 || file.size > limit) throw new Error(`Select a nonempty file with at most ${limit.toLocaleString()} bytes.`)
    const bytes = new Uint8Array(await file.arrayBuffer()); if (disposed || token !== revision) return
    if (kind === 'native') { const text = new TextDecoder().decode(bytes); JSON.parse(text); nativeDocument.value = { filename: file.name, text }; return }
    const item = { filename: file.name, content_base64: encode(bytes), byte_length: bytes.length }
    if (kind === 'source') source.value = item; else companion.value = item
  } catch (failure) { if (!disposed && token === revision) error.value = failureMessage(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
function request(mapped = false) {
  if (!source.value) throw new Error('Select an original source file.')
  const mappings = Object.fromEntries(Object.entries(form.mappings).map(([key, item]) => [key, mapped && item.name ? { name: item.name, unit: item.unit, polarity: Number(item.polarity) } : null]))
  if (mapped && form.format !== 'log') for (const key of ['time', 'voltage', 'current']) { if (!mappings[key]?.name || !mappings[key]?.unit) throw new Error(`Declare the ${quantityLabels[key]} trace and unit.`) }
  if (mapped && mappings.gate && !mappings.gate.unit) throw new Error('Declare the gate voltage unit.')
  const log = Object.fromEntries(Object.entries(form.log).map(([key, value]) => [key, value || (key.endsWith('_unit') ? ({ energy_unit: 'J', current_unit: 'A', bus_unit: 'V' })[key] : null)]))
  if (mapped && form.format === 'log') {
    if (!log.energy_on && !log.energy_off) throw new Error('Select at least one supplied energy measurement.')
    if (!form.log.energy_unit) throw new Error('Declare the supplied energy unit.')
    if (log.current && !form.log.current_unit) throw new Error('Declare the step current unit.')
    if (log.bus && !form.log.bus_unit) throw new Error('Declare the step bus voltage unit.')
  }
  return { format: form.format, filename: source.value.filename, content_base64: source.value.content_base64, source_kind: form.sourceKind, disclosure: form.disclosure || null,
    companion_log_base64: companion.value?.content_base64 ?? null, companion_log_filename: companion.value?.filename ?? null,
    step_index: form.format === 'csv' || (!mapped && form.stepIndex.trim() === '') ? null : finite(form.stepIndex, 'Step index'), mappings, csv: { encoding: form.encoding, delimiter: form.delimiter, decimal: form.decimal, step_column: form.stepColumn || null, step_value: form.stepValue || null }, log }
}
async function inspect(mapped = false) {
  invalidate(); const token = revision; busy.value = true
  try { const data = await measurementApi.preview(request(mapped)); if (disposed || token !== revision) return; discovery.value = data; inspectedInterpretation.value = interpretation(); if (mapped) { preview.value = data; stale.value = false; setRange(data.waveform) } }
  catch (failure) { if (!disposed && token === revision) error.value = failureMessage(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
function conditions() {
  const values = Object.fromEntries(Object.entries(form.conditions).map(([key, item]) => [key, item.known ? finite(item.value, conditionLabels[key]) : null]))
  if (form.format !== 'log') for (const key of ['bus_voltage_v', 'v_g_on', 'v_g_off']) if (values[key] === null) throw new Error(`Declare ${conditionLabels[key]} for gate-edge extraction.`)
  return { ...values, commutation_device: form.commutation || null, manual_override: form.manualOverride || null }
}
async function analyze() {
  if (!preview.value) return
  if (form.format !== 'log' && !preview.value.waveform?.gate_v) { error.value = 'Map a gate waveform before local-edge extraction.'; return }
  const token = ++revision; result.value = null; error.value = ''; busy.value = true
  try { const data = await measurementApi.extract({ ...request(true), conditions: conditions(), device_document_text: nativeDocument.value?.text ?? null }); if (disposed || token !== revision) return; result.value = data; stale.value = false; setRange(data.waveform) }
  catch (failure) { if (!disposed && token === revision) error.value = failureMessage(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
function setRange(waveform) { rangeStart.value = waveform?.time_s?.[0] ?? null; rangeEnd.value = waveform?.time_s?.at(-1) ?? null }
const shown = computed(() => result.value ?? preview.value)
const waveform = computed(() => shown.value?.waveform)
const sourceIdentity = computed(() => shown.value?.source ?? (inspectedInterpretation.value === interpretation() ? discovery.value?.source : null))
const records = computed(() => shown.value?.log_records ?? [])
const events = computed(() => Object.entries(result.value?.analysis?.event_evidence ?? {}))
const traces = computed(() => (discovery.value?.available?.traces ?? []).map(item => typeof item === 'string' ? item : item.name))
const measures = computed(() => (discovery.value?.available?.measurements ?? []).map(item => typeof item === 'string' ? item : item.name))
const stepOptions = computed(() => discovery.value?.available?.steps ?? [])
const xMin = computed(() => rangeStart.value !== '' && rangeEnd.value !== '' && rangeStart.value !== null && rangeEnd.value !== null && Number.isFinite(Number(rangeEnd.value)) && Number.isFinite(Number(rangeStart.value)) && Number(rangeStart.value) < Number(rangeEnd.value) ? Number(rangeStart.value) : null)
const xMax = computed(() => xMin.value !== null ? Number(rangeEnd.value) : null)
const plots = computed(() => ['voltage_v', 'current_a', 'gate_v'].map((key, i) => ({ key, title: ['Device voltage', 'Device current', 'Gate voltage'][i], unit: ['V', 'A', 'V'][i], datasets: waveform.value?.[key] ? [{ label: ['Device voltage', 'Signed current', 'Gate evidence'][i], pointRadius: 0, borderColor: ['#296797', '#ae4734', '#568345'][i], data: waveform.value.time_s.map((time, index) => ({ x: time, y: waveform.value[key][index] })) }] : [] })))
const energies = computed(() => Object.entries(result.value?.points ?? {}).flatMap(([event, points]) => points.map((point, index) => ({ event, index, point }))))
function number(value) { return value === null || value === undefined ? 'Unknown' : typeof value === 'number' ? value.toPrecision(8) : String(value) }
function save(value, name, type = 'application/json') { const url = URL.createObjectURL(new Blob([typeof value === 'string' ? value : JSON.stringify(value, null, 2)], { type })); const link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 0) }
function reviewDownload() { if (result.value) save(result.value.review_json_text ?? { ...result.value, import_request: request(true), native_input_text: nativeDocument.value?.text ?? null }, 'measurement-review.json') }
function csvCell(value) { let text = value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value); if (typeof value === 'string' && /^[\s]*[=+\-@\t\r]/.test(value)) text = "'" + text; return '"' + text.replaceAll('"', '""') + '"' }
function exportCsv() { if (!result.value) return; const rows = records.value.length ? records.value : energies.value.map(({event, point}) => ({event, ...point})); if (!rows.length) return; const keys = [...new Set(rows.flatMap(row => Object.keys(row)))]; save([keys.map(csvCell).join(','), ...rows.map(row => keys.map(key => csvCell(row[key])).join(','))].join('\r\n'), 'measurement-energies.csv', 'text/csv;charset=utf-8') }
</script>
<template>
  <div class="measurement-workbench" data-testid="measurement-workbench">
    <header><h2>Measurement import workbench</h2><p>Import CSV, LTspice RAW, or measurement logs. Review source mappings before Python analysis.</p></header>
    <section class="panel"><h3>1. Original source and interpretation</h3><div class="form-grid">
      <label>Source format<select v-model="form.format" data-testid="measurement-format"><option value="csv">CSV waveform</option><option value="raw">LTspice RAW waveform</option><option value="log">LTspice supplied-integral log</option></select></label>
      <label>Original source file<input type="file" data-testid="measurement-file" @change="upload($event)"></label>
      <label>Companion step log (RAW)<input type="file" accept=".log,.txt" data-testid="measurement-companion" @change="upload($event, 'companion')"></label>
      <label>Source type<select v-model="form.sourceKind" data-testid="measurement-source-kind"><option value="unknown">Unknown</option><option value="measurement">Physical measurement</option><option value="simulation">Simulation</option><option value="manual">Manual record</option></select></label>
      <label class="wide">Source / model disclosure<textarea v-model="form.disclosure" data-testid="measurement-disclosure" placeholder="State the instrument or simulation model and its limits."></textarea></label>
    </div><p v-if="source">Selected source: {{ source.filename }} · {{ source.byte_length }} original bytes.</p><p v-if="companion">Companion log: {{ companion.filename }}.</p>
      <p>Source limit: 8,000,000 bytes. Mapped waveform limit: {{ capabilities?.max_points?.toLocaleString() ?? 'server-declared' }} samples. Native archive limit: 40,000,000 bytes.</p>
      <p>Generic LTspice simulation is not a calibrated device measurement. Upload does not start a simulator.</p>
      <div v-if="form.format === 'csv'" class="form-grid"><label>CSV encoding<input v-model="form.encoding" data-testid="measurement-encoding"></label><label>Field separator<select v-model="form.delimiter" data-testid="measurement-delimiter"><option value=",">Comma</option><option value=";">Semicolon</option><option :value="'\t'">Tab</option></select></label><label>Decimal separator<select v-model="form.decimal" data-testid="measurement-decimal"><option value=".">Point</option><option value=",">Comma</option></select></label></div>
      <button @click="inspect(false)" :disabled="!source || busy" data-testid="measurement-inspect">Inspect source columns and steps</button>
      <details v-if="capabilities"><summary>Server parser capabilities</summary><pre>{{ JSON.stringify(capabilities, null, 2) }}</pre></details>
    </section>
    <p v-if="busy" role="status">Reading or analyzing the selected source…</p><p v-if="stale" class="notice" data-testid="measurement-stale">Inputs changed. Preview the mapping again before analysis or export.</p><p v-if="error" role="alert">{{ error }}</p>
    <section v-if="sourceIdentity" class="panel"><h3>Source identity</h3><dl><dt>Original filename</dt><dd>{{ sourceIdentity.filename }}</dd><dt>SHA-256</dt><dd class="hash" data-testid="measurement-hash">{{ sourceIdentity.sha256 }}</dd><dt>Parser / version</dt><dd>{{ sourceIdentity.parser }} / {{ sourceIdentity.parser_version }}</dd><dt>Source type</dt><dd>{{ sourceIdentity.source_kind }}</dd><dt>Model disclosure</dt><dd>{{ sourceIdentity.disclosure ?? 'Unknown' }}</dd><dt>Selected source step</dt><dd>{{ sourceIdentity.step_index ?? 'No indexed step' }}</dd><template v-for="(value,name) in sourceIdentity.step_values" :key="name"><dt>Recorded step variable · {{ name }}</dt><dd>{{ number(value) }} · original source value, unit from mapping</dd></template><template v-if="sourceIdentity.companion_log_filename"><dt>Companion log</dt><dd>{{ sourceIdentity.companion_log_filename }}</dd><dt>Companion SHA-256</dt><dd class="hash">{{ sourceIdentity.companion_log_sha256 }}</dd></template></dl><p v-for="(normalization,index) in sourceIdentity.parser_normalizations" :key="index">Parser operation: {{ normalization }}</p></section>
    <section v-if="discovery" class="panel"><h3>2. Explicit source mapping</h3>
      <p>No units or probe directions are inferred. Original order and sample alignment remain unchanged.</p>
      <p v-if="form.format === 'raw'">RAW stores SI traces. Declare time in s, voltage in V, and current in A.</p><template v-if="form.format !== 'log'"><div class="mapping-grid" v-for="(item, key) in form.mappings" :key="key"><label>{{ quantityLabels[key] }} trace<select v-model="item.name" :data-testid="'measurement-map-' + key"><option value="">Select a source trace</option><option v-for="name in traces" :key="name" :value="name">{{ name }}</option></select></label><label>Recorded unit<select v-model="item.unit" :data-testid="'measurement-unit-' + key"><option value="">Declare unit</option><option v-for="unit in units[key]" :key="unit" :value="unit">{{ unit }}</option></select></label><label>Probe direction<select v-model="item.polarity" :data-testid="'measurement-polarity-' + key"><option :value="1">As recorded</option><option :value="-1" :disabled="key === 'time'">Invert recorded sign</option></select></label></div>
      <div class="form-grid"><label v-if="form.format === 'raw'">Selected RAW step index<input v-model="form.stepIndex" type="number" min="0" step="1" data-testid="measurement-step-index"></label><template v-if="form.format === 'csv'"><label>Step column (if present)<select v-model="form.stepColumn" data-testid="measurement-step-column"><option value="">No step column</option><option v-for="name in traces" :key="name">{{ name }}</option></select></label><label>Exact selected step value<input v-model="form.stepValue" data-testid="measurement-step-value"></label></template></div>
      </template>
      <template v-else><label>Selected log step index for derived point<input v-model="form.stepIndex" type="number" min="0" step="1" data-testid="measurement-step-index"></label><p>Preview lists all source records. Analysis creates core points only for the selected step.</p><div class="form-grid"><label v-for="(label, key) in { energy_on: 'Turn-on supplied energy measure', energy_off: 'Turn-off supplied energy measure', current: 'Current step variable / measure', bus: 'Bus voltage step variable / measure' }" :key="key">{{ label }}<select v-model="form.log[key]" :data-testid="'measurement-log-' + key"><option value="">Not mapped · unknown</option><option v-for="name in [...new Set([...measures, ...(discovery.available?.step_variables ?? [])])]" :key="name">{{ name }}</option></select></label><label>Supplied energy unit<select v-model="form.log.energy_unit" data-testid="measurement-log-energy-unit"><option value="">Declare unit</option><option v-for="unit in ['J','mJ','uJ','nJ']" :key="unit">{{ unit }}</option></select></label><label>Step current unit<select v-model="form.log.current_unit" data-testid="measurement-log-current-unit"><option value="">Declare unit</option><option v-for="unit in ['A','mA','kA']" :key="unit">{{ unit }}</option></select></label><label>Step bus voltage unit<select v-model="form.log.bus_unit" data-testid="measurement-log-bus-unit"><option value="">Declare unit</option><option v-for="unit in ['V','mV','kV']" :key="unit">{{ unit }}</option></select></label></div><p>Supplied integrals retain their source FROM/TO windows and expressions. They do not contain reconstructed waveforms.</p></template>
      <details v-if="stepOptions.length"><summary>Recorded step conditions ({{ stepOptions.length }})</summary><pre>{{ JSON.stringify(stepOptions, null, 2) }}</pre></details>
      <button @click="inspect(true)" :disabled="busy" data-testid="measurement-preview">Preview mapped SI data</button>
    </section>
    <section v-if="discovery" class="panel"><h3>3. Test conditions and native device archive</h3><p>Select Known only when the condition is supplied. A blank known value is invalid. Unknown values remain unknown.</p>
      <div class="form-grid"><div v-for="(item, key) in form.conditions" :key="key" class="condition"><label>{{ conditionLabels[key] }} status<select v-model="item.known" :data-testid="'measurement-known-' + key"><option :value="false">Unknown</option><option :value="true">Known</option></select></label><label v-if="item.known">{{ conditionLabels[key] }}<input v-model="item.value" type="number" step="any" :data-testid="'measurement-condition-' + key"></label></div><label>Commutation device / circuit<input v-model="form.commutation" data-testid="measurement-commutation"></label><label>Manual condition override / reason<input v-model="form.manualOverride" data-testid="measurement-manual-override"></label><label>Existing native device JSON (optional)<input type="file" accept=".json" data-testid="measurement-native-file" @change="upload($event, 'native')"></label></div>
      <p v-if="nativeDocument">Native device archive: {{ nativeDocument.filename }}. Review output does not save it to the database.</p>
      <p v-if="form.format !== 'log'">The accepted DPT engine uses the first falling gate edge and the next rising edge. Gate threshold is the midpoint of declared gate rails. The integration window is the union of local 1–99% voltage/current transitions. There is no conduction subtraction.</p>
      <button @click="analyze" :disabled="!preview || busy" data-testid="measurement-analyze">Analyze and prepare review</button>
    </section>
    <template v-if="shown">
      <section class="panel"><h3>Mapped source preview in SI units</h3><p v-for="(warning, index) in shown.warnings" :key="index" class="notice">{{ warning }}</p>
        <template v-if="waveform"><p data-testid="measurement-sample-count">{{ waveform.time_s.length }} aligned source samples. Display range controls all three time axes.</p><div class="form-grid"><label>Display start time (s)<input v-model="rangeStart" type="number" step="any" data-testid="measurement-range-start"></label><label>Display end time (s)<input v-model="rangeEnd" type="number" step="any" data-testid="measurement-range-end"></label></div><CurvePlotter v-for="plot in plots" :key="plot.key" :title="plot.title" :datasets="plot.datasets" x-label="Time (s)" :y-label="plot.unit" :allow-zoom="false" :x-min="xMin" :x-max="xMax" />
          <div class="table-scroll"><table data-testid="measurement-samples"><caption>First 200 original samples, converted to declared SI units</caption><thead><tr><th>Index</th><th>Time (s)</th><th>Voltage (V)</th><th>Current (A)</th><th>Gate (V)</th></tr></thead><tbody><tr v-for="(time,index) in waveform.time_s.slice(0,200)" :key="index"><td>{{ index }}</td><td>{{ number(time) }}</td><td>{{ number(waveform.voltage_v[index]) }}</td><td>{{ number(waveform.current_a[index]) }}</td><td>{{ number(waveform.gate_v?.[index]) }}</td></tr></tbody></table></div>
        </template>
        <div v-if="shown.statistics && waveform" class="table-scroll"><table data-testid="measurement-statistics"><caption>Python statistics · exact time-weighted piecewise-linear integrals</caption><thead><tr><th>Signal / SI unit</th><th>Mean</th><th>RMS</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody><tr v-for="(stats,name) in shown.statistics" :key="name"><th>{{ { voltage_v:'Device voltage (V)', current_a:'Signed current (A)', gate_v:'Gate voltage (V)' }[name] }}</th><td v-for="key in ['mean','rms','minimum','maximum']" :key="key">{{ number(stats[key]) }}</td></tr></tbody></table></div>
        <div v-if="records.length" class="table-scroll"><table data-testid="measurement-log-records"><caption>{{ records.length }} supplied integral records · failed records retain their status</caption><thead><tr><th>Event</th><th>Step</th><th>Status</th><th>Energy (J)</th><th>Current (A)</th><th>Bus (V)</th><th>Window (s)</th><th>Source expression</th><th>Failure reason</th></tr></thead><tbody><tr v-for="(record,index) in records" :key="index"><td>{{ record.event }}</td><td>{{ record.step_index }}</td><td>{{ record.status }}</td><td>{{ number(record.energy_j) }}</td><td>{{ number(record.current_a) }}</td><td>{{ number(record.bus_voltage_v) }}</td><td>{{ record.window_s?.map(number).join(' → ') ?? 'Unknown' }}</td><td>{{ record.expression ?? 'Unknown' }}</td><td>{{ record.error ?? '—' }}</td></tr></tbody></table></div>
      </section>
    </template>
    <section v-if="result" class="panel" data-testid="measurement-result"><h3>4. Review extracted records</h3><p>{{ result.status }}. Review the raw record and core-derived points before use.</p><p v-if="!result.core_json_document">Upload an existing native device JSON and analyze again to prepare a reloadable native device archive.</p>
      <p v-if="result.analysis">{{ result.analysis.energy_window_convention }}</p><div v-if="result.analysis" class="table-scroll"><table data-testid="measurement-analysis"><thead><tr><th>Quantity</th><th>Python result</th></tr></thead><tbody><tr v-for="(label,key) in { e_on:'Turn-on energy (J)', e_off:'Turn-off energy (J)', i_on:'Local turn-on current (A)', i_off:'Local turn-off current (A)', i_peak:'Peak current (A)', v_overshoot:'Absolute device voltage peak (V)', t_r:'Current rise time (s)', t_f:'Current fall time (s)', di_dt_on:'Turn-on current slope (A/s)', dv_dt_off:'Turn-off voltage slope (V/s)' }" :key="key"><th>{{ label }}</th><td>{{ number(result.analysis[key]) }}</td></tr></tbody></table></div>
      <div v-for="([event,evidence]) in events" :key="event" class="event"><h4>{{ event === 'turn_on' ? 'Turn-on' : 'Turn-off' }} local event evidence</h4><dl><dt>Gate edge (s)</dt><dd>{{ number(evidence.gate_edge_s) }}</dd><dt>Absolute integration window (s)</dt><dd>{{ evidence.integration_window_s?.map(number).join(' → ') }}</dd><dt>Local current low / high (A)</dt><dd>{{ number(evidence.local_current_low_a) }} / {{ number(evidence.local_current_high_a) }}</dd><dt>Local voltage low / high (V)</dt><dd>{{ number(evidence.local_vds_low_v) }} / {{ number(evidence.local_vds_high_v) }}</dd><dt>Current threshold times (s)</dt><dd>{{ evidence.current_threshold_times_s?.map(number).join(', ') }}</dd><dt>Voltage 1–99% times (s)</dt><dd>{{ evidence.voltage_1_99_times_s?.map(number).join(', ') }}</dd></dl></div>
      <div class="actions"><button @click="reviewDownload" data-testid="measurement-review-download">Download full measurement review JSON</button><button @click="save(result.core_json_text ?? result.core_json_document, 'measurement-native-device.json')" :disabled="!result.core_json_document" data-testid="measurement-native-download">Download native device JSON</button><button @click="exportCsv" data-testid="measurement-csv-download">Download energy CSV</button></div>
      <details><summary>Raw record, derived points, and exact provenance</summary><pre data-testid="measurement-record-json">{{ JSON.stringify({ raw_record: result.raw_record, points: result.points, source: result.source }, null, 2) }}</pre></details>
    </section>
  </div>
</template>
<style scoped>
.measurement-workbench { max-width: 1250px; margin: auto; padding: 1.2rem; min-width: 0; overflow-wrap: anywhere; } h2,h3,h4 { margin-bottom: .6rem; } p { margin: .7rem 0; } .panel { min-width: 0; padding: 1.1rem; border: 1px solid #bdcad9; border-radius: 8px; background: var(--bg-secondary,white); margin: 1rem 0; } .form-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .8rem; margin: .8rem 0; } .mapping-grid { display:grid;grid-template-columns: minmax(0,2fr) minmax(0,1fr) minmax(0,1fr);gap:.8rem;margin:.8rem 0; } label,.condition { display:flex;flex-direction:column;gap:.3rem;min-width:0; } input,select,textarea { min-width:0;max-width:100%;width:100%;padding:.5rem;font:inherit;color:inherit;background:var(--bg-primary,white);border:1px solid #aab9ca;border-radius:4px; } .wide { grid-column:1/-1; } button { padding:.65rem .8rem;font:inherit;color:inherit;background:var(--bg-primary,white);border:1px solid #859db8;border-radius:5px;cursor:pointer; } button:disabled { opacity:.5;cursor:default; } .actions { display:flex;flex-wrap:wrap;gap:.6rem;margin:.9rem 0; } dl { display:grid;grid-template-columns:minmax(0,1fr) minmax(0,3fr);gap:.5rem 1rem; } dt {font-weight:600;} dd,.hash {min-width:0;margin:0;overflow-wrap:anywhere;} .table-scroll { min-width:0;max-width:100%;overflow-x:auto;max-height:450px;margin:.8rem 0; } table { width:100%;min-width:650px;border-collapse:collapse; } th,td {padding:.5rem;border:1px solid #c2cddb;text-align:left;vertical-align:top;} th {background:var(--bg-tertiary,#eef3f8);} caption {text-align:left;} pre {max-height:500px;overflow:auto;max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;} summary {cursor:pointer;font-weight:600;} .notice {border-left:3px solid #b47718;padding-left:.7rem;} [role=alert] {color:#b42323;} .event {margin:1rem 0;}
table[data-testid="measurement-analysis"] { min-width: 0; table-layout: fixed; } table[data-testid="measurement-analysis"] th { width: 55%; } table[data-testid="measurement-analysis"] td { overflow-wrap: anywhere; }
@media(max-width:650px) {.measurement-workbench {padding:.7rem;} .panel {padding:.8rem;} .form-grid,.mapping-grid,dl {grid-template-columns:minmax(0,1fr);} dd {margin-bottom:.6rem;} .actions {flex-direction:column;} }
</style>





