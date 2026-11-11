<script setup>
import { computed, ref, watch } from 'vue'
import CurvePlotter from './CurvePlotter.vue'
import WaveformThermalTransfer from './WaveformThermalTransfer.vue'
import { analysisCsv } from './analysisFiles.js'
import { returnedPlotBounds } from './returnedPlotBounds.js'
const props = defineProps({ result: { type: Object, required: true }, acceptedRequest: { type: Object, default: null }, identity: { type: String, default: '' } })
const emit = defineEmits(['thermal-handoff'])
const result = computed(() => props.result)
const rangeStart = ref(null), rangeEnd = ref(null)
const cursorIndex = ref(0)
watch(() => props.result, value => { rangeStart.value = value.signals.time_s[0]; rangeEnd.value = value.signals.time_s.at(-1); cursorIndex.value = 0 }, { immediate: true })
const cursorTime = computed(() => result.value.signals.time_s[cursorIndex.value])
const componentNames = { conduction: 'Conduction', turn_on: 'Turn-on', turn_off: 'Turn-off' }
const componentColors = { conduction: '#147d92', turn_on: '#cf6922', turn_off: '#8355b4' }
const componentStack = computed(() => {
  let offset = 0
  return Object.entries(result.value.components).filter(([, value]) => value.status === 'computed' && value.power_w !== null).map(([key, value]) => {
    const width = result.value.known_component_power_w > 0 ? 600 * value.power_w / result.value.known_component_power_w : 0
    const item = { key, watts: value.power_w, x: offset, width }
    offset += width
    return item
  })
})
const rangeValid = computed(() => Number.isFinite(rangeStart.value) && Number.isFinite(rangeEnd.value) && rangeStart.value < rangeEnd.value)
const xMin = computed(() => rangeValid.value ? rangeStart.value : null)
const xMax = computed(() => rangeValid.value ? rangeEnd.value : null)
function signal(name, label) {
  return result.value?.signals[name] ? [{ label, pointRadius: 0, data: result.value.signals.time_s.map((time, index) => ({ x: time, y: result.value.signals[name][index] })) }] : []
}
function series(rows, x, y, label, extras = {}) { return rows.length ? [{ label, pointRadius: 0, data: rows.map(row => ({ x: row[x], y: row[y] })), ...extras }] : [] }
const current = computed(() => signal('current_a', 'Signed current'))
const gate = computed(() => signal('gate_v', 'Gate evidence'))
const voltage = computed(() => signal('device_voltage_v', 'Measured device voltage'))
const power = computed(() => series(result.value?.conduction_plot ?? [], 'time_s', 'power_w', 'Exact conduction power'))
const cumulative = computed(() => [
  ...series(result.value?.conduction_plot ?? [], 'time_s', 'cumulative_j', 'Conduction energy'),
  ...series(result.value?.switching_cumulative_plot ?? [], 'time_s', 'energy_j', 'Known switching event energy'),
])
const events = computed(() => ['turn_on', 'turn_off'].map(name => ({ label: name === 'turn_on' ? 'Turn-on' : 'Turn-off', showLine: false, pointRadius: 5,
  data: (result.value?.events ?? []).filter(event => event.event === name).map(event => ({ x: event.time_s, y: event.energy_j })) })))
const timeBounds = computed(() => ({
  current: returnedPlotBounds(current.value), voltage: returnedPlotBounds(voltage.value), gate: returnedPlotBounds(gate.value),
  power: returnedPlotBounds(power.value, true), cumulative: returnedPlotBounds(cumulative.value, true), events: returnedPlotBounds(events.value, true),
}))
const spectrum = computed(() => result.value?.spectrum ? [{ label: 'One-sided peak current amplitude', pointRadius: 0,
  data: result.value.spectrum.frequency_hz.map((frequency, index) => ({ x: frequency, y: result.value.spectrum.amplitude_a[index] })) }] : [])
const harmonics = computed(() => (result.value.spectrum?.selected_bin_indices ?? []).slice(0, 10))
const cursorPanels = computed(() => [
  ['Signed current', current.value, 'A'], ...(voltage.value.length ? [['Measured device voltage', voltage.value, 'V']] : []),
  ...(gate.value.length ? [['Gate event evidence', gate.value, 'V']] : []),
  ...(result.value.power_segments.length ? [['Exact conduction power', power.value, 'W']] : []),
  ['Cumulative component energies', cumulative.value, 'J'], ['Switching energy events', events.value, 'J'],
].map(([label, datasets, unit]) => ({ label, unit, values: datasets.flatMap(dataset => dataset.data
  .filter(point => point.x === cursorTime.value).map(point => ({ label: dataset.label, value: point.y }))) })))
function number(value) { return value === null || value === undefined ? '—' : Number(value).toPrecision(6) }
function download(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}
function csvExport(rows, filename) {
  if (!rows.length) return
  download(analysisCsv(rows), filename, 'text/csv')
}
</script>

<template>
    <section v-if="result" class="results" data-testid="waveform-results">
      <p v-if="identity" class="identity">Selected source: {{ identity }}</p>
      <p>Waveform source: {{ result.waveform_provenance?.source ?? 'Unknown' }}<template v-if="result.waveform_provenance?.imported_filename"> · {{ result.waveform_provenance.imported_filename }}</template></p>
      <h3>Losses over the complete {{ number(result.duration_s) }} s interval</h3>
      <p>Known component sum: <strong>{{ number(result.known_component_power_w) }} W</strong>. Full total: <strong>{{ number(result.total_power_w) }} W</strong> ({{ result.total_status }}). An omitted component remains unknown, never zero.</p>
      <figure class="component-stack" data-testid="loss-component-stack">
        <figcaption>Average component power over the complete interval (W)</figcaption>
        <svg viewBox="0 0 600 44" role="img" :aria-label="`Known component stack ${number(result.known_component_power_w)} W; ${result.total_status}`">
          <rect x="0" y="0" width="600" height="32" fill="var(--bg-tertiary,#eee)"/>
          <rect v-for="item in componentStack" :key="item.key" :data-component="item.key" :data-watts="item.watts" :x="item.x" y="0" :width="item.width" height="32" :fill="componentColors[item.key]"><title>{{ componentNames[item.key] }}: {{ number(item.watts) }} W</title></rect>
        </svg>
        <ul><li v-for="(value, name) in result.components" :key="name"><span class="swatch" :style="{ backgroundColor: componentColors[name] }"></span>{{ componentNames[name] }}: <strong>{{ value.power_w === null ? 'Unavailable' : number(value.power_w) + ' W' }}</strong> · {{ value.status }} {{ value.reason ?? '' }}</li></ul>
        <p>Known subtotal: {{ number(result.known_component_power_w) }} W · Full total: {{ result.total_power_w === null ? 'Unavailable' : number(result.total_power_w) + ' W' }}. Omitted components have no bar.</p>
      </figure>
      <div class="table-scroll"><table><thead><tr><th>Component</th><th>Status</th><th>Energy (J)</th><th>Power (W)</th><th>Events / rate (Hz)</th></tr></thead><tbody><tr v-for="(value, name) in result.components" :key="name"><th>{{ ({ conduction: 'Conduction', turn_on: 'Turn-on', turn_off: 'Turn-off' })[name] }}</th><td>{{ value.status }} {{ value.reason ?? '' }}</td><td>{{ number(value.energy_j) }}</td><td>{{ number(value.power_w) }}</td><td>{{ value.event_count ?? '—' }} / {{ number(value.event_rate_hz) }}</td></tr></tbody></table></div>
      <p>Time-weighted current mean {{ number(result.current_statistics.mean) }} A · RMS {{ number(result.current_statistics.rms) }} A · {{ result.current_statistics.method }}.</p>
      <div class="actions"><button @click="download(JSON.stringify({ request: result.request, response: result }, null, 2), 'waveform-analysis.json', 'application/json')">Download request + result JSON</button><button :disabled="!result.events.length" @click="csvExport(result.events, 'waveform-events.csv')">Download event CSV</button><button :disabled="!result.power_segments.length" @click="csvExport(result.power_segments, 'waveform-intervals.csv')">Download interval CSV</button></div>
      <WaveformThermalTransfer :analysis="result" :accepted-request="acceptedRequest || result.request" @thermal-handoff="emit('thermal-handoff', $event)"/>
      <div class="panel range"><h4>Linked visual time range</h4><p>This selection changes all time plots together. Calculated energy and average power still use the complete input interval.</p><p>Time plots use full-source Y bounds from all returned plot points. Constant values receive axis padding. The frequency plot keeps its independent scale.</p><div class="columns"><label>Visual start (s)<input v-model.number="rangeStart" type="number" step="any" data-testid="range-start"></label><label>Visual end (s)<input v-model.number="rangeEnd" type="number" step="any" data-testid="range-end"></label></div><p v-if="!rangeValid" role="alert">Enter a finite visual start smaller than the end.</p></div>
      <section class="panel" data-testid="shared-time-cursor"><h4>Shared time cursor</h4>
        <label>Cursor at original returned sample (s)<select v-model.number="cursorIndex" data-testid="time-cursor-sample"><option v-for="(time, index) in result.signals.time_s" :key="index" :value="index">{{ time }} s · sample {{ index }}</option></select></label>
        <p>Cursor time: <strong data-testid="time-cursor-value">{{ cursorTime }} s</strong>. The purple line uses this SI time on every time plot. The frequency plot is independent.</p>
        <p>Readouts retain exact Python points at this time, including both sides of an event jump. No interpolation or resampling applies.</p>
        <div class="cursor-readouts"><section v-for="panel in cursorPanels" :key="panel.label" :aria-label="`${panel.label} cursor readout`"><h5>{{ panel.label }}</h5><ul v-if="panel.values.length"><li v-for="(value, index) in panel.values" :key="index">{{ value.label }}: {{ number(value.value) }} {{ panel.unit }}</li></ul><p v-else>{{ panel.label === 'Switching energy events' ? 'No returned event at this sample time.' : 'No returned point at this sample time.' }}</p></section></div>
      </section>
      <div class="plots">
        <CurvePlotter title="Signed current" :datasets="current" :y-min="timeBounds.current.min" :y-max="timeBounds.current.max" x-label="Time (s)" y-label="Current (A)" :x-min="xMin" :x-max="xMax" :cursor-x="cursorTime" :allow-zoom="false"/>
        <CurvePlotter v-if="voltage.length" title="Measured device voltage" :datasets="voltage" :y-min="timeBounds.voltage.min" :y-max="timeBounds.voltage.max" x-label="Time (s)" y-label="Device voltage (V)" :x-min="xMin" :x-max="xMax" :cursor-x="cursorTime" :allow-zoom="false"/>
        <CurvePlotter v-if="gate.length" title="Gate event evidence" :datasets="gate" :y-min="timeBounds.gate.min" :y-max="timeBounds.gate.max" x-label="Time (s)" y-label="Gate voltage (V)" :x-min="xMin" :x-max="xMax" :cursor-x="cursorTime" :allow-zoom="false"/>
        <CurvePlotter v-if="result.power_segments.length" title="Exact conduction power" :datasets="power" :y-min="timeBounds.power.min" :y-max="timeBounds.power.max" x-label="Time (s)" y-label="Power (W)" :x-min="xMin" :x-max="xMax" :cursor-x="cursorTime" :allow-zoom="false"/>
        <CurvePlotter title="Cumulative component energies" :datasets="cumulative" :y-min="timeBounds.cumulative.min" :y-max="timeBounds.cumulative.max" x-label="Time (s)" y-label="Energy (J)" :x-min="xMin" :x-max="xMax" :cursor-x="cursorTime" :allow-zoom="false"/>
        <CurvePlotter title="Switching energy events (impulses)" :datasets="events" :y-min="timeBounds.events.min" :y-max="timeBounds.events.max" x-label="Pre-edge time (s)" y-label="Event energy (J)" :x-min="xMin" :x-max="xMax" :cursor-x="cursorTime" :allow-zoom="false"/>
      </div>
      <template v-if="result.spectrum"><h4>Current spectrum — independent frequency axis</h4><p>{{ result.spectrum.normalization }}. N={{ result.spectrum.n }}, dt={{ number(result.spectrum.sample_interval_s) }} s, sequence period N·dt={{ number(result.spectrum.sequence_period_s) }} s. {{ result.spectrum.window }} window; detrend {{ result.spectrum.detrend }}. Discrete input RMS {{ number(result.spectrum.discrete_input_rms_a) }} A; windowed RMS {{ number(result.spectrum.windowed_discrete_rms_a) }} A; Parseval RMS {{ number(result.spectrum.parseval_rms_a) }} A. These describe discrete samples, separately from the continuous piecewise-linear current RMS above.</p><CurvePlotter title="Uniform-sample current spectrum" :datasets="spectrum" x-label="Frequency (Hz)" y-label="Peak amplitude (A)"/></template>
      <section v-if="result.spectrum?.selected_bin_indices" class="panel" data-testid="current-harmonics"><h4>Selected current harmonics</h4><p>Threshold {{ result.spectrum.relative_threshold }} of maximum amplitude. Up to 10 selected bins appear in frequency order. Phase uses the {{ result.spectrum.phase_reference }} and radians. Zero-magnitude phase is unavailable.</p><div v-if="harmonics.length" class="table-scroll"><table><thead><tr><th>Bin</th><th>Frequency (Hz)</th><th>Magnitude (A)</th><th>Real (A)</th><th>Imaginary (A)</th><th>Phase (rad)</th></tr></thead><tbody><tr v-for="index in harmonics" :key="index"><td>{{ index }}</td><td>{{ number(result.spectrum.frequency_hz[index]) }}</td><td>{{ number(result.spectrum.amplitude_a[index]) }}</td><td>{{ number(result.spectrum.real_a[index]) }}</td><td>{{ number(result.spectrum.imaginary_a[index]) }}</td><td>{{ number(result.spectrum.phase_rad[index]) }}</td></tr></tbody></table></div><p v-else>No bins exceed the selected threshold.</p></section>
      <h4>Event audit</h4><div v-if="result.events.length" class="table-scroll"><table><thead><tr><th>Event</th><th>Index / bracket (s)</th><th>Current (A)</th><th>Energy (J)</th><th>Source / mode</th></tr></thead><tbody><tr v-for="event in result.events" :key="event.event + event.sample_index"><td>{{ event.event }}</td><td>{{ event.sample_index }} / {{ event.bracket_s.join(' → ') }}</td><td>{{ number(event.current_a) }}</td><td>{{ number(event.energy_j) }}</td><td>{{ event.source }} · {{ event.mode }}</td></tr></tbody></table></div><p v-else-if="result.event_policy.mode === 'omitted'">Event detection omitted: {{ result.event_policy.omission_reason }}.</p><p v-else>No observed events. Requested components with supported models have computed zero; omitted components stay null.</p>
      <details><summary>Assumptions, exact conditions and full evidence</summary><ul><li v-for="assumption in result.assumptions" :key="assumption">{{ assumption }}</li></ul><pre>{{ JSON.stringify(result.request, null, 2) }}</pre></details>
    </section>
</template>

<style scoped>
.waveform-workbench { max-width: 1440px; margin: auto; padding: 1.25rem; color: var(--text-primary); }
.component-stack { margin:1rem 0; border:1px solid var(--border-color,#aaa); padding:1rem; border-radius:8px; }.component-stack svg { display:block; width:100%; height:64px; }.component-stack ul,.cursor-readouts ul { list-style:none; padding:0; }.component-stack li { margin:.4rem 0; }.swatch { display:inline-block; width:.8rem; height:.8rem; margin-right:.4rem; }.cursor-readouts { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr)); gap:.7rem; }.cursor-readouts section { padding:.6rem; border:1px solid var(--border-color,#aaa); min-width:0; overflow-wrap:anywhere; }
h2 { font-size: 2rem; margin: .3rem 0; } .eyebrow { color: #0369a1; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
p { line-height: 1.6; } .intro { background: #e0f2fe; border-left: 4px solid #0284c7; padding: 1rem; border-radius: 5px; }
.panel { background: var(--bg-secondary); border: 1px solid #cbd5e1; padding: 1.2rem; border-radius: 12px; margin: 1rem 0; min-width: 0; }
label { display: grid; gap: .4rem; margin: .6rem 0; } label.check { display: flex; align-items: baseline; gap: .5rem; }
input:not([type=checkbox]), select, textarea { width: 100%; box-sizing: border-box; min-width: 0; border: 1px solid #94a3b8; padding: .6rem; border-radius: 6px; font: inherit; }
textarea { font-family: ui-monospace, monospace; } .columns, .plots { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 1rem; }
button { border: 1px solid #94a3b8; border-radius: 6px; padding: .65rem .9rem; background: #fff; color: #0f172a; cursor: pointer; font: inherit; } button:disabled { opacity: .5; cursor: default; }
.primary { background: #0369a1; color: white; border-color: #0369a1; font-weight: 700; }.actions { display: flex; flex-wrap: wrap; align-items: center; gap: .7rem; }
.error { color: #991b1b; background: #fee2e2; padding: 1rem; overflow-wrap: anywhere; } .table-scroll { overflow-x: auto; max-width: 100%; }
table { width: 100%; border-collapse: collapse; background: var(--bg-secondary); } th, td { text-align: left; padding: .7rem; border-bottom: 1px solid #cbd5e1; vertical-align: top; } pre { white-space: pre-wrap; overflow-wrap: anywhere; } code { overflow-wrap: anywhere; } summary { cursor: pointer; padding: .5rem 0; }
@media(max-width: 780px) { .columns, .plots { grid-template-columns: 1fr; } .waveform-workbench { padding: .5rem; } .panel { padding: .8rem; } h2 { font-size: 1.6rem; } }
</style>
