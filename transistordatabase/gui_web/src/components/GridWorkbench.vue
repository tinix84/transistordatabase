<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { analysisApi } from '../services/api.js'
import WaveformModelEditor from './WaveformModelEditor.vue'
import WaveformAnalysisView from './WaveformAnalysisView.vue'
import GroupedSeriesEvidence from './GroupedSeriesEvidence.vue'
import MeritAnalysisEvidence from './MeritAnalysisEvidence.vue'
import { model, branch, finite } from './waveformModels.js'
import { analysisCsv, analysisError, displayNumber as number, downloadAnalysis, finiteDecimal } from './analysisFiles.js'
const emit = defineEmits(['thermal-handoff'])
function configuration(id = 'config-1') {
  return { id, source: 'Editable independent hand fixture; not measured device data',
    positive: model(), negative: model(false, false), duty: 1,
    onOmit: true, offOmit: true, onReason: 'Explicit conduction-only fixture', offReason: 'Explicit conduction-only fixture',
    onPositive: model(true), onNegative: model(true, false), offPositive: model(true), offNegative: model(true, false) }
}
function point(id = 'point-1') { return { id, source: 'Editable sampled hand fixture; no measurement identity', weight: 1,
  csv: 'time_s,current_a,gate_v,bus_voltage_v\n0,2,0,400\n0.001,4,1,400\n0.002,6,0,400',
  eventMode: 'gate', threshold: .5, omission: 'No event evidence supplied', fft: true, window: 'rectangular', detrend: 'none', harmonicThreshold: .1 } }
const input = reactive({ configurations: [configuration()], points: [point()], metric: 'conduction_power_w',
  direction: 'minimize', summary: 'full', zeroWorst: false, groupedExports: false, merit: false,
  objectives: [{ name: 'conduction_power_w', direction: 'minimize', weight: 1 }, { name: 'turn_on_power_w', direction: 'minimize', weight: 3 }], advanced: false, json: '' })
const metricNames = { total_power_w: 'Complete total power (W)', conduction_power_w: 'Conduction power (W)', turn_on_power_w: 'Turn-on power (W)', turn_off_power_w: 'Turn-off power (W)' }
const result = ref(null), acceptedRequest = ref(null), error = ref(''), busy = ref(false), stale = ref(false)
const selectedIndex = ref(null), filter = ref('all')
let revision = 0, disposed = false
watch(input, () => { revision++; stale.value = !!result.value || busy.value; result.value = null; acceptedRequest.value = null; selectedIndex.value = null; error.value = ''; busy.value = false }, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
function samples(text) {
  const lines = text.trim().split(/\r?\n/), header = lines.shift().split(',').map(value => value.trim())
  const allowed = ['time_s', 'current_a', 'gate_v', 'device_voltage_v', 'bus_voltage_v']
  if (!header.includes('time_s') || !header.includes('current_a') || new Set(header).size !== header.length || header.some(key => !allowed.includes(key))) throw new Error('Use unique time_s,current_a columns and the documented optional SI columns.')
  const columns = Object.fromEntries(header.map(key => [key, []]))
  for (const [index, line] of lines.entries()) {
    const cells = line.split(',').map(value => value.trim())
    if (cells.length !== header.length) throw new Error(`Sample row ${index + 2} does not match its header.`)
    header.forEach((key, col) => columns[key].push(finite(cells[col], `Sample row ${index + 2}`)))
  }
  return columns
}
function request() {
  if (input.advanced) return JSON.parse(input.json)
  return { configurations: input.configurations.map(value => ({ id: value.id, source: value.source,
    conduction: { positive: branch(value.positive), negative: branch(value.negative), duty: finite(value.duty, 'Duty') },
    switching: Object.fromEntries(['on', 'off'].map(side => [side === 'on' ? 'turn_on' : 'turn_off',
      value[side + 'Omit'] ? { omit: true, omission_reason: value[side + 'Reason'] } : { positive: branch(value[side + 'Positive'], true), negative: branch(value[side + 'Negative'], true) }])) })),
  operating_points: input.points.map(value => ({ id: value.id, source: value.source, weight: finiteDecimal(value.weight, 'Operating-point weight'), waveform: samples(value.csv),
    events: value.eventMode === 'gate' ? { mode: 'gate', threshold_v: finite(value.threshold, 'Gate threshold') } : { mode: 'omitted', omission_reason: value.omission },
    provenance: { source: value.source, calibration_notes: 'User supplied SI samples', imported_filename: null, edited_after_import: false },
    ...(value.fft ? { spectrum: { window: value.window, detrend: value.detrend, ...harmonicThreshold(value.harmonicThreshold) } } : {}) })),
  metric: { name: input.metric, unit: 'W', direction: input.direction }, summary_mode: input.summary,
  include_zero_weight_in_worst: input.zeroWorst, grouped_series_exports: input.groupedExports,
  ...(input.merit ? { objectives: meritObjectives() } : {}) }
}
function meritObjectives() {
  if (!input.objectives.length || input.objectives.length > 4) throw new Error('Select between one and four merit objectives.')
  const names = new Set()
  const objectives = input.objectives.map(value => {
    if (!Object.hasOwn(metricNames, value.name) || names.has(value.name)) throw new Error('Each merit objective must use a different power metric.')
    names.add(value.name)
    const weight = finiteDecimal(value.weight, 'Merit objective weight')
    if (weight < 0) throw new Error('Merit objective weights must be nonnegative.')
    return { name: value.name, unit: 'W', direction: value.direction, weight }
  })
  if (!objectives.some(value => value.weight > 0)) throw new Error('At least one merit objective must have a positive weight.')
  return objectives
}
function addObjective() {
  const name = Object.keys(metricNames).find(name => !input.objectives.some(value => value.name === name))
  if (name) input.objectives.push({ name, direction: 'minimize', weight: 1 })
}
function harmonicThreshold(value) {
  const threshold = finiteDecimal(value, 'Harmonic threshold')
  if (threshold < 0 || threshold > 1) throw new Error('Harmonic threshold must be between 0 and 1.')
  return threshold === .1 ? {} : { relative_threshold: threshold }
}
async function calculate() {
  const token = ++revision; error.value = ''; result.value = null; selectedIndex.value = null; busy.value = true; stale.value = false
  try {
    const payload = input.advanced ? input.json : JSON.parse(JSON.stringify(request())), response = await analysisApi.grid(payload)
    if (disposed || token !== revision) return
    acceptedRequest.value = typeof payload === 'string' ? response.inputs : payload; result.value = response
    selectedIndex.value = response.rows.findIndex(row => row.status === 'computed')
  } catch (failure) { if (!disposed && token === revision) error.value = analysisError(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
function prepareJson() { try { input.json = JSON.stringify(request(), null, 2) } catch (failure) { error.value = failure.message } }
const selected = computed(() => result.value?.rows[selectedIndex.value] ?? null)
const filteredRows = computed(() => (result.value?.rows ?? []).map((row, index) => ({ row, index })).filter(({ row }) => filter.value === 'all' || row.status === filter.value))
const objective = computed(() => result.value?.metric.name)
const barMaximum = computed(() => Math.max(1, ...(result.value?.ranking ?? []).map(row => Math.abs(row.value))))
function pair(configId, pointId) { return result.value.rows.find(row => row.configuration_id === configId && row.operating_point_id === pointId) }
function selectPair(row) { selectedIndex.value = result.value.rows.indexOf(row) }
function exportRows() { downloadAnalysis(analysisCsv(result.value.rows.map(row => ({ configuration: row.configuration_id, operating_point: row.operating_point_id,
  weight: row.weight, status: row.status, metric: result.value.metric.name, value_w: row.metrics[objective.value].value,
  error: row.error, input_sha256: row.input_sha256 }))), 'analysis-grid.csv', 'text/csv') }
</script>

<template>
  <section class="analysis-workbench">
    <header><p>Finite configuration × operating-point analysis</p><h2>Analysis grid</h2><p>Compare explicit model configurations against a finite set of sampled operating points. Python calculates every pair independently.</p></header>
    <form @submit.prevent="calculate">
      <section class="panel"><h3>Configurations</h3><p>Each configuration has independent direction models and explicit switching omissions. Editable defaults are hand fixtures.</p>
        <details v-for="(value, index) in input.configurations" :key="index" :open="index === 0" class="entry"><summary>Configuration {{ index + 1 }} · {{ value.id }}</summary>
          <div class="fields"><label>Configuration ID<input v-model="value.id" :aria-label="`Configuration ${index + 1} ID`" required></label><label>Configuration source<input v-model="value.source" required></label><label>Conduction duty<input v-model.number="value.duty" type="number" min="0" max="1" step="any"></label></div>
          <div class="fields"><WaveformModelEditor v-model="value.positive" :title="`${value.id} positive conduction`"/><WaveformModelEditor v-model="value.negative" :title="`${value.id} negative conduction`"/></div>
          <details><summary>Independent turn-on and turn-off source models</summary><div v-for="side in ['on', 'off']" :key="side"><h4>Turn-{{ side }}</h4><label class="check"><input v-model="value[side + 'Omit']" type="checkbox">Explicitly omit turn-{{ side }} energy</label><label v-if="value[side + 'Omit']">Omission reason<input v-model="value[side + 'Reason']"></label><div v-else class="fields"><WaveformModelEditor v-model="value[side + 'Positive']" :title="`${value.id} positive turn-${side}`" energy/><WaveformModelEditor v-model="value[side + 'Negative']" :title="`${value.id} negative turn-${side}`" energy/></div></div></details>
        </details><button type="button" @click="input.configurations.push(configuration(`config-${input.configurations.length + 1}`))">Add configuration</button>
      </section>
      <section class="panel"><h3>Operating points</h3><p>CSV uses time_s,current_a. Optional columns: gate_v,device_voltage_v,bus_voltage_v. Use the waveform workbench for explicit masks and more import options.</p>
        <details v-for="(value, index) in input.points" :key="index" :open="index === 0" class="entry"><summary>Operating point {{ index + 1 }} · {{ value.id }}</summary>
          <div class="fields"><label>Operating-point ID<input v-model="value.id" required></label><label>Waveform source<input v-model="value.source" required></label><label>Operating-point weight<input :value="value.weight" @input="value.weight = $event.target.value" type="number" min="0" step="any" :aria-label="`Point ${index + 1} weight`"></label></div>
          <label>Waveform CSV<textarea v-model="value.csv" rows="5" spellcheck="false" :aria-label="`Point ${index + 1} waveform CSV`"></textarea></label>
          <div class="fields"><label>Event policy<select v-model="value.eventMode"><option value="gate">Gate threshold</option><option value="omitted">Explicitly omit events</option></select></label><label v-if="value.eventMode === 'gate'">Gate threshold (V)<input v-model.number="value.threshold" type="number" step="any"></label><label v-else>Event omission reason<input v-model="value.omission"></label>
          <label class="check"><input v-model="value.fft" type="checkbox">Uniform-sample FFT (all supplied physical series in grouped exports)</label><template v-if="value.fft"><label>FFT window<select v-model="value.window"><option value="rectangular">Rectangular</option><option value="hann">Hann</option></select></label><label>FFT detrend<select v-model="value.detrend"><option value="none">None</option><option value="mean">Subtract mean</option></select></label><label>Harmonic threshold for all supplied physical series<input v-model="value.harmonicThreshold" type="number" min="0" max="1" step="any" :aria-label="`Point ${index + 1} harmonic threshold`"></label></template></div>
        </details><button type="button" @click="input.points.push(point(`point-${input.points.length + 1}`))">Add operating point</button>
      </section>
      <section class="panel"><h3>Comparison policy</h3><div class="fields"><label>Metric<select v-model="input.metric"><option value="total_power_w">Complete total power (W)</option><option value="conduction_power_w">Conduction power (W)</option><option value="turn_on_power_w">Turn-on power (W)</option><option value="turn_off_power_w">Turn-off power (W)</option></select></label><label>Objective direction<select v-model="input.direction"><option value="minimize">Minimize</option><option value="maximize">Maximize</option></select></label><label>Selected summary<select v-model="input.summary"><option value="full">Full weighted mean</option><option value="successful-subset">Successful subset, with coverage</option></select></label></div>
        <label class="check"><input v-model="input.zeroWorst" type="checkbox">Include zero-weight points in the worst-point search</label><p>The full mean is unavailable if a positive-weight metric is missing. A subset result always shows retained weight.</p>
        <label class="check"><input v-model="input.groupedExports" type="checkbox">Include grouped series statistics and harmonic exports</label><p>Grouped exports use supplied current (A), gate voltage (V), device voltage (V), and bus voltage (V). Each pair retains its own source.</p>
        <label class="check"><input v-model="input.merit" type="checkbox">Compare weighted merit penalties</label>
        <section v-if="input.merit" class="entry" data-testid="merit-controls"><h4>Weighted comparison policy</h4><p>Define a dimensionless penalty for this comparison population. Lower is better. These selected weights express your comparison policy.</p><p>Python uses complete full means for every active objective over all positive-weight operating points. Zero objective weights are inactive.</p>
          <div v-for="(value, index) in input.objectives" :key="index" class="fields"><label>Objective metric<select v-model="value.name" :aria-label="`Merit objective ${index + 1} metric`"><option v-for="(label, name) in metricNames" :key="name" :value="name">{{ label }}</option></select></label><label>Objective direction<select v-model="value.direction" :aria-label="`Merit objective ${index + 1} direction`"><option value="minimize">Minimize</option><option value="maximize">Maximize</option></select></label><label>Nonnegative objective weight<input :value="value.weight" @input="value.weight = $event.target.value" type="number" min="0" step="any" :aria-label="`Merit objective ${index + 1} weight`"></label><button type="button" @click="input.objectives.splice(index, 1)">Remove objective {{ index + 1 }}</button></div>
          <button type="button" :disabled="input.objectives.length >= 4" @click="addObjective">Add merit objective</button>
        </section>
        <details><summary>Advanced canonical request, including explicit event masks</summary><button type="button" @click="prepareJson">Copy form into grid request editor</button><label class="check"><input v-model="input.advanced" type="checkbox">Use grid JSON as input</label><label>Grid request JSON<textarea v-model="input.json" rows="10" aria-label="Grid request JSON"></textarea></label></details>
      </section><button type="submit" :disabled="busy">{{ busy ? 'Calculating pairs…' : 'Calculate analysis grid' }}</button>
    </form><p v-if="error" role="alert">{{ error }}</p><p v-if="stale" role="status">Inputs changed. Calculate again to restore results and exports.</p>
    <section v-if="result" data-testid="grid-results"><h3>Grid result · {{ result.metric.name }} (W)</h3><p>Objective: {{ result.metric.direction }} · summary: {{ result.summary_mode }} · method: {{ result.method_version }}</p><p class="hash">Input SHA-256: {{ result.input_sha256 }}</p>
      <div class="actions"><button @click="downloadAnalysis({ request: acceptedRequest, response: result }, 'analysis-grid.json')">Download grid request + result JSON</button><button @click="exportRows">Download grid CSV</button></div>
      <GroupedSeriesEvidence v-if="result.grouped_series_exports" :projection="result.grouped_series_exports"/>
      <MeritAnalysisEvidence v-if="result.merit" :merit="result.merit"/>
      <h4>Configuration × operating-point matrix</h4><div class="table-scroll"><table class="matrix"><thead><tr><th>Configuration</th><th v-for="p in result.inputs.operating_points" :key="p.id">{{ p.id }} · weight {{ p.weight }}</th></tr></thead><tbody><tr v-for="c in result.inputs.configurations" :key="c.id"><th>{{ c.id }}</th><td v-for="p in result.inputs.operating_points" :key="p.id"><button @click="selectPair(pair(c.id, p.id))">{{ number(pair(c.id, p.id).metrics[objective].value) }} W<br>{{ pair(c.id, p.id).status }}</button></td></tr></tbody></table></div>
      <h4>Selected-summary ranking</h4><p v-if="!result.ranking.length">No configuration has an available selected summary.</p><ol class="bars"><li v-for="item in result.ranking" :key="item.configuration_id"><span>{{ item.configuration_id }} · {{ number(item.value) }} W</span><div class="bar-track"><div :style="{ width: `${100 * Math.abs(item.value) / barMaximum}%` }"></div></div></li></ol>
      <div class="table-scroll"><table><thead><tr><th>Configuration</th><th>Full mean (W)</th><th>Subset mean (W)</th><th>Selected (W)</th><th>Coverage / retained weight</th><th>Worst (W) / point IDs</th></tr></thead><tbody><tr v-for="summary in result.summaries" :key="summary.configuration_id"><th>{{ summary.configuration_id }}</th><td>{{ number(summary.metrics[objective].full_mean) }}</td><td>{{ number(summary.metrics[objective].subset_mean) }}</td><td>{{ number(summary.metrics[objective].selected_mean) }}</td><td>{{ number(summary.metrics[objective].coverage) }} · {{ summary.metrics[objective].retained_weight }} / {{ summary.metrics[objective].total_weight }}</td><td>{{ number(summary.metrics[objective].worst_value) }} · {{ summary.metrics[objective].worst_point_ids.join(', ') }}<br>{{ summary.metrics[objective].worst_complete ? 'Complete' : 'Incomplete' }} worst search · zero weights {{ summary.metrics[objective].zero_weight_in_worst ? 'included' : 'excluded' }}</td></tr></tbody></table></div>
      <label>Filter pair diagnostics<select v-model="filter"><option value="all">All pairs</option><option value="computed">Computed pairs</option><option value="unsupported">Unsupported pairs</option></select></label>
      <ul class="pair-list"><li v-for="{ row, index } in filteredRows" :key="index"><button @click="selectedIndex = index">{{ row.configuration_id }} × {{ row.operating_point_id }} · {{ row.status }}</button><span v-if="row.error" class="diagnostic">{{ row.error }}</span></li></ul>
      <section v-if="selected" class="panel" data-testid="selected-grid-pair"><h3>{{ selected.configuration_id }} × {{ selected.operating_point_id }}</h3><p class="hash">Pair SHA-256: {{ selected.input_sha256 }}</p><p v-if="selected.error" role="alert">{{ selected.error }}</p><WaveformAnalysisView v-if="selected.result" :key="selected.input_sha256" :result="selected.result" :accepted-request="selected.inputs.waveform_request" :identity="`${selected.configuration_id} × ${selected.operating_point_id}`" @thermal-handoff="emit('thermal-handoff', $event)"/><details><summary>Immutable pair inputs and diagnostics</summary><pre>{{ JSON.stringify(selected, null, 2) }}</pre></details></section>
    </section>
  </section>
</template>

<style scoped>
.analysis-workbench { max-width:1440px; margin:auto; padding:1.2rem; min-width:0; color:var(--text-primary); }header,h3,h4 { margin:.7rem 0; }p { line-height:1.6; margin:.5rem 0; }
.panel,.entry { border:1px solid var(--border-color,#aaa); border-radius:8px; padding:1rem; margin:1rem 0; min-width:0; background:var(--bg-secondary); }.fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,250px),1fr)); gap:1rem; }
label { display:grid; gap:.35rem; margin:.6rem 0; }.check { display:flex; gap:.5rem; align-items:baseline; }input:not([type=checkbox]),select,textarea { min-width:0; width:100%; padding:.55rem; border:1px solid var(--border-color,#aaa); border-radius:4px; background:var(--bg-secondary); color:inherit; font:inherit; }textarea { font-family:monospace; }summary { cursor:pointer; font-weight:600; }button { padding:.55rem .8rem; margin:.3rem; border:1px solid #7c98ae; border-radius:5px; color:inherit; background:var(--bg-secondary); cursor:pointer; }button:disabled { opacity:.6; cursor:wait; }
.table-scroll { max-width:100%; overflow-x:auto; }table { border-collapse:collapse; width:100%; }th,td { padding:.6rem; text-align:left; vertical-align:top; border-bottom:1px solid var(--border-color,#aaa); overflow-wrap:anywhere; }table:not(.matrix) { min-width:640px; }.matrix { min-width:100%; }.matrix button { min-width:100px; }pre { white-space:pre-wrap; overflow-wrap:anywhere; }.hash,.diagnostic { overflow-wrap:anywhere; }.pair-list { list-style:none; }.bars { padding-left:1.5rem; }.bars li { margin:.7rem 0; }.bar-track { background:var(--bg-tertiary); height:12px; width:100%; }.bar-track div { height:100%; background:#1686bd; }[role=alert] { color:#b02828; overflow-wrap:anywhere; }
@media(max-width:600px) { .analysis-workbench { padding:.5rem; }.panel,.entry { padding:.6rem; } }
</style>
