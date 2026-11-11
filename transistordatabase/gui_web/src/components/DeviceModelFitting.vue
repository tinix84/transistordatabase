<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { deviceModelApi } from '../services/api.js'
import { analysisCsv, analysisError, displayNumber as number, downloadAnalysis, finiteDecimal } from './analysisFiles.js'
import { coordinateLabels, fittingFixture, modelFields, modelHeader, nullableDecimal, parseObservations } from './deviceModelInput.js'
import DeviceModelFitPlot from './DeviceModelFitPlot.vue'

const input = reactive({ family: 'ron', degree: '1', event: 'e_on', source: '', series: '', csv: '',
  conditions: { v_g_v: '', current_a: '' }, anchor: { current_a: '', voltage_v: '', t_j_c: '', r_g_ohm: '' } })
const query = reactive({ output: 'resistance', coordinates: { t_j_c: '', current_a: '', voltage_v: '', v_g_v: '', r_g_ohm: '' } })
const fit = ref(null), prediction = ref(null), fitJson = ref(null), predictionJson = ref(null)
const importText = ref(''), imported = ref(null), importError = ref('')
const error = ref(''), queryError = ref(''), busy = ref(false), queryBusy = ref(false), stale = ref(false)
let fitRevision = 0, queryRevision = 0, disposed = false
function invalidateQuery() { queryRevision++; prediction.value = null; predictionJson.value = null; queryError.value = ''; queryBusy.value = false }
watch(input, () => {
  fitRevision++; stale.value ||= !!fit.value || busy.value; fit.value = null; fitJson.value = null; imported.value = null; error.value = ''; busy.value = false; invalidateQuery()
}, { deep: true, flush: 'sync' })
watch(query, invalidateQuery, { deep: true, flush: 'sync' })
watch(importText, () => { fitRevision++; fit.value = null; fitJson.value = null; imported.value = null; busy.value = false; error.value = ''; importError.value = ''; invalidateQuery() }, { flush: 'sync' })
const predictionFamily = computed(() => imported.value?.preview.family || input.family)
watch(predictionFamily, family => { query.output = family === 'ron' ? 'resistance' : family === 'diode' ? 'voltage' : 'energy' }, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; fitRevision++; queryRevision++ })
const header = computed(() => modelHeader(input.family).join(','))
const unit = computed(() => input.family === 'ron' ? 'ohm' : input.family === 'diode' ? 'V' : 'J')
const conditionKeys = computed(() => input.family === 'ron' ? ['v_g_v', 'current_a'] : input.family === 'energy' ? ['v_g_v'] : [])
const queryKeys = computed(() => predictionFamily.value === 'ron' ? ['t_j_c', 'v_g_v', 'current_a']
  : predictionFamily.value === 'energy' ? [...modelFields.energy, 'v_g_v']
    : [query.output === 'current' ? 'voltage_v' : 'current_a', 't_j_c', 'v_g_v'])
const nullableQueryKeys = computed(() => ['v_g_v', ...(predictionFamily.value === 'ron' ? ['current_a'] : predictionFamily.value === 'diode' ? ['t_j_c'] : [])])
function example() {
  const fixture = fittingFixture(input.family)
  input.source = fixture.source; input.series = fixture.series; input.csv = fixture.csv; input.degree = '1'
  input.conditions.v_g_v = ''; input.conditions.current_a = ''; input.event = 'e_on'
  Object.assign(input.anchor, { current_a: '20', voltage_v: '400', t_j_c: '25', r_g_ohm: '4' })
  Object.assign(query.coordinates, { t_j_c: input.family === 'ron' ? '50' : input.family === 'energy' ? '75' : '25', current_a: input.family === 'energy' ? '15' : input.family === 'diode' ? '2' : '', voltage_v: input.family === 'energy' ? '500' : '.70', v_g_v: '', r_g_ohm: '6' })
}
function trainingRequest() {
  const degree = finiteDecimal(input.degree, 'Polynomial degree')
  if (![1, 2, 3].includes(degree)) throw new Error('Select polynomial degree 1, 2, or 3.')
  return { family: input.family, degree, event: input.family === 'energy' ? input.event : null,
    fixed_conditions: Object.fromEntries(conditionKeys.value.map(key => [key, nullableDecimal(input.conditions[key], coordinateLabels[key])])),
    anchor: input.family === 'energy' ? Object.fromEntries(Object.entries(input.anchor).map(([key, value]) => [key, finiteDecimal(value, `Anchor ${coordinateLabels[key]}`)])) : null,
    observations: parseObservations(input.csv, input.family, input.source, input.series) }
}
async function train() {
  const token = ++fitRevision; invalidateQuery(); fit.value = null; fitJson.value = null; imported.value = null; error.value = ''; busy.value = true; stale.value = false
  try {
    const response = await deviceModelApi.train(trainingRequest())
    if (disposed || token !== fitRevision) return
    fit.value = response.data; fitJson.value = response.rawJson
  } catch (failure) { if (!disposed && token === fitRevision) error.value = analysisError(failure) }
  finally { if (!disposed && token === fitRevision) busy.value = false }
}
async function evaluate() {
  if (!fit.value && !imported.value) return
  const token = ++queryRevision, modelToken = fitRevision
  prediction.value = null; predictionJson.value = null; queryError.value = ''; queryBusy.value = true
  try {
    const coordinates = Object.fromEntries(queryKeys.value.map(key => [key, nullableQueryKeys.value.includes(key)
      ? nullableDecimal(query.coordinates[key], coordinateLabels[key]) : finiteDecimal(query.coordinates[key], coordinateLabels[key])]))
    const queries = [{ id: 'query-1', output: query.output, coordinates }]
    const response = imported.value ? await deviceModelApi.evaluateModelJson(imported.value.rawJson, queries)
      : await deviceModelApi.evaluate({ model: fit.value.model, queries })
    if (disposed || token !== queryRevision || modelToken !== fitRevision) return
    prediction.value = response.data; predictionJson.value = response.rawJson
  } catch (failure) { if (!disposed && token === queryRevision && modelToken === fitRevision) queryError.value = analysisError(failure) }
  finally { if (!disposed && token === queryRevision && modelToken === fitRevision) queryBusy.value = false }
}
function useImportedModel() {
  fitRevision++; fit.value = null; fitJson.value = null; busy.value = false; error.value = ''; importError.value = ''; imported.value = null; invalidateQuery()
  try {
    if (importText.value.length > 4 * 1024 * 1024) throw new Error('Model JSON exceeds the 4 MiB limit.')
    const preview = JSON.parse(importText.value)
    if (!preview || Array.isArray(preview) || !['ron', 'diode', 'energy'].includes(preview.family)) throw new Error('Supply a downloaded device model JSON object with a supported family.')
    imported.value = { preview, rawJson: importText.value }
    stale.value = false
  } catch (failure) { importError.value = failure.message }
}
const residualRows = computed(() => (fit.value?.diagnostics || []).flatMap(diagnostic => ['train', 'holdout'].flatMap(split =>
  (diagnostic[split]?.rows || []).map(row => ({ output: diagnostic.output, slice: diagnostic.slice, split, ...row })))))
function exportResiduals() {
  downloadAnalysis(analysisCsv(residualRows.value.map(row => ({ ...row, unit: row.output === 'current' ? 'A' : row.output === 'voltage' ? 'V' : unit.value,
    model_sha256: fit.value.evidence.model_sha256, result_sha256: fit.value.evidence.result_sha256 }))), 'device-model-residuals.csv', 'text/csv')
}
function plotQueries(plot) {
  // Only display direct one-dimensional predictions. Energy query factors require a separate backend projection.
  if (input.family === 'energy' || !prediction.value) return []
  return prediction.value.result.predictions.filter(row => row.status === 'predicted' && row.value !== null && row.output === plot.output).flatMap(row => {
    const coordinates = prediction.value.inputs.queries.find(item => item.id === row.id)?.coordinates
    if (!coordinates || Object.entries(plot.conditions || {}).some(([key, value]) => coordinates[key] !== value)) return []
    return [{ x: coordinates[plot.axis], y: row.value }]
  })
}
</script>

<template>
  <section class="device-model-fitting" data-testid="model-fitting">
    <h3>Device model fitting</h3>
    <p>Fit raw observations in Python. Evaluate the resulting model within its recorded support and exact conditions.</p>
    <p>Training diagnostics describe the fit. Held-out diagnostics use rows excluded from fitting. Neither establishes device accuracy.</p>
    <form @submit.prevent="train">
      <fieldset><legend>1. Family and observations</legend>
        <div class="fields">
          <label>Model family<select v-model="input.family" data-testid="fit-family"><option value="ron">On resistance R_on(Tj)</option><option value="diode">Diode forward and inverse</option><option value="energy">Switching energy</option></select></label>
          <label>Polynomial degree<select v-model="input.degree" data-testid="fit-degree"><option v-for="degree in [1,2,3]" :key="degree" :value="String(degree)">{{ degree }}</option></select></label>
          <label v-if="input.family === 'energy'">Switching event<select v-model="input.event" data-testid="fit-event"><option value="e_on">E_on · separate model</option><option value="e_off">E_off · separate model</option></select></label>
        </div>
        <button type="button" @click="example" data-testid="fit-example">Load hand fixture</button>
        <p>The fixture verifies algebra. It contains no measured device data. Loading it replaces the visible inputs.</p>
        <div class="fields"><label>Common source<input v-model="input.source" data-testid="fit-source" placeholder="Used only when a row source is blank"></label><label>Common series<input v-model="input.series" data-testid="fit-series" placeholder="Used only when a row series is blank"></label></div>
        <p>Each row requires a unique ID, source, series, and explicit train or holdout split. Row labels override common labels.</p>
        <p>Required CSV header: <code data-testid="fit-header">{{ header }}</code>. Target value unit: {{ unit }}.</p>
        <p>Coordinate units: current_a (A), voltage_v and v_g_v (V), t_j_c (°C), r_g_ohm (ohm).</p>
        <label>Raw observation CSV<textarea v-model="input.csv" rows="10" spellcheck="false" :placeholder="header" data-testid="fit-observations"></textarea></label>
        <p v-if="input.family === 'diode'">Temperature and gate voltage define exact, separate slices. Use blank or null for either unknown condition.</p>
        <p>Degree is explicit. Held-out rows do not select the degree or change the coefficients.</p>
      </fieldset>
      <fieldset v-if="conditionKeys.length"><legend>2. Fixed conditions</legend><p>A blank or null condition remains unknown. Unknown is distinct from zero.</p><div class="fields"><label v-for="key in conditionKeys" :key="key">{{ coordinateLabels[key] }}<input v-model="input.conditions[key]" inputmode="decimal" :data-testid="'fit-condition-' + key" placeholder="Unknown"></label></div></fieldset>
      <fieldset v-if="input.family === 'energy'"><legend>3. Explicit energy anchor</legend>
        <p>Supply real, isolated axis sweeps at this anchor. The nominal observation can be shared across sweeps. No synthetic reference is added.</p>
        <div class="fields"><label v-for="(_, key) in input.anchor" :key="key">Anchor {{ coordinateLabels[key] }}<input v-model="input.anchor[key]" inputmode="decimal" :data-testid="'fit-anchor-' + key"></label></div>
        <p>Current supplies the baseline energy. Other supported axes supply normalized factors. Missing axes do not imply a measured four-dimensional surface.</p>
      </fieldset>
      <button type="submit" :disabled="busy" data-testid="fit-train">{{ busy ? 'Fitting…' : 'Fit device model' }}</button>
    </form>
    <p v-if="stale" role="status" data-testid="fit-stale">Inputs changed. Fit again before prediction, review, or download.</p>
    <p v-if="error" role="alert">{{ error }}</p>
    <section v-if="fit" data-testid="fit-results">
      <h4>Fit, held-out diagnostics, and support</h4><p>{{ fit.scope }}</p><p>SI targets: resistance (ohm), voltage (V), current (A), energy (J).</p>
      <p>Signed residual = observation − prediction. Unavailable R² means insufficient rows or constant targets.</p>
      <div v-for="(diagnostic, index) in fit.diagnostics" :key="index" class="panel">
        <h4>{{ diagnostic.output }} ({{ fit.units[diagnostic.output] }}) · {{ diagnostic.slice }}</h4>
        <div class="fields"><article v-for="split in ['train','holdout']" :key="split" :data-testid="'fit-metrics-' + split"><h5>{{ split === 'train' ? 'Training fit' : 'Held-out evaluation' }}</h5>
          <template v-if="diagnostic[split]"><p>Supplied: {{ diagnostic[split].rows.length }} · Supported: {{ diagnostic[split].count }} · Excluded: {{ diagnostic[split].rows.length - diagnostic[split].count }}</p><p>RMSE: {{ number(diagnostic[split].rmse) }} · MAE: {{ number(diagnostic[split].mae) }} · R²: {{ number(diagnostic[split].r2) }}</p></template><p v-else>No held-out observations.</p>
        </article></div>
      </div>
      <div class="table-wrap"><table data-testid="fit-residuals"><thead><tr><th>Output / slice</th><th>ID / split</th><th>Source / series</th><th>Observed</th><th>Predicted</th><th>Signed residual</th><th>Status / exclusion</th></tr></thead>
        <tbody><tr v-for="(row, index) in residualRows" :key="index"><td>{{ row.output }} ({{ fit.units[row.output] }}) / {{ row.slice }}</td><td>{{ row.id }} / {{ row.split }}</td><td>{{ row.source }} / {{ row.series }}</td><td>{{ number(row.observed) }}</td><td>{{ number(row.predicted) }}</td><td>{{ number(row.residual) }}</td><td>{{ row.status }} {{ row.reason }}</td></tr></tbody></table></div>
      <div class="plots"><DeviceModelFitPlot v-for="plot in fit.plots || []" :key="plot.id" :plot="plot" :queries="plotQueries(plot)" /></div>
      <details><summary>Model basis, coefficients, support, conditions, and training sources</summary><pre data-testid="fit-model">{{ JSON.stringify(fit.model, null, 2) }}</pre></details>
      <details><summary>Accepted input and calculation evidence</summary><pre>{{ JSON.stringify({ inputs: fit.inputs, evidence: fit.evidence }, null, 2) }}</pre></details>
      <p>These downloads are the only persistence. Fitting does not change repository devices or stored core models.</p>
      <div class="downloads"><button type="button" @click="downloadAnalysis(fitJson, 'device-model-training-review.json', 'application/json;charset=utf-8')" data-testid="fit-json">Download exact training review JSON</button><button type="button" @click="downloadAnalysis(fit.model_json, 'device-model.json', 'application/json;charset=utf-8')" data-testid="fit-model-json">Download model JSON</button><button type="button" @click="exportResiduals" data-testid="fit-csv">Download residuals CSV</button></div>
    </section>
    <details class="panel"><summary>Reuse a saved device model JSON</summary>
      <p>Use the model JSON downloaded from this tool. Python validates its structure, training evidence, and coefficients before prediction.</p>
      <p>The preview is unvalidated. Importing a model supplies no held-out accuracy evidence.</p>
      <label>Saved model JSON<textarea v-model="importText" rows="7" spellcheck="false" data-testid="fit-import-json"></textarea></label>
      <button type="button" @click="useImportedModel" data-testid="fit-import">Use saved model for prediction</button><p v-if="importError" role="alert">{{ importError }}</p>
      <details v-if="imported"><summary>Unvalidated model preview</summary><pre>{{ JSON.stringify(imported.preview, null, 2) }}</pre></details>
    </details>
    <section v-if="fit || imported">
      <form class="panel" @submit.prevent="evaluate"><h4>{{ imported ? 'Predict with imported model · validation required' : 'Predict with this fitted model' }}</h4>
        <p v-if="imported" data-testid="fit-import-status">Imported family: {{ predictionFamily }}. The next request validates the original JSON in Python.</p>
        <label v-if="predictionFamily === 'diode'">Prediction direction<select v-model="query.output" data-testid="fit-query-output"><option value="voltage">Forward voltage (V) from current (A)</option><option value="current">Inverse current (A) from voltage (V)</option></select></label>
        <p>Supply exact fixed conditions and a coordinate inside the supported domain. Blank fixed conditions remain unknown.</p>
        <p v-if="predictionFamily === 'diode'">Blank temperature matches only a diode slice with unknown temperature. It does not match zero or a known temperature.</p>
        <div class="fields"><label v-for="key in queryKeys" :key="key">Query {{ coordinateLabels[key] }}<input v-model="query.coordinates[key]" inputmode="decimal" :data-testid="'fit-query-' + key" :placeholder="nullableQueryKeys.includes(key) ? 'Unknown' : 'Required'"></label></div>
        <button type="submit" :disabled="queryBusy" data-testid="fit-predict">{{ queryBusy ? 'Predicting…' : 'Predict with fitted model' }}</button>
      </form>
      <p v-if="queryError" role="alert">{{ queryError }}</p>
      <section v-if="prediction" class="panel" data-testid="fit-predictions"><h4>Model prediction</h4>
        <p v-for="row in prediction.result.predictions" :key="row.id" data-testid="fit-prediction">{{ row.id }} · {{ row.output }}: {{ number(row.value) }} {{ row.unit }} · {{ row.status }}<span v-if="row.reason"> · {{ row.reason }}</span></p>
        <p>{{ prediction.result.scope }}</p><details><summary>Prediction input, validated model, and evidence</summary><pre>{{ JSON.stringify(prediction, null, 2) }}</pre></details>
        <button type="button" @click="downloadAnalysis(predictionJson, 'device-model-prediction-review.json', 'application/json;charset=utf-8')" data-testid="fit-prediction-json">Download exact prediction review JSON</button>
      </section>
    </section>
  </section>
</template>

<style scoped>
.device-model-fitting { padding: 1rem; border: 1px solid #cbd5df; border-radius: 8px; background: white; min-width: 0; color: #263444; }
fieldset,.panel { min-width: 0; border: 1px solid #cbd5df; border-radius: 6px; padding: 1rem; margin: 1rem 0; } legend { font-weight: 600; }
.fields,.plots { display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,280px),1fr)); gap: 1rem; }.plots { margin: 1rem 0; grid-template-columns: repeat(auto-fit,minmax(min(100%,480px),1fr)); }
label { display: flex; flex-direction: column; gap: .4rem; margin: .7rem 0; } input,select,textarea { box-sizing: border-box; width: 100%; min-width: 0; padding: .65rem; font: inherit; border: 1px solid #aab8c5; border-radius: 4px; }
textarea { font-family: monospace; resize: vertical; } button { padding: .7rem 1rem; cursor: pointer; max-width: 100%; } button:disabled { cursor: wait; } p { line-height: 1.5; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; } code,p { overflow-wrap: anywhere; } .table-wrap { overflow-x: auto; } table { border-collapse: collapse; width: 100%; } td,th { text-align: left; padding: .6rem; border-bottom: 1px solid #cbd5df; min-width: 105px; }
h5 { font-size: 1rem; margin: .5rem 0; } .downloads { display: flex; flex-wrap: wrap; gap: .6rem; margin: 1rem 0; } [role=alert] { color: #af2b24; } [data-testid=fit-stale] { background: #fff0cf; padding: .7rem; }
@media(max-width:600px) { .device-model-fitting { padding: .5rem; } fieldset,.panel { padding: .65rem; } }
</style>
