<script setup>
import { computed, onBeforeUnmount, reactive, ref, shallowRef, watch } from 'vue'
import { matApi } from '../services/api.js'
const props = defineProps({ source: { type: Object, required: true }, document: { type: Object, required: true } })
const identityFields = [['name', 'Declared part name'], ['type', 'Device type'], ['manufacturer', 'Manufacturer'], ['housing_type', 'Housing type'], ['author', 'Declaration author']]
const ratingFields = [['v_abs_max', 'Absolute voltage limit (V)'], ['i_abs_max', 'Absolute current limit (A)'], ['i_cont', 'Continuous current limit (A)'], ['t_j_max', 'Maximum junction temperature (°C)']]
const areaFields = [['housing_area', 'Housing area (m²)'], ['cooling_area', 'Cooling area (m²)']]
const optionalThermal = [['r_th_cs', 'Common case interface (K/W)'], ['r_th_switch_cs', 'Switch case interface (K/W)'], ['r_th_diode_cs', 'Diode case interface (K/W)'], ['t_c_max', 'Maximum case temperature (°C)']]
const conditionFields = [['t_j_c', 'Junction temperature (°C)'], ['gate_voltage_v', 'Gate voltage (V)'], ['gate_off_voltage_v', 'Gate-off voltage (V)'], ['gate_resistance_ohm', 'Gate resistance (Ω)'], ['supply_voltage_v', 'Original supply reference (V)']]
const mechanisms = ['forward', 'reverse'].flatMap(branch => [['conduction', 'Conduction'], ['turn-on', 'TurnOn'], ['turn-off', 'TurnOff']].map(([kind, stem]) => ({ branch, kind, key: `${branch}${stem}Trace`, label: `${branch} ${kind}` })))
function fresh() {
  return { identity: Object.fromEntries(identityFields.map(([key]) => [key, ''])), ratings: {}, areas: {}, units: false, declarationSource: '',
    rows: mechanisms.map(mechanism => ({ ...mechanism, selected: false, target: '', source: '', convention: '', indices: '', order: '', zero: '', status: 'recorded-source-values', scope: '', acknowledgement: '', conditions: {}, commutation: '', reason: '', evidence: '', extrapolation: 'reject' })) }
}
const input = reactive(fresh()), result = shallowRef(null), error = ref(''), busy = ref(false)
const finalizedRequest = shallowRef(null), calculation = shallowRef(null), calculateError = ref(''), calculating = ref(false), point = reactive({ target: '', current: '' })
let revision = 0, calculationRevision = 0, disposed = false
function invalidatePoint() { calculationRevision++; calculation.value = null; calculateError.value = ''; calculating.value = false }
watch(point, invalidatePoint, { deep: true, flush: 'sync' })
watch(input, () => { revision++; result.value = null; finalizedRequest.value = null; error.value = ''; busy.value = false; invalidatePoint() }, { deep: true, flush: 'sync' })
watch(() => props.source, () => { revision++; Object.assign(input, fresh()); result.value = null; error.value = '' }, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++; calculationRevision++ })
function numeric(value, label, nullable = false) {
  if (value === undefined || String(value).trim() === '') { if (nullable) return null; throw new Error(`Declare ${label}`) }
  if (!Number.isFinite(Number(value))) throw new Error(`Declare a finite ${label}`)
  return Number(value)
}
function reference(row) {
  const fields = props.document.variables.scData.fields
  const node = row.source === 'original-trace' ? fields[row.key]?.fields?.testVoltage : fields[row.kind === 'turn-on' ? 'turnOnVoltage' : 'turnOffVoltage']
  return node?.values?.flat(Infinity).join(', ') || 'Unknown'
}
async function finalize() {
  error.value = ''; result.value = null; busy.value = true; const token = ++revision
  try {
    if (!input.units) throw new Error('Confirm the proposed SI units explicitly')
    if (!input.declarationSource.trim()) throw new Error('Give the source for the independent declarations')
    const ratings = Object.fromEntries(ratingFields.map(([key, label]) => [key, numeric(input.ratings[key], label)]))
    const thermal_properties = Object.fromEntries(areaFields.map(([key, label]) => [key, numeric(input.areas[key], label)]))
    for (const [key, label] of optionalThermal) thermal_properties[key] = numeric(input.areas[key], label, true)
    const curves = [], omission_reasons = {}
    for (const row of input.rows) {
      if (!row.selected) { if (!row.reason.trim()) throw new Error(`Give an omission reason for ${row.label}`); omission_reasons[row.key] = row.reason; continue }
      if (![row.target, row.source, row.convention, row.order, row.zero].every(Boolean)) throw new Error(`Complete the explicit mapping for ${row.label}`)
      if (!row.evidence.trim()) throw new Error(`Give the evidence reference for ${row.label}`)
      const indices = row.indices.trim() ? row.indices.split(',').map(value => { if (!/^\s*\d+\s*$/.test(value)) throw new Error('Sample indices must be comma-separated nonnegative integers'); return Number(value) }) : null
      const conditions = Object.fromEntries(conditionFields.map(([key, label]) => [key, numeric(row.conditions[key], `${row.label} ${label}`, key !== 't_j_c')]))
      conditions.commutation_device = row.commutation.trim() ? row.commutation : null
      curves.push({ kind: row.kind, branch: row.branch, target: row.target, source: row.source, coordinate_convention: row.convention,
        sample_indices: indices, sample_order: row.order, zero_policy: row.zero, conditions, data_status: row.status,
        evidence_reference: row.evidence, extrapolation_policy: row.extrapolation,
        ideal_model_scope: row.status === 'declared-ideal-model' ? row.scope : null,
        source_zero_acknowledgement: row.status === 'source-zero-unverified' ? row.acknowledgement : null })
    }
    if (!curves.length) throw new Error('Select at least one supported branch curve')
    const request = { ...props.source, identity: { ...input.identity }, ratings, thermal_properties, curves, omission_reasons, units: 'SC-editor-SI-confirmed', declaration_source: input.declarationSource }
    const response = await matApi.finalize(request)
    if (!disposed && token === revision) { result.value = response; finalizedRequest.value = request }
  } catch (failure) {
    if (!disposed && token === revision) {
      const detail = failure.response?.data?.detail
      error.value = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(row => `${row.loc?.join('.')}: ${row.msg}`).join('; ') : failure.message
    }
  } finally { if (!disposed && token === revision) busy.value = false }
}
function listValues(node) { return node?.kind === 'list' ? node.values.slice(0, 200).map(listValues) : node }
const records = computed(() => {
  if (!result.value) return []
  const root = result.value.core_json_document.transistor.fields, rows = []
  for (const component of ['switch', 'diode']) for (const mechanism of ['channel_data', 'e_on_data', 'e_off_data', 'e_rr_data']) {
    for (const record of root[component]?.fields?.[mechanism]?.values ?? []) {
      const fields = record.fields, graph = fields.graph_v_i ?? fields.graph_i_e
      rows.push({ component, mechanism, temperature: fields.t_j, gate: fields.v_g, supply: fields.v_supply, graph: listValues(graph?.values) ?? [] })
    }
  }
  return rows
})
function download() {
  if (!result.value) return
  const blob = new Blob([JSON.stringify(result.value.core_json_document, null, 2)], { type: 'application/json' }), url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = 'legacy-mat-declared-core.json'; link.click(); URL.revokeObjectURL(url)
}
async function calculate() {
  invalidatePoint(); calculating.value = true; const token = calculationRevision
  try {
    if (!finalizedRequest.value || !point.target) throw new Error('Validate the declarations and select a mapped conduction target')
    const response = await matApi.calculate({ ...finalizedRequest.value, target: point.target, current_a: numeric(point.current, 'positive current (A)') })
    if (!disposed && token === calculationRevision) calculation.value = response
  } catch (failure) {
    if (!disposed && token === calculationRevision) { const detail = failure.response?.data?.detail; calculateError.value = typeof detail === 'string' ? detail : failure.message }
  } finally { if (!disposed && token === calculationRevision) calculating.value = false }
}
function downloadCalculation(format) {
  if (!calculation.value) return
  const data = calculation.value, content = format === 'csv' ? `current_A,voltage_V,resistance_ohm,coverage_min_A,coverage_max_A\r\n${[data.current_a, data.voltage_v, data.resistance_ohm, ...data.current_coverage_a].join(',')}\r\n` : JSON.stringify(data, null, 2)
  const url = URL.createObjectURL(new Blob([content], { type: format === 'csv' ? 'text/csv' : 'application/json' })), link = document.createElement('a')
  link.href = url; link.download = `legacy-mat-bounded-point.${format}`; link.click(); URL.revokeObjectURL(url)
}
</script>
<template>
  <section class="mapping" data-testid="mat-mapping">
    <h3>Declare a supported core model</h3>
    <p>The core accepts one branch per component mechanism. Declare each missing value independently. The original archive retains every omitted branch.</p>
    <p>Prefer original traces when combined arrays contain synthetic points or conflicting references. Negative energy remains unsupported for passive heat.</p>
    <form @submit.prevent="finalize">
      <fieldset><legend>Independent identity, ratings, and physical areas</legend><div class="grid">
        <label v-for="[key, label] in identityFields" :key="key">{{ label }}<input v-model="input.identity[key]" :data-testid="`mat-identity-${key}`" required></label>
        <label v-for="[key, label] in ratingFields" :key="key">{{ label }}<input v-model="input.ratings[key]" type="number" step="any" :data-testid="`mat-rating-${key}`" required></label>
        <label v-for="[key, label] in areaFields" :key="key">{{ label }}<input v-model="input.areas[key]" type="number" step="any" :data-testid="`mat-area-${key}`" required></label>
      </div><label>Source for identity, ratings, and areas<textarea v-model="input.declarationSource" required data-testid="mat-declaration-source" placeholder="Document, revision, figure or table, and declaration basis" /></label><details><summary>Optional explicit thermal declarations</summary><div class="grid"><label v-for="[key, label] in optionalThermal" :key="key">{{ label }}<input v-model="input.areas[key]" type="number" step="any"></label></div><p>A blank optional field remains unknown. Zero is a separate ideal-path declaration.</p></details></fieldset>
      <fieldset v-for="row in input.rows" :key="row.key" :data-testid="`mat-map-${row.key}`"><legend>{{ row.label }}</legend>
        <label class="check"><input v-model="row.selected" type="checkbox" :data-testid="`mat-select-${row.key}`"> Map this source mechanism</label>
        <div v-if="row.selected">
          <div class="grid">
            <label>Core target<select v-model="row.target" :data-testid="`mat-target-${row.key}`" required><option value="">Select target</option><option value="switch">Switch</option><option v-if="row.kind !== 'turn-on'" value="diode">Diode{{ row.kind === 'turn-off' ? ' recovery' : '' }}</option></select></label>
            <label>Source curve<select v-model="row.source" :data-testid="`mat-source-${row.key}`" required><option value="">Select source</option><option value="original-trace">Original branch trace (preferred)</option><option value="combined">Original combined signed array</option></select></label>
            <label>Coordinate convention<select v-model="row.convention" :data-testid="`mat-convention-${row.key}`" required><option value="">Select convention</option><option v-if="row.source !== 'combined'" value="trace-positive-magnitudes">Trace already uses positive magnitudes</option><option value="signed-passive">Signed passive branch convention</option></select></label>
            <label>Sample order<select v-model="row.order" :data-testid="`mat-order-${row.key}`" required><option value="">Select order</option><option value="recorded-order">Preserve recorded increasing current</option><option value="sort-distinct-current">Explicitly sort distinct current samples</option></select></label>
            <label>Zero-current policy<select v-model="row.zero" :data-testid="`mat-zero-${row.key}`" required><option value="">Select policy</option><option value="reject-zero">Reject zero-current samples</option><option value="recorded-branch-value">Use this branch's recorded zero value</option></select></label>
            <label>Original sample indices (zero-based)<input v-model="row.indices" :data-testid="`mat-indices-${row.key}`" placeholder="Blank selects all before branch separation"><small>Use comma-separated indices. Explicitly exclude combined ±1e−6 A synthetic candidates.</small></label>
            <label>Source interpretation<select v-model="row.status" :data-testid="`mat-status-${row.key}`"><option value="recorded-source-values">Recorded source values · measurement status unknown</option><option v-if="row.kind !== 'conduction'" value="source-zero-unverified">Recorded zero curve · evidence unverified</option><option value="declared-ideal-model">Explicitly declared ideal model</option></select></label>
            <label>Outside-calibration points<select v-model="row.extrapolation" :data-testid="`mat-extrapolation-${row.key}`"><option value="reject">Reject outside-calibration samples</option><option value="explicit-estimate">Explicitly accept a calibration extrapolation estimate</option></select></label>
          </div>
          <label>Source evidence and condition reference<textarea v-model="row.evidence" required :data-testid="`mat-evidence-${row.key}`" placeholder="Source figure, independent conditions, and known evidence limitations" /></label>
          <label v-if="row.status === 'source-zero-unverified'">Acknowledge absent measurement evidence<textarea v-model="row.acknowledgement" required :data-testid="`mat-ack-${row.key}`" /></label>
          <label v-if="row.status === 'declared-ideal-model'">Ideal model scope<textarea v-model="row.scope" required :data-testid="`mat-scope-${row.key}`" /></label>
          <p v-if="row.kind !== 'conduction'">Original voltage reference: {{ reference(row) }} V. Declare the same reference. No voltage scaling occurs.</p>
          <h4>Independent {{ row.label }} conditions</h4><div class="grid">
            <label v-for="[key, label] in conditionFields" :key="key">{{ label }}<input v-model="row.conditions[key]" type="number" step="any" :data-testid="`mat-condition-${row.key}-${key}`" :required="key === 't_j_c'"></label>
            <label>Commutation device<input v-model="row.commutation" :data-testid="`mat-commutation-${row.key}`"></label>
          </div><p>Blank optional conditions remain unknown. Energy requires gate voltage, gate resistance, and positive supply voltage.</p>
        </div>
        <label v-else>Reason to preserve this mechanism only in the archive<textarea v-model="row.reason" required :data-testid="`mat-omit-${row.key}`" /></label>
      </fieldset>
      <label class="check"><input v-model="input.units" type="checkbox" data-testid="mat-confirm-units"> I confirm SC editor SI units: A, V, J, K/W, and m².</label>
      <p>No filename value supplies a part name, temperature, rating, gate condition, or measurement status.</p>
      <button type="submit" :disabled="busy" data-testid="mat-finalize">{{ busy ? 'Validating in Python…' : 'Validate explicit core model' }}</button>
    </form>
    <p v-if="error" role="alert">{{ error }}</p>
    <section v-if="result" data-testid="mat-core-result"><h3>Declared core · {{ result.status }}</h3>
      <p>{{ result.declarations.identity.name }} · {{ result.core_json_document.schema }} version {{ result.core_json_document.version }}</p>
      <ul><li v-for="limit in result.limitations" :key="limit">{{ limit }}</li></ul>
      <div v-for="(record, i) in records" :key="i" class="table-scroll"><table><caption>Python finalized {{ record.component }} {{ record.mechanism }} · Tj {{ record.temperature }} °C · gate {{ record.gate ?? 'unknown' }} V · supply {{ record.supply ?? 'unknown' }} V. First 200 recorded rows. The native JSON retains the complete selection.</caption><thead><tr><th>Index</th><th>{{ record.mechanism === 'channel_data' ? 'Voltage (V)' : 'Current (A)' }}</th><th>{{ record.mechanism === 'channel_data' ? 'Current (A)' : 'Energy (J)' }}</th></tr></thead><tbody><tr v-for="(value, j) in record.graph[0] ?? []" :key="j"><td>{{ j }}</td><td>{{ value }}</td><td>{{ record.graph[1][j] }}</td></tr></tbody></table></div>
      <button @click="download" data-testid="mat-core-json">Download native core JSON</button>
      <p>The native document includes the original MAT bytes and declarations. No database write occurs.</p>
      <form @submit.prevent="calculate" class="bounded-point"><h4>Bounded conduction lookup in Python</h4><div class="grid">
        <label>Mapped conduction target<select v-model="point.target" data-testid="mat-calculate-target" required><option value="">Select target</option><option v-for="curve in result.declarations.curves.filter(curve => curve.kind === 'conduction')" :key="curve.target" :value="curve.target">{{ curve.target }}</option></select></label>
        <label>Current (A)<input v-model="point.current" type="number" step="any" min="0" required data-testid="mat-calculate-current"></label>
      </div><button type="submit" :disabled="calculating" data-testid="mat-calculate">{{ calculating ? 'Calculating in Python…' : 'Calculate bounded point' }}</button></form>
      <p v-if="calculateError" role="alert">{{ calculateError }}</p>
      <section v-if="calculation" data-testid="mat-calculation"><h4>Python result · {{ calculation.status }}</h4><dl><dt>Current (A)</dt><dd>{{ calculation.current_a }}</dd><dt>Voltage (V)</dt><dd data-testid="mat-calculation-voltage">{{ calculation.voltage_v }}</dd><dt>Resistance (Ω)</dt><dd data-testid="mat-calculation-resistance">{{ calculation.resistance_ohm }}</dd><dt>Recorded current coverage (A)</dt><dd>{{ calculation.current_coverage_a.join(' to ') }}</dd><dt>Source interpretation</dt><dd>{{ calculation.data_status }} · calibration extrapolation {{ calculation.extrapolation_policy }}</dd><dt>Junction temperature / gate voltage</dt><dd>{{ calculation.conditions?.t_j_c }} °C / {{ calculation.conditions?.gate_voltage_v ?? 'unknown' }} V</dd><dt>Condition evidence</dt><dd>{{ calculation.evidence_reference }}</dd><dt>Declaration source</dt><dd>{{ calculation.declaration_source }}</dd><dt>Method</dt><dd>{{ calculation.method }} · version {{ calculation.method_version }}</dd></dl>
        <p v-if="calculation.extrapolation_policy === 'explicit-estimate'">This result uses an explicitly declared calibration extrapolation estimate. It does not establish measured status.</p>
        <p>No evaluation outside the selected recorded current coverage occurs.</p><button @click="downloadCalculation('csv')" data-testid="mat-calculation-csv">Download point CSV</button> <button @click="downloadCalculation('json')" data-testid="mat-calculation-json">Download point JSON</button>
      </section>
    </section>
  </section>
</template>
<style scoped>
.mapping { min-width: 0; } h3,h4 { margin: .8rem 0; } p { margin: .6rem 0; } fieldset { min-width: 0; margin: 1rem 0; padding: 1rem; border: 1px solid #a7b7c9; border-radius: 6px; }
.grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .8rem; } label { display: flex; flex-direction: column; min-width: 0; margin: .5rem 0; gap: .3rem; } input,select,textarea { min-width: 0; width: 100%; box-sizing: border-box; padding: .6rem; font: inherit; color: inherit; background: var(--bg-primary,white); border: 1px solid #a0b0c2; border-radius: 4px; } .check { flex-direction: row; align-items: flex-start; } .check input { width: auto; flex: 0 0 auto; margin-top: .4rem; } small { font-weight: normal; } button { padding: .7rem; border: 1px solid #8198b1; border-radius: 5px; background: var(--bg-secondary,white); color: inherit; cursor: pointer; } button:disabled { opacity: .5; } [role=alert] { color: #b42323; overflow-wrap: anywhere; }
.table-scroll { max-width: 100%; overflow: auto; max-height: 420px; margin: 1rem 0; } table { min-width: 420px; width: 100%; border-collapse: collapse; } th,td { padding: .5rem; text-align: left; border-bottom: 1px solid #bdc9d7; } caption { text-align: left; } ul { margin-left: 1.4rem; } @media(max-width:850px) { .grid { grid-template-columns: repeat(2,minmax(0,1fr)); } } @media(max-width:550px) { .grid { grid-template-columns: minmax(0,1fr); } fieldset { padding: .7rem; } }
</style>
