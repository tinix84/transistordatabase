<template>
  <section class="biela-calculator">
    <h3>Christen–Biela half-bridge estimate</h3>
    <p>Enter charge-equivalent capacitances, the charge of one switch, transfer coefficients and circuit parasitics.
      Capacitances and Qoss must describe the stated bus voltage and temperature. Transfer measurement Vds stays separate.
      The initial values are an algebra
      verification example, with no claim to describe a commercial device.</p>
    <section v-if="staged" aria-label="Review extraction application">
      <h4>Review extracted parameters</h4>
      <p>Source: {{ staged.context.source }}. Recorded conditions: {{ staged.context.conditions }}.</p>
      <p>Apply only the listed parameters. The circuit, capacitances, Qoss and other parameter sources keep their current values.
        Transfer measurement Vds is independent of the model bus voltage.</p>
      <table><thead><tr><th>Parameter</th><th>Current value (SI)</th><th>Extracted value (SI)</th></tr></thead>
        <tbody><tr v-for="(value, key) in staged.result.biela_parameters" :key="key"><td>{{ key }}</td><td>{{ params[key] ?? recovery[key] }}</td><td>{{ value }}</td></tr></tbody>
      </table>
      <p v-if="staged.operation === 'reverse-recovery-extraction'">Application selects charge-control recovery.
        Source Qoss is {{ staged.inputs.q_oss_c }} C at {{ staged.inputs.conditions.q_oss_v_ref_v }} V / {{ staged.inputs.conditions.q_oss_t_j_c }} °C.
        Model Qoss is {{ params.q_oss }} C at {{ params.q_oss_v_ref }} V / {{ params.parameter_t_j }} °C.
        Supply compatible model charge and capacitances before calculation. No capacitance is inferred.</p>
      <button @click="applyExtraction">Apply these extracted parameters</button>
      <button @click="staged = null">Cancel extraction application</button>
    </section>
    <section v-if="evidence.length" aria-label="Applied extraction evidence">
      <h4>Extraction source records</h4>
      <div v-for="item in evidence" :key="item.inputs.operation">
        <p>{{ item.inputs.operation }}: {{ item.status === 'applied' ? 'Applied extracted parameters' : 'Edited manual parameters. Original extraction is historical evidence, not current calibration' }}.
          Source: {{ item.inputs.source }}. Recorded conditions: {{ item.inputs.conditions }}.</p>
        <details><summary>Original extraction samples, diagnostics and coverage</summary><pre>{{ JSON.stringify(item.envelope, null, 2) }}</pre></details>
      </div>
    </section>
    <div class="inputs">
      <label v-for="field in fields" :key="field.key">{{ field.label }} ({{ field.unit }})
        <input v-model.number="params[field.key]" :name="field.key" type="number" step="any" />
      </label>
    </div>
    <label class="source">Parameter source / provenance
      <textarea v-model="params.source" name="source" rows="2" />
    </label>
    <label>Reverse recovery
      <select v-model="params.recovery_mode" name="recovery_mode">
        <option value="omit">Explicitly omit reverse recovery</option>
        <option value="charge_control">Charge-control estimate with verified constants</option>
      </select>
    </label>
    <div v-if="params.recovery_mode === 'charge_control'" class="inputs">
      <label v-for="field in recoveryFields" :key="field.key">{{ field.label }} ({{ field.unit }})
        <input v-model.number="recovery[field.key]" :name="field.key" type="number" step="any" />
      </label>
    </div>
    <label class="choice"><input v-model="sweepSelected" type="checkbox" /> Include a bounded current sweep</label>
    <div v-if="sweepSelected" class="inputs">
      <label>Sweep minimum (A)<input v-model.number="sweep.i_min" name="i_min" type="number" step="any" /></label>
      <label>Sweep maximum (A)<input v-model.number="sweep.i_max" name="i_max" type="number" step="any" /></label>
      <label>Sweep points<input v-model.number="sweep.i_points" name="i_points" type="number" min="2" max="200" /></label>
    </div>
    <label class="choice"><input v-model="estimateSelected" type="checkbox" /> Use the corrected analytical half-bridge estimate</label>
    <button :disabled="calculating || !estimateSelected" @click="calculate">{{ calculating ? 'Calculating…' : 'Calculate half-bridge estimate' }}</button>
    <p v-if="error" role="alert">{{ error }}</p>
    <div v-if="result" aria-label="Biela estimate results">
      <h4>Estimated switching energy</h4>
      <div class="summary">
        <p v-for="key in ['e_on', 'e_off', 'e_total']" :key="key">{{ key }}: <strong>{{ fmt(result.results[key] * 1e6) }} µJ</strong></p>
      </div>
      <ul><li v-for="assumption in result.assumptions" :key="assumption">{{ assumption }}</li></ul>
      <p>Conditions: {{ result.operating_point.v_dc }} V, {{ result.operating_point.i_load }} A,
        {{ result.operating_point.t_j }} °C. Source: {{ result.parameter_inputs_si.source }}</p>
      <h4>Energy partitions</h4>
      <table><thead><tr><th>Event / phase</th><th>Energy (µJ)</th></tr></thead><tbody>
        <template v-for="event in ['turn_on', 'turn_off']" :key="event">
          <tr v-for="(value, key) in result.results.diagnostics[event + '_energy_partitions_j']" :key="event + key">
            <td>{{ event }} / {{ key }}</td><td>{{ fmt(value * 1e6) }}</td>
          </tr>
        </template>
        <tr><td>Complementary body diode recovery energy (separate)</td><td>{{ fmt(result.results.diagnostics.body_diode_energy_j * 1e6) }}</td></tr>
      </tbody></table>
      <div class="plots">
        <BielaEnergyBreakdown :diagnostics="result.results.diagnostics" />
        <CurvePlotter v-if="result.sweep" title="Supported current-sweep points" :datasets="sweepData" x-label="Load current (A)" y-label="Switching energy (µJ)" />
      </div>
      <h4>Phase values and convergence</h4>
      <table><thead><tr><th>Quantity</th><th>SI value</th></tr></thead><tbody>
        <tr v-for="(value, key) in scalarDetails" :key="key"><td>{{ key }}</td><td>{{ fmt(value) }}</td></tr>
      </tbody></table>
      <p v-for="event in ['turn_on', 'turn_off']" :key="event">{{ event }} solver:
        {{ result.results.diagnostics[event].iterations }} iterations;
        relative gm residual {{ fmt(result.results.diagnostics[event].gm_relative_residual) }};
        equation residual {{ fmt(result.results.diagnostics[event].equation_residual_a) }} A.</p>
      <p>Charge-equivalent consistency: relative difference
        {{ fmt(result.results.diagnostics.charge_consistency?.relative_difference || 0) }};
        accepted source-rounding tolerance 1%. The boundary uses the same Qoss as the Miller equation.</p>
      <section v-if="result.sweep" aria-label="Biela sweep evidence">
        <h4>Current sweep: {{ result.sweep.supported_points }} / {{ result.sweep.requested_points }} supported points</h4>
        <p>Unsupported points are excluded from the plotted curves. A failure describes this estimate's domain;
          it does not establish that the actual circuit or device is impossible.</p>
        <table><thead><tr><th>Current (A)</th><th>Status</th><th>Energy / diagnostic</th></tr></thead><tbody>
          <tr v-for="point in result.sweep.points" :key="point.current_a">
            <td>{{ fmt(point.current_a) }}</td><td>{{ point.status }}</td>
            <td>{{ point.status === 'supported' ? fmt(point.results.e_total * 1e6) + ' µJ' : point.error }}</td>
          </tr>
        </tbody></table>
      </section>
      <button @click="download">Download reproducible estimate JSON</button>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { transistorApi } from '../services/api'
import CurvePlotter from './CurvePlotter.vue'
import BielaEnergyBreakdown from './BielaEnergyBreakdown.vue'

const props = defineProps({ extraction: { type: Object, default: null } })
const staged = ref(null), evidence = ref([])
watch(() => props.extraction, value => { staged.value = value ? JSON.parse(JSON.stringify(value)) : null }, { immediate: true })

const fields = [
  ['c_gs', 'Charge-equivalent Cgs', 'F'], ['c_ds', 'Charge-equivalent Cds', 'F'], ['c_gd', 'Charge-equivalent Cgd', 'F'],
  ['q_oss', 'Single-device Qoss', 'C'], ['q_oss_v_ref', 'Qoss reference bus voltage', 'V'],
  ['k_1', 'Transfer coefficient k1', 'A / V^x'], ['k_2', 'Transfer offset k2', 'A'], ['x', 'Transfer exponent x', '1'],
  ['v_th', 'Known threshold Vth', 'V'], ['gm_i_min', 'Transfer-current coverage minimum', 'A'],
  ['gm_i_max', 'Transfer-current coverage maximum', 'A'], ['v_dc', 'DC bus voltage', 'V'], ['i_load', 'Load current', 'A'],
  ['v_driver_on', 'Driver Von', 'V'], ['v_driver_off', 'Driver Voff', 'V'], ['r_g', 'Total gate resistance', 'Ω'],
  ['l_s', 'Source inductance', 'H'], ['l_d', 'Drain inductance', 'H'], ['t_j', 'Requested junction temperature', '°C'],
  ['parameter_t_j', 'Parameter junction temperature', '°C']
].map(([key, label, unit]) => ({ key, label, unit }))
const recoveryFields = [['tau_rr', 'Recovery time constant', 's'], ['tau_c', 'Carrier lifetime', 's'],
  ['t_m', 'Transit time', 's'], ['q_rr_star', 'Corrected recovery charge', 'C']].map(([key, label, unit]) => ({ key, label, unit }))
const params = ref({ c_gs: 1e-9, c_ds: 1e-9, c_gd: 1e-9, q_oss: 200e-9, q_oss_v_ref: 100,
  k_1: 10, k_2: 0, x: 1, v_th: 3, gm_i_min: 1.3, gm_i_max: 50,
  v_dc: 100, i_load: 10, v_driver_on: 15, v_driver_off: 0, r_g: 10, l_s: 0, l_d: 0,
  t_j: 25, parameter_t_j: 25, source: 'Independent constant-gm algebra verification fixture; illustrative constants', recovery_mode: 'omit' })
const recovery = ref({ tau_rr: 2e-9, tau_c: 5e-9, t_m: 3.3333333333333334e-9, q_rr_star: 88e-9 })
const sweep = ref({ i_min: 0, i_max: 20, i_points: 11 })
const estimateSelected = ref(false), sweepSelected = ref(false), calculating = ref(false), error = ref(''), result = ref(null)
let revision = 0
watch([params, recovery, sweep, estimateSelected, sweepSelected], () => {
  revision++; result.value = null; error.value = ''; calculating.value = false
  for (const item of evidence.value) {
    if (item.status === 'applied' && (Object.entries(item.envelope.result.biela_parameters)
      .some(([key, value]) => (params.value[key] ?? recovery.value[key]) !== value)
      || (item.inputs.operation === 'reverse-recovery-extraction' && params.value.recovery_mode !== 'charge_control'))) item.status = 'edited'
  }
}, { deep: true, flush: 'sync' })
function applyExtraction() {
  if (!staged.value) return
  const envelope = JSON.parse(JSON.stringify(staged.value))
  const target = envelope.operation === 'transfer-fit' ? params.value : recovery.value
  Object.assign(target, envelope.result.biela_parameters)
  if (envelope.operation === 'reverse-recovery-extraction') params.value.recovery_mode = 'charge_control'
  evidence.value = [...evidence.value.filter(item => item.inputs.operation !== envelope.operation),
    { inputs: envelope.inputs, status: 'applied', envelope }]
  staged.value = null
  revision++; result.value = null; error.value = ''; calculating.value = false
}
function evidenceError() {
  for (const item of evidence.value.filter(item => item.status === 'applied')) {
    const condition = item.inputs.conditions
    if (condition.t_j_c !== params.value.parameter_t_j) return 'Applied extraction temperature differs from the model. Supply compatible model inputs.'
    if (item.inputs.operation === 'reverse-recovery-extraction' && (item.inputs.q_oss_c !== params.value.q_oss
      || condition.q_oss_v_ref_v !== params.value.q_oss_v_ref || condition.v_supply_v !== params.value.v_dc)) {
      return 'Applied recovery source Qoss or voltage differs from the model. Supply compatible Qoss and charge-equivalent capacitances.'
    }
  }
  return ''
}
const fmt = value => Number.isFinite(value) ? Number(value).toPrecision(8) : 'Unsupported'
const scalarDetails = computed(() => Object.fromEntries(Object.entries(result.value?.results || {}).filter(([, v]) => typeof v === 'number')))
const colors = ['#205f95', '#c75d20', '#29936c']
const sweepData = computed(() => ['e_on', 'e_off', 'e_total'].map((key, index) => ({
  label: key, borderColor: colors[index], spanGaps: false,
  data: result.value.sweep.points.map(p => ({ x: p.current_a, y: p.status === 'supported' ? p.results[key] * 1e6 : null }))
})))
async function calculate() {
  if (!estimateSelected.value) return
  const token = ++revision
  result.value = null; error.value = ''
  const mismatch = evidenceError()
  if (mismatch) { error.value = mismatch; return }
  const payload = { ...params.value, model_kind: 'christen_biela_estimate',
    extraction_evidence: evidence.value.map(({ inputs, status }) => ({ inputs, status })),
    ...(params.value.recovery_mode === 'charge_control' ? recovery.value : {}), ...(sweepSelected.value ? sweep.value : {}) }
  if (Object.entries(payload).some(([key, v]) => !['source', 'recovery_mode', 'model_kind', 'extraction_evidence'].includes(key) && (typeof v !== 'number' || !Number.isFinite(v)))) {
    error.value = 'Provide finite numeric parameters.'; return
  }
  calculating.value = true
  try {
    const response = await transistorApi.calculateBiela(payload)
    // API quantities remain SI. Guard the separate µJ display/chart conversion.
    const energies = data => [data.e_on, data.e_off, data.e_total,
      ...Object.values(data.diagnostics.turn_on_energy_partitions_j),
      ...Object.values(data.diagnostics.turn_off_energy_partitions_j), data.diagnostics.body_diode_energy_j]
    const displayValues = [...energies(response.results), ...(response.sweep?.points || [])
      .filter(point => point.status === 'supported').flatMap(point => energies(point.results))]
    if (displayValues.some(value => !Number.isFinite(value * 1e6))) {
      throw new Error('Energy exceeds the finite µJ display range. Use the SI Python/API result for this scale.')
    }
    if (token === revision) result.value = response
  } catch (err) {
    if (token !== revision) return
    const detail = err.response?.data?.detail
    error.value = Array.isArray(detail) ? detail.map(v => `${v.loc?.slice(-1)}: ${v.msg}`).join('; ') : detail || err.message
  } finally { if (token === revision) calculating.value = false }
}
function download() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(result.value, null, 2)], { type: 'application/json' }))
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'christen-biela-estimate.json'; anchor.click(); URL.revokeObjectURL(url)
}
</script>

<style scoped>
.biela-calculator { max-width: 1400px; margin: auto; }
.inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 270px), 1fr)); gap: 14px; margin: 20px 0; }
label { display: flex; flex-direction: column; gap: 6px; }
input, textarea, select { padding: 8px; min-width: 0; max-width: 100%; box-sizing: border-box; }
.choice { display: flex; flex-direction: row; margin: 18px 0; align-items: center; }
button { padding: 12px 18px; margin: 12px 0; cursor: pointer; }
.summary { display: flex; flex-wrap: wrap; gap: 24px; }
.plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr)); gap: 20px; }
table { width: 100%; border-collapse: collapse; overflow-wrap: anywhere; font-size: .9rem; margin: 18px 0; }
th, td { text-align: left; border-bottom: 1px solid #ccd2da; padding: 8px; }
[role=alert] { color: #982b20; background: #fff0ea; padding: 12px; }
</style>
