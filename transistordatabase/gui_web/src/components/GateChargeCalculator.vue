<template>
  <section class="gate-calculator">
    <h3>Gate charge timing estimate</h3>
    <p>Qgs is measured from zero gate voltage to the plateau. Qg must be specified at the chosen Von.
      Charge values and the plateau must describe the same device operating condition.</p>
    <div class="gate-inputs">
      <label v-for="field in fields" :key="field.key">
        {{ field.label }} ({{ field.unit }})
        <input v-model.number="params[field.key]" :name="field.key" type="number" :step="field.step" />
      </label>
    </div>
    <label class="estimate-choice">
      <input v-model="estimateSelected" type="checkbox" />
      Use the constant-capacitance RC / Miller estimate
    </label>
    <label class="estimate-choice"><input v-model="settlingEnabled" type="checkbox" data-testid="gate-settling-enabled">Also estimate gate settling after the Miller plateau</label>
    <label v-if="settlingEnabled">Explicit target gate voltage (V)<input v-model="settlingTarget" inputmode="decimal" data-testid="gate-settling-target"></label>
    <p v-if="settlingEnabled">The target must remain strictly between Vplateau and Von. Qg remains measured at Von. This separate gate phase does not add drain switching energy.</p>
    <p>Normalized Id assumes linear transconductance. Normalized Vds is linear during Miller.
      These illustrations do not predict physical drain amplitudes, slew rates or switching energy.
      Negative Voff extends the constant-capacitance estimate below zero.</p>
    <button :disabled="calculating || !estimateSelected" @click="calculate">
      {{ calculating ? 'Calculating…' : 'Calculate switching intervals' }}
    </button>
    <p v-if="error" role="alert">{{ error }}</p>
    <div v-if="result" class="gate-results" aria-label="Gate timing results">
      <h4>Estimated switching intervals</h4>
      <p>Turn-on ends at Miller end, Vg = Vplateau. Turn-off ends at Vth.
        Zero-charge stages appear as instantaneous jumps.</p>
      <ul><li v-for="assumption in result.estimate.assumptions" :key="assumption">{{ assumption }}</li></ul>
      <section v-for="direction in directions" :key="direction.key" class="gate-direction">
        <h4>{{ direction.label }}</h4>
        <div class="gate-table-scroll"><table>
          <thead><tr><th>Interval</th><th>Duration (ns)</th><th>Start (ns)</th><th>End (ns)</th></tr></thead>
          <tbody>
            <tr v-for="phase in result.estimate[direction.key].phases" :key="phase.name">
              <td>{{ phaseLabels[phase.name] }}{{ phase.instantaneous ? ' (instantaneous)' : '' }}</td>
              <td>{{ ((phase.end_s - phase.start_s) * 1e9).toFixed(6) }}</td>
              <td>{{ (phase.start_s * 1e9).toFixed(6) }}</td>
              <td>{{ (phase.end_s * 1e9).toFixed(6) }}</td>
            </tr>
            <tr><th>Total {{ direction.label.toLowerCase() }}</th><td colspan="3">{{ result[direction.key + '_times'].t_total.toFixed(6) }} ns</td></tr>
          </tbody>
        </table></div>
        <div class="gate-plots">
          <CurvePlotter :title="direction.label + ' gate voltage estimate'" :datasets="datasets(direction.key, ['v_gate_v'])"
            x-label="Time (ns)" y-label="Gate voltage (V)" />
          <CurvePlotter :title="direction.label + ' gate current estimate'" :datasets="datasets(direction.key, ['i_gate_a'])"
            x-label="Time (ns)" y-label="Gate current (A)" />
          <CurvePlotter :title="direction.label + ' NORMALIZED drain illustration'" :datasets="datasets(direction.key, ['id_normalized', 'vds_normalized'])"
            x-label="Time (ns)" y-label="Normalized drain value (0–1)" />
        </div>
      </section>
      <section v-if="result.estimate.turn_on_settling" data-testid="gate-settling-results">
        <h4>Separate post-Miller gate settling estimate</h4>
        <p>{{ result.estimate.turn_on_settling.scope }}. {{ result.estimate.turn_on_settling.charge_reference }}.</p>
        <p>{{ result.estimate.turn_on_settling.time_basis }}. {{ result.estimate.turn_on_settling.assumption }}.</p>
        <div class="gate-table-scroll"><table><tbody>
          <tr><th>Status</th><td>{{ result.estimate.turn_on_settling.status }}</td></tr>
          <tr><th>Target gate voltage (V)</th><td>{{ displayNumber(result.estimate.turn_on_settling.target_gate_v) }}</td></tr>
          <tr><th>Additional gate settling duration (s)</th><td data-testid="gate-settling-duration">{{ displayNumber(result.estimate.turn_on_settling.duration_s) }}</td></tr>
          <tr><th>Incremental gate charge (C)</th><td>{{ displayNumber(result.estimate.turn_on_settling.incremental_charge_c) }}</td></tr>
          <tr><th>Post-plateau capacitance (F)</th><td>{{ displayNumber(result.estimate.turn_on_settling.post_plateau_capacitance_f) }}</td></tr>
          <tr><th>Existing Miller-end anchor (s)</th><td>{{ displayNumber(result.estimate.turn_on_settling.anchor_s) }}</td></tr>
          <tr><th>Absolute target timestamp (s)</th><td>{{ displayNumber(result.estimate.turn_on_settling.absolute_end_s) }} — {{ result.estimate.turn_on_settling.absolute_time_status }}</td></tr>
          <tr><th>Initial / final gate current (A)</th><td>{{ displayNumber(result.estimate.turn_on_settling.initial_gate_current_a) }} / {{ displayNumber(result.estimate.turn_on_settling.final_gate_current_a) }}</td></tr>
        </tbody></table></div>
        <div class="gate-plots">
          <CurvePlotter title="Post-Miller gate voltage estimate" :datasets="settlingDatasets('v_gate_v')" x-label="Relative time after Miller end (s)" y-label="Gate voltage (V)" :allow-zoom="false" />
          <CurvePlotter v-if="!result.estimate.turn_on_settling.instantaneous" title="Post-Miller gate current estimate" :datasets="settlingDatasets('i_gate_a')" x-label="Relative time after Miller end (s)" y-label="Gate current (A)" :allow-zoom="false" />
          <p v-else>No finite-time gate current pulse is defined for the ideal zero-charge crossing.</p>
        </div>
        <button @click="downloadAnalysis(result, 'gate-settling-evidence.json')" data-testid="gate-settling-json">Export gate request and result JSON</button>
        <button @click="exportSettlingCsv" data-testid="gate-settling-csv">Export relative gate settling trace CSV</button>
      </section>
    </div>
  </section>
</template>

<script setup>
import { ref, watch } from 'vue'
import { transistorApi } from '../services/api'
import CurvePlotter from './CurvePlotter.vue'
import { finiteDecimal, displayNumber, analysisCsv, downloadAnalysis } from './analysisFiles.js'

const fields = [
  { key: 'q_gs', label: 'Qgs: zero to plateau', unit: 'C', step: '1e-9' },
  { key: 'q_gd', label: 'Qgd: Miller charge', unit: 'C', step: '1e-9' },
  { key: 'q_g', label: 'Qg: total at Von', unit: 'C', step: '1e-9' },
  { key: 'v_th', label: 'Vth: threshold', unit: 'V', step: '0.1' },
  { key: 'v_plateau', label: 'Vplateau', unit: 'V', step: '0.1' },
  { key: 'r_g', label: 'External Rg', unit: 'Ω', step: '0.5' },
  { key: 'r_g_int', label: 'Internal Rg', unit: 'Ω', step: '0.1' },
  { key: 'v_driver', label: 'Von: driver on voltage', unit: 'V', step: '1' },
  { key: 'v_off', label: 'Voff: driver off voltage', unit: 'V', step: '1' }
]
const params = ref({ q_gs: 10e-9, q_gd: 15e-9, q_g: 50e-9, v_th: 3,
  v_plateau: 5, r_g: 6, r_g_int: 0, v_driver: 15, v_off: 0 })
const estimateSelected = ref(false)
const settlingEnabled = ref(false), settlingTarget = ref('')
const calculating = ref(false)
const result = ref(null)
const error = ref('')
let revision = 0
watch([params, estimateSelected, settlingEnabled, settlingTarget], () => {
  revision++
  result.value = null
  error.value = ''
  calculating.value = false
}, { deep: true, flush: 'sync' })
const directions = [{ key: 'turn_on', label: 'Turn-on' }, { key: 'turn_off', label: 'Turn-off' }]
const phaseLabels = { t_delay: 'Gate delay', t_rise: 'Current rise illustration',
  t_fall_v: 'Miller / voltage fall illustration', t_rise_v: 'Miller / voltage rise illustration',
  t_fall_i: 'Current fall illustration' }
const traceLabels = { v_gate_v: 'Vg estimate', i_gate_a: 'Ig estimate',
  id_normalized: 'NORMALIZED Id', vds_normalized: 'NORMALIZED Vds' }
function datasets(direction, keys) {
  const trace = result.value.estimate[direction]
  return keys.map((key, index) => ({ label: traceLabels[key],
    data: trace.time_s.map((t, i) => ({ x: t * 1e9, y: trace[key][i] })),
    borderColor: index ? '#d66826' : '#2875b8', pointRadius: 1 }))
}
function settlingDatasets(key) {
  const phase = result.value.estimate.turn_on_settling
  return [{ label: key === 'v_gate_v' ? 'Vg estimate' : 'Ig estimate',
    data: phase.relative_time_s.map((time, i) => ({ x: time, y: phase[key][i] })),
    borderColor: '#2875b8', pointRadius: phase.instantaneous ? 3 : 1, showLine: !phase.instantaneous }]
}
function exportSettlingCsv() {
  const phase = result.value.estimate.turn_on_settling
  const rows = phase.relative_time_s.map((time, i) => ({ ...result.value.parameters,
    relative_time_s: time, v_gate_v: phase.v_gate_v[i], i_gate_a: phase.i_gate_a[i],
    anchor_s: phase.anchor_s, duration_s: phase.duration_s, incremental_charge_c: phase.incremental_charge_c,
    absolute_end_s: phase.absolute_end_s, absolute_time_status: phase.absolute_time_status,
    status: phase.status, method: phase.method, time_basis: phase.time_basis, scope: phase.scope }))
  downloadAnalysis(analysisCsv(rows), 'gate-settling-trace.csv', 'text/csv')
}
async function calculate() {
  if (!estimateSelected.value) return
  const requestRevision = ++revision
  result.value = null
  error.value = ''
  if (Object.values(params.value).some(value => typeof value !== 'number' || !Number.isFinite(value))) {
    error.value = 'Provide a finite number for every gate parameter.'
    return
  }
  let optional = {}
  try {
    if (settlingEnabled.value) optional = { settling_target_v: finiteDecimal(settlingTarget.value, 'Gate settling target') }
  } catch (failure) { error.value = failure.message; return }
  calculating.value = true
  try {
    const response = await transistorApi.calculateGateCharge({ ...params.value, ...optional })
    if (revision === requestRevision) result.value = response
  } catch (err) {
    if (revision !== requestRevision) return
    const detail = err.response?.data?.detail
    error.value = Array.isArray(detail) ? detail.map(item => `${item.loc?.slice(-1)}: ${item.msg}`).join('; ')
      : detail || err.message || 'Gate estimate failed'
  } finally {
    if (revision === requestRevision) calculating.value = false
  }
}
</script>

<style scoped>
.gate-calculator { min-width: 0; padding: 20px; background: white; border: 1px solid #ddd; border-radius: 8px; }
.gate-calculator p, li { line-height: 1.5; overflow-wrap: anywhere; }
.gate-results, .gate-direction, .gate-plots { min-width: 0; max-width: 100%; }
.gate-table-scroll { min-width: 0; width: 100%; max-width: 100%; overflow-x: auto; }
.gate-inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 24px 0; }
.gate-inputs label { display: grid; gap: 6px; }
input[type=number], input[inputmode=decimal] { padding: 10px; width: 100%; box-sizing: border-box; }
.estimate-choice { display: flex; align-items: center; gap: 10px; font-weight: bold; }
button { padding: 12px 20px; background: #2875b8; color: white; border: 0; border-radius: 4px; cursor: pointer; }
button:disabled { background: #888; cursor: default; }
[role=alert] { background: #fce8e6; padding: 16px; color: #85271e; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; }
th, td { padding: 10px; border-bottom: 1px solid #ddd; text-align: left; }
.gate-plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 16px; }
.gate-direction { margin-top: 32px; }
</style>
