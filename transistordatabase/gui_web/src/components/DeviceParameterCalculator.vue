<template>
  <section class="parameter-workbench">
    <h3>Device parameter workbench</h3>
    <p><a href="#rg-energy">Go to repository Rg–energy queries</a></p>
    <p>Fit and evaluate supplied device data in SI units. Conditions must match exactly:
      <code>null</code> means unknown and never matches a known condition. No nearest-temperature
      selection or curve extrapolation is applied to measured tables.</p>
    <label>Calculation
      <select v-model="operation" aria-label="Parameter calculation">
        <option v-for="item in examples" :key="item.operation" :value="item.operation">{{ item.label }}</option>
      </select>
      <span>Changing the quantity loads illustrative sample points and resets their source. Replace them with your measured data.</span>
    </label>
    <p>{{ example.help }}</p>
    <label v-if="operation === 'measured-table'">Measured quantity
      <select :value="inputData.table?.kind" aria-label="Measured quantity" @change="changeTableKind($event.target.value)">
        <option value="energy">Switching energy versus current (A, J)</option>
        <option value="channel">Channel voltage versus current (A, V)</option>
        <option value="diode">Diode voltage versus current (A, V)</option>
        <option value="resistance">Resistance versus temperature (°C, Ω)</option>
        <option value="r_factor">Resistance factor versus temperature (°C, 1)</option>
      </select>
    </label>
    <div class="parameter-fields">
      <label v-for="field in simpleFields" :key="field.key">
        {{ label(field.key) }}
        <input v-if="typeof field.value === 'number' || field.value === null" :value="scaled(field.key, field.value)" type="number" step="any"
          :name="field.key" @input="setField(field.path, unscaled(field.key, numeric($event.target.value)))" />
        <input v-else :value="field.value" :name="field.key" @input="setField(field.path, $event.target.value)" />
      </label>
    </div>
    <fieldset v-for="group in conditionGroups" :key="group.path">
      <legend>{{ group.title }}</legend>
      <div class="parameter-fields">
        <div v-for="(value, key) in group.values" :key="key" class="condition-field">
          <label>{{ label(key) }}
            <input :value="value" type="number" step="any" :disabled="value === null && !isExtraction"
              :aria-label="group.title + ': ' + label(key)" @input="setField(group.path + '.' + key, numeric($event.target.value))" />
          </label>
          <label v-if="!isExtraction" class="unknown-choice"><input type="checkbox" :checked="value === null"
            :aria-label="group.title + ': ' + label(key) + ' unknown'"
            @change="setField(group.path + '.' + key, $event.target.checked ? null : 0)" />Unknown</label>
        </div>
      </div>
    </fieldset>
    <fieldset v-if="operation === 'transfer-fit'">
      <legend>Recorded channel curves</legend>
      <p>Each gate voltage has paired Vds/current samples. A single sample must equal the selected Vds exactly.
        Multiple samples use bounded linear Vds interpolation. All curves must have the stated temperature.</p>
      <p v-if="!editableCurves">Supply an array of curve objects with voltage and current sample arrays. Use the advanced editor to correct the shape.</p>
      <section v-for="(curve, index) in editableCurves || []" :key="index" :aria-label="'Transfer curve ' + (index + 1)">
        <div class="parameter-fields">
          <label>Gate voltage (V)<input type="number" step="any" :value="curve.v_g_v" :aria-label="'Curve ' + (index + 1) + ' gate voltage (V)'" @input="setField('curves.' + index + '.v_g_v', numeric($event.target.value))" /></label>
          <label>Recorded junction temperature (°C)<input type="number" step="any" :value="curve.t_j_c" :aria-label="'Curve ' + (index + 1) + ' temperature (°C)'" @input="setField('curves.' + index + '.t_j_c', numeric($event.target.value))" /></label>
        </div>
        <div class="sample-columns">
          <label>Recorded Vds samples (V)<textarea :value="curve.voltage_v.join('\n')" :aria-label="'Curve ' + (index + 1) + ' Vds samples (V)'" rows="3" @input="invalidate" @change="setSamples('curves.' + index + '.voltage_v', $event.target.value)" /></label>
          <label>Recorded current samples (A)<textarea :value="curve.current_a.join('\n')" :aria-label="'Curve ' + (index + 1) + ' current samples (A)'" rows="3" @input="invalidate" @change="setSamples('curves.' + index + '.current_a', $event.target.value)" /></label>
        </div>
        <button :disabled="editableCurves.length <= 4" @click="removeCurve(index)">Remove curve {{ index + 1 }}</button>
      </section>
      <button :disabled="!editableCurves || editableCurves.length >= 100" @click="addCurve">Add a recorded curve</button>
    </fieldset>
    <fieldset v-if="samplePair">
      <legend>{{ samplePair.title }}</legend>
      <p>Enter matching columns, separated by commas or new lines. Each row is one paired measurement.</p>
      <div class="sample-columns">
        <label v-for="column in samplePair.columns" :key="column.path">{{ column.label }}
          <textarea :value="column.values.join('\n')" :aria-label="column.label" rows="7"
            @input="invalidate" @change="setSamples(column.path, $event.target.value)" />
        </label>
      </div>
    </fieldset>
    <details class="data-editor"><summary>Advanced reproducible input (JSON; SI units)</summary>
      <label>All supplied data, sources and conditions
        <textarea v-model="inputText" aria-label="Parameter input data" spellcheck="false" rows="16" />
      </label>
    </details>
    <label v-if="requiresEstimate" class="estimate-choice">
      <input v-model="estimateSelected" type="checkbox" />
      Use the {{ example.label.toLowerCase() }} with its stated assumptions
    </label>
    <button :disabled="busy || (requiresEstimate && !estimateSelected)" @click="calculate">
      {{ busy ? 'Calculating…' : 'Calculate device parameters' }}
    </button>
    <p v-if="error" role="alert">{{ error }}</p>
    <section v-if="response" aria-label="Device parameter results">
      <h4>Result and diagnostics</h4>
      <p>{{ response.condition_policy }}. {{ response.units }}.</p>
      <template v-if="response.operation === 'gate-resistance-estimate'">
        <p>Source: {{ response.context.source }}</p>
        <p>Each selected value is the minimum total series resistance that keeps the predicted slew at or below the target throughout the plateau interval.
          Greater resistance reduces the predicted slew. The min/max fields describe the range of required resistance across plateau voltages.</p>
        <p>Total series resistance includes driver, internal and external resistance.</p>
        <ul><li v-for="assumption in response.result.assumptions" :key="assumption">{{ assumption }}</li></ul>
      </template>
      <dl><template v-for="[key, value] in scalarResults" :key="key"><dt>{{ label(key) }}</dt><dd>{{ displayValue(value) }}</dd></template></dl>
      <template v-if="response.result.diagnostics">
        <h4>Fit diagnostics and coverage</h4>
        <dl><template v-for="(value, key) in response.result.diagnostics" :key="key"><dt>{{ label(key) }}</dt><dd>{{ displayValue(value) }}</dd></template></dl>
      </template>
      <div v-if="response.operation === 'transfer-fit'" class="table-scroll" role="region" aria-label="Scrollable transfer fit samples" tabindex="0">
      <table aria-label="Exact transfer fit samples">
        <thead><tr><th>Vgs (V)</th><th>Recorded current (A)</th><th>Fitted current (A)</th><th>Residual (A)</th><th>Secant gm (S)</th></tr></thead>
        <tbody><tr v-for="(gate, index) in response.result.gate_voltage_v" :key="gate"><td>{{ displayValue(gate) }}</td><td>{{ displayValue(response.result.current_a[index]) }}</td><td>{{ displayValue(response.result.fitted_current_a[index]) }}</td><td>{{ displayValue(response.result.residual_a[index]) }}</td><td>{{ displayValue(response.result.secant_gm_s[index]) }}</td></tr></tbody>
      </table>
      </div>
      <dl v-if="response.operation === 'reverse-recovery-extraction'" aria-label="Recovery readable units">
        <template v-for="key in ['tau_rr', 'tau_c', 't_m']" :key="key"><dt>{{ label(key) }} (ns)</dt><dd>{{ displayValue(response.result[key] * 1e9) }}</dd></template>
        <dt>Corrected recovery charge (nC)</dt><dd>{{ displayValue(response.result.q_rr_star * 1e9) }}</dd>
      </dl>
      <div class="parameter-plots">
        <CurvePlotter v-for="plot in plots" :key="plot.title" :title="plot.title"
          :x-label="plot.x_label" :y-label="plot.y_label" :datasets="plot.datasets"
          :compact-scientific-y-ticks="plot.compactScientificYTicks" />
      </div>
      <details><summary>All result values, source and assumptions</summary><pre>{{ JSON.stringify(response, null, 2) }}</pre></details>
      <button @click="download">Download reproducible result</button>
      <button v-if="response.csv" @click="downloadCsv">Download exact SI points CSV</button>
      <button v-if="response.result.biela_parameters" @click="stageBiela">Review extracted parameters in Biela</button>
    </section>
    <RgEnergyPanel />
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import api from '../services/api'
import CurvePlotter from './CurvePlotter.vue'
import RgEnergyPanel from './RgEnergyPanel.vue'

const emit = defineEmits(['apply-biela', 'withdraw-biela'])

const context = { source: 'User supplied example — replace with your source', conditions: { t_j_c: 25, v_g_v: 15 } }
const examples = [
  { operation: 'transfer-fit', label: 'Nonlinear transfer fit for Biela', help: 'Fit I=k1*(Vgs−Vth)^x+k2 at a known transfer Vds/Tj and known Vth. The illustrative samples verify algebra. They are not commercial-device measurements. Transfer Vds is independent of the Biela bus voltage.',
    data: { source: 'Independent quadratic transfer verification fixture; replace with recorded device data', conditions: { v_ds_v: 10, t_j_c: 25 }, v_th_v: 3, curves: [2.1, 8.1, 18.1, 32.1, 50.1].map((current, index) => ({ v_g_v: index + 4, t_j_c: 25, voltage_v: [10], current_a: [current] })) } },
  { operation: 'reverse-recovery-extraction', label: 'Reverse-recovery charge-control extraction', help: 'Supply Qrr, peak Irr, positive current-slope magnitude and forward test current at one known test condition. Qoss is for one device at that voltage and temperature. The illustrative fixture verifies equations, not measured device accuracy. Trr is optional compatibility metadata and does not truncate the exponential tail.',
    data: { source: 'Independent recovery extraction verification fixture; replace with recorded device data', conditions: { t_j_c: 25, v_supply_v: 100, q_oss_v_ref_v: 100, q_oss_t_j_c: 25 }, q_rr_c: 100e-9, i_rr_a: 10, di_dt_a_s: 2.4e9, q_oss_c: 12e-9, i_forward_test_a: 20, t_rr_s: null } },
  { operation: 'linear-fit', label: 'Linear V-I fit with strict current bounds', help: 'Threshold-plus-resistance fit. Bounds exclude their endpoints; inspect residuals and sample count.',
    data: { ...context, current_a: [0, 5, 10, 20], voltage_v: [1.2, 1.35, 1.5, 1.8], lower_a: 0, upper_a: 21 } },
  { operation: 'terminal-capacitances', label: 'Ciss / Coss / Crss decomposition', help: 'All three capacitances must describe the same voltage, frequency, temperature and bias condition.',
    data: { source: context.source, conditions: { v_ds_v: 200, frequency_hz: 1e6, t_j_c: 25, v_g_v: 0 }, c_iss_f: 2e-9, c_oss_f: 5e-10, c_rss_f: 1e-10 } },
  { operation: 'capacitance-integration', label: 'Bounded C(V) charge and energy', help: 'Piecewise-linear capacitance integration uses interpolated endpoints and rejects extrapolation. Q and E are relative to start_v.',
    data: { source: context.source, conditions: { t_j_c: 25, v_g_v: 0 }, voltage_v: [0, 100, 200], capacitance_f: [1e-9, 5e-10, 2e-10], start_v: 0, end_v: 200 } },
  { operation: 'gate-slew-estimate', label: 'Signed gate current and slew estimate', help: 'Constant transconductance/capacitance estimate: positive gate current raises collector current and lowers collector voltage. No parasitic transient prediction.',
    data: { ...context, v_th_v: 4, current_a: 20, gm_s: 10, r_g_ohm: 5, c_ies_f: 2e-9, c_res_f: 1e-10, v_drive_v: 15 } },
  { operation: 'gate-resistance-estimate', label: 'Gate resistance from target voltage slew', help: 'Estimate total gate-loop series resistance from positive |dV/dt| in V/s and minimum Crss in F. Driver rails must enclose the supplied Miller plateau interval. The selected values are minimum total resistance requirements for the full plateau interval. Greater resistance reduces the predicted slew. These values include driver and internal resistance. They do not specify an external resistor or calibrated switching loss.',
    data: { source: 'Independent gate resistance algebra fixture; replace with recorded capacitance and plateau evidence', conditions: { t_j_c: 25, v_supply_v: 400, current_a: null }, target_dv_dt_v_s: 20e9, c_rss_min_f: 100e-12, v_drive_high_v: 15, v_drive_low_v: -5, v_plateau_low_v: 4, v_plateau_high_v: 7 } },
  { operation: 'measured-table', label: 'Bounded condition-specific data table', help: 'Kinds: energy (A/J), channel or diode (A/V), resistance (°C/Ω), r_factor (°C/1, requires positive r_nom_ohm). Unknown conditions must remain null.',
    data: { table: { kind: 'energy', x: [0, 10, 20], y: [0.001, 0.002, 0.004], source: context.source, conditions: { t_j_c: 25, v_supply_v: 400, v_g_v: 15, r_g_ohm: null } }, query: 5, conditions: { t_j_c: 25, v_supply_v: 400, v_g_v: 15, r_g_ohm: null }, r_nom_ohm: null } },
  { operation: 'energy-current-coefficient', label: 'Workbook energy-per-current coefficient', help: 'Stores J/A with its test current and conditions. This coefficient is not an event energy or a measured E(I) curve. Missing switching components remain missing.',
    data: { source: context.source, conditions: { t_j_c: 175, v_supply_v: 600, v_g_v: null, r_g_ohm: null }, energy_j: 0.000632, reference_current_a: 36 } },
  { operation: 'resistance-temperature-estimate', label: 'Resistance temperature / maximum estimate', help: 'Explicit linear estimate, with extrapolation disclosed. Source points must share gate bias. Optional max_typ_ratio transfers an assumed maximum/typical ratio.',
    data: { source: context.source, conditions: { v_g_v: 15 }, t1_c: 25, r1_ohm: 0.04, t2_c: 125, r2_ohm: 0.06, target_c: 150, max_typ_ratio: 1.2 } },
  { operation: 'separable-energy-estimate', label: 'Separable switching-energy estimate', help: 'Independent multiplicative condition effects are assumed. Supply bounded isolated energy-condition sweeps at a reference current, with other conditions matching baseline. Residuals disclose disagreement at the shared reference; interactions are unvalidated.',
    data: { current_table: { kind: 'energy', x: [0, 10, 20], y: [0.001, 0.002, 0.004], conditions: { t_j_c: 25, v_supply_v: 400, r_g_ohm: 5 }, source: context.source }, current_a: 5, conditions: { t_j_c: 75, v_supply_v: 400, r_g_ohm: 5 }, slices: { t_j_c: { kind: 'energy-condition', x: [25, 125], y: [0.002, 0.003], conditions: { current_a: 10, v_supply_v: 400, r_g_ohm: 5 }, source: context.source } } } }
]
const operation = ref('linear-fit')
const example = computed(() => examples.find(item => item.operation === operation.value))
const inputText = ref(JSON.stringify(example.value.data, null, 2))
const estimateSelected = ref(false)
const requiresEstimate = computed(() => operation.value.endsWith('-estimate'))
const isExtraction = computed(() => ['transfer-fit', 'reverse-recovery-extraction'].includes(operation.value))
const response = ref(null), error = ref(''), busy = ref(false)
const labels = {
  q_rr_c: 'Recorded recovery charge Qrr (nC)', i_rr_a: 'Peak recovery current Irr (A)',
  q_oss_c: 'Single-device Qoss (nC)',
  i_forward_test_a: 'Forward test current (A)', t_rr_s: 'Optional threshold-defined Trr (ns)',
  q_oss_v_ref_v: 'Qoss reference voltage (V)', q_oss_t_j_c: 'Qoss junction temperature (°C)',
  current_min: 'Transfer-current coverage minimum (A)', current_max: 'Transfer-current coverage maximum (A)',
  tau_rr: 'Recovery time constant', tau_c: 'Carrier lifetime', t_m: 'Transit time',
  q_rr_star: 'Corrected recovery charge (C)', rmse_a: 'Fit RMSE (A)', max_abs_residual_a: 'Maximum absolute fit residual (A)',
  rank: 'Jacobian rank', jacobian_condition: 'Jacobian condition number',
  source: 'Data source / reference', lower_a: 'Lower current bound, excluded (A)', upper_a: 'Upper current bound, excluded (A)',
  c_iss_f: 'Input capacitance Ciss (F)', c_oss_f: 'Output capacitance Coss (F)', c_rss_f: 'Reverse transfer capacitance Crss (F)',
  c_gs_f: 'Gate-source capacitance Cgs (F)', c_ds_f: 'Drain-source capacitance Cds (F)', c_gd_f: 'Gate-drain capacitance Cgd (F)',
  start_v: 'Integration start voltage (V)', end_v: 'Integration end voltage (V)', v_th_v: 'Threshold voltage (V)',
  current_a: 'Collector / drain current (A)', gm_s: 'Transconductance (S)', r_g_ohm: 'Gate resistance (Ω)', c_ies_f: 'Input capacitance Cies (F)',
  c_res_f: 'Reverse transfer capacitance Cres (F)', v_drive_v: 'Driver voltage (V)', query: 'Requested current (A) or temperature (°C)',
  r_nom_ohm: 'Nominal resistance for factor conversion (Ω)', energy_j: 'Event energy (J)', reference_current_a: 'Reference test current (A)',
  t1_c: 'First reference temperature (°C)', t2_c: 'Second reference temperature (°C)', target_c: 'Requested temperature (°C)',
  r1_ohm: 'Resistance at first temperature (Ω)', r2_ohm: 'Resistance at second temperature (Ω)', max_typ_ratio: 'Assumed maximum / typical ratio (1)',
  t_j_c: 'Junction temperature (°C)', v_g_v: 'Gate bias (V)', v_ds_v: 'Drain-source voltage (V)', frequency_hz: 'Measurement frequency (Hz)',
  v_supply_v: 'Switching supply voltage (V)', v_g_off_v: 'Turn-off gate voltage (V)',
  v0_v: 'Fitted threshold voltage (V)', resistance_ohm: 'Resistance (Ω)', rmse_v: 'Fit RMS residual (V)', max_residual_v: 'Maximum fit residual (V)',
  r_squared: 'Fit coefficient of determination', sample_count: 'Samples inside bounds', degrees_of_freedom: 'Fit degrees of freedom',
  scope: 'Calculation scope', charge_c: 'Integrated charge (C)', v_plateau_v: 'Plateau voltage (V)', gate_current_a: 'Signed gate current (A)',
  di_dt_a_s: 'Signed collector current slew (A/s)', dv_ce_dt_v_s: 'Signed collector voltage slew (V/s)',
  target_dv_dt_v_s: 'Target voltage-slew magnitude |dV/dt| (V/s)', c_rss_min_f: 'Minimum reverse transfer capacitance Crss (F)',
  v_drive_high_v: 'Driver high rail (V)', v_drive_low_v: 'Driver low rail (V)',
  v_plateau_low_v: 'Miller plateau lower bound (V)', v_plateau_high_v: 'Miller plateau upper bound (V)',
  r_on_ohm: 'Minimum turn-on total series resistance (Ω)', r_off_ohm: 'Minimum turn-off total series resistance (Ω)',
  r_on_min_ohm: 'Turn-on required resistance range minimum (Ω)', r_on_max_ohm: 'Turn-on required resistance range maximum (Ω)',
  r_off_min_ohm: 'Turn-off required resistance range minimum (Ω)', r_off_max_ohm: 'Turn-off required resistance range maximum (Ω)',
  gate_current_magnitude_a: 'Gate-current magnitude at target slew (A)', sizing_policy: 'Resistance selection policy', model_source: 'Model equation source',
  coefficient_j_per_a: 'Energy-per-current coefficient (J/A)', reference_energy_j: 'Reference test energy (J)', unit: 'Result unit',
  value: 'Evaluated value', mode: 'Calculation mode', kind: 'Recorded quantity', query_unit: 'Query unit',
  factor: 'Interpolated resistance factor (1)', factor_unit: 'Factor unit', estimated_max_ohm: 'Estimated maximum resistance (Ω)', extrapolated: 'Extrapolated beyond reference temperatures'
}
function label(key) {
  if (operation.value === 'reverse-recovery-extraction' && key === 'di_dt_a_s') return 'Positive current-slope magnitude (GA/s)'
  if (operation.value === 'transfer-fit' && key === 'v_ds_v') return 'Transfer measurement Vds (V)'
  if (operation.value === 'transfer-fit' && key === 'v_th_v') return 'Known threshold Vth (V)'
  return labels[key] || key.replaceAll('_', ' ').replace(/^./, char => char.toUpperCase())
}
function displayValue(value) { return typeof value === 'number' ? Number(value.toPrecision(8)).toString() : value === null ? 'Undefined for these samples' : String(value) }
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value)
const inputData = computed(() => { try { const parsed = JSON.parse(inputText.value); return isObject(parsed) ? parsed : {} } catch { return {} } })
const editableCurves = computed(() => Array.isArray(inputData.value.curves) && inputData.value.curves.every(curve => isObject(curve)
  && Array.isArray(curve.voltage_v) && Array.isArray(curve.current_a)) ? inputData.value.curves : null)
let revision = 0
watch(operation, () => { inputText.value = JSON.stringify(example.value.data, null, 2); estimateSelected.value = false })
watch([inputText, operation, estimateSelected], invalidate, { flush: 'sync' })
const simpleFields = computed(() => {
  const data = inputData.value
  const fields = Object.entries(data).filter(([key, value]) => key !== 'r_nom_ohm' && key !== 'selected' && (['number', 'string'].includes(typeof value) || value === null)).map(([key, value]) => ({ key, value, path: key }))
  if (isObject(data.table)) {
    fields.push({ key: 'source', value: data.table.source, path: 'table.source' })
    if (data.table.kind === 'r_factor') fields.push({ key: 'r_nom_ohm', value: data.r_nom_ohm, path: 'r_nom_ohm' })
  }
  return fields
})
const conditionGroups = computed(() => {
  const data = inputData.value, groups = []
  if (isObject(data.conditions)) groups.push({ path: 'conditions', title: data.table || data.current_table ? 'Requested conditions' : 'Source conditions', values: data.conditions })
  if (isObject(data.table?.conditions)) groups.push({ path: 'table.conditions', title: 'Recorded table conditions', values: data.table.conditions })
  if (isObject(data.current_table?.conditions)) groups.push({ path: 'current_table.conditions', title: 'Baseline conditions', values: data.current_table.conditions })
  return groups
})
const samplePair = computed(() => {
  const data = inputData.value
  if (operation.value === 'linear-fit' && Array.isArray(data.current_a) && Array.isArray(data.voltage_v)) return { title: 'Measured V-I points', columns: [{ path: 'current_a', label: 'Measured current samples (A)', values: data.current_a }, { path: 'voltage_v', label: 'Measured voltage samples (V)', values: data.voltage_v }] }
  if (operation.value === 'capacitance-integration' && Array.isArray(data.voltage_v) && Array.isArray(data.capacitance_f)) return { title: 'Measured capacitance points', columns: [{ path: 'voltage_v', label: 'Measured voltage samples (V)', values: data.voltage_v }, { path: 'capacitance_f', label: 'Measured capacitance samples (F)', values: data.capacitance_f }] }
  if (isObject(data.table) && Array.isArray(data.table.x) && Array.isArray(data.table.y)) {
    const kind = data.table.kind, temperature = ['resistance', 'r_factor'].includes(kind)
    return { title: 'Measured table points', columns: [{ path: 'table.x', label: temperature ? 'Measured temperature samples (°C)' : 'Measured current samples (A)', values: data.table.x || [] }, { path: 'table.y', label: kind === 'energy' ? 'Measured energy samples (J)' : kind === 'resistance' ? 'Measured resistance samples (Ω)' : kind === 'r_factor' ? 'Measured resistance factors (1)' : 'Measured voltage samples (V)', values: data.table.y || [] }] }
  }
  return null
})
const scalarResults = computed(() => Object.entries(response.value?.result || {}).filter(([, value]) => value === null || typeof value !== 'object'))
function numeric(value) { return value === '' ? null : Number(value) }
const scales = { q_rr_c: 1e9, q_oss_c: 1e9, di_dt_a_s: 1e-9, t_rr_s: 1e9 }
function scale(key) { return operation.value === 'reverse-recovery-extraction' ? scales[key] || 1 : 1 }
function scaled(key, value) { return value === null ? null : value * scale(key) }
function unscaled(key, value) { return value === null ? null : value / scale(key) }
function addCurve() { const data = JSON.parse(inputText.value); data.curves.push({ v_g_v: null, t_j_c: data.conditions.t_j_c, voltage_v: [], current_a: [] }); inputText.value = JSON.stringify(data, null, 2) }
function removeCurve(index) { const data = JSON.parse(inputText.value); data.curves.splice(index, 1); inputText.value = JSON.stringify(data, null, 2) }
function stageBiela() { if (response.value?.result.biela_parameters) emit('apply-biela', JSON.parse(JSON.stringify(response.value))) }
function invalidate() { revision++; response.value = null; error.value = ''; busy.value = false; emit('withdraw-biela') }
function setField(path, value) { const data = JSON.parse(inputText.value), keys = path.split('.'); let target = data; for (const key of keys.slice(0, -1)) target = target[key]; target[keys.at(-1)] = value; inputText.value = JSON.stringify(data, null, 2) }
function setSamples(path, value) {
  const tokens = value.trim().split(/[\s,;]+/)
  if (tokens.some(token => !token || !Number.isFinite(Number(token)))) { setField(path, tokens.map(token => token && Number.isFinite(Number(token)) ? Number(token) : null)); error.value = 'Enter finite numeric samples separated by commas or new lines.'; return }
  setField(path, tokens.map(Number))
}
function changeTableKind(kind) {
  const data = JSON.parse(inputText.value)
  data.table.kind = kind
  data.table.source = context.source
  if (['resistance', 'r_factor'].includes(kind)) { data.table.x = [25, 125]; data.table.y = kind === 'r_factor' ? [1, 1.5] : [0.04, 0.06]; data.table.conditions = { v_g_v: 15 }; data.query = 75; data.r_nom_ohm = kind === 'r_factor' ? .04 : null }
  else { data.table.x = [0, 10, 20]; data.table.y = kind === 'energy' ? [.001, .002, .004] : [.7, 1, 1.3]; data.table.conditions = { t_j_c: 25, v_g_v: kind === 'diode' ? null : 15 }; if (kind === 'energy') Object.assign(data.table.conditions, { v_supply_v: 400, r_g_ohm: null }); data.query = 5; data.r_nom_ohm = null }
  data.conditions = { ...data.table.conditions }
  inputText.value = JSON.stringify(data, null, 2)
}
const plots = computed(() => (response.value?.plots || []).map(plot => ({ ...plot,
  compactScientificYTicks: response.value.operation === 'transfer-fit' && plot.title === 'Transfer residual',
  datasets: Object.entries(plot.series).map(([label, values], i) => ({ label, data: plot.x.map((x, j) => ({ x, y: values[j] })), borderColor: i ? '#ca6424' : '#2679b8', pointRadius: 3 }))
})))
async function calculate() {
  const version = ++revision
  error.value = ''; response.value = null
  try {
    const data = JSON.parse(inputText.value)
    if (requiresEstimate.value && !estimateSelected.value) return
    const payload = { ...data, operation: operation.value }
    if (requiresEstimate.value) payload.selected = true
    busy.value = true
    const result = await api.post('/api/parameters/calculate', payload)
    if (version === revision) response.value = result.data
  } catch (err) {
    if (version !== revision) return
    const detail = err.response?.data?.detail
    error.value = Array.isArray(detail) ? detail.map(item => `${item.loc?.slice(-1)}: ${item.msg}`).join('; ') : detail || err.message
  } finally { if (version === revision) busy.value = false }
}
function download() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(response.value, null, 2)], { type: 'application/json' }))
  const link = document.createElement('a'); link.href = url; link.download = `device-parameters-${operation.value}.json`; link.click(); URL.revokeObjectURL(url)
}
function downloadCsv() {
  if (!response.value?.csv) return
  const url = URL.createObjectURL(new Blob([response.value.csv], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a'); link.href = url; link.download = `device-parameters-${operation.value}.csv`; link.click(); URL.revokeObjectURL(url)
}
</script>

<style scoped>
.parameter-workbench { background: white; padding: 20px; border: 1px solid #ddd; border-radius: 8px; min-width: 0; max-width: 100%; box-sizing: border-box; }
p { line-height: 1.5; } label { display: grid; gap: 6px; min-width: 0; } input, select, textarea { padding: 10px; box-sizing: border-box; width: 100%; min-width: 0; max-width: 100%; }
.parameter-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: 14px; margin: 20px 0; }
.data-editor { margin: 20px 0; } textarea, pre { font-family: monospace; font-size: 13px; } pre { white-space: pre-wrap; overflow-wrap: anywhere; }
fieldset { border: 1px solid #cbd4df; border-radius: 6px; margin: 20px 0; min-width: 0; } legend { font-weight: 600; padding: 0 8px; }
.sample-columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.unknown-choice { display: flex; align-items: center; gap: 8px; margin-top: 6px; } .unknown-choice input { width: auto; }
.estimate-choice { display: flex; align-items: center; margin: 18px 0; } .estimate-choice input { width: auto; }
button { padding: 12px 20px; background: #2875b8; color: white; border: 0; border-radius: 4px; cursor: pointer; margin: 10px 0; } button:disabled { background: #888; }
[role=alert] { background: #fce8e6; padding: 15px; color: #85271e; }
dl { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 10px; } dt, dd { margin: 0; overflow-wrap: anywhere; }
.parameter-plots { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 16px; }
table { width: 100%; border-collapse: collapse; } th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
.table-scroll { max-width: 100%; min-width: 0; overflow-x: auto; } .table-scroll table { min-width: 520px; }
</style>
