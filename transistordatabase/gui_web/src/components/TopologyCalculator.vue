<template>
  <div class="topology-calculator">
    <div class="calc-header">
      <h2>Topology Calculator</h2>
      <div class="header-actions">
        <button class="btn btn-secondary" type="button" @click="reset">Reset</button>
        <button class="btn btn-info" type="button" :disabled="!hasExportableResults" @click="saveResults">Download JSON</button>
      </div>
    </div>

    <div class="calc-content">
      <section class="topology-panel">
        <h3>Converter topology</h3>
        <div class="topology-grid">
          <button
            v-for="topology in topologies"
            :key="topology.id"
            class="topology-card"
            :class="{ active: selectedTopology === topology.id, unsupported: !topology.supported }"
            type="button"
            :aria-pressed="selectedTopology === topology.id"
            @click="selectTopology(topology.id)"
          >
            <span class="topology-name">{{ topology.name }}</span>
            <span class="topology-desc">{{ topology.description }}</span>
            <span v-if="!topology.supported" class="unsupported-label">Unsupported: isolated topology model is not implemented</span>
          </button>
        </div>
      </section>

      <section class="parameters-panel">
        <h3>Operating point</h3>
        <div class="param-grid">
          <label class="param-item">Calculation mode
            <select v-model="form.mode">
              <option value="estimate">Ideal operating-point estimate</option>
              <option value="conduction-only">TDB channel conduction only</option>
              <option value="detailed">TDB curve based losses</option>
              <option value="loss-balanced">Loss-balanced channel model</option>
            </select>
          </label>
          <label class="param-item">Conduction regime
            <select v-model="form.conduction_regime">
              <option value="CCM">Continuous conduction (CCM)</option>
              <option value="DCM">Discontinuous conduction (DCM)</option>
            </select>
          </label>
          <label v-if="selectedTopology === 'flyback'" class="param-item">Flyback turns ratio (Np/Ns)
            <input v-model.number="form.turns_ratio_nps" type="number" min="0.001" step="any">
          </label>
          <label class="param-item">Input voltage
            <input v-model.number="form.v_in" type="number" min="0.001" step="any"><span class="unit">V</span>
          </label>
          <label class="param-item">Output voltage magnitude
            <input v-model.number="form.v_out" type="number" min="0.001" step="any"><span class="unit">V</span>
          </label>
          <label class="param-item">Output power
            <input v-model.number="form.p_out" type="number" min="0.001" step="any"><span class="unit">W</span>
          </label>
          <label class="param-item">Switching frequency
            <input v-model.number="form.f_sw_khz" type="number" min="0.001" step="any"><span class="unit">kHz</span>
          </label>
          <label class="param-item">Inductance
            <input v-model.number="form.l_value_uh" type="number" min="0.001" step="any"><span class="unit">µH</span>
          </label>
          <label class="param-item">Output capacitance
            <input v-model.number="form.c_out_uf" type="number" min="0.001" step="any"><span class="unit">µF</span>
          </label>
          <label class="param-item">Parallel switches
            <input v-model.number="form.parallel_switches" type="number" min="1" step="1">
          </label>
          <label class="param-item">Parallel diodes
            <input v-model.number="form.parallel_diodes" type="number" min="1" step="1">
          </label>
        </div>

        <div class="param-section">
          <h4>Device selection</h4>
          <div class="param-grid">
            <label class="param-item">Switch transistor
              <select v-model="form.switch_transistor_id">
                <option value="">Select a TDB device</option>
                <option v-for="device in transistors" :key="deviceId(device)" :value="deviceId(device)">{{ deviceId(device) }}</option>
              </select>
            </label>
            <label class="param-item">Diode transistor
              <select v-model="form.diode_transistor_id">
                <option value="">Select a TDB device</option>
                <option v-for="device in transistors" :key="deviceId(device)" :value="deviceId(device)">{{ deviceId(device) }}</option>
              </select>
            </label>
          </div>
          <p class="hint">Estimate mode does not use device curves. Conduction-only mode requires exact-condition switch and diode channel curves. Detailed mode also requires exact-condition turn-on and turn-off curves. Loss-balanced mode requires exact-condition switch and diode channel curves. Buck and Boost support scalar DCM feedback; DCM feedback for other topologies and all curve extrapolation are rejected.</p>
        </div>

        <div v-if="form.mode !== 'estimate'" class="param-section">
          <h4>Curve and thermal conditions</h4>
          <div class="param-grid">
            <label class="param-item">Junction temperature
              <input v-model.number="form.t_j" type="number" step="any"><span class="unit">°C</span>
            </label>
            <label class="param-item">Case / sink reference temperature
              <input v-model.number="form.t_case" type="number" step="any"><span class="unit">°C</span>
            </label>
            <label class="param-item">Switch gate voltage
              <input v-model.number="form.v_g" type="number" step="any"><span class="unit">V</span>
            </label>
            <label class="param-item">Diode channel gate voltage (optional; required for gate-conditioned curves)
              <input v-model.number="form.diode_v_g" type="number" step="any"><span class="unit">V</span>
            </label>
            <label class="param-item">Turn-off gate voltage (optional)
              <input v-model.number="form.v_g_off" type="number" step="any"><span class="unit">V</span>
            </label>
            <label class="param-item">Turn-on gate resistance (optional)
              <input v-model.number="form.r_g_on" type="number" min="0" step="any"><span class="unit">Ω</span>
            </label>
            <label class="param-item">Turn-off gate resistance (optional)
              <input v-model.number="form.r_g_off" type="number" min="0" step="any"><span class="unit">Ω</span>
            </label>
          </div>
        </div>

        <div class="param-section">
          <h4>Python generated sweep</h4>
          <label class="sweep-toggle"><input v-model="form.sweep_enabled" type="checkbox"> Include line or contour sweep</label>
          <p v-if="form.mode === 'loss-balanced'" class="hint">Loss-balanced sweeps solve each CCM point from exact-condition channel curves. Every point must be supported; no curve extrapolation is used.</p>
          <p v-if="form.mode === 'loss-balanced' && form.conduction_regime !== 'CCM'" class="hint">Loss-balanced sweeps support CCM only. Buck and Boost DCM feedback are available for scalar operating points.</p>
          <div v-if="form.sweep_enabled" class="param-grid">
            <label class="param-item">Sweep type
              <select v-model="form.sweep_kind"><option value="line">Line</option><option value="contour">Contour</option></select>
            </label>
            <label class="param-item">X axis
              <select v-model="form.x_axis"><option v-for="axis in axes" :key="axis.id" :value="axis.id">{{ axis.label }}</option></select>
            </label>
            <label class="param-item">X minimum ({{ axisUnit(form.x_axis) }})
              <input v-model.number="form.x_min" type="number" min="0.001" step="any">
            </label>
            <label class="param-item">X maximum ({{ axisUnit(form.x_axis) }})
              <input v-model.number="form.x_max" type="number" min="0.001" step="any">
            </label>
            <label class="param-item">X points
              <input v-model.number="form.x_points" type="number" min="2" max="200" step="1">
            </label>
            <template v-if="form.sweep_kind === 'contour'">
              <label class="param-item">Y axis
                <select v-model="form.y_axis"><option v-for="axis in availableYAxes" :key="axis.id" :value="axis.id">{{ axis.label }}</option></select>
              </label>
              <label class="param-item">Y minimum ({{ axisUnit(form.y_axis) }})
                <input v-model.number="form.y_min" type="number" min="0.001" step="any">
              </label>
              <label class="param-item">Y maximum ({{ axisUnit(form.y_axis) }})
                <input v-model.number="form.y_max" type="number" min="0.001" step="any">
              </label>
              <label class="param-item">Y points
                <input v-model.number="form.y_points" type="number" min="2" max="200" step="1">
              </label>
            </template>
            <label class="param-item">Plotted metric
              <select v-model="form.metric"><option v-for="metric in metrics" :key="metric.id" :value="metric.id">{{ metric.label }}</option></select>
            </label>
          </div>
          <p class="hint">Contour requests are limited to 2,500 operating points. Every sweep point must be supported; invalid regions return an error instead of extrapolated values.</p>
        </div>

        <div class="calculation-actions">
          <button class="btn btn-primary" type="button" :disabled="loading" @click="calculate">{{ loading ? 'Calculating…' : 'Calculate' }}</button>
        <button class="btn btn-secondary" type="button" :disabled="loading || (form.mode === 'loss-balanced' && form.conduction_regime !== 'CCM')" @click="compare">Compare topologies</button>
        </div>
        <div v-if="errorMessage" class="error-message" role="alert">{{ errorMessage }}</div>
        <p v-if="form.mode === 'loss-balanced' && form.conduction_regime !== 'CCM'" class="hint">Topology comparison is unavailable for loss-balanced DCM. Only scalar Buck or Boost DCM feedback is supported.</p>
      </section>

      <section v-if="hasResults" class="results-panel">
        <h3>Calculation results</h3>
        <p class="model-label">{{ modelLabel }}</p>
        <p v-if="results.model.output_polarity === -1" class="hint">Inverting buck-boost output: voltage and current values below are magnitudes; signed output voltage is {{ format(results.calculated.v_out_signed_v, 2) }} V.</p>
        <ul class="assumptions"><li v-for="assumption in results.model.assumptions" :key="assumption">{{ assumption }}</li></ul>

        <div class="results-grid">
          <div class="result-item"><label>Duty cycle</label><span class="value">{{ format(results.calculated.duty_cycle, 4) }}</span></div>
          <div class="result-item"><label>{{ results.model.mode === 'estimate' ? 'Average input current' : results.model.mode === 'loss-balanced' ? 'Average input current (loss-balanced)' : 'Average input current (ideal lossless point)' }}</label><span class="value">{{ format(results.calculated.i_in_avg_a, 3) }} A</span></div>
          <div class="result-item"><label>Average output current</label><span class="value">{{ format(results.calculated.i_out_avg_a, 3) }} A</span></div>
          <div class="result-item"><label>Average inductor current</label><span class="value">{{ format(results.calculated.i_l_avg_a, 3) }} A</span></div>
          <div class="result-item"><label>Peak inductor current</label><span class="value">{{ format(results.calculated.i_l_peak_a, 3) }} A</span></div>
          <div class="result-item"><label>Inductor current ripple (p-p)</label><span class="value">{{ format(results.calculated.i_l_ripple_pp_a, 3) }} A</span></div>
          <div class="result-item"><label>Output voltage ripple (p-p)</label><span class="value">{{ format(results.calculated.v_out_ripple_pp_v, 4) }} V</span></div>
          <div class="result-item"><label>Switch blocking voltage</label><span class="value">{{ format(results.calculated.v_switch_block_v, 2) }} V</span></div>
        </div>

        <div v-if="results.losses" class="results-section">
          <h4>{{ results.model.mode === 'loss-balanced' ? 'Included device losses' : 'Partial device losses' }}</h4>
          <div class="loss-grid">
            <div class="loss-item"><label>Switch conduction</label><span class="value">{{ format(results.losses.conduction_switch_w, 3) }} W</span></div>
            <div class="loss-item"><label>Diode conduction</label><span class="value">{{ format(results.losses.conduction_diode_w, 3) }} W</span></div>
            <div v-if="results.model.mode === 'detailed'" class="loss-item"><label>Switching on</label><span class="value">{{ format(results.losses.switching_on_w, 3) }} W</span></div>
            <div v-if="results.model.mode === 'detailed'" class="loss-item"><label>Switching off</label><span class="value">{{ format(results.losses.switching_off_w, 3) }} W</span></div>
            <div class="loss-item total"><label>Included partial loss</label><span class="value">{{ format(results.losses.total_w, 3) }} W</span></div>
            <div class="loss-item"><label>{{ results.model.mode === 'loss-balanced' ? 'Loss-balanced input power' : 'Partial input power estimate' }}</label><span class="value">{{ format(results.performance.p_in_w, 3) }} W</span></div>
            <div class="loss-item"><label>{{ results.model.mode === 'loss-balanced' ? 'Loss-balanced efficiency' : 'Partial efficiency estimate' }}</label><span class="value">{{ format(results.performance.efficiency_percent, 3) }}%</span></div>
          </div>
          <p v-if="results.performance.power_balance_residual_w != null" class="hint">Power-balance residual: {{ format(results.performance.power_balance_residual_w, 8) }} W.</p>
          <p v-if="results.calculated.inductor_volt_second_residual_v != null" class="hint">Averaged inductor volt-second residual: {{ format(results.calculated.inductor_volt_second_residual_v, 8) }} V.</p>
          <p v-if="results.losses.scope" class="hint">{{ results.losses.scope }}.</p>
          <p v-if="results.performance.input_power_scope" class="hint">{{ results.performance.input_power_scope }}.</p>
          <p v-if="results.temperatures?.t_case_c != null" class="hint">
            Fixed case/sink reference {{ format(results.temperatures.t_case_c, 1) }} °C —
            switch {{ format(results.temperatures.switch_c, 1) }} °C<span v-if="results.temperatures.switch_unavailable_reason"> ({{ results.temperatures.switch_unavailable_reason }})</span>,
            diode {{ format(results.temperatures.diode_c, 1) }} °C<span v-if="results.temperatures.diode_unavailable_reason"> ({{ results.temperatures.diode_unavailable_reason }})</span>.
          </p>
          <p v-else-if="results.temperatures" class="hint">
            Temperature output unavailable: {{ results.temperatures.unavailable_reason || [results.temperatures.switch_unavailable_reason, results.temperatures.diode_unavailable_reason].filter(Boolean).join('; ') }}
          </p>
          <p v-if="results.temperatures?.thermal_closure_reason" class="hint">Electrothermal closure: {{ results.temperatures.thermal_closure_reason }}</p>
        </div>

        <div v-if="results.waveforms" class="charts-section">
          <h4>{{ results.topology === 'flyback' ? 'Primary magnetizing current over one switching period' : 'Inductor current over one switching period' }}</h4>
          <div class="chart-container"><canvas ref="waveformCanvas" role="img" :aria-label="`${results.topology === 'flyback' ? 'Primary magnetizing' : 'Inductor'} current over one switching period`"></canvas></div>
        </div>
        <div v-if="results.sweep?.kind === 'line'" class="charts-section">
          <h4>{{ results.sweep.metric }} vs {{ results.sweep.x_axis }}</h4>
          <p v-if="results.sweep.loss_scope" class="hint">Sweep scope: {{ results.sweep.loss_scope }}. {{ results.sweep.voltage_law_note }}</p>
          <p v-if="results.sweep.voltage_law_residual_values_v" class="hint">Hover a point to inspect its averaged volt-second residual on the right-hand axis.</p>
          <div class="chart-container"><canvas ref="sweepCanvas" role="img" :aria-label="`${results.sweep.metric} vs ${results.sweep.x_axis}`"></canvas></div>
        </div>
        <div v-if="results.sweep?.kind === 'contour'" class="charts-section">
          <h4>{{ results.sweep.metric }} contour</h4>
          <p v-if="results.sweep.loss_scope" class="hint">Sweep scope: {{ results.sweep.loss_scope }}. {{ results.sweep.voltage_law_note }}</p>
          <div class="contour-scroll"><div class="contour-grid" role="img" :aria-label="`${results.sweep.metric} contour plot`" :style="{ gridTemplateColumns: `repeat(${results.sweep.x_values.length}, minmax(18px, 1fr))` }">
            <span v-for="cell in contourCells" :key="cell.key" :title="cell.title" :style="{ backgroundColor: cell.color }"></span>
          </div></div>
          <p class="hint">X: {{ results.sweep.x_axis }} ({{ results.sweep.x_unit }}); Y: {{ results.sweep.y_axis }} ({{ results.sweep.y_unit }}); color shows {{ results.sweep.metric }} ({{ results.sweep.metric_unit }}).</p>
        </div>
        <details v-if="results.model.curve_conditions"><summary>Curve conditions and provenance</summary><pre>{{ JSON.stringify(results.model.curve_conditions, null, 2) }}</pre></details>
      </section>

      <section v-if="comparison" class="results-panel">
        <h3>Topology comparison</h3>
        <p v-if="comparison.comparison?.reason" class="hint">{{ comparison.comparison.reason }}</p>
        <p v-if="comparison.comparison?.loss_scope" class="hint">
          Ranking uses {{ comparison.comparison.loss_scope }} losses only.
        </p>
        <div v-for="name in ['buck', 'boost', 'buck-boost', 'flyback']" :key="name" class="comparison-row">
          <div class="comparison-summary">
            <strong>{{ name }}</strong>
            <span v-if="comparison[name]?.error" class="error-text">{{ comparison[name].error }}</span>
            <span v-else-if="comparison[name]?.supported === false" class="hint">{{ comparison[name]?.reason || 'Unsupported operating point.' }}</span>
            <span v-else-if="!comparison[name]" class="hint">No operating point was returned.</span>
            <span v-else-if="comparison[name]?.performance?.efficiency_percent == null" class="hint">
              {{ comparison[name]?.model?.mode === 'estimate' ? 'Estimate mode; efficiency is not calculated.' : 'Efficiency is not reported for this mode.' }}
            </span>
            <span v-else>{{ format(comparison[name].performance.efficiency_percent, 3) }}%</span>
          </div>
          <div v-if="isLossBalancedCcmComparison(comparison[name])" class="comparison-details">
            <p><strong>Power-balance residual:</strong> {{ format(comparison[name].performance?.power_balance_residual_w, 8) }} W</p>
            <p><strong>Averaged inductor volt-second residual:</strong> {{ format(comparison[name].calculated?.inductor_volt_second_residual_v, 8) }} V</p>
            <div v-if="comparison[name].temperatures" class="comparison-thermal">
              <p><strong>Thermal status:</strong> {{ comparison[name].model?.thermal_closure || comparison[name].temperatures.thermal_closure || 'not reported' }}</p>
              <p v-if="comparison[name].temperatures.t_case_c != null">Fixed case/sink reference: {{ format(comparison[name].temperatures.t_case_c, 1) }} °C.</p>
              <p v-if="comparison[name].temperatures.switch_c != null">Switch junction estimate: {{ format(comparison[name].temperatures.switch_c, 1) }} °C.</p>
              <p v-else>Switch temperature unavailable: {{ comparison[name].temperatures.switch_unavailable_reason || comparison[name].temperatures.unavailable_reason || 'Reason not reported.' }}</p>
              <p v-if="comparison[name].temperatures.diode_c != null">Diode junction estimate: {{ format(comparison[name].temperatures.diode_c, 1) }} °C.</p>
              <p v-else>Diode temperature unavailable: {{ comparison[name].temperatures.diode_unavailable_reason || comparison[name].temperatures.unavailable_reason || 'Reason not reported.' }}</p>
              <p v-if="comparison[name].temperatures.thermal_closure_reason">Thermal status reason: {{ comparison[name].temperatures.thermal_closure_reason }}</p>
            </div>
            <p v-else>Thermal data not reported.</p>
            <div class="comparison-limitations">
              <strong>Model assumptions and limitations</strong>
              <ul v-if="lossBalancedAssumptions(comparison[name]).length">
                <li v-for="assumption in lossBalancedAssumptions(comparison[name])" :key="assumption">{{ assumption }}</li>
              </ul>
              <p v-else>No model assumptions or limitations were reported in this response.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, reactive, ref, watch } from 'vue'
import Chart from 'chart.js/auto'
import { transistorApi } from '../services/api.js'

const props = defineProps({ transistors: { type: Array, default: () => [] } })
const emit = defineEmits(['transistor-selected'])

const topologies = [
  { id: 'buck', name: 'Buck', description: 'Step-down, non-isolated PWM converter', supported: true },
  { id: 'boost', name: 'Boost', description: 'Step-up, non-isolated PWM converter', supported: true },
  { id: 'buck-boost', name: 'Inverting buck-boost', description: 'Inverting, non-isolated PWM converter', supported: true },
  { id: 'flyback', name: 'Flyback', description: 'Isolated flyback with ideal transformer ratio', supported: true }
]
const axes = [
  { id: 'v_in', label: 'Input voltage' },
  { id: 'v_out', label: 'Output voltage magnitude' },
  { id: 'p_out', label: 'Output power' },
  { id: 'f_sw', label: 'Switching frequency' },
  { id: 'l_value', label: 'Inductance' },
  { id: 'c_out', label: 'Output capacitance' }
]
const metrics = [
  { id: 'duty_cycle', label: 'Duty cycle' },
  { id: 'i_in_avg_a', label: 'Average input current' },
  { id: 'i_out_avg_a', label: 'Average output current' },
  { id: 'i_l_peak_a', label: 'Peak inductor current' },
  { id: 'i_l_ripple_pp_a', label: 'Inductor current ripple' },
  { id: 'v_out_ripple_pp_v', label: 'Output voltage ripple' },
  { id: 'p_loss_total_w', label: 'Included partial device losses (curve modes)' },
  { id: 'efficiency_percent', label: 'Efficiency from included partial losses (curve modes)' },
  { id: 't_switch_c', label: 'Switch junction temperature (when thermal data is available)' },
  { id: 't_diode_c', label: 'Diode junction temperature (when thermal data is available)' }
]
const form = reactive({
  mode: 'estimate', conduction_regime: 'CCM', turns_ratio_nps: 1,
  v_in: 400, v_out: 48, p_out: 1000, f_sw_khz: 100,
  l_value_uh: 100, c_out_uf: 1000, parallel_switches: 1, parallel_diodes: 1,
  switch_transistor_id: '', diode_transistor_id: '', t_j: 25, t_case: 85,
  v_g: 15, diode_v_g: '', v_g_off: -5, r_g_on: 10, r_g_off: 10,
  sweep_enabled: false, sweep_kind: 'line', x_axis: 'f_sw', x_min: 50,
  x_max: 200, x_points: 31, y_axis: 'v_in', y_min: 300, y_max: 500,
  y_points: 21, metric: 'i_l_ripple_pp_a'
})
const selectedTopology = ref('buck')
const results = ref(null)
const comparison = ref(null)
const comparisonRequest = ref(null)
const loading = ref(false)
const errorMessage = ref('')
const waveformCanvas = ref(null)
const sweepCanvas = ref(null)
let waveformChart = null
let sweepChart = null
let requestSequence = 0

const transistors = computed(() => props.transistors || [])
const hasResults = computed(() => results.value !== null)
const hasExportableResults = computed(() => hasResults.value || comparison.value !== null)
const modelLabel = computed(() => {
  const mode = results.value?.model?.mode
  if (mode === 'loss-balanced') {
    const regime = results.value?.model?.conduction_regime || 'CCM'
    const topology = ['buck', 'boost'].includes(results.value?.topology) && regime === 'DCM'
      ? `${results.value.topology === 'buck' ? 'Buck' : 'Boost'} DCM`
      : regime
    return `${topology} TDB channel loss-balanced averaged model`
  }
  if (mode === 'detailed') return 'TDB curve based partial device model'
  if (mode === 'conduction-only') return 'TDB channel conduction-only partial model'
  return `Ideal ${results.value?.model?.conduction_regime || 'CCM'} operating-point estimate`
})
const availableYAxes = computed(() => axes.filter(axis => axis.id !== form.x_axis))
const contourCells = computed(() => {
  const sweep = results.value?.sweep
  if (!sweep || sweep.kind !== 'contour') return []
  const flat = sweep.values.flat()
  const minimum = Math.min(...flat)
  const maximum = Math.max(...flat)
  return sweep.values.flatMap((row, yIndex) => row.map((value, xIndex) => {
    const fraction = maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum)
    const hue = 220 - fraction * 220
    return {
      key: `${yIndex}-${xIndex}`,
      color: `hsl(${hue}, 78%, 53%)`,
      title: `x=${sweep.x_values[xIndex]} ${sweep.x_unit}; y=${sweep.y_values[yIndex]} ${sweep.y_unit}; ${sweep.metric}=${value} ${sweep.metric_unit}${sweep.voltage_law_residual_values_v ? `; inductor_volt_second_residual=${sweep.voltage_law_residual_values_v[yIndex][xIndex]} V` : ''}`
    }
  }))
})

function deviceId(device) {
  return device?.id || device?.metadata?.name || device?.name || ''
}
function selectTopology(id) {
  selectedTopology.value = id
  errorMessage.value = ''
}
function format(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : '—'
}
function isLossBalancedCcmComparison(result) {
  return result?.model?.mode === 'loss-balanced'
    && result.model.conduction_regime === 'CCM'
    && !result.error
    && result.supported !== false
}
function lossBalancedAssumptions(result) {
  const assumptions = result?.model?.assumptions
  if (!Array.isArray(assumptions)) return []
  return assumptions.filter(assumption => (
    typeof assumption === 'string' && assumption.trim().length > 0
  ))
}
function axisUnit(axis) {
  return ({ v_in: 'V', v_out: 'V', p_out: 'W', f_sw: 'kHz', l_value: 'µH', c_out: 'µF' })[axis]
}
function toSi(axis, value) {
  const scale = { f_sw: 1000, l_value: 1e-6, c_out: 1e-6 }[axis] || 1
  return Number(value) * scale
}
function optionalNumber(value) {
  if (value == null || (typeof value === 'string' && value.trim() === '')) return undefined
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}
function buildSweep() {
  if (!form.sweep_enabled) return undefined
  const sweep = {
    x_axis: form.x_axis, x_min: toSi(form.x_axis, form.x_min),
    x_max: toSi(form.x_axis, form.x_max), x_points: Number(form.x_points), metric: form.metric
  }
  if (form.sweep_kind === 'contour') {
    Object.assign(sweep, {
      y_axis: form.y_axis, y_min: toSi(form.y_axis, form.y_min),
      y_max: toSi(form.y_axis, form.y_max), y_points: Number(form.y_points)
    })
  }
  return sweep
}
function buildPayload(includeSweep = true) {
  return {
    topology: selectedTopology.value,
    mode: form.mode,
    conduction_regime: form.conduction_regime,
    ...(selectedTopology.value === 'flyback' ? { turns_ratio_nps: Number(form.turns_ratio_nps) } : {}),
    v_in: Number(form.v_in), v_out: Number(form.v_out), p_out: Number(form.p_out),
    f_sw: toSi('f_sw', form.f_sw_khz),
    l_value: toSi('l_value', form.l_value_uh), c_out: toSi('c_out', form.c_out_uf),
    parallel_switches: Number(form.parallel_switches), parallel_diodes: Number(form.parallel_diodes),
    switch_transistor_id: form.switch_transistor_id || undefined,
    diode_transistor_id: form.diode_transistor_id || undefined,
    t_j: form.mode !== 'estimate' ? Number(form.t_j) : undefined,
    t_case: form.mode !== 'estimate' ? optionalNumber(form.t_case) : undefined,
    v_g: form.mode !== 'estimate' ? Number(form.v_g) : undefined,
    diode_v_g: form.mode !== 'estimate' ? optionalNumber(form.diode_v_g) : undefined,
    v_g_off: form.mode === 'detailed' ? optionalNumber(form.v_g_off) : undefined,
    r_g_on: form.mode === 'detailed' ? optionalNumber(form.r_g_on) : undefined,
    r_g_off: form.mode === 'detailed' ? optionalNumber(form.r_g_off) : undefined,
    ...(includeSweep && buildSweep() ? { sweep: buildSweep() } : {})
  }
}
async function calculate() {
  const sequence = ++requestSequence
  loading.value = true
  errorMessage.value = ''
  try {
    const payload = buildPayload()
    const response = await transistorApi.calculateTopology(payload)
    if (sequence !== requestSequence) return
    results.value = { request: payload, ...response }
    comparison.value = null
    comparisonRequest.value = null
    await renderCharts()
  } catch (error) {
    if (sequence !== requestSequence) return
    results.value = null
    errorMessage.value = error.response?.data?.detail || error.message || 'Topology calculation failed'
  } finally {
    if (sequence === requestSequence) loading.value = false
  }
}
async function compare() {
  const sequence = ++requestSequence
  loading.value = true
  errorMessage.value = ''
  try {
    const payload = buildPayload(false)
    const { topology: ignoredTopology, ...parameters } = payload
    const request = {
      parameters,
      switch_transistor_id: payload.switch_transistor_id,
      diode_transistor_id: payload.diode_transistor_id
    }
    const response = await transistorApi.compareTopologies(request)
    if (sequence !== requestSequence) return
    comparison.value = response
    comparisonRequest.value = request
  } catch (error) {
    if (sequence === requestSequence) errorMessage.value = error.response?.data?.detail || error.message || 'Topology comparison failed'
  } finally {
    if (sequence === requestSequence) loading.value = false
  }
}
function destroyCharts() {
  waveformChart?.destroy()
  sweepChart?.destroy()
  waveformChart = null
  sweepChart = null
}
async function renderCharts() {
  destroyCharts()
  await nextTick()
  if (results.value?.waveforms && waveformCanvas.value) {
    waveformChart = new Chart(waveformCanvas.value, {
      type: 'line',
      data: { labels: results.value.waveforms.time_s, datasets: [{
        label: 'Inductor current (A)', data: results.value.waveforms.inductor_current_a,
        borderColor: '#0d6efd', pointRadius: 0, borderWidth: 2
      }] },
      options: { responsive: true, plugins: { legend: { display: true } }, scales: {
        x: { title: { display: true, text: 'Time (s)' } },
        y: { title: { display: true, text: 'Current (A)' } }
      } }
    })
  }
  if (results.value?.sweep?.kind === 'line' && sweepCanvas.value) {
    const sweep = results.value.sweep
    const datasets = [{
      label: `${sweep.metric} (${sweep.metric_unit})`, data: sweep.values,
      borderColor: '#198754', pointRadius: 2, borderWidth: 2
    }]
    const scales = {
      x: { title: { display: true, text: `${sweep.x_axis} (${sweep.x_unit})` } },
      y: { title: { display: true, text: `${sweep.metric} (${sweep.metric_unit})` } }
    }
    if (sweep.voltage_law_residual_values_v) {
      datasets.push({
        label: 'Inductor volt-second residual (V)',
        data: sweep.voltage_law_residual_values_v,
        borderColor: '#dc3545', pointRadius: 2, borderWidth: 2,
        yAxisID: 'residual'
      })
      scales.residual = {
        type: 'linear', position: 'right',
        title: { display: true, text: 'Volt-second residual (V)' },
        grid: { drawOnChartArea: false }
      }
    }
    sweepChart = new Chart(sweepCanvas.value, {
      type: 'line',
      data: { labels: sweep.x_values, datasets },
      options: { responsive: true, plugins: { legend: { display: true } }, scales }
    })
  }
}
function saveResults() {
  if (!hasExportableResults.value) return
  const exportData = {
    topology: results.value?.topology ?? null,
    request: results.value?.request ?? comparisonRequest.value,
    result: results.value,
    comparison: comparison.value,
    comparison_request: comparisonRequest.value,
    exported_at: new Date().toISOString()
  }
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  const exportName = results.value?.topology || 'comparison'
  link.download = `topology-${exportName}-${new Date().toISOString().slice(0, 10)}.json`
  link.click()
  URL.revokeObjectURL(url)
}
function reset() {
  requestSequence += 1
  selectedTopology.value = 'buck'
  Object.assign(form, {
    mode: 'estimate', conduction_regime: 'CCM', turns_ratio_nps: 1,
    v_in: 400, v_out: 48, p_out: 1000, f_sw_khz: 100,
    l_value_uh: 100, c_out_uf: 1000, parallel_switches: 1, parallel_diodes: 1,
    switch_transistor_id: '', diode_transistor_id: '', t_j: 25, t_case: 85,
    v_g: 15, diode_v_g: '', v_g_off: -5, r_g_on: 10, r_g_off: 10,
    sweep_enabled: false, sweep_kind: 'line', x_axis: 'f_sw', x_min: 50,
    x_max: 200, x_points: 31, y_axis: 'v_in', y_min: 300, y_max: 500,
    y_points: 21, metric: 'i_l_ripple_pp_a'
  })
  results.value = null
  comparison.value = null
  comparisonRequest.value = null
  errorMessage.value = ''
  loading.value = false
  destroyCharts()
}
function invalidateResults() {
  requestSequence += 1
  results.value = null
  comparison.value = null
  comparisonRequest.value = null
  errorMessage.value = ''
  loading.value = false
  destroyCharts()
}
watch([form, selectedTopology], invalidateResults, { deep: true })
watch(() => form.x_axis, axis => {
  if (form.y_axis === axis) form.y_axis = axes.find(item => item.id !== axis)?.id || 'v_in'
})
onBeforeUnmount(destroyCharts)
</script>

<style scoped>
.topology-calculator {
  max-width: 1600px;
  margin: 0 auto;
  padding: 1rem;
}

.calc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding: 1rem;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header-actions {
  display: flex;
  gap: 1rem;
}

.calc-content {
  display: grid;
  grid-template-columns: 350px 400px 1fr;
  gap: 2rem;
}

.topology-panel,
.parameters-panel,
.results-panel {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  height: fit-content;
}

.topology-panel h3,
.parameters-panel h3,
.results-panel h3 {
  margin: 0 0 1rem 0;
  color: #2c3e50;
  border-bottom: 2px solid #3498db;
  padding-bottom: 0.5rem;
}

.comparison-row {
  padding: 0.75rem 0;
  border-bottom: 1px solid #e5e7eb;
}

.comparison-row:last-child {
  border-bottom: 0;
}

.comparison-summary {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 0.5rem 1rem;
}

.comparison-details {
  margin: 0.75rem 0 0 1rem;
  padding-left: 0.75rem;
  border-left: 2px solid #dbeafe;
  color: #4b5563;
  font-size: 0.9rem;
}

.comparison-details p {
  margin: 0.35rem 0;
}

.comparison-limitations {
  margin-top: 0.75rem;
}

.comparison-limitations ul {
  margin: 0.35rem 0 0;
  padding-left: 1.25rem;
}

.topology-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 2rem;
}

.topology-card {
  padding: 1rem;
  border: 2px solid #ddd;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  text-align: center;
}

.topology-card:hover {
  border-color: #007bff;
  transform: translateY(-2px);
}

.topology-card.active {
  border-color: #007bff;
  background: #f8f9ff;
}

.topology-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.topology-name {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.topology-desc {
  font-size: 0.8rem;
  color: #666;
}

.topology-info {
  margin-top: 1rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 4px;
}

.topology-info h4 {
  margin: 0 0 1rem 0;
  color: #2c3e50;
}

.circuit-svg {
  text-align: center;
}

.circuit-svg svg {
  width: 100%;
  max-width: 300px;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.parameter-sections {
  margin-bottom: 2rem;
}

.param-section {
  margin-bottom: 2rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 4px;
}

.param-section h4 {
  margin: 0 0 1rem 0;
  color: #2c3e50;
  font-size: 1rem;
}

.param-grid {
  display: grid;
  gap: 1rem;
}

.param-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.param-item label {
  min-width: 140px;
  font-size: 0.9rem;
  color: #555;
}

.param-item input {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.9rem;
}

.param-item .unit {
  min-width: 30px;
  font-size: 0.8rem;
  color: #666;
  font-weight: 500;
}

.calculation-actions {
  display: flex;
  gap: 1rem;
}

.results-panel {
  max-height: 80vh;
  overflow-y: auto;
}

.no-results {
  text-align: center;
  color: #999;
  font-style: italic;
  padding: 2rem;
}

.results-content {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.results-section {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 4px;
}

.results-section h4 {
  margin: 0 0 1rem 0;
  color: #2c3e50;
  font-size: 1rem;
}

.results-grid,
.requirements-grid,
.loss-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.result-item,
.req-item,
.loss-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: white;
  border-radius: 4px;
  border-left: 3px solid #007bff;
}

.loss-item.total {
  grid-column: 1 / -1;
  border-left-color: #28a745;
  font-weight: 600;
}

.result-item label,
.req-item label,
.loss-item label {
  font-size: 0.9rem;
  color: #555;
}

.value {
  font-weight: 600;
  color: #2c3e50;
}

.note,
.percent {
  font-size: 0.8rem;
  color: #666;
  margin-left: 0.5rem;
}

.no-recommendations {
  text-align: center;
  color: #999;
  font-style: italic;
  padding: 1rem;
}

.recommendations-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.recommendation-item {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  transition: all 0.3s;
}

.recommendation-item.best {
  border-color: #ffc107;
  background: #fff8e1;
}

.recommendation-item:hover {
  border-color: #007bff;
  transform: translateY(-1px);
}

.rec-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.rec-rank {
  font-size: 1.2rem;
}

.rec-score {
  margin-left: auto;
  font-size: 0.8rem;
  color: #666;
  background: #e9ecef;
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
}

.rec-details {
  display: flex;
  gap: 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  color: #666;
}

.rec-actions {
  display: flex;
  gap: 0.5rem;
}

.charts-section {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 4px;
}

.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.chart-container {
  background: white;
  padding: 1rem;
  border-radius: 4px;
  height: 300px;
}

.chart-container canvas {
  max-height: 100%;
}

.contour-scroll {
  overflow-x: auto;
}

.contour-grid {
  display: grid;
  grid-auto-rows: 24px;
  gap: 2px;
}

.contour-grid span {
  display: block;
  min-height: 24px;
}

.btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s;
  text-decoration: none;
  display: inline-block;
  font-size: 0.9rem;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #007bff;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #0056b3;
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover:not(:disabled) {
  background: #545b62;
}

.btn-info {
  background: #17a2b8;
  color: white;
}

.btn-info:hover:not(:disabled) {
  background: #138496;
}

.btn-success {
  background: #28a745;
  color: white;
}

.btn-success:hover:not(:disabled) {
  background: #1e7e34;
}

.btn-sm {
  padding: 0.25rem 0.5rem;
  font-size: 0.8rem;
}

@media (max-width: 1400px) {
  .calc-content {
    grid-template-columns: 1fr;
    gap: 1rem;
  }
}

@media (max-width: 768px) {
  .calc-header {
    flex-direction: column;
    gap: 1rem;
  }
  
  .topology-grid {
    grid-template-columns: 1fr;
  }
  
  .results-grid,
  .requirements-grid,
  .loss-grid {
    grid-template-columns: 1fr;
  }
  
  .charts-grid {
    grid-template-columns: 1fr;
  }
  
  .calculation-actions {
    flex-direction: column;
  }
}
</style>
