<template>
  <div class="analytical-calculator">
    <h2>Analytical Loss Models</h2>
    <p class="description">
      Physics-based switching loss and timing calculations when empirical data is unavailable.
    </p>

    <!-- Model Selector -->
    <div class="model-tabs">
      <button
        :class="{ active: activeModel === 'biela' }"
        @click="activeModel = 'biela'"
        class="tab-btn"
      >
        Biela Switching Loss Model
      </button>
      <button
        :class="{ active: activeModel === 'gate_charge' }"
        @click="activeModel = 'gate_charge'"
        class="tab-btn"
      >
        Gate Charge Timing Model
      </button>
    </div>

    <!-- Biela Model -->
    <div v-if="activeModel === 'biela'" class="model-panel">
      <h3>Biela Inductive Switching Loss Model</h3>
      <p class="model-description">
        Analytical model for turn-on and turn-off energy based on device capacitances,
        gate charge, and circuit conditions. Suitable for hard-switched converters.
      </p>

      <div class="form-grid">
        <div class="form-section">
          <h4>Device Parameters</h4>
          <div class="form-row">
            <label>Output Capacitance (C_oss):</label>
            <input v-model.number="bielaParams.c_oss" type="number" step="1e-12" />
            <span class="unit">F</span>
          </div>
          <div class="form-row">
            <label>Total Gate Charge (Q_g):</label>
            <input v-model.number="bielaParams.q_g" type="number" step="1e-9" />
            <span class="unit">C</span>
          </div>
          <div class="form-row">
            <label>Plateau Voltage (V_plateau):</label>
            <input v-model.number="bielaParams.v_plateau" type="number" step="0.1" />
            <span class="unit">V</span>
          </div>
          <div class="form-row">
            <label>R_g Turn-On:</label>
            <input v-model.number="bielaParams.r_g_on" type="number" step="0.5" />
            <span class="unit">Ω</span>
          </div>
          <div class="form-row">
            <label>R_g Turn-Off:</label>
            <input v-model.number="bielaParams.r_g_off" type="number" step="0.5" />
            <span class="unit">Ω</span>
          </div>
        </div>

        <div class="form-section">
          <h4>Operating Conditions</h4>
          <div class="form-row">
            <label>DC Bus Voltage (V_dc):</label>
            <input v-model.number="bielaParams.v_dc" type="number" step="10" />
            <span class="unit">V</span>
          </div>
          <div class="form-row">
            <label>Load Current (I_load):</label>
            <input v-model.number="bielaParams.i_load" type="number" step="1" />
            <span class="unit">A</span>
          </div>
          <div class="form-row">
            <label>Junction Temperature (T_j):</label>
            <input v-model.number="bielaParams.t_j" type="number" step="5" />
            <span class="unit">°C</span>
          </div>
        </div>

        <div class="form-section">
          <h4>Curve Generation (Optional)</h4>
          <div class="form-row">
            <label>Min Current:</label>
            <input v-model.number="bielaParams.i_min" type="number" step="1" />
            <span class="unit">A</span>
          </div>
          <div class="form-row">
            <label>Max Current:</label>
            <input v-model.number="bielaParams.i_max" type="number" step="1" />
            <span class="unit">A</span>
          </div>
          <div class="form-row">
            <label>Points:</label>
            <input v-model.number="bielaParams.i_points" type="number" step="10" min="10" max="1000" />
          </div>
        </div>
      </div>

      <button @click="calculateBiela" :disabled="calculating" class="btn-calculate">
        {{ calculating ? 'Calculating...' : 'Calculate Biela Model' }}
      </button>

      <!-- Biela Results -->
      <div v-if="bielaResults" class="results-panel">
        <h4>Calculation Results</h4>
        <div class="results-grid">
          <div class="result-card">
            <div class="result-label">Turn-On Energy (E_on)</div>
            <div class="result-value">{{ (bielaResults.results.e_on * 1e6).toFixed(3) }} µJ</div>
          </div>
          <div class="result-card">
            <div class="result-label">Turn-Off Energy (E_off)</div>
            <div class="result-value">{{ (bielaResults.results.e_off * 1e6).toFixed(3) }} µJ</div>
          </div>
          <div class="result-card">
            <div class="result-label">Total Energy (E_total)</div>
            <div class="result-value">{{ (bielaResults.results.e_total * 1e6).toFixed(3) }} µJ</div>
          </div>
        </div>

        <!-- Curve Plot -->
        <div v-if="bielaResults.curve" class="curve-plot">
          <h4>Switching Loss Curves</h4>
          <CurvePlotter
            :datasets="bielaLossCurves"
            x-label="Current (A)"
            y-label="Energy (µJ)"
            :show-legend="true"
          />
        </div>
      </div>
    </div>

    <!-- Gate Charge Model -->
    <div v-if="activeModel === 'gate_charge'" class="model-panel">
      <h3>Gate Charge Switching Time Model</h3>
      <p class="model-description">
        Estimates switching transition times from gate charge characteristics.
        Useful for predicting di/dt and dv/dt during switching.
      </p>

      <div class="form-grid">
        <div class="form-section">
          <h4>Gate Charge Parameters</h4>
          <div class="form-row">
            <label>Gate-Source Charge (Q_gs):</label>
            <input v-model.number="gateChargeParams.q_gs" type="number" step="1e-9" />
            <span class="unit">C</span>
          </div>
          <div class="form-row">
            <label>Gate-Drain Charge (Q_gd):</label>
            <input v-model.number="gateChargeParams.q_gd" type="number" step="1e-9" />
            <span class="unit">C</span>
          </div>
          <div class="form-row">
            <label>Total Gate Charge (Q_g):</label>
            <input v-model.number="gateChargeParams.q_g" type="number" step="1e-9" />
            <span class="unit">C</span>
          </div>
          <div class="form-row">
            <label>Threshold Voltage (V_th):</label>
            <input v-model.number="gateChargeParams.v_th" type="number" step="0.1" />
            <span class="unit">V</span>
          </div>
          <div class="form-row">
            <label>Plateau Voltage (V_plateau):</label>
            <input v-model.number="gateChargeParams.v_plateau" type="number" step="0.1" />
            <span class="unit">V</span>
          </div>
        </div>

        <div class="form-section">
          <h4>Gate Driver & Resistance</h4>
          <div class="form-row">
            <label>External R_g:</label>
            <input v-model.number="gateChargeParams.r_g" type="number" step="0.5" />
            <span class="unit">Ω</span>
          </div>
          <div class="form-row">
            <label>Internal R_g:</label>
            <input v-model.number="gateChargeParams.r_g_int" type="number" step="0.1" />
            <span class="unit">Ω</span>
          </div>
          <div class="form-row">
            <label>Driver Voltage (V_driver):</label>
            <input v-model.number="gateChargeParams.v_driver" type="number" step="1" />
            <span class="unit">V</span>
          </div>
          <div class="form-row">
            <label>Off-State Voltage (V_off):</label>
            <input v-model.number="gateChargeParams.v_off" type="number" step="1" />
            <span class="unit">V</span>
          </div>
        </div>
      </div>

      <button @click="calculateGateCharge" :disabled="calculating" class="btn-calculate">
        {{ calculating ? 'Calculating...' : 'Calculate Switching Times' }}
      </button>

      <!-- Gate Charge Results -->
      <div v-if="gateChargeResults" class="results-panel">
        <h4>Switching Time Results</h4>

        <div class="timing-section">
          <h5>Turn-On Timing</h5>
          <div class="results-grid">
            <div class="result-card">
              <div class="result-label">Delay Time (t_d)</div>
              <div class="result-value">{{ gateChargeResults.turn_on_times.t_delay.toFixed(2) }} ns</div>
            </div>
            <div class="result-card">
              <div class="result-label">Current Rise (t_ri)</div>
              <div class="result-value">{{ gateChargeResults.turn_on_times.t_rise.toFixed(2) }} ns</div>
            </div>
            <div class="result-card">
              <div class="result-label">Voltage Fall (t_fv)</div>
              <div class="result-value">{{ gateChargeResults.turn_on_times.t_fall_v.toFixed(2) }} ns</div>
            </div>
            <div class="result-card highlight">
              <div class="result-label">Total Turn-On</div>
              <div class="result-value">{{ gateChargeResults.turn_on_times.t_total.toFixed(2) }} ns</div>
            </div>
          </div>
        </div>

        <div class="timing-section">
          <h5>Turn-Off Timing</h5>
          <div class="results-grid">
            <div class="result-card">
              <div class="result-label">Delay Time (t_d)</div>
              <div class="result-value">{{ gateChargeResults.turn_off_times.t_delay.toFixed(2) }} ns</div>
            </div>
            <div class="result-card">
              <div class="result-label">Voltage Rise (t_rv)</div>
              <div class="result-value">{{ gateChargeResults.turn_off_times.t_rise_v.toFixed(2) }} ns</div>
            </div>
            <div class="result-card">
              <div class="result-label">Current Fall (t_fi)</div>
              <div class="result-value">{{ gateChargeResults.turn_off_times.t_fall_i.toFixed(2) }} ns</div>
            </div>
            <div class="result-card highlight">
              <div class="result-label">Total Turn-Off</div>
              <div class="result-value">{{ gateChargeResults.turn_off_times.t_total.toFixed(2) }} ns</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Error Display -->
    <div v-if="error" class="error-message">
      <strong>Error:</strong> {{ error }}
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { transistorApi } from '../services/api'
import CurvePlotter from './CurvePlotter.vue'

export default {
  name: 'AnalyticalCalculator',
  components: { CurvePlotter },
  setup() {
    const activeModel = ref('biela')
    const calculating = ref(false)
    const error = ref(null)

    // Biela Model Parameters
    const bielaParams = ref({
      c_oss: 150e-12,  // 150 pF
      q_g: 50e-9,      // 50 nC
      v_plateau: 5.0,
      r_g_on: 10.0,
      r_g_off: 10.0,
      v_driver_on: 15.0,
      v_driver_off: -5.0,
      v_dc: 400.0,
      i_load: 20.0,
      t_j: 25.0,
      i_min: 0,
      i_max: 50,
      i_points: 50
    })
    const bielaResults = ref(null)

    // Gate Charge Model Parameters
    const gateChargeParams = ref({
      q_gs: 10e-9,     // 10 nC
      q_gd: 15e-9,     // 15 nC
      q_g: 50e-9,      // 50 nC
      v_th: 2.5,
      v_plateau: 5.0,
      r_g: 10.0,
      r_g_int: 1.0,
      v_driver: 15.0,
      v_off: 0.0
    })
    const gateChargeResults = ref(null)

    // Computed curve data for Biela results
    const bielaLossCurves = computed(() => {
      if (!bielaResults.value?.curve) return []

      const curve = bielaResults.value.curve
      return [
        {
          label: 'E_on',
          data: curve.currents.map((i, idx) => ({
            x: i,
            y: curve.e_on[idx] * 1e6  // Convert to µJ
          })),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
        },
        {
          label: 'E_off',
          data: curve.currents.map((i, idx) => ({
            x: i,
            y: curve.e_off[idx] * 1e6
          })),
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
        },
        {
          label: 'E_total',
          data: curve.currents.map((i, idx) => ({
            x: i,
            y: curve.e_total[idx] * 1e6
          })),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
        }
      ]
    })

    async function calculateBiela() {
      calculating.value = true
      error.value = null

      try {
        const result = await transistorApi.calculateBiela(bielaParams.value)
        bielaResults.value = result
      } catch (err) {
        error.value = err.message || 'Calculation failed'
      } finally {
        calculating.value = false
      }
    }

    async function calculateGateCharge() {
      calculating.value = true
      error.value = null

      try {
        const result = await transistorApi.calculateGateCharge(gateChargeParams.value)
        gateChargeResults.value = result
      } catch (err) {
        error.value = err.message || 'Calculation failed'
      } finally {
        calculating.value = false
      }
    }

    return {
      activeModel,
      calculating,
      error,
      bielaParams,
      bielaResults,
      bielaLossCurves,
      gateChargeParams,
      gateChargeResults,
      calculateBiela,
      calculateGateCharge
    }
  }
}
</script>

<style scoped>
.analytical-calculator {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

h2 {
  font-size: 24px;
  margin-bottom: 10px;
}

.description {
  color: #666;
  margin-bottom: 30px;
}

/* Model Tabs */
.model-tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 30px;
  border-bottom: 2px solid #e0e0e0;
}

.tab-btn {
  padding: 12px 24px;
  background: none;
  border: none;
  border-bottom: 3px solid transparent;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s;
}

.tab-btn.active {
  border-bottom-color: #4CAF50;
  font-weight: bold;
}

.tab-btn:hover {
  background: #f0f0f0;
}

/* Model Panel */
.model-panel {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.model-panel h3 {
  font-size: 20px;
  margin-bottom: 10px;
}

.model-description {
  color: #666;
  margin-bottom: 30px;
}

/* Form Grid */
.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 30px;
  margin-bottom: 30px;
}

.form-section h4 {
  font-size: 16px;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #e0e0e0;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 120px 40px;
  gap: 10px;
  align-items: center;
  margin-bottom: 15px;
}

.form-row label {
  font-size: 14px;
}

.form-row input {
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

.form-row .unit {
  font-size: 14px;
  color: #666;
}

/* Calculate Button */
.btn-calculate {
  width: 100%;
  padding: 15px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-calculate:hover:not(:disabled) {
  background: #45a049;
}

.btn-calculate:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* Results Panel */
.results-panel {
  margin-top: 40px;
  padding-top: 30px;
  border-top: 2px solid #e0e0e0;
}

.results-panel h4 {
  font-size: 18px;
  margin-bottom: 20px;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.result-card {
  background: #f9f9f9;
  padding: 20px;
  border-radius: 8px;
  text-align: center;
}

.result-card.highlight {
  background: #e3f2fd;
  border: 2px solid #2196F3;
}

.result-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 10px;
}

.result-value {
  font-size: 24px;
  font-weight: bold;
  color: #333;
}

.timing-section {
  margin-bottom: 30px;
}

.timing-section h5 {
  font-size: 16px;
  margin-bottom: 15px;
  color: #555;
}

.curve-plot {
  margin-top: 30px;
}

.curve-plot h4 {
  font-size: 16px;
  margin-bottom: 15px;
}

/* Error Message */
.error-message {
  margin-top: 20px;
  padding: 15px;
  background: #f8d7da;
  border-left: 4px solid #f44336;
  border-radius: 4px;
  color: #721c24;
}
</style>
