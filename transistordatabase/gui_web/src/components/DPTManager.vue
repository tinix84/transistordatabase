<template>
  <div class="dpt-manager">
    <h2>Double Pulse Test (DPT) Manager</h2>
    <p class="description">
      Generate LTspice netlists for double pulse testing and validate measurement data
      against datasheet specifications.
    </p>

    <!-- Transistor Selector -->
    <div class="transistor-selector">
      <label>Select Transistor:</label>
      <select v-model="selectedTransistorId" @change="loadDPTData">
        <option value="">-- Choose a transistor --</option>
        <option v-for="t in transistorList" :key="t.id" :value="t.id">
          {{ t.name }}
        </option>
      </select>
    </div>

    <div v-if="selectedTransistorId" class="dpt-content">
      <!-- Tabs -->
      <div class="tab-nav">
        <button
          :class="{ active: activeTab === 'data' }"
          @click="activeTab = 'data'"
          class="tab-btn"
        >
          📊 DPT Data
        </button>
        <button
          :class="{ active: activeTab === 'netlist' }"
          @click="activeTab = 'netlist'"
          class="tab-btn"
        >
          📝 Generate Netlist
        </button>
        <button
          :class="{ active: activeTab === 'validate' }"
          @click="activeTab = 'validate'"
          class="tab-btn"
        >
          ✅ Validate Data
        </button>
      </div>

      <!-- DPT Data Tab -->
      <div v-if="activeTab === 'data'" class="tab-panel">
        <h3>Measurement Data Overview</h3>

        <div v-if="dptData" class="data-overview">
          <div class="status-grid">
            <div class="status-card" :class="{ available: dptData.has_e_on }">
              <div class="status-icon">{{ dptData.has_e_on ? '✅' : '❌' }}</div>
              <div class="status-label">Turn-On Energy (E_on)</div>
              <div class="status-text">
                {{ dptData.has_e_on ? 'Available' : 'Not available' }}
              </div>
            </div>

            <div class="status-card" :class="{ available: dptData.has_e_off }">
              <div class="status-icon">{{ dptData.has_e_off ? '✅' : '❌' }}</div>
              <div class="status-label">Turn-Off Energy (E_off)</div>
              <div class="status-text">
                {{ dptData.has_e_off ? 'Available' : 'Not available' }}
              </div>
            </div>

            <div class="status-card" :class="{ available: dptData.has_e_rr }">
              <div class="status-icon">{{ dptData.has_e_rr ? '✅' : '❌' }}</div>
              <div class="status-label">Reverse Recovery (E_rr)</div>
              <div class="status-text">
                {{ dptData.has_e_rr ? 'Available' : 'Not available' }}
              </div>
            </div>
          </div>

          <!-- Metadata -->
          <div v-if="dptData.metadata" class="metadata-section">
            <h4>Measurement Metadata</h4>
            <div class="metadata-grid">
              <div class="metadata-item">
                <span class="metadata-label">Junction Temperature:</span>
                <span class="metadata-value">{{ dptData.metadata.t_j }}°C</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Supply Voltage:</span>
                <span class="metadata-value">{{ dptData.metadata.v_supply }}V</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Gate Voltage:</span>
                <span class="metadata-value">{{ dptData.metadata.v_g }}V</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Gate Resistance:</span>
                <span class="metadata-value">{{ dptData.metadata.r_g }}Ω</span>
              </div>
            </div>
            <div v-if="dptData.metadata.comment" class="metadata-comment">
              <strong>Comment:</strong> {{ dptData.metadata.comment }}
            </div>
          </div>
        </div>

        <div v-else class="no-data">
          <p>No DPT data available for this transistor.</p>
        </div>
      </div>

      <!-- Generate Netlist Tab -->
      <div v-if="activeTab === 'netlist'" class="tab-panel">
        <h3>LTspice DPT Netlist Generator</h3>

        <div class="config-form">
          <div class="form-grid">
            <div class="form-section">
              <h4>Circuit Parameters</h4>
              <div class="form-row">
                <label>DC Bus Voltage:</label>
                <input v-model.number="dptConfig.v_dc" type="number" step="10" />
                <span class="unit">V</span>
              </div>
              <div class="form-row">
                <label>Target Current:</label>
                <input v-model.number="dptConfig.i_target" type="number" step="1" />
                <span class="unit">A</span>
              </div>
              <div class="form-row">
                <label>Load Inductance:</label>
                <input v-model.number="dptConfig.l_load" type="number" step="10e-6" />
                <span class="unit">H</span>
              </div>
            </div>

            <div class="form-section">
              <h4>Gate Drive</h4>
              <div class="form-row">
                <label>R_g Turn-On:</label>
                <input v-model.number="dptConfig.r_g_on" type="number" step="0.5" />
                <span class="unit">Ω</span>
              </div>
              <div class="form-row">
                <label>R_g Turn-Off:</label>
                <input v-model.number="dptConfig.r_g_off" type="number" step="0.5" />
                <span class="unit">Ω</span>
              </div>
              <div class="form-row">
                <label>V_gate On:</label>
                <input v-model.number="dptConfig.v_g_on" type="number" step="1" />
                <span class="unit">V</span>
              </div>
              <div class="form-row">
                <label>V_gate Off:</label>
                <input v-model.number="dptConfig.v_g_off" type="number" step="1" />
                <span class="unit">V</span>
              </div>
            </div>

            <div class="form-section">
              <h4>Timing</h4>
              <div class="form-row">
                <label>Dead Time:</label>
                <input v-model.number="dptConfig.t_dead" type="number" step="1e-6" />
                <span class="unit">s</span>
              </div>
              <div class="form-row">
                <label>Pulse 2 Duration:</label>
                <input v-model.number="dptConfig.t_pulse2" type="number" step="1e-6" />
                <span class="unit">s</span>
              </div>
            </div>
          </div>

          <button @click="generateNetlist" :disabled="generating" class="btn-generate">
            {{ generating ? 'Generating...' : 'Generate LTspice Netlist' }}
          </button>
        </div>

        <!-- Netlist Display -->
        <div v-if="generatedNetlist" class="netlist-display">
          <div class="netlist-header">
            <h4>Generated Netlist</h4>
            <div class="netlist-actions">
              <button @click="copyNetlist" class="btn-copy">
                📋 Copy to Clipboard
              </button>
              <button @click="downloadNetlist" class="btn-download">
                💾 Download .asc file
              </button>
            </div>
          </div>
          <pre class="netlist-content">{{ generatedNetlist.netlist }}</pre>
        </div>
      </div>

      <!-- Validate Tab -->
      <div v-if="activeTab === 'validate'" class="tab-panel">
        <h3>Validate DPT Data</h3>
        <p class="validation-description">
          Check DPT measurement data against datasheet specifications to identify
          potential issues or out-of-range test conditions.
        </p>

        <button @click="validateData" :disabled="validating" class="btn-validate">
          {{ validating ? 'Validating...' : 'Run Validation Checks' }}
        </button>

        <!-- Validation Results -->
        <div v-if="validationResults" class="validation-results">
          <div class="validation-summary" :class="validationResults.valid ? 'valid' : 'invalid'">
            <div class="summary-icon">
              {{ validationResults.valid ? '✅' : '⚠️' }}
            </div>
            <div class="summary-text">
              <strong>{{ validationResults.valid ? 'Valid' : 'Issues Found' }}</strong>
              <p>{{ validationResults.valid ? 'All checks passed' : 'See details below' }}</p>
            </div>
          </div>

          <!-- Statistics -->
          <div v-if="validationResults.statistics" class="statistics-section">
            <h4>Test Conditions Statistics</h4>
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-label">Supply Voltage Range</div>
                <div class="stat-value">
                  {{ validationResults.statistics.v_supply_range[0] }} -
                  {{ validationResults.statistics.v_supply_range[1] }} V
                </div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Max Test Current</div>
                <div class="stat-value">
                  {{ validationResults.statistics.i_max_tested.toFixed(1) }} A
                </div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Temperature Range</div>
                <div class="stat-value">
                  {{ validationResults.statistics.t_j_range[0] }} -
                  {{ validationResults.statistics.t_j_range[1] }} °C
                </div>
              </div>
            </div>
          </div>

          <!-- Errors -->
          <div v-if="validationResults.errors.length > 0" class="error-list">
            <h4>Errors ({{ validationResults.errors.length }})</h4>
            <div v-for="(error, index) in validationResults.errors" :key="index" class="error-item">
              ❌ {{ error }}
            </div>
          </div>

          <!-- Warnings -->
          <div v-if="validationResults.warnings.length > 0" class="warning-list">
            <h4>Warnings ({{ validationResults.warnings.length }})</h4>
            <div v-for="(warning, index) in validationResults.warnings" :key="index" class="warning-item">
              ⚠️ {{ warning }}
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
import { ref, onMounted } from 'vue'
import { transistorApi } from '../services/api'

export default {
  name: 'DPTManager',
  setup() {
    const transistorList = ref([])
    const selectedTransistorId = ref('')
    const activeTab = ref('data')
    const dptData = ref(null)
    const generating = ref(false)
    const validating = ref(false)
    const error = ref(null)

    // DPT Configuration
    const dptConfig = ref({
      v_dc: 400.0,
      i_target: 20.0,
      l_load: 100e-6,
      r_g_on: 10.0,
      r_g_off: 10.0,
      v_g_on: 15.0,
      v_g_off: -5.0,
      t_dead: 5e-6,
      t_pulse2: 2e-6
    })

    const generatedNetlist = ref(null)
    const validationResults = ref(null)

    onMounted(async () => {
      try {
        const transistors = await transistorApi.getAll()
        transistorList.value = transistors.map(t => ({
          id: t.metadata?.name || t.name,
          name: t.metadata?.name || t.name
        }))
      } catch (err) {
        error.value = 'Failed to load transistor list'
      }
    })

    async function loadDPTData() {
      if (!selectedTransistorId.value) return

      try {
        error.value = null
        dptData.value = await transistorApi.getDPTData(selectedTransistorId.value)
      } catch (err) {
        error.value = 'Failed to load DPT data'
        dptData.value = null
      }
    }

    async function generateNetlist() {
      if (!selectedTransistorId.value) return

      generating.value = true
      error.value = null

      try {
        generatedNetlist.value = await transistorApi.generateDPTNetlist(
          selectedTransistorId.value,
          dptConfig.value
        )
      } catch (err) {
        error.value = 'Failed to generate netlist: ' + err.message
      } finally {
        generating.value = false
      }
    }

    async function validateData() {
      if (!selectedTransistorId.value) return

      validating.value = true
      error.value = null

      try {
        validationResults.value = await transistorApi.validateDPTData(selectedTransistorId.value)
      } catch (err) {
        error.value = 'Failed to validate data: ' + err.message
      } finally {
        validating.value = false
      }
    }

    function copyNetlist() {
      if (!generatedNetlist.value) return

      navigator.clipboard.writeText(generatedNetlist.value.netlist).then(() => {
        alert('Netlist copied to clipboard!')
      })
    }

    function downloadNetlist() {
      if (!generatedNetlist.value) return

      const blob = new Blob([generatedNetlist.value.netlist], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${generatedNetlist.value.transistor_name}_DPT.asc`
      link.click()
      URL.revokeObjectURL(url)
    }

    return {
      transistorList,
      selectedTransistorId,
      activeTab,
      dptData,
      dptConfig,
      generating,
      validating,
      error,
      generatedNetlist,
      validationResults,
      loadDPTData,
      generateNetlist,
      validateData,
      copyNetlist,
      downloadNetlist
    }
  }
}
</script>

<style scoped>
.dpt-manager {
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

/* Transistor Selector */
.transistor-selector {
  margin-bottom: 30px;
}

.transistor-selector label {
  display: block;
  font-weight: bold;
  margin-bottom: 10px;
}

.transistor-selector select {
  width: 100%;
  max-width: 400px;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

/* Tabs */
.tab-nav {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  border-bottom: 2px solid #e0e0e0;
}

.tab-btn {
  padding: 12px 20px;
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

/* Tab Panel */
.tab-panel {
  background: white;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.tab-panel h3 {
  font-size: 20px;
  margin-bottom: 20px;
}

/* Data Overview */
.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.status-card {
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  background: #f5f5f5;
  border: 2px solid #e0e0e0;
}

.status-card.available {
  background: #e8f5e9;
  border-color: #4CAF50;
}

.status-icon {
  font-size: 36px;
  margin-bottom: 10px;
}

.status-label {
  font-weight: bold;
  margin-bottom: 5px;
}

.status-text {
  color: #666;
  font-size: 14px;
}

/* Metadata */
.metadata-section {
  padding: 20px;
  background: #f9f9f9;
  border-radius: 6px;
}

.metadata-section h4 {
  font-size: 16px;
  margin-bottom: 15px;
}

.metadata-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 15px;
}

.metadata-item {
  display: flex;
  justify-content: space-between;
}

.metadata-label {
  font-weight: bold;
}

.metadata-comment {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #e0e0e0;
}

/* Config Form */
.config-form {
  margin-bottom: 30px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
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
  grid-template-columns: 1fr 100px 40px;
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

/* Buttons */
.btn-generate,
.btn-validate {
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

.btn-generate:hover:not(:disabled),
.btn-validate:hover:not(:disabled) {
  background: #45a049;
}

.btn-generate:disabled,
.btn-validate:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* Netlist Display */
.netlist-display {
  margin-top: 30px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  overflow: hidden;
}

.netlist-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
}

.netlist-header h4 {
  margin: 0;
  font-size: 16px;
}

.netlist-actions {
  display: flex;
  gap: 10px;
}

.btn-copy,
.btn-download {
  padding: 8px 16px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
}

.btn-copy:hover,
.btn-download:hover {
  background: #1976D2;
}

.netlist-content {
  padding: 20px;
  background: #fafafa;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  overflow-x: auto;
  max-height: 500px;
  overflow-y: auto;
}

/* Validation Results */
.validation-description {
  color: #666;
  margin-bottom: 20px;
}

.validation-results {
  margin-top: 30px;
}

.validation-summary {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 30px;
}

.validation-summary.valid {
  background: #e8f5e9;
  border-left: 4px solid #4CAF50;
}

.validation-summary.invalid {
  background: #fff3e0;
  border-left: 4px solid #ff9800;
}

.summary-icon {
  font-size: 48px;
}

.summary-text strong {
  font-size: 20px;
}

.summary-text p {
  margin: 5px 0 0 0;
  color: #666;
}

/* Statistics */
.statistics-section {
  margin-bottom: 30px;
}

.statistics-section h4 {
  font-size: 16px;
  margin-bottom: 15px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.stat-card {
  padding: 15px;
  background: #f9f9f9;
  border-radius: 6px;
  text-align: center;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 20px;
  font-weight: bold;
}

/* Error and Warning Lists */
.error-list,
.warning-list {
  padding: 15px;
  border-radius: 6px;
  margin-bottom: 20px;
}

.error-list {
  background: #ffebee;
  border-left: 4px solid #f44336;
}

.warning-list {
  background: #fff3e0;
  border-left: 4px solid #ff9800;
}

.error-list h4,
.warning-list h4 {
  font-size: 16px;
  margin-bottom: 10px;
}

.error-item,
.warning-item {
  padding: 8px 0;
  border-bottom: 1px solid rgba(0,0,0,0.1);
}

.error-item:last-child,
.warning-item:last-child {
  border-bottom: none;
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

.no-data {
  text-align: center;
  padding: 40px;
  color: #999;
}
</style>
