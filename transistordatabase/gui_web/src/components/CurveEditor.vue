<template>
  <div class="curve-editor">
    <div class="editor-header">
      <h3>{{ editMode ? 'Edit' : 'Add' }} {{ curveTypeLabel }}</h3>
      <button @click="$emit('cancel')" class="btn-close">✕</button>
    </div>

    <div class="editor-content">
      <!-- Curve Parameters Section -->
      <div class="section parameters-section">
        <h4>📋 Curve Parameters</h4>
        <div class="form-grid">
          <div class="form-group">
            <label>Junction Temperature (T_j) *</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.t_j"
                type="number"
                step="1"
                required
              />
              <span class="unit">°C</span>
            </div>
          </div>

          <div v-if="requiresVg" class="form-group">
            <label>Gate Voltage (V_g) *</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.v_g"
                type="number"
                step="0.1"
                required
              />
              <span class="unit">V</span>
            </div>
          </div>

          <div v-if="curveType === 'switching_loss'" class="form-group">
            <label>Supply Voltage (V_supply) *</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.v_supply"
                type="number"
                step="1"
                required
              />
              <span class="unit">V</span>
            </div>
          </div>

          <div v-if="curveType === 'switching_loss'" class="form-group">
            <label>Gate Resistance (R_g)</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.r_g"
                type="number"
                step="0.1"
              />
              <span class="unit">Ω</span>
            </div>
          </div>

          <div v-if="curveType === 'gate_charge'" class="form-group">
            <label>Channel Current (I_channel) *</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.i_channel"
                type="number"
                step="0.1"
                required
              />
              <span class="unit">A</span>
            </div>
          </div>

          <div v-if="curveType === 'gate_charge'" class="form-group">
            <label>Supply Voltage (V_supply) *</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.v_supply"
                type="number"
                step="1"
                required
              />
              <span class="unit">V</span>
            </div>
          </div>

          <div v-if="curveType === 'soa'" class="form-group">
            <label>Case Temperature (T_c)</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.t_c"
                type="number"
                step="1"
              />
              <span class="unit">°C</span>
            </div>
          </div>

          <div v-if="curveType === 'soa'" class="form-group">
            <label>Pulse Duration</label>
            <div class="input-with-unit">
              <input
                v-model.number="formData.time_pulse"
                type="number"
                step="0.001"
              />
              <span class="unit">s</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Data Input Method Selection -->
      <div class="section data-method-section">
        <h4>📊 Data Input Method</h4>
        <div class="method-tabs">
          <button
            @click="dataInputMethod = 'upload'"
            :class="['method-tab', { active: dataInputMethod === 'upload' }]"
          >
            📁 Upload CSV
          </button>
          <button
            @click="dataInputMethod = 'manual'"
            :class="['method-tab', { active: dataInputMethod === 'manual' }]"
          >
            ✏️ Manual Entry
          </button>
        </div>

        <!-- CSV Upload -->
        <div v-if="dataInputMethod === 'upload'" class="upload-container">
          <CurveUploader
            :x-label="xAxisLabel"
            :y-label="yAxisLabel"
            :x-unit="xUnit"
            :y-unit="yUnit"
            @use-data="handleUploadedData"
            @error="handleUploadError"
          />
        </div>

        <!-- Manual Entry -->
        <div v-else class="manual-entry-container">
          <div class="manual-header">
            <div class="column-labels">
              <span>{{ xAxisLabel }} ({{ xUnit }})</span>
              <span>{{ yAxisLabel }} ({{ yUnit }})</span>
            </div>
            <button @click="addDataPoint" class="btn btn-sm btn-primary">
              ➕ Add Point
            </button>
          </div>

          <div class="data-points-list">
            <div
              v-for="(point, index) in manualDataPoints"
              :key="index"
              class="data-point-row"
            >
              <input
                v-model.number="point.x"
                type="number"
                step="any"
                placeholder="0.0"
              />
              <input
                v-model.number="point.y"
                type="number"
                step="any"
                placeholder="0.0"
              />
              <button
                @click="removeDataPoint(index)"
                class="btn-icon btn-danger-icon"
                title="Remove"
              >
                🗑️
              </button>
            </div>
          </div>

          <div v-if="manualDataPoints.length === 0" class="empty-manual">
            Click "Add Point" to start entering data manually
          </div>
        </div>
      </div>

      <!-- Live Preview -->
      <div v-if="hasValidData" class="section preview-section">
        <h4>📈 Live Preview</h4>
        <CurvePlotter
          :title="`${curveTypeLabel} Preview`"
          :datasets="previewDatasets"
          :x-label="`${xAxisLabel} (${xUnit})`"
          :y-label="`${yAxisLabel} (${yUnit})`"
        />
      </div>

      <!-- Validation Status -->
      <div v-if="validationMessage" :class="['validation-message', validationClass]">
        {{ validationMessage }}
      </div>
    </div>

    <div class="editor-footer">
      <button @click="$emit('cancel')" class="btn btn-secondary">
        Cancel
      </button>
      <button
        @click="submitCurve"
        :disabled="!canSubmit"
        class="btn btn-primary"
      >
        {{ editMode ? '💾 Update' : '✅ Add' }} Curve
      </button>
    </div>
  </div>
</template>

<script>
import CurveUploader from './CurveUploader.vue'
import CurvePlotter from './CurvePlotter.vue'

export default {
  name: 'CurveEditor',
  components: {
    CurveUploader,
    CurvePlotter
  },
  props: {
    curveType: {
      type: String,
      required: true,
      // 'channel', 'switching_loss', 'gate_charge', 'soa', 'capacitance'
    },
    component: {
      type: String,
      default: 'switch',
      // 'switch' or 'diode'
    },
    lossType: {
      type: String,
      default: null,
      // 'e_on', 'e_off', 'e_rr'
    },
    capacitanceType: {
      type: String,
      default: null,
      // 'c_oss', 'c_iss', 'c_rss'
    },
    editMode: {
      type: Boolean,
      default: false
    },
    initialData: {
      type: Object,
      default: null
    }
  },
  emits: ['submit', 'cancel'],
  data() {
    return {
      dataInputMethod: 'upload',
      formData: {
        t_j: 25,
        v_g: null,
        v_supply: null,
        r_g: null,
        i_channel: null,
        t_c: null,
        time_pulse: null
      },
      manualDataPoints: [],
      uploadedData: null,
      validationMessage: null,
      validationClass: 'info'
    }
  },
  computed: {
    curveTypeLabel() {
      const labels = {
        channel: 'Channel Characteristic',
        switching_loss: `Switching Loss (${this.lossType?.toUpperCase() || 'E_x'})`,
        gate_charge: 'Gate Charge Curve',
        soa: 'Safe Operating Area',
        capacitance: `Capacitance (${this.capacitanceType?.toUpperCase() || 'C_x'})`
      }
      return labels[this.curveType] || 'Curve'
    },

    requiresVg() {
      return this.curveType === 'channel' && this.component === 'switch'
    },

    xAxisLabel() {
      if (this.curveType === 'gate_charge') return 'Charge (Q_g)'
      if (this.curveType === 'soa') return 'Voltage (V_ds)'
      if (this.curveType === 'capacitance') return 'Voltage (V_ds)'
      return 'Voltage (V)'
    },

    yAxisLabel() {
      if (this.curveType === 'gate_charge') return 'Voltage (V_gs)'
      if (this.curveType === 'soa') return 'Current (I_d)'
      if (this.curveType === 'capacitance') return 'Capacitance (C)'
      return 'Current (I)'
    },

    xUnit() {
      if (this.curveType === 'gate_charge') return 'C'
      if (this.curveType === 'capacitance') return 'V'
      if (this.curveType === 'soa') return 'V'
      return 'V'
    },

    yUnit() {
      if (this.curveType === 'gate_charge') return 'V'
      if (this.curveType === 'capacitance') return 'F'
      return 'A'
    },

    currentData() {
      if (this.dataInputMethod === 'upload' && this.uploadedData) {
        return this.uploadedData
      }
      if (this.dataInputMethod === 'manual') {
        return this.manualDataPoints
          .filter(p => !isNaN(p.x) && !isNaN(p.y))
          .map(p => [p.x, p.y])
      }
      return []
    },

    hasValidData() {
      return this.currentData.length >= 2
    },

    previewDatasets() {
      if (!this.hasValidData) return []

      return [
        {
          label: this.curveTypeLabel,
          data: this.currentData.map(([x, y]) => ({ x, y })),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)'
        }
      ]
    },

    canSubmit() {
      if (!this.hasValidData) return false
      if (this.formData.t_j === null) return false
      if (this.requiresVg && this.formData.v_g === null) return false
      if (this.curveType === 'switching_loss' && this.formData.v_supply === null) return false
      if (this.curveType === 'gate_charge') {
        if (this.formData.i_channel === null) return false
        if (this.formData.v_supply === null) return false
      }
      return true
    }
  },
  mounted() {
    if (this.initialData) {
      this.loadInitialData()
    }
  },
  methods: {
    handleUploadedData(data) {
      this.uploadedData = data
      this.validateData()
    },

    handleUploadError(error) {
      this.validationMessage = error
      this.validationClass = 'error'
    },

    addDataPoint() {
      this.manualDataPoints.push({ x: 0, y: 0 })
    },

    removeDataPoint(index) {
      this.manualDataPoints.splice(index, 1)
    },

    validateData() {
      if (this.currentData.length < 2) {
        this.validationMessage = '⚠️ At least 2 data points required'
        this.validationClass = 'warning'
        return false
      }

      // Check for monotonic increase in x-axis
      const xValues = this.currentData.map(([x]) => x)
      const isMonotonic = xValues.every((val, i) => i === 0 || val > xValues[i - 1])

      if (!isMonotonic) {
        this.validationMessage = '⚠️ X-axis values should be monotonically increasing'
        this.validationClass = 'warning'
      } else {
        this.validationMessage = `✅ ${this.currentData.length} data points ready`
        this.validationClass = 'success'
      }

      return true
    },

    submitCurve() {
      if (!this.validateData()) return

      const curveData = {
        t_j: this.formData.t_j,
        v_data: this.currentData.map(([x]) => x),
        i_data: this.currentData.map(([, y]) => y)
      }

      // Add optional parameters
      if (this.formData.v_g !== null) curveData.v_g = this.formData.v_g
      if (this.formData.v_supply !== null) curveData.v_supply = this.formData.v_supply
      if (this.formData.r_g !== null) curveData.r_g = this.formData.r_g
      if (this.formData.i_channel !== null) curveData.i_channel = this.formData.i_channel
      if (this.formData.t_c !== null) curveData.t_c = this.formData.t_c
      if (this.formData.time_pulse !== null) curveData.time_pulse = this.formData.time_pulse

      // Rename for gate charge
      if (this.curveType === 'gate_charge') {
        curveData.q_data = curveData.v_data
        curveData.v_data = curveData.i_data
        delete curveData.i_data
      }

      // Rename for capacitance
      if (this.curveType === 'capacitance') {
        curveData.c_data = curveData.i_data
        delete curveData.i_data
      }

      // Add dataset_type for switching losses
      if (this.curveType === 'switching_loss') {
        curveData.dataset_type = 'graph_i_e'
      }

      this.$emit('submit', curveData)
    },

    loadInitialData() {
      // Load initial data for edit mode
      Object.assign(this.formData, this.initialData)
    }
  }
}
</script>

<style scoped>
.curve-editor {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  max-width: 1200px;
  margin: 0 auto;
}

.editor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 2px solid #e5e7eb;
}

.editor-header h3 {
  margin: 0;
  color: #1f2937;
  font-size: 20px;
}

.btn-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #6b7280;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s;
}

.btn-close:hover {
  background: #f3f4f6;
  color: #1f2937;
}

.editor-content {
  padding: 24px;
}

.section {
  margin-bottom: 32px;
}

.section h4 {
  margin: 0 0 16px 0;
  color: #374151;
  font-size: 16px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  color: #374151;
  font-weight: 500;
  font-size: 14px;
}

.input-with-unit {
  display: flex;
  align-items: center;
  gap: 8px;
}

.input-with-unit input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
}

.input-with-unit input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.unit {
  color: #6b7280;
  font-weight: 500;
  min-width: 30px;
}

.method-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
}

.method-tab {
  flex: 1;
  padding: 12px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.method-tab:hover {
  border-color: #3b82f6;
}

.method-tab.active {
  border-color: #3b82f6;
  background: #eff6ff;
  color: #3b82f6;
}

.manual-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.column-labels {
  display: flex;
  gap: 100px;
  font-weight: 500;
  color: #374151;
  font-size: 14px;
}

.data-points-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.data-point-row {
  display: flex;
  gap: 12px;
  align-items: center;
}

.data-point-row input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
}

.btn-icon {
  background: none;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.btn-danger-icon:hover {
  background: #fef2f2;
  border-color: #ef4444;
}

.empty-manual {
  text-align: center;
  padding: 40px;
  color: #9ca3af;
  font-style: italic;
}

.validation-message {
  padding: 12px 16px;
  border-radius: 6px;
  font-size: 14px;
  margin-top: 16px;
}

.validation-message.success {
  background: #d1fae5;
  border: 1px solid #6ee7b7;
  color: #047857;
}

.validation-message.warning {
  background: #fef3c7;
  border: 1px solid #fcd34d;
  color: #92400e;
}

.validation-message.error {
  background: #fee2e2;
  border: 1px solid #fca5a5;
  color: #dc2626;
}

.editor-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 20px 24px;
  border-top: 2px solid #e5e7eb;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  font-size: 14px;
  transition: all 0.2s;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 13px;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
}

.btn-primary:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}
</style>
