<template>
  <div class="transistor-comparison-enhanced">
    <div class="comparison-header">
      <h2>🔍 Advanced Transistor Comparison</h2>
      <div class="header-actions">
        <button @click="clearAll" class="btn btn-secondary">
          🗑️ Clear All
        </button>
        <button
          @click="runComparison"
          :disabled="selectedTransistors.length < 2"
          class="btn btn-primary"
        >
          🚀 Run Comparison
        </button>
      </div>
    </div>

    <!-- Transistor Selection Section -->
    <div class="selection-section">
      <h3>📌 Select Transistors ({{ selectedTransistors.length }}/3)</h3>

      <div class="transistor-slots">
        <div
          v-for="slot in 3"
          :key="slot"
          class="transistor-slot"
          :class="{ filled: selectedTransistors[slot - 1] }"
        >
          <div class="slot-header">
            <h4>Transistor {{ slot }}</h4>
            <button
              v-if="selectedTransistors[slot - 1]"
              @click="removeTransistor(slot - 1)"
              class="btn-icon btn-danger-small"
            >
              ✕
            </button>
          </div>

          <div v-if="selectedTransistors[slot - 1]" class="selected-transistor">
            <div class="transistor-info">
              <strong>{{ selectedTransistors[slot - 1].metadata.name }}</strong>
              <p>{{ selectedTransistors[slot - 1].metadata.type }} - {{ selectedTransistors[slot - 1].metadata.manufacturer }}</p>
              <div class="specs">
                <span>{{ selectedTransistors[slot - 1].electrical.v_abs_max }}V</span>
                <span>{{ selectedTransistors[slot - 1].electrical.i_abs_max }}A</span>
              </div>
            </div>

            <!-- Configuration Panel -->
            <div class="config-panel">
              <div class="config-row">
                <label>V_supply (V):</label>
                <input
                  v-model.number="configurations[slot - 1].v_supply"
                  type="number"
                  step="10"
                  min="0"
                  @change="onConfigChange"
                />
              </div>

              <div class="config-row">
                <label>R_g,on (Ω):</label>
                <input
                  v-model.number="configurations[slot - 1].r_g_on"
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  @input="onConfigChange"
                />
                <span class="value">{{ configurations[slot - 1].r_g_on }}Ω</span>
              </div>

              <div class="config-row">
                <label>R_g,off (Ω):</label>
                <input
                  v-model.number="configurations[slot - 1].r_g_off"
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  @input="onConfigChange"
                />
                <span class="value">{{ configurations[slot - 1].r_g_off }}Ω</span>
              </div>

              <div class="config-row">
                <label>Parallel Count:</label>
                <select
                  v-model.number="configurations[slot - 1].parallel_count"
                  @change="onConfigChange"
                >
                  <option :value="1">x1</option>
                  <option :value="2">x2</option>
                  <option :value="4">x4</option>
                  <option :value="8">x8</option>
                </select>
              </div>
            </div>
          </div>

          <div v-else class="empty-slot">
            <button @click="openSelector(slot - 1)" class="btn-add">
              ➕ Select Transistor
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Global Configuration -->
    <div v-if="selectedTransistors.length >= 2" class="global-config-section">
      <h3>⚙️ Global Settings</h3>
      <div class="global-config-grid">
        <div class="config-item">
          <label>Junction Temperature (T_j):</label>
          <input
            v-model.number="globalConfig.t_j"
            type="number"
            step="5"
            @change="onConfigChange"
          />
          <span class="unit">°C</span>
        </div>

        <div class="config-item">
          <label>Channel Current (I_channel):</label>
          <input
            v-model.number="globalConfig.i_channel"
            type="number"
            step="5"
            @change="onConfigChange"
          />
          <span class="unit">A</span>
        </div>
      </div>
    </div>

    <!-- Comparison Results - 9 Plot Types -->
    <div v-if="comparisonResults" class="results-section">
      <h3>📊 Comparison Results</h3>

      <div class="plot-tabs">
        <button
          v-for="(plotType, index) in plotTypes"
          :key="index"
          @click="activePlot = plotType.key"
          :class="['plot-tab', { active: activePlot === plotType.key }]"
        >
          {{ plotType.icon }} {{ plotType.label }}
        </button>
      </div>

      <div class="plot-container">
        <PlotCard
          v-if="activePlot === 'channel'"
          title="Channel Characteristics Comparison"
          :plot-data="comparisonResults.channel_characteristics"
          data-requirement="channel characteristic"
        />

        <PlotCard
          v-else-if="activePlot === 'eon'"
          title="Turn-On Switching Losses (E_on)"
          :plot-data="comparisonResults.switching_losses_eon"
          data-requirement="E_on switching loss"
        />

        <PlotCard
          v-else-if="activePlot === 'eoff'"
          title="Turn-Off Switching Losses (E_off)"
          :plot-data="comparisonResults.switching_losses_eoff"
          data-requirement="E_off switching loss"
        />

        <PlotCard
          v-else-if="activePlot === 'gate_charge'"
          title="Gate Charge Comparison"
          :plot-data="comparisonResults.gate_charge"
          data-requirement="gate charge"
        />

        <PlotCard
          v-else-if="activePlot === 'soa'"
          title="Safe Operating Area Comparison"
          :plot-data="comparisonResults.soa"
          data-requirement="SOA"
        />

        <PlotCard
          v-else-if="activePlot === 'thermal'"
          title="Thermal Impedance Comparison"
          :plot-data="comparisonResults.thermal_impedance"
          data-requirement="thermal model"
        />

        <PlotCard
          v-else-if="activePlot === 'capacitances'"
          title="Capacitance Comparison"
          :plot-data="comparisonResults.capacitances"
          data-requirement="capacitance"
        />

        <PlotCard
          v-else-if="activePlot === 'loss_breakdown'"
          title="Loss Breakdown"
          :plot-data="comparisonResults.loss_breakdown"
          data-requirement="loss"
        />

        <PlotCard
          v-else-if="activePlot === 'efficiency'"
          title="Efficiency vs Load Current"
          :plot-data="comparisonResults.efficiency"
          data-requirement="efficiency"
        />
      </div>
    </div>

    <!-- Transistor Selector Modal -->
    <div v-if="showSelector" class="modal-overlay" @click="closeSelector">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h3>Select Transistor {{ currentSlot + 1 }}</h3>
          <button @click="closeSelector" class="btn-close">✕</button>
        </div>

        <div class="selector-filters">
          <input
            v-model="searchQuery"
            type="text"
            placeholder="🔍 Search by name, type, manufacturer..."
            class="search-input"
          />
          <select v-model="typeFilter">
            <option value="">All Types</option>
            <option v-for="type in availableTypes" :key="type" :value="type">
              {{ type }}
            </option>
          </select>
        </div>

        <div class="transistor-list">
          <div
            v-for="(transistor, index) in filteredTransistors"
            :key="index"
            @click="selectTransistor(transistor)"
            class="transistor-item"
            :class="{ disabled: isAlreadySelected(transistor) }"
          >
            <div class="item-name">{{ transistor.metadata.name }}</div>
            <div class="item-specs">
              <span>{{ transistor.metadata.type }}</span>
              <span>{{ transistor.electrical.v_abs_max }}V</span>
              <span>{{ transistor.electrical.i_abs_max }}A</span>
            </div>
            <div v-if="isAlreadySelected(transistor)" class="already-selected">
              ✅ Selected
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Loading Indicator -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <p>Running comparison...</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { transistorApi } from '../services/api.js'
import PlotCard from './PlotCard.vue'

// State
const allTransistors = ref([])
const selectedTransistors = ref([])
const configurations = ref([
  { v_supply: 600, r_g_on: 10, r_g_off: 10, parallel_count: 1 },
  { v_supply: 600, r_g_on: 10, r_g_off: 10, parallel_count: 1 },
  { v_supply: 600, r_g_on: 10, r_g_off: 10, parallel_count: 1 }
])
const globalConfig = ref({
  t_j: 25,
  i_channel: 50
})
const comparisonResults = ref(null)
const activePlot = ref('channel')
const isLoading = ref(false)

// Selector Modal
const showSelector = ref(false)
const currentSlot = ref(0)
const searchQuery = ref('')
const typeFilter = ref('')

// Plot types
const plotTypes = [
  { key: 'channel', label: 'Channel', icon: '⚡' },
  { key: 'eon', label: 'E_on', icon: '🔼' },
  { key: 'eoff', label: 'E_off', icon: '🔽' },
  { key: 'gate_charge', label: 'Gate Charge', icon: '🔋' },
  { key: 'soa', label: 'SOA', icon: '🛡️' },
  { key: 'thermal', label: 'Thermal', icon: '🌡️' },
  { key: 'capacitances', label: 'Capacitances', icon: '🔌' },
  { key: 'loss_breakdown', label: 'Loss Breakdown', icon: '📊' },
  { key: 'efficiency', label: 'Efficiency', icon: '📈' }
]

// Computed
const availableTypes = computed(() => {
  const types = new Set(allTransistors.value.map(t => t.metadata.type))
  return Array.from(types).sort()
})

const filteredTransistors = computed(() => {
  let filtered = allTransistors.value

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(t =>
      t.metadata.name.toLowerCase().includes(query) ||
      t.metadata.type.toLowerCase().includes(query) ||
      t.metadata.manufacturer.toLowerCase().includes(query)
    )
  }

  if (typeFilter.value) {
    filtered = filtered.filter(t => t.metadata.type === typeFilter.value)
  }

  return filtered
})

// Methods
onMounted(async () => {
  try {
    allTransistors.value = await transistorApi.getAll()
  } catch (error) {
    console.error('Failed to load transistors:', error)
  }
})

const openSelector = (slot) => {
  currentSlot.value = slot
  showSelector.value = true
  searchQuery.value = ''
  typeFilter.value = ''
}

const closeSelector = () => {
  showSelector.value = false
}

const selectTransistor = (transistor) => {
  if (isAlreadySelected(transistor)) return

  selectedTransistors.value[currentSlot.value] = transistor
  closeSelector()

  // Auto-run comparison if 2+ transistors selected
  if (selectedTransistors.value.filter(t => t).length >= 2) {
    setTimeout(() => runComparison(), 300)
  }
}

const removeTransistor = (slot) => {
  selectedTransistors.value[slot] = null
  comparisonResults.value = null
}

const isAlreadySelected = (transistor) => {
  return selectedTransistors.value.some(
    t => t && t.metadata.name === transistor.metadata.name
  )
}

const clearAll = () => {
  selectedTransistors.value = []
  comparisonResults.value = null
}

const onConfigChange = () => {
  // Debounced re-run of comparison
  if (selectedTransistors.value.filter(t => t).length >= 2 && comparisonResults.value) {
    setTimeout(() => runComparison(), 500)
  }
}

const runComparison = async () => {
  const selected = selectedTransistors.value.filter(t => t)
  if (selected.length < 2) return

  isLoading.value = true

  try {
    const transistorIds = selected.map(t => t.metadata.name)
    const config = {
      t_j: globalConfig.value.t_j,
      i_channel: globalConfig.value.i_channel,
      v_supply: configurations.value.slice(0, selected.length).map(c => c.v_supply),
      r_g_on: configurations.value.slice(0, selected.length).map(c => c.r_g_on),
      r_g_off: configurations.value.slice(0, selected.length).map(c => c.r_g_off),
      parallel_count: configurations.value.slice(0, selected.length).map(c => c.parallel_count)
    }

    comparisonResults.value = await transistorApi.compareAdvanced(transistorIds, config)
  } catch (error) {
    console.error('Comparison failed:', error)
    alert('Comparison failed: ' + error.message)
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.transistor-comparison-enhanced {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

.comparison-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
}

.comparison-header h2 {
  margin: 0;
  color: #1f2937;
  font-size: 28px;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.2s;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
  transform: translateY(-1px);
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

.selection-section {
  margin-bottom: 32px;
}

.selection-section h3 {
  margin-bottom: 20px;
  color: #374151;
}

.transistor-slots {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 20px;
}

.transistor-slot {
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  padding: 20px;
  background: white;
  transition: all 0.2s;
}

.transistor-slot.filled {
  border-color: #3b82f6;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
}

.slot-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.slot-header h4 {
  margin: 0;
  color: #374151;
  font-size: 16px;
}

.btn-icon {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
}

.btn-danger-small {
  color: #ef4444;
}

.selected-transistor {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.transistor-info strong {
  display: block;
  color: #1f2937;
  font-size: 16px;
  margin-bottom: 6px;
}

.transistor-info p {
  margin: 0;
  color: #6b7280;
  font-size: 14px;
}

.specs {
  display: flex;
  gap: 12px;
  margin-top: 8px;
}

.specs span {
  background: #eff6ff;
  color: #1e40af;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 500;
}

.config-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
}

.config-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.config-row label {
  flex: 0 0 120px;
  font-size: 13px;
  color: #374151;
  font-weight: 500;
}

.config-row input[type="number"],
.config-row select {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 13px;
}

.config-row input[type="range"] {
  flex: 1;
}

.config-row .value {
  flex: 0 0 50px;
  text-align: right;
  font-weight: 600;
  color: #3b82f6;
  font-size: 13px;
}

.empty-slot {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 200px;
}

.btn-add {
  padding: 12px 24px;
  background: #dbeafe;
  color: #1e40af;
  border: 2px dashed #3b82f6;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;
}

.btn-add:hover {
  background: #bfdbfe;
  border-color: #2563eb;
}

.global-config-section {
  margin-bottom: 32px;
  padding: 20px;
  background: white;
  border-radius: 12px;
  border: 2px solid #e5e7eb;
}

.global-config-section h3 {
  margin-top: 0;
  margin-bottom: 16px;
  color: #374151;
}

.global-config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
}

.config-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.config-item label {
  font-size: 14px;
  color: #374151;
  font-weight: 500;
  white-space: nowrap;
}

.config-item input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
}

.config-item .unit {
  color: #6b7280;
  font-size: 14px;
}

.results-section {
  margin-top: 40px;
}

.results-section h3 {
  margin-bottom: 20px;
  color: #374151;
}

.plot-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 24px;
}

.plot-tab {
  padding: 10px 16px;
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  font-size: 14px;
  color: #374151;
  transition: all 0.2s;
}

.plot-tab:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.plot-tab.active {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
}

.plot-container {
  margin-top: 20px;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 16px;
  padding: 0;
  max-width: 800px;
  width: 90%;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px;
  border-bottom: 2px solid #e5e7eb;
}

.modal-header h3 {
  margin: 0;
  color: #1f2937;
}

.btn-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #6b7280;
  padding: 0;
}

.btn-close:hover {
  color: #1f2937;
}

.selector-filters {
  display: flex;
  gap: 12px;
  padding: 20px 24px;
  background: #f9fafb;
}

.search-input {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
}

.selector-filters select {
  padding: 10px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  min-width: 150px;
}

.transistor-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px 24px;
}

.transistor-item {
  padding: 16px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.transistor-item:hover:not(.disabled) {
  border-color: #3b82f6;
  background: #eff6ff;
}

.transistor-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.item-name {
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 6px;
}

.item-specs {
  display: flex;
  gap: 12px;
  font-size: 13px;
  color: #6b7280;
}

.already-selected {
  position: absolute;
  top: 16px;
  right: 16px;
  background: #d1fae5;
  color: #047857;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}

/* Loading Overlay */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.95);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.loading-spinner {
  width: 60px;
  height: 60px;
  border: 4px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-overlay p {
  margin-top: 20px;
  font-size: 18px;
  color: #374151;
  font-weight: 500;
}
</style>
