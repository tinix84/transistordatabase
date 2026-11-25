<template>
  <div class="transistor-comparison">
    <div class="comparison-header">
      <h2>Transistor Comparison</h2>
      <div class="header-actions">
        <button @click="clearSelection" class="btn btn-secondary">
          Clear Selection
        </button>
        <button @click="exportComparison" :disabled="selectedTransistors.length < 2" class="btn btn-primary">
          Export Comparison
        </button>
      </div>
    </div>

    <div class="selection-section">
      <h3>Select Transistors to Compare ({{ selectedTransistors.length }}/{{ maxComparisons }})</h3>
      <div class="transistor-selector">
        <div class="selector-filters">
          <input 
            v-model="searchQuery" 
            type="text" 
            placeholder="Search transistors..."
            aria-label="Search transistors"
            class="search-input"
          />
          <select v-model="typeFilter" class="filter-select" aria-label="Filter by transistor type">
            <option value="">All Types</option>
            <option v-for="type in availableTypes" :key="type" :value="type">{{ type }}</option>
          </select>
          <select v-model="manufacturerFilter" class="filter-select" aria-label="Filter by manufacturer">
            <option value="">All Manufacturers</option>
            <option v-for="mfg in availableManufacturers" :key="mfg" :value="mfg">{{ mfg }}</option>
          </select>
        </div>

        <div class="transistor-grid">
          <div 
            v-for="(transistor, index) in filteredTransistors" 
            :key="index"
            role="button"
            :tabindex="!isSelected(transistor) && selectedTransistors.length >= maxComparisons ? -1 : 0"
            :aria-label="transistor.metadata.name"
            :aria-pressed="isSelected(transistor)"
            :aria-disabled="!isSelected(transistor) && selectedTransistors.length >= maxComparisons"
            @click="toggleTransistorSelection(transistor)"
            @keydown.enter.prevent="toggleTransistorSelection(transistor)"
            @keydown.space.prevent="toggleTransistorSelection(transistor)"
            :class="['transistor-card', { 
              selected: isSelected(transistor),
              disabled: !isSelected(transistor) && selectedTransistors.length >= maxComparisons
            }]"
          >
            <div class="card-header">
              <h4>{{ transistor.metadata.name }}</h4>
              <div class="selection-indicator">
                {{ isSelected(transistor) ? 'Selected' : 'Select' }}
              </div>
            </div>
            <div class="card-content">
              <p><strong>Type:</strong> {{ transistor.metadata.type }}</p>
              <p><strong>Manufacturer:</strong> {{ transistor.metadata.manufacturer }}</p>
              <p><strong>Max V:</strong> {{ transistor.electrical.v_abs_max }}V</p>
              <p><strong>Max I:</strong> {{ transistor.electrical.i_abs_max }}A</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="selectedTransistors.length >= 2" class="comparison-results">
      <h3>Comparison Results</h3>
      
      <!-- Summary Statistics -->
      <div class="comparison-summary">
        <div class="summary-card">
          <h4>Electrical Ratings</h4>
          <div class="stat-row">
            <span>Voltage Range:</span>
            <span>{{ Math.min(...selectedTransistors.map(t => t.electrical.v_abs_max)) }}V - {{ Math.max(...selectedTransistors.map(t => t.electrical.v_abs_max)) }}V</span>
          </div>
          <div class="stat-row">
            <span>Current Range:</span>
            <span>{{ Math.min(...selectedTransistors.map(t => t.electrical.i_abs_max)) }}A - {{ Math.max(...selectedTransistors.map(t => t.electrical.i_abs_max)) }}A</span>
          </div>
          <div class="stat-row">
            <span>Temperature Range:</span>
            <span>{{ Math.min(...selectedTransistors.map(t => t.electrical.t_j_max)) }}°C - {{ Math.max(...selectedTransistors.map(t => t.electrical.t_j_max)) }}°C</span>
          </div>
        </div>

        <div class="summary-card">
          <h4>Thermal Properties</h4>
          <div class="stat-row">
            <span>Thermal Resistance Range:</span>
            <span>{{ thermalRange }}</span>
          </div>
          <div class="stat-row">
            <span>Lowest Recorded Case-to-Sink Resistance:</span>
            <span>{{ getLowestRecordedCaseSink()?.metadata.name || 'Unknown' }} ({{ thermalCandidates.length }}/{{ selectedTransistors.length }} known)</span>
          </div>
        </div>

        <div class="summary-card">
          <h4>Recorded Rating Limits</h4>
          <div class="stat-row">
            <span>Highest Voltage:</span>
            <span>{{ getHighestVoltageTransistor().metadata.name }} ({{ getHighestVoltageTransistor().electrical.v_abs_max }}V)</span>
          </div>
          <div class="stat-row">
            <span>Highest Current:</span>
            <span>{{ getHighestCurrentTransistor().metadata.name }} ({{ getHighestCurrentTransistor().electrical.i_abs_max }}A)</span>
          </div>
          <p>Ratings do not establish efficiency, loss, power density, or application performance.</p>
        </div>
      </div>

      <!-- Detailed Comparison Table -->
      <div class="comparison-table">
        <h4>Detailed Comparison</h4>
        <table>
          <thead>
            <tr>
              <th>Property</th>
              <th v-for="transistor in selectedTransistors" :key="transistor.metadata.name">
                {{ transistor.metadata.name }}
              </th>
            </tr>
          </thead>
          <tbody>
            <!-- Metadata -->
            <tr class="section-header">
              <td colspan="100%"><strong>Metadata</strong></td>
            </tr>
            <tr>
              <td>Type</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name">
                <span :class="['type-badge', transistor.metadata.type.toLowerCase()]">
                  {{ transistor.metadata.type }}
                </span>
              </td>
            </tr>
            <tr>
              <td>Manufacturer</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name">
                {{ transistor.metadata.manufacturer }}
              </td>
            </tr>
            <tr>
              <td>Housing Type</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name">
                {{ transistor.metadata.housing_type }}
              </td>
            </tr>

            <!-- Electrical -->
            <tr class="section-header">
              <td colspan="100%"><strong>Electrical Ratings</strong></td>
            </tr>
            <tr>
              <td>Max Voltage (V)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name"
                  :class="{ 'best-value': transistor.electrical.v_abs_max === Math.max(...selectedTransistors.map(t => t.electrical.v_abs_max)) }">
                {{ transistor.electrical.v_abs_max }}V
              </td>
            </tr>
            <tr>
              <td>Max Current (A)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name"
                  :class="{ 'best-value': transistor.electrical.i_abs_max === Math.max(...selectedTransistors.map(t => t.electrical.i_abs_max)) }">
                {{ transistor.electrical.i_abs_max }}A
              </td>
            </tr>
            <tr>
              <td>Continuous Current (A)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name"
                  :class="{ 'best-value': transistor.electrical.i_cont === Math.max(...selectedTransistors.map(t => t.electrical.i_cont)) }">
                {{ transistor.electrical.i_cont }}A
              </td>
            </tr>
            <tr>
              <td>Max Junction Temp (°C)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name"
                  :class="{ 'best-value': transistor.electrical.t_j_max === Math.max(...selectedTransistors.map(t => t.electrical.t_j_max)) }">
                {{ transistor.electrical.t_j_max }}°C
              </td>
            </tr>
            <tr>
              <td>Power Rating (W)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name">Unavailable · no recorded power rating</td>
            </tr>

            <!-- Thermal -->
            <tr class="section-header">
              <td colspan="100%"><strong>Thermal Properties</strong></td>
            </tr>
            <tr>
              <td>Thermal Resistance (K/W)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name"
                  :class="{ 'best-value': knownNumber(transistor.thermal.r_th_cs) && transistor.thermal.r_th_cs === getLowestRecordedCaseSink()?.thermal.r_th_cs }">
                {{ formatNumber(transistor.thermal.r_th_cs) }}
              </td>
            </tr>
            <tr>
              <td>Housing Area (cm²)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name">
                {{ transistor.thermal.housing_area }}
              </td>
            </tr>
            <tr>
              <td>Cooling Area (cm²)</td>
              <td v-for="transistor in selectedTransistors" :key="transistor.metadata.name">
                {{ transistor.thermal.cooling_area }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Visual Comparison Charts -->
      <div class="comparison-charts">
        <h4>Visual Comparison</h4>
        <div class="charts-grid">
          <div class="chart-container">
            <canvas id="voltageCurrentChart" width="400" height="300"></canvas>
            <p>Voltage vs Current Rating</p>
          </div>
          <p>Power and performance charts are unavailable. The core record has no power rating or conditioned performance estimate.</p>
        </div>
      </div>

      <p>Use Advanced Compare for recorded curves and estimates at explicit operating conditions.</p>
    </div>

    <div v-else-if="selectedTransistors.length === 1" class="single-selection">
      <p>Select at least one more transistor to start comparison.</p>
    </div>

    <div v-else class="no-selection">
      <p>Select at least 2 transistors to compare their specifications.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onBeforeUnmount, nextTick } from 'vue'
import Chart from 'chart.js/auto'

const props = defineProps({
  transistors: Array
})

// Reactive data
const selectedTransistors = ref([])
const searchQuery = ref('')
const typeFilter = ref('')
const manufacturerFilter = ref('')
const maxComparisons = 5

// Chart instances
let voltageCurrentChart = null
onBeforeUnmount(() => voltageCurrentChart?.destroy())
function knownNumber(value) { return typeof value === 'number' && Number.isFinite(value) }
function formatNumber(value) { return knownNumber(value) ? value.toFixed(3) : 'Unknown' }
const thermalCandidates = computed(() => selectedTransistors.value.filter(device => knownNumber(device.thermal.r_th_cs)))
const thermalRange = computed(() => {
  if (thermalCandidates.value.length !== selectedTransistors.value.length) return 'Unavailable · one or more recorded values are unknown'
  const values = thermalCandidates.value.map(device => device.thermal.r_th_cs)
  return `${Math.min(...values).toFixed(2)} - ${Math.max(...values).toFixed(2)} K/W`
})

// Computed properties
const availableTypes = computed(() => {
  return [...new Set(props.transistors.map(t => t.metadata.type))].sort()
})

const availableManufacturers = computed(() => {
  return [...new Set(props.transistors.map(t => t.metadata.manufacturer))].sort()
})

const filteredTransistors = computed(() => {
  return props.transistors.filter(transistor => {
    const matchesSearch = transistor.metadata.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
                         transistor.metadata.manufacturer.toLowerCase().includes(searchQuery.value.toLowerCase())
    const matchesType = !typeFilter.value || transistor.metadata.type === typeFilter.value
    const matchesManufacturer = !manufacturerFilter.value || transistor.metadata.manufacturer === manufacturerFilter.value
    
    return matchesSearch && matchesType && matchesManufacturer
  })
})

// Watchers
watch(selectedTransistors, () => {
  if (selectedTransistors.value.length >= 2) {
    nextTick(() => {
      updateComparisonCharts()
    })
  }
}, { deep: true })

// Methods
function toggleTransistorSelection(transistor) {
  const index = selectedTransistors.value.findIndex(t => t.metadata.name === transistor.metadata.name)
  
  if (index >= 0) {
    // Remove if already selected
    selectedTransistors.value.splice(index, 1)
  } else if (selectedTransistors.value.length < maxComparisons) {
    // Add if not at max capacity
    selectedTransistors.value.push(transistor)
  }
}

function isSelected(transistor) {
  return selectedTransistors.value.some(t => t.metadata.name === transistor.metadata.name)
}

function clearSelection() {
  selectedTransistors.value = []
}

function getLowestRecordedCaseSink() {
  return thermalCandidates.value.reduce((lowest, current) =>
    lowest === null || current.thermal.r_th_cs < lowest.thermal.r_th_cs ? current : lowest, null)
}

function getHighestVoltageTransistor() {
  return selectedTransistors.value.reduce((highest, current) => 
    current.electrical.v_abs_max > highest.electrical.v_abs_max ? current : highest
  )
}

function getHighestCurrentTransistor() {
  return selectedTransistors.value.reduce((highest, current) => 
    current.electrical.i_abs_max > highest.electrical.i_abs_max ? current : highest
  )
}

function updateComparisonCharts() {
  updateVoltageCurrentChart()
}

function updateVoltageCurrentChart() {
  const canvas = document.getElementById('voltageCurrentChart')
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  if (voltageCurrentChart) {
    voltageCurrentChart.destroy()
  }
  
  const datasets = selectedTransistors.value.map((transistor, index) => ({
    label: transistor.metadata.name,
    data: [{
      x: transistor.electrical.v_abs_max,
      y: transistor.electrical.i_abs_max
    }],
    backgroundColor: getChartColor(index),
    borderColor: getChartColor(index),
    pointRadius: 8
  }))
  
  voltageCurrentChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Voltage vs Current Rating'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Max Voltage (V)'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Max Current (A)'
          }
        }
      }
    }
  })
}

function getChartColor(index, alpha = 1) {
  const colors = [
    `rgba(255, 99, 132, ${alpha})`,
    `rgba(54, 162, 235, ${alpha})`,
    `rgba(255, 205, 86, ${alpha})`,
    `rgba(75, 192, 192, ${alpha})`,
    `rgba(153, 102, 255, ${alpha})`
  ]
  return colors[index % colors.length]
}

async function exportComparison() {
  try {
    const comparisonData = {
      selectedTransistors: selectedTransistors.value,
      summary: {
        voltageRange: `${Math.min(...selectedTransistors.value.map(t => t.electrical.v_abs_max))}V - ${Math.max(...selectedTransistors.value.map(t => t.electrical.v_abs_max))}V`,
        currentRange: `${Math.min(...selectedTransistors.value.map(t => t.electrical.i_abs_max))}A - ${Math.max(...selectedTransistors.value.map(t => t.electrical.i_abs_max))}A`,
        lowestRecordedCaseToSinkResistance: getLowestRecordedCaseSink()?.metadata.name ?? null,
        thermalCoverage: { known_count: thermalCandidates.value.length, total_count: selectedTransistors.value.length },
        ratedPowerW: null,
        ratedPowerReason: "The core record has no power rating",
        applicationPerformance: null,
        highestVoltage: getHighestVoltageTransistor().metadata.name,
        highestCurrent: getHighestCurrentTransistor().metadata.name
      },
      timestamp: new Date().toISOString()
    }
    
    const blob = new Blob([JSON.stringify(comparisonData, null, 2)], { type: 'application/json' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', `transistor_comparison_${new Date().toISOString().split('T')[0]}.json`)
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  } catch (error) {
    alert('Export failed: ' + error.message)
  }
}

</script>

<style scoped>
.transistor-comparison {
  max-width: 1400px;
  margin: 0 auto;
  padding: .85rem;
}

.comparison-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: .65rem;
  padding: 1rem;
  background: var(--bg-secondary);
  border-radius: 5px;
  border: 1px solid var(--border-color);
}

.header-actions {
  display: flex;
  gap: .55rem;
}

.selection-section {
  background: var(--bg-secondary);
  padding: .75rem;
  border-radius: 5px;
  border: 1px solid var(--border-color);
  margin-bottom: .65rem;
}

.selector-filters {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: .55rem;
  margin-bottom: .65rem;
}

.search-input, .filter-select {
  padding: 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: 4px;
}

.transistor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 250px), 1fr));
  gap: .55rem;
  max-height: 400px;
  overflow-y: auto;
}

.transistor-card {
  border: 2px solid var(--border-color);
  border-radius: 5px;
  padding: 1rem;
  cursor: pointer;
  transition: all 0.3s;
}

.transistor-card:hover {
  border-color: var(--accent-blue);
}

.transistor-card.selected {
  border-color: var(--accent-blue);
  background: var(--accent-blue-light);
}

.transistor-card.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.card-header h4 {
  margin: 0;
  font-size: 1rem;
  color: var(--text-primary);
}

.selection-indicator {
  font-size: .85rem;
}

.card-content p {
  margin: 0.25rem 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.comparison-results {
  background: var(--bg-secondary);
  padding: .75rem;
  border-radius: 5px;
  border: 1px solid var(--border-color);
}

.comparison-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: .55rem;
  margin-bottom: .65rem;
}

.summary-card {
  background: var(--bg-tertiary);
  padding: 1.5rem;
  border-radius: 5px;
  border-left: 4px solid var(--accent-blue);
}

.summary-card h4 {
  margin: 0 0 1rem 0;
  color: var(--text-primary);
}

.stat-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-color);
}

.stat-row:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.comparison-table {
  margin: 2rem 0;
}

.comparison-table table {
  width: 100%;
  border-collapse: collapse;
  background: var(--bg-secondary);
  border-radius: 5px;
  overflow: hidden;
  border: 1px solid var(--border-color);
}

.comparison-table th,
.comparison-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.comparison-table th {
  background: var(--bg-tertiary);
  font-weight: 600;
  position: sticky;
  top: 0;
}

.section-header td {
  background: #e9ecef;
  font-weight: 600;
  color: #495057;
}

.best-value {
  background: #d4edda;
  font-weight: 600;
  color: #155724;
}

.type-badge {
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  font-size: 0.8rem;
  font-weight: 500;
  text-transform: uppercase;
}

.type-badge.igbt {
  background: #e3f2fd;
  color: #1976d2;
}

.type-badge.mosfet {
  background: #f3e5f5;
  color: #7b1fa2;
}

.type-badge.diode {
  background: #e8f5e8;
  color: #388e3c;
}

.comparison-charts {
  margin: 2rem 0;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
  margin-top: 1rem;
}

.chart-container {
  text-align: center;
}

.chart-container canvas {
  max-width: 100%;
  height: auto;
}

.chart-container p {
  margin-top: 0.5rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.similar-suggestions {
  margin-top: 2rem;
  padding: 1.5rem;
  background: var(--bg-tertiary);
  border-radius: 5px;
}

.similar-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: .55rem;
  margin-top: 1rem;
}

.similar-card {
  background: var(--bg-secondary);
  padding: 1rem;
  border-radius: 4px;
  border-left: 4px solid #ffc107;
}

.similar-card h5 {
  margin: 0 0 0.5rem 0;
  color: var(--text-primary);
}

.similar-card p {
  margin: 0.25rem 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.similarity-score {
  font-weight: 600;
  color: var(--text-primary);
}

.no-selection, .single-selection {
  text-align: center;
  padding: .85rem;
  color: var(--text-secondary);
  font-size: .85rem;
  background: var(--bg-secondary);
  border-radius: 5px;
  border: 1px solid var(--border-color);
}

.btn {
  padding: .3rem .6rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s;
  text-decoration: none;
  display: inline-block;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--accent-blue);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--action-hover);
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.btn-secondary:hover:not(:disabled) {
  border-color: var(--accent-blue);
}

@media (max-width: 768px) {
  .comparison-header {
    flex-direction: column;
    gap: .55rem;
  }

  .selector-filters {
    grid-template-columns: 1fr;
  }

  .transistor-grid {
    grid-template-columns: 1fr;
  }

  .comparison-summary {
    grid-template-columns: 1fr;
  }

  .charts-grid {
    grid-template-columns: 1fr;
  }
}
.comparison-header h2 { font-size: 1.25rem; }
.selection-section h3, .comparison-results h3 { font-size: 1rem; }
.card-header { flex-wrap: wrap; gap: .3rem .55rem; }
.card-header h4 { font: 600 .82rem ui-monospace, monospace; overflow-wrap: anywhere; }
.transistor-card { padding: .55rem; border-width: 1px; }
.transistor-card:focus-visible { outline: 2px solid var(--accent-blue); outline-offset: -2px; }
.card-content { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: .45rem; }
.card-content p { font-size: .8rem; }
.comparison-table { overflow-x: auto; }
.comparison-table table { min-width: 32rem; }
.btn-primary { background: var(--action-bg); border: 1px solid var(--action-bg); }
</style>
