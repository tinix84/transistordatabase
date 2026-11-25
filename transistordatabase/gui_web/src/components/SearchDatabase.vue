<template>
  <div class="search-database">
    <div class="search-header">
      <h2>Search Database</h2>
      <div class="header-actions">
        <button @click="resetFilters" class="btn btn-secondary">
          Reset Filters
        </button>
        <button @click="exportResults" :disabled="filteredTransistors.length === 0" class="btn btn-primary">
          Export Results
        </button>
      </div>
    </div>

    <div class="search-content">
      <!-- Filter Panel -->
      <aside class="filter-panel" aria-labelledby="search-filters">
        <h3 id="search-filters">Advanced Filters</h3>
        
        <!-- Basic Filters -->
        <div class="filter-section">
          <h4>Basic Properties</h4>
          <div class="filter-grid">
            <div class="filter-item">
              <label>
                <input type="checkbox" v-model="filters.name.enabled" />
                Name Contains:
              </label>
              <input 
                type="text" 
                v-model="filters.name.value" 
                :disabled="!filters.name.enabled"
                placeholder="Enter name filter..."
              />
            </div>

            <div class="filter-item">
              <label>
                <input type="checkbox" v-model="filters.type.enabled" />
                Type:
              </label>
              <select v-model="filters.type.value" :disabled="!filters.type.enabled">
                <option value="">All Types</option>
                <option v-for="type in availableTypes" :key="type" :value="type">
                  {{ type }}
                </option>
              </select>
            </div>

            <div class="filter-item">
              <label>
                <input type="checkbox" v-model="filters.manufacturer.enabled" />
                Manufacturer:
              </label>
              <select v-model="filters.manufacturer.value" :disabled="!filters.manufacturer.enabled">
                <option value="">All Manufacturers</option>
                <option v-for="mfg in availableManufacturers" :key="mfg" :value="mfg">
                  {{ mfg }}
                </option>
              </select>
            </div>

            <div class="filter-item">
              <label>
                <input type="checkbox" v-model="filters.housing_type.enabled" />
                Housing Type:
              </label>
              <select v-model="filters.housing_type.value" :disabled="!filters.housing_type.enabled">
                <option value="">All Housing Types</option>
                <option v-for="housing in availableHousingTypes" :key="housing" :value="housing">
                  {{ housing }}
                </option>
              </select>
            </div>
          </div>
        </div>

        <!-- Electrical Ratings Filters -->
        <div class="filter-section">
          <h4>Electrical Ratings</h4>
          <div class="filter-grid">
            <div class="filter-item range-filter">
              <label>
                <input type="checkbox" v-model="filters.v_abs_max.enabled" />
                Max Voltage (V):
              </label>
              <div class="range-inputs">
                <input 
                  type="number" 
                  v-model.number="filters.v_abs_max.min" 
                  :disabled="!filters.v_abs_max.enabled"
                  placeholder="Min"
                />
                <span>to</span>
                <input 
                  type="number" 
                  v-model.number="filters.v_abs_max.max" 
                  :disabled="!filters.v_abs_max.enabled"
                  placeholder="Max"
                />
              </div>
            </div>

            <div class="filter-item range-filter">
              <label>
                <input type="checkbox" v-model="filters.i_abs_max.enabled" />
                Max Current (A):
              </label>
              <div class="range-inputs">
                <input 
                  type="number" 
                  v-model.number="filters.i_abs_max.min" 
                  :disabled="!filters.i_abs_max.enabled"
                  placeholder="Min"
                />
                <span>to</span>
                <input 
                  type="number" 
                  v-model.number="filters.i_abs_max.max" 
                  :disabled="!filters.i_abs_max.enabled"
                  placeholder="Max"
                />
              </div>
            </div>

            <div class="filter-item range-filter">
              <label>
                <input type="checkbox" v-model="filters.i_cont.enabled" />
                Continuous Current (A):
              </label>
              <div class="range-inputs">
                <input 
                  type="number" 
                  v-model.number="filters.i_cont.min" 
                  :disabled="!filters.i_cont.enabled"
                  placeholder="Min"
                />
                <span>to</span>
                <input 
                  type="number" 
                  v-model.number="filters.i_cont.max" 
                  :disabled="!filters.i_cont.enabled"
                  placeholder="Max"
                />
              </div>
            </div>

            <div class="filter-item range-filter">
              <label>
                <input type="checkbox" v-model="filters.t_j_max.enabled" />
                Max Junction Temperature (°C):
              </label>
              <div class="range-inputs">
                <input 
                  type="number" 
                  v-model.number="filters.t_j_max.min" 
                  :disabled="!filters.t_j_max.enabled"
                  placeholder="Min"
                />
                <span>to</span>
                <input 
                  type="number" 
                  v-model.number="filters.t_j_max.max" 
                  :disabled="!filters.t_j_max.enabled"
                  placeholder="Max"
                />
              </div>
            </div>
          </div>
        </div>

        <!-- Thermal Properties Filters -->
        <div class="filter-section">
          <h4>Thermal Properties</h4>
          <div class="filter-grid">
            <div class="filter-item range-filter">
              <label>
                <input type="checkbox" v-model="filters.r_th_cs.enabled" />
                Thermal Resistance (K/W):
              </label>
              <div class="range-inputs">
                <input 
                  type="number" 
                  v-model.number="filters.r_th_cs.min" 
                  :disabled="!filters.r_th_cs.enabled"
                  placeholder="Min"
                  step="0.001"
                />
                <span>to</span>
                <input 
                  type="number" 
                  v-model.number="filters.r_th_cs.max" 
                  :disabled="!filters.r_th_cs.enabled"
                  placeholder="Max"
                  step="0.001"
                />
              </div>
            </div>

            <div class="filter-item range-filter">
              <label>
                <input type="checkbox" v-model="filters.housing_area.enabled" />
                Housing Area (m²):
              </label>
              <div class="range-inputs">
                <input 
                  type="number" 
                  v-model.number="filters.housing_area.min" 
                  :disabled="!filters.housing_area.enabled"
                  placeholder="Min"
                  step="0.000001"
                />
                <span>to</span>
                <input 
                  type="number" 
                  v-model.number="filters.housing_area.max" 
                  :disabled="!filters.housing_area.enabled"
                  placeholder="Max"
                  step="0.000001"
                />
              </div>
            </div>
          </div>
        </div>

        <!-- Quick Filters -->
        <div class="filter-section">
          <h4>Quick Filters</h4>
          <div class="quick-filters">
            <button @click="applyQuickFilter('high_voltage')" class="btn btn-quick">
              High Voltage (>1000V)
            </button>
            <button @click="applyQuickFilter('high_current')" class="btn btn-quick">
              High Current (>100A)
            </button>
            <button @click="applyQuickFilter('sic_mosfet')" class="btn btn-quick">
              SiC MOSFETs
            </button>
            <button @click="applyQuickFilter('igbt')" class="btn btn-quick">
              IGBTs
            </button>
            <button @click="applyQuickFilter('low_thermal')" class="btn btn-quick">
              Low Thermal Resistance
            </button>
          </div>
        </div>
      </aside>

      <!-- Results Panel -->
      <div class="results-panel">
        <div class="results-header">
          <h3>Search Results ({{ filteredTransistors.length }} devices)</h3>
          <div class="view-controls">
            <button 
              @click="viewMode = 'table'" 
              :class="{ active: viewMode === 'table' }"
              :aria-pressed="viewMode === 'table'"
              class="btn btn-sm"
            >
              Table
            </button>
            <button 
              @click="viewMode = 'cards'" 
              :class="{ active: viewMode === 'cards' }"
              :aria-pressed="viewMode === 'cards'"
              class="btn btn-sm"
            >
              Cards
            </button>
          </div>
        </div>

        <!-- Table View -->
        <div v-if="viewMode === 'table'" class="table-view" tabindex="0" role="region" aria-label="Device results table">
          <table class="results-table">
            <thead>
              <tr>
                <th @click="sortBy('name')" class="sortable">
                  Name {{ getSortIcon('name') }}
                </th>
                <th @click="sortBy('manufacturer')" class="sortable">
                  Manufacturer {{ getSortIcon('manufacturer') }}
                </th>
                <th @click="sortBy('type')" class="sortable">
                  Type {{ getSortIcon('type') }}
                </th>
                <th @click="sortBy('v_abs_max')" class="sortable">
                  Max V {{ getSortIcon('v_abs_max') }}
                </th>
                <th @click="sortBy('i_abs_max')" class="sortable">
                  Max I {{ getSortIcon('i_abs_max') }}
                </th>
                <th @click="sortBy('r_th_cs')" class="sortable">
                  R_th {{ getSortIcon('r_th_cs') }}
                </th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="transistor in paginatedTransistors"
                :key="transistor.metadata.name"
                data-testid="transistor-row"
                :data-transistor-name="transistor.metadata.name"
              >
                <td class="transistor-name">{{ transistor.metadata.name }}</td>
                <td>{{ transistor.metadata.manufacturer }}</td>
                <td>
                  <span :class="['type-badge', transistor.metadata.type.toLowerCase()]">
                    {{ transistor.metadata.type }}
                  </span>
                </td>
                <td>{{ transistor.electrical.v_abs_max }}V</td>
                <td>{{ transistor.electrical.i_abs_max }}A</td>
                <td>{{ formatNumber(transistor.thermal.r_th_cs) }} K/W</td>
                <td class="actions">
                  <button @click="loadToExportingTools(transistor)" class="btn btn-sm">
                    Export
                  </button>
                  <button @click="loadToComparison(transistor)" class="btn btn-sm">
                    Compare
                  </button>
                  <button @click="loadToTopology(transistor)" class="btn btn-sm">
                    Topology
                  </button>
                  <button @click="emit('edit-transistor', transistor)" class="btn btn-sm">
                    Edit
                  </button>
                  <button @click="emit('delete-transistor', transistor)" class="btn btn-sm row-delete">
                    Delete
                  </button>
                </td>
              </tr>
            </tbody>
          </table>

          <!-- Pagination -->
          <div class="pagination" v-if="totalPages > 1">
            <button @click="currentPage = 1" :disabled="currentPage === 1" class="btn btn-sm">
              « First
            </button>
            <button @click="currentPage--" :disabled="currentPage === 1" class="btn btn-sm">
              ‹ Prev
            </button>
            <span class="page-info">
              Page {{ currentPage }} of {{ totalPages }}
            </span>
            <button @click="currentPage++" :disabled="currentPage === totalPages" class="btn btn-sm">
              Next ›
            </button>
            <button @click="currentPage = totalPages" :disabled="currentPage === totalPages" class="btn btn-sm">
              Last »
            </button>
          </div>
        </div>

        <!-- Cards View -->
        <div v-if="viewMode === 'cards'" class="cards-view">
          <div class="transistor-cards">
            <div 
              v-for="transistor in paginatedTransistors" 
              :key="transistor.metadata.name"
              class="transistor-card"
              data-testid="transistor-card"
              :data-transistor-name="transistor.metadata.name"
            >
              <div class="card-header">
                <h4>{{ transistor.metadata.name }}</h4>
                <span :class="['type-badge', transistor.metadata.type.toLowerCase()]">
                  {{ transistor.metadata.type }}
                </span>
              </div>
              <div class="card-content">
                <div class="card-info">
                  <p><strong>Manufacturer:</strong> {{ transistor.metadata.manufacturer }}</p>
                  <p><strong>Housing:</strong> {{ transistor.metadata.housing_type }}</p>
                  <p><strong>Max Voltage:</strong> {{ transistor.electrical.v_abs_max }}V</p>
                  <p><strong>Max Current:</strong> {{ transistor.electrical.i_abs_max }}A</p>
                  <p><strong>Thermal Resistance:</strong> {{ formatNumber(transistor.thermal.r_th_cs) }} K/W</p>
                </div>
                <div class="card-actions">
                  <button @click="loadToExportingTools(transistor)" class="btn btn-sm">
                    Export
                  </button>
                  <button @click="loadToComparison(transistor)" class="btn btn-sm">
                    Compare
                  </button>
                  <button @click="loadToTopology(transistor)" class="btn btn-sm">
                    Topology
                  </button>
                  <button @click="emit('edit-transistor', transistor)" class="btn btn-sm">
                    Edit
                  </button>
                  <button @click="emit('delete-transistor', transistor)" class="btn btn-sm row-delete">
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'

const props = defineProps({
  transistors: Array
})

const emit = defineEmits([
  'load-to-exporting',
  'load-to-comparison',
  'load-to-topology',
  'edit-transistor',
  'delete-transistor'
])

// Debug logging
onMounted(() => {
  console.log('[SearchDatabase] Mounted')
  console.log('[SearchDatabase] props.transistors:', props.transistors)
  console.log('[SearchDatabase] transistors type:', typeof props.transistors, 'Is Array:', Array.isArray(props.transistors))
  console.log('[SearchDatabase] transistors length:', props.transistors?.length)
})

// Watch for prop changes
watch(() => props.transistors, (newVal, oldVal) => {
  console.log('[SearchDatabase] transistors prop changed')
  console.log('[SearchDatabase] Old length:', oldVal?.length, 'New length:', newVal?.length)
  console.log('[SearchDatabase] New value sample:', newVal?.[0])
})

// Filter state
const filters = ref({
  name: { enabled: false, value: '' },
  type: { enabled: false, value: '' },
  manufacturer: { enabled: false, value: '' },
  housing_type: { enabled: false, value: '' },
  v_abs_max: { enabled: false, min: null, max: null },
  i_abs_max: { enabled: false, min: null, max: null },
  i_cont: { enabled: false, min: null, max: null },
  t_j_max: { enabled: false, min: null, max: null },
  r_th_cs: { enabled: false, min: null, max: null },
  housing_area: { enabled: false, min: null, max: null }
})

// View state
const viewMode = ref('table')
const sortField = ref('')
const sortDirection = ref('asc')
const currentPage = ref(1)
const itemsPerPage = ref(20)

// Computed properties for filter options
const availableTypes = computed(() => {
  if (!props.transistors || !Array.isArray(props.transistors)) {
    console.warn('[SearchDatabase] availableTypes: transistors is not an array:', props.transistors)
    return []
  }
  return [...new Set(props.transistors.map(t => t.metadata.type))].sort()
})

const availableManufacturers = computed(() => {
  if (!props.transistors || !Array.isArray(props.transistors)) {
    console.warn('[SearchDatabase] availableManufacturers: transistors is not an array:', props.transistors)
    return []
  }
  return [...new Set(props.transistors.map(t => t.metadata.manufacturer))].sort()
})

const availableHousingTypes = computed(() => {
  if (!props.transistors || !Array.isArray(props.transistors)) {
    console.warn('[SearchDatabase] availableHousingTypes: transistors is not an array:', props.transistors)
    return []
  }
  return [...new Set(props.transistors.map(t => t.metadata.housing_type))].sort()
})

// Filtered and sorted transistors
const filteredTransistors = computed(() => {
  if (!props.transistors || !Array.isArray(props.transistors)) {
    console.warn('[SearchDatabase] filteredTransistors: transistors is not an array:', props.transistors)
    return []
  }
  console.log('[SearchDatabase] filteredTransistors computing with', props.transistors.length, 'transistors')
  let result = props.transistors.filter(transistor => {
    // Name filter
    if (filters.value.name.enabled) {
      if (!transistor.metadata.name.toLowerCase().includes(filters.value.name.value.toLowerCase())) {
        return false
      }
    }

    // Type filter
    if (filters.value.type.enabled && filters.value.type.value) {
      if (transistor.metadata.type !== filters.value.type.value) {
        return false
      }
    }

    // Manufacturer filter
    if (filters.value.manufacturer.enabled && filters.value.manufacturer.value) {
      if (transistor.metadata.manufacturer !== filters.value.manufacturer.value) {
        return false
      }
    }

    // Housing type filter
    if (filters.value.housing_type.enabled && filters.value.housing_type.value) {
      if (transistor.metadata.housing_type !== filters.value.housing_type.value) {
        return false
      }
    }

    // Range filters
    const rangeFilters = ['v_abs_max', 'i_abs_max', 'i_cont', 't_j_max', 'r_th_cs', 'housing_area']
    for (const field of rangeFilters) {
      const filter = filters.value[field]
      if (filter.enabled) {
        let value
        if (field === 'housing_area') {
          value = transistor.thermal[field]
        } else if (field === 'r_th_cs') {
          value = transistor.thermal[field]
        } else {
          value = transistor.electrical[field]
        }

        if (typeof value !== 'number' || !Number.isFinite(value)) return false
        if (filter.min !== null && value < filter.min) return false
        if (filter.max !== null && value > filter.max) return false
      }
    }

    return true
  })

  // Apply sorting
  if (sortField.value) {
    result.sort((a, b) => {
      let aVal, bVal
      
      if (sortField.value === 'name' || sortField.value === 'manufacturer' || sortField.value === 'type' || sortField.value === 'housing_type') {
        aVal = a.metadata[sortField.value]
        bVal = b.metadata[sortField.value]
      } else if (sortField.value === 'r_th_cs' || sortField.value === 'housing_area') {
        aVal = a.thermal[sortField.value]
        bVal = b.thermal[sortField.value]
      } else {
        aVal = a.electrical[sortField.value]
        bVal = b.electrical[sortField.value]
      }

      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase()
        bVal = bVal.toLowerCase()
      }

      if (aVal < bVal) return sortDirection.value === 'asc' ? -1 : 1
      if (aVal > bVal) return sortDirection.value === 'asc' ? 1 : -1
      return 0
    })
  }

  return result
})

// Pagination
const totalPages = computed(() => Math.ceil(filteredTransistors.value.length / itemsPerPage.value))

const paginatedTransistors = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage.value
  const end = start + itemsPerPage.value
  return filteredTransistors.value.slice(start, end)
})

// Methods
function formatNumber(value, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'Unknown'
}

function resetFilters() {
  filters.value = {
    name: { enabled: false, value: '' },
    type: { enabled: false, value: '' },
    manufacturer: { enabled: false, value: '' },
    housing_type: { enabled: false, value: '' },
    v_abs_max: { enabled: false, min: null, max: null },
    i_abs_max: { enabled: false, min: null, max: null },
    i_cont: { enabled: false, min: null, max: null },
    t_j_max: { enabled: false, min: null, max: null },
    r_th_cs: { enabled: false, min: null, max: null },
    housing_area: { enabled: false, min: null, max: null }
  }
  currentPage.value = 1
}

function applyQuickFilter(filterType) {
  resetFilters()
  
  switch (filterType) {
    case 'high_voltage':
      filters.value.v_abs_max.enabled = true
      filters.value.v_abs_max.min = 1000
      break
    case 'high_current':
      filters.value.i_abs_max.enabled = true
      filters.value.i_abs_max.min = 100
      break
    case 'sic_mosfet':
      filters.value.type.enabled = true
      filters.value.type.value = 'SiC-MOSFET'
      break
    case 'igbt':
      filters.value.type.enabled = true
      filters.value.type.value = 'IGBT'
      break
    case 'low_thermal':
      filters.value.r_th_cs.enabled = true
      filters.value.r_th_cs.max = 0.5
      break
  }
}

function sortBy(field) {
  if (sortField.value === field) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortField.value = field
    sortDirection.value = 'asc'
  }
  currentPage.value = 1
}

function getSortIcon(field) {
  if (sortField.value !== field) return ''
  return sortDirection.value === 'asc' ? '↑' : '↓'
}

function loadToExportingTools(transistor) {
  emit('load-to-exporting', transistor)
}

function loadToComparison(transistor) {
  emit('load-to-comparison', transistor)
}

function loadToTopology(transistor) {
  emit('load-to-topology', transistor)
}

function exportResults() {
  const data = {
    filters: filters.value,
    results: filteredTransistors.value,
    totalCount: filteredTransistors.value.length,
    exportDate: new Date().toISOString()
  }
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', `transistor_search_results_${new Date().toISOString().split('T')[0]}.json`)
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
}

// Reset page when filters change
watch(filters, () => {
  currentPage.value = 1
}, { deep: true })
</script>

<style scoped>
.search-database {
  max-width: 1680px;
  margin: 0 auto;
  padding: .85rem;
  font-size: .85rem;
}

.search-header,
.results-header,
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: .55rem;
}

.search-header { margin-bottom: .7rem; }
.search-header h2 { font-size: 1.25rem; }
.header-actions, .view-controls, .card-actions {
  display: flex;
  flex-wrap: wrap;
  gap: .35rem;
}

.search-content {
  display: grid;
  grid-template-columns: clamp(210px, 20vw, 278px) minmax(0, 1fr);
  gap: .7rem;
  align-items: start;
}

.filter-panel, .results-panel {
  min-width: 0;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 5px;
}

.filter-panel h3, .results-header {
  padding: .5rem .7rem;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-tertiary);
}

.filter-panel h3, .results-header h3 { font-size: .85rem; }

.filter-section { padding: .55rem .7rem .7rem; }
.filter-section + .filter-section { border-top: 1px solid var(--border-color); }

.filter-section h4 {
  margin-bottom: .4rem;
  color: var(--text-secondary);
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .045em;
  text-transform: uppercase;
}

.filter-grid { display: grid; gap: .45rem; }
.filter-item { display: grid; gap: .18rem; }

.filter-item label {
  display: flex;
  align-items: center;
  gap: .4rem;
  font-weight: 500;
}

.filter-item input:not([type="checkbox"]), .filter-item select {
  min-width: 0;
  width: 100%;
  padding: .22rem .35rem;
  font: inherit;
}

.filter-item input:disabled, .filter-item select:disabled {
  background: var(--bg-tertiary);
  color: var(--text-muted);
}

.range-inputs {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: center;
  gap: .35rem;
  color: var(--text-secondary);
}

.quick-filters {
  display: flex;
  flex-wrap: wrap;
  gap: .3rem;
}

.search-database .btn { padding: .29rem .55rem; font-size: .78rem; }
.search-database .btn-sm, .search-database .btn-quick { padding: .18rem .36rem; }
.search-database .btn-primary { background: var(--action-bg); border-color: var(--action-bg); }
.search-database .btn-primary:hover:not(:disabled) { background: var(--action-hover); }
.search-database .btn.active {
  background: var(--accent-blue-light);
  color: var(--accent-blue);
}

.table-view { overflow-x: auto; }
.results-table { width: 100%; font-variant-numeric: tabular-nums; }

.results-table th, .results-table td {
  border: 0;
  border-bottom: 1px solid var(--border-color);
  padding: .42rem .55rem;
  text-align: left;
  vertical-align: middle;
}

.results-table th { background: var(--bg-tertiary); white-space: nowrap; }
.results-table th.sortable { cursor: pointer; user-select: none; }
.results-table th.sortable:hover, .results-table tbody tr:hover { background: var(--accent-blue-light); }
.results-table td:nth-child(n+4):nth-child(-n+6) { white-space: nowrap; }

.transistor-name {
  color: var(--accent-blue);
  font: 600 .8rem ui-monospace, monospace;
  overflow-wrap: anywhere;
}

.type-badge {
  display: inline-block;
  padding: .05rem .3rem;
  border: 1px solid var(--border-color);
  border-radius: 3px;
  background: var(--bg-tertiary);
  font-size: .72rem;
  text-transform: uppercase;
  white-space: nowrap;
}

.actions { min-width: 12rem; }
.actions .btn { margin: .1rem .15rem .1rem 0; }
.row-delete { color: var(--danger-color); }

.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: .45rem;
  padding: .5rem;
}

.page-info, .card-info { color: var(--text-secondary); }
.cards-view { padding: .7rem; }

.transistor-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 245px), 1fr));
  gap: .55rem;
}

.transistor-card {
  min-width: 0;
  padding: .65rem;
  border: 1px solid var(--border-color);
  border-radius: 4px;
}

.transistor-card:hover { border-color: var(--accent-blue); }
.card-header h4 { overflow-wrap: anywhere; }
.card-content { display: grid; gap: .5rem; margin-top: .35rem; }

@media (max-width: 760px) {
  .search-content { grid-template-columns: minmax(0, 1fr); }
  .filter-panel { max-height: min(42vh, 25rem); overflow: auto; }
}

@media (pointer: coarse) {
  .search-database .btn { min-height: 32px; }
}
</style>
