<template>
  <div class="batch-export-panel">
    <div class="panel-header">
      <h2>📦 Batch Export</h2>
      <p class="subtitle">Export multiple transistors at once</p>
    </div>

    <!-- Selection Summary -->
    <div class="selection-summary">
      <div class="summary-card">
        <div class="summary-icon">✅</div>
        <div class="summary-content">
          <div class="summary-number">{{ selectedCount }}</div>
          <div class="summary-label">Transistors Selected</div>
        </div>
      </div>

      <div class="summary-card">
        <div class="summary-icon">📊</div>
        <div class="summary-content">
          <div class="summary-number">{{ estimatedSize }}</div>
          <div class="summary-label">Estimated Size</div>
        </div>
      </div>

      <div class="summary-card">
        <div class="summary-icon">⏱️</div>
        <div class="summary-content">
          <div class="summary-number">{{ estimatedTime }}</div>
          <div class="summary-label">Estimated Time</div>
        </div>
      </div>
    </div>

    <!-- Transistor Selection -->
    <div class="selection-section">
      <div class="selection-header">
        <h3>Select Transistors</h3>
        <div class="selection-actions">
          <button @click="selectAll" class="btn btn-sm btn-secondary">
            Select All
          </button>
          <button @click="clearSelection" class="btn btn-sm btn-secondary">
            Clear Selection
          </button>
        </div>
      </div>

      <!-- Filter -->
      <div class="filters">
        <input
          v-model="searchQuery"
          type="text"
          placeholder="🔍 Search transistors..."
          class="search-input"
        />
        <select v-model="typeFilter" class="filter-select">
          <option value="">All Types</option>
          <option v-for="type in availableTypes" :key="type" :value="type">
            {{ type }}
          </option>
        </select>
      </div>

      <!-- Transistor List -->
      <div class="transistor-list">
        <div
          v-for="(transistor, index) in filteredTransistors"
          :key="index"
          @click="toggleSelection(transistor)"
          :class="['transistor-item', { selected: isSelected(transistor) }]"
        >
          <div class="checkbox">
            <input
              type="checkbox"
              :checked="isSelected(transistor)"
              @click.stop="toggleSelection(transistor)"
            />
          </div>
          <div class="transistor-info">
            <div class="transistor-name">{{ transistor.metadata.name }}</div>
            <div class="transistor-specs">
              <span>{{ transistor.metadata.type }}</span>
              <span>•</span>
              <span>{{ transistor.electrical.v_abs_max }}V</span>
              <span>•</span>
              <span>{{ transistor.electrical.i_abs_max }}A</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Export Options -->
    <div class="export-options">
      <h3>Export Settings</h3>

      <div class="options-grid">
        <div class="option-group">
          <label>Export Format</label>
          <select v-model="exportFormat" class="format-select">
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
            <option value="spice">SPICE</option>
            <option value="plecs">PLECS XML</option>
            <option value="matlab">MATLAB (.mat)</option>
            <option value="gecko">GeckoCIRCUITS</option>
            <option value="ltspice">LTspice</option>
          </select>
        </div>

        <div class="option-group">
          <label>Archive Name</label>
          <input
            v-model="archiveName"
            type="text"
            placeholder="transistor_exports"
            class="text-input"
          />
        </div>
      </div>

      <!-- Format-specific options (expandable) -->
      <details v-if="exportFormat === 'plecs'" class="format-options">
        <summary>PLECS Options</summary>
        <div class="options-content">
          <label class="checkbox-label">
            <input type="checkbox" v-model="plecsOptions.recheck" />
            Recheck data before export
          </label>
          <label class="checkbox-label">
            <input type="checkbox" v-model="plecsOptions.include_thermal" />
            Include thermal model
          </label>
          <label class="checkbox-label">
            <input type="checkbox" v-model="plecsOptions.include_losses" />
            Include loss data
          </label>
        </div>
      </details>
    </div>

    <!-- Progress (when exporting) -->
    <div v-if="isExporting" class="progress-section">
      <div class="progress-header">
        <h3>⏳ Exporting...</h3>
        <p>{{ exportProgress.current }} / {{ exportProgress.total }}</p>
      </div>
      <div class="progress-bar">
        <div
          class="progress-fill"
          :style="{ width: `${(exportProgress.current / exportProgress.total) * 100}%` }"
        ></div>
      </div>
      <div class="progress-message">
        {{ exportProgress.message }}
      </div>
    </div>

    <!-- Error Display -->
    <div v-if="error" class="error-message">
      <span class="error-icon">⚠️</span>
      <div>
        <strong>Export Failed</strong>
        <p>{{ error }}</p>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="action-buttons">
      <button @click="$emit('close')" class="btn btn-secondary">
        Cancel
      </button>
      <button
        @click="startExport"
        :disabled="selectedCount === 0 || isExporting"
        class="btn btn-primary"
      >
        {{ isExporting ? '⏳ Exporting...' : `💾 Export ${selectedCount} Transistor${selectedCount !== 1 ? 's' : ''}` }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { transistorApi } from '../services/api.js'

const emit = defineEmits(['close', 'export-complete'])

// State
const allTransistors = ref([])
const selectedTransistors = ref([])
const searchQuery = ref('')
const typeFilter = ref('')
const exportFormat = ref('json')
const archiveName = ref('transistor_exports')
const isExporting = ref(false)
const error = ref(null)
const exportProgress = ref({
  current: 0,
  total: 0,
  message: ''
})

// PLECS-specific options
const plecsOptions = ref({
  recheck: true,
  include_thermal: true,
  include_losses: true
})

// Computed
const selectedCount = computed(() => selectedTransistors.value.length)

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

const estimatedSize = computed(() => {
  if (selectedCount.value === 0) return '0 KB'

  // Rough estimates per format
  const sizePerTransistor = {
    json: 50,  // KB
    csv: 2,
    spice: 5,
    plecs: 30,
    matlab: 40,
    gecko: 25,
    ltspice: 10
  }

  const totalKB = (sizePerTransistor[exportFormat.value] || 20) * selectedCount.value
  if (totalKB < 1024) return `${totalKB} KB`
  return `${(totalKB / 1024).toFixed(1)} MB`
})

const estimatedTime = computed(() => {
  if (selectedCount.value === 0) return '0s'

  // Rough time estimates (seconds per transistor)
  const timePerTransistor = {
    json: 0.1,
    csv: 0.05,
    spice: 0.2,
    plecs: 0.5,
    matlab: 0.4,
    gecko: 0.3,
    ltspice: 0.2
  }

  const totalSeconds = (timePerTransistor[exportFormat.value] || 0.2) * selectedCount.value
  if (totalSeconds < 60) return `${Math.ceil(totalSeconds)}s`
  return `${Math.ceil(totalSeconds / 60)}m`
})

// Methods
onMounted(async () => {
  try {
    allTransistors.value = await transistorApi.getAll()
  } catch (err) {
    error.value = 'Failed to load transistors: ' + err.message
  }
})

const toggleSelection = (transistor) => {
  const index = selectedTransistors.value.findIndex(
    t => t.metadata.name === transistor.metadata.name
  )

  if (index >= 0) {
    selectedTransistors.value.splice(index, 1)
  } else {
    selectedTransistors.value.push(transistor)
  }
}

const isSelected = (transistor) => {
  return selectedTransistors.value.some(
    t => t.metadata.name === transistor.metadata.name
  )
}

const selectAll = () => {
  selectedTransistors.value = [...filteredTransistors.value]
}

const clearSelection = () => {
  selectedTransistors.value = []
}

const startExport = async () => {
  if (selectedCount.value === 0) return

  isExporting.value = true
  error.value = null
  exportProgress.value = {
    current: 0,
    total: selectedCount.value,
    message: 'Preparing export...'
  }

  try {
    const transistorIds = selectedTransistors.value.map(t => t.metadata.name)
    const options = exportFormat.value === 'plecs' ? plecsOptions.value : {}

    // Simulate progress (since batch export is a single API call)
    const progressInterval = setInterval(() => {
      if (exportProgress.value.current < exportProgress.value.total - 1) {
        exportProgress.value.current++
        exportProgress.value.message = `Exporting ${exportProgress.value.current}/${exportProgress.value.total}...`
      }
    }, 500)

    const blob = await transistorApi.batchExport(transistorIds, exportFormat.value, options)

    clearInterval(progressInterval)
    exportProgress.value.current = exportProgress.value.total
    exportProgress.value.message = 'Creating ZIP archive...'

    // Download the ZIP file
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${archiveName.value}_${exportFormat.value}.zip`
    link.click()
    URL.revokeObjectURL(url)

    emit('export-complete', {
      count: selectedCount.value,
      format: exportFormat.value
    })

    // Reset after short delay
    setTimeout(() => {
      isExporting.value = false
      exportProgress.value = { current: 0, total: 0, message: '' }
    }, 1000)

  } catch (err) {
    error.value = err.response?.data?.detail || err.message
    isExporting.value = false
  }
}
</script>

<style scoped>
.batch-export-panel {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.panel-header {
  margin-bottom: 32px;
}

.panel-header h2 {
  margin: 0 0 8px 0;
  color: #1f2937;
  font-size: 28px;
}

.subtitle {
  margin: 0;
  color: #6b7280;
  font-size: 16px;
}

.selection-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
}

.summary-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: white;
  border-radius: 12px;
  border: 2px solid #e5e7eb;
}

.summary-icon {
  font-size: 32px;
}

.summary-number {
  font-size: 28px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1;
}

.summary-label {
  font-size: 13px;
  color: #6b7280;
  margin-top: 4px;
}

.selection-section {
  background: white;
  border-radius: 12px;
  border: 2px solid #e5e7eb;
  padding: 24px;
  margin-bottom: 24px;
}

.selection-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.selection-header h3 {
  margin: 0;
  color: #374151;
  font-size: 18px;
}

.selection-actions {
  display: flex;
  gap: 8px;
}

.filters {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.search-input {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
}

.filter-select {
  padding: 10px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  min-width: 150px;
}

.transistor-list {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}

.transistor-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid #e5e7eb;
  cursor: pointer;
  transition: background 0.2s;
}

.transistor-item:last-child {
  border-bottom: none;
}

.transistor-item:hover {
  background: #f9fafb;
}

.transistor-item.selected {
  background: #eff6ff;
  border-left: 4px solid #3b82f6;
}

.checkbox input {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.transistor-info {
  flex: 1;
}

.transistor-name {
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 4px;
}

.transistor-specs {
  display: flex;
  gap: 8px;
  font-size: 13px;
  color: #6b7280;
}

.export-options {
  background: white;
  border-radius: 12px;
  border: 2px solid #e5e7eb;
  padding: 24px;
  margin-bottom: 24px;
}

.export-options h3 {
  margin: 0 0 16px 0;
  color: #374151;
  font-size: 18px;
}

.options-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
}

.option-group label {
  display: block;
  margin-bottom: 8px;
  color: #374151;
  font-weight: 500;
  font-size: 14px;
}

.format-select,
.text-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
}

.format-options {
  margin-top: 16px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
}

.format-options summary {
  cursor: pointer;
  font-weight: 500;
  color: #374151;
  margin-bottom: 12px;
}

.options-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #374151;
  font-size: 14px;
  cursor: pointer;
}

.checkbox-label input {
  width: 16px;
  height: 16px;
}

.progress-section {
  background: white;
  border-radius: 12px;
  border: 2px solid #e5e7eb;
  padding: 24px;
  margin-bottom: 24px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.progress-header h3 {
  margin: 0;
  color: #374151;
}

.progress-bar {
  height: 12px;
  background: #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #2563eb);
  transition: width 0.3s ease;
}

.progress-message {
  text-align: center;
  color: #6b7280;
  font-size: 14px;
}

.error-message {
  display: flex;
  gap: 12px;
  padding: 16px;
  background: #fef2f2;
  border: 1px solid #fca5a5;
  border-radius: 8px;
  margin-bottom: 24px;
}

.error-icon {
  font-size: 24px;
}

.error-message strong {
  color: #dc2626;
  display: block;
  margin-bottom: 4px;
}

.error-message p {
  margin: 0;
  color: #991b1b;
  font-size: 14px;
}

.action-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
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
