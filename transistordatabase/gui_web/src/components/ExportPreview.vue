<template>
  <div class="export-preview-modal" v-if="show" @click="close">
    <div class="modal-container" @click.stop>
      <div class="modal-header">
        <h2>📄 Export Preview</h2>
        <button @click="close" class="btn-close">✕</button>
      </div>

      <div class="modal-body">
        <!-- Format Selection -->
        <div class="format-section">
          <h3>Export Format</h3>
          <div class="format-buttons">
            <button
              v-for="fmt in availableFormats"
              :key="fmt.value"
              @click="selectFormat(fmt.value)"
              :class="['format-btn', { active: selectedFormat === fmt.value }]"
            >
              {{ fmt.icon }} {{ fmt.label }}
            </button>
          </div>
        </div>

        <!-- Format Description -->
        <div v-if="selectedFormat" class="format-info">
          <div class="info-card">
            <h4>{{ getFormatInfo(selectedFormat).title }}</h4>
            <p>{{ getFormatInfo(selectedFormat).description }}</p>
            <div class="info-tags">
              <span v-for="tag in getFormatInfo(selectedFormat).tags" :key="tag" class="tag">
                {{ tag }}
              </span>
            </div>
          </div>
        </div>

        <!-- Preview Content -->
        <div v-if="isLoading" class="loading-section">
          <div class="spinner"></div>
          <p>Loading preview...</p>
        </div>

        <div v-else-if="previewData" class="preview-section">
          <div class="preview-header">
            <h3>Preview</h3>
            <div class="preview-meta">
              <span class="meta-item">
                📄 {{ previewData.file_name }}
              </span>
              <span class="meta-item">
                📊 {{ previewData.total_lines }} lines
              </span>
              <span class="meta-item">
                💾 {{ formatFileSize(previewData.file_size) }}
              </span>
              <span v-if="previewData.truncated" class="meta-item warning">
                ⚠️ Showing first {{ previewData.preview_lines }} lines
              </span>
            </div>
          </div>

          <div class="preview-content">
            <pre><code>{{ previewData.preview }}</code></pre>
          </div>

          <div v-if="previewData.truncated" class="truncation-notice">
            <p>
              Preview is truncated. Full content will be included in the download.
            </p>
          </div>
        </div>

        <div v-else-if="error" class="error-section">
          <div class="error-message">
            <span class="error-icon">⚠️</span>
            <div>
              <h4>Preview Failed</h4>
              <p>{{ error }}</p>
            </div>
          </div>
        </div>
      </div>

      <div class="modal-footer">
        <button @click="close" class="btn btn-secondary">
          Cancel
        </button>
        <button
          v-if="previewData"
          @click="copyToClipboard"
          class="btn btn-secondary"
        >
          📋 Copy to Clipboard
        </button>
        <button
          @click="downloadExport"
          :disabled="!selectedFormat || isDownloading"
          class="btn btn-primary"
        >
          {{ isDownloading ? '⏳ Downloading...' : '💾 Download Export' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { transistorApi } from '../services/api.js'

const props = defineProps({
  show: {
    type: Boolean,
    required: true
  },
  transistorId: {
    type: String,
    required: true
  },
  initialFormat: {
    type: String,
    default: 'json'
  }
})

const emit = defineEmits(['close', 'download-complete'])

// State
const selectedFormat = ref(props.initialFormat)
const previewData = ref(null)
const isLoading = ref(false)
const isDownloading = ref(false)
const error = ref(null)

// Available formats
const availableFormats = [
  { value: 'json', label: 'JSON', icon: '📋' },
  { value: 'csv', label: 'CSV', icon: '📊' },
  { value: 'spice', label: 'SPICE', icon: '⚡' },
  { value: 'plecs', label: 'PLECS', icon: '🔌' },
  { value: 'matlab', label: 'MATLAB', icon: '🔢' },
  { value: 'gecko', label: 'GeckoCIRCUITS', icon: '🦎' },
  { value: 'ltspice', label: 'LTspice', icon: '📐' }
]

// Format information
const formatInfo = {
  json: {
    title: 'JSON Export',
    description: 'Human-readable JSON format with complete transistor data including all curves and metadata.',
    tags: ['Portable', 'Re-importable', 'Human-readable']
  },
  csv: {
    title: 'CSV Export',
    description: 'Comma-separated values format for spreadsheet applications. Includes basic electrical specifications.',
    tags: ['Excel-compatible', 'Lightweight', 'Tabular']
  },
  spice: {
    title: 'SPICE Model',
    description: 'SPICE netlist format for circuit simulation tools like LTspice, PSPICE, and HSPICE.',
    tags: ['Simulation', 'Industry-standard', 'Circuit analysis']
  },
  plecs: {
    title: 'PLECS XML',
    description: 'PLECS thermal description XML format for power electronics simulation with thermal modeling.',
    tags: ['Thermal simulation', 'Loss calculation', 'PLECS-compatible']
  },
  matlab: {
    title: 'MATLAB .mat File',
    description: 'MATLAB binary format (.mat) for data analysis and visualization in MATLAB/Octave.',
    tags: ['MATLAB', 'Data analysis', 'Plotting']
  },
  gecko: {
    title: 'GeckoCIRCUITS',
    description: 'GeckoCIRCUITS semiconductor library format (.scl) for power electronics simulation.',
    tags: ['GeckoCIRCUITS', 'Power electronics', 'Simulation']
  },
  ltspice: {
    title: 'LTspice DPT',
    description: 'LTspice double pulse test netlist for switching loss characterization.',
    tags: ['LTspice', 'DPT', 'Switching analysis']
  }
}

// Methods
const selectFormat = async (format) => {
  selectedFormat.value = format
  await loadPreview()
}

const loadPreview = async () => {
  if (!selectedFormat.value) return

  // Only text formats support preview
  const previewableFormats = ['json', 'csv', 'spice', 'plecs', 'ltspice']
  if (!previewableFormats.includes(selectedFormat.value)) {
    previewData.value = null
    error.value = 'Preview not available for binary formats. Download to view.'
    return
  }

  isLoading.value = true
  error.value = null

  try {
    previewData.value = await transistorApi.exportPreview(
      props.transistorId,
      selectedFormat.value
    )
  } catch (err) {
    error.value = err.response?.data?.detail || err.message
    previewData.value = null
  } finally {
    isLoading.value = false
  }
}

const downloadExport = async () => {
  if (!selectedFormat.value) return

  isDownloading.value = true

  try {
    const blob = await transistorApi.export(props.transistorId, selectedFormat.value)

    // Create download link
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url

    // Determine file extension
    const extensions = {
      json: 'json',
      csv: 'csv',
      spice: 'spice',
      plecs: 'xml',
      matlab: 'mat',
      gecko: 'scl',
      ltspice: 'asc'
    }
    const ext = extensions[selectedFormat.value] || 'txt'
    link.download = `${props.transistorId}.${ext}`

    link.click()
    URL.revokeObjectURL(url)

    emit('download-complete', selectedFormat.value)
  } catch (err) {
    error.value = err.response?.data?.detail || err.message
  } finally {
    isDownloading.value = false
  }
}

const copyToClipboard = async () => {
  if (!previewData.value) return

  try {
    await navigator.clipboard.writeText(previewData.value.preview)
    // Show success toast (you can implement a toast notification)
    alert('Preview copied to clipboard!')
  } catch (err) {
    alert('Failed to copy to clipboard: ' + err.message)
  }
}

const getFormatInfo = (format) => {
  return formatInfo[format] || {
    title: format.toUpperCase(),
    description: 'Export in ' + format + ' format',
    tags: []
  }
}

const formatFileSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const close = () => {
  emit('close')
}

// Watch for show changes
watch(() => props.show, (newVal) => {
  if (newVal) {
    selectedFormat.value = props.initialFormat
    loadPreview()
  }
})
</script>

<style scoped>
.export-preview-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 20px;
}

.modal-container {
  background: white;
  border-radius: 16px;
  width: 100%;
  max-width: 1000px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px 28px;
  border-bottom: 2px solid #e5e7eb;
}

.modal-header h2 {
  margin: 0;
  color: #1f2937;
  font-size: 24px;
}

.btn-close {
  background: none;
  border: none;
  font-size: 28px;
  cursor: pointer;
  color: #6b7280;
  padding: 0;
  line-height: 1;
}

.btn-close:hover {
  color: #1f2937;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px 28px;
}

.format-section h3 {
  margin: 0 0 16px 0;
  color: #374151;
  font-size: 16px;
}

.format-buttons {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}

.format-btn {
  padding: 12px 16px;
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  font-size: 14px;
  color: #374151;
  transition: all 0.2s;
}

.format-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.format-btn.active {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
}

.format-info {
  margin-top: 24px;
}

.info-card {
  padding: 20px;
  background: #f9fafb;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.info-card h4 {
  margin: 0 0 8px 0;
  color: #1f2937;
  font-size: 16px;
}

.info-card p {
  margin: 0 0 12px 0;
  color: #6b7280;
  font-size: 14px;
  line-height: 1.6;
}

.info-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.tag {
  display: inline-block;
  padding: 4px 12px;
  background: #dbeafe;
  color: #1e40af;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.loading-section,
.error-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: #6b7280;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-message {
  display: flex;
  gap: 16px;
  padding: 20px;
  background: #fef2f2;
  border: 1px solid #fca5a5;
  border-radius: 8px;
}

.error-icon {
  font-size: 32px;
}

.error-message h4 {
  margin: 0 0 8px 0;
  color: #dc2626;
}

.error-message p {
  margin: 0;
  color: #991b1b;
  font-size: 14px;
}

.preview-section {
  margin-top: 24px;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.preview-header h3 {
  margin: 0;
  color: #374151;
  font-size: 16px;
}

.preview-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.meta-item {
  font-size: 13px;
  color: #6b7280;
}

.meta-item.warning {
  color: #f59e0b;
  font-weight: 500;
}

.preview-content {
  background: #1f2937;
  border-radius: 8px;
  padding: 20px;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}

.preview-content pre {
  margin: 0;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.6;
}

.preview-content code {
  color: #e5e7eb;
}

.truncation-notice {
  margin-top: 12px;
  padding: 12px 16px;
  background: #fef3c7;
  border: 1px solid #fcd34d;
  border-radius: 8px;
}

.truncation-notice p {
  margin: 0;
  color: #92400e;
  font-size: 14px;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 20px 28px;
  border-top: 2px solid #e5e7eb;
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
