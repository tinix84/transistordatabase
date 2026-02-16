<template>
  <div class="plecs-importer">
    <h2>PLECS XML Library Importer</h2>
    <p class="description">
      Import transistor data from PLECS XML semiconductor library files.
      Supports 372+ devices from major manufacturers.
    </p>

    <!-- Upload Zone -->
    <div
      class="upload-zone"
      :class="{ 'drag-over': isDragging }"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="handleDrop"
      @click="triggerFileInput"
    >
      <div class="upload-icon">📁</div>
      <p class="upload-text">
        <strong>Drop PLECS XML files here</strong> or click to browse
      </p>
      <p class="upload-subtext">Supports .xml files up to 10MB each</p>
      <input
        ref="fileInput"
        type="file"
        accept=".xml"
        multiple
        style="display: none"
        @change="handleFileSelect"
      />
    </div>

    <!-- File List -->
    <div v-if="files.length > 0" class="file-list">
      <h3>Files to Import ({{ files.length }})</h3>
      <div v-for="(file, index) in files" :key="index" class="file-item">
        <div class="file-info">
          <span class="file-name">{{ file.name }}</span>
          <span class="file-size">{{ formatBytes(file.size) }}</span>
        </div>
        <div class="file-status">
          <span v-if="file.status === 'pending'" class="status-pending">⏳ Pending</span>
          <span v-else-if="file.status === 'uploading'" class="status-uploading">
            ⏫ Uploading...
          </span>
          <span v-else-if="file.status === 'success'" class="status-success">
            ✅ {{ file.imported }} transistor(s)
          </span>
          <span v-else-if="file.status === 'error'" class="status-error">
            ❌ {{ file.error }}
          </span>
        </div>
        <button v-if="file.status === 'pending'" @click="removeFile(index)" class="btn-remove">
          🗑️
        </button>
      </div>
    </div>

    <!-- Import Controls -->
    <div v-if="files.length > 0" class="import-controls">
      <button
        @click="startImport"
        :disabled="importing || files.every(f => f.status !== 'pending')"
        class="btn-primary btn-import"
      >
        {{ importing ? 'Importing...' : `Import ${pendingCount} file(s)` }}
      </button>
      <button @click="clearAll" :disabled="importing" class="btn-secondary">
        Clear All
      </button>
    </div>

    <!-- Import Results -->
    <div v-if="importResults.length > 0" class="import-results">
      <h3>Import Results</h3>
      <div class="results-summary">
        <div class="summary-card">
          <div class="summary-number">{{ totalImported }}</div>
          <div class="summary-label">Transistors Imported</div>
        </div>
        <div class="summary-card">
          <div class="summary-number">{{ successfulFiles }}</div>
          <div class="summary-label">Files Processed</div>
        </div>
        <div class="summary-card" v-if="totalErrors > 0">
          <div class="summary-number error">{{ totalErrors }}</div>
          <div class="summary-label">Errors</div>
        </div>
      </div>

      <!-- Imported Transistors List -->
      <div class="transistor-list">
        <h4>Imported Transistors</h4>
        <div class="transistor-chips">
          <span
            v-for="id in allImportedIds"
            :key="id"
            class="transistor-chip"
            @click="$emit('view-transistor', id)"
          >
            {{ id }}
          </span>
        </div>
      </div>

      <!-- Errors and Warnings -->
      <div v-if="allErrors.length > 0" class="error-list">
        <h4>Errors ({{ allErrors.length }})</h4>
        <div v-for="(error, index) in allErrors" :key="index" class="error-item">
          {{ error }}
        </div>
      </div>

      <div v-if="allWarnings.length > 0" class="warning-list">
        <h4>Warnings ({{ allWarnings.length }})</h4>
        <div v-for="(warning, index) in allWarnings" :key="index" class="warning-item">
          {{ warning }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { transistorApi } from '../services/api'

export default {
  name: 'PLECSImporter',
  emits: ['import-complete', 'view-transistor'],
  setup(props, { emit }) {
    const files = ref([])
    const isDragging = ref(false)
    const importing = ref(false)
    const fileInput = ref(null)
    const importResults = ref([])

    const pendingCount = computed(() =>
      files.value.filter(f => f.status === 'pending').length
    )

    const totalImported = computed(() =>
      importResults.value.reduce((sum, r) => sum + (r.transistors_imported || 0), 0)
    )

    const successfulFiles = computed(() =>
      importResults.value.filter(r => r.success).length
    )

    const totalErrors = computed(() =>
      importResults.value.reduce((sum, r) => sum + (r.errors?.length || 0), 0)
    )

    const allImportedIds = computed(() => {
      const ids = []
      importResults.value.forEach(r => {
        if (r.transistor_ids) {
          ids.push(...r.transistor_ids)
        }
      })
      return ids
    })

    const allErrors = computed(() => {
      const errors = []
      importResults.value.forEach(r => {
        if (r.errors) {
          errors.push(...r.errors)
        }
      })
      return errors
    })

    const allWarnings = computed(() => {
      const warnings = []
      importResults.value.forEach(r => {
        if (r.warnings) {
          warnings.push(...r.warnings)
        }
      })
      return warnings
    })

    function triggerFileInput() {
      fileInput.value.click()
    }

    function handleFileSelect(event) {
      const selectedFiles = Array.from(event.target.files)
      addFiles(selectedFiles)
      event.target.value = '' // Reset input
    }

    function handleDrop(event) {
      isDragging.value = false
      const droppedFiles = Array.from(event.dataTransfer.files)
      const xmlFiles = droppedFiles.filter(f => f.name.endsWith('.xml'))
      addFiles(xmlFiles)
    }

    function addFiles(newFiles) {
      newFiles.forEach(file => {
        if (file.size > 10 * 1024 * 1024) {
          alert(`File ${file.name} is too large (max 10MB)`)
          return
        }
        files.value.push({
          name: file.name,
          size: file.size,
          file: file,
          status: 'pending',
          imported: 0,
          error: null
        })
      })
    }

    function removeFile(index) {
      files.value.splice(index, 1)
    }

    function clearAll() {
      files.value = []
      importResults.value = []
    }

    async function startImport() {
      importing.value = true
      importResults.value = []

      for (const fileItem of files.value) {
        if (fileItem.status !== 'pending') continue

        fileItem.status = 'uploading'

        try {
          const result = await transistorApi.importPLECS(fileItem.file)
          fileItem.status = result.success ? 'success' : 'error'
          fileItem.imported = result.transistors_imported
          fileItem.error = result.errors?.[0] || null
          importResults.value.push(result)
        } catch (error) {
          fileItem.status = 'error'
          fileItem.error = error.message
          importResults.value.push({
            success: false,
            transistors_imported: 0,
            transistor_ids: [],
            errors: [error.message]
          })
        }
      }

      importing.value = false
      emit('import-complete', {
        total: totalImported.value,
        ids: allImportedIds.value
      })
    }

    function formatBytes(bytes) {
      if (bytes < 1024) return bytes + ' B'
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
    }

    return {
      files,
      isDragging,
      importing,
      fileInput,
      importResults,
      pendingCount,
      totalImported,
      successfulFiles,
      totalErrors,
      allImportedIds,
      allErrors,
      allWarnings,
      triggerFileInput,
      handleFileSelect,
      handleDrop,
      removeFile,
      clearAll,
      startImport,
      formatBytes
    }
  }
}
</script>

<style scoped>
.plecs-importer {
  padding: 20px;
  max-width: 1200px;
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

/* Upload Zone */
.upload-zone {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 60px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
  background: #f9f9f9;
}

.upload-zone:hover,
.upload-zone.drag-over {
  border-color: #4CAF50;
  background: #f0fff0;
}

.upload-icon {
  font-size: 48px;
  margin-bottom: 15px;
}

.upload-text {
  font-size: 16px;
  margin: 0 0 5px 0;
}

.upload-subtext {
  font-size: 14px;
  color: #999;
  margin: 0;
}

/* File List */
.file-list {
  margin-top: 30px;
}

.file-list h3 {
  font-size: 18px;
  margin-bottom: 15px;
}

.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 15px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  margin-bottom: 10px;
  background: white;
}

.file-info {
  flex: 1;
}

.file-name {
  font-weight: 500;
  margin-right: 10px;
}

.file-size {
  color: #999;
  font-size: 14px;
}

.file-status span {
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 14px;
  margin-right: 10px;
}

.status-pending {
  background: #f0f0f0;
}

.status-uploading {
  background: #fff3cd;
}

.status-success {
  background: #d4edda;
  color: #155724;
}

.status-error {
  background: #f8d7da;
  color: #721c24;
}

.btn-remove {
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  opacity: 0.6;
}

.btn-remove:hover {
  opacity: 1;
}

/* Import Controls */
.import-controls {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}

.btn-primary,
.btn-secondary {
  padding: 12px 24px;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  border: none;
  transition: all 0.2s;
}

.btn-primary {
  background: #4CAF50;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #45a049;
}

.btn-primary:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.btn-secondary {
  background: #f0f0f0;
}

.btn-secondary:hover:not(:disabled) {
  background: #e0e0e0;
}

/* Import Results */
.import-results {
  margin-top: 40px;
  padding-top: 30px;
  border-top: 2px solid #e0e0e0;
}

.import-results h3 {
  font-size: 20px;
  margin-bottom: 20px;
}

.results-summary {
  display: flex;
  gap: 20px;
  margin-bottom: 30px;
}

.summary-card {
  flex: 1;
  background: #f9f9f9;
  padding: 20px;
  border-radius: 8px;
  text-align: center;
}

.summary-number {
  font-size: 36px;
  font-weight: bold;
  color: #4CAF50;
}

.summary-number.error {
  color: #f44336;
}

.summary-label {
  font-size: 14px;
  color: #666;
  margin-top: 5px;
}

.transistor-list {
  margin-top: 30px;
}

.transistor-list h4 {
  font-size: 16px;
  margin-bottom: 15px;
}

.transistor-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.transistor-chip {
  display: inline-block;
  padding: 6px 12px;
  background: #e3f2fd;
  border-radius: 16px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.transistor-chip:hover {
  background: #2196F3;
  color: white;
}

/* Error and Warning Lists */
.error-list,
.warning-list {
  margin-top: 20px;
  padding: 15px;
  border-radius: 6px;
}

.error-list {
  background: #fff3cd;
  border-left: 4px solid #ffc107;
}

.warning-list {
  background: #fff3cd;
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
  font-size: 14px;
  border-bottom: 1px solid #e0e0e0;
}

.error-item:last-child,
.warning-item:last-child {
  border-bottom: none;
}
</style>
