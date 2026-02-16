<template>
  <div class="curve-uploader">
    <div
      @drop.prevent="handleDrop"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      :class="['upload-zone', { dragging: isDragging, error: uploadError }]"
    >
      <div class="upload-content">
        <div class="upload-icon">📁</div>
        <p class="upload-text">
          <strong>Drag and drop CSV file here</strong>
        </p>
        <p class="upload-subtext">or</p>
        <button @click="triggerFileInput" class="btn btn-upload">
          Browse Files
        </button>
        <input
          ref="fileInput"
          type="file"
          accept=".csv,.txt"
          @change="handleFileSelect"
          style="display: none"
        />
      </div>

      <div v-if="uploadError" class="error-message">
        ⚠️ {{ uploadError }}
      </div>
    </div>

    <div v-if="parsedData" class="preview-section">
      <div class="preview-header">
        <h4>📊 Preview ({{ parsedData.length }} data points)</h4>
        <div class="preview-actions">
          <button @click="clearData" class="btn btn-secondary btn-sm">
            🗑️ Clear
          </button>
          <button @click="$emit('use-data', parsedData)" class="btn btn-primary btn-sm">
            ✅ Use This Data
          </button>
        </div>
      </div>

      <div class="data-table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th v-for="(col, index) in columns" :key="index">
                {{ col }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, rowIndex) in previewRows" :key="rowIndex">
              <td class="row-number">{{ rowIndex + 1 }}</td>
              <td v-for="(value, colIndex) in row" :key="colIndex">
                {{ formatNumber(value) }}
              </td>
            </tr>
            <tr v-if="parsedData.length > maxPreviewRows">
              <td :colspan="columns.length + 1" class="more-rows">
                ... and {{ parsedData.length - maxPreviewRows }} more rows
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="file-info">
        <span><strong>File:</strong> {{ fileName }}</span>
        <span><strong>Size:</strong> {{ fileSize }}</span>
        <span><strong>Format:</strong> {{ detectedFormat }}</span>
      </div>
    </div>

    <div class="format-help">
      <details>
        <summary>📋 CSV Format Guidelines</summary>
        <div class="help-content">
          <h5>Expected Format:</h5>
          <ul>
            <li>Two columns of numerical data</li>
            <li>First column: {{ xLabel }} ({{ xUnit }})</li>
            <li>Second column: {{ yLabel }} ({{ yUnit }})</li>
            <li>Comma, semicolon, or tab separated</li>
            <li>Optional header row (will be auto-detected)</li>
          </ul>
          <h5>Example:</h5>
          <pre>{{ exampleCSV }}</pre>
        </div>
      </details>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CurveUploader',
  props: {
    xLabel: {
      type: String,
      default: 'X'
    },
    yLabel: {
      type: String,
      default: 'Y'
    },
    xUnit: {
      type: String,
      default: ''
    },
    yUnit: {
      type: String,
      default: ''
    },
    maxPreviewRows: {
      type: Number,
      default: 10
    }
  },
  emits: ['use-data', 'error'],
  data() {
    return {
      isDragging: false,
      parsedData: null,
      fileName: '',
      fileSize: '',
      detectedFormat: '',
      uploadError: null,
      columns: []
    }
  },
  computed: {
    previewRows() {
      if (!this.parsedData) return []
      return this.parsedData.slice(0, this.maxPreviewRows)
    },
    exampleCSV() {
      return `${this.xLabel},${this.yLabel}
0.1,5.2
0.2,10.5
0.3,15.8
0.4,21.3`
    }
  },
  methods: {
    triggerFileInput() {
      this.$refs.fileInput.click()
    },

    handleFileSelect(event) {
      const file = event.target.files[0]
      if (file) {
        this.processFile(file)
      }
    },

    handleDrop(event) {
      this.isDragging = false
      const files = event.dataTransfer.files
      if (files.length > 0) {
        this.processFile(files[0])
      }
    },

    async processFile(file) {
      this.uploadError = null
      this.fileName = file.name
      this.fileSize = this.formatFileSize(file.size)

      // Check file type
      if (!file.name.match(/\.(csv|txt)$/i)) {
        this.uploadError = 'Only CSV and TXT files are supported'
        this.$emit('error', this.uploadError)
        return
      }

      try {
        const content = await this.readFileContent(file)
        this.parseCSV(content)
      } catch (error) {
        this.uploadError = `Failed to read file: ${error.message}`
        this.$emit('error', this.uploadError)
      }
    },

    readFileContent(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => resolve(e.target.result)
        reader.onerror = reject
        reader.readAsText(file)
      })
    },

    parseCSV(content) {
      const lines = content.trim().split(/\r?\n/)

      if (lines.length < 2) {
        this.uploadError = 'File must contain at least 2 rows of data'
        this.$emit('error', this.uploadError)
        return
      }

      // Detect delimiter
      const delimiter = this.detectDelimiter(lines[0])
      this.detectedFormat = `CSV (${delimiter === ',' ? 'comma' : delimiter === ';' ? 'semicolon' : 'tab'} separated)`

      // Check if first line is header
      const firstLineParts = lines[0].split(delimiter)
      const hasHeader = this.looksLikeHeader(firstLineParts)

      if (hasHeader) {
        this.columns = firstLineParts.map(col => col.trim())
        lines.shift() // Remove header
      } else {
        this.columns = [this.xLabel, this.yLabel]
      }

      // Parse data rows
      const parsedData = []
      const errors = []

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim()
        if (!line) continue

        const parts = line.split(delimiter)
        if (parts.length < 2) {
          errors.push(`Row ${i + 1}: Expected 2 columns, got ${parts.length}`)
          continue
        }

        const x = parseFloat(parts[0])
        const y = parseFloat(parts[1])

        if (isNaN(x) || isNaN(y)) {
          errors.push(`Row ${i + 1}: Invalid numeric values`)
          continue
        }

        parsedData.push([x, y])
      }

      if (parsedData.length === 0) {
        this.uploadError = 'No valid data rows found'
        this.$emit('error', this.uploadError)
        return
      }

      if (errors.length > 0 && errors.length < 5) {
        console.warn('Parse warnings:', errors)
      }

      this.parsedData = parsedData
      this.uploadError = null
    },

    detectDelimiter(line) {
      const delimiters = [',', ';', '\t']
      let maxCount = 0
      let detectedDelimiter = ','

      for (const delimiter of delimiters) {
        const count = (line.match(new RegExp('\\' + delimiter, 'g')) || []).length
        if (count > maxCount) {
          maxCount = count
          detectedDelimiter = delimiter
        }
      }

      return detectedDelimiter
    },

    looksLikeHeader(parts) {
      // If first part contains non-numeric characters, likely a header
      return parts.some(part => isNaN(parseFloat(part)))
    },

    formatFileSize(bytes) {
      if (bytes < 1024) return `${bytes} B`
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    },

    formatNumber(value) {
      if (typeof value !== 'number') return value
      return value.toExponential(3)
    },

    clearData() {
      this.parsedData = null
      this.fileName = ''
      this.uploadError = null
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = ''
      }
    }
  }
}
</script>

<style scoped>
.curve-uploader {
  padding: 16px;
}

.upload-zone {
  border: 3px dashed #d1d5db;
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  background: #f9fafb;
  transition: all 0.3s;
  cursor: pointer;
}

.upload-zone.dragging {
  border-color: #3b82f6;
  background: #eff6ff;
}

.upload-zone.error {
  border-color: #ef4444;
  background: #fef2f2;
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.upload-icon {
  font-size: 48px;
}

.upload-text {
  margin: 0;
  color: #1f2937;
  font-size: 16px;
}

.upload-subtext {
  margin: 0;
  color: #6b7280;
  font-size: 14px;
}

.btn-upload {
  background: #3b82f6;
  color: white;
  border: none;
  padding: 10px 24px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-upload:hover {
  background: #2563eb;
}

.error-message {
  margin-top: 16px;
  padding: 12px;
  background: #fee2e2;
  border: 1px solid #fca5a5;
  border-radius: 6px;
  color: #dc2626;
  font-size: 14px;
}

.preview-section {
  margin-top: 24px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: white;
  overflow: hidden;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.preview-header h4 {
  margin: 0;
  color: #1f2937;
}

.preview-actions {
  display: flex;
  gap: 8px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
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

.btn-primary:hover {
  background: #2563eb;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.data-table-wrapper {
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.data-table th {
  position: sticky;
  top: 0;
  background: #f3f4f6;
  padding: 10px;
  text-align: left;
  font-weight: 600;
  color: #374151;
  border-bottom: 2px solid #d1d5db;
  z-index: 1;
}

.data-table td {
  padding: 8px 10px;
  border-bottom: 1px solid #e5e7eb;
  color: #1f2937;
  font-family: 'Courier New', monospace;
}

.row-number {
  color: #9ca3af;
  font-weight: 500;
}

.more-rows {
  text-align: center;
  color: #6b7280;
  font-style: italic;
  padding: 16px !important;
}

.file-info {
  display: flex;
  gap: 24px;
  padding: 12px 16px;
  background: #f9fafb;
  border-top: 1px solid #e5e7eb;
  font-size: 13px;
  color: #6b7280;
}

.format-help {
  margin-top: 16px;
}

.format-help details {
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 12px;
  background: #f9fafb;
}

.format-help summary {
  cursor: pointer;
  font-weight: 500;
  color: #374151;
  user-select: none;
}

.help-content {
  margin-top: 12px;
  color: #6b7280;
  font-size: 13px;
}

.help-content h5 {
  margin: 12px 0 8px 0;
  color: #1f2937;
  font-size: 14px;
}

.help-content ul {
  margin: 8px 0;
  padding-left: 20px;
}

.help-content li {
  margin: 4px 0;
}

.help-content pre {
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  padding: 12px;
  overflow-x: auto;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #1f2937;
}
</style>
