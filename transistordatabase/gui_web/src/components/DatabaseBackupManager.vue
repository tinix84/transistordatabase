<script setup>
import { ref } from 'vue'
import { transistorApi } from '../services/api.js'

const emit = defineEmits(['restored'])
const selectedFile = ref(null)
const validation = ref(null)
const restoreResult = ref(null)
const message = ref('')
const messageType = ref('status')
const isValidating = ref(false)
const isRestoring = ref(false)
const isDownloading = ref(false)

async function responseDetail(error) {
  let responseBody = error.response?.data
  if (responseBody instanceof Blob) {
    try {
      responseBody = JSON.parse(await responseBody.text())
    } catch {
      responseBody = null
    }
  }
  const detail = responseBody?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') {
    const entries = detail.errors || detail.records || []
    const summary = entries.map(entry => {
      const identity = entry.name || entry.file || `record ${Number(entry.index) + 1}`
      return `${identity}: ${entry.error}`
    }).join('; ')
    return [detail.message, summary].filter(Boolean).join(' ')
  }
  return error.message || 'The database operation failed.'
}

function handleFileChange(event) {
  selectedFile.value = event.target.files?.[0] || null
  validation.value = null
  restoreResult.value = null
  message.value = ''
}

async function validateBackup() {
  if (!selectedFile.value) return
  validation.value = null
  restoreResult.value = null
  message.value = ''
  isValidating.value = true
  try {
    validation.value = await transistorApi.validateDatabaseBackup(selectedFile.value)
  } catch (error) {
    validation.value = { valid: false }
    messageType.value = 'error'
    message.value = await responseDetail(error)
  } finally {
    isValidating.value = false
  }
}

async function restoreBackup() {
  if (!selectedFile.value || !validation.value?.valid || validation.value.restore_count < 1) return
  const count = validation.value.restore_count
  const confirmed = window.confirm(
    `Add ${count} transistor record${count === 1 ? '' : 's'} from this backup? Existing records will remain unchanged and name conflicts will be skipped.`
  )
  if (!confirmed) return

  isRestoring.value = true
  restoreResult.value = null
  message.value = ''
  try {
    restoreResult.value = await transistorApi.restoreDatabaseBackup(selectedFile.value)
    messageType.value = restoreResult.value.errors?.length ? 'error' : 'status'
    message.value = restoreResult.value.errors?.length
      ? 'Restore finished with file publication errors; review the details below.'
      : 'Restore finished. Existing-name conflicts were skipped.'
    if (restoreResult.value.restored.length) emit('restored')
  } catch (error) {
    messageType.value = 'error'
    message.value = await responseDetail(error)
  } finally {
    isRestoring.value = false
  }
}

async function downloadBackup() {
  message.value = ''
  isDownloading.value = true
  try {
    const blob = await transistorApi.downloadDatabaseBackup()
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'transistordatabase-backup.json'
    link.click()
    URL.revokeObjectURL(url)
    messageType.value = 'status'
    message.value = 'Database backup downloaded.'
  } catch (error) {
    messageType.value = 'error'
    message.value = await responseDetail(error)
  } finally {
    isDownloading.value = false
  }
}
</script>

<template>
  <section class="database-backup" aria-labelledby="database-backup-title">
    <h2 id="database-backup-title">Database Backup and Restore</h2>
    <p class="backup-description">
      Download or restore a versioned JSON backup of the full database. Raw legacy fields are preserved.
      Restore adds new records and skips names already in this database.
    </p>

    <div class="backup-action">
      <button class="btn btn-primary" :disabled="isDownloading" @click="downloadBackup">
        {{ isDownloading ? 'Preparing backup…' : 'Download database backup' }}
      </button>
    </div>

    <div class="backup-action restore-action">
      <label for="database-backup-file">Select a database backup (.json)</label>
      <input
        id="database-backup-file"
        type="file"
        accept=".json,application/json"
        @change="handleFileChange"
      >
      <button
        class="btn btn-secondary"
        :disabled="!selectedFile || isValidating || isRestoring"
        @click="validateBackup"
      >
        {{ isValidating ? 'Validating…' : 'Validate backup' }}
      </button>
    </div>

    <div v-if="validation?.valid" class="backup-validation" role="status">
      <h3>Backup is valid</h3>
      <p>Format: {{ validation.format }}, version {{ validation.version }}.</p>
      <p>{{ validation.record_count }} records found; {{ validation.restore_count }} will be added.</p>
      <ul v-if="validation.conflicts.length" class="backup-list">
        <li>Existing-name conflicts will be skipped: {{ validation.conflicts.join(', ') }}</li>
      </ul>
      <ul v-if="validation.restore_names.length" class="backup-list">
        <li>New records: {{ validation.restore_names.join(', ') }}</li>
      </ul>
      <button
        class="btn btn-primary"
        :disabled="validation.restore_count < 1 || isRestoring"
        @click="restoreBackup"
      >
        {{ isRestoring ? 'Restoring…' : 'Restore new records' }}
      </button>
    </div>

    <div v-if="restoreResult" class="backup-result" role="status">
      <h3>Restore results</h3>
      <p>
        {{ restoreResult.restored.length }} record{{ restoreResult.restored.length === 1 ? '' : 's' }} added.
      </p>
      <ul v-if="restoreResult.restored.length" class="backup-list">
        <li>Added: {{ restoreResult.restored.join(', ') }}</li>
      </ul>
      <ul v-if="restoreResult.conflicts.length" class="backup-list">
        <li>Skipped existing names: {{ restoreResult.conflicts.join(', ') }}</li>
      </ul>
      <ul v-if="restoreResult.errors?.length" class="backup-list error-list">
        <li v-for="item in restoreResult.errors" :key="item.name">
          {{ item.name }}: {{ item.error }}
        </li>
      </ul>
    </div>

    <p v-if="message" :role="messageType === 'error' ? 'alert' : 'status'" :class="['backup-message', messageType]">
      {{ message }}
    </p>
  </section>
</template>

<style scoped>
.database-backup {
  max-width: 900px;
  margin: 1.5rem auto;
  padding: 1.5rem;
  border-radius: 8px;
  background: var(--card-background, #fff);
  box-shadow: 0 2px 8px rgb(0 0 0 / 8%);
}

.database-backup h2,
.database-backup h3 {
  margin-bottom: 0.75rem;
}

.backup-description {
  margin-bottom: 1.25rem;
}

.backup-action {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin: 1rem 0;
}

.restore-action label {
  flex-basis: 100%;
  font-weight: 600;
}

.backup-validation,
.backup-result {
  margin-top: 1.25rem;
  padding: 1rem;
  border: 1px solid #b7d8c1;
  border-radius: 6px;
  background: #f2faf4;
}

.backup-list {
  margin: 0.75rem 0 1rem 1.25rem;
  overflow-wrap: anywhere;
}

.error-list,
.backup-message.error {
  color: #9d2525;
}

.backup-message {
  margin-top: 1rem;
}
</style>
