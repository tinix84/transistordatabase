<template>
  <section class="batch-export-panel">
    <h2>Batch Export</h2>
    <p v-if="error" role="alert">{{ error }}</p>
    <ExportDownloadPanel :transistors="transistors" batch
      @export-complete="$emit('export-complete', $event)"
      @navigate-workbench="$emit('navigate-workbench', $event)" />
    <button type="button" @click="$emit('close')">Close batch export</button>
  </section>
</template>
<script setup>
import { onMounted, ref } from 'vue'
import { transistorApi } from '../services/api.js'
import ExportDownloadPanel from './ExportDownloadPanel.vue'
defineEmits(['close', 'export-complete', 'navigate-workbench'])
const transistors = ref([])
const error = ref('')
onMounted(async () => {
  try { transistors.value = await transistorApi.getAll() }
  catch (err) { error.value = err.message || 'Cannot load devices.' }
})
</script>
<style scoped>
.batch-export-panel { padding: 1.5rem; max-width: 1100px; margin: auto; }
</style>
