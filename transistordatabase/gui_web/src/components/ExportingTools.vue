<template>
  <div class="exporting-tools">
    <h2>Exporting Tools</h2>
    <ExportDownloadPanel :transistors="transistors" :selected-from-search="selectedFromSearch"
      @open-database-search="$emit('open-database-search')"
      @navigate-workbench="$emit('navigate-workbench', $event)" />
  </div>
</template>
<script setup>
import ExportDownloadPanel from './ExportDownloadPanel.vue'
defineProps({ transistors: { type: Array, default: () => [] }, selectedFromSearch: { type: Array, default: () => [] } })
defineEmits(['open-database-search', 'navigate-workbench'])
</script>
<style scoped>
.exporting-tools { padding: 1.5rem; }
@media (max-width: 500px) { .exporting-tools { padding: .75rem; } }
</style>
