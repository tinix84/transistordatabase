<script setup>
import { computed, ref } from 'vue'
import { displayNumber as number } from './analysisFiles.js'
const props = defineProps({ source: { type: Object, required: true } })
const open = ref(false), page = ref(0)
const rows = computed(() => props.source.current_a.slice(page.value * 50, (page.value + 1) * 50))
</script>

<template>
  <details @toggle="open = $event.target.open">
    <summary>Original points · {{ source.id }} · {{ source.source_kind }} · {{ source.voltage_v }} V</summary>
    <div v-if="open"><p>{{ source.source }}</p><pre>{{ JSON.stringify(source.conditions, null, 2) }}</pre>
      <p>All {{ source.current_a.length }} original points remain in the download. This table shows 50 points per page.</p>
      <div v-if="source.current_a.length > 50"><button type="button" :disabled="page === 0" @click="page--">Previous slice points</button><span>Page {{ page + 1 }} / {{ Math.ceil(source.current_a.length / 50) }}</span><button type="button" :disabled="(page + 1) * 50 >= source.current_a.length" @click="page++">Next slice points</button></div>
      <table><thead><tr><th>Current (A)</th><th>Energy (J)</th></tr></thead><tbody><tr v-for="(current, index) in rows" :key="index"><td>{{ number(current) }}</td><td>{{ number(source.energy_j[page * 50 + index]) }}</td></tr></tbody></table>
    </div>
  </details>
</template>

<style scoped>
summary { cursor:pointer; font-weight:600; }pre { white-space:pre-wrap; overflow-wrap:anywhere; }table { width:100%; border-collapse:collapse; }th,td { text-align:left; padding:.6rem; border-bottom:1px solid var(--border-color,#aaa); }button { padding:.5rem; margin:.25rem; color:inherit; background:var(--bg-secondary); border:1px solid #7b97ab; border-radius:5px; }
</style>
