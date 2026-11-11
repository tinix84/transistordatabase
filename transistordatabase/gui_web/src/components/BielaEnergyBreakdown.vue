<template>
  <section class="energy-bars">
    <h4>Estimated energy by switching phase</h4>
    <div class="canvas"><canvas ref="canvas" aria-label="Switching phase energy bars" /></div>
    <p>Blue: turn-on. Orange: turn-off. Recovery contributions are zero only when explicitly omitted.</p>
  </section>
</template>
<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Chart, registerables } from 'chart.js/auto'
Chart.register(...registerables)
const props = defineProps({ diagnostics: { type: Object, required: true } })
const canvas = ref(null)
let chart
function render() {
  chart?.destroy()
  const points = ['turn_on', 'turn_off'].flatMap(event => Object.entries(props.diagnostics[event + '_energy_partitions_j'])
    .map(([phase, energy]) => ({ label: event.replaceAll('_', ' ') + ' / ' + phase.replaceAll('_', ' '),
      energy: energy * 1e6, color: event === 'turn_on' ? '#205f95' : '#c75d20' })))
  chart = new Chart(canvas.value, { type: 'bar',
    data: { labels: points.map(p => p.label), datasets: [{ label: 'Estimated phase energy (µJ)',
      data: points.map(p => p.energy), backgroundColor: points.map(p => p.color) }] },
    options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      animation: false, plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, title: { display: true, text: 'Energy (µJ)' } } } } })
}
onMounted(render)
watch(() => props.diagnostics, render, { deep: true })
onBeforeUnmount(() => chart?.destroy())
</script>
<style scoped>
.energy-bars { background: white; padding: 14px; border: 1px solid #d9dee6; border-radius: 6px; }
.canvas { position: relative; height: 340px; width: 100%; }
p { font-size: .8rem; }
</style>
