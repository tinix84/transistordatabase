<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { Chart, registerables } from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'

Chart.register(...registerables, zoomPlugin)
const props = defineProps({
  title: String, points: Array, xKey: String, yKey: String,
  metrics: Object, selectedId: String, currency: String,
  relative: Boolean, matchedCount: Number, excludedCount: Number, exclusions: Array,
})
const emit = defineEmits(['select'])
const canvas = ref(null)
const logX = ref(false)
const logY = ref(false)
let chart
const colors = { Si: '#4264c4', SiC: '#088275', GaN: '#b55816', IGBT: '#8b4eb0' }
const sourceColor = id => {
  let hash = 0
  for (const char of id) hash = (hash * 31 + char.charCodeAt(0)) >>> 0
  return `hsl(${hash % 360} 55% 42%)`
}
const axisScales = computed(() => Object.fromEntries([props.xKey, props.yKey].map(key => {
  const factor = { qg_c: 1e9, ciss_f: 1e12, coss_f: 1e12, rds_qg: 1e9, rds_coss: 1e12 }[key] || 1
  return [key, props.points.some(p => p.metrics[key] != null && !Number.isFinite(p.metrics[key] * factor)) ? 1 : factor]
})))
const scale = key => axisScales.value[key]
const unit = key => (scale(key) !== 1 && { qg_c: 'nC', ciss_f: 'pF', coss_f: 'pF', rds_qg: 'Ω·nC', rds_coss: 'Ω·pF' }[key])
  || props.metrics[key]?.unit.replace('source currency', props.currency).replace('currency', props.currency)
const label = key => `${props.metrics[key]?.label} (${props.relative ? 'relative to positive minimum' : unit(key)})`
const fmt = n => Math.abs(n) < 0.001 || Math.abs(n) >= 1e6 ? n.toExponential(2) : Number(n.toPrecision(4)).toString()
const value = (p, key) => props.relative ? p.normalization?.[key]?.relative : p.metrics[key] == null ? null : p.metrics[key] * scale(key)
const completePairs = computed(() => props.points.filter(p =>
  [value(p, props.xKey), value(p, props.yKey)].every(n => Number.isFinite(n) && n >= 0)))
const plotted = computed(() => completePairs.value.filter(p =>
  (!logX.value || value(p, props.xKey) > 0) && (!logY.value || value(p, props.yKey) > 0)))

function render() {
  if (!canvas.value) return
  chart?.destroy()
  const groupKey = p => p.source_id || p.technology
  const groups = [...new Set(plotted.value.map(groupKey))]
  chart = new Chart(canvas.value, {
    type: 'scatter',
    data: { datasets: groups.map(group => ({
      label: group,
      data: plotted.value.filter(p => groupKey(p) === group).map(p => ({
        x: value(p, props.xKey), y: value(p, props.yKey), record: p,
      })),
      backgroundColor: plotted.value.some(p => p.source_id === group) ? sourceColor(group) : colors[group] || '#64748b',
      pointRadius: c => c.raw?.record.id === props.selectedId ? 9 : 4,
      pointStyle: c => c.raw?.record.pareto ? 'triangle' : 'circle',
      pointBorderColor: c => c.raw?.record.id === props.selectedId ? '#e11d48' : '#ffffff',
      pointBorderWidth: c => c.raw?.record.id === props.selectedId ? 3 : 0.8,
      pointHoverRadius: 8,
    })) },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      onClick: (event, elements) => {
        if (elements.length) {
          const e = elements[0]
          emit('select', chart.data.datasets[e.datasetIndex].data[e.index].record.id)
        }
      },
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true } },
        tooltip: { callbacks: {
          title: items => items[0].raw.record.part_number,
          label: c => [`${label(props.xKey)}: ${fmt(c.parsed.x)}`, `${label(props.yKey)}: ${fmt(c.parsed.y)}`],
          afterLabel: c => [c.raw.record.manufacturer, c.raw.record.technology, `${c.raw.record.source_id} · row ${c.raw.record.row}`, c.raw.record.pareto == null ? 'Pareto unavailable: chosen objectives' : c.raw.record.pareto ? 'Pareto: chosen objectives' : 'Dominated: chosen objectives'],
        } },
        zoom: {
          zoom: { wheel: { enabled: true, modifierKey: 'ctrl' }, pinch: { enabled: true }, mode: 'xy' },
          pan: { enabled: true, modifierKey: 'shift', mode: 'xy' },
        },
      },
      scales: {
        x: { type: logX.value ? 'logarithmic' : 'linear', title: { display: true, text: label(props.xKey) }, ticks: { callback: fmt } },
        y: { type: logY.value ? 'logarithmic' : 'linear', title: { display: true, text: label(props.yKey) }, ticks: { callback: fmt } },
      },
    },
  })
}
watch(() => [props.points, props.xKey, props.yKey, props.relative, logX.value, logY.value], () => nextTick(render), { immediate: true })
watch(() => props.selectedId, () => chart?.update('none'))
onBeforeUnmount(() => chart?.destroy())
function download() {
  const a = document.createElement('a')
  a.href = chart.toBase64Image()
  a.download = `catalog-${props.xKey}-${props.yKey}.png`
  a.click()
}
</script>

<template>
  <article class="scatter-card">
    <div class="scatter-heading"><h3>{{ title }}</h3><span>{{ plotted.length }} points</span></div>
    <p class="axis-labels">X: {{ label(xKey) }} · Y: {{ label(yKey) }}</p>
    <p class="coverage">{{ matchedCount }} candidates · {{ excludedCount }} unavailable physical pairs<span v-if="relative"> · {{ points.length - completePairs.length }} unavailable relative pairs</span><span v-if="completePairs.length !== plotted.length"> · {{ completePairs.length - plotted.length }} pairs outside the selected axis scale</span></p>
    <details v-if="exclusions?.length"><summary>Excluded pairs and reasons</summary><p v-for="entry in exclusions" :key="entry.id">{{ entry.id }}: {{ Object.entries(entry.reasons).map(([key, reason]) => `${key}: ${reason}`).join('; ') }}</p></details>
    <div class="scatter-actions">
      <label><input v-model="logX" type="checkbox"> Log X</label>
      <label><input v-model="logY" type="checkbox"> Log Y</label>
      <button type="button" @click="chart?.resetZoom()">Reset zoom</button>
      <button type="button" :disabled="!plotted.length" @click="download">PNG</button>
    </div>
    <div class="scatter-canvas"><canvas ref="canvas" :aria-label="title" role="img"></canvas></div>
    <p v-if="!plotted.length" class="empty-plot">{{ completePairs.length ? 'All complete pairs fall outside the selected axis scale.' : 'No complete pairs for these axes.' }}</p>
  </article>
</template>

<style scoped>
.scatter-card { position: relative; background: var(--fom-card); border: 1px solid var(--fom-line); border-radius: 12px; padding: 18px; min-width: 0; }
.scatter-heading, .scatter-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.scatter-heading { justify-content: space-between; }
h3 { font-size: 15px; margin: 0 0 12px; color: var(--fom-text); }
.scatter-heading span { font-size: 12px; color: var(--fom-muted); }
.axis-labels { font-size: 12px; line-height: 1.5; color: var(--fom-muted); overflow-wrap: anywhere; }
.scatter-actions { font-size: 12px; margin-bottom: 12px; }
.scatter-actions label { display: flex; align-items: center; gap: 5px; }
button { background: var(--fom-bg); border: 1px solid var(--fom-line); color: var(--fom-text); padding: 5px 9px; border-radius: 5px; cursor: pointer; }
.scatter-canvas { height: 300px; }
.empty-plot { position: absolute; inset: 52% 15% auto; text-align: center; background: var(--fom-card); }
</style>
