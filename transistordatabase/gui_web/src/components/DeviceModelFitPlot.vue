<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { displayNumber as number } from './analysisFiles.js'
const props = defineProps({ plot: { type: Object, required: true }, queries: { type: Array, default: () => [] } })
const container = ref(null), width = ref(700)
let observer
onMounted(() => {
  observer = new ResizeObserver(entries => { if (entries[0].contentRect.width > 0) width.value = Math.max(280, entries[0].contentRect.width) })
  observer.observe(container.value)
})
onBeforeUnmount(() => observer?.disconnect())
const right = computed(() => width.value - 25), plotWidth = computed(() => right.value - 88)
const finite = point => Number.isFinite(point.x) && Number.isFinite(point.y)
const observations = computed(() => props.plot.observations.filter(finite))
const samples = computed(() => props.plot.samples.filter(finite))
const queryPoints = computed(() => props.queries.filter(finite))
const bounds = computed(() => {
  const points = [...samples.value, ...observations.value, ...queryPoints.value]
  const range = key => {
    const values = points.map(point => point[key]), low = Math.min(...values), high = Math.max(...values)
    const padding = (high - low) * .08 || Math.abs(low) * .08 || 1
    return [low - padding, high + padding]
  }
  return { x: range('x'), y: range('y') }
})
const x = value => 88 + (value - bounds.value.x[0]) / (bounds.value.x[1] - bounds.value.x[0]) * plotWidth.value
const y = value => 280 - (value - bounds.value.y[0]) / (bounds.value.y[1] - bounds.value.y[0]) * 245
const line = computed(() => samples.value.map(point => `${x(point.x)},${y(point.y)}`).join(' '))
const triangle = point => `${x(point.x)},${y(point.y) - 6} ${x(point.x) - 6},${y(point.y) + 5} ${x(point.x) + 6},${y(point.y) + 5}`
const description = point => `${point.id}: ${point.source}; ${point.series}; ${point.split}; (${number(point.x)}, ${number(point.y)})`
</script>

<template>
  <figure ref="container" class="model-fit-plot">
    <figcaption>{{ plot.id }} · {{ plot.output }} <span>{{ plot.conditions }}</span></figcaption>
    <p>Support: {{ plot.support.map(number).join(' to ') }} · {{ plot.x_label }}<template v-if="plot.normalization !== null"> · Normalization: {{ number(plot.normalization) }} J</template></p>
    <svg v-if="samples.length || observations.length" :viewBox="`0 0 ${width} 345`" role="img" :aria-label="`${plot.output} fit against ${plot.x_label}`" data-testid="fit-plot">
      <path :d="`M88 30V280H${right}`" fill="none" stroke="#68788a" />
      <template v-for="fraction in [0, .5, 1]" :key="fraction">
        <text :x="88 + fraction * plotWidth" y="301" :text-anchor="fraction === 1 ? 'end' : fraction === 0 ? 'start' : 'middle'">{{ number(bounds.x[0] + fraction * (bounds.x[1] - bounds.x[0])) }}</text>
        <text x="80" :y="284 - fraction * 245" text-anchor="end">{{ number(bounds.y[0] + fraction * (bounds.y[1] - bounds.y[0])) }}</text>
      </template>
      <polyline :points="line" fill="none" stroke="#2674aa" stroke-width="2" data-kind="fit" />
      <template v-for="point in observations" :key="point.id">
        <circle v-if="point.split === 'train'" :cx="x(point.x)" :cy="y(point.y)" r="4" fill="#176644" data-kind="train"><title>{{ description(point) }}</title></circle>
        <polygon v-else :points="triangle(point)" fill="#b85612" data-kind="holdout"><title>{{ description(point) }}</title></polygon>
      </template>
      <path v-for="(point, index) in queryPoints" :key="index" :d="`M${x(point.x)-6} ${y(point.y)}h12M${x(point.x)} ${y(point.y)-6}v12`" stroke="#8a2ba5" stroke-width="3" data-kind="query"><title>Query: {{ number(point.x) }}, {{ number(point.y) }}</title></path>
      <text :x="88 + plotWidth / 2" y="335" text-anchor="middle">{{ plot.x_label }}</text>
      <text transform="translate(17 157) rotate(-90)" text-anchor="middle">{{ plot.y_label }}</text>
    </svg>
    <p class="legend">● Training observations · ▲ Held-out observations · blue line: Python model samples · + Query</p>
  </figure>
</template>

<style scoped>
.model-fit-plot { margin: 0; border: 1px solid #cbd5df; border-radius: 6px; padding: .8rem; min-width: 0; background: white; }
figcaption { font-weight: 600; overflow-wrap: anywhere; } figcaption span { font-weight: normal; } svg { width: 100%; display: block; } text { font-size: 12px; fill: #263444; }
p { font-size: .85rem; line-height: 1.4; overflow-wrap: anywhere; } .legend { margin-bottom: 0; }
</style>
