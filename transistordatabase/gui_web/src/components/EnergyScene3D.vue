<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { buildScene, defaultCamera, FACE_GRID_LIMIT, moveCamera, normalize, pickPoint, POINT_LIMIT, projectNormalized, projectScene, sourceObservations } from './energyScene.js'

const props = defineProps({ result: { type: Object, required: true }, mode: { type: String, required: true } })
const canvas = ref(null), camera = ref(defaultCamera()), presentation = ref('scatter'), selectedId = ref(null), page = ref(0)
const size = ref({ width: 720, height: 430 })
const observations = computed(() => sourceObservations(props.result, props.mode))
const scene = computed(() => buildScene(observations.value))
const projected = computed(() => projectScene(scene.value, camera.value, size.value.width, size.value.height))
const selected = computed(() => observations.value.find(point => point.id === selectedId.value))
const coincident = computed(() => selected.value ? observations.value.filter(row => row.current_a === selected.value.current_a &&
  row.voltage_v === selected.value.voltage_v && row.energy_j === selected.value.energy_j) : [])
const rows = computed(() => observations.value.slice(page.value * 50, (page.value + 1) * 50))
const hash = computed(() => props.mode === 'xyz' ? props.result.source.sha256 : props.result.input_sha256)
const pages = computed(() => Math.ceil(observations.value.length / 50))
const display = value => value === null || value === undefined ? 'NULL' : String(value)
let observer, dragging = null
function select(point) {
  selectedId.value = point?.id ?? null
  if (point) page.value = Math.floor(observations.value.findIndex(row => row.id === point.id) / 50)
}
function color(point) { return `hsl(${195 - 145 * ((normalize(point.energy_j, scene.value.energies) + 1) / 2)} 70% 45%)` }
function draw() {
  const node = canvas.value
  if (!node) return
  const ctx = node.getContext('2d')
  if (!ctx) return
  const { width, height } = size.value, dpr = window.devicePixelRatio || 1
  node.width = Math.round(width * dpr); node.height = Math.round(height * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.fillStyle = '#f5f8fc'; ctx.fillRect(0, 0, width, height)
  const origin = projectNormalized(-1, -1, -1, camera.value, width, height)
  ctx.font = '12px sans-serif'; ctx.lineWidth = 1
  for (const [x, y, z, label, values, stroke] of [[1,-1,-1,'I [A]',scene.value.currents,'#286aa2'],[-1,1,-1,'V [V]',scene.value.voltages,'#38734c'],[-1,-1,1,'E [J]',scene.value.energies,'#974b3a']]) {
    const end = projectNormalized(x, y, z, camera.value, width, height)
    ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(end.x, end.y); ctx.strokeStyle = stroke; ctx.stroke()
    ctx.fillStyle = stroke; ctx.textAlign = 'center'
    ctx.fillText(label, Math.max(28, Math.min(width - 28, end.x)), Math.max(14, Math.min(height - 8, end.y - 12)))
    ctx.fillText(values.length ? `${values[0]} … ${values[values.length - 1]}` : 'No finite values', Math.max(75, Math.min(width - 75, end.x)), Math.max(27, Math.min(height - 8, end.y + 5)))
  }
  if (presentation.value === 'surface') for (const face of projected.value.faces) {
    ctx.beginPath(); ctx.moveTo(face[0].x, face[0].y); ctx.lineTo(face[1].x, face[1].y); ctx.lineTo(face[2].x, face[2].y); ctx.closePath()
    ctx.fillStyle = color(face[0]); ctx.globalAlpha = .24; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = '#68929e'; ctx.stroke()
  }
  for (const point of projected.value.points) {
    ctx.beginPath(); ctx.arc(point.x, point.y, point.id === selectedId.value ? 7 : 4.5, 0, Math.PI * 2)
    ctx.fillStyle = color(point); ctx.fill(); ctx.strokeStyle = point.id === selectedId.value ? '#111' : point.available ? '#fff' : '#a92828'; ctx.lineWidth = point.available ? 1.5 : 2.5; ctx.stroke()
  }
}
function change(key) { camera.value = moveCamera(camera.value, key) }
function keyboard(event) {
  if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','r','R','Home'].includes(event.key)) { event.preventDefault(); change(event.key) }
}
function down(event) { dragging = { x: event.clientX, y: event.clientY, moved: false }; canvas.value.setPointerCapture?.(event.pointerId) }
function move(event) {
  if (!dragging) return
  const dx = event.clientX - dragging.x, dy = event.clientY - dragging.y
  if (Math.abs(dx) + Math.abs(dy) < 3) return
  camera.value = { ...camera.value, yaw: camera.value.yaw + dx * .008, pitch: Math.max(-1.4, Math.min(1.4, camera.value.pitch + dy * .008)) }
  dragging = { x: event.clientX, y: event.clientY, moved: true }
}
function up(event) {
  if (dragging && !dragging.moved) {
    const box = canvas.value.getBoundingClientRect()
    select(pickPoint(projected.value.points, (event.clientX - box.left) * size.value.width / box.width, (event.clientY - box.top) * size.value.height / box.height))
  }
  dragging = null
}
watch(() => props.result, () => { camera.value = defaultCamera(); selectedId.value = null; page.value = 0 })
watch([projected, presentation, selectedId], draw, { flush: 'post' })
onMounted(() => {
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(entries => { size.value = { width: Math.max(220, entries[0].contentRect.width), height: 430 } })
    observer.observe(canvas.value)
  }
  draw()
})
onBeforeUnmount(() => observer?.disconnect())
</script>

<template>
  <section class="scene" data-testid="energy-3d">
    <h4>Three-axis energy observations</h4>
    <p>X: I [A] · Y: V [V] · Z: E [J]. All axes use linear numeric spacing. Constant axes place observations at the center.</p>
    <p>Each finite original source row has one point, including repeated coordinates. Unknown energy has no point or substituted zero.</p>
    <p class="hash">{{ mode === 'xyz' ? 'Source' : 'Input' }} SHA-256: {{ hash }}</p>
    <label>3D presentation<select v-model="presentation" aria-label="3D presentation"><option value="scatter">Source scatter</option><option value="surface">Source scatter with visual surface connections</option></select></label>
    <p>Surface connections are visual only. They are not measured or interpolated energy. Faces require four unique available original corners.</p>
    <p v-if="mode === 'prepared'">Only accepted source slices appear. Overlapping source domains make corners ambiguous. Displayed interpolated cells and estimates supply no points.</p>
    <p data-testid="energy-3d-counts">{{ observations.length }} source rows · {{ scene.points.length }} finite points · {{ scene.renderedPoints.length }} rendered points · {{ scene.faces.length }} eligible triangles · {{ presentation === 'surface' ? scene.faces.length : 0 }} drawn triangles · {{ observations.length - scene.points.length }} unknown energy rows</p>
    <p v-if="scene.capped" role="status">The {{ POINT_LIMIT }} point render limit is exceeded. The canvas contains no observations or faces. All rows remain in the table and downloads.</p>
    <p v-if="scene.faceCapped" role="status">The {{ FACE_GRID_LIMIT }} coordinate-grid limit is exceeded. Surface connections are disabled. Original rows remain available.</p>
    <div class="controls" aria-label="3D camera controls"><button type="button" @click="change('ArrowLeft')">Rotate left</button><button type="button" @click="change('ArrowRight')">Rotate right</button><button type="button" @click="change('ArrowUp')">Tilt up</button><button type="button" @click="change('ArrowDown')">Tilt down</button><button type="button" @click="change('+')">Zoom in</button><button type="button" @click="change('-')">Zoom out</button><button type="button" @click="change('r')">Reset 3D view</button></div>
    <p>Drag to rotate. Select a point or a source row. Keyboard: arrow keys rotate, + and - zoom, R or Home resets.</p>
    <canvas ref="canvas" class="plot" tabindex="0" role="img" aria-label="Interactive three-axis energy source observations. Use the linked source table for values and selection." :data-yaw="camera.yaw" :data-pitch="camera.pitch" :data-zoom="camera.zoom" @keydown="keyboard" @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="dragging = null">The linked source table contains each original observation.</canvas>
    <p aria-live="polite" data-testid="energy-3d-selection">{{ selected ? `Selected ${selected.id}: I ${display(selected.current_a)} A, V ${display(selected.voltage_v)} V, E ${display(selected.energy_j)} J, ${selected.cell_status}.` : 'No source row selected.' }}</p>
    <section v-if="selected" class="evidence" data-testid="energy-3d-evidence"><h5>Selected original source row: {{ selected.id }}</h5><pre>{{ JSON.stringify(selected, null, 2) }}</pre>
      <div v-if="coincident.length > 1" data-testid="energy-3d-coincident"><p>{{ coincident.length }} original rows share these exact I, V and E coordinates. Select an ID below. All IDs remain in the source table.</p><button v-for="row in coincident.slice(0,50)" :key="row.id" type="button" @click="select(row)">{{ row.id }}</button><p v-if="coincident.length > 50">This list shows the first 50 IDs. Use the paged source table for the remaining IDs.</p></div>
    </section>
    <details open><summary>Linked accessible 3D source table</summary><p>All {{ observations.length }} original rows are available in pages of 50. NULL means unknown. Recorded 0 remains 0.</p>
      <div v-if="pages > 1" class="controls"><button type="button" :disabled="page === 0" @click="page--">Previous 3D rows</button><span>Page {{ page + 1 }} / {{ pages }}</span><button type="button" :disabled="page + 1 >= pages" @click="page++">Next 3D rows</button></div>
      <div class="table-scroll"><table aria-label="3D original source observations"><thead><tr><th scope="col">Source row</th><th scope="col">I [A]</th><th scope="col">V [V]</th><th scope="col">E [J]</th><th scope="col">Cell status / source</th></tr></thead><tbody><tr v-for="row in rows" :key="row.id" :class="{ selected: row.id === selectedId }"><th scope="row"><button type="button" :aria-pressed="row.id === selectedId" :data-source-row="row.id" @click="select(row)">{{ row.id }}</button></th><td>{{ display(row.current_a) }}</td><td>{{ display(row.voltage_v) }}</td><td>{{ display(row.energy_j) }}</td><td>{{ row.cell_status }} / {{ row.source }} / {{ row.source_kind }}</td></tr></tbody></table></div>
    </details>
  </section>
</template>

<style scoped>
.scene { margin:1.5rem 0; min-width:0; border:1px solid var(--border-color,#aaa); border-radius:8px; padding:1rem; }p { line-height:1.6; }.hash,pre { overflow-wrap:anywhere; }pre { white-space:pre-wrap; }.plot { display:block; width:100%; height:430px; touch-action:none; border:1px solid #9badbe; border-radius:6px; }.plot:focus { outline:3px solid #357caf; outline-offset:2px; }.controls { display:flex; flex-wrap:wrap; align-items:center; gap:.3rem; }button,select { padding:.5rem; color:inherit; background:var(--bg-secondary); border:1px solid #7b97ab; border-radius:4px; }button { cursor:pointer; }button:disabled { opacity:.5; }label { display:grid; gap:.4rem; }select { max-width:100%; }.table-scroll { overflow-x:auto; max-width:100%; }table { width:100%; border-collapse:collapse; }th,td { border:1px solid var(--border-color,#aaa); padding:.4rem; text-align:left; overflow-wrap:anywhere; }.selected { background:rgba(13,148,180,.18); }.evidence { border-left:3px solid #357caf; padding-left:.8rem; }summary { cursor:pointer; font-weight:600; }@media(max-width:600px) { .scene { padding:.5rem; } }
</style>
