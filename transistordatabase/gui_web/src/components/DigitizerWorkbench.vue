<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { digitizerApi } from '../services/api.js'

const blankAxis = () => ({ pixel_1: '', value_1: '', pixel_2: '', value_2: '', scale: 'linear', unit: '' })
const axes = reactive({ x: blankAxis(), y: blankAxis() })
const image = ref(null), points = ref([]), history = ref([]), result = ref(null)
const source = ref(''), allowExtrapolation = ref(false), mode = ref('point')
const loading = ref(false), busy = ref(false), error = ref(''), notice = ref(''), stale = ref(false)
const manual = reactive({ x: '', y: '' })
let revision = 0, uploadSerial = 0, disposed = false
let pointerCoordinates = null, pointerStart = null, ignoreClick = false
function invalidate() {
  revision++
  stale.value = !!result.value || stale.value || busy.value
  result.value = null
  error.value = ''
  busy.value = false
}
watch([axes, points, source, allowExtrapolation], invalidate, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++; uploadSerial++ })
const markerSize = computed(() => image.value ? Math.max(image.value.metadata.width, image.value.metadata.height) / 100 : 5)
const imageHref = computed(() => image.value ? `data:image/${image.value.metadata.format.toLowerCase()};base64,${image.value.content_base64}` : '')
const axisNames = { x: 'X', y: 'Y' }
function message(failure) {
  const detail = failure.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map(row => row.msg ?? row.message ?? JSON.stringify(row)).join('; ')
  return detail?.message ?? failure.message
}
async function upload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  const previous = image.value
  const token = ++uploadSerial
  invalidate(); image.value = null; loading.value = true; notice.value = ''
  try {
    if (file.size > 8 * 1024 * 1024) throw new Error('Choose an image of 8 MiB or less.')
    const bytes = new Uint8Array(await file.arrayBuffer())
    if (disposed || token !== uploadSerial) return
    let binary = ''
    for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768))
    const content_base64 = btoa(binary)
    const response = await digitizerApi.image({ filename: file.name, content_base64 })
    if (disposed || token !== uploadSerial) return
    const metadata = response.image ?? response
    const identical = previous?.metadata.sha256 === metadata.sha256 && previous.metadata.width === metadata.width && previous.metadata.height === metadata.height
    if (!identical) {
      axes.x = blankAxis(); axes.y = blankAxis(); points.value = []; history.value = []
      allowExtrapolation.value = false; source.value = ''; mode.value = 'point'
    }
    image.value = { metadata, filename: file.name, content_base64 }
    stale.value = false
    notice.value = identical
      ? 'The image bytes are identical. Calibration and selected points are retained. Map again to refresh exports.'
      : 'New image identity. Enter calibration and source information before mapping points.'
  } catch (failure) {
    if (!disposed && token === uploadSerial) {
      axes.x = blankAxis(); axes.y = blankAxis(); points.value = []; history.value = []
      error.value = message(failure)
    }
  } finally { if (!disposed && token === uploadSerial) loading.value = false }
}
function addPoint(point) {
  if (points.value.length >= 20000) { error.value = 'The image supports at most 20000 selected points.'; return }
  history.value.push({ action: 'remove', index: points.value.length })
  points.value.push(point)
}
function editPoint(index, axis, event) {
  history.value.push({ action: 'edit', index, axis, value: points.value[index][axis] })
  points.value[index][axis] = event.target.valueAsNumber
}
function removePoint(index) {
  history.value.push({ action: 'insert', index, point: { ...points.value[index] } })
  points.value.splice(index, 1)
}
function undo() {
  const change = history.value.pop()
  if (!change) return
  if (change.action === 'remove') points.value.splice(change.index, 1)
  else if (change.action === 'insert') points.value.splice(change.index, 0, change.point)
  else points.value[change.index][change.axis] = change.value
}
function startPointer(event) {
  pointerCoordinates = null
  pointerStart = event.isPrimary && event.button === 0 ? { x: event.clientX, y: event.clientY, id: event.pointerId, moved: false } : null
  ignoreClick = !pointerStart
}
function movePointer(event) {
  if (pointerStart && event.pointerId === pointerStart.id && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4) pointerStart.moved = true
}
function rememberPointer(event) {
  movePointer(event)
  if (pointerStart && !pointerStart.moved && event.isPrimary && event.button === 0 && event.pointerId === pointerStart.id) {
    pointerCoordinates = { x: event.clientX, y: event.clientY }
  } else { pointerCoordinates = null; ignoreClick = true }
  pointerStart = null
}
function cancelPointer() {
  pointerStart = null; pointerCoordinates = null; ignoreClick = true
}
function selectPixel(event) {
  if (!image.value || loading.value || event.button !== 0) return
  if (ignoreClick) { ignoreClick = false; pointerCoordinates = null; return }
  const svg = event.currentTarget
  const transform = svg.getScreenCTM()
  if (!transform) return
  const screen = pointerCoordinates ?? { x: event.clientX, y: event.clientY }
  pointerCoordinates = null
  const pixel = new DOMPoint(screen.x, screen.y).matrixTransform(transform.inverse())
  const { width, height } = image.value.metadata
  if (pixel.x < 0 || pixel.x > width - 1 || pixel.y < 0 || pixel.y > height - 1) return
  if (mode.value === 'point') addPoint({ x: pixel.x, y: pixel.y })
  else {
    const [axis, anchor] = mode.value.split(':')
    axes[axis][`pixel_${anchor}`] = pixel[axis]
  }
}
async function transformPoints() {
  if (!image.value || loading.value) return
  error.value = ''; result.value = null; stale.value = false
  const token = ++revision; busy.value = true
  const { metadata, filename, content_base64 } = image.value
  const request = {
    filename, content_base64,
    calibration: {
      image: { sha256: metadata.sha256, width: metadata.width, height: metadata.height },
      x: { ...axes.x }, y: { ...axes.y }, allow_extrapolation: allowExtrapolation.value,
    },
    points: points.value.map(point => ({ ...point })), source: source.value.trim() ? source.value : 'Unknown',
  }
  try {
    const response = await digitizerApi.transform(request)
    if (!disposed && token === revision) result.value = response
  } catch (failure) { if (!disposed && token === revision) error.value = message(failure) }
  finally { if (!disposed && token === revision) busy.value = false }
}
function download(format) {
  if (!result.value) return
  const body = format === 'csv' ? result.value.csv : JSON.stringify(result.value, null, 2)
  const blob = new Blob([body], { type: format === 'csv' ? 'text/csv;charset=utf-8' : 'application/json' })
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = `digitized-points.${format}`; link.click(); URL.revokeObjectURL(url)
}
</script>

<template>
  <section class="digitizer-workbench">
    <h2>Image digitizer</h2>
    <p>Calibrate each axis, select curve points, then map and export the values. Image calibration does not establish device accuracy.</p>
    <div class="card">
      <label>Source image <input data-testid="digitizer-file" type="file" accept="image/png,image/jpeg,.png,.jpg,.jpeg" @change="upload"></label>
      <p>PNG or JPEG · maximum 8 MiB · 16 million pixels · 8192 pixels per side. Rotation metadata must indicate normal orientation. Animated images are unsupported.</p>
      <p v-if="loading" role="status">Checking image bytes and dimensions…</p>
      <p v-if="notice" role="status">{{ notice }}</p>
    </div>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <template v-if="image">
      <div class="card identity">
        <h3>Image identity</h3>
        <p>{{ image.filename }} · {{ image.metadata.width }} × {{ image.metadata.height }} pixels · {{ image.metadata.byte_size }} bytes</p>
        <p>SHA-256: <code data-testid="digitizer-hash">{{ image.metadata.sha256 }}</code></p>
        <p>Natural image pixels. Origin: top left. X increases right. Y increases down. Pixel centers span 0 to width − 1 and 0 to height − 1. The original image bytes are retained.</p>
      </div>
      <div class="workspace">
        <div class="card calibration">
          <h3>Axis calibration</h3>
          <p>Set each pixel anchor numerically, or select its button and click the image. X and Y anchors are independent.</p>
          <fieldset v-for="(label, axis) in axisNames" :key="axis">
            <legend>{{ label }} axis</legend>
            <div class="axis-settings">
              <label>Scale <select v-model="axes[axis].scale" :data-testid="`digitizer-${axis}-scale`"><option value="linear">Linear</option><option value="log">Logarithmic (base 10)</option></select></label>
              <label>Unit <input v-model="axes[axis].unit" :data-testid="`digitizer-${axis}-unit`" placeholder="For example: V, A, nF, or 1"></label>
            </div>
            <div v-for="anchor in [1, 2]" :key="anchor" class="anchor">
              <button type="button" :aria-pressed="mode === `${axis}:${anchor}`" :data-testid="`digitizer-${axis}-anchor-${anchor}`" @click="mode = `${axis}:${anchor}`">Select {{ label }} anchor {{ anchor }}</button>
              <label>Pixel {{ anchor }} <input v-model.number="axes[axis][`pixel_${anchor}`]" :data-testid="`digitizer-${axis}-pixel-${anchor}`" type="number" step="any"></label>
              <label>Value {{ anchor }} <input v-model.number="axes[axis][`value_${anchor}`]" :data-testid="`digitizer-${axis}-value-${anchor}`" type="number" step="any"></label>
            </div>
          </fieldset>
          <label class="checkbox"><input v-model="allowExtrapolation" data-testid="digitizer-extrapolate" type="checkbox"> Allow points outside the calibrated intervals</label>
          <label>Source, curve, and test conditions <textarea v-model="source" data-testid="digitizer-source" maxlength="2000" rows="3" placeholder="Document, page, curve label, temperature, gate voltage. State unknown conditions explicitly."></textarea></label>
          <p>Units and conditions are supplied by you. No device field or unit conversion is inferred.</p>
        </div>
        <div class="card image-panel">
          <h3>Select image pixels</h3>
          <button type="button" :aria-pressed="mode === 'point'" data-testid="digitizer-point-mode" @click="mode = 'point'">Select curve points</button>
          <p role="status">{{ mode === 'point' ? 'Click to add a curve point.' : `Click to set ${mode.startsWith('x') ? 'X' : 'Y'} anchor ${mode.endsWith('1') ? '1' : '2'}.` }}</p>
          <svg data-testid="digitizer-image" :viewBox="`0 0 ${image.metadata.width} ${image.metadata.height}`" :style="{ aspectRatio: `${image.metadata.width} / ${image.metadata.height}` }" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Source image with calibration anchors and selected points" @pointerdown="startPointer" @pointermove="movePointer" @pointerup="rememberPointer" @pointercancel="cancelPointer" @click="selectPixel">
            <image :href="imageHref" :width="image.metadata.width" :height="image.metadata.height" />
            <g class="x-anchors"><line v-for="anchor in [1, 2]" v-show="axes.x[`pixel_${anchor}`] !== ''" :key="anchor" :x1="axes.x[`pixel_${anchor}`]" :x2="axes.x[`pixel_${anchor}`]" y1="0" :y2="image.metadata.height" /></g>
            <g class="y-anchors"><line v-for="anchor in [1, 2]" v-show="axes.y[`pixel_${anchor}`] !== ''" :key="anchor" x1="0" :x2="image.metadata.width" :y1="axes.y[`pixel_${anchor}`]" :y2="axes.y[`pixel_${anchor}`]" /></g>
            <g v-for="(point, index) in points" :key="index" class="point-marker"><circle :cx="point.x" :cy="point.y" :r="markerSize" /><text :x="point.x + markerSize" :y="point.y - markerSize" :font-size="markerSize * 2">{{ index + 1 }}</text></g>
          </svg>
          <p>Clicks use the original image coordinates at every display size. Colored lines show X (blue) and Y (orange) anchors.</p>
        </div>
      </div>
      <section class="card">
        <h3>Selected points</h3>
        <form class="point-entry" @submit.prevent="addPoint({ x: Number(manual.x), y: Number(manual.y) })">
          <label>X pixel <input v-model="manual.x" data-testid="digitizer-manual-x" type="number" step="any" required></label>
          <label>Y pixel <input v-model="manual.y" data-testid="digitizer-manual-y" type="number" step="any" required></label>
          <button type="submit">Add pixel point</button>
          <button type="button" :disabled="!history.length" data-testid="digitizer-undo" @click="undo">Undo point change</button>
        </form>
        <p>Points retain selection order. Maximum 20000 points. Edit pixel coordinates or remove individual points, then map again.</p>
        <div class="table-scroll"><table>
          <thead><tr><th>Order</th><th>X pixel</th><th>Y pixel</th><th>X {{ result ? `(${result.units.x})` : '' }}</th><th>Y {{ result ? `(${result.units.y})` : '' }}</th><th>Action</th></tr></thead>
          <tbody><tr v-for="(point, index) in points" :key="index" data-testid="digitizer-point-row">
            <td>{{ index + 1 }}</td>
            <td><input :value="point.x" type="number" step="any" :aria-label="`Point ${index + 1} X pixel`" @change="editPoint(index, 'x', $event)"></td>
            <td><input :value="point.y" type="number" step="any" :aria-label="`Point ${index + 1} Y pixel`" @change="editPoint(index, 'y', $event)"></td>
            <td data-testid="digitizer-physical-x">{{ result?.points[index]?.x ?? 'Map required' }}</td>
            <td data-testid="digitizer-physical-y">{{ result?.points[index]?.y ?? 'Map required' }}</td>
            <td><button type="button" :aria-label="`Remove point ${index + 1}`" @click="removePoint(index)">Remove</button></td>
          </tr></tbody>
        </table></div>
        <div class="actions">
          <button type="button" :disabled="busy || loading" data-testid="digitizer-map" @click="transformPoints">{{ busy ? 'Mapping…' : 'Map calibrated points' }}</button>
          <button type="button" :disabled="!result" data-testid="digitizer-csv" @click="download('csv')">Download physical CSV</button>
          <button type="button" :disabled="!result" data-testid="digitizer-json" @click="download('json')">Download points and calibration JSON</button>
        </div>
        <p v-if="stale" role="status">Inputs changed. Map again to refresh values and exports.</p>
        <p v-if="result" role="status" data-testid="digitizer-results">{{ result.points.length }} points mapped by Python. CSV and JSON contain these same physical values.</p>
        <details v-if="result"><summary>Calibration and source metadata</summary><pre>{{ JSON.stringify({ image: result.image, calibration: result.calibration, source: result.source, units: result.units }, null, 2) }}</pre></details>
      </section>
    </template>
  </section>
</template>

<style scoped>
.digitizer-workbench { max-width: 1600px; margin: auto; padding: 1.5rem; display: grid; gap: 1rem; }
.card { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; min-width: 0; }
.workspace { display: grid; grid-template-columns: minmax(340px, 0.9fr) minmax(0, 1.4fr); gap: 1rem; align-items: start; }
h3 { margin-bottom: .6rem; } p { margin: .4rem 0 .8rem; }
label { display: flex; flex-direction: column; gap: .25rem; margin: .4rem 0; min-width: 0; }
input, select, textarea { color: var(--text-primary); background: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 4px; padding: .5rem; min-width: 0; max-width: 100%; font: inherit; }
input[type=file] { width: 100%; } textarea { width: 100%; }
fieldset { border: 1px solid var(--border-color); border-radius: 6px; padding: .7rem; margin: .7rem 0; min-width: 0; }
.axis-settings, .anchor { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: .5rem; align-items: end; }
.axis-settings input, .axis-settings select, .anchor input { width: 100%; }
.anchor button { grid-column: 1 / -1; justify-self: start; }
.checkbox { flex-direction: row; align-items: center; gap: .5rem; }
button { background: var(--accent-blue); color: white; border: 1px solid transparent; border-radius: 4px; padding: .55rem .75rem; cursor: pointer; font: inherit; }
button[aria-pressed=true] { outline: 3px solid #f6b846; outline-offset: 1px; } button:disabled { opacity: .5; cursor: default; }
.identity code, pre { overflow-wrap: anywhere; white-space: pre-wrap; } .identity code { font-size: .85rem; }
svg { display: block; width: 100%; max-height: 70vh; background: #e5e5e5; cursor: crosshair; touch-action: manipulation; }
svg line { stroke-width: 1.5; vector-effect: non-scaling-stroke; stroke-dasharray: 5 4; pointer-events: none; }
.x-anchors line { stroke: #0065d1; } .y-anchors line { stroke: #e56b00; }
.point-marker { pointer-events: none; } .point-marker circle { fill: #df1748; stroke: white; stroke-width: 1; vector-effect: non-scaling-stroke; }
.point-marker text { fill: #b00030; stroke: white; stroke-width: .4px; paint-order: stroke; font-weight: bold; }
.point-entry, .actions { display: flex; flex-wrap: wrap; gap: .6rem; align-items: end; margin: .7rem 0; }
.point-entry label { flex: 1 1 120px; max-width: 210px; } .table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; } th, td { padding: .5rem; border-bottom: 1px solid var(--border-color); text-align: left; }
td input { width: 125px; } .error { color: #b62222; border: 1px solid currentColor; border-radius: 5px; padding: .7rem; }
@media(max-width: 900px) { .workspace { grid-template-columns: 1fr; } .digitizer-workbench { padding: .75rem; } }
</style>
