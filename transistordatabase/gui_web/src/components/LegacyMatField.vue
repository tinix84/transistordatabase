<script setup>
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from 'vue'
import { matApi } from '../services/api.js'
const props = defineProps({ node: { type: Object, required: true }, name: { type: String, required: true }, source: { type: Object, required: true }, sourceHash: { type: String, required: true } })
const resolved = shallowRef(null), error = ref(''), busy = ref(false), page = ref(0), canvas = ref(null)
let revision = 0, disposed = false
const values = computed(() => props.node.values?.flat(Infinity) ?? [])
const count = computed(() => props.node.shape.reduce((a, b) => a * b, 1))
const text = computed(() => props.node.dtype?.includes('U') && !resolved.value ? values.value.join('') : null)
const visible = computed(() => values.value.slice(page.value * 100, (page.value + 1) * 100))
watch(() => [props.node, props.source], () => { revision++; resolved.value = null; error.value = ''; busy.value = false; page.value = 0 }, { flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
function decode(blob) {
  const bytes = Uint8Array.from(atob(blob.data_base64), c => c.charCodeAt(0))
  if (bytes.length !== blob.byte_length || blob.source_sha256 !== props.sourceHash) throw new Error('The array does not match the current source')
  const match = /^([<>=|])([biufU])(\d+)$/.exec(blob.dtype)
  if (!match) throw new Error(`Unsupported array display dtype ${blob.dtype}. The original bytes remain in the archive.`)
  const [, endian, kind, widthText] = match, width = blob.itemsize ?? Number(widthText) * (kind === 'U' ? 4 : 1), view = new DataView(bytes.buffer), little = endian !== '>'
  if (bytes.length % width) throw new Error('Array byte count does not match its dtype')
  const methods = { f4: 'getFloat32', f8: 'getFloat64', i1: 'getInt8', i2: 'getInt16', i4: 'getInt32', i8: 'getBigInt64', u1: 'getUint8', u2: 'getUint16', u4: 'getUint32', u8: 'getBigUint64', b1: 'getUint8' }
  if (kind !== 'U' && !methods[kind + width]) throw new Error('This array dtype has no browser display adapter')
  const length = bytes.length / width
  // Keep the full array in binary form. Decode only the selected table page.
  return { bytes, length, value(index) {
    if (kind === 'U') {
      let result = ''
      for (let j = 0; j < width; j += 4) result += String.fromCodePoint(view.getUint32(index * width + j, little))
      return result.replace(/\0+$/, '')
    }
    return view[methods[kind + width]](index * width, little).toString()
  } }
}
const binaryRows = computed(() => {
  const data = resolved.value
  if (!data) return []
  return Array.from({ length: Math.min(100, data.length - page.value * 100) }, (_, i) => data.value(i + page.value * 100))
})
const shown = computed(() => resolved.value ? binaryRows.value : visible.value)
async function inspect() {
  error.value = ''; busy.value = true; const token = ++revision
  try {
    const blob = await matApi.array({ ...props.source, array_path: props.node.source_array_path })
    if (disposed || token !== revision) return
    resolved.value = decode(blob); page.value = 0
    await nextTick()
    if (disposed || token !== revision || !canvas.value) return
    const [height, width, channels = 1] = blob.shape
    if (!['|u1', '<u1', '>u1'].includes(blob.dtype) || ![1, 3, 4].includes(channels) || blob.shape.length > 3 || width * height * channels !== blob.byte_length) throw new Error('Image display requires an original uint8 grayscale, RGB, or RGBA array')
    canvas.value.width = width; canvas.value.height = height
    const ctx = canvas.value.getContext('2d'), pixels = ctx.createImageData(width, height), source = resolved.value.bytes
    for (let i = 0; i < width * height; i++) {
      pixels.data[i * 4] = source[i * channels]
      pixels.data[i * 4 + 1] = source[i * channels + (channels === 1 ? 0 : 1)]
      pixels.data[i * 4 + 2] = source[i * channels + (channels === 1 ? 0 : 2)]
      pixels.data[i * 4 + 3] = channels === 4 ? source[i * 4 + 3] : 255
    }
    ctx.putImageData(pixels, 0, 0)
  } catch (failure) { if (!disposed && token === revision) error.value = failure.response?.data?.detail ?? failure.message }
  finally { if (!disposed && token === revision) busy.value = false }
}
</script>
<template>
  <details class="mat-field">
    <summary>{{ name }} · {{ node.kind === 'struct' ? `${Object.keys(node.fields).length} fields` : `${node.dtype} [${node.shape.join(' × ')}]` }}</summary>
    <template v-if="node.kind === 'struct'">
      <LegacyMatField v-for="(child, key) in node.fields" :key="key" :node="child" :name="key" :source="source" :source-hash="sourceHash" />
    </template>
    <template v-else>
      <p>{{ node.path }} · {{ count }} original elements · {{ node.itemsize }} bytes per element<span v-if="node.nonfinite"> · Nonfinite values: unsupported for scientific finalization</span></p>
      <p v-if="count === 0">Empty source array. The value is unknown, not zero.</p>
      <pre v-else-if="text !== null">{{ text }}</pre>
      <p v-if="node.sha256" class="hash">Array SHA-256 {{ node.sha256 }} · {{ node.byte_length }} bytes</p>
      <button v-if="node.source_array_path" type="button" :disabled="busy" data-testid="mat-inspect-array" @click="inspect">{{ busy ? 'Reading original array…' : node.role === 'original-image' ? 'View original image' : 'Inspect original array' }}</button>
      <figure v-if="resolved && node.role === 'original-image'"><canvas ref="canvas" data-testid="mat-original-image" /><figcaption>Original pixels. No calibration or image transformation is applied.</figcaption></figure>
      <template v-if="text === null && shown.length && node.role !== 'original-image'">
        <div class="table-scroll"><table><caption>Original flattened C-order values · page {{ page + 1 }}</caption><thead><tr><th>Original index</th><th>Value</th></tr></thead><tbody><tr v-for="(value, i) in shown" :key="i"><td>{{ page * 100 + i }}</td><td class="literal">{{ value }}</td></tr></tbody></table></div>
        <div class="pages"><button :disabled="page === 0" @click="page--">Previous 100</button><button :disabled="(page + 1) * 100 >= count" @click="page++">Next 100</button></div>
      </template>
      <p v-if="node.encoding && !resolved">Values remain in the source. Request this array to inspect bounded data.</p>
      <p v-if="error" role="alert">{{ error }}</p>
    </template>
  </details>
</template>
<style scoped>
.mat-field { min-width: 0; margin: .55rem 0; padding: .5rem .7rem; border-left: 2px solid #c0ccda; overflow-wrap: anywhere; }
summary { cursor: pointer; font-weight: 600; } pre,.literal { white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }
.hash { font-size: .85rem; overflow-wrap: anywhere; } canvas { display: block; max-width: 100%; height: auto; } figure { margin: 1rem 0; }
.table-scroll { overflow-x: auto; max-width: 100%; } table { width: 100%; border-collapse: collapse; min-width: 280px; } th,td { text-align: left; padding: .3rem; border-bottom: 1px solid #c0ccda; }
button { margin: .35rem .35rem .35rem 0; padding: .5rem; color: inherit; background: var(--bg-secondary); border: 1px solid #899fb6; border-radius: 4px; cursor: pointer; } button:disabled { opacity: .5; } [role=alert] { color: #b42323; }
</style>
