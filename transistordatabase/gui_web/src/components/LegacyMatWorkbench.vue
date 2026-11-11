<script setup>
import { computed, onBeforeUnmount, ref, shallowRef } from 'vue'
import { matApi } from '../services/api.js'
import LegacyMatField from './LegacyMatField.vue'
import LegacyMatMapping from './LegacyMatMapping.vue'
import CurvePlotter from './CurvePlotter.vue'
const preview = shallowRef(null), source = shallowRef(null), error = ref(''), busy = ref(false), exporting = ref(''), filename = ref('')
let revision = 0, disposed = false
onBeforeUnmount(() => { disposed = true; revision++ })
function encode(bytes) {
  const chunks = []
  for (let i = 0; i < bytes.length; i += 32768) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 32768)))
  return btoa(chunks.join(''))
}
async function message(failure) {
  let detail = failure.response?.data
  if (detail instanceof Blob) { try { detail = JSON.parse(await detail.text()) } catch { detail = null } }
  detail = detail?.detail
  return typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(row => row.msg).join('; ') : failure.message
}
async function upload(event) {
  const file = event.target.files?.[0], token = ++revision
  preview.value = null; source.value = null; error.value = ''; exporting.value = ''; filename.value = file?.name ?? ''; busy.value = !!file
  if (!file) return
  try {
    if (!file.size || file.size > 24 * 1024 * 1024) throw new Error('Select a nonempty MAT file within the 24 MiB source budget')
    const bytes = new Uint8Array(await file.arrayBuffer())
    if (disposed || token !== revision) return
    const request = { source_name: file.name, mat_base64: encode(bytes) }
    const response = await matApi.preview(request)
    if (!disposed && token === revision) { source.value = request; preview.value = response }
  } catch (failure) { const detail = await message(failure); if (!disposed && token === revision) error.value = detail }
  finally { if (!disposed && token === revision) busy.value = false }
}
function save(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url)
}
async function exportMat(mode) {
  if (!source.value || !preview.value) return
  const token = revision; exporting.value = mode; error.value = ''
  try {
    const blob = await matApi.export({ ...source.value, mode })
    if (!disposed && token === revision) save(blob, mode === 'source-preserved' ? 'original-source.mat' : 'legacy-sc-data-0.2.mat')
  } catch (failure) { const detail = await message(failure); if (!disposed && token === revision) error.value = detail }
  finally { if (!disposed && token === revision) exporting.value = '' }
}
function archive() {
  if (preview.value) save(new Blob([JSON.stringify(preview.value, null, 2)], { type: 'application/json' }), 'legacy-mat-source-archive.json')
}
const fields = computed(() => preview.value?.variables.scData?.fields ?? preview.value?.variables.hsData?.fields ?? preview.value?.variables ?? {})
const fieldLabels = { name: 'Original identity', blockingVoltage: 'Blocking voltage', cost: 'Cost · currency unknown', weight: 'Weight · unit unknown', fileFormatVersion: 'Source format version', onStateVoltage: 'Conduction voltage', onStateCurrent: 'Signed conduction current', turnOnEnergy: 'Turn-on energy', turnOnCurrent: 'Signed turn-on current', turnOnVoltage: 'Combined turn-on reference', turnOffEnergy: 'Turn-off energy', turnOffCurrent: 'Signed turn-off current', turnOffVoltage: 'Combined turn-off reference' }
function label(key) { return fieldLabels[key] ?? key.replace(/([a-z])([A-Z])/g, '$1 $2') }
function smallValue(node) {
  if (node.kind === 'struct') return `${Object.keys(node.fields).length} preserved fields`
  const values = node.values?.flat(Infinity)
  if (!node.shape.reduce((a, b) => a * b, 1)) return 'Empty · unknown'
  if (node.dtype.includes('U')) return values ? values.join('') : 'Source array reference'
  return values?.length <= 6 ? values.join(', ') : 'Inspect original array below'
}
const curves = computed(() => {
  const sourceFields = fields.value
  return [['onState', 'Conduction', 'Voltage (V)'], ['turnOn', 'Turn-on', 'Energy (J)'], ['turnOff', 'Turn-off', 'Energy (J)']].map(([stem, title, unit]) => {
    const current = sourceFields[stem + 'Current']?.values?.flat(Infinity), ordinate = sourceFields[stem + (stem === 'onState' ? 'Voltage' : 'Energy')]?.values?.flat(Infinity)
    const pairs = current && ordinate && current.length === ordinate.length ? current.map((value, index) => ({ x: value, y: ordinate[index] })) : []
    return { stem, title, unit, pairs, datasets: [{ label: `${title} · original recorded values`, data: pairs, showLine: false, pointRadius: 3, borderColor: '#296797' }] }
  })
})
</script>
<template>
  <div class="mat-workbench" data-testid="mat-workbench">
    <header><h2>Legacy MAT workbench</h2><p>Inspect original SC data, digitizer traces, heat-sink coefficients, and numeric workspaces. Scientific use requires separate declarations.</p></header>
    <section class="panel"><label class="upload">Original MAT file<input type="file" accept=".mat" data-testid="mat-file" @change="upload"></label>
      <p>Upload exact original bytes. Maximum source size: 24 MiB. Parsing and calculations run in Python.</p><p v-if="busy" role="status">Reading {{ filename }}…</p>
      <p v-if="error" role="alert">{{ error }}</p>
    </section>
    <template v-if="preview">
      <section class="panel" data-testid="mat-preview"><h3>Source identity and parser report</h3><dl class="source-grid">
        <dt>Original filename</dt><dd>{{ preview.source.name }}</dd><dt>Original SHA-256</dt><dd class="hash" data-testid="mat-hash">{{ preview.source.sha256 }}</dd>
        <dt>Original bytes</dt><dd>{{ preview.source.byte_length }}</dd><dt>Container format</dt><dd>{{ preview.source.format }}</dd><dt>Workspace schema</dt><dd>{{ preview.workspace_schema }}</dd>
        <dt>Archive / parser version</dt><dd>{{ preview.schema }} v{{ preview.version }} / parser v{{ preview.parser_version }}</dd><dt>Original SC version</dt><dd>{{ preview.source_version ?? 'Absent · unknown (compatibility conversion treats a missing SC version as 0.0)' }}</dd>
      </dl><p>The filename supplies provenance only. Preview success does not establish physical curve validity or measurement accuracy.</p>
        <p v-if="preview.workspace_schema !== 'scData'" class="notice">Preview and original-byte archive only. This workspace has no supported SC core mapping.</p>
        <p v-if="preview.workspace_schema === 'hsData'">Heat-sink interface coefficients use K·m²/W. An explicit area is necessary to derive a K/W resistance. The filename does not declare an ideal path.</p>
        <div class="actions"><button @click="archive" data-testid="mat-archive-json">Download source archive JSON</button><button @click="exportMat('source-preserved')" :disabled="!!exporting" data-testid="mat-source-export">Download exact original MAT</button><button v-if="preview.workspace_schema === 'scData'" @click="exportMat('legacy-mat5')" :disabled="!!exporting" data-testid="mat-legacy-export">Download semantic MAT5 v0.2</button></div>
        <p>MAT5 export preserves source data semantically. New declarations require native core JSON. Exact archival bytes remain separate.</p>
        <ul v-if="preview.conversion_history.length"><li v-for="(step, i) in preview.conversion_history" :key="i">{{ step.from }} → {{ step.to }}: {{ step.operation }}</li></ul>
      </section>
      <section class="panel"><h3>Source diagnostics</h3><p v-if="!preview.diagnostics.length">No container diagnostic was returned. Conditions and measurement status still require review.</p>
        <div v-else class="table-scroll"><table data-testid="mat-diagnostics"><thead><tr><th>Source field</th><th>Diagnostic</th><th>Meaning</th></tr></thead><tbody><tr v-for="(diagnostic, i) in preview.diagnostics" :key="i"><td>{{ diagnostic.path }}</td><td>{{ diagnostic.code }}</td><td>{{ diagnostic.message }}</td></tr></tbody></table></div>
      </section>
      <section class="panel"><h3>Original fields and proposed units</h3><p>{{ preview.proposed_units.provenance }}</p><p>Unknown conditions remain unknown. Empty values do not mean zero. Recorded thermal zero remains distinct.</p>
        <div class="table-scroll"><table><thead><tr><th>Source quantity</th><th>Original value / structure</th><th>Proposed unit</th><th>Dtype and original shape</th></tr></thead><tbody><tr v-for="(node, key) in fields" :key="key"><td>{{ label(key) }}<small>{{ key }}</small></td><td>{{ smallValue(node) }}</td><td>{{ preview.proposed_units.values[key] ?? 'Unknown / trace labels' }}</td><td>{{ node.dtype ?? 'struct' }} [{{ node.shape.join(' × ') }}]</td></tr></tbody></table></div>
        <p>Known SC fields: {{ preview.known_fields.length }}. Unknown extras: {{ preview.unknown_fields.join(', ') || 'None' }}. The tree preserves all source fields.</p>
        <LegacyMatField v-for="(node, name) in preview.variables" :key="name" :node="node" :name="name" :source="source" :source-hash="preview.source.sha256" />
      </section>
      <section v-if="preview.workspace_schema === 'scData'" class="panel"><h3>Original signed combined curves</h3><p>These points retain source signs and order. Lines do not join branches. Synthetic candidates and negative energy remain visible.</p>
        <details v-for="curve in curves" :key="curve.stem" class="curve"><summary>{{ curve.title }} · {{ curve.pairs.length }} inline source points</summary>
          <template v-if="curve.pairs.length"><CurvePlotter :title="`${curve.title} source preview`" :datasets="curve.datasets" x-label="Signed current (A)" :y-label="curve.unit" :allow-zoom="false" />
            <div class="table-scroll"><table><caption>Original indices and values · first 200 rows. The source tree provides paged array inspection.</caption><thead><tr><th>Original index</th><th>Signed current (A)</th><th>{{ curve.unit }}</th></tr></thead><tbody><tr v-for="(point, i) in curve.pairs.slice(0,200)" :key="i"><td>{{ i }}</td><td>{{ point.x }}</td><td>{{ point.y }}</td></tr></tbody></table></div>
          </template><p v-else>Use the field tree to inspect source arrays. No matching inline curve pair is available.</p>
        </details>
      </section>
      <section class="panel"><h3>Original image and calibration evidence</h3><p>Expand a trace image in the field tree to view original pixels. The original MAT always retains image bytes and anchors.</p>
        <p data-testid="mat-coordinate-convention">Historical coordinates: {{ preview.legacy_coordinates }}. X locations multiply by image height. Y locations multiply by image width.</p>
        <p>Python reconstructs values only for comparison. Stored source values remain authoritative. Outside-calibration samples require exclusion or an explicit estimate declaration.</p>
        <p v-if="!Object.keys(preview.calibrations).length">No recognized calibration evidence exists in this workspace.</p>
        <details v-for="(calibration, path) in preview.calibrations" :key="path" class="calibration"><summary>{{ path }}</summary>
          <p v-if="calibration.status">{{ calibration.status }}</p>
          <div v-for="(axis, name) in calibration.axes" :key="name"><h4>{{ name.toUpperCase() }} axis · {{ axis.label ?? 'Unknown label' }}</h4><p>{{ axis.status }} · scale {{ axis.scale ?? 'unknown' }} · maximum absolute residual {{ axis.max_absolute_residual ?? 'unknown' }}</p><p v-if="axis.message">{{ axis.message }}</p><p>Outside-calibration indices: {{ axis.outside_calibration_indices?.join(', ') || 'None reported' }}</p>
            <div v-if="axis.reconstructed_values" class="table-scroll"><table><caption>Python calibration comparison · first 100 rows</caption><thead><tr><th>Index</th><th>Reconstructed value</th><th>Residual from stored value</th><th>Historical pixel location</th></tr></thead><tbody><tr v-for="(value, i) in axis.reconstructed_values.slice(0,100)" :key="i"><td>{{ i }}</td><td>{{ value }}</td><td>{{ axis.residuals[i] }}</td><td>{{ axis.legacy_pixel_locations?.[i] ?? 'No image' }}</td></tr></tbody></table></div>
          </div>
        </details>
      </section>
      <section v-if="preview.workspace_schema === 'scData'" class="panel"><LegacyMatMapping :source="source" :document="preview" /></section>
    </template>
    <details class="panel"><summary>Supported subset and resource limits</summary><p>MAT5 numeric arrays, character arrays, scalar structs, and the reviewed literal Octave text subset support preview. HDF5 v7.3, objects, handles, sparse, complex, cell, and general executable workspaces are unsupported.</p>
      <p>Source 24 MiB. Source plus expanded MAT data 96 MiB. One array 8,000,000 elements. Aggregate arrays 24,000,000 elements. One requested array blob 24 MiB.</p>
      <p>Maximum 128 variables, 4,096 fields, and nesting depth 16. Inline previews retain at most 4,096 values per array and 200,000 total. Calibration comparisons retain at most 20,000 locations. Backend errors report limits without partial scientific results.</p>
    </details>
  </div>
</template>
<style scoped>
.mat-workbench { max-width: 1250px; margin: auto; padding: 1.2rem; min-width: 0; overflow-wrap: anywhere; } header { margin-bottom: 1rem; } h2,h3,h4 { margin-bottom: .6rem; } p { margin: .6rem 0; }
.panel { min-width: 0; background: var(--bg-secondary,white); border: 1px solid #bdcad9; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; } .upload { display: flex; flex-direction: column; gap: .5rem; min-width: 0; font-weight: 600; } input { min-width: 0; max-width: 100%; width: 100%; font: inherit; } .hash,dd { min-width: 0; overflow-wrap: anywhere; } .source-grid { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,3fr); gap: .5rem 1rem; } dt { font-weight: 600; } dd { margin: 0; }
.table-scroll { max-width: 100%; overflow-x: auto; max-height: 500px; margin: .8rem 0; } table { width: 100%; min-width: 620px; border-collapse: collapse; } th,td { text-align: left; padding: .6rem; border: 1px solid #c2cddb; vertical-align: top; } th { background: var(--bg-tertiary,#eef3f8); } td small { display: block; font-size: .8rem; } caption { text-align: left; } button { padding: .65rem .8rem; font: inherit; color: inherit; background: var(--bg-primary,white); border: 1px solid #859db8; border-radius: 5px; cursor: pointer; } button:disabled { opacity: .5; } .actions { display: flex; flex-wrap: wrap; gap: .6rem; margin: .9rem 0; } summary { cursor: pointer; font-weight: 600; } .curve,.calibration { min-width: 0; margin: 1rem 0; padding: .5rem; border-left: 2px solid #adbed1; } .notice { border-left: 3px solid #b47718; padding-left: .7rem; } [role=alert] { color: #b42323; } ul { margin-left: 1.4rem; }
@media(max-width:550px) { .mat-workbench { padding: .7rem; } .panel { padding: .8rem; } .source-grid { grid-template-columns: minmax(0,1fr); gap: .2rem; } dd { margin-bottom: .6rem; } .actions { flex-direction: column; } }
</style>
