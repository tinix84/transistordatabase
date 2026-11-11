<template>
  <section class="download-panel">
    <p>Download source records from Python. These data exports do not establish simulator acceptance.</p>
    <label>Filter devices <input v-model="query" type="search" placeholder="Name or manufacturer" /></label>
    <div class="actions">
      <button type="button" @click="selectVisible">Select visible devices</button>
      <button type="button" @click="selected = []">Clear selection</button>
      <button v-if="!batch" type="button" @click="$emit('open-database-search')">Search database</button>
    </div>
    <fieldset class="devices">
      <legend>Devices ({{ selected.length }} selected)</legend>
      <label v-for="device in filtered" :key="device.metadata.name">
        <input v-model="selected" type="checkbox" :value="device.metadata.name" />
        {{ device.metadata.name }} · {{ device.metadata.manufacturer || 'Unknown manufacturer' }}
      </label>
      <p v-if="!filtered.length">No transistors match this filter.</p>
    </fieldset>
    <label>Download format
      <select v-model="format" data-testid="export-format">
        <option value="json">Native JSON · complete source archive</option>
        <option value="csv">CSV · ratings summary only</option>
        <option value="matlab">MATLAB MAT5 · numeric view and core_json</option>
        <option value="gecko">GeckoCIRCUITS · switch SCL and evidence manifest ZIP</option>
      </select>
    </label>
    <p v-if="format === 'json'">The original download bytes preserve integer precision, array shapes, conditions, and evidence.</p>
    <p v-if="format === 'csv'">CSV contains ratings only. It does not contain channel curves, switching losses, or thermal models.</p>
    <p v-if="format === 'matlab'">MAT5 contains the legacy numeric view and complete core_json. It is not SC scData or a Simulink model.</p>
    <fieldset v-if="format === 'gecko'" class="conditions">
      <legend>Exact recorded switch conditions</legend>
      <p>All selected devices must contain this slice. No nearest-condition search or inferred losses apply.</p>
      <label>Component <select disabled><option>switch</option></select></label>
      <label v-for="field in fields" :key="field.key">{{ field.label }}
        <input v-model="conditions[field.key]" :data-testid="`gecko-${field.key}`" type="text"
          :placeholder="field.nullable ? 'Number or null' : 'Number'" />
      </label>
      <p>Enter null only when the source records an unknown gate condition. Empty fields are not unknown values.</p>
      <p>Vg on matches the recorded on-gate voltage for both energy events. Vg off is a separate recorded condition.</p>
      <p v-if="conditionError" role="status">{{ conditionError }}</p>
    </fieldset>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-if="success" role="status">{{ success }}</p>
    <button type="button" data-testid="download-export" :disabled="!selected.length || busy || !!conditionError" @click="download">
      {{ busy ? 'Preparing download…' : `Download ${selected.length} selected device${selected.length === 1 ? '' : 's'}` }}
    </button>
    <aside>
      <h3>Scientific workbenches</h3>
      <p>PLECS requires source table selection and explicit omissions. DPT requires explicit operating conditions and a selected template.</p>
      <div class="actions">
        <button type="button" @click="$emit('navigate-workbench', 'plecs-data')">Open PLECS data workbench</button>
        <button type="button" @click="$emit('navigate-workbench', 'dpt')">Open DPT workbench</button>
      </div>
      <p>Generic SPICE and Simulink device models are unavailable. Ratings do not establish calibrated simulator models.</p>
    </aside>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { transistorApi } from '../services/api.js'

const props = defineProps({
  transistors: { type: Array, default: () => [] },
  selectedFromSearch: { type: Array, default: () => [] },
  batch: Boolean,
})
const emit = defineEmits(['open-database-search', 'navigate-workbench', 'export-complete'])
const selected = ref([])
const query = ref('')
const format = ref('json')
const busy = ref(false)
const error = ref('')
const success = ref('')
const fields = [
  { key: 't_j', label: 'Junction temperature (°C)' },
  { key: 'v_supply', label: 'Supply voltage (V)' },
  { key: 'v_g_on', label: 'Vg on (V)', nullable: true },
  { key: 'v_g_off', label: 'Vg off (V)', nullable: true },
  { key: 'r_g_on', label: 'Rg on (Ω)', nullable: true },
  { key: 'r_g_off', label: 'Rg off (Ω)', nullable: true },
]
const conditions = ref(Object.fromEntries(fields.map(field => [field.key, ''])))
watch(() => props.selectedFromSearch, devices => {
  selected.value = [...new Set([...selected.value, ...devices.map(device => device.metadata.name)])]
}, { immediate: true, deep: true })
const devices = computed(() => [...new Map([...props.transistors, ...props.selectedFromSearch]
  .map(device => [device.metadata.name, device])).values()])
const filtered = computed(() => devices.value.filter(device =>
  `${device.metadata.name} ${device.metadata.manufacturer || ''}`.toLowerCase().includes(query.value.toLowerCase())))
function selectVisible() {
  selected.value = [...new Set([...selected.value, ...filtered.value.map(device => device.metadata.name)])]
}
const selection = computed(() => {
  const result = { component: 'switch' }
  for (const field of fields) {
    const raw = String(conditions.value[field.key]).trim()
    if (field.nullable && raw === 'null') result[field.key] = null
    else {
      if (!raw || !Number.isFinite(Number(raw))) throw new Error(`Enter ${field.label}${field.nullable ? ' or null' : ''}.`)
      result[field.key] = Number(raw)
      if (field.key === 'v_supply' && result[field.key] <= 0) throw new Error('Supply voltage must be positive.')
      if (field.key.startsWith('r_g') && result[field.key] < 0) throw new Error('Gate resistance must be nonnegative.')
    }
  }
  return result
})
const conditionError = computed(() => {
  if (format.value !== 'gecko') return ''
  try { selection.value; return '' } catch (err) { return err.message }
})
async function download() {
  if (!selected.value.length || busy.value || conditionError.value) return
  busy.value = true; error.value = ''; success.value = ''
  try {
    const ids = [...selected.value]
    const selectedFormat = format.value
    const multiple = props.batch || ids.length > 1
    const options = selectedFormat === 'gecko' ? { ...selection.value } : {}
    const blob = multiple
      ? await transistorApi.batchExport(ids, selectedFormat, selectedFormat === 'gecko' ? { selection: options } : {})
      : await transistorApi.export(ids[0], selectedFormat, options)
    const extension = multiple || selectedFormat === 'gecko' ? 'zip' : { json: 'json', csv: 'csv', matlab: 'mat' }[selectedFormat]
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = multiple ? `transistor_exports_${selectedFormat}.zip` : `${ids[0]}_${selectedFormat}.${extension}`
    document.body.appendChild(link); link.click(); link.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    success.value = `Download prepared for ${ids.length} device${ids.length === 1 ? '' : 's'}.`
    emit('export-complete', { count: ids.length, format: selectedFormat })
  } catch (err) { error.value = err.message || 'Export failed.' }
  finally { busy.value = false }
}
</script>

<style scoped>
.download-panel { display: grid; gap: 1rem; max-width: 1100px; margin: auto; }
label { display: grid; gap: .4rem; }
input, select, button { padding: .65rem; max-width: 100%; min-width: 0; font: inherit; }
button { cursor: pointer; border: 1px solid #cbd5e1; border-radius: .4rem; background: #eff6ff; color: #172554; }
button:disabled { cursor: default; opacity: .55; }
fieldset { border: 1px solid #cbd5e1; border-radius: .5rem; padding: 1rem; min-width: 0; }
.devices { max-height: 20rem; overflow: auto; }
.devices label { display: flex; align-items: center; margin: .6rem 0; overflow-wrap: anywhere; }
.actions { display: flex; flex-wrap: wrap; gap: .6rem; }
.conditions { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; }
.conditions p { grid-column: 1 / -1; margin: 0; }
aside { border-top: 1px solid #cbd5e1; padding-top: 1rem; }
[role=alert] { color: #b91c1c; white-space: pre-wrap; }
@media (max-width: 500px) { .conditions { grid-template-columns: 1fr; } }
</style>
