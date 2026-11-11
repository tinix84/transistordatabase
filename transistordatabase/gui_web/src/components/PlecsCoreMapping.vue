<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { plecsDataApi } from '../services/api.js'
const props = defineProps({ document: { type: Object, required: true }, sourceRequest: { type: Object, required: true } })
const fields = [
  ['v_abs_max', 'Absolute voltage limit (V)'], ['i_abs_max', 'Absolute current limit (A)'],
  ['i_cont', 'Continuous current limit (A)'], ['t_j_max', 'Maximum junction temperature (°C)'],
  ['housing_area', 'Housing area (m²)'], ['cooling_area', 'Cooling area (m²)'],
]
const input = reactive({ package: '', declarations: Object.fromEntries(fields.map(([key]) => [key, ''])), targets: {}, gates: {}, convention: '', thermal: '' })
const selected = computed(() => props.document.packages.find(pkg => pkg.id === input.package))
const result = ref(null), error = ref(''), busy = ref(false)
let revision = 0, disposed = false
watch(input, () => { revision++; result.value = null; error.value = '' }, { deep: true, flush: 'sync' })
watch(() => input.package, () => { input.targets = Object.fromEntries((selected.value?.blocks ?? []).map(block => [block.id, ''])); input.gates = {}; input.thermal = '' })
onBeforeUnmount(() => { disposed = true; revision++ })
function number(value, label) {
  if (String(value).trim() === '' || !Number.isFinite(Number(value))) throw new Error(`Declare a finite ${label}`)
  return Number(value)
}
async function finalize() {
  error.value = ''; result.value = null; const token = ++revision; busy.value = true
  try {
    if (!selected.value) throw new Error('Select the package to map')
    const ratings = {}, thermal_properties = {}, block_targets = {}, gate_voltage_by_table = {}
    for (const [key, label] of fields) (key.endsWith('_area') ? thermal_properties : ratings)[key] = number(input.declarations[key], label)
    for (const block of selected.value.blocks) {
      if (!input.targets[block.id]) throw new Error(`Select the core target for block ${block.id}`)
      block_targets[block.id] = input.targets[block.id]
    }
    for (const [path, value] of Object.entries(input.gates)) if (String(value).trim() !== '') gate_voltage_by_table[path] = number(value, 'gate voltage')
    if (!input.convention) throw new Error('Select the source recovery voltage convention')
    if (selected.value.thermal.length && !input.thermal) throw new Error('Select the owner of the package thermal network')
    const response = await plecsDataApi.finalize({ ...props.sourceRequest, package_id: input.package,
      ratings, thermal_properties, block_targets, gate_voltage_by_table,
      recovery_voltage_convention: input.convention, package_thermal_target: input.thermal || null })
    if (!disposed && token === revision) result.value = response
  } catch (failure) {
    if (!disposed && token === revision) {
      const detail = failure.response?.data?.detail
      error.value = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(row => row.msg).join('; ') : failure.message
    }
  } finally { if (!disposed) busy.value = false }
}
function save(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a')
  link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url)
}
function json() { save(new Blob([JSON.stringify(result.value.core_json_document, null, 2)], { type: 'application/json' }), 'plecs-declared-core.json') }
function xml() {
  const bytes = Uint8Array.from(atob(result.value.roundtrip_xml_base64), character => character.charCodeAt(0))
  save(new Blob([bytes], { type: 'application/xml' }), 'plecs-source-backed-core.xml')
}
</script>
<template>
  <details class="core-mapping" data-testid="plecs-core-mapping">
    <summary>Optional explicit mapping into the transistor core</summary>
    <p>Supply independent datasheet declarations and deliberate block assignments. These values are recorded separately from the original XML. The result is reviewable and is not saved to the database.</p>
    <form @submit.prevent="finalize">
      <label>Package to map <select v-model="input.package" data-testid="plecs-map-package"><option value="">Select package</option><option v-for="pkg in document.packages" :key="pkg.id" :value="pkg.id">{{ pkg.partnumber }} · Package {{ pkg.id }}</option></select></label>
      <template v-if="selected">
        <div class="declarations"><label v-for="[key, label] in fields" :key="key">{{ label }}<input v-model="input.declarations[key]" :data-testid="`plecs-declare-${key}`" type="number" step="any"></label></div>
        <fieldset v-for="block in selected.blocks" :key="block.id"><legend>Block {{ block.id }} · {{ block.device_type }}</legend>
          <label>Core target <select v-model="input.targets[block.id]" :data-testid="`plecs-target-${block.id}`"><option value="">Select target</option><option value="switch">Switch</option><option value="diode">Diode</option></select></label>
          <label v-for="table in block.tables.filter(item => item.gate_voltage_v === null)" :key="table.source_path">Declared gate for {{ table.kind }} (V)<small>{{ table.source_path }}</small><input v-model="input.gates[table.source_path]" data-testid="plecs-declare-gate" type="number" step="any"></label>
          <p>Leave an unknown gate blank only when the mapped core representation does not require it. Python validates each selected branch.</p>
        </fieldset>
        <label>Recovery supply convention <select v-model="input.convention" data-testid="plecs-recovery-convention"><option value="">Select convention</option><option value="preserve-signed">Preserve source signed voltage</option><option value="negative-to-positive-bus">Explicitly map nonpositive diode recovery supply to positive bus</option></select></label>
        <label v-if="selected.thermal.length">Package thermal owner <select v-model="input.thermal" data-testid="plecs-thermal-owner"><option value="">Select owner</option><option value="switch">Switch</option><option value="diode">Diode</option></select></label>
        <button type="submit" :disabled="busy">{{ busy ? 'Mapping…' : 'Validate explicit core mapping' }}</button>
      </template>
    </form>
    <p v-if="error" role="alert">{{ error }}</p>
    <section v-if="result" data-testid="plecs-core-result">
      <h4>Declared core · {{ result.status }}</h4>
      <p>{{ result.core.metadata.name }} · Source SHA-256 {{ result.source.sha256 }}</p>
      <dl><template v-for="[key, label] in fields" :key="key"><dt>{{ label }}</dt><dd>{{ result.declarations.ratings[key] ?? result.declarations.thermal_properties[key] }}</dd></template></dl>
      <div class="table-scroll"><table><caption>Mapped recorded curve counts</caption><thead><tr><th>Component</th><th>Conduction</th><th>Turn-on</th><th>Turn-off</th><th>Recovery</th></tr></thead><tbody><tr v-for="component in ['switch', 'diode']" :key="component"><td>{{ component }}</td><td>{{ result.core[component].channel_data?.length ?? 0 }}</td><td>{{ result.core[component].e_on_data?.length ?? 0 }}</td><td>{{ result.core[component].e_off_data?.length ?? 0 }}</td><td>{{ result.core[component].e_rr_data?.length ?? 0 }}</td></tr></tbody></table></div>
      <button @click="json">Download declared core JSON</button> <button @click="xml">Download source-backed core XML</button>
    </section>
  </details>
</template>
<style scoped>
.core-mapping { margin: 1rem 0; padding: 1rem; border: 1px solid #aeb9c8; border-radius: 8px; }
summary { font-weight: 600; cursor: pointer; }
label { display: flex; flex-direction: column; gap: .35rem; margin-bottom: .85rem; font-weight: 600; }
input, select { box-sizing: border-box; width: 100%; min-width: 0; padding: .6rem; color: inherit; background: var(--bg-primary, white); border: 1px solid #aeb9c8; border-radius: 5px; font: inherit; }
.declarations { display: grid; grid-template-columns: repeat(auto-fit,minmax(min(210px,100%),1fr)); gap: .8rem; }
fieldset { min-width: 0; padding: .8rem; margin: .8rem 0; } small { font-weight: normal; overflow-wrap: anywhere; }
button { padding: .6rem .85rem; border: 1px solid #8ba0b9; border-radius: 5px; background: var(--bg-secondary, #eef4fa); color: inherit; cursor: pointer; }
button:disabled { opacity: .5; } [role=alert] { color: #b52222; }
dt { font-weight: 600; } dd { margin: .2rem 0 .6rem; }
.table-scroll { max-width: 100%; overflow-x: auto; margin: 1rem 0; } table { min-width: 550px; width: 100%; border-collapse: collapse; }
th,td { text-align: left; padding: .6rem; border: 1px solid #c6cfdb; } caption { text-align: left; }
</style>
