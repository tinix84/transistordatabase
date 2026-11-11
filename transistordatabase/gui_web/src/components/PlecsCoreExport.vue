<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { plecsDataApi, transistorApi } from '../services/api.js'
import CurvePlotter from './CurvePlotter.vue'
const kinds = [['conduction', 'Conduction'], ['turn_on', 'Turn-on'], ['turn_off', 'Turn-off / diode recovery']]
const devices = ref([]), contexts = ref(null), result = ref(null), error = ref(''), busy = ref(false), loading = ref(false)
const input = reactive({ device: '', component: '', selection: Object.fromEntries(kinds.map(([key]) => [key, { include: false, context: '', reason: '' }])), thermal: false, thermalReason: '', convention: '' })
let revision = 0, disposed = false
watch(input, () => { revision++; result.value = null; error.value = '' }, { deep: true, flush: 'sync' })
watch(() => input.device, loadContexts)
watch(() => input.component, () => { for (const row of Object.values(input.selection)) row.context = '' })
onBeforeUnmount(() => { disposed = true; revision++ })
onMounted(async () => { try { const values = await transistorApi.getAll(); if (!disposed) devices.value = values } catch (failure) { if (!disposed) error.value = failure.message } })
async function loadContexts() {
  contexts.value = null; input.component = ''
  for (const row of Object.values(input.selection)) { row.include = false; row.context = ''; row.reason = '' }
  input.thermal = false; input.thermalReason = ''; input.convention = ''
  if (!input.device) return
  const token = revision; loading.value = true
  try { const data = await plecsDataApi.contexts(input.device); if (!disposed && token === revision) contexts.value = data }
  catch (failure) { if (!disposed && token === revision) error.value = failure.response?.data?.detail ?? failure.message }
  finally { if (!disposed) loading.value = false }
}
const available = computed(() => contexts.value?.components?.[input.component])
const describe = values => Object.entries(values ?? {}).map(([key, value]) => `${key}: ${value === null ? 'unknown' : typeof value === 'object' ? JSON.stringify(value) : value}`).join('; ')
function request() {
  if (!input.device || !input.component || !available.value) throw new Error('Select a stored device and component')
  if (!input.convention) throw new Error('Select the recovery voltage convention')
  const result = { component: input.component, include_thermal: input.thermal, recovery_voltage_convention: input.convention, omission_reasons: {} }
  for (const [key, label] of kinds) {
    const row = input.selection[key]
    if (!row.include) {
      if (!row.reason.trim()) throw new Error(`Explain the ${label} omission`)
      result[key] = null; result.omission_reasons[key] = row.reason.trim()
    } else {
      if (row.context === '') throw new Error(`Select an exact recorded ${label} context`)
      const context = available.value[key][Number(row.context)]
      result[key] = { gate_voltage_v: context.gate_voltage_v, ...(key === 'conduction' ? {} : { gate_resistance_ohm: context.gate_resistance_ohm }) }
    }
  }
  if (!input.thermal) {
    if (!input.thermalReason.trim()) throw new Error('Explain the thermal omission')
    result.omission_reasons.thermal = input.thermalReason.trim()
  }
  return result
}
async function preview() {
  result.value = null; error.value = ''; const token = ++revision; busy.value = true
  try { const data = await plecsDataApi.previewCore(input.device, request()); if (!disposed && token === revision) result.value = data }
  catch (failure) { if (!disposed && token === revision) { const detail = failure.response?.data?.detail; error.value = typeof detail === 'string' ? detail : failure.message } }
  finally { if (!disposed) busy.value = false }
}
function save(blob, name) { const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url) }
function json() { save(new Blob([JSON.stringify(result.value, null, 2)], { type: 'application/json' }), 'plecs-selected-core-evidence.json') }
async function xml() {
  error.value = ''; const token = revision; busy.value = true
  try { const data = await plecsDataApi.exportCore(input.device, request()); if (!disposed && token === revision) save(data, 'plecs-selected-core.xml') }
  catch (failure) {
    if (!disposed && token === revision) {
      let detail = failure.response?.data
      if (detail instanceof Blob) { try { detail = JSON.parse(await detail.text()) } catch { detail = null } }
      if (!disposed && token === revision) error.value = detail?.detail ?? failure.message
    }
  } finally { if (!disposed) busy.value = false }
}
const plotted = computed(() => (result.value?.document.packages ?? []).flatMap(pkg => pkg.blocks.flatMap(block => block.tables.map(table => ({
  key: table.source_path, kind: table.kind, datasets: table.slices.map((slice, index) => ({ label: slice.label, data: slice.plot_data.map(([x, y]) => ({ x, y })), borderColor: ['#2469b2', '#b54335', '#26844a', '#8d4da6'][index % 4] })), slices: table.slices,
})))))
</script>
<template>
  <section class="core-export" data-testid="plecs-core-export">
    <h3>Export recorded curves from the current database</h3>
    <p>Select exact recorded gate contexts and include each mechanism deliberately. Missing rectangles, unequal current axes and ambiguous source conditions are reported by Python; no interpolation or nearest record is selected.</p>
    <form @submit.prevent="preview">
      <label>Stored device <select v-model="input.device" data-testid="plecs-existing-device"><option value="">Select device</option><option v-for="device in devices" :key="device.metadata.name" :value="device.metadata.name">{{ device.metadata.name }}</option></select></label>
      <p v-if="loading" role="status">Loading recorded contexts…</p>
      <template v-if="contexts">
        <section v-if="contexts.diagnostics?.length" aria-label="Recorded context diagnostics"><h4>Source records needing attention</h4><ul><li v-for="(item, index) in contexts.diagnostics" :key="index">{{ item.component }} · {{ item.kind }} · Record {{ item.record_index ?? 'collection' }}: {{ item.message }} ({{ item.code }})</li></ul><p>These records remain source evidence. A listed context does not guarantee that the full selected rectangle can be exported.</p></section>
        <label>Physical component <select v-model="input.component" data-testid="plecs-existing-component"><option value="">Select component</option><option value="switch">Switch</option><option value="diode">Diode</option></select></label>
        <template v-if="available">
          <fieldset v-for="[key, label] in kinds" :key="key"><legend>{{ label }}</legend>
            <label class="check"><input v-model="input.selection[key].include" type="checkbox" :data-testid="`plecs-include-${key}`"> Include recorded {{ label.toLowerCase() }}</label>
            <template v-if="input.selection[key].include">
              <label>Exact recorded gate context <select v-model="input.selection[key].context" :data-testid="`plecs-context-${key}`"><option value="">Select recorded context</option><option v-for="(context, index) in available[key]" :key="index" :value="String(index)">{{ describe(context) }}</option></select></label>
              <p v-if="!available[key]?.length">No recorded context is available for this mechanism.</p>
            </template>
            <label v-else>Omission reason <input v-model="input.selection[key].reason" :data-testid="`plecs-omit-${key}`"></label>
          </fieldset>
          <label class="check"><input v-model="input.thermal" type="checkbox" data-testid="plecs-include-thermal"> Include component Foster model</label>
          <p>Recorded thermal availability: {{ available.thermal_available ? 'Available' : 'Unavailable' }}</p>
          <label v-if="!input.thermal">Thermal omission reason <input v-model="input.thermalReason" data-testid="plecs-omit-thermal"></label>
          <label>Recovery supply convention <select v-model="input.convention" data-testid="plecs-core-convention"><option value="">Select convention</option><option value="preserve-signed">Preserve recorded signed voltage</option><option value="positive-bus-to-negative-plecs">Explicitly map nonnegative core diode bus to negative PLECS supply</option></select></label>
          <button type="submit" :disabled="busy">Preview selected core export</button>
        </template>
      </template>
    </form>
    <p v-if="error" role="alert">{{ error }}</p>
    <section v-if="result" data-testid="plecs-core-export-result">
      <h4>{{ result.report.status }}</h4><p>Simulator validation: {{ result.report.simulator_validation }}</p>
      <p>Omitted mechanisms: {{ result.report.omitted_mechanisms.join(', ') || 'None' }}. Omissions remain outside this selected subset.</p>
      <p>{{ result.report.fixed_conditions_storage }}</p>
      <button @click="json">Download selected core evidence JSON</button> <button :disabled="busy" @click="xml">Download selected core PLECS XML</button>
      <div class="table-scroll"><table><caption>Exact selected recorded coverage</caption><thead><tr><th>Kind</th><th>Gate context</th><th>Additional recorded conditions</th><th>Tj (°C)</th><th>Supply (V)</th><th>Curve count</th></tr></thead><tbody><tr v-for="row in result.report.evidence" :key="row.kind"><td>{{ row.kind }}</td><td>{{ describe(row.conditions) }}</td><td>{{ describe(row.additional_conditions) || 'Not applicable' }}</td><td>{{ row.temperatures_c.join(', ') }}</td><td>{{ row.voltages_v?.join(', ') ?? 'Not applicable' }}</td><td>{{ row.curve_count }}</td></tr></tbody></table></div>
      <section v-for="table in plotted" :key="table.key"><CurvePlotter :title="`Selected ${table.kind}`" :datasets="table.datasets" x-label="Current (A)" :y-label="table.kind === 'conduction' ? 'Voltage drop (V)' : 'Energy (J)'" />
        <details><summary>Recorded SI points</summary><div v-for="(slice, index) in table.slices" :key="index" class="table-scroll"><table><caption>{{ slice.label }}</caption><thead><tr><th>Current (A)</th><th>{{ slice.unit }}</th></tr></thead><tbody><tr v-for="(point, i) in slice.plot_data" :key="i"><td>{{ point[0] }}</td><td>{{ point[1] }}</td></tr></tbody></table></div></details>
      </section>
    </section>
  </section>
</template>
<style scoped>
.core-export { margin: 1.5rem 0; padding: 1rem; border: 1px solid #aeb9c8; border-radius: 8px; overflow-wrap: anywhere; }
label { display: flex; flex-direction: column; gap: .35rem; margin-bottom: .85rem; font-weight: 600; }
input,select { box-sizing: border-box; width: 100%; min-width: 0; padding: .6rem; color: inherit; background: var(--bg-primary, white); border: 1px solid #aeb9c8; border-radius: 5px; font: inherit; }
.check { flex-direction: row; align-items: center; } .check input { width: auto; }
fieldset { min-width: 0; padding: .8rem; margin: .8rem 0; }
button { padding: .6rem .85rem; border: 1px solid #8ba0b9; border-radius: 5px; background: var(--bg-secondary, #eef4fa); color: inherit; cursor: pointer; }
button:disabled { opacity: .5; } [role=alert] { color: #b52222; }
.table-scroll { max-width: 100%; overflow-x: auto; margin: 1rem 0; } table { min-width: 650px; width: 100%; border-collapse: collapse; }
th,td { min-width: 90px; text-align: left; padding: .6rem; border: 1px solid #c6cfdb; } caption { text-align: left; }
details { margin: .8rem 0; } summary { cursor: pointer; font-weight: 600; }
</style>
