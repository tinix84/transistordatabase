<template>
  <section class="transistor-comparison-enhanced" data-testid="advanced-comparison">
    <header class="comparison-header"><h2>Advanced transistor comparison</h2><div class="header-actions"><button @click="clearAll">Clear selection</button><button data-testid="run-comparison" @click="runComparison" :disabled="selectedCount < 2 || busy">Run comparison</button></div></header>
    <p>Recorded curves retain their original conditions. Estimate settings do not select nearby measurements.</p>
    <div class="transistor-slots">
      <article v-for="slot in 3" :key="slot" class="transistor-slot">
        <label>Transistor {{ slot }}<select :value="selectedIds[slot - 1]" :data-testid="`comparison-device-${slot}`" @change="selectedIds[slot - 1] = $event.target.value"><option value="">Select a stored transistor</option><option v-for="device in allTransistors" :key="device.metadata.name" :value="device.metadata.name" :disabled="selectedIds.includes(device.metadata.name) && selectedIds[slot - 1] !== device.metadata.name">{{ device.metadata.name }}</option></select></label>
        <fieldset v-if="estimateSelected && selectedIds[slot - 1]" class="config-panel"><legend>Conditions for this device</legend>
          <label>Supply voltage (V)<input type="number" step="any" v-model="configurations[slot - 1].v_supply" :data-testid="`comparison-supply-${slot}`"></label>
          <label>Gate on voltage (V)<input type="number" step="any" v-model="configurations[slot - 1].v_g"></label>
          <label>Gate off voltage <select v-model="configurations[slot - 1].offKnown"><option :value="false">Explicitly unknown</option><option :value="true">Known</option></select><input v-if="configurations[slot - 1].offKnown" type="number" step="any" v-model="configurations[slot - 1].v_g_off" aria-label="Gate off voltage (V)"></label>
          <label>Gate resistance on (Ω)<input type="number" step="any" v-model="configurations[slot - 1].r_g_on"></label>
          <label>Gate resistance off (Ω)<input type="number" step="any" v-model="configurations[slot - 1].r_g_off"></label>
          <label>Identical parallel device count<input type="number" min="1" step="1" v-model="configurations[slot - 1].parallel_count"></label>
        </fieldset>
      </article>
    </div>
    <label class="estimate-choice"><input type="checkbox" v-model="estimateSelected" data-testid="comparison-estimate"> I select the held-current hard-switching PWM estimate</label>
    <div v-if="estimateSelected" class="global-config-section">
      <p>Current is the total shared current. Each identical device carries current / count; all losses are multiplied by count. Duty applies to conduction; 0 &lt; duty &lt; 1 selects one on/off event per switching period. D = 0, D = 1 or fs = 0 has no switching events.</p>
      <div class="global-config-grid"><label v-for="field in globalFields" :key="field[0]">{{ field[1] }}<input type="number" step="any" v-model="globalConfig[field[0]]" :data-testid="`comparison-${field[0]}`"></label></div>
      <label>Output power (W), optional semiconductor-only efficiency estimate<input type="number" step="any" v-model="globalConfig.output_power_w" data-testid="comparison-output_power_w"></label>
    </div>
    <p v-if="busy" role="status">Evaluating comparison…</p><p v-if="stale" role="status">Inputs changed. Run comparison for current settings.</p><p v-if="error" role="alert">{{ error }}</p>
    <section v-if="comparisonResults" class="results-section" data-testid="comparison-results">
      <h3>Comparison results</h3><p>{{ comparisonResults.observation_policy }}</p><button @click="download">Download evidence JSON</button>
      <nav class="plot-tabs"><button v-for="panel in panels" :key="panel[0]" @click="activePlot = panel[0]" :class="['plot-tab', { active: activePlot === panel[0] }]">{{ panel[1] }}</button></nav>
      <section v-if="activePanel"><h3>{{ activePanel.title }}</h3>
        <template v-if="activePanel.curves">
          <p>{{ activePanel.status }}<span v-if="activePanel.reason">: {{ activePanel.reason }}</span></p>
          <CurvePlotter v-if="activePanel.curves.length" :datasets="chartDatasets" :title="activePanel.title" :x-label="activePanel.xlabel" :y-label="activePanel.ylabel" :x-log-scale="activePanel.log_scale_x || false" />
          <div v-if="activePanel.curves.length" class="comparison-table-scroll"><table><thead><tr><th>Device / record</th><th>Actual recorded conditions</th><th>Samples</th></tr></thead><tbody><tr v-for="(curve, index) in activePanel.curves" :key="index"><td>{{ curve.device }} / {{ curve.record_index }}</td><td>{{ conditions(curve.conditions) }}</td><td>{{ samples(curve) }}</td></tr></tbody></table></div>
          <p v-for="(diagnostic, index) in activePanel.diagnostics || []" :key="index">{{ diagnostic.name }}: {{ diagnostic.status }} — {{ diagnostic.reason }}</p>
        </template>
        <template v-else-if="activePlot === 'loss_breakdown'">
          <p>Held-current estimate; source coverage and exact conditions are required for every active component.</p>
          <div class="comparison-table-scroll"><table><thead><tr><th>Device</th><th>Status / reason</th><th>Conduction (W)</th><th>On (W)</th><th>Off (W)</th><th>Total (W)</th></tr></thead><tbody><tr v-for="row in activePanel.rows" :key="row.name"><td>{{ row.name }}</td><td>{{ row.status }}<p v-if="row.reason">{{ row.reason }}</p></td><td>{{ number(row.conduction) }}</td><td>{{ number(row.switching_on) }}</td><td>{{ number(row.switching_off) }}</td><td>{{ number(row.total) }}</td></tr></tbody></table></div>
          <p v-for="row in activePanel.rows.filter(row => row.calculation)" :key="row.name">{{ row.name }}: {{ conditions(row.calculation.conditions) }}. {{ row.calculation.switching_status }}. Ideal equal current sharing, count {{ row.parallel_count }}.</p>
        </template>
        <template v-else><p>{{ activePanel.scope }}</p><div class="comparison-table-scroll"><table><thead><tr><th>Device</th><th>Status / reason</th><th>Efficiency (%)</th></tr></thead><tbody><tr v-for="row in activePanel.rows" :key="row.name"><td>{{ row.name }}</td><td>{{ row.status }}<p v-if="row.reason">{{ row.reason }}</p></td><td>{{ number(row.efficiency_percent) }}</td></tr></tbody></table></div></template>
      </section>
    </section>
  </section>
</template>

<script setup>
import { ref, computed, onMounted, watch, onBeforeUnmount } from 'vue'
import { transistorApi } from '../services/api.js'
import CurvePlotter from './CurvePlotter.vue'
const allTransistors = ref([]), selectedIds = ref(['', '', ''])
const configurations = ref(Array.from({ length: 3 }, () => ({ v_supply: '', v_g: '', v_g_off: '', offKnown: false, r_g_on: '', r_g_off: '', parallel_count: 1 })))
const globalConfig = ref({ t_j: '', i_channel: '', f_sw: '', duty: '', output_power_w: '' })
const estimateSelected = ref(false), comparisonResults = ref(null), busy = ref(false), error = ref(''), stale = ref(false)
const activePlot = ref('channel_characteristics')
const selectedCount = computed(() => selectedIds.value.filter(Boolean).length)
const globalFields = [['t_j', 'Junction temperature (°C)'], ['i_channel', 'Total shared current (A)'], ['f_sw', 'Switching frequency (Hz)'], ['duty', 'Conduction duty (0–1)']]
const panels = [['channel_characteristics', 'Channel'], ['switching_losses_eon', 'E_on'], ['switching_losses_eoff', 'E_off'], ['gate_charge', 'Gate charge'], ['soa', 'SOA'], ['thermal_impedance', 'Thermal'], ['capacitances', 'Capacitances'], ['loss_breakdown', 'Loss breakdown'], ['efficiency', 'Efficiency']]
const activePanel = computed(() => comparisonResults.value?.[activePlot.value])
let revision = 0, disposed = false
watch([selectedIds, configurations, globalConfig, estimateSelected], () => { revision++; stale.value = comparisonResults.value !== null || busy.value || stale.value; comparisonResults.value = null; error.value = '' }, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { disposed = true; revision++ })
onMounted(async () => { try { allTransistors.value = await transistorApi.getAll() } catch (e) { error.value = e.message } })
function number(value) { return value == null ? 'Unavailable' : Number(value).toPrecision(7) }
function conditions(value = {}) { return Object.entries(value).map(([key, item]) => `${key}: ${item == null ? 'unknown' : item}`).join('; ') || 'No condition recorded' }
function samples(curve) { return Object.values(curve).find(value => Array.isArray(value))?.length ?? 0 }
const chartDatasets = computed(() => (activePanel.value?.curves || []).map((curve, index) => {
  const keys = curve.i_data && curve.e_data ? ['i_data', 'e_data'] : curve.q_data ? ['q_data', 'v_data'] : curve.c_data ? ['v_data', 'c_data'] : curve.time_data ? ['time_data', 'z_th_data'] : ['v_data', 'i_data']
  return { label: curve.name, data: curve[keys[0]].map((x, i) => ({ x, y: curve[keys[1]][i] })), borderColor: ['#2563eb', '#dc2626', '#16a34a', '#9333ea'][index % 4], pointRadius: 2 }
}))
function finite(value, label) { if (value === '' || value === null || !Number.isFinite(Number(value))) throw new Error(`${label} needs a finite number`); return Number(value) }
function clearAll() { selectedIds.value = ['', '', ''] }
async function runComparison() {
  const chosen = selectedIds.value.map((id, index) => ({ id, index })).filter(item => item.id)
  if (chosen.length < 2) return
  const version = revision; error.value = ''; comparisonResults.value = null; busy.value = true
  try {
    let config = {}
    if (estimateSelected.value) {
      config = Object.fromEntries(globalFields.map(([key, label]) => [key, finite(globalConfig.value[key], label)]))
      for (const key of ['v_supply', 'v_g', 'r_g_on', 'r_g_off', 'parallel_count']) config[key] = chosen.map(({ index }) => finite(configurations.value[index][key], key))
      config.v_g_off = chosen.map(({ index }) => configurations.value[index].offKnown ? finite(configurations.value[index].v_g_off, 'Gate off voltage') : null)
      if (globalConfig.value.output_power_w !== '') config.output_power_w = finite(globalConfig.value.output_power_w, 'Output power')
    }
    const data = await transistorApi.compareAdvanced(chosen.map(item => item.id), config)
    if (!disposed && revision === version) { comparisonResults.value = data; stale.value = false }
  } catch (e) { if (!disposed && revision === version) error.value = e.response?.data?.detail || e.message }
  finally { if (!disposed) busy.value = false }
}
function download() { const url = URL.createObjectURL(new Blob([JSON.stringify(comparisonResults.value, null, 2)], { type: 'application/json' })); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'semiconductor-comparison-evidence.json'; anchor.click(); URL.revokeObjectURL(url) }
</script>

<style scoped>
.transistor-comparison-enhanced { overflow-wrap: anywhere; }
.transistor-comparison-enhanced label { display: grid; gap: .4rem; margin: .8rem 0; }
.transistor-comparison-enhanced select, .transistor-comparison-enhanced input:not([type=checkbox]) { width: 100%; min-width: 0; padding: .5rem; box-sizing: border-box; }
.transistor-comparison-enhanced .estimate-choice { display: flex; align-items: baseline; }
.comparison-table-scroll { max-width: 100%; overflow: auto; }
.comparison-table-scroll table { min-width: 650px; width: 100%; border-collapse: collapse; }
.comparison-table-scroll th, .comparison-table-scroll td { min-width: 110px; text-align: left; padding: .65rem; border: 1px solid #cbd5e1; vertical-align: top; }
@media (max-width: 650px) { .comparison-header { flex-direction: column; align-items: stretch !important; gap: 1rem; } }
.transistor-comparison-enhanced {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

.comparison-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
}

.comparison-header h2 {
  margin: 0;
  color: #1f2937;
  font-size: 28px;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.2s;
}

.btn-primary {
  background: #3b82f6;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
  transform: translateY(-1px);
}

.btn-primary:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.selection-section {
  margin-bottom: 32px;
}

.selection-section h3 {
  margin-bottom: 20px;
  color: #374151;
}

.transistor-slots {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 20px;
}

.transistor-slot {
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  padding: 20px;
  background: white;
  transition: all 0.2s;
}

.transistor-slot.filled {
  border-color: #3b82f6;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
}

.slot-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.slot-header h4 {
  margin: 0;
  color: #374151;
  font-size: 16px;
}

.btn-icon {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
}

.btn-danger-small {
  color: #ef4444;
}

.selected-transistor {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.transistor-info strong {
  display: block;
  color: #1f2937;
  font-size: 16px;
  margin-bottom: 6px;
}

.transistor-info p {
  margin: 0;
  color: #6b7280;
  font-size: 14px;
}

.specs {
  display: flex;
  gap: 12px;
  margin-top: 8px;
}

.specs span {
  background: #eff6ff;
  color: #1e40af;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 500;
}

.config-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
}

.config-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.config-row label {
  flex: 0 0 120px;
  font-size: 13px;
  color: #374151;
  font-weight: 500;
}

.config-row input[type="number"],
.config-row select {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 13px;
}

.config-row input[type="range"] {
  flex: 1;
}

.config-row .value {
  flex: 0 0 50px;
  text-align: right;
  font-weight: 600;
  color: #3b82f6;
  font-size: 13px;
}

.empty-slot {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 200px;
}

.btn-add {
  padding: 12px 24px;
  background: #dbeafe;
  color: #1e40af;
  border: 2px dashed #3b82f6;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;
}

.btn-add:hover {
  background: #bfdbfe;
  border-color: #2563eb;
}

.global-config-section {
  margin-bottom: 32px;
  padding: 20px;
  background: white;
  border-radius: 12px;
  border: 2px solid #e5e7eb;
}

.global-config-section h3 {
  margin-top: 0;
  margin-bottom: 16px;
  color: #374151;
}

.global-config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
}

.config-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.config-item label {
  font-size: 14px;
  color: #374151;
  font-weight: 500;
  white-space: nowrap;
}

.config-item input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 14px;
}

.config-item .unit {
  color: #6b7280;
  font-size: 14px;
}

.results-section {
  margin-top: 40px;
}

.results-section h3 {
  margin-bottom: 20px;
  color: #374151;
}

.plot-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 24px;
}

.plot-tab {
  padding: 10px 16px;
  background: white;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  font-size: 14px;
  color: #374151;
  transition: all 0.2s;
}

.plot-tab:hover {
  border-color: #3b82f6;
  color: #3b82f6;
}

.plot-tab.active {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
}

.plot-container {
  margin-top: 20px;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 16px;
  padding: 0;
  max-width: 800px;
  width: 90%;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px;
  border-bottom: 2px solid #e5e7eb;
}

.modal-header h3 {
  margin: 0;
  color: #1f2937;
}

.btn-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #6b7280;
  padding: 0;
}

.btn-close:hover {
  color: #1f2937;
}

.selector-filters {
  display: flex;
  gap: 12px;
  padding: 20px 24px;
  background: #f9fafb;
}

.search-input {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
}

.selector-filters select {
  padding: 10px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  min-width: 150px;
}

.transistor-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px 24px;
}

.transistor-item {
  padding: 16px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.transistor-item:hover:not(.disabled) {
  border-color: #3b82f6;
  background: #eff6ff;
}

.transistor-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.item-name {
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 6px;
}

.item-specs {
  display: flex;
  gap: 12px;
  font-size: 13px;
  color: #6b7280;
}

.already-selected {
  position: absolute;
  top: 16px;
  right: 16px;
  background: #d1fae5;
  color: #047857;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}

/* Loading Overlay */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.95);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.loading-spinner {
  width: 60px;
  height: 60px;
  border: 4px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-overlay p {
  margin-top: 20px;
  font-size: 18px;
  color: #374151;
  font-weight: 500;
}
</style>
